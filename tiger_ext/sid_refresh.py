"""E8 -- SID refresh: versioned, auditable re-generation of item Semantic IDs.

Contract: ``integration/plan.md`` E8 table.

    | 输入   | 当前迭代的 TIGER RQ-VAE checkpoint; ``data/<D>/item_emb.parquet`` |
    | 输出   | 新的全量 item SID ``.npy`` ``(n_items, 4)`` int64                  |
    | 新增   | 每轮迭代的 SID 版本化命名与登记(写进 manifest, 记录 ckpt/commit/时间) |
    | 修改   | 无(复用现有 ``generate_code.py``, 通过 CLI 传参)                      |

So this module **adds no SID encoding mechanism of its own**. It drives the
frozen ``rqvae/generate_code.py`` through its existing CLI and adds the two
things the plan asks for on top: versioned naming and a provenance manifest.

Acceptance criteria implemented here (plan.md E8 验收标准):

  ① every item is covered            -> ``validate_sid`` checks n_rows == n_items
  ② frozen collision handling intact -> reuse ``generate_code.py`` verbatim; the
                                        result must contain no duplicate rows
  ③ all tokens in [0, 256)           -> ``validate_sid`` range check
  ④ diff rate vs the previous SID    -> ``diff_rate``, recorded in the manifest
  ⑤ SID <-> checkpoint traceable     -> manifest records the checkpoint sha256,
                                        the generator's sha256, the repo commit and
                                        the device

Two operational facts this module had to accommodate (both measured, see
``integration/exp_e8_sid_refresh.py``):

* **``generate_code.py`` cannot run under ``python -I``.** It does a bare
  ``from datasets import EmbDataset``, which resolves to the HuggingFace
  ``datasets`` package unless the script directory is on ``sys.path``. Isolated
  mode also drops ``PYTHONPATH``, so the only reliable invocation is a plain
  interpreter with ``cwd=rqvae/``. ``run_generator`` does exactly that.
* **Historical checkpoints store a relative ``args.data_path``.** The frozen
  generator reads that field verbatim, so with ``cwd=rqvae/`` a repo-relative
  ``data/Toys/...`` resolves to ``rqvae/data/Toys/...`` and fails. Two relative
  conventions exist in this repo (627 checkpoints store ``data/...``, 61 store
  ``../data/...``, 20 are already absolute). ``resolve_data_path`` resolves both
  against fixed roots -- never against the caller's cwd -- and ``run_generator``
  hands the generator a throwaway checkpoint copy whose ``args.data_path`` is
  absolute, because ``rqvae/`` is frozen and its CLI has no data-path override.
* **The SID is device-dependent.** Regenerating from the same checkpoint is
  bitwise reproducible on one device, but CPU and CUDA disagree on ~3% of items
  (only in the last-layer / collision columns, where the SK path decides
  borderline items). The device is therefore recorded in the manifest, and the
  measured diff rate must be read against that noise floor.

Nothing in ``model/`` or ``rqvae/`` imports this module; it is called by the
E9 scheduler and by the E8 verification script.
"""
import copy
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
SID_COLUMNS = 4          # 3 RQ levels + 1 duplicate-resolution column
CODEBOOK_SIZE = 256      # plan.md E8 验收 ③
SID_VERSION_DIR = "sid_versions"
SID_SUFFIX = ".sid.json"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVQAE_DIR = os.path.join(REPO, "rqvae")
GENERATOR_REL = os.path.join("rqvae", "generate_code.py")
GENERATOR = os.path.join(REPO, GENERATOR_REL)

# checkpoint args worth keeping for provenance (the rest is training config)
_CKPT_ARG_KEYS = ("data_path", "num_emb_list", "e_dim", "layers", "loss_type",
                  "kmeans_init", "kmeans_iters", "sk_iters", "sk_epsilons",
                  "seed", "epochs")

_NAME_RE = re.compile(r"^(?P<dataset>.+)_sid_v(?P<version>\d{4})_"
                      r"(?P<ckpt_sha8>[0-9a-f]{8})\.npy$")


class SidRefreshError(RuntimeError):
    """Base class for E8 failures."""


class SidValidationError(SidRefreshError):
    """The generated SID does not satisfy the E8 acceptance criteria."""


class SidManifestMismatch(SidRefreshError):
    """A resume/reuse check found a SID that does not match its manifest."""


# --------------------------------------------------------------------------- #
# hashing / provenance
# --------------------------------------------------------------------------- #
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(repo=REPO):
    """(commit, dirty) for the working tree, or (None, None) outside git."""
    try:
        commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    check=True).stdout.strip())
        return commit, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def ckpt_args(ckpt_path):
    """The provenance-relevant subset of the checkpoint's stored args."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck.get("args")
    out = {}
    for k in _CKPT_ARG_KEYS:
        if a is not None and hasattr(a, k):
            v = getattr(a, k)
            out[k] = list(v) if isinstance(v, (list, tuple)) else v
    return out


def ckpt_provenance(ckpt_path):
    """Everything needed to say *which* checkpoint produced a SID (验收 ⑤)."""
    if not os.path.isfile(ckpt_path):
        raise SidRefreshError(f"checkpoint not found: {ckpt_path}")
    commit, dirty = git_commit()
    return {
        "ckpt_path": os.path.abspath(ckpt_path),
        "ckpt_sha256": sha256_file(ckpt_path),
        "ckpt_args": ckpt_args(ckpt_path),
        "generator": GENERATOR_REL,
        "generator_sha256": sha256_file(GENERATOR),
        "repo_commit": commit,
        "repo_dirty": dirty,
    }


# --------------------------------------------------------------------------- #
# versioned naming
# --------------------------------------------------------------------------- #
def version_dir(base_dir):
    return os.path.join(base_dir, SID_VERSION_DIR)


def sid_name(dataset, version, ckpt_sha):
    return f"{dataset}_sid_v{int(version):04d}_{ckpt_sha[:8]}.npy"


def sid_path(base_dir, dataset, version, ckpt_sha):
    return os.path.join(version_dir(base_dir), sid_name(dataset, version, ckpt_sha))


def sidecar_path(path):
    """Stage-manifest style sidecar, same convention as E7's ``<ckpt>.stage.json``."""
    return path + SID_SUFFIX


def parse_sid_name(name):
    m = _NAME_RE.match(os.path.basename(name))
    if not m:
        return None
    d = m.groupdict()
    return {"dataset": d["dataset"], "version": int(d["version"]),
            "ckpt_sha8": d["ckpt_sha8"]}


def list_versions(base_dir, dataset):
    """Existing versions for a dataset, oldest first."""
    vd = version_dir(base_dir)
    if not os.path.isdir(vd):
        return []
    out = []
    for fn in sorted(os.listdir(vd)):
        info = parse_sid_name(fn)
        if info and info["dataset"] == dataset and fn.endswith(".npy"):
            info["path"] = os.path.join(vd, fn)
            out.append(info)
    return sorted(out, key=lambda r: r["version"])


def next_version(base_dir, dataset):
    existing = list_versions(base_dir, dataset)
    return (max(v["version"] for v in existing) + 1) if existing else 1


# --------------------------------------------------------------------------- #
# SID loading / validation  (验收 ①②③)
# --------------------------------------------------------------------------- #
def load_sid(path):
    if not os.path.isfile(path):
        raise SidRefreshError(f"SID not found: {path}")
    try:
        codes = np.load(path)
    except Exception as exc:                       # truncated / not an npy
        raise SidValidationError(f"unreadable SID {path}: {exc}") from exc
    return codes


def collision_stats(codes):
    """Collision picture of a SID array.

    ``raw_duplicate_rate`` uses the 3 codebook columns only (what the SID looked
    like before the 4th column did its job); ``resolved_duplicate_rows`` counts
    rows that are still identical across all 4 columns. The frozen pipeline is
    only correct when the latter is 0.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != SID_COLUMNS:
        raise SidValidationError(f"expected (*, {SID_COLUMNS}) SID, got {codes.shape}")
    raw = codes[:, :3]
    _, raw_counts = np.unique(raw, axis=0, return_counts=True)
    _, full_counts = np.unique(codes, axis=0, return_counts=True)
    return {
        "n_items": int(codes.shape[0]),
        "unique_rows": int(len(full_counts)),
        "max_duplicate_group": int(full_counts.max()) if len(full_counts) else 0,
        "resolved_duplicate_rows": int(codes.shape[0] - len(full_counts)),
        "raw_duplicate_rate": float(1.0 - len(raw_counts) / codes.shape[0]) if codes.shape[0] else 0.0,
        "duplicate_column_max": int(codes[:, 3].max()) if codes.shape[0] else 0,
    }


def validate_sid(codes, n_items=None, strict=True):
    """Check the E8 acceptance criteria ①②③ and return the measured statistics.

    ``n_items`` (the item-embedding row count) enables the coverage check ①.
    With ``strict=False`` violations are collected in ``errors`` instead of
    raised, which the verification script uses to prove the checks have teeth.
    """
    errors = []
    codes = np.asarray(codes)

    if codes.ndim != 2 or codes.shape[1] != SID_COLUMNS:
        errors.append(f"shape {codes.shape} != (*, {SID_COLUMNS})")
    if not np.issubdtype(codes.dtype, np.integer):
        errors.append(f"dtype {codes.dtype} is not integer")

    stats = {}
    if not errors:
        lo, hi = int(codes.min()), int(codes.max())
        per_col_max = codes.max(0).tolist()
        per_col_min = codes.min(0).tolist()
        if hi >= CODEBOOK_SIZE or lo < 0:                 # ③
            errors.append(f"token out of range: min {lo}, max {hi}, "
                          f"per-column max {per_col_max} (need all in [0,{CODEBOOK_SIZE}))")
        stats = collision_stats(codes)
        stats["token_min"] = lo
        stats["token_max"] = hi
        stats["per_column_max"] = per_col_max
        stats["per_column_min"] = per_col_min
        if stats["resolved_duplicate_rows"] > 0:          # ②
            errors.append(f"{stats['resolved_duplicate_rows']} duplicate rows survive "
                          "collision handling")
        if n_items is not None and codes.shape[0] != n_items:   # ①
            errors.append(f"coverage {codes.shape[0]} != n_items {n_items}")

    stats["n_items_expected"] = n_items
    stats["errors"] = errors
    stats["ok"] = not errors
    if errors and strict:
        raise SidValidationError("; ".join(errors))
    return stats


def diff_rate(new, previous):
    """④ -- how much the SID moved between two refresh rounds."""
    new = np.asarray(new)
    previous = np.asarray(previous)
    if new.shape != previous.shape:
        raise SidValidationError(f"cannot diff {new.shape} against {previous.shape}")
    per_col = (new != previous)
    return {
        "rows_total": int(new.shape[0]),
        "rows_changed": int(per_col.any(axis=1).sum()),
        "rows_changed_rate": float(per_col.any(axis=1).mean()),
        "per_column_rate": [float(x) for x in per_col.mean(axis=0)],
        "exact_match": bool(np.array_equal(new, previous)),
    }


# --------------------------------------------------------------------------- #
# manifest  (验收 ④⑤)
# --------------------------------------------------------------------------- #
def write_sid_manifest(path, record):
    side = sidecar_path(path)
    with open(side, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True, default=str)
    return side


def read_sid_manifest(path):
    side = path if str(path).endswith(SID_SUFFIX) else sidecar_path(path)
    with open(side) as f:
        return json.load(f)


def verify_sid_manifest(record, ckpt_path=None, sid_path_=None):
    """Refuse to treat a SID as belonging to a checkpoint it did not come from."""
    if ckpt_path is not None:
        want = sha256_file(ckpt_path)
        got = record.get("ckpt_sha256")
        if want != got:
            raise SidManifestMismatch(
                f"checkpoint {ckpt_path} sha256 {want[:12]} != manifest {str(got)[:12]}")
    if sid_path_ is not None:
        if not os.path.isfile(sid_path_):
            raise SidManifestMismatch(f"SID missing: {sid_path_}")
        if sha256_file(sid_path_) != record.get("sid_sha256"):
            raise SidManifestMismatch(f"SID content changed since registration: {sid_path_}")
    return True


# --------------------------------------------------------------------------- #
# checkpoint data_path resolution  (B3)
# --------------------------------------------------------------------------- #
def resolve_data_path(data_path, repo=REPO):
    """Absolute embedding table for a checkpoint's stored ``args.data_path``.

    The resolution never depends on the caller's cwd:

    * an **absolute** path is returned unchanged (normalised) -- new checkpoints
      written by ``qspace_train``/``alt_pipeline`` already store absolute paths,
      and their meaning must not shift;
    * a **relative** path must be one of the two historical conventions and is
      resolved against the fixed roots that convention was written for --
      ``data/...`` against the repository root, ``../data/...`` against
      ``rqvae/`` (``rqvae/main.py``'s own default). Exactly one candidate has to
      exist; anything else is an error rather than a guess.

    Raises:
        SidRefreshError: missing/empty/ambiguous path -- never falls back to a
            different embedding table.
    """
    if not data_path:
        raise SidRefreshError("the checkpoint has no args.data_path")
    if os.path.isabs(data_path):
        return os.path.normpath(data_path)
    rqvae_dir = os.path.join(repo, "rqvae")
    candidates = [os.path.normpath(os.path.join(repo, data_path)),
                  os.path.normpath(os.path.join(rqvae_dir, data_path))]
    hits = [c for c in candidates if os.path.isfile(c)]
    if len(hits) > 1:
        same = all(os.path.samefile(hits[0], h) for h in hits[1:])
        if not same:
            raise SidRefreshError(
                f"checkpoint data_path {data_path!r} is ambiguous: both "
                f"{candidates[0]} and {candidates[1]} exist and differ")
        return hits[0]
    if len(hits) == 1:
        return hits[0]
    raise SidRefreshError(
        f"checkpoint data_path {data_path!r} is relative and neither "
        f"{candidates[0]} nor {candidates[1]} exists; refusing to guess")


def ckpt_data_path(ckpt_path):
    """``(stored value, resolved absolute path)`` for a checkpoint's embedding."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stored = getattr(ck.get("args"), "data_path", None)
    return stored, resolve_data_path(stored)


def ckpt_with_absolute_data_path(ckpt_path, workdir):
    """A checkpoint the frozen generator can read from ``cwd=rqvae/``.

    ``generate_code.py`` reads ``args.data_path`` verbatim and ``rqvae/`` is
    frozen (no CLI override for it), so a relative path can only be repaired in
    the payload. The copy changes exactly one field -- ``args.data_path`` -- and
    leaves every weight byte-identical; when the stored path is already the
    resolved absolute one, the original file is passed through untouched.

    Returns:
        ``(path_to_pass, stored_value, resolved_value, rewritten)``
    """
    import torch
    stored, resolved = ckpt_data_path(ckpt_path)
    if stored == resolved:
        return os.path.abspath(ckpt_path), stored, resolved, False
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = copy.copy(ck["args"])            # shallow: every other field verbatim
    args.data_path = resolved
    out = os.path.join(workdir, f"abs_{os.path.basename(ckpt_path)}")
    payload = dict(ck)
    payload["args"] = args
    torch.save(payload, out)
    return out, stored, resolved, True


# --------------------------------------------------------------------------- #
# driving the frozen generator
# --------------------------------------------------------------------------- #
def run_generator(ckpt_path, output_file, dataset, device, timeout=None):
    """Invoke ``rqvae/generate_code.py`` exactly as the frozen pipeline does.

    Plain interpreter (never ``-I``, see the module docstring) with
    ``cwd=rqvae/``; the parsed CLI of that script is used unchanged. The
    checkpoint handed to it carries an absolute ``args.data_path`` (see
    :func:`ckpt_with_absolute_data_path`), so historical checkpoints work from
    any caller cwd. ``ckpt_path`` in the result is always the original.
    """
    if not os.path.isfile(ckpt_path):
        raise SidRefreshError(f"checkpoint not found: {ckpt_path}")
    env = dict(os.environ)
    env["PYTHONPATH"] = RVQAE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    with tempfile.TemporaryDirectory(prefix="sid_refresh_") as workdir:
        gen_ckpt, stored, resolved, rewritten = ckpt_with_absolute_data_path(
            ckpt_path, workdir)
        if not os.path.isfile(resolved):
            raise SidRefreshError(
                f"checkpoint data_path does not exist: {resolved} "
                f"(stored {stored!r})")
        cmd = [sys.executable, "generate_code.py",
               "--dataset", str(dataset),
               "--ckpt_path", os.path.abspath(gen_ckpt),
               "--output_file", os.path.abspath(output_file),
               "--device", str(device)]
        proc = subprocess.run(cmd, cwd=RVQAE_DIR, env=env, capture_output=True,
                              text=True, timeout=timeout)
    ok = proc.returncode == 0 and os.path.isfile(output_file)
    return {"cmd": cmd, "returncode": proc.returncode, "ok": ok,
            "ckpt_path": os.path.abspath(ckpt_path),
            "ckpt_for_generator": os.path.abspath(gen_ckpt),
            "data_path": resolved, "data_path_stored": stored,
            "data_path_rewritten": rewritten,
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-8:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-8:])}


def publish_sid(sid_path_, target):
    """Atomically point the canonical path (what T5 reads) at this version."""
    if not os.path.isfile(sid_path_):
        raise SidRefreshError(f"cannot publish missing SID: {sid_path_}")
    parent = os.path.dirname(os.path.abspath(target))
    if not os.path.isdir(parent):
        raise SidRefreshError(f"publish target directory does not exist: {parent}")
    tmp = f"{target}.tmp-{os.getpid()}"
    shutil.copyfile(sid_path_, tmp)
    os.replace(tmp, target)          # atomic: readers never see a half file
    return target


def n_items_of(ckpt_path):
    """Item count of the embedding the checkpoint was trained on.

    Returns ``(n_items, resolved_absolute_data_path)``. The path is resolved by
    :func:`resolve_data_path`, so a historical repo-relative checkpoint yields
    the same table no matter which directory the caller runs from.
    """
    import pandas as pd
    stored, data_path = ckpt_data_path(ckpt_path)
    if not os.path.isfile(data_path):
        raise SidRefreshError(
            f"checkpoint data_path missing/unreadable: {data_path} (stored {stored!r})")
    return len(pd.read_parquet(data_path)), data_path


def refresh_sid(dataset, ckpt_path, base_dir, device="cuda:0", previous=None,
                force=False, publish=None, timeout=None):
    """Run one SID refresh round: generate -> validate -> version -> register.

    ``base_dir`` is required on purpose: the plan says the location and the
    versioning are E8's contribution, but it does not name a directory, and
    silently writing into ``data/`` or into the read-only ``clean_tiger/`` would
    be a design decision this Task is not allowed to make. The caller (E9's
    scheduler) owns it.

    ``previous`` -- SID to diff against (④). Defaults to the newest registered
    version for this dataset, which is the natural "previous round".
    ``publish``  -- optional canonical path to atomically point at the new SID.
    Returns the manifest record that was written next to the SID.
    """
    if not os.path.isdir(version_dir(base_dir)):
        os.makedirs(version_dir(base_dir), exist_ok=True)

    prov = ckpt_provenance(ckpt_path)
    n_items, data_path = n_items_of(ckpt_path)
    version = next_version(base_dir, dataset)
    final = sid_path(base_dir, dataset, version, prov["ckpt_sha256"])
    if os.path.exists(final) and not force:
        raise SidRefreshError(f"refusing to overwrite an existing SID: {final}")

    tmp_dir = os.path.join(version_dir(base_dir), "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_out = os.path.join(tmp_dir, f"{dataset}_v{version:04d}_{os.getpid()}.npy")

    record = {
        "dataset": dataset,
        "version": version,
        "device": str(device),
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "sid_path": os.path.abspath(final),
        **prov,
        "n_items": n_items,
        # what the run actually read, and what the checkpoint stored verbatim
        "data_path": data_path,
        "ckpt_data_path_stored": prov["ckpt_args"].get("data_path"),
    }

    try:
        gen = run_generator(ckpt_path, tmp_out, dataset, device, timeout=timeout)
        record["generator_returncode"] = gen["returncode"]
        record["generator_cmd"] = [c for c in gen["cmd"]]
        record["generator_ckpt"] = gen["ckpt_path"]
        record["data_path_rewritten"] = gen["data_path_rewritten"]
        if gen["data_path"] != data_path:
            raise SidRefreshError(
                f"the generator would read {gen['data_path']} but n_items was "
                f"counted from {data_path}; refusing to register a mixed SID")
        if not gen["ok"]:
            record["generator_stderr_tail"] = gen["stderr_tail"]
            raise SidRefreshError(f"generate_code.py failed (rc={gen['returncode']}): "
                                  f"{gen['stderr_tail'] or gen['stdout_tail']}")

        codes = load_sid(tmp_out)
        stats = validate_sid(codes, n_items=n_items)        # ①②③ (raises on failure)
        record["sid_stats"] = stats

        prev_path = previous
        if prev_path is None:
            versions = list_versions(base_dir, dataset)
            prev_path = versions[-1]["path"] if versions else None
        if prev_path is not None and os.path.isfile(prev_path):
            record["previous_sid_path"] = os.path.abspath(prev_path)
            record["previous_sid_sha256"] = sha256_file(prev_path)
            record["diff"] = diff_rate(codes, load_sid(prev_path))   # ④
        else:
            record["previous_sid_path"] = None
            record["diff"] = None

        os.replace(tmp_out, final)                          # atomic commit
        record["sid_sha256"] = sha256_file(final)
        record["sid_bytes"] = os.path.getsize(final)
        if publish is not None:
            record["published_to"] = publish_sid(final, publish)
        record["manifest"] = write_sid_manifest(final, record)
    except BaseException:
        # failure state: no versioned SID, no sidecar, no temp left behind
        for p in (tmp_out, sidecar_path(final)):
            try:
                os.remove(p)
            except OSError:
                pass
        raise
    finally:
        try:
            if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
                os.rmdir(tmp_dir)
        except OSError:
            pass
    return record
