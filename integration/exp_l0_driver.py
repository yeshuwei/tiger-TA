#!/usr/bin/env python
"""EXP-L0 driver (R3) -- the single execution entry point for the five frozen
arms of plan.md §19.11.13.

Run:
  python -I integration/exp_l0_driver.py --dry-run
  python -I integration/exp_l0_driver.py                # all five arms
  python -I integration/exp_l0_driver.py --arms lc1e-3 le1.0

Design constraints (R3 task):

* No production file is modified.  The driver only calls the frozen CLIs
  ``python -m tiger_ext.qspace_train`` (kebab-case flags) and
  ``python -m tiger_ext.fixed_train`` (snake_case flags), plus the
  ``tiger_ext.sid_refresh`` library.
* No reuse of old products.  Every invocation gets its own ``--run-id`` and
  every arm gets its own ``<run_root>/<arm>/{ckpt,sid,run,logs}`` tree.  The
  driver refuses to start if any of those directories already holds files.
* Products are identified dynamically: the checkpoint timestamp directory is
  found by diffing ``<ckpt_parent>`` before/after training and the SID version
  is found by diffing ``sid_refresh.list_versions`` before/after the refresh.
  Neither the timestamp nor the version number is ever hardcoded.
* Strict stage ordering: tokenizer -> refresh -> recommender for the lambda_c
  arms; recommender only for the lambda_e arms.  The first failure stops the
  arm *and* every later arm; there is no fallback to an old checkpoint/SID.
* A whole-run record is written to
  ``integration/artifacts/exp_l0/driver_run.json`` (plus an immutable copy
  under the run root) with parameters, commands, exit codes, paths, hashes,
  per-stage status and timestamps.
* ``--dry-run`` prints the plan and the fully resolved commands but executes
  no subprocess and writes nothing to disk.

lambda_csg is recorded in the run record because the frozen RQVAE checkpoint
args do not carry it (R1 residual gap, plan.md §19.11.12).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tiger_ext import sid_refresh as SR  # noqa: E402

# --------------------------------------------------------------------------- #
# frozen recipe (plan.md §19.11.13 -- do not edit without a new R-task)
# --------------------------------------------------------------------------- #
DATASET = "Toys"
DATA_ROOT = "data"
EMBEDDING_PATH = "data/Toys/item_emb_768d_contentonly.parquet"
RELATION_DIR = "integration/artifacts/collab_relation"
RELATION_K = 20

RQVAE_EPOCHS = 3000
RQVAE_BATCH = 1024
RQVAE_EVAL_STEP = 50
RQVAE_WORKERS = 4
RQVAE_WARMUP = 50
SEED_RQVAE = 2024

BASELINE_CKPT = ("rqvae/ckpt/Toys_Content768d_Baseline/"
                 "Jul-31-2026_21-37-33/best_collision_model.pth")
BASELINE_SID = "data/Toys/Toys_content768d_baseline_sid.npy"

SEED_T5 = 2025
T5_EPOCHS = 200
T5_EARLY_STOP = 10
T5_BATCH = 256
T5_INFER = 96
T5_BEAM = 30
T5_LR = "1e-4"
T5_WORKERS = 1
TOPK_LIST = [5, 10, 20]

DEFAULT_RUN_ROOT = os.path.join(REPO, "integration/artifacts/exp_l0/runs")
DEFAULT_RECORD = os.path.join(REPO, "integration/artifacts/exp_l0/driver_run.json")


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class ArmSpec:
    tag: str
    kind: str                      # "csg" | "ete"
    label: str
    lambda_csg: Optional[float]
    lambda_csg_text: str
    lambda_sia: float
    lambda_psa: float
    lambda_sia_text: str
    lambda_psa_text: str

    def as_dict(self) -> dict:
        return {
            "tag": self.tag,
            "kind": self.kind,
            "label": self.label,
            "lambda_csg": self.lambda_csg,
            "lambda_sia": self.lambda_sia,
            "lambda_psa": self.lambda_psa,
        }


ARM_SPECS: Tuple[ArmSpec, ...] = (
    ArmSpec("lc1e-4", "csg", "EXP-L0 λc-1e-4", 1e-4, "1e-4", 0.0, 0.0, "0", "0"),
    ArmSpec("lc1e-3", "csg", "EXP-L0 λc-1e-3", 1e-3, "1e-3", 0.0, 0.0, "0", "0"),
    ArmSpec("lc1e-2", "csg", "EXP-L0 λc-1e-2", 1e-2, "1e-2", 0.0, 0.0, "0", "0"),
    ArmSpec("le0.1", "ete", "EXP-L0 λe-0.1", None, "", 0.1, 0.1, "0.1", "0.1"),
    ArmSpec("le1.0", "ete", "EXP-L0 λe-1.0", None, "", 1.0, 1.0, "1.0", "1.0"),
)
ARM_BY_TAG: Dict[str, ArmSpec] = {a.tag: a for a in ARM_SPECS}
ARM_ORDER: List[str] = [a.tag for a in ARM_SPECS]


class DriverError(RuntimeError):
    """A precondition, product-identification or provenance check failed."""


# --------------------------------------------------------------------------- #
# paths / small helpers
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class ArmPaths:
    arm_dir: str
    ckpt_parent: Optional[str]
    sid_base: Optional[str]
    run_dir: str
    log_dir: str
    refresh_manifest: Optional[str]


def arm_paths(run_root: str, spec: ArmSpec) -> ArmPaths:
    arm_dir = os.path.join(run_root, spec.tag)
    if spec.kind == "csg":
        ckpt_parent = os.path.join(arm_dir, "ckpt")
        sid_base = os.path.join(arm_dir, "sid")
        refresh_manifest = os.path.join(arm_dir, "refresh_manifest.json")
    else:
        ckpt_parent = None
        sid_base = None
        refresh_manifest = None
    return ArmPaths(
        arm_dir=arm_dir,
        ckpt_parent=ckpt_parent,
        sid_base=sid_base,
        run_dir=os.path.join(arm_dir, "run"),
        log_dir=os.path.join(arm_dir, "logs"),
        refresh_manifest=refresh_manifest,
    )


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _rel(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    rel = os.path.relpath(path, REPO)
    # Outside the repo a pile of ../../ is noise; show the absolute path instead.
    return os.path.abspath(path) if rel.startswith(os.pardir + os.sep) else rel


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def list_subdirs(path: Optional[str]) -> List[str]:
    if not path or not os.path.isdir(path):
        return []
    return sorted(os.path.join(path, name) for name in os.listdir(path)
                  if os.path.isdir(os.path.join(path, name)))


def embedding_info(path: str) -> dict:
    """Path, sha256, row count and width of the 768d content-only table."""
    info = {"path": _rel(_abs(path)), "abspath": _abs(path)}
    if not os.path.isfile(_abs(path)):
        info["error"] = "missing"
        return info
    info["sha256"] = sha256_file(_abs(path))
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(_abs(path))
        info["n_rows"] = parquet.metadata.num_rows
        first = parquet.read_row_group(0, columns=["embedding"]).column(0)[0]
        info["width"] = len(first.as_py())
    except Exception as exc:                     # pragma: no cover - optional
        info["error"] = f"width unavailable: {exc}"
    return info


# --------------------------------------------------------------------------- #
# commands (frozen shapes -- see plan.md §19.11.13)
# --------------------------------------------------------------------------- #
def tokenizer_command(spec: ArmSpec, paths: ArmPaths, python: str,
                      device: str) -> List[str]:
    if spec.kind != "csg" or not paths.ckpt_parent:
        raise DriverError(f"{spec.tag} has no tokenizer stage")
    return [
        python, "-m", "tiger_ext.qspace_train",
        "--dataset", DATASET, "--data-root", DATA_ROOT,
        "--artifact-dir", RELATION_DIR, "--k", str(RELATION_K),
        "--lambda-csg", spec.lambda_csg_text,
        "--epochs", str(RQVAE_EPOCHS), "--batch-size", str(RQVAE_BATCH),
        "--eval-step", str(RQVAE_EVAL_STEP),
        "--num-workers", str(RQVAE_WORKERS),
        "--warmup-epochs", str(RQVAE_WARMUP),
        "--embedding-path", EMBEDDING_PATH,
        "--device", device, "--seed", str(SEED_RQVAE),
        "--ckpt-dir", paths.ckpt_parent,
    ]


def recommender_command(spec: ArmSpec, paths: ArmPaths, sid_path: str,
                        ckpt_path: str, python: str, device: str) -> List[str]:
    return [
        python, "-m", "tiger_ext.fixed_train",
        "--dataset", DATASET, "--data_root", DATA_ROOT,
        "--embedding_path", EMBEDDING_PATH,
        "--sid_path", sid_path,
        "--sid_manifest", sid_path + ".sid.json",
        "--tokenizer_ckpt", ckpt_path,
        "--arm", spec.label,
        "--lambda_sia", spec.lambda_sia_text,
        "--lambda_psa", spec.lambda_psa_text,
        "--seed", str(SEED_T5), "--num_epochs", str(T5_EPOCHS),
        "--early_stop", str(T5_EARLY_STOP),
        "--batch_size", str(T5_BATCH), "--infer_size", str(T5_INFER),
        "--beam_size", str(T5_BEAM), "--lr", T5_LR,
        "--num_workers", str(T5_WORKERS), "--device", device,
        "--run_dir", paths.run_dir,
    ]


REFRESH_SNIPPET = (
    "import json, sys\n"
    "from tiger_ext import sid_refresh as SR\n"
    "dataset, ckpt, base, device, out = sys.argv[1:6]\n"
    "rec = SR.refresh_sid(dataset, ckpt, base, device=device)\n"
    "with open(out, 'w') as fh:\n"
    "    json.dump(rec, fh, indent=2, ensure_ascii=False, default=str)\n"
    "print(json.dumps({k: rec.get(k) for k in ('version', 'sid_path', "
    "'sid_sha256', 'ckpt_sha256', 'n_items', 'sid_stats')}, "
    "ensure_ascii=False))\n"
)


def refresh_command(spec: ArmSpec, paths: ArmPaths, ckpt_path: str,
                    python: str, device: str) -> List[str]:
    if spec.kind != "csg" or not paths.sid_base or not paths.refresh_manifest:
        raise DriverError(f"{spec.tag} has no refresh stage")
    return [
        python, "-c", REFRESH_SNIPPET,
        DATASET, ckpt_path, paths.sid_base, device, paths.refresh_manifest,
    ]


# --------------------------------------------------------------------------- #
# subprocess execution
# --------------------------------------------------------------------------- #
def run_process(command: Sequence[str], log_path: str, cwd: str = REPO,
                echo: bool = True) -> Tuple[int, str]:
    """Run one stage, streaming the complete log to ``log_path``."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    header = (f"\n===== {now_iso()} =====\n$ {shlex.join(command)}\n")
    tail: List[str] = []
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(header)
        log.flush()
        if echo:
            sys.stdout.write(header)
            sys.stdout.flush()
        proc = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)
        rc = proc.wait()
        footer = f"===== exit {rc} at {now_iso()} =====\n"
        log.write(footer)
        if echo:
            sys.stdout.write(footer)
            sys.stdout.flush()
    return rc, "".join(tail)


Runner = Callable[..., Tuple[int, str]]


def _no_runner(command, log_path, **kwargs):      # pragma: no cover - safety
    raise DriverError(f"dry-run must not execute: {shlex.join(command)}")


# --------------------------------------------------------------------------- #
# product identification / provenance
# --------------------------------------------------------------------------- #
def identify_new_ckpt(ckpt_parent: str, before: Sequence[str]) -> Tuple[str, str]:
    """The one timestamped checkpoint directory created by this run."""
    after = list_subdirs(ckpt_parent)
    new = [p for p in after if p not in set(before)]
    if len(new) != 1:
        raise DriverError(
            f"expected exactly one new checkpoint directory under {ckpt_parent}, "
            f"found {new}")
    best = os.path.join(new[0], "best_collision_model.pth")
    if not os.path.isfile(best):
        raise DriverError(f"new checkpoint directory has no best collision "
                          f"checkpoint: {best}")
    return new[0], best


def identify_new_sid(sid_base: str, before: Sequence[dict]) -> dict:
    """The one SID version registered by this run (parsed, never hardcoded)."""
    before_paths = {v["path"] for v in before}
    after = SR.list_versions(sid_base, DATASET)
    new = [v for v in after if v["path"] not in before_paths]
    if len(new) != 1:
        raise DriverError(
            f"expected exactly one new SID under {sid_base}/sid_versions, "
            f"found {[v['path'] for v in new]}")
    return new[0]


def verify_sid_provenance(manifest_path: str, sid_path: str,
                          ckpt_path: str) -> dict:
    """Cross-check the refresh manifest against the actual products."""
    if not os.path.isfile(manifest_path):
        raise DriverError(f"refresh manifest missing: {manifest_path}")
    if not os.path.isfile(sid_path):
        raise DriverError(f"SID missing: {sid_path}")
    if not os.path.isfile(sid_path + ".sid.json"):
        raise DriverError(f"SID sidecar missing: {sid_path}.sid.json")
    with open(manifest_path) as fh:
        rec = json.load(fh)

    problems: List[str] = []
    if os.path.realpath(rec.get("sid_path", "")) != os.path.realpath(sid_path):
        problems.append(f"manifest sid_path {rec.get('sid_path')} != {sid_path}")
    if sha256_file(sid_path) != rec.get("sid_sha256"):
        problems.append("SID sha256 does not match the refresh manifest")
    if sha256_file(ckpt_path) != rec.get("ckpt_sha256"):
        problems.append("checkpoint sha256 does not match the refresh manifest")
    parsed = SR.parse_sid_name(os.path.basename(sid_path))
    if not parsed or parsed["version"] != rec.get("version"):
        problems.append("SID filename version does not match the manifest")
    stats = rec.get("sid_stats") or {}
    if stats.get("ok") is False:
        problems.append(f"SID validation stats failed: {stats.get('errors')}")
    if problems:
        raise DriverError("SID provenance check failed: " + "; ".join(problems))
    return {
        "manifest_path": _rel(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sid_version": rec.get("version"),
        "sid_sha256": rec.get("sid_sha256"),
        "ckpt_sha256": rec.get("ckpt_sha256"),
        "n_items": rec.get("n_items"),
        "verified": True,
        "manifest_verified": True,
    }


# --------------------------------------------------------------------------- #
# record writing
# --------------------------------------------------------------------------- #
def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def base_record(run_id: str, run_root: str, record_path: str, device: str,
                python: str, dry_run: bool) -> dict:
    commit, dirty = SR.git_commit(REPO)
    return {
        "suite": "EXP-L0",
        "driver": _rel(os.path.abspath(__file__)),
        "driver_sha256": sha256_file(os.path.abspath(__file__)),
        "run_id": run_id,
        "run_root": os.path.abspath(run_root),
        "record_path": os.path.abspath(record_path),
        "dry_run": dry_run,
        "status": "running",
        "started_at": now_iso(),
        "finished_at": None,
        "repo_commit": commit,
        "repo_dirty": dirty,
        "environment": {
            "python": python,
            "python_version": sys.version.split()[0],
            "device": device,
        },
        "frozen_recipe": {
            "dataset": DATASET,
            "data_root": DATA_ROOT,
            "embedding_path": EMBEDDING_PATH,
            "relation_artifact_dir": RELATION_DIR,
            "relation_k": RELATION_K,
            "seed_rqvae": SEED_RQVAE,
            "seed_t5": SEED_T5,
            "rqvae": {
                "epochs": RQVAE_EPOCHS,
                "batch_size": RQVAE_BATCH,
                "eval_step": RQVAE_EVAL_STEP,
                "num_workers": RQVAE_WORKERS,
                "warmup_epochs": RQVAE_WARMUP,
            },
            "t5": {
                "num_epochs": T5_EPOCHS,
                "early_stop": T5_EARLY_STOP,
                "batch_size": T5_BATCH,
                "infer_size": T5_INFER,
                "beam_size": T5_BEAM,
                "lr": T5_LR,
                "num_workers": T5_WORKERS,
                "topk_list": list(TOPK_LIST),
            },
            "baseline_ckpt": BASELINE_CKPT,
            "baseline_sid": BASELINE_SID,
        },
        "arms": [],
    }


# --------------------------------------------------------------------------- #
# planning (dry-run)
# --------------------------------------------------------------------------- #
def plan_arm(spec: ArmSpec, paths: ArmPaths, python: str, device: str) -> dict:
    stages = []
    if spec.kind == "csg":
        stages.append({
            "name": "tokenizer",
            "depends_on": None,
            "command": tokenizer_command(spec, paths, python, device),
            "outputs": {"ckpt_parent": _rel(paths.ckpt_parent)},
        })
        stages.append({
            "name": "refresh",
            "depends_on": "tokenizer",
            "command": refresh_command(
                spec, paths,
                os.path.join(paths.ckpt_parent,
                             "<timestamp>/best_collision_model.pth"),
                python, device),
            "outputs": {
                "sid_base": _rel(paths.sid_base),
                "manifest": _rel(paths.refresh_manifest),
            },
        })
        stages.append({
            "name": "recommender",
            "depends_on": "refresh",
            "command": recommender_command(
                spec, paths,
                "<resolved-from-sid_versions>/<dataset>_sid_v<NNNN>_<sha8>.npy",
                os.path.join(paths.ckpt_parent,
                             "<timestamp>/best_collision_model.pth"),
                python, device),
            "outputs": {"run_dir": _rel(paths.run_dir)},
        })
    else:
        stages.append({
            "name": "recommender",
            "depends_on": None,
            "command": recommender_command(
                spec, paths, BASELINE_SID, BASELINE_CKPT, python, device),
            "outputs": {"run_dir": _rel(paths.run_dir)},
        })
    return {
        "arm": spec.as_dict(),
        "paths": {
            "arm_dir": _rel(paths.arm_dir),
            "ckpt_parent": _rel(paths.ckpt_parent),
            "sid_base": _rel(paths.sid_base),
            "run_dir": _rel(paths.run_dir),
            "log_dir": _rel(paths.log_dir),
        },
        "stages": [{
            "name": s["name"],
            "depends_on": s["depends_on"],
            "command": s["command"],
            "command_text": shlex.join(s["command"]),
            "outputs": s["outputs"],
        } for s in stages],
        "status": "planned",
    }


def dry_run_plan(specs: Sequence[ArmSpec], run_root: str, python: str,
                 device: str) -> dict:
    return {
        "suite": "EXP-L0",
        "mode": "dry-run",
        "run_root": os.path.abspath(run_root),
        "execution_order": [s.tag for s in specs],
        "note": ("dry-run: no subprocess is executed, no directory is created, "
                 "no SID is refreshed"),
        "arms": [plan_arm(s, arm_paths(run_root, s), python, device)
                 for s in specs],
    }


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def _preflight(spec: ArmSpec, paths: ArmPaths) -> None:
    """Refuse to run into a directory tree that already holds products."""
    for label, path in (("arm", paths.arm_dir),):
        if os.path.isdir(path) and os.listdir(path):
            raise DriverError(
                f"{spec.tag}: {label} directory is not empty ({path}); "
                f"old products are never reused -- pick a new --run-id")
    if os.path.exists(paths.run_dir):
        raise DriverError(f"{spec.tag}: run_dir already exists: {paths.run_dir}")
    if os.path.exists(paths.log_dir) and os.listdir(paths.log_dir):
        raise DriverError(f"{spec.tag}: log_dir is not empty: {paths.log_dir}")
    if spec.kind == "csg":
        if os.path.exists(paths.refresh_manifest):
            raise DriverError(
                f"{spec.tag}: refresh manifest already exists: "
                f"{paths.refresh_manifest}")
        if os.path.isdir(paths.sid_base):
            versions = SR.list_versions(paths.sid_base, DATASET)
            if versions:
                raise DriverError(
                    f"{spec.tag}: sid_base already has versions: "
                    f"{[v['path'] for v in versions]}")


def stage_record(name: str, command: Sequence[str], started_at: str) -> dict:
    return {
        "name": name,
        "status": "running",
        "command": list(command),
        "command_text": shlex.join(command),
        "exit_code": None,
        "started_at": started_at,
        "finished_at": None,
        "log_path": None,
        "failure_reason": None,
        "inputs": {},
        "outputs": {},
    }


def _execute_stage(stage: dict, log_path: str, runner: Runner,
                   echo: bool) -> Tuple[int, str]:
    rc, tail = runner(list(stage["command"]), log_path, echo=echo)
    stage["exit_code"] = rc
    stage["finished_at"] = now_iso()
    stage["log_path"] = _rel(log_path)
    if rc != 0:
        stage["status"] = "failed"
        stage["failure_reason"] = (f"exit code {rc}; last log lines:\n"
                                   + "\n".join(tail.splitlines()[-10:]))
    else:
        stage["status"] = "completed"
    return rc, tail


def execute_arm(spec: ArmSpec, paths: ArmPaths, ctx: dict,
                record_arm: dict) -> bool:
    """Run one arm. Returns True on success; on failure the record is filled."""
    python = ctx["python"]
    device = ctx["device"]
    runner: Runner = ctx["runner"]
    echo = ctx["echo"]
    os.makedirs(paths.arm_dir, exist_ok=True)
    os.makedirs(paths.log_dir, exist_ok=True)

    record_arm["paths"] = {
        "arm_dir": _rel(paths.arm_dir),
        "ckpt_parent": _rel(paths.ckpt_parent),
        "sid_base": _rel(paths.sid_base),
        "run_dir": _rel(paths.run_dir),
        "log_dir": _rel(paths.log_dir),
    }
    record_arm["started_at"] = now_iso()
    record_arm["status"] = "running"
    products: dict = {}
    record_arm["products"] = products

    def fail(stage: dict, reason: str) -> bool:
        stage["status"] = "failed"
        stage["failure_reason"] = reason
        record_arm["status"] = "failed"
        record_arm["failed_stage"] = stage["name"]
        record_arm["failure_reason"] = reason
        record_arm["finished_at"] = now_iso()
        return False

    ckpt_path = BASELINE_CKPT
    sid_path = BASELINE_SID

    if spec.kind == "csg":
        # ---- stage 1: frozen 3000-epoch tokenizer ------------------------- #
        before_ckpts = list_subdirs(paths.ckpt_parent)
        stage = stage_record("tokenizer",
                             tokenizer_command(spec, paths, python, device),
                             now_iso())
        stage["inputs"] = {
            "embedding_path": EMBEDDING_PATH,
            "lambda_csg": spec.lambda_csg,
            "seed": SEED_RQVAE,
        }
        record_arm["stages"].append(stage)
        rc, _ = _execute_stage(stage, os.path.join(paths.log_dir,
                                                   "tokenizer.log"),
                               runner, echo)
        if rc != 0:
            return fail(stage, stage["failure_reason"])
        try:
            ckpt_dir, ckpt_path = identify_new_ckpt(paths.ckpt_parent,
                                                    before_ckpts)
        except DriverError as exc:
            return fail(stage, str(exc))
        products["ckpt_dir"] = _rel(ckpt_dir)
        products["ckpt_sha256"] = sha256_file(ckpt_path)
        products["ckpt_is_new"] = True
        stage["outputs"] = {"ckpt_dir": _rel(ckpt_dir),
                            "ckpt_sha256": products["ckpt_sha256"]}

        # ---- stage 2: refresh the SID from that checkpoint ---------------- #
        before_sids = SR.list_versions(paths.sid_base, DATASET) \
            if os.path.isdir(paths.sid_base) else []
        stage = stage_record("refresh",
                             refresh_command(spec, paths, ckpt_path, python,
                                             device),
                             now_iso())
        stage["inputs"] = {"ckpt_path": _rel(ckpt_path),
                           "sid_versions_before": [v["version"] for v in before_sids]}
        record_arm["stages"].append(stage)
        rc, _ = _execute_stage(stage, os.path.join(paths.log_dir, "refresh.log"),
                               runner, echo)
        if rc != 0:
            return fail(stage, stage["failure_reason"])
        try:
            version = identify_new_sid(paths.sid_base, before_sids)
            sid_path = version["path"]
            proven = verify_sid_provenance(paths.refresh_manifest, sid_path,
                                           ckpt_path)
        except DriverError as exc:
            return fail(stage, str(exc))
        products.update(proven)
        products["sid_base_dir"] = _rel(paths.sid_base)
        products["sid_path"] = _rel(sid_path)
        products["sid_is_new"] = True
        products["sid_versions_before"] = [v["version"] for v in before_sids]
        stage["outputs"] = {
            "sid_version": version["version"],
            "sid_path": _rel(sid_path),
            "sid_sha256": products["sid_sha256"],
            "manifest_sha256": products["manifest_sha256"],
        }

    # ---- stage 3: frozen 200-epoch recommender ---------------------------- #
    stage = stage_record("recommender",
                         recommender_command(spec, paths, sid_path, ckpt_path,
                                             python, device),
                         now_iso())
    stage["inputs"] = {
        "sid_path": _rel(_abs(sid_path)),
        "sid_manifest": _rel(_abs(sid_path)) + ".sid.json",
        "tokenizer_ckpt": _rel(_abs(ckpt_path)),
        "lambda_sia": spec.lambda_sia,
        "lambda_psa": spec.lambda_psa,
        "seed": SEED_T5,
    }
    record_arm["stages"].append(stage)
    rc, _ = _execute_stage(stage, os.path.join(paths.log_dir, "recommender.log"),
                           runner, echo)
    if rc != 0:
        return fail(stage, stage["failure_reason"])

    run_json = os.path.join(paths.run_dir, "run.json")
    if not os.path.isfile(run_json):
        return fail(stage, f"run.json missing: {_rel(run_json)}")
    with open(run_json) as fh:
        result = json.load(fh).get("result", {})
    products["run_dir"] = _rel(paths.run_dir)
    products["run_json"] = _rel(run_json)
    products["run_json_sha256"] = sha256_file(run_json)
    products["best_ndcg@20"] = result.get("best_ndcg@20")
    products["best_epoch"] = result.get("best_epoch")
    products["num_epochs_executed"] = result.get("num_epochs_executed")
    products["early_stopped"] = result.get("early_stopped")
    stage["outputs"] = {"run_dir": _rel(paths.run_dir),
                        "run_json": _rel(run_json),
                        "best_ndcg@20": products["best_ndcg@20"]}
    record_arm["status"] = "completed"
    record_arm["finished_at"] = now_iso()
    return True


def _running_stage(arm_record: dict) -> str:
    """Name of the stage that was in flight when the arm raised."""
    stages = arm_record.get("stages") or []
    for stage in reversed(stages):
        if stage.get("status") == "running":
            return stage.get("name") or "driver"
    return stages[-1]["name"] if stages else "driver"


def _mark_raised_stage(arm_record: dict, exc: BaseException) -> str:
    """Attribute an exception to its stage and leave no stage 'running'."""
    for stage in reversed(arm_record.get("stages") or []):
        if stage.get("status") == "running":
            stage["status"] = "failed"
            stage["finished_at"] = now_iso()
            stage["failure_reason"] = f"{type(exc).__name__}: {exc}"
            return stage.get("name") or "driver"
    return _running_stage(arm_record)


def execute_run(specs: Sequence[ArmSpec], run_root: str, record_path: str,
                python: str = "python", device: str = "cuda:0",
                runner: Runner = run_process, echo: bool = True,
                dry_run: bool = False) -> dict:
    run_id = os.path.basename(os.path.normpath(run_root))
    record = base_record(run_id, run_root, record_path, device, python, dry_run)

    if dry_run:
        record["status"] = "dry-run"
        record["finished_at"] = now_iso()
        return record

    if not specs:
        # M1: an empty selection must never become a zero-arm "completed" run.
        raise DriverError("no arms selected; refusing to start an empty run")

    os.makedirs(run_root, exist_ok=True)
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    record["embedding"] = embedding_info(EMBEDDING_PATH)
    ctx = {"python": python, "device": device, "runner": runner, "echo": echo}
    record["failure"] = None

    def finalize(status: str) -> None:
        """Close the whole-run record; a failure is never rewritten as success."""
        record["status"] = status
        record["finished_at"] = now_iso()
        record["arms_completed"] = [a["arm"]["tag"] for a in record["arms"]
                                    if a.get("status") == "completed"]
        record["arms_not_started"] = [s.tag for s in specs
                                      if s.tag not in {a["arm"]["tag"]
                                                       for a in record["arms"]}]
        write_json_atomic(record_path, record)
        run_copy = os.path.join(run_root, "driver_run.json")
        if os.path.abspath(run_copy) != os.path.abspath(record_path):
            write_json_atomic(run_copy, record)

    ok = True
    try:
        for spec in specs:
            paths = arm_paths(run_root, spec)
            try:
                _preflight(spec, paths)
            except DriverError as exc:
                record["arms"].append({
                    "arm": spec.as_dict(),
                    "paths": {"arm_dir": _rel(paths.arm_dir),
                              "ckpt_parent": _rel(paths.ckpt_parent),
                              "sid_base": _rel(paths.sid_base),
                              "run_dir": _rel(paths.run_dir),
                              "log_dir": _rel(paths.log_dir)},
                    "stages": [],
                    "status": "failed",
                    "failed_stage": "preflight",
                    "failure_reason": str(exc),
                    "started_at": now_iso(),
                    "finished_at": now_iso(),
                    "products": {},
                })
                record["failure"] = {
                    "arm": spec.tag,
                    "stage": "preflight",
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                ok = False
                break

            arm_record = {"arm": spec.as_dict(), "stages": [], "products": {},
                          "failed_stage": None, "failure_reason": None}
            record["arms"].append(arm_record)
            try:
                arm_ok = execute_arm(spec, paths, ctx, arm_record)
            except BaseException as exc:
                # M2: anything raised by a stage (Popen failure, corrupt
                # run.json, KeyboardInterrupt, ...) is attributed, recorded and
                # still stops every later arm.
                stage_name = _mark_raised_stage(arm_record, exc)
                arm_record["status"] = "failed"
                arm_record["failed_stage"] = stage_name
                arm_record["failure_reason"] = f"{type(exc).__name__}: {exc}"
                arm_record["finished_at"] = now_iso()
                record["failure"] = {
                    "arm": spec.tag,
                    "stage": stage_name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                raise
            if not arm_ok:
                ok = False
                break
    except BaseException:
        ok = False
        try:
            finalize("failed")
        except Exception as write_exc:      # pragma: no cover - disk failure
            print(f"ERROR: could not write run record: {write_exc}",
                  file=sys.stderr)
        raise

    finalize("completed" if ok else "failed")
    return record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -I integration/exp_l0_driver.py",
        description="EXP-L0 (R3) driver: five frozen arms from plan.md §19.11.13")
    p.add_argument("--run-id", default=None,
                   help="unique run id (default: UTC timestamp); a fresh run "
                        "root is required -- old products are never reused")
    p.add_argument("--run-root", default=None,
                   help=f"output root (default: {_rel(DEFAULT_RUN_ROOT)}/<run-id>)")
    p.add_argument("--record", default=DEFAULT_RECORD,
                   help=f"whole-run record (default: {_rel(DEFAULT_RECORD)})")
    p.add_argument("--arms", default="all",
                   help="comma-separated subset of " + ",".join(ARM_ORDER)
                        + " or 'all' (default)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--python", default="python")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and commands; execute nothing, write nothing")
    p.add_argument("--quiet", action="store_true",
                   help="do not echo subprocess output to stdout (still logged)")
    return p


def select_specs(value: str) -> List[ArmSpec]:
    if value == "all":
        return list(ARM_SPECS)
    tags = [t.strip() for t in value.split(",") if t.strip()]
    if not tags:
        raise DriverError("no arms selected; pass --arms all or a subset of "
                          + ",".join(ARM_ORDER))
    unknown = [t for t in tags if t not in ARM_BY_TAG]
    if unknown:
        raise DriverError(f"unknown arm(s): {unknown}; choose from {ARM_ORDER}")
    return [ARM_BY_TAG[t] for t in tags]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        specs = select_specs(args.arms)
    except DriverError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    run_id = args.run_id or _dt.datetime.now(_dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    run_root = args.run_root or os.path.join(DEFAULT_RUN_ROOT, run_id)
    run_root = _abs(run_root)
    record_path = _abs(args.record)

    if args.dry_run:
        plan = dry_run_plan(specs, run_root, args.python, args.device)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    try:
        record = execute_run(specs, run_root, record_path,
                             python=args.python, device=args.device,
                             echo=not args.quiet)
    except DriverError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR: filesystem error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:                    # M2: record is already closed
        print(f"ERROR: arm execution aborted: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    print(f"\nEXP-L0 run {record['run_id']}: {record['status']}")
    print(f"arms completed: {record.get('arms_completed', [])}")
    if record.get("arms_not_started"):
        print(f"arms not started: {record['arms_not_started']}")
    print(f"record: {_rel(record_path)}")
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
