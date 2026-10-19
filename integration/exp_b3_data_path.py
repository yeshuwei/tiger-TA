#!/usr/bin/env python
"""EXP-B3 -- historical checkpoints store a relative ``args.data_path``.

Run:  python -I integration/exp_b3_data_path.py

The gate (plan.md 19.11.4) registered B3: the frozen generator reads
``args.data_path`` verbatim and runs with ``cwd=rqvae/``, so a checkpoint that
recorded ``data/Toys/...`` looks for ``rqvae/data/Toys/...`` and dies. Two
relative conventions exist in this repo (measured over 708 RQ-VAE checkpoints):
``data/...`` (627, repo-root relative) and ``../data/...`` (61, rqvae/
relative); 20 are already absolute and must keep their meaning.

This harness asks what the fix has to answer:

  [1] the resolver: absolute paths untouched, both historical conventions
      mapped to the fixed roots, missing/ambiguous paths are errors, and the
      answer does not depend on the caller's cwd
  [2] real historical checkpoints resolve to real embedding tables
  [3] the bug has teeth: the frozen generator really fails on the raw
      (unresolved) checkpoint
  [4] a historical relative checkpoint now produces a valid SID end to end
  [5] an absolute checkpoint is passed through untouched, and the two forms
      generate the same SID (the rewrite changes no weights)
  [6] a freshly trained (absolute-path) checkpoint from ``qspace_train`` works,
      from any caller cwd
  [7] nothing writes a ``rqvae/data`` symlink, and the frozen E8 fixture
      resolves to exactly the path it recorded

Nothing here launches EXP-L0 / EXP-B1; the only training is one 1-epoch
tokenizer on 8192 rows (a few seconds on CPU).
"""
import glob
import os
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from tiger_ext import sid_refresh as SR                                # noqa: E402
import tiger_ext.qspace_train as QT                                   # noqa: E402

RESULTS = []

DEFAULT_768D = os.path.join(REPO, "data", "Toys", "item_emb_768d_contentonly.parquet")
# a real historical checkpoint that recorded the repo-relative form
HIST_REL_CKPT = os.path.join(REPO, "rqvae", "ckpt", "Toys_TA_SID_seed2025",
                             "Sep-03-2026_22-07-02", "best_collision_model.pth")
# E8's frozen fixture (absolute path, known-good SID)
E8_CKPT = os.path.join(REPO, "clean_tiger", "ckpt", "Toys", "rqvae",
                       "Oct-14-2026_05-55-09", "best_collision_model.pth")
SYMLINK = os.path.join(REPO, "rqvae", "data")


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def section(title):
    print(f"\n[{title}]")


def raises(fn, exc):
    try:
        fn()
    except exc as e:
        return True, str(e)
    except BaseException as e:                                   # noqa: BLE001
        return False, f"wrong exception {type(e).__name__}: {e}"
    return False, "no exception"


# --------------------------------------------------------------------------- #
# [1] the resolver
# --------------------------------------------------------------------------- #
def part1_resolver(root):
    section("1 the path resolver never trusts the caller's cwd")

    # a synthetic repo with both historical layouts on disk
    repo = os.path.join(root, "repo")
    os.makedirs(os.path.join(repo, "data", "D"), exist_ok=True)
    os.makedirs(os.path.join(repo, "rqvae", "data", "D"), exist_ok=True)
    repo_rel = os.path.join(repo, "data", "D", "repo_rel.parquet")
    rqvae_rel = os.path.join(repo, "data", "D", "rqvae_rel.parquet")
    open(repo_rel, "w").close()
    open(rqvae_rel, "w").close()

    got = SR.resolve_data_path("data/D/repo_rel.parquet", repo=repo)
    check("a `data/...` checkpoint path resolves against the repository root",
          got == os.path.normpath(repo_rel), got)

    # `../data/...` written by rqvae/main.py is rqvae/-relative: <repo>/rqvae/../data
    got = SR.resolve_data_path("../data/D/rqvae_rel.parquet", repo=repo)
    check("a `../data/...` checkpoint path resolves against rqvae/",
          got == os.path.normpath(rqvae_rel), got)

    abs_in = os.path.join(repo, "data", "D", "missing_but_absolute.parquet")
    got = SR.resolve_data_path(abs_in + "/../" + "missing_but_absolute.parquet",
                               repo=repo)
    check("an absolute path is returned normalised and unchanged",
          got == os.path.normpath(abs_in), got)

    ok, msg = raises(lambda: SR.resolve_data_path("data/D/nope.parquet", repo=repo),
                     SR.SidRefreshError)
    check("a relative path that exists nowhere is an error, not a guess",
          ok and "refusing to guess" in msg, msg[:110])

    ok, msg = raises(lambda: SR.resolve_data_path(None), SR.SidRefreshError)
    check("a checkpoint with no data_path is an error", ok, msg[:80])

    # ambiguity: the same relative name exists under both roots, with different
    # content -> refuse rather than pick one
    amb = os.path.join(repo, "data", "D", "amb.parquet")
    os.makedirs(os.path.dirname(amb), exist_ok=True)
    with open(amb, "w") as f:
        f.write("repo")
    open(os.path.join(repo, "rqvae", "data", "D", "amb.parquet"), "w").close()
    ok, msg = raises(lambda: SR.resolve_data_path("data/D/amb.parquet", repo=repo),
                     SR.SidRefreshError)
    check("an ambiguous relative path (both roots, different files) is refused",
          ok and "ambiguous" in msg, msg[:110])

    # cwd independence, on the real repository
    rel = "data/Toys/item_emb_768d_contentonly.parquet"
    before = SR.resolve_data_path(rel)
    cwd = os.getcwd()
    try:
        os.chdir(root)
        after = SR.resolve_data_path(rel)
        ninfo = SR.n_items_of(HIST_REL_CKPT)
    finally:
        os.chdir(cwd)
    check("resolving from a foreign cwd gives the same absolute table",
          before == after == os.path.normpath(DEFAULT_768D), after)
    hist_table = SR.resolve_data_path(SR.ckpt_data_path(HIST_REL_CKPT)[0])
    check("n_items_of is cwd-independent too (returned table + row count)",
          ninfo[1] == hist_table and ninfo[0] > 0,
          f"{ninfo[0]} rows -> {os.path.basename(ninfo[1])}")


# --------------------------------------------------------------------------- #
# [2] real historical checkpoints
# --------------------------------------------------------------------------- #
def part2_real_checkpoints():
    section("2 every stored convention is handled over the whole corpus")

    import torch
    paths = sorted(glob.glob(os.path.join(REPO, "rqvae", "ckpt", "**", "*.pth"),
                             recursive=True))
    buckets = {"absolute": [], "data/": [], "../data/": []}
    for p in paths:
        try:
            dp = torch.load(p, map_location="cpu", weights_only=False)["args"].data_path
        except Exception:                                        # noqa: BLE001
            continue
        key = ("absolute" if os.path.isabs(dp) else
               "../data/" if str(dp).startswith("../") else
               "data/" if str(dp).startswith("data/") else None)
        if key:
            buckets[key].append((p, dp))

    check(f"the corpus still contains all three conventions ({len(paths)} checkpoints)",
          all(buckets[k] for k in buckets),
          ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))

    refused = []
    for key, items in buckets.items():
        good, bad = 0, []
        for p, dp in items:
            try:
                resolved = SR.resolve_data_path(dp)
            except SR.SidRefreshError as e:
                # a loud refusal is a valid outcome (the table may be retired);
                # it is audited separately below
                refused.append((p, dp, str(e)))
                if "refusing to guess" not in str(e) and "ambiguous" not in str(e):
                    bad.append((p, dp, str(e)))
                continue
            if not os.path.isfile(resolved):
                bad.append((p, dp, f"resolved to a missing file: {resolved}"))
            elif key == "absolute" and resolved != os.path.normpath(dp):
                bad.append((p, dp, "an absolute path was changed"))
            else:
                good += 1
        check(f"every {key} checkpoint resolves to a real table or refuses loudly "
              f"({good}/{len(items)})",
              not bad,
              bad[0][2][:110] if bad else
              f"all ok; {len(items) - good} refused (no silent fallback)")

    tables = sorted({dp for _, dp, _ in refused})
    check("the only unresolvable ones point at the retired `data/Sports` table",
          all(dp.startswith("../data/Sports/") and "refusing to guess" in msg
              for _, dp, msg in refused),
          f"{len(refused)} ckpt(s); tables: {', '.join(tables)[:110] or 'none'}")


# --------------------------------------------------------------------------- #
# [3] the bug has teeth
# --------------------------------------------------------------------------- #
def part3_teeth(root):
    section("3 the frozen generator really fails on the raw checkpoint")

    stored, resolved = SR.ckpt_data_path(HIST_REL_CKPT)
    check("the historical checkpoint stores a relative data_path",
          not os.path.isabs(stored), stored)

    out = os.path.join(root, "raw.npy")
    env = dict(os.environ)
    env["PYTHONPATH"] = SR.RVQAE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "generate_code.py", "--dataset", "Toys",
         "--ckpt_path", os.path.abspath(HIST_REL_CKPT),
         "--output_file", out, "--device", "cpu"],
        cwd=SR.RVQAE_DIR, env=env, capture_output=True, text=True, timeout=600)
    wrong = os.path.join(SR.RVQAE_DIR, stored)
    check("unresolved, the generator looks under rqvae/ and FileNotFoundErrors",
          proc.returncode != 0 and "FileNotFoundError" in proc.stderr
          and stored in proc.stderr,
          f"rc={proc.returncode} opened {stored!r} from cwd=rqvae/ "
          f"(= {os.path.normpath(wrong)})")
    check("the resolver points at the file that actually exists",
          os.path.isfile(resolved) and resolved != os.path.normpath(wrong), resolved)


# --------------------------------------------------------------------------- #
# [4] end to end from the historical relative checkpoint
# --------------------------------------------------------------------------- #
def part4_historical_end_to_end(root):
    section("4 a historical relative checkpoint produces a valid SID")

    base = os.path.join(root, "hist_base")
    os.makedirs(base, exist_ok=True)
    t0 = time.time()
    rec = SR.refresh_sid("Toys", HIST_REL_CKPT, base, device="cpu", timeout=1800)
    dt = time.time() - t0

    check("refresh_sid succeeds without any symlink",
          rec["sid_stats"]["ok"] and rec["n_items"] == 11924,
          f"{rec['n_items']} rows, collision {rec['sid_stats']['raw_duplicate_rate']:.6f}, "
          f"{dt:.1f}s")
    return rec


# --------------------------------------------------------------------------- #
# [5] absolute checkpoint: untouched, and the same SID
# --------------------------------------------------------------------------- #
def part5_absolute_twin(root, rec_rel):
    section("5 an absolute checkpoint is passed through untouched")

    work = os.path.join(root, "abs_ckpt")
    os.makedirs(work, exist_ok=True)
    abs_ckpt, stored, resolved, rewritten = SR.ckpt_with_absolute_data_path(
        HIST_REL_CKPT, work)
    check("the rewritten copy exists and stores an absolute data_path",
          rewritten and os.path.isfile(abs_ckpt)
          and os.path.isabs(SR.ckpt_data_path(abs_ckpt)[0]),
          os.path.basename(abs_ckpt))

    import torch
    a = torch.load(HIST_REL_CKPT, map_location="cpu", weights_only=False)
    b = torch.load(abs_ckpt, map_location="cpu", weights_only=False)
    same = (a["state_dict"].keys() == b["state_dict"].keys()
            and all(torch.equal(a["state_dict"][k], b["state_dict"][k])
                    for k in a["state_dict"]))
    check("the copy changes no weight (state_dict is bit-identical)", same,
          f"{len(a['state_dict'])} tensors")

    base = os.path.join(root, "abs_base")
    os.makedirs(base, exist_ok=True)
    rec_abs = SR.refresh_sid("Toys", abs_ckpt, base, device="cpu", timeout=1800)
    check("an absolute checkpoint is not rewritten and is passed through",
          rec_abs["data_path_rewritten"] is False
          and os.path.samefile(rec_abs["generator_ckpt"], os.path.abspath(abs_ckpt)),
          f"generator_ckpt={os.path.basename(rec_abs['generator_ckpt'])}")

    same_sid = np.array_equal(SR.load_sid(rec_rel["sid_path"]),
                              SR.load_sid(rec_abs["sid_path"]))
    check("both forms generate the same SID (the rewrite is semantics-free)",
          same_sid, "bitwise equal" if same_sid else "DIFFERENT")
    return rec_abs


# --------------------------------------------------------------------------- #
# [6] a freshly trained, absolute-path checkpoint
# --------------------------------------------------------------------------- #
def part6_new_checkpoint(root):
    section("6 a qspace_train checkpoint (absolute path) works from any cwd")

    ckpt_dir = os.path.join(root, "new_ckpt")
    cfg = QT.CSGTrainerConfig(dataset="Toys", data_root=os.path.join(REPO, "data"),
                              embedding_path=DEFAULT_768D, ckpt_dir=ckpt_dir,
                              device="cpu", epochs=1, eval_step=1, batch_size=1024,
                              num_workers=0, limit=8192, lambda_csg=0.0)
    report = QT.train_tokenizer(cfg, hook=QT.CSGHook(lambda_csg=0.0))
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "**", "*.pth"), recursive=True))
    check("qspace_train wrote a checkpoint", bool(ckpts) and report["in_dim"] == 768,
          f"{len(ckpts)} file(s), in_dim={report['in_dim']}")
    if not ckpts:
        return

    new_ckpt = ckpts[-1]
    stored, resolved = SR.ckpt_data_path(new_ckpt)
    check("the new checkpoint stores an absolute data_path and needs no rewrite",
          os.path.isabs(stored) and stored == resolved, stored)

    cwd = os.getcwd()
    try:
        os.chdir(root)                       # a caller cwd far from the repo root
        out = os.path.join(root, "new_sid.npy")
        gen = SR.run_generator(new_ckpt, out, "Toys", "cpu", timeout=1800)
        n_items, table = SR.n_items_of(new_ckpt)
    finally:
        os.chdir(cwd)
    check("run_generator works from a foreign caller cwd",
          gen["ok"] and os.path.isfile(out), f"rc={gen['returncode']}")
    check("... without rewriting or copying the checkpoint",
          gen["data_path_rewritten"] is False
          and os.path.samefile(gen["ckpt_for_generator"], os.path.abspath(new_ckpt)),
          os.path.basename(gen["ckpt_for_generator"]))
    check("the generated SID covers every item in the table",
          np.load(out).shape[0] == n_items == 11924, f"{np.load(out).shape}")


# --------------------------------------------------------------------------- #
# [7] no symlink, and the frozen fixture is untouched
# --------------------------------------------------------------------------- #
def part7_hygiene():
    section("7 hygiene: no symlink, frozen fixture unchanged")

    check("no rqvae/data symlink was created", not os.path.lexists(SYMLINK), SYMLINK)
    stored, resolved = SR.ckpt_data_path(E8_CKPT)
    check("E8's absolute fixture resolves to exactly what it recorded",
          stored == resolved and os.path.isfile(resolved),
          os.path.basename(resolved))
    check("the repository still has no rqvae/data directory",
          not os.path.isdir(os.path.join(REPO, "rqvae", "data")), "absent")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    root = tempfile.mkdtemp(prefix="exp_b3_")
    print(f"EXP-B3 workdir: {root}")
    try:
        part1_resolver(root)
        part2_real_checkpoints()
        part3_teeth(root)
        rec_rel = part4_historical_end_to_end(root)
        hist_table = os.path.join(REPO, "data", "Toys", "fused_item_emb_768d.parquet")
        check("the historical run records resolved/stored data_path + rewrite flag",
              rec_rel["data_path"] == os.path.normpath(hist_table)
              and rec_rel["ckpt_data_path_stored"] == "data/Toys/fused_item_emb_768d.parquet"
              and rec_rel["data_path_rewritten"] is True
              and os.path.samefile(rec_rel["generator_ckpt"], HIST_REL_CKPT)
              and rec_rel["manifest"].endswith(".sid.json"),
              f"stored={rec_rel['ckpt_data_path_stored']} rewritten="
              f"{rec_rel['data_path_rewritten']}")
        part5_absolute_twin(root, rec_rel)
        part6_new_checkpoint(root)
        part7_hygiene()
    except BaseException:                                        # noqa: BLE001
        traceback.print_exc()
        RESULTS.append(("harness completed", False, "crashed"))
    finally:
        pass

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nEXP-B3: {passed}/{len(RESULTS)} checks passed")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} -- {detail}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
