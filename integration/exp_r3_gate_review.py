#!/usr/bin/env python
"""Independent gate review for EXP-L0 (R3). Reads the driver as a library,
re-derives the five frozen commands from plan §19.11.13, parses them with the
real CLIs (training entry points spied), checks preconditions and probes the
failure paths. Nothing is trained; no formal SID is refreshed.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shlex
import sys
import tempfile
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

SPEC = importlib.util.spec_from_file_location(
    "exp_l0_driver", os.path.join(REPO, "integration/exp_l0_driver.py"))
drv = importlib.util.module_from_spec(SPEC)
sys.modules["exp_l0_driver"] = drv
SPEC.loader.exec_module(drv)

import tiger_ext.qspace_train as QT
import tiger_ext.fixed_train as FT
from tiger_ext import sid_refresh as SR

R = []


def ck(name, ok, detail=""):
    R.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1. commands derived straight from plan §19.11.13 text
# --------------------------------------------------------------------------- #
PLAN_CSG = (
    "python -m tiger_ext.qspace_train --dataset Toys --data-root data "
    "--artifact-dir integration/artifacts/collab_relation --k 20 "
    "--lambda-csg {lam} --epochs 3000 --batch-size 1024 --eval-step 50 "
    "--num-workers 4 --warmup-epochs 50 "
    "--embedding-path data/Toys/item_emb_768d_contentonly.parquet "
    "--device cuda:0 --seed 2024 --ckpt-dir {ckpt}"
)
PLAN_REC = (
    "python -m tiger_ext.fixed_train --dataset Toys --data_root data "
    "--embedding_path data/Toys/item_emb_768d_contentonly.parquet "
    "--sid_path {sid} --sid_manifest {sid}.sid.json "
    "--tokenizer_ckpt {ckpt} --arm \"{arm}\" "
    "--lambda_sia {l} --lambda_psa {l} "
    "--seed 2025 --num_epochs 200 --early_stop 10 "
    "--batch_size 256 --infer_size 96 --beam_size 30 --lr 1e-4 "
    "--num_workers 1 --device cuda:0 --run_dir {run}"
)


def section(t):
    print(f"\n== {t} ==")


def part1():
    section("1. driver commands vs plan §19.11.13 (independent re-derivation)")
    for spec in drv.ARM_SPECS:
        paths = drv.arm_paths("/tmp/rr", spec)
        if spec.kind == "csg":
            want = shlex.split(PLAN_CSG.format(lam=spec.lambda_csg_text,
                                               ckpt=paths.ckpt_parent))
            got = drv.tokenizer_command(spec, paths, "python", "cuda:0")
            ck(f"{spec.tag}: tokenizer token-identical to plan, ckpt-dir={paths.ckpt_parent}",
               got == want, str([x for x in zip(got, want) if x[0] != x[1]][:2]))
            sid = "/tmp/rr/sid/sid_versions/x.npy"
            ckp = "/tmp/rr/ckpt/stamp/best_collision_model.pth"
            want = shlex.split(PLAN_REC.format(sid=sid, ckpt=ckp, arm=spec.label,
                                               l="0", run=paths.run_dir))
            got = drv.recommender_command(spec, paths, sid, ckp, "python", "cuda:0")
            ck(f"{spec.tag}: recommender == plan λc template (lambdas 0)", got == want,
               str([x for x in zip(got, want) if x[0] != x[1]][:2]))
        else:
            want = shlex.split(PLAN_REC.format(
                sid=drv.BASELINE_SID, ckpt=drv.BASELINE_CKPT, arm=spec.label,
                l=spec.lambda_sia_text, run=paths.run_dir))
            got = drv.recommender_command(spec, paths, drv.BASELINE_SID,
                                          drv.BASELINE_CKPT, "python", "cuda:0")
            ck(f"{spec.tag}: recommender == plan λe template", got == want,
               str([x for x in zip(got, want) if x[0] != x[1]][:2]))


# --------------------------------------------------------------------------- #
# 2. real parsers accept the driver commands
# --------------------------------------------------------------------------- #
def run_qspace(argv):
    cap = {}

    def fake(config):
        cap["cfg"] = config
        return {"dataset": config.dataset, "lambda_csg": config.lambda_csg,
                "device": config.device, "seed": config.seed,
                "epochs": config.epochs, "batch_size": config.batch_size,
                "warmup_epochs": config.warmup_epochs,
                "embedding_path": config.embedding_path,
                "embedding_path_source": "spy", "limit": config.limit,
                "n_steps": 0, "best_loss": 0, "best_collision_rate": 0,
                "ckpt_dir": config.ckpt_dir, "hook": "spy",
                "seeding": {"seed": config.seed, "torch_initial_seed": 0,
                            "shuffle_generator_seed": 0,
                            "kmeans_random_state_calls": 0}}
    orig = QT.train_tokenizer
    QT.train_tokenizer = fake
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rc = QT.main(argv)
    finally:
        QT.train_tokenizer = orig
    return rc, cap.get("cfg")


def run_fixed(argv):
    cap = {}

    class Spy:
        def __init__(self, *a, **k):
            cap["a"], cap["k"] = a, k

        def run(self):
            return {"arm": cap["k"].get("arm"), "dataset": cap["a"][0],
                    "embedding_path": cap["a"][2], "in_dim": 768,
                    "num_epochs_executed": 0, "best_ndcg@20": 0.0,
                    "best_ckpt": "", "last_ckpt": "", "n_best_saves": 0,
                    "n_test_evals": 0, "tokenizer_unchanged": True,
                    "sid_unchanged": True, "embedding_unchanged": True,
                    "mounts": {"lambda_sia": cap["k"]["lambda_sia"],
                               "lambda_psa": cap["k"]["lambda_psa"]}}
    orig = FT.FixedSidTrainer
    FT.FixedSidTrainer = Spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rc = FT.main(argv)
    finally:
        FT.FixedSidTrainer = orig
    return rc, cap


def part2():
    section("2. real CLI parsers accept the exact driver tokens")
    for spec in drv.ARM_SPECS:
        paths = drv.arm_paths("/tmp/rr", spec)
        if spec.kind == "csg":
            cmd = drv.tokenizer_command(spec, paths, "python", "cuda:0")
            rc, cfg = run_qspace(cmd[3:])
            ok = rc == 0 and cfg is not None and (
                cfg.dataset == "Toys" and cfg.lambda_csg == spec.lambda_csg
                and cfg.epochs == 3000 and cfg.batch_size == 1024
                and cfg.eval_step == 50 and cfg.num_workers == 4
                and cfg.warmup_epochs == 50 and cfg.seed == 2024
                and cfg.device == "cuda:0" and cfg.k == 20
                and cfg.embedding_path.endswith("item_emb_768d_contentonly.parquet")
                and cfg.ckpt_dir == paths.ckpt_parent
                and cfg.artifact_dir.endswith("collab_relation"))
            ck(f"{spec.tag}: qspace_train parses; recipe fields match", ok,
               f"rc={rc} lc={getattr(cfg,'lambda_csg',None)} ep={getattr(cfg,'epochs',None)}")
        sid = "/tmp/rr/sid/sid_versions/x.npy" if spec.kind == "csg" else drv.BASELINE_SID
        ckp = "/tmp/rr/ckpt/stamp/best_collision_model.pth" if spec.kind == "csg" else drv.BASELINE_CKPT
        cmd = drv.recommender_command(spec, paths, sid, ckp, "python", "cuda:0")
        rc, cap = run_fixed(cmd[3:])
        k = cap.get("k", {})
        ok = rc == 0 and k and (
            k.get("lambda_sia") == spec.lambda_sia
            and k.get("lambda_psa") == spec.lambda_psa
            and k.get("seed") == 2025 and k.get("num_epochs") == 200
            and k.get("early_stop") == 10 and k.get("batch_size") == 256
            and k.get("infer_size") == 96 and k.get("beam_size") == 30
            and k.get("lr") == 1e-4 and k.get("num_workers") == 1
            and k.get("device") == "cuda:0"
            and k.get("arm") == spec.label
            and not k.get("allow_unregistered_sid", False)
            and str(k.get("embedding_path", "")).endswith("item_emb_768d_contentonly.parquet")
            and os.path.abspath(k.get("sid_manifest", "")) == os.path.abspath(sid) + ".sid.json"
            and cap["a"][3] == ckp)
        ck(f"{spec.tag}: fixed_train parses; recipe fields match", ok,
           f"rc={rc} sia={k.get('lambda_sia')} psa={k.get('lambda_psa')} "
           f"ep={k.get('num_epochs')} mk={k.get('sid_manifest','')[-40:]}")


# --------------------------------------------------------------------------- #
# 3. preconditions
# --------------------------------------------------------------------------- #
def part3():
    section("3. startup preconditions")
    import pyarrow.parquet as pq
    import numpy as np
    import torch

    emb = os.path.join(REPO, drv.EMBEDDING_PATH)
    pf = pq.ParquetFile(emb)
    width = len(pf.read_row_group(0, columns=["embedding"]).column(0)[0].as_py())
    ck(f"embedding table exists, 768d, {pf.metadata.num_rows} rows",
       width == 768 and pf.metadata.num_rows == 11924, f"width={width}")

    ckp = os.path.join(REPO, drv.BASELINE_CKPT)
    ck(f"baseline tokenizer ckpt exists ({os.path.getsize(ckp)/1e6:.1f} MB)",
       os.path.isfile(ckp))
    sd = torch.load(ckp, map_location="cpu", weights_only=False)["state_dict"]
    w = sd["encoder.mlp_layers.1.weight"].shape
    ck("baseline ckpt encoder in_dim == 768 (matches embedding table)",
       w[1] == 768, str(tuple(w)))

    sid = os.path.join(REPO, drv.BASELINE_SID)
    codes = np.load(sid)
    ck("baseline SID shape (N,4) and N == embedding rows",
       codes.ndim == 2 and codes.shape[1] == 4 and codes.shape[0] == 11924,
       str(codes.shape))
    man = SR.read_sid_manifest(sid)
    try:
        SR.verify_sid_manifest(man, ckpt_path=ckp, sid_path_=sid)
        ok = True
    except Exception as e:
        ok = False
        print("   ", e)
    ck("baseline SID sidecar verifies against the baseline ckpt (B4 contract)", ok)
    ck("baseline SID manifest sid_stats.ok", man.get("sid_stats", {}).get("ok") is True)

    rel = os.path.join(REPO, "integration/artifacts/collab_relation/Toys_k20.json")
    ids = os.path.join(REPO, "integration/artifacts/collab_relation/Toys_k20.item_ids.npy")
    meta = json.load(open(rel))
    ck("relation artifact present (Toys, k=20, 11924 item ids)",
       os.path.isfile(rel) and os.path.isfile(ids)
       and len(np.load(ids)) == 11924,
       f"meta.k={meta.get('k')} dataset={meta.get('dataset')}")

    ck("SID generator present", os.path.isfile(SR.GENERATOR), SR.GENERATOR)
    ck("torch + CUDA available", torch.cuda.is_available(),
       f"{torch.__version__}")
    for p in ("data/Toys/train.parquet", "data/Toys/valid.parquet",
              "data/Toys/test.parquet", "data/Toys/item_mapping.npy",
              "data/Toys/item_emb.parquet"):
        ck(f"data file present: {p}", os.path.isfile(os.path.join(REPO, p)))
    free = __import__("shutil").disk_usage(REPO).free / 1e9
    ck("free disk >= 20 GB", free >= 20, f"{free:.1f} GB")


# --------------------------------------------------------------------------- #
# 4. failure-path probes against the real driver code
# --------------------------------------------------------------------------- #
def part4(tmp):
    section("4. failure-path probes (synthetic; no training)")

    def mkroot(tag):
        return os.path.join(tmp, tag, "runs", "run")

    # (a) dry-run writes nothing
    root = mkroot("dry")
    rec = drv.execute_run(drv.ARM_SPECS, root, os.path.join(tmp, "dry", "r.json"),
                          runner=drv._no_runner, dry_run=True)
    ck("dry-run: status + no directories + no record",
       rec["status"] == "dry-run" and not os.path.exists(root)
       and not os.path.exists(os.path.join(tmp, "dry", "r.json")))

    # (b) tokenizer returns 0 but produces no checkpoint dir
    class NoCkpt:
        calls = []

        def __call__(self, cmd, log, echo=False):
            self.calls.append(cmd)
            os.makedirs(os.path.dirname(log), exist_ok=True)
            open(log, "a").write("x\n")
            return 0, "ok\n"
    spy = NoCkpt()
    root = mkroot("nockpt")
    rec = drv.execute_run([drv.ARM_BY_TAG["lc1e-4"]], root,
                          os.path.join(tmp, "nockpt", "r.json"), runner=spy, echo=False)
    arm = rec["arms"][0]
    ck("rc=0 but no new ckpt dir -> arm fails (does not proceed)",
       rec["status"] == "failed" and arm["failed_stage"] == "tokenizer"
       and "exactly one new checkpoint" in arm["failure_reason"], arm["failure_reason"][:80])
    ck("rc=0 but no new ckpt dir -> no refresh/recommender call",
       len(spy.calls) == 1)

    # (c) recommender rc=0 but no run.json
    class NoRunJson:
        def __init__(self):
            self.calls = []

        def __call__(self, cmd, log, echo=False):
            self.calls.append(cmd)
            os.makedirs(os.path.dirname(log), exist_ok=True)
            if "tiger_ext.fixed_train" in cmd:
                os.makedirs(cmd[cmd.index("--run_dir") + 1], exist_ok=True)
                return 0, "ok\n"
            return 0, "ok\n"
    # reuse the R3 SpyRunner shape is not possible here without importing it;
    # instead fail at recommender and assert the arm is failed + record written
    root = mkroot("norunjson")
    spy = NoRunJson()
    rec = drv.execute_run([drv.ARM_BY_TAG["le0.1"]], root,
                          os.path.join(tmp, "norunjson", "r.json"), runner=spy, echo=False)
    arm = rec["arms"][0]
    ck("recommender rc=0 but missing run.json -> arm fails",
       rec["status"] == "failed" and "run.json missing" in (arm["failure_reason"] or ""),
       (arm.get("failure_reason") or "")[:90])

    # (d) exception raised by a stage -> is a record written?
    class Boom:
        def __call__(self, cmd, log, echo=False):
            raise RuntimeError("boom")
    root = mkroot("boom")
    rec_path = os.path.join(tmp, "boom", "r.json")
    raised = None
    try:
        drv.execute_run([drv.ARM_BY_TAG["le0.1"]], root, rec_path,
                        runner=Boom(), echo=False)
    except Exception as e:
        raised = e
    ck("stage exception propagates out of execute_run", raised is not None, repr(raised))
    ck("stage exception -> whole-run record written?", os.path.isfile(rec_path),
       "NOT WRITTEN" if not os.path.isfile(rec_path) else "written")

    # (e) manifest pointing at a different SID path is rejected
    class TamperSidPath:
        def __init__(self):
            self.n = 0

        def __call__(self, cmd, log, echo=False):
            self.n += 1
            os.makedirs(os.path.dirname(log), exist_ok=True)
            if "tiger_ext.qspace_train" in cmd:
                parent = cmd[cmd.index("--ckpt-dir") + 1]
                ckdir = os.path.join(parent, "Oct-19-2026_12-00-00")
                os.makedirs(ckdir, exist_ok=True)
                open(os.path.join(ckdir, "best_collision_model.pth"), "wb").write(b"ck")
                return 0, ""
            ckpt, base, out = cmd[4], cmd[5], cmd[7]
            vd = os.path.join(base, "sid_versions")
            os.makedirs(vd, exist_ok=True)
            p = os.path.join(vd, f"Toys_sid_v0001_{sha(ckpt)[:8]}.npy")
            open(p, "wb").write(b"sid")
            open(p + ".sid.json", "w").write("{}")
            json.dump({"version": 1, "sid_path": "/somewhere/else.npy",
                       "sid_sha256": sha(p), "ckpt_sha256": sha(ckpt),
                       "sid_stats": {"ok": True}, "n_items": 11924}, open(out, "w"))
            return 0, ""
    root = mkroot("badpath")
    rec = drv.execute_run([drv.ARM_BY_TAG["lc1e-4"]], root,
                          os.path.join(tmp, "badpath", "r.json"),
                          runner=TamperSidPath(), echo=False)
    ck("manifest sid_path mismatch -> refresh stage fails",
       rec["status"] == "failed"
       and "manifest sid_path" in (rec["arms"][0].get("failure_reason") or ""),
       (rec["arms"][0].get("failure_reason") or "")[:80])

    # (f) rerunning into the same root is refused before any subprocess
    class Counter:
        calls = 0

        def __call__(self, *a, **k):
            Counter.calls += 1
            return 0, ""
    root = mkroot("reuse")
    drv.execute_run([drv.ARM_BY_TAG["le0.1"]], root,
                    os.path.join(tmp, "reuse", "r1.json"), runner=Counter(), echo=False)
    before = Counter.calls
    rec = drv.execute_run([drv.ARM_BY_TAG["le0.1"]], root,
                          os.path.join(tmp, "reuse", "r2.json"), runner=Counter(), echo=False)
    ck("rerun same run root refused (preflight, zero new subprocesses)",
       rec["status"] == "failed" and Counter.calls == before,
       rec["arms"][0].get("failure_reason", "")[:70])

    # (g) M1: an empty selection must not be accepted as a (completed) run
    try:
        picked = drv.select_specs("")
        rejected = len(picked) > 0
        detail = f"select_specs('') returned {len(picked)} arms"
    except drv.DriverError:
        rejected, detail = True, "DriverError"
    ck("empty --arms selection is rejected (no zero-arm 'completed' run)",
       rejected, detail)


# --------------------------------------------------------------------------- #
# 5. hashes / git
# --------------------------------------------------------------------------- #
def part5():
    section("5. production files and git state")
    want = {
        "tiger_ext/qspace_train.py": "305c50a7bf530f537da23a578950eb5eef74a7de094e6f77cc05045936b6193c",
        "tiger_ext/fixed_train.py": "389e90854b5f6f13371c00942c82fd5a779011635fa8eaa18bf40c5a3a88ce28",
        "tiger_ext/sid_refresh.py": "983121cb3d9cd17bbb337fc8eb7da7e6e6a26df732f19533c8367c4218272bc3",
        "tiger_ext/alt_pipeline.py": "fc1b8652a72c04cec685192604841a3f8fcd64c8641e3f1c61e635cee3a86677",
    }
    for rel, h in want.items():
        ck(f"unchanged {rel}", sha(os.path.join(REPO, rel)) == h)
    import subprocess
    st = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                        capture_output=True, text=True).stdout
    expect = (" M model/main.py\n M rqvae/models/rq.py\n"
              " M rqvae/models/vq.py\n M rqvae/trainer.py\n"
              "?? integration/\n?? tiger_ext/\n")
    ck("git status matches the pre-task baseline", st == expect, repr(st))
    ck("no EXP-L0 run root exists",
       not os.path.exists(os.path.join(REPO, "integration/artifacts/exp_l0")))


def main():
    tmp = tempfile.mkdtemp(prefix="r3gate_")
    try:
        part1()
        part2()
        part3()
        part4(tmp)
        part5()
    except Exception:
        traceback.print_exc()
        ck("review script completed", False, traceback.format_exc().splitlines()[-1])
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    ok = sum(1 for _, o, _ in R if o)
    print(f"\nREVIEW {ok}/{len(R)}")
    os.makedirs(os.path.join(REPO, "integration/artifacts/r3_gate_review"), exist_ok=True)
    json.dump({"checks": [{"name": n, "ok": o, "detail": d} for n, o, d in R],
               "passed": ok, "total": len(R)},
              open(os.path.join(REPO, "integration/artifacts/r3_gate_review/exp_r3_gate_review.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
