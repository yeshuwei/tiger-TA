#!/usr/bin/env python
"""EXP-R2 -- freeze the five EXP-L0 command lines (3 CSG + 2 ETE).

Run:  python -I integration/exp_r2_exp_l0_cli.py

R2 (plan.md 19.11.9.4) recorded that **both** CLIs have defaults that do not
equal the frozen EXP-L0 recipe, so any omitted flag silently drifts away from
plan.md 19.1 / 19.3.  This harness freezes one complete command line per arm --
three CSG tokenizer runs (lambda_csg 1e-4 / 1e-3 / 1e-2) and two ETE recommender
runs (lambda_sia = lambda_psa = 0.1 / 1.0) -- and verifies, without launching
EXP-L0:

  [1] inventory     five logical commands, correct entry points, no shell
                    metacharacters, no reference to the 896d decoy table
  [2] CSG parse     the real ``qspace_train.main`` parses each command and
                    builds a ``CSGTrainerConfig``; every recipe field is checked
                    against plan.md 19.1, and the resulting ``args`` namespace
                    is checked field by field
  [3] ETE parse     the real ``fixed_train.main`` parses each command; the
                    constructor kwargs are captured with a spy (no model is
                    built) and checked against plan.md 19.3 / 19.11.3
  [4] paths         every path exists from the repo root; the 768d table is
                    really 768d, the baseline SID manifest verifies against the
                    baseline checkpoint, the relation artifacts exist
  [5] defaults      the R2 premise: the parser defaults alone would land on the
                    4096d table / zero lambdas / 3 epochs, and the dashed vs
                    underscored flag spellings are not interchangeable
  [6] scope         no production file changed, no driver added, nothing ran

Nothing here trains: the CSG entry point is monkeypatched at
``QT.train_tokenizer`` and the ETE entry point at ``FT.FixedSidTrainer``.
"""
import contextlib
import io
import json
import os
import shlex
import sys

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import tiger_ext.fixed_train as FT                                  # noqa: E402
import tiger_ext.qspace_train as QT                                 # noqa: E402
from tiger_ext import alt_pipeline as AP                            # noqa: E402
from tiger_ext.collab_relation import RELATION_K                    # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def section(title):
    print(f"\n{title}")


# --------------------------------------------------------------------------- #
# the five frozen commands (runnable from the repo root)
# --------------------------------------------------------------------------- #
CSG_ARMS = ["1e-4", "1e-3", "1e-2"]

CSG_COMMANDS = {
    lam: (
        "python -m tiger_ext.qspace_train "
        "--dataset Toys --data-root data "
        "--artifact-dir integration/artifacts/collab_relation --k 20 "
        f"--lambda-csg {lam} "
        "--epochs 3000 --batch-size 1024 --eval-step 50 "
        "--num-workers 4 --warmup-epochs 50 "
        "--embedding-path data/Toys/item_emb_768d_contentonly.parquet "
        f"--device cuda:0 --seed 2024 --ckpt-dir rqvae/ckpt/EXP_L0_lc{lam}"
    )
    for lam in CSG_ARMS
}

ETE_ARMS = [("le0.1", "0.1", "EXP-L0 λe-0.1"),
            ("le1.0", "1.0", "EXP-L0 λe-1.0")]

ETE_COMMANDS = {
    tag: (
        "python -m tiger_ext.fixed_train "
        "--dataset Toys --data_root data "
        "--embedding_path data/Toys/item_emb_768d_contentonly.parquet "
        "--sid_path data/Toys/Toys_content768d_baseline_sid.npy "
        "--sid_manifest data/Toys/Toys_content768d_baseline_sid.npy.sid.json "
        "--tokenizer_ckpt "
        "rqvae/ckpt/Toys_Content768d_Baseline/Jul-31-2026_21-37-33/"
        "best_collision_model.pth "
        f"--arm \"{label}\" "
        f"--lambda_sia {lam} --lambda_psa {lam} "
        "--seed 2025 --num_epochs 200 --early_stop 10 "
        "--batch_size 256 --infer_size 96 --beam_size 30 --lr 1e-4 "
        "--num_workers 1 --device cuda:0 "
        f"--run_dir integration/artifacts/exp_l0/{tag}/run"
    )
    for tag, lam, label in ETE_ARMS
}

EMBEDDING_768 = "data/Toys/item_emb_768d_contentonly.parquet"
BASELINE_CKPT = ("rqvae/ckpt/Toys_Content768d_Baseline/"
                 "Jul-31-2026_21-37-33/best_collision_model.pth")
BASELINE_SID = "data/Toys/Toys_content768d_baseline_sid.npy"
BASELINE_MANIFEST = BASELINE_SID + ".sid.json"

#: plan.md 19.1 -- the frozen RQ-VAE recipe the CSG arms must reproduce.
FROZEN_RQVAE = {
    "lr": 1e-3, "batch_size": 1024, "learner": "AdamW",
    "lr_scheduler_type": "linear", "weight_decay": 1e-4,
    "dropout_prob": 0.0, "bn": False, "loss_type": "mse",
    "kmeans_init": True, "kmeans_iters": 100,
    "sk_epsilons": [0.0, 0.0, 0.003], "sk_iters": 50,
    "beta": 0.25, "quant_loss_weight": 1.0, "save_limit": 5,
    "num_emb_list": [256, 256, 256], "e_dim": 32,
    "layers": [512, 256, 128, 64],
}
FROZEN_CSG = {"epochs": 3000, "eval_step": 50, "num_workers": 4,
              "warmup_epochs": 50, "seed": 2024, "device": "cuda:0"}
FROZEN_ETE = {"seed": 2025, "num_epochs": 200, "early_stop": 10,
              "batch_size": 256, "infer_size": 96, "beam_size": 30,
              "lr": 1e-4, "num_workers": 1, "device": "cuda:0"}

#: the minimal argv the fixed_train parser needs, for the spelling checks.
BASE_ETE_ARGV = ["--sid_path", BASELINE_SID,
                 "--tokenizer_ckpt", BASELINE_CKPT,
                 "--run_dir", "/tmp/exp_r2_probe"]


def repo(rel):
    return rel if os.path.isabs(rel) else os.path.join(REPO, rel)


def parquet_width(rel):
    table = pq.read_table(repo(rel), columns=["embedding"]).slice(0, 1)
    return len(table.column("embedding")[0].as_py())


# --------------------------------------------------------------------------- #
# [2]/[3] execute the parsers with the training entry points replaced by spies
# --------------------------------------------------------------------------- #
def run_csg_cli(command):
    argv = shlex.split(command)
    assert argv[:3] == ["python", "-m", "tiger_ext.qspace_train"], argv[:3]
    captured = {}

    def fake_train(config):
        captured["config"] = config
        return {
            "dataset": config.dataset, "lambda_csg": config.lambda_csg,
            "device": config.device, "seed": config.seed,
            "epochs": config.epochs, "batch_size": config.batch_size,
            "warmup_epochs": config.warmup_epochs,
            "embedding_path": config.embedding_path,
            "embedding_path_source": "spy", "limit": config.limit,
            "n_steps": 0, "best_loss": float("inf"),
            "best_collision_rate": float("inf"),
            "ckpt_dir": config.ckpt_dir, "hook": "spy",
            "seeding": {"seed": config.seed, "torch_initial_seed": 0,
                        "shuffle_generator_seed": 0,
                        "kmeans_random_state_calls": 0},
        }

    original = QT.train_tokenizer
    QT.train_tokenizer = fake_train
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = QT.main(argv[3:])
    finally:
        QT.train_tokenizer = original
    return rc, captured.get("config"), out.getvalue()


def run_ete_cli(command):
    argv = shlex.split(command)
    assert argv[:3] == ["python", "-m", "tiger_ext.fixed_train"], argv[:3]
    captured = {}

    class SpyTrainer:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def run(self):
            return {
                "arm": captured["kwargs"].get("arm"),
                "dataset": captured["args"][0],
                "embedding_path": captured["args"][2],
                "in_dim": 768, "num_epochs_executed": 0,
                "best_ndcg@20": 0.0, "best_ckpt": "", "last_ckpt": "",
                "n_best_saves": 0, "n_test_evals": 0,
                "tokenizer_unchanged": True, "sid_unchanged": True,
                "embedding_unchanged": True,
                "mounts": {"lambda_sia": captured["kwargs"]["lambda_sia"],
                           "lambda_psa": captured["kwargs"]["lambda_psa"]},
            }

    original = FT.FixedSidTrainer
    FT.FixedSidTrainer = SpyTrainer
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = FT.main(argv[3:])
    finally:
        FT.FixedSidTrainer = original
    return rc, captured, out.getvalue()


# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("EXP-R2 -- freeze the five EXP-L0 CLI commands")
    print("=" * 78)

    # ------------------------------------------------------------------ [1]
    section("[1] command inventory (5 commands, repo-root relative paths)")
    commands = list(CSG_COMMANDS.items()) + list(ETE_COMMANDS.items())
    check("exactly five commands", len(commands) == 5, str(len(commands)))
    check("three CSG commands", len(CSG_COMMANDS) == 3, ",".join(CSG_COMMANDS))
    check("two ETE commands", len(ETE_COMMANDS) == 2, ",".join(ETE_COMMANDS))
    meta = (";", "&&", "||", "|", ">", "<", "`", "$(")
    for tag, command in commands:
        argv = shlex.split(command)
        check(f"{tag}: starts with 'python -m'",
              argv[:2] == ["python", "-m"], " ".join(argv[:3]))
        check(f"{tag}: no shell metacharacters",
              not any(m in command for m in meta))
        check(f"{tag}: no 896d decoy table",
              "fused_item_emb_768d" not in command)
        check(f"{tag}: no absolute repo path",
              not any(tok.startswith("/data/") for tok in argv))

    # ------------------------------------------------------------------ [2]
    section("[2] CSG arms -- real qspace_train parser -> CSGTrainerConfig -> args")
    for lam in CSG_ARMS:
        command = CSG_COMMANDS[lam]
        rc, config, printed = run_csg_cli(command)
        check(f"lc{lam}: main() returns 0", rc == 0, str(rc))
        check(f"lc{lam}: config built", config is not None)
        if config is None:
            continue
        check(f"lc{lam}: dataset Toys", config.dataset == "Toys", config.dataset)
        check(f"lc{lam}: lambda_csg {lam}", abs(config.lambda_csg - float(lam)) < 1e-18,
              f"{config.lambda_csg!r}")
        check(f"lc{lam}: epochs 3000", config.epochs == FROZEN_CSG["epochs"], str(config.epochs))
        check(f"lc{lam}: batch_size 1024", config.batch_size == 1024, str(config.batch_size))
        check(f"lc{lam}: eval_step 50", config.eval_step == 50, str(config.eval_step))
        check(f"lc{lam}: num_workers 4", config.num_workers == 4, str(config.num_workers))
        check(f"lc{lam}: warmup_epochs 50", config.warmup_epochs == 50,
              str(config.warmup_epochs))
        check(f"lc{lam}: seed 2024", config.seed == 2024, str(config.seed))
        check(f"lc{lam}: device cuda:0", config.device == "cuda:0", config.device)
        check(f"lc{lam}: validation on", config.validate is True)
        check(f"lc{lam}: full dataset (limit None)", config.limit is None)
        resolved = config.resolved_embedding_path()
        check(f"lc{lam}: embedding path is the 768d table",
              resolved == os.path.abspath(repo(EMBEDDING_768)), resolved)
        check(f"lc{lam}: embedding table exists and is 768d",
              os.path.isfile(resolved) and parquet_width(EMBEDDING_768) == 768,
              f"width={parquet_width(EMBEDDING_768)}")
        check(f"lc{lam}: validate_embedding_path accepts it",
              config.validate_embedding_path() == resolved)
        check(f"lc{lam}: ckpt parent dir is arm-specific",
              config.ckpt_dir == f"rqvae/ckpt/EXP_L0_lc{lam}", str(config.ckpt_dir))

        ns = config.namespace()
        for key, want in FROZEN_RQVAE.items():
            check(f"lc{lam}: args.{key} == frozen {want!r}",
                  getattr(ns, key) == want, repr(getattr(ns, key)))
        for key, want in FROZEN_CSG.items():
            got = getattr(ns, key)
            check(f"lc{lam}: args.{key} == {want!r}", got == want, repr(got))
        check(f"lc{lam}: args.data_path is the 768d table",
              ns.data_path == os.path.abspath(repo(EMBEDDING_768)), ns.data_path)
        check(f"lc{lam}: args.data_path is not item_emb.parquet",
              not ns.data_path.endswith("item_emb.parquet"), ns.data_path)
        check(f"lc{lam}: args.ckpt_dir is the arm parent",
              ns.ckpt_dir == os.path.abspath(repo(f"rqvae/ckpt/EXP_L0_lc{lam}")),
              ns.ckpt_dir)
        # R1: the auxiliary-loss knobs are absent and fall back to frozen values.
        for key, want in (("tau", 0.1), ("lambda_trans", 0.0),
                          ("lambda_balance", 0.0), ("trans_graph_path", None),
                          ("align_weight", 0.0), ("align_tau", 0.1),
                          ("align_data_path", None)):
            check(f"lc{lam}: getattr(args, {key!r}) == frozen {want!r}",
                  getattr(ns, key, want) == want, repr(getattr(ns, key, want)))

    # ------------------------------------------------------------------ [3]
    section("[3] ETE arms -- real fixed_train parser -> FixedSidTrainer kwargs")
    for tag, lam, label in ETE_ARMS:
        command = ETE_COMMANDS[tag]
        rc, captured, printed = run_ete_cli(command)
        kwargs = captured.get("kwargs", {})
        args = captured.get("args", ())
        check(f"{tag}: main() returns 0", rc == 0, str(rc))
        check(f"{tag}: trainer constructed once", len(args) == 5, str(len(args)))
        if len(args) != 5:
            continue
        check(f"{tag}: dataset Toys", args[0] == "Toys", args[0])
        check(f"{tag}: data_root data", args[1] == "data", args[1])
        check(f"{tag}: sid_path is the frozen baseline SID",
              args[2] == BASELINE_SID, args[2])
        check(f"{tag}: tokenizer is the frozen baseline ckpt",
              args[3] == BASELINE_CKPT, args[3])
        check(f"{tag}: run_dir is arm-specific",
              args[4] == f"integration/artifacts/exp_l0/{tag}/run", args[4])
        check(f"{tag}: embedding_path explicitly the 768d table",
              kwargs.get("embedding_path") == EMBEDDING_768,
              str(kwargs.get("embedding_path")))
        check(f"{tag}: embedding table is 768d",
              parquet_width(EMBEDDING_768) == 768)
        check(f"{tag}: sid_manifest is the registered sidecar",
              kwargs.get("sid_manifest") == BASELINE_MANIFEST,
              str(kwargs.get("sid_manifest")))
        check(f"{tag}: allow_unregistered_sid is False",
              kwargs.get("allow_unregistered_sid") is False)
        check(f"{tag}: lambda_sia == {lam}",
              abs(kwargs.get("lambda_sia", -1) - float(lam)) < 1e-18,
              str(kwargs.get("lambda_sia")))
        check(f"{tag}: lambda_psa == {lam}",
              abs(kwargs.get("lambda_psa", -1) - float(lam)) < 1e-18,
              str(kwargs.get("lambda_psa")))
        check(f"{tag}: lambda_sia == lambda_psa",
              kwargs.get("lambda_sia") == kwargs.get("lambda_psa"))
        for key, want in FROZEN_ETE.items():
            got = kwargs.get(key)
            check(f"{tag}: {key} == {want!r}", got == want, repr(got))
        check(f"{tag}: arm label", kwargs.get("arm") == label, str(kwargs.get("arm")))
        check(f"{tag}: topk_list left at the frozen default (5,10,20)",
              kwargs.get("topk_list") is None)
        check(f"{tag}: rqvae_config left at the frozen default",
              kwargs.get("rqvae_config") is None)

    # ------------------------------------------------------------------ [4]
    section("[4] paths and widths")
    widths = {
        EMBEDDING_768: parquet_width(EMBEDDING_768),
        "data/Toys/fused_item_emb_768d.parquet":
            parquet_width("data/Toys/fused_item_emb_768d.parquet"),
        "data/Toys/item_emb.parquet": parquet_width("data/Toys/item_emb.parquet"),
    }
    check("content-only table is 768d", widths[EMBEDDING_768] == 768,
          str(widths[EMBEDDING_768]))
    check("fused_item_emb_768d is 896d (decoy, never used)",
          widths["data/Toys/fused_item_emb_768d.parquet"] == 896,
          str(widths["data/Toys/fused_item_emb_768d.parquet"]))
    check("default item_emb.parquet is 4096d (the R2 hazard)",
          widths["data/Toys/item_emb.parquet"] == 4096,
          str(widths["data/Toys/item_emb.parquet"]))
    for rel in (EMBEDDING_768, BASELINE_SID, BASELINE_MANIFEST, BASELINE_CKPT,
                "data/Toys/train.parquet", "data/Toys/valid.parquet",
                "data/Toys/test.parquet",
                "integration/artifacts/collab_relation/Toys_k20.pairs.npy",
                "integration/artifacts/collab_relation/Toys_k20.item_ids.npy",
                "integration/artifacts/collab_relation/Toys_k20.neighbors.npy",
                "integration/artifacts/collab_relation/Toys_k20.json"):
        check(f"exists: {rel}", os.path.exists(repo(rel)))
    check("default k is the k20 relation", RELATION_K == 20, str(RELATION_K))

    import torch
    ckpt_object = torch.load(repo(BASELINE_CKPT), map_location="cpu",
                             weights_only=False)
    dim = FT.tokenizer_input_dim(ckpt_object)
    check("baseline tokenizer in_dim == 768", dim == 768, str(dim))
    check("baseline ckpt's stored data_path is the 768d table",
          ckpt_object["args"].data_path.endswith(
              "data/Toys/item_emb_768d_contentonly.parquet"),
          ckpt_object["args"].data_path)
    provenance = FT.verify_sid_provenance(repo(BASELINE_SID), repo(BASELINE_CKPT),
                                          manifest_path=repo(BASELINE_MANIFEST),
                                          dataset="Toys")
    check("baseline SID manifest verifies against the baseline ckpt",
          provenance["verified"] is True, provenance["mode"])
    check("baseline SID manifest mode is 'manifest' (registered)",
          provenance["mode"] == "manifest", provenance["mode"])
    check("baseline SID covers 11924 items", provenance["n_items"] == 11924,
          str(provenance["n_items"]))
    check("no output directory has been created yet",
          not os.path.exists(repo("integration/artifacts/exp_l0"))
          and not os.path.exists(repo("rqvae/ckpt/EXP_L0_lc1e-4")))

    # ------------------------------------------------------------------ [5]
    section("[5] R2 premise -- defaults are not the EXP-L0 recipe")
    qdefaults = QT.build_parser().parse_args([])
    check("qspace default epochs=3 (recipe 3000)", qdefaults.epochs == 3)
    check("qspace default batch-size=256 (recipe 1024)", qdefaults.batch_size == 256)
    check("qspace default eval-step=1 (recipe 50)", qdefaults.eval_step == 1)
    check("qspace default warmup-epochs=0 (recipe 50)", qdefaults.warmup_epochs == 0)
    check("qspace default num-workers=1 (recipe 4)", qdefaults.num_workers == 1)
    check("qspace default device=cpu (recipe cuda:0)", qdefaults.device == "cpu")
    check("qspace default embedding-path=None", qdefaults.embedding_path is None)
    check("qspace default ckpt-dir=None", qdefaults.ckpt_dir is None)
    default_cfg = QT.CSGTrainerConfig()
    check("qspace default embedding resolves to the 4096d item_emb.parquet",
          default_cfg.resolved_embedding_path()
          == os.path.abspath(repo("data/Toys/item_emb.parquet")),
          default_cfg.resolved_embedding_path())

    # fixed_train builds its parser inside main(); recover the defaults by
    # parsing an argv that supplies only the required flags.
    rc, captured, _ = run_ete_cli(
        "python -m tiger_ext.fixed_train --sid_path data/Toys/"
        "Toys_content768d_baseline_sid.npy --tokenizer_ckpt "
        "rqvae/ckpt/Toys_Content768d_Baseline/Jul-31-2026_21-37-33/"
        "best_collision_model.pth --run_dir /tmp/exp_r2_probe")
    dkw = captured.get("kwargs", {})
    check("fixed default lambda_sia=0 (recipe 0.1/1.0)", dkw.get("lambda_sia") == 0.0)
    check("fixed default lambda_psa=0 (recipe 0.1/1.0)", dkw.get("lambda_psa") == 0.0)
    check("fixed default embedding_path=None (resolves to 4096d)",
          dkw.get("embedding_path") is None)
    check("fixed default allow_unregistered_sid=False", dkw.get("allow_unregistered_sid") is False)

    # flag spellings: argparse does not alias '-' and '_'
    def rejects(parser, argv):
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                parser.parse_args(argv)
            except SystemExit as exc:
                return exc.code == 2
        return False

    qsp = QT.build_parser()
    check("qspace_train rejects the underscored --embedding_path",
          rejects(qsp, ["--embedding_path", "x"]))
    check("qspace_train accepts the dashed --embedding-path",
          qsp.parse_args(["--embedding-path", "x"]).embedding_path == "x")
    check("fixed_train rejects the dashed --embedding-path (parser spelling is _)",
          _fixed_main_rejects(BASE_ETE_ARGV + ["--embedding-path", "x"]))
    check("fixed_train accepts the underscored --embedding_path",
          _fixed_main_captures(BASE_ETE_ARGV + ["--embedding_path", "x"])
          .get("kwargs", {}).get("embedding_path") == "x")

    # ------------------------------------------------------------------ [6]
    section("[6] scope -- nothing ran, no production file changed, no driver")
    import hashlib

    def sha16(rel):
        with open(repo(rel), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]

    expected = {"tiger_ext/qspace_train.py": "305c50a7bf530f53",
                "tiger_ext/fixed_train.py": "389e90854b5f6f13",
                "tiger_ext/sid_refresh.py": "983121cb3d9cd17b",
                "tiger_ext/alt_pipeline.py": "fc1b8652a72c04ce"}
    for rel, want in expected.items():
        got = sha16(rel)
        check(f"unchanged: {rel}", got == want, got)
    import subprocess
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout
    want_status = (" M model/main.py\n M rqvae/models/rq.py\n"
                   " M rqvae/models/vq.py\n M rqvae/trainer.py\n"
                   "?? integration/\n?? tiger_ext/\n")
    check("git status unchanged", status == want_status, repr(status))
    drivers = [p for p in os.listdir(repo("integration"))
               if p.endswith(".sh") and "l0" in p.lower()]
    check("no EXP-L0 driver added (R3 still open)", not drivers, str(drivers))
    check("no ckpt was written by this harness",
          not os.path.exists(repo("rqvae/ckpt/EXP_L0_lc1e-4")))

    # ------------------------------------------------------------------ [7]
    section("[7] artifact")
    artifact = {
        "suite": "EXP-R2",
        "commands": {"csg": CSG_COMMANDS, "ete": ETE_COMMANDS},
        "frozen_rqvae": FROZEN_RQVAE,
        "frozen_csg": FROZEN_CSG,
        "frozen_ete": FROZEN_ETE,
        "paths": {"embedding_768d": EMBEDDING_768,
                  "baseline_ckpt": BASELINE_CKPT,
                  "baseline_sid": BASELINE_SID,
                  "baseline_manifest": BASELINE_MANIFEST},
        "widths": widths,
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in RESULTS],
    }
    out_dir = repo("integration/artifacts/r2_exp_l0_cli")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp_r2_audit.json")
    with open(out_path, "w") as fh:
        json.dump(artifact, fh, indent=2, ensure_ascii=False)
    check("audit json written", os.path.isfile(out_path), out_path)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("\n" + "=" * 78)
    print(f"EXP-R2: {passed}/{total} checks passed")
    print(f"R2 verdict: {'CLOSED-BY-COMMAND-FREEZE' if passed == total else 'BLOCKED'}")
    print("=" * 78)
    return 0 if passed == total else 1


def _fixed_main_captures(argv):
    """Run the real ``fixed_train.main`` with the trainer replaced by a spy."""
    captured = {}

    class SpyTrainer:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        def run(self):
            return {"arm": captured["kwargs"].get("arm"),
                    "dataset": captured["args"][0],
                    "embedding_path": captured["args"][2], "in_dim": 768,
                    "num_epochs_executed": 0, "best_ndcg@20": 0.0,
                    "best_ckpt": "", "last_ckpt": "", "n_best_saves": 0,
                    "n_test_evals": 0, "tokenizer_unchanged": True,
                    "sid_unchanged": True, "embedding_unchanged": True,
                    "mounts": {"lambda_sia": captured["kwargs"]["lambda_sia"],
                               "lambda_psa": captured["kwargs"]["lambda_psa"]}}

    original = FT.FixedSidTrainer
    FT.FixedSidTrainer = SpyTrainer
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            FT.main(argv)
    finally:
        FT.FixedSidTrainer = original
    return captured


def _fixed_main_rejects(argv):
    """True when the real parser rejects ``argv`` with SystemExit(2)."""
    original = FT.FixedSidTrainer
    FT.FixedSidTrainer = lambda *a, **kw: None            # never reached
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            FT.main(argv)
    except SystemExit as exc:
        return exc.code == 2
    finally:
        FT.FixedSidTrainer = original
    return False


if __name__ == "__main__":
    raise SystemExit(main())
