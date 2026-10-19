#!/usr/bin/env python
"""EXP-P3 -- the Stage 0bis tokenizer entry: embedding table + warm-up + CLI.

Run:  python -I integration/exp_p3_qspace_cli.py

The gate (plan.md 19.11.4) registered three blockers in
``tiger_ext/qspace_train.py``:

  B1  the embedding table was hard-coded to ``data_root/<dataset>/item_emb.parquet``
      (4096d on Toys) with no override -> a 768d arm could not be launched.
  B2  ``warmup_epochs`` existed on the config but had no CLI, so the frozen
      3000-epoch recipe's ``warmup_epochs=50`` was unreachable (default 0).
  B5  the module had no ``if __name__ == "__main__"`` guard, so
      ``python -m tiger_ext.qspace_train`` exited 0 without running anything.

This harness asks the questions the fix has to answer:

  [1] the defaults are what they were -- every parser default, and the default
      embedding path still resolves to the pre-change formula
  [2] an explicit embedding path is used **verbatim** and reaches the dataset
      the trainer actually reads (a decoy ``item_emb.parquet`` of a different
      dimension sits next to it; reading the wrong one is caught by ``in_dim``)
  [3] ``--warmup-epochs 50`` reaches the frozen Trainer's schedule, measured on
      the optimizer's learning rate -- not read off a log line
  [4] a missing table is an error, never a silent switch to another file
  [5] the CLI really runs (B5), and ``import`` does not train anything

Nothing here launches EXP-L0: every training run is <= 1 epoch on <= 64 rows.
"""
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import tiger_ext.qspace_train as QT                                   # noqa: E402
from tiger_ext.alt_pipeline import NUM_WORKERS, PHASE_WARMUP_EPOCHS   # noqa: E402
from tiger_ext.collab_relation import RELATION_K                      # noqa: E402

RESULTS = []

DEFAULT_768D = os.path.join(REPO, "data", "Toys", "item_emb_768d_contentonly.parquet")


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def section(title):
    print(f"\n{title}")


# --------------------------------------------------------------------------- #
# fixture: a synthetic dataset dir holding a decoy table beside a custom one
# --------------------------------------------------------------------------- #
SYN_DATASET = "SynA"
SYN_ROWS = 64
SYN_DIM_DEFAULT = 8        # what data_root/<dataset>/item_emb.parquet carries
SYN_DIM_CUSTOM = 16        # what an explicit --embedding-path must select


def write_emb_table(path, rows, dim, seed):
    rng = np.random.RandomState(seed)
    emb = rng.randn(rows, dim).astype(np.float32)
    pd.DataFrame({"embedding": [row for row in emb]}).to_parquet(path)


def build_synth(root):
    data_dir = os.path.join(root, SYN_DATASET)
    os.makedirs(data_dir, exist_ok=True)
    write_emb_table(os.path.join(data_dir, "item_emb.parquet"),
                    SYN_ROWS, SYN_DIM_DEFAULT, seed=0)
    write_emb_table(os.path.join(data_dir, "custom_emb.parquet"),
                    SYN_ROWS, SYN_DIM_CUSTOM, seed=1)
    os.makedirs(os.path.join(root, "SynEmpty"), exist_ok=True)
    return data_dir


def synth_config(root, *, warmup, embedding_path, tag, epochs=1, batch=8):
    return QT.CSGTrainerConfig(
        dataset=SYN_DATASET, data_root=root, lambda_csg=0.0, device="cpu",
        seed=1234, epochs=epochs, eval_step=epochs, batch_size=batch,
        num_workers=0, warmup_epochs=warmup, embedding_path=embedding_path,
        limit=SYN_ROWS, ckpt_dir=os.path.join(root, f"ckpt_{tag}"))


def run_synth(root, *, warmup, embedding_path, tag, epochs=1, batch=8):
    """One tiny real training run; returns (report, seen, trainer, config)."""
    seen, holder = {}, {}

    def dataset_factory(mods, path, return_index):
        seen["path"] = path
        seen["return_index"] = return_index
        return mods["EmbDataset"](path, return_index=return_index)

    def trainer_factory(args, model, data_num, csg):
        seen["args"] = args
        seen["data_num"] = data_num
        trainer = QT.frozen_tokenizer_modules()["RQVAETrainer"](
            args, model, data_num, csg)
        holder["trainer"] = trainer
        return trainer

    cfg = synth_config(root, warmup=warmup, embedding_path=embedding_path,
                       tag=tag, epochs=epochs, batch=batch)
    # a disabled hook: these runs exercise the *entry*, not L_csg, so no C1/C2
    # artifact is needed for a synthetic dataset name.
    hook = QT.CSGHook(lambda_csg=0.0)
    report = QT.train_tokenizer(cfg, hook=hook, dataset_factory=dataset_factory,
                                trainer_factory=trainer_factory)
    return report, seen, holder["trainer"], cfg


def run_cli(args, *, timeout=900, env_extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    env.update(env_extra or {})
    proc = subprocess.run([sys.executable, "-m", "tiger_ext.qspace_train", *args],
                          cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=timeout)
    return proc


def ckpt_args(path):
    """The ``args`` an RQ-VAE checkpoint carries (the trainer's own record)."""
    from tiger_ext.sid_refresh import ckpt_args as _ckpt_args
    return _ckpt_args(path)


def ckpt_full_args(path):
    """The **unfiltered** ``args`` in a checkpoint (E8's reader keeps a subset)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("args")


def arg_of(record, name, default=None):
    """Read one field from a checkpoint's ``args`` (dict or Namespace)."""
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def find_ckpt(ckpt_dir):
    for base, _dirs, files in os.walk(ckpt_dir):
        for name in files:
            if name.endswith(".pth"):
                return os.path.join(base, name)
    return None


# --------------------------------------------------------------------------- #
# [1] static shape and defaults
# --------------------------------------------------------------------------- #
def part1_defaults():
    section("[1] the entry point and the defaults")

    src_path = os.path.join(REPO, "tiger_ext", "qspace_train.py")
    with open(src_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    last = tree.body[-1]
    guarded = (isinstance(last, ast.If) and "__main__" in ast.unparse(last.test)
               and "SystemExit(main())" in ast.unparse(last))
    check("B5: the module ends with `if __name__ == \"__main__\": raise SystemExit(main())`",
          guarded, f"last statement: {ast.unparse(last)[:60]}")

    imp = subprocess.run(
        [sys.executable, "-c", "import tiger_ext.qspace_train as q; print('imported')"],
        cwd=REPO, capture_output=True, text=True, timeout=300,
        env=dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", "")))
    check("importing the module does not start training",
          imp.returncode == 0 and imp.stdout.strip() == "imported",
          f"rc={imp.returncode} out={imp.stdout.strip()!r}")

    args = QT.build_parser().parse_args([])
    expected = {
        "dataset": "Toys", "lambda_csg": 0.0, "k": RELATION_K,
        "device": "cpu", "seed": 2024, "epochs": 3, "eval_step": 1,
        "batch_size": 256, "num_workers": NUM_WORKERS, "limit": None,
        "ckpt_dir": None, "embedding_path": None,
    }
    mismatched = {k: (getattr(args, k), v) for k, v in expected.items()
                  if getattr(args, k) != v}
    check("parser defaults are the pre-change values", not mismatched, str(mismatched))
    check("--warmup-epochs defaults to PHASE_WARMUP_EPOCHS (0)",
          args.warmup_epochs == PHASE_WARMUP_EPOCHS == 0,
          f"{args.warmup_epochs}")

    cfg = QT.CSGTrainerConfig()
    old_formula = os.path.abspath(os.path.join(REPO, "data", "Toys", "item_emb.parquet"))
    check("the default table still resolves to data_root/<dataset>/item_emb.parquet",
          cfg.resolved_embedding_path() == old_formula
          and cfg.report()["embedding_path_source"] == "default",
          cfg.resolved_embedding_path())
    check("the default warm-up is unchanged (0)",
          cfg.warmup_epochs == 0 and cfg.namespace().warmup_epochs == 0)

    help_out = build_help()
    check("--help exposes --embedding-path and --warmup-epochs",
          "--embedding-path" in help_out and "--warmup-epochs" in help_out,
          f"{len(help_out)} chars")


def build_help():
    from tiger_ext.qspace_train import build_parser
    text = build_parser().format_help()
    return text


# --------------------------------------------------------------------------- #
# [2] explicit table + warm-up, measured where they are consumed
# --------------------------------------------------------------------------- #
def part2_explicit(root):
    section("[2] an explicit embedding table and --warmup-epochs 50")

    custom = os.path.join(root, SYN_DATASET, "custom_emb.parquet")
    rel = os.path.relpath(custom, os.getcwd())
    cfg = synth_config(root, warmup=50, embedding_path=rel, tag="resolve")
    check("an explicit relative path is resolved to an absolute one",
          cfg.resolved_embedding_path() == os.path.abspath(custom)
          and cfg.report()["embedding_path_source"] == "explicit",
          cfg.resolved_embedding_path())

    ns = cfg.namespace()
    check("the namespace the Trainer reads carries the explicit table",
          ns.data_path == os.path.abspath(custom), ns.data_path)
    check("the namespace carries warmup_epochs=50 (and the other knobs unchanged)",
          ns.warmup_epochs == 50 and ns.epochs == 1 and ns.batch_size == 8,
          f"warmup={ns.warmup_epochs} epochs={ns.epochs} batch={ns.batch_size}")

    rep, seen, trainer, _cfg = run_synth(root, warmup=50,
                                         embedding_path=custom, tag="w50")
    check("the dataset was built from the explicit table, not the decoy",
          seen["path"] == os.path.abspath(custom)
          and rep["in_dim"] == SYN_DIM_CUSTOM,
          f"path={seen['path']} in_dim={rep['in_dim']} (decoy dim={SYN_DIM_DEFAULT})")
    check("the report names the source", rep["embedding_path_source"] == "explicit"
          and rep["embedding_path"] == os.path.abspath(custom))

    n_batches = rep["n_batches"]
    check("the frozen Trainer's warm-up budget is 50 epochs worth of steps",
          trainer.warmup_steps == 50 * n_batches
          and trainer.max_steps == 1 * n_batches,
          f"warmup_steps={trainer.warmup_steps} max_steps={trainer.max_steps} "
          f"n_batches={n_batches}")

    lr_w50 = trainer.optimizer.param_groups[0]["lr"]
    expected_lr = (n_batches / trainer.warmup_steps) * float(QT.RQVAE_CONFIG["lr"])
    check("the warm-up reached the optimizer, not just the config",
          abs(lr_w50 - expected_lr) < 1e-9,
          f"final lr={lr_w50:.3e} expected {expected_lr:.3e}")

    _rep0, _seen0, trainer0, _cfg0 = run_synth(root, warmup=0, embedding_path=custom,
                                               tag="w0")
    lr_w0 = trainer0.optimizer.param_groups[0]["lr"]
    check("warmup_epochs=0 keeps the old schedule (linear decay to 0)",
          trainer0.warmup_steps == 0 and lr_w0 == 0.0,
          f"warmup_steps={trainer0.warmup_steps} lr={lr_w0:.3e}")
    check("the two warm-up settings give measurably different schedules",
          lr_w50 > lr_w0, f"{lr_w50:.3e} vs {lr_w0:.3e}")

    ck = find_ckpt(_cfg.ckpt_dir)
    recorded = ckpt_full_args(ck) if ck else None
    check("the checkpoint records both values (provenance)",
          recorded is not None
          and arg_of(recorded, "data_path") == os.path.abspath(custom)
          and arg_of(recorded, "warmup_epochs") == 50,
          f"data_path={arg_of(recorded, 'data_path')} "
          f"warmup={arg_of(recorded, 'warmup_epochs')}")


def part3_default_path(root):
    section("[3] the default layout is still the default")
    rep, seen, _t, _c = run_synth(root, warmup=0, embedding_path=None, tag="default")
    expected = os.path.abspath(os.path.join(root, SYN_DATASET, "item_emb.parquet"))
    check("with no override the run reads data_root/<dataset>/item_emb.parquet",
          seen["path"] == expected and rep["in_dim"] == SYN_DIM_DEFAULT,
          f"path={seen['path']} in_dim={rep['in_dim']}")
    check("the report marks the source as the default",
          rep["embedding_path_source"] == "default", rep["embedding_path"])


# --------------------------------------------------------------------------- #
# [4] a missing table is an error, never a fallback
# --------------------------------------------------------------------------- #
def part4_no_fallback(root):
    section("[4] a missing table is refused, never replaced")

    missing = os.path.join(root, SYN_DATASET, "does_not_exist.parquet")
    cfg = synth_config(root, warmup=0, embedding_path=missing, tag="missing")
    try:
        cfg.validate_embedding_path()
        check("a missing explicit table raises", False, "no exception")
    except QT.QSpaceTrainError as exc:
        text = str(exc)
        check("a missing explicit table raises QSpaceTrainError naming it",
              missing in text and "refusing to fall back" in text, text[:110])

    called = {"n": 0}

    def factory(_mods, _path, _return_index):
        called["n"] += 1
        raise AssertionError("must not be reached")

    try:
        QT.train_tokenizer(cfg, hook=QT.CSGHook(lambda_csg=0.0),
                           dataset_factory=factory)
        check("train_tokenizer refuses before touching any dataset", False,
              "no exception")
    except QT.QSpaceTrainError:
        check("train_tokenizer refuses before touching any dataset",
              called["n"] == 0,
              f"the decoy item_emb.parquet in the same dir was NOT read "
              f"(factory calls={called['n']})")

    dir_as_table = synth_config(root, warmup=0,
                                embedding_path=os.path.join(root, SYN_DATASET),
                                tag="dir")
    try:
        dir_as_table.validate_embedding_path()
        check("a directory is not accepted as an embedding table", False, "no error")
    except QT.QSpaceTrainError:
        check("a directory is not accepted as an embedding table", True)


# --------------------------------------------------------------------------- #
# [5] the CLI really runs
# --------------------------------------------------------------------------- #
def part5_cli(root):
    section("[5] the CLI entry point")

    ckpt_dir = os.path.join(root, "cli_768d")
    proc = run_cli([
        "--dataset", "Toys", "--data-root", os.path.join(REPO, "data"),
        "--embedding-path", DEFAULT_768D,
        "--warmup-epochs", "50", "--epochs", "1", "--eval-step", "1",
        "--batch-size", "8", "--limit", "16", "--num-workers", "0",
        "--lambda-csg", "0", "--device", "cpu", "--ckpt-dir", ckpt_dir,
    ])
    ok = proc.returncode == 0
    check("`python -m tiger_ext.qspace_train` runs the real entry (B5)",
          ok, f"rc={proc.returncode} err={proc.stderr.strip().splitlines()[-1:] }")
    if ok:
        import re as _re
        check("the CLI reports the explicit 768d table and warmup 50",
              _re.search(r"embedding_path_source\s+explicit", proc.stdout) is not None
              and DEFAULT_768D in proc.stdout
              and _re.search(r"warmup_epochs\s+50\b", proc.stdout) is not None,
              "stdout summary")
        ck = find_ckpt(ckpt_dir)
        recorded = ckpt_full_args(ck) if ck else None
        check("the CLI run trained against the 768d table with warm-up 50",
              recorded is not None
              and os.path.samefile(arg_of(recorded, "data_path"), DEFAULT_768D)
              and arg_of(recorded, "warmup_epochs") == 50,
              f"data_path={arg_of(recorded, 'data_path')} "
              f"warmup={arg_of(recorded, 'warmup_epochs')}")
    else:
        check("the CLI reports the explicit 768d table and warmup 50", False, "skipped")
        check("the CLI run trained against the 768d table with warm-up 50",
              False, "skipped")

    ckpt_dir2 = os.path.join(root, "cli_default")
    proc2 = run_cli([
        "--dataset", "Toys", "--data-root", os.path.join(REPO, "data"),
        "--epochs", "1", "--eval-step", "1", "--batch-size", "8",
        "--limit", "16", "--num-workers", "0", "--lambda-csg", "0",
        "--device", "cpu", "--ckpt-dir", ckpt_dir2,
    ])
    ok2 = proc2.returncode == 0
    check("the CLI without --embedding-path still uses the old default table",
          ok2, f"rc={proc2.returncode} err={proc2.stderr.strip().splitlines()[-1:]}")
    if ok2:
        ck2 = find_ckpt(ckpt_dir2)
        recorded2 = ckpt_full_args(ck2) if ck2 else None
        old_default = os.path.join(REPO, "data", "Toys", "item_emb.parquet")
        check("the default-layout CLI run records data_path=<repo>/data/Toys/item_emb.parquet "
              "and warmup 0",
              recorded2 is not None
              and os.path.samefile(arg_of(recorded2, "data_path"), old_default)
              and arg_of(recorded2, "warmup_epochs") == 0,
              f"data_path={arg_of(recorded2, 'data_path')} "
              f"warmup={arg_of(recorded2, 'warmup_epochs')}")
    else:
        check("the default-layout CLI run records data_path=<repo>/data/Toys/"
              "item_emb.parquet and warmup 0", False, "skipped")

    bad_dir = os.path.join(root, "cli_missing")
    missing = os.path.join(root, "nope.parquet")
    proc3 = run_cli([
        "--dataset", "Toys", "--data-root", os.path.join(REPO, "data"),
        "--embedding-path", missing, "--epochs", "1", "--batch-size", "8",
        "--limit", "8", "--num-workers", "0", "--lambda-csg", "0",
        "--device", "cpu", "--ckpt-dir", bad_dir,
    ], timeout=600)
    check("the CLI fails loudly on a missing explicit table",
          proc3.returncode != 0
          and "embedding table is missing" in proc3.stderr
          and "refusing to fall back" in proc3.stderr
          and not os.path.isdir(bad_dir),
          f"rc={proc3.returncode}")

    empty_root = os.path.join(root, "empty_root")
    os.makedirs(os.path.join(empty_root, "NoSuchDataset"), exist_ok=True)
    proc4 = run_cli([
        "--dataset", "NoSuchDataset", "--data-root", empty_root,
        "--epochs", "1", "--batch-size", "8", "--limit", "8",
        "--num-workers", "0", "--lambda-csg", "0", "--device", "cpu",
        "--ckpt-dir", os.path.join(root, "cli_missing_default"),
    ], timeout=600)
    check("the CLI fails loudly when the *default* table is missing too",
          proc4.returncode != 0
          and "embedding table is missing" in proc4.stderr
          and "frozen default layout" in proc4.stderr,
          f"rc={proc4.returncode}")


# --------------------------------------------------------------------------- #
def main():
    print(f"repo  -> {REPO}")
    print(f"768d  -> {DEFAULT_768D} (exists={os.path.isfile(DEFAULT_768D)})")
    root = tempfile.mkdtemp(prefix="p3_qspace_")
    try:
        build_synth(root)
        print(f"fixture -> {root}")
        part1_defaults()
        part2_explicit(root)
        part3_default_path(root)
        part4_no_fallback(root)
        part5_cli(root)
    except Exception:                                    # noqa: BLE001
        traceback.print_exc()
        RESULTS.append(("harness crashed", False, "see traceback"))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\nEXP-P3: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} -- {detail}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
