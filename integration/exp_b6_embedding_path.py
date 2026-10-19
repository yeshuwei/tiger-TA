#!/usr/bin/env python
"""EXP-B6 -- ``FixedSidTrainer`` must be able to read an explicit item table.

Run:  python -I integration/exp_b6_embedding_path.py

The gate (plan.md 19.11.9) found B6: ``FixedSidTrainer.__init__`` hardcoded
``data_root/<dataset>/item_emb.parquet``. On Toys that file is the **4096d**
table, while every R13 tokenizer is 768d, so ``load_state_dict`` failed with
``[512,768] vs [512,4096]`` and all five EXP-L0 arms were blocked.

This suite asks whether the new ``embedding_path`` / ``--embedding_path``:

  [1] keeps the historical default byte-for-byte (no argument -> same path,
      same tensor, same manifest as before)
  [2] actually reads the explicit table (the loaded tensor equals the explicit
      file, not the default one, and the run manifest says which path was used)
  [3] refuses a table whose width differs from the checkpoint's ``in_dim``
      **at construction**, naming both files and both widths
  [4] does not fall back to the default when an explicit table is given (a
      decoy default table of the wrong width is ignored)
  [5] does not bypass B4: a mispaired checkpoint or a manifest-less SID is still
      refused with an explicit table
  [6] the CLI exposes ``--embedding_path`` with the same semantics
  [7] the real blocked case -- the real 768d Toys checkpoint + the real
      ``fused_item_emb_768d.parquet`` -- constructs **without** any shadow data
      root

Nothing here measures recommendation quality. Training is one T5 epoch on 120
synthetic sequences; the real-input check runs zero epochs.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from tiger_ext import sid_refresh as SR                                  # noqa: E402
from tiger_ext.alt_pipeline import frozen_modules, rqvae_namespace       # noqa: E402
from tiger_ext.fixed_train import (                                      # noqa: E402
    SID_MANIFEST_MISSING, FixedSidError, FixedSidTrainer, SidProvenanceError,
    tokenizer_input_dim,
)

DATASET = "B6Syn"
DECOY = "B6Decoy"
DEVICE = os.environ.get("EXP_B6_DEVICE", "cpu")

SYN = dict(n_items=128, dim=32, centres=8, codes=[64, 64, 64], rqvae_batch=128,
           seq_len=6, n_train=120, n_valid=24, n_test=24, seed=0)
RQVAE_CFG = dict(num_emb_list=list(SYN["codes"]), e_dim=32)
T5_CFG = dict(num_layers=2, num_decoder_layers=2, d_model=64, d_ff=256,
              num_heads=4, d_kv=16)
SEED = 2025

# The real 768d baseline pair. NB: ``data/Toys/fused_item_emb_768d.parquet`` is
# **896d** despite its name (see [[toys-rqvae-fixture-dims]]); the table this
# checkpoint actually read is ``item_emb_768d_contentonly.parquet``.
REAL_CKPT = os.path.join(REPO, "rqvae", "ckpt", "Toys_Content768d_Baseline",
                         "Jul-31-2026_21-37-33", "best_collision_model.pth")
REAL_TABLE = os.path.join(REPO, "data", "Toys",
                          "item_emb_768d_contentonly.parquet")

RESULTS = []


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
# fixture
# --------------------------------------------------------------------------- #
def write_table(path, emb):
    pd.DataFrame({"ItemID": np.arange(1, len(emb) + 1, dtype=np.int64),
                  "embedding": [emb[i] for i in range(len(emb))]}).to_parquet(
        path, index=False)


def build_dataset(root, dataset=DATASET):
    rng = np.random.default_rng(SYN["seed"])
    n, dim = SYN["n_items"], SYN["dim"]
    centres = rng.normal(size=(SYN["centres"], dim)).astype(np.float32) * 2.0
    which = rng.integers(0, SYN["centres"], size=n)
    emb = centres[which] + rng.normal(size=(n, dim)).astype(np.float32) * 0.3

    data_dir = os.path.join(root, dataset)
    os.makedirs(data_dir, exist_ok=True)
    write_table(os.path.join(data_dir, "item_emb.parquet"), emb)

    def interactions(rows):
        history, target = [], []
        for _ in range(rows):
            seq = rng.integers(1, n + 1, size=SYN["seq_len"]).tolist()
            history.append(seq[:-1])
            target.append(seq[-1])
        return pd.DataFrame({"history": history, "target": target})

    for split, rows in (("train", SYN["n_train"]), ("valid", SYN["n_valid"]),
                        ("test", SYN["n_test"])):
        interactions(rows).to_parquet(os.path.join(data_dir, f"{split}.parquet"),
                                      index=False)
    return data_dir, emb


def train_tokenizer(root, data_dir, tag="a"):
    """One real RQ-VAE epoch through the frozen Trainer -- a genuine checkpoint."""
    mods = frozen_modules()
    torch = mods["torch"]
    cfg = dict(RQVAE_CFG)
    rqvae = mods["RQVAE"](
        in_dim=SYN["dim"], num_emb_list=cfg["num_emb_list"], e_dim=cfg["e_dim"],
        layers=[512, 256, 128, 64], dropout_prob=0.0, bn=False, loss_type="mse",
        quant_loss_weight=1.0, beta=0.25, kmeans_init=True, kmeans_iters=100,
        sk_epsilons=[0, 0, 0.003], sk_iters=50)
    ckpt_dir = os.path.join(root, f"rqvae_ckpt_{tag}")
    args = rqvae_namespace(os.path.join(data_dir, "item_emb.parquet"), ckpt_dir,
                           DEVICE, SEED, epochs=1, eval_step=1,
                           batch_size=SYN["rqvae_batch"], num_workers=0,
                           config=cfg)
    data = mods["EmbDataset"](os.path.join(data_dir, "item_emb.parquet"))
    loader = torch.utils.data.DataLoader(data, num_workers=0,
                                         batch_size=args.batch_size, shuffle=True)
    trainer = mods["RQVAETrainer"](args, rqvae, len(loader))
    trainer.fit(loader)
    path = os.path.join(trainer.ckpt_dir, "best_collision_model.pth")
    if not os.path.isfile(path):
        raise RuntimeError(f"fixture tokenizer saved no checkpoint at {path}")
    return path


def build_trainer(root, tag, sid_path, ckpt, dataset=DATASET, **kwargs):
    params = dict(device=DEVICE, rqvae_config=RQVAE_CFG, t5_config=T5_CFG,
                  seed=SEED, num_workers=0, batch_size=64, infer_size=64,
                  beam_size=5, num_epochs=1, early_stop=99)
    params.update(kwargs)
    return FixedSidTrainer(dataset, root, sid_path, ckpt,
                           os.path.join(root, "run", tag), **params)


def load_run_json(root, tag):
    with open(os.path.join(root, "run", tag, "run.json")) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# [1] the default is unchanged
# --------------------------------------------------------------------------- #
def part1_default(root, sid_path, ckpt, default_emb):
    section("1 no --embedding_path: the historical default, unchanged")

    tr = build_trainer(root, "default", sid_path, ckpt)
    expected = os.path.join(root, DATASET, "item_emb.parquet")
    check("the default path is data_root/<dataset>/item_emb.parquet",
          tr.embedding_path == os.path.abspath(expected)
          and tr.embedding_path_explicit is False, tr.embedding_path)
    check("the loaded tensor is the default table",
          np.array_equal(tr.item_embeddings.cpu().numpy(), default_emb),
          f"in_dim={tr.in_dim}")
    check("the checkpoint's input width was read and matches",
          tr.tokenizer_input_dim == SYN["dim"]
          and tokenizer_input_dim(tr.tokenizer_ckpt_data) == SYN["dim"],
          f"in_dim={tr.tokenizer_input_dim}")

    res = tr.run()
    cfg = load_run_json(root, "default")["config"]
    check("run.json records the default table, its width and its hash",
          cfg["embedding_path"] == os.path.abspath(expected)
          and cfg["embedding_path_explicit"] is False
          and cfg["embedding_dim"] == SYN["dim"]
          and cfg["embedding_n_items"] == SYN["n_items"]
          and cfg["embedding_sha256"] == SR.sha256_file(expected)
          and cfg["tokenizer_input_dim"] == SYN["dim"],
          f"{os.path.basename(cfg['embedding_path'])} dim={cfg['embedding_dim']}")
    check("the run records the table was unchanged",
          res["embedding_unchanged"] is True
          and res["embedding_sha256_after"] == cfg["embedding_sha256"])


# --------------------------------------------------------------------------- #
# [2] an explicit table is actually read
# --------------------------------------------------------------------------- #
def part2_explicit(root, sid_path, ckpt, alt_path, alt_emb, default_emb):
    section("2 an explicit table is read (not the default) and recorded")

    tr = build_trainer(root, "explicit", sid_path, ckpt, embedding_path=alt_path)
    check("the trainer uses the explicit path as given",
          tr.embedding_path == os.path.abspath(alt_path)
          and tr.embedding_path_explicit is True, tr.embedding_path)
    check("the loaded tensor is the explicit file",
          np.array_equal(tr.item_embeddings.cpu().numpy(), alt_emb),
          f"sha={tr.embedding_sha256[:12]}")
    check("... and it differs from the default table (the fixture is a real switch)",
          not np.array_equal(alt_emb, default_emb))
    check("the explicit file's sha256 is what the trainer recorded",
          tr.embedding_sha256 == SR.sha256_file(alt_path))

    res = tr.run()
    cfg = load_run_json(root, "explicit")["config"]
    check("run.json records the explicit path, width and hash",
          cfg["embedding_path"] == os.path.abspath(alt_path)
          and cfg["embedding_path_explicit"] is True
          and cfg["embedding_dim"] == SYN["dim"]
          and cfg["embedding_sha256"] == SR.sha256_file(alt_path),
          cfg["embedding_path"])
    check("the run still reports the tokenizer/SID frozen",
          res["tokenizer_unchanged"] is True and res["sid_unchanged"] is True)
    check("the stage manifest carries the embedding identity",
          _stage_manifest_has_embedding(res["best_ckpt"], alt_path),
          res["best_ckpt"] or "(no improvement branch reached)")


def _stage_manifest_has_embedding(ckpt_path, alt_path):
    if not ckpt_path:
        return True                      # no save -> nothing to inspect
    side = ckpt_path + ".stage.json"
    if not os.path.isfile(side):
        return False
    with open(side) as fh:
        rec = json.load(fh)
    return (rec.get("embedding_path") == os.path.abspath(alt_path)
            and rec.get("embedding_dim") == SYN["dim"])


# --------------------------------------------------------------------------- #
# [3] a wrong-width table fails before training
# --------------------------------------------------------------------------- #
def part3_dimension_mismatch(root, sid_path, ckpt, wrong_path):
    section("3 a table whose width differs from the checkpoint fails early")

    run_dir = os.path.join(root, "run", "wrong_dim")
    ok, msg = raises(lambda: build_trainer(root, "wrong_dim", sid_path, ckpt,
                                           embedding_path=wrong_path),
                     FixedSidError)
    check("construction refuses the 48d table for a 32d checkpoint",
          ok and "embedding dimension mismatch" in msg, msg[:200])
    check("the error names the table, both widths and the checkpoint",
          ok and os.path.abspath(wrong_path) in msg and "48 columns" in msg
          and "in_dim=32" in msg and os.path.basename(ckpt) in msg)
    check("nothing was trained and no run directory was produced",
          not os.path.exists(run_dir))

    # the same failure via the default path, where the hint about how to fix it
    # is part of the message
    ok, msg = raises(lambda: build_trainer(root, "wrong_default", sid_path, ckpt,
                                           dataset=DECOY),
                     FixedSidError)
    check("a wrong-width *default* table is refused too, with a fix hint",
          ok and "embedding dimension mismatch" in msg
          and "embedding_path=" in msg, msg[:200])


# --------------------------------------------------------------------------- #
# [4] the explicit path wins over a decoy default
# --------------------------------------------------------------------------- #
def part4_no_fallback(root, sid_path, ckpt, correct_path):
    section("4 an explicit table is never replaced by the default")

    tr = build_trainer(root, "decoy_ok", sid_path, ckpt, dataset=DECOY,
                       embedding_path=correct_path)
    check("a decoy 48d default is ignored when a 32d table is given",
          tr.embedding_path == os.path.abspath(correct_path)
          and tr.in_dim == SYN["dim"], f"in_dim={tr.in_dim}")
    check("the dataset name still drives the interaction splits",
          os.path.abspath(tr.data_dir) == os.path.abspath(os.path.join(root, DECOY)))
    check("the dataset-name mismatch in the manifest is surfaced, not fatal",
          tr.provenance["verified"] is True
          and tr.provenance["dataset_matches"] is False,
          f"dataset={tr.provenance['dataset']}")


# --------------------------------------------------------------------------- #
# [5] B4 is unchanged
# --------------------------------------------------------------------------- #
def part5_provenance(root, sid_path, side, ckpt_a, ckpt_b, alt_path):
    section("5 the explicit table does not bypass SID provenance (B4)")

    ok, msg = raises(lambda: build_trainer(root, "bad_ckpt", sid_path, ckpt_b,
                                           embedding_path=alt_path),
                     SidProvenanceError)
    check("a mispaired checkpoint is still refused with an explicit table",
          ok and "does not belong" in msg, msg[:150])
    check("no run directory for the mispaired run",
          not os.path.exists(os.path.join(root, "run", "bad_ckpt")))

    plain_dir = os.path.join(root, "plain")
    os.makedirs(plain_dir, exist_ok=True)
    plain = os.path.join(plain_dir, f"{DATASET}_sid.npy")
    shutil.copyfile(sid_path, plain)
    ok, msg = raises(lambda: build_trainer(root, "plain", plain, ckpt_a,
                                           embedding_path=alt_path),
                     SidProvenanceError)
    check("a manifest-less SID is still refused by default",
          ok and SID_MANIFEST_MISSING in msg and "allow_unregistered_sid" in msg,
          msg[:140])

    tr = build_trainer(root, "plain_ok", plain, ckpt_a,
                       embedding_path=alt_path, allow_unregistered_sid=True)
    check("the explicit opt-out still records mode=unregistered",
          tr.provenance["mode"] == "unregistered"
          and tr.provenance["verified"] is False)


# --------------------------------------------------------------------------- #
# [6] CLI
# --------------------------------------------------------------------------- #
def part6_cli(root, sid_path, ckpt, alt_path, wrong_path):
    section("6 the CLI exposes --embedding_path with the same semantics")

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    rcfg = json.dumps(RQVAE_CFG)

    def run_cli(tag, embedding, extra=()):
        cmd = [sys.executable, "-m", "tiger_ext.fixed_train",
               "--dataset", DATASET, "--data_root", root,
               "--sid_path", sid_path, "--tokenizer_ckpt", ckpt,
               "--run_dir", os.path.join(root, "run", tag),
               "--device", "cpu", "--num_workers", "0",
               "--num_epochs", "1", "--batch_size", "64", "--infer_size", "64",
               "--beam_size", "5", "--rqvae_config", rcfg, *extra]
        if embedding is not None:
            cmd += ["--embedding_path", embedding]
        return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=900)

    proc = run_cli("cli_explicit", alt_path)
    cfg = load_run_json(root, "cli_explicit")["config"] \
        if proc.returncode == 0 else {}
    check("the CLI runs with --embedding_path and records that path",
          proc.returncode == 0
          and cfg.get("embedding_path") == os.path.abspath(alt_path)
          and cfg.get("embedding_path_explicit") is True,
          f"rc={proc.returncode} "
          f"{proc.stderr.strip().splitlines()[-1:] or ''}")

    proc = run_cli("cli_bad_dim", wrong_path)
    check("the CLI fails non-zero on a width mismatch",
          proc.returncode != 0 and "embedding dimension mismatch" in proc.stderr,
          f"rc={proc.returncode}")

    proc = run_cli("cli_default", None)
    cfg = load_run_json(root, "cli_default")["config"] \
        if proc.returncode == 0 else {}
    check("omitting the flag keeps the default path",
          proc.returncode == 0
          and cfg.get("embedding_path") == os.path.abspath(
              os.path.join(root, DATASET, "item_emb.parquet"))
          and cfg.get("embedding_path_explicit") is False,
          f"rc={proc.returncode}")


# --------------------------------------------------------------------------- #
# [7] the case B6 blocked: a real 768d checkpoint, no shadow root
# --------------------------------------------------------------------------- #
def part7_real(root):
    section("7 the real 768d Toys pair constructs without a shadow data root")

    sid_root = os.path.join(root, "real_sid")
    rec = SR.refresh_sid("Toys", REAL_CKPT, sid_root, device="cpu", timeout=1800)
    ckpt_dim = tokenizer_input_dim(
        torch.load(REAL_CKPT, map_location="cpu", weights_only=False))
    check("the real checkpoint is 768d", ckpt_dim == 768, f"in_dim={ckpt_dim}")

    # data_root is the real repo `data/` directory: no symlinks, no shadow root.
    tr = FixedSidTrainer("Toys", os.path.join(REPO, "data"), rec["sid_path"],
                         REAL_CKPT, os.path.join(root, "run", "real"),
                         device="cpu", num_epochs=0,
                         embedding_path=REAL_TABLE)
    check("the 768d table is accepted at construction (B6 fixed)",
          tr.in_dim == 768 and tr.n_items == 11924
          and tr.data_dir == os.path.abspath(os.path.join(REPO, "data", "Toys")),
          f"in_dim={tr.in_dim} n_items={tr.n_items}")
    check("the run records the real table, not the 4096d default",
          tr.embedding_path == os.path.abspath(REAL_TABLE)
          and tr.embedding_path_explicit is True
          and "item_emb.parquet" not in os.path.basename(tr.embedding_path),
          os.path.basename(tr.embedding_path))

    ok, msg = raises(lambda: FixedSidTrainer(
        "Toys", os.path.join(REPO, "data"), rec["sid_path"], REAL_CKPT,
        os.path.join(root, "run", "real_default"), device="cpu", num_epochs=0),
        FixedSidError)
    check("the 4096d default is now refused with a clear message (not a raw "
          "size mismatch)",
          ok and "embedding dimension mismatch" in msg
          and "4096 columns" in msg and "in_dim=768" in msg, msg[:220])


# --------------------------------------------------------------------------- #
def main():
    root = tempfile.mkdtemp(prefix="exp_b6_")
    print(f"EXP-B6 workdir: {root}")
    try:
        data_dir, emb = build_dataset(root)
        decoy_dir = os.path.join(root, DECOY)
        os.makedirs(decoy_dir, exist_ok=True)
        for split in ("train", "valid", "test"):
            shutil.copyfile(os.path.join(data_dir, f"{split}.parquet"),
                            os.path.join(decoy_dir, f"{split}.parquet"))
        wrong48 = np.pad(emb, ((0, 0), (0, 16)), mode="constant")
        write_table(os.path.join(decoy_dir, "item_emb.parquet"), wrong48)

        alt_emb = (emb + np.float32(0.25)).astype(np.float32)
        alt_path = os.path.join(root, "explicit_32d.parquet")
        write_table(alt_path, alt_emb)
        wrong_path = os.path.join(root, "wrong_48d.parquet")
        write_table(wrong_path, wrong48)
        correct_path = os.path.join(data_dir, "item_emb.parquet")

        ckpt_a = train_tokenizer(root, data_dir, tag="a")
        ckpt_b = train_tokenizer(root, data_dir, tag="b")
        rec = SR.refresh_sid(DATASET, ckpt_a, root, device="cpu", timeout=900)
        sid_path = rec["sid_path"]
        side = SR.sidecar_path(sid_path)
        check("the fixture SID was registered by refresh_sid",
              os.path.isfile(side) and rec["n_items"] == SYN["n_items"],
              f"{os.path.basename(sid_path)} ({rec['n_items']} rows)")

        part1_default(root, sid_path, ckpt_a, emb)
        part2_explicit(root, sid_path, ckpt_a, alt_path, alt_emb, emb)
        part3_dimension_mismatch(root, sid_path, ckpt_a, wrong_path)
        part4_no_fallback(root, sid_path, ckpt_a, correct_path)
        part5_provenance(root, sid_path, side, ckpt_a, ckpt_b, alt_path)
        part6_cli(root, sid_path, ckpt_a, alt_path, wrong_path)
        part7_real(root)
    except BaseException:                                        # noqa: BLE001
        traceback.print_exc()
        RESULTS.append(("harness completed", False, "crashed"))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nEXP-B6: {passed}/{len(RESULTS)} checks passed")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name} -- {detail}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
