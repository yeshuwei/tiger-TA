#!/usr/bin/env python
"""
verify_clean.py — prove that the Clean TIGER RQ-VAE path loads no historical
experiment mechanism.

This is a *runtime* check, not a documentation claim. It:

  1. instruments every pandas.read_parquet / numpy.load call and records the
     paths actually touched;
  2. builds the Clean TIGER RQ-VAE stack (dataset + model + Trainer) with the
     historical switches pinned to 0;
  3. installs tripwires on L_trans / L_balance / L_align so that *calling* any
     of them raises immediately;
  4. runs a couple of real training steps;
  5. asserts that nothing forbidden was read, no tripwire fired, and every
     historical mechanism is inactive on the Trainer object.

A non-zero exit code means Clean TIGER is contaminated.

Usage:
    python -I clean_tiger/verify_clean.py --root . --data_path data/Toys/item_emb.parquet
"""
import argparse
import os
import sys


FORBIDDEN_SUBSTRINGS = (
    "behavior_emb",     # behavior embedding (PPMI / SVD / itemcf / rawcooc / ...)
    "fused_item_emb",   # content + behavior fusion  -> the TA-SID input
    "transition_pairs", # transition / co-occurrence graph (Track B)
    "trans_graph",
    "ppmi",
    "_svd",
    "b2align",          # E50 b2 L_align runs
    "ta_sid",
)


def parse_args():
    p = argparse.ArgumentParser(description="Verify the Clean TIGER baseline is uncontaminated.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--data_path", type=str, required=True,
                   help="content-only item embedding parquet (item_emb.parquet)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--items", type=int, default=2048,
                   help="number of items to use for the smoke training steps")
    p.add_argument("--epochs", type=int, default=1)
    return p.parse_args()


class Tripwire:
    """Callable that fails loudly if a disabled mechanism is ever invoked."""

    def __init__(self, name):
        self.name = name
        self.fired = False

    def __call__(self, *a, **k):
        self.fired = True
        raise AssertionError(
            f"CONTAMINATED: disabled mechanism '{self.name}' was invoked during Clean TIGER training"
        )


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    data_path = os.path.abspath(args.data_path)
    results, failures = [], []

    def check(ok, label, detail=""):
        results.append((ok, label, detail))
        if not ok:
            failures.append(label)

    # ---- 1. instrument file reads ------------------------------------------
    import numpy as np
    import pandas as pd

    reads = []
    _pd_read_parquet = pd.read_parquet

    def spy_read_parquet(path, *a, **k):
        reads.append(str(path))
        return _pd_read_parquet(path, *a, **k)

    pd.read_parquet = spy_read_parquet

    _np_load = np.load

    def spy_np_load(file, *a, **k):
        reads.append(str(file))
        return _np_load(file, *a, **k)

    np.load = spy_np_load

    # ---- 2. build the Clean TIGER RQ-VAE stack -----------------------------
    os.chdir(root)
    sys.path.insert(0, os.path.join(root, "rqvae"))

    import torch
    from torch.utils.data import DataLoader, Subset

    from datasets import EmbDataset          # noqa: E402  (rqvae/datasets.py)
    from models.rqvae import RQVAE           # noqa: E402
    from trainer import Trainer              # noqa: E402

    emb_file = os.path.basename(data_path)
    check(not any(s in emb_file for s in FORBIDDEN_SUBSTRINGS),
          "input embedding is content-only (not fused / behavior)",
          f"basename={emb_file}")

    data = EmbDataset(data_path)
    n = min(args.items, len(data))
    subset = Subset(data, list(range(n)))
    loader = DataLoader(subset, num_workers=0, batch_size=args.batch_size,
                        shuffle=False, pin_memory=False)

    model = RQVAE(in_dim=data.dim, num_emb_list=[256, 256, 256], e_dim=32,
                  layers=[512, 256, 128, 64], dropout_prob=0.0, bn=False,
                  loss_type="mse", quant_loss_weight=1.0, beta=0.25,
                  kmeans_init=True, kmeans_iters=100,
                  sk_epsilons=[0.0, 0.0, 0.003], sk_iters=50)

    # exactly the Clean TIGER configuration: every historical switch is 0
    from types import SimpleNamespace
    targs = SimpleNamespace(
        lr=1e-3, learner="AdamW", lr_scheduler_type="linear", weight_decay=1e-4,
        epochs=args.epochs, warmup_epochs=0, save_limit=1, eval_step=1,
        device=args.device, num_emb_list=[256, 256, 256], num_workers=0,
        ckpt_dir=os.path.join(root, "clean_tiger", "logs", "_verify_ckpt"),
        lambda_trans=0.0, lambda_balance=0.0, tau=0.1, trans_graph_path=None,
        align_weight=0.0, align_data_path=None, align_tau=0.1,
    )
    trainer = Trainer(targs, model, len(loader))

    check(trainer.lambda_trans == 0.0, "L_trans disabled", f"lambda_trans={trainer.lambda_trans}")
    check(trainer.lambda_balance == 0.0, "L_balance disabled", f"lambda_balance={trainer.lambda_balance}")
    check(trainer.align_weight == 0.0, "L_align disabled", f"align_weight={trainer.align_weight}")
    check(trainer.trans_dict is None, "no transition graph loaded")
    check(trainer.align_table is None, "no behavior alignment target loaded")
    check(trainer.align_proj is None, "no L_align projection head built")

    # ---- 3. tripwires ------------------------------------------------------
    tw_trans = Tripwire("L_trans (ContrastiveLoss)")
    tw_balance = Tripwire("L_balance (BalanceLoss)")
    tw_align = Tripwire("L_align (_get_align_loss)")
    trainer.contrastive_loss = tw_trans
    trainer.balance_loss = tw_balance
    trainer._get_align_loss = tw_align

    # ---- 4. real training steps -------------------------------------------
    try:
        for epoch in range(args.epochs):
            trainer._train_epoch(loader, epoch)
        train_ok, train_err = True, ""
    except AssertionError as e:
        train_ok, train_err = False, str(e)
    except Exception as e:                                   # noqa: BLE001
        train_ok, train_err = False, f"{type(e).__name__}: {e}"

    check(train_ok, "RQ-VAE training steps completed", train_err)
    for tw in (tw_trans, tw_balance, tw_align):
        check(not tw.fired, f"tripwire never fired: {tw.name}")

    # ---- 5. nothing forbidden was read ------------------------------------
    forbidden_reads = [p for p in reads
                       if any(s in os.path.basename(p).lower() or s in p.lower()
                              for s in FORBIDDEN_SUBSTRINGS)]
    check(not forbidden_reads, "no behavior / fused / transition file was read",
          f"hits={forbidden_reads}" if forbidden_reads else f"read {len(reads)} file(s): "
          + ", ".join(sorted(set(os.path.basename(p) for p in reads))))

    # ---- 6. report + known divergence -------------------------------------
    print("=" * 78)
    print("Clean TIGER verification")
    print("=" * 78)
    for ok, label, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   ({detail})" if detail else ""))

    # Latent divergence from upstream TIGER: RQVAE.forward now defaults to
    # return_z=True (4-tuple) instead of the original 3-tuple. No Clean TIGER
    # code path relies on the default (trainer passes return_z explicitly and
    # SID generation uses get_indices), so this is inert — but it is recorded
    # here so the frozen baseline documents it explicitly.
    with torch.no_grad():
        out = model(torch.zeros(2, data.dim, device=args.device))
    arity = len(out) if isinstance(out, tuple) else 1
    print(f"  [INFO] RQVAE.forward default return arity = {arity} "
          f"(upstream TIGER = 3). Inert: no Clean TIGER path uses the default.")

    print("=" * 78)
    if failures:
        print(f"RESULT: FAIL — {len(failures)} check(s) failed: {failures}")
        return 1
    print(f"RESULT: PASS — {len(results)}/{len(results)} checks green. "
          "No historical mechanism participates in Clean TIGER.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
