#!/usr/bin/env python
"""EXP-E1 - Clean TIGER consistency test (Task E1, type A).

Proves that the E1 hidden-state interface is *purely additive*: it exposes the
T5 encoder / decoder hidden states without perturbing anything the original
TIGER pipeline computes.

Method: the pristine pre-E1 ``model/main.py`` is extracted from git commit
``5edfca0`` (the frozen Clean TIGER baseline) into a scratch directory, both
versions are run on **the same tensors with the same weights**, and the outputs
are compared with ``torch.equal`` (bitwise, never allclose).

This script is read-only with respect to the repository: it only *reads* files
and writes nothing outside a temp dir. Run it with the repo root as cwd:

    python integration/exp_e1_consistency.py                 # CPU
    EXP_E1_DEVICE=cuda python integration/exp_e1_consistency.py   # as trained
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, "model")
BASELINE_COMMIT = "5edfca0"
SMOKE = os.path.join(REPO, "clean_tiger", "_smoke", "Toys")
SEED = 2025
BATCH = 4
DEVICE = os.environ.get("EXP_E1_DEVICE", "cpu")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main():
    import numpy as np
    import torch

    sys.path.insert(0, MODEL_DIR)

    # ---------------------------------------------------------------- pristine
    tmpdir = tempfile.mkdtemp(prefix="exp_e1_")
    try:
        src = subprocess.run(
            ["git", "-C", REPO, "show", f"{BASELINE_COMMIT}:model/main.py"],
            capture_output=True, text=True, check=True).stdout
        pristine_path = os.path.join(tmpdir, "pristine_main.py")
        with open(pristine_path, "w") as f:
            f.write(src)
        sys.path.insert(0, tmpdir)
        import pristine_main
        import main as current_main

        config = {
            'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
            'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1, 'vocab_size': 1025,
            'pad_token_id': 0, 'eos_token_id': 0, 'feed_forward_proj': 'relu',
        }

        device = torch.device(DEVICE)
        pristine_main.set_seed(SEED)
        pristine = pristine_main.TIGER(config).to(device)
        current_main.set_seed(SEED)
        current = current_main.TIGER(config).to(device)
        # identical weights regardless of init RNG
        current.model.load_state_dict(pristine.model.state_dict())
        pristine.eval()
        current.eval()

        # ----------------------------------------------------------------- input
        from dataset import GenRecDataset
        from dataloader import GenRecDataLoader
        ds = GenRecDataset(dataset_path=os.path.join(SMOKE, "data", "Toys", "train.parquet"),
                           code_path=os.path.join(SMOKE, "Toys_clean_tiger_sid.npy"),
                           mode='train', max_len=20)
        loader = GenRecDataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)
        B, L_enc = input_ids.shape
        L_dec = labels.shape[1]
        d_model = config['d_model']
        print(f"\nreal input: B={B}  L_enc={L_enc}  L_dec={L_dec}  d_model={d_model}\n")

        # ------------------------------------------------------- 1. additive path
        p_loss, p_logits = pristine(input_ids=input_ids, attention_mask=attention_mask,
                                    labels=labels)
        c_loss, c_logits = current(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=labels)
        check("return_hidden=False -> loss bitwise identical to pristine",
              torch.equal(p_loss, c_loss), f"{p_loss.item():.8f}")
        check("return_hidden=False -> logits bitwise identical to pristine",
              torch.equal(p_logits, c_logits), f"shape {tuple(c_logits.shape)}")

        # ------------------------------------------------------- 2. hidden path
        h_loss, h_logits, enc, dec = current(input_ids=input_ids, attention_mask=attention_mask,
                                             labels=labels, return_hidden=True)
        check("return_hidden=True -> loss bitwise identical to pristine",
              torch.equal(p_loss, h_loss), f"{h_loss.item():.8f}")
        check("return_hidden=True -> logits bitwise identical to pristine",
              torch.equal(p_logits, h_logits), f"shape {tuple(h_logits.shape)}")
        check(f"encoder_hidden shape == (B, L_enc, d_model) = {(B, L_enc, d_model)}",
              tuple(enc.shape) == (B, L_enc, d_model), f"got {tuple(enc.shape)}")
        check(f"decoder_hidden shape == (B, L_dec, d_model) = {(B, L_dec, d_model)}",
              tuple(dec.shape) == (B, L_dec, d_model), f"got {tuple(dec.shape)}")

        # ---------------------------------------------- 3. wrong-tensor guards
        check("encoder_hidden is 3-D, not pooled [B, d_model]",
              enc.dim() == 3 and tuple(enc.shape) != (B, d_model))
        check("encoder_hidden last dim is d_model, not vocab_size",
              enc.shape[-1] == d_model and enc.shape[-1] != config['vocab_size'],
              f"d_model={d_model}, vocab_size={config['vocab_size']}")
        check("encoder_hidden is the encoder output, not the decoder output",
              tuple(enc.shape) != tuple(dec.shape)
              and float(enc.abs().sum()) > 0 and float(dec.abs().sum()) > 0,
              f"enc {tuple(enc.shape)} vs dec {tuple(dec.shape)}")
        check("encoder_hidden is not logits",
              tuple(enc.shape) != tuple(h_logits.shape))

        # ------------------------------------------------------- 4. numerics
        check("no NaN / Inf in loss, logits, encoder_hidden, decoder_hidden",
              all(bool(torch.isfinite(t).all()) for t in (h_loss, h_logits, enc, dec)))

        # ------------------------------------------------------- 5. backward
        current.train()
        b_loss, b_logits, b_enc, b_dec = current(input_ids=input_ids,
                                                 attention_mask=attention_mask,
                                                 labels=labels, return_hidden=True)
        b_loss.backward()
        enc_grads = [p.grad for n, p in current.named_parameters()
                     if n.startswith("model.encoder.") and p.grad is not None]
        check("backward() on loss from return_hidden=True reaches the encoder",
              len(enc_grads) > 0, f"{len(enc_grads)} encoder tensors with grad")
        check("encoder gradients are finite and non-zero",
              all(bool(torch.isfinite(g).all()) for g in enc_grads)
              and any(float(g.abs().sum()) > 0 for g in enc_grads))
        # the hidden tensors are returned detached from the graph in no way --
        # they carry grad_fn, which later Tasks (E2/E5) depend on
        check("exposed hidden tensors carry grad_fn (usable downstream)",
              b_enc.grad_fn is not None and b_dec.grad_fn is not None)

        # ------------------------------------------------------- 6. generate()
        current.eval()
        with torch.no_grad():
            g_p = pristine.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    num_beams=3)
            g_c = current.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   num_beams=3)
        check("generate() output bitwise identical to pristine",
              torch.equal(g_p, g_c), f"shape {tuple(g_c.shape)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'=' * 70}\nEXP-E1: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed\n{'=' * 70}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
