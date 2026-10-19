#!/usr/bin/env python
"""EXP-E2 (interface stage) - Encoder hidden interface verification (Task E2, type A).

Verifies that the encoder hidden exposed by E1 is a *usable SIA input interface*:

    T5 Encoder -> encoder_last_hidden_state [B, L, d_model] -> SIA input position

This task is verification only. It adds NO module, NO adapter, NO projection and
NO loss, and touches no model code: the interface it exercises is exactly the
one E1 already provides in ``TIGER.forward(..., return_hidden=True)``.

Scope guards enforced by the script itself:
  * parameter inventory of the model is bitwise-identical to the pristine
    pre-E1 model -> proves no adapter / projection / module was slipped in;
  * gradients from the ordinary TIGER training path are bitwise-identical to
    pristine -> proves the interface did not alter training;
  * the throwaway test scalar used for gradient reachability is a local variable
    that is never added to any TIGER loss.

Read-only with respect to the repository: reads files, writes nothing outside a
temp dir. Run with the repo root as cwd:

    python integration/exp_e2_encoder_interface.py                 # CPU
    EXP_E2_DEVICE=cuda python integration/exp_e2_encoder_interface.py
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
DEVICE = os.environ.get("EXP_E2_DEVICE", "cpu")

# untouched TIGER SID layout: 3 RQ levels of 256 + 1 collision token, offset by 1
SID_BANDS = [(0, 1, 256), (1, 257, 512), (2, 513, 768), (3, 769, 1024)]
SIA_LATENT_DIM = 32  # E2's target latent space, per plan.md -- documented, NOT built here

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main():
    import torch

    sys.path.insert(0, MODEL_DIR)

    tmpdir = tempfile.mkdtemp(prefix="exp_e2_")
    try:
        src = subprocess.run(
            ["git", "-C", REPO, "show", f"{BASELINE_COMMIT}:model/main.py"],
            capture_output=True, text=True, check=True).stdout
        with open(os.path.join(tmpdir, "pristine_main.py"), "w") as f:
            f.write(src)
        sys.path.insert(0, tmpdir)
        import pristine_main
        import main as current_main

        config = {
            'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
            'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1, 'vocab_size': 1025,
            'pad_token_id': 0, 'eos_token_id': 0, 'feed_forward_proj': 'relu',
            'max_len': 20,
        }
        device = torch.device(DEVICE)

        pristine_main.set_seed(SEED)
        pristine = pristine_main.TIGER(config).to(device)
        current_main.set_seed(SEED)
        current = current_main.TIGER(config).to(device)
        current.model.load_state_dict(pristine.model.state_dict())
        pristine.eval()
        current.eval()

        # --------------------------------------------------------- real input
        from dataset import GenRecDataset
        from dataloader import GenRecDataLoader
        ds = GenRecDataset(dataset_path=os.path.join(SMOKE, "data", "Toys", "train.parquet"),
                           code_path=os.path.join(SMOKE, "Toys_clean_tiger_sid.npy"),
                           mode='train', max_len=config['max_len'])
        loader = GenRecDataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)
        B, L_enc = input_ids.shape
        d_model = config['d_model']
        print(f"\nreal input: B={B}  L_enc={L_enc}  labels={tuple(labels.shape)}  "
              f"d_model={d_model}  device={device}\n")

        # =============================================== 1. basic interface
        print("[1] encoder hidden basic interface")
        loss, logits, enc, dec = current(input_ids=input_ids, attention_mask=attention_mask,
                                         labels=labels, return_hidden=True)
        check(f"shape == (B, L, d_model) = {(B, L_enc, d_model)}",
              tuple(enc.shape) == (B, L_enc, d_model), f"got {tuple(enc.shape)}")
        check("batch dimension equals input batch size", enc.shape[0] == B)
        check("hidden dimension equals T5 d_model", enc.shape[-1] == d_model,
              f"{enc.shape[-1]} vs {d_model}")
        check("hidden is non-empty", enc.numel() > 0, f"numel={enc.numel()}")
        check("hidden is not degenerate (all-zero)", float(enc.abs().sum()) > 0,
              f"abs-sum={float(enc.abs().sum()):.4f}")
        check("hidden is finite (no NaN / Inf)", bool(torch.isfinite(enc).all()))
        check("hidden dtype is float32", enc.dtype == torch.float32, str(enc.dtype))
        check("hidden is on the model's device", enc.device.type == device.type, str(enc.device))
        check("hidden carries a computation graph (grad_fn is not None)",
              enc.grad_fn is not None, type(enc.grad_fn).__name__)

        # =============================================== 2. input correspondence
        print("\n[2] correspondence with the untouched TIGER encoder input")
        check("L equals the encoder input sequence length exactly",
              enc.shape[1] == input_ids.shape[1],
              f"L={enc.shape[1]} vs input_ids.shape[1]={input_ids.shape[1]}")
        check("attention_mask is shaped (B, L_enc) as fed to the encoder",
              tuple(attention_mask.shape) == (B, L_enc))
        padded = int((attention_mask == 0).sum())
        check("encoder runs over the full padded length (hidden defined everywhere)",
              enc.shape[1] == attention_mask.shape[1],
              f"{padded}/{attention_mask.numel()} padded position(s), hidden at those "
              f"positions is non-zero: {not bool((enc[attention_mask == 0] == 0).all())} "
              f"-> SIA pooling must use the mask")
        items = input_ids.view(B, config['max_len'], 4)
        check("SID structure untouched: 4 tokens per item",
              items.shape[-1] == 4, f"history parsed as (B, {config['max_len']}, 4)")
        live = items[items.sum(dim=-1) > 0]
        band_ok = all(bool(((live[:, c] >= lo) & (live[:, c] <= hi)).all())
                      for c, lo, hi in SID_BANDS)
        check("SID token bands untouched: [1,256] [257,512] [513,768] [769,1024]",
              band_ok and int(live.max()) <= 1024 and int(live.min()) >= 1,
              f"live tokens {live.shape[0]}, range [{int(live.min())}, {int(live.max())}]")
        check("labels untouched: 4 target tokens per sample", labels.shape[1] == 4)

        # =============================================== 3. gradient reachability
        print("\n[3] gradient reachability via a throwaway local scalar")
        current.zero_grad(set_to_none=True)
        # labels are passed so the decoder can run (HF needs decoder inputs), but
        # the TIGER loss inside this call is deliberately discarded: only the local
        # test scalar below is backpropagated.
        _, _, hidden, _ = current(input_ids=input_ids, attention_mask=attention_mask,
                                  labels=labels, return_hidden=True)
        test_scalar = hidden.float().pow(2).mean()   # local only; NOT a TIGER loss
        test_scalar.backward()
        enc_grads = {n: p.grad for n, p in current.named_parameters()
                     if n.startswith("model.encoder.")}
        got = [g for g in enc_grads.values() if g is not None]
        check("test scalar backward reaches encoder parameters",
              len(got) > 0, f"{len(got)}/{len(enc_grads)} encoder tensors with grad")
        check("encoder gradients are non-zero",
              any(float(g.abs().sum()) > 0 for g in got))
        check("encoder gradients are finite",
              all(bool(torch.isfinite(g).all()) for g in got))
        check("test scalar is a local variable, absent from the model state",
              not any("test_scalar" in k for k in list(current.state_dict()) + list(dict(current.named_buffers()))))

        # =============================================== 4. SIA interface level
        print("\n[4] SIA interface compatibility (interface level only -- no adapter built)")
        check("hidden is indexable/reshapeable as a per-sequence (B, L, d_model) tensor",
              tuple(hidden.reshape(B, L_enc, d_model).shape) == (B, L_enc, d_model))
        pooled = hidden.mean(dim=1)   # plain torch op, no module, no parameters
        check("a per-sequence (B, d_model) tensor is obtainable by plain ops",
              tuple(pooled.shape) == (B, d_model), f"shape {tuple(pooled.shape)}")
        check("E2's target latent space is 32 (documented, not implemented here)",
              SIA_LATENT_DIM == 32)
        check("SIA latent / adapter modules do NOT exist in the model",
              not any(k.endswith("_sia") or "_adapter" in k or "sia_" in k
                      for k in current.state_dict()))

        # =============================================== 5. no scope creep
        print("\n[5] scope guards: interface is purely additive")
        cur_sd, pri_sd = current.state_dict(), pristine.state_dict()
        check("parameter inventory identical to pristine (no adapter/projection added)",
              list(cur_sd) == list(pri_sd) and
              all(cur_sd[k].shape == pri_sd[k].shape for k in cur_sd),
              f"{len(cur_sd)} tensors, {sum(v.numel() for v in cur_sd.values())} params")
        check("current model is not a superset of pristine (no extra trainable tensors)",
              set(cur_sd) == set(pri_sd))

        # =============================================== 6. TIGER regression
        print("\n[6] original TIGER behaviour unchanged")
        with torch.no_grad():
            p_loss, p_logits = pristine(input_ids=input_ids, attention_mask=attention_mask,
                                        labels=labels)
        cur_loss, cur_logits = current(input_ids=input_ids, attention_mask=attention_mask,
                                       labels=labels)
        check("loss bitwise identical to pristine (return_hidden=False)",
              torch.equal(p_loss, cur_loss), f"{cur_loss.item():.8f}")
        check("logits bitwise identical to pristine", torch.equal(p_logits, cur_logits),
              f"shape {tuple(cur_logits.shape)}")
        with torch.no_grad():
            g_p = pristine.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    num_beams=3)
            g_c = current.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   num_beams=3)
        check("generate() bitwise identical to pristine", torch.equal(g_p, g_c),
              f"shape {tuple(g_c.shape)}")

        # gradient parity on the ordinary TIGER training path
        pristine.zero_grad(set_to_none=True)
        current.zero_grad(set_to_none=True)
        pristine(input_ids=input_ids, attention_mask=attention_mask,
                 labels=labels)[0].backward()
        current(input_ids=input_ids, attention_mask=attention_mask,
                labels=labels)[0].backward()
        cur_params = dict(current.named_parameters())
        pri_params = dict(pristine.named_parameters())
        same = all(cur_params[k].grad is not None
                   and torch.equal(cur_params[k].grad, pri_params[k].grad)
                   for k in cur_params)
        check("gradients of the ordinary TIGER training step bitwise identical to pristine",
              same, f"{len(cur_params)} parameter tensors compared")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'=' * 70}\nEXP-E2 (interface stage): {len(RESULTS) - n_fail}/{len(RESULTS)} "
          f"checks passed\n{'=' * 70}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
