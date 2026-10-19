#!/usr/bin/env python
"""EXP-E2 - SIA adapter interface sanity test (Task E2, type A).

Exercises the E2 adapter (``tiger_ext/sia.py``) against the E1 interface on a
real TIGER batch, and checks the three items frozen in experiments.md:

    * output is (B, 32)
    * gradients to the adapter parameters are non-None and non-zero
    * padding positions do not affect the output of valid samples

The adapter is a standalone module: it is NOT wired into TIGER, and the TIGER
side is re-verified bitwise against the pre-E1 baseline to prove E2 touched
nothing in the recommendation path.

Read-only with respect to the repository: reads files, writes nothing outside a
temp dir. Run with the repo root as cwd:

    python integration/exp_e2_sia_adapter.py                 # CPU
    EXP_E2_DEVICE=cuda python integration/exp_e2_sia_adapter.py
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
PAD_TOL = 1e-6  # float32 reduction-order tolerance for the padding test

RESULTS = []


def module_facts(path):
    """Parse a module and return (top-level defs, imported modules, used names).

    Substring-scanning source text is not a check: docstrings mention RQVAE and
    codebooks in prose while the code never touches them. This looks at the AST.
    """
    import ast
    with open(path) as f:
        tree = ast.parse(f.read())
    defs = [n.name for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))] + \
           [t.id for n in tree.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)]
    imports = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imports.append(n.module or "")
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    calls = {n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    calls |= {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return defs, imports, names, calls


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main():
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, MODEL_DIR)
    sys.path.insert(0, REPO)
    from tiger_ext.sia import SIAAdapter, masked_mean_pool, SIA_LATENT_DIM

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

        # ================================================== 1. consumes E1 output
        print("\n[1] adapter input is the E1 encoder hidden")
        loss, logits, enc, dec = current(input_ids=input_ids, attention_mask=attention_mask,
                                         labels=labels, return_hidden=True)
        check("encoder hidden fed to the adapter has shape (B, L, d_model)",
              tuple(enc.shape) == (B, L_enc, d_model), f"got {tuple(enc.shape)}")
        check("adapter input is the encoder hidden, not the decoder hidden",
              enc.shape[1] == L_enc and dec.shape[1] != L_enc,
              f"enc L={enc.shape[1]}, dec L={dec.shape[1]}")
        adapter = SIAAdapter(d_model).to(device)
        pooled = masked_mean_pool(enc, attention_mask)
        out = adapter(enc, attention_mask)
        check("forward runs end to end on the real batch", out is not None)

        # ================================================== 2/3/4. shapes
        print("\n[2] pooling and projection shapes")
        check(f"pooling output shape == (B, d_model) = {(B, d_model)}",
              tuple(pooled.shape) == (B, d_model), f"got {tuple(pooled.shape)}")
        check(f"adapter output shape == (B, {SIA_LATENT_DIM}) = {(B, SIA_LATENT_DIM)}",
              tuple(out.shape) == (B, SIA_LATENT_DIM), f"got {tuple(out.shape)}")
        check("latent dim equals TIGER RQ-VAE e_dim = 32",
              SIA_LATENT_DIM == 32 and adapter.latent_dim == 32)
        check("linear layer is exactly Linear(d_model, 32)",
              tuple(adapter.proj.weight.shape) == (SIA_LATENT_DIM, d_model),
              f"weight {tuple(adapter.proj.weight.shape)}")

        # ================================================== 3. mask correctness
        print("\n[3] attention_mask actually participates in pooling")
        ref = (enc.double() * attention_mask.double().unsqueeze(-1)).sum(1) / \
              attention_mask.double().sum(1, keepdim=True).clamp(min=1.0)
        check("masked mean matches an independent float64 reference",
              bool(torch.allclose(pooled.double(), ref, atol=1e-6, rtol=0)),
              f"max abs diff {float((pooled.double() - ref).abs().max()):.3e}")
        unmasked = enc.mean(dim=1)
        gap = float((unmasked - pooled).abs().max())
        check("unmasked mean would differ materially (proves the mask matters)",
              gap > 1e-3, f"max abs diff unmasked-vs-masked = {gap:.4f}")
        check("padding count is non-trivial on this real batch",
              int((attention_mask == 0).sum()) > 0,
              f"{int((attention_mask == 0).sum())}/{attention_mask.numel()} positions padded")

        # ================================================== 4. padding consistency
        print("\n[4] padding consistency test (frozen EXP-E2 criterion 3)")
        valid = enc[0]                                # (L_enc, d_model)
        real = attention_mask[0].bool()
        core = valid[real]                            # (n_valid, d_model), no padding
        n_valid = core.shape[0]

        # (a) same valid block, different amounts of leading padding
        rows, masks = [], []
        for pad in (0, 3, 11):
            rows.append(torch.cat([torch.randn(pad, d_model, device=device), core], dim=0))
            masks.append(torch.cat([torch.zeros(pad, device=device),
                                    torch.ones(n_valid, device=device)]).long())
        diffs = []
        for i in range(1, len(rows)):
            d = (masked_mean_pool(rows[i].unsqueeze(0), masks[i].unsqueeze(0))
                 - masked_mean_pool(rows[0].unsqueeze(0), masks[0].unsqueeze(0))).abs().max()
            diffs.append(float(d))
        check("more padding does not dilute the pooled result",
              max(diffs) <= PAD_TOL,
              f"pad 0 vs 3 vs 11 -> max abs diff {max(diffs):.3e} "
              f"({'bitwise identical' if max(diffs) == 0 else 'within float32 reduction order'})")

        # (b) same layout, garbage content at the padded positions
        noisy = rows[1].clone()
        noisy[:3] = torch.randn_like(noisy[:3]) * 100.0
        d_noise = (masked_mean_pool(noisy.unsqueeze(0), masks[1].unsqueeze(0))
                   - masked_mean_pool(rows[1].unsqueeze(0), masks[1].unsqueeze(0))).abs().max()
        check("changing the CONTENT of padded positions does not change the output",
              float(d_noise) == 0.0, f"max abs diff {float(d_noise):.3e}")

        # (c) end-to-end through the adapter: same valid block, same mask, three
        #     different padding contents, all stacked into one batch
        L_pad = n_valid + 5
        mask_pad = torch.cat([torch.zeros(5, device=device),
                              torch.ones(n_valid, device=device)]).long()
        batch_rows = torch.stack([
            torch.cat([torch.randn(5, d_model, device=device), core]),           # noise pad
            torch.cat([torch.zeros(5, d_model, device=device), core]),           # zero pad
            torch.cat([torch.randn(5, d_model, device=device) * 1000.0, core]),  # huge pad
        ])
        batch_mask = mask_pad.unsqueeze(0).expand(3, L_pad).contiguous()
        with torch.no_grad():
            a_out = adapter(batch_rows, batch_mask)
            bare = adapter(core.unsqueeze(0), torch.ones(n_valid, device=device,
                                                         dtype=torch.long).unsqueeze(0))
        check("adapter output identical across the three padding variants",
              torch.equal(a_out[0], a_out[1]) and torch.equal(a_out[1], a_out[2]),
              f"max abs diff {float((a_out - a_out[1]).abs().max()):.3e}")
        # cross-shape comparison: a (3, L_pad, d) batch vs a (1, n_valid, d) batch
        # cannot be bitwise (different float32 reduction kernels), so this one is
        # tolerance-based. Dilution would show up at O(1) relative scale, not 1e-7.
        gap_pad = float((a_out[0] - bare[0]).abs().max())
        check("adapter output with 5 padding positions == output with none (float32 tol)",
              gap_pad <= PAD_TOL,
              f"max abs diff {gap_pad:.3e} on values of scale "
              f"{float(bare[0].abs().max()):.2f} -> relative {gap_pad / float(bare[0].abs().max()):.1e}")

        # ================================================== 5. gradients
        print("\n[5] backward / gradient")
        adapter.zero_grad(set_to_none=True)
        current.zero_grad(set_to_none=True)
        scalar = adapter(enc, attention_mask).float().pow(2).mean()   # local, not a TIGER loss
        scalar.backward()
        a_grads = {n: p.grad for n, p in adapter.named_parameters()}
        check("adapter parameters receive gradients",
              all(g is not None for g in a_grads.values()), f"{len(a_grads)} tensors")
        check("adapter gradients are non-zero",
              all(float(g.abs().sum()) > 0 for g in a_grads.values()),
              ", ".join(f"{n}:{float(g.abs().sum()):.3e}" for n, g in a_grads.items()))
        check("adapter gradients are finite",
              all(bool(torch.isfinite(g).all()) for g in a_grads.values()))
        enc_grads = [p.grad for n, p in current.named_parameters()
                     if n.startswith("model.encoder.") and p.grad is not None]
        check("gradient flows back through the adapter into the T5 encoder",
              len(enc_grads) > 0 and all(bool(torch.isfinite(g).all()) for g in enc_grads),
              f"{len(enc_grads)} encoder tensors with grad")

        # ================================================== 6. output is finite
        check("adapter output is finite (no NaN / Inf)", bool(torch.isfinite(out).all()))
        check("adapter output is not degenerate (all-zero)",
              float(out.abs().sum()) > 0, f"abs-sum {float(out.abs().sum()):.4f}")

        # ================================================== 7. scope guards
        print("\n[6] scope guards")
        names = list(dict(adapter.named_parameters()))
        check("adapter holds exactly the 2 Linear tensors (weight + bias)",
              names == ["proj.weight", "proj.bias"], f"{names}")
        defs, imports, names, calls = module_facts(os.path.join(REPO, "tiger_ext", "sia.py"))
        check("adapter never touches RQ-VAE (AST: no rqvae/RQVAE anywhere in code)",
              not any("rqvae" in i.lower() for i in imports) and "RQVAE" not in names,
              f"imports={imports}")
        check("adapter is not wired into the TIGER module tree",
              not any("sia" in n.lower() for n in
                      [k for k, _ in current.named_modules()]))
        cur_sd, pri_sd = current.state_dict(), pristine.state_dict()
        check("TIGER parameter inventory still identical to pristine (no module added)",
              list(cur_sd) == list(pri_sd) and
              all(cur_sd[k].shape == pri_sd[k].shape for k in cur_sd),
              f"{len(cur_sd)} tensors")
        check("no SIA loss / code logits / PSA in the module (AST: no loss or code ops)",
              not any(c in calls for c in ("kl_div", "log_softmax", "softmax",
                                           "cross_entropy", "nll_loss"))
              and set(defs) == {"SIA_LATENT_DIM", "masked_mean_pool", "SIAAdapter"},
              f"defs={defs}, calls={sorted(calls)}")

        # ================================================== 8. TIGER regression
        print("\n[7] original TIGER recommendation flow unchanged")
        with torch.no_grad():
            p_loss, p_logits = pristine(input_ids=input_ids, attention_mask=attention_mask,
                                        labels=labels)
            c_loss, c_logits = current(input_ids=input_ids, attention_mask=attention_mask,
                                       labels=labels)
            g_p = pristine.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    num_beams=3)
            g_c = current.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   num_beams=3)
        check("loss bitwise identical to pristine", torch.equal(p_loss, c_loss),
              f"{c_loss.item():.8f}")
        check("logits bitwise identical to pristine", torch.equal(p_logits, c_logits),
              f"shape {tuple(c_logits.shape)}")
        check("generate() bitwise identical to pristine", torch.equal(g_p, g_c),
              f"shape {tuple(g_c.shape)}")

        pristine.zero_grad(set_to_none=True)
        current.zero_grad(set_to_none=True)
        pristine(input_ids=input_ids, attention_mask=attention_mask,
                 labels=labels)[0].backward()
        current(input_ids=input_ids, attention_mask=attention_mask,
                labels=labels)[0].backward()
        cp, pp = dict(current.named_parameters()), dict(pristine.named_parameters())
        check("gradients of the ordinary TIGER training step bitwise identical to pristine",
              all(cp[k].grad is not None and torch.equal(cp[k].grad, pp[k].grad) for k in cp),
              f"{len(cp)} parameter tensors compared")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'=' * 70}\nEXP-E2 (SIA adapter): {len(RESULTS) - n_fail}/{len(RESULTS)} "
          f"checks passed\n{'=' * 70}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
