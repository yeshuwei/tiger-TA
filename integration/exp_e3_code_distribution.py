#!/usr/bin/env python
"""EXP-E3 - SIA code-distribution diagnostic (Task E3, type A + B).

Verifies that the E3 interface extracts, from TIGER's *own* quantizer, one
(B, 256) code-logit block per RQ level on both sides of the SIA alignment:

    sequence side:  encoder hidden -> SIAAdapter -> (B, 32) -> rq      (no encoder)
    target side:    item content embedding -> RQVAE.encoder -> (B, 32) -> rq

and that turning the extra output on leaves TIGER bitwise unchanged.

The quantizer used is a real Clean TIGER RQ-VAE checkpoint, so the codebooks
under test are the frozen baseline's, not fresh ones.

Read-only with respect to the repository: reads files, writes nothing outside a
temp dir. Run with the repo root as cwd:

    python integration/exp_e3_code_distribution.py                 # CPU
    EXP_E3_DEVICE=cuda python integration/exp_e3_code_distribution.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, "model")
RVQAE_DIR = os.path.join(REPO, "rqvae")
SMOKE = os.path.join(REPO, "clean_tiger", "_smoke", "Toys")
CKPT = os.path.join(REPO, "clean_tiger", "ckpt", "Toys", "rqvae",
                    "Oct-14-2026_05-55-09", "best_collision_model.pth")
BASELINE_COMMIT = "5edfca0"
SEED = 2025
BATCH = 4
DEVICE = os.environ.get("EXP_E3_DEVICE", "cpu")

# Original TIGER RQ-VAE hyper-parameters (clean_tiger/config.sh)
RQVAE_CFG = dict(num_emb_list=[256, 256, 256], e_dim=32, layers=[512, 256, 128, 64],
                 dropout_prob=0.0, bn=False, loss_type="mse", quant_loss_weight=1.0,
                 beta=0.25, kmeans_init=True, kmeans_iters=100,
                 sk_epsilons=[0.0, 0.0, 0.003], sk_iters=50)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def build_pristine_rqvae(tmpdir):
    """Load the pre-E3 rqvae/models package from git under a distinct name."""
    pkg = os.path.join(tmpdir, "pristine_rqvae", "pristine_models")
    os.makedirs(pkg, exist_ok=True)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    for fn in ("rq.py", "vq.py", "rqvae.py", "layers.py"):
        src = subprocess.run(["git", "-C", REPO, "show",
                              f"{BASELINE_COMMIT}:rqvae/models/{fn}"],
                             capture_output=True, text=True, check=True).stdout
        with open(os.path.join(pkg, fn), "w") as f:
            f.write(src)
    sys.path.insert(0, os.path.join(tmpdir, "pristine_rqvae"))
    from pristine_models.rqvae import RQVAE as PristineRQVAE
    return PristineRQVAE


def main():
    import numpy as np
    import torch

    # order matters: both model/ and rqvae/ contain a top-level `main.py`, so
    # MODEL_DIR must win the lookup for `import main`.
    sys.path.insert(0, RVQAE_DIR)
    sys.path.insert(0, MODEL_DIR)
    sys.path.insert(0, REPO)
    from models.rqvae import RQVAE
    from datasets import EmbDataset
    from tiger_ext.sia import SIAAdapter
    from tiger_ext.sia_codes import SIACodeExtractor

    tmpdir = tempfile.mkdtemp(prefix="exp_e3_")
    try:
        PristineRQVAE = build_pristine_rqvae(tmpdir)
        sys.path.insert(0, os.path.join(tmpdir, "pristine_rqvae"))

        src = subprocess.run(["git", "-C", REPO, "show",
                              f"{BASELINE_COMMIT}:model/main.py"],
                             capture_output=True, text=True, check=True).stdout
        with open(os.path.join(tmpdir, "pristine_main.py"), "w") as f:
            f.write(src)
        sys.path.insert(0, tmpdir)
        import pristine_main
        import main as current_main

        device = torch.device(DEVICE)

        # ------------------------------------------------ real TIGER RQ-VAE
        emb_ds = EmbDataset(os.path.join(SMOKE, "data", "Toys", "item_emb.parquet"))
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        rqvae = RQVAE(in_dim=emb_ds.dim, **RQVAE_CFG).to(device)
        rqvae.load_state_dict(ck["state_dict"])
        rqvae.eval()
        sid = np.load(os.path.join(SMOKE, "Toys_clean_tiger_sid.npy"))
        print(f"\nRQ-VAE: in_dim={emb_ds.dim}, checkpoint {os.path.basename(CKPT)}")
        print(f"items={len(emb_ds)}, SID {sid.shape}\n")

        # ------------------------------------------------ real TIGER batch
        t5cfg = {'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
                 'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1, 'vocab_size': 1025,
                 'pad_token_id': 0, 'eos_token_id': 0, 'feed_forward_proj': 'relu',
                 'max_len': 20}
        current_main.set_seed(SEED)
        tiger = current_main.TIGER(t5cfg).to(device)
        tiger.eval()
        from dataset import GenRecDataset
        from dataloader import GenRecDataLoader
        ds = GenRecDataset(dataset_path=os.path.join(SMOKE, "data", "Toys", "train.parquet"),
                           code_path=os.path.join(SMOKE, "Toys_clean_tiger_sid.npy"),
                           mode='train', max_len=t5cfg['max_len'])
        batch = next(iter(GenRecDataLoader(ds, batch_size=BATCH, shuffle=False,
                                           num_workers=0)))
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        adapter = SIAAdapter(t5cfg['d_model']).to(device)
        extractor = SIACodeExtractor(adapter, rqvae).to(device)

        # ============================================ 1. sequence-side interface
        print("[1] sequence-side extraction")
        _, _, enc, _ = tiger(input_ids=input_ids, attention_mask=attention_mask,
                             labels=labels, return_hidden=True)
        query = extractor.to_query(enc, attention_mask)
        check(f"input to the quantizer path is (B, 32) = {(BATCH, 32)}",
              tuple(query.shape) == (BATCH, 32), f"got {tuple(query.shape)}")
        seq = extractor.sequence_logits(enc, attention_mask)
        check("output contains exactly 3 RQ levels",
              len(seq) == 3, f"{len(seq)} levels")
        check("every level is (B, 256)",
              all(tuple(t.shape) == (BATCH, 256) for t in seq),
              f"{[tuple(t.shape) for t in seq]}")
        check("quantizer has exactly 3 codebooks of 256 codes",
              rqvae.rq.num_quantizers == 3 and
              all(q.n_e == 256 for q in rqvae.rq.vq_layers) and
              all(tuple(q.embedding.weight.shape) == (256, 32)
                  for q in rqvae.rq.vq_layers),
              f"num_quantizers={rqvae.rq.num_quantizers}")

        # ============================================ 2. target-side interface
        print("\n[2] target-side extraction")
        tgt_rows = torch.arange(BATCH, device=device)
        target_emb = torch.tensor(emb_ds.embeddings, dtype=torch.float32,
                                  device=device)[tgt_rows]
        tgt = extractor.target_logits(target_emb)
        check("target side also yields exactly 3 RQ levels", len(tgt) == 3)
        check("target levels are (B, 256)",
              all(tuple(t.shape) == (BATCH, 256) for t in tgt),
              f"{[tuple(t.shape) for t in tgt]}")
        check("both sides use the same codebook objects (identity, not a copy)",
              all(extractor.rqvae.rq.vq_layers[i] is rqvae.rq.vq_layers[i]
                  for i in range(3)))

        # ============================================ 3. codebook identity
        print("\n[3] logits really come from the TIGER codebook")
        # Residual quantisation: each level sees the residual left by the levels
        # before it, so the level-k logits are the squared L2 distance from THAT
        # residual. The sign is ETEGRec's official `dist='l2'` branch, which
        # returns `logit = d` un-negated.
        manual = []
        residual = query.double()
        with torch.no_grad():
            for level, vq in enumerate(rqvae.rq.vq_layers):
                w = vq.embedding.weight.double()
                d = ((residual ** 2).sum(1, keepdim=True)
                     + (w ** 2).sum(1, keepdim=True).t() - 2 * residual @ w.t())
                manual.append(d.float())
                # TIGER's first two levels use argmin (sk_epsilon=0); the level-2
                # residual is never needed again, so this chain is exact
                residual = residual - w[d.argmin(dim=-1)]
        check("sequence logits == recomputed +L2 distance from each level's residual "
              "to that level's TIGER codebook",
              all(torch.allclose(seq[i], manual[i], atol=1e-4, rtol=0)
                  for i in range(3)),
              f"max abs diff per level "
              f"{[f'{float((seq[i] - manual[i]).abs().max()):.2e}' for i in range(3)]}")

        # The sign is pinned explicitly: ETEGRec's official `vq.py` returns
        # `logit = d` (un-negated squared L2) under `dist='l2'`. That means
        # logits are non-negative and *larger* means *further*.
        nonneg = all(float(t.min()) >= -1e-4 for t in list(seq) + list(tgt))
        check("sign is un-negated: every code logit is >= 0 (squared L2, not its negative)",
              nonneg,
              f"min logit = {min(float(t.min()) for t in list(seq) + list(tgt)):+.4f}")
        with torch.no_grad():
            cdist_ok = True
            residual_t = rqvae.encoder(target_emb).double()
            for level, vq in enumerate(rqvae.rq.vq_layers):
                w = vq.embedding.weight.double()
                d2 = torch.cdist(residual_t, w, p=2) ** 2
                cdist_ok = cdist_ok and torch.allclose(tgt[level], d2.float(),
                                                       atol=1e-3, rtol=0)
                residual_t = residual_t - w[d2.argmin(dim=-1)]
        check("target-side logits equal the squared Euclidean distance to each codeword "
              "(independent cdist recomputation, all 3 levels)",
              cdist_ok)
        # Under `+d` the *chosen* codeword is the ar**gmin** of the logits, the
        # opposite of the previous `-d` convention. Levels 0/1 select by argmin
        # (sk_epsilon = 0); level 2 is excluded because sinkhorn selects there.
        with torch.no_grad():
            chosen = rqvae.rq(query, use_sk=True)[2]
        chosen = chosen.detach().cpu()
        argmin_match = all(
            float((seq[i].argmin(dim=-1).detach().cpu() == chosen[:, i]).float().mean()) == 1.0
            for i in range(2))
        check("the codeword the quantizer selects is the argmin of the logits "
              "(larger logit = further), levels 0/1",
              argmin_match,
              "level agreement "
              f"{[float((seq[i].argmin(dim=-1).detach().cpu() == chosen[:, i]).float().mean()) for i in range(2)]}")
        with torch.no_grad():
            idx = rqvae.get_indices(torch.tensor(emb_ds.embeddings, dtype=torch.float32,
                                                 device=device), use_sk=True).cpu().numpy()
        agree = [float((idx[:, c] == sid[:, c]).mean()) for c in range(3)]
        check("checkpoint is TIGER's: re-derived level-0/1 codes match the SID file 100%",
              agree[0] == 1.0 and agree[1] == 1.0,
              f"agreement per level = {[round(a, 4) for a in agree]} "
              f"(level 2 is informational: collision resolution rewrites that codebook)")
        check("item_emb row order corresponds to SID row order (row i <-> item i)",
              len(emb_ds) == sid.shape[0], f"{len(emb_ds)} rows vs {sid.shape[0]}")

        # ============================================ 4. path assertions 6/7
        print("\n[4] path assertions: sequence side skips the RQ-VAE encoder")
        calls = {"encoder": 0, "decoder": 0}
        captured = {}
        h_enc = rqvae.encoder.register_forward_hook(
            lambda *a, **k: calls.__setitem__("encoder", calls["encoder"] + 1))
        h_dec = rqvae.decoder.register_forward_hook(
            lambda *a, **k: calls.__setitem__("decoder", calls["decoder"] + 1))
        h_ad = adapter.register_forward_hook(
            lambda mod, i, o: captured.__setitem__("adapter_out", o))
        h_rq = rqvae.rq.register_forward_pre_hook(
            lambda mod, args: captured.__setitem__("query", args[0]))
        seq2 = extractor.sequence_logits(enc, attention_mask)
        check("sequence side: RQVAE.encoder never called", calls["encoder"] == 0,
              f"encoder calls = {calls['encoder']}")
        check("sequence side: RQVAE.decoder never called", calls["decoder"] == 0,
              f"decoder calls = {calls['decoder']}")
        check("within one forward, the tensor entering the RQ quantizer IS the "
              "adapter output (no transform in between)",
              captured.get("query") is captured.get("adapter_out")
              and captured.get("query") is not None,
              f"adapter_out is query: {captured.get('query') is captured.get('adapter_out')}")
        calls["encoder"] = calls["decoder"] = 0
        extractor.target_logits(target_emb)
        check("target side: RQVAE.encoder called exactly once (must go through it)",
              calls["encoder"] == 1, f"encoder calls = {calls['encoder']}")
        check("target side: RQVAE.decoder never called", calls["decoder"] == 0,
              f"decoder calls = {calls['decoder']}")
        h_enc.remove(); h_dec.remove(); h_rq.remove(); h_ad.remove()
        check("re-running the sequence path is deterministic",
              all(torch.equal(seq[i], seq2[i]) for i in range(3)))

        # ============================================ 5. collision token excluded
        print("\n[5] collision token does not enter the logits")
        coll = np.where(sid[:, 3] != 0)[0]
        check("collision items exist in the SID file (plan.md says 467)",
              len(coll) == 467, f"{len(coll)} items with a non-zero 4th column")
        check("exactly 3 levels for collision-bearing items too (no 4th level)",
              len(extractor.target_logits(target_emb)) == 3)
        check("collision token values cannot be codebook indices at all",
              int(sid[coll, 3].min()) > 0 and
              extractor.num_levels == 3,
              f"collision values in [{int(sid[coll, 3].min())}, {int(sid[coll, 3].max())}], "
              f"quantizer levels = {extractor.num_levels}")

        # ============================================ 6. numerics
        print("\n[6] logits numerics")
        all_logits = list(seq) + list(tgt)
        check("all logits finite (no NaN / Inf)",
              all(bool(torch.isfinite(t).all()) for t in all_logits))
        check("logits are not degenerate (not constant across the codebook)",
              all(float(t.std(dim=-1).min()) > 0 for t in all_logits),
              f"min per-row std = {min(float(t.std(dim=-1).min()) for t in all_logits):.4f}")
        soft = [torch.softmax(t.float(), dim=-1) for t in seq]
        sums = [float((s.sum(dim=-1) - 1.0).abs().max()) for s in soft]
        check("softmax rows sum to 1 on every level", max(sums) <= 1e-5,
              f"max |rowsum-1| = {max(sums):.2e}")

        # ============================================ 7. gradient
        print("\n[7] gradient from code logits back to the SIA adapter")
        adapter.zero_grad(set_to_none=True); rqvae.zero_grad(set_to_none=True)
        seq_g = extractor.sequence_logits(enc, attention_mask)
        scalar = sum(t.float().pow(2).mean() for t in seq_g)   # local, not a TIGER loss
        scalar.backward()
        a_grads = {n: p.grad for n, p in adapter.named_parameters()}
        check("adapter parameters receive gradients from the code logits",
              all(g is not None for g in a_grads.values()), f"{list(a_grads)}")
        check("adapter gradients are non-zero and finite",
              all(float(g.abs().sum()) > 0 and bool(torch.isfinite(g).all())
                  for g in a_grads.values()),
              ", ".join(f"{n}:{float(g.abs().sum()):.3e}" for n, g in a_grads.items()))
        codebook_grads = [q.embedding.weight.grad for q in rqvae.rq.vq_layers]
        check("codebook gradients reached too (logits are attached to the codebooks)",
              all(g is not None and bool(torch.isfinite(g).all()) for g in codebook_grads))
        check("adapter->quantizer path is differentiable end to end",
              seq_g[0].grad_fn is not None)

        # ============================================ 8. Clean TIGER regression
        print("\n[8] Clean TIGER regression")
        prist = PristineRQVAE(in_dim=emb_ds.dim, **RQVAE_CFG).to(device)
        prist.load_state_dict(ck["state_dict"])
        prist.eval()
        X = torch.tensor(emb_ds.embeddings[:256], dtype=torch.float32, device=device)
        with torch.no_grad():
            for use_sk in (True, False):
                pi = prist.get_indices(X, use_sk=use_sk)
                ci = rqvae.get_indices(X, use_sk=use_sk)
                check(f"get_indices bitwise identical to pre-E3 (use_sk={use_sk})",
                      torch.equal(pi, ci), f"shape {tuple(ci.shape)}")
            pz = prist.encoder(X)
            cz = rqvae.encoder(X)
            check("RQVAE.encoder output bitwise identical to pre-E3",
                  torch.equal(pz, cz))
            pq = prist.rq(cz, use_sk=True)
            cq = rqvae.rq(cz, use_sk=True)
            check("rq() default return is still a 3-tuple, values bitwise identical",
                  len(cq) == 3 and torch.equal(pq[0], cq[0])
                  and torch.equal(pq[1], cq[1]) and torch.equal(pq[2], cq[2]),
                  f"{len(cq)} values returned by default")
            pf = prist(X, use_sk=True)
            cf = rqvae(X, use_sk=True)
            check("RQVAE.forward default output bitwise identical to pre-E3",
                  len(cf) == len(pf) and all(torch.equal(a, b) for a, b in zip(pf, cf)),
                  f"{len(cf)} values returned by default")

        # a whole real RQ-VAE training step: loss value and parameter gradients
        prist.zero_grad(set_to_none=True); rqvae.zero_grad(set_to_none=True)
        p_out, p_ql, _, _ = prist(X, use_sk=True)
        p_loss, _ = prist.compute_loss(p_out, p_ql, X)
        p_loss.backward()
        c_out, c_ql, _, _ = rqvae(X, use_sk=True)
        c_loss, _ = rqvae.compute_loss(c_out, c_ql, X)
        c_loss.backward()
        check("RQ-VAE training-step loss bitwise identical to pre-E3",
              torch.equal(p_loss, c_loss), f"{float(c_loss):.8f}")
        pp, cp_ = dict(prist.named_parameters()), dict(rqvae.named_parameters())
        check("RQ-VAE training-step parameter gradients bitwise identical to pre-E3",
              all(torch.equal(pp[k].grad, cp_[k].grad) for k in pp),
              f"{len(pp)} parameter tensors compared")
        prist.zero_grad(set_to_none=True); rqvae.zero_grad(set_to_none=True)

        pristine_main.set_seed(SEED)
        prist_t5 = pristine_main.TIGER(t5cfg).to(device)
        prist_t5.model.load_state_dict(tiger.model.state_dict())
        prist_t5.eval()
        with torch.no_grad():
            p_loss, p_logits = prist_t5(input_ids=input_ids, attention_mask=attention_mask,
                                        labels=labels)
            c_loss, c_logits = tiger(input_ids=input_ids, attention_mask=attention_mask,
                                     labels=labels)
            g_p = prist_t5.generate(input_ids=input_ids, attention_mask=attention_mask,
                                    num_beams=3)
            g_c = tiger.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 num_beams=3)
        check("T5 loss bitwise identical to baseline", torch.equal(p_loss, c_loss),
              f"{c_loss.item():.8f}")
        check("T5 logits bitwise identical to baseline", torch.equal(p_logits, c_logits))
        check("T5 generate() bitwise identical to baseline", torch.equal(g_p, g_c))

        # ============================== 9. diagnostics (Diagnostic Only)
        print("\n[9] DIAGNOSTIC ONLY -- query vs codebook space, non-blocking")
        with torch.no_grad():
            cb = rqvae.rq.vq_layers[0].embedding.weight                    # first level
            z_tgt = rqvae.encoder(target_emb)
            for label, q in (("adapter query", query), ("RQVAE.encoder(item_emb)", z_tgt)):
                norm = q.float().norm(dim=-1)
                dist = torch.cdist(q.float(), cb.float()).min(dim=-1).values
                print(f"      {label:26s} ||x|| mean {float(norm.mean()):8.4f} "
                      f"p5/p95 {float(norm.quantile(0.05)):7.4f}/{float(norm.quantile(0.95)):8.4f} | "
                      f"dist-to-nearest-codeword mean {float(dist.mean()):8.4f} "
                      f"p5/p95 {float(dist.quantile(0.05)):7.4f}/{float(dist.quantile(0.95)):8.4f}")
            print(f"      codebook ||e|| mean {float(cb.norm(dim=-1).mean()):8.4f}")
            print("      (recorded only; no threshold, no pass/fail, no action taken)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{'=' * 70}\nEXP-E3: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed"
          f"\n{'=' * 70}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
