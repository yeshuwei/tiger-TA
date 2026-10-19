#!/usr/bin/env python
"""EXP-E4 - SIA loss mechanism test (Task E4, type A).

Checks the E4 mechanism only -- forward, both-side paths, backward, the
``lambda_sia = 0`` equivalence, and the loss's own correctness. Training the
model on real data (the type B part of EXP-E4 in integration/experiments.md) is
*not* in scope here; E4 is a mechanism Task.

Everything runs on a real Clean TIGER RQ-VAE checkpoint and a real TIGER batch,
so the codebooks under test are the frozen baseline's.

Read-only with respect to the repository: reads files, writes nothing outside a
temp dir. Run with the repo root as cwd:

    python -I integration/exp_e4_sia_loss.py                  # CPU
    EXP_E4_DEVICE=cuda python -I integration/exp_e4_sia_loss.py
"""
import os
import sys
import tempfile
import subprocess

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, "model")
RVQAE_DIR = os.path.join(REPO, "rqvae")
SMOKE = os.path.join(REPO, "clean_tiger", "_smoke", "Toys")
CKPT = os.path.join(REPO, "clean_tiger", "ckpt", "Toys", "rqvae",
                    "Oct-14-2026_05-55-09", "best_collision_model.pth")
BASELINE_COMMIT = "5edfca0"
SEED = 2025
BATCH = 8
LAMBDA = 0.1
DEVICE = os.environ.get("EXP_E4_DEVICE", "cpu")

RQVAE_CFG = dict(num_emb_list=[256, 256, 256], e_dim=32, layers=[512, 256, 128, 64],
                 dropout_prob=0.0, bn=False, loss_type="mse", quant_loss_weight=1.0,
                 beta=0.25, kmeans_init=True, kmeans_iters=100,
                 sk_epsilons=[0.0, 0.0, 0.003], sk_iters=50)
T5CFG = {'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
         'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1, 'vocab_size': 1025,
         'pad_token_id': 0, 'eos_token_id': 0, 'feed_forward_proj': 'relu',
         'max_len': 20}

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def ref_sym_kl(seq, tgt, temperature=1.0):
    """Independent numpy reference: symmetric KL per sample, shape (B,)."""
    def log_softmax(x):
        x = x / temperature
        m = x.max(axis=-1, keepdims=True)
        e = np.exp(x - m)
        return (x - m) - np.log(e.sum(axis=-1, keepdims=True))
    lp, lq = log_softmax(np.asarray(seq, dtype=np.float64)), log_softmax(np.asarray(tgt, dtype=np.float64))
    p, q = np.exp(lp), np.exp(lq)
    return (p * (lp - lq)).sum(-1) + (q * (lq - lp)).sum(-1)


def build_pristine(tmpdir):
    """Pre-E4 rqvae/models package and pre-E1..E4 model/main.py, from git."""
    pkg = os.path.join(tmpdir, "pristine_rqvae", "pristine_models")
    os.makedirs(pkg, exist_ok=True)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    for fn in ("rq.py", "vq.py", "rqvae.py", "layers.py"):
        src = subprocess.run(["git", "-C", REPO, "show",
                              f"{BASELINE_COMMIT}:rqvae/models/{fn}"],
                             capture_output=True, text=True, check=True).stdout
        with open(os.path.join(pkg, fn), "w") as f:
            f.write(src)
    src = subprocess.run(["git", "-C", REPO, "show", f"{BASELINE_COMMIT}:model/main.py"],
                         capture_output=True, text=True, check=True).stdout
    with open(os.path.join(tmpdir, "pristine_main.py"), "w") as f:
        f.write(src)
    sys.path.insert(0, os.path.join(tmpdir, "pristine_rqvae"))
    sys.path.insert(0, tmpdir)
    from pristine_models.rqvae import RQVAE as PristineRQVAE
    import pristine_main
    return PristineRQVAE, pristine_main


def main():
    # order matters: model/ and rqvae/ both contain a top-level `main.py`
    sys.path.insert(0, RVQAE_DIR)
    sys.path.insert(0, MODEL_DIR)
    sys.path.insert(0, REPO)
    from models.rqvae import RQVAE
    from datasets import EmbDataset
    from dataset import GenRecDataset
    from dataloader import GenRecDataLoader
    import main as current_main
    from tiger_ext.sia import SIAAdapter
    from tiger_ext.sia_codes import SIACodeExtractor
    from tiger_ext.sia_loss import SIALoss, SIAMount, symmetric_kl

    device = torch.device(DEVICE)
    tmpdir = tempfile.mkdtemp(prefix="exp_e4_")
    PristineRQVAE, pristine_main = build_pristine(tmpdir)

    sid_path = os.path.join(SMOKE, "Toys_clean_tiger_sid.npy")
    emb_ds = EmbDataset(os.path.join(SMOKE, "data", "Toys", "item_emb.parquet"))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    rqvae = RQVAE(in_dim=emb_ds.dim, **RQVAE_CFG).to(device)
    rqvae.load_state_dict(ck["state_dict"])
    rqvae.eval()
    sid_tokens = torch.as_tensor(np.load(sid_path), dtype=torch.long)
    item_embs = torch.as_tensor(emb_ds.embeddings, dtype=torch.float32)
    print(f"\nRQ-VAE: in_dim={emb_ds.dim} | items={len(emb_ds)} | SID {tuple(sid_tokens.shape)}")

    ds = GenRecDataset(dataset_path=os.path.join(SMOKE, "data", "Toys", "train.parquet"),
                       code_path=sid_path, mode='train', max_len=T5CFG['max_len'])
    batch = next(iter(GenRecDataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)))
    input_ids = batch['history'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['target'].to(device)

    current_main.set_seed(SEED)
    tiger = current_main.TIGER(T5CFG).to(device)
    adapter = SIAAdapter(T5CFG['d_model']).to(device)
    mount = SIAMount(adapter, rqvae, item_embs, sid_tokens,
                     lambda_sia=LAMBDA, temperature=1.0).to(device)
    print(f"batch: history {tuple(input_ids.shape)} attention_mask {tuple(attention_mask.shape)} "
          f"target {tuple(labels.shape)} | lambda_sia={mount.lambda_sia} enabled={mount.enabled}\n")

    # =========================================================== 1. forward
    print("[1] forward on a real batch")
    captured = {"rq_queries": []}

    def note(name):
        def hook(module, inputs, output=None):
            captured[name] = captured.get(name, 0) + 1
        return hook

    def note_adapter(module, inputs, output):
        captured["adapter_out"] = output

    def note_encoder(module, inputs, output):
        captured["enc_out"] = output

    def note_rq_query(module, args):
        captured["rq_queries"].append(args[0])

    handles = [adapter.register_forward_hook(note_adapter),
               rqvae.encoder.register_forward_hook(note_encoder),
               rqvae.rq.register_forward_pre_hook(note_rq_query),
               rqvae.decoder.register_forward_hook(note("decoder_calls"))]

    t5_loss, logits, encoder_hidden, _ = tiger(input_ids=input_ids, attention_mask=attention_mask,
                                               labels=labels, return_hidden=True)
    sia_total = mount(encoder_hidden, attention_mask, labels)
    for h in handles:
        h.remove()

    query = mount.extractor.to_query(encoder_hidden, attention_mask)
    rows = mount.rows_for_labels(labels)
    target_latent = rqvae.encoder(mount.target_embeddings[rows])
    seq_logits = mount.extractor.sequence_logits(encoder_hidden, attention_mask)
    tgt_logits = mount.target_side_logits(labels)
    l_sia = mount.loss_fn(seq_logits, tgt_logits)

    check("sequence query shape is (B, 32)=(8, 32)", tuple(query.shape) == (BATCH, 32),
          f"got {tuple(query.shape)}")
    check("target latent shape is (B, 32)=(8, 32)  [RQVAE.encoder output]",
          tuple(target_latent.shape) == (BATCH, 32), f"got {tuple(target_latent.shape)}")
    check("both sides yield 3 levels of (B, 256)",
          len(seq_logits) == 3 and len(tgt_logits) == 3 and
          all(tuple(t.shape) == (BATCH, 256) for t in seq_logits + tgt_logits),
          f"seq {[tuple(t.shape) for t in seq_logits]}")
    check("SIA loss is a scalar (0-dim tensor)", l_sia.dim() == 0, f"dim={l_sia.dim()}")
    check("SIA loss is finite (no NaN / Inf)",
          bool(torch.isfinite(l_sia).all()), f"L_SIA={float(l_sia):.6f}")
    check("weighted SIA term = lambda_sia * L_SIA, finite",
          bool(torch.isfinite(sia_total).all()) and
          abs(float(sia_total) - LAMBDA * float(l_sia)) < 1e-9,
          f"lambda*L_SIA={float(sia_total):.6f}")
    check("all logits finite (no NaN / Inf on either side)",
          all(bool(torch.isfinite(t).all()) for t in seq_logits + tgt_logits))
    check("recommendation loss still computes alongside SIA",
          bool(torch.isfinite(t5_loss).all()), f"L_T5={float(t5_loss):.6f}")

    # ================================================== 2. two-side path checks
    print("\n[2] path assertions (sequence side skips RQVAE.encoder)")
    rq_queries = captured.get("rq_queries", [])
    check("sequence side: adapter output IS the tensor that enters the RQ quantizer",
          len(rq_queries) == 2 and rq_queries[0] is captured["adapter_out"],
          f"rq calls={len(rq_queries)}, identity={len(rq_queries) == 2 and rq_queries[0] is captured.get('adapter_out')}")
    check("target side: RQVAE.encoder output IS the tensor that enters the RQ quantizer",
          len(rq_queries) == 2 and rq_queries[1] is captured["enc_out"])
    check("the two legs feed different tensors into the same quantizer (not confused)",
          len(rq_queries) == 2 and rq_queries[0] is not rq_queries[1])
    check("both legs use the same RQ module / same codebooks (identity, not a copy)",
          mount.extractor.rqvae is rqvae and rqvae.rq.num_quantizers == 3)

    seq_captured = {}
    h_enc_c = rqvae.encoder.register_forward_hook(
        lambda m, i, o: seq_captured.update(encoder_calls=seq_captured.get("encoder_calls", 0) + 1))
    h_dec_c = rqvae.decoder.register_forward_hook(
        lambda m, i, o: seq_captured.update(decoder_calls=seq_captured.get("decoder_calls", 0) + 1))
    mount.extractor.sequence_logits(encoder_hidden, attention_mask)
    h_enc_c.remove(); h_dec_c.remove()
    check("sequence side never calls RQVAE.encoder",
          seq_captured.get("encoder_calls", 0) == 0,
          f"encoder calls = {seq_captured.get('encoder_calls', 0)}")

    tgt_captured = {}
    h_enc_c = rqvae.encoder.register_forward_hook(
        lambda m, i, o: tgt_captured.update(encoder_calls=tgt_captured.get("encoder_calls", 0) + 1))
    mount.target_side_logits(labels)
    h_enc_c.remove()
    check("target side calls RQVAE.encoder exactly once",
          tgt_captured.get("encoder_calls", 0) == 1,
          f"encoder calls = {tgt_captured.get('encoder_calls', 0)}")
    check("no side touches RQVAE.decoder during the SIA path",
          captured.get("decoder_calls", 0) == 0 and seq_captured.get("decoder_calls", 0) == 0)

    check("aligned space is the codebook (256), NOT the T5 vocabulary (1025)",
          seq_logits[0].shape[-1] == 256 and T5CFG['vocab_size'] != 256,
          f"logit width {seq_logits[0].shape[-1]} vs vocab {T5CFG['vocab_size']}")

    # ============================================================ 3. backward
    print("\n[3] backward through the SIA loss")
    adapter.zero_grad(); rqvae.zero_grad(); tiger.zero_grad()
    sia_total.backward()
    ad_grads = {n: p.grad for n, p in adapter.named_parameters()}
    enc_grads = [p.grad for p in tiger.model.encoder.parameters()]
    # `decoder.embed_tokens.weight` IS `model.shared`, the encoder/decoder tied
    # input embedding, so the encoder path legitimately touches it; the decoder
    # stack itself must stay untouched.
    dec_named = {n: p.grad for n, p in tiger.model.decoder.named_parameters()
                 if not n.startswith("embed_tokens")}
    rq_grads = [p.grad for p in rqvae.parameters()]
    dec_grads = list(dec_named.values())
    check("backward succeeds from L_SIA",
          all(g is not None for g in ad_grads.values()))
    check("adapter parameters receive gradients",
          all(g is not None for g in ad_grads.values()), f"{len(ad_grads)} tensors")
    check("adapter gradients are non-zero",
          all(float(g.abs().sum()) > 0 for g in ad_grads.values()),
          ", ".join(f"{n}:{float(g.abs().sum()):.3e}" for n, g in ad_grads.items()))
    check("T5 encoder path receives gradients",
          sum(g is not None for g in enc_grads) > 0,
          f"{sum(g is not None for g in enc_grads)}/{len(enc_grads)} encoder tensors with grad")
    check("T5 decoder stack gets NO gradient from L_SIA (it is not a decoder-hidden loss)",
          all(g is None for g in dec_grads),
          f"{sum(g is not None for g in dec_grads)}/{len(dec_grads)} decoder tensors with grad "
          f"(excluding tied embed_tokens)")
    check("target-side RQ-VAE path also gets gradients (freeze policy is E7, not E4)",
          sum(g is not None for g in rq_grads) > 0,
          f"{sum(g is not None for g in rq_grads)}/{len(rq_grads)} RQ-VAE tensors with grad")
    all_grads = [g for g in list(ad_grads.values()) + enc_grads + dec_grads + rq_grads if g is not None]
    check("every gradient is finite (no NaN / Inf)",
          all(bool(torch.isfinite(g).all()) for g in all_grads),
          f"{len(all_grads)} gradient tensors checked")

    # =========================================================== 5. loss correctness
    print("\n[5] SIA loss correctness (independent reference + reduction)")
    torch.manual_seed(0)
    s = torch.randn(6, 256, device=device) * 2.0
    t = torch.randn(6, 256, device=device) * 2.0
    one = SIALoss()
    check("identical distributions give L_SIA = 0 exactly",
          float(one([s, s, s], [s, s, s])) == 0.0)
    # the reference is float64; run the module in float64 too so the comparison
    # tests the formula rather than float32 rounding
    got = symmetric_kl(s.double(), t.double(), 1.0).cpu().numpy()
    want = ref_sym_kl(s.cpu().numpy(), t.cpu().numpy(), 1.0)
    check("matches the independent numpy reference (T=1, float64)",
          float(np.abs(got - want).max()) < 1e-12, f"max abs diff {np.abs(got - want).max():.2e}")
    got2 = symmetric_kl(s.double(), t.double(), 2.5).cpu().numpy()
    want2 = ref_sym_kl(s.cpu().numpy(), t.cpu().numpy(), 2.5)
    check("temperature enters at the softmax, matching the reference (T=2.5, float64)",
          float(np.abs(got2 - want2).max()) < 1e-12, f"max abs diff {np.abs(got2 - want2).max():.2e}")
    f32_gap = float(np.abs(symmetric_kl(s, t, 1.0).double().cpu().numpy() - want).max())
    check("float32 path agrees with the reference to float32 precision",
          f32_gap < 1e-5, f"max abs diff {f32_gap:.2e} on values of order {want.mean():.2f}")
    check("symmetry: KL(s,t) == KL(t,s)",
          float((symmetric_kl(s, t, 1.0) - symmetric_kl(t, s, 1.0)).abs().max()) == 0.0)
    check("non-negative everywhere",
          float(symmetric_kl(s, t, 1.0).min()) >= 0.0)

    # float32 reductions over 256 classes cannot agree tighter than ~1e-6 here,
    # so these use a relative tolerance; the exact formula is pinned above in float64
    per_sample = [float(one([s[i:i + 1]] * 3, [t[i:i + 1]] * 3)) for i in range(6)]
    full = float(one([s, s, s], [t, t, t]))
    check("batch mean equals the mean of per-sample losses",
          abs(full - float(np.mean(per_sample))) < 1e-5 * max(abs(full), 1.0),
          f"batch {full:.8f} vs singles {np.mean(per_sample):.8f}")
    dup = [torch.cat([s, s]), torch.cat([s, s]), torch.cat([s, s])]
    dpt = [torch.cat([t, t]), torch.cat([t, t]), torch.cat([t, t])]
    check("batch dimension is reduced, not summed (duplicating the batch keeps the value)",
          abs(float(one(dup, dpt)) - full) < 1e-5 * max(abs(full), 1.0),
          f"B=6 {full:.8f} vs B=12 {float(one(dup, dpt)):.8f}")
    perm = torch.randperm(6, device=device)
    check("permutation invariant over the batch",
          abs(float(one([s[perm]] * 3, [t[perm]] * 3)) - full) < 1e-5 * max(abs(full), 1.0),
          f"permuted {float(one([s[perm]] * 3, [t[perm]] * 3)):.8f} vs plain {full:.8f}")

    levels_seq = [s, s * 1.0, s]
    knob = SIALoss(level_reduction="sum")
    three = one(levels_seq, [t, t, t])
    summed = knob(levels_seq, [t, t, t])
    check("level reduction: mean == per-level average, sum == 3 x mean",
          abs(float(summed) - 3 * float(three)) < 1e-6,
          f"mean {float(three):.6f} sum {float(summed):.6f}")

    noise = torch.randn_like(s)
    prev, monotone = -1.0, True
    for alpha in (0.0, 0.05, 0.2, 0.5, 1.0, 2.0):
        val = float(one([s, s, s], [s + alpha * noise, s + alpha * noise, s + alpha * noise]))
        if val < prev - 1e-9:
            monotone = False
        prev = val
    check("L_SIA increases monotonically as the distributions diverge", monotone)

    check("loss on the real batch is not degenerate (non-zero)",
          float(l_sia) > 0.0, f"L_SIA={float(l_sia):.6f}")

    # ================================================= 4. zero-lambda equivalence
    print("\n[4] lambda_sia = 0 equivalence (recommender loop)")
    from torch.utils.data import Subset
    small = Subset(ds, list(range(0, 4 * BATCH)))

    def run_step(lam):
        """One real epoch of model/main.py train(); lam=None means SIA not passed in."""
        current_main.set_seed(SEED)
        m = current_main.TIGER(T5CFG).to(device)
        sia = None
        calls = {"mount": 0}
        if lam is not None:
            ad = SIAAdapter(T5CFG['d_model']).to(device)
            sia = SIAMount(ad, rqvae, item_embs, sid_tokens, lambda_sia=lam).to(device)
            orig_forward = SIAMount.forward

            def counting_forward(self, *args, **kwargs):
                calls["mount"] += 1
                return orig_forward(self, *args, **kwargs)

            sia.forward = counting_forward.__get__(sia, SIAMount)
        current_main.set_seed(SEED + 1)
        loader = GenRecDataLoader(small, batch_size=BATCH, shuffle=False, num_workers=0)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        avg = current_main.train(m, loader, opt, device, sia=sia)
        grads = {n: (p.grad.clone() if p.grad is not None else None)
                 for n, p in m.named_parameters()}
        return avg, grads, calls["mount"]

    clean_avg, clean_grads, clean_calls = run_step(None)
    zero_avg, zero_grads, zero_calls = run_step(0.0)
    check("lambda_sia = 0 leaves the epoch loss bitwise identical to Clean TIGER",
          clean_avg == zero_avg, f"clean {clean_avg!r} vs lambda=0 {zero_avg!r}")
    same = all((clean_grads[n] is None and zero_grads[n] is None) or
               (clean_grads[n] is not None and zero_grads[n] is not None and
                torch.equal(clean_grads[n], zero_grads[n])) for n in clean_grads)
    check("lambda_sia = 0 leaves every parameter gradient bitwise identical",
          same, f"{len(clean_grads)} parameter tensors compared")
    check("lambda_sia = 0 never enters the SIA path (mount not invoked)",
          clean_calls == 0 and zero_calls == 0, f"clean {clean_calls}, lambda=0 {zero_calls}")

    current_main.set_seed(SEED)
    m_a = current_main.TIGER(T5CFG).to(device)
    m_b = current_main.TIGER(T5CFG).to(device)
    m_b.load_state_dict(m_a.state_dict())
    m_a.train(); m_b.train()
    current_main.set_seed(SEED + 7)
    la, loa = m_a(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    current_main.set_seed(SEED + 7)
    lb, lob, _, _ = m_b(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                        return_hidden=True)
    check("in train mode, return_hidden=True does not perturb loss/logits (dropout RNG untouched)",
          torch.equal(la, lb) and torch.equal(loa, lob), f"loss {float(la):.8f}")

    # ================================================= 6. original TIGER regression
    print("\n[6] original TIGER recommendation path")
    current_main.set_seed(SEED)
    cur = current_main.TIGER(T5CFG).to(device)
    current_main.set_seed(SEED)
    pri = pristine_main.TIGER(T5CFG).to(device)
    pri.load_state_dict(cur.state_dict())
    cur.eval(); pri.eval()
    lc, oc = cur(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    lp, op = pri(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    check("loss / logits bitwise identical to pre-E4 TIGER",
          torch.equal(lc, lp) and torch.equal(oc, op), f"loss {float(lc):.8f}")
    gc = cur.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5)
    gp = pri.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5)
    check("generate() bitwise identical to pre-E4 TIGER",
          torch.equal(gc, gp), f"shape {tuple(gc.shape)}")
    check("SIA adds no parameter to the TIGER module (adapter lives outside it)",
          sum(p.numel() for p in cur.parameters()) == sum(p.numel() for p in pri.parameters()),
          f"{sum(p.numel() for p in cur.parameters())} params")
    check("no SIA loss was added to the default forward (still 2 values out)",
          len(cur(input_ids=input_ids, attention_mask=attention_mask, labels=labels)) == 2)

    # ============================================================ 7. diagnostics
    print("\n[7] DIAGNOSTIC ONLY -- query / target scale (recorded, no action)")
    q = query.detach().float().cpu().numpy()
    tl = target_latent.detach().float().cpu().numpy()
    cb = rqvae.rq.vq_layers[0].embedding.weight.detach().float().cpu().numpy()
    d_q = np.linalg.norm(q[:, None, :] - cb[None, :, :], axis=-1).min(axis=-1)
    d_t = np.linalg.norm(tl[:, None, :] - cb[None, :, :], axis=-1).min(axis=-1)
    ad_w = float(adapter.proj.weight.grad.abs().sum())
    print(f"      adapter query            ||x|| mean {np.linalg.norm(q, axis=-1).mean():7.4f} | "
          f"dist-to-nearest-codeword mean {d_q.mean():7.4f}")
    print(f"      RQVAE.encoder(item_emb)  ||x|| mean {np.linalg.norm(tl, axis=-1).mean():7.4f} | "
          f"dist-to-nearest-codeword mean {d_t.mean():7.4f}")
    print(f"      L_SIA (lambda={LAMBDA})        {float(l_sia):.6f}   per level "
          f"{[round(float(v), 6) for v in torch.stack([symmetric_kl(a, b).mean() for a, b in zip(seq_logits, tgt_logits)]).cpu()]}")
    print(f"      grad scale: adapter proj.weight {ad_w:.3e}, proj.bias "
          f"{float(adapter.proj.bias.grad.abs().sum()):.3e}, "
          f"T5 encoder {sum(float(g.abs().sum()) for g in enc_grads if g is not None):.3e}, "
          f"RQ-VAE {sum(float(g.abs().sum()) for g in rq_grads if g is not None):.3e}")
    print("      (recorded only; no threshold, no pass/fail, no action taken)")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 70)
    print(f"EXP-E4: {passed}/{len(RESULTS)} checks passed")
    print("=" * 70)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
