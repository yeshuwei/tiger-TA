#!/usr/bin/env python
"""EXP-E5 - PSA decoder interface + loss mechanism test (Task E5/E6, type A).

Checks the PSA mechanism only, on a real Toys batch and a real Clean TIGER
RQ-VAE checkpoint:

    sequence side:  T5 decoder_hidden_states[-1][:, 0, :] -> PSAAdapter -> (B, D_target)
    target side:    item content embedding -> RQVAE.forward -> out = decoder(x_q)

plus the frozen E6 loss (symmetric contrastive loss, in-batch negatives) and the
``lambda_psa = 0`` equivalence of the recommender loop. Training on real data
(the type B half of EXP-E5/E6) is *not* in scope.

The loss follows ETEGRec's official implementation exactly: cosine similarity
(ETEGRec ``sim: cos``), temperature 0.07 (its call site's static default), and
the two cross-entropy directions summed with no 0.5 factor.

Read-only with respect to the repository: reads files, writes nothing outside a
temp dir. Run with the repo root as cwd:

    python -I integration/exp_e5_psa.py                  # CPU
    EXP_E5_DEVICE=cuda python -I integration/exp_e5_psa.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, "model")
RVQAE_DIR = os.path.join(REPO, "rqvae")
SMOKE = os.path.join(REPO, "clean_tiger", "_smoke", "Toys")

# ---------------------------------------------------------------- fixture
# Default = the fixture that matches frozen plan.md §1.3 (Toys D_target = 4096,
# i.e. Clean TIGER's content-only data/Toys/item_emb.parquet) with the only
# 4096-d Toys RQ-VAE that exists in the repo (the clean-tiger smoke checkpoint).
# Every path can be overridden so the same checks can run against another real
# fixture (e.g. the trained 896-d TA-SID RQ-VAE, where the decoder is healthy).
DATA_DIR = os.environ.get("EXP_E5_DATA_DIR", os.path.join(SMOKE, "data", "Toys"))
EMB_PATH = os.environ.get("EXP_E5_EMB", os.path.join(DATA_DIR, "item_emb.parquet"))
SID_PATH = os.environ.get("EXP_E5_SID",
                          os.path.join(SMOKE, "Toys_clean_tiger_sid.npy"))
CKPT = os.environ.get("EXP_E5_RQVAE_CKPT",
                      os.path.join(REPO, "clean_tiger", "ckpt", "Toys", "rqvae",
                                   "Oct-14-2026_05-55-09", "best_collision_model.pth"))
FIXTURE_TAG = os.environ.get("EXP_E5_FIXTURE_TAG", os.path.relpath(CKPT, REPO))
BASELINE_COMMIT = "5edfca0"
SEED = 2025
BATCH = 8
LAMBDA = 0.5
DEVICE = os.environ.get("EXP_E5_DEVICE", "cpu")

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


def ref_symmetric_contrastive(seq, tgt, temperature=0.07):
    """Independent numpy reference for the frozen E6 loss (float64).

    ETEGRec fidelity: cosine similarity (both sides L2-normalised),
    temperature 0.07 (the call site's static default), and the two
    cross-entropy directions *summed* -- ``cl(a, b) + cl(b, a)`` has no 0.5.
    """
    seq = np.asarray(seq, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    seq = seq / np.linalg.norm(seq, axis=-1, keepdims=True)
    tgt = tgt / np.linalg.norm(tgt, axis=-1, keepdims=True)
    logits = seq @ tgt.T / temperature

    def ce(x):
        m = x.max(axis=-1, keepdims=True)
        lse = m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True))
        return float((lse - np.diag(x)[:, None]).mean())

    return ce(logits) + ce(logits.T)


def build_pristine(tmpdir):
    """Pre-E1..E5 model/main.py from git, for the bitwise regression check."""
    src = subprocess.run(["git", "-C", REPO, "show", f"{BASELINE_COMMIT}:model/main.py"],
                         capture_output=True, text=True, check=True).stdout
    with open(os.path.join(tmpdir, "pristine_main.py"), "w") as f:
        f.write(src)
    sys.path.insert(0, tmpdir)
    import pristine_main
    return pristine_main


def main():
    sys.path.insert(0, RVQAE_DIR)
    sys.path.insert(0, MODEL_DIR)
    sys.path.insert(0, REPO)
    from models.rqvae import RQVAE
    from datasets import EmbDataset
    from dataset import GenRecDataset
    from dataloader import GenRecDataLoader
    import main as current_main
    from tiger_ext.psa import PSAAdapter, psa_position0
    from tiger_ext.psa_loss import PSALoss, PSAMount, symmetric_contrastive_loss

    device = torch.device(DEVICE)
    tmpdir = tempfile.mkdtemp(prefix="exp_e5_")
    pristine_main = build_pristine(tmpdir)

    sid_path = SID_PATH
    emb_ds = EmbDataset(EMB_PATH)
    rqvae = RQVAE(in_dim=emb_ds.dim, **RQVAE_CFG).to(device)
    rqvae.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"])
    for _vq in rqvae.rq.vq_layers:
        _vq.initted = True     # ckpt does not carry this flag; without it a later
                               # train() would lazily re-run k-means over the codebook
    rqvae.eval()
    sid_tokens = torch.as_tensor(np.load(sid_path), dtype=torch.long)
    item_embs = torch.as_tensor(emb_ds.embeddings, dtype=torch.float32)
    d_target = emb_ds.dim
    print(f"\nfixture: {FIXTURE_TAG}")
    print(f"  emb  {os.path.relpath(EMB_PATH, REPO)}")
    print(f"  ckpt {os.path.relpath(CKPT, REPO)}")
    print(f"  sid  {os.path.relpath(sid_path, REPO)}")
    print(f"  ints {os.path.relpath(DATA_DIR, REPO)}/train.parquet")
    print(f"RQ-VAE: in_dim={d_target} (= D_target) | items={len(emb_ds)} | "
          f"SID {tuple(sid_tokens.shape)}")

    # Fixture-level diagnostic (recorded, not a gate): how much of the item
    # content does this decoder actually reconstruct? E6's target is exactly
    # that reconstruction, so a decoder that outputs ~one constant vector makes
    # L_PSA near-trivial. Measured on the first 2048 items.
    with torch.no_grad():
        _X = item_embs[:2048].to(device)
        _out, _, _, _ = rqvae(_X)
        _mse = float(((_out - _X) ** 2).mean())
        _var = float(_X.var())
        print(f"  recon fidelity (2048 items): MSE {_mse:.8f} vs raw var {_var:.8f} "
              f"-> R2 {1 - _mse / _var:+.4f} | |recon|/|x| "
              f"{float(_out.norm(dim=1).mean() / _X.norm(dim=1).mean()):.4f}")

    ds = GenRecDataset(dataset_path=os.path.join(DATA_DIR, "train.parquet"),
                       code_path=sid_path, mode='train', max_len=T5CFG['max_len'])
    batch = next(iter(GenRecDataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)))
    input_ids = batch['history'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['target'].to(device)

    current_main.set_seed(SEED)
    tiger = current_main.TIGER(T5CFG).to(device)
    tiger.eval()
    adapter = PSAAdapter(T5CFG['d_model'], d_target).to(device)
    mount = PSAMount(adapter, rqvae, item_embs, sid_tokens,
                     lambda_psa=LAMBDA).to(device)   # temperature: ETEGRec default 0.07
    print(f"batch: history {tuple(input_ids.shape)} target {tuple(labels.shape)} | "
          f"D_target={d_target} lambda_psa={mount.lambda_psa} enabled={mount.enabled}\n")

    # ================================================== 1. decoder hidden source
    print("[1] decoder hidden acquisition")
    captured = {}

    def note_decoder(module, inputs, output):
        captured["decoder_hidden_states"] = output.hidden_states

    def note_adapter_input(module, args):
        captured["adapter_in"] = args[0]

    h_dec = tiger.model.decoder.register_forward_hook(note_decoder)
    h_psi = adapter.register_forward_pre_hook(note_adapter_input)
    t5_loss, logits, encoder_hidden, decoder_hidden = tiger(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_hidden=True)
    psa_total = mount(decoder_hidden, labels)
    h_dec.remove(); h_psi.remove()

    hs = captured["decoder_hidden_states"]
    check("decoder hidden has shape (B, L_dec, d_model) = (8, 4, 128)",
          tuple(decoder_hidden.shape) == (BATCH, 4, T5CFG['d_model']),
          f"got {tuple(decoder_hidden.shape)}")
    check("PSA input IS decoder_hidden_states[-1] (last layer, identity within the forward)",
          hs[-1] is decoder_hidden)
    check("PSA input is NOT decoder layer 0",
          hs[0] is not decoder_hidden and not torch.equal(hs[0], decoder_hidden))
    check("PSA input is NOT the encoder hidden",
          tuple(encoder_hidden.shape) != tuple(decoder_hidden.shape))
    check("PSA adapter receives the decoder hidden tensor itself (no re-selection upstream)",
          captured["adapter_in"] is decoder_hidden)

    psa_rep = adapter(decoder_hidden)
    proj = adapter.proj
    with torch.no_grad():
        manual_pos0 = proj(decoder_hidden[:, 0, :])
        manual_pos1 = proj(decoder_hidden[:, 1, :])
    check("position 0 is what is projected: adapter(h) == proj(h[:, 0, :]) exactly",
          torch.equal(psa_rep, manual_pos0))
    check("... and it is not some other position (proj(h[:, 1, :]) differs)",
          not torch.equal(psa_rep, manual_pos1))
    check("psa_position0 keeps only the sequence dim (B, d_model)",
          tuple(psa_position0(decoder_hidden).shape) == (BATCH, T5CFG['d_model']))

    # ============================================================ 2. both sides
    print("\n[2] PSA representation and target-side reconstruction")
    check(f"PSA representation shape is (B, D_target) = ({BATCH}, {d_target})",
          tuple(psa_rep.shape) == (BATCH, d_target), f"got {tuple(psa_rep.shape)}")

    rows = mount.rows_for_labels(labels)
    raw_target = mount.target_embeddings[rows]
    scale_captured = {}

    def note_encoder_call(module, inputs, output):
        scale_captured["encoder_calls"] = scale_captured.get("encoder_calls", 0) + 1

    def note_decoder_input(module, args):
        scale_captured["into_rqvae_decoder"] = args[0]

    h_e = rqvae.encoder.register_forward_hook(note_encoder_call)
    h_d = rqvae.decoder.register_forward_pre_hook(note_decoder_input)
    target_rep = mount.target_representation(labels)
    h_e.remove(); h_d.remove()

    check(f"target representation shape is (B, D_target) = ({BATCH}, {d_target})",
          tuple(target_rep.shape) == (BATCH, d_target), f"got {tuple(target_rep.shape)}")
    check("target side goes through RQVAE.encoder exactly once",
          scale_captured.get("encoder_calls", 0) == 1,
          f"encoder calls = {scale_captured.get('encoder_calls', 0)}")
    check("RQ-VAE decoder is fed the quantised latent x_q (reconstruction, not the raw input)",
          scale_captured["into_rqvae_decoder"] is not None and
          tuple(scale_captured["into_rqvae_decoder"].shape) == (BATCH, RQVAE_CFG["e_dim"]),
          f"shape {tuple(scale_captured['into_rqvae_decoder'].shape)} vs raw (8, {d_target})")
    with torch.no_grad():
        out_check = rqvae(raw_target)[0]
    check("target representation IS RQVAE.forward's first return value (decoder(x_q))",
          torch.equal(target_rep, out_check))
    check("target representation is a reconstruction, not the raw content embedding",
          not torch.equal(target_rep, raw_target),
          f"mean |x - recon| = {float((raw_target - target_rep).abs().mean()):.4f} on "
          f"raw scale {float(raw_target.abs().mean()):.4f}")

    # ============================================================ 3. the loss
    print("\n[3] PSA loss correctness")
    l_psa = mount.loss_fn(psa_rep, target_rep)
    check("L_PSA is a scalar (0-dim tensor)", l_psa.dim() == 0, f"dim={l_psa.dim()}")
    check("L_PSA is finite (no NaN / Inf)", bool(torch.isfinite(l_psa).all()),
          f"L_PSA={float(l_psa):.6f}")
    check("weighted PSA term = lambda_psa * L_PSA",
          abs(float(psa_total) - LAMBDA * float(l_psa)) < 1e-6,
          f"lambda*L_PSA={float(psa_total):.6f}")

    torch.manual_seed(0)
    s = (torch.randn(6, 32, device=device) * 0.5).double()
    t = (torch.randn(6, 32, device=device) * 0.5).double()
    got = float(symmetric_contrastive_loss(s, t).cpu())
    want = ref_symmetric_contrastive(s.cpu().numpy(), t.cpu().numpy())
    check("matches the independent numpy reference (cosine, T=0.07, float64)",
          abs(got - want) < 1e-9, f"got {got:.9f} vs ref {want:.9f}")
    got25 = float(symmetric_contrastive_loss(s, t, 2.5).cpu())
    want25 = ref_symmetric_contrastive(s.cpu().numpy(), t.cpu().numpy(), 2.5)
    check("temperature divides the cosine similarity, matching the reference (T=2.5)",
          abs(got25 - want25) < 1e-9, f"got {got25:.9f} vs ref {want25:.9f}")

    # ----- the three ETEGRec fidelity points this revision pinned down -----
    check("default temperature is 0.07 everywhere (ETEGRec's call site never overrides it)",
          symmetric_contrastive_loss.__defaults__[0] == 0.07 and
          PSALoss().temperature == 0.07 and
          PSAMount.__init__.__defaults__[-1] == 0.07 and
          mount.loss_fn.temperature == 0.07,
          f"fn {symmetric_contrastive_loss.__defaults__[0]} | PSALoss {PSALoss().temperature} "
          f"| mount ctor {PSAMount.__init__.__defaults__[-1]} | live mount "
          f"{mount.loss_fn.temperature}")

    sn = F.normalize(s, dim=-1)
    tn = F.normalize(t, dim=-1)
    cos_logits = sn @ tn.t() / 0.07
    pos = torch.arange(6, device=device)
    ce_fwd = F.cross_entropy(cos_logits, pos)
    ce_bwd = F.cross_entropy(cos_logits.t(), pos)
    check("symmetric term is the SUM of the two directions, not their average "
          "(ETEGRec computes cl(a,b) + cl(b,a))",
          abs(got - float(ce_fwd + ce_bwd)) < 1e-12 and
          abs(got - 0.5 * float(ce_fwd + ce_bwd)) > 1e-6,
          f"got {got:.9f} | ce_fwd+ce_bwd {float(ce_fwd + ce_bwd):.9f} | "
          f"0.5*sum {0.5 * float(ce_fwd + ce_bwd):.9f}")

    rescaled = float(symmetric_contrastive_loss(s * 7.0, t * 0.01))
    check("similarity is cosine, not a raw dot product: rescaling either side "
          "leaves the loss unchanged",
          abs(rescaled - got) < 1e-9,
          f"rescaling seq x7 and tgt x0.01 -> {rescaled:.9f} vs {got:.9f}")
    check("symmetry: swapping the two sides gives the same loss",
          abs(float(symmetric_contrastive_loss(s, t, 1.0)) -
              float(symmetric_contrastive_loss(t, s, 1.0))) == 0.0)

    # in-batch correspondence: the positive of row i is row i, and nothing else
    one = PSALoss()
    aligned = one(s, s)
    perm = torch.randperm(6, device=device)
    shuffled = one(s, s[perm])
    check("row i's positive is row i (misaligning the pairs raises the loss)",
          float(shuffled) > float(aligned) + 1e-6,
          f"aligned {float(aligned):.6f} vs misaligned {float(shuffled):.6f}")
    check("perfect alignment is the minimum over target permutations",
          all(float(aligned) <= float(one(s, s[p])) + 1e-9
              for p in (torch.roll(perm, 1), perm, perm.flip(0))),
          f"aligned {float(aligned):.6f}")
    joint = one(s[perm], s[perm])
    check("permuting sequences and targets jointly leaves the loss unchanged",
          abs(float(joint) - float(aligned)) < 1e-12)
    big = torch.randn(6, 32, device=device).double()
    check("unrelated targets cost more than aligned ones",
          float(one(s, big)) > float(aligned), f"{float(one(s, big)):.6f}")

    # temperature and batch-size behaviour (acceptance ③), against a closed form.
    # Under cosine the similarity matrix is built directly: pairs whose cosine is
    # exactly `c` on the diagonal and 0 off it give, in both directions,
    # CE = log(1 + (n-1) * exp(-c / T)); the loss sums the two, hence the 2x.
    def closed_form(n, c, T):
        return 2.0 * float(np.log(1.0 + (n - 1) * np.exp(-c / T)))

    def controlled(n, c):
        """(n, 2n) pairs with cosine c on the diagonal and exactly 0 elsewhere.

        Built in float64 throughout: an accidental float32 rounding of the
        ``sqrt(1 - c^2)`` block already shifts the cosine by ~1e-8, which the
        closed-form comparison at T=0.07 would then read as a real mismatch.
        """
        eye = torch.eye(n, device=device, dtype=torch.float64)
        tgt = torch.cat([eye, torch.zeros(n, n, device=device, dtype=torch.float64)], dim=1)
        seq = torch.cat([c * eye, float(np.sqrt(1.0 - c ** 2)) * eye], dim=1)
        return seq, tgt

    check("controlled construction really has cosine c on the diagonal and 0 off it",
          all(abs(float((F.normalize(controlled(n, c)[0], dim=-1) @
                         F.normalize(controlled(n, c)[1], dim=-1).t())[i, j]) -
                  (c if i == j else 0.0)) < 1e-12
              for n, c in ((6, 0.5), (4, 1.0)) for i in range(n) for j in range(n)))

    exact_ok = True
    for n, c, T in ((6, 0.5, 1.0), (6, 0.5, 0.07), (16, 0.75, 0.5), (4, 0.25, 0.25)):
        seqc, tgtc = controlled(n, c)
        got_c = float(PSALoss(temperature=T)(seqc, tgtc))
        exact_ok = exact_ok and abs(got_c - closed_form(n, c, T)) < 1e-9
    check("matches the closed form 2*log(1 + (n-1) exp(-cos/T)) on a controlled cosine matrix",
          exact_ok)
    temps = {T: float(PSALoss(temperature=T)(*controlled(6, 0.5))) for T in (0.25, 1.0, 4.0)}
    two_log6 = 2.0 * np.log(6)
    check("temperature behaviour is sane: high T -> uniform (2 log n), low T -> sharper",
          temps[0.25] < temps[1.0] < temps[4.0] < two_log6 + 1e-9 and
          abs(temps[4.0] - two_log6) < abs(temps[0.25] - two_log6),
          " ".join(f"T={t}:{v:.4f}" for t, v in temps.items()) + f" | 2 log 6 {two_log6:.4f}")
    sizes = {n: float(one(*controlled(n, 0.5))) for n in (2, 4, 8, 16)}
    check("batch-size behaviour is sane: more in-batch negatives -> higher loss, bounded by 2 log B",
          all(sizes[a] < sizes[b] for a, b in ((2, 4), (4, 8), (8, 16))) and
          all(sizes[n] <= 2.0 * np.log(n) + 1e-9 for n in sizes),
          ", ".join(f"B={n}:{v:.4f}" for n, v in sizes.items()))
    gap = [(c, float(one(*controlled(6, c)))) for c in (0.1, 0.4, 0.7, 1.0)]
    check("better-aligned cosine pairs cost strictly less, approaching 0 as cos -> 1",
          all(a[1] > b[1] for a, b in zip(gap, gap[1:])) and gap[-1][1] < 1e-5,
          ", ".join(f"cos={c:.1f}:{v:.3e}" for c, v in gap))
    check("loss has a gradient with respect to both sides",
          symmetric_contrastive_loss(s.clone().requires_grad_(True),
                                     t.clone().requires_grad_(True)).requires_grad)

    # =========================================================== 4. backward
    print("\n[4] backward through the PSA loss")
    adapter.zero_grad(); rqvae.zero_grad(); tiger.zero_grad()
    psa_total.backward()
    psa_grads = {n: p.grad for n, p in adapter.named_parameters()}
    dec_named = {n: p.grad for n, p in tiger.model.decoder.named_parameters()
                 if not n.startswith("embed_tokens")}
    enc_grads = [p.grad for p in tiger.model.encoder.parameters()]
    rq_grads = [p.grad for p in rqvae.parameters()]
    check("PSA projection head receives gradients (non-zero, finite)",
          all(g is not None and float(g.abs().sum()) > 0 and bool(torch.isfinite(g).all())
              for g in psa_grads.values()),
          ", ".join(f"{n}:{float(g.abs().sum()):.3e}" for n, g in psa_grads.items()))
    dec_with_grad = [g for g in dec_named.values() if g is not None]
    check("T5 decoder stack receives gradients (this is a decoder-hidden loss)",
          len(dec_with_grad) > 0 and all(float(g.abs().sum()) >= 0 and
                                         bool(torch.isfinite(g).all()) for g in dec_with_grad),
          f"{len(dec_with_grad)}/{len(dec_named)} decoder tensors with grad")
    check("T5 encoder also receives gradients (decoder cross-attends to it)",
          sum(g is not None for g in enc_grads) > 0,
          f"{sum(g is not None for g in enc_grads)}/{len(enc_grads)} encoder tensors with grad")
    check("target-side RQ-VAE path also receives gradients (freeze policy is E7, not E5/E6)",
          sum(g is not None for g in rq_grads) > 0,
          f"{sum(g is not None for g in rq_grads)}/{len(rq_grads)} RQ-VAE tensors with grad")
    all_grads = [g for g in list(psa_grads.values()) + list(dec_named.values()) +
                 enc_grads + rq_grads if g is not None]
    check("every gradient is finite (no NaN / Inf)",
          all(bool(torch.isfinite(g).all()) for g in all_grads),
          f"{len(all_grads)} gradient tensors checked")
    check("no NaN / Inf anywhere in the PSA forward outputs",
          all(bool(torch.isfinite(x).all()) for x in (psa_rep, target_rep, l_psa)))
    check("recommendation loss still computes alongside PSA", bool(torch.isfinite(t5_loss).all()),
          f"L_T5={float(t5_loss):.6f}")

    # ================================================== 5. zero-lambda equivalence
    print("\n[5] lambda_psa = 0 equivalence (recommender loop)")
    from torch.utils.data import Subset
    small = Subset(ds, list(range(0, 4 * BATCH)))

    def count_calls(mount_obj, cls, key, calls):
        """Wrap an instance's forward so the test can prove it was never entered."""
        orig = cls.forward

        def wrapper(self, *args, **kwargs):
            calls[key] += 1
            calls.setdefault(f"{key}_input_requires_grad", []).append(
                bool(args[0].requires_grad) if args else None)
            return orig(self, *args, **kwargs)

        mount_obj.forward = wrapper.__get__(mount_obj, cls)

    def run_step(lam_sia, lam_psa):
        current_main.set_seed(SEED)
        m = current_main.TIGER(T5CFG).to(device)
        sia = psa = None
        calls = {"sia": 0, "psa": 0, "generate": 0}
        if lam_sia is not None:
            from tiger_ext.sia import SIAAdapter
            from tiger_ext.sia_loss import SIAMount
            sia = SIAMount(SIAAdapter(T5CFG['d_model']).to(device), rqvae, item_embs,
                           sid_tokens, lambda_sia=lam_sia).to(device)
            count_calls(sia, SIAMount, "sia", calls)
        if lam_psa is not None:
            psa = PSAMount(PSAAdapter(T5CFG['d_model'], d_target).to(device), rqvae,
                           item_embs, sid_tokens, lambda_psa=lam_psa).to(device)
            count_calls(psa, PSAMount, "psa", calls)
        # plan.md E5 acceptance ④: the training path must be the teacher-forcing
        # `forward`, never `generate()` (beam search does not expose decoder hidden).
        gen_orig = current_main.TIGER.generate

        def counted_generate(self, *a, **k):
            calls["generate"] += 1
            return gen_orig(self, *a, **k)

        m.generate = counted_generate.__get__(m, current_main.TIGER)
        current_main.set_seed(SEED + 1)
        loader = GenRecDataLoader(small, batch_size=BATCH, shuffle=False, num_workers=0)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        avg = current_main.train(m, loader, opt, device, sia=sia, psa=psa)
        grads = {n: (p.grad.clone() if p.grad is not None else None)
                 for n, p in m.named_parameters()}
        return avg, grads, calls

    clean_avg, clean_grads, clean_calls = run_step(None, None)
    p0_avg, p0_grads, p0_calls = run_step(None, 0.0)
    both_avg, both_grads, both_calls = run_step(0.0, 0.0)
    check("lambda_psa = 0 leaves the epoch loss bitwise identical to Clean TIGER",
          clean_avg == p0_avg, f"clean {clean_avg!r} vs lambda=0 {p0_avg!r}")
    same = all((clean_grads[n] is None and p0_grads[n] is None) or
               (clean_grads[n] is not None and p0_grads[n] is not None and
                torch.equal(clean_grads[n], p0_grads[n])) for n in clean_grads)
    check("lambda_psa = 0 leaves every parameter gradient bitwise identical", same,
          f"{len(clean_grads)} parameter tensors compared")
    check("lambda_psa = 0 never enters the PSA path (mount not invoked)",
          p0_calls["psa"] == 0 and clean_calls["psa"] == 0,
          f"clean {clean_calls['psa']}, lambda=0 {p0_calls['psa']}")
    check("both weights at 0 is still bitwise identical to Clean TIGER (SIA mount attached too)",
          both_avg == clean_avg and
          all((clean_grads[n] is None and both_grads[n] is None) or
              (clean_grads[n] is not None and both_grads[n] is not None and
               torch.equal(clean_grads[n], both_grads[n])) for n in clean_grads),
          f"clean {clean_avg!r} vs both-zero {both_avg!r}")
    check("neither mount was invoked when both weights are 0",
          both_calls["sia"] == 0 and both_calls["psa"] == 0)

    live_avg, _, live_calls = run_step(0.0, LAMBDA)
    check("with lambda_psa > 0 the loop actually runs the PSA path and stays finite",
          live_calls["psa"] == 4 and np.isfinite(live_avg), f"psa calls {live_calls['psa']}")
    check("training never calls generate(): PSA is fed by the teacher-forcing forward",
          live_calls["generate"] == 0, f"generate() calls = {live_calls['generate']}")
    check("the tensor PSA receives is a live forward output (requires_grad), not a detached cache",
          all(live_calls.get("psa_input_requires_grad", [])) and
          len(live_calls.get("psa_input_requires_grad", [])) == 4,
          f"{live_calls.get('psa_input_requires_grad')}")

    # ================================================= 6. original TIGER regression
    print("\n[6] original TIGER path")
    current_main.set_seed(SEED)
    cur = current_main.TIGER(T5CFG).to(device)
    current_main.set_seed(SEED)
    pri = pristine_main.TIGER(T5CFG).to(device)
    pri.load_state_dict(cur.state_dict())
    cur.eval(); pri.eval()
    lc, oc = cur(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    lp, op = pri(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    check("loss / logits bitwise identical to pre-E1..E5 TIGER",
          torch.equal(lc, lp) and torch.equal(oc, op), f"loss {float(lc):.8f}")
    gc = cur.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5)
    gp = pri.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5)
    check("generate() bitwise identical to pre-E1..E5 TIGER",
          torch.equal(gc, gp), f"shape {tuple(gc.shape)}")
    check("default forward still returns exactly (loss, logits)",
          len(cur(input_ids=input_ids, attention_mask=attention_mask, labels=labels)) == 2)
    check("PSA adds no parameter to the TIGER module (adapter lives outside it)",
          sum(p.numel() for p in cur.parameters()) == sum(p.numel() for p in pri.parameters()),
          f"{sum(p.numel() for p in cur.parameters())} params")
    with torch.no_grad():
        sd_now = {k: tuple(v.shape) for k, v in rqvae.state_dict().items()}
    check("RQ-VAE structure / codebooks unchanged (23 tensors, 3 x (256, 32))",
          len(sd_now) == 23 and
          all(tuple(sd_now[k]) == (256, RQVAE_CFG["e_dim"]) for k in sd_now if "embedding" in k),
          f"{len(sd_now)} tensors")
    with torch.no_grad():
        enc_latent = rqvae.encoder(raw_target)
        idx_default = rqvae.get_indices(raw_target)
    check("SID / collision layout untouched: quantiser still yields 3 levels of hard indices",
          tuple(idx_default.shape) == (BATCH, 3) and int(idx_default.max()) < 256,
          f"indices {tuple(idx_default.shape)}, max {int(idx_default.max())}")

    # ============================================================ 7. diagnostics
    print("\n[7] DIAGNOSTIC -- frozen target (reconstruction) vs the task message's variant")
    with torch.no_grad():
        l_raw = float(mount.loss_fn(psa_rep, raw_target))
        cos = F.cosine_similarity(psa_rep, target_rep).mean()
        cos_raw = F.cosine_similarity(psa_rep, raw_target).mean()
        sim = F.normalize(psa_rep, dim=-1) @ F.normalize(target_rep, dim=-1).t()
        print(f"      frozen E6 target = RQ-VAE reconstruction : L_PSA = {float(l_psa):.6f} "
              f"| mean cos(psa, target) = {float(cos):+.4f}")
        print(f"      variant target   = raw item embedding    : L_PSA = {l_raw:.6f} "
              f"| mean cos(psa, target) = {float(cos_raw):+.4f}")
        print(f"      (the frozen plan.md E6 explicitly says the target is the reconstruction, "
              f"'不是原始 content embedding'; variant shown for comparison only)")
        print(f"      reconstruction fidelity: mean |x - recon| = "
              f"{float((raw_target - target_rep).abs().mean()):.4f} vs raw scale "
              f"{float(raw_target.abs().mean()):.4f}")
        rec_std = target_rep.std(dim=0).mean()
        raw_std = raw_target.std(dim=0).mean()
        pairwise = F.normalize(target_rep, dim=-1) @ F.normalize(target_rep, dim=-1).t()
        off = pairwise[~torch.eye(BATCH, dtype=torch.bool)]
        print(f"      does the reconstruction still differ across items? across-batch std "
              f"recon {float(rec_std):.5f} vs raw {float(raw_std):.5f} "
              f"(ratio {float(rec_std / raw_std):.3f}); pairwise cos(recon_i, recon_j) "
              f"mean {float(off.mean()):+.4f} (raw {float((F.normalize(raw_target, dim=-1) @ F.normalize(raw_target, dim=-1).t())[~torch.eye(BATCH, dtype=torch.bool)].mean()):+.4f})")
        print(f"      cosine matrix diag mean {float(sim.diag().mean()):.4f} vs "
              f"off-diag mean {float(sim[~torch.eye(BATCH, dtype=torch.bool)].mean()):.4f} "
              f"(temperature {mount.loss_fn.temperature})")
        print(f"      grad scale: psa.proj.weight "
              f"{float(adapter.proj.weight.grad.abs().sum()):.3e}, psa.proj.bias "
              f"{float(adapter.proj.bias.grad.abs().sum()):.3e}, "
              f"T5 decoder {sum(float(g.abs().sum()) for g in dec_with_grad):.3e}, "
              f"T5 encoder {sum(float(g.abs().sum()) for g in enc_grads if g is not None):.3e}")
        print("      (recorded only; no threshold, no pass/fail, no action taken)")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 70)
    print(f"EXP-E5: {passed}/{len(RESULTS)} checks passed")
    print("=" * 70)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
