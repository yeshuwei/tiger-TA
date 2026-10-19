#!/usr/bin/env python
"""EXP-E10 -- ETEGRec integration smoke test (Task E10, type A + C candidate).

Run:  python -I integration/exp_e10_smoke.py
      EXP_E10_DEVICE=cuda python -I integration/exp_e10_smoke.py
      EXP_E10_REGRESSION=skip python -I integration/exp_e10_smoke.py

The plan's E10 acceptance:

  1. the whole chain runs end to end and the process exits 0
  2. every module the port added is *actually executed* (call counters > 0)
  3. G0 -- with every new switch off, Clean TIGER is bitwise unchanged
  4. G1 -- no historical mechanism (L_trans / L_balance / L_align / behavior /
     fused) is activated

Unlike EXP-E7/E8/E9, which each verify one layer, this is the integration gate:
the whole E1-E9 chain is driven once, with counters and tripwires installed, so
that "it is wired up" is a measurement rather than a reading of the source.

Five parts, cheapest first:

  [1] the probe harness and its **negative controls**. A probe that has never
      been seen to fire proves nothing, so each one is shown to have teeth: a
      counter that catches a real call, a tripwire that fires when a historical
      mechanism is genuinely switched on, a read-spy that flags a forbidden file.
  [2] G0: the pristine sources from ``5edfca0`` are imported side by side with
      the current ones *on identical weights and inputs*, and compared with
      ``torch.equal`` -- including the whole 11924-item Toys code assignment and
      a real Semantic-ID regeneration against the frozen baseline artefact.
  [3] the full chain: a real 2-cycle Tokenizer -> SID Refresh -> Recommender ->
      Evaluation run on a synthetic fixture, with the probes live; plus the
      switches-off control, a resume and an idempotency re-run.
  [4] scope: the frozen directories are untouched.
  [5] regression: E1-E9 re-run.

Nothing here measures recommendation quality; that is EXP-E7-E9 (type B/C).
"""
import copy
import functools
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)        # -I does not add the script's directory
BASELINE_COMMIT = "5edfca0"
SMOKE = os.path.join(REPO, "clean_tiger", "_smoke", "Toys")
SMOKE_SID = os.path.join(SMOKE, "Toys_clean_tiger_sid.npy")

DEVICE = os.environ.get("EXP_E10_DEVICE", "cpu")
REGRESSION = os.environ.get("EXP_E10_REGRESSION", "all").lower()
DATASET = "E10Syn"
SEED = 2025

# the tracked files E1-E6 legitimately modified; E10 must not add to this set
PRE_E10_MODIFIED = {"model/main.py", "rqvae/models/rq.py", "rqvae/models/vq.py"}

# clean_tiger/config.sh's forbidden-input guard, as a substring list (this is
# the same list clean_tiger/verify_clean.py uses for the frozen baseline)
FORBIDDEN_SUBSTRINGS = ("behavior_emb", "fused_item_emb", "transition_pairs",
                        "trans_graph", "ppmi", "_svd", "b2align", "ta_sid")

# the real Clean TIGER hyper-parameters, used for the G0 comparison
RQVAE_CFG = dict(num_emb_list=[256, 256, 256], e_dim=32, layers=[512, 256, 128, 64],
                 dropout_prob=0.0, bn=False, loss_type="mse", quant_loss_weight=1.0,
                 beta=0.25, kmeans_init=True, kmeans_iters=100,
                 sk_epsilons=[0.0, 0.0, 0.003], sk_iters=50)
T5CFG = {'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
         'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1, 'vocab_size': 1025,
         'pad_token_id': 0, 'eos_token_id': 0, 'feed_forward_proj': 'relu',
         'max_len': 20}
BATCH = 8

# tiny fixture for the alternating smoke run (E9's sizing, for the same reasons:
# k-means needs n_e <= batch samples, and E8 rejects an unresolved duplicate)
SYN = dict(n_items=1024, dim=32, n_users=24, seq_len=12, n_valid=8)
SYN_RQVAE = dict(num_emb_list=[128, 128, 128], e_dim=32, layers=[16, 8],
                 kmeans_init=True, kmeans_iters=5, sk_iters=5,
                 sk_epsilons=[0.0, 0.0, 0.003], batch_size=1024, lr=5e-3)
SYN_T5 = dict(num_layers=1, num_decoder_layers=1, d_model=16, d_ff=32,
              num_heads=2, d_kv=4, dropout_rate=0.0, vocab_size=1025,
              pad_token_id=0, eos_token_id=0, feed_forward_proj="relu", max_len=20)

RESULTS = []

# the (sia, psa) arguments every call to the frozen model/main.py::train carried
train_kwargs = []
_train_kwargs_seen = [[]]        # the same, captured from the switches-on run


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:                                            # noqa: BLE001
        return False
    return False


def same_bytes(x, y):
    """Bitwise equality for a float, without going through allclose."""
    return np.float64(x).tobytes() == np.float64(y).tobytes()


def tensors_bitwise_equal(a, b):
    return len(a) == len(b) and all(torch.equal(x.detach().cpu(), y.detach().cpu())
                                    for x, y in zip(a, b))


def state_dict_of(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


# --------------------------------------------------------------------------- #
# [1] the probe harness
# --------------------------------------------------------------------------- #
class Probe:
    """A call counter installed *in place* on a class or module attribute.

    ``predicate(args, kwargs)`` makes the counter selective (used for E1, whose
    interface is a keyword argument rather than a separate entry point).
    """

    def __init__(self, owner, attr, label, predicate=None, on_call=None):
        self.owner, self.attr, self.label = owner, attr, label
        self.predicate = predicate
        self.on_call = on_call
        self.calls = 0
        self.wraps_calls = 0
        self._orig = getattr(owner, attr)

    def install(self):
        outer = self

        @functools.wraps(self._orig)
        def wrapper(*a, **k):
            outer.wraps_calls += 1
            if outer.predicate is None or outer.predicate(a, k):
                outer.calls += 1
            if outer.on_call is not None:
                outer.on_call(a, k)
            return outer._orig(*a, **k)

        setattr(self.owner, self.attr, wrapper)
        return self

    def restore(self):
        setattr(self.owner, self.attr, self._orig)

    def reset(self):
        self.calls = 0
        self.wraps_calls = 0

    @property
    def installed(self):
        return getattr(self.owner, self.attr) is not self._orig


class Tripwire:
    """A callable that counts, then fails loudly if a disabled mechanism runs."""

    def __init__(self, name):
        self.name = name
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        raise AssertionError(
            f"CONTAMINATED: disabled mechanism {self.name!r} was invoked "
            "during the E10 smoke run")


def install_probes(mods):
    """One counter per module the port added (E1-E9)."""
    from tiger_ext import sia, sia_codes, sia_loss, psa, psa_loss
    from tiger_ext import alternating as A, alt_pipeline as AP
    from tiger_ext.stage import StageFreeze

    current_main = mods["tiger_main"]
    probes = {
        # E1: the hidden-state interface is a kwarg on TIGER.forward
        "E1  TIGER.forward(return_hidden=True)": Probe(
            current_main.TIGER, "forward", "E1",
            predicate=lambda a, k: bool(k.get("return_hidden")
                                        or (len(a) > 3 and a[3]))),
        # E2
        "E2  SIAAdapter.forward": Probe(sia.SIAAdapter, "forward", "E2"),
        "E2  masked_mean_pool": Probe(sia, "masked_mean_pool", "E2"),
        # E3
        "E3  SIACodeExtractor.sequence_logits": Probe(
            sia_codes.SIACodeExtractor, "sequence_logits", "E3"),
        "E3  SIACodeExtractor.target_logits": Probe(
            sia_codes.SIACodeExtractor, "target_logits", "E3"),
        # E4
        "E4  SIAMount.forward": Probe(sia_loss.SIAMount, "forward", "E4"),
        "E4  SIALoss.forward": Probe(sia_loss.SIALoss, "forward", "E4"),
        "E4  symmetric_kl": Probe(sia_loss, "symmetric_kl", "E4"),
        # E5
        "E5  PSAAdapter.forward": Probe(psa.PSAAdapter, "forward", "E5"),
        # E6
        "E6  PSAMount.forward": Probe(psa_loss.PSAMount, "forward", "E6"),
        "E6  PSALoss.forward": Probe(psa_loss.PSALoss, "forward", "E6"),
        "E6  symmetric_contrastive_loss": Probe(
            psa_loss, "symmetric_contrastive_loss", "E6"),
        # E7
        "E7  StageFreeze.apply": Probe(StageFreeze, "apply", "E7"),
        "E7  StageFreeze.trainable_parameters": Probe(
            StageFreeze, "trainable_parameters", "E7"),
        "E7  StageFreeze.mounts_for_stage": Probe(
            StageFreeze, "mounts_for_stage", "E7"),
        # E8 -- patched where the pipeline actually calls it. alt_pipeline does
        # ``from .sid_refresh import refresh_sid``, so the module-global binding
        # is the live one and patching tiger_ext.sid_refresh would observe nothing.
        "E8  sid_refresh.refresh_sid": Probe(AP, "refresh_sid", "E8"),
        # E9
        "E9  AlternatingScheduler.run": Probe(
            A.AlternatingScheduler, "run", "E9"),
        "E9  AlternatingScheduler.run_cycle": Probe(
            A.AlternatingScheduler, "run_cycle", "E9"),
        "E9  pipeline.tokenizer_phase": Probe(
            AP.TigerAlternatingPipeline, "tokenizer_phase", "E9"),
        "E9  pipeline.refresh_phase": Probe(
            AP.TigerAlternatingPipeline, "refresh_phase", "E9"),
        "E9  pipeline.recommender_phase": Probe(
            AP.TigerAlternatingPipeline, "recommender_phase", "E9"),
        "E9  pipeline.eval_phase": Probe(
            AP.TigerAlternatingPipeline, "eval_phase", "E9"),
    }
    for probe in probes.values():
        probe.install()
    return probes


class ReadSpy:
    """Records every parquet / npy path the process opens."""

    def __init__(self):
        import pandas as pd
        self.paths = []
        self._pd_read_parquet = pd.read_parquet
        self._np_load = np.load
        self._pd = pd

    def start(self):
        outer = self

        def spy_parquet(path, *a, **k):
            outer.paths.append(str(path))
            return outer._pd_read_parquet(path, *a, **k)

        def spy_load(file, *a, **k):
            outer.paths.append(str(file))
            return outer._np_load(file, *a, **k)

        self._pd.read_parquet = spy_parquet
        np.load = spy_load

    def stop(self):
        self._pd.read_parquet = self._pd_read_parquet
        np.load = self._np_load

    def forbidden(self):
        return [p for p in self.paths
                if any(s in p.lower() for s in FORBIDDEN_SUBSTRINGS)]


def part1_probe_harness(tmp):
    print("\n[1] the probe harness, and its negative controls")
    # --- the counter catches a call it is aimed at ---------------------------
    class Target:
        def method(self, x):
            return x + 1

    probe = Probe(Target, "method", "selftest").install()
    check("a fresh probe is really installed (the attribute is replaced)",
          probe.installed and probe.calls == 0)
    Target().method(1)
    check("the negative control: the counter increments when the target is called",
          probe.calls == 1, f"calls={probe.calls}")
    Target().method(1)
    check("... and it keeps counting", probe.calls == 2)
    selective = Probe(Target, "method", "selftest", predicate=lambda a, k: a[1] == 7)
    selective.install()
    Target().method(1)
    Target().method(7)
    check("a selective probe only counts the calls it was aimed at "
          "(the E1 interface is a kwarg, not an entry point)",
          selective.calls == 1 and selective.wraps_calls == 2,
          f"counted {selective.calls} of {selective.wraps_calls} calls")
    selective.restore()
    probe.restore()

    # --- the read spy flags a forbidden file ---------------------------------
    spy = ReadSpy()
    spy.start()
    try:
        bad = os.path.join(tmp, "behavior_emb_probe.npy")
        np.save(bad, np.zeros(2))
        np.load(bad)
    finally:
        spy.stop()
    check("the read spy catches a forbidden file deliberately opened "
          "(behavior_emb / fused / transition / ... )",
          len(spy.forbidden()) == 1, f"{spy.forbidden()}")


# --------------------------------------------------------------------------- #
# [2] G0 -- bitwise equivalence with the pristine 5edfca0 sources
# --------------------------------------------------------------------------- #
def build_pristine(tmpdir):
    """The pre-E1 rqvae/models package and model/main.py, straight from git."""
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


def smoke_ckpt():
    """The same Clean TIGER tokenizer checkpoint the E8 suite regenerates from."""
    hits = sorted(glob.glob(os.path.join(REPO, "clean_tiger", "ckpt", "Toys", "rqvae",
                                         "*", "best_collision_model.pth")))
    return hits[-1] if hits else None


def part2_g0(tmp, mods, PristineRQVAE, pristine_main):
    print("\n[2] G0 -- with every switch off, Clean TIGER is bitwise unchanged")
    from datasets import EmbDataset
    from dataset import GenRecDataset
    from dataloader import GenRecDataLoader
    from models.rqvae import RQVAE
    current_main = mods["tiger_main"]
    device = torch.device(DEVICE)

    ckpt_path = smoke_ckpt()
    check("the frozen Clean TIGER smoke artefacts are present to compare against",
          ckpt_path is not None and os.path.isfile(SMOKE_SID),
          f"{os.path.relpath(ckpt_path, REPO) if ckpt_path else 'MISSING'}")
    check("the frozen checkpoint itself records all three historical switches at 0 "
          "-- the baseline these ported modules must not perturb never used them",
          (lambda a: a.lambda_trans == 0 and a.lambda_balance == 0
           and a.align_weight == 0)(
               torch.load(ckpt_path, map_location="cpu", weights_only=False)["args"]))

    # ---------------------------------------------------------- RQ-VAE -> codes
    emb_ds = EmbDataset(os.path.join(SMOKE, "data", "Toys", "item_emb.parquet"))
    xs = torch.as_tensor(emb_ds.embeddings, dtype=torch.float32).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def build(cls):
        m = cls(in_dim=emb_ds.dim, **RQVAE_CFG).to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        return m

    pristine_rq, current_rq = build(PristineRQVAE), build(RQVAE)
    check("the two RQ-VAEs start from identical weights",
          tensors_bitwise_equal(list(pristine_rq.state_dict().values()),
                                list(current_rq.state_dict().values())))

    with torch.no_grad():
        for use_sk in (True, False):
            codes_p = pristine_rq.get_indices(xs, use_sk=use_sk)
            codes_c = current_rq.get_indices(xs, use_sk=use_sk)
            check(f"all {len(emb_ds)} items get bitwise identical codes "
                  f"(get_indices, use_sk={use_sk})",
                  torch.equal(codes_p.cpu(), codes_c.cpu())
                  and tuple(codes_p.shape) == (len(emb_ds), 3),
                  f"{tuple(codes_c.shape)}")
        batch = xs[:BATCH]
        out_p = pristine_rq(batch)
        out_c = current_rq(batch)
        check("RQVAE.forward returns the same arity and bitwise identical tensors "
              "(the default switches are inert)",
              len(out_p) == len(out_c)
              and all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(out_p, out_c)),
              f"{len(out_p)}-tuple each")

    stale = sys.modules.pop("pristine_models", None)
    del stale

    # ------------------------------------------------------------- codes -> SID
    def blob(path):
        return subprocess.run(["git", "-C", REPO, "hash-object", path],
                              capture_output=True, text=True, check=True).stdout.strip()

    def baseline_blob(rel):
        return subprocess.run(["git", "-C", REPO, "rev-parse",
                               f"{BASELINE_COMMIT}:{rel}"],
                              capture_output=True, text=True, check=True).stdout.strip()

    gen_rel = "rqvae/generate_code.py"
    check("the SID generator itself is byte-identical to the frozen baseline "
          "(identical codes therefore give identical .npy bytes)",
          blob(os.path.join(REPO, gen_rel)) == baseline_blob(gen_rel),
          f"{gen_rel} @ {blob(os.path.join(REPO, gen_rel))[:8]}")

    # a real regeneration, against the frozen pipeline's own artefact
    from tiger_ext import sid_refresh as SR
    work = os.path.join(tmp, "g0_sid")
    rec = SR.refresh_sid("Toys", ckpt_path, work, device=DEVICE, timeout=600)
    fresh = SR.load_sid(rec["sid_path"])
    frozen = SR.load_sid(SMOKE_SID)
    changed_rows = int((fresh != frozen).any(axis=1).sum())
    per_column = [int((fresh[:, c] != frozen[:, c]).sum())
                  for c in range(frozen.shape[1])]
    if str(DEVICE).startswith("cuda"):
        check("regenerating the Semantic ID from the frozen checkpoint on the "
              "baseline's own device reproduces it bitwise",
              np.array_equal(fresh, frozen),
              f"{changed_rows}/{frozen.shape[0]} rows differ")
    else:
        # The SID is device dependent -- E8's §3 measures exactly this as a noise
        # floor (the SK path decides borderline items differently on CPU). So the
        # claim here is E8's own: the difference is real but *structural*, i.e.
        # confined to the last RQ level and the collision-removal column, and
        # levels 0/1 -- where the semantics live -- are untouched.
        rate = changed_rows / frozen.shape[0]
        check("on CPU the regenerated Semantic ID differs from the CUDA baseline "
              "only the way E8's device-noise floor says it does: levels 0-1 "
              "identical, the difference confined to level 2 + the collision "
              "column, and a few percent at most",
              per_column[0] == 0 and per_column[1] == 0 and changed_rows > 0
              and rate <= 0.05,
              f"{changed_rows}/{frozen.shape[0]} rows differ ({rate:.2%}), "
              f"per column {per_column}")

    # ------------------------------------------------------------------- T5
    ds = GenRecDataset(dataset_path=os.path.join(SMOKE, "data", "Toys",
                                                 "train.parquet"),
                       code_path=SMOKE_SID, mode="train", max_len=T5CFG["max_len"])
    batch = next(iter(GenRecDataLoader(ds, batch_size=BATCH, shuffle=False,
                                       num_workers=0)))
    input_ids = batch["history"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["target"].to(device)

    torch.manual_seed(SEED)
    ref = pristine_main.TIGER(T5CFG).to(device)
    cur = current_main.TIGER(T5CFG).to(device)
    cur.model.load_state_dict(ref.model.state_dict())
    ref.train(), cur.train()

    ref_grads, cur_grads, ref_losses, cur_losses = [], [], [], []
    for opt in ("ref", "cur"):
        model = ref if opt == "ref" else cur
        torch.manual_seed(SEED + 1)          # identical dropout masks on both sides
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        before = state_dict_of(model)
        loss = (pristine_main if opt == "ref" else current_main).train(
            model,
            [(dict(history=input_ids, attention_mask=attention_mask, target=labels))],
            optim, device)
        grads = [p.grad.detach().clone() if p.grad is not None else None
                 for p in model.parameters()]
        after = state_dict_of(model)
        sizes = [p.numel() for p in model.parameters()]
        if opt == "ref":
            ref_losses, ref_grads, ref_state, ref_sizes = [loss], grads, after, sizes
        else:
            cur_losses, cur_grads, cur_state, cur_sizes = [loss], grads, after, sizes
    check("switches off: one Clean TIGER training epoch produces a bitwise "
          "identical mean loss",
          same_bytes(ref_losses[0], cur_losses[0]),
          f"{ref_losses[0]!r} vs {cur_losses[0]!r}")
    check("switches off: every parameter gradient is bitwise identical "
          "(count, shapes and values)",
          ref_sizes == cur_sizes
          and all((a is None) == (b is None) and (a is None or torch.equal(a, b))
                  for a, b in zip(ref_grads, cur_grads)),
          f"{sum(g is not None for g in cur_grads)} tensors carry a gradient")
    check("switches off: the optimizer step leaves every weight bitwise identical",
          all(torch.equal(ref_state[k], cur_state[k]) for k in ref_state)
          and set(ref_state) == set(cur_state),
          f"{len(cur_state)} state tensors compared")

    ref.eval(), cur.eval()          # same weights (checked above) -> same decoding
    with torch.no_grad():
        gen_ref = ref.generate(input_ids, attention_mask, num_beams=2)
        gen_cur = cur.generate(input_ids, attention_mask, num_beams=2)
    check("switches off: generate() returns bitwise identical tokens",
          torch.equal(gen_ref, gen_cur),
          f"{tuple(gen_cur.shape)}")

    eval_ds = GenRecDataset(dataset_path=os.path.join(SMOKE, "data", "Toys",
                                                      "valid.parquet"),
                            code_path=SMOKE_SID, mode="evaluation",
                            max_len=T5CFG["max_len"])
    eval_loader = GenRecDataLoader(eval_ds, batch_size=2, shuffle=False, num_workers=0)
    with torch.no_grad():
        r_ref, n_ref = pristine_main.evaluate(ref, eval_loader, [5, 10, 20], 2, device)
        r_cur, n_cur = current_main.evaluate(cur, eval_loader, [5, 10, 20], 2, device)
    check("switches off: Recall@5/10/20 and NDCG@5/10/20 are bitwise identical",
          set(r_ref) == set(r_cur) and set(n_ref) == set(n_cur)
          and all(same_bytes(r_ref[k], r_cur[k]) for k in r_ref)
          and all(same_bytes(n_ref[k], n_cur[k]) for k in n_ref),
          f"{ {k: round(v, 6) for k, v in n_cur.items()} }")


# --------------------------------------------------------------------------- #
# [3] the full chain
# --------------------------------------------------------------------------- #
def build_fixture(root, rng):
    import pandas as pd
    data_dir = os.path.join(root, DATASET)
    os.makedirs(data_dir, exist_ok=True)
    n_items, dim = SYN["n_items"], SYN["dim"]
    centres = rng.normal(size=(8, dim)).astype(np.float32) * 1.5
    which = rng.integers(0, 8, size=n_items)
    emb = (centres[which] + rng.normal(size=(n_items, dim)).astype(np.float32)
           ).astype(np.float32)
    pd.DataFrame({"embedding": [emb[i] for i in range(n_items)]}).to_parquet(
        os.path.join(data_dir, "item_emb.parquet"), index=False)

    def interactions(n_rows):
        history, target = [], []
        for _ in range(n_rows):
            seq = rng.integers(1, n_items + 1, size=SYN["seq_len"]).tolist()
            history.append(seq[:-1])
            target.append(seq[-1])
        return pd.DataFrame({"history": history, "target": target})

    for split, rows in (("train", SYN["n_users"]), ("valid", SYN["n_valid"]),
                        ("test", SYN["n_valid"])):
        interactions(rows).to_parquet(os.path.join(data_dir, f"{split}.parquet"),
                                      index=False)
    return data_dir


def build_pipeline(root, sid_base, run_dir, **kwargs):
    AP = sys.modules["tiger_ext.alt_pipeline"]
    cfg = dict(device=DEVICE, seed=SEED, rqvae_config=SYN_RQVAE, t5_config=SYN_T5,
               lambda_sia=0.1, lambda_psa=0.1, batch_size=32, infer_size=4,
               beam_size=2)
    cfg.update(kwargs)
    return AP.TigerAlternatingPipeline(DATASET, root, sid_base, run_dir, **cfg)


def part3_smoke(tmp, mods, probes, wires, trainer_log, spy, recording):
    from tiger_ext import alternating as A, alt_pipeline as AP
    from tiger_ext import sid_refresh as SR
    from tiger_ext.stage import STAGE_RECOMMENDER, STAGE_TOKENIZER

    print("\n[3] the full chain: 2 real cycles, probes and tripwires live")
    recording["on"] = True       # from here on, every RQ-VAE Trainer is audited
    root = os.path.join(tmp, "data")
    build_fixture(root, np.random.default_rng(0))
    sid_base = os.path.join(tmp, "sid")
    run_dir = os.path.join(tmp, "run")

    spy.paths.clear()
    boot = build_pipeline(root, os.path.join(tmp, "sid_boot"),
                          os.path.join(tmp, "bootstrap"))
    t0 = time.time()
    initial = boot.tokenizer_phase(0, 0, None)

    pipeline = build_pipeline(root, sid_base, run_dir)
    scheduler = AP.build_scheduler(pipeline, run_dir, 2, initial["ckpt_path"])
    crashed = None
    try:
        records = scheduler.run()
    except Exception as exc:                                     # noqa: BLE001
        crashed = f"{type(exc).__name__}: {exc}"
        records = []
    elapsed = time.time() - t0

    check("(1) the whole chain ran end to end without raising (exit 0)",
          crashed is None and len(records) == 2
          and all(r["state"] == "recommender_done" for r in records),
          crashed or f"2 cycles in {elapsed:.1f}s on {DEVICE}")

    log = [(e["phase"], e["cycle_index"], e["epoch_index"]) for e in pipeline.phase_log]
    expected = [("refresh", -1, -1),
                ("tokenizer", 0, 0), ("refresh", 0, 0), ("recommender", 0, 1),
                ("evaluation", 0, 1),
                ("tokenizer", 1, 2), ("refresh", 1, 2), ("recommender", 1, 3),
                ("evaluation", 1, 3)]
    check("the real phase order is Tokenizer -> Refresh -> Recommender -> Eval "
          "per cycle, after one initial refresh",
          log == expected, f"{log}")

    # ------------------------------------------------------ (2) the artefacts
    artefacts = []
    for cycle, rec in enumerate(records):
        for key in ("tokenizer", "recommender"):
            ckpt = rec[key]["ckpt_path"]
            artefacts += [ckpt, ckpt + ".stage.json"]
        artefacts += [rec["sid"]["sid_path"], SR.sidecar_path(rec["sid"]["sid_path"]),
                      A.cycle_manifest_path(run_dir, cycle)]
    artefacts += [os.path.join(run_dir, "initial_refresh.json"),
                  A.cycle_manifest_path(run_dir, 0)]
    versions = SR.list_versions(sid_base, DATASET)
    check("the run left a complete artefact chain on disk (checkpoints, stage "
          "records, SID files, SID sidecars, cycle manifests)",
          all(os.path.isfile(p) for p in artefacts),
          f"{len(artefacts)} files, {len(versions)} SID versions")
    check("one initial refresh plus one per cycle, each a new immutable version",
          len(versions) == 3 and [v["version"] for v in versions] == [1, 2, 3],
          f"v{[v['version'] for v in versions]}")

    for cycle, rec in enumerate(records):
        check(f"cycle {cycle}: the recommender consumed this cycle's SID "
              f"(version and sha256)",
              rec["recommender"]["sid_version_used"] == rec["sid"]["version"] == 2 + cycle
              and rec["recommender"]["sid_sha256_used"] == rec["sid"]["sid_sha256"],
              f"v{rec['recommender']['sid_version_used']}")
        check(f"cycle {cycle}: the dataset, both mounts and the SID file agree",
              rec["recommender"]["handoff"]["mismatches"] == {"sia": 0, "psa": 0}
              and rec["recommender"]["handoff"]["sid_sha256"] == rec["sid"]["sid_sha256"],
              f"{rec['recommender']['handoff']['consistent_items']} items checked")
        tok_stage = json.load(open(rec["tokenizer"]["ckpt_path"] + ".stage.json"))
        rec_stage = json.load(open(rec["recommender"]["ckpt_path"] + ".stage.json"))
        check(f"cycle {cycle}: the checkpoints carry the right stage record "
              f"(and the SID sidecar names the tokenizer that produced it)",
              tok_stage["freeze"]["stage"] == STAGE_TOKENIZER
              and rec_stage["freeze"]["stage"] == STAGE_RECOMMENDER
              and SR.read_sid_manifest(rec["sid"]["sid_path"])["ckpt_sha256"]
              == rec["tokenizer"]["sha256"])
        check(f"cycle {cycle}: evaluation ran, on the recommender epoch, and is "
              f"recorded in that cycle's manifest",
              A.phase_of(rec["evaluation"]["seed"] - SEED) == STAGE_RECOMMENDER
              and rec["evaluation"]["ndcgs"].get("NDCG@20") is not None
              and rec["evaluation"]["cycle_index"] == cycle)

    # the optimizer parameter set is what the stage contract says it is
    freeze = pipeline.freeze
    freeze.apply(STAGE_RECOMMENDER)
    trainable = {id(p) for p in freeze.trainable_parameters(STAGE_RECOMMENDER)}
    tiger_ids = {id(p) for p in pipeline.tiger.parameters()}
    adapter_ids = {id(p) for p in list(pipeline.sia_adapter.parameters())
                   + list(pipeline.psa_adapter.parameters())}
    rqvae_ids = {id(p) for p in pipeline.rqvae.parameters()}
    check("no evaluation is ever recorded for a tokenizer epoch",
          all(A.phase_of(e["epoch_index"]) == STAGE_RECOMMENDER
              for e in pipeline.phase_log if e["phase"] == "evaluation"))
    check("the recommender's trainable set is exactly T5 + adapters, and the "
          "RQ-VAE is excluded",
          trainable == (tiger_ids | adapter_ids) and not (trainable & rqvae_ids)
          and len(trainable) > len(tiger_ids),
          f"{len(tiger_ids)} T5 + {len(adapter_ids)} adapter tensors, "
          f"{len(rqvae_ids)} RQ-VAE tensors frozen")

    # ------------------------------------------- (2) every module was executed
    print("      -- execution evidence (acceptance 2)")
    empty = {k: p for k, p in probes.items() if p.calls == 0}
    check("every module the port added was actually executed: each of E1-E9 has "
          "a call counter > 0",
          not empty,
          ", ".join(f"{k}={p.calls}" for k, p in sorted(probes.items())))
    check("the loss modules were entered on both cycles, not once at bootstrap",
          all(p.calls >= 2 for k, p in probes.items() if k.startswith(("E4 ", "E6 "))),
          f"SIALoss {probes['E4  SIALoss.forward'].calls} calls, "
          f"PSALoss {probes['E6  PSALoss.forward'].calls} calls")

    # ------------------------------------------------- the tripwire evidence
    print("      -- G1 evidence gathered during that run")
    fired = [w.name for w in wires.values() if w.calls]
    check("(4) no historical mechanism ran during the whole chain "
          "(L_trans / L_balance / historical L_align)",
          not fired, f"fired: {fired}" if fired else "0 activations")
    bad_reads = spy.forbidden()
    check("(4) no behavior / fused / transition file was read by the chain",
          not bad_reads,
          f"forbidden reads: {bad_reads}" if bad_reads
          else f"{len(spy.paths)} file(s) read, all clean")
    check("(4) every RQ-VAE Trainer the chain built had all historical switches at 0",
          len(trainer_log) >= 1 and all(
              t.lambda_trans == 0.0 and t.lambda_balance == 0.0
              and t.align_weight == 0.0 and t.trans_dict is None
              and t.align_table is None and t.align_proj is None
              for t in trainer_log),
          f"{len(trainer_log)} trainer(s) inspected")

    # ------------------------------------------- G0: the switches-off control
    print("      -- G0: the same machinery with every switch off")
    from dataset import GenRecDataset
    from dataloader import GenRecDataLoader

    on_calls = {k: p.calls for k, p in probes.items()}
    _train_kwargs_seen[0] = list(train_kwargs)
    train_kwargs.clear()
    for probe in probes.values():
        probe.reset()

    off = build_pipeline(root, os.path.join(tmp, "sid_off"),
                         os.path.join(tmp, "run_off"), lambda_sia=0.0,
                         lambda_psa=0.0, batch_size=SYN["n_users"])
    # a recommender stage requires a real, trained tokenizer (StageFreeze refuses
    # to seal uninitialised codebooks), so hand it the very tokenizer that
    # produced the SID we are about to train on
    off._load_tokenizer(records[-1]["tokenizer"]["ckpt_path"])
    off.freeze.apply(STAGE_RECOMMENDER)
    start_state = state_dict_of(off.tiger)
    sid_rec = records[-1]["sid"]

    # The fixture has exactly one batch, but ``recommender_phase`` builds its
    # loader with shuffle=True, and a reordered batch sums in a different order,
    # so a bitwise comparison needs the *same* permutation. Seeding identically
    # and replicating the construction order reproduces it. Everything that
    # consumes RNG must therefore sit in the same place: both replica models are
    # built before the seed is set (their init draws from the global stream), and
    # nothing between set_seed_for() and a loader's first __iter__ draws at all
    # (dropout_rate is 0; Adam draws nothing). recommender_phase re-seeds to the
    # same value internally, so all three runs see one permutation.
    pristine_main = sys.modules["pristine_main"]     # imported by build_pristine()
    replica_ds = GenRecDataset(
        dataset_path=os.path.join(root, DATASET, "train.parquet"),
        code_path=sid_rec["sid_path"], mode="train", max_len=SYN_T5["max_len"])
    replicas = {}
    for tag, tiger_mod in (("current", mods["tiger_main"]),
                           ("pristine", pristine_main)):
        model = tiger_mod.TIGER(SYN_T5).to(DEVICE)
        model.load_state_dict(copy.deepcopy(start_state))
        off.set_seed_for(1)
        loader = GenRecDataLoader(replica_ds, batch_size=SYN["n_users"],
                                  shuffle=True, num_workers=0)
        replicas[tag] = (model,
                         tiger_mod.train(model, loader,
                                         torch.optim.Adam(model.parameters(),
                                                          lr=off.lr),
                                         torch.device(DEVICE)))

    off_out = off.recommender_phase(20, 1, sid_rec["sid_path"], sid_rec["version"])

    sia_calls = sum(p.calls for k, p in probes.items() if k.startswith(("E2 ", "E3 ", "E4 ")))
    psa_calls = sum(p.calls for k, p in probes.items() if k.startswith(("E5 ", "E6 ")))
    check("switches off: the SIA and PSA paths are provably not entered "
          "(the same counters were non-zero on the switches-on run, so the "
          "zero is a measurement, not an untested probe)",
          sia_calls == 0 and psa_calls == 0
          and sum(v for k, v in on_calls.items() if k.startswith(("E2 ", "E3 ", "E4 ",
                                                                  "E5 ", "E6 "))) > 0,
          f"SIA={sia_calls} PSA={psa_calls} calls off; "
          f"{sum(v for k, v in on_calls.items() if k[0] == 'E')} module calls on")
    check("switches off: TIGER.forward(return_hidden=True) is never called, so "
          "the E1/E4/E6 code path is not even entered",
          probes["E1  TIGER.forward(return_hidden=True)"].calls == 0)
    check("switches off: the mounts reached the frozen loop switched off, so its "
          "own ``sia.enabled`` / ``psa.enabled`` gate is False -- the ported code "
          "was not entered at all, rather than entered with zero weight",
          train_kwargs and all(s is not True and p is not True for s, p in train_kwargs)
          and any(s is True and p is True for s, p in _train_kwargs_seen[0]),
          f"off (enabled flags)={sorted(set(train_kwargs), key=str)}, "
          f"on={sorted({(s is True, p is True) for s, p in _train_kwargs_seen[0]})}")
    for tag, (model, loss) in replicas.items():
        check(f"switches off: the E9 recommender machinery reproduces the {tag} "
              f"Clean TIGER epoch bitwise -- mean loss and every resulting weight",
              same_bytes(loss, off_out["train_loss"])
              and tensors_bitwise_equal(list(model.state_dict().values()),
                                        list(off.tiger.state_dict().values())),
              f"E9 {off_out['train_loss']!r} vs {tag} {loss!r}")

    # ------------------------------------------------------ resume / idempotency
    print("      -- resume and idempotency")
    before_versions = len(SR.list_versions(sid_base, DATASET))
    before_refresh = probes["E8  sid_refresh.refresh_sid"].calls
    again = scheduler.run()
    check("re-running a completed scheduler is a no-op: no new SID version, no "
          "new refresh, every cycle still complete",
          len(again) == 2 and all(r["state"] == "recommender_done" for r in again)
          and len(SR.list_versions(sid_base, DATASET)) == before_versions
          and probes["E8  sid_refresh.refresh_sid"].calls == before_refresh,
          f"{before_versions} SID versions before and after")

    crash_run = os.path.join(tmp, "run_crash")
    crash_pipe = build_pipeline(root, os.path.join(tmp, "sid_crash"), crash_run)
    real_refresh = crash_pipe.refresh_phase      # the real bound method
    # the patch has to precede build_scheduler: the scheduler captures the bound
    # method, so patching the instance afterwards would change nothing
    fail_once = {"cycle0": True}

    def flaky(cycle_index, epoch_index, tokenizer_ckpt):
        if cycle_index == 0 and fail_once["cycle0"]:
            fail_once["cycle0"] = False
            raise SR.SidRefreshError("simulated generator failure for E10")
        return real_refresh(cycle_index, epoch_index, tokenizer_ckpt)

    crash_pipe.refresh_phase = flaky
    crash_sched = AP.build_scheduler(crash_pipe, crash_run, 2, initial["ckpt_path"])
    aborted = raises(crash_sched.run, A.RefreshAborted)
    m0 = A.CycleManifest.load(A.cycle_manifest_path(crash_run, 0))
    check("a refresh failure aborts the cycle before the recommender runs",
          aborted and m0.state == "tokenizer_done"
          and m0.get("recommender") is None
          and [e["kind"] for e in m0.get("errors", [])] == ["SidRefreshError"],
          f"state={m0.state!r}")

    probes["E9  pipeline.tokenizer_phase"].reset()
    crash_sched.run()
    m0 = A.CycleManifest.load(A.cycle_manifest_path(crash_run, 0))
    check("resuming re-enters at the recorded state and finishes the cycle",
          m0.state == "recommender_done" and m0.get("sid") is not None
          and m0.get("evaluation") is not None,
          f"state={m0.state!r}")
    check("... without re-training the tokenizer epoch it had already recorded",
          probes["E9  pipeline.tokenizer_phase"].calls == 1,
          f"{probes['E9  pipeline.tokenizer_phase'].calls} tokenizer phase(s) "
          "on the resumed run (cycle 1 only)")

    return records, pipeline


# --------------------------------------------------------------------------- #
# [4] scope
# --------------------------------------------------------------------------- #
def part4_scope():
    print("\n[4] scope: the frozen directories are untouched")
    status = subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                            capture_output=True, text=True).stdout.splitlines()
    modified = {l[3:].strip() for l in status if l[:2].strip() == "M"}
    check("model/ and rqvae/ carry no E10 change",
          modified <= PRE_E10_MODIFIED, f"modified: {sorted(modified)}")
    check("clean_tiger/ is untouched by E10",
          not [l for l in status if "clean_tiger" in l])
    grep = subprocess.run(
        ["grep", "-rn", "--include=*.py", "-E",
         "^[[:space:]]*(from|import)[[:space:]]+tiger_ext",
         os.path.join(REPO, "model"), os.path.join(REPO, "rqvae")],
        capture_output=True, text=True)
    check("nothing under model/ or rqvae/ imports tiger_ext",
          grep.returncode == 1 and not grep.stdout.strip())
    hits = subprocess.run(
        ["grep", "-rn", "-E", "|".join(FORBIDDEN_SUBSTRINGS),
         os.path.join(REPO, "model", "main.py")],
        capture_output=True, text=True)
    check("model/main.py carries no historical mechanism at all, so the T5 side "
          "has nothing to trip",
          hits.returncode == 1 and not hits.stdout.strip(),
          "the historical set lives entirely in rqvae/trainer.py")
    # G1 also asks about the CSG4Rec line (C1-C6), which is not built yet: the
    # strongest statement available is that no trace of it exists to be run
    hist = subprocess.run(
        ["grep", "-rniE", "csg_|lightgcn|graph_conv",
         os.path.join(REPO, "tiger_ext"), os.path.join(REPO, "model"),
         os.path.join(REPO, "rqvae")],
        capture_output=True, text=True)
    check("nothing from the CSG4Rec line (C1-C6) exists in any runtime module, "
          "so it cannot be silently active",
          hist.returncode == 1 and not hist.stdout.strip(),
          "no csg/lightgcn/graph_conv symbol under tiger_ext/, model/, rqvae/")


# --------------------------------------------------------------------------- #
# [5] regression
# --------------------------------------------------------------------------- #
REGRESSION_SUITES = ["exp_e1_consistency", "exp_e2_encoder_interface",
                     "exp_e2_sia_adapter", "exp_e3_code_distribution",
                     "exp_e4_sia_loss", "exp_e5_psa", "exp_e7_stage_freeze",
                     "exp_e8_sid_refresh", "exp_e9_alternating"]
COUNT_RE = re.compile(r"(\d+)/(\d+) checks passed")


def run_regression():
    print("\n[5] regression: the E1-E9 suites, re-run after E10")
    total_ok = total_n = 0
    for name in REGRESSION_SUITES:
        env = dict(os.environ)
        env.pop("EXP_E10_DEVICE", None)
        if name == "exp_e9_alternating":
            env["EXP_E9_REGRESSION"] = "skip"    # this script runs E1-E8 itself
        proc = subprocess.run([sys.executable, "-I", f"integration/{name}.py"],
                              cwd=REPO, env=env, capture_output=True, text=True)
        match = None
        for line in proc.stdout.splitlines():
            m = COUNT_RE.search(line)
            if m:
                match = m
        ok, n = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
        total_ok += ok
        total_n += n
        passed = proc.returncode == 0 and match is not None and ok == n
        detail = "" if passed else " | ".join(
            ln for ln in proc.stderr.splitlines() if ln.strip())[-200:]
        check(f"{name}: {ok}/{n}", passed, detail)
    check(f"the E1-E9 suites are unaffected by E10 ({total_ok}/{total_n})",
          total_ok == total_n and total_n > 380, f"{total_ok}/{total_n} checks")


# --------------------------------------------------------------------------- #
def main():
    print("=" * 76)
    print(f"EXP-E10 -- ETEGRec integration smoke test "
          f"(device={DEVICE}, regression={REGRESSION})")

    from tiger_ext import alt_pipeline as AP
    mods = AP.frozen_modules()               # sets sys.path for model/ + rqvae/

    tmp = tempfile.mkdtemp(prefix="exp_e10_")
    spy = ReadSpy()
    try:
        part1_probe_harness(tmp)
        PristineRQVAE, pristine_main = build_pristine(tmp)

        # --- the probes and tripwires are installed once, before any real run --
        probes = install_probes(mods)
        # what the switch state actually looked like when it reached the frozen
        # loop: model/main.py::train gates on ``sia is not None and sia.enabled``,
        # so the *enabled* flag is the thing that decides whether the ported code
        # runs -- None (no mount at all) and False (mount present but switched
        # off) are both inert, True is not
        def _switched(mount):
            return None if mount is None else bool(mount.enabled)

        probes["G0  tiger_main.train(sia, psa)"] = Probe(
            mods["tiger_main"], "train", "G0",
            on_call=lambda a, k: train_kwargs.append(
                (_switched(k.get("sia")), _switched(k.get("psa"))))
        ).install()
        wires = {"L_trans (ContrastiveLoss)": Tripwire("L_trans"),
                 "L_balance (BalanceLoss)": Tripwire("L_balance"),
                 "historical L_align (_get_align_loss)": Tripwire("L_align")}
        trainer_log = []
        RQVAETrainer = mods["RQVAETrainer"]
        orig_init = RQVAETrainer.__init__

        recording = {"on": False}     # off while the negative controls run

        def patched_init(self, *a, **k):
            orig_init(self, *a, **k)
            self.contrastive_loss = wires["L_trans (ContrastiveLoss)"]
            self.balance_loss = wires["L_balance (BalanceLoss)"]
            self._get_align_loss = wires["historical L_align (_get_align_loss)"]
            if recording["on"]:
                trainer_log.append(self)

        RQVAETrainer.__init__ = patched_init
        try:
            print("\n[1b] negative controls: the tripwires fire when the "
                  "historical mechanisms are genuinely switched on")
            train_negative_controls(mods, wires, tmp)

            spy.start()
            part2_g0(tmp, mods, PristineRQVAE, pristine_main)
            part3_smoke(tmp, mods, probes, wires, trainer_log, spy, recording)
            spy.stop()
        finally:
            RQVAETrainer.__init__ = orig_init
            for probe in probes.values():
                probe.restore()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    part4_scope()
    if REGRESSION != "skip":
        run_regression()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 76)
    print(f"EXP-E10: {passed}/{len(RESULTS)} checks passed  (device={DEVICE})")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  -- {detail}")
    print("=" * 76)
    return 0 if passed == len(RESULTS) else 1


class _Args:
    """The attribute set rqvae/trainer.py::Trainer reads."""

    def __init__(self, ckpt_dir, device, num_emb_list, **over):
        self.lr = 1e-3
        self.learner = "AdamW"
        self.lr_scheduler_type = "linear"
        self.weight_decay = 1e-4
        self.epochs = 1
        self.warmup_epochs = 0
        self.save_limit = 1
        self.eval_step = 1
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.num_emb_list = num_emb_list
        self.lambda_trans = 0.0
        self.lambda_balance = 0.0
        self.tau = 0.1
        self.trans_graph_path = None
        self.align_weight = 0.0
        self.align_data_path = None
        self.align_tau = 0.1
        for k, v in over.items():
            setattr(self, k, v)


def train_negative_controls(mods, wires, tmp):
    """Switch each historical mechanism ON and show its tripwire actually fires.

    The point is not that the wire can raise -- it is that the wire sits on the
    path the frozen loop really takes when the switch is on, so "0 activations"
    in the smoke run is a measurement rather than an assumption.
    """
    from datasets import EmbDataset
    from models.rqvae import RQVAE
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cpu")
    emb_ds = EmbDataset(os.path.join(SMOKE, "data", "Toys", "item_emb.parquet"))
    xs = torch.as_tensor(emb_ds.embeddings[:64], dtype=torch.float32)
    # every one of the three historical mechanisms sets is_track_b, and that path
    # unpacks (idx, embedding) batches -- the index is the parquet row
    trackb = DataLoader(TensorDataset(torch.arange(64), xs), batch_size=8,
                        shuffle=False)

    def new_model():
        m = RQVAE(in_dim=emb_ds.dim, **RQVAE_CFG)
        m.eval()
        return m

    def run(args, loader, n=1):
        model = new_model()
        model.train()
        trainer = mods["RQVAETrainer"](args, model, len(loader))
        try:
            for epoch in range(n):
                trainer._train_epoch(loader, epoch)
        except AssertionError:
            return trainer, None
        except Exception as exc:                                 # noqa: BLE001
            return trainer, f"{type(exc).__name__}: {exc}"
        return trainer, None

    def one(label, wire_name, args):
        for wire in wires.values():
            wire.calls = 0
        _, err = run(args, trackb)
        check(f"negative control: {label} really fires "
              "(so a zero on the switches-off run is a measurement)",
              wires[wire_name].calls >= 1,
              f"{wires[wire_name].calls} activation(s)"
              + (f" [{err}]" if err else ""))
        for wire in wires.values():
            wire.calls = 0

    # L_balance needs no extra input beyond the switch
    one("with lambda_balance=1.0 the L_balance tripwire",
        "L_balance (BalanceLoss)",
        _Args(os.path.join(tmp, "nc_bal"), "cpu", RQVAE_CFG["num_emb_list"],
              lambda_balance=1.0))

    # L_trans needs a transition graph; make every consecutive pair a relation
    trans = os.path.join(tmp, "nc_trans.npy")
    np.save(trans, np.array([[i, i + 1] for i in range(63)], dtype=np.int64))
    one("with lambda_trans=1.0 and a transition graph the L_trans tripwire",
        "L_trans (ContrastiveLoss)",
        _Args(os.path.join(tmp, "nc_trans"), "cpu", RQVAE_CFG["num_emb_list"],
              lambda_trans=1.0, trans_graph_path=trans))

    # L_align needs a behaviour table -- the archived E50 b2 path, which reads a
    # parquet of per-item target embeddings
    import pandas as pd
    table = os.path.join(tmp, "nc_align.parquet")
    rng = np.random.default_rng(0)
    pd.DataFrame({"embedding": [rng.normal(size=128).astype(np.float32)
                                for _ in range(64)]}).to_parquet(table, index=False)
    one("with align_weight=1.0 and an alignment table the historical L_align "
        "tripwire",
        "historical L_align (_get_align_loss)",
        _Args(os.path.join(tmp, "nc_align"), "cpu", RQVAE_CFG["num_emb_list"],
              align_weight=1.0, align_data_path=table, align_tau=0.1))


if __name__ == "__main__":
    sys.exit(main())
