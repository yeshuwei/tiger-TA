"""C6 -- CSG smoke test (CSG4Rec -> TIGER port).

Contract: ``integration/plan.md`` C6 table.

C6 is a **smoke test**, not an effect experiment. It answers exactly four
questions and nothing else:

1. **Do the four weights run?** ``lambda_csg in {0, 1e-4, 1e-3, 1e-2}`` each
   drive the whole chain RQ-VAE -> SID -> T5 -> recommendation -> evaluation.
2. **Does the constraint really enter training?** The hook is called on every
   step, its term lands in the total loss, the codebooks receive gradient and
   the encoder receives none from it.
3. **Is the TIGER pipeline still intact?** ``lambda_csg = 0`` is bitwise-equal to
   the Clean TIGER chain, the E7 stage contract holds, and no user information
   reaches the SID.
4. **Which** ``kmeans_init`` **setting is safe at this budget?**

Nothing here is new plumbing. :func:`tiger_ext.qspace_train.train_tokenizer`
already runs the frozen tokenizer loop with the hook attached (C5), and
:class:`tiger_ext.alt_pipeline.TigerAlternatingPipeline` already runs the other
three phases (E9). This module only *composes* the two and measures the four
things above.

Why the composition and not an edit
-----------------------------------
``alt_pipeline.py`` constructs its tokenizer ``Trainer`` without a constraint at
its own line and has been frozen since E9; editing it would reopen E9 and break
the C1-C4 sha256 pins. So C6 runs C5's tokenizer entry to produce the
checkpoint, then loads that checkpoint into the pipeline
(``_load_tokenizer``, which also restores ``vq.initted``) and lets the
pipeline's own ``refresh_phase`` / ``recommender_phase`` / ``eval_phase`` run the
rest. **No existing runtime file is modified.**

The arms are TIGER + CSG and nothing else: ``lambda_sia = lambda_psa = 0.0``, so
the historical mounts stay disabled and the only difference between the arms is
the one switch that C5 owns.

What this module deliberately does **not** claim
------------------------------------------------
Recall/NDCG reported here prove only that the full pipeline runs with the
constraint switched on. They are smoke output and must never be cited for or
against CSG; the operating point and the real comparison belong to the effect
experiment that starts after C6 passes. ``CST_SMOKE_PENDING`` records that.

No identifier in this module contains the substring the E10 guard greps for; the
guard scans ``tiger_ext/`` and this file lives there.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .alt_pipeline import (
    NUM_WORKERS, RQVAE_CONFIG, T5_CONFIG, T5_TOPK, TigerAlternatingPipeline,
)
from .collab_relation import RELATION_K, build_relation
from .qspace_constraint import (
    REBUILD_ATOL, REBUILD_RTOL, rebuild_quantized,
)
from .qspace_train import (
    CSGHook, CSGTrainerConfig, frozen_tokenizer_modules, train_tokenizer,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------- #
# the pinned smoke contract
# --------------------------------------------------------------------------- #

#: The four arms. Inherited verbatim from C5's candidate list plus the off arm;
#: C6 adds no lambda of its own and runs no hyper-parameter search.
CST_SMOKE_LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2)

#: C2's ``--knn_k``; unchanged, and checked against ``RELATION_K`` below.
CST_SMOKE_K = 20

CST_SMOKE_ARM_NOTE = (
    "A = Clean TIGER with the constraint switched off (lambda_csg = 0); "
    "B/C/D = the same chain with lambda_csg = 1e-4 / 1e-3 / 1e-2. The three "
    "weights are C5's measured candidates (relative contribution 1.551% / "
    "15.535% / 154.972%); C6 neither re-searches them nor picks a winner."
)

#: The synthetic fixture -- E10's shape, with an ``ItemID`` column added so C1's
#: canonical-item lookup and C2's relation builder both work on it.
CST_SMOKE_SYN = dict(n_items=1024, dim=32, n_users=24, seq_len=12,
                     n_valid=8, n_test=8, n_centres=8)

CST_SMOKE_SYN_RQVAE = dict(
    num_emb_list=[128, 128, 128], e_dim=32, layers=[16, 8], kmeans_init=True,
    kmeans_iters=5, sk_iters=5, sk_epsilons=[0.0, 0.0, 0.003],
    batch_size=1024, lr=5e-3,
)

CST_SMOKE_SYN_T5 = dict(
    num_layers=1, num_decoder_layers=1, d_model=16, d_ff=32, num_heads=2,
    d_kv=4, dropout_rate=0.0, vocab_size=1025, pad_token_id=0, eos_token_id=0,
    feed_forward_proj="relu", max_len=20,
)

#: The real-data arm's budget == ``clean_tiger/smoke_test.sh``'s RQ-VAE budget.
#: ``limit_train`` is the pipeline's own knob and caps the recommender *and* the
#: evaluation loader, which is what keeps the arm to minutes.
CST_SMOKE_TOYS = dict(rqvae_epochs=2, rqvae_batch=1024, limit_train=64,
                      t5_batch=32, infer_size=32, beam_size=10)

CST_SMOKE_TOYS_NOTE = (
    "The real-data arm is Toys because C1/C2's artifacts exist only for "
    "Toys / Sport / Beauty and Musical_Instruments has neither "
    "behavior_emb_itemcf.parquet nor a SASRec checkpoint, so the constraint "
    "cannot be loaded for it. Toys carries two known confounds that C6 records "
    "rather than resolves: the 2-epoch 4096-d tokenizer budget is far below the "
    "clean baseline's, and the first-batch geometry interacts with "
    "kmeans_init (see CST_SMOKE_PENDING). Its Recall/NDCG are engineering "
    "evidence that the chain ran, nothing more."
)

#: What C6 hands to the effect experiment. Listed here so the boundary is in the
#: code and not only in a report.
CST_SMOKE_PENDING = (
    {
        "id": "lambda-operating-point",
        "left_to": "the effect experiment, after C6 passes",
        "why": "C6 shows four arms run; it does not choose among them",
    },
    {
        "id": "recall-ndcg-as-evidence",
        "left_to": "the effect experiment",
        "why": "C6's Recall/NDCG are pipeline-smoke output; citing them for or "
               "against CSG is exactly what the brief forbids",
    },
    {
        "id": "tokenizer-structure-metrics",
        "left_to": "the effect experiment",
        "why": "codebook usage / collision trends are recorded here as "
               "engineering readings, not compared as an outcome",
    },
    {
        "id": "toys-confounds",
        "left_to": "the effect experiment (or the dataset choice that precedes it)",
        "why": "2-epoch 4096-d under-training and the first-batch / k-means "
               "geometry are confounds of the real-data arm",
    },
    {
        "id": "straight-through-to-encoder",
        "left_to": "a ruling; unchanged from C5",
        "why": "C5 measured the path as reachable but did not adopt it "
               "(it needs either a redefined objective or a core edit)",
    },
)


class QSpaceSmokeError(RuntimeError):
    """The smoke chain was asked for something it must not do silently."""


# --------------------------------------------------------------------------- #
# the chain spec
# --------------------------------------------------------------------------- #
@dataclass
class SmokeSpec:
    """Everything one arm of the chain needs, and nothing about the weight."""

    name: str
    dataset: str
    data_root: str
    artifact_dir: str = None
    relation_prefix: str = None
    rqvae_config: dict = field(default_factory=dict)
    t5_config: dict = field(default_factory=dict)
    rqvae_epochs: int = 1
    rqvae_batch: int = 1024
    rqvae_limit: int = None
    device: str = "cpu"
    seed: int = 2024
    limit_train: int = None
    t5_batch: int = 32
    infer_size: int = 8
    beam_size: int = 2
    t5_lr: float = 1e-4
    topk_list: list = None
    k: int = CST_SMOKE_K
    note: str = ""

    @property
    def embedding_path(self):
        return os.path.join(self.data_root, self.dataset, "item_emb.parquet")

    def report(self):
        return {"name": self.name, "dataset": self.dataset,
                "data_root": self.data_root, "device": self.device,
                "seed": int(self.seed), "limit_train": self.limit_train,
                "rqvae_epochs": int(self.rqvae_epochs),
                "rqvae_batch": int(self.rqvae_batch), "k": int(self.k),
                "beam_size": int(self.beam_size), "note": self.note}


def synthetic_spec(root, *, device="cpu", seed=2024, name="C6Syn"):
    """The synthetic fixture's spec (the fixture itself is :func:`build_fixture`)."""
    return SmokeSpec(
        name=name, dataset=name, data_root=root, device=device, seed=seed,
        rqvae_config=dict(CST_SMOKE_SYN_RQVAE), t5_config=dict(CST_SMOKE_SYN_T5),
        rqvae_epochs=1, rqvae_batch=1024, t5_batch=32, infer_size=8, beam_size=2,
        limit_train=None, topk_list=list(T5_TOPK),
        note="synthetic fixture; isolates the switch from the dataset confounds")


def toys_spec(data_root, *, device="cpu", seed=2024, artifact_dir=None,
              relation_prefix=None, name="Toys"):
    """The real-data arm's spec: the frozen Clean TIGER shapes at smoke budget."""
    cfg = dict(CST_SMOKE_TOYS)
    return SmokeSpec(
        name=f"{name}-smoke", dataset=name, data_root=data_root,
        artifact_dir=artifact_dir, relation_prefix=relation_prefix,
        device=device, seed=seed, rqvae_config=dict(RQVAE_CONFIG),
        t5_config=dict(T5_CONFIG), rqvae_limit=None, topk_list=list(T5_TOPK),
        note=CST_SMOKE_TOYS_NOTE, **cfg)


# --------------------------------------------------------------------------- #
# the synthetic fixture
# --------------------------------------------------------------------------- #
def build_fixture(root, spec=None, *, rng=None):
    """Write an E10-shaped synthetic dataset under ``root``; return its data dir.

    The fixture is data plumbing, not a training loop, so it is written here
    rather than imported from a harness -- ``integration/exp_*.py`` files are
    checks, not libraries, and importing one would tie C6 to E10's internals.

    Unlike E10's fixture this one carries an ``ItemID`` column, because C1's
    canonical-item lookup and C2's relation builder both key on it, and the C2
    relation artifact is produced by **C2's own builder** rather than by hand.
    """
    import pandas as pd

    spec = spec or SmokeSpec(name="C6Syn", dataset="C6Syn", data_root=root)
    cfg = CST_SMOKE_SYN
    rng = rng if rng is not None else np.random.default_rng(0)
    data_dir = os.path.join(root, spec.dataset)
    os.makedirs(data_dir, exist_ok=True)

    n_items, dim, centres_n = cfg["n_items"], cfg["dim"], cfg["n_centres"]
    centres = rng.normal(size=(centres_n, dim)).astype(np.float32) * 1.5
    which = rng.integers(0, centres_n, size=n_items)
    emb = (centres[which]
           + rng.normal(size=(n_items, dim)).astype(np.float32)).astype(np.float32)

    item_ids = np.arange(1, n_items + 1, dtype=np.int64)
    pd.DataFrame({"ItemID": item_ids,
                  "embedding": [emb[i] for i in range(n_items)]}).to_parquet(
        os.path.join(data_dir, "item_emb.parquet"), index=False)

    def interactions(n_rows):
        history, target = [], []
        for _ in range(n_rows):
            seq = rng.integers(1, n_items + 1, size=cfg["seq_len"]).tolist()
            history.append(seq[:-1])
            target.append(seq[-1])
        return pd.DataFrame({"history": history, "target": target})

    for split, rows in (("train", cfg["n_users"]), ("valid", cfg["n_valid"]),
                        ("test", cfg["n_test"])):
        interactions(rows).to_parquet(os.path.join(data_dir, f"{split}.parquet"),
                                      index=False)

    prefix = os.path.join(root, "collab_relation", f"{spec.dataset}_k{spec.k}")
    relation = build_relation(emb, item_ids, k=spec.k, source="c6_synthetic")
    relation.save(prefix)
    spec.relation_prefix = prefix
    spec.artifact_dir = os.path.join(root, "collab_relation")
    return data_dir


# --------------------------------------------------------------------------- #
# one full chain
# --------------------------------------------------------------------------- #
def pinned_loader(data, config):
    """A shuffle-free loader: every arm sees the same rows in the same order."""
    return torch.utils.data.DataLoader(data, batch_size=config.batch_size,
                                       shuffle=False, num_workers=0)


def build_hook(spec, lambda_csg):
    """C5's hook for this spec's relation, through C5's own constructor."""
    if int(spec.k) != int(RELATION_K):
        raise QSpaceSmokeError(
            f"the smoke pins k={CST_SMOKE_K} (C2's own default); spec asks {spec.k}")
    return CSGHook.for_dataset(
        spec.dataset, lambda_csg=float(lambda_csg), k=int(spec.k),
        data_dir=spec.data_root, artifact_dir=spec.artifact_dir,
        relation_prefix=spec.relation_prefix)


def run_chain(spec, *, lambda_csg, hook=None, tag="arm", pristine=False,
              workdir=None, keep=False):
    """Run RQ-VAE -> SID -> T5 -> recommendation -> evaluation exactly once.

    Args:
        spec: :class:`SmokeSpec`.
        lambda_csg: the arm's weight. ``0.0`` switches the constraint off.
        hook: a pre-built hook (the caller keeps its counters). Built from
            ``spec`` and ``lambda_csg`` when omitted.
        tag: the arm's name, used for the work directory.
        pristine: build the tokenizer's ``Trainer`` **without** the constraint
            argument at all -- the exact Clean TIGER call signature, used by the
            bitwise check. The hook object still exists and stays uncalled.
        workdir: where the tokenizer checkpoint, SID and run directory go.
        keep: leave ``workdir`` in place (the harness reads its artifacts).

    Returns:
        A report dict with the phase order, every artifact path and hash, the
        hook counters, the evaluation numbers and the wall-clock split.
    """
    if hook is None:
        hook = build_hook(spec, lambda_csg)

    workdir = workdir or tempfile.mkdtemp(prefix=f"c6_{tag}_")
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    config = CSGTrainerConfig(
        dataset=spec.dataset, data_root=spec.data_root,
        artifact_dir=spec.artifact_dir, relation_prefix=spec.relation_prefix,
        lambda_csg=float(lambda_csg), k=int(spec.k), validate=True,
        device=spec.device, seed=int(spec.seed), epochs=int(spec.rqvae_epochs),
        eval_step=1, batch_size=int(spec.rqvae_batch), num_workers=1,
        warmup_epochs=0, ckpt_dir=os.path.join(workdir, "tokenizer"),
        limit=spec.rqvae_limit, rqvae_config=dict(spec.rqvae_config))

    trainer_factory = None
    if pristine:
        frozen = frozen_tokenizer_modules()["RQVAETrainer"]

        def trainer_factory(args, model, data_num, csg):
            # no constraint argument: the pristine signature, csg defaults None
            return frozen(args, model, data_num)

    t0 = time.time()
    report = train_tokenizer(config, hook=hook, trainer_factory=trainer_factory,
                             loader_factory=pinned_loader)
    tokenizer_seconds = time.time() - t0

    ckpt_path = os.path.join(report["ckpt_dir"], "best_collision_model.pth")
    if not os.path.isfile(ckpt_path):
        raise QSpaceSmokeError(f"the tokenizer stage wrote no checkpoint: {ckpt_path}")

    t1 = time.time()
    pipeline = TigerAlternatingPipeline(
        spec.dataset, spec.data_root, os.path.join(workdir, "sid"),
        os.path.join(workdir, "run"), device=spec.device,
        rqvae_config=dict(spec.rqvae_config), t5_config=dict(spec.t5_config),
        lambda_sia=0.0, lambda_psa=0.0, seed=int(spec.seed),
        batch_size=int(spec.t5_batch), infer_size=int(spec.infer_size),
        beam_size=int(spec.beam_size), topk_list=list(spec.topk_list or T5_TOPK),
        lr=float(spec.t5_lr), num_workers=NUM_WORKERS,
        limit_train=spec.limit_train)
    pipeline._load_tokenizer(ckpt_path)
    refresh = pipeline.refresh_phase(0, 0, ckpt_path)
    recommender = pipeline.recommender_phase(0, 0, refresh["sid_path"],
                                             refresh["version"])
    evaluation = pipeline.eval_phase(0, 0, recommender["ckpt_path"],
                                     refresh["sid_path"])
    downstream_seconds = time.time() - t1

    from .sid_refresh import sha256_file
    from .stage import read_stage_manifest, stage_manifest_path

    def manifest_of(path):
        mp = stage_manifest_path(path)
        return read_stage_manifest(mp) if os.path.isfile(mp) else None

    result = {
        "tag": tag, "spec": spec.report(), "lambda_csg": float(lambda_csg),
        "pristine": bool(pristine), "workdir": os.path.abspath(workdir),
        "tokenizer": {"ckpt_path": os.path.abspath(ckpt_path),
                      "sha256": sha256_file(ckpt_path),
                      "best_loss": report["best_loss"],
                      "best_collision_rate": report["best_collision_rate"],
                      "n_batches": report["n_batches"],
                      "n_steps": report["n_steps"],
                      "n_rows": report["n_rows"], "in_dim": report["in_dim"],
                      "optimizer_params": report["optimizer_params"],
                      "hook_in_optimizer": report["hook_in_optimizer"]},
        "counters": dict(hook.counters),
        "history": [dict(h) for h in hook.history],
        "last_diagnostics": dict(hook.last_diagnostics or {}),
        "refresh": {"sid_path": refresh["sid_path"],
                    "sid_sha256": refresh["sid_sha256"],
                    "version": refresh["version"],
                    "device": refresh["device"],
                    "stats": refresh["sid_stats"], "diff": refresh["diff"],
                    "manifest_path": refresh["manifest_path"]},
        "recommender": {"ckpt_path": recommender["ckpt_path"],
                        "sha256": recommender["sha256"],
                        "train_loss": recommender["train_loss"],
                        "sid_version_used": recommender["sid_version_used"],
                        "sid_sha256_used": recommender["sid_sha256_used"],
                        "n_train_sequences": recommender["n_train_sequences"],
                        "n_train_batches": recommender["n_train_batches"],
                        "mounts": dict(recommender["mounts"]),
                        "handoff": dict(recommender["handoff"]),
                        "freeze": recommender["freeze"],
                        "losses": recommender["losses"]},
        "evaluation": {"recalls": dict(evaluation["recalls"]),
                       "ndcgs": dict(evaluation["ndcgs"]),
                       "n_eval_sequences": evaluation["n_eval_sequences"],
                       "beam_size": evaluation["beam_size"],
                       "sid_sha256": evaluation["sid_sha256"]},
        "phase_order": [(e["phase"], e["cycle_index"], e["epoch_index"])
                        for e in pipeline.phase_log],
        "stage_manifests": {"tokenizer": manifest_of(ckpt_path),
                            "recommender": manifest_of(recommender["ckpt_path"])},
        "seconds": {"tokenizer": tokenizer_seconds,
                    "downstream": downstream_seconds,
                    "total": tokenizer_seconds + downstream_seconds},
    }
    if not keep:
        shutil.rmtree(workdir, ignore_errors=True)
        result["workdir_removed"] = True
    return result


# --------------------------------------------------------------------------- #
# the k-means / first-batch geometry probe
# --------------------------------------------------------------------------- #
def probe_kmeans(spec, *, kmeans_init, epochs=None, batch_size=None,
                 tag=None, workdir=None):
    """Measure what C6's §三 ruling needs, on the tokenizer stage alone.

    Four readings, in the order the brief asks for them:

    * ``first_batch_rebuild_delta`` -- ``max |x_q - sum_l W_l[I_l]|`` on the
      first batch. This is the quantity C3's reconstruction guard bounds
      (``REBUILD_ATOL = 1e-4``), so it is the one the safe/unsafe test uses.
    * ``first_batch_latent_delta`` -- ``max |x_q - z|`` on the same batch, i.e.
      whether the codebook reproduces the encoder's own latent exactly.
    * ``step0_codebook_grad`` -- the per-level gradient norm of the codebooks on
      the first optimiser step, plus whether *any* encoder parameter received
      one.
    * ``epoch_recon`` / ``usage`` -- the per-epoch mean reconstruction loss, and
      the per-level code usage on the first batch and over the first epoch.

    The model is not touched and the loop is not forked: the forward output is
    read through a wrapper installed on a ``trainer_factory``-built instance,
    and the gradient through tensor hooks on the codebooks themselves.
    """
    mods = frozen_tokenizer_modules()
    base = mods["RQVAETrainer"]
    cfg = dict(spec.rqvae_config)
    cfg["kmeans_init"] = bool(kmeans_init)
    epochs = int(epochs if epochs is not None else spec.rqvae_epochs)
    batch_size = int(batch_size if batch_size is not None else spec.rqvae_batch)
    tag = tag or f"{spec.name}-kmeans{int(bool(kmeans_init))}"
    workdir = workdir or tempfile.mkdtemp(prefix=f"c6_{tag}_")

    config = CSGTrainerConfig(
        dataset=spec.dataset, data_root=spec.data_root,
        artifact_dir=spec.artifact_dir, relation_prefix=spec.relation_prefix,
        lambda_csg=0.0, k=int(spec.k), validate=True, device=spec.device,
        seed=int(spec.seed), epochs=epochs, eval_step=1, batch_size=batch_size,
        num_workers=1, warmup_epochs=0,
        ckpt_dir=os.path.join(workdir, "tokenizer"),
        limit=spec.rqvae_limit, rqvae_config=cfg)

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    seen = {"first": None, "recon": [], "steps": 0, "batches": [],
            "e0_start": None, "e0_end": None}
    grads = []
    enc_grads = []
    holder = {}

    class Recorder(base):
        """The frozen Trainer, with its per-epoch reconstruction loss kept."""

        def _train_epoch(self, train_data, epoch_idx):
            if epoch_idx == 0:
                seen["e0_start"] = len(seen["batches"])
            total, recon = super()._train_epoch(train_data, epoch_idx)
            if epoch_idx == 0:
                seen["e0_end"] = len(seen["batches"])
            n = max(1, len(train_data))
            seen["recon"].append(float(recon) / n)
            return total, recon

    def factory(args, model, data_num, csg):
        original = model.forward
        state = {"step": 0}

        def wrapped(x, *a, **kw):
            want_z = bool(kw.get("return_z", False))
            want_xq = bool(kw.get("return_xq", False))
            out = original(x, *a, **dict(kw, return_z=True, return_xq=True))
            captured = {"x_q": out[4].detach().clone(), "z": out[3].detach().clone(),
                        "indices": out[2].detach().clone()}
            if state["step"] == 0 and seen["first"] is None:
                # ``sum_l W_l[I_l]`` has to be gathered from the codebooks as they
                # stood at *this* forward. Recomputing it after training would
                # compare a step-0 ``x_q`` against an epoch-3 codebook and measure
                # the optimiser, not C3's reconstruction guard.
                with torch.no_grad():
                    captured["rebuilt"] = rebuild_quantized(model, captured["indices"])
                seen["first"] = captured
            seen["batches"].append(captured)
            state["step"] += 1
            seen["steps"] = state["step"]
            width = 3 + int(want_z) + int(want_xq)
            return tuple(out[:width])

        model.forward = wrapped
        holder["model"] = model
        for level, layer in enumerate(model.rq.vq_layers):
            layer.embedding.weight.register_hook(
                lambda g, lv=level: grads.append((lv, float(g.detach().norm()))))
        for name, param in model.encoder.named_parameters():
            param.register_hook(
                lambda g, nm=name: enc_grads.append((nm, float(g.detach().norm()))))
        return Recorder(args, model, data_num, csg)

    report = train_tokenizer(config, trainer_factory=factory,
                             loader_factory=pinned_loader)

    first = seen["first"]
    if first is None:
        raise QSpaceSmokeError(f"{tag}: no forward pass was captured")

    # The two guards C3's QSpaceBatch.validate applies to the first batch, with
    # the polarities it uses. They point in **opposite** directions, and the one
    # C5 tripped is the second:
    #   * rebuild -- x_q must equal sum_l W_l[I_l]; deviation must be SMALL
    #     (validate raises when allclose fails, i.e. when it is large);
    #   * latent  -- x_q must NOT be indistinguishable from z; deviation must
    #     EXCEED REBUILD_ATOL (validate raises when it does not).
    # The degenerate first batch has x_q == z exactly, so it is the latent guard
    # that rejects it. Reading the rebuild delta as the safe/unsafe test would
    # invert the verdict.
    with torch.no_grad():
        rebuilt = first["rebuilt"]              # gathered at the first forward
        rebuild = float((first["x_q"] - rebuilt).abs().max())
        rebuild_ok = bool(torch.allclose(first["x_q"], rebuilt,
                                         atol=REBUILD_ATOL, rtol=REBUILD_RTOL))
        latent = float((first["x_q"] - first["z"]).abs().max())
        latent_ok = bool(latent > REBUILD_ATOL)

    usage = _usage(seen)
    window = usage.get("epoch0") or usage.get("first_batch") or {}
    usage_ok = bool(window.get("distinct_min", 0) > 1
                    and window.get("max_share_max", 1.0) < 1.0)
    n_levels = int(len(cfg["num_emb_list"]))
    n_enc = len(list(holder["model"].encoder.named_parameters()))
    step0 = [g for _, g in grads[:n_levels]]
    step0_enc = [g for _, g in enc_grads[:n_enc]]
    out = {
        "tag": tag, "kmeans_init": bool(kmeans_init), "epochs": epochs,
        "batch_size": batch_size, "dataset": spec.dataset,
        "first_batch_rows": int(first["indices"].shape[0]),
        "first_batch_rebuild_delta": rebuild,
        "first_batch_latent_delta": latent,
        "first_batch_rebuild_ok": rebuild_ok,
        "first_batch_latent_ok": latent_ok,
        "step0_codebook_grad": step0,
        "step0_codebook_grad_sum": float(sum(step0)),
        "step0_encoder_grad": step0_enc,
        "encoder_grad_seen": bool(any(g > 0.0 for g in step0_enc)),
        "epoch_recon": list(seen["recon"]),
        "epoch_recon_moves": bool(len(set(seen["recon"])) > 1),
        "usage": usage,
        "usage_ok": usage_ok,
        "best_loss": report["best_loss"],
        "best_collision_rate": report["best_collision_rate"],
        "n_batches": report["n_batches"], "n_rows": report["n_rows"],
        "ckpt_dir": report["ckpt_dir"],
    }
    out["safe"] = bool(rebuild_ok and latent_ok
                       and out["step0_codebook_grad_sum"] > 0.0
                       and out["epoch_recon_moves"] and out["usage_ok"])
    shutil.rmtree(workdir, ignore_errors=True)
    return out


def rebuild_delta_of(captured):
    """``max |x_q - sum_l W_l[I_l]|`` on one captured batch.

    This is the **rebuild** identity: a *small* value is the healthy reading,
    because C3's ``validate`` raises when ``x_q`` and the codebook sum fail
    ``allclose`` -- when this number is *large*. It is not the safe/unsafe test
    for a first batch; the latent gap (``|x_q - z|``) is, and it points the other
    way. Reported here so both can be read side by side.

    It reads the codebook sum the capture already carries rather than re-gathering
    it here: the optimiser moves the codebooks from the first step onward, so a
    late gather would compare a step-0 ``x_q`` against a later codebook and report
    the training rather than the identity. Captures without that tensor therefore
    raise instead of silently measuring the wrong thing.
    """
    if "rebuilt" not in captured:
        raise QSpaceSmokeError(
            "this capture did not record sum_l W_l[I_l] at its own forward, so "
            "the rebuild identity cannot be evaluated on it")
    return float((captured["x_q"] - captured["rebuilt"]).abs().max())


def _usage(seen):
    """Per-level code usage: distinct codes and the largest single-code share."""
    from collections import Counter

    start, end = seen.get("e0_start"), seen.get("e0_end")
    first_epoch = (seen["batches"][start:end]
                   if start is not None and end is not None
                   else seen["batches"])
    out = {}
    for label, batches in (("first_batch", seen["batches"][:1]),
                           ("epoch0", first_epoch)):
        if not batches:
            continue
        codes = torch.cat([b["indices"] for b in batches], dim=0)
        per_level = []
        for level in range(codes.shape[1]):
            counts = Counter(int(c) for c in codes[:, level].tolist())
            total = sum(counts.values())
            per_level.append({
                "level": level, "distinct": len(counts), "rows": total,
                "max_share": max(counts.values()) / total if total else 0.0,
            })
        out[label] = {
            "levels": per_level,
            "distinct_min": min(p["distinct"] for p in per_level),
            "distinct_max": max(p["distinct"] for p in per_level),
            "max_share_max": max(p["max_share"] for p in per_level),
            "rows": int(codes.shape[0]),
        }
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="python -I -m tiger_ext.qspace_smoke",
        description="C6 -- one smoke chain (RQ-VAE -> SID -> T5 -> eval) with "
                    "the Q-space constraint switched on or off")
    p.add_argument("--dataset", default="Toys",
                   help="Toys (real artifacts) or a synthetic fixture name")
    p.add_argument("--data-root", default=os.path.join(REPO, "data"))
    p.add_argument("--lambda-csg", type=float, default=0.0, dest="lambda_csg",
                   help="0 switches the constraint off (C5's default)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--synthetic", action="store_true",
                   help="build the E10-shaped fixture under --data-root first")
    p.add_argument("--probe-kmeans", choices=["true", "false", "both"],
                   default="both", dest="probe_kmeans",
                   help="also run the §三 first-batch geometry probe")
    p.add_argument("--json", default=None, help="write the report here")
    p.add_argument("--keep", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = args.data_root
    if args.synthetic:
        spec = synthetic_spec(root, device=args.device, seed=args.seed)
        build_fixture(root, spec, rng=np.random.default_rng(args.seed))
    else:
        spec = toys_spec(root, device=args.device, seed=args.seed)
    result = run_chain(spec, lambda_csg=args.lambda_csg, tag=spec.name,
                       workdir=None if not args.keep else os.path.join(
                           root, f"run_{spec.name}"), keep=args.keep)
    if args.probe_kmeans != "false":
        which = (True, False) if args.probe_kmeans == "both" else (
            args.probe_kmeans == "true",)
        result["kmeans"] = [probe_kmeans(spec, kmeans_init=flag) for flag in which]
    text = json.dumps(result, indent=2, default=str)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
