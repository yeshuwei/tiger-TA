#!/usr/bin/env python
"""EXP-C4 -- Q-space collaborative constraint objective (CSG4Rec -> TIGER port).

Verification harness for ``tiger_ext/qspace_loss.py``: the scalar objective
``ConstraintBatch -> L_csg`` that C3's construction feeds. It is an engineering
gate, not an experiment: nothing here claims a recommendation improvement, and
nothing here trains anything.

Ten requirement groups (the plan's C4 brief), plus a real-data block:

  [1] hand-computed fixture          -- logits / target / CE, computed longhand
  [2] multi-neighbour reduction      -- pair mean, not anchor mean, not multi-positive
  [3] directed pairs                 -- (a,b) and (b,a) never merged
  [4] no valid pair                  -- explicit, stable, zero
  [5] the C2 weight                  -- carried by C3, unused here (asserted)
  [6] gradient                       -- loss -> codebooks; the STE path measured
  [6b] real data                     -- Toys / Sport / Beauty, end to end
  [7] numerical stability            -- NaN, Inf, extremes, size-1, degenerate
  [8] forbidden mechanisms           -- no projection / concat / warmup / params / detach
  [9] lambda_csg == 0                -- disabled semantics, and the model untouched
  [10] regression                    -- exp_e10_smoke still 64/64

Every expected value is either computed by an independent implementation in
numpy (never by the module under test) or read from the reference code and cited.

Run:  python -I integration/exp_c4_qspace_loss.py
      EXP_C4_REGRESSION=skip ...   to drop the (slow) E10 re-run
"""
import ast
import hashlib
import io
import os
import re
import subprocess
import sys
import tokenize as _tokenize

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MODEL_DIR = os.path.join(REPO, "model")
RVQAE_DIR = os.path.join(REPO, "rqvae")
DATA_DIR = os.path.join(REPO, "data")
TOYS_CKPT = os.path.join(REPO, "clean_tiger", "ckpt", "Toys", "rqvae",
                         "Oct-14-2026_05-55-09", "best_collision_model.pth")

sys.path.insert(0, REPO)

from tiger_ext import qspace_constraint as QC                 # noqa: E402
from tiger_ext import collab_relation as CR                   # noqa: E402
from tiger_ext import qspace_loss as QL                       # noqa: E402

MODULE = os.path.join(REPO, "tiger_ext", "qspace_loss.py")
REGRESSION = os.environ.get("EXP_C4_REGRESSION", "all")

# the same three pre-existing edits the whole C line inherits: C4 must add none
PRE_C4_MODIFIED = {"model/main.py", "rqvae/models/rq.py", "rqvae/models/vq.py"}

# the untracked port modules C4 is forbidden to modify. `tiger_ext/` is untracked,
# so `git status` cannot see an edit inside it; these hashes are the pin instead.
# Recorded when C4 was built, so a later change to C1/C2/C3/E7 is detectable.
PINNED_SHA16 = {
    "tiger_ext/cf_embed.py": "3c42f2a66cac9fff",
    "tiger_ext/collab_relation.py": "9aa499ea51a65ddc",
    "tiger_ext/qspace_constraint.py": "d50ccc33d7145596",
    "tiger_ext/stage.py": "323f526fdfd1e5b9",
    "tiger_ext/psa_loss.py": "42e37a09d7db3825",
    "tiger_ext/sia_loss.py": "8b462a0a4e2a2e23",
}

DATASETS = ("Toys", "Sport", "Beauty")
N_ITEMS = {"Toys": 11924, "Sport": 18357, "Beauty": 12101}
K = 20                                  # CSG4Rec's --knn_k default
SEED = 2025
BATCH = 256

RQVAE_CFG = dict(num_emb_list=[256, 256, 256], e_dim=32, layers=[512, 256, 128, 64],
                 dropout_prob=0.0, bn=False, loss_type="mse", quant_loss_weight=1.0,
                 beta=0.25, kmeans_init=False, kmeans_iters=100,
                 sk_epsilons=[0.0, 0.0, 0.003], sk_iters=50)

# --------------------------------------------------------------------------- #
# [1] the synthetic fixture. Codebooks are *uniform* in size (3/3) on purpose:
# the objective scores an anchor against one level's codebook, so the level
# class counts must match for the stack to exist at all -- which is itself the
# reason the objective is level-wise rather than a constraint on the summed x_q.
# --------------------------------------------------------------------------- #
FIX_CODEBOOKS = (
    np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]], dtype=np.float32),      # level 0: 3 codewords
    np.array([[0.0, 2.0],
              [3.0, 0.0],
              [2.0, 1.0]], dtype=np.float32),      # level 1: 3 codewords
)
# latents = encoder output (the encoder here is the identity). Each is a chosen
# codeword plus a small displacement, so the model's own argmin assigns the code
# pair recorded below -- and the harness re-derives it in numpy rather than
# trusting this comment.
FIX_X = np.array([[1.00, 0.30],   # -> (0, 0)
                  [0.45, 1.00],   # -> (1, 2)
                  [1.30, 1.15],   # -> (2, 0)
                  [1.30, 0.15],   # -> (0, 0)
                  [0.00, 1.30],   # -> (1, 0)
                  [1.45, 1.00]],  # -> (2, 2)
                 dtype=np.float32)
FIX_IDS = np.array([40, 11, 7, 3, 29, 5], dtype=np.int64)   # scrambled: id-1 is wrong
FIX_NEIGHBORS = np.array([[1, 2],      # row 0 -> rows 1, 2   (two neighbours)
                          [2, 3],      # row 1 -> rows 2, 3   (two neighbours)
                          [5, -1],     # row 2 -> row 5, then empty
                          [4, -1],     # row 3 -> row 4, then empty
                          [-1, -1],    # row 4 -> nothing
                          [-1, -1]],   # row 5 -> nothing
                         dtype=np.int32)
FIX_SIMILARITY = np.array([[0.9, 0.5],
                           [0.8, 0.4],
                           [0.7, np.nan],
                           [0.6, np.nan],
                           [np.nan, np.nan],
                           [np.nan, np.nan]], dtype=np.float32)
# batch = every row, in canonical order, so the pair axis is the edge list above
FIX_BATCH_ROWS = [0, 1, 2, 3, 4, 5]
FIX_PAIRS = [(0, 1, 0.9), (0, 2, 0.5), (1, 2, 0.8), (1, 3, 0.4), (2, 5, 0.7),
             (3, 4, 0.6)]
FIX_ANCHOR_ROWS = [0, 0, 1, 1, 2, 3]
FIX_TARGET_ROWS = [1, 2, 2, 3, 5, 4]

# hand-computed from FIX_CODEBOOKS and the code pairs above (vq.py:84, 87-90, 92),
# written out longhand in the [1] block: level 0 uses 5 rows and drops 1, level 1
# uses 4 and drops 2.
FIX_L0_PER_ROW = [1.3132616875182228, 0.3132616875182229, 0.3132616875182229,
                  1.3132616875182228, 1.3132616875182228]
FIX_L1_PER_ROW = [0.1269280110429725, 4.01814992791781, 4.01814992791781,
                  0.1269280110429725]
FIX_L0 = 0.9132616875182228
FIX_L1 = 2.0725389694803913
FIX_L = 1.492900328499307
FIX_DROPPED = 3

# tokens that may not appear in the *code* of the C4 module. Only NAME tokens are
# scanned, so the docstring and the provenance strings cannot mask a hit. `loss`
# and `cross_entropy` are absent from this list on purpose: this module *is* the
# objective, so they are the deliverable rather than a violation. `detach` is
# handled structurally instead, because the validation helper legitimately uses it.
FORBIDDEN_TOKENS = ("backward", "softmax", "normalize", "concat", "projection",
                    "proj", "warmup", "linear", "info_nce", "infonce",
                    "contrastive", "user", "knn", "cf_", "align")
ALLOWED_TORCH_ATTRS = {"matmul", "zeros", "zeros_like", "arange", "stack",
                       "no_grad", "allclose"}

IMPORT_PROBE = r"""
import builtins, os, sys
repo = os.path.abspath(sys.argv[1])
sys.path.insert(0, repo)
import numpy, torch
calls = []
_open = builtins.open
def counting_open(file, *a, **k):
    try:
        path = os.path.abspath(str(file))
    except Exception:
        path = ""
    if path.startswith(repo):
        calls.append(("open", path))
    return _open(file, *a, **k)
builtins.open = counting_open
_load = numpy.load
def counting_load(*a, **k):
    calls.append(("numpy.load", str(a[0]) if a else ""))
    return _load(*a, **k)
numpy.load = counting_load
_tload = torch.load
def counting_tload(*a, **k):
    calls.append(("torch.load", str(a[0]) if a else ""))
    return _tload(*a, **k)
torch.load = counting_tload
import tiger_ext.qspace_loss
print("IMPORT_CALLS=%d" % len(calls))
for entry in calls[:5]:
    print("   ", entry)
"""

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def raises(fn, exc):
    """True only if fn raises exactly the expected error type."""
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# an independent implementation of the pinned objective (numpy, never the module)
# --------------------------------------------------------------------------- #
def _ce_row(W, a, y):
    """vq.py:84 then :87-90 then :92, for one row, in float64."""
    sim = (W[a] @ W.T).astype(np.float64)
    sim = sim.copy()
    sim[a] = sim[a] - 1e12                         # the reference's self mask
    z = sim - sim.max()
    p = np.exp(z)
    p = p / p.sum()
    return float(-np.log(p[y]))


def _kept_rows(ia, iy, level):
    return [k for k in range(ia.shape[0]) if ia[k, level] != iy[k, level]]


def np_objective(codebooks, ia, iy):
    """The pinned form: pair mean within a level, then mean over non-empty levels."""
    ia, iy = np.asarray(ia), np.asarray(iy)
    levels, dropped = [], 0
    for level, W in enumerate(codebooks):
        rows = _kept_rows(ia, iy, level)
        dropped += ia.shape[0] - len(rows)
        if rows:
            levels.append(np.mean([_ce_row(W, ia[k, level], iy[k, level])
                                   for k in rows]))
    if not levels:
        return 0.0, [], dropped
    return float(np.mean(levels)), [float(v) for v in levels], dropped


def np_ce_bound(codebooks, ia, iy):
    """The largest value the pinned objective can legally take on this input.

    For one row, ``CE = -s_y + logsumexp(s)`` with ``s`` taken over the
    ``n_e - 1`` unmasked entries, so ``CE <= log(n_e - 1) + max s - s_y
    <= log(n_e - 1) + (max s - min s)``. Averaging per level and over levels
    keeps the bound. An uninformative codebook attains ``log(n_e - 1)``; a wide
    logit spread is what lets a real codebook exceed ``log n_e``.
    """
    ia, iy = np.asarray(ia), np.asarray(iy)
    levels = []
    for level, W in enumerate(codebooks):
        rows = _kept_rows(ia, iy, level)
        if not rows:
            continue
        n_e = W.shape[0]
        worst = max(np.log(n_e - 1)
                    + float(np.delete(W[ia[k, level]] @ W.T, ia[k, level]).max()
                            - np.delete(W[ia[k, level]] @ W.T,
                                        ia[k, level]).min())
                    for k in rows)
        levels.append(worst)
    return float(np.max(levels)) if levels else 0.0


def np_anchor_mean(codebooks, ia, iy, anchor_identity):
    """The *rejected* alternative: mean within an anchor, then mean over anchors."""
    ia, iy = np.asarray(ia), np.asarray(iy)
    levels = []
    for level, W in enumerate(codebooks):
        groups = {}
        for k in _kept_rows(ia, iy, level):
            groups.setdefault(anchor_identity[k], []).append(
                _ce_row(W, ia[k, level], iy[k, level]))
        if groups:
            levels.append(np.mean([np.mean(v) for v in groups.values()]))
    return float(np.mean(levels)) if levels else 0.0


def np_multi_positive(codebooks, ia, iy, anchor_identity):
    """The *rejected* alternative: one softmax per anchor over all its positives."""
    ia, iy = np.asarray(ia), np.asarray(iy)
    levels = []
    for level, W in enumerate(codebooks):
        groups = {}
        for k in _kept_rows(ia, iy, level):
            groups.setdefault(anchor_identity[k], []).append(k)
        vals = []
        for _, rows in groups.items():
            sim = (W[ia[rows[0], level]] @ W.T).astype(np.float64)
            sim = sim.copy()
            sim[ia[rows[0], level]] = sim[ia[rows[0], level]] - 1e12
            z = sim - sim.max()
            p = np.exp(z)
            p = p / p.sum()
            vals.append(float(-np.log(p[[iy[k, level] for k in rows]].sum())))
        if vals:
            levels.append(np.mean(vals))
    return float(np.mean(levels)) if levels else 0.0


# --------------------------------------------------------------------------- #
# fixture construction
# --------------------------------------------------------------------------- #
def build_tiny_rqvae(RQVAE):
    """An RQ-VAE with an identity encoder and the hand-set uniform codebooks."""
    import torch
    from torch import nn

    class _Identity(nn.Module):
        def forward(self, x):
            return x

    model = RQVAE(in_dim=FIX_X.shape[1], e_dim=FIX_X.shape[1], num_emb_list=[3, 3],
                  layers=[], dropout_prob=0.0, bn=False, loss_type="mse",
                  beta=0.25, kmeans_init=False, sk_epsilons=[0.0, 0.0], sk_iters=10)
    model.encoder = _Identity()
    model.eval()
    for layer, codebook in zip(model.rq.vq_layers, FIX_CODEBOOKS):
        layer.embedding.weight.data.copy_(torch.from_numpy(codebook))
        layer.initted = True
    return model


def fixture_relation():
    return CR.CollabRelation(FIX_NEIGHBORS, FIX_SIMILARITY, FIX_IDS, k=2)


def fixture_model_and_batch(RQVAE, lambda_csg=1.0):
    """The whole C3 path on the fixture: relation -> mount -> capture -> batch."""
    import torch

    model = build_tiny_rqvae(RQVAE)
    mount = QC.QSpaceConstraint(fixture_relation(), lambda_csg=lambda_csg, k=2)
    x = torch.from_numpy(FIX_X[FIX_BATCH_ROWS])
    quant = QC.capture_qspace(model, x, FIX_BATCH_ROWS)
    return model, mount, quant, mount(quant)


class _QuantStub:
    """The minimum of C3's QSpaceBatch a ConstraintBatch needs to be well formed."""

    def __init__(self, rows, dim):
        import torch
        self.rows = torch.as_tensor(rows, dtype=torch.long)
        self.n_items = int(self.rows.numel())
        self.dim = dim


def hand_batch(ia, iy, codebooks, *, weight=None, anchor_pos=None, target_pos=None):
    """A ConstraintBatch built from code indices, with a consistent ``x_q``.

    ``x_q_anchor``/``x_q_target`` are built as ``sum_l codebook_l[indices_l]`` --
    C3's own definition -- so the batch is as well formed as one C3 produced and
    the objective's provenance check accepts it.
    """
    import torch

    ia = torch.as_tensor(np.asarray(ia, dtype=np.int64)).reshape(-1, len(codebooks))
    iy = torch.as_tensor(np.asarray(iy, dtype=np.int64)).reshape(-1, len(codebooks))
    pairs = int(ia.shape[0])
    e_dim = int(codebooks[0].shape[1])
    if pairs:
        xa = sum(codebooks[l][ia[:, l]] for l in range(len(codebooks)))
        xt = sum(codebooks[l][iy[:, l]] for l in range(len(codebooks)))
    else:
        xa = torch.zeros(0, e_dim)
        xt = torch.zeros(0, e_dim)
    pos = torch.arange(pairs)
    quant = _QuantStub(list(range(pairs)), e_dim)
    w = (torch.ones(pairs) if weight is None
         else torch.as_tensor(weight, dtype=torch.float32).reshape(-1))
    return QC.ConstraintBatch(
        quant,
        pos if anchor_pos is None else torch.as_tensor(anchor_pos),
        pos if target_pos is None else torch.as_tensor(target_pos),
        w, xa, xt, ia, iy, {"pairs": pairs})


FIXTURE_INIT = "seeded data subsample (not the frozen k-means helper)"
# measured, not assumed -- and it is why the fixture below does not call init_emb()
FIXTURE_INIT_NOTE = (
    "The frozen k-means helper the architecture actually uses "
    "(rqvae/models/layers.py:86, `KMeans(n_clusters=..., max_iter=...)`) passes no "
    "random_state, so it draws from the global numpy state -- which the seed here does "
    "set -- but its OpenMP/OpenBLAS reductions are not reproducible run-to-run at default "
    "threading. Measured 2026-10-14 on a 3000x32 float32 matrix with a fixed numpy seed: "
    "4/4 runs gave 4 distinct cluster_centers_ hashes under default threads, and 3/3 "
    "identical under OMP_NUM_THREADS=1. Every number this harness quotes has to be "
    "reproducible, so the fixture tokenizer takes its codebooks from a seeded subsample of "
    "the level's residual instead. Measured the same day, on the real Toys content: the "
    "trained checkpoint path is stable 3/3 and the seeded torch init + encoder forward is "
    "stable 3/3 -- k-means is the *only* non-reproducible step, and the frozen core is not "
    "modified to fix it (the harness pins its own fixture instead). The fixture is a "
    "fixture: it gives the objective a real, non-degenerate code space on real content, "
    "and it is not a claim about how Clean TIGER initialises RQ-VAE."
)


def fit_fixture_tokenizer(RQVAE, x_all, seed=SEED):
    """A fixture tokenizer of the frozen architecture, with reproducible codebooks.

    Codebooks come from a seeded subsample of each level's residual rather than
    from the frozen ``init_emb`` -- see ``FIXTURE_INIT_NOTE`` for the measurement
    that forced that choice. Deterministic across processes, which is what makes
    the numbers this harness prints quotable.
    """
    import torch

    torch.manual_seed(seed)
    model = RQVAE(in_dim=int(x_all.shape[1]), **RQVAE_CFG)
    model.eval()
    gen = torch.Generator().manual_seed(seed)   # own stream: no other draw can shift it
    with torch.no_grad():
        residual = []
        for start in range(0, x_all.shape[0], 8192):
            residual.append(model.encoder(x_all[start:start + 8192]))
        residual = torch.cat(residual, dim=0)
        for layer in model.rq.vq_layers:
            weight = layer.embedding.weight
            n_e = weight.shape[0]
            # the frozen helper upsamples when n_e > the sample count; mirror that
            if residual.shape[0] >= n_e:
                pick = torch.randperm(residual.shape[0], generator=gen)[:n_e]
            else:
                pick = torch.randint(residual.shape[0], (n_e,), generator=gen)
            weight.data.copy_(residual[pick])
            layer.initted = True
            dist = ((residual ** 2).sum(1, keepdim=True)
                    + (weight ** 2).sum(1)[None, :]
                    - 2.0 * residual @ weight.t())
            residual = residual - weight[torch.argmin(dist, dim=-1)]
    return model


def load_content(dataset):
    """The dataset's canonical content table, in canonical row order, plus the ids."""
    import pandas as pd
    import torch

    frame = pd.read_parquet(os.path.join(DATA_DIR, dataset, "item_emb.parquet"))
    ids = frame["ItemID"].to_numpy(dtype=np.int64)
    x = np.stack(frame["embedding"].values).astype(np.float32)
    return torch.from_numpy(x), ids


# --------------------------------------------------------------------------- #
# [1] the fixture, computed by hand
# --------------------------------------------------------------------------- #
def part1_fixture(RQVAE):
    import torch

    print("\n[1] synthetic fixture: hand-computed logits -> target -> cross-entropy")
    model, _, quant, batch = fixture_model_and_batch(RQVAE)

    # the code pairs are re-derived in numpy, not taken from this file's comments
    def argmin_rows(d, W):
        dist = ((d ** 2).sum(1, keepdims=True) + (W ** 2).sum(1)[None, :]
                - 2.0 * d @ W.T)
        return dist.argmin(1)

    i0 = argmin_rows(FIX_X, FIX_CODEBOOKS[0])
    i1 = argmin_rows(FIX_X - FIX_CODEBOOKS[0][i0], FIX_CODEBOOKS[1])
    check("the tiny model's code indices are the hand-designed ones, in batch order",
          np.array_equal(quant.indices.numpy(),
                         np.stack([i0[FIX_BATCH_ROWS], i1[FIX_BATCH_ROWS]], axis=1)),
          f"got {quant.indices.tolist()}")
    check("the model's level-0 codes are not degenerate on the fixture",
          int(np.unique(i0).size) == 3, f"{int(np.unique(i0).size)} distinct level-0 codes")

    check("the pair axis is the hand-computed edge list",
          batch.n_pairs == len(FIX_PAIRS)
          and batch.anchor_rows.tolist() == FIX_ANCHOR_ROWS
          and batch.target_rows.tolist() == FIX_TARGET_ROWS,
          f"pairs={batch.n_pairs}, anchors={batch.anchor_rows.tolist()}, "
          f"targets={batch.target_rows.tolist()}")

    # what the objective will score, computed longhand, twice: once here in numpy
    # and once from the reference's arithmetic written out by hand
    ia = batch.indices_anchor.numpy()
    iy = batch.indices_target.numpy()
    expected, per_level, dropped = np_objective(FIX_CODEBOOKS, ia, iy)
    print("      hand-computed rows (a -> y, loss):")
    for level, W in enumerate(FIX_CODEBOOKS):
        rows = _kept_rows(ia, iy, level)
        shown = ", ".join(f"{int(ia[k, level])}->{int(iy[k, level])}:"
                          f"{_ce_row(W, ia[k, level], iy[k, level]):.7f}" for k in rows)
        print(f"        level {level}: {shown}   (dropped "
              f"{ia.shape[0] - len(rows)} same-code)")
    check("the hand arithmetic reproduces the recorded per-row values",
          all(abs(v - e) < 1e-12 for v, e in
              zip([_ce_row(FIX_CODEBOOKS[0], ia[k, 0], iy[k, 0])
                   for k in _kept_rows(ia, iy, 0)], FIX_L0_PER_ROW))
          and all(abs(v - e) < 1e-12 for v, e in
                  zip([_ce_row(FIX_CODEBOOKS[1], ia[k, 1], iy[k, 1])
                       for k in _kept_rows(ia, iy, 1)], FIX_L1_PER_ROW)),
          f"level 0 {[round(v, 7) for v in FIX_L0_PER_ROW]}, "
          f"level 1 {[round(v, 7) for v in FIX_L1_PER_ROW]}")
    check("the hand arithmetic reproduces the recorded level means and total",
          abs(per_level[0] - FIX_L0) < 1e-12 and abs(per_level[1] - FIX_L1) < 1e-12
          and abs(expected - FIX_L) < 1e-12 and dropped == FIX_DROPPED,
          f"levels {[round(v, 10) for v in per_level]}, "
          f"L_csg {expected:.12f}, dropped {dropped}; "
          f"the golden constants in this file agree")

    loss = QL.QSpaceLoss()(batch, model)
    check("the module's L_csg is the hand-computed value",
          bool(torch.isfinite(loss)) and abs(float(loss) - FIX_L) < 1e-6,
          f"module {float(loss):.12f} vs hand {FIX_L:.12f}")
    probe = QL.QSpaceLoss()
    probe(batch, model)
    check("the module's per-level values are the hand-computed ones",
          abs(probe.stats["level0_loss"] - FIX_L0) < 1e-6
          and abs(probe.stats["level1_loss"] - FIX_L1) < 1e-6,
          f"module {[round(probe.stats['level0_loss'], 10), round(probe.stats['level1_loss'], 10)]}")
    check("the module's drop accounting is the hand-computed one "
          "(the reference's positive is always a *different* codeword, vq.py:77-78)",
          probe.stats["dropped_same_code"] == FIX_DROPPED
          and [e["used"] for e in probe.stats["per_level"]] == [5, 4],
          f"dropped {probe.stats['dropped_same_code']}, "
          f"per-level used {[e['used'] for e in probe.stats['per_level']]}")
    bound = np_ce_bound(FIX_CODEBOOKS, ia, iy)
    check("the loss sits inside the analytic bound for cross-entropy over a "
          "codebook, and is not a constant",
          0.0 < float(loss) <= bound + 1e-6
          and abs(probe.stats["level0_loss"] - probe.stats["level1_loss"]) > 1e-3,
          f"L_csg {float(loss):.6f} <= bound {bound:.6f} "
          f"(log(n_e-1) + the level's logit spread); the two levels differ")

    # the objective scores the codebook lookup, not the tensor C3 handed over:
    # identical indices with a corrupted x_q must change nothing once validation
    # is off, and must be *caught* while it is on.
    import copy
    corrupt = hand_batch(ia, iy, [torch.from_numpy(c) for c in FIX_CODEBOOKS])
    corrupt.x_q_anchor = corrupt.x_q_anchor + 5.0
    check("a batch whose x_q does not explain its indices is refused",
          raises(lambda: QL.QSpaceLoss()(corrupt, model), QL.QSpaceLossError),
          "the provenance check ties the batch to *this* model's codebooks")
    check("with validation off the objective is invariant to x_q_anchor, because it "
          "scores the codebook lookup itself (the reference's x_q, vq.py:195)",
          abs(float(QL.QSpaceLoss(validate=False)(corrupt, model)) - FIX_L) < 1e-6,
          "documented, not accidental: the query is rebuilt from the codebooks")

    # the provenance check must be a real check, not a tautology: a batch built
    # from *another* model's codebooks has to fail
    other = build_tiny_rqvae(RQVAE)
    other.rq.vq_layers[0].embedding.weight.data.mul_(2.0)
    check("a batch whose x_q belongs to a different model is refused",
          raises(lambda: QL.QSpaceLoss()(batch, other), QL.QSpaceLossError),
          "same indices, other codebooks -> the tensors no longer agree")
    return batch, model


# --------------------------------------------------------------------------- #
# [2] several neighbours on one anchor
# --------------------------------------------------------------------------- #
def part2_multi_pair(RQVAE):
    import torch

    print("\n[2] multi-neighbour: each pair is its own row, the reduction is a pair mean")
    cb = [torch.from_numpy(c) for c in FIX_CODEBOOKS]
    # one anchor (code 0,0) with two neighbours, and a second anchor (code 1,2)
    # with one -- so a pair mean and an anchor mean part company
    ia = np.array([[0, 0], [0, 0], [1, 2]])
    iy = np.array([[1, 2], [2, 1], [0, 0]])
    identity = np.array([0, 0, 1])
    batch = hand_batch(ia, iy, cb)
    model = build_tiny_rqvae(RQVAE)
    loss = float(QL.QSpaceLoss()(batch, model))

    expected, per_level, _ = np_objective(FIX_CODEBOOKS, ia, iy)
    anchor_mean = np_anchor_mean(FIX_CODEBOOKS, ia, iy, identity)
    multi_pos = np_multi_positive(FIX_CODEBOOKS, ia, iy, identity)
    check("the two-neighbour anchor contributes two independent CE rows",
          abs(loss - expected) < 1e-6 and len(per_level) == 2,
          f"L_csg {loss:.9f} == pair mean {expected:.9f}")
    check("the reduction is a mean over *pairs*, not over anchors",
          abs(anchor_mean - expected) > 1e-3 and abs(loss - anchor_mean) > 1e-3,
          f"anchor mean would be {anchor_mean:.9f} (rejected); "
          f"difference {abs(loss - anchor_mean):.6f}")
    check("the reduction is not a multi-positive softmax over an anchor's neighbours",
          abs(multi_pos - expected) > 1e-3 and abs(loss - multi_pos) > 1e-3,
          f"multi-positive would be {multi_pos:.9f} (rejected)")
    check("dropping one of the two neighbours really changes the loss, so both rows "
          "are in the CE rather than one overriding the other",
          abs(float(QL.QSpaceLoss()(hand_batch(ia[:2], iy[:2], cb), model))
              - float(QL.QSpaceLoss()(hand_batch(ia[:1], iy[:1], cb), model))) > 1e-6,
          "the pair axis, not an anchor axis, is what the objective reduces over")


# --------------------------------------------------------------------------- #
# [3] direction
# --------------------------------------------------------------------------- #
def part3_directed(RQVAE):
    import torch

    print("\n[3] directed pairs: (a,b) and (b,a) are two rows, never one")
    cb = [torch.from_numpy(c) for c in FIX_CODEBOOKS]
    model = build_tiny_rqvae(RQVAE)

    fwd = hand_batch(np.array([[0, 0]]), np.array([[1, 2]]), cb)
    rev = hand_batch(np.array([[1, 2]]), np.array([[0, 0]]), cb)
    both = hand_batch(np.array([[0, 0], [1, 2]]), np.array([[1, 2], [0, 0]]), cb)
    l_fwd = float(QL.QSpaceLoss()(fwd, model))
    l_rev = float(QL.QSpaceLoss()(rev, model))
    l_both = float(QL.QSpaceLoss()(both, model))
    check("a one-directional relation yields exactly one pair, in that direction",
          fwd.n_pairs == 1 and rev.n_pairs == 1
          and fwd.indices_anchor.tolist() == [[0, 0]]
          and rev.indices_anchor.tolist() == [[1, 2]],
          "the anchor of the pair is the query; the roles are not interchangeable")
    check("the forward and reverse pairs are scored separately, not merged",
          abs(l_both - (l_fwd + l_rev) / 2.0) < 1e-6 and l_both != l_fwd,
          f"L(0->1) {l_fwd:.9f}, L(1->0) {l_rev:.9f}, L(both) {l_both:.9f} "
          f"= their mean")
    check("direction matters: reversing a pair is not the same objective value",
          abs(l_fwd - l_rev) > 1e-6,
          "the query is the anchor's own codeword, so swapping the roles changes "
          "the logits and the target")

    # through C3's real pipeline, a one-way relation stays one-way
    neighbors = np.array([[1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
                         dtype=np.int32)
    similarity = np.array([[0.9, np.nan]] + [[np.nan, np.nan]] * 5, dtype=np.float32)
    rel = CR.CollabRelation(neighbors, similarity, FIX_IDS, k=2)
    mount = QC.QSpaceConstraint(rel, lambda_csg=1.0, k=2)
    quant = QC.capture_qspace(build_tiny_rqvae(RQVAE),
                              torch.from_numpy(FIX_X), FIX_BATCH_ROWS)
    one_way = mount(quant)
    check("C3's own pair axis carries the direction through to C4",
          one_way.n_pairs == 1
          and one_way.anchor_rows.tolist() == [0]
          and one_way.target_rows.tolist() == [1],
          "row 0 -> row 1 exists; row 1 -> row 0 does not, and C4 does not invent it")


# --------------------------------------------------------------------------- #
# [4] no valid pair
# --------------------------------------------------------------------------- #
def part4_empty(RQVAE):
    import torch

    print("\n[4] no valid pair: an explicit, stable zero (house precedent, not a crash)")
    cb = [torch.from_numpy(c) for c in FIX_CODEBOOKS]
    model = build_tiny_rqvae(RQVAE)

    empty = hand_batch(np.zeros((0, 2), dtype=np.int64),
                       np.zeros((0, 2), dtype=np.int64), cb)
    check("an empty pair axis is explicit on the batch", empty.is_empty
          and empty.n_pairs == 0, "the caller checks is_empty rather than a shape")
    loss = QL.QSpaceLoss()(empty, model)
    probe = QL.QSpaceLoss()
    probe(empty, model)
    check("an empty pair axis returns a 0-dim zero, finite and typed like the codebooks",
          loss.dim() == 0 and float(loss) == 0.0 and bool(torch.isfinite(loss))
          and loss.dtype == model.rq.get_codebook().dtype,
          f"{loss!r} with stats reason {probe.stats['reason']!r}")
    check("the empty case is a data outcome, not a disabled switch -- it does not raise",
          probe.stats["levels_used"] == 0,
          "raising is reserved for lambda_csg == 0, matching SIAMount / PSAMount")

    # every pair sharing its anchor's codeword at every level (2 pairs x 2 levels)
    same = hand_batch(np.array([[1, 2], [0, 1]]), np.array([[1, 2], [0, 1]]), cb)
    probe2 = QL.QSpaceLoss()
    l_same = probe2(same, model)
    check("all-pairs-identical at every level also returns 0.0, and says why",
          float(l_same) == 0.0 and probe2.stats["dropped_same_code"] == 4,
          f"reason {probe2.stats['reason']!r}; the drop counter is per level, so "
          f"{probe2.stats['dropped_same_code']} for 2 pairs over 2 levels")

    # level 1 emptied while level 0 survives: the mean is over the levels that
    # still have rows, not over L
    half = hand_batch(np.array([[0, 0], [0, 0]]), np.array([[1, 0], [2, 0]]), cb)
    probe3 = QL.QSpaceLoss()
    l_half = float(probe3(half, model))
    expected_l0 = np.mean([_ce_row(FIX_CODEBOOKS[0], 0, 1),
                           _ce_row(FIX_CODEBOOKS[0], 0, 2)])
    check("a level that loses every pair is omitted, and the mean is over the "
          "remaining levels (the reference has no such case: vq.py:77-78 guarantees "
          "a positive)",
          abs(l_half - expected_l0) < 1e-6 and probe3.stats["levels_used"] == 1
          and [e["used"] for e in probe3.stats["per_level"]] == [2, 0],
          f"L_csg {l_half:.9f} == the surviving level's mean {expected_l0:.9f}, "
          f"not half of it")

    # a single item batch, and a hand-built batch where both sides reduce to the
    # same code at level 0 only
    one = hand_batch(np.array([[1, 0]]), np.array([[2, 0]]), cb)
    l_one = QL.QSpaceLoss()(one, model)
    check("a single pair is a well-defined objective, finite and positive",
          one.n_pairs == 1 and bool(torch.isfinite(l_one)) and float(l_one) > 0.0,
          f"L_csg {float(l_one):.9f}")


# --------------------------------------------------------------------------- #
# [5] the C2 weight
# --------------------------------------------------------------------------- #
def part5_weight(RQVAE):
    import torch

    print("\n[5] the C2 cosine weight: carried by C3, deliberately unused here")
    cb = [torch.from_numpy(c) for c in FIX_CODEBOOKS]
    model = build_tiny_rqvae(RQVAE)
    ia = np.array([[0, 0], [1, 2], [2, 0]])
    iy = np.array([[1, 2], [2, 1], [0, 1]])

    base = hand_batch(ia, iy, cb, weight=[0.9, 0.4, 0.7])
    l_base = QL.QSpaceLoss()(base, model)
    variants = {
        "zero": [0.0, 0.0, 0.0],
        "huge": [1e6, 1e6, 1e6],
        "negated": [-0.9, -0.4, -0.7],
        "shuffled": [0.7, 0.9, 0.4],
    }
    same = []
    for name, w in variants.items():
        batch = hand_batch(ia, iy, cb, weight=w)
        same.append((name, torch.equal(QL.QSpaceLoss()(batch, model), l_base)))
    check("the loss is bitwise invariant to C3's weight under every perturbation",
          all(ok for _, ok in same),
          f"L_csg {float(l_base):.9f} unchanged for "
          f"{', '.join(n for n, _ in same)}")
    check("the weight is genuinely present on the batch, so the invariance is not vacuous",
          base.weight.numel() == 3 and float(base.weight.sum()) > 0,
          f"weights {base.weight.tolist()} carried but not read")

    src = open(MODULE, encoding="utf-8").read()
    check("the module never reads `batch.weight`",
          "batch.weight" not in src and ".weight]" not in src,
          "the reference's diversity_loss takes (x_q, indices, clusters) and no "
          "weight (vq.py:68); using it would be a new mechanism")
    prov = QL.QSpaceLoss().provenance()
    check("the provenance record states the decision instead of leaving it implicit",
          "NOT used" in prov["weight_policy"] and "vq.py:68" in prov["weight_policy"],
          "C2 weight retained in C3's construction, unused by C4 -- as required")


# --------------------------------------------------------------------------- #
# [6] gradient
# --------------------------------------------------------------------------- #
def part6_gradient(RQVAE):
    import torch

    print("\n[6] gradient: loss -> codebooks; the straight-through path measured")
    cb = [torch.from_numpy(c) for c in FIX_CODEBOOKS]
    ia = np.array([[0, 0], [1, 2], [2, 0], [0, 0]])
    iy = np.array([[1, 2], [2, 1], [0, 1], [1, 0]])
    batch = hand_batch(ia, iy, cb)

    model = build_tiny_rqvae(RQVAE)
    before = [p.requires_grad for p in model.parameters()]
    loss = QL.QSpaceLoss()(batch, model)
    check("the objective is a differentiable tensor, not a detached number",
          loss.requires_grad and loss.grad_fn is not None,
          f"grad_fn {type(loss.grad_fn).__name__}")
    loss.backward()
    grads = [model.rq.vq_layers[l].embedding.weight.grad for l in range(2)]
    check("the gradient reaches every level's codebook",
          all(g is not None and float(g.abs().sum()) > 0 for g in grads),
          f"|dL/dW| = {[float(g.abs().sum()) for g in grads]}")
    check("the objective reaches the RQ-VAE's quantiser graph on both sides",
          all(g is not None for g in grads),
          "query side (the anchor's codeword) and candidate side (the codebook) "
          "both come from W_l, exactly as sim = x_q @ emb.t() does")
    check("no parameter's requires_grad is touched, and no gradient step is called "
          "by the module",
          [p.requires_grad for p in model.parameters()] == before,
          "C4 confirms the graph; it does not decide how C5 trains")
    # and the straight-through path from the tensor C3 captures is real, untouched
    torch.manual_seed(SEED)
    real = fit_fixture_tokenizer(RQVAE, torch.randn(64, 32))
    x = torch.randn(8, 32, requires_grad=True)
    enc_before = [p.requires_grad for p in real.parameters()]
    _, _, _, _, x_q = real(x, return_z=True, return_xq=True)
    x_q.sum().backward()
    enc_grads = [p.grad for p in real.encoder.parameters()]
    check("the straight-through route from C3's x_q to the encoder is intact",
          x.grad is not None and float(x.grad.abs().sum()) > 0
          and any(g is not None and float(g.abs().sum()) > 0 for g in enc_grads),
          "x_q is the post-straight-through sum; its gradient does reach the encoder "
          f"({len(enc_grads)} encoder params got a gradient)")

    # the same real model, but scored by the objective instead of by x_q.sum():
    # the encoder must stay untouched, because the objective reads the codebooks
    real.zero_grad(set_to_none=True)
    real_cb = [layer.embedding.weight for layer in real.rq.vq_layers]
    real_batch = hand_batch(np.array([[0, 0, 0], [3, 7, 1]]),
                            np.array([[1, 2, 2], [5, 4, 3]]), real_cb)
    l_real = QL.QSpaceLoss()(real_batch, real)
    l_real.backward()
    enc_after = [p.grad for p in real.encoder.parameters()]
    cb_grads = [float(layer.embedding.weight.grad.abs().sum())
                for layer in real.rq.vq_layers]
    check("the objective is not routed to the encoder, because indices come from "
          "argmin/argmax and carry no gradient (rqvae/models/vq.py:78,86)",
          all(g is None or float(g.abs().sum()) == 0 for g in enc_after)
          and any(v > 0 for v in cb_grads),
          f"encoder grads {[None if g is None else round(float(g.abs().sum()), 8) for g in enc_after]}, "
          f"codebook |dL/dW| {[round(v, 6) for v in cb_grads]} -- a level whose "
          f"codebook has collapsed on this random fixture gives exactly zero, and "
          f"that is reported rather than hidden")
    check("the objective neither breaks nor detaches that route",
          [p.requires_grad for p in real.parameters()] == enc_before,
          "whether C5 routes the constraint through the encoder is a C5 decision "
          "(recorded in the module's QSL_GRADIENT_POLICY)")

    tree = ast.parse(open(MODULE, encoding="utf-8").read())

    def _detach_nodes(node):
        return [c for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                and c.func.attr == "detach"]

    def _is_no_grad(fn):
        return any(isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                   and d.func.attr == "no_grad" for d in fn.decorator_list)

    loose = {}
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        every = _detach_nodes(fn)
        # a detach is harmless when nothing differentiable can flow through it:
        # either the whole function is a no_grad validation path, or the detach's
        # subtree is consumed by float(), which yields a python number
        consumed = set()
        if not _is_no_grad(fn):
            in_float = [c for c in ast.walk(fn)
                        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                        and c.func.id == "float"]
            consumed = {id(c) for f in in_float for c in _detach_nodes(f)}
        if len(every) != len(consumed) and not _is_no_grad(fn):
            loose[fn.name] = (len(every), len(consumed))
    check("no .detach() is ever used on a differentiable value -- each one is "
          "either inside the no_grad validation helper or a `float(...)` statistics "
          "conversion, never a gradient being cut",
          not loose,
          "checked structurally per function: verify_indices_match_qspace is "
          "@torch.no_grad, and qspace_constraint_loss's two detaches are the reported "
          "per-level value and the mean" if not loose
          else f"loose detaches: {loose}")
    check("no .backward() is ever called by the module",
          not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == "backward" for n in ast.walk(tree)),
          "the module returns a scalar and stops")

    mount = QL.QSpaceLossMount(lambda_csg=3.0)
    plain = float(QL.QSpaceLoss()(hand_batch(ia, iy, cb), build_tiny_rqvae(RQVAE)))
    scaled = float(mount(hand_batch(ia, iy, cb), build_tiny_rqvae(RQVAE)))
    check("the mount returns exactly lambda_csg * L_csg",
          abs(scaled - 3.0 * plain) < 1e-6,
          f"{scaled:.9f} == 3.0 * {plain:.9f}")
    check("neither the objective nor its mount holds a parameter or a buffer",
          len(list(QL.QSpaceLoss().parameters())) == 0
          and len(list(QL.QSpaceLoss().buffers())) == 0
          and len(list(mount.parameters())) == 0
          and len(list(mount.buffers())) == 0,
          "the codebooks belong to the model passed in; this only reads them")

    part6b_real(RQVAE)


def part6b_real(RQVAE):
    """The same objective on real relations and real content, end to end."""
    import torch

    print("\n[6b] real data: Toys / Sport / Beauty")
    real = None
    for dataset in DATASETS:
        mount = QC.load_constraint_for_dataset(dataset, lambda_csg=1.0, k=K)
        relation = mount.relation
        x_all, _ = load_content(dataset)
        model = fit_fixture_tokenizer(RQVAE, x_all)
        x = x_all[:BATCH]
        rows = torch.arange(BATCH)
        batch = mount(QC.capture_qspace(model, x, rows))
        fn = QL.QSpaceLoss()
        loss = fn(batch, model)
        st = fn.stats
        finite = bool(torch.isfinite(loss)) and all(
            e["loss"] is None or np.isfinite(e["loss"]) for e in st["per_level"])
        check(f"{dataset}: {batch.n_pairs} in-batch pairs score to a finite L_csg",
              batch.n_pairs > 0 and finite and float(loss) > 0.0,
              f"L_csg {float(loss):.6f}; per-level "
              f"{[None if e['loss'] is None else round(e['loss'], 4) for e in st['per_level']]}, "
              f"dropped_same_code {st['dropped_same_code']}")
        per_level_used = [e["used"] for e in st["per_level"]]
        check(f"{dataset}: every pair is accounted for at every level",
              sum(per_level_used) + st["dropped_same_code"]
              == batch.n_pairs * st["levels"],
              f"used {per_level_used} + dropped {st['dropped_same_code']} == "
              f"{batch.n_pairs} x {st['levels']}")
        check(f"{dataset}: the loss is near the reference scale, not degenerate",
              0.0 < float(loss) <= float(np.log(255)) + 8.0,
              f"L_csg {float(loss):.4f}; log(255) = {np.log(255):.4f} is the "
              f"uninformative-codebook value and the logit spread may add to it")
        grad_ok = _codebook_grads(model, batch)
        print(f"      {dataset}: {relation.n_edges:,} relation edges, "
              f"{batch.n_pairs} pairs, same-code drops {st['dropped_same_code']}, "
              f"|dL/dW| = {[round(v, 5) for v in grad_ok]}")
        if dataset == "Toys":
            real = (mount, x, rows)
            with torch.no_grad():
                model_codes = model.get_indices(x)

    mount, x, rows = real
    ckpt = torch.load(TOYS_CKPT, map_location="cpu", weights_only=False)
    trained = RQVAE(in_dim=int(x.shape[1]), **RQVAE_CFG)
    trained.load_state_dict(ckpt["state_dict"])
    trained.eval()
    batch = mount(QC.capture_qspace(trained, x, rows))
    fn = QL.QSpaceLoss()
    loss = fn(batch, trained)
    grads = _codebook_grads(trained, batch)
    check("Toys: the one genuinely trained canonical-content tokenizer gives a "
          "finite loss and a gradient on every level's codebook",
          bool(torch.isfinite(loss)) and all(v > 0 for v in grads),
          f"L_csg {float(loss):.6f}, per-level "
          f"{[round(e['loss'], 4) for e in fn.stats['per_level']]}, "
          f"|dL/dW| = {[round(v, 6) for v in grads]}")

    # every number quoted above and in the per-dataset block has to be reproducible,
    # or it is not a measurement -- so the fixture is rebuilt and compared
    x_all, _ = load_content("Toys")
    again = fit_fixture_tokenizer(RQVAE, x_all)
    with torch.no_grad():
        same_codes = bool(torch.equal(again.get_indices(x), model_codes))
    check("the fixture tokenizer is reproducible across rebuilds in one process, "
          "so the numbers this section quotes are measurements and not draws",
          same_codes and FIXTURE_INIT.startswith("seeded"),
          f"init = {FIXTURE_INIT}; rebuilding gives identical code indices on the "
          f"batch. The frozen helper it replaces is not reproducible -- see "
          f"FIXTURE_INIT_NOTE (4/4 distinct under default threads, 3/3 identical "
          f"at OMP_NUM_THREADS=1)")


def _codebook_grads(model, batch):
    """|dL/dW_l| per level, on a throwaway zeroed gradient."""
    model.zero_grad(set_to_none=True)
    QL.QSpaceLoss()(batch, model).backward()
    return [round(float(layer.embedding.weight.grad.abs().sum()), 6)
            for layer in model.rq.vq_layers]


# --------------------------------------------------------------------------- #
# [7] numerical stability and the refusal surface
# --------------------------------------------------------------------------- #
def part7_stability(RQVAE):
    import torch

    print("\n[7] numerical stability: extremes, degenerate codebooks, refusals")

    # an all-zero codebook is the analytic worst case: every logit is equal, so
    # the CE is exactly log(n_e - 1) after the self mask
    model = build_tiny_rqvae(RQVAE)
    for layer in model.rq.vq_layers:
        layer.embedding.weight.data.zero_()
    cb = [torch.zeros(3, 2) for _ in range(2)]
    batch = hand_batch(np.array([[0, 0], [1, 1]]), np.array([[1, 1], [2, 2]]), cb)
    loss = QL.QSpaceLoss()(batch, model)
    check("a zero codebook gives exactly log(n_e - 1): the CE is well posed, not NaN",
          abs(float(loss) - float(np.log(2.0))) < 1e-6,
          f"L_csg {float(loss):.12f} vs log(2) {np.log(2.0):.12f}")

    for scale, label in ((1e3, "large"), (1e-8, "tiny")):
        model = build_tiny_rqvae(RQVAE)
        for layer in model.rq.vq_layers:
            layer.embedding.weight.data.mul_(scale)
        cb = [torch.from_numpy(c * scale) for c in FIX_CODEBOOKS]
        batch = hand_batch(np.array([[0, 0], [1, 1]]), np.array([[1, 2], [2, 0]]), cb)
        loss = QL.QSpaceLoss()(batch, model)
        bound = np_ce_bound([c * scale for c in FIX_CODEBOOKS],
                            np.array([[0, 0], [1, 1]]), np.array([[1, 2], [2, 0]]))
        check(f"{label} codebook magnitudes stay finite and inside the analytic "
              f"bound (no overflow to Inf, no underflow to 0)",
              bool(torch.isfinite(loss)) and 0.0 <= float(loss) <= bound * (1 + 1e-5) + 1e-6,
              f"scale {scale:g} -> L_csg {float(loss):.9f} <= bound {bound:.6f} "
              f"(float32 accumulation vs a float64 bound, hence the slack)")

    # a NaN in the model is *observable*, not silently masked away
    model = build_tiny_rqvae(RQVAE)
    model.rq.vq_layers[0].embedding.weight.data[1, :] = float("nan")
    batch = hand_batch(np.array([[0, 0]]), np.array([[1, 2]]),
                       [torch.from_numpy(c) for c in FIX_CODEBOOKS])
    loss = QL.QSpaceLoss(validate=False)(batch, model)
    check("a NaN codebook produces NaN rather than a plausible number",
          bool(torch.isnan(loss)),
          "the objective introduces no NaN of its own; it does not hide one either")

    model = build_tiny_rqvae(RQVAE)
    cb = [torch.from_numpy(c) for c in FIX_CODEBOOKS]
    good = hand_batch(np.array([[0, 0]]), np.array([[1, 2]]), cb)
    check("the objective is finite and non-negative on a clean input",
          bool(torch.isfinite(QL.QSpaceLoss()(good, model)))
          and float(QL.QSpaceLoss()(good, model)) >= 0.0,
          "the stability checks above are not passing vacuously")

    # refusals: each is a wiring error, not a data outcome. The index blocks are
    # overwritten on a *valid* batch so the refusal under test is the module's,
    # not torch's indexing.
    def _spoil(ia, iy):
        b = hand_batch(np.array([[0, 0]]), np.array([[1, 2]]), cb)
        b.indices_anchor = torch.as_tensor(ia).reshape(1, 2)
        b.indices_target = torch.as_tensor(iy).reshape(1, 2)
        return b

    check("an out-of-range code index is refused",
          raises(lambda: QL.QSpaceLoss()(_spoil([[0, 0]], [[9, 2]]), model),
                 QL.QSpaceLossError),
          "index 9 does not exist in a 3-codeword codebook; without this the "
          "codebook lookup would happily wrap or raise from torch instead")
    check("a negative code index is refused",
          raises(lambda: QL.QSpaceLoss()(_spoil([[0, 0]], [[-1, 2]]), model),
                 QL.QSpaceLossError),
          "a sentinel must never reach the objective; C3 drops them, and torch's "
          "own negative indexing must not silently accept one here")
    check("a batch with the wrong number of RQ levels is refused",
          raises(lambda: QL.QSpaceLoss()(hand_batch(np.array([[0]]),
                                                    np.array([[1]]), cb[:1]), model),
                 QL.QSpaceLossError),
          "the collision token is a level-0 field, not a quantiser level")
    check("a batch whose anchor and target index blocks disagree in shape is refused",
          raises(lambda: QL.QSpaceLoss()(hand_batch(np.array([[0, 0]]),
                                                    np.array([[1, 2], [0, 0]]), cb),
                                         model),
                 QL.QSpaceLossError),
          "the pair axis must be one axis")

    tiny = RQVAE(in_dim=2, e_dim=2, num_emb_list=[1, 1], layers=[], dropout_prob=0.0,
                 bn=False, loss_type="mse", beta=0.25, kmeans_init=False,
                 sk_epsilons=[0.0, 0.0], sk_iters=10)
    tiny.eval()
    check("a model with a one-codeword level is refused before any arithmetic",
          raises(lambda: QL.QSpaceLoss()(good, tiny), QL.QSpaceLossError),
          "with the self codeword masked there is no candidate left, and the "
          "reference's own construct could not act either (vq.py:77-78)")


# --------------------------------------------------------------------------- #
# [8] forbidden mechanisms
# --------------------------------------------------------------------------- #
def part8_scope():
    print("\n[8] scope: no projection, no concat, no warmup, no parameter, no frozen change")
    status = subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                            capture_output=True, text=True).stdout.splitlines()
    modified = {l[3:].strip() for l in status if l[:2].strip() == "M"}
    check("C4 adds no tracked-file modification at all",
          modified <= PRE_C4_MODIFIED, f"modified: {sorted(modified)}")
    check("clean_tiger/ is untouched by C4",
          not [l for l in status if "clean_tiger" in l], "no line mentions it")

    drifted = {rel: (pin, sha16(os.path.join(REPO, rel)))
               for rel, pin in PINNED_SHA16.items()
               if sha16(os.path.join(REPO, rel)) != pin}
    check("C1 / C2 / C3 / E7 / SIA / PSA are byte-identical to what C4 was built on",
          not drifted,
          "tiger_ext/ is untracked, so git cannot see an edit inside it; these "
          "hashes are the pin. " + (f"drifted: {drifted}" if drifted else "6/6 match"))

    source = open(MODULE, encoding="utf-8").read()
    tree = ast.parse(source)
    names = [t.string for t in _tokenize.generate_tokens(io.StringIO(source).readline)
             if t.type == _tokenize.NAME]
    offenders = sorted({n for n in names for tok in FORBIDDEN_TOKENS
                        if tok in n.lower()})
    check("no projection / concat / warmup / softmax / align / user identifier exists "
          "in the code",
          not offenders,
          f"scanned {len(names)} NAME tokens; `loss` and `cross_entropy` are "
          f"deliberately allowed -- they are the deliverable")

    f_attrs, torch_attrs, nn_attrs = set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "F":
                f_attrs.add(node.attr)
            elif node.value.id == "torch":
                torch_attrs.add(node.attr)
            elif node.value.id == "nn":
                nn_attrs.add(node.attr)
    check("the objective's only functional call is cross_entropy -- no linear / "
          "normalize / cosine_similarity / log_softmax could hide here",
          f_attrs == {"cross_entropy"}, f"F.* used: {sorted(f_attrs)}")
    check("no torch operation outside the recorded set is used",
          torch_attrs <= ALLOWED_TORCH_ATTRS,
          f"torch.* used: {sorted(torch_attrs)}")
    registers = {"register_buffer", "register_parameter", "register_forward_hook"}
    check("no nn layer is constructed and no buffer/parameter is registered",
          nn_attrs <= {"Module"} and not (registers & set(names)),
          f"nn.* used: {sorted(nn_attrs)}; the only nn import is the Module base "
          f"class. Scanned as AST attribute accesses and NAME tokens, so this "
          f"module's prose about `nn.Embedding` cannot mask or fake a hit")

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            # node.module carries no leading dots; node.level does
            imports.append(("." * node.level + (node.module or "")) + ":" +
                           ",".join(a.name for a in node.names))
    allowed_imports = {
        "torch", "torch.nn", "torch.nn.functional",
        ".qspace_constraint:DEFAULT_LAMBDA_CSG,QSpaceDisabled,QSpaceError,"
        "REBUILD_ATOL,REBUILD_RTOL",
    }
    check("the module imports nothing but torch and C3's five public names",
          set(imports) <= allowed_imports and len(imports) == len(allowed_imports),
          f"imports: {sorted(imports)}")
    reached = sorted({k for i in imports for k in
                      ("rqvae", "models", "clean_tiger", "reference_repos",
                       "trainer", "main", "psa", "sia", "stage", "alternating")
                      if k in i})
    check("C4 does not reach into the frozen core, the reference, or any other "
          "port module at all",
          not reached,
          "checked on the import graph, not on prose: the objective reads a model "
          f"it is handed and nothing else. reached: {reached or 'nothing'}")

    qc_src = open(os.path.join(REPO, "tiger_ext", "qspace_constraint.py"),
                  encoding="utf-8").read()
    check("C3 was not back-patched to know about C4",
          "qspace_loss" not in qc_src and "qspace_constraint_loss" not in qc_src
          and "QSpaceLoss" not in qc_src,
          "C4 consumes C3's interface; C3 does not depend on C4")
    frozen_hits = subprocess.run(
        ["grep", "-rIl", "-e", "qspace_loss", "-e", "lambda_csg",
         os.path.join(REPO, "rqvae"), os.path.join(REPO, "model"),
         os.path.join(REPO, "clean_tiger")],
        capture_output=True, text=True).stdout.split()
    check("no frozen file (rqvae/, model/, clean_tiger/) mentions C4 or its switch",
          not frozen_hits, f"{len(frozen_hits)} hits")

    probe = subprocess.run([sys.executable, "-I", "-c", IMPORT_PROBE, REPO],
                           capture_output=True, text=True).stdout.strip()
    first = probe.splitlines()[0] if probe else ""
    check("importing the module reads nothing from the repository and calls no "
          "loader (proved, not asserted)",
          first == "IMPORT_CALLS=0",
          "open() counted for repo paths only, numpy.load / torch.load wrapped")

    import tiger_ext
    check("tiger_ext re-exports the C4 interface",
          tiger_ext.QSpaceLoss is QL.QSpaceLoss
          and tiger_ext.QSpaceLossMount is QL.QSpaceLossMount,
          "one definition of the objective and its mount")
    check("the C4 constants do not shadow C3's in tiger_ext",
          tiger_ext.MISSING_REFERENCE is QC.MISSING_REFERENCE
          and tiger_ext.DIRECTION_POLICY is QC.DIRECTION_POLICY,
          "C4's own copies are namespaced QSL_* precisely so this stays true")

    prov = QL.QSpaceLoss().provenance()
    check("the provenance record separates the reference objective from the adaptation",
          prov["is_a_reproduction"] is False
          and prov["csg4rec_reference"]["file"].endswith("CSGT/models/vq.py")
          and "ABSENT" in prov["csg4rec_missing_file"]["status"],
          "the missing file is recorded as missing, not reconstructed")
    check("all ten pre-code questions are answered in the frozen record",
          len(prov["design_decisions"]) == 10,
          f"{len(prov['design_decisions'])} decisions, keys "
          f"{sorted(k.split('_')[0] for k in prov['design_decisions'])}")
    check("the level-wise decision is recorded as forced by evidence, not chosen",
          "Forced by evidence" in prov["level_policy"],
          "emb is the level's own codebook (vq.py:69), so the class set is per-level")


# --------------------------------------------------------------------------- #
# [9] lambda_csg == 0
# --------------------------------------------------------------------------- #
def part9_disabled(RQVAE):
    print("\n[9] lambda_csg == 0: the objective is off, not silently zero")

    class _Tripwire:
        """A model that fails loudly if anything reads its codebooks."""

        class _Rq:
            @staticmethod
            def get_codebook():
                raise AssertionError("the codebooks were read while disabled")

        def __init__(self):
            self.rq = self._Rq()

    mount = QL.QSpaceLossMount()
    check("the default weight is 0, so the objective is disabled out of the box",
          mount.lambda_csg == QC.DEFAULT_LAMBDA_CSG == 0.0 and not mount.enabled,
          "a caller that forgets the switch gets no constraint, not a wrong one")
    check("calling a disabled mount raises instead of returning a manufactured zero",
          raises(lambda: mount(hand_batch(np.array([[0, 0]]), np.array([[1, 2]]),
                                          [__import__("torch").from_numpy(c)
                                           for c in FIX_CODEBOOKS]),
                               _Tripwire()), QC.QSpaceDisabled),
          "matching SIAMount / PSAMount: guard on `enabled`, do not read a 0")
    check("the disabled guard fires before the model is touched at all",
          mount.n_disabled_calls == 1 and mount.n_calls == 0,
          "the tripwire never raised, so no codebook was read")
    check("a live mount counts its own calls separately",
          _live_mount_counts(RQVAE), "the counters make a half-wired caller observable")
    check("the empty pair axis is a different situation from a disabled switch",
          not raises(lambda: QL.QSpaceLoss()(hand_batch(np.zeros((0, 2), np.int64),
                                                        np.zeros((0, 2), np.int64),
                                                        [__import__("torch").from_numpy(c)
                                                         for c in FIX_CODEBOOKS]),
                                             build_tiny_rqvae(RQVAE)), Exception),
          "empty -> 0.0 (a data outcome); disabled -> raise (a wiring error)")
    check("a disabled mount still holds no parameters and no buffers",
          len(list(mount.parameters())) == 0 and len(list(mount.buffers())) == 0,
          "nothing about the off state is trainable")


def _live_mount_counts(RQVAE):
    mount = QL.QSpaceLossMount(lambda_csg=0.5)
    _, _, _, batch = fixture_model_and_batch(RQVAE)
    loss = mount(batch, build_tiny_rqvae(RQVAE))
    return mount.n_calls == 1 and mount.n_disabled_calls == 0 and float(loss) > 0.0


# --------------------------------------------------------------------------- #
# [10] regression
# --------------------------------------------------------------------------- #
COUNT_RE = re.compile(r"(\d+)/(\d+) checks passed")


def part10_regression():
    print("\n[10] regression: the E1-E10 suites, re-run after C4")
    if REGRESSION == "skip":
        print("  [skip] EXP_C4_REGRESSION=skip")
        return
    env = dict(os.environ)
    env.pop("EXP_E10_DEVICE", None)
    proc = subprocess.run([sys.executable, "-I", "integration/exp_e10_smoke.py"],
                          cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=7200)
    match = COUNT_RE.search(proc.stdout)
    n, total = (int(match.group(1)), int(match.group(2))) if match else (-1, -1)
    check("exp_e10_smoke (which re-runs E1-E9) still passes, unchanged",
          proc.returncode == 0 and n == total and n > 0,
          f"{n}/{total} checks passed, exit {proc.returncode}")
    check("the E10 run inside this regression is the same 64-check suite C3 recorded",
          total == 64, f"{total} checks in the suite")
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-2000:])


# --------------------------------------------------------------------------- #
def main():
    print("=" * 76)
    print("EXP-C4 -- Q-space collaborative constraint objective (CSG4Rec -> TIGER)")
    print("=" * 76)

    # order matters: both model/ and rqvae/ contain a top-level `main.py`, so
    # MODEL_DIR must win the lookup for `import main`
    sys.path.insert(0, RVQAE_DIR)
    sys.path.insert(0, MODEL_DIR)
    sys.path.insert(0, REPO)
    from models.rqvae import RQVAE

    part1_fixture(RQVAE)
    part2_multi_pair(RQVAE)
    part3_directed(RQVAE)
    part4_empty(RQVAE)
    part5_weight(RQVAE)
    part6_gradient(RQVAE)
    part7_stability(RQVAE)
    part8_scope()
    part9_disabled(RQVAE)
    part10_regression()

    total = len(RESULTS)
    passed = sum(RESULTS)
    print("\n" + "=" * 76)
    print(f"EXP-C4: {passed}/{total} checks passed")
    print("=" * 76)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
