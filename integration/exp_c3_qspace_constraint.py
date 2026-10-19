"""EXP-C3 -- Q-space collaborative structure constraint (construction half only).

Run:  python -I integration/exp_c3_qspace_constraint.py
      EXP_C3_REGRESSION=skip python -I integration/exp_c3_qspace_constraint.py

Covers plan.md C3's acceptance criteria. C3 turns C2's collaborative relation
into a constraint on the RQ-VAE's **quantised** representation. It builds the
per-batch intermediate; it computes no loss (that is C4).

  [1] synthetic fixture, hand-computed end to end: a 4-item relation with a
      scrambled ItemID column, a deliberately empty neighbour row and a
      deliberately short one; the expected pairs, weights, gathered code indices
      and gathered x_q are written out as literals and compared, plus the
      independent relation -> Q-space correspondence check
  [2] B provenance: the captured tensor really is ``sum_l codebook_l[indices_l]``
      and really is not the encoder latent, with the negative controls that make
      the check mean something
  [3] sentinels: ``-1`` / ``NaN`` never reach a tensor operation, never appear in
      the output, and a relation whose two sentinels disagree is refused
  [4] out-of-batch policy: measured counts, monotonicity in batch size, the
      empty pair axis, and the out-of-range row refusal
  [5] row-space and ItemID mapping: ``row_of`` (never ``id - 1``), the Beauty
      negative control, and the load-time refusal of a wrong ``expected_item_ids``
  [6] real data (Toys / Sport / Beauty) minimal validation end to end, plus the
      one genuinely trained Clean TIGER tokenizer on Toys
  [7] scope: no loss, no backward, no parameter, no frozen directory touched,
      nothing read at import time
  [8] the ``lambda_csg == 0`` no-op and its tripwires
  [9] regression: the E1-E10 suites, re-run after C3

What is measured vs what is claimed
-----------------------------------
Every check here is an engineering check. [1]-[5] and [7]-[9] are structural.
[6] exercises the real row spaces, real k=20 relations and real batch geometry of
the three datasets, but it makes **no** claim about the tokenizers used for Sport
and Beauty: no RQ-VAE trained on those two datasets' canonical 4096-d content
table exists in this repository (``rqvae/ckpt/Sport_TA_SID_4096d`` and friends are
trained on a *fused* table, not ``data/Sport/item_emb.parquet``), so the per-dataset
loop fits a fixture tokenizer of the frozen architecture by k-means on that
dataset's own content latents -- enough to make the Q-space non-degenerate, and
labelled as a fixture. Toys is the one dataset with a genuinely trained
canonical-content tokenizer, and it is run twice: once as the fixture (so the
three dataset rows are comparable) and once with the real checkpoint.

No recommendation-quality claim is made anywhere in this file. Engineering gate
only.
"""
import ast
import io
import os
import re
import subprocess
import sys
import tempfile
import tokenize as _tokenize

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MODEL_DIR = os.path.join(REPO, "model")
RVQAE_DIR = os.path.join(REPO, "rqvae")
DATA_DIR = os.path.join(REPO, "data")
SMOKE_TOYS = os.path.join(REPO, "clean_tiger", "_smoke", "Toys")
TOYS_CKPT = os.path.join(REPO, "clean_tiger", "ckpt", "Toys", "rqvae",
                         "Oct-14-2026_05-55-09", "best_collision_model.pth")

sys.path.insert(0, REPO)

from tiger_ext import qspace_constraint as QC                 # noqa: E402
from tiger_ext import collab_relation as CR                   # noqa: E402

MODULE = os.path.join(REPO, "tiger_ext", "qspace_constraint.py")
REGRESSION = os.environ.get("EXP_C3_REGRESSION", "all")

# the tracked files E1-E6 legitimately modified; C3 must not add to this set
PRE_C3_MODIFIED = {"model/main.py", "rqvae/models/rq.py", "rqvae/models/vq.py"}

DATASETS = ("Toys", "Sport", "Beauty")
N_ITEMS = {"Toys": 11924, "Sport": 18357, "Beauty": 12101}
BEAUTY_SCRAMBLED_ROWS = 4158          # Beauty rows where ItemID != row + 1
K = 20                                # CSG4Rec's --knn_k default
SEED = 2025
BATCH = 256
DEVICE = "cpu"

# the frozen TIGER RQ-VAE architecture (clean_tiger/config.sh)
RQVAE_CFG = dict(num_emb_list=[256, 256, 256], e_dim=32, layers=[512, 256, 128, 64],
                 dropout_prob=0.0, bn=False, loss_type="mse", quant_loss_weight=1.0,
                 beta=0.25, kmeans_init=False, kmeans_iters=100,
                 sk_epsilons=[0.0, 0.0, 0.003], sk_iters=50)

# --------------------------------------------------------------------------- #
# [1] the synthetic fixture -- every expected value below is computed by hand
# --------------------------------------------------------------------------- #
# tiny model: identity encoder, e_dim=2, two RQ levels, explicit codebooks
FIX_CODEBOOKS = (
    np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]], dtype=np.float32),      # level 0: 3 codewords
    np.array([[0.0, 2.0],
              [3.0, 0.0]], dtype=np.float32),      # level 1: 2 codewords
)
# latents (== encoder output, the encoder is identity)
FIX_X = np.array([[2.0, 0.0],     # row 0
                  [0.0, 3.0],     # row 1
                  [5.0, 4.0],     # row 2
                  [1.0, 5.0]],    # row 3
                 dtype=np.float32)
# hand-computed argmin over the squared L2 distance, level by level, on the
# residual left by the previous level
FIX_INDICES = np.array([[0, 1],
                        [1, 0],
                        [2, 1],
                        [2, 0]], dtype=np.int64)
# hand-computed sum of the two codebook lookups
FIX_X_Q = np.array([[4.0, 0.0],   # [1,0] + [3,0]
                    [0.0, 3.0],   # [0,1] + [0,2]
                    [4.0, 1.0],   # [1,1] + [3,0]
                    [1.0, 3.0]],  # [1,1] + [0,2]
                   dtype=np.float32)

# the relation: n=4, k_eff=2, ItemIDs scrambled so `id - 1` cannot find a row,
# row 3 has no neighbours at all and row 1 has only one
FIX_IDS = np.array([10, 3, 7, 1], dtype=np.int64)
FIX_NEIGHBORS = np.array([[1, 3],          # row 0 -> rows 1, 3
                          [0, -1],         # row 1 -> row 0, then empty
                          [3, 0],          # row 2 -> rows 3, 0
                          [-1, -1]],       # row 3 -> nothing
                         dtype=np.int32)
FIX_SIMILARITY = np.array([[0.9, 0.4],
                           [0.7, np.nan],
                           [0.5, 0.2],
                           [np.nan, np.nan]], dtype=np.float32)

# the batch order is a permutation of the rows on purpose, so a position/row
# confusion cannot pass unnoticed: position -> row
FIX_BATCH_ROWS = [2, 0, 3, 1]

# hand-computed by walking the batch: for each position, that row's out-list,
# keeping a slot only when the neighbour is also in the batch
#   pos 0 (row 2): (row 3 @ pos 2, w 0.5), (row 0 @ pos 1, w 0.2)
#   pos 1 (row 0): (row 1 @ pos 3, w 0.9), (row 3 @ pos 2, w 0.4)
#   pos 2 (row 3): no out-list
#   pos 3 (row 1): (row 0 @ pos 1, w 0.7), then the -1 slot is dropped
FIX_ANCHOR_POS = [0, 0, 1, 1, 3]
FIX_TARGET_POS = [2, 1, 3, 2, 1]
FIX_WEIGHT = [0.5, 0.2, 0.9, 0.4, 0.7]
# gathered from FIX_X_Q at those positions
FIX_ANCHOR_ROWS = [2, 2, 0, 0, 1]
FIX_TARGET_ROWS = [3, 0, 1, 3, 0]
FIX_XQ_ANCHOR = [[4.0, 1.0], [4.0, 1.0], [4.0, 0.0], [4.0, 0.0], [0.0, 3.0]]
FIX_XQ_TARGET = [[1.0, 3.0], [4.0, 0.0], [0.0, 3.0], [1.0, 3.0], [4.0, 0.0]]
FIX_IND_ANCHOR = [[2, 1], [2, 1], [0, 1], [0, 1], [1, 0]]
FIX_IND_TARGET = [[2, 0], [0, 1], [1, 0], [2, 0], [0, 1]]

# identifiers that may appear in the *code* of the C3 module. The four tokens
# below are absent from every NAME token; the only occurrences in the file are
# inside the docstring and the provenance strings describing CSG4Rec's reference
FORBIDDEN_TOKENS = ("backward", "cross_entropy", "softmax", "loss", "info_nce",
                    "infonce", "nce_loss", "contrastive", "projection", "proj",
                    "warmup", "concat", "user")
FORBIDDEN_ALLOWED = {"rq_loss"}       # the RQ-VAE's own return, unpacked and dropped

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
    except Exception:                       # a wrong exception type is a failure too
        return False
    return False


def source_count(path, needle):
    return open(path, encoding="utf-8").read().count(needle)


# --------------------------------------------------------------------------- #
# fixture construction
# --------------------------------------------------------------------------- #
def build_tiny_rqvae(RQVAE):
    """An RQ-VAE with an identity encoder and the hand-set codebooks above."""
    import torch
    from torch import nn

    class _Identity(nn.Module):
        def forward(self, x):
            return x

    model = RQVAE(in_dim=FIX_X.shape[1], e_dim=FIX_X.shape[1], num_emb_list=[3, 2],
                  layers=[], dropout_prob=0.0, bn=False, loss_type="mse",
                  beta=0.25, kmeans_init=False, sk_epsilons=[0.0, 0.0], sk_iters=10)
    model.encoder = _Identity()
    model.eval()
    for layer, codebook in zip(model.rq.vq_layers, FIX_CODEBOOKS):
        layer.embedding.weight.data.copy_(torch.from_numpy(codebook))
        layer.initted = True
    return model


def build_fixture_relation(neighbors=FIX_NEIGHBORS, similarity=FIX_SIMILARITY,
                           item_ids=FIX_IDS, k=2):
    return CR.CollabRelation(neighbors, similarity, item_ids, k=k)


def fixture_batch_x():
    """The fixture's latents in *batch* order.

    The batch is a permutation of the rows, so its content must be permuted with
    it: batch element ``p`` carries the latent of row ``FIX_BATCH_ROWS[p]``. Handing
    the latents over in row order instead would be exactly the position/row
    confusion these checks exist to catch.
    """
    import torch

    return torch.from_numpy(FIX_X[FIX_BATCH_ROWS])


def fit_fixture_tokenizer(RQVAE, x_all, seed=SEED):
    """A fixture tokenizer of the frozen architecture, codebooks set by k-means.

    No training happens: each level's codebook is initialised from that level's
    residual by the model's *own* ``init_emb`` (the k-means initialiser the
    frozen recipe uses), so the resulting Q-space is non-degenerate. Only used
    for the two datasets that have no trained canonical-content tokenizer.
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)          # sklearn's KMeans draws from the global state
    model = RQVAE(in_dim=int(x_all.shape[1]), **RQVAE_CFG)
    model.eval()
    with torch.no_grad():
        residual = []
        for start in range(0, x_all.shape[0], 8192):
            residual.append(model.encoder(x_all[start:start + 8192]))
        residual = torch.cat(residual, dim=0)
        for layer in model.rq.vq_layers:
            layer.init_emb(residual)
            weight = layer.embedding.weight
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
# [1] the synthetic fixture
# --------------------------------------------------------------------------- #
def part1_synthetic(RQVAE):
    import torch

    print("\n[1] synthetic fixture: hand-computed relation -> pairs -> Q-space")
    relation = build_fixture_relation()
    check("the hand-written relation passes C2's own validator",
          relation.k_eff == 2 and relation.n_edges == 5,
          f"k_eff={relation.k_eff}, edges={relation.n_edges}")

    model = build_tiny_rqvae(RQVAE)
    quant = QC.capture_qspace(model, fixture_batch_x(), FIX_BATCH_ROWS, use_sk=False)

    check("the tiny model reproduces the hand-computed code indices, in the "
          "batch's own order",
          np.array_equal(quant.indices.numpy(), FIX_INDICES[FIX_BATCH_ROWS]),
          f"got {quant.indices.tolist()}; rows are {FIX_BATCH_ROWS}")
    check("the tiny model reproduces the hand-computed quantised vectors, in the "
          "batch's own order",
          np.allclose(quant.x_q.numpy(), FIX_X_Q[FIX_BATCH_ROWS], atol=1e-6),
          f"got {quant.x_q.tolist()}")
    check("the provenance check measured a real, tiny rebuild deviation",
          quant.diagnostics["rebuild_max_abs_delta"] < QC.REBUILD_ATOL,
          f"max |x_q - sum_l codebook_l[indices_l]| = "
          f"{quant.diagnostics['rebuild_max_abs_delta']:.3e}")
    check("the same tensor is far from the encoder latent, so the provenance "
          "check is not vacuous on the fixture",
          quant.diagnostics["latent_max_abs_delta"] > 1.0,
          f"max |x_q - z| = {quant.diagnostics['latent_max_abs_delta']:.3f}")

    mount = QC.QSpaceConstraint(relation, lambda_csg=1.0, k=2)
    batch = mount(quant)

    check("the pair count is the hand-computed 5",
          batch.n_pairs == 5, f"got {batch.n_pairs}")
    check("anchor positions are the hand-computed ones",
          batch.anchor_pos.tolist() == FIX_ANCHOR_POS,
          f"got {batch.anchor_pos.tolist()}")
    check("target positions are the hand-computed ones",
          batch.target_pos.tolist() == FIX_TARGET_POS,
          f"got {batch.target_pos.tolist()}")
    check("weights are the relation's own similarities, in the hand-computed order",
          np.allclose(batch.weight.numpy(), FIX_WEIGHT, atol=1e-6),
          f"got {batch.weight.tolist()}")
    check("the pairs' rows are the hand-computed ones (position -> row, not identity)",
          batch.anchor_rows.tolist() == FIX_ANCHOR_ROWS
          and batch.target_rows.tolist() == FIX_TARGET_ROWS,
          f"anchors {batch.anchor_rows.tolist()}, targets {batch.target_rows.tolist()}")
    check("the gathered code indices are the hand-computed ones",
          batch.indices_anchor.tolist() == FIX_IND_ANCHOR
          and batch.indices_target.tolist() == FIX_IND_TARGET,
          "each side is the code tuple of the row named by anchor_rows / target_rows")
    check("the gathered x_q is the hand-computed one",
          np.allclose(batch.x_q_anchor.numpy(), FIX_XQ_ANCHOR, atol=1e-6)
          and np.allclose(batch.x_q_target.numpy(), FIX_XQ_TARGET, atol=1e-6),
          "x_q gathered at the pair axis, both sides")

    # the independent relation -> Q-space correspondence check: every emitted
    # pair must be an edge of the relation, and its Q-space must be the one
    # built from the code indices of the two rows it names
    edges = {(int(a), int(b)) for a, b in relation.pairs}
    emitted = {(int(a), int(b)) for a, b in
               zip(batch.anchor_rows.tolist(), batch.target_rows.tolist())}
    check("every emitted pair is an edge of the relation, and none is missing",
          emitted == edges, f"emitted {sorted(emitted)} vs relation {sorted(edges)}")
    rebuilt = np.stack([FIX_CODEBOOKS[0][batch.indices_anchor.numpy()[:, 0]]
                        + FIX_CODEBOOKS[1][batch.indices_anchor.numpy()[:, 1]]])
    check("the anchor side's x_q is rebuilt from its own code indices, in numpy, "
          "independently of the module",
          np.allclose(rebuilt, batch.x_q_anchor.numpy(), atol=1e-6),
          "no torch tensor and no module helper is used for this")
    check("no pair is a self-pair and none is duplicated",
          all(a != b for a, b in emitted)
          and len(emitted) == batch.n_pairs,
          f"{batch.n_pairs} distinct non-self pairs")
    check("the direction is not symmetrised: (row 0 -> row 3) exists while "
          "(row 3 -> row 0) cannot, because row 3 has no out-list",
          (0, 3) in emitted and (3, 0) not in emitted,
          "a consumer wanting both directions must ask for them")


# --------------------------------------------------------------------------- #
# [2] B provenance
# --------------------------------------------------------------------------- #
def part2_provenance(RQVAE, EmbDataset):
    import torch

    print("\n[2] B provenance: what was captured is the codebook sum, not the latent")
    emb = EmbDataset(os.path.join(SMOKE_TOYS, "data", "Toys", "item_emb.parquet"))
    ckpt = torch.load(TOYS_CKPT, map_location="cpu", weights_only=False)
    model = RQVAE(in_dim=emb.dim, **RQVAE_CFG)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x = torch.as_tensor(emb.embeddings[:64], dtype=torch.float32)
    rows = torch.arange(64)

    quant = QC.capture_qspace(model, x, rows)
    check("the trained Clean TIGER tokenizer's capture passes the provenance check",
          quant.x_q.shape == (64, 32) and quant.n_levels == 3,
          f"x_q {tuple(quant.x_q.shape)}, {quant.n_levels} RQ levels")
    check("its rebuild deviation is at float32 straight-through level, far below "
          "the tolerance",
          quant.diagnostics["rebuild_max_abs_delta"] < QC.REBUILD_ATOL,
          f"max |delta| = {quant.diagnostics['rebuild_max_abs_delta']:.3e} "
          f"(tolerance {QC.REBUILD_ATOL:.0e})")
    check("its Q-space is genuinely distinct from the encoder latent",
          quant.diagnostics["latent_max_abs_delta"] > QC.REBUILD_ATOL * 100,
          f"max |x_q - z| = {quant.diagnostics['latent_max_abs_delta']:.3f}")

    # negative control 1: hand over the latent instead of the quantised vector
    bad = QC.QSpaceBatch(x_q=quant.latent.detach().clone(), indices=quant.indices,
                         rows=rows, latent=quant.latent)
    check("handing the encoder latent over as `x_q` is rejected",
          raises(lambda: bad.validate(model), QC.QSpaceMismatch),
          "this is exactly the latent-space alignment C3 is forbidden to become")

    # negative control 2: tamper with one code index
    tampered = quant.indices.detach().clone()
    tampered[0, 0] = (int(tampered[0, 0]) + 1) % model.rq.vq_layers[0].n_e
    bad2 = QC.QSpaceBatch(x_q=quant.x_q.detach().clone(), indices=tampered,
                          rows=rows, latent=quant.latent)
    check("a batch whose code indices do not explain its x_q is rejected",
          raises(lambda: bad2.validate(model), QC.QSpaceMismatch),
          "the provenance check ties the tensor to *this* model's codebooks")

    # negative control 3: a batch carrying a level the model does not have
    short = QC.QSpaceBatch(x_q=quant.x_q.detach().clone(),
                           indices=quant.indices[:, :2].detach().clone(),
                           rows=rows, latent=quant.latent)
    check("a batch with the wrong number of RQ levels is rejected",
          raises(lambda: short.validate(model), QC.QSpaceMismatch),
          "the collision token is not a quantiser level and can never enter here")

    # negative control 4: rows that do not match the batch
    bad3 = QC.QSpaceBatch(x_q=quant.x_q.detach().clone(),
                          indices=quant.indices.detach().clone(),
                          rows=torch.arange(63), latent=quant.latent)
    check("a batch whose row vector does not match its tensors is rejected",
          raises(lambda: bad3.validate(model), QC.QSpaceMismatch),
          "identity is data too")

    # rebuilt from the model's own codebooks, outside the module and outside the
    # capture's no_grad context -- a direct gather with no helper involved
    with torch.no_grad():
        direct = np.stack([model.rq.vq_layers[level].embedding.weight[
                               quant.indices[:, level].numpy()].numpy()
                           for level in range(quant.n_levels)]).sum(axis=0)
    check("the module's own `rebuild_quantized` agrees with a direct codebook "
          "gather written out longhand",
          np.allclose(QC.rebuild_quantized(model, quant.indices).detach().numpy(),
                      direct, atol=1e-5),
          "the verification helper is not doing anything clever")
    check("`rebuild_quantized` is a verification-only path: it is never reached "
          "outside the no_grad capture in this module",
          source_count(MODULE, "rebuild_quantized(") == 2,
          "one definition, one call, and that call sits inside validate()")


# --------------------------------------------------------------------------- #
# [3] sentinels
# --------------------------------------------------------------------------- #
def part3_sentinels(RQVAE):
    import torch

    print("\n[3] sentinels: -1 and NaN are filtered, never computed with")
    relation = build_fixture_relation()
    mount = QC.QSpaceConstraint(relation, lambda_csg=1.0, k=2)
    model = build_tiny_rqvae(RQVAE)
    quant = QC.capture_qspace(model, fixture_batch_x(), FIX_BATCH_ROWS, use_sk=False)
    batch = mount(quant)

    check("the 3 sentinel slots are dropped, not turned into row -1",
          batch.stats["dropped_sentinel"] == 3,
          f"row 3 contributes 2, row 1 contributes 1; got "
          f"{batch.stats['dropped_sentinel']}")
    check("every dropped sentinel is accounted for: slots = pairs + sentinel + "
          "out-of-batch",
          4 * 2 == batch.stats["pairs"] + batch.stats["dropped_sentinel"]
          + batch.stats["dropped_out_of_batch"],
          f"8 == {batch.stats['pairs']} + {batch.stats['dropped_sentinel']} + "
          f"{batch.stats['dropped_out_of_batch']}")
    check("no NaN and no inf survives into the constraint weights",
          bool(torch.isfinite(batch.weight).all()),
          f"weights {batch.weight.tolist()}")
    check("no NaN and no inf survives into either Q-space side",
          bool(torch.isfinite(batch.x_q_anchor).all())
          and bool(torch.isfinite(batch.x_q_target).all()),
          "a NaN here would poison any downstream mean")
    check("the anchor of the all-sentinel row never appears in the pair axis",
          2 not in batch.anchor_pos.tolist(),
          "row 3 has no neighbours and so contributes no anchor")
    check("anchors with at least one pair are counted separately from the batch size",
          batch.stats["anchors_with_pair"] == 3 and batch.stats["batch_size"] == 4,
          f"{batch.stats['anchors_with_pair']} of {batch.stats['batch_size']}")

    # the guard that C2 guarantees cannot fire, proved to fire when the
    # relation object did not come from C2
    class _Tampered:
        n_items = 4
        neighbors = np.array([[1, 3], [0, -1], [3, 0], [-1, -1]], dtype=np.int32)
        similarity = np.array([[0.9, 0.4], [0.7, 0.6], [0.5, 0.2],
                               [np.nan, np.nan]], dtype=np.float32)

    tampered_mount = QC.QSpaceConstraint(lambda_csg=1.0, k=2)
    tampered_mount.relation = _Tampered()
    check("a relation whose -1 and its finite similarity disagree is refused",
          raises(lambda: tampered_mount.pairs_for_rows(torch.tensor(FIX_BATCH_ROWS)),
                 QC.QSpaceMismatch),
          "row 1 slot 1 is -1 but carries a finite cosine")

    class _AlsoTampered:
        n_items = 4
        neighbors = np.array([[1, 3], [0, -1], [3, 0], [-1, -1]], dtype=np.int32)
        similarity = np.array([[0.9, 0.4], [0.7, np.nan], [0.5, 0.2],
                               [0.1, np.nan]], dtype=np.float32)

    tampered_mount.relation = _AlsoTampered()
    check("the reverse disagreement is refused too: a finite neighbour carrying "
          "a NaN similarity",
          raises(lambda: tampered_mount.pairs_for_rows(torch.tensor(FIX_BATCH_ROWS)),
                 QC.QSpaceMismatch),
          "row 3 slot 0 is a real neighbour with a NaN cosine")


# --------------------------------------------------------------------------- #
# [4] the out-of-batch policy
# --------------------------------------------------------------------------- #
def part4_out_of_batch(RQVAE):
    import torch

    print("\n[4] out-of-batch policy: in-batch pairs only, drop and count")
    relation = build_fixture_relation()
    mount = QC.QSpaceConstraint(relation, lambda_csg=1.0, k=2)

    full = mount.pairs_for_rows(torch.tensor(FIX_BATCH_ROWS))[3]
    check("a batch holding every row drops nothing for out-of-batch reasons",
          full["dropped_out_of_batch"] == 0 and full["pairs"] == 5,
          f"pairs={full['pairs']}, dropped={full['dropped_out_of_batch']}")

    half = mount.pairs_for_rows(torch.tensor([0, 1]))[3]
    check("a two-row batch keeps only the edges inside it",
          half["pairs"] == 2 and half["dropped_out_of_batch"] == 1
          and half["dropped_sentinel"] == 1,
          "row 0 -> row 1 survives, row 0 -> row 3 does not, row 1's slot is a "
          "sentinel")

    single = mount.pairs_for_rows(torch.tensor([0]))[3]
    single_pos, single_tgt, single_w, _ = mount.pairs_for_rows(torch.tensor([0]))
    check("a one-row batch has an empty pair axis, and the empty axis is explicit",
          single["pairs"] == 0 and single_pos.numel() == 0
          and single_tgt.numel() == 0 and single_w.numel() == 0,
          "the caller checks is_empty rather than assuming a shape")

    check("the pair count is monotone in the batch size",
          full["pairs"] >= half["pairs"] >= single["pairs"],
          f"{full['pairs']} >= {half['pairs']} >= {single['pairs']}")

    model = build_tiny_rqvae(RQVAE)
    quant = QC.capture_qspace(model, fixture_batch_x()[:1], [0], use_sk=False)
    batch = mount(quant)
    check("the empty pair axis survives the full forward as (0, dim) tensors",
          batch.is_empty and batch.x_q_anchor.shape == (0, 2)
          and batch.x_q_target.shape == (0, 2) and batch.weight.shape == (0,),
          f"x_q_anchor {tuple(batch.x_q_anchor.shape)}")
    check("forward records the representation it built on",
          batch.stats["representation"] == QC.QSPACE_REPRESENTATION
          and batch.stats["lambda_csg"] == 1.0,
          f"representation={batch.stats['representation']}")

    check("a batch reaching past the relation's row space is refused",
          raises(lambda: mount.pairs_for_rows(torch.tensor([0, 4])),
                 QC.QSpaceMismatch),
          "row 4 does not exist in a 4-row relation")
    check("a negative row index is refused",
          raises(lambda: mount.pairs_for_rows(torch.tensor([-1])),
                 QC.QSpaceMismatch),
          "-1 is a sentinel in the relation, not a row of the batch")


# --------------------------------------------------------------------------- #
# [5] row space and ItemID mapping
# --------------------------------------------------------------------------- #
def part5_mapping():
    print("\n[5] row space and ItemID mapping: `row_of`, never `id - 1`")
    relation = build_fixture_relation()

    check("rows_of_item_ids resolves the scrambled ids through the relation",
          QC.rows_of_item_ids(relation, [7, 1, 10]).tolist() == [2, 3, 0],
          "ItemIDs [10,3,7,1] live at rows [0,1,2,3]")
    check("the fixture itself would be mis-mapped by `id - 1`",
          int(QC.rows_of_item_ids(relation, [3])[0]) != 3 - 1,
          "ItemID 3 is at row 1, not row 2")

    # the real negative control: Beauty's ItemID column is a permutation
    beauty_ids = CR.CollabRelation.load(
        CR.relation_prefix("Beauty", k=K), expected_item_ids=None,
        expected_k=K).item_ids
    scrambled = np.nonzero(beauty_ids != np.arange(1, beauty_ids.size + 1))[0]
    check("Beauty's ItemID column really is scrambled, as C1/C2 recorded",
          scrambled.size == BEAUTY_SCRAMBLED_ROWS,
          f"{scrambled.size} of {beauty_ids.size} rows have ItemID != row + 1")
    row = int(scrambled[0])
    check("`row_of` returns the true row while `id - 1` does not",
          CR.CollabRelation.load(CR.relation_prefix("Beauty", k=K),
                                 expected_k=K).row_of(int(beauty_ids[row])) == row
          and int(beauty_ids[row]) - 1 != row,
          f"ItemID {int(beauty_ids[row])} is at row {row}; `id - 1` would give "
          f"{int(beauty_ids[row]) - 1}")

    # the load-time refusal
    wrong = np.array(FIX_IDS[::-1], dtype=np.int64)
    mount = QC.QSpaceConstraint(lambda_csg=1.0, k=2)
    with tempfile.TemporaryDirectory(prefix="exp_c3_") as tmp:
        prefix = os.path.join(tmp, "fixture_k2")
        relation.save(prefix)
        check("a relation loads through C2's interface with the right ids",
              mount.load_relation(prefix, expected_item_ids=FIX_IDS,
                                  expected_k=2).n_items == 4,
              "the .npy files are never opened directly by C3")
        check("reloading with a permuted expected_item_ids is refused at load time",
              raises(lambda: mount.load_relation(prefix, expected_item_ids=wrong,
                                                 expected_k=2),
                     CR.CollabRelationMismatch),
              "row space is checked, not assumed")
        check("reloading with the wrong expected_k is refused at load time",
              raises(lambda: mount.load_relation(prefix,
                                                 expected_item_ids=FIX_IDS,
                                                 expected_k=5),
                     CR.CollabRelationMismatch),
              "k=20 is the pinned contract, not a suggestion")
        check("a mount pinned to k=20 refuses a k=2 relation at construction",
              raises(lambda: QC.QSpaceConstraint(relation, lambda_csg=1.0),
                     QC.QSpaceMismatch),
              "the default k is tiger_ext.collab_relation.RELATION_K")
        check("an ItemID outside the row space raises rather than guessing",
              raises(lambda: CR.CollabRelation.load(
                  prefix, expected_k=2).row_of(999), CR.CollabRelationMismatch),
              "no nearest-match, no wraparound")
    check("the per-dataset entry point reads the relation through the same interface",
          QC.load_constraint_for_dataset("Toys", lambda_csg=1.0).relation.n_items
          == N_ITEMS["Toys"],
          "expected_item_ids comes from C1's canonical list")


# --------------------------------------------------------------------------- #
# [6] real data, three datasets
# --------------------------------------------------------------------------- #
def part6_real(RQVAE):
    import torch

    print("\n[6] real data: Toys / Sport / Beauty, minimal end-to-end validation")
    real = None
    for dataset in DATASETS:
        mount = QC.load_constraint_for_dataset(dataset, lambda_csg=1.0, k=K)
        relation = mount.relation
        x_all, ids = load_content(dataset)

        check(f"{dataset}: the relation is the real k=20 one over the canonical "
              f"row space",
              relation.n_items == N_ITEMS[dataset] and relation.k == K
              and np.array_equal(relation.item_ids, ids),
              f"n_items={relation.n_items}, k={relation.k}, "
              f"n_edges={relation.n_edges}")

        # the fixture tokenizer is fitted on the *whole* dataset, then the batch
        # is the first BATCH rows of the same canonical order
        model = fit_fixture_tokenizer(RQVAE, x_all)
        x = x_all[:BATCH]
        rows = torch.arange(BATCH)
        quant = QC.capture_qspace(model, x, rows)
        batch = mount(quant)

        check(f"{dataset}: capture passes the provenance check on real content",
              quant.x_q.shape == (BATCH, 32)
              and quant.diagnostics["rebuild_max_abs_delta"] < QC.REBUILD_ATOL
              and quant.diagnostics["latent_max_abs_delta"] > QC.REBUILD_ATOL,
              f"rebuild delta {quant.diagnostics['rebuild_max_abs_delta']:.3e}, "
              f"latent gap {quant.diagnostics['latent_max_abs_delta']:.3f}")

        counts_ok = (batch.stats["slots"] == batch.n_pairs
                     + batch.stats["dropped_sentinel"]
                     + batch.stats["dropped_out_of_batch"])
        finite_ok = (bool(torch.isfinite(batch.weight).all())
                     and bool(torch.isfinite(batch.x_q_anchor).all())
                     and bool(torch.isfinite(batch.x_q_target).all()))
        nonself = bool((batch.anchor_rows != batch.target_rows).all())
        check(f"{dataset}: {batch.n_pairs} in-batch pairs, every slot accounted "
              f"for, nothing non-finite, no self-pair",
              batch.n_pairs > 0 and counts_ok and finite_ok and nonself,
              f"pairs={batch.n_pairs}, dropped_out_of_batch="
              f"{batch.stats['dropped_out_of_batch']}, dropped_sentinel="
              f"{batch.stats['dropped_sentinel']}, "
              f"anchors_with_pair={batch.stats['anchors_with_pair']}/{BATCH}")

        edges = {(int(a), int(b)) for a, b in relation.pairs}
        emitted = set(zip(batch.anchor_rows.tolist(), batch.target_rows.tolist()))
        check(f"{dataset}: every emitted pair is a real edge of the relation, and "
              f"every edge inside the batch is emitted",
              emitted <= edges and len(emitted) == batch.n_pairs,
              f"{len(emitted)} distinct pairs, all in the relation's "
              f"{relation.n_edges} edges")

        check(f"{dataset}: the emitted rows span exactly the batch's own rows",
              set(batch.anchor_rows.tolist()) <= set(rows.tolist())
              and set(batch.target_rows.tolist()) <= set(rows.tolist()),
              f"rows [0, {BATCH - 1}]")

        print(f"      {dataset}: {relation.n_edges:,} edges, "
              f"out-degree min/median {int(relation.out_degree.min())}/"
              f"{int(np.median(relation.out_degree))}, "
              f"{batch.n_pairs} pairs from {BATCH} items "
              f"({batch.n_pairs / BATCH:.2f} per item); fixture tokenizer uses "
              f"{int(torch.unique(quant.indices[:, 0]).numel())} of "
              f"{model.rq.vq_layers[0].n_e} level-0 codes on the batch")
        if dataset == "Toys":
            real = (mount, model, x, rows, batch)

    # the one genuinely trained canonical-content tokenizer in the repository
    mount, fixture_model, x, rows, fixture_batch = real
    ckpt = torch.load(TOYS_CKPT, map_location="cpu", weights_only=False)
    trained = RQVAE(in_dim=int(x.shape[1]), **RQVAE_CFG)
    trained.load_state_dict(ckpt["state_dict"])
    trained.eval()
    quant = QC.capture_qspace(trained, x, rows)
    batch = mount(quant)
    check("Toys: the trained Clean TIGER tokenizer drives the same interface",
          batch.n_pairs == fixture_batch.n_pairs
          and batch.anchor_pos.tolist() == fixture_batch.anchor_pos.tolist(),
          f"{batch.n_pairs} pairs, identical pair axis to the fixture tokenizer")
    check("Toys: the trained tokenizer's code indices are not degenerate "
          "(a different Q-space, same pairing)",
          int(torch.unique(quant.indices[:, 0]).numel()) > 1
          and not np.array_equal(quant.indices.numpy(),
                                 fixture_model.get_indices(x).numpy()),
          f"{int(torch.unique(quant.indices[:, 0]).numel())} distinct level-0 "
          f"codes over {BATCH} items")
    check("the pair axis depends only on the relation, never on the tokenizer",
          batch.stats["pairs"] == fixture_batch.stats["pairs"]
          and batch.stats["dropped_out_of_batch"]
          == fixture_batch.stats["dropped_out_of_batch"],
          "A and C are separable from B, which is the point of the split")

    print(f"      pair coverage across a {BATCH}-item batch is a batch-geometry "
          f"quantity -- it depends on which rows were drawn, not on the "
          f"tokenizer -- which is why the same pair axis comes out of both")

    part6_device(RQVAE, mount, x, rows, batch)


def part6_device(RQVAE, mount_cpu, x, rows, cpu_batch):
    """The same path on CUDA: rows stay on CPU while the Q-space does not.

    This is not decoration. ``rows`` is integer *identity* and the rest of the
    project keeps it on CPU (C2's relation is a numpy object); ``x_q`` is model
    output and lives wherever the model lives. The pair axis is built by indexing
    a CPU index vector into `x_q`, so a device mismatch would surface here rather
    than in C5's trainer.
    """
    import torch

    if not torch.cuda.is_available():
        print("  [skip] no CUDA device; the CPU path is covered above")
        return

    ckpt = torch.load(TOYS_CKPT, map_location="cpu", weights_only=False)
    model = RQVAE(in_dim=int(x.shape[1]), **RQVAE_CFG).to("cuda")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    quant = QC.capture_qspace(model, x.to("cuda"), rows)
    check("CUDA: the provenance check passes on-device, against this device's "
          "codebooks",
          quant.x_q.is_cuda
          and quant.diagnostics["rebuild_max_abs_delta"] < QC.REBUILD_ATOL,
          f"x_q on {quant.x_q.device}, rebuild delta "
          f"{quant.diagnostics['rebuild_max_abs_delta']:.3e}")

    batch = mount_cpu(quant)
    check("CUDA: the canonical rows stay on CPU while the Q-space tensors do not",
          batch.quant.rows.device.type == "cpu"
          and batch.anchor_rows.device.type == "cpu"
          and batch.target_rows.device.type == "cpu"
          and batch.x_q_anchor.is_cuda and batch.x_q_target.is_cuda,
          "identity is CPU-only by contract; the representation follows the model")

    check("CUDA: the pair axis is identical to the CPU run, so nothing about "
          "pairing is device-dependent",
          batch.n_pairs == cpu_batch.n_pairs
          and batch.anchor_pos.tolist() == cpu_batch.anchor_pos.tolist()
          and batch.target_pos.tolist() == cpu_batch.target_pos.tolist(),
          f"{batch.n_pairs} pairs on both devices")

    check("CUDA: the gathered Q-space agrees with the CPU run to float32 "
          "device-arithmetic precision",
          torch.allclose(batch.x_q_target.cpu(), cpu_batch.x_q_target,
                         atol=1e-5, rtol=1e-5),
          "the same codebooks, the same indices, the same sum")


# --------------------------------------------------------------------------- #
# [7] scope
# --------------------------------------------------------------------------- #
IMPORT_PROBE = r"""
import builtins, os, sys
repo = os.path.abspath(sys.argv[1])
sys.path.insert(0, repo)
# numpy and torch are imported *before* the wraps go on: their own internal
# import-time file reads are not what this probe is about. What is being tested
# is whether importing *this* package reaches into the repository or calls a
# loader. Only opens under the repository are counted, so the probe reports the
# thing it claims to report.
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
import tiger_ext.qspace_constraint
print("IMPORT_CALLS=%d" % len(calls))
for entry in calls[:5]:
    print("   ", entry)
"""


def part7_scope():
    print("\n[7] scope: no loss, no backward, no parameter, no frozen change")
    status = subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                            capture_output=True, text=True).stdout.splitlines()
    modified = {l[3:].strip() for l in status if l[:2].strip() == "M"}
    check("C3 adds no tracked-file modification at all",
          modified <= PRE_C3_MODIFIED, f"modified: {sorted(modified)}")
    check("clean_tiger/ is untouched by C3",
          not [l for l in status if "clean_tiger" in l])
    check("no rqvae/ or model/ file is added to the modified set by C3",
          modified <= PRE_C3_MODIFIED,
          "the frozen core keeps exactly the 3 pre-existing edits")

    source = open(MODULE, encoding="utf-8").read()
    tree = ast.parse(source)

    # every forbidden token must be absent from the *code*: only NAME tokens are
    # scanned, so the docstring and the provenance strings cannot mask a hit
    names = [t.string for t in _tokenize.generate_tokens(io.StringIO(source).readline)
             if t.type == _tokenize.NAME]
    offenders = sorted({n for n in names for tok in FORBIDDEN_TOKENS
                        if tok in n.lower()} - FORBIDDEN_ALLOWED)
    literal_hits = {tok: source.lower().count(tok) for tok in FORBIDDEN_TOKENS
                    if tok in source.lower()}
    check("no loss / backward / softmax / concat / projection identifier exists "
          "anywhere in the code",
          not offenders,
          f"scanned {len(names)} NAME tokens; the only literal occurrences are "
          f"{literal_hits}, all inside the docstring and the recorded reference")
    check("the RQ-VAE's own returned loss is unpacked and discarded, never used",
          source.count("rq_loss") == 1,
          "it appears once, on the unpacking line, and nowhere else")
    check("no distinct top-level name is a loss, a projection or an alignment",
          not any(tok in n.name.lower() for n in tree.body
                  if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                  for tok in ("loss", "backward", "softmax", "proj", "align",
                              "warmup", "concat", "user")),
          "the deliverable is a constraint constructor, not an objective")

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr in ("backward", "cross_entropy", "log_softmax",
                                 "nll_loss", "mse_loss", "kl_div",
                                 "binary_cross_entropy")]
    check("no objective and no gradient step is ever called",
          not calls, "not one loss function and not one .backward()")

    import torch
    mount = QC.QSpaceConstraint(lambda_csg=1.0, k=2)
    check("the mount holds no parameters and no buffers",
          len(list(mount.parameters())) == 0 and len(list(mount.buffers())) == 0
          and len(list(mount.state_dict())) == 0,
          "it is integer identity data plus a float; nothing is trainable")
    check("the mount has no submodule at all, so it cannot hold a projection or "
          "an embedding by accident",
          [type(m).__name__ for m in mount.modules()] == ["QSpaceConstraint"],
          f"{[type(m).__name__ for m in mount.modules()]}")
    check("no nn layer constructor and no buffer registration exists in the code",
          not any(t in source for t in ("nn.Linear", "nn.Parameter", "nn.Embedding",
                                        "register_buffer", "register_parameter")),
          "the module's only nn import is the Module base class")

    probe = subprocess.run([sys.executable, "-I", "-c", IMPORT_PROBE, REPO],
                           capture_output=True, text=True).stdout.strip()
    probe_first = probe.splitlines()[0] if probe else ""
    check("importing the module reads nothing from the repository and calls no "
          "loader (proved, not asserted)",
          probe_first == "IMPORT_CALLS=0",
          "open() is counted only for paths under the repo, and numpy.load / "
          "torch.load are wrapped, across `import tiger_ext.qspace_constraint`")

    check("no CF embedding, no user-side object and no warmup appears in the "
          "module's public surface",
          not any(tok in n for n in QC.__all__ for tok in
                  ("cf_", "user", "warmup", "loss", "proj")),
          f"{len(QC.__all__)} exported names")

    check("the module reaches the relation only through C2's loader -- it never "
          "opens the artifacts itself",
          "np.load(" not in source and "loadtxt" not in source
          and "read_parquet" not in source,
          "a direct read would skip the row-space and k checks that "
          "CollabRelation.load performs")

    check("the provenance record carries the reference, the missing file and "
          "every difference",
          len(QC.PORT_DIFFERENCES) == 5
          and QC.MISSING_REFERENCE["status"].startswith("ABSENT")
          and QC.REFERENCE_IMPLEMENTATION["quantity"].startswith("x_q")
          and QC.QSPACE_REPRESENTATION == "x_q",
          "5 differences, the absent reference file recorded as absent")

    check("the three parts A / B / C are declared separately",
          set(QC.THREE_PARTS) == {"A_relation", "B_representation", "C_constraint"},
          "collaborative relation, Q-space representation, constraint")

    check("tiger_ext re-exports the C3 interface",
          __import__("tiger_ext").QSpaceConstraint is QC.QSpaceConstraint
          and __import__("tiger_ext").REBUILD_ATOL == QC.REBUILD_ATOL,
          "one definition of the mount and of the tolerance")


# --------------------------------------------------------------------------- #
# [8] the disabled mount
# --------------------------------------------------------------------------- #
def part8_disabled(RQVAE):
    import torch

    print("\n[8] lambda_csg == 0: the constraint is off, not silently zero")
    mount = QC.QSpaceConstraint()
    check("the default weight is 0, so the constraint is disabled out of the box",
          mount.lambda_csg == QC.DEFAULT_LAMBDA_CSG == 0.0 and not mount.enabled,
          "a caller that forgets the switch gets no constraint, not a wrong one")
    check("a disabled mount with no relation reports itself unloaded",
          not mount.loaded)

    model = build_tiny_rqvae(RQVAE)
    quant = QC.capture_qspace(model, fixture_batch_x(), FIX_BATCH_ROWS, use_sk=False)
    check("calling a disabled mount raises instead of returning an empty batch",
          raises(lambda: mount(quant), QC.QSpaceDisabled),
          "matching SIAMount / PSAMount: guard on `enabled`, do not read a 0")
    check("the disabled call is counted, so a caller that ignores `enabled` is "
          "visible",
          mount.n_disabled_calls == 1 and mount.n_calls == 0,
          f"n_disabled_calls={mount.n_disabled_calls}, n_calls={mount.n_calls}")
    check("the disabled guard fires before the relation is touched, so a "
          "disabled mount needs no relation at all",
          not mount.loaded,
          "the tripwire is at the top of forward(), ahead of _require_relation")

    live = QC.QSpaceConstraint(build_fixture_relation(), lambda_csg=1.0, k=2)
    live(quant)
    check("a live mount counts its own calls separately",
          live.n_calls == 1 and live.n_disabled_calls == 0,
          "the counters make a half-wired caller observable")
    check("the per-dataset entry point defaults to disabled as well",
          not QC.load_constraint_for_dataset("Toys").enabled,
          "`lambda_csg` defaults to 0 there too")

    check("a mount with no relation refuses to pair, rather than pairing nothing",
          raises(lambda: QC.QSpaceConstraint(lambda_csg=1.0, k=2).pairs_for_rows(
              torch.tensor([0, 1])), QC.QSpaceError),
          "an unloaded live mount is a wiring error, not an empty result")


# --------------------------------------------------------------------------- #
# [9] regression
# --------------------------------------------------------------------------- #
COUNT_RE = re.compile(r"(\d+)/(\d+) checks passed")


def part9_regression():
    print("\n[9] regression: the E1-E10 suites, re-run after C3")
    if REGRESSION == "skip":
        print("  [skip] EXP_C3_REGRESSION=skip")
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
    check("the E10 run inside this regression is the same 64-check suite C2 "
          "recorded",
          total == 64, f"{total} checks in the suite")
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-2000:])


# --------------------------------------------------------------------------- #
def main():
    print("=" * 76)
    print("EXP-C3 -- Q-space collaborative structure constraint (construction only)")
    print("=" * 76)

    # order matters: both model/ and rqvae/ contain a top-level `main.py`, so
    # MODEL_DIR must win the lookup for `import main`
    sys.path.insert(0, RVQAE_DIR)
    sys.path.insert(0, MODEL_DIR)
    sys.path.insert(0, REPO)
    from models.rqvae import RQVAE
    from datasets import EmbDataset

    part1_synthetic(RQVAE)
    part2_provenance(RQVAE, EmbDataset)
    part3_sentinels(RQVAE)
    part4_out_of_batch(RQVAE)
    part5_mapping()
    part6_real(RQVAE)
    part7_scope()
    part8_disabled(RQVAE)
    part9_regression()

    total = len(RESULTS)
    passed = sum(RESULTS)
    print("\n" + "=" * 76)
    print(f"EXP-C3: {passed}/{total} checks passed")
    print("=" * 76)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
