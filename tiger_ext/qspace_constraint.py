"""C3 -- Q-space collaborative structure constraint (CSG4Rec -> TIGER port).

Contract: ``integration/plan.md`` C3 table.

    | 来源机制 | CSG4Rec 的协同约束（移植目标是它作用在量化空间这一形式）              |
    | 输入     | C2 的关系对；一批 item 的量化表示                                 |
    | 输出     | 每 batch 的约束张量/标量中间量                                    |
    | 新增     | 约束构造模块：以 kNN 关系为监督，在量化空间上约束                    |
    |          | "协同近邻 → 码结构/量化表示相近"                                   |
    | 修改     | 受开关守卫的挂载点（``lambda_csg`` 默认 0）                        |

This module is the **construction half only**. It pairs C2's collaborative
relation with the RQ-VAE's quantised representation and hands the result to C4.
It computes **no loss**, takes no gradient step, adds no projection, no warmup and
no regularisation, and does not touch the quantiser's arithmetic. Nothing is
concatenated into the RQ-VAE input, and no CF-embedding-to-latent alignment is
performed anywhere -- the CF table is not even visible here (it is consumed inside
C2 to *build* the relation and is never seen again).

The three things the Task requires be kept separate
---------------------------------------------------
The plan forbids collapsing these into one "embedding alignment". They are three
distinct objects, each independently constructible and testable:

* **A -- collaborative relation.** Owned entirely by C2. This module never reads a
  ``.npy``: it goes through :meth:`tiger_ext.collab_relation.CollabRelation.load`
  with ``expected_item_ids=`` and ``expected_k=20``, so a relation belonging to a
  different row space is refused rather than silently mis-paired.
* **B -- Q-space representation.** Captured *only* from the RQ-VAE's own outputs,
  by :func:`capture_qspace`, which then **asserts** the tensor really is the
  quantised representation and not the encoder latent. See below.
* **C -- constraint.** The projection of A onto B: in-batch ``(anchor, target)``
  position pairs, the gathered Q-space tensors for each side, and the relation's
  own CF cosine carried along as a weight. That is an *intermediate*, not an
  objective.

Which Q-space tensor, and why (code basis, not preference)
----------------------------------------------------------
The plan names two candidates -- ``x_q`` and the E3 code logits. Only one is
usable, and the reason is in the code:

* **``x_q`` is chosen.** CSG4Rec's *only* surviving Q-space constraint consumes
  exactly this quantity. ``VectorQuantizer.diversity_loss``
  (``CSGT/models/vq.py:68-106``) is handed ``x_q`` -- the codebook lookup
  ``x_q = self.embedding(indices).view(x.shape)`` at ``:195``, *before* the
  straight-through rewrite at ``:213`` -- and scores it against every codeword
  with ``sim = torch.matmul(x_q, emb.t())`` (``:84``). The quantity is
  post-quantisation and it comes from the codebook. TIGER has the handle:
  ``RQVAE.forward(..., return_xq=True)`` (``rqvae/models/rqvae.py:61-79``).
* **the E3 code logits are excluded.** They are not a Q-space quantity at all:
  ``rqvae/models/vq.py`` computes them from ``latent`` -- the *encoder output*,
  pre-quantisation -- as ``raw_d / logit_temperature`` (``:76``, ``:107``).
  Constraining them would constrain the encoder's output vector, i.e. it would be
  precisely the "CF embedding <-> RQ-VAE latent alignment" that the plan's C3
  section forbids. Recording this is the point: the two candidates look similar
  and are not.

The per-level code assignment ``indices`` is carried alongside ``x_q`` (not
instead of it). CSG4Rec's constraint targets a *code index* -- ``y_true`` is a
codeword id, fed to ``F.cross_entropy`` -- so a consumer that wants the code-side
"码结构" half of the plan's sentence needs them, and they are free: the same
forward pass returns them.

What CSG4Rec actually does, and what is missing from the snapshot
-----------------------------------------------------------------
**This is the honest part of the report, and it is not a footnote.**

Recoverable from the snapshot, and reproduced here: the CF neighbours are built
over the CF table with cosine kNN ``knn_k=20`` (``CSGT/datasets.py:36-38``,
already ported as C2); they travel through the batch dict kept on CPU
(``CSGT/trainer.py:155,192``); they are consumed by
``self.model.compute_loss(batch_items, epoch_idx, epochs, labels)``
(``CSGT/trainer.py:159``); the objective is weighted by ``--knn_weight``, whose
value in the released recipe is **0.01** (``CSGT/train_tokenizer.sh``); and the
per-epoch printout has a dedicated ``knn`` slot (``CSGT/trainer.py:241``).

**Not recoverable.** ``CSGT/models/rqvae.py`` -- the file that defines
``RQVAE``, ``compute_loss`` and ``get_indices``, and the only place
``global_knn``, ``use_knn``, ``use_knn_argument``, ``use_cf_loss``, ``use_pairwise``
and ``use_kd`` are ever read -- **is absent from the snapshot**. ``CSGT/main.py:9``
imports it (``from models.rqvae import RQVAE``) and ``CSGT/models/`` contains only
``layers.py``, ``rq.py`` and ``vq.py``. There is no archive, no ``.pyc`` and no
second copy anywhere on this machine. **The exact functional form of CSG4Rec's
kNN loss therefore cannot be read off the reference**, and nothing in this module
pretends otherwise.

What that means for this Task, stated plainly:

* C3 does **not** invent a loss form to fill the gap. It is scoped by the plan to
  produce the *intermediate*, and the objective is C4's Task. The gap does not
  block C3, because C3 is not asked to reproduce the missing formula.
* Everything C3 does decide is anchored on code that *is* present -- the
  post-quantisation quantity and its scoring form from
  ``CSGT/models/vq.py::diversity_loss``, and the in-batch pairing policy from this
  repository's own existing relation-to-Q-space construct (below).
* The two adaptations that are this module's own choice are labelled as such in
  ``PORT_DIFFERENCES`` and repeated in the Task report.

The in-batch policy (an adaptation, and why it is the minimal one)
------------------------------------------------------------------
A batch is a sample of items, so most of a row's 20 neighbours are not in it. The
policy here is **in-batch pairs only**: a pair survives iff both endpoints are in
the batch, and everything else is dropped and *counted*. This is not a free
invention -- ``rqvae/trainer.py::_get_transition_loss:213-219`` (the repository's
existing relation-to-Q-space construct) does exactly this, dropping a target with
``if target_idx in idx_to_pos``. It is also the only policy that needs no extra
information: no second forward pass, no neighbour-side embedding table, no extra
state. CSG4Rec's own policy is unknown (missing file) and is recorded as unknown
rather than guessed.

Because the relation is **directed** (C2: ``i -> j`` does not imply ``j -> i``,
reciprocal fraction 0.61-0.69), the pairs are generated from each anchor's own
out-list and are not symmetrised. A consumer that wants both directions must ask
for them; assuming symmetry here would silently double-count.

Sentinels (C2's contract, honoured rather than assumed)
-------------------------------------------------------
C2 marks "no neighbour" with ``neighbor == -1`` and ``similarity = NaN``, and the
two always agree. Both are filtered before anything reaches a tensor operation:
a ``-1`` would otherwise index the *last* row of the batch, and a ``NaN`` would
otherwise poison a downstream mean. The counts of each are reported. The guard is
written to fail loudly on the one thing C2 guarantees cannot happen -- a ``-1``
that does not carry a ``NaN``, or vice versa -- because that would mean the
relation object was tampered with or built elsewhere.

Engineering gate only -- nothing here says anything about recommendation quality.
"""
import os

import numpy as np
import torch
import torch.nn as nn

from .cf_embed import canonical_item_ids
from .collab_relation import (CollabRelation, CollabRelationMismatch,
                              RELATION_K, relation_prefix)

# --------------------------------------------------------------------------- #
# pinned definition of the constraint interface
# --------------------------------------------------------------------------- #

#: which Q-space tensor the constraint is built on. Justified above: it is the
#: quantity CSG4Rec's surviving constraint consumes, and it is post-quantisation.
QSPACE_REPRESENTATION = "x_q"

#: the candidate the plan also named, and why it is *not* used. The reason is
#: structural (it is computed from the encoder output), not a preference.
QSPACE_EXCLUDED = (
    "E3 code logits (``rqvae/models/vq.py:107``): computed from ``latent``, the "
    "encoder output *before* quantisation, so constraining them constrains the "
    "encoder vector -- i.e. it is the latent-space alignment the plan's C3 "
    "section forbids, wearing a Q-space name")

QSPACE_SOURCE = ("the RQ-VAE's own forward pass with ``return_xq=True`` "
                 "(``rqvae/models/rqvae.py:61-79``); the value is the sum over RQ "
                 "levels of the codebook lookups, carried by straight-through")

#: the assertion that makes "Q-space" a measured property rather than a label
QSPACE_PROVENANCE_CHECK = (
    "the captured tensor is rebuilt from the model's own codebooks, "
    "``sum_l codebook_l[indices_l]``, and must agree with the returned x_q to "
    "float32 straight-through precision; the encoder latent ``z`` from the same "
    "forward pass is checked to be *different* from it")

#: A, B, C kept as three objects -- the plan's explicit requirement
THREE_PARTS = {
    "A_relation": "C2's ``CollabRelation``, loaded via ``CollabRelation.load("
                  "prefix, expected_item_ids=..., expected_k=20)``; no ``.npy`` "
                  "is read directly by this module",
    "B_representation": "``QSpaceBatch`` -- x_q (+ per-level code indices), from "
                        "the RQ-VAE's own output only",
    "C_constraint": "``ConstraintBatch`` -- A projected onto B: in-batch "
                    "``(anchor, target)`` position pairs, the gathered Q-space "
                    "tensors per side, and the relation's own cosine as a weight",
}

BATCH_POLICY = (
    "in-batch pairs only: a pair survives iff both endpoints are in the batch; "
    "the rest are dropped and counted. Precedent in this repository: "
    "``rqvae/trainer.py::_get_transition_loss:213-219``. CSG4Rec's own policy is "
    "unrecoverable -- see the module docstring")

SENTINEL_POLICY = (
    "C2's sentinels are filtered before any tensor operation: ``neighbor == -1`` "
    "and non-finite ``similarity``; the two must agree, and a disagreement raises")

DIRECTION_POLICY = (
    "directed, not symmetrised: pairs come from each anchor's own out-list, "
    "matching C2 (reciprocal fraction 0.61-0.69). Symmetrising would silently "
    "double-count")

REFERENCE_IMPLEMENTATION = {
    "file": "reference_repos/CSG4Rec_original/CSGT/models/vq.py",
    "lines": "68-106,195,198-200",
    "quantity": "x_q = the codebook lookup, before straight-through",
    "scoring": "sim = torch.matmul(x_q, emb.t())  # dot product with every codeword",
    "objective_form": "F.cross_entropy(sim, y_true) -- y_true is a *codeword id*",
    "levels": "per RQ level; averaged by rq.py:64, weighted by --beta",
    "relation_source_in_reference": "code-cluster membership (constrained k-means "
                                    "over the codebooks), NOT the CF kNN",
}

#: **the missing file.** Recorded here so no reader of this port can mistake the
#: adaptation below for a faithful reproduction.
MISSING_REFERENCE = {
    "file": "CSGT/models/rqvae.py",
    "status": "ABSENT from the snapshot; no archive, no .pyc, no second copy",
    "importers": ["CSGT/main.py:9 (from models.rqvae import RQVAE)",
                  "CSGT/generate_indices.py:15"],
    "what_it_held": "RQVAE, compute_loss, get_indices -- and the only reads of "
                    "global_knn, use_knn, use_knn_argument, use_cf_loss, "
                    "use_pairwise, use_kd",
    "consequence": "the exact functional form of CSG4Rec's kNN loss cannot be "
                   "recovered from the reference and is NOT reproduced here",
}

PORT_DIFFERENCES = [
    {"what": "the constraint's supervision",
     "reference": "code-to-code: the positive is another codeword in the same "
                  "constrained-k-means cluster of the codebook (vq.py:72-79)",
     "here": "item-to-item: the positive is a C2 collaborative neighbour of the "
             "anchor item",
     "why": "the plan's C3 sources the constraint from 'C2 的关系对'; the "
            "collaborative relation is the Task's input. The reference's "
            "cluster-of-codes relation is a different object and is not used."},
    {"what": "the loss",
     "reference": "computed inline, ``beta * diversity_loss`` inside the "
                  "quantiser's own ``forward`` (vq.py:198-209)",
     "here": "not computed at all -- C3 emits the intermediate, C4 owns the "
             "objective",
     "why": "the plan splits them into two Tasks; computing a loss here would "
            "be doing C4's work and would also be the one place the missing "
            "reference file would have to be guessed"},
    {"what": "out-of-batch neighbours",
     "reference": "unknown -- the consuming code is in the missing file",
     "here": "dropped, and counted",
     "why": "the only policy needing no extra information; precedent in this "
            "repository at rqvae/trainer.py::_get_transition_loss:213-219"},
    {"what": "which quantity is constrained",
     "reference": "x_q, the pre-straight-through codebook lookup (vq.py:195)",
     "here": "x_q as returned by ``forward(return_xq=True)``, whose *value* is "
             "the same sum, with the straight-through gradient path recorded "
             "for C4",
     "why": "TIGER exposes the post-straight-through tensor; its value equals "
            "the codebook sum, so the constraint sees the same quantity. The "
            "gradient routing differs and C4 must know that."},
    {"what": "the switch",
     "reference": "--use_knn / --knn_weight (knn_weight=0.01 in the released "
                  "recipe)",
     "here": "``lambda_csg``, default 0, on this module's own mount",
     "why": "the plan requires an independent switch; the historical "
            "``trans_graph_path`` semantics are explicitly not reused"},
]

DEFAULT_LAMBDA_CSG = 0.0

#: float32 straight-through arithmetic means ``x + (y - x)`` is not exactly
#: ``y``; the provenance rebuild is therefore compared within this tolerance
#: rather than bitwise. Measured headroom is reported by EXP-C3.
REBUILD_ATOL = 1e-4
REBUILD_RTOL = 1e-4


class QSpaceError(Exception):
    """Base class for C3 failures."""


class QSpaceMismatch(QSpaceError):
    """A tensor or relation is not what it claims to be."""


class QSpaceDisabled(QSpaceError):
    """Called while ``lambda_csg == 0``; the caller must guard on ``enabled``."""


# --------------------------------------------------------------------------- #
# B -- the Q-space representation
# --------------------------------------------------------------------------- #
def _codebooks_of(rqvae):
    """The per-level codebook weight matrices, in RQ level order."""
    return [layer.embedding.weight for layer in rqvae.rq.vq_layers]


def rebuild_quantized(rqvae, indices):
    """``sum_l codebook_l[indices_l]`` -- the quantised vector, from the model.

    This is the definition the provenance check compares against. Note that
    ``rqvae/models/rqvae.py``'s docstring warns against re-gathering in order to
    *backpropagate* (that would cut the encoder's gradient path); here it is used
    for verification only, on a detached copy.
    """
    total = None
    for level, weight in enumerate(_codebooks_of(rqvae)):
        block = weight[indices[:, level]]
        total = block if total is None else total + block
    return total


@torch.no_grad()
def capture_qspace(rqvae, x, rows, *, use_sk=True):
    """Run the RQ-VAE and capture its Q-space output for a batch.

    Args:
        rqvae: TIGER ``RQVAE``.
        x: ``(B, in_dim)`` item content embeddings for this batch.
        rows: ``(B,)`` canonical row indices -- the batch's identity in C1/C2's
            shared row space. Kept on CPU; it is identity, not data.
        use_sk: forwarded to the quantiser (whether Sinkhorn assigns codes).

    Returns:
        :class:`QSpaceBatch`, already validated against the model it came from.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    out, rq_loss, indices, z, x_q = rqvae(x, use_sk=use_sk,
                                          return_z=True, return_xq=True)
    batch = QSpaceBatch(x_q=x_q, indices=indices, rows=rows, latent=z)
    batch.diagnostics = batch.validate(rqvae)
    return batch


class QSpaceBatch:
    """One batch's quantised representation, with its provenance checked.

    Construct through :func:`capture_qspace`; constructing it by hand is allowed
    (the tests do it, to check that a tampered batch is rejected) but it is only
    trustworthy once :meth:`validate` has passed.
    """

    def __init__(self, x_q, indices, rows, *, latent=None):
        self.x_q = x_q
        self.indices = indices
        self.latent = latent          # kept only so validate() can prove x_q != z
        self.rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1).cpu()
        self.diagnostics = None       # set by capture_qspace / validate()

    @property
    def n_items(self):
        return int(self.x_q.shape[0])

    @property
    def n_levels(self):
        return int(self.indices.shape[1])

    @property
    def dim(self):
        return int(self.x_q.shape[1])

    def validate(self, rqvae):
        """Assert the tensor really is the quantised representation.

        This is plan.md C3 acceptance ① made mechanical. It checks, in order:

        1. ``indices`` is ``(B, L)`` with ``L == len(num_emb_list)`` -- the
           collision token is not a quantiser level and must never appear;
        2. ``rows`` matches the batch size;
        3. ``x_q`` equals ``sum_l codebook_l[indices_l]`` computed from *this*
           model's codebooks, within float32 straight-through tolerance -- a
           tensor that is not the codebook sum (e.g. the encoder latent, or
           anything else a caller might hand over) fails here;
        4. ``x_q`` is *not* the encoder latent from the same forward pass;
           otherwise "Q-space" would be a label with no content.

        Returns a dict of measured diagnostics rather than a bare ``True``, so a
        caller can record how far apart the two tensors actually were.
        """
        if self.indices.dim() != 2:
            raise QSpaceMismatch(
                f"indices must be (B, L), got shape {tuple(self.indices.shape)}")
        n_levels = len(rqvae.rq.vq_layers)
        if self.n_levels != n_levels:
            raise QSpaceMismatch(
                f"batch carries {self.n_levels} RQ levels but the model has "
                f"{n_levels}; the collision token is not a quantiser level")
        if self.x_q.shape[0] != self.indices.shape[0]:
            raise QSpaceMismatch(
                f"x_q has {self.x_q.shape[0]} rows but indices has "
                f"{self.indices.shape[0]}")
        if self.rows.numel() != self.x_q.shape[0]:
            raise QSpaceMismatch(
                f"rows has {self.rows.numel()} entries for a batch of "
                f"{self.x_q.shape[0]}")

        rebuilt = rebuild_quantized(rqvae, self.indices.detach())
        deviation = float((self.x_q.detach().cpu() - rebuilt.detach().cpu())
                          .abs().max())
        if not torch.allclose(self.x_q.detach().cpu(), rebuilt.detach().cpu(),
                              atol=REBUILD_ATOL, rtol=REBUILD_RTOL):
            raise QSpaceMismatch(
                "the captured tensor is not the quantised representation: it "
                f"does not equal sum_l codebook_l[indices_l] (max |delta| "
                f"= {deviation:.3e}). The encoder latent, or any other tensor, "
                "fails this check on purpose.")

        latent_gap = None
        if self.latent is not None:
            latent_gap = float((self.x_q.detach().cpu()
                                - self.latent.detach().cpu()).abs().max())
            if latent_gap <= REBUILD_ATOL:
                raise QSpaceMismatch(
                    "the captured tensor is indistinguishable from the encoder "
                    f"latent (max |delta| = {latent_gap:.3e}); a Q-space "
                    "constraint on it would be a latent-space constraint")
        return {"rebuild_max_abs_delta": deviation, "latent_max_abs_delta": latent_gap}

    def __repr__(self):
        return (f"QSpaceBatch(B={self.n_items}, dim={self.dim}, "
                f"levels={self.n_levels})")


# --------------------------------------------------------------------------- #
# C -- the constraint: A projected onto B
# --------------------------------------------------------------------------- #
class ConstraintBatch:
    """The per-batch constraint intermediate handed to C4.

    Everything is aligned along the pair axis ``P``: ``anchor_pos`` and
    ``target_pos`` are positions **within the batch**, ``anchor_rows`` and
    ``target_rows`` are the canonical rows those positions hold, and the
    Q-space tensors are gathered at those positions. The pair axis may be empty
    (``P == 0``); callers should check :attr:`is_empty` rather than assuming.
    """

    def __init__(self, quant, anchor_pos, target_pos, weight,
                 x_q_anchor, x_q_target, indices_anchor, indices_target, stats):
        self.quant = quant
        self.anchor_pos = anchor_pos
        self.target_pos = target_pos
        self.weight = weight
        self.x_q_anchor = x_q_anchor
        self.x_q_target = x_q_target
        self.indices_anchor = indices_anchor
        self.indices_target = indices_target
        self.stats = dict(stats)

    @property
    def n_pairs(self):
        return int(self.anchor_pos.numel())

    @property
    def is_empty(self):
        return self.n_pairs == 0

    @property
    def anchor_rows(self):
        return self.quant.rows[self.anchor_pos]

    @property
    def target_rows(self):
        return self.quant.rows[self.target_pos]

    def __repr__(self):
        return (f"ConstraintBatch(pairs={self.n_pairs}, "
                f"of {self.quant.n_items} items, "
                f"dropped_out_of_batch={self.stats.get('dropped_out_of_batch')}, "
                f"dropped_sentinel={self.stats.get('dropped_sentinel')})")


class QSpaceConstraint(nn.Module):
    """The ``lambda_csg``-guarded mount that builds :class:`ConstraintBatch`.

    Registered as an ``nn.Module`` for consistency with the other C-line mounts,
    but it holds **no parameters and no buffers**: it is integer identity data
    (which row's neighbour is which row) plus a weight. Nothing about it is
    trainable, and it is not part of any model's ``parameters()``.

    ``lambda_csg`` defaults to 0, in which case :attr:`enabled` is False, the
    relation is never read and the quantiser is never touched -- the caller is
    expected to guard on :attr:`enabled`, and :meth:`forward` raises rather than
    quietly returning a zero, matching ``SIAMount`` / ``PSAMount``.

    Args:
        relation: C2's :class:`CollabRelation`, or None for a disabled mount.
        lambda_csg: weight; ``> 0`` enables the constraint path.
        k: the ``k`` the relation must have been built with (default
            :data:`tiger_ext.collab_relation.RELATION_K`, i.e. 20).
    """

    def __init__(self, relation=None, *, lambda_csg=DEFAULT_LAMBDA_CSG, k=RELATION_K):
        super().__init__()
        self.lambda_csg = float(lambda_csg)
        self.k = int(k)
        self.relation = relation
        if relation is not None and relation.k != self.k:
            raise QSpaceMismatch(
                f"constraint pinned to k={self.k} but the relation was built "
                f"with k={relation.k}")
        self.n_calls = 0
        self.n_disabled_calls = 0

    # -- the guard ---------------------------------------------------------- #
    @property
    def enabled(self):
        """The constraint runs only when the weight is strictly positive."""
        return self.lambda_csg > 0.0

    @property
    def loaded(self):
        """Whether a relation is actually in hand (independent of ``enabled``)."""
        return self.relation is not None

    # -- A, read through C2's interface only -------------------------------- #
    def load_relation(self, prefix, *, expected_item_ids, expected_k=RELATION_K):
        """Load C2's relation through ``CollabRelation.load``, with checks.

        The ``.npy`` files are never opened here: going through the interface is
        what makes the row-space check possible, and a relation that belongs to a
        different row space raises instead of mis-pairing items.
        """
        self.relation = CollabRelation.load(
            prefix, expected_item_ids=expected_item_ids,
            expected_k=expected_k)
        self.k = int(expected_k)
        return self.relation

    def _require_relation(self):
        if self.relation is None:
            raise QSpaceError(
                "no relation loaded; call load_relation(prefix, "
                "expected_item_ids=..., expected_k=20) first")
        return self.relation

    # -- C, the projection of A onto B -------------------------------------- #
    def pairs_for_rows(self, rows):
        """In-batch ``(anchor_pos, target_pos, weight)`` plus the drop counts.

        Split out from :meth:`forward` so the pairing can be tested on its own --
        it needs no model and no Q-space tensor, which is what keeps A separable
        from B.
        """
        relation = self._require_relation()
        rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1).cpu()
        B = int(rows.numel())
        n_items = relation.n_items
        if B and (int(rows.min()) < 0 or int(rows.max()) >= n_items):
            raise QSpaceMismatch(
                f"batch rows span [{int(rows.min())}, {int(rows.max())}] but the "
                f"relation covers [0, {n_items - 1}]; the batch is not in this "
                "relation's row space")

        neighbours = torch.from_numpy(
            np.asarray(relation.neighbors[rows.numpy()], dtype=np.int64))
        similarity = torch.from_numpy(
            np.asarray(relation.similarity[rows.numpy()], dtype=np.float32))

        # C2 guarantees -1 <=> non-finite; a disagreement means someone built a
        # relation this module was not told about
        sentinel = neighbours < 0
        not_finite = ~torch.isfinite(similarity)
        if not torch.equal(sentinel, not_finite):
            raise QSpaceMismatch(
                "the relation's -1 sentinel and its non-finite similarities "
                "disagree; this relation did not come from C2 unchanged")

        # position of every row inside this batch
        pos_of = torch.full((n_items,), -1, dtype=torch.long)
        pos_of[rows] = torch.arange(B, dtype=torch.long)
        safe = neighbours.clamp(min=0)
        target_pos = pos_of[safe]

        in_batch = (~sentinel) & (target_pos >= 0)
        dropped_sentinel = int(sentinel.sum())
        dropped_out_of_batch = int(((~sentinel) & (target_pos < 0)).sum())

        anchor_pos, slot = torch.nonzero(in_batch, as_tuple=True)
        stats = {
            "batch_size": B,
            "k": int(neighbours.shape[1]),
            "slots": int(neighbours.numel()),
            "pairs": int(anchor_pos.numel()),
            "dropped_sentinel": dropped_sentinel,
            "dropped_out_of_batch": dropped_out_of_batch,
            "anchors_with_pair": int(torch.unique(anchor_pos).numel())
                                if anchor_pos.numel() else 0,
        }
        return anchor_pos, target_pos[anchor_pos, slot], \
            similarity[anchor_pos, slot], stats

    def forward(self, quant):
        """Build the constraint intermediate for one batch.

        Raises if called while disabled -- guard on :attr:`enabled`.
        """
        if not self.enabled:
            self.n_disabled_calls += 1
            raise QSpaceDisabled(
                "the Q-space constraint is disabled (lambda_csg == 0); the "
                "caller is expected to guard on `enabled` instead of relying on "
                "an empty return value")
        self.n_calls += 1
        relation = self._require_relation()
        if quant.rows.numel() and int(quant.rows.max()) >= relation.n_items:
            raise QSpaceMismatch(
                f"the batch's rows reach {int(quant.rows.max())} but the "
                f"relation covers [0, {relation.n_items - 1}]")

        anchor_pos, target_pos, weight, stats = self.pairs_for_rows(quant.rows)
        stats["representation"] = QSPACE_REPRESENTATION
        stats["lambda_csg"] = self.lambda_csg
        return ConstraintBatch(
            quant=quant,
            anchor_pos=anchor_pos,
            target_pos=target_pos,
            weight=weight,
            x_q_anchor=quant.x_q[anchor_pos],
            x_q_target=quant.x_q[target_pos],
            indices_anchor=quant.indices[anchor_pos],
            indices_target=quant.indices[target_pos],
            stats=stats,
        )

    def provenance(self):
        """Everything a reader needs to know what this constraint is built on."""
        relation = self.relation
        return {
            "task": "C3",
            "method": "Q-space collaborative structure constraint (interface only)",
            "encodes_no_loss": True,
            "representation": QSPACE_REPRESENTATION,
            "representation_excluded": QSPACE_EXCLUDED,
            "representation_source": QSPACE_SOURCE,
            "provenance_check": QSPACE_PROVENANCE_CHECK,
            "three_parts": THREE_PARTS,
            "batch_policy": BATCH_POLICY,
            "sentinel_policy": SENTINEL_POLICY,
            "direction_policy": DIRECTION_POLICY,
            "csg4rec_reference": REFERENCE_IMPLEMENTATION,
            "csg4rec_missing_file": MISSING_REFERENCE,
            "differences_from_reference": list(PORT_DIFFERENCES),
            "lambda_csg": self.lambda_csg,
            "enabled": self.enabled,
            "k": self.k,
            "relation": None if relation is None else {
                "n_items": relation.n_items,
                "k": relation.k,
                "k_eff": relation.k_eff,
                "n_edges": relation.n_edges,
                "item_ids_sha256": relation.item_ids_sha256,
                "cf_source": relation.source,
            },
        }


# --------------------------------------------------------------------------- #
# convenience entry points
# --------------------------------------------------------------------------- #
def load_constraint_for_dataset(dataset, *, lambda_csg=DEFAULT_LAMBDA_CSG,
                                k=RELATION_K, data_dir="data",
                                artifact_dir=None):
    """Build the mount for a dataset, loading C2's relation through its interface.

    ``expected_item_ids`` comes from C1's canonical list, so the two row spaces
    are pinned to the same definition rather than compared by eye.
    """
    if artifact_dir is None:
        prefix = relation_prefix(dataset, k=k)
    else:
        prefix = relation_prefix(dataset, k=k, artifact_dir=artifact_dir)
    mount = QSpaceConstraint(lambda_csg=lambda_csg, k=k)
    mount.load_relation(prefix,
                        expected_item_ids=canonical_item_ids(dataset,
                                                             data_dir=data_dir),
                        expected_k=k)
    return mount


def rows_of_item_ids(relation, item_ids):
    """Canonical rows for a list of ItemIDs, via C2's mapping -- never ``id - 1``."""
    return np.asarray([relation.row_of(int(i)) for i in np.asarray(item_ids).reshape(-1)],
                      dtype=np.int64)


__all__ = [
    "QSpaceConstraint", "QSpaceBatch", "ConstraintBatch",
    "QSpaceError", "QSpaceMismatch", "QSpaceDisabled",
    "capture_qspace", "rebuild_quantized", "load_constraint_for_dataset",
    "rows_of_item_ids",
    "QSPACE_REPRESENTATION", "QSPACE_EXCLUDED", "QSPACE_SOURCE",
    "QSPACE_PROVENANCE_CHECK", "THREE_PARTS", "BATCH_POLICY", "SENTINEL_POLICY",
    "DIRECTION_POLICY", "REFERENCE_IMPLEMENTATION", "MISSING_REFERENCE",
    "PORT_DIFFERENCES", "DEFAULT_LAMBDA_CSG", "REBUILD_ATOL", "REBUILD_RTOL",
]
