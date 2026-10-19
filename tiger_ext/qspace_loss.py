"""C4 -- Q-space collaborative constraint objective (CSG4Rec -> TIGER port).

Contract: ``integration/plan.md`` C4 table (``L_CSG``, a scalar). C3
(``tiger_ext/qspace_constraint.py``) delivered the *intermediate* -- a
:class:`~tiger_ext.qspace_constraint.ConstraintBatch`. This module turns that
intermediate into a scalar and stops there: no trainer, no stage switch, no
optimiser, no backward call. Wiring it into a training loop is C5's Task.

    ConstraintBatch  ->  Q-space constraint loss  ->  scalar

What is the objective, and on what evidence
-------------------------------------------
``CSGT/models/vq.py::VectorQuantizer.diversity_loss`` (``:68-106``) is the one
Q-space constraint that survives in the CSG4Rec snapshot. Read literally:

    emb  = self.embedding.weight                      # :69  the level's OWN codebook
    y    = a codeword id (another member of the same cluster)   # :72-81
    sim  = torch.matmul(x_q, emb.t())                 # :84  raw dot product
    sim[idx, indices[idx]] = 1e12; sim -= sim_self    # :87-90  mask the anchor's own code
    loss = F.cross_entropy(sim, y)                    # :92

called **once per RQ level** (``rq.py:51-61``), averaged over levels
(``rq.py:64``) and added to the quantiser loss with the ``--beta`` weight
(``vq.py:209``). Four properties of that snippet are load-bearing here, and all
four are read off the code rather than chosen:

* **``emb`` is the level's codebook, not the anchor's or the target's vector.**
  So the candidate set is a codebook, and the class count is that codebook's
  size. This settles the plan's C4 question "query / candidate / target" from
  evidence.
* **The objective is level-wise.** ``emb`` being *this level's* codebook is only
  meaningful if ``x_q`` is that level's quantised output -- which it is
  (``vq.py:195`` is inside one ``VectorQuantizer``, whose input is the residual
  left by the previous levels, ``rq.py:57-58``). A constraint over the *summed*
  ``x_q`` has no matching ``emb`` at all, so level-wise is not a preference: it
  is what the reference's scoring form can express. See ``QSL_LEVEL_POLICY``.
* **The similarity is a bare dot product with temperature 1.** The cosine
  variant is *commented out* at ``vq.py:83`` and ``temp = 1`` is hardcoded at
  ``:70`` and applied as ``sim / temp`` at ``:91``. There is therefore no cosine
  and no temperature to port, and inventing one would be a new mechanism.
* **No weight is used.** ``diversity_loss``'s parameters are
  ``(x_q, indices, indices_cluster, indices_list)`` -- the relation's own
  similarity never reaches it. C2's cosine therefore survives C3's construction
  and is **not** consumed here; see ``QSL_WEIGHT_POLICY``.

What is *not* recoverable, and what that means for this Task
------------------------------------------------------------
``CSGT/models/rqvae.py`` is **absent from the snapshot** (recorded by C3 in
``qspace_constraint.MISSING_REFERENCE``; ``CSGT/main.py:9`` imports it, ``CSGT/models/`` holds only
``layers.py``/``rq.py``/``vq.py``, and there is no archive, ``.pyc`` or second
copy). It is the only reader of ``global_knn`` / ``use_knn`` /
``use_knn_argument`` / ``use_cf_loss`` / ``use_pairwise`` / ``use_kd`` and the
only definition of ``compute_loss``. **CSG4Rec's actual kNN loss is therefore
not reproduced here, and this module does not pretend it is.** What *is*
reproduced is the *form* of the surviving constraint above -- scoring a
quantised vector against a codebook with a code-index target -- applied to C3's
relation-driven pair axis instead of the reference's code-cluster positives.

That substitution is the one deliberate adaptation and it is the plan's own
requirement (C3 sources the constraint from "C2 的关系对"). Every remaining
choice that the reference does not force is listed in ``QSL_DESIGN_DECISIONS`` with
its reason, and every difference from the reference is in ``_PORT_DIFFERENCES``.

The gradient path, stated exactly
---------------------------------
The query here is the codebook lookup ``W_l[Ia[:, l]]`` -- the level-``l``
component of the very ``x_q`` C3 captured, obtained by the same lookup the
quantiser itself performs (``rqvae/models/vq.py:90``). Two consequences, both
measured by EXP-C4 rather than asserted:

* ``dL/dW_l != 0`` for every level that has pairs: the objective reaches the
  RQ-VAE's quantiser graph (the codebooks), on both the query and the candidate
  side, exactly as ``sim = x_q @ emb.t()`` does in the reference.
* There is **no path to the encoder**, because ``indices`` is the output of
  ``argmin``/``argmax`` (``rqvae/models/vq.py:78,86``) and carries no gradient.
  This is the reference's own routing -- it scores the *pre*-straight-through
  tensor and only rewrites ``x_q`` at ``vq.py:213``, after the loss. C3's
  ``PORT_DIFFERENCES`` item 4 and ``plan.md`` §15.9 already flag that the tensor
  C3 hands over (``forward(return_xq=True)``) has the straight-through path
  available; that path is real and EXP-C4 verifies it is intact, but the
  objective does not route through it, because the reference's does not either.
  **Nothing is detached to make this convenient** -- a source-level check
  asserts the objective calls no ``.detach()``. Whether C5 wants the constraint
  to reach the encoder as well is a C5 decision, and it cannot be made inside
  the reference's scoring form; recorded, not invented.

An empty pair axis is a data outcome, a disabled switch is a wiring error
-----------------------------------------------------------------------
Two different "nothing to do" situations, deliberately handled differently and
both by precedent in this repository:

* ``lambda_csg == 0`` -> :class:`~tiger_ext.qspace_constraint.QSpaceDisabled`,
  reusing C3's exception so "disabled" means one thing across the C line,
  matching ``SIAMount`` / ``PSAMount`` (guard on ``enabled``; never return a
  manufactured zero).
* no usable pair, or every pair dropped at every level -> ``0.0`` scalar, the
  house behaviour for a relation-driven loss with nothing to say:
  ``rqvae/trainer.py::_get_transition_loss`` (``:191-192,213-214``) and
  ``rqvae/trans_loss.py::ContrastiveLoss`` both return a zero tensor rather than
  raising. Raising here would turn an ordinary batch geometry into a crash.

Engineering gate only -- no recommendation-quality claim is made anywhere.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qspace_constraint import (DEFAULT_LAMBDA_CSG, QSpaceDisabled, QSpaceError,
                                REBUILD_ATOL, REBUILD_RTOL)

# --------------------------------------------------------------------------- #
# the pinned objective
# --------------------------------------------------------------------------- #

#: the objective, in one line, so no reader has to infer it from the code
QSL_OBJECTIVE_FORM = (
    "L_csg = mean over the levels that have pairs of  CE_l ;  "
    "CE_l = F.cross_entropy(sim_l, y_l) ;  "
    "sim_l = W_l[Ia[:, l]] @ W_l.T  with the anchor's own codeword masked,  "
    "y_l = Iy[:, l]")

#: the reference's own arithmetic, quoted so the two can be diffed by eye
QSL_REFERENCE_OBJECTIVE = {
    "file": "reference_repos/CSG4Rec_original/CSGT/models/vq.py",
    "lines": "68-106 (diversity_loss); 195 (the x_q it consumes); 198-200 (call "
             "site); 209 (beta * it, added to the quantiser loss)",
    "call_structure": "once per RQ level (CSGT/models/rq.py:51-61), averaged "
                      "over levels (rq.py:64)",
    "query": "x_q = self.embedding(indices).view(x.shape) -- the codeword the "
             "anchor itself was assigned at this level (vq.py:195)",
    "candidate": "emb = self.embedding.weight -- the level's OWN codebook "
                 "(vq.py:69), scored by sim = torch.matmul(x_q, emb.t()) (:84)",
    "target": "y_true = a codeword id (:72-81, :92) -- an int class label",
    "similarity": "raw dot product; the cosine variant is commented out at "
                  "vq.py:83",
    "temperature": "hardcoded temp = 1 (vq.py:70), applied as sim = sim / temp "
                   "(:91) -- i.e. no temperature",
    "self_masking": "sim_self[idx, indices[idx]] = 1e12; sim = sim - sim_self "
                    "(vq.py:87-90)",
    "reduction": "F.cross_entropy's default -- mean over the rows of sim",
    "weight": "none: diversity_loss's parameters are (x_q, indices, "
              "indices_cluster, indices_list). The relation's own similarity is "
              "never passed to it",
    "supervision_source": "code-cluster membership (constrained k-means over "
                          "the codebooks, vq.py:53-65, 72-79), NOT the CF kNN",
}

#: **the missing file.** Recorded here so the adaptation below cannot be
#: mistaken for a faithful reproduction of CSG4Rec's kNN loss.
QSL_MISSING_REFERENCE = {
    "file": "CSGT/models/rqvae.py",
    "status": "ABSENT from the snapshot; no archive, no .pyc, no second copy",
    "importers": ["CSGT/main.py:9 (from models.rqvae import RQVAE)",
                  "CSGT/generate_indices.py:15"],
    "what_it_held": "RQVAE, compute_loss, get_indices -- and the only reads of "
                    "global_knn, use_knn, use_knn_argument, use_cf_loss, "
                    "use_pairwise, use_kd",
    "consequence": "CSG4Rec's actual kNN loss cannot be recovered from the "
                   "reference and is NOT reproduced here. What is reproduced is "
                   "the *form* of the surviving constraint, vq.py:68-106, driven "
                   "by C2's relation instead of by code clusters.",
}

#: C2's cosine weight reaches C3 but not the objective. Saying so beats
#: inventing a place to spend it.
QSL_WEIGHT_POLICY = (
    "NOT used. C3 carries C2's cosine similarity on the pair axis, and this "
    "objective ignores it: the reference's diversity_loss is a function of "
    "(x_q, indices, indices_cluster, indices_list) only (vq.py:68) -- no "
    "weight, no scaling of the per-pair term. Using it would require a new "
    "mechanism (a weighting, or a soft target), which this Task is forbidden "
    "to add. EXP-C4 asserts the loss is bitwise invariant to the weight.")

QSL_SELF_MASK = 1e12              # vq.py:89, verbatim
QSL_SELF_MASK_RULE = (
    "sim_self[idx, indices[idx]] = 1e12; sim = sim - sim_self (vq.py:87-90): the "
    "anchor's own codeword is pushed off the softmax because it is trivially the "
    "nearest one. Kept as a large finite constant, not -inf, to stay literal")

QSL_SAME_CODE_POLICY = (
    "a pair whose target shares the anchor's codeword at this level is dropped "
    "for this level and counted. The reference's positive is defined to be a "
    "*different* codeword -- vq.py:77-78 loops 'while random_element == "
    "indices[idx]' -- so the case has no counterpart there. Keeping such a pair "
    "would ask the softmax to rank the masked self-column first, giving a loss "
    "of order QSL_SELF_MASK")

QSL_LEVEL_POLICY = (
    "level-wise, one CE per RQ level over the levels that have pairs. Forced by "
    "evidence, not chosen: the reference's `emb` is the codebook of the level "
    "whose residual quantiser produced x_q (vq.py:69,195 together with "
    "rq.py:51-61), so the class set is per-level. The summed x_q has no "
    "matching `emb`, so a constraint on the sum is not expressible in the "
    "reference's scoring form. No level-wise loss is *added* beyond this: there "
    "is one objective, averaged over levels exactly as rq.py:64 averages")

QSL_MULTI_NEIGHBOUR_POLICY = (
    "each (anchor, neighbour) pair is its own row of the CE, i.e. an anchor with "
    "n neighbours contributes n rows, and the per-level reduction is the mean "
    "over the pair axis. The reference has exactly one positive per row "
    "(vq.py:72-81), so its row axis *is* the pair axis when every anchor has one "
    "neighbour; this is that case generalised, not a multi-positive softmax "
    "(which would be a different formula the reference does not contain)")

QSL_DIRECTION_POLICY = (
    "directed, never merged: C3 emits the pairs from each anchor's own out-list, "
    "and each pair is scored with its own anchor as the query. (a -> b) and "
    "(b -> a) therefore become two independent CE rows, and neither is inferred "
    "from the other")

QSL_EMPTY_POLICY = (
    "a zero scalar, not an exception: no pair at all (C3's empty pair axis), or "
    "every pair dropped at every level, returns 0.0. Precedent in this "
    "repository for a relation-driven loss with no pairs: "
    "rqvae/trainer.py::_get_transition_loss:191-192,213-214 and "
    "rqvae/trans_loss.py::ContrastiveLoss. Guarding on `enabled` -- not on an "
    "empty pair axis -- is the caller's job, and that failure is a "
    "QSpaceDisabled, reusing C3's exception")

QSL_GRADIENT_POLICY = (
    "the objective differentiates into the RQ-VAE's codebooks, where the "
    "reference's does: dL/dW_l has both a query-side and a candidate-side term, "
    "mirroring sim = x_q @ emb.t(). It does not reach the encoder, because "
    "indices come from argmin/argmax (rqvae/models/vq.py:78,86) and carry no "
    "gradient; the reference is identical (it scores the pre-straight-through "
    "tensor, vq.py:195, and rewrites x_q only at :213, after the loss). The "
    "straight-through route from the tensor C3 captured is real and EXP-C4 "
    "verifies it end to end, but routing the objective through it is a decision "
    "the reference's scoring form does not make -- recorded for C5. No "
    "requires_grad is touched and nothing is detached")

_PORT_DIFFERENCES = [
    {"what": "the loss itself",
     "reference": "unknown -- the kNN-loss body lives in the missing "
                  "CSGT/models/rqvae.py",
     "here": "not a reproduction. The *form* of the surviving constraint "
             "(vq.py:68-106) is applied to C2's relation: codebook-scored "
             "cross-entropy with a code-index target",
     "why": "the reference file is absent; inventing a replacement would be "
            "fabrication. C3's MISSING_REFERENCE records the gap and this "
            "module repeats it"},
    {"what": "the positive",
     "reference": "another codeword in the same constrained-k-means cluster "
                  "(vq.py:72-79)",
     "here": "the codeword that C2's collaborative neighbour was assigned at "
             "this level",
     "why": "the plan sources C4 from C3's intermediate, and C3's pair axis is "
            "item-to-item. The substitution is the Task's premise, not a "
            "mechanism change: the scored quantity, the candidate set, the "
            "similarity and the target *type* all stay as the reference has them"},
    {"what": "the query's gradient routing",
     "reference": "the pre-straight-through codebook lookup (vq.py:195), so "
                  "gradient reaches the codebook only",
     "here": "the level component of x_q obtained by the same lookup "
             "(rqvae/models/vq.py:90) -- the same routing",
     "why": "the value is identical; C3 additionally hands over the "
            "post-straight-through tensor, whose encoder path is verified but "
            "unused. C3's PORT_DIFFERENCES item 4 and plan.md 15.9 flagged this "
            "for C4; C4 keeps the reference's routing and records the choice"},
    {"what": "the positive is guaranteed to differ from the anchor's codeword",
     "reference": "guaranteed by construction (vq.py:77-78)",
     "here": "not guaranteed, so pairs that collide at a level are dropped for "
             "that level and counted",
     "why": "the reference has no arithmetic for a self-target; asking the "
            "masked column to win would score ~1e12"},
    {"what": "the levels",
     "reference": "every level always contributes (a positive always exists)",
     "here": "a level with no surviving pair is omitted and the mean is taken "
             "over the remaining levels",
     "why": "a per-level CE the reference never constructs; averaging over the "
            "defined terms keeps the scale comparable to the reference's, "
            "instead of silently diluting it with manufactured zeros"},
    {"what": "the switch",
     "reference": "--use_knn / --knn_weight / --use_constrain_loss / --beta "
                  "(train_tokenizer.sh)",
     "here": "lambda_csg on this module's mount, default 0, not wired into any "
             "total loss (that wiring is C5)",
     "why": "the plan requires an independent switch and forbids the C4 Task "
            "from touching a trainer"},
]

#: the plan's own pre-code questions (section 3 of the C4 brief), answered here
#: rather than only in prose, so the frozen definition travels with the code
QSL_DESIGN_DECISIONS = {
    "1_query": "the level-l component of the anchor item's quantised vector, "
               "W_l[Ia[:, l]] -- the codeword the RQ-VAE assigned the anchor at "
               "level l, which is what vq.py:195 hands to the reference",
    "2_candidate": "the level's codebook, W_l (n_e_l codewords). Evidence: "
                   "emb = self.embedding.weight (vq.py:69)",
    "3_similarity": "raw dot product sim_l = X_l @ W_l.T, no normalisation and "
                    "no temperature. Evidence: vq.py:84, with the cosine line "
                    "commented out at :83 and temp hardcoded to 1 at :70",
    "4_y_true": "a codeword id: the code index the collaborative neighbour was "
                "assigned at that level, Iy[:, l] (C3's indices_target). Target "
                "type matches the reference's int class label (vq.py:92)",
    "5_multiple_neighbours": QSL_MULTI_NEIGHBOUR_POLICY,
    "6_directed_pairs": QSL_DIRECTION_POLICY,
    "7_c2_weight": QSL_WEIGHT_POLICY,
    "8_reduction": "mean over the pair axis within a level "
                   "(F.cross_entropy's default, as in the reference), then mean "
                   "over the levels that have pairs (the counterpart of "
                   "rq.py:64's stack().mean()). Not an anchor mean: the "
                   "reference's row axis is the anchor axis only because it has "
                   "one positive per anchor",
    "9_no_valid_pair": QSL_EMPTY_POLICY,
    "10_multi_positive": "neither multi-positive nor a softmax over positives: "
                         "n rows for n neighbours, so the per-level mean weights "
                         "every pair equally rather than every anchor equally. "
                         "That is what a mean over the reference's row axis "
                         "becomes here",
}

QSL_REFERENCE_SCALE = "log(256) = 5.5452"      # what an uninformative codebook scores


class QSpaceLossError(QSpaceError):
    """The batch and the model handed to the objective do not belong together."""


# --------------------------------------------------------------------------- #
# the objective
# --------------------------------------------------------------------------- #
def codebook_stack(rqvae):
    """The RQ-VAE's per-level codebooks as one ``(L, n_e, e_dim)`` tensor.

    Uses the model's own public accessor (``ResidualVectorQuantizer.get_codebook``
    -> ``VectorQuantizer.get_codebook``), which is ``torch.stack``ed from the
    ``nn.Embedding`` weights, so the tensor stays attached to the parameters and
    gradients flow into the quantiser.
    """
    return rqvae.rq.get_codebook()


def _checked_indices(batch):
    """C3's pair-axis indices, with the shapes frozen by that interface."""
    ia, iy = batch.indices_anchor, batch.indices_target
    if ia.dim() != 2 or iy.dim() != 2:
        raise QSpaceLossError(
            f"indices must be (P, L), got {tuple(ia.shape)} and {tuple(iy.shape)}")
    if ia.shape != iy.shape:
        raise QSpaceLossError(
            f"anchor and target indices differ in shape: {tuple(ia.shape)} vs "
            f"{tuple(iy.shape)}")
    if ia.shape[0] != int(batch.anchor_pos.numel()):
        raise QSpaceLossError(
            f"{ia.shape[0]} index rows for {int(batch.anchor_pos.numel())} pairs")
    return ia, iy


@torch.no_grad()
def verify_indices_match_qspace(rqvae, batch, *, atol=REBUILD_ATOL,
                               rtol=REBUILD_RTOL):
    """Assert C3's indices explain C3's tensors *under this model*.

    The objective rebuilds the query from the codebooks rather than from
    ``batch.x_q_anchor``, so the link between the two has to be checked or the
    objective could silently score a batch produced by a different tokenizer.
    The identity is C3's own definition -- ``x_q = sum_l codebook_l[indices_l]``
    -- recomputed here on detached copies, so nothing is detached in the
    objective's own path.

    Returns the measured deviations; raises :class:`QSpaceLossError` if either
    side fails.
    """
    codebooks = codebook_stack(rqvae)
    ia, iy = _checked_indices(batch)
    out = {}
    for side, ind, x_q in (("anchor", ia, batch.x_q_anchor),
                           ("target", iy, batch.x_q_target)):
        if ind.shape[1] != codebooks.shape[0]:
            raise QSpaceLossError(
                f"the {side} side carries {ind.shape[1]} RQ levels but the model "
                f"has {codebooks.shape[0]}; the collision token is not a "
                "quantiser level")
        if int(ind.numel()) and int(ind.max()) >= codebooks.shape[1]:
            raise QSpaceLossError(
                f"the {side} side carries code index {int(ind.max())} but the "
                f"codebooks hold {codebooks.shape[1]} codewords per level")
        if int(ind.numel()) and int(ind.min()) < 0:
            raise QSpaceLossError(f"the {side} side carries a negative code index")
        rebuilt = sum(codebooks[l][ind[:, l]] for l in range(codebooks.shape[0]))
        if int(ind.numel()) == 0:
            # nothing to compare, and nothing was claimed by C3 either: an empty
            # pair axis is a legitimate batch, not a failure of this check
            out[f"{side}_rebuild_max_abs_delta"] = 0.0
            continue
        deviation = float((x_q.detach().cpu() - rebuilt.detach().cpu()).abs().max())
        out[f"{side}_rebuild_max_abs_delta"] = deviation
        if not torch.allclose(x_q.detach().cpu(), rebuilt.detach().cpu(),
                              atol=atol, rtol=rtol):
            raise QSpaceLossError(
                f"the {side} side's x_q is not built from the model's codebooks "
                f"by its own indices (max |delta| = {deviation:.3e}); the batch "
                "and the model were not produced together")
    return out


def qspace_constraint_loss(batch, rqvae, *, validate=True):
    """``L_csg`` for one batch, plus the diagnostics of how it was assembled.

    Args:
        batch: C3's :class:`ConstraintBatch`.
        rqvae: the TIGER ``RQVAE`` the batch was captured from -- the source of
            the codebooks the objective scores against.
        validate: run :func:`verify_indices_match_qspace` first. On by default;
            it is the only thing tying the batch to this model.

    Returns:
        ``(loss, stats)``. ``loss`` is a 0-dim tensor, differentiable into
        ``rqvae``'s codebooks; ``stats`` is a plain dict of counts and detached
        per-level values.
    """
    codebooks = codebook_stack(rqvae)
    n_levels, n_e, e_dim = (int(codebooks.shape[0]), int(codebooks.shape[1]),
                            int(codebooks.shape[2]))
    if n_e < 2:
        raise QSpaceLossError(
            f"a level holds {n_e} codeword(s); with the anchor's own codeword "
            "masked there is no candidate left, and the reference's own "
            "construct cannot act either (vq.py:77-78 would not terminate)")
    ia, iy = _checked_indices(batch)
    if ia.shape[1] != n_levels:
        raise QSpaceLossError(
            f"the batch carries {ia.shape[1]} RQ levels but the model has "
            f"{n_levels}")
    if int(batch.x_q_anchor.shape[-1]) != e_dim:
        raise QSpaceLossError(
            f"x_q_anchor has dimension {int(batch.x_q_anchor.shape[-1])} but the "
            f"codebooks have e_dim = {e_dim}")

    diagnostics = verify_indices_match_qspace(rqvae, batch) if validate else {}
    stats = {
        "objective": QSL_OBJECTIVE_FORM,
        "pairs": int(batch.anchor_pos.numel()),
        "levels": n_levels,
        "n_e": n_e,
        "per_level": [],
    }
    if batch.is_empty:
        stats["reason"] = "no pairs"
        stats["levels_used"] = 0
        stats["dropped_same_code"] = 0
        stats.update(diagnostics)
        stats.update(_as_floats(stats.pop("per_level")))
        return _zero(codebooks), stats

    ia = ia.to(codebooks.device)
    iy = iy.to(codebooks.device)
    arange = None
    level_losses = []
    dropped_total = 0
    for level in range(n_levels):
        anchor_code = ia[:, level]
        target_code = iy[:, level]
        # the reference's positive is a *different* codeword (vq.py:77-78)
        keep = anchor_code != target_code
        n_dropped = int((~keep).sum())
        dropped_total += n_dropped
        entry = {"level": level, "pairs": int(anchor_code.numel()),
                 "dropped_same_code": n_dropped, "used": int(keep.sum())}
        if int(keep.sum()) == 0:
            entry["loss"] = None
            stats["per_level"].append(entry)
            continue
        anchor_code = anchor_code[keep]
        target_code = target_code[keep]
        weight_l = codebooks[level]                       # (n_e, e_dim), differentiable
        query = weight_l[anchor_code]                     # (P_l, e_dim)
        sim = torch.matmul(query, weight_l.t())           # (P_l, n_e)
        # vq.py:87-90, verbatim: push the anchor's own codeword off the softmax
        if arange is None or arange.numel() != sim.shape[0]:
            arange = torch.arange(sim.shape[0], device=sim.device)
        self_scores = torch.zeros_like(sim)
        self_scores[arange, anchor_code] = QSL_SELF_MASK
        sim = sim - self_scores
        loss_l = F.cross_entropy(sim, target_code)        # mean over the pair axis
        level_losses.append(loss_l)
        entry["loss"] = float(loss_l.detach())
        stats["per_level"].append(entry)

    if not level_losses:
        stats["reason"] = "every pair shares its anchor's codeword at every level"
        stats["levels_used"] = 0
        stats["dropped_same_code"] = dropped_total
        stats.update(diagnostics)
        stats.update(_as_floats(stats.pop("per_level")))
        return _zero(codebooks), stats

    loss = torch.stack(level_losses).mean()
    stats["levels_used"] = len(level_losses)
    stats["dropped_same_code"] = dropped_total
    stats["mean"] = float(loss.detach())
    stats["reference_scale"] = QSL_REFERENCE_SCALE
    stats.update(diagnostics)
    stats.update(_as_floats(stats.pop("per_level")))
    return loss, stats


def _zero(codebooks):
    """The ``0.0`` the empty cases return, typed like the codebooks."""
    return torch.zeros((), device=codebooks.device, dtype=codebooks.dtype)


def _as_floats(per_level):
    """Flatten the per-level list into stable dict keys."""
    out = {"per_level": per_level}
    for entry in per_level:
        out[f"level{entry['level']}_loss"] = entry["loss"]
    return out


class QSpaceLoss(nn.Module):
    """The standalone objective: ``(ConstraintBatch, RQVAE) -> scalar``.

    Holds no parameters and no buffers -- the codebooks belong to the model
    passed in, and this module only reads them. :attr:`stats` carries the
    diagnostics of the most recent call (the counterpart of
    :attr:`~tiger_ext.qspace_constraint.QSpaceBatch.diagnostics`).
    """

    def __init__(self, *, validate=True):
        super().__init__()
        self.validate = bool(validate)
        self.stats = None

    def forward(self, batch, rqvae):
        loss, stats = qspace_constraint_loss(batch, rqvae, validate=self.validate)
        self.stats = stats
        return loss

    def provenance(self):
        """Everything a reader needs to know what this objective is built on."""
        return {
            "task": "C4",
            "objective": QSL_OBJECTIVE_FORM,
            "csg4rec_reference": QSL_REFERENCE_OBJECTIVE,
            "csg4rec_missing_file": QSL_MISSING_REFERENCE,
            "design_decisions": QSL_DESIGN_DECISIONS,
            "weight_policy": QSL_WEIGHT_POLICY,
            "self_mask": QSL_SELF_MASK,
            "level_policy": QSL_LEVEL_POLICY,
            "empty_policy": QSL_EMPTY_POLICY,
            "gradient_policy": QSL_GRADIENT_POLICY,
            "differences_from_reference": list(_PORT_DIFFERENCES),
            "is_a_reproduction": False,
        }


class QSpaceLossMount(nn.Module):
    """The ``lambda_csg``-guarded wrapper, so a caller has one line to write.

    Mirrors ``SIAMount`` / ``PSAMount``: it returns ``lambda_csg * L_csg`` and
    **raises** when the switch is off, rather than returning a manufactured
    zero. It wires into no total loss, holds no relation and touches no trainer
    -- that is C5. ``lambda_csg`` fills the role of the reference's ``--beta``
    (``vq.py:209``), which is likewise the weight the constraint carries into a
    total loss.

    Args:
        lambda_csg: weight; ``> 0`` enables the objective, 0 (the default)
            leaves it disabled. Default is C3's
            :data:`~tiger_ext.qspace_constraint.DEFAULT_LAMBDA_CSG`.
        validate: forwarded to :class:`QSpaceLoss`.
    """

    def __init__(self, *, lambda_csg=DEFAULT_LAMBDA_CSG, validate=True):
        super().__init__()
        self.lambda_csg = float(lambda_csg)
        self.loss_fn = QSpaceLoss(validate=validate)
        self.n_calls = 0
        self.n_disabled_calls = 0

    @property
    def enabled(self):
        """The objective runs only when the weight is strictly positive."""
        return self.lambda_csg > 0.0

    @property
    def stats(self):
        """The diagnostics of the most recent successful call."""
        return self.loss_fn.stats

    def forward(self, batch, rqvae):
        """Return ``lambda_csg * L_csg`` (a scalar tensor).

        Raises :class:`~tiger_ext.qspace_constraint.QSpaceDisabled` if called
        while disabled -- guard on :attr:`enabled`, do not read a zero.
        """
        if not self.enabled:
            self.n_disabled_calls += 1
            raise QSpaceDisabled(
                "the Q-space constraint objective is disabled (lambda_csg == 0); "
                "the caller is expected to guard on `enabled` instead of relying "
                "on a manufactured zero")
        self.n_calls += 1
        return self.lambda_csg * self.loss_fn(batch, rqvae)


__all__ = [
    "QSpaceLoss", "QSpaceLossMount",
    "QSpaceLossError", "qspace_constraint_loss", "codebook_stack",
    "verify_indices_match_qspace",
    "QSL_OBJECTIVE_FORM", "QSL_REFERENCE_OBJECTIVE", "QSL_MISSING_REFERENCE",
    "QSL_DESIGN_DECISIONS", "QSL_WEIGHT_POLICY", "QSL_SELF_MASK", "QSL_SELF_MASK_RULE",
    "QSL_SAME_CODE_POLICY", "QSL_LEVEL_POLICY", "QSL_MULTI_NEIGHBOUR_POLICY",
    "QSL_DIRECTION_POLICY", "QSL_EMPTY_POLICY", "QSL_GRADIENT_POLICY",
    "QSL_REFERENCE_SCALE", "DEFAULT_LAMBDA_CSG",
]
