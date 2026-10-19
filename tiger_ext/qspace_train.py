"""C5 -- Q-space constraint training integration (CSG4Rec -> TIGER port).

Contract: ``integration/plan.md`` C5 table.

    L_total = L_RQVAE + lambda_csg * L_csg

C3 built the constraint, C4 built the objective. This module is the wiring: it
is the **new entry point and the switch group** the plan asks for, and it is the
only thing that puts ``L_csg`` into a ``backward()``. It does not redefine
``L_csg`` (C4 owns that), does not re-implement a training step (the frozen
``rqvae/trainer.py`` owns that), and adds no warm-up, no regularisation, no
optimiser and no new parameter.

What is new here, exactly
-------------------------
* :class:`CSGHook` -- a parameter-free ``nn.Module`` holding C3's
  :class:`~tiger_ext.qspace_constraint.QSpaceConstraint` and C4's
  :class:`~tiger_ext.qspace_loss.QSpaceLossMount`, both pinned to the same
  ``lambda_csg``. ``forward(x_q, indices, latent, rows, rqvae)`` builds C3's
  ``QSpaceBatch`` **from the training step's own forward pass**, validates its
  provenance, and returns ``lambda_csg * L_csg``.
* :func:`train_tokenizer` -- assembles the frozen pieces (``EmbDataset`` with
  ``return_index=True``, ``RQVAE``, the frozen ``Trainer``) with the hook
  attached as ``csg=``, and calls ``Trainer.fit``. It re-implements nothing.
* :func:`main` -- the CLI entry, independent of ``rqvae/main.py``.

The one mount point, and why it is safe
---------------------------------------
``rqvae/trainer.py::_train_epoch`` already carried three optional weighted terms
(``L_trans``, ``L_balance``, ``L_align``), each in the same three-line shape. C5
adds a fourth in that same shape and nothing else. ``lambda_csg`` defaults to
**0**, so out of the box the hook is never called, no codebook is read and the
historical path is bitwise unchanged -- the same global gate (G0) every other
mechanism in this port obeys.

The gradient route, audited rather than assumed
-----------------------------------------------
Three routes were measured before one was adopted (see ``CST_GRADIENT_ROUTES``):

* **P1, codebook only -- adopted.** The objective's query is the codebook lookup
  ``W_l[Ia[:, l]]`` and its candidate is the whole ``W_l``, so ``dL/dW_l != 0``
  while every encoder parameter keeps ``.grad is None``. This is the reference's
  own routing: ``CSGT/models/vq.py`` scores the *pre*-straight-through tensor
  (``:195``) and rewrites ``x_q`` only afterwards (``:213``), strictly *after*
  the loss.
* **P2, straight-through into the encoder -- measured, not adopted.** The tensor
  the hook receives (``forward(return_xq=True)``) does carry the encoder path --
  ``L_align`` already uses it -- and the harness measures that it is intact.
  Routing ``L_csg`` through it was rejected because it cannot be done inside the
  reference's scoring form: the objective's query is defined as the codebook
  lookup, so either the query is swapped for the straight-through tensor (which
  changes ``L_csg``, C4's frozen deliverable) or the per-level straight-through
  values are exposed from ``rqvae/models/rq.py`` / ``vq.py`` (which edits the
  frozen RQ-VAE core). Both are out of bounds for C5, so it is recorded as
  ``CST_PENDING_DECISIONS`` item 1 rather than silently chosen.
* **P3, the baseline -- measured.** ``lambda_csg=0`` means ``enabled`` is False,
  the trainer's guard short-circuits, ``n_calls == 0`` and the switch-off run is
  bitwise equal to the pristine trainer.

The switch contract is C4's, unchanged
--------------------------------------
``CSGHook.enabled`` is ``lambda_csg > 0``. The trainer guards on it, so a
disabled hook is never called. If it *is* called while disabled the C3/C4
``QSpaceDisabled`` is raised -- a manufactured zero is never returned. An empty
pair axis is a different situation and stays C4's ``0.0`` with a reason.

Engineering gate only -- nothing here claims a recommendation improvement, and
no final-recommendation experiment is run.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn as nn

from .alt_pipeline import (
    NUM_WORKERS, PHASE_WARMUP_EPOCHS, RQVAE_CONFIG, rqvae_namespace,
)
from .collab_relation import RELATION_K
from .qspace_constraint import (
    DEFAULT_LAMBDA_CSG, QSpaceBatch, QSpaceConstraint, QSpaceDisabled, QSpaceError,
    load_constraint_for_dataset,
)
from .qspace_loss import QSpaceLossMount

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVQAE_DIR = os.path.join(REPO, "rqvae")

# --------------------------------------------------------------------------- #
# the pinned integration contract
# --------------------------------------------------------------------------- #

#: the total loss, in one line, so no reader has to infer it from the code
CST_TOTAL_LOSS_FORM = (
    "L_total = L_RQVAE + lambda_csg * L_csg   where   "
    "L_RQVAE = L_recon + quant_loss_weight * rq_loss (unchanged, "
    "rqvae/models/rqvae.py:87-97) and L_csg is C4's frozen objective")

#: one line per route, with the measurement that settled it
CST_GRADIENT_ROUTES = {
    "P1_codebook_only": {
        "status": "ADOPTED",
        "what": "the trainer passes the step's own x_q / indices / z to the hook; "
                "the hook scores C4's objective, whose query is the codebook "
                "lookup W_l[Ia[:, l]] and whose candidate set is the whole W_l",
        "measured": "sum |dL_csg/dW_l| > 0 on every level that has pairs; "
                    "p.grad is None for every encoder parameter",
        "why": "the reference routes the same way: CSGT/models/vq.py:195 scores "
               "the pre-straight-through tensor and :213 rewrites x_q after it",
    },
    "P2_straight_through_to_encoder": {
        "status": "MEASURED, NOT ADOPTED",
        "what": "x_q as returned by forward(return_xq=True) carries the "
                "straight-through path back into the encoder (the route "
                "L_align uses)",
        "measured": "sum(x_q).backward() reaches encoder parameters -- the path "
                    "is intact on the tensor the hook receives",
        "why": "not routed through: the objective's query is *defined* as the "
               "codebook lookup, so using the straight-through tensor would "
               "change L_csg (C4's frozen deliverable), and exposing the "
               "per-level straight-through values would edit the frozen RQ-VAE "
               "core. Recorded in CST_PENDING_DECISIONS, not chosen here",
    },
    "P3_baseline_disabled": {
        "status": "ADOPTED (default)",
        "what": "lambda_csg == 0",
        "measured": "the hook's n_calls stays 0, no codebook is read, and the "
                    "run is bitwise equal to the pristine trainer",
        "why": "the global gate: every mechanism in this port defaults to off",
    },
}

CST_GRADIENT_POLICY = (
    "codebook-only by default, because that is the reference's own arithmetic. "
    "The straight-through route to the encoder exists on the tensor the hook "
    "receives and is measured by EXP-C5, but adopting it would require changing "
    "L_csg or the frozen RQ-VAE core, so it is recorded as a pending decision")

#: weight candidates. `1e-4` is not an arbitrary guess: CSGT/train_tokenizer.sh
#: publishes `--beta 0.0001` for `vq.py:209`'s `beta * diversity_loss` -- the
#: same-shaped term, whose value is of the order log(n_e) = 5.54, which is what
#: C4 measured for L_csg. 1e-3 and 1e-2 bracket it from above.
CST_LAMBDA_CANDIDATES = (1e-4, 1e-3, 1e-2)

CST_LAMBDA_POLICY = (
    "the candidates are scale-matched, not copied: the reference's own weight "
    "for the same-shaped term is beta = 1e-4 (CSGT/train_tokenizer.sh:8), and "
    "L_csg sits at the same scale as the reference's diversity_loss (log n_e). "
    "C5 reports lambda * E[L_csg] / E[L_RQVAE] per candidate instead of "
    "declaring a winner; picking the operating point is deferred")

CST_REFERENCE_WEIGHT = {
    "reference": "reference_repos/CSG4Rec_original/CSGT/train_tokenizer.sh",
    "line": "--alpha 0.01 --use_constrain_loss true --beta 0.0001 --use_knn true "
            "--knn_weight 0.01",
    "which_term": "beta weights diversity_loss (CSGT/models/vq.py:209), the "
                  "surviving constraint C4 adapted",
    "recovered": 1e-4,
    "note": "CSG4Rec's own total-loss formula is NOT recoverable -- "
            "CSGT/models/rqvae.py, the only definition of compute_loss, is "
            "absent from the snapshot. C5 therefore uses the plan's "
            "L_RQVAE + lambda_csg * L_csg and does not claim to reproduce the "
            "reference's total",
}

CST_SWITCH_CONTRACT = (
    "lambda_csg default 0 (C3's DEFAULT_LAMBDA_CSG); enabled == lambda_csg > 0. "
    "The trainer guards on `enabled`, so a disabled hook is never invoked. If a "
    "disabled hook is invoked anyway it raises C3/C4's QSpaceDisabled rather "
    "than returning a manufactured zero. An empty pair axis is a different "
    "situation and stays C4's 0.0 scalar with a reason")

CST_STAGE_NOTE = (
    "no E7 change is needed and none was made: L_csg is a term of L_RQVAE, and "
    "E7's STAGE_CONTRACT already restricts L_RQVAE to the Tokenizer stage. The "
    "Recommender stage inherits the restriction automatically. "
    "alt_pipeline.tokenizer_phase still constructs the trainer without a csg "
    "argument, so E9/E10 are unaffected")

CST_ENTRY_NOTE = (
    "this module is the new driver the plan asks for; rqvae/main.py is not "
    "modified and neither is any other existing entry point")

CST_PENDING_DECISIONS = [
    {"item": "routing L_csg through the straight-through path into the encoder",
     "why not here": "it cannot be done inside C4's frozen scoring form. The "
                     "objective's query is defined as the codebook lookup, so "
                     "substituting the straight-through tensor changes L_csg, "
                     "and exposing the per-level straight-through values means "
                     "editing rqvae/models/rq.py or vq.py. Both are outside "
                     "C5's authorized file set",
     "state": "measured and recorded, not adopted"},
    {"item": "the lambda_csg operating point for a recommendation experiment",
     "why not here": "C5 is an integration sanity check; the brief forbids "
                     "running the final recommendation experiment",
     "state": "the harness reports relative contributions; nothing is chosen"},
    {"item": "CSG4Rec's exact total-loss formula and its `knn` term",
     "why not here": "CSGT/models/rqvae.py is absent from the snapshot (and "
                     "from CSG4Rec-main.zip), so the reference's compute_loss "
                     "cannot be read off",
     "state": "C5 uses the plan's L_RQVAE + lambda_csg * L_csg and says so"},
    {"item": "the Q-space constraint on the Recommender stage",
     "why not here": "not expressible: L_csg is a term of L_RQVAE, which E7's "
                     "contract already confines to the Tokenizer stage",
     "state": "no E7 change made"},
    {"item": "the batch size C3's Q-space provenance check requires",
     "why not here": "the guard rejects a batch whose x_q is indistinguishable "
                     "from the encoder latent (max |x_q - z| <= REBUILD_ATOL = "
                     "1e-4). With the frozen recipe's kmeans_init=True the first "
                     "batch hits exactly that case when it holds no more distinct "
                     "points than the codebook has codewords: k-means++ then seeds "
                     "one centre per point, every RQ residual is reproduced exactly "
                     "and the quantised sum lands back on the latent. Measured by "
                     "EXP-C5 section [11]: 256 rows -> max |x_q - z| = 0.0e+00 "
                     "(refused), 1024 rows -> 4.2e-02 (fine). Section [6] (c0) "
                     "exercises the same degeneracy end to end -- a run whose whole "
                     "dataset is one item is refused at its first step -- and [6] "
                     "(c) turns k-means initialisation off for that one run so the "
                     "empty-pair path can be reached. C5 runs its sanity check at "
                     "the frozen batch size instead of loosening the guard, which "
                     "would mean editing C3",
     "state": "measured and recorded; C3's guard and the frozen recipe are "
              "unchanged"},
    {"item": "the five pre-C5 harnesses' tracked-file guard",
     "why not here": "C5's authorized mount is a fourth modified tracked file, "
                     "so each older harness reports exactly one guard failure",
     "state": "registered in integration/status.md as an authorized C5 "
              "modification; no harness's judgement logic was edited"},
]

CST_DEFAULT_SEED = 2024          # the frozen tokenizer seed (alt_pipeline.SEED_RQVAE)
CST_SANITY_SUBSET = 2048         # rows used by the integration sanity check

#: B7 -- what "seeded" means here, and what it does not claim
CST_SEED_CONTRACT = (
    "config.seed is applied *before* the dataset, the model and the first "
    "k-means step: python `random`, numpy, torch CPU and torch CUDA are seeded "
    "exactly as rqvae/main.py seeds them (cuDNN deterministic, benchmark off); "
    "the DataLoader's shuffle order comes from its own seeded generator; and "
    "the frozen kmeans() in rqvae/models/layers.py -- which never passes "
    "random_state -- is handed an explicit per-call random_state drawn from "
    "RandomState(seed). Two CPU runs with the same seed therefore start from "
    "the same weights, the same k-means centres and the same batch order. CUDA "
    "runs are seeded too but are not claimed bitwise reproducible")

CST_CUDA_NONDETERMINISM = (
    "remains possible with a fixed seed: CUDA embedding-backward atomics "
    "(index_add/gather reduction order), cuDNN kernel selection on some ops, "
    "multi-stream/multi-GPU reduction order, and DataLoader worker arrival order "
    "when num_workers > 0 (the project default is 1). "
    "torch.use_deterministic_algorithms is deliberately NOT enabled -- it can "
    "abort the frozen loop on an operation the recipe relies on, and B7 does "
    "not change the recipe")


class QSpaceTrainError(QSpaceError):
    """The integration was asked for something it must not do silently."""


# --------------------------------------------------------------------------- #
# the frozen RQ-VAE side, imported on demand
# --------------------------------------------------------------------------- #
_MODULES = {}


def frozen_tokenizer_modules() -> dict:
    """Import the frozen RQ-VAE side only -- never the T5 side.

    ``rqvae/datasets.py`` shadows the HuggingFace ``datasets`` package, so
    ``rqvae/`` must win the lookup and the resolved module is checked to really
    live there.
    """
    if _MODULES:
        return _MODULES
    for path in (RVQAE_DIR, REPO):
        if path not in sys.path:
            sys.path.insert(0, path)
    from datasets import EmbDataset
    from models.rqvae import RQVAE
    from trainer import Trainer as RQVAETrainer

    if os.path.dirname(os.path.abspath(sys.modules["datasets"].__file__)) != RVQAE_DIR:
        raise ImportError(
            "`datasets` resolved to "
            f"{sys.modules['datasets'].__file__!r}, not rqvae/datasets.py")
    _MODULES.update(EmbDataset=EmbDataset, RQVAE=RQVAE, RQVAETrainer=RQVAETrainer)
    return _MODULES


# --------------------------------------------------------------------------- #
# B7 -- make the seed actually control the random processes
# --------------------------------------------------------------------------- #
def apply_seed(seed):
    """Seed python, numpy, torch CPU and torch CUDA as ``rqvae/main.py`` does.

    The baseline entry point seeds in exactly this order and also pins cuDNN
    (``deterministic=True``, ``benchmark=False``); B7 mirrors it so "seed=2024"
    means the same thing here as it does in the frozen tokenizer run.

    Returns the record that is written into the run report. Calling it again
    with the same seed resets every stream, so a run never inherits random
    state from a previous run in the same process.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "seed": seed,
        "python_random": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda": True,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_initial_seed": int(torch.initial_seed()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


@contextlib.contextmanager
def seeded_kmeans(seed):
    """Hand every frozen ``KMeans`` call an explicit ``random_state``.

    ``rqvae/models/layers.py:86`` builds ``KMeans(n_clusters=..., max_iter=...)``
    with no ``random_state``, so k-means++ initialisation consumes numpy's
    *global* RNG. Seeding numpy is necessary but not sufficient evidence of
    control: the stream is shared with every other numpy consumer between the
    seed and the first training step, and there is one k-means call per RQ
    level, repeated on each initialisation -- so any extra draw shifts every
    later centre.

    This wrapper gives each call a ``random_state`` drawn from a private
    ``RandomState(seed)``: the choice is explicit, reproducible and independent
    of whatever else touched the global stream. The frozen file is not edited
    -- the module-level ``KMeans`` name in ``rqvae/models/layers.py`` is
    swapped for the duration of the run and restored in ``finally``.

    Yields:
        The list of ``random_state`` values actually passed, in call order.
    """
    frozen_tokenizer_modules()
    layers = sys.modules.get("models.layers")
    if layers is None:                                       # pragma: no cover
        import models.layers as _layers
        layers = _layers
    original = layers.KMeans
    stream = np.random.RandomState(int(seed))
    used = []

    def factory(*args, **kwargs):
        if kwargs.get("random_state") is None:
            kwargs["random_state"] = int(stream.randint(0, np.iinfo(np.int32).max))
        used.append(int(kwargs["random_state"]))
        return original(*args, **kwargs)

    layers.KMeans = factory
    try:
        yield used
    finally:
        layers.KMeans = original


# --------------------------------------------------------------------------- #
# the hook
# --------------------------------------------------------------------------- #
class CSGHook(nn.Module):
    """``(x_q, indices, latent, rows, rqvae) -> lambda_csg * L_csg``.

    Parameter-free and buffer-free on purpose: it owns no representation of its
    own, so it must never enter a model's ``parameters()`` and the trainer's
    optimiser/clip sets stay exactly what they were.

    Args:
        constraint: C3's :class:`~tiger_ext.qspace_constraint.QSpaceConstraint`,
            or None for a disabled hook. Its ``lambda_csg`` must equal this
            hook's -- two different weights for one switch would be a wiring bug.
        lambda_csg: the weight; default 0 (disabled).
        validate: forwarded to C4's :class:`QSpaceLossMount`.
    """

    def __init__(self, constraint=None, *, lambda_csg=DEFAULT_LAMBDA_CSG,
                 validate=True):
        super().__init__()
        self.lambda_csg = float(lambda_csg)
        if constraint is None:
            constraint = QSpaceConstraint(lambda_csg=self.lambda_csg)
        if float(constraint.lambda_csg) != self.lambda_csg:
            raise QSpaceTrainError(
                f"the hook is pinned to lambda_csg={self.lambda_csg} but the "
                f"constraint carries {constraint.lambda_csg}; one switch, one "
                "value")
        self.constraint = constraint
        self.mount = QSpaceLossMount(lambda_csg=self.lambda_csg, validate=validate)

        # counters -- the switch-off proof and the training-sanity record
        self.n_calls = 0            # times the hook actually ran
        self.n_disabled_calls = 0   # times it was entered while disabled (a bug)
        self.n_zero = 0             # ran, but C4's empty policy returned 0.0
        self.n_nonempty = 0         # ran and produced a positive objective
        self.n_levels_total = 0
        self.n_pairs_total = 0
        #: one record per call: the unweighted objective, the weighted term the
        #: trainer added, and C4's counts. This is the trace the harness checks
        #: ``L_total - L_RQVAE == lambda_csg * L_csg`` against.
        self.history = []
        #: C3's provenance measurement for the most recent call.
        self.last_diagnostics = {}

    # -- the switch --------------------------------------------------------- #
    @property
    def enabled(self):
        """The objective runs only when the weight is strictly positive."""
        return self.lambda_csg > 0.0

    @property
    def loaded(self):
        """Whether a relation is actually in hand (independent of ``enabled``)."""
        return self.constraint.loaded

    @property
    def stats(self):
        """C4's diagnostics for the most recent call."""
        return self.mount.stats

    @property
    def counters(self):
        return {"n_calls": self.n_calls, "n_disabled_calls": self.n_disabled_calls,
                "n_zero": self.n_zero, "n_nonempty": self.n_nonempty,
                "n_levels_total": self.n_levels_total,
                "n_pairs_total": self.n_pairs_total,
                "constraint_calls": self.constraint.n_calls,
                "mount_calls": self.mount.n_calls,
                "lambda_csg": self.lambda_csg, "enabled": self.enabled,
                "loaded": self.loaded}

    # -- construction ------------------------------------------------------- #
    @classmethod
    def for_dataset(cls, dataset, *, lambda_csg=DEFAULT_LAMBDA_CSG, k=RELATION_K,
                    data_dir="data", artifact_dir=None, validate=True,
                    relation_prefix=None):
        """Build the hook for a dataset, loading C2's relation through its interface."""
        if relation_prefix is None:
            constraint = load_constraint_for_dataset(
                dataset, lambda_csg=lambda_csg, k=k, data_dir=data_dir,
                artifact_dir=artifact_dir)
        else:
            constraint = QSpaceConstraint(lambda_csg=lambda_csg, k=k)
            from .cf_embed import canonical_item_ids
            constraint.load_relation(
                relation_prefix,
                expected_item_ids=canonical_item_ids(dataset, data_dir=data_dir),
                expected_k=k)
        return cls(constraint, lambda_csg=lambda_csg, validate=validate)

    # -- the objective, on the training step's own tensors ------------------ #
    def forward(self, x_q=None, indices=None, latent=None, rows=None, rqvae=None):
        """Return ``lambda_csg * L_csg`` for one training step.

        The Q-space batch is built from the **training step's own** forward
        output. C3's :func:`~tiger_ext.qspace_constraint.capture_qspace` is
        ``@torch.no_grad()`` and exists for verification; it is deliberately not
        used here, because a no-grad capture would sever the very graph the
        objective has to differentiate through.
        """
        if not self.enabled:
            self.n_disabled_calls += 1
            raise QSpaceDisabled(
                "the Q-space constraint hook is disabled (lambda_csg == 0); the "
                "caller is expected to guard on `enabled`")
        for name, value in (("x_q", x_q), ("indices", indices), ("rows", rows),
                            ("rqvae", rqvae)):
            if value is None:
                raise QSpaceTrainError(
                    f"the hook was given {name}=None; the training step must "
                    "forward with return_xq=True and the dataset must return "
                    "row indices")
        if latent is None:
            raise QSpaceTrainError(
                "the hook was given latent=None; without z the Q-space "
                "provenance check (x_q != z) cannot run")

        quant = QSpaceBatch(x_q=x_q, indices=indices, rows=rows, latent=latent)
        # C3's provenance check, run on the training step's own tensors. Kept
        # (not just asserted) so the training record says *how far* x_q sat from
        # the encoder latent and from the codebook rebuild on every step.
        quant.diagnostics = quant.validate(rqvae)
        self.last_diagnostics = dict(quant.diagnostics or {})

        self.n_calls += 1
        batch = self.constraint(quant)
        term = self.mount(batch, rqvae)
        stats = self.mount.stats
        self.n_levels_total += int(stats.get("levels_used", 0))
        self.n_pairs_total += int(stats.get("pairs", 0))
        unweighted = float(term.detach()) / self.lambda_csg
        if float(term.detach()) == 0.0:
            self.n_zero += 1
        else:
            self.n_nonempty += 1
        self.history.append({
            "call": self.n_calls, "loss": unweighted,
            "weighted": float(term.detach()), "pairs": int(stats.get("pairs", 0)),
            "levels_used": int(stats.get("levels_used", 0)),
            "reason": stats.get("reason"), "requires_grad": bool(term.requires_grad),
            "rebuild_max_abs_delta": self.last_diagnostics.get(
                "rebuild_max_abs_delta"),
            "latent_max_abs_delta": self.last_diagnostics.get(
                "latent_max_abs_delta"),
        })
        return term

    def provenance(self):
        """Everything a reader needs to know how this integration is wired."""
        return {
            "task": "C5",
            "total_loss": CST_TOTAL_LOSS_FORM,
            "gradient_routes": {k: dict(v) for k, v in CST_GRADIENT_ROUTES.items()},
            "gradient_policy": CST_GRADIENT_POLICY,
            "lambda_candidates": list(CST_LAMBDA_CANDIDATES),
            "lambda_policy": CST_LAMBDA_POLICY,
            "reference_weight": dict(CST_REFERENCE_WEIGHT),
            "switch_contract": CST_SWITCH_CONTRACT,
            "stage_note": CST_STAGE_NOTE,
            "entry_note": CST_ENTRY_NOTE,
            "pending_decisions": [dict(d) for d in CST_PENDING_DECISIONS],
            "mount_point": "rqvae/trainer.py::_train_epoch, one guarded term in "
                           "the same shape as L_trans / L_balance / L_align",
            "constraint": self.constraint.provenance(),
            "objective": self.mount.loss_fn.provenance(),
            "counters": self.counters,
        }


# --------------------------------------------------------------------------- #
# the new entry point
# --------------------------------------------------------------------------- #
@dataclass
class CSGTrainerConfig:
    """Everything :func:`train_tokenizer` needs. ``lambda_csg`` defaults to 0."""

    dataset: str = "Toys"
    data_root: str = os.path.join(REPO, "data")
    artifact_dir: str = None
    relation_prefix: str = None
    lambda_csg: float = DEFAULT_LAMBDA_CSG
    k: int = RELATION_K
    validate: bool = True
    device: str = "cpu"
    seed: int = CST_DEFAULT_SEED
    epochs: int = 3
    eval_step: int = 1
    batch_size: int = 256
    num_workers: int = NUM_WORKERS
    warmup_epochs: int = PHASE_WARMUP_EPOCHS
    ckpt_dir: str = None
    limit: int = None
    rqvae_config: dict = field(default_factory=dict)
    #: explicit embedding table. ``None`` keeps the frozen layout
    #: (``data_root/<dataset>/item_emb.parquet``) so every pre-existing caller
    #: is unaffected; an explicit value is used **verbatim** -- there is no
    #: fallback to another table if it is missing (see ``validate_embedding_path``).
    embedding_path: str = None

    def resolved_embedding_path(self):
        """The embedding table this run will actually read (never a fallback)."""
        if self.embedding_path is not None:
            return os.path.abspath(self.embedding_path)
        return os.path.abspath(
            os.path.join(self.data_root, self.dataset, "item_emb.parquet"))

    def validate_embedding_path(self):
        """Fail loudly on a missing table instead of silently reading another one."""
        path = self.resolved_embedding_path()
        if not os.path.isfile(path):
            source = ("--embedding-path" if self.embedding_path is not None
                      else "the frozen default layout data_root/<dataset>/"
                           "item_emb.parquet")
            raise QSpaceTrainError(
                f"the embedding table is missing: {path} (from {source}); "
                "refusing to fall back to a different embedding file")
        return path

    def namespace(self, ckpt_dir=None):
        """The ``args`` the frozen ``Trainer`` reads (frozen Clean TIGER config)."""
        return rqvae_namespace(
            self.resolved_embedding_path(),
            ckpt_dir or self.ckpt_dir or os.path.join(REPO, "ckpt", "c5_sanity"),
            self.device, self.seed, epochs=self.epochs, eval_step=self.eval_step,
            batch_size=self.batch_size, num_workers=self.num_workers,
            config=dict(self.rqvae_config), warmup_epochs=self.warmup_epochs)

    def report(self):
        return {"dataset": self.dataset, "lambda_csg": float(self.lambda_csg),
                "k": int(self.k), "validate": bool(self.validate),
                "device": self.device, "seed": int(self.seed),
                "epochs": int(self.epochs), "batch_size": int(self.batch_size),
                "warmup_epochs": int(self.warmup_epochs),
                "limit": self.limit,
                "embedding_path": self.resolved_embedding_path(),
                "embedding_path_source": ("explicit" if self.embedding_path is not None
                                          else "default")}


def build_model(config, in_dim, *, namespace=None):
    """The frozen ``RQVAE``, built from the same ``args`` the Trainer will read."""
    mods = frozen_tokenizer_modules()
    args = namespace if namespace is not None else config.namespace()
    return mods["RQVAE"](
        in_dim=int(in_dim), num_emb_list=args.num_emb_list, e_dim=args.e_dim,
        layers=args.layers, dropout_prob=args.dropout_prob, bn=args.bn,
        loss_type=args.loss_type, quant_loss_weight=args.quant_loss_weight,
        beta=args.beta, kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters, sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters)


def build_dataset(config, *, hook=None, dataset_factory=None):
    """The frozen ``EmbDataset``, with row indices iff the hook needs them."""
    mods = frozen_tokenizer_modules()
    return_index = bool(hook is not None and hook.enabled)
    source = mods["EmbDataset"]
    path = config.resolved_embedding_path()
    data = (dataset_factory(mods, path, return_index)
            if dataset_factory is not None
            else source(path, return_index=return_index))
    if hook is not None and hook.enabled and not getattr(data, "return_index",
                                                         return_index):
        # An enabled hook over a dataset that carries no row indices would make
        # the trainer's `batch_indices is not None` guard drop the term
        # **silently**. That is a wiring error, not a data outcome.
        raise QSpaceTrainError(
            "the hook is enabled but the dataset does not return row indices, "
            "so the constraint would be silently skipped")
    return data


def train_tokenizer(config, *, hook=None, trainer_factory=None, dataset_factory=None,
                    loader_factory=None):
    """Run the frozen tokenizer loop with the Q-space constraint attached.

    Nothing about the loop is re-implemented: the frozen
    ``rqvae/trainer.py::Trainer`` does the stepping, and the only difference
    from a baseline run is the ``csg=`` argument.

    Args:
        config: :class:`CSGTrainerConfig`.
        hook: optional pre-built :class:`CSGHook` (the caller keeps its
            counters). Built from ``config`` when omitted.
        trainer_factory: ``(args, model, data_num, csg) -> Trainer``. Defaults to
            the frozen class; the harness passes a recording subclass so it can
            read the per-step loss sequence without forking the loop.
        dataset_factory: ``(mods, path, return_index) -> Dataset``.
        loader_factory: ``(dataset, config) -> DataLoader``.

    Returns:
        A report dict: the config, the hook counters, the frozen ``fit``'s
        losses and the checkpoint it wrote.
    """
    config.validate_embedding_path()
    # B7: applied first, before the dataset, the model, the loader and -- most
    # importantly -- the lazy k-means initialisation on the first training step.
    seeding = apply_seed(config.seed)
    if hook is None:
        hook = CSGHook.for_dataset(
            config.dataset, lambda_csg=config.lambda_csg, k=config.k,
            data_dir=config.data_root, artifact_dir=config.artifact_dir,
            validate=config.validate, relation_prefix=config.relation_prefix)

    data = build_dataset(config, hook=hook, dataset_factory=dataset_factory)
    if config.limit is not None:
        data = torch.utils.data.Subset(data, list(range(int(config.limit))))

    if loader_factory is not None:
        # A caller-supplied loader owns its own shuffle order; the report says
        # so instead of claiming a generator that was never used.
        loader = loader_factory(data, config)
        seeding["shuffle_generator_seed"] = None
        seeding["shuffle_source"] = "caller-supplied loader_factory"
    else:
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(int(config.seed))
        loader = torch.utils.data.DataLoader(
            data, num_workers=config.num_workers, batch_size=config.batch_size,
            shuffle=True, pin_memory=False, generator=shuffle_generator)
        seeding["shuffle_generator_seed"] = int(config.seed)
        seeding["shuffle_source"] = "torch.Generator seeded with config.seed"

    in_dim = int(getattr(data, "dim", None)
                 or (data.dataset.dim if isinstance(data, torch.utils.data.Subset)
                     else 0))
    ckpt_dir = config.ckpt_dir or os.path.join(REPO, "ckpt", "c5_sanity")
    args = config.namespace(ckpt_dir)

    # The frozen kmeans() runs lazily on the first training forward, not at
    # model construction, so the explicit random_state wrapper has to span the
    # whole fit, not just build_model().
    with seeded_kmeans(config.seed) as kmeans_random_states:
        model = build_model(config, in_dim, namespace=args)

        factory = trainer_factory or frozen_tokenizer_modules()["RQVAETrainer"]
        trainer = factory(args, model, len(loader), hook)

        internal = {id(p) for g in trainer.optimizer.param_groups
                    for p in g["params"]}
        module = {id(p) for p in model.parameters()}
        if internal != module:
            raise QSpaceTrainError(
                "the hook leaked into the optimiser; it must be parameter-free")

        best_loss, best_collision = trainer.fit(loader)

    seeding["kmeans_random_state_calls"] = list(kmeans_random_states)
    seeding["kmeans_random_state_source"] = (
        "numpy RandomState(config.seed), one explicit random_state per frozen "
        "KMeans call (rqvae/models/layers.py is not edited)")
    # layers.kmeans() subsamples with a hard-coded RandomState(0) when the
    # batch holds fewer rows than the codebook has codewords. That is frozen
    # behaviour and is *not* seed-controlled; recorded rather than hidden.
    seeding["kmeans_subsample_random_state"] = 0
    seeding["cuda_nondeterminism"] = CST_CUDA_NONDETERMINISM

    report = dict(config.report())
    report.update(
        hook=hook.counters,
        n_steps=int(len(loader)) * int(config.epochs),
        n_batches=int(len(loader)),
        n_rows=len(data),
        in_dim=in_dim,
        best_loss=float(best_loss),
        best_collision_rate=float(best_collision),
        ckpt_dir=os.path.abspath(trainer.ckpt_dir),
        optimizer_params=len(internal),
        hook_in_optimizer=False,
        seeding=seeding,
    )
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="python -I -m tiger_ext.qspace_train",
        description="C5 -- train the frozen RQ-VAE with the Q-space constraint "
                    "attached. lambda_csg defaults to 0 (off).")
    p.add_argument("--dataset", default="Toys")
    p.add_argument("--data-root", default=os.path.join(REPO, "data"))
    p.add_argument("--artifact-dir", default=None)
    p.add_argument("--relation-prefix", default=None)
    p.add_argument("--lambda-csg", type=float, default=DEFAULT_LAMBDA_CSG,
                   help="weight of L_csg; 0 (default) disables it entirely")
    p.add_argument("--k", type=int, default=RELATION_K)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=CST_DEFAULT_SEED,
                   help="seed applied to python/numpy/torch/cuda, the DataLoader "
                        "shuffle generator and every frozen k-means call before "
                        "training starts (B7)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--eval-step", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    p.add_argument("--warmup-epochs", type=int, default=PHASE_WARMUP_EPOCHS,
                   help="LR-schedule warm-up epochs handed to the frozen Trainer "
                        "(default 0; the frozen 3000-epoch recipe uses 50)")
    p.add_argument("--embedding-path", default=None,
                   help="explicit embedding table (e.g. the 768d content-only "
                        "parquet). Omitted = the frozen layout "
                        "data_root/<dataset>/item_emb.parquet. An explicit path "
                        "that does not exist is an error, never a fallback")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ckpt-dir", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = CSGTrainerConfig(
        dataset=args.dataset, data_root=args.data_root,
        artifact_dir=args.artifact_dir, relation_prefix=args.relation_prefix,
        lambda_csg=args.lambda_csg, k=args.k, validate=not args.no_validate,
        device=args.device, seed=args.seed, epochs=args.epochs,
        eval_step=args.eval_step, batch_size=args.batch_size,
        num_workers=args.num_workers, warmup_epochs=args.warmup_epochs,
        embedding_path=args.embedding_path,
        limit=args.limit, ckpt_dir=args.ckpt_dir)
    report = train_tokenizer(config)
    print("=" * 72)
    print("C5 -- Q-space constraint training integration")
    print(CST_TOTAL_LOSS_FORM)
    print("=" * 72)
    for key in ("dataset", "lambda_csg", "device", "seed", "epochs",
                "batch_size", "warmup_epochs", "embedding_path",
                "embedding_path_source", "limit", "n_steps", "best_loss",
                "best_collision_rate", "ckpt_dir"):
        print(f"  {key:20s} {report[key]}")
    print(f"  {'hook':20s} {report['hook']}")
    print(f"  {'seeding':20s} seed={report['seeding']['seed']} "
          f"torch_initial_seed={report['seeding']['torch_initial_seed']} "
          f"shuffle_generator_seed={report['seeding']['shuffle_generator_seed']} "
          f"kmeans_random_state_calls="
          f"{report['seeding']['kmeans_random_state_calls']}")
    return 0


__all__ = [
    "CSGHook", "CSGTrainerConfig", "QSpaceTrainError",
    "train_tokenizer", "build_model", "frozen_tokenizer_modules",
    "build_parser", "main",
    "apply_seed", "seeded_kmeans",
    "CST_TOTAL_LOSS_FORM", "CST_GRADIENT_ROUTES", "CST_GRADIENT_POLICY",
    "CST_LAMBDA_CANDIDATES", "CST_LAMBDA_POLICY", "CST_REFERENCE_WEIGHT",
    "CST_SWITCH_CONTRACT", "CST_STAGE_NOTE", "CST_ENTRY_NOTE",
    "CST_PENDING_DECISIONS", "CST_DEFAULT_SEED", "CST_SANITY_SUBSET",
    "CST_SEED_CONTRACT", "CST_CUDA_NONDETERMINISM",
]


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
