"""Stage freeze interface -- Task E7 of the ETEGRec -> TIGER port.

Implements the *phase split* of ETEGRec's Alternating Optimization: which parts
of the stack are trainable and which losses are live, per stage. It does NOT
implement the alternation itself (that is E9) nor the SID refresh (E8) -- this
module only answers "given the stage name, what is frozen and what is active?"

    stage "tokenizer"    : RQ-VAE trainable, T5 frozen, adapters frozen
                           L_RQ-VAE active; L_T5 / L_SIA / L_PSA inactive
    stage "recommender"  : RQ-VAE frozen, T5 trainable, adapters trainable
                           L_T5 / L_SIA / L_PSA active; L_RQ-VAE inactive

Source of the contract
----------------------
``integration/plan.md`` §1.4 and the E7 table (identical), which is the frozen
E7 implementation contract. The stage split is *not* invented here.

ETEGRec's own scheduler (``trainer.py::train``, ``trainer.py:513-547``) selects
the phase with ``epoch_idx % self.cycle == 0`` -> tokenizer, else recommender,
and per phase:

    tokenizer phase : model_rec.requires_grad = False (every parameter)
                      model_id.requires_grad  = True  (every parameter)
    recommender phase: model_rec -> True except `semantic_embedding`
                      model_id.parameters() -> False

That freeze/unfreeze split is what this module reproduces. Two ETEGRec details
are deliberately **not** reproduced, both already recorded as out of scope:

  * ``warm_epoch`` -- ETEGRec zeroes the id-phase extra losses for the first
    ``warm_epoch`` epochs (``config: warm_epoch: 10``). No warm-up is ported.
  * ETEGRec also computes ``kl_loss`` / ``dec_cl_loss`` in the *tokenizer*
    phase (weights ``id_kl_loss`` / ``id_dec_cl_loss``, both non-zero in
    ``config/scientific.yaml``), where they regularise ``model_id`` through the
    frozen recommender's latents. ``plan.md`` §1.4 explicitly puts ``L_SIA`` /
    ``L_PSA`` in the recommender stage only, and the port's SIA/PSA sequence
    sides are T5-side tensors, so the tokenizer stage here runs the RQ-VAE loop
    alone. This is a plan-level scoping decision, not an oversight; it is
    reported in the E7 revision record.
  * The ``semantic_embedding`` carve-out has no counterpart: that table is on
    plan.md §0.2's "not ported" list, so the recommender is unfrozen whole.

Frozen-but-used vs. frozen-and-idle
-----------------------------------
"Frozen" here means exactly ``requires_grad = False`` -- the parameter is still
read by the forward passes. In the recommender stage the RQ-VAE is frozen but
*is* executed: SIA's target side runs ``RQVAE.encoder`` and PSA's target side
runs the whole ``RQVAE.forward``. Freezing must therefore not be confused with
``eval()``; this module never changes a module's training mode.

Why freezing alone is not enough for the codebook
-------------------------------------------------
``VectorQuantizer.forward`` lazily re-initialises its codebook whenever
``not self.initted and self.training`` (``rqvae/models/vq.py:67-68``) by
calling ``init_emb``, which does a raw ``weight.data.copy_(centers)``. That
write ignores ``requires_grad`` and never goes through an optimizer, so a
"frozen" codebook could still be overwritten by k-means on the first batch that
reaches the quantizer. ``apply("recommender")`` therefore also *seals* the
codebooks (see :func:`seal_codebooks`), which is the whole point of the
``vq.initted`` handling in E7.

Nothing in this module is imported by ``model/`` or ``rqvae/``: constructing a
:class:`StageFreeze` has no effect until :meth:`StageFreeze.apply` is called,
so leaving E7 unused is bit-for-bit Clean TIGER.
"""
import json
import logging
import os
import time

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Stage names and the frozen contract (plan.md §1.4 / E7)
# --------------------------------------------------------------------------- #
STAGE_TOKENIZER = "tokenizer"
STAGE_RECOMMENDER = "recommender"
STAGES = (STAGE_TOKENIZER, STAGE_RECOMMENDER)

GROUP_TOKENIZER = "tokenizer"
GROUP_RECOMMENDER = "recommender"
GROUP_ADAPTER = "adapter"
GROUP_ORDER = (GROUP_TOKENIZER, GROUP_RECOMMENDER, GROUP_ADAPTER)

GROUP_LABEL = {
    GROUP_TOKENIZER: "RQ-VAE",
    GROUP_RECOMMENDER: "T5",
    GROUP_ADAPTER: "adapter (E2/E5)",
}

LOSS_ORDER = ("L_RQVAE", "L_T5", "L_SIA", "L_PSA")

# Which parameter groups a loss needs to be *trainable* for the (参数状态 × loss
# 状态) consistency rule to hold -- E7 acceptance ⑤, "不得出现参数已冻结但 loss
# 仍在生效之类的错配". Each entry is the group the loss reads through.
LOSS_REQUIRES = {
    "L_RQVAE": (GROUP_TOKENIZER,),
    "L_T5": (GROUP_RECOMMENDER,),
    "L_SIA": (GROUP_RECOMMENDER, GROUP_ADAPTER),
    "L_PSA": (GROUP_RECOMMENDER, GROUP_ADAPTER),
}

STAGE_CONTRACT = {
    STAGE_TOKENIZER: {
        "trainable": (GROUP_TOKENIZER,),
        "frozen": (GROUP_RECOMMENDER, GROUP_ADAPTER),
        "losses": {"L_RQVAE": True, "L_T5": False, "L_SIA": False, "L_PSA": False},
    },
    STAGE_RECOMMENDER: {
        "trainable": (GROUP_RECOMMENDER, GROUP_ADAPTER),
        "frozen": (GROUP_TOKENIZER,),
        "losses": {"L_RQVAE": False, "L_T5": True, "L_SIA": True, "L_PSA": True},
    },
}


class StageError(ValueError):
    """Raised for an unknown stage or an ill-formed controller."""


class StageInconsistent(RuntimeError):
    """Raised when the live parameter/loss state contradicts the stage contract."""


class StageMismatch(RuntimeError):
    """Raised when a checkpoint's recorded stage does not match the resume stage."""


def _require_stage(stage) -> str:
    if stage not in STAGES:
        raise StageError(f"unknown stage {stage!r}; expected one of {STAGES}")
    return stage


# --------------------------------------------------------------------------- #
# Codebook sealing / loading (the `vq.initted` trap)
# --------------------------------------------------------------------------- #
def codebook_layers(rqvae) -> list:
    """The ``VectorQuantizer`` of every RQ level, in level order."""
    return list(rqvae.rq.vq_layers)


def codebook_initted(rqvae) -> list:
    """Per-level ``initted`` flags. This flag is a plain python attribute."""
    return [bool(vq.initted) for vq in codebook_layers(rqvae)]


def codebook_is_zero(rqvae) -> list:
    """Per-level "codebook is still the all-zero placeholder" flags.

    With ``kmeans_init=True`` a fresh ``VectorQuantizer`` zeroes its embedding
    and sets ``initted=False``; ``init_emb`` is what fills it. An all-zero
    codebook therefore means "never initialised", which is the one case where
    sealing would silently freeze garbage instead of a trained codebook.
    """
    flags = []
    for vq in codebook_layers(rqvae):
        w = vq.embedding.weight.detach()
        flags.append(bool(torch.count_nonzero(w) == 0))
    return flags


def seal_codebooks(rqvae, strict: bool = True) -> dict:
    """Mark every codebook as settled so the lazy k-means path can never fire.

    Sets ``vq.initted = True`` on every level. This is the *state* half of the
    freeze: ``requires_grad=False`` stops gradient updates but does not stop
    ``init_emb``'s raw ``weight.data.copy_``.

    Args:
        rqvae: the TIGER ``RQVAE``.
        strict: when True (default), raise if any codebook is still the all-zero
            placeholder, rather than sealing a model that was never trained.

    Returns:
        dict with the flags observed before/after and the zero-codebook list.
    """
    before = codebook_initted(rqvae)
    zeros = codebook_is_zero(rqvae)
    if strict and any(zeros):
        bad = [i for i, z in enumerate(zeros) if z]
        raise StageError(
            f"refusing to seal RQ-VAE levels {bad}: their codebooks are still the "
            f"all-zero placeholder, i.e. the tokenizer was never initialised. "
            f"Sealing here would freeze an untrained codebook instead of a trained one."
        )
    for vq in codebook_layers(rqvae):
        vq.initted = True
    return {"initted_before": before, "initted_after": codebook_initted(rqvae),
            "zero_codebooks": zeros, "sealed": True}


def load_tokenizer_state_dict(rqvae, state_dict, strict: bool = True,
                             strict_codebook: bool = True) -> dict:
    """Load an RQ-VAE checkpoint and restore the non-persisted ``initted`` flags.

    ``initted`` is a plain python attribute: it is not in ``state_dict``, so a
    freshly built ``RQVAE`` (``kmeans_init=True``) still reports ``False`` after
    ``load_state_dict``. The next ``train()``-mode forward then re-runs k-means
    and overwrites the *loaded* codebooks. Restoring the flag is what makes a
    resume a resume.

    Returns a report with the load result and the seal report.
    """
    load_result = rqvae.load_state_dict(state_dict, strict=strict)
    seal = seal_codebooks(rqvae, strict=strict_codebook)
    missing = list(getattr(load_result, "missing_keys", []) or [])
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    return {"missing_keys": missing, "unexpected_keys": unexpected, **seal}


# --------------------------------------------------------------------------- #
# The control layer
# --------------------------------------------------------------------------- #
class StageFreeze:
    """Holds the module groups and flips ``requires_grad`` / loss switches.

    Args:
        tokenizer: the TIGER ``RQVAE`` (optionally preceded by the plain
            ``nn.Module`` that owns it, e.g. a trainer's ``.model``).
        recommender: the ``TIGER`` module from ``model/main.py`` -- its only
            parameters are the T5's, so "freeze the recommender" == "freeze T5".
        adapters: the E2 ``SIAAdapter`` and E5 ``PSAAdapter``. Pass the adapter
            modules themselves, **not** the ``SIAMount`` / ``PSAMount`` wrappers:
            a mount also holds a reference to the RQ-VAE, so handing a mount in
            would put the tokenizer in the adapter group too.
        stage: optional initial stage; applied immediately when given.
        seal_codebook: whether ``apply("recommender")`` also seals the codebook
            (default True -- see the module docstring).
    """

    def __init__(self, tokenizer=None, recommender=None, adapters=(), stage=None,
                 seal_codebook: bool = True):
        self._groups = {GROUP_TOKENIZER: [], GROUP_RECOMMENDER: [], GROUP_ADAPTER: []}
        self._adapter_names = []
        if tokenizer is not None:
            self.add_tokenizer(tokenizer)
        if recommender is not None:
            self.add_recommender(recommender)
        for adapter in adapters:
            self.add_adapter(adapter)
        self.seal_codebook = bool(seal_codebook)
        self._stage = None
        self.seal_report = None
        self._check_disjoint()
        if stage is not None:
            self.apply(stage)

    # ---------------------------------------------------------------- wiring
    def add_tokenizer(self, module: nn.Module) -> "StageFreeze":
        self._groups[GROUP_TOKENIZER].append(module)
        self._check_disjoint()
        return self

    def add_recommender(self, module: nn.Module) -> "StageFreeze":
        self._groups[GROUP_RECOMMENDER].append(module)
        self._check_disjoint()
        return self

    def add_adapter(self, module: nn.Module, name: str = None) -> "StageFreeze":
        self._groups[GROUP_ADAPTER].append(module)
        self._adapter_names.append(name or type(module).__name__)
        self._check_disjoint()
        return self

    def _check_disjoint(self) -> None:
        """A parameter must belong to exactly one group, or freezing is ambiguous."""
        owners = {}
        for group in GROUP_ORDER:
            for p in self._group_params(group):
                owners.setdefault(id(p), []).append(group)
        clashes = {k: v for k, v in owners.items() if len(v) > 1}
        if clashes:
            raise StageError(
                f"{len(clashes)} parameter(s) belong to more than one stage group, "
                f"e.g. {sorted({tuple(v) for v in clashes.values()})}; the same module "
                f"was probably registered twice (a mount holds the RQ-VAE as well as "
                f"the adapter)."
            )

    def _group_params(self, group: str) -> list:
        seen, out = set(), []
        for module in self._groups.get(group, ()):
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    def _tokenizer(self):
        mods = self._groups[GROUP_TOKENIZER]
        if not mods:
            raise StageError("no tokenizer registered; cannot seal its codebook")
        return mods[0]

    # ------------------------------------------------------------- properties
    @property
    def stage(self):
        """The currently applied stage, or None if :meth:`apply` was never called."""
        return self._stage

    def loss_flags(self, stage: str = None) -> dict:
        """The active/inactive loss switches of a stage (static contract)."""
        stage = self._stage if stage is None else stage
        return dict(STAGE_CONTRACT[_require_stage(stage)]["losses"])

    def loss_enabled(self, name: str, stage: str = None) -> bool:
        return bool(self.loss_flags(stage)[name])

    def trainable_parameters(self, stage: str = None) -> list:
        """Every parameter that currently has ``requires_grad=True``."""
        self.assert_consistent(stage)
        return [p for p in self.all_parameters() if p.requires_grad]

    def all_parameters(self) -> list:
        """Every parameter the controller manages, in group order."""
        seen, out = set(), []
        for group in GROUP_ORDER:
            for p in self._group_params(group):
                if id(p) not in seen:
                    seen.add(id(p))
                    out.append(p)
        return out

    def mounts_for_stage(self, sia=None, psa=None, stage: str = None) -> dict:
        """Loss switches for a stage, as kwargs for ``model.main.train()``.

        In the tokenizer stage both are dropped to ``None`` -- that is the
        "explicitly reset the loss switches" half of a stage switch. The caller
        passes the result straight through, so neither training loop needs to
        know about stages.
        """
        flags = self.loss_flags(stage)
        return {"sia": sia if (flags["L_SIA"] and sia is not None) else None,
                "psa": psa if (flags["L_PSA"] and psa is not None) else None}

    # ------------------------------------------------------------ the switch
    def apply(self, stage: str, log: bool = True) -> dict:
        """Enter ``stage``: reset ``requires_grad`` and the loss switches.

        Returns the live report (see :meth:`report`). Logs the
        (parameter state x loss state) matrix at INFO level, which is E7
        acceptance ④/⑤'s audit trail.
        """
        stage = _require_stage(stage)
        trainable = set(STAGE_CONTRACT[stage]["trainable"])
        for group in GROUP_ORDER:
            desired = group in trainable
            for p in self._group_params(group):
                p.requires_grad = desired

        self.seal_report = None
        if stage == STAGE_RECOMMENDER and self.seal_codebook and self._groups[GROUP_TOKENIZER]:
            self.seal_report = seal_codebooks(self._tokenizer())

        self._stage = stage
        self.assert_consistent(stage)
        report = self.report(stage)
        if log:
            logger.info("[tiger_ext.stage] entered stage %r\n%s", stage, self.matrix(stage))
        return report

    # -------------------------------------------------------------- reporting
    def report(self, stage: str = None) -> dict:
        """Live state: measured ``requires_grad`` counts + the loss switches."""
        stage = self._stage if stage is None else _require_stage(stage)
        groups = {}
        for group in GROUP_ORDER:
            params = self._group_params(group)
            n_train = sum(1 for p in params if p.requires_grad)
            groups[group] = {
                "tensors": len(params),
                "elements": sum(p.numel() for p in params),
                "trainable_tensors": n_train,
                "frozen_tensors": len(params) - n_train,
                "all_trainable": bool(params) and n_train == len(params),
                "all_frozen": bool(params) and n_train == 0,
            }
        return {
            "stage": stage,
            "groups": groups,
            "losses": self.loss_flags(stage),
            "codebook_initted": codebook_initted(self._tokenizer())
            if self._groups[GROUP_TOKENIZER] else None,
        }

    def matrix(self, stage: str = None) -> str:
        """The (parameter state x loss state) table for a stage, as text.

        Two columns: the static contract, and what is measured right now. They
        must agree -- :meth:`assert_consistent` is what enforces it.
        """
        stage = self._stage if stage is None else _require_stage(stage)
        live = self.report(stage)
        width = max(len(GROUP_LABEL[g]) for g in GROUP_ORDER) + 4
        lines = [f"stage = {stage}   (plan.md §1.4 / E7 contract; "
                 f"left = contract, right = measured)"]
        head = "  " + "component".ljust(width) + "| " + "".join(
            f"{s:^26}" for s in STAGES) + "| measured now"
        lines.append(head)
        lines.append("  " + "-" * (len(head) - 2))

        def cell(group, s):
            trainable = group in STAGE_CONTRACT[s]["trainable"]
            return ("trainable" if trainable else "frozen").center(26)

        for group in GROUP_ORDER:
            g = live["groups"][group]
            if g["tensors"] == 0:
                now = "n/a (not registered)"
            elif g["all_trainable"]:
                now = f"trainable ({g['tensors']} tensors, {g['elements']} elems)"
            elif g["all_frozen"]:
                now = f"frozen ({g['tensors']} tensors, {g['elements']} elems)"
            else:
                now = f"MIXED {g['trainable_tensors']}/{g['tensors']} trainable"
            lines.append("  " + GROUP_LABEL[group].ljust(width) + "| " +
                         "".join(cell(group, s) for s in STAGES) + f"| {now}")

        for name in LOSS_ORDER:
            row = "".join(
                ("active" if STAGE_CONTRACT[s]["losses"][name] else "inactive").center(26)
                for s in STAGES)
            lines.append("  " + name.ljust(width) + "| " + row + "| " +
                         ("active" if live["losses"][name] else "inactive"))
        if live["codebook_initted"] is not None:
            lines.append("  " + "codebook initted".ljust(width) + "| " +
                         " " * (26 * len(STAGES)) + "| " + str(live["codebook_initted"]))
        return "\n".join(lines)

    # ------------------------------------------------------------ consistency
    def assert_consistent(self, stage: str = None) -> bool:
        """Verify the live state equals the contract; raise otherwise (⑤)."""
        stage = self._stage if stage is None else _require_stage(stage)
        contract = STAGE_CONTRACT[stage]
        problems = []
        for group in GROUP_ORDER:
            params = self._group_params(group)
            if not params:
                continue
            want = group in contract["trainable"]
            bad = sum(1 for p in params if bool(p.requires_grad) != want)
            if bad:
                problems.append(
                    f"group {group!r}: {bad}/{len(params)} tensors have the wrong "
                    f"requires_grad (contract says {'trainable' if want else 'frozen'})")
        for name in LOSS_ORDER:
            if not contract["losses"][name]:
                continue
            for group in LOSS_REQUIRES[name]:
                if group not in contract["trainable"]:
                    problems.append(
                        f"{name} is active but its group {group!r} is frozen in stage "
                        f"{stage!r} (parameter/loss mismatch)")
        if problems:
            raise StageInconsistent(
                f"stage {stage!r} is inconsistent:\n  - " + "\n  - ".join(problems))
        return True

    # --------------------------------------------------------------- manifest
    def signature(self, stage: str = None) -> dict:
        """A JSON-able, auditable record of this stage's freeze state."""
        stage = self._stage if stage is None else _require_stage(stage)
        rep = self.report(stage)
        return {
            "stage": stage,
            "groups": {g: {k: v for k, v in rep["groups"][g].items()} for g in GROUP_ORDER},
            "losses": rep["losses"],
            "codebook_initted": rep["codebook_initted"],
            "adapters": list(self._adapter_names),
        }

    def extra_repr(self) -> str:
        return (f"stage={self._stage!r}, groups="
                f"{ {g: len(m) for g, m in self._groups.items()} }, "
                f"seal_codebook={self.seal_codebook}")


# --------------------------------------------------------------------------- #
# Checkpoint-side audit trail (E7 acceptance ④) and resume check
# --------------------------------------------------------------------------- #
def stage_manifest_path(ckpt_path: str) -> str:
    """Where the stage record for ``ckpt_path`` lives."""
    return f"{ckpt_path}.stage.json"


def write_stage_manifest(ckpt_path: str, freeze: StageFreeze, stage: str = None,
                         extra: dict = None, path: str = None) -> str:
    """Write the freeze signature beside a checkpoint.

    The two training loops are not touched, so the stage record travels as a
    sidecar rather than as a new key inside the saved dict.
    """
    record = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "baseline_contract": "plan.md §1.4 / E7",
              "freeze": freeze.signature(stage)}
    if extra:
        record["extra"] = dict(extra)
    path = path or stage_manifest_path(ckpt_path)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    return path


def read_stage_manifest(path: str) -> dict:
    """Read a stage record; accepts either the ckpt path or the sidecar path."""
    if not str(path).endswith(".json"):
        sidecar = stage_manifest_path(path)
        if os.path.exists(sidecar):
            path = sidecar
    with open(path) as fh:
        return json.load(fh)


def verify_stage_manifest(manifest, freeze: StageFreeze, stage: str = None) -> dict:
    """Resume-time check: does the recorded freeze state match the one to resume with?

    Raises :class:`StageMismatch` on a different stage or a different
    trainable/frozen split, so a resume cannot silently continue under the wrong
    phase.
    """
    record = read_stage_manifest(manifest) if isinstance(manifest, str) else manifest
    recorded = record.get("freeze", record)
    stage = freeze.stage if stage is None else _require_stage(stage)
    if recorded.get("stage") != stage:
        raise StageMismatch(
            f"checkpoint was written in stage {recorded.get('stage')!r} but the "
            f"resume asks for {stage!r}")
    live = freeze.signature(stage)
    for group, want in recorded.get("groups", {}).items():
        got = live["groups"].get(group)
        if got is None or got["trainable_tensors"] != want.get("trainable_tensors") \
                or got["tensors"] != want.get("tensors"):
            raise StageMismatch(
                f"group {group!r} freeze state differs: checkpoint "
                f"{want.get('trainable_tensors')}/{want.get('tensors')} trainable, "
                f"now {None if got is None else got['trainable_tensors']}/"
                f"{None if got is None else got['tensors']}")
    return record
