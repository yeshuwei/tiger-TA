"""P2 -- fixed-SID recommender training entry (R13 cells #3 / #5).

``TigerAlternatingPipeline.recommender_phase`` is a **one-epoch phase body**: it
builds a fresh optimizer, trains one epoch and hands control back to the
scheduler, which may then refine the tokenizer and refresh the SID. R13's
``A=0`` cells (#1, #2, #3, #5) freeze the tokenizer and the SID for the whole
run, so they need the *other* shape of the same thing: one long-lived optimizer
driving ``model/main.py``'s own epoch loop against one immutable SID. Calling
``recommender_phase`` 200 times would not be that -- every call rebuilds the
optimizer (and would re-seed, rebuild the dataset and rebuild both mounts), so
"200 x 1 epoch" is a different training regime from "1 x 200 epochs".

This module is that entry point. It adds no loss, no schedule and no model:

    tokenizer      : loaded once from a checkpoint and frozen (E7's
                     ``StageFreeze.apply("recommender")`` -- which also seals the
                     codebooks, so the lazy k-means path cannot rewrite them)
    item table     : ``data_root/<dataset>/item_emb.parquet`` by default, or an
                     explicit ``embedding_path`` (R13's 768d tables). The path,
                     its sha256 and its width are recorded; an explicit table
                     whose width differs from the checkpoint's ``in_dim`` is
                     refused at construction, before any model is built
    SID            : one ``.npy`` read once, validated against the item table,
                     and re-hashed after every epoch
    recommender    : ``model/main.py::train`` once per epoch, one ``Adam`` built
                     once, then ``model/main.py::evaluate`` -- i.e. exactly the
                     loop ``model/main.py::__main__`` runs
    SIA / PSA      : ``tiger_ext.sia_loss.SIAMount`` / ``psa_loss.PSAMount``,
                     mounted only when their weight is > 0

Why CSG is *not* a parameter here
---------------------------------
CSG (C) is a **tokenizer-stage** factor: ``L_csg`` enters
``rqvae/trainer.py::_train_epoch``. R13 cell #5 (CSG+ETE) therefore differs
from cell #3 (ETE) only in *which tokenizer checkpoint and which SID file* are
fed in; the recommender stage is identical. ``csg=`` is accepted only so that
this can be stated as a refusal: passing a hook raises, because keeping a live
CSG hook here would either do nothing (the trainer is not running) or tempt a
caller into training the tokenizer while the SID is supposed to be frozen.

What is verified at runtime (and recorded in the run manifest)
--------------------------------------------------------------
  * **the SID belongs to the tokenizer checkpoint** (B4): before anything is
    trained, ``sid_refresh.verify_sid_manifest`` is run against the SID's
    sidecar manifest (``<sid>.sid.json``, or an explicit ``sid_manifest``).
    A mismatch in either the checkpoint hash or the SID hash is fatal. A SID
    with **no** manifest is refused by default; ``allow_unregistered_sid=True``
    is the explicit opt-out (recorded in ``run.json``) for legacy SIDs that
    predate ``refresh_sid`` -- it still checks the versioned file-name
    convention when the name encodes a checkpoint hash;
  * the tokenizer's ``state_dict`` digest and the SID file's sha256 are
    identical before and after training, checked **after every epoch**;
  * the optimizer's parameter set equals T5 + the two adapters and contains no
    RQ-VAE parameter;
  * every SID row the dataset knows maps back to itself through each live
    mount's routing table (the same hand-off proof E9 makes each cycle);
  * the epoch loop records train loss, valid metrics, whether the epoch
    improved, and whether a test evaluation + checkpoint write happened.

Nothing in this module is imported by ``model/``, ``rqvae/`` or ``clean_tiger/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from .alt_pipeline import (
    NUM_WORKERS, RQVAE_CONFIG, SEED_T5, T5_BATCH, T5_BEAM, T5_CONFIG,
    T5_INFER_SIZE, T5_LR, T5_TOPK, frozen_modules,
)
from .stage import (
    STAGE_RECOMMENDER, StageFreeze, codebook_initted, load_tokenizer_state_dict,
    write_stage_manifest,
)
from .sid_refresh import (
    SidManifestMismatch, load_sid, parse_sid_name, sha256_file, sidecar_path,
    validate_sid, verify_sid_manifest,
)

DEFAULT_RUN_NAME = "fixed_sid"


class FixedSidError(RuntimeError):
    """Raised when the frozen-SID contract would be violated."""


class SidProvenanceError(FixedSidError):
    """The frozen SID cannot be tied to the supplied tokenizer checkpoint."""


# --------------------------------------------------------------------------- #
# frozen-state evidence
# --------------------------------------------------------------------------- #
def state_digest(module) -> str:
    """Order-sensitive sha256 over a module's ``state_dict``.

    Used to prove the tokenizer did not move. Bitwise (``tobytes``) rather than
    a float summary, so a 1-ulp change cannot hide.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tokenizer_input_dim(ckpt) -> int | None:
    """The checkpoint's input width, read from its first encoder linear layer.

    ``RQVAE`` builds ``encoder = MLPLayers([in_dim] + layers + [e_dim])``; the
    first ``nn.Linear`` is ``mlp_layers.1`` with shape ``(layers[0], in_dim)``
    (index 0 is a ``Dropout``). Returns ``None`` when the key is absent (e.g. a
    bare ``state_dict`` that was stripped), in which case the ordinary
    ``load_state_dict`` shape check still refuses a mismatch.
    """
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt \
        else ckpt
    weight = state.get("encoder.mlp_layers.1.weight") if isinstance(state, dict) \
        else None
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return int(shape[1])


def tokenizer_input_shape(ckpt):
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt \
        else ckpt
    weight = state.get("encoder.mlp_layers.1.weight") if isinstance(state, dict) \
        else None
    shape = getattr(weight, "shape", None)
    return None if shape is None else tuple(int(s) for s in shape)


def verify_sid_routing(dataset, mounts, sid_tokens) -> dict:
    """Prove the dataset and every live mount were built from *this* SID.

    ``GenRecDataset`` maps item ``i+1`` to ``code + level*256 + 1``
    (``model/dataset.py:84-87``); both mounts key their SID -> row table on the
    same offsets. A dataset or mount built from a different file fails here.
    """
    expected = int(sid_tokens.shape[0])
    if len(dataset.item_to_code) != expected:
        raise FixedSidError(
            f"the dataset knows {len(dataset.item_to_code)} items but the SID has "
            f"{expected} rows; they came from different files")
    mismatches, checked = {}, []
    for name, mount in mounts.items():
        if mount is None:
            mismatches[name] = None
            continue
        lookup = mount._lookup
        if len(lookup) != expected:
            raise FixedSidError(
                f"{name} lookup has {len(lookup)} entries, SID has {expected} rows")
        bad = 0
        for row in range(expected):
            key = tuple(int(t) for t in dataset.item_to_code[row + 1])
            if lookup.get(key) != row:
                bad += 1
        mismatches[name] = bad
        checked.append(name)
    if any(v for v in mismatches.values() if v):
        raise FixedSidError(
            f"the dataset and {mismatches} disagree about the SID; they were built "
            "from different files")
    return {"n_items": expected, "mismatches": mismatches, "checked": checked}


# --------------------------------------------------------------------------- #
# SID <-> tokenizer provenance  (B4)
# --------------------------------------------------------------------------- #
SID_MANIFEST_MISSING = "no SID manifest"


def resolve_sid_manifest(sid_path, manifest_path=None):
    """Where this SID's provenance record lives, or ``None`` if there is none.

    ``refresh_sid`` writes the sidecar next to the versioned SID
    (``<sid>.sid.json``); that is the default. ``manifest_path`` overrides it
    for a SID that was published (``publish_sid`` copies the ``.npy`` only) or
    whose sidecar was moved.
    """
    if manifest_path is not None:
        path = os.path.abspath(manifest_path)
        if not os.path.isfile(path):
            raise SidProvenanceError(f"SID manifest not found: {path}")
        return path
    side = sidecar_path(sid_path)
    return side if os.path.isfile(side) else None


def verify_sid_provenance(sid_path, ckpt_path, manifest_path=None,
                          allow_unregistered=False, dataset=None):
    """Tie a SID to the tokenizer checkpoint that generated it (B4).

    This is the guard ``FixedSidTrainer`` runs before training. It reuses
    ``sid_refresh.verify_sid_manifest`` rather than re-implementing the hash
    comparison: a manifest carries the checkpoint's sha256 and the SID's
    sha256, so either pairing the wrong checkpoint with a SID or editing the
    SID after registration is refused with that module's own error type.

    A SID without a manifest is **not** silently accepted. Two legitimate
    no-manifest situations exist in this repo:

      1. legacy SIDs produced before ``sid_refresh`` existed (e.g. the
         ``*_content768d_*_sid.npy`` files under ``data/``) -- R13's ``λe``
         arms read one of these;
      2. a SID copied out of the versioned directory, e.g. by ``publish_sid``,
         which takes the ``.npy`` but not its sidecar.

    Case 2 is solved by ``manifest_path``. Case 1 needs the explicit
    ``allow_unregistered=True`` opt-out; the run then records
    ``mode="unregistered"`` and this function still checks the versioned
    naming convention (``{dataset}_sid_v{version}_{ckpt_sha8}.npy``) when the
    file name carries one, so a mispaired *versioned* SID is still refused.

    Returns the evidence dict stored in ``run.json``.
    """
    if not os.path.isfile(sid_path):
        raise SidProvenanceError(f"SID not found: {sid_path}")
    if not os.path.isfile(ckpt_path):
        raise SidProvenanceError(f"tokenizer checkpoint not found: {ckpt_path}")

    side = resolve_sid_manifest(sid_path, manifest_path)
    ckpt_sha = sha256_file(ckpt_path)
    sid_sha = sha256_file(sid_path)

    if side is None:
        if not allow_unregistered:
            raise SidProvenanceError(
                f"{SID_MANIFEST_MISSING} for {sid_path} (expected "
                f"{sidecar_path(sid_path)}); refusing to train on a SID that cannot "
                f"be tied to {ckpt_path}. Register the SID with "
                "tiger_ext.sid_refresh.refresh_sid, pass sid_manifest=..., or -- "
                "for a legacy SID that predates refresh_sid -- opt in explicitly "
                "with allow_unregistered_sid=True.")
        named = parse_sid_name(sid_path)
        if named is not None and named["ckpt_sha8"] != ckpt_sha[:8]:
            raise SidProvenanceError(
                f"the SID file name says checkpoint {named['ckpt_sha8']} but "
                f"{ckpt_path} is {ckpt_sha[:8]}; even an unregistered SID must "
                "match its versioned name")
        return {
            "mode": "unregistered",
            "verified": False,
            "manifest": None,
            "sid_path": os.path.abspath(sid_path),
            "sid_sha256": sid_sha,
            "ckpt_path": os.path.abspath(ckpt_path),
            "ckpt_sha256": ckpt_sha,
            "name_checked": named is not None,
            "warning": "no SID manifest: provenance was not verified against a "
                       "registration record (explicit allow_unregistered_sid)",
        }

    try:
        # an explicit manifest may have any file name, so read it directly
        # (``read_sid_manifest`` appends the sidecar suffix to non-.sid.json paths)
        with open(side) as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SidProvenanceError(
            f"SID manifest is unreadable ({side}): {exc}") from exc
    try:
        verify_sid_manifest(record, ckpt_path=ckpt_path, sid_path_=sid_path)
    except SidManifestMismatch as exc:
        raise SidProvenanceError(
            f"{sid_path} does not belong to {ckpt_path}: {exc}") from exc

    # The dataset *name* is not part of the hash linkage: a SID may legitimately
    # be reused under another name when the item space is the same (P2's
    # ``P2SynSmall`` replays P2Syn's tokenizer and SID over a second split).
    # Record a mismatch so it is auditable, but decide on the hashes.
    recorded_dataset = record.get("dataset")
    dataset_matches = (dataset is None or recorded_dataset is None
                       or recorded_dataset == dataset)

    return {
        "mode": "manifest",
        "verified": True,
        "manifest": side,
        "dataset": recorded_dataset,
        "dataset_matches": dataset_matches,
        "version": record.get("version"),
        "n_items": record.get("n_items"),
        "data_path": record.get("data_path"),
        "sid_path": os.path.abspath(sid_path),
        "sid_sha256": sid_sha,
        "ckpt_path": os.path.abspath(ckpt_path),
        "ckpt_sha256": ckpt_sha,
    }


# --------------------------------------------------------------------------- #
# the trainer
# --------------------------------------------------------------------------- #
class FixedSidTrainer:
    """Train the T5 recommender against one frozen tokenizer and one frozen SID.

    Args:
        dataset: dataset name; ``data_root/<dataset>/{item_emb,train,valid,test}``.
        data_root: root holding the dataset directory.
        embedding_path: the item-embedding table to read. Default ``None`` keeps
            the historical ``data_root/<dataset>/item_emb.parquet``; pass an
            explicit path (e.g. R13's 768d table) when the tokenizer checkpoint
            was trained on a table that does not live at the default path. The
            table's width must equal the checkpoint's ``in_dim``.
        sid_path: the ``.npy`` SID for the whole run. Read once, hashed, and
            re-hashed after every epoch.
        tokenizer_ckpt: the RQ-VAE checkpoint the SID was generated from. Loaded
            once; ``initted`` flags are restored (E7) and the codebooks are
            sealed, so no lazy k-means can fire later.
        run_dir: where ``best.pth`` / ``last.pth`` / ``run.json`` go.
        lambda_sia / lambda_psa: ETE weights. Both default to **0**, in which
            case the mount is not even constructed and ``model/main.py::train``
            receives ``None`` -- the Clean TIGER code path. R13 cells #3 / #5
            pass values > 0.
        csg: **rejected unless None** -- see the module docstring. CSG belongs to
            the tokenizer stage; a cell #5 run bakes it into ``tokenizer_ckpt``
            and ``sid_path`` instead.
        num_epochs / early_stop: the ``model/main.py::__main__`` protocol
            (200 epochs, patience 10 on valid NDCG@20) unless overridden.
        limit_train / limit_eval: optional caps for smoke runs; recorded in the
            manifest so a truncated run cannot be mistaken for a full one.
        sid_manifest: explicit path to the SID's provenance sidecar. Default:
            ``<sid_path>.sid.json``. Use this when the SID was published (the
            ``.npy`` was copied without its sidecar) or the sidecar was moved.
        allow_unregistered_sid: explicit opt-out for a **legacy SID that has no
            manifest** (see :func:`verify_sid_provenance`). Default ``False`` so
            a missing manifest is a hard error, never a silent skip.
    """

    def __init__(self, dataset, data_root, sid_path, tokenizer_ckpt, run_dir,
                 device="cpu", rqvae_config=None, t5_config=None,
                 lambda_sia=0.0, lambda_psa=0.0, seed=SEED_T5,
                 batch_size=T5_BATCH, infer_size=T5_INFER_SIZE, beam_size=T5_BEAM,
                 topk_list=None, lr=T5_LR, num_workers=NUM_WORKERS,
                 limit_train=None, limit_eval=None, num_epochs=200, early_stop=10,
                 csg=None, arm=None, sid_manifest=None,
                 allow_unregistered_sid=False, embedding_path=None):
        mods = frozen_modules()
        torch = mods["torch"]

        if csg is not None:
            raise FixedSidError(
                "csg= is not accepted by the fixed-SID entry point: CSG is a "
                "tokenizer-stage loss (rqvae/trainer.py), and this trainer must "
                "never update the tokenizer while the SID is frozen. For R13 cell "
                "#5 (CSG+ETE), pass the CSG-trained tokenizer checkpoint and the "
                "SID generated from it; the recommender stage is then identical "
                "to cell #3.")

        self.dataset = dataset
        self.data_root = os.path.abspath(data_root)
        self.data_dir = os.path.join(self.data_root, dataset)
        # Default keeps the historical path exactly; an explicit table is used
        # as given (never silently replaced by -- or merged with -- the default).
        self.embedding_path_explicit = embedding_path is not None
        self.embedding_path = (
            os.path.join(self.data_dir, "item_emb.parquet")
            if embedding_path is None else os.path.abspath(embedding_path))
        self.sid_path = os.path.abspath(sid_path)
        self.tokenizer_ckpt = os.path.abspath(tokenizer_ckpt)
        self.run_dir = os.path.abspath(run_dir)
        self.device = str(device)
        self.torch_device = torch.device(device)
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.infer_size = int(infer_size)
        self.beam_size = int(beam_size)
        self.topk_list = list(topk_list or T5_TOPK)
        self.lr = float(lr)
        self.num_workers = int(num_workers)
        self.limit_train = limit_train
        self.limit_eval = limit_eval
        self.num_epochs = int(num_epochs)
        self.early_stop = int(early_stop)
        self.arm = None if arm is None else str(arm)
        self.rqvae_config = dict(RQVAE_CONFIG, **(rqvae_config or {}))
        self.t5_config = dict(T5_CONFIG, **(t5_config or {}))
        self.lambda_sia = float(lambda_sia)
        self.lambda_psa = float(lambda_psa)

        for path, what in ((self.embedding_path, "item embeddings"),
                           (self.sid_path, "SID"),
                           (self.tokenizer_ckpt, "tokenizer checkpoint")):
            if not os.path.isfile(path):
                raise FixedSidError(f"{what} not found: {path}")

        # --- B4: the SID must belong to this tokenizer, before anything trains --
        self.sid_manifest = (None if sid_manifest is None
                             else os.path.abspath(sid_manifest))
        self.allow_unregistered_sid = bool(allow_unregistered_sid)
        self.provenance = self._verify_provenance("construction")

        # --- the frozen inputs, read exactly once ------------------------------
        self.sid_codes = load_sid(self.sid_path)
        self.item_embeddings = torch.as_tensor(
            mods["EmbDataset"](self.embedding_path).embeddings,
            dtype=torch.float32).to(self.torch_device)
        self.in_dim = int(self.item_embeddings.shape[1])
        self.n_items = int(self.item_embeddings.shape[0])
        self.embedding_sha256 = sha256_file(self.embedding_path)
        self.sid_stats = validate_sid(self.sid_codes, n_items=self.n_items)
        self.sid_tokens = torch.as_tensor(
            np.asarray(self.sid_codes), dtype=torch.long)

        # --- the checkpoint is read once, and its width checked *before* the
        # model is built, so a 4096d checkpoint cannot hide behind a 768d table
        self.tokenizer_ckpt_data = self._read_tokenizer_checkpoint()
        self.tokenizer_input_dim = tokenizer_input_dim(self.tokenizer_ckpt_data)
        self.tokenizer_input_shape = tokenizer_input_shape(self.tokenizer_ckpt_data)
        self._assert_embedding_matches_tokenizer()

        # --- models, built once ------------------------------------------------
        mods["tiger_main"].set_seed(self.seed)
        cfg = self.rqvae_config
        self.rqvae = mods["RQVAE"](
            in_dim=self.in_dim, num_emb_list=cfg["num_emb_list"], e_dim=cfg["e_dim"],
            layers=cfg["layers"], dropout_prob=cfg["dropout_prob"], bn=cfg["bn"],
            loss_type=cfg["loss_type"], quant_loss_weight=cfg["quant_loss_weight"],
            beta=cfg["beta"], kmeans_init=cfg["kmeans_init"],
            kmeans_iters=cfg["kmeans_iters"], sk_epsilons=cfg["sk_epsilons"],
            sk_iters=cfg["sk_iters"]).to(self.torch_device)
        self.tokenizer_load = self._load_tokenizer()

        self.tiger = mods["tiger_main"].TIGER(self.t5_config).to(self.torch_device)

        from .psa import PSAAdapter
        from .sia import SIAAdapter
        self.sia_adapter = SIAAdapter(self.t5_config["d_model"]).to(self.torch_device)
        self.psa_adapter = PSAAdapter(self.t5_config["d_model"],
                                      self.in_dim).to(self.torch_device)

        # --- the freeze contract: RQ-VAE frozen, T5 + adapters trainable -------
        self.freeze = StageFreeze(self.rqvae, self.tiger,
                                  [self.sia_adapter, self.psa_adapter])
        self.freeze_report = self.freeze.apply(STAGE_RECOMMENDER)
        self._assert_tokenizer_frozen()

        # --- the ETE mounts: only constructed when the weight is live ----------
        self.sia = None
        self.psa = None
        if self.lambda_sia > 0:
            from .sia_loss import SIAMount
            self.sia = SIAMount(self.sia_adapter, self.rqvae, self.item_embeddings,
                                self.sid_tokens, lambda_sia=self.lambda_sia)
        if self.lambda_psa > 0:
            from .psa_loss import PSAMount
            self.psa = PSAMount(self.psa_adapter, self.rqvae, self.item_embeddings,
                                self.sid_tokens, lambda_psa=self.lambda_psa)
        self.mounts = self.freeze.mounts_for_stage(sia=self.sia, psa=self.psa)
        if (self.mounts["sia"] is not None) != (self.lambda_sia > 0) or \
                (self.mounts["psa"] is not None) != (self.lambda_psa > 0):
            raise FixedSidError(
                f"the stage contract dropped a live mount: {self.mounts} with "
                f"lambda_sia={self.lambda_sia}, lambda_psa={self.lambda_psa}")

        # --- one optimizer for the whole run -----------------------------------
        self.optimizer = torch.optim.Adam(
            self.freeze.trainable_parameters(STAGE_RECOMMENDER), lr=self.lr)
        self._assert_optimizer_scope()

        #: set by :meth:`run`; the frozen-state checks compare against these
        self.tokenizer_digest_before = state_digest(self.rqvae)
        self.sid_sha256_before = sha256_file(self.sid_path)
        self.codebook_flags_before = codebook_initted(self.rqvae)

    # ------------------------------------------------------------------ guards
    def _verify_provenance(self, where):
        """Re-prove the SID <-> tokenizer link (B4). Cheap: two file hashes."""
        try:
            return verify_sid_provenance(
                self.sid_path, self.tokenizer_ckpt,
                manifest_path=self.sid_manifest,
                allow_unregistered=self.allow_unregistered_sid,
                dataset=self.dataset)
        except SidProvenanceError as exc:
            raise SidProvenanceError(f"{where}: {exc}") from exc

    def _read_tokenizer_checkpoint(self):
        """Read the checkpoint exactly once (also used for the width check)."""
        torch = frozen_modules()["torch"]
        try:
            return torch.load(self.tokenizer_ckpt, map_location="cpu",
                              weights_only=False)
        except Exception as exc:                              # noqa: BLE001
            raise FixedSidError(
                f"the tokenizer checkpoint could not be read "
                f"({self.tokenizer_ckpt}): {exc}") from exc

    def _assert_embedding_matches_tokenizer(self) -> None:
        """Refuse a table whose width differs from the checkpoint's ``in_dim``.

        Runs at construction, before ``RQVAE`` is built, so the error names
        both inputs instead of surfacing as a ``size mismatch`` deep inside
        ``load_state_dict``.
        """
        expected = self.tokenizer_input_dim
        if expected is None or expected == self.in_dim:
            return
        source = "explicit --embedding_path" if self.embedding_path_explicit \
            else "default data_root/<dataset>/item_emb.parquet"
        hint = ""
        if not self.embedding_path_explicit:
            hint = ("; pass embedding_path=<the table this checkpoint was trained "
                    "on> (e.g. the 768d content table) instead of relying on the "
                    "default")
        raise FixedSidError(
            f"embedding dimension mismatch: {self.embedding_path} ({source}) has "
            f"{self.in_dim} columns, but {self.tokenizer_ckpt} was built with "
            f"in_dim={expected} (encoder.mlp_layers.1.weight "
            f"{self.tokenizer_input_shape}){hint}")

    def _load_tokenizer(self):
        """Load the checkpoint and restore ``vq.initted`` (E7's seal-then-load)."""
        ckpt = self.tokenizer_ckpt_data
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt \
            else ckpt
        try:
            report = load_tokenizer_state_dict(self.rqvae, state)
        except Exception as exc:                              # noqa: BLE001
            raise FixedSidError(
                "the tokenizer checkpoint does not fit the RQ-VAE built from "
                f"rqvae_config (num_emb_list={self.rqvae_config['num_emb_list']}, "
                f"e_dim={self.rqvae_config['e_dim']}): {exc}") from exc
        report["has_args"] = bool(isinstance(ckpt, dict) and "args" in ckpt)
        report["ckpt_sha256"] = sha256_file(self.tokenizer_ckpt)
        return report

    def _assert_tokenizer_frozen(self) -> None:
        trainable = [name for name, p in self.rqvae.named_parameters()
                     if p.requires_grad]
        if trainable:
            raise FixedSidError(
                f"{len(trainable)} RQ-VAE tensors are still trainable, e.g. "
                f"{trainable[:3]}; the tokenizer must be frozen for a fixed SID")
        flags = codebook_initted(self.rqvae)
        if not all(flags):
            raise FixedSidError(
                f"codebooks are not sealed ({flags}); a training-mode forward could "
                "re-run k-means and rewrite them")

    def _assert_optimizer_scope(self) -> None:
        opt = {id(p) for g in self.optimizer.param_groups for p in g["params"]}
        tiger = {id(p) for p in self.tiger.parameters()}
        adapters = {id(p) for p in list(self.sia_adapter.parameters())
                    + list(self.psa_adapter.parameters())}
        rqvae = {id(p) for p in self.rqvae.parameters()}
        if opt & rqvae:
            raise FixedSidError(
                f"{len(opt & rqvae)} RQ-VAE tensors are in the optimizer; the "
                "tokenizer and the SID must stay frozen")
        if opt != (tiger | adapters):
            raise FixedSidError(
                "the optimizer must cover exactly T5 + adapters "
                f"(T5 {len(tiger)}, adapters {len(adapters)}, "
                f"optimizer {len(opt)}, extra {len(opt - tiger - adapters)}, "
                f"missing {len((tiger | adapters) - opt)})")

    def _check_frozen(self, where: str) -> dict:
        """Re-prove the frozen inputs did not move."""
        digest = state_digest(self.rqvae)
        sid_sha = sha256_file(self.sid_path)
        emb_sha = sha256_file(self.embedding_path)
        flags = codebook_initted(self.rqvae)
        if digest != self.tokenizer_digest_before:
            raise FixedSidError(
                f"{where}: the tokenizer's state_dict changed during recommender "
                "training; the SID is supposed to be frozen")
        if sid_sha != self.sid_sha256_before:
            raise FixedSidError(f"{where}: {self.sid_path} was rewritten ({sid_sha})")
        if emb_sha != self.embedding_sha256:
            raise FixedSidError(
                f"{where}: the item table {self.embedding_path} was rewritten "
                f"({emb_sha}); the run must keep the table it was built from")
        if flags != self.codebook_flags_before:
            raise FixedSidError(f"{where}: codebook initted flags changed {flags}")
        return {"tokenizer_digest": digest, "sid_sha256": sid_sha,
                "embedding_sha256": emb_sha}

    # ------------------------------------------------------------------- data
    def _build_loader(self, split, mode, batch_size, shuffle, limit):
        mods = frozen_modules()
        torch = mods["torch"]
        dataset = mods["GenRecDataset"](
            dataset_path=os.path.join(self.data_dir, f"{split}.parquet"),
            code_path=self.sid_path, mode=mode,
            max_len=self.t5_config["max_len"])
        if os.path.abspath(dataset.code_path) != self.sid_path:
            raise FixedSidError(
                f"{split}: the dataset carries {dataset.code_path!r}, "
                f"not the frozen SID {self.sid_path!r}")
        source = dataset
        truncated = False
        if limit is not None and len(dataset) > limit:
            source = torch.utils.data.Subset(dataset, list(range(limit)))
            truncated = True
        loader = mods["GenRecDataLoader"](source, batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=self.num_workers)
        return dataset, loader, truncated

    # ------------------------------------------------------------- reporting
    def run_config(self) -> dict:
        """Everything a re-run needs. No results, only inputs and knobs."""
        return {
            "entry": "tiger_ext.fixed_train",
            "arm": self.arm,
            "dataset": self.dataset,
            "data_root": self.data_root,
            "embedding_path": self.embedding_path,
            "embedding_path_explicit": self.embedding_path_explicit,
            "embedding_sha256": self.embedding_sha256,
            "embedding_dim": self.in_dim,
            "embedding_n_items": self.n_items,
            "tokenizer_input_dim": self.tokenizer_input_dim,
            "tokenizer_input_shape": self.tokenizer_input_shape,
            "sid_path": self.sid_path,
            "sid_sha256": self.sid_sha256_before,
            "sid_stats": {k: v for k, v in self.sid_stats.items() if k != "errors"},
            "tokenizer_ckpt": self.tokenizer_ckpt,
            "tokenizer_ckpt_sha256": self.tokenizer_load["ckpt_sha256"],
            "tokenizer_args_in_ckpt": self.tokenizer_load["has_args"],
            "sid_manifest": self.provenance["manifest"],
            "sid_provenance_mode": self.provenance["mode"],
            "allow_unregistered_sid": self.allow_unregistered_sid,
            "sid_provenance": self.provenance,
            "run_dir": self.run_dir,
            "device": self.device, "seed": self.seed,
            "batch_size": self.batch_size, "infer_size": self.infer_size,
            "beam_size": self.beam_size, "topk_list": self.topk_list,
            "lr": self.lr, "num_workers": self.num_workers,
            "limit_train": self.limit_train, "limit_eval": self.limit_eval,
            "num_epochs": self.num_epochs, "early_stop": self.early_stop,
            "lambda_sia": self.lambda_sia, "lambda_psa": self.lambda_psa,
            "rqvae_config": self.rqvae_config, "t5_config": self.t5_config,
            "in_dim": self.in_dim, "n_items": self.n_items,
            "protocol": "model/main.py::__main__ loop (train 1 ep -> valid NDCG@20 "
                        "-> test + best save on improvement -> early stop)",
        }

    # ------------------------------------------------------------- the loop
    def run(self) -> dict:
        """Run the ``model/main.py::__main__`` epoch loop against the frozen SID.

        One optimizer, one dataset per split, one SID. Every epoch re-proves the
        tokenizer and the SID are unchanged before the next epoch starts.
        """
        mods = frozen_modules()
        torch = mods["torch"]

        # re-assert the contract in case a caller touched files in between
        self.provenance = self._verify_provenance("run start")
        self.freeze_report = self.freeze.apply(STAGE_RECOMMENDER)
        self._assert_tokenizer_frozen()
        self._assert_optimizer_scope()
        self._check_frozen("run start")

        train_dataset, train_loader, train_trunc = self._build_loader(
            "train", "train", self.batch_size, True, self.limit_train)
        _, valid_loader, valid_trunc = self._build_loader(
            "valid", "evaluation", self.infer_size, False, self.limit_eval)
        _, test_loader, test_trunc = self._build_loader(
            "test", "evaluation", self.infer_size, False, self.limit_eval)

        routing = verify_sid_routing(
            train_dataset, {"sia": self.sia, "psa": self.psa}, self.sid_tokens)

        os.makedirs(self.run_dir, exist_ok=True)
        best_ckpt = os.path.join(self.run_dir, "best.pth")
        last_ckpt = os.path.join(self.run_dir, "last.pth")

        # The loop starts from a known RNG state. ``model/main.py::__main__``
        # seeds once, before building the model; the draw that matters for
        # comparability between arms is the shuffle order, so the seed is also
        # (re)applied here, after all construction is done.
        mods["tiger_main"].set_seed(self.seed)

        best_ndcg = 0.0
        best_epoch = None
        counter = 0
        n_best_saves = 0
        n_test_evals = 0
        records = []
        started = time.time()

        for epoch in range(self.num_epochs):
            epoch_started = time.time()
            train_loss = mods["tiger_main"].train(
                self.tiger, train_loader, self.optimizer, self.torch_device,
                **self.mounts)
            avg_recalls, avg_ndcgs = mods["tiger_main"].evaluate(
                self.tiger, valid_loader, self.topk_list, self.beam_size,
                self.torch_device)
            ndcg20 = float(avg_ndcgs["NDCG@20"])
            improved = ndcg20 > best_ndcg

            record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "valid_recalls": {k: float(v) for k, v in avg_recalls.items()},
                "valid_ndcgs": {k: float(v) for k, v in avg_ndcgs.items()},
                "ndcg@20": ndcg20,
                "improved": bool(improved),
                "mounts": {"sia": self.mounts["sia"] is not None,
                           "psa": self.mounts["psa"] is not None},
                "elapsed_seconds": round(time.time() - epoch_started, 3),
            }

            if improved:
                best_ndcg = ndcg20
                best_epoch = epoch
                counter = 0
                test_recalls, test_ndcgs = mods["tiger_main"].evaluate(
                    self.tiger, test_loader, self.topk_list, self.beam_size,
                    self.torch_device)
                n_test_evals += 1
                torch.save(self.tiger.state_dict(), best_ckpt)
                torch.save(self.sia_adapter.state_dict(),
                           os.path.join(self.run_dir, "best.sia_adapter.pth"))
                torch.save(self.psa_adapter.state_dict(),
                           os.path.join(self.run_dir, "best.psa_adapter.pth"))
                n_best_saves += 1
                record["test_recalls"] = {k: float(v) for k, v in test_recalls.items()}
                record["test_ndcgs"] = {k: float(v) for k, v in test_ndcgs.items()}
                record["best_ckpt"] = best_ckpt
            else:
                counter += 1
                record["early_stop_counter"] = counter

            record["frozen"] = self._check_frozen(f"epoch {epoch}")
            records.append(record)

            if counter >= self.early_stop:
                break

        torch.save(self.tiger.state_dict(), last_ckpt)
        torch.save(self.sia_adapter.state_dict(),
                   os.path.join(self.run_dir, "last.sia_adapter.pth"))
        torch.save(self.psa_adapter.state_dict(),
                   os.path.join(self.run_dir, "last.psa_adapter.pth"))

        final = self._check_frozen("run end")
        if protocol_best := (best_ckpt if n_best_saves else None):
            write_stage_manifest(
                protocol_best, self.freeze, stage=STAGE_RECOMMENDER,
                extra={"entry": "tiger_ext.fixed_train", "arm": self.arm,
                       "embedding_path": self.embedding_path,
                       "embedding_sha256": self.embedding_sha256,
                       "embedding_dim": self.in_dim,
                       "tokenizer_input_dim": self.tokenizer_input_dim,
                       "sid_path": self.sid_path,
                       "sid_sha256": self.sid_sha256_before,
                       "sid_provenance_mode": self.provenance["mode"],
                       "sid_manifest": self.provenance["manifest"],
                       "tokenizer_ckpt": self.tokenizer_ckpt,
                       "best_epoch": best_epoch,
                       "best_ndcg@20": best_ndcg,
                       "num_epochs_executed": len(records)})

        result = {
            "entry": "tiger_ext.fixed_train",
            "arm": self.arm,
            "dataset": self.dataset,
            "device": self.device,
            "embedding_path": self.embedding_path,
            "embedding_path_explicit": self.embedding_path_explicit,
            "embedding_sha256": self.embedding_sha256,
            "embedding_torch_dtype": str(self.item_embeddings.dtype),
            "in_dim": self.in_dim,
            "tokenizer_input_dim": self.tokenizer_input_dim,
            "sid_path": self.sid_path,
            "sid_sha256": self.sid_sha256_before,
            "sid_sha256_after": final["sid_sha256"],
            "sid_unchanged": final["sid_sha256"] == self.sid_sha256_before,
            "embedding_sha256_after": final["embedding_sha256"],
            "embedding_unchanged": final["embedding_sha256"]
            == self.embedding_sha256,
            "tokenizer_ckpt": self.tokenizer_ckpt,
            "tokenizer_ckpt_sha256": self.tokenizer_load["ckpt_sha256"],
            "sid_provenance": self.provenance,
            "tokenizer_digest_before": self.tokenizer_digest_before,
            "tokenizer_digest_after": final["tokenizer_digest"],
            "tokenizer_unchanged": final["tokenizer_digest"]
            == self.tokenizer_digest_before,
            "tokenizer_frozen": self.freeze_report["groups"]["tokenizer"],
            "adapters": self.freeze_report["groups"]["adapter"],
            "n_items": self.n_items,
            "n_train_sequences": len(train_loader.dataset),
            "n_valid_sequences": len(valid_loader.dataset),
            "n_test_sequences": len(test_loader.dataset),
            "truncated": {"train": train_trunc, "valid": valid_trunc,
                          "test": test_trunc},
            "limit_train": self.limit_train, "limit_eval": self.limit_eval,
            "num_epochs_requested": self.num_epochs,
            "num_epochs_executed": len(records),
            "early_stopped": len(records) < self.num_epochs,
            "best_epoch": best_epoch,
            "best_ndcg@20": best_ndcg,
            "best_ckpt": best_ckpt if n_best_saves else None,
            "last_ckpt": last_ckpt,
            "n_best_saves": n_best_saves,
            "n_test_evals": n_test_evals,
            "mounts": {"sia": self.sia is not None,
                       "psa": self.psa is not None,
                       "sia_constructed": self.sia is not None,
                       "psa_constructed": self.psa is not None,
                       "lambda_sia": self.lambda_sia,
                       "lambda_psa": self.lambda_psa},
            "routing": routing,
            "losses": self.freeze_report["losses"],
            "epochs": records,
            "elapsed_seconds": round(time.time() - started, 3),
        }
        with open(os.path.join(self.run_dir, "run.json"), "w") as fh:
            json.dump({"config": self.run_config(), "result": result}, fh,
                      indent=2, sort_keys=True, default=str)
        return result


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def main(argv=None):
    """``python -m tiger_ext.fixed_train --dataset ... --sid_path ...``.

    One call = one R13 cell on one dataset/seed. ``--limit_train`` /
    ``--limit_eval`` exist for smoke runs and are recorded in ``run.json``.
    """
    import logging

    parser = argparse.ArgumentParser(
        description="P2 -- fixed-SID recommender training (R13 cells #3 / #5)")
    parser.add_argument("--dataset", default="Toys")
    parser.add_argument("--data_root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    parser.add_argument("--embedding_path", default=None,
                        help="item-embedding table to read (default: "
                             "data_root/<dataset>/item_emb.parquet); its width "
                             "must equal the tokenizer checkpoint's in_dim")
    parser.add_argument("--sid_path", required=True)
    parser.add_argument("--tokenizer_ckpt", required=True)
    parser.add_argument("--sid_manifest", default=None,
                        help="SID provenance sidecar (default: <sid_path>.sid.json)")
    parser.add_argument("--allow_unregistered_sid", action="store_true",
                        help="explicitly accept a legacy SID with no manifest "
                             "(recorded in run.json; blocked by default)")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--arm", default=None,
                        help="R13 cell label, e.g. '#3 ETE' or '#5 CSG+ETE'")
    parser.add_argument("--device", default="cuda:0" if _cuda() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED_T5)
    parser.add_argument("--lambda_sia", type=float, default=0.0)
    parser.add_argument("--lambda_psa", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=T5_BATCH)
    parser.add_argument("--infer_size", type=int, default=T5_INFER_SIZE)
    parser.add_argument("--beam_size", type=int, default=T5_BEAM)
    parser.add_argument("--lr", type=float, default=T5_LR)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_eval", type=int, default=None)
    parser.add_argument("--rqvae_config", default=None,
                        help="JSON overrides for the RQ-VAE config; must match "
                             "the checkpoint (default: RQVAE_CONFIG)")
    args = parser.parse_args(argv)
    rqvae_config = None
    if args.rqvae_config:
        try:
            rqvae_config = json.loads(args.rqvae_config)
        except ValueError as exc:
            parser.error(f"--rqvae_config is not valid JSON: {exc}")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    trainer = FixedSidTrainer(
        args.dataset, args.data_root, args.sid_path, args.tokenizer_ckpt,
        args.run_dir, device=args.device, rqvae_config=rqvae_config,
        lambda_sia=args.lambda_sia,
        lambda_psa=args.lambda_psa, seed=args.seed,
        batch_size=args.batch_size, infer_size=args.infer_size,
        beam_size=args.beam_size, lr=args.lr, num_workers=args.num_workers,
        limit_train=args.limit_train, limit_eval=args.limit_eval,
        num_epochs=args.num_epochs, early_stop=args.early_stop, arm=args.arm,
        sid_manifest=args.sid_manifest,
        allow_unregistered_sid=args.allow_unregistered_sid,
        embedding_path=args.embedding_path)
    result = trainer.run()
    summary = {k: result[k] for k in
               ("arm", "dataset", "embedding_path", "in_dim",
                "num_epochs_executed", "best_ndcg@20",
                "best_ckpt", "last_ckpt", "n_best_saves", "n_test_evals",
                "tokenizer_unchanged", "sid_unchanged", "embedding_unchanged")}
    summary["lambda_sia"] = result["mounts"]["lambda_sia"]
    summary["lambda_psa"] = result["mounts"]["lambda_psa"]
    print(json.dumps(summary, indent=2, default=str))
    return 0


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


__all__ = ["FixedSidTrainer", "FixedSidError", "SidProvenanceError",
           "state_digest", "tokenizer_input_dim", "tokenizer_input_shape",
           "verify_sid_routing", "verify_sid_provenance",
           "resolve_sid_manifest", "main", "DEFAULT_RUN_NAME"]


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
