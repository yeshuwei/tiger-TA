"""E9 -- the real phase bodies behind :class:`tiger_ext.alternating.AlternatingScheduler`.

The scheduler owns the loop, the state machine and the hand-off check; this
module owns the four phase bodies, each of which calls *frozen* code:

    tokenizer phase    -> ``rqvae/trainer.py::Trainer.fit``      (1 epoch)
    refresh phase      -> ``tiger_ext.sid_refresh.refresh_sid``  (E8, unchanged)
    recommender phase  -> ``model/main.py::train``               (1 epoch)
    evaluation phase   -> ``model/main.py::evaluate``            (after recommender)

Nothing here re-implements a training step. What it *does* add is the wiring the
plan asks for:

  * **one stage at a time.** Every phase starts with
    ``StageFreeze.apply(stage)`` and its optimizer is built from
    ``StageFreeze.trainable_parameters(stage)``.
  * **the SID hand-off.** Each cycle rebuilds the recommender's dataset, its SIA
    and PSA mounts and its SID->item routing table from *that cycle's* SID file,
    and proves the three agree (dataset offsets == mount lookup == manifest
    hash) before training. A recommender that silently kept reading the previous
    SID cannot pass.
  * **a per-cycle checkpoint chain.** The tokenizer checkpoint, the SID version
    and the recommender checkpoint are all recorded with their sha256.

Two structural facts about this port, both already established by E7/E8, shape
the code below:

  * ``model/main.py::__main__`` builds ``optim.Adam(model.parameters())``, which
    does **not** contain the adapters (they live outside ``TIGER``). E9 therefore
    drives ``model/main.py::train`` directly instead of using that entry point,
    which is also why ``model/main.py`` needs no edit.
  * ``rqvae/generate_code.py`` runs as a *subprocess* under a plain interpreter
    with ``cwd=rqvae/`` (E8's ``run_generator``). The RQ-VAE checkpoint it reads
    must therefore carry a picklable ``args`` object; a plain
    ``argparse.Namespace`` is used so the child can unpickle it without
    importing anything from this package.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from .alternating import (
    CYCLE_LENGTH, DEFAULT_SEED, EVAL_STEP, AlternatingScheduler, phase_seed,
)
from .sid_refresh import parse_sid_name, refresh_sid, sha256_file
from .stage import (
    STAGE_RECOMMENDER, STAGE_TOKENIZER, StageFreeze, load_tokenizer_state_dict,
    write_stage_manifest,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, "model")
RVQAE_DIR = os.path.join(REPO, "rqvae")

# --------------------------------------------------------------------------- #
# Frozen Clean TIGER defaults (clean_tiger/config.sh, the single source of truth)
# --------------------------------------------------------------------------- #
RQVAE_CONFIG = dict(
    lr=1e-3, batch_size=1024, learner="AdamW", lr_scheduler_type="linear",
    weight_decay=1e-4, dropout_prob=0.0, bn=False, loss_type="mse",
    kmeans_init=True, kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.003],
    sk_iters=50, beta=0.25, quant_loss_weight=1.0, save_limit=5,
    num_emb_list=[256, 256, 256], e_dim=32, layers=[512, 256, 128, 64],
)

T5_CONFIG = dict(num_layers=4, num_decoder_layers=4, d_model=128, d_ff=1024,
                 num_heads=6, d_kv=64, dropout_rate=0.1, vocab_size=1025,
                 pad_token_id=0, eos_token_id=0, feed_forward_proj="relu",
                 max_len=20)

#: ``model/main.py``'s own default; argparse's ``type=list`` makes ``--topk_list``
#: unusable, and NDCG@20 is required, so the default is used verbatim.
T5_TOPK = [5, 10, 20]
T5_LR = 1e-4
T5_BATCH = 256
T5_INFER_SIZE = 96
T5_BEAM = 30

#: Per-phase warm-up. The frozen ``Trainer`` rebuilds its optimizer and its
#: linear schedule at every ``__init__``, and a phase is one epoch, so a
#: 50-epoch warm-up (``clean_tiger/config.sh:48``) would exceed the phase's own
#: step budget. E9 runs each phase with **no** warm-up and records the value.
PHASE_WARMUP_EPOCHS = 0

SEED_RQVAE = 2024
SEED_T5 = 2025
NUM_WORKERS = 1          # the project's reproducibility lesson: never leave it at 4


# --------------------------------------------------------------------------- #
# importing the frozen loops
# --------------------------------------------------------------------------- #
_MODULES = {}


def _ensure_paths():
    """Put ``model/`` before ``rqvae/`` (both contain a top-level ``main.py``)."""
    for path in (RVQAE_DIR, MODEL_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)


def frozen_modules() -> dict:
    """Import the frozen training entry points once, in a safe order.

    ``rqvae/datasets.py`` shadows the HuggingFace ``datasets`` package, so the
    RQ-VAE modules are imported first and the resolved ``EmbDataset`` is checked
    to actually come from ``rqvae/``.
    """
    if _MODULES:
        return _MODULES
    _ensure_paths()
    import torch                                     # noqa: F401
    from datasets import EmbDataset
    from models.rqvae import RQVAE
    from trainer import Trainer as RQVAETrainer

    if os.path.dirname(os.path.abspath(sys.modules["datasets"].__file__)) != RVQAE_DIR:
        raise ImportError(
            "`datasets` resolved to "
            f"{sys.modules['datasets'].__file__!r}, not rqvae/datasets.py; "
            "model/ and rqvae/ must precede site-packages on sys.path")

    from dataset import GenRecDataset
    from dataloader import GenRecDataLoader
    import main as tiger_main

    _MODULES.update(torch=torch, EmbDataset=EmbDataset, RQVAE=RQVAE,
                    RQVAETrainer=RQVAETrainer, GenRecDataset=GenRecDataset,
                    GenRecDataLoader=GenRecDataLoader, tiger_main=tiger_main)
    return _MODULES


def rqvae_namespace(data_path, ckpt_dir, device, seed, epochs=1, eval_step=1,
                    batch_size=None, num_workers=NUM_WORKERS, config=None,
                    warmup_epochs=PHASE_WARMUP_EPOCHS):
    """The ``args`` object an RQ-VAE checkpoint carries.

    Read back by ``rqvae/trainer.py::Trainer`` *and* by ``rqvae/generate_code.py``
    (which rebuilds the model from ``args``), so it must carry the full model
    configuration -- not just the training knobs. A plain ``argparse.Namespace``
    keeps it unpicklable-free of this package in the generator subprocess.
    """
    cfg = dict(RQVAE_CONFIG)
    cfg.update(config or {})
    cfg["batch_size"] = int(batch_size if batch_size is not None else cfg["batch_size"])
    return argparse.Namespace(
        data_path=os.path.abspath(data_path),
        ckpt_dir=os.path.abspath(ckpt_dir),
        device=str(device),
        seed=int(seed),
        epochs=int(epochs),
        eval_step=int(eval_step),
        num_workers=int(num_workers),
        warmup_epochs=int(warmup_epochs),
        **cfg,
    )


def _validate_csg(csg):
    """P1: accept C5's hook shape and reject anything the Trainer cannot use.

    The frozen Trainer guards on ``csg.enabled`` and calls
    ``csg(x_q=, indices=, latent=, rows=, rqvae=)``, so the object has to expose
    both. It must also be parameter-free (C5's invariant): a hook with
    parameters would never enter the tokenizer optimizer, which is built from
    ``model.parameters()``, so its weights would silently never train.
    """
    if csg is None:
        return None
    if not hasattr(csg, "enabled"):
        raise TypeError(
            "csg must be a tiger_ext.qspace_train.CSGHook (it exposes `enabled` "
            f"and returns lambda_csg * L_csg); got {type(csg).__name__}")
    params = getattr(csg, "parameters", None)
    if callable(params):
        n_params = sum(1 for _ in params())
        if n_params:
            raise ValueError(
                f"the CSG hook carries {n_params} parameters; it must be "
                "parameter-free, or the tokenizer optimizer would never train them")
    return csg


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #
class TigerAlternatingPipeline:
    """Builds the models once and implements the four E9 phases.

    Args:
        dataset: dataset name, also the SID file-name prefix (E8 convention).
        data_root: directory holding ``<dataset>/{train,valid,test}.parquet`` and
            ``<dataset>/item_emb.parquet``.
        sid_base_dir: where E8 writes ``sid_versions/``. Owned by the caller.
        run_dir: where checkpoints, stage records and cycle manifests go.
        device: torch device string.
        rqvae_config / t5_config: overrides of the frozen Clean TIGER config.
        lambda_sia / lambda_psa: SIA / PSA weights. Both must be > 0 for the
            mounts to be live (``SIAMount.enabled`` / ``PSAMount.enabled``).
        csg: optional C5 hook (``tiger_ext.qspace_train.CSGHook``) attached to
            the **tokenizer** phase only, in exactly the shape C5 attaches it to
            ``rqvae/trainer.py::Trainer``. ``None`` (the default) reproduces the
            pre-P1 call site exactly: no ``return_index`` on the dataset and no
            fourth argument to the Trainer. A hook whose ``lambda_csg == 0`` is
            carried but never invoked (the Trainer guards on ``enabled``), and
            the phase still runs the default path.
        limit_train: optional cap on the number of training sequences (a small
            offline run); ``None`` uses everything.
    """

    def __init__(self, dataset, data_root, sid_base_dir, run_dir, device="cpu",
                 rqvae_config=None, t5_config=None, lambda_sia=0.1, lambda_psa=0.1,
                 seed=SEED_T5, batch_size=T5_BATCH, infer_size=T5_INFER_SIZE,
                 beam_size=T5_BEAM, topk_list=None, lr=T5_LR,
                 num_workers=NUM_WORKERS, limit_train=None, csg=None):
        mods = frozen_modules()
        torch = mods["torch"]

        self.dataset = dataset
        self.data_root = os.path.abspath(data_root)
        self.sid_base_dir = os.path.abspath(sid_base_dir)
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
        self.rqvae_config = dict(RQVAE_CONFIG, **(rqvae_config or {}))
        self.t5_config = dict(T5_CONFIG, **(t5_config or {}))
        self.lambda_sia = float(lambda_sia)
        self.lambda_psa = float(lambda_psa)
        #: P1 -- the tokenizer-phase CSG hook. ``None`` by default, which is
        #: what E9 / E10 / C6 construct, so their call path is unchanged.
        self.csg = _validate_csg(csg)

        self.data_dir = os.path.join(self.data_root, dataset)
        self.embedding_path = os.path.join(self.data_dir, "item_emb.parquet")

        # --- models, built once and carried across the cycles ------------------
        mods["tiger_main"].set_seed(self.seed)
        # kept on the training device: the SIA / PSA mounts register it as a
        # buffer and index it with a batch's target tokens, so a CPU table would
        # break the moment the batch is on CUDA.
        self.item_embeddings = torch.as_tensor(
            mods["EmbDataset"](self.embedding_path).embeddings,
            dtype=torch.float32).to(self.torch_device)
        self.in_dim = int(self.item_embeddings.shape[1])

        cfg = self.rqvae_config
        self.rqvae = mods["RQVAE"](
            in_dim=self.in_dim, num_emb_list=cfg["num_emb_list"], e_dim=cfg["e_dim"],
            layers=cfg["layers"], dropout_prob=cfg["dropout_prob"], bn=cfg["bn"],
            loss_type=cfg["loss_type"], quant_loss_weight=cfg["quant_loss_weight"],
            beta=cfg["beta"], kmeans_init=cfg["kmeans_init"],
            kmeans_iters=cfg["kmeans_iters"], sk_epsilons=cfg["sk_epsilons"],
            sk_iters=cfg["sk_iters"]).to(self.torch_device)

        self.tiger = mods["tiger_main"].TIGER(self.t5_config).to(self.torch_device)

        from .psa import PSAAdapter
        from .sia import SIAAdapter
        self.sia_adapter = SIAAdapter(self.t5_config["d_model"]).to(self.torch_device)
        self.psa_adapter = PSAAdapter(self.t5_config["d_model"],
                                      self.in_dim).to(self.torch_device)

        self.freeze = StageFreeze(self.rqvae, self.tiger,
                                  [self.sia_adapter, self.psa_adapter])
        self.phase_log = []

    # ------------------------------------------------------------------ utils
    def set_seed_for(self, epoch_index):
        mods = frozen_modules()
        seed = phase_seed(self.seed, epoch_index)
        mods["tiger_main"].set_seed(seed)
        return seed

    def run_config(self) -> dict:
        """Everything a re-run needs, for the run manifest.

        P1: the CSG block is added **only** when a hook is attached, so a
        pipeline built without one produces exactly the pre-P1 manifest.
        """
        config = {
            "dataset": self.dataset, "data_root": self.data_root,
            "sid_base_dir": self.sid_base_dir, "run_dir": self.run_dir,
            "device": self.device, "seed": self.seed,
            "batch_size": self.batch_size, "infer_size": self.infer_size,
            "beam_size": self.beam_size, "topk_list": self.topk_list, "lr": self.lr,
            "num_workers": self.num_workers, "limit_train": self.limit_train,
            "lambda_sia": self.lambda_sia, "lambda_psa": self.lambda_psa,
            "rqvae_config": self.rqvae_config, "t5_config": self.t5_config,
            "in_dim": self.in_dim, "n_items": int(self.item_embeddings.shape[0]),
            "phase_warmup_epochs": PHASE_WARMUP_EPOCHS,
            "cycle_length": CYCLE_LENGTH, "eval_step": EVAL_STEP,
        }
        if self.csg is not None:
            constraint = getattr(self.csg, "constraint", None)
            config["csg"] = {
                "enabled": bool(self.csg.enabled),
                "lambda_csg": float(getattr(self.csg, "lambda_csg", 0.0)),
                "stage": "tokenizer",
                "relation_loaded": bool(getattr(constraint, "loaded", False)),
            }
        return config

    def _load_tokenizer(self, ckpt_path):
        """Load an RQ-VAE checkpoint *and* restore its ``vq.initted`` flags."""
        mods = frozen_modules()
        ckpt = mods["torch"].load(ckpt_path, map_location="cpu", weights_only=False)
        return load_tokenizer_state_dict(self.rqvae, ckpt["state_dict"])

    def _sid_loader(self, sid_path):
        """The SID array the dataset and the mounts both key their routing on."""
        return frozen_modules()["torch"].as_tensor(
            np.load(sid_path), dtype=frozen_modules()["torch"].long)

    def _maybe_limit(self, dataset, loader_cls, **kwargs):
        """Optionally cap a training set (offline runs); the SID check uses the full one."""
        torch = frozen_modules()["torch"]
        source = dataset
        if self.limit_train is not None and len(dataset) > self.limit_train:
            source = torch.utils.data.Subset(dataset, list(range(self.limit_train)))
        return loader_cls(source, **kwargs)

    # ------------------------------------------------------- 1. tokenizer phase
    def tokenizer_phase(self, cycle_index, epoch_index, resume_ckpt=None):
        """One RQ-VAE epoch through the frozen ``rqvae/trainer.py``.

        The whole stack is re-frozen to the tokenizer contract first, and the
        optimizer the Trainer builds is *checked* to cover exactly
        ``freeze.trainable_parameters("tokenizer")`` -- the frozen loop owns its
        optimizer (``rqvae/trainer.py:119-120``), so the contract is asserted
        rather than injected.

        P1: when the pipeline was built with a hook, it is passed as the
        Trainer's ``csg=`` and the dataset starts returning row indices. Both
        are required -- the frozen step guards on ``batch_indices is not None``,
        so an enabled hook over an index-less dataset would drop ``L_csg``
        silently. With ``csg=None`` neither happens and the phase runs the
        pre-P1 code path unchanged.
        """
        mods = frozen_modules()
        torch = mods["torch"]
        seed = self.set_seed_for(epoch_index)
        report = self.freeze.apply(STAGE_TOKENIZER)

        if resume_ckpt:
            self._load_tokenizer(resume_ckpt)

        ckpt_dir = os.path.join(self.run_dir, "tokenizer", f"cycle_{cycle_index:03d}")
        args = rqvae_namespace(self.embedding_path, ckpt_dir, self.device, seed,
                               epochs=1, eval_step=1,
                               batch_size=self.rqvae_config["batch_size"],
                               config=self.rqvae_config)

        csg = self.csg
        return_index = bool(csg is not None and csg.enabled)
        data = mods["EmbDataset"](self.embedding_path, return_index=return_index)
        if return_index and not getattr(data, "return_index", return_index):
            raise RuntimeError(
                "the CSG hook is enabled but EmbDataset carries no row indices, "
                "so L_csg would be silently skipped (C5's build_dataset guard)")
        loader = torch.utils.data.DataLoader(
            data, num_workers=self.num_workers,
            batch_size=args.batch_size, shuffle=True, pin_memory=True)

        trainer = mods["RQVAETrainer"](args, self.rqvae, len(loader), csg=csg)
        if csg is not None and trainer.csg is not csg:
            raise RuntimeError(
                "the CSG hook did not reach the trainer; the tokenizer phase "
                "would run without L_csg")

        internal = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
        contract = {id(p) for p in self.freeze.trainable_parameters(STAGE_TOKENIZER)}
        if internal != contract:
            raise AssertionError(
                f"tokenizer optimizer covers {len(internal)} parameters but the "
                f"stage contract has {len(contract)} trainable; "
                f"{len(internal - contract)} unexpected, {len(contract - internal)} missing")

        best_loss, best_collision = trainer.fit(loader)

        ckpt_path = os.path.join(trainer.ckpt_dir, "best_collision_model.pth")
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(f"the tokenizer epoch saved no checkpoint at {ckpt_path}")
        write_stage_manifest(ckpt_path, self.freeze, stage=STAGE_TOKENIZER,
                             extra={"cycle_index": cycle_index,
                                    "epoch_index": epoch_index, "seed": seed,
                                    "best_loss": float(best_loss),
                                    "best_collision_rate": float(best_collision)})

        result = {
            "ckpt_path": os.path.abspath(ckpt_path),
            "sha256": sha256_file(ckpt_path),
            "cycle_index": cycle_index, "epoch_index": epoch_index, "seed": seed,
            "best_loss": float(best_loss),
            "best_collision_rate": float(best_collision),
            "optimizer_params": len(contract),
            "freeze": report["groups"],
            "losses": report["losses"],
        }
        if csg is not None:
            # P1 provenance -- only present when a hook was attached, so the
            # csg=None result dict stays byte-identical to the pre-P1 one.
            result["csg"] = {
                "enabled": bool(csg.enabled),
                "lambda_csg": float(getattr(csg, "lambda_csg", 0.0)),
                "attached": trainer.csg is csg,
                "return_index": return_index,
                "counters": (dict(csg.counters)
                             if hasattr(csg, "counters") else None),
            }
        self.phase_log.append({"phase": "tokenizer", **{k: result[k] for k in
                               ("cycle_index", "epoch_index", "seed", "ckpt_path")}})
        return result

    # --------------------------------------------------------- 2. refresh phase
    def refresh_phase(self, cycle_index, epoch_index, tokenizer_ckpt):
        """E8's ``refresh_sid``, called unchanged; the scheduler reads the rest.

        ``publish`` is left at its default (off): the canonical SID under
        ``data/<D>/`` is the frozen baseline's artifact and the hand-off here is
        the per-cycle ``--code_path``, not a global overwrite.
        """
        record = refresh_sid(self.dataset, tokenizer_ckpt, self.sid_base_dir,
                             device=self.device)
        self.phase_log.append({"phase": "refresh", "cycle_index": cycle_index,
                               "epoch_index": epoch_index,
                               "version": record["version"],
                               "sid_path": record["sid_path"]})
        return {
            "version": record["version"],
            "sid_path": record["sid_path"],
            "sid_sha256": record["sid_sha256"],
            "ckpt_path": record["ckpt_path"],
            "ckpt_sha256": record["ckpt_sha256"],
            "device": record["device"],
            "manifest_path": record["manifest"],
            "diff": record["diff"],
            "sid_stats": record["sid_stats"],
        }

    # ----------------------------------------------------- 3. recommender phase
    def recommender_phase(self, cycle_index, epoch_index, sid_path, sid_version,
                          resume_ckpt=None):
        """One T5 epoch on this cycle's SID, with SIA + PSA mounted."""
        mods = frozen_modules()
        torch = mods["torch"]
        from .psa_loss import PSAMount
        from .sia_loss import SIAMount

        seed = self.set_seed_for(epoch_index)
        report = self.freeze.apply(STAGE_RECOMMENDER)

        sid_tokens = self._sid_loader(sid_path)
        dataset = mods["GenRecDataset"](
            dataset_path=os.path.join(self.data_dir, "train.parquet"),
            code_path=sid_path, mode="train", max_len=self.t5_config["max_len"])
        loader = self._maybe_limit(dataset, mods["GenRecDataLoader"],
                                   batch_size=self.batch_size,
                                   shuffle=True, num_workers=self.num_workers)
        n_sequences = len(loader.dataset)

        sia = SIAMount(self.sia_adapter, self.rqvae, self.item_embeddings, sid_tokens,
                       lambda_sia=self.lambda_sia)
        psa = PSAMount(self.psa_adapter, self.rqvae, self.item_embeddings, sid_tokens,
                       lambda_psa=self.lambda_psa)
        mounts = self.freeze.mounts_for_stage(sia=sia, psa=psa)

        handoff = self.verify_sid_consistency(cycle_index, dataset, sia, psa, sid_path,
                                              sid_tokens)

        if resume_ckpt:
            ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt \
                else ckpt
            self.tiger.load_state_dict(state)

        optimizer = torch.optim.Adam(
            self.freeze.trainable_parameters(STAGE_RECOMMENDER), lr=self.lr)

        trainable = {id(p) for p in self.freeze.trainable_parameters(STAGE_RECOMMENDER)}
        tiger_ids = {id(p) for p in self.tiger.parameters()}
        adapter_ids = {id(p) for p in list(self.sia_adapter.parameters())
                       + list(self.psa_adapter.parameters())}
        if trainable != (tiger_ids | adapter_ids):
            raise AssertionError(
                "the recommender optimizer must cover T5 + adapters and nothing else "
                f"(T5 {len(tiger_ids)}, adapters {len(adapter_ids)}, "
                f"union {len(tiger_ids | adapter_ids)}, optimizer {len(trainable)})")

        train_loss = mods["tiger_main"].train(self.tiger, loader, optimizer,
                                              self.torch_device, **mounts)

        ckpt_path = os.path.join(self.run_dir, "recommender",
                                 f"cycle_{cycle_index:03d}_epoch_{epoch_index:03d}.pth")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"state_dict": self.tiger.state_dict(),
                    "cycle_index": cycle_index, "epoch_index": epoch_index,
                    "sid_path": os.path.abspath(sid_path),
                    "sid_version": sid_version,
                    "sid_sha256": handoff["sid_sha256"]}, ckpt_path)
        write_stage_manifest(ckpt_path, self.freeze, stage=STAGE_RECOMMENDER,
                             extra={"cycle_index": cycle_index, "epoch_index": epoch_index,
                                    "seed": seed, "sid_path": os.path.abspath(sid_path),
                                    "sid_version": sid_version,
                                    "train_loss": float(train_loss)})

        result = {
            "ckpt_path": os.path.abspath(ckpt_path),
            "sha256": sha256_file(ckpt_path),
            "cycle_index": cycle_index, "epoch_index": epoch_index, "seed": seed,
            "train_loss": float(train_loss),
            "sid_path_used": os.path.abspath(sid_path),
            "sid_version_used": int(sid_version),
            "sid_sha256_used": handoff["sid_sha256"],
            "n_train_sequences": n_sequences,
            "n_train_batches": len(loader),
            "mounts": {"sia": mounts["sia"] is not None, "psa": mounts["psa"] is not None},
            "code_path_seen_by_dataset": dataset.code_path,
            "handoff": handoff,
            "freeze": report["groups"],
            "losses": report["losses"],
        }
        self.phase_log.append({"phase": "recommender", "cycle_index": cycle_index,
                               "epoch_index": epoch_index, "seed": seed,
                               "ckpt_path": result["ckpt_path"],
                               "sid_version": sid_version})
        return result

    # ------------------------------------------------------ 4. evaluation phase
    def eval_phase(self, cycle_index, epoch_index, recommender_ckpt, sid_path):
        """Evaluate a saved recommender checkpoint on the cycle's SID.

        Only ever called from ``recommender_done`` (the scheduler enforces it),
        so an evaluation can never land after a tokenizer epoch.
        """
        mods = frozen_modules()
        torch = mods["torch"]
        seed = self.set_seed_for(epoch_index)

        ckpt = torch.load(recommender_ckpt, map_location="cpu", weights_only=False)
        self.tiger.load_state_dict(ckpt["state_dict"])

        dataset = mods["GenRecDataset"](
            dataset_path=os.path.join(self.data_dir, "valid.parquet"),
            code_path=sid_path, mode="evaluation", max_len=self.t5_config["max_len"])
        loader = self._maybe_limit(dataset, mods["GenRecDataLoader"],
                                   batch_size=self.infer_size,
                                   shuffle=False, num_workers=self.num_workers)
        recalls, ndcgs = mods["tiger_main"].evaluate(
            self.tiger, loader, self.topk_list, self.beam_size, self.torch_device)
        result = {"split": "valid", "recalls": recalls, "ndcgs": ndcgs,
                  "n_eval_sequences": len(loader.dataset), "seed": seed,
                  "sid_path": os.path.abspath(sid_path),
                  "sid_sha256": sha256_file(sid_path),
                  "beam_size": self.beam_size, "cycle_index": cycle_index}
        self.phase_log.append({"phase": "evaluation", "cycle_index": cycle_index,
                               "epoch_index": epoch_index, "seed": seed,
                               "ndcg@20": ndcgs.get("NDCG@20")})
        return result

    # ------------------------------------------------------ hand-off evidence
    def verify_sid_consistency(self, cycle_index, dataset, sia, psa, sid_path,
                               sid_tokens):
        """Prove the dataset, both mounts and the file all use *this* SID.

        ``GenRecDataset`` turns item ``i+1`` into ``code + level*256 + 1``
        (``model/dataset.py:84-87``), and both mounts key their SID->row table on
        exactly those offsets. So for every item the dataset knows, the mount
        must map that item's offsets back to row ``i``. A dataset or a mount
        built from a different SID fails here.
        """
        expected = sid_tokens.shape[0]
        lookups = {"sia": sia._lookup, "psa": psa._lookup}
        mismatches = {name: 0 for name in lookups}
        for name, lookup in lookups.items():
            if len(lookup) != expected:
                raise AssertionError(
                    f"{name} lookup has {len(lookup)} entries, SID has {expected} rows")
            for row in range(expected):
                key = tuple(int(t) for t in dataset.item_to_code[row + 1])
                if lookup.get(key) != row:
                    mismatches[name] += 1
        if any(mismatches.values()):
            raise AssertionError(
                f"cycle {cycle_index}: the dataset and the {mismatches} mounts "
                f"disagree about the SID; they were built from different files")
        return {
            "sid_path": os.path.abspath(sid_path),
            "sid_sha256": sha256_file(sid_path),
            "sid_version": parse_sid_name(sid_path)["version"],
            "n_items": expected,
            "n_train_sequences": len(dataset),
            "consistent_items": expected,
            "mismatches": mismatches,
        }


def build_scheduler(pipeline, run_dir, n_cycles, initial_tokenizer_ckpt,
                    initial_recommender_ckpt=None, seed=None, device=None,
                    log_every_cycle=True):
    """A scheduler wired to a real pipeline (the entry point the E9 runner uses)."""
    return AlternatingScheduler(
        run_dir=run_dir, dataset=pipeline.dataset, n_cycles=n_cycles,
        tokenizer_phase=pipeline.tokenizer_phase,
        refresh_phase=pipeline.refresh_phase,
        recommender_phase=pipeline.recommender_phase,
        eval_phase=pipeline.eval_phase,
        seed=pipeline.seed if seed is None else seed,
        device=pipeline.device if device is None else device,
        initial_tokenizer_ckpt=initial_tokenizer_ckpt,
        initial_recommender_ckpt=initial_recommender_ckpt)


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def main(argv=None):
    """``python -m tiger_ext.alt_pipeline --dataset ... --n_cycles 2``.

    Small offline runs (``--limit_train``) exist so the loop can be exercised
    end to end without a full training budget; a production run simply omits it.
    """
    import json
    import logging

    parser = argparse.ArgumentParser(description="E9 alternating training scheduler")
    parser.add_argument("--dataset", default="Toys")
    parser.add_argument("--data_root", default=os.path.join(REPO, "data"))
    parser.add_argument("--sid_base_dir", required=True,
                        help="where E8 writes sid_versions/ (caller-owned)")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--n_cycles", type=int, default=2)
    parser.add_argument("--initial_rqvae_ckpt", required=True)
    parser.add_argument("--initial_t5_ckpt", default=None)
    parser.add_argument("--device", default="cuda:0" if _cuda() else "cpu")
    parser.add_argument("--seed", type=int, default=SEED_T5)
    parser.add_argument("--lambda_sia", type=float, default=0.1)
    parser.add_argument("--lambda_psa", type=float, default=0.1)
    parser.add_argument("--limit_train", type=int, default=None,
                        help="cap the training sequences (offline / smoke runs)")
    parser.add_argument("--dry_run", action="store_true",
                        help="print the phase plan and the resume plan, then stop")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    pipeline = TigerAlternatingPipeline(
        args.dataset, args.data_root, args.sid_base_dir, args.run_dir,
        device=args.device, seed=args.seed, lambda_sia=args.lambda_sia,
        lambda_psa=args.lambda_psa, limit_train=args.limit_train)
    scheduler = build_scheduler(pipeline, args.run_dir, args.n_cycles,
                                args.initial_rqvae_ckpt, args.initial_t5_ckpt)

    from .alternating import render_plan
    print("phase plan:\n" + render_plan(scheduler.plan))
    print("resume plan:\n" + json.dumps(scheduler.resume_plan(), indent=2))
    if args.dry_run:
        return 0

    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, "run.json"), "w") as fh:
        json.dump({"config": pipeline.run_config(),
                   "plan": scheduler.plan,
                   "initial_rqvae_ckpt": args.initial_rqvae_ckpt,
                   "initial_t5_ckpt": args.initial_t5_ckpt}, fh, indent=2,
                  sort_keys=True, default=str)

    records = scheduler.run()
    for record in records:
        print(json.dumps({"cycle": record["cycle_index"], "state": record["state"],
                          "sid_version": (record.get("sid") or {}).get("version"),
                          "ndcgs": (record.get("evaluation") or {}).get("ndcgs"),
                          "errors": [e["kind"] for e in record["errors"]]}, indent=2))
    return 0


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


__all__ = ["TigerAlternatingPipeline", "build_scheduler", "rqvae_namespace",
           "frozen_modules", "main", "RQVAE_CONFIG", "T5_CONFIG",
           "T5_LR", "T5_BATCH", "T5_INFER_SIZE", "T5_BEAM", "T5_TOPK",
           "PHASE_WARMUP_EPOCHS", "SEED_RQVAE", "SEED_T5", "NUM_WORKERS"]
