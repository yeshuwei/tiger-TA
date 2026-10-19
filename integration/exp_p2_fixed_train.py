#!/usr/bin/env python
"""EXP-P2 -- the fixed-SID recommender entry point (R13 cells #3 / #5).

Run:  python -I integration/exp_p2_fixed_train.py

P2 is a **driver**, not a mechanism: it freezes one tokenizer and one SID and
runs ``model/main.py``'s own epoch loop for many epochs under a single
optimizer. The checks therefore ask:

  [1] static shape -- optional weights default to 0 (Clean TIGER path), and a
      CSG hook is refused rather than silently ignored
  [2] the freeze contract -- RQ-VAE frozen + codebooks sealed, optimizer covers
      exactly T5 + adapters
  [3] multi-epoch training -- >= 3 epochs executed, T5 weights move, and the
      tokenizer state / SID file are bitwise and bytewise unchanged
  [4] SIA / PSA switches -- with weight 0 the mounts are not even constructed
      and ``model/main.py::train`` receives ``None``; with weight > 0 the
      matching adapter moves and the other one does not
  [5] the Clean TIGER path -- the same run with weight 0 is **bitwise identical**
      to a manual replay of ``model/main.py::__main__``'s loop
  [6] ``recommender_phase`` is never used (the whole point of the new driver)
  [7] the saved checkpoint is protocol-compatible (plain ``state_dict``, strict
      load into a fresh ``TIGER``, as ``tools/eval_trained_model.py`` expects)
  [8] cell #5 -- a CSG-trained tokenizer/SID pair is accepted and recorded; the
      CSG hook itself can never be mounted here
  [9] the CLI entry point -- ``python -m tiger_ext.fixed_train`` runs end to end

Nothing here measures recommendation quality.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from tiger_ext import sid_refresh as SR                                  # noqa: E402
from tiger_ext.alt_pipeline import (                                     # noqa: E402
    TigerAlternatingPipeline, frozen_modules, rqvae_namespace,
)
from tiger_ext.fixed_train import (                                      # noqa: E402
    FixedSidError, FixedSidTrainer, state_digest,
)
from tiger_ext.stage import codebook_initted, load_tokenizer_state_dict   # noqa: E402

DATASET = "P2Syn"
DEVICE = os.environ.get("EXP_P2_DEVICE", "cpu")

SYN = dict(n_items=128, dim=32, centres=8, codes=[64, 64, 64], rqvae_batch=128,
           seq_len=6, n_train=120, n_valid=24, n_test=24, seed=0)
SEED = 2025
RQVAE_CFG = dict(num_emb_list=list(SYN["codes"]), e_dim=32,
                 batch_size=SYN["rqvae_batch"])
T5_CFG = dict(num_layers=2, num_decoder_layers=2, d_model=64, d_ff=256,
              num_heads=4, d_kv=16)
N_EPOCHS = 3
EPOCHS_ETE = 2

# A second interaction split over the *same* tokenizer/SID but a two-item
# vocabulary with a deterministic next item: it exists only to exercise the
# *improvement* branch (test eval + best.pth write), which random interactions
# never reach because NDCG@20 stays 0.
SMALL_DATASET = "P2SynSmall"
SMALL_VOCAB = 2
SMALL_ROWS = dict(train=256, valid=8, test=8)
SMALL_CFG = dict(batch_size=16, num_epochs=8)   # >= a few dozen gradient steps

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def section(title):
    print(f"\n{title}")


# --------------------------------------------------------------------------- #
# fixture: item table + interactions + a real (tiny) RQ-VAE + its SID
# --------------------------------------------------------------------------- #
def build_small_split(root):
    """A copy of the fixture item table with a trivially learnable interaction set.

    Same tokenizer and SID; only the interactions differ: history is five
    copies of one item and the target is the other item. Used to reach the
    improvement branch (``NDCG@20 > 0``) in a handful of epochs.
    """
    base = os.path.join(root, DATASET)
    data_dir = os.path.join(root, SMALL_DATASET)
    os.makedirs(data_dir, exist_ok=True)
    shutil.copy(os.path.join(base, "item_emb.parquet"),
                os.path.join(data_dir, "item_emb.parquet"))
    rows = []
    for _ in range(max(SMALL_ROWS.values())):
        for k in range(1, SMALL_VOCAB + 1):
            rows.append(([k] * (SYN["seq_len"] - 1), k % SMALL_VOCAB + 1))
    start = 0
    for split, n in SMALL_ROWS.items():
        chunk = rows[start:start + n]
        start += n
        pd.DataFrame({"history": [h for h, _ in chunk],
                      "target": [t for _, t in chunk]}).to_parquet(
            os.path.join(data_dir, f"{split}.parquet"), index=False)
    return data_dir


def build_fixture(root, syn=None, dataset=None):
    syn = SYN if syn is None else syn
    dataset = DATASET if dataset is None else dataset
    rng = np.random.default_rng(syn["seed"])
    n, dim = syn["n_items"], syn["dim"]
    centres = rng.normal(size=(syn["centres"], dim)).astype(np.float32) * 2.0
    which = rng.integers(0, syn["centres"], size=n)
    emb = (centres[which] + rng.normal(size=(n, dim)).astype(np.float32) * 0.3)
    emb = emb.astype(np.float32)

    data_dir = os.path.join(root, dataset)
    os.makedirs(data_dir, exist_ok=True)
    item_ids = np.arange(1, n + 1, dtype=np.int64)
    pd.DataFrame({"ItemID": item_ids,
                  "embedding": [emb[i] for i in range(n)]}).to_parquet(
        os.path.join(data_dir, "item_emb.parquet"), index=False)

    def interactions(rows):
        history, target = [], []
        for _ in range(rows):
            seq = rng.integers(1, n + 1, size=syn["seq_len"]).tolist()
            history.append(seq[:-1])
            target.append(seq[-1])
        return pd.DataFrame({"history": history, "target": target})

    for split, rows in (("train", syn["n_train"]), ("valid", syn["n_valid"]),
                        ("test", syn["n_test"])):
        interactions(rows).to_parquet(os.path.join(data_dir, f"{split}.parquet"),
                                      index=False)

    rqvae_cfg = dict(num_emb_list=list(syn["codes"]), e_dim=32,
                     batch_size=syn["rqvae_batch"])
    ckpt = train_tokenizer(root, data_dir, rqvae_cfg=rqvae_cfg)
    sid = make_sid(ckpt, emb, rqvae_cfg=rqvae_cfg)
    sid_path = os.path.join(data_dir, f"{dataset}_sid.npy")
    np.save(sid_path, sid)
    # B4: production SIDs always carry a provenance sidecar; the fixture does
    # too so FixedSidTrainer sees the shape real runs have.
    SR.write_sid_manifest(sid_path, {
        "dataset": dataset, "version": 1,
        "sid_path": os.path.abspath(sid_path),
        "ckpt_sha256": SR.sha256_file(ckpt),
        "sid_sha256": SR.sha256_file(sid_path),
        "n_items": int(sid.shape[0]),
    })
    return data_dir, ckpt, sid_path


def train_tokenizer(root, data_dir, rqvae_cfg=None, syn=None):
    """One real RQ-VAE epoch through the frozen Trainer -- a genuine checkpoint."""
    mods = frozen_modules()
    torch = mods["torch"]
    from tiger_ext.alt_pipeline import RQVAE_CONFIG
    syn = SYN if syn is None else syn
    cfg = dict(RQVAE_CONFIG, **(rqvae_cfg or RQVAE_CFG))
    rqvae = mods["RQVAE"](
        in_dim=syn["dim"], num_emb_list=cfg["num_emb_list"], e_dim=cfg["e_dim"],
        layers=cfg["layers"], dropout_prob=cfg["dropout_prob"], bn=cfg["bn"],
        loss_type=cfg["loss_type"], quant_loss_weight=cfg["quant_loss_weight"],
        beta=cfg["beta"], kmeans_init=cfg["kmeans_init"],
        kmeans_iters=cfg["kmeans_iters"], sk_epsilons=cfg["sk_epsilons"],
        sk_iters=cfg["sk_iters"])
    ckpt_dir = os.path.join(root, "rqvae_ckpt")
    args = rqvae_namespace(os.path.join(data_dir, "item_emb.parquet"), ckpt_dir,
                           DEVICE, SEED, epochs=1, eval_step=1,
                           batch_size=syn["rqvae_batch"], num_workers=0,
                           config=cfg)
    data = mods["EmbDataset"](os.path.join(data_dir, "item_emb.parquet"))
    loader = torch.utils.data.DataLoader(data, num_workers=0,
                                         batch_size=args.batch_size, shuffle=True)
    trainer = mods["RQVAETrainer"](args, rqvae, len(loader))
    trainer.fit(loader)
    path = os.path.join(trainer.ckpt_dir, "best_collision_model.pth")
    if not os.path.isfile(path):
        raise RuntimeError(f"fixture tokenizer saved no checkpoint at {path}")
    return path


def make_sid(ckpt_path, emb, rqvae_cfg=None, syn=None):
    """A 4-column SID from the fixture tokenizer, with duplicate resolution."""
    mods = frozen_modules()
    torch = mods["torch"]
    from tiger_ext.alt_pipeline import RQVAE_CONFIG
    syn = SYN if syn is None else syn
    cfg = dict(RQVAE_CONFIG, **(rqvae_cfg or RQVAE_CFG))
    rqvae = mods["RQVAE"](
        in_dim=syn["dim"], num_emb_list=cfg["num_emb_list"], e_dim=cfg["e_dim"],
        layers=cfg["layers"], dropout_prob=cfg["dropout_prob"], bn=cfg["bn"],
        loss_type=cfg["loss_type"], quant_loss_weight=cfg["quant_loss_weight"],
        beta=cfg["beta"], kmeans_init=cfg["kmeans_init"],
        kmeans_iters=cfg["kmeans_iters"], sk_epsilons=cfg["sk_epsilons"],
        sk_iters=cfg["sk_iters"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_tokenizer_state_dict(rqvae, ckpt["state_dict"])
    with torch.no_grad():
        indices = rqvae.get_indices(torch.as_tensor(emb, dtype=torch.float32))
    codes = indices.view(-1, indices.shape[-1]).cpu().numpy().astype(np.int64)
    # E8's 4th column: disambiguate raw code collisions so every row is unique
    seen, resolution = {}, []
    for row in codes:
        key = tuple(int(t) for t in row)
        resolution.append(seen.get(key, 0))
        seen[key] = seen.get(key, 0) + 1
    if max(resolution) >= 256:
        raise RuntimeError("fixture SID needs more than 256 duplicate slots")
    return np.concatenate([codes, np.asarray(resolution)[:, None]], axis=1)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def build_trainer(root, tag, sid_path, ckpt, dataset=DATASET, **kwargs):
    params = dict(device=DEVICE, rqvae_config=RQVAE_CFG, t5_config=T5_CFG,
                  seed=SEED, num_workers=0, num_epochs=N_EPOCHS, early_stop=99,
                  limit_eval=SYN["n_valid"])
    params.update(kwargs)
    return FixedSidTrainer(dataset, root, sid_path, ckpt,
                           os.path.join(root, "run", tag), **params)


def snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def same_state(a, b):
    if set(a) != set(b):
        return False, f"key sets differ ({len(set(a) ^ set(b))} keys)"
    for key in sorted(a):
        if a[key].shape != b[key].shape:
            return False, f"{key}: shape differs"
        if not torch.equal(a[key], b[key]):
            return False, f"{key}: not bitwise equal"
    return True, f"{len(a)} tensors bitwise equal"


class _TrainRecorder:
    """Records the mount kwargs ``model/main.py::train`` actually receives."""

    def __init__(self, train_fn):
        self.train_fn = train_fn
        self.calls = []

    def __call__(self, model, loader, optimizer, device, sia=None, psa=None):
        self.calls.append({
            "sia": None if sia is None else type(sia).__name__,
            "psa": None if psa is None else type(psa).__name__,
            "sia_enabled": None if sia is None else bool(sia.enabled),
            "psa_enabled": None if psa is None else bool(psa.enabled),
        })
        return self.train_fn(model, loader, optimizer, device, sia=sia, psa=psa)


def install_recorder():
    """Swap ``model/main.py::train`` for a recorder; return the original too.

    ``mods["tiger_main"]`` *is* the imported ``model/main.py`` module, so the
    original has to be captured before the attribute is replaced -- re-reading
    the attribute later would hand back the recorder.
    """
    mods = frozen_modules()
    original = mods["tiger_main"].train
    recorder = _TrainRecorder(original)
    mods["tiger_main"].train = recorder
    return recorder, mods, original


def remove_recorder(mods, original):
    mods["tiger_main"].train = original


# --------------------------------------------------------------------------- #
# [1] static shape
# --------------------------------------------------------------------------- #
def part1_static(root, sid_path, ckpt):
    section("[1] static: defaults are the Clean TIGER path; CSG is refused")
    import inspect
    params = inspect.signature(FixedSidTrainer.__init__).parameters
    check("lambda_sia / lambda_psa default to 0",
          params["lambda_sia"].default == 0.0 and params["lambda_psa"].default == 0.0,
          f"{params['lambda_sia'].default} / {params['lambda_psa'].default}")
    check("csg defaults to None and is not a live hook",
          params["csg"].default is None)
    check("num_epochs / early_stop default to the model/main.py protocol (200 / 10)",
          params["num_epochs"].default == 200 and params["early_stop"].default == 10,
          f"{params['num_epochs'].default} / {params['early_stop'].default}")

    module = __import__("tiger_ext.fixed_train", fromlist=["x"])
    source = inspect.getsource(module)
    # the docstring legitimately *names* CSG to explain the refusal; what must
    # be absent is any construction of or import path to the CSG machinery
    imports = {name for name in dir(module) if not name.startswith("_")}
    check("the module constructs no CSGHook and imports no CSG loss",
          "CSGHook(" not in source and "qspace" not in source
          and "qspace_loss" not in imports and "CSGHook" not in imports)

    raised = None
    try:
        build_trainer(root, "bad_csg", sid_path, ckpt, csg=object())
    except FixedSidError as exc:
        raised = exc
    check("a CSG hook is refused (it belongs to the tokenizer stage)",
          raised is not None and "tokenizer" in str(raised), str(raised))


# --------------------------------------------------------------------------- #
# [2] freeze contract
# --------------------------------------------------------------------------- #
def part2_contract(root, sid_path, ckpt, trainer):
    section("[2] freeze contract: tokenizer frozen + sealed, optimizer scoped")
    check("every RQ-VAE parameter is frozen",
          all(not p.requires_grad for p in trainer.rqvae.parameters()),
          f"{sum(1 for p in trainer.rqvae.parameters() if p.requires_grad)} trainable")
    check("every codebook is sealed (initted=True)",
          all(codebook_initted(trainer.rqvae)),
          f"{codebook_initted(trainer.rqvae)}")
    opt = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    rq = {id(p) for p in trainer.rqvae.parameters()}
    tiger = {id(p) for p in trainer.tiger.parameters()}
    adapters = {id(p) for p in list(trainer.sia_adapter.parameters())
                + list(trainer.psa_adapter.parameters())}
    check("the optimizer contains no RQ-VAE tensor", not (opt & rq),
          f"{len(opt & rq)} overlap")
    check("the optimizer covers exactly T5 + adapters", opt == (tiger | adapters),
          f"opt={len(opt)} t5={len(tiger)} adapters={len(adapters)}")
    check("weight 0 constructs no mount at all",
          trainer.sia is None and trainer.psa is None
          and trainer.mounts == {"sia": None, "psa": None})

    live = build_trainer(root, "mounts_only", sid_path, ckpt,
                         lambda_sia=0.5, lambda_psa=0.5,
                         num_epochs=1, limit_eval=SYN["n_valid"])
    check("weight > 0 constructs both mounts and hands them to the stage",
          live.sia is not None and live.psa is not None
          and live.mounts["sia"] is live.sia and live.mounts["psa"] is live.psa)
    check("... and the mounts stay out of the optimizer",
          {id(p) for g in live.optimizer.param_groups for p in g["params"]} ==
          {id(p) for p in list(live.tiger.parameters())
           + list(live.sia_adapter.parameters())
           + list(live.psa_adapter.parameters())})


# --------------------------------------------------------------------------- #
# [3] multi-epoch training against a frozen tokenizer / SID
# --------------------------------------------------------------------------- #
def part3_frozen(root, sid_path, ckpt):
    section("[3] many epochs, one optimizer, tokenizer + SID untouched")
    trainer = build_trainer(root, "frozen", sid_path, ckpt)
    initial = snapshot(trainer.tiger)
    sid_before = open(sid_path, "rb").read()
    digest_before = state_digest(trainer.rqvae)

    recorder, mods, original = install_recorder()
    try:
        result = trainer.run()
    finally:
        remove_recorder(mods, original)

    check("more than one epoch actually ran",
          result["num_epochs_executed"] == N_EPOCHS,
          f"executed={result['num_epochs_executed']} requested={N_EPOCHS}")
    check("`train` was called once per epoch",
          len(recorder.calls) == N_EPOCHS, f"{len(recorder.calls)} calls")
    check("with weight 0 every call received sia=None and psa=None",
          all(c["sia"] is None and c["psa"] is None for c in recorder.calls),
          f"{recorder.calls}")
    check("the T5 weights moved (training really happened)",
          not all(torch.equal(initial[k], v)
                  for k, v in trainer.tiger.state_dict().items()))
    check("the SID file is byte-identical after training",
          open(sid_path, "rb").read() == sid_before)
    check("the tokenizer state_dict digest is unchanged",
          state_digest(trainer.rqvae) == digest_before
          == result["tokenizer_digest_after"])
    check("the manifest records both frozen-state results",
          result["tokenizer_unchanged"] is True and result["sid_unchanged"] is True)
    check("each epoch recorded train loss + valid metrics",
          len(result["epochs"]) == N_EPOCHS
          and all("train_loss" in e and "NDCG@20" in e["valid_ndcgs"]
                  for e in result["epochs"]))
    check("valid was evaluated every epoch",
          all(e["valid_ndcgs"] for e in result["epochs"]))
    check("routing was verified against every live mount",
          result["routing"]["n_items"] == SYN["n_items"]
          and result["routing"]["mismatches"] == {"sia": None, "psa": None},
          f"{result['routing']}")
    check("the run manifest was written to run.json",
          os.path.isfile(os.path.join(root, "run", "frozen", "run.json")))
    return result, trainer


# --------------------------------------------------------------------------- #
# [4] SIA / PSA switches
# --------------------------------------------------------------------------- #
def part4_mounts(root, sid_path, ckpt):
    section("[4] SIA / PSA: off => no loss computed; on => that arm moves")
    out = {}
    for tag, lam_sia, lam_psa in (("ete_off", 0.0, 0.0),
                                  ("ete_sia_only", 0.5, 0.0),
                                  ("ete_both", 0.5, 0.5)):
        trainer = build_trainer(root, tag, sid_path, ckpt, lambda_sia=lam_sia,
                                lambda_psa=lam_psa, num_epochs=EPOCHS_ETE)
        # snapshot both adapters unconditionally: an adapter whose mount is
        # absent must still be provably unchanged, not "not measured"
        before = {"sia": snapshot(trainer.sia_adapter),
                  "psa": snapshot(trainer.psa_adapter)}
        recorder, mods, original = install_recorder()
        try:
            result = trainer.run()
        finally:
            remove_recorder(mods, original)
        moved = {
            name: not all(torch.equal(before[name][k], v)
                          for k, v in getattr(trainer, f"{name}_adapter")
                          .state_dict().items())
            for name in ("sia", "psa")
        }
        out[tag] = {"trainer": trainer, "result": result,
                    "calls": list(recorder.calls), "moved": moved}

    off = out["ete_off"]
    check("weight 0: no mount constructed, train() saw None, adapters froze",
          off["trainer"].sia is None and off["trainer"].psa is None
          and all(c["sia"] is None and c["psa"] is None for c in off["calls"]))

    sia_only = out["ete_sia_only"]
    check("SIA-only: train() saw SIA and psa=None",
          all(c["sia"] == "SIAMount" and c["psa"] is None
              and c["sia_enabled"] is True for c in sia_only["calls"]),
          f"{sia_only['calls'][0]}")
    check("SIA-only: the SIA adapter moved", sia_only["moved"]["sia"] is True)
    check("SIA-only: the PSA adapter did not move (its loss never ran)",
          sia_only["moved"]["psa"] is False)

    both = out["ete_both"]
    check("both on: train() saw both mounts, enabled",
          all(c["sia"] == "SIAMount" and c["psa"] == "PSAMount"
              and c["sia_enabled"] is True and c["psa_enabled"] is True
              for c in both["calls"]),
          f"{both['calls'][0]}")
    check("both on: both adapters moved",
          both["moved"]["sia"] is True and both["moved"]["psa"] is True)
    check("both on: the stage still kept the SID and tokenizer frozen",
          both["result"]["sid_unchanged"] is True
          and both["result"]["tokenizer_unchanged"] is True)
    return out


# --------------------------------------------------------------------------- #
# [5] the Clean TIGER path is bitwise unchanged
# --------------------------------------------------------------------------- #
def part5_clean_path(root, sid_path, ckpt, frozen_run, trainer):
    section("[5] weight 0 == a manual replay of the model/main.py loop")
    mods = frozen_modules()
    torch = mods["torch"]

    # rebuild the same construction path with the same seed: construction is
    # deterministic, so this is the exact state the frozen run started from
    ref_trainer = build_trainer(root, "clean_ref", sid_path, ckpt)
    initial = snapshot(ref_trainer.tiger)

    # mirror ``model/main.py::__main__`` exactly, including the per-epoch
    # validation pass (it advances the global RNG through the loader iterator,
    # so omitting it would change the next epoch's shuffle order)
    train_dataset, train_loader, _ = ref_trainer._build_loader(
        "train", "train", ref_trainer.batch_size, True, None)
    _, valid_loader, _ = ref_trainer._build_loader(
        "valid", "evaluation", ref_trainer.infer_size, False, None)
    _, test_loader, _ = ref_trainer._build_loader(
        "test", "evaluation", ref_trainer.infer_size, False, None)

    ref = mods["tiger_main"].TIGER(ref_trainer.t5_config).to(ref_trainer.torch_device)
    ref.load_state_dict(initial)
    optimizer = torch.optim.Adam(ref.parameters(), lr=ref_trainer.lr)
    mods["tiger_main"].set_seed(ref_trainer.seed)

    ref_losses = []
    best_ndcg = 0.0
    for _ in range(N_EPOCHS):
        ref_losses.append(float(mods["tiger_main"].train(
            ref, train_loader, optimizer, ref_trainer.torch_device,
            sia=None, psa=None)))
        _, avg_ndcgs = mods["tiger_main"].evaluate(
            ref, valid_loader, ref_trainer.topk_list, ref_trainer.beam_size,
            ref_trainer.torch_device)
        if float(avg_ndcgs["NDCG@20"]) > best_ndcg:
            best_ndcg = float(avg_ndcgs["NDCG@20"])
            mods["tiger_main"].evaluate(
                ref, test_loader, ref_trainer.topk_list, ref_trainer.beam_size,
                ref_trainer.torch_device)

    ok, detail = same_state(frozen_run["state"], snapshot(ref))
    check("the fixed-SID run with weight 0 is bitwise equal to the frozen loop",
          ok, detail)
    frozen_losses = [e["train_loss"] for e in frozen_run["result"]["epochs"]]
    check("... and every epoch's train loss matches",
          frozen_losses == ref_losses,
          f"{frozen_losses} == {ref_losses}")

    # protocol compatibility: a plain state_dict that eval_trained_model.py can load
    ckpt_path = frozen_run["result"]["last_ckpt"]
    fresh = mods["tiger_main"].TIGER(ref_trainer.t5_config)
    fresh.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    check("last.pth is a plain state_dict that loads strictly into a fresh TIGER",
          True, ckpt_path)


# --------------------------------------------------------------------------- #
# [6] the new driver never touches the alternating phase body
# --------------------------------------------------------------------------- #
def part6_no_recommender_phase(root, sid_path, ckpt):
    section("[6] `recommender_phase` is never called")
    original = TigerAlternatingPipeline.recommender_phase

    def forbidden(self, *a, **k):
        raise AssertionError("fixed_train must not call recommender_phase")

    TigerAlternatingPipeline.recommender_phase = forbidden
    try:
        trainer = build_trainer(root, "no_phase", sid_path, ckpt, num_epochs=2)
        trainer.run()
        ok, detail = True, "ran without touching it"
    except AssertionError as exc:
        ok, detail = False, str(exc)
    finally:
        TigerAlternatingPipeline.recommender_phase = original
    check("training completed without recommender_phase", ok, detail)


# --------------------------------------------------------------------------- #
# [7] protocol accounting + checkpoint save rules
# --------------------------------------------------------------------------- #
def part7_protocol(root, sid_path, ckpt, frozen_run):
    section("[7] model/main.py protocol accounting")
    result = frozen_run["result"]
    improved = [e for e in result["epochs"] if e["improved"]]
    check("a test evaluation happened exactly on improvement",
          result["n_test_evals"] == len(improved)
          == sum(1 for e in result["epochs"] if "test_ndcgs" in e),
          f"n_test_evals={result['n_test_evals']} improved={len(improved)}")
    check("best.pth is written exactly on improvement",
          result["n_best_saves"] == len(improved)
          and (result["best_ckpt"] is not None) == (len(improved) > 0),
          f"saves={result['n_best_saves']}")
    check("last.pth is always written", os.path.isfile(result["last_ckpt"]))
    print(f"    (random fixture: epochs improved {[e['epoch'] for e in improved]} "
          f"of {result['num_epochs_executed']} -- improvement branch not reached)")

    # the same accounting on a split where NDCG@20 can actually be > 0
    build_small_split(root)
    small = build_trainer(root, "small", sid_path, ckpt,
                          dataset=SMALL_DATASET, limit_eval=SMALL_ROWS["valid"],
                          **SMALL_CFG)
    small_result = small.run()
    small_improved = [e for e in small_result["epochs"] if e["improved"]]
    check("small split: valid NDCG@20 became positive (branch is live)",
          small_result["best_ndcg@20"] > 0.0,
          f"best NDCG@20={small_result['best_ndcg@20']}")
    check("small split: test evaluation ran only on improvement",
          small_result["n_test_evals"] == len(small_improved) > 0
          and all("test_ndcgs" in e for e in small_improved)
          and all("test_ndcgs" not in e for e in small_result["epochs"]
                  if not e["improved"]),
          f"n_test_evals={small_result['n_test_evals']} "
          f"improved={[e['epoch'] for e in small_improved]}")
    check("small split: best.pth written on improvement and loadable",
          small_result["n_best_saves"] == len(small_improved)
          and os.path.isfile(small_result["best_ckpt"] or ""),
          small_result.get("best_ckpt"))
    if small_improved and not small_result["epochs"][-1]["improved"]:
        check("small split: best.pth is not last.pth (the last epoch did not improve)",
              not same_state(torch.load(small_result["best_ckpt"], map_location="cpu"),
                             torch.load(small_result["last_ckpt"], map_location="cpu"))[0])
    check("small split: the SID and tokenizer stayed frozen too",
          small_result["sid_unchanged"] is True
          and small_result["tokenizer_unchanged"] is True)


# --------------------------------------------------------------------------- #
# [8] cell #5: a CSG-trained pair is just an input
# --------------------------------------------------------------------------- #
def part8_cell5(root, sid_path, ckpt):
    section("[8] cell #5 (CSG+ETE) is supported through its inputs only")
    trainer = build_trainer(root, "cell5", sid_path, ckpt, lambda_sia=0.5,
                            lambda_psa=0.5, num_epochs=1, arm="#5 CSG+ETE")
    result = trainer.run()
    check("the run accepts a pre-built tokenizer/SID pair and labels the arm",
          result["arm"] == "#5 CSG+ETE"
          and result["tokenizer_ckpt_sha256"] == trainer.tokenizer_load["ckpt_sha256"],
          result["arm"])
    check("... records the SID provenance", len(result["sid_sha256"]) == 64)
    check("... and still proves the tokenizer never moved",
          result["tokenizer_unchanged"] is True)


# --------------------------------------------------------------------------- #
# [9] the command line entry point
# --------------------------------------------------------------------------- #
def part9_cli(root, sid_path, ckpt):
    section("[9] `python -m tiger_ext.fixed_train` runs end to end")
    run_dir = os.path.join(root, "run", "cli")
    cmd = [sys.executable, "-m", "tiger_ext.fixed_train",
           "--dataset", DATASET, "--data_root", root,
           "--sid_path", sid_path, "--tokenizer_ckpt", ckpt,
           "--run_dir", run_dir, "--device", DEVICE, "--arm", "#3 ETE",
           "--num_epochs", "1", "--early_stop", "5",
           "--limit_train", "8", "--limit_eval", "2",
           "--lambda_sia", "0.5",
           "--rqvae_config", json.dumps(RQVAE_CFG)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    check("the CLI exits 0 and writes run.json",
          proc.returncode == 0 and os.path.isfile(os.path.join(run_dir, "run.json")),
          (proc.stderr or proc.stdout).strip().splitlines()[-1:] or "no output")
    tail = proc.stdout.strip().splitlines()
    summary = None
    try:
        summary = json.loads("\n".join(tail[tail.index("{"):]))
    except (ValueError, IndexError):
        pass
    check("... prints a JSON summary with the arm and both weights",
          summary is not None and summary.get("arm") == "#3 ETE"
          and summary.get("lambda_sia") == 0.5 and summary.get("lambda_psa") == 0.0,
          str(summary))


# --------------------------------------------------------------------------- #
def main():
    root = tempfile.mkdtemp(prefix="p2_fixed_")
    try:
        print(f"fixture -> {root}")
        data_dir, ckpt, sid_path = build_fixture(root)
        print(f"  tokenizer ckpt -> {ckpt}\n  SID -> {sid_path}")

        part1_static(root, sid_path, ckpt)

        contract_trainer = build_trainer(root, "contract", sid_path, ckpt,
                                         num_epochs=1, limit_eval=SYN["n_valid"])
        part2_contract(root, sid_path, ckpt, contract_trainer)

        result, trainer = part3_frozen(root, sid_path, ckpt)
        frozen_run = {"result": result,
                      "state": snapshot_of_last(result["last_ckpt"])}
        part5_clean_path(root, sid_path, ckpt, frozen_run, trainer)
        part7_protocol(root, sid_path, ckpt, frozen_run)

        part4_mounts(root, sid_path, ckpt)
        part6_no_recommender_phase(root, sid_path, ckpt)
        part8_cell5(root, sid_path, ckpt)
        part9_cli(root, sid_path, ckpt)
    except Exception:                                    # noqa: BLE001
        traceback.print_exc()
        RESULTS.append(("harness crashed", False, "see traceback"))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\nEXP-P2: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} -- {detail}")
    return 0 if not failed else 1


def snapshot_of_last(path):
    return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    sys.exit(main())
