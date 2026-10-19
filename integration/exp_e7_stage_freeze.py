#!/usr/bin/env python
"""EXP-E7 - tokenizer / recommender freeze interface (Task E7, type A).

Covers the E7 mechanism: the stage contract, the ``requires_grad`` switch, the
loss switches, the (parameter state x loss state) consistency rule, the
``vq.initted`` trap, the checkpoint stage record, and short-range runs of *both*
real training loops under their stage. It does not train anything to
convergence and produces no Recall / NDCG -- that is EXP-E7-E9 (type B/C).

Both frozen sides are exercised through their unmodified loops:
  * tokenizer stage -> ``rqvae/trainer.py::Trainer.fit`` on a 2-batch subset
  * recommender stage -> ``model/main.py::train`` with SIA + PSA mounted

Read-only with respect to the repository: reads files, writes only into a temp
dir. Run with the repo root as cwd:

    python -I integration/exp_e7_stage_freeze.py                  # CPU
    EXP_E7_DEVICE=cuda python -I integration/exp_e7_stage_freeze.py
"""
import argparse
import copy
import glob
import inspect
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO, "model")
RVQAE_DIR = os.path.join(REPO, "rqvae")
SMOKE = os.path.join(REPO, "clean_tiger", "_smoke", "Toys")
CKPT = os.path.join(REPO, "clean_tiger", "ckpt", "Toys", "rqvae",
                    "Oct-14-2026_05-55-09", "best_collision_model.pth")
BASELINE_COMMIT = "5edfca0"
SEED = 2025
BATCH = 8
LAMBDA_SIA = 0.1
LAMBDA_PSA = 0.1
DEVICE = os.environ.get("EXP_E7_DEVICE", "cpu")

RQVAE_CFG = dict(num_emb_list=[256, 256, 256], e_dim=32, layers=[512, 256, 128, 64],
                 dropout_prob=0.0, bn=False, loss_type="mse", quant_loss_weight=1.0,
                 beta=0.25, kmeans_init=True, kmeans_iters=100,
                 sk_epsilons=[0.0, 0.0, 0.003], sk_iters=50)
T5CFG = {'num_layers': 4, 'num_decoder_layers': 4, 'd_model': 128, 'd_ff': 1024,
         'num_heads': 6, 'd_kv': 64, 'dropout_rate': 0.1, 'vocab_size': 1025,
         'pad_token_id': 0, 'eos_token_id': 0, 'feed_forward_proj': 'relu',
         'max_len': 20}

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def as_modules(obj):
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def snap(obj):
    """Ordered clones of every parameter of the given module(s)."""
    out = []
    for m in as_modules(obj):
        out.extend(p.detach().clone() for p in m.parameters())
    return out


def unchanged(before, obj):
    now = snap(obj)
    return len(before) == len(now) and all(torch.equal(a, b) for a, b in zip(before, now))


def n_changed(before, obj):
    now = snap(obj)
    return sum(1 for a, b in zip(before, now) if not torch.equal(a, b))


def flags(module_or_objs):
    out = []
    for m in as_modules(module_or_objs):
        out.extend(bool(p.requires_grad) for p in m.parameters())
    return out


def codebooks(rqvae, clone=True):
    out = []
    for vq in rqvae.rq.vq_layers:
        w = vq.embedding.weight.detach()
        out.append(w.clone() if clone else w)
    return out


def codes_equal(a, b):
    # .cpu() only: checkpoints load on CPU while the model may be on CUDA, and
    # torch.equal refuses mixed devices. Values are compared bitwise either way.
    return len(a) == len(b) and all(torch.equal(x.detach().cpu(), y.detach().cpu())
                                    for x, y in zip(a, b))


def contract_violations(contract, loss_requires):
    """Static check of a (stage -> trainable/losses) contract.

    Mirrors E7 acceptance ⑤: an active loss whose group is frozen in the same
    stage is a mismatch. Used both on the real contract and on a corrupted copy,
    so the check itself is shown to have teeth.
    """
    bad = []
    for stage, spec in contract.items():
        for name, active in spec["losses"].items():
            if not active:
                continue
            for group in loss_requires[name]:
                if group not in spec["trainable"]:
                    bad.append((stage, name, group))
    return bad


class RQVaeArgs:
    """The attribute set rqvae/trainer.py::Trainer reads, with 1-epoch budgets."""

    def __init__(self, ckpt_dir, device, num_emb_list):
        self.lr = 1e-3
        self.learner = "AdamW"
        self.lr_scheduler_type = "linear"
        self.weight_decay = 1e-4
        self.epochs = 1
        self.warmup_epochs = 0
        self.save_limit = 1
        self.eval_step = 1
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.num_emb_list = num_emb_list
        self.lambda_trans = 0.0
        self.lambda_balance = 0.0
        self.tau = 0.1
        self.trans_graph_path = None
        self.align_weight = 0.0
        self.align_data_path = None
        self.align_tau = 0.1


def build_pristine(tmpdir):
    """Pre-E1..E7 rqvae/models package and model/main.py, straight from git."""
    pkg = os.path.join(tmpdir, "pristine_rqvae", "pristine_models")
    os.makedirs(pkg, exist_ok=True)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    for fn in ("rq.py", "vq.py", "rqvae.py", "layers.py"):
        src = subprocess.run(["git", "-C", REPO, "show",
                              f"{BASELINE_COMMIT}:rqvae/models/{fn}"],
                             capture_output=True, text=True, check=True).stdout
        with open(os.path.join(pkg, fn), "w") as f:
            f.write(src)
    src = subprocess.run(["git", "-C", REPO, "show", f"{BASELINE_COMMIT}:model/main.py"],
                         capture_output=True, text=True, check=True).stdout
    with open(os.path.join(tmpdir, "pristine_main.py"), "w") as f:
        f.write(src)
    sys.path.insert(0, os.path.join(tmpdir, "pristine_rqvae"))
    sys.path.insert(0, tmpdir)
    from pristine_models.rqvae import RQVAE as PristineRQVAE
    import pristine_main
    return PristineRQVAE, pristine_main


def main():
    # order matters: model/ and rqvae/ both contain a top-level `main.py`
    sys.path.insert(0, RVQAE_DIR)
    sys.path.insert(0, MODEL_DIR)
    sys.path.insert(0, REPO)
    from models.rqvae import RQVAE
    from datasets import EmbDataset
    from dataset import GenRecDataset
    from dataloader import GenRecDataLoader
    from trainer import Trainer as RQVAETrainer
    import main as current_main
    from tiger_ext.sia import SIAAdapter
    from tiger_ext.sia_loss import SIAMount
    from tiger_ext.psa import PSAAdapter
    from tiger_ext.psa_loss import PSAMount
    from tiger_ext import stage as S

    device = torch.device(DEVICE)
    tmpdir = tempfile.mkdtemp(prefix="exp_e7_")
    PristineRQVAE, pristine_main = build_pristine(tmpdir)

    sid_path = os.path.join(SMOKE, "Toys_clean_tiger_sid.npy")
    emb_ds = EmbDataset(os.path.join(SMOKE, "data", "Toys", "item_emb.parquet"))
    item_embs = torch.as_tensor(emb_ds.embeddings, dtype=torch.float32)
    sid_tokens = torch.as_tensor(np.load(sid_path), dtype=torch.long)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)

    print(f"\nRQ-VAE ckpt: in_dim={emb_ds.dim} | items={len(emb_ds)} | "
          f"SID {tuple(sid_tokens.shape)} | device={device}")

    # a real batch, for the recommender-side runs
    ds = GenRecDataset(dataset_path=os.path.join(SMOKE, "data", "Toys", "train.parquet"),
                       code_path=sid_path, mode='train', max_len=T5CFG['max_len'])
    batch = next(iter(GenRecDataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)))
    input_ids = batch['history'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['target'].to(device)

    # two models, so the group separation is observable: one real RQ-VAE and one
    # real TIGER, each with its own adapters
    def new_rqvae(load=False):
        m = RQVAE(in_dim=emb_ds.dim, **RQVAE_CFG).to(device)
        if load:
            m.load_state_dict(ck["state_dict"])
        return m

    current_main.set_seed(SEED)
    tiger = current_main.TIGER(T5CFG).to(device)
    rqvae = new_rqvae(load=True)
    sia_adapter = SIAAdapter(T5CFG['d_model']).to(device)
    psa_adapter = PSAAdapter(T5CFG['d_model'], emb_ds.dim).to(device)

    ctrl = S.StageFreeze(rqvae, tiger, [sia_adapter, psa_adapter])

    # ==================================================== 1. contract and API
    print("\n[1] stage contract and API")
    check("stage names are exactly ('tokenizer', 'recommender')",
          S.STAGES == ("tokenizer", "recommender"), f"{S.STAGES}")
    ok = False
    try:
        ctrl.apply("tokeniser")
    except S.StageError:
        ok = True
    check("an unknown stage name is rejected", ok)
    ok = False
    try:
        ctrl.loss_flags("eval")
    except S.StageError:
        ok = True
    check("loss_flags rejects an unknown stage", ok)

    ok = False
    try:
        # a mount carries the RQ-VAE as well as the adapter, so registering one
        # would put the tokenizer in the adapter group as well
        S.StageFreeze(rqvae, tiger, [SIAMount(sia_adapter, rqvae, item_embs, sid_tokens)])
    except S.StageError as exc:
        ok = "more than one stage group" in str(exc)
    check("registering a mount (which also holds the RQ-VAE) is refused as ambiguous", ok)

    check("the frozen contract is self-consistent (no active loss over a frozen group)",
          contract_violations(S.STAGE_CONTRACT, S.LOSS_REQUIRES) == [],
          f"violations={contract_violations(S.STAGE_CONTRACT, S.LOSS_REQUIRES)}")
    broken = copy.deepcopy(S.STAGE_CONTRACT)
    broken["tokenizer"]["trainable"] = (S.GROUP_RECOMMENDER,)
    bad = contract_violations(broken, S.LOSS_REQUIRES)
    check("that self-consistency rule fires on a corrupted contract (has teeth)",
          len(bad) > 0, f"e.g. {bad[:2]}")

    # =============================================== 2. requires_grad per stage
    print("\n[2] requires_grad per stage")
    ctrl.apply("tokenizer")
    check("tokenizer stage: RQ-VAE fully trainable", all(flags(rqvae)), f"{sum(flags(rqvae))}/{len(flags(rqvae))}")
    check("tokenizer stage: T5 fully frozen", set(flags(tiger)) == {False},
          f"{sum(flags(tiger))} trainable of {len(flags(tiger))}")
    check("tokenizer stage: adapters frozen",
          set(flags([sia_adapter, psa_adapter])) == {False})
    rep_tok = ctrl.report("tokenizer")
    check("report is complete (trainable + frozen == total for every group)",
          all(g["trainable_tensors"] + g["frozen_tensors"] == g["tensors"]
              for g in rep_tok["groups"].values()),
          f"RQ-VAE {rep_tok['groups'][S.GROUP_TOKENIZER]['tensors']} tensors, "
          f"T5 {rep_tok['groups'][S.GROUP_RECOMMENDER]['tensors']}, "
          f"adapter {rep_tok['groups'][S.GROUP_ADAPTER]['tensors']}")

    ctrl.apply("recommender")
    check("recommender stage: RQ-VAE fully frozen", set(flags(rqvae)) == {False},
          f"{sum(flags(rqvae))} trainable of {len(flags(rqvae))}")
    check("recommender stage: T5 fully trainable", all(flags(tiger)),
          f"{sum(flags(tiger))}/{len(flags(tiger))}")
    check("recommender stage: adapters trainable", all(flags([sia_adapter, psa_adapter])))

    sequence = []
    for st in ("tokenizer", "recommender", "tokenizer", "recommender"):
        ctrl.apply(st)
        sequence.append((flags(rqvae), flags(tiger), flags([sia_adapter, psa_adapter])))
    check("switching back and forth is deterministic (no drift)",
          sequence[0] == sequence[2] and sequence[1] == sequence[3])

    # ====================================================== 3. loss switches
    print("\n[3] loss switches per stage")
    check("tokenizer stage: only L_RQ-VAE is active",
          ctrl.loss_flags("tokenizer") == {"L_RQVAE": True, "L_T5": False,
                                           "L_SIA": False, "L_PSA": False},
          f"{ctrl.loss_flags('tokenizer')}")
    check("recommender stage: L_T5 / L_SIA / L_PSA active, L_RQ-VAE not",
          ctrl.loss_flags("recommender") == {"L_RQVAE": False, "L_T5": True,
                                             "L_SIA": True, "L_PSA": True},
          f"{ctrl.loss_flags('recommender')}")

    sia_mount = SIAMount(sia_adapter, rqvae, item_embs, sid_tokens,
                         lambda_sia=LAMBDA_SIA).to(device)
    psa_mount = PSAMount(psa_adapter, rqvae, item_embs, sid_tokens,
                         lambda_psa=LAMBDA_PSA).to(device)
    kw_tok = ctrl.mounts_for_stage(sia=sia_mount, psa=psa_mount, stage="tokenizer")
    check("tokenizer stage resets both loss switches to None even when mounts are given",
          kw_tok == {"sia": None, "psa": None}, f"{ {k: type(v).__name__ for k, v in kw_tok.items()} }")
    kw_rec = ctrl.mounts_for_stage(sia=sia_mount, psa=psa_mount, stage="recommender")
    check("recommender stage passes the mounts through unchanged",
          kw_rec["sia"] is sia_mount and kw_rec["psa"] is psa_mount)
    train_sig = inspect.signature(current_main.train).parameters
    check("mounts_for_stage returns exactly the loss kwargs of model/main.py train()",
          set(kw_rec) <= set(train_sig),
          f"keys {sorted(kw_rec)} vs train() params {sorted(train_sig)}")

    # ================================================ 4. parameter/loss coupling
    print("\n[4] (parameter state x loss state) consistency  [E7 acceptance 5]")
    for st in ("tokenizer", "recommender"):
        ctrl.apply(st)
        check(f"assert_consistent passes in stage {st}", ctrl.assert_consistent(st) is True)
    ctrl.apply("tokenizer")
    victim = next(iter(tiger.parameters()))
    victim.requires_grad = True                      # deliberately corrupt the state
    ok = False
    try:
        ctrl.assert_consistent("tokenizer")
    except S.StageInconsistent:
        ok = True
    check("it catches a trainable parameter inside a group the stage freezes", ok)
    victim.requires_grad = False
    check("back to consistent once the state is restored", ctrl.assert_consistent("tokenizer") is True)
    tok_trainable = sum(rep_tok["groups"][g]["elements"]
                        for g in ("tokenizer", "recommender", "adapter")
                        if g in S.STAGE_CONTRACT["tokenizer"]["trainable"])
    check("trainable_parameters() equals the contract's trainable element count",
          sum(p.numel() for p in ctrl.trainable_parameters("tokenizer")) == tok_trainable,
          f"{sum(p.numel() for p in ctrl.trainable_parameters('tokenizer'))} of {tok_trainable}")

    # ==================================================== 5. the initted trap
    print("\n[5] codebook init / seal (the vq.initted trap)")
    fresh = new_rqvae(load=False)
    check("a fresh kmeans_init RQ-VAE reports initted=False and an all-zero codebook",
          S.codebook_initted(fresh) == [False] * 3 and all(S.codebook_is_zero(fresh)),
          f"initted={S.codebook_initted(fresh)}")

    safe = new_rqvae(load=False)
    rep = S.load_tokenizer_state_dict(safe, ck["state_dict"])
    check("load_tokenizer_state_dict restores initted=True on every level",
          S.codebook_initted(safe) == [True] * 3, f"{S.codebook_initted(safe)}")
    check("it loads every key (no missing / unexpected)",
          not rep["missing_keys"] and not rep["unexpected_keys"],
          f"missing={len(rep['missing_keys'])} unexpected={len(rep['unexpected_keys'])}")
    ck_codebooks = [ck["state_dict"][f"rq.vq_layers.{i}.embedding.weight"] for i in range(3)]
    check("the loaded codebooks are bitwise the checkpoint's",
          codes_equal(codebooks(safe), ck_codebooks))

    # negative control: prove the hazard is real, not hypothetical
    naive = new_rqvae(load=False)
    naive.load_state_dict(ck["state_dict"])          # initted stays False
    probe = item_embs[:512].to(device)
    before = codebooks(naive)
    naive.train()
    with torch.no_grad():
        naive(probe)
    naive_changed = sum(1 for a, b in zip(before, codebooks(naive)) if not torch.equal(a, b))
    check("CONTROL: plain load_state_dict + a train-mode forward DOES overwrite the codebook",
          naive_changed > 0,
          f"{naive_changed}/3 levels rewritten by lazy k-means (this is the trap)")

    before = codebooks(safe)
    safe.train()
    with torch.no_grad():
        safe(probe)
    check("after load_tokenizer_state_dict the same forward leaves every codebook bitwise intact",
          codes_equal(before, codebooks(safe)),
          f"{sum(1 for a, b in zip(before, codebooks(safe)) if not torch.equal(a, b))}/3 levels changed")

    ok = False
    try:
        S.seal_codebooks(new_rqvae(load=False))
    except S.StageError:
        ok = True
    check("seal_codebooks refuses an un-initialised (all-zero) codebook", ok)
    soft = S.seal_codebooks(new_rqvae(load=False), strict=False)
    check("with strict=False it seals but reports which levels were still zero",
          soft["sealed"] and all(soft["zero_codebooks"]),
          f"zero={soft['zero_codebooks']}")

    zeroed = copy.deepcopy(ck["state_dict"])
    for i in range(3):
        zeroed[f"rq.vq_layers.{i}.embedding.weight"] = torch.zeros_like(
            zeroed[f"rq.vq_layers.{i}.embedding.weight"])
    ok = False
    try:
        S.load_tokenizer_state_dict(new_rqvae(load=False), zeroed)
    except S.StageError:
        ok = True
    check("loading a checkpoint whose codebook is all-zero raises instead of sealing garbage", ok)

    sealed = new_rqvae(load=False)
    ctrl2 = S.StageFreeze(sealed, current_main.TIGER(T5CFG).to(device),
                          [SIAAdapter(T5CFG['d_model']).to(device)])
    ctrl2.apply("tokenizer")
    check("the tokenizer stage does NOT seal (its own stage may initialise the codebook)",
          ctrl2.seal_report is None and S.codebook_initted(sealed) == [False] * 3,
          f"initted={S.codebook_initted(sealed)}")
    sealed.train()
    with torch.no_grad():
        sealed(probe)
    check("inside the tokenizer stage a train-mode forward does initialise it (as TIGER did)",
          S.codebook_initted(sealed) == [True] * 3)
    before = codebooks(sealed)
    ctrl2.apply("recommender")
    check("entering the recommender stage seals every level",
          ctrl2.seal_report is not None and S.codebook_initted(sealed) == [True] * 3,
          f"seal report zero_codebooks={None if ctrl2.seal_report is None else ctrl2.seal_report['zero_codebooks']}")
    sealed.train()
    with torch.no_grad():
        sealed(probe)
    check("a frozen-and-sealed codebook survives a train-mode forward bitwise",
          codes_equal(before, codebooks(sealed)))
    del ctrl2, sealed

    # ================================== 6. the real tokenizer training loop
    print("\n[6] tokenizer stage through the unmodified rqvae/trainer.py")
    tok_rq = new_rqvae(load=False)
    current_main.set_seed(SEED)
    tok_tiger = current_main.TIGER(T5CFG).to(device)
    tok_adapters = [SIAAdapter(T5CFG['d_model']).to(device),
                    PSAAdapter(T5CFG['d_model'], emb_ds.dim).to(device)]
    ctrl_tok = S.StageFreeze(tok_rq, tok_tiger, tok_adapters)
    ctrl_tok.apply("tokenizer")

    t5_calls = {"n": 0}
    handles = [tok_tiger.model.encoder.register_forward_pre_hook(
                   lambda m, a: t5_calls.update(n=t5_calls["n"] + 1)),
               tok_tiger.model.decoder.register_forward_pre_hook(
                   lambda m, a: t5_calls.update(n=t5_calls["n"] + 1))]

    chunks = [item_embs[i * 1024:(i + 1) * 1024].to(device) for i in range(2)]
    tok_rq_before = snap(tok_rq)
    tok_t5_before = snap(tok_tiger)
    tok_ad_before = snap(tok_adapters)
    trainer = RQVAETrainer(RQVaeArgs(os.path.join(tmpdir, "rqvae_ckpt"), device,
                                     RQVAE_CFG["num_emb_list"]),
                           tok_rq, len(chunks))
    best_loss, best_collision = trainer.fit(chunks)
    for h in handles:
        h.remove()

    check("tokenizer stage runs end to end (real Trainer.fit, 1 epoch, 2 batches)",
          np.isfinite(best_loss), f"best_loss={best_loss:.6f} best_collision={best_collision:.6f}")
    check("the RQ-VAE really trained (its parameters changed)",
          n_changed(tok_rq_before, tok_rq) > 0,
          f"{n_changed(tok_rq_before, tok_rq)}/{len(tok_rq_before)} tensors moved")
    check("T5 is bitwise untouched by a tokenizer-stage epoch",
          unchanged(tok_t5_before, tok_tiger),
          f"{n_changed(tok_t5_before, tok_tiger)} tensors moved")
    check("adapters are bitwise untouched by a tokenizer-stage epoch",
          unchanged(tok_ad_before, tok_adapters),
          f"{n_changed(tok_ad_before, tok_adapters)} tensors moved")
    check("the tokenizer stage never runs the T5 forward at all", t5_calls["n"] == 0,
          f"{t5_calls['n']} T5 stack calls")

    saved = sorted(glob.glob(os.path.join(tmpdir, "rqvae_ckpt", "*",
                                          "best_collision_model.pth")))
    restored = new_rqvae(load=False)
    rep = S.load_tokenizer_state_dict(restored, torch.load(saved[-1], map_location="cpu",
                                                          weights_only=False)["state_dict"])
    check("the epoch's checkpoint round-trips: initted restored, codebooks bitwise equal",
          S.codebook_initted(restored) == [True] * 3 and
          codes_equal(codebooks(restored), codebooks(tok_rq)) and not rep["missing_keys"],
          f"saved={os.path.relpath(saved[-1], tmpdir)}")

    # ================================= 7. the real recommender training loop
    print("\n[7] recommender stage through the unmodified model/main.py train()")
    rec_rq = new_rqvae(load=True)
    current_main.set_seed(SEED)
    rec_tiger = current_main.TIGER(T5CFG).to(device)
    rec_sia_ad = SIAAdapter(T5CFG['d_model']).to(device)
    rec_psa_ad = PSAAdapter(T5CFG['d_model'], emb_ds.dim).to(device)
    rec_sia = SIAMount(rec_sia_ad, rec_rq, item_embs, sid_tokens, lambda_sia=LAMBDA_SIA).to(device)
    rec_psa = PSAMount(rec_psa_ad, rec_rq, item_embs, sid_tokens, lambda_psa=LAMBDA_PSA).to(device)
    ctrl_rec = S.StageFreeze(rec_rq, rec_tiger, [rec_sia_ad, rec_psa_ad])
    ctrl_rec.apply("recommender")

    rq_calls = {"enc": 0, "dec": 0}
    handles = [rec_rq.encoder.register_forward_hook(
                   lambda m, i, o: rq_calls.update(enc=rq_calls["enc"] + 1)),
               rec_rq.decoder.register_forward_hook(
                   lambda m, i, o: rq_calls.update(dec=rq_calls["dec"] + 1))]

    from torch.utils.data import Subset
    small = Subset(ds, list(range(0, 4 * BATCH)))
    loader = GenRecDataLoader(small, batch_size=BATCH, shuffle=False, num_workers=0)

    trainable_ids = {id(p) for p in ctrl_rec.trainable_parameters("recommender")}
    tiger_ids = {id(p) for p in rec_tiger.parameters()}
    adapter_ids = {id(p) for p in list(rec_sia_ad.parameters()) + list(rec_psa_ad.parameters())}
    check("the recommender optimizer's parameter set is T5 + adapters, with no RQ-VAE",
          trainable_ids == tiger_ids | adapter_ids
          and not (trainable_ids & {id(p) for p in rec_rq.parameters()}),
          f"{len(trainable_ids)} params = {len(tiger_ids)} T5 + {len(adapter_ids)} adapter")
    check("the adapters live outside TIGER.parameters(), so they must be added to the "
          "recommender optimizer explicitly (the frozen entry point does not)",
          not (adapter_ids & tiger_ids),
          f"{len(adapter_ids)} adapter tensors, none inside TIGER")
    opt = torch.optim.Adam(ctrl_rec.trainable_parameters("recommender"), lr=1e-4)

    rq_before = snap(rec_rq)
    t5_before = snap(rec_tiger)
    ad_before = snap([rec_sia_ad, rec_psa_ad])
    epoch_loss = current_main.train(rec_tiger, loader, opt, device,
                                    **ctrl_rec.mounts_for_stage(sia=rec_sia, psa=rec_psa))
    for h in handles:
        h.remove()

    check("recommender stage runs end to end (SIA + PSA mounted, lambda > 0)",
          np.isfinite(epoch_loss), f"epoch loss {epoch_loss!r}")
    check("the T5 really trained (its parameters changed)",
          n_changed(t5_before, rec_tiger) > 0,
          f"{n_changed(t5_before, rec_tiger)}/{len(t5_before)} tensors moved")
    check("the adapters really trained (their parameters changed)",
          n_changed(ad_before, [rec_sia_ad, rec_psa_ad]) > 0,
          f"{n_changed(ad_before, [rec_sia_ad, rec_psa_ad])} tensors moved")
    check("the RQ-VAE is bitwise untouched across a lambda>0 recommender epoch",
          unchanged(rq_before, rec_rq),
          f"{n_changed(rq_before, rec_rq)}/{len(rq_before)} RQ-VAE tensors moved "
          f"(was 10/13 with gradients before E7)")
    check("its target-side forward still ran (frozen but used, not eval()'d away)",
          rq_calls["enc"] > 0 and rq_calls["dec"] > 0,
          f"encoder {rq_calls['enc']} calls, decoder {rq_calls['dec']} calls")
    check("every RQ-VAE parameter ends the epoch with grad None",
          all(p.grad is None for p in rec_rq.parameters()),
          f"{sum(p.grad is not None for p in rec_rq.parameters())}/{len(list(rec_rq.parameters()))} with grad")

    # ==================== 8. one optimizer across a stage switch
    print("\n[8] optimizer parameter coverage across a stage switch")
    sw_rq = new_rqvae(load=True)
    current_main.set_seed(SEED)
    sw_tiger = current_main.TIGER(T5CFG).to(device)
    sw_adapter = SIAAdapter(T5CFG['d_model']).to(device)
    ctrl_sw = S.StageFreeze(sw_rq, sw_tiger, [sw_adapter])
    shared_opt = torch.optim.Adam(ctrl_sw.all_parameters(), lr=1e-4)
    opt_ids_before = [id(p) for g in shared_opt.param_groups for p in g["params"]]

    ctrl_sw.apply("recommender")
    t5_before, rq_before = snap(sw_tiger), snap(sw_rq)
    current_main.train(sw_tiger, GenRecDataLoader(Subset(ds, list(range(0, 2 * BATCH))),
                                                  batch_size=BATCH, shuffle=False,
                                                  num_workers=0),
                       shared_opt, device)
    check("one Adam instance serves the recommender step: T5 moves, RQ-VAE does not",
          n_changed(t5_before, sw_tiger) > 0 and unchanged(rq_before, sw_rq),
          f"T5 moved {n_changed(t5_before, sw_tiger)}, "
          f"RQ-VAE moved {n_changed(rq_before, sw_rq)}")
    check("Adam keeps no state for the frozen RQ-VAE (frozen params are simply skipped)",
          not any(id(p) in {id(q) for q in shared_opt.state} for p in sw_rq.parameters()),
          f"{len(shared_opt.state)} optimizer state entries, none for the RQ-VAE")

    ctrl_sw.apply("tokenizer")
    t5_before, rq_before = snap(sw_tiger), snap(sw_rq)
    x = item_embs[:512].to(device)
    out, rq_loss, _ = sw_rq(x, return_z=False)          # the Track-A step body
    loss, _ = sw_rq.compute_loss(out, rq_loss, xs=x)
    shared_opt.zero_grad()
    loss.backward()
    shared_opt.step()
    check("the same Adam instance also serves the tokenizer step: RQ-VAE moves, T5 does not",
          n_changed(rq_before, sw_rq) > 0 and unchanged(t5_before, sw_tiger),
          f"RQ-VAE moved {n_changed(rq_before, sw_rq)}, "
          f"T5 moved {n_changed(t5_before, sw_tiger)}")
    check("an over-covering optimizer needs no rebuild: its parameter list is unchanged "
          "across the switch",
          [id(p) for g in shared_opt.param_groups for p in g["params"]] == opt_ids_before,
          f"{len(opt_ids_before)} parameters, frozen and trainable alike")

    # ============================================= 9. checkpoint stage record
    print("\n[9] checkpoint stage record and resume check")
    ckpt_path = os.path.join(tmpdir, "tiger_epoch1.pth")
    torch.save({"state_dict": rec_tiger.state_dict()}, ckpt_path)
    written = S.write_stage_manifest(ckpt_path, ctrl_rec, stage="recommender",
                                     extra={"dataset": "Toys(smoke-subset)",
                                            "lambda_sia": LAMBDA_SIA,
                                            "lambda_psa": LAMBDA_PSA})
    record = S.read_stage_manifest(ckpt_path)
    check("a stage record is written beside the checkpoint and round-trips",
          os.path.exists(written) and record["freeze"]["stage"] == "recommender",
          f"{os.path.basename(written)}")
    check("it records the per-group freeze split and the loss switches",
          record["freeze"]["groups"]["tokenizer"]["trainable_tensors"] == 0
          and record["freeze"]["losses"] == ctrl_rec.loss_flags("recommender"))
    check("it records the codebook initted state (auditable resume evidence)",
          record["freeze"]["codebook_initted"] == [True] * 3,
          f"{record['freeze']['codebook_initted']}")
    check("verify_stage_manifest accepts a resume under the same stage",
          S.verify_stage_manifest(record, ctrl_rec, stage="recommender")["freeze"]["stage"]
          == "recommender")
    ok = False
    try:
        S.verify_stage_manifest(record, ctrl_rec, stage="tokenizer")
    except S.StageMismatch:
        ok = True
    check("it refuses to resume the checkpoint under the other stage", ok)
    tampered = copy.deepcopy(record)
    tampered["freeze"]["groups"]["tokenizer"]["trainable_tensors"] = 15
    tampered["freeze"]["groups"]["tokenizer"]["tensors"] = 15
    ok = False
    try:
        S.verify_stage_manifest(tampered, ctrl_rec, stage="recommender")
    except S.StageMismatch:
        ok = True
    check("it refuses a checkpoint whose freeze split differs from the live one", ok)

    # ============================================= 10. E7-off = clean TIGER
    print("\n[10] with E7 unused, Clean TIGER is unchanged")
    off_rq = new_rqvae(load=True)
    current_main.set_seed(SEED)
    off_tiger = current_main.TIGER(T5CFG).to(device)
    off_ad = SIAAdapter(T5CFG['d_model']).to(device)
    before = (flags(off_rq), flags(off_tiger), flags(off_ad))
    constructed = S.StageFreeze(off_rq, off_tiger, [off_ad])
    check("constructing StageFreeze touches no requires_grad flag",
          (flags(off_rq), flags(off_tiger), flags(off_ad)) == before
          and constructed.stage is None,
          f"stage={constructed.stage!r}")

    current_main.set_seed(SEED)
    cur = current_main.TIGER(T5CFG).to(device)
    current_main.set_seed(SEED)
    pri = pristine_main.TIGER(T5CFG).to(device)
    pri.load_state_dict(cur.state_dict())
    cur.eval(); pri.eval()
    lc, oc = cur(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    lp, op = pri(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    check("loss / logits bitwise identical to pre-E7 TIGER",
          torch.equal(lc, lp) and torch.equal(oc, op), f"loss {float(lc):.8f}")
    gc = cur.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5)
    gp = pri.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5)
    check("generate() bitwise identical to pre-E7 TIGER", torch.equal(gc, gp),
          f"shape {tuple(gc.shape)}")
    check("E7 adds no parameter to the TIGER module",
          sum(p.numel() for p in cur.parameters()) == sum(p.numel() for p in pri.parameters()),
          f"{sum(p.numel() for p in cur.parameters())} params")

    hits = subprocess.run(
        ["grep", "-rn", "--include=*.py", "-E",
         r"^[[:space:]]*(from|import)[[:space:]]+tiger_ext",
         os.path.join(REPO, "model"), os.path.join(REPO, "rqvae")],
        capture_output=True, text=True).stdout.strip().splitlines()
    check("nothing under model/ or rqvae/ imports the new package (no runtime coupling)",
          hits == [], f"{hits[:2]}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 70)
    print(f"EXP-E7: {passed}/{len(RESULTS)} checks passed  (device={device})")
    print("=" * 70)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
