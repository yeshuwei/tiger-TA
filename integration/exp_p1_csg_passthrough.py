#!/usr/bin/env python
"""EXP-P1 -- the CSG hook's passthrough into ``alt_pipeline.tokenizer_phase``.

Run:  python -I integration/exp_p1_csg_passthrough.py

P1 is a **wiring** change, not a mechanism change: ``TigerAlternatingPipeline``
grows an optional ``csg=`` and hands it to the frozen
``rqvae/trainer.py::Trainer`` the same way C5 does. Nothing about ``L_csg``,
the RQ-VAE core or the trainer is touched. The checks therefore ask three
questions:

  [1] the change is what it claims -- signature and static import
  [2] ``csg=None`` is inert -- the objects built and the training run are
      bitwise identical to the **pre-P1 call sites**
      (``EmbDataset(path)`` / ``Trainer(args, model, n)``)
  [3] an **enabled** hook actually reaches the trainer and is invoked on every
      batch; a **disabled** hook is carried but never invoked

The pre-P1 call sites are reproduced by shimming ``frozen_modules()`` with
three-argument and one-argument wrappers that assert what the P1 code passed
(``csg is None``, ``return_index is False``). If P1 leaked a fourth argument or
a ``return_index=True`` into the default path, the assertion fires.

Nothing here measures recommendation quality.
"""
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from tiger_ext.alt_pipeline import (                                     # noqa: E402
    TigerAlternatingPipeline, frozen_modules, _validate_csg,
)
from tiger_ext.collab_relation import build_relation                     # noqa: E402
from tiger_ext.qspace_train import CSGHook                               # noqa: E402

DATASET = "P1Syn"
DEVICE = os.environ.get("EXP_P1_DEVICE", "cpu")

#: small enough to run four pipelines on CPU in well under a minute, large
#: enough that the codebook sees more than one batch and the k-means init is
#: not degenerate
SYN = dict(n_items=512, dim=32, k=20, codes=[64, 64, 64], batch=128, seed=0)
SEED = 2025
LAMBDA_CSG = 1e-3

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def section(title):
    print(f"\n{title}")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def build_synth(root):
    """A tiny item table plus a real C2 relation, both through the real builders."""
    rng = np.random.default_rng(SYN["seed"])
    n, dim = SYN["n_items"], SYN["dim"]
    # clustered embeddings so k-NN neighbours are meaningful rather than noise
    centres = rng.normal(size=(8, dim)).astype(np.float32) * 2.0
    which = rng.integers(0, 8, size=n)
    emb = (centres[which] + rng.normal(size=(n, dim)).astype(np.float32) * 0.3)
    emb = emb.astype(np.float32)

    item_ids = np.arange(1, n + 1, dtype=np.int64)
    data_dir = os.path.join(root, DATASET)
    os.makedirs(data_dir, exist_ok=True)
    pd.DataFrame({"ItemID": item_ids,
                  "embedding": [emb[i] for i in range(n)]}).to_parquet(
        os.path.join(data_dir, "item_emb.parquet"), index=False)

    relation = build_relation(emb, item_ids, k=SYN["k"], source="p1_synthetic")
    artifact_dir = os.path.join(root, "collab_relation")
    prefix = os.path.join(artifact_dir, f"{DATASET}_k{SYN['k']}")
    relation.save(prefix)
    return data_dir, artifact_dir, prefix


def build_pipeline(root, tag, csg=None):
    """A fresh pipeline: identical seed, identical config, one hook or none."""
    rqvae_config = dict(num_emb_list=list(SYN["codes"]), e_dim=32,
                        batch_size=SYN["batch"])
    return TigerAlternatingPipeline(
        DATASET, root, os.path.join(root, "sid"), os.path.join(root, "run", tag),
        device=DEVICE, rqvae_config=rqvae_config, lambda_sia=0.0, lambda_psa=0.0,
        seed=SEED, num_workers=0, csg=csg)


def state_dict_of(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt


def same_state(a, b):
    """Bitwise ``torch.equal`` on every tensor; returns (ok, detail)."""
    if set(a) != set(b):
        return False, f"key sets differ ({len(set(a) ^ set(b))} keys)"
    for key in sorted(a):
        x, y = a[key], b[key]
        if x.shape != y.shape:
            return False, f"{key}: shape {tuple(x.shape)} vs {tuple(y.shape)}"
        if not torch.equal(x, y):
            return False, f"{key}: not bitwise equal"
    return True, f"{len(a)} tensors bitwise equal"


def run_phase(root, tag, csg=None, pre_p1=False):
    """One tokenizer phase; ``pre_p1`` shims the module table to the old calls."""
    mods = frozen_modules()
    saved = (mods["EmbDataset"], mods["RQVAETrainer"])
    calls = {"dataset": [], "trainer": []}

    if pre_p1:
        def pre_emb_dataset(path, return_index=False):
            calls["dataset"].append({"return_index": return_index})
            if return_index:
                raise AssertionError("pre-P1 EmbDataset has no return_index path")
            return saved[0](path)

        def pre_trainer(args, model, data_num, csg=None):
            calls["trainer"].append({"csg": csg, "n_args": 4})
            if csg is not None:
                raise AssertionError("pre-P1 Trainer got a csg hook")
            return saved[1](args, model, data_num)

        mods["EmbDataset"], mods["RQVAETrainer"] = pre_emb_dataset, pre_trainer
    try:
        pipeline = build_pipeline(root, tag, csg=csg)
        result = pipeline.tokenizer_phase(0, 0)
        config = pipeline.run_config()
    finally:
        mods["EmbDataset"], mods["RQVAETrainer"] = saved

    return {
        "result": result, "config": config,
        "state": state_dict_of(result["ckpt_path"]),
        "calls": calls,
    }


# --------------------------------------------------------------------------- #
# [1] static shape of the change
# --------------------------------------------------------------------------- #
def part1_static():
    section("[1] static: the constructor grows one optional keyword, nothing else")
    import inspect
    sig = inspect.signature(TigerAlternatingPipeline.__init__)
    params = sig.parameters
    check("`csg` is an optional constructor keyword",
          "csg" in params and params["csg"].default is None,
          f"default={params['csg'].default!r}" if "csg" in params else "absent")
    check("`csg` is the last parameter (no positional shuffling)",
          list(params)[-1] == "csg", f"last={list(params)[-1]}")
    check("`tokenizer_phase` keeps its pre-P1 signature",
          list(inspect.signature(
              TigerAlternatingPipeline.tokenizer_phase).parameters
               ) == ["self", "cycle_index", "epoch_index", "resume_ckpt"])
    check("`_validate_csg(None) is None`", _validate_csg(None) is None)


# --------------------------------------------------------------------------- #
# [2] csg=None: inert, and identical to the pre-P1 call sites
# --------------------------------------------------------------------------- #
def part2_default(root):
    section("[2] csg=None: pre-P1 call sites and bitwise-identical training")

    base = run_phase(root, "base")
    pre = run_phase(root, "pre", pre_p1=True)

    # two constructions per pipeline, both pre-P1: the constructor's
    # ``self.item_embeddings`` and the phase's training set
    check("pre-P1 shim saw `EmbDataset(path)` only (return_index False)",
          len(pre["calls"]["dataset"]) == 2
          and all(c["return_index"] is False for c in pre["calls"]["dataset"]),
          f"{pre['calls']['dataset']}")
    check("pre-P1 shim saw `Trainer(args, model, n)` (csg is None)",
          pre["calls"]["trainer"] == [{"csg": None, "n_args": 4}],
          f"{pre['calls']['trainer']}")

    ok, detail = same_state(base["state"], pre["state"])
    check("csg=None trains bitwise the same as the pre-P1 call sites", ok, detail)
    check("... and reaches the same best_loss",
          base["result"]["best_loss"] == pre["result"]["best_loss"],
          f"{base['result']['best_loss']!r} vs {pre['result']['best_loss']!r}")
    check("... and the same collision rate",
          base["result"]["best_collision_rate"] == pre["result"]["best_collision_rate"])

    pre_p1_keys = {"ckpt_path", "sha256", "cycle_index", "epoch_index", "seed",
                   "best_loss", "best_collision_rate", "optimizer_params",
                   "freeze", "losses"}
    check("the csg=None result dict has exactly the pre-P1 keys",
          set(base["result"]) == pre_p1_keys,
          f"extra={sorted(set(base['result']) - pre_p1_keys)}")
    check("the csg=None run manifest carries no `csg` key",
          "csg" not in base["config"])
    return base


# --------------------------------------------------------------------------- #
# [3] a hook: enabled reaches the trainer; disabled stays inert
# --------------------------------------------------------------------------- #
def part3_hooks(root, baseline):
    section("[3] a hook is carried to the trainer; only an enabled one acts")

    artifact_dir = os.path.join(root, "collab_relation")
    prefix = os.path.join(artifact_dir, f"{DATASET}_k{SYN['k']}")

    def make_hook(lam):
        return CSGHook.for_dataset(DATASET, lambda_csg=lam, k=SYN["k"],
                                   data_dir=root, artifact_dir=artifact_dir,
                                   relation_prefix=prefix)

    # --- disabled: carried, never invoked ---------------------------------- #
    off_hook = make_hook(0.0)
    off = run_phase(root, "hook_off", csg=off_hook)
    ok, detail = same_state(off["state"], baseline["state"])
    check("a lambda_csg=0 hook is inert (bitwise identical to csg=None)", ok, detail)
    check("... it still reached the trainer (`attached`)",
          off["result"].get("csg", {}).get("attached") is True)
    check("... but was never invoked",
          off_hook.counters["n_calls"] == 0
          and off_hook.counters["n_disabled_calls"] == 0,
          f"n_calls={off_hook.counters['n_calls']} "
          f"n_disabled={off_hook.counters['n_disabled_calls']}")
    check("... and the dataset stayed index-free",
          off["result"]["csg"]["return_index"] is False)

    # --- enabled: reaches the trainer and runs on every batch --------------- #
    on_hook = make_hook(LAMBDA_CSG)
    on = run_phase(root, "hook_on", csg=on_hook)
    counters = on_hook.counters
    expected_steps = int(np.ceil(SYN["n_items"] / SYN["batch"]))

    check("an enabled hook reaches the trainer (`attached`)",
          on["result"]["csg"]["attached"] is True)
    check("the dataset carries row indices when the hook is enabled",
          on["result"]["csg"]["return_index"] is True)
    check("the hook ran once per batch",
          counters["n_calls"] == expected_steps,
          f"n_calls={counters['n_calls']} expected={expected_steps}")
    check("no call happened while disabled",
          counters["n_disabled_calls"] == 0,
          f"n_disabled_calls={counters['n_disabled_calls']}")
    check("the constraint was reached through C3/C4",
          counters["constraint_calls"] == expected_steps
          and counters["mount_calls"] == expected_steps,
          f"constraint={counters['constraint_calls']} mount={counters['mount_calls']}")
    check("at least one step produced a non-zero, differentiable L_csg",
          any(rec["weighted"] > 0 and rec["requires_grad"]
              for rec in on_hook.history),
          f"n_nonempty={counters['n_nonempty']} n_zero={counters['n_zero']}")
    ok, detail = same_state(on["state"], baseline["state"])
    check("the enabled hook actually changes training (not bitwise equal)", not ok,
          "state_dict differs as expected" if not ok else "IDENTICAL -- hook had no effect")

    # --- provenance is recorded -------------------------------------------- #
    check("the result dict records the hook when one is attached",
          on["result"]["csg"]["lambda_csg"] == LAMBDA_CSG
          and on["result"]["csg"]["enabled"] is True)
    check("the run manifest records the hook",
          on["config"].get("csg", {}).get("lambda_csg") == LAMBDA_CSG
          and on["config"]["csg"]["stage"] == "tokenizer",
          f"{on['config'].get('csg')}")


# --------------------------------------------------------------------------- #
# [4] negative controls
# --------------------------------------------------------------------------- #
def part4_negative(root):
    section("[4] negative controls: unusable hook objects are rejected")

    raised = None
    try:
        _validate_csg(object())
    except TypeError as exc:
        raised = exc
    check("a non-hook object raises TypeError", raised is not None, str(raised))

    class _ParamHook(CSGHook):
        def __init__(self):
            super().__init__(lambda_csg=0.0)
            self.w = torch.nn.Parameter(torch.zeros(1))

    raised = None
    try:
        _validate_csg(_ParamHook())
    except ValueError as exc:
        raised = exc
    check("a hook carrying parameters raises ValueError", raised is not None,
          str(raised))

    raised = None
    try:
        TigerAlternatingPipeline(
            DATASET, root, os.path.join(root, "sid"), os.path.join(root, "run", "bad"),
            device=DEVICE, seed=SEED, num_workers=0, csg=object())
    except TypeError as exc:
        raised = exc
    check("the constructor rejects it too (fails before any training)",
          raised is not None, str(raised))


# --------------------------------------------------------------------------- #
def main():
    root = tempfile.mkdtemp(prefix="p1_csg_")
    try:
        print(f"fixture -> {root}")
        build_synth(root)
        part1_static()
        baseline = part2_default(root)
        part3_hooks(root, baseline)
        part4_negative(root)
    except Exception:                                    # noqa: BLE001
        traceback.print_exc()
        RESULTS.append(("harness crashed", False, "see traceback"))
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\nEXP-P1: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED: {name} -- {detail}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
