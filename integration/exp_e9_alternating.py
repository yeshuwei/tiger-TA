#!/usr/bin/env python
"""EXP-E9 -- the alternating scheduler (Task E9, type A).

Run:  python -I integration/exp_e9_alternating.py
      EXP_E9_DEVICE=cuda python -I integration/exp_e9_alternating.py
      EXP_E9_REGRESSION=skip python -I integration/exp_e9_alternating.py

Covers the twelve checks E9 was specified with:

   1. >=2 cycles run end to end
   2. the Tokenizer -> Refresh -> Recommender order holds
   3. the *new* SID is what the next recommender phase consumes
   4. a refresh failure stops the cycle before the recommender
   5. resuming from `refresh_done` does not refresh a second time
   6. `recommender_done` advances to the next cycle
   7. the tokenizer phase updates the RQ-VAE and nothing else
   8. the recommender phase leaves the RQ-VAE bitwise unchanged
   9. the recommender phase updates T5 *and* the adapters
  10. evaluation happens only after a recommender phase
  11. the seed control is deterministic and per-phase
  12. the E1-E8 suites still pass afterwards

Four parts, cheapest first:

  [1] the pure schedule -- no torch, no models
  [2] the scheduler with **injected fake phases** -- branch, failure, resume
      and hand-off logic, exercised without a single training step
  [3] the **real** pipeline on a small synthetic fixture -- both frozen loops,
      the real E8 refresh (a real ``generate_code.py`` subprocess) and the real
      SID hand-off, for two full cycles
  [4] regression: re-run E1-E8, and check the frozen directories are untouched

Nothing here measures recommendation quality; that is EXP-E7-E9 (type B/C).
"""
import ast
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from tiger_ext import alternating as A                                   # noqa: E402
from tiger_ext import alt_pipeline as AP                                 # noqa: E402
from tiger_ext import sid_refresh as SR                                  # noqa: E402
from tiger_ext.stage import STAGE_RECOMMENDER, STAGE_TOKENIZER           # noqa: E402

DEVICE = os.environ.get("EXP_E9_DEVICE", "cpu")
REGRESSION = os.environ.get("EXP_E9_REGRESSION", "all").lower()
DATASET = "E9Syn"
SEED = 2025

# the tracked files E1-E6 legitimately modified; E9 must not add to this set
PRE_E9_MODIFIED = {"model/main.py", "rqvae/models/rq.py", "rqvae/models/vq.py"}

# Small but not degenerate: 1024 items so the k-means codebooks can actually
# separate them (a code book larger than the item count would make collisions
# systematic rather than incidental, and E8 rejects an unresolved duplicate).
SYN = dict(n_items=1024, dim=32, n_users=24, seq_len=12, n_valid=8)
# e_dim stays 32: the SIA sequence-side query is ``SIA_LATENT_DIM`` = 32 wide and
# goes straight into ``rqvae.rq`` (tiger_ext/sia_codes.py), which requires the
# quantizer's own e_dim. That is the E2/E3 contract, not a free knob.
SYN_RQVAE = dict(num_emb_list=[128, 128, 128], e_dim=32, layers=[16, 8],
                 kmeans_init=True, kmeans_iters=5, sk_iters=5,
                 sk_epsilons=[0.0, 0.0, 0.003], batch_size=1024, lr=5e-3)
SYN_T5 = dict(num_layers=1, num_decoder_layers=1, d_model=16, d_ff=32,
              num_heads=2, d_kv=4, dropout_rate=0.0, vocab_size=1025,
              pad_token_id=0, eos_token_id=0, feed_forward_proj="relu", max_len=20)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def snap(obj):
    mods = obj if isinstance(obj, (list, tuple)) else [obj]
    out = []
    for m in mods:
        out.extend(p.detach().clone() for p in m.parameters())
    return out


def unchanged(before, obj):
    now = snap(obj)
    return len(before) == len(now) and all(torch.equal(a, b) for a, b in zip(before, now))


def n_changed(before, obj):
    return sum(1 for a, b in zip(before, snap(obj)) if not torch.equal(a, b))


# --------------------------------------------------------------------------- #
# [1] the pure schedule
# --------------------------------------------------------------------------- #
def part1_schedule():
    print("\n[1] the phase schedule (pure; the official modulo rule)")
    check("epoch 0 is the tokenizer phase, epoch 1 the recommender",
          A.phase_of(0) == STAGE_TOKENIZER and A.phase_of(1) == STAGE_RECOMMENDER)
    seq = [A.phase_of(e) for e in range(6)]
    check("cycle=2 gives exactly T, R, T, R, T, R",
          seq == [STAGE_TOKENIZER, STAGE_RECOMMENDER] * 3, f"{seq}")
    check("the epoch index is 0-based: phase_of(0) is the *first* epoch",
          A.phase_seed(SEED, 0) == SEED, "off-by-one here swaps both phases")

    four = [A.phase_of(e, 4) for e in range(9)]
    check("cycle is a modulo period, not a block length: one tokenizer epoch "
          "per `cycle` epochs, the rest recommender",
          four == [STAGE_TOKENIZER, STAGE_RECOMMENDER, STAGE_RECOMMENDER,
                   STAGE_RECOMMENDER, STAGE_TOKENIZER, STAGE_RECOMMENDER,
                   STAGE_RECOMMENDER, STAGE_RECOMMENDER, STAGE_TOKENIZER],
          f"cycle=4 -> {four}")

    check("the 1-based reading is provably wrong (it inverts every phase)",
          [A.phase_of(e + 1) for e in range(2)] == [STAGE_RECOMMENDER, STAGE_TOKENIZER],
          "a `Epoch {n+1}` label must not be fed to phase_of")
    check("an unknown stage / a negative epoch is rejected, not guessed",
          raises(lambda: A.epoch_index_of(0, "eval"), A.ScheduleError)
          and raises(lambda: A.phase_of(-1), A.ScheduleError))

    check("epoch_index_of maps cycle c to epochs 2c (tokenizer) and 2c+1 (recommender)",
          [A.epoch_index_of(c, STAGE_TOKENIZER) for c in range(3)] == [0, 2, 4]
          and [A.epoch_index_of(c, STAGE_RECOMMENDER) for c in range(3)] == [1, 3, 5])
    check("a cycle longer than 2 epochs is refused (E9 fixes 1T + 1R per cycle)",
          raises(lambda: A.epoch_index_of(0, STAGE_TOKENIZER, cycle_length=4),
                 A.ScheduleError))

    plan = A.phase_plan(2)
    check("a 2-cycle plan is exactly the 4 epochs T, R, T, R",
          [(p["cycle"], p["epoch"], p["stage"]) for p in plan] ==
          [(0, 0, STAGE_TOKENIZER), (0, 1, STAGE_RECOMMENDER),
           (1, 2, STAGE_TOKENIZER), (1, 3, STAGE_RECOMMENDER)])
    check("phase_seed is deterministic and distinct per epoch",
          [A.phase_seed(SEED, e) for e in range(4)] == [SEED, SEED + 1, SEED + 2, SEED + 3])

    check("the official eval cadence invariant cycle % eval_step == 0 holds "
          "(trainer.py:56) and 2/2 keeps evaluation on a recommender epoch",
          A.check_eval_cadence(2, 2) is None
          and A.phase_of(1, 2) == STAGE_RECOMMENDER
          and raises(lambda: A.check_eval_cadence(2, 3), A.ScheduleError))
    check("the cycle length and eval step are the official values",
          A.CYCLE_LENGTH == 2 and A.EVAL_STEP == 2)
    check("the cycle state machine is exactly the four specified states",
          A.PHASE_STATES == ("started", "tokenizer_done", "refresh_done",
                             "recommender_done"))


# --------------------------------------------------------------------------- #
# [2] the scheduler, with injected phases
# --------------------------------------------------------------------------- #
class FakePhases:
    """Records every call and mints real (if meaningless) artifacts on disk."""

    def __init__(self, root):
        self.root = root
        self.calls = []
        self.refresh_calls = {}
        self.fail_refresh = set()
        self.fail_recommender = set()
        self.break_handoff = set()
        self.next_version = 1

    def _artifact(self, rel, payload):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(payload)
        return path

    def tokenizer(self, cycle_index, epoch_index, resume_ckpt):
        self.calls.append(("tokenizer", cycle_index, epoch_index))
        path = self._artifact(f"tok/c{cycle_index}.pth", f"tok {cycle_index}".encode())
        return {"ckpt_path": path, "sha256": SR.sha256_file(path)}

    def refresh(self, cycle_index, epoch_index, tokenizer_ckpt):
        self.calls.append(("refresh", cycle_index, epoch_index))
        self.refresh_calls[cycle_index] = self.refresh_calls.get(cycle_index, 0) + 1
        if cycle_index in self.fail_refresh:
            raise SR.SidRefreshError(f"simulated generator failure in cycle {cycle_index}")
        version = self.next_version
        self.next_version += 1
        rel = f"sid/{DATASET}_sid_v{version:04d}_deadbeef.npy"
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.full((4, 4), version, dtype=int))
        return {"version": version, "sid_path": path,
                "sid_sha256": SR.sha256_file(path)}

    def recommender(self, cycle_index, epoch_index, sid_path, sid_version, resume_ckpt):
        self.calls.append(("recommender", cycle_index, epoch_index))
        if cycle_index in self.fail_recommender:
            raise RuntimeError(f"simulated recommender failure in cycle {cycle_index}")
        path = self._artifact(f"rec/c{cycle_index}.pth", f"rec {cycle_index}".encode())
        broken = cycle_index in self.break_handoff
        other = self._artifact(f"sid/stale_{cycle_index}.npy", b"stale")
        return {"ckpt_path": path, "sha256": SR.sha256_file(path),
                "sid_path_used": other if broken else sid_path,
                "sid_version_used": sid_version - 1 if broken else sid_version,
                "sid_sha256_used": SR.sha256_file(other) if broken
                else SR.sha256_file(sid_path)}

    def evaluate(self, cycle_index, epoch_index, recommender_ckpt, sid_path):
        self.calls.append(("evaluation", cycle_index, epoch_index))
        return {"ndcgs": {"NDCG@20": 0.5}, "epoch_stage": A.phase_of(epoch_index)}


def make_scheduler(run_dir, fake, n_cycles=2, seed=SEED,
                   initial_ckpt="/nonexistent/initial.pth"):
    return A.AlternatingScheduler(
        run_dir=run_dir, dataset=DATASET, n_cycles=n_cycles, seed=seed, device="cpu",
        tokenizer_phase=fake.tokenizer, refresh_phase=fake.refresh,
        recommender_phase=fake.recommender, eval_phase=fake.evaluate,
        initial_tokenizer_ckpt=initial_ckpt)


def part2_scheduler_fakes(tmp):
    print("\n[2] the scheduler with injected phases (branch / failure / resume)")
    root = os.path.join(tmp, "sched")
    run_dir = os.path.join(root, "run")
    fake = FakePhases(root)
    scheduler = make_scheduler(run_dir, fake)
    records = scheduler.run()

    check("a 2-cycle run completes (1)",
          len(records) == 2 and all(r["state"] == "recommender_done" for r in records),
          f"{[r['state'] for r in records]}")
    expected = [("refresh", -1, -1),                       # initial refresh, no cycle
                ("tokenizer", 0, 0), ("refresh", 0, 0), ("recommender", 0, 1),
                ("evaluation", 0, 1),
                ("tokenizer", 1, 2), ("refresh", 1, 2), ("recommender", 1, 3),
                ("evaluation", 1, 3)]
    check("the call order is exactly initial-refresh, then T->refresh->R->eval per cycle",
          fake.calls == expected, f"{fake.calls}")
    check("the initial refresh is recorded but belongs to no cycle (E9 spec 4)",
          os.path.isfile(scheduler.initial_manifest_path)
          and json.load(open(scheduler.initial_manifest_path))["version"] == 1
          and records[0]["sid"]["version"] == 2,
          "initial -> v1, cycle 0 -> v2, cycle 1 -> v3")
    check("the state machine only ever moves forward through the four states",
          [h["state"] for h in records[0]["history"]] == list(A.PHASE_STATES)
          and len(records[0]["history"]) == 4,
          f"{[h['state'] for h in records[0]['history']]}")

    for cycle in (0, 1):
        order = [c[0] for c in fake.calls if c[1] == cycle]
        check(f"cycle {cycle}: tokenizer -> refresh -> recommender, in that order (2)",
              order == ["tokenizer", "refresh", "recommender", "evaluation"], f"{order}")

    evals = [c for c in fake.calls if c[0] == "evaluation"]
    check("evaluation always follows the recommender of its own cycle (10)",
          all(c[2] == A.epoch_index_of(c[1], STAGE_RECOMMENDER) for c in evals)
          and all(A.phase_of(c[2]) == STAGE_RECOMMENDER for c in evals),
          f"eval epochs {[c[2] for c in evals]} (recommender epochs are odd)")
    check("no evaluation is ever recorded for a tokenizer epoch",
          all(A.phase_of(c[2]) != STAGE_TOKENIZER for c in evals))
    check("each cycle's evaluation is stored in that cycle's manifest (6/10)",
          all(r["evaluation"] and r["evaluation"]["ndcgs"]["NDCG@20"] == 0.5
              for r in records))
    check("cycle 1 exists and is complete, i.e. recommender_done advances (6)",
          records[1]["cycle_index"] == 1 and records[1]["state"] == "recommender_done")

    plan = scheduler.resume_plan()
    check("a completed run has nothing left to do",
          all(a["action"] == "next_cycle" for a in plan["actions"]),
          f"{[a['action'] for a in plan['actions']]}")

    # ---------------------------------------------------------- hand-off check
    print("      -- the hand-off check")
    root2 = os.path.join(tmp, "sched_handoff")
    fake2 = FakePhases(root2)
    fake2.break_handoff.add(0)
    sched2 = make_scheduler(os.path.join(root2, "run"), fake2)
    check("a recommender that used a different SID is caught (3, negative control)",
          raises(sched2.run, A.HandoffError))
    m0 = A.CycleManifest.load(A.cycle_manifest_path(os.path.join(root2, "run"), 0))
    check("... and the cycle is left at refresh_done, not marked complete",
          m0.state == "refresh_done" and m0.get("recommender") is None,
          f"state={m0.state!r}")

    # --------------------------------------------------------- refresh failure
    print("      -- failure policy")
    root3 = os.path.join(tmp, "sched_fail")
    fake3 = FakePhases(root3)
    fake3.fail_refresh.add(1)
    sched3 = make_scheduler(os.path.join(root3, "run"), fake3)
    check("a refresh failure aborts the run (4)", raises(sched3.run, A.RefreshAborted))
    check("... the recommender phase of that cycle never ran (4)",
          [c for c in fake3.calls if c[0] == "recommender" and c[1] == 1] == [],
          f"recommender calls: {[c[1] for c in fake3.calls if c[0] == 'recommender']}")
    m1 = A.CycleManifest.load(A.cycle_manifest_path(os.path.join(root3, "run"), 1))
    m0 = A.CycleManifest.load(A.cycle_manifest_path(os.path.join(root3, "run"), 0))
    check("... cycle 1 is parked at tokenizer_done with the error recorded",
          m1.state == "tokenizer_done" and m1.get("sid") is None
          and [e["kind"] for e in m1.get("errors")] == ["SidRefreshError"],
          f"state={m1.state!r} errors={[e['kind'] for e in m1.get('errors')]}")
    check("... the already-complete cycle 0 is untouched",
          m0.state == "recommender_done" and m0.get("recommender") is not None)

    fake3.fail_refresh.clear()
    sched3.run()
    m1 = A.CycleManifest.load(A.cycle_manifest_path(os.path.join(root3, "run"), 1))
    check("re-running after the failure finishes the cycle",
          m1.state == "recommender_done" and m1.get("recommender") is not None)
    check("... and the tokenizer of cycle 1 was not re-trained on the retry "
          "(it resumed at the recorded state)",
          len([c for c in fake3.calls if c == ("tokenizer", 1, 2)]) == 1,
          f"{len([c for c in fake3.calls if c == ('tokenizer', 1, 2)])} tokenizer calls")

    # --------------------------------------------- resume from refresh_done (5)
    print("      -- resume")
    root4 = os.path.join(tmp, "sched_resume")
    fake4 = FakePhases(root4)
    fake4.fail_recommender.add(0)
    run4 = os.path.join(root4, "run")
    check("a recommender failure aborts the run",
          raises(make_scheduler(run4, fake4).run, RuntimeError))
    m0 = A.CycleManifest.load(A.cycle_manifest_path(run4, 0))
    version_before = m0.get("sid")["version"]
    check("the cycle is parked at refresh_done", m0.state == "refresh_done")
    check("resume_plan reports that cycle 0 re-enters at the recommender",
          [a for a in make_scheduler(run4, fake4).resume_plan()["actions"]
           if a["cycle"] == 0][0]["action"] == "recommender")

    fake4.fail_recommender.clear()
    make_scheduler(run4, fake4).run()
    m0 = A.CycleManifest.load(A.cycle_manifest_path(run4, 0))
    check("resuming from refresh_done does NOT refresh a second time (5)",
          fake4.refresh_calls[0] == 1,
          f"cycle 0 refreshed {fake4.refresh_calls[0]} time(s)")
    check("... and the SID version is the one already recorded, not a new one",
          m0.get("sid")["version"] == version_before, f"v{version_before}")
    check("... and the already-done initial refresh was not repeated either",
          fake4.refresh_calls.get(-1, 0) == 1,
          f"the initial refresh ran {fake4.refresh_calls.get(-1, 0)} time(s) across "
          f"the crash and the resume")
    check("... the recommender then runs and the cycle completes",
          m0.state == "recommender_done")

    # ------------------------------------------------- resumed tokenizer_done
    # parked at tokenizer_done (the refresh failed), so a resume must re-enter
    # the refresh branch -- and that branch re-checks the checkpoint it is about
    # to encode from.
    root5 = os.path.join(tmp, "sched_tokchanged")
    fake5 = FakePhases(root5)
    fake5.fail_refresh.add(0)
    run5 = os.path.join(root5, "run")
    raises(make_scheduler(run5, fake5).run, A.RefreshAborted)
    with open(os.path.join(root5, "tok", "c0.pth"), "ab") as fh:
        fh.write(b"tampered")
    fake5.fail_refresh.clear()
    check("a tokenizer checkpoint that changed since it was recorded is refused "
          "rather than silently re-used",
          raises(make_scheduler(run5, fake5).run, A.ResumeError))
    check("... and the SID was never generated from the swapped-in checkpoint",
          fake5.refresh_calls.get(0, 0) == 1,
          "the first (failed) refresh is the only one; the tampered resume was "
          "stopped before the generator ran")

    check("advance() refuses to skip a state",
          raises(lambda: A.CycleManifest.create(tmp, 77, DATASET).advance("refresh_done"),
                 A.ResumeError))
    check("advance() refuses to move backwards too",
          raises(lambda: A.CycleManifest.load(
              A.cycle_manifest_path(run_dir, 0)).advance("tokenizer_done"), A.ResumeError))
    bad = os.path.join(tmp, "cycles", "cycle_099.json")
    os.makedirs(os.path.dirname(bad), exist_ok=True)
    with open(bad, "w") as fh:
        fh.write(json.dumps({"schema": "something/else", "state": "started"}))
    check("an unknown manifest schema is refused, not interpreted",
          raises(lambda: A.CycleManifest.load(bad), A.ResumeError))
    with open(bad, "w") as fh:
        fh.write(json.dumps({"schema": A.CycleManifest.SCHEMA, "state": "finished"}))
    check("an unknown cycle state is refused",
          raises(lambda: A.CycleManifest.load(bad), A.ResumeError))
    os.remove(bad)


# --------------------------------------------------------------------------- #
# [3] the real pipeline on a synthetic fixture
# --------------------------------------------------------------------------- #
def build_fixture(root, rng):
    """A tiny but genuine dataset: item embeddings + train/valid/test parquets."""
    data_dir = os.path.join(root, DATASET)
    os.makedirs(data_dir, exist_ok=True)
    n_items, dim = SYN["n_items"], SYN["dim"]

    # mostly noise with a mild cluster signal: enough for the RQ-VAE to have
    # something to learn, not so concentrated that whole groups collide into one
    # code (E8 rejects a SID whose duplicates cannot be resolved).
    centres = rng.normal(size=(8, dim)).astype(np.float32) * 1.5
    which = rng.integers(0, 8, size=n_items)
    emb = (centres[which] + rng.normal(size=(n_items, dim)).astype(np.float32)).astype(
        np.float32)
    import pandas as pd
    pd.DataFrame({"embedding": [emb[i] for i in range(n_items)]}).to_parquet(
        os.path.join(data_dir, "item_emb.parquet"), index=False)

    def interactions(n_rows, lo, hi):
        history, target = [], []
        for _ in range(n_rows):
            seq = rng.integers(lo, hi, size=SYN["seq_len"]).tolist()
            history.append(seq[:-1])
            target.append(seq[-1])
        return pd.DataFrame({"history": history, "target": target})

    interactions(SYN["n_users"], 1, n_items + 1).to_parquet(
        os.path.join(data_dir, "train.parquet"), index=False)
    interactions(SYN["n_valid"], 1, n_items + 1).to_parquet(
        os.path.join(data_dir, "valid.parquet"), index=False)
    interactions(SYN["n_valid"], 1, n_items + 1).to_parquet(
        os.path.join(data_dir, "test.parquet"), index=False)
    return data_dir, emb


def build_pipeline(tmp, root, sid_base, run_dir, **kwargs):
    return AP.TigerAlternatingPipeline(
        DATASET, root, sid_base, run_dir, device=DEVICE, seed=SEED,
        rqvae_config=SYN_RQVAE, t5_config=SYN_T5, lambda_sia=0.1, lambda_psa=0.1,
        batch_size=32, infer_size=4, beam_size=2, **kwargs)


def part3_real_pipeline(tmp):
    print("\n[3] the real pipeline: 2 cycles over a synthetic fixture")
    root = os.path.join(tmp, "data")
    rng = np.random.default_rng(0)
    build_fixture(root, rng)
    sid_base = os.path.join(tmp, "sid")
    run_dir = os.path.join(tmp, "run")

    # --- a starting tokenizer, produced by the same frozen loop --------------
    boot = build_pipeline(tmp, root, os.path.join(tmp, "sid_boot"),
                          os.path.join(tmp, "bootstrap"))
    started = time.time()
    initial = boot.tokenizer_phase(0, 0, None)
    print(f"      (bootstrap RQ-VAE epoch {time.time() - started:.1f}s, "
          f"ckpt {os.path.relpath(initial['ckpt_path'], tmp)})")

    pipeline = build_pipeline(tmp, root, sid_base, run_dir)
    scheduler = AP.build_scheduler(pipeline, run_dir, 2, initial["ckpt_path"])
    t0 = time.time()
    records = scheduler.run()
    elapsed = time.time() - t0
    print(f"      (2 cycles took {elapsed:.1f}s on {DEVICE})")

    versions = SR.list_versions(sid_base, DATASET)
    check("2 cycles ran end to end (1), producing the whole artifact chain",
          len(records) == 2 and all(r["state"] == "recommender_done" for r in records),
          f"{len(records)} cycles, {elapsed:.1f}s")

    log = [(e["phase"], e["cycle_index"], e["epoch_index"]) for e in pipeline.phase_log]
    expected = [("refresh", -1, -1),
                ("tokenizer", 0, 0), ("refresh", 0, 0), ("recommender", 0, 1),
                ("evaluation", 0, 1),
                ("tokenizer", 1, 2), ("refresh", 1, 2), ("recommender", 1, 3),
                ("evaluation", 1, 3)]
    check("the real phase order is the specified one (2)", log == expected, f"{log}")

    check("exactly one initial refresh plus one refresh per cycle (1)",
          len(versions) == 3 and [v["version"] for v in versions] == [1, 2, 3],
          f"versions {[v['version'] for v in versions]}")

    # -------------------------------------------------- (3) the SID hand-off
    print("      -- SID hand-off (3)")
    for cycle, record in enumerate(records):
        sid = record["sid"]
        rec = record["recommender"]
        n_items = SR.load_sid(sid["sid_path"]).shape[0]
        check(f"cycle {cycle}: the recommender consumed this cycle's SID version",
              rec["sid_version_used"] == sid["version"] == 2 + cycle
              and rec["sid_sha256_used"] == sid["sid_sha256"],
              f"v{rec['sid_version_used']} (refreshed v{sid['version']})")
        check(f"cycle {cycle}: the dataset, both mounts and the file agree on that SID",
              rec["handoff"]["mismatches"] == {"sia": 0, "psa": 0}
              and rec["handoff"]["consistent_items"] == n_items
              and rec["handoff"]["sid_sha256"] == sid["sid_sha256"],
              f"{n_items} items checked against the sia and psa routing tables")
        check(f"cycle {cycle}: the dataset was built from that exact path",
              os.path.abspath(rec["code_path_seen_by_dataset"])
              == os.path.abspath(sid["sid_path"]))
        check(f"cycle {cycle}: the SID file's own E8 manifest names this ckpt",
              os.path.isfile(SR.sidecar_path(sid["sid_path"]))
              and SR.read_sid_manifest(sid["sid_path"])["ckpt_sha256"]
              == record["tokenizer"]["sha256"])
        check(f"cycle {cycle}: the E8 manifest refuses a different checkpoint",
              raises(lambda p=sid["sid_path"], r=record: SR.verify_sid_manifest(
                  SR.read_sid_manifest(p), ckpt_path=initial["ckpt_path"]),
                  SR.SidManifestMismatch)
              if record["tokenizer"]["sha256"] != initial["sha256"] else True)

    diff = records[1]["sid"]["diff"]
    check("cycle 1's refresh is a genuinely new encoding of the trained tokenizer, "
          "not a re-published old file",
          records[1]["sid"]["sid_sha256"] != records[0]["sid"]["sid_sha256"]
          and diff is not None and diff["rows_total"] == SYN["n_items"]
          and diff["rows_changed"] > 0,
          f"v{records[0]['sid']['version']} -> v{records[1]['sid']['version']}, "
          f"{diff['rows_changed']}/{diff['rows_total']} rows changed "
          f"({diff['rows_changed_rate']:.1%})")

    # negative control for the hand-off guard itself
    sid_a = records[0]["sid"]["sid_path"]
    sid_b = records[1]["sid"]["sid_path"]
    from tiger_ext.psa_loss import PSAMount
    from tiger_ext.sia_loss import SIAMount
    mods = AP.frozen_modules()
    ds_a = mods["GenRecDataset"](dataset_path=os.path.join(root, DATASET, "train.parquet"),
                                 code_path=sid_a, mode="train",
                                 max_len=SYN_T5["max_len"])
    tok_b = pipeline._sid_loader(sid_b)
    sia_b = SIAMount(pipeline.sia_adapter, pipeline.rqvae, pipeline.item_embeddings,
                     tok_b, lambda_sia=0.1)
    psa_b = PSAMount(pipeline.psa_adapter, pipeline.rqvae, pipeline.item_embeddings,
                     tok_b, lambda_psa=0.1)
    check("the hand-off guard fires when the dataset and the mounts disagree "
          "(dataset on v2, mounts on v3)",
          raises(lambda: pipeline.verify_sid_consistency(0, ds_a, sia_b, psa_b, sid_a,
                                                         tok_b), AssertionError))

    # ------------------------------------- (7)(8)(9) what each phase updates
    print("      -- freeze and optimizer (7/8/9)")
    pipeline.freeze.apply(STAGE_TOKENIZER)
    rq_b, t5_b, ad_b = snap(pipeline.rqvae), snap(pipeline.tiger), \
        snap([pipeline.sia_adapter, pipeline.psa_adapter])
    tok_out = pipeline.tokenizer_phase(90, 0, records[-1]["tokenizer"]["ckpt_path"])
    check("(7) the tokenizer phase really trains the RQ-VAE",
          n_changed(rq_b, pipeline.rqvae) > 0,
          f"{n_changed(rq_b, pipeline.rqvae)}/{len(rq_b)} RQ-VAE tensors moved")
    check("(7) ... and leaves T5 bitwise untouched",
          unchanged(t5_b, pipeline.tiger))
    check("(7) ... and leaves both adapters bitwise untouched",
          unchanged(ad_b, [pipeline.sia_adapter, pipeline.psa_adapter]))
    check("(7) the tokenizer phase's checkpoint stores a loadable stage record",
          json.load(open(tok_out["ckpt_path"] + ".stage.json"))["freeze"]["stage"]
          == STAGE_TOKENIZER
          and tok_out["losses"] == {"L_RQVAE": True, "L_T5": False,
                                    "L_SIA": False, "L_PSA": False})

    sid_rec = records[-1]["sid"]
    pipeline.freeze.apply(STAGE_RECOMMENDER)
    # the tokenizer epoch above left stale gradients on the RQ-VAE; clear them so
    # a gradient appearing after the recommender epoch can only come from it
    pipeline.rqvae.zero_grad(set_to_none=True)
    rq_b, t5_b, ad_b = snap(pipeline.rqvae), snap(pipeline.tiger), \
        snap([pipeline.sia_adapter, pipeline.psa_adapter])
    rec_out = pipeline.recommender_phase(91, 1, sid_rec["sid_path"], sid_rec["version"],
                                         records[-1]["recommender"]["ckpt_path"])
    check("(8) the recommender phase leaves the RQ-VAE bitwise unchanged",
          unchanged(rq_b, pipeline.rqvae),
          f"{n_changed(rq_b, pipeline.rqvae)}/{len(rq_b)} tensors moved")
    check("(8) ... even though the frozen RQ-VAE is still executed (PSA's target side)",
          all(p.grad is None for p in pipeline.rqvae.parameters()),
          f"{sum(p.grad is not None for p in pipeline.rqvae.parameters())} frozen "
          f"parameters carry a gradient")
    check("(9) the recommender phase trains T5",
          n_changed(t5_b, pipeline.tiger) > 0,
          f"{n_changed(t5_b, pipeline.tiger)}/{len(t5_b)} T5 tensors moved")
    check("(9) ... and the adapters, which are outside TIGER.parameters()",
          n_changed(ad_b, [pipeline.sia_adapter, pipeline.psa_adapter]) > 0,
          f"{n_changed(ad_b, [pipeline.sia_adapter, pipeline.psa_adapter])} adapter "
          f"tensors moved (they must be added to the optimizer explicitly)")
    check("(9) the recommender's stage record records the SID it consumed",
          json.load(open(rec_out["ckpt_path"] + ".stage.json"))["freeze"]["stage"]
          == STAGE_RECOMMENDER
          and rec_out["sid_version_used"] == sid_rec["version"])
    check("(10) every recorded evaluation is on a recommender epoch of its own cycle",
          all(A.phase_of(e["epoch_index"]) == STAGE_RECOMMENDER
              and e["cycle_index"] < len(records) for e in pipeline.phase_log
              if e["phase"] == "evaluation")
          and [e["cycle_index"] for e in pipeline.phase_log
               if e["phase"] == "evaluation"] == [0, 1])

    # the run manifest is a complete chain (acceptance 2)
    chain_ok = True
    for record in records:
        sidecar = SR.read_sid_manifest(record["sid"]["sid_path"])
        chain_ok &= (sidecar["ckpt_sha256"] == record["tokenizer"]["sha256"]
                     and sidecar["sid_sha256"] == record["sid"]["sid_sha256"]
                     and os.path.isfile(record["recommender"]["ckpt_path"])
                     and record["evaluation"]["ndcgs"].get("NDCG@20") is not None)
    check("every cycle's ckpt -> SID -> ckpt -> metric chain is complete and "
          "hash-linked (acceptance 2)", chain_ok)

    # ---------------------------------------------------------- (11) the seed
    print("      -- seed control (11)")
    seeds = {(e["phase"], e["epoch_index"]): e["seed"] for e in pipeline.phase_log
             if "seed" in e}
    check("(11) every seeded phase uses base + its global epoch index",
          all(seed == SEED + epoch for (_, epoch), seed in seeds.items())
          and len(seeds) >= 6,
          f"{sorted((k[1], v) for k, v in seeds.items())}")
    training_seeds = sorted(v for (phase, _), v in seeds.items()
                            if phase in ("tokenizer", "recommender"))
    check("(11) the four training epochs each get a distinct seed",
          len(set(training_seeds)) == 4, f"{training_seeds}")

    sid_base2 = os.path.join(tmp, "sid_repeat")
    run_dir2 = os.path.join(tmp, "run_repeat")
    pipeline2 = build_pipeline(tmp, root, sid_base2, run_dir2)
    AP.build_scheduler(pipeline2, run_dir2, 2, initial["ckpt_path"]).run()
    same = all(np.array_equal(
        SR.load_sid(records[c]["sid"]["sid_path"]),
        SR.load_sid(SR.list_versions(sid_base2, DATASET)[c + 1]["path"]))
        for c in range(2))
    check("(11) a second run with the same seed regenerates the SIDs bitwise",
          same,
          "the RQ-VAE (1 batch, seeded) and the generator are both deterministic")

    return root, pipeline


# --------------------------------------------------------------------------- #
# [4] regression and scope
# --------------------------------------------------------------------------- #
REGRESSION_SUITES = ["exp_e1_consistency", "exp_e2_encoder_interface",
                     "exp_e2_sia_adapter", "exp_e3_code_distribution",
                     "exp_e4_sia_loss", "exp_e5_psa", "exp_e7_stage_freeze",
                     "exp_e8_sid_refresh"]
COUNT_RE = re.compile(r"(\d+)/(\d+) checks passed")


def run_regression():
    print("\n[4b] regression: the E1-E8 suites, re-run after E9")
    total_ok = total_n = 0
    for name in REGRESSION_SUITES:
        path = os.path.join("integration", f"{name}.py")
        env = dict(os.environ)
        env.pop("EXP_E9_DEVICE", None)
        proc = subprocess.run([sys.executable, "-I", path], cwd=REPO, env=env,
                              capture_output=True, text=True)
        match = None
        for line in proc.stdout.splitlines():
            m = COUNT_RE.search(line)
            if m:
                match = m
        ok, n = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
        total_ok += ok
        total_n += n
        passed = proc.returncode == 0 and match is not None and ok == n
        # only on failure: the child's last stderr line is usually a tqdm bar,
        # and its carriage returns would clobber this line on a terminal
        detail = "" if passed else " | ".join(
            ln for ln in proc.stderr.splitlines() if ln.strip())[-200:]
        check(f"{name}: {ok}/{n}", passed, detail)
    check(f"(12) the E1-E8 suites are unaffected by E9 ({total_ok}/{total_n})",
          total_ok == total_n and total_n > 300, f"{total_ok}/{total_n} checks")


def part4_scope():
    print("\n[4a] scope: frozen directories untouched, no runtime coupling")
    status = subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                            capture_output=True, text=True).stdout.splitlines()
    modified = {l[3:].strip() for l in status if l[:2].strip() == "M"}
    check("model/ and rqvae/ carry no E9 change",
          modified <= PRE_E9_MODIFIED, f"modified: {sorted(modified)}")
    check("clean_tiger/ is untouched by E9",
          not [l for l in status if "clean_tiger" in l])
    grep = subprocess.run(
        ["grep", "-rn", "--include=*.py", "-E",
         "^[[:space:]]*(from|import)[[:space:]]+tiger_ext",
         os.path.join(REPO, "model"), os.path.join(REPO, "rqvae")],
        capture_output=True, text=True)
    check("nothing under model/ or rqvae/ imports tiger_ext",
          grep.returncode == 1 and not grep.stdout.strip())

    sched_src = open(A.__file__).read()
    check("the scheduler module imports no numerical stack: it only orchestrates",
          "import torch" not in sched_src and "import numpy" not in sched_src,
          "the phase bodies live in tiger_ext/alt_pipeline.py")
    check("the scheduler never writes a SID itself",
          "np.save" not in sched_src and "generate_code" not in sched_src)
    tree = ast.parse(open(AP.__file__).read())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "refresh_sid"]
    check("E9 never publishes over the canonical Clean TIGER SID: every refresh is "
          "an immutable versioned file handed over by --code_path",
          len(calls) == 1 and not [k for c in calls for k in c.keywords
                                   if k.arg == "publish"],
          f"{len(calls)} refresh_sid() call, publish never passed")
    def root_name(node):
        """The module alias behind an attribute access: ``m.x`` or ``mods['m'].x``."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            return node.slice.value
        return None

    called = {(root_name(n.func.value), n.func.attr) for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    check("the pipeline drives the frozen model/main.py loops directly, so the "
          "adapter-blind __main__ optimizer is never in the path",
          ("tiger_main", "train") in called and ("tiger_main", "evaluate") in called,
          f"{sorted(c for c in called if c[0] == 'tiger_main')}")


# --------------------------------------------------------------------------- #
def main():
    print("=" * 72)
    print(f"EXP-E9 -- alternating scheduler  (device={DEVICE}, regression={REGRESSION})")
    part1_schedule()
    tmp = tempfile.mkdtemp(prefix="exp_e9_")
    try:
        part2_scheduler_fakes(tmp)
        os.environ["EXP_E9_DEVICE"] = DEVICE
        part3_real_pipeline(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    part4_scope()
    if REGRESSION != "skip":
        run_regression()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 72)
    print(f"EXP-E9: {passed}/{len(RESULTS)} checks passed  (device={DEVICE})")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  -- {detail}")
    print("=" * 72)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
