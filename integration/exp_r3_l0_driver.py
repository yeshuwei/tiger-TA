#!/usr/bin/env python
"""EXP-R3: mechanism tests for integration/exp_l0_driver.py.

Run:
  python -I integration/exp_r3_l0_driver.py

No model is trained and no formal SID is refreshed.  A spy runner replaces the
subprocess layer and writes synthetic products into a temporary directory.
"""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

DRIVER_PATH = os.path.join(HERE, "exp_l0_driver.py")
_spec = importlib.util.spec_from_file_location("exp_l0_driver", DRIVER_PATH)
drv = importlib.util.module_from_spec(_spec)
sys.modules["exp_l0_driver"] = drv          # dataclasses needs this on py3.10
_spec.loader.exec_module(drv)

RESULTS = []
LOG_LINES = []


def emit(line=""):
    print(line)
    LOG_LINES.append(line)


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    emit(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def section(title):
    emit(f"\n{title}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data if isinstance(data, bytes) else data.encode())
    return path


# --------------------------------------------------------------------------- #
# spy runner: writes the products each stage is expected to produce
# --------------------------------------------------------------------------- #
class SpyRunner:
    def __init__(self, fail=None, tamper=None, timestamp="Oct-19-2026_12-00-00"):
        self.calls = []
        self.fail = fail or {}          # {(arm_tag, stage): rc}
        self.tamper = tamper or {}      # {arm_tag: mode}
        self.timestamp = timestamp
        self.stage_of_call = []

    @staticmethod
    def _arg(command, flag):
        return command[command.index(flag) + 1]

    def arm_tag(self, stage, command):
        if stage == "tokenizer":
            parent = self._arg(command, "--ckpt-dir")
        elif stage == "refresh":
            parent = command[5]                     # [-c, code, ds, ckpt, base, ...]
        else:
            parent = self._arg(command, "--run_dir")
        return os.path.basename(os.path.dirname(os.path.normpath(parent)))

    def __call__(self, command, log_path, echo=False):
        command = list(command)
        if "tiger_ext.qspace_train" in command:
            stage = "tokenizer"
        elif "tiger_ext.fixed_train" in command:
            stage = "recommender"
        else:
            stage = "refresh"
        tag = self.arm_tag(stage, command)
        self.calls.append(command)
        self.stage_of_call.append((tag, stage))
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as fh:
            fh.write(f"$ {shlex.join(command)}\n")
            fh.write("  best_loss            0.1234\n")
            fh.write("  best_collision_rate  0.000419\n")

        rc = self.fail.get((tag, stage), 0)
        if rc != 0:
            with open(log_path, "a") as fh:
                fh.write("injected failure\n")
            return rc, "injected failure\n"

        mode = self.tamper.get(tag)
        if stage == "tokenizer":
            parent = self._arg(command, "--ckpt-dir")
            ckpt_dir = os.path.join(parent, self.timestamp)
            write(os.path.join(ckpt_dir, "best_collision_model.pth"),
                  f"ckpt-{tag}")
        elif stage == "refresh":
            ckpt, base, manifest_out = command[4], command[5], command[7]
            version_dir = os.path.join(base, "sid_versions")
            os.makedirs(version_dir, exist_ok=True)
            existing = sorted(glob.glob(os.path.join(version_dir, "Toys_sid_v*.npy")))
            version = (int(os.path.basename(existing[-1])[10:14]) + 1) if existing else 1
            if mode != "no_sid":
                sha8 = sha256(ckpt)[:8]
                sid_path = os.path.join(
                    version_dir, f"Toys_sid_v{version:04d}_{sha8}.npy")
                write(sid_path, f"sid-{tag}-{version}")
                if mode != "no_sidecar":
                    write(sid_path + ".sid.json", "{}")
                if mode == "two_sids":
                    write(os.path.join(
                        version_dir,
                        f"Toys_sid_v{version + 1:04d}_{sha8}.npy"),
                        f"sid-{tag}-extra")
                sid_sha = sha256(sid_path)
            else:
                sid_path, sid_sha = "", ""
            record = {
                "version": version + (1 if mode == "bad_version" else 0),
                "sid_path": os.path.abspath(sid_path),
                "sid_sha256": ("0" * 64 if mode == "bad_sid_sha" else sid_sha),
                "ckpt_sha256": ("0" * 64 if mode == "bad_ckpt_sha"
                                else sha256(ckpt)),
                "sid_stats": {"ok": True},
                "n_items": 11924,
            }
            with open(manifest_out, "w") as fh:
                json.dump(record, fh)
        elif stage == "recommender":
            run_dir = self._arg(command, "--run_dir")
            os.makedirs(run_dir, exist_ok=True)
            write(os.path.join(run_dir, "best.pth"), b"best")
            with open(os.path.join(run_dir, "run.json"), "w") as fh:
                json.dump({"config": {}, "result": {
                    "best_ndcg@20": 0.0377, "best_epoch": 12,
                    "num_epochs_executed": 22, "early_stopped": True}}, fh)
        return 0, "ok\n"


def run(tmp):
    run_root = os.path.join(tmp, "runs", "run-test")
    record_path = os.path.join(tmp, "driver_run.json")
    return run_root, record_path


# --------------------------------------------------------------------------- #
# [1] arm configuration
# --------------------------------------------------------------------------- #
def part1():
    section("[1] five arms -- frozen configuration")
    specs = drv.ARM_SPECS
    check("exactly five arms", len(specs) == 5, str([s.tag for s in specs]))
    check("execution order", drv.ARM_ORDER ==
          ["lc1e-4", "lc1e-3", "lc1e-2", "le0.1", "le1.0"], str(drv.ARM_ORDER))
    expected = {
        "lc1e-4": ("csg", 1e-4, "1e-4", 0.0, 0.0, "EXP-L0 λc-1e-4"),
        "lc1e-3": ("csg", 1e-3, "1e-3", 0.0, 0.0, "EXP-L0 λc-1e-3"),
        "lc1e-2": ("csg", 1e-2, "1e-2", 0.0, 0.0, "EXP-L0 λc-1e-2"),
        "le0.1": ("ete", None, "", 0.1, 0.1, "EXP-L0 λe-0.1"),
        "le1.0": ("ete", None, "", 1.0, 1.0, "EXP-L0 λe-1.0"),
    }
    for tag, (kind, lc, lc_text, sia, psa, label) in expected.items():
        spec = drv.ARM_BY_TAG[tag]
        ok = (spec.kind == kind and spec.lambda_csg == lc
              and spec.lambda_csg_text == lc_text
              and spec.lambda_sia == sia and spec.lambda_psa == psa
              and spec.label == label)
        check(f"{tag}: kind/lambdas/label", ok,
              f"{spec.kind} lc={spec.lambda_csg} sia={spec.lambda_sia} psa={spec.lambda_psa}")
    check("csg arms use lambda_sia=lambda_psa=0",
          all(s.lambda_sia == 0.0 and s.lambda_psa == 0.0
              for s in specs if s.kind == "csg"))


# --------------------------------------------------------------------------- #
# [2] commands match the R2 freeze
# --------------------------------------------------------------------------- #
def part2():
    section("[2] commands vs EXP-R2 frozen commands")
    r2 = json.load(open(os.path.join(
        HERE, "artifacts/r2_exp_l0_cli/exp_r2_audit.json")))
    frozen = r2["commands"]

    def diff_only(actual, expected, flag):
        """"True iff the two token lists differ only in the value of ``flag``."""
        if len(actual) != len(expected) or actual.count(flag) != 1:
            return False, "length/flag-count mismatch"
        i = actual.index(flag)
        a, e = list(actual), list(expected)
        if a[:i] != e[:i] or a[i + 2:] != e[i + 2:]:
            return False, f"tokens outside {flag} differ"
        return a[i] == e[i], f"{flag}: {a[i + 1]} vs {e[i + 1]}"

    for tag in ("lc1e-4", "lc1e-3", "lc1e-2"):
        spec = drv.ARM_BY_TAG[tag]
        paths = drv.arm_paths("/tmp/run", spec)
        actual = drv.tokenizer_command(spec, paths, "python", "cuda:0")
        expected = shlex.split(frozen["csg"][spec.lambda_csg_text])
        ok, detail = diff_only(actual, expected, "--ckpt-dir")
        check(f"{tag}: tokenizer == R2 except run-scoped --ckpt-dir", ok, detail)

    for tag in ("le0.1", "le1.0"):
        spec = drv.ARM_BY_TAG[tag]
        paths = drv.arm_paths("/tmp/run", spec)
        actual = drv.recommender_command(spec, paths, drv.BASELINE_SID,
                                         drv.BASELINE_CKPT, "python", "cuda:0")
        expected = shlex.split(frozen["ete"][tag])
        ok, detail = diff_only(actual, expected, "--run_dir")
        check(f"{tag}: recommender == R2 except run-scoped --run_dir", ok, detail)

    for tag in ("lc1e-4", "lc1e-3", "lc1e-2"):
        spec = drv.ARM_BY_TAG[tag]
        paths = drv.arm_paths("/tmp/run", spec)
        actual = drv.recommender_command(
            spec, paths, "/tmp/run/lc/sid/sid_versions/x.npy",
            "/tmp/run/lc/ckpt/stamp/best_collision_model.pth", "python", "cuda:0")
        expected = shlex.split(frozen["ete"]["le1.0"])
        for flag, wanted in (
                ("--sid_path", "/tmp/run/lc/sid/sid_versions/x.npy"),
                ("--sid_manifest", "/tmp/run/lc/sid/sid_versions/x.npy.sid.json"),
                ("--tokenizer_ckpt",
                 "/tmp/run/lc/ckpt/stamp/best_collision_model.pth"),
                ("--arm", spec.label),
                ("--lambda_sia", "0"), ("--lambda_psa", "0"),
                ("--run_dir", paths.run_dir),
                ("--seed", "2025"), ("--num_epochs", "200"),
                ("--early_stop", "10"), ("--batch_size", "256"),
                ("--infer_size", "96"), ("--beam_size", "30"),
                ("--lr", "1e-4"), ("--num_workers", "1")):
            i = expected.index(flag)
            expected[i + 1] = wanted
        check(f"{tag}: recommender == R2 template with lc substitutions",
              actual == expected, f"{len(actual)} tokens")

    all_commands = []
    for spec in drv.ARM_SPECS:
        paths = drv.arm_paths("/tmp/run", spec)
        if spec.kind == "csg":
            all_commands.append(drv.tokenizer_command(spec, paths, "python", "cuda:0"))
            all_commands.append(drv.refresh_command(spec, paths, "/tmp/ckpt.pth",
                                                    "python", "cuda:0"))
        all_commands.append(drv.recommender_command(
            spec, paths, drv.BASELINE_SID, drv.BASELINE_CKPT, "python", "cuda:0"))
    check("no --allow_unregistered_sid anywhere",
          all("--allow_unregistered_sid" not in c for c in all_commands))
    qspace = [c for c in all_commands if "tiger_ext.qspace_train" in c]
    fixed = [c for c in all_commands if "tiger_ext.fixed_train" in c]
    check("qspace_train uses --embedding-path (kebab-case)",
          all("--embedding-path" in c and "--embedding_path" not in c
              for c in qspace))
    check("fixed_train uses --embedding_path (snake_case)",
          all("--embedding_path" in c and "--embedding-path" not in c
              for c in fixed))
    check("all commands carry the frozen 768d embedding path",
          all(drv.EMBEDDING_PATH in c for c in qspace + fixed))
    check("lambda_c arms never pass baseline SID to the recommender",
          all(drv.BASELINE_SID not in c
              for c in (drv.recommender_command(
                  drv.ARM_BY_TAG[t], drv.arm_paths("/tmp/run", drv.ARM_BY_TAG[t]),
                  "/tmp/new.npy", "/tmp/new.pth", "python", "cuda:0")
                  for t in ("lc1e-4", "lc1e-3", "lc1e-2"))))


# --------------------------------------------------------------------------- #
# [3] dry-run
# --------------------------------------------------------------------------- #
def part3(tmp):
    section("[3] dry-run -- plan only, no side effects")
    run_root, record_path = run(tmp)
    plan = drv.dry_run_plan(drv.ARM_SPECS, run_root, "python", "cuda:0")
    check("dry-run plan lists five arms", len(plan["arms"]) == 5)
    check("dry-run plan lists 11 stages (3x3 csg + 2 ete)",
          sum(len(a["stages"]) for a in plan["arms"]) == 11,
          str([len(a["stages"]) for a in plan["arms"]]))
    check("execution order lc x3 then le x2",
          plan["execution_order"] == drv.ARM_ORDER)
    csg = [a for a in plan["arms"] if a["arm"]["kind"] == "csg"]
    check("csg stages are tokenizer -> refresh -> recommender",
          all([s["name"] for s in a["stages"]] ==
              ["tokenizer", "refresh", "recommender"] for a in csg))
    check("csg dependencies are chained",
          all([s["depends_on"] for s in a["stages"]] ==
              [None, "tokenizer", "refresh"] for a in csg))
    check("ete arms only have the recommender",
          all([s["name"] for s in a["stages"]] == ["recommender"]
              for a in plan["arms"] if a["arm"]["kind"] == "ete"))
    check("no directory created by dry_run_plan", not os.path.exists(run_root))
    check("no directory created by execute_run(dry_run=True)",
          not os.path.exists(run_root) and
          drv.execute_run(drv.ARM_SPECS, run_root, record_path,
                          runner=drv._no_runner, dry_run=True)["status"] == "dry-run"
          and not os.path.exists(run_root))
    calls = []
    drv.execute_run(drv.ARM_SPECS, run_root, record_path,
                    runner=lambda *a, **k: calls.append(a), dry_run=True)
    check("dry-run never calls the runner", calls == [])
    check("dry-run does not write the record", not os.path.exists(record_path))


# --------------------------------------------------------------------------- #
# [4] happy path with synthetic products
# --------------------------------------------------------------------------- #
def part4(tmp):
    section("[4] execution -- synthetic fixture, dynamic product identification")
    run_root, record_path = run(tmp)
    spy = SpyRunner()
    record = drv.execute_run(drv.ARM_SPECS, run_root, record_path,
                             runner=spy, echo=False)
    check("run status completed", record["status"] == "completed",
          record["status"])
    check("all five arms completed",
          record["arms_completed"] == drv.ARM_ORDER,
          str(record["arms_completed"]))
    check("11 stage calls (3x3 csg + 2 ete)", len(spy.calls) == 11,
          str(len(spy.calls)))
    expected_order = []
    for tag in drv.ARM_ORDER:
        if drv.ARM_BY_TAG[tag].kind == "csg":
            expected_order += [(tag, "tokenizer"), (tag, "refresh"),
                               (tag, "recommender")]
        else:
            expected_order += [(tag, "recommender")]
    check("per-arm stage order", spy.stage_of_call == expected_order,
          str(spy.stage_of_call))

    arms = {a["arm"]["tag"]: a for a in record["arms"]}
    for tag in drv.ARM_ORDER:
        arm = arms[tag]
        check(f"{tag}: record carries lambda_csg/sia/psa",
              arm["arm"]["lambda_csg"] == drv.ARM_BY_TAG[tag].lambda_csg
              and arm["arm"]["lambda_sia"] == drv.ARM_BY_TAG[tag].lambda_sia
              and arm["arm"]["lambda_psa"] == drv.ARM_BY_TAG[tag].lambda_psa)
    check("csg checkpoint identified from the new timestamp dir",
          all(arms[t]["products"].get("ckpt_is_new") is True
              and arms[t]["products"]["ckpt_dir"].endswith("Oct-19-2026_12-00-00")
              for t in ("lc1e-4", "lc1e-3", "lc1e-2")))
    check("csg SID identified as a new version",
          all(arms[t]["products"].get("sid_is_new") is True
              and arms[t]["products"]["sid_version"] == 1
              for t in ("lc1e-4", "lc1e-3", "lc1e-2")))
    check("csg recommender consumed the identified SID",
          all(arms[t]["stages"][-1]["inputs"]["sid_path"]
              == arms[t]["products"]["sid_path"]
              for t in ("lc1e-4", "lc1e-3", "lc1e-2")))
    check("csg recommender consumed the identified checkpoint",
          all(arms[t]["stages"][-1]["inputs"]["tokenizer_ckpt"]
              .startswith(arms[t]["products"]["ckpt_dir"])
              for t in ("lc1e-4", "lc1e-3", "lc1e-2")))
    check("manifest verification recorded for all csg arms",
          all(arms[t]["products"].get("manifest_verified") is True
              for t in ("lc1e-4", "lc1e-3", "lc1e-2")))
    check("ete arms use the frozen baseline SID/checkpoint",
          all(arms[t]["stages"][0]["inputs"]["sid_path"]
              == drv.BASELINE_SID and
              arms[t]["stages"][0]["inputs"]["tokenizer_ckpt"]
              == drv.BASELINE_CKPT for t in ("le0.1", "le1.0")))
    check("stage exit codes recorded",
          all(s["exit_code"] == 0 for a in record["arms"] for s in a["stages"]))
    check("stage start/finish timestamps recorded",
          all(s["started_at"] and s["finished_at"]
              for a in record["arms"] for s in a["stages"]))
    check("recommender metrics recorded",
          arms["lc1e-4"]["products"]["best_ndcg@20"] == 0.0377)
    check("embedding 768d path recorded",
          record["frozen_recipe"]["embedding_path"] == drv.EMBEDDING_PATH
          and record.get("embedding", {}).get("width") == 768,
          str(record.get("embedding", {}).get("width")))
    check("seed/device recorded",
          record["frozen_recipe"]["seed_rqvae"] == 2024
          and record["frozen_recipe"]["seed_t5"] == 2025
          and record["environment"]["device"] == "cuda:0")
    check("hashes match the synthetic products",
          all(arms[t]["products"]["ckpt_sha256"] ==
              sha256(os.path.join(REPO, arms[t]["products"]["ckpt_dir"],
                                  "best_collision_model.pth"))
              for t in ("lc1e-4", "lc1e-3", "lc1e-2")))
    check("canonical record written", os.path.isfile(record_path))
    check("immutable per-run record written",
          os.path.isfile(os.path.join(run_root, "driver_run.json")))
    check("record status/finish fields present",
          record["finished_at"] and record["status"] == "completed")


# --------------------------------------------------------------------------- #
# [5] dynamic SID version resolution (parsed, not hardcoded)
# --------------------------------------------------------------------------- #
def part5(tmp):
    section("[5] dynamic SID version resolution")
    base = os.path.join(tmp, "dyn", "sid")
    vd = os.path.join(base, "sid_versions")
    write(os.path.join(vd, "Toys_sid_v0001_aaaaaaaa.npy"), b"v1")
    write(os.path.join(vd, "Toys_sid_v0003_bbbbbbbb.npy"), b"v3")
    write(os.path.join(vd, "Toys_sid_v0007_notasha.npy"), b"ignored")
    before = drv.SR.list_versions(base, drv.DATASET)
    write(os.path.join(vd, "Toys_sid_v0008_cccccccc.npy"), b"v8")
    new = drv.identify_new_sid(base, before)
    check("new version parsed dynamically (v0008, not max-hardcoded v0003)",
          new["version"] == 8 and new["path"].endswith("v0008_cccccccc.npy"),
          str(new))
    check("non-matching filenames ignored",
          all("notasha" not in v["path"] for v in drv.SR.list_versions(base, drv.DATASET)))

    # checkpoint timestamp is also dynamic: a different stamp is picked up
    parent = os.path.join(tmp, "dyn", "ckpt")
    os.makedirs(parent)
    before_ckpts = drv.list_subdirs(parent)
    stamp = "Jan-01-2027_00-00-01"
    write(os.path.join(parent, stamp, "best_collision_model.pth"), b"new")
    ckpt_dir, ckpt = drv.identify_new_ckpt(parent, before_ckpts)
    check("new checkpoint timestamps identified dynamically",
          ckpt_dir.endswith(stamp) and os.path.isfile(ckpt), ckpt_dir)


# --------------------------------------------------------------------------- #
# [6] failure injection
# --------------------------------------------------------------------------- #
def part6(tmp):
    section("[6] failure stops the arm and all later arms")
    cases = [
        ("tokenizer failure on arm 2",
         {("lc1e-3", "tokenizer"): 3}, None, drv.ARM_SPECS,
         "lc1e-3", "tokenizer", ["lc1e-4"]),
        ("refresh failure on arm 1",
         {("lc1e-4", "refresh"): 1}, None, drv.ARM_SPECS,
         "lc1e-4", "refresh", []),
        ("recommender failure on le0.1",
         {("le0.1", "recommender"): 2}, None,
         [drv.ARM_BY_TAG["le0.1"], drv.ARM_BY_TAG["le1.0"]],
         "le0.1", "recommender", []),
    ]
    for name, fail, _, specs, bad_tag, bad_stage, completed in cases:
        root = os.path.join(tmp, "fail", name.replace(" ", "_"))
        spy = SpyRunner(fail=fail)
        rec = drv.execute_run(specs, root, os.path.join(root, "record.json"),
                              runner=spy, echo=False)
        tags = [a["arm"]["tag"] for a in rec["arms"]]
        bad = [a for a in rec["arms"] if a["arm"]["tag"] == bad_tag][0]
        check(f"{name}: status failed", rec["status"] == "failed")
        check(f"{name}: failed stage = {bad_stage}",
              bad["failed_stage"] == bad_stage and bad["status"] == "failed")
        check(f"{name}: later arms not started",
              tags == [s.tag for s in specs[:specs.index(drv.ARM_BY_TAG[bad_tag]) + 1]],
              str(tags))
        check(f"{name}: completed arms recorded",
              rec["arms_completed"] == completed, str(rec["arms_completed"]))

    section("[6b] provenance tampering fails the arm")
    for mode, needle in (
            ("bad_sid_sha", "SID sha256"),
            ("bad_ckpt_sha", "checkpoint sha256"),
            ("bad_version", "version"),
            ("no_sidecar", "sidecar"),
            ("two_sids", "exactly one new SID"),
            ("no_sid", "exactly one new SID")):
        root = os.path.join(tmp, "tamper", mode)
        specs = [drv.ARM_BY_TAG["lc1e-4"]]
        spy = SpyRunner(tamper={"lc1e-4": mode})
        rec = drv.execute_run(specs, root, os.path.join(root, "record.json"),
                              runner=spy, echo=False)
        arm = rec["arms"][0]
        check(f"tamper {mode}: refresh stage fails",
              rec["status"] == "failed" and arm["failed_stage"] == "refresh",
              arm.get("failure_reason", ""))
        check(f"tamper {mode}: reason names the problem",
              needle in (arm.get("failure_reason") or ""),
              (arm.get("failure_reason") or "")[:120])
        check(f"tamper {mode}: recommender never runs",
              not any(s["name"] == "recommender" for s in arm["stages"]))


# --------------------------------------------------------------------------- #
# [7] no reuse of old products
# --------------------------------------------------------------------------- #
def part7(tmp):
    section("[7] old products are never reused")
    # 7a: a non-empty arm directory blocks the run before any subprocess
    root = os.path.join(tmp, "reuse", "runs", "old-run")
    stale = os.path.join(root, "lc1e-4", "run")
    write(os.path.join(stale, "run.json"), b"{}")
    spy = SpyRunner()
    rec = drv.execute_run(drv.ARM_SPECS, root, os.path.join(tmp, "reuse", "rec.json"),
                          runner=spy, echo=False)
    check("non-empty arm dir -> preflight failure",
          rec["status"] == "failed" and rec["arms"][0]["failed_stage"] == "preflight")
    check("preflight failure starts no subprocess", spy.calls == [])
    check("preflight failure records zero completed arms",
          rec["arms_completed"] == [])

    # 7b: pre-existing SID versions in the arm's own sid_base block the run
    root = os.path.join(tmp, "reuse2", "runs", "run")
    write(os.path.join(root, "lc1e-4", "sid", "sid_versions",
                       "Toys_sid_v0001_aaaaaaaa.npy"), b"old")
    spy = SpyRunner()
    rec = drv.execute_run([drv.ARM_BY_TAG["lc1e-4"]], root,
                          os.path.join(tmp, "reuse2", "rec.json"),
                          runner=spy, echo=False)
    check("pre-existing SID version -> preflight failure, no subprocess",
          rec["status"] == "failed"
          and rec["arms"][0]["failed_stage"] == "preflight"
          and spy.calls == [] and not rec["arms_completed"],
          rec["arms"][0].get("failure_reason", ""))

    # 7c: products from an older run root are never referenced
    old_root = os.path.join(tmp, "old", "runs", "old-run")
    old_ckpt = write(os.path.join(old_root, "lc1e-4", "ckpt", "OLD",
                                  "best_collision_model.pth"), b"old")
    old_sid = write(os.path.join(old_root, "lc1e-4", "sid", "sid_versions",
                                 "Toys_sid_v0009_deadbeef.npy"), b"old")
    root, record_path = run(os.path.join(tmp, "fresh"))
    spy = SpyRunner()
    drv.execute_run([drv.ARM_BY_TAG["lc1e-4"]], root, record_path,
                    runner=spy, echo=False)
    referenced = " ".join(" ".join(c) for c in spy.calls)
    check("old checkpoint is never referenced", old_ckpt not in referenced)
    check("old SID is never referenced", old_sid not in referenced)
    check("all new products live under this run root",
          all(root in " ".join(c) for c in spy.calls if "fixed_train" in c))


# --------------------------------------------------------------------------- #
# [8] CLI dry-run (end to end)
# --------------------------------------------------------------------------- #
def part8(tmp):
    section("[8] CLI dry-run (subprocess, no training)")
    root = os.path.join(tmp, "cli", "runs", "cli-run")
    proc = subprocess.run(
        [sys.executable, "-I", DRIVER_PATH, "--dry-run",
         "--run-root", root],
        cwd=REPO, capture_output=True, text=True)
    check("CLI dry-run exit 0", proc.returncode == 0,
          proc.stderr.strip()[-200:])
    try:
        plan = json.loads(proc.stdout)
        ok = plan["mode"] == "dry-run" and len(plan["arms"]) == 5
        commands = [s["command"] for a in plan["arms"] for s in a["stages"]]
    except Exception:
        ok, commands = False, []
    check("CLI dry-run prints a parseable five-arm plan", ok)
    check("CLI dry-run creates no run root", not os.path.exists(root))

    def value(command, flag):
        return command[command.index(flag) + 1] if flag in command else None

    check("CLI dry-run commands use the frozen lambdas",
          any(value(c, "--lambda-csg") == "1e-4" for c in commands)
          and any(value(c, "--lambda-csg") == "1e-2" for c in commands)
          and any(value(c, "--lambda_sia") == "0.1"
                  and value(c, "--lambda_psa") == "0.1" for c in commands)
          and any(value(c, "--lambda_sia") == "1.0"
                  and value(c, "--lambda_psa") == "1.0" for c in commands))


# --------------------------------------------------------------------------- #
# [9] production files / git state
# --------------------------------------------------------------------------- #
def part9():
    section("[9] production files untouched")
    expected = {
        "tiger_ext/qspace_train.py": "305c50a7bf530f537da23a578950eb5eef74a7de094e6f77cc05045936b6193c",
        "tiger_ext/fixed_train.py": "389e90854b5f6f13371c00942c82fd5a779011635fa8eaa18bf40c5a3a88ce28",
        "tiger_ext/sid_refresh.py": "983121cb3d9cd17bbb337fc8eb7da7e6e6a26df732f19533c8367c4218272bc3",
        "tiger_ext/alt_pipeline.py": "fc1b8652a72c04cec685192604841a3f8fcd64c8641e3f1c61e635cee3a86677",
    }
    for rel, want in expected.items():
        got = sha256(os.path.join(REPO, rel))
        check(f"unchanged: {rel}", got == want, got[:16])
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout
    want_status = (" M model/main.py\n M rqvae/models/rq.py\n"
                   " M rqvae/models/vq.py\n M rqvae/trainer.py\n"
                   "?? integration/\n?? tiger_ext/\n")
    check("git status unchanged", status == want_status, repr(status))
    check("no EXP-L0 artifacts created by the tests",
          not os.path.exists(os.path.join(
              REPO, "integration/artifacts/exp_l0")))


# --------------------------------------------------------------------------- #
# [10] M1/M2 regression: empty selection and exceptions raised by a stage
# --------------------------------------------------------------------------- #
def part10(tmp):
    section("[10] M1 -- an empty arm selection is rejected before any execution")
    for value in ("", ",", " , "):
        try:
            picked = drv.select_specs(value)
            raised, detail = False, f"returned {len(picked)} arms"
        except drv.DriverError as exc:
            raised, detail = True, str(exc)
        check(f"select_specs({value!r}) raises DriverError", raised, detail)

    root = os.path.join(tmp, "m1", "runs", "empty")
    rec_path = os.path.join(tmp, "m1", "driver_run.json")
    rc = drv.main(["--arms", "", "--run-root", root, "--record", rec_path,
                   "--quiet"])
    check("main(--arms '') -> rc 2", rc == 2, f"rc={rc}")
    check("empty selection writes no record", not os.path.exists(rec_path))
    check("empty selection creates no run root", not os.path.exists(root))

    root2 = os.path.join(tmp, "m1", "runs", "empty2")
    rec2 = os.path.join(tmp, "m1", "rec2.json")
    raised = False
    try:
        drv.execute_run([], root2, rec2, runner=SpyRunner(), echo=False)
    except drv.DriverError:
        raised = True
    check("execute_run([]) raises DriverError", raised)
    check("execute_run([]) writes no record and no run root",
          not os.path.exists(rec2) and not os.path.exists(root2))

    section("[10b] M2 -- a stage exception still closes the whole-run record")

    class Boom:
        def __init__(self):
            self.calls = 0

        def __call__(self, command, log_path, echo=False):
            self.calls += 1
            raise OSError("cannot spawn python")

    root = os.path.join(tmp, "m2", "runs", "boom")
    rec_path = os.path.join(tmp, "m2", "driver_run.json")
    spy = Boom()
    raised = None
    try:
        drv.execute_run(drv.ARM_SPECS, root, rec_path, runner=spy, echo=False)
    except OSError as exc:
        raised = exc
    check("Popen failure propagates", isinstance(raised, OSError), repr(raised))
    check("Popen failure -> record written", os.path.isfile(rec_path))
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    check("Popen failure -> run status failed", rec.get("status") == "failed",
          str(rec.get("status")))
    arm = (rec.get("arms") or [{}])[0]
    check("Popen failure -> arm failed at tokenizer",
          arm.get("status") == "failed" and arm.get("failed_stage") == "tokenizer",
          f"{arm.get('status')}/{arm.get('failed_stage')}")
    failure = rec.get("failure") or {}
    check("Popen failure -> type/arm/message/traceback recorded",
          failure.get("type") == "OSError"
          and failure.get("arm") == "lc1e-4"
          and failure.get("stage") == "tokenizer"
          and "cannot spawn python" in (failure.get("message") or "")
          and bool((failure.get("traceback") or "").strip()),
          f"{failure.get('type')} {failure.get('arm')} {failure.get('stage')}")
    check("Popen failure -> no stage left 'running'",
          all(s.get("status") != "running" for s in (arm.get("stages") or [])))
    check("Popen failure -> zero arms completed",
          rec.get("arms_completed") == [], str(rec.get("arms_completed")))
    check("Popen failure -> later arms not started",
          rec.get("arms_not_started") == ["lc1e-3", "lc1e-2", "le0.1", "le1.0"],
          str(rec.get("arms_not_started")))
    check("Popen failure -> only the first stage ever ran", spy.calls == 1,
          str(spy.calls))
    check("Popen failure -> run-root copy written",
          os.path.isfile(os.path.join(root, "driver_run.json")))

    class CorruptRunJson:
        def __init__(self):
            self.calls = 0

        def __call__(self, command, log_path, echo=False):
            self.calls += 1
            run_dir = command[command.index("--run_dir") + 1]
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "run.json"), "w") as fh:
                fh.write("{not json")
            return 0, "ok\n"

    root = os.path.join(tmp, "m2", "runs", "corrupt")
    rec_path = os.path.join(tmp, "m2", "corrupt.json")
    specs = [drv.ARM_BY_TAG["le0.1"], drv.ARM_BY_TAG["le1.0"]]
    spy = CorruptRunJson()
    raised = None
    try:
        drv.execute_run(specs, root, rec_path, runner=spy, echo=False)
    except ValueError as exc:
        raised = exc
    check("corrupt run.json -> JSONDecodeError propagates", raised is not None,
          repr(raised))
    check("corrupt run.json -> record written", os.path.isfile(rec_path))
    rec = json.load(open(rec_path)) if os.path.isfile(rec_path) else {}
    arm = (rec.get("arms") or [{}])[0]
    check("corrupt run.json -> run status failed", rec.get("status") == "failed")
    check("corrupt run.json -> arm failed at recommender",
          arm.get("status") == "failed"
          and arm.get("failed_stage") == "recommender",
          f"{arm.get('status')}/{arm.get('failed_stage')}")
    check("corrupt run.json -> reason names JSONDecodeError",
          "JSONDecodeError" in (arm.get("failure_reason") or ""),
          (arm.get("failure_reason") or "")[:90])
    check("corrupt run.json -> failure records the arm",
          (rec.get("failure") or {}).get("arm") == "le0.1"
          and (rec.get("failure") or {}).get("type") == "JSONDecodeError")
    check("corrupt run.json -> next arm not started",
          rec.get("arms_not_started") == ["le1.0"] and spy.calls == 1,
          f"{rec.get('arms_not_started')} calls={spy.calls}")


# --------------------------------------------------------------------------- #
def main():
    tmp = tempfile.mkdtemp(prefix="exp_r3_")
    try:
        part1()
        part2()
        part3(os.path.join(tmp, "p3"))
        part4(os.path.join(tmp, "p4"))
        part5(os.path.join(tmp, "p5"))
        part6(os.path.join(tmp, "p6"))
        part7(os.path.join(tmp, "p7"))
        part8(os.path.join(tmp, "p8"))
        part9()
        part10(os.path.join(tmp, "p10"))
    except Exception:
        traceback.print_exc()
        check("harness completed without an exception", False,
              traceback.format_exc().splitlines()[-1])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    verdict = "CLOSED-BY-MECHANISM-TESTS" if passed == total else "BLOCKED"
    emit(f"\nEXP-R3: {passed}/{total} checks passed -- {verdict}")
    out_dir = os.path.join(HERE, "artifacts/r3_l0_driver")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp_r3.log"), "w") as fh:
        fh.write("\n".join(LOG_LINES) + "\n")
    with open(os.path.join(out_dir, "exp_r3_audit.json"), "w") as fh:
        json.dump({
            "suite": "EXP-R3",
            "verdict": verdict,
            "passed": passed,
            "total": total,
            "checks": [{"name": n, "ok": ok, "detail": d}
                       for n, ok, d in RESULTS],
        }, fh, indent=2, ensure_ascii=False)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
