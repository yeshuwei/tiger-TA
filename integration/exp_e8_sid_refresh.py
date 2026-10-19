"""EXP-E8 -- SID refresh: engineering / acceptance verification (A class).

Run:  python -I integration/exp_e8_sid_refresh.py
      EXP_E8_DEVICE=cpu python -I integration/exp_e8_sid_refresh.py
      EXP_E8_FAST=1 python -I integration/exp_e8_sid_refresh.py   # skip 2nd device

Covers plan.md E8's five acceptance criteria plus the operational concerns this
Task was asked to check (overwrite risk, failure state, ckpt<->SID traceability):

  [1] API, versioned naming and version discovery
  [2] REAL generation from a real checkpoint -> acceptance (1) coverage,
      (2) collision handling preserved, (3) token range, (5) traceability;
      plus bitwise equality with the frozen pipeline's own SID on the same device
  [3] acceptance (4): diff rate within a device (=0) and across devices (noise floor)
  [4] checkpoint <-> SID traceability; the manifest refuses to be reused
  [5] overwrite risk: versioned paths are never written in place
  [6] failure state: a failed round leaves no SID, no sidecar, no temp file
  [7] publish (atomic hand-off to the canonical path T5 reads)
  [8] scope: no new encoding mechanism, no runtime coupling, frozen dirs untouched

Engineering gate only -- nothing here says anything about recommendation quality.
Fixture: Toys (11924 items, 4096-d), the same checkpoint the frozen baseline used.
"""
import copy
import glob
import os
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

from tiger_ext import sid_refresh as SR          # noqa: E402

HAVE_CUDA = torch.cuda.is_available()
DEVICE = os.environ.get("EXP_E8_DEVICE") or ("cuda:0" if HAVE_CUDA else "cpu")
OTHER_DEVICE = "cpu" if str(DEVICE).startswith("cuda") else "cuda:0"
FAST = bool(os.environ.get("EXP_E8_FAST")) or (OTHER_DEVICE.startswith("cuda")
                                              and not HAVE_CUDA)
GEN_TIMEOUT = 1800

# frozen clean_tiger fixture: the SID + checkpoint the baseline pipeline produced
SMOKE_DIR = os.path.join(REPO, "clean_tiger", "_smoke", "Toys")
SMOKE_SID = os.path.join(SMOKE_DIR, "Toys_clean_tiger_sid.npy")
SMOKE_CKPT = os.path.join(REPO, "clean_tiger", "ckpt", "Toys", "rqvae",
                          "Oct-14-2026_05-55-09", "best_collision_model.pth")
# the tracked files E1-E6 legitimately modified; E8 must not add to this set
PRE_E8_MODIFIED = {"model/main.py", "rqvae/models/rq.py", "rqvae/models/vq.py"}

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def raises(fn, exc):
    """True only if fn raises exactly the expected E8 error type."""
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def count(pattern):
    return len(glob.glob(pattern, recursive=True))


def main():
    print("=" * 70)
    print(f"EXP-E8 -- SID refresh  (device={DEVICE}, other={OTHER_DEVICE}, fast={FAST})")
    for p in (SMOKE_SID, SMOKE_CKPT):
        if not os.path.exists(p):
            print(f"!! missing fixture {p}")
            return 2

    # ============================================================ 1. API / naming
    print("\n[1] API, versioned naming and version discovery")
    check("the generator is the frozen rqvae/generate_code.py",
          SR.GENERATOR_REL == os.path.join("rqvae", "generate_code.py")
          and os.path.isfile(SR.GENERATOR), SR.GENERATOR_REL)
    check("SID layout constants come from the plan (4 columns, 256 codebook)",
          SR.SID_COLUMNS == 4 and SR.CODEBOOK_SIZE == 256)

    name = SR.sid_name("Toys", 3, "abcdef0123456789")
    check("a versioned name carries dataset, version and checkpoint prefix",
          SR.parse_sid_name(name) == {"dataset": "Toys", "version": 3,
                                      "ckpt_sha8": "abcdef01"}, name)
    check("the version field is zero-padded so names sort by version",
          SR.sid_name("Toys", 3, "a" * 40) < SR.sid_name("Toys", 10, "a" * 40))
    check("foreign / malformed names are rejected, not guessed",
          SR.parse_sid_name("Toys_clean_tiger_sid.npy") is None
          and SR.parse_sid_name("Toys_sid_v1_0123abcd.npy") is None)
    check("a name whose checkpoint prefix changed is a different path",
          SR.sid_path("/b", "Toys", 1, "a" * 40) != SR.sid_path("/b", "Toys", 1, "b" * 40))
    check("the sidecar name follows E7's '<artifact>.json' convention",
          SR.sidecar_path("/x/a.npy") == "/x/a.npy.sid.json")

    with tempfile.TemporaryDirectory() as td:
        check("an empty base dir has no versions -> next is v1",
              SR.list_versions(td, "Toys") == [] and SR.next_version(td, "Toys") == 1)
        os.makedirs(SR.version_dir(td), exist_ok=True)
        for v in (1, 2):
            np.save(SR.sid_path(td, "Toys", v, "deadbeef" * 5),
                    np.zeros((2, 4), dtype=int))
        np.save(SR.sid_path(td, "Beauty", 7, "deadbeef" * 5),
                np.zeros((2, 4), dtype=int))
        got = [v["version"] for v in SR.list_versions(td, "Toys")]
        check("versions are discovered per dataset and ordered",
              got == [1, 2] and SR.next_version(td, "Toys") == 3
              and SR.next_version(td, "Beauty") == 8, f"Toys={got}")

    # ====================================================== 2. real generation
    print("\n[2] REAL SID generation from the frozen checkpoint (acceptance 1/2/3/5)")
    work = tempfile.mkdtemp(prefix="exp_e8_")
    try:
        t0 = time.time()
        rec = SR.refresh_sid("Toys", SMOKE_CKPT, work, device=DEVICE,
                             timeout=GEN_TIMEOUT)
        print(f"       (real generation took {time.time() - t0:.1f}s)")
        sid = rec["sid_path"]
        codes = SR.load_sid(sid)
        st = rec["sid_stats"]

        check("(1) every item is covered",
              codes.shape[0] == rec["n_items"] == st["n_items"] == 11924,
              f"{codes.shape[0]} rows == n_items {rec['n_items']}")
        check("the SID has the frozen shape and dtype",
              codes.shape[1] == 4 and np.issubdtype(codes.dtype, np.integer),
              f"{codes.shape} {codes.dtype}")
        check("(3) every token is inside [0, 256)",
              st["token_min"] >= 0 and st["token_max"] < 256,
              f"min {st['token_min']} max {st['token_max']} "
              f"per-column max {st['per_column_max']}")
        check("    ... and that bound is not vacuous: codebook columns really reach 255",
              max(st["per_column_max"][:3]) == 255,
              f"cols 0-2 max {st['per_column_max'][:3]}, col3 max {st['per_column_max'][3]}")
        check("(2) frozen collision handling preserved: no duplicate rows survive",
              st["resolved_duplicate_rows"] == 0 and st["unique_rows"] == st["n_items"],
              f"unique {st['unique_rows']}/{st['n_items']}, raw dup rate "
              f"{st['raw_duplicate_rate']:.4f}, largest group {st['max_duplicate_group']}")

        frozen = np.load(SMOKE_SID)
        dr = SR.diff_rate(codes, frozen)
        if str(DEVICE).startswith("cuda"):
            check("the refresh reproduces the frozen pipeline's own SID bitwise",
                  dr["exact_match"], f"rows differing {dr['rows_changed_rate']:.6f}")
        else:
            check("on CPU the refresh matches the frozen (cuda) SID except in the "
                  "last-layer / collision columns",
                  dr["per_column_rate"][0] == 0.0 and dr["per_column_rate"][1] == 0.0
                  and 0.0 < dr["rows_changed_rate"] < 0.05,
                  f"rows differing {dr['rows_changed_rate']:.6f} per-column "
                  f"{[round(x, 5) for x in dr['per_column_rate']]}")

        check("(5) the manifest records the checkpoint and generator sha256",
              rec["ckpt_sha256"] == SR.sha256_file(SMOKE_CKPT)
              and rec["generator_sha256"] == SR.sha256_file(SR.GENERATOR),
              f"ckpt {rec['ckpt_sha256'][:12]} generator {rec['generator_sha256'][:12]}")
        check("(5) the manifest records the repo commit, dirty flag, time and device",
              rec.get("repo_commit") and rec.get("created_at")
              and rec["device"] == str(DEVICE) and isinstance(rec.get("repo_dirty"), bool),
              f"commit {str(rec['repo_commit'])[:8]} dirty={rec['repo_dirty']} "
              f"{rec['created_at']} device={rec['device']}")
        check("(5) the manifest records the checkpoint's own training args",
              rec["ckpt_args"]["num_emb_list"] == [256, 256, 256]
              and rec["ckpt_args"]["sk_epsilons"] == [0.0, 0.0, 0.003]
              and rec["data_path"] == rec["ckpt_args"]["data_path"],
              f"data_path={os.path.basename(rec['data_path'])}")
        check("the recorded command uses only the generator's existing CLI flags",
              os.path.basename(rec["generator_cmd"][1]) == "generate_code.py"
              and set(rec["generator_cmd"][2::2]) == {"--dataset", "--ckpt_path",
                                                      "--output_file", "--device"},
              " ".join(rec["generator_cmd"][2:]))
        check("the sidecar is on disk beside the SID and round-trips",
              os.path.isfile(SR.sidecar_path(sid))
              and SR.read_sid_manifest(sid)["sid_sha256"] == rec["sid_sha256"],
              os.path.basename(SR.sidecar_path(sid)))
        check("the SID is written into the versioned directory, not the base dir",
              os.path.dirname(sid) == SR.version_dir(work)
              and count(os.path.join(work, "*.npy")) == 0)
        check("a first refresh has no previous SID to diff against",
              rec["previous_sid_path"] is None and rec["diff"] is None)

        # ============================================== 3. acceptance (4) diff rate
        print("\n[3] acceptance (4): diff rate, and its device noise floor")
        rec2 = SR.refresh_sid("Toys", SMOKE_CKPT, work, device=DEVICE,
                              timeout=GEN_TIMEOUT)
        check("a second round auto-discovers the previous version and diffs against it",
              rec2["version"] == 2 and rec2["previous_sid_path"] == sid
              and rec2["previous_sid_sha256"] == rec["sid_sha256"],
              f"v{rec2['version']} <- v{rec['version']}")
        check("the same-device diff rate is exactly 0 (generation is deterministic)",
              rec2["diff"]["exact_match"] and rec2["diff"]["rows_changed_rate"] == 0.0,
              f"rows_changed {rec2['diff']['rows_changed']} / "
              f"{rec2['diff']['rows_total']}")

        if FAST:
            print("  [skip] cross-device round (no second device available / EXP_E8_FAST=1)")
        else:
            rec3 = SR.refresh_sid("Toys", SMOKE_CKPT, work, device=OTHER_DEVICE,
                                  timeout=GEN_TIMEOUT)
            d = rec3["diff"]
            check("a different device changes the SID: this diff rate is a device "
                  "noise floor, not a codebook change",
                  d["rows_changed_rate"] > 0.0,
                  f"{OTHER_DEVICE} vs {DEVICE}: {d['rows_changed_rate']:.6f} rows "
                  f"({d['rows_changed']}/{d['rows_total']})")
            check("that device difference is confined to the last-layer/collision "
                  "columns (the SK path decides borderline items)",
                  d["per_column_rate"][0] == 0.0 and d["per_column_rate"][1] == 0.0,
                  f"per-column {[round(x, 5) for x in d['per_column_rate']]}")
            check("the manifest records which device produced each version",
                  rec3["device"] == OTHER_DEVICE and rec["device"] == str(DEVICE),
                  f"v{rec['version']}={rec['device']} v{rec3['version']}={rec3['device']}")

        # ============================================== 4. traceability / resume
        print("\n[4] checkpoint <-> SID traceability (acceptance 5)")
        man = SR.read_sid_manifest(sid)
        check("verify_sid_manifest accepts its own checkpoint and SID",
              SR.verify_sid_manifest(man, ckpt_path=SMOKE_CKPT, sid_path_=sid) is True)
        bad_ckpt = os.path.join(work, "other.pth")
        shutil.copyfile(SMOKE_CKPT, bad_ckpt)
        with open(bad_ckpt, "ab") as f:
            f.write(b"\x00")                    # same shape, different bytes
        check("it refuses a checkpoint whose bytes differ from the registered one",
              raises(lambda: SR.verify_sid_manifest(man, ckpt_path=bad_ckpt),
                     SR.SidManifestMismatch))
        tampered = os.path.join(work, "tampered.npy")
        arr = SR.load_sid(sid)
        arr[0, 0] = (arr[0, 0] + 1) % 256
        np.save(tampered, arr)
        check("it refuses a SID whose contents changed since registration",
              raises(lambda: SR.verify_sid_manifest(man, sid_path_=tampered),
                     SR.SidManifestMismatch))
        check("a missing SID is refused rather than silently accepted",
              raises(lambda: SR.verify_sid_manifest(
                  man, sid_path_=os.path.join(work, "nope.npy")), SR.SidManifestMismatch))

        # ================================================== 5. overwrite risk
        print("\n[5] overwrite risk")
        canon = os.path.join(work, "canonical", "Toys_clean_tiger_sid.npy")
        os.makedirs(os.path.dirname(canon), exist_ok=True)
        np.save(canon, np.full((2, 4), 7, dtype=int))
        before = np.load(canon).copy()
        SR.refresh_sid("Toys", SMOKE_CKPT, work, device=DEVICE, timeout=GEN_TIMEOUT)
        check("a refresh does NOT silently rewrite the canonical path T5 reads",
              np.array_equal(np.load(canon), before),
              "publishing is opt-in (publish=...), never implicit")

        clash_dir = tempfile.mkdtemp(prefix="exp_e8_clash_")
        try:
            vd = SR.version_dir(clash_dir)
            os.makedirs(vd, exist_ok=True)
            sha = SR.sha256_file(SMOKE_CKPT)
            occupied = SR.sid_path(clash_dir, "Toys", 1, sha)
            np.save(occupied, np.full((3, 4), 3, dtype=int))     # a real v1 already sits here
            real_next = SR.next_version
            SR.next_version = lambda base, ds: 1                 # simulate the collision
            try:
                ok = raises(lambda: SR.refresh_sid("Toys", SMOKE_CKPT, clash_dir,
                                                   device=DEVICE, timeout=GEN_TIMEOUT),
                            SR.SidRefreshError)
            finally:
                SR.next_version = real_next
            check("an existing versioned SID at the target path is refused, not overwritten",
                  ok)
            check("the occupied SID is bitwise intact after the refusal",
                  np.array_equal(SR.load_sid(occupied), np.full((3, 4), 3, dtype=int)))
            check("the refused round left no sidecar and no temp file behind",
                  count(os.path.join(vd, "*.sid.json")) == 0
                  and count(os.path.join(vd, "_tmp", "*")) == 0)
        finally:
            shutil.rmtree(clash_dir, ignore_errors=True)

        expected_v = 4 if not FAST else 3
        check("each successful round consumes exactly one new version number",
              len(SR.list_versions(work, "Toys")) == expected_v,
              f"{len(SR.list_versions(work, 'Toys'))} versions on disk, expected {expected_v}")

        # ==================================================== 6. failure state
        print("\n[6] failure state")
        broken = os.path.join(work, "broken_ckpt.pth")
        ck = torch.load(SMOKE_CKPT, map_location="cpu", weights_only=False)
        sd = copy.deepcopy(ck["state_dict"])
        victim = next(k for k in sd if k.endswith("embedding.weight"))
        sd[victim] = torch.zeros(1, 2)        # wrong shape -> load_state_dict raises
        torch.save({"args": ck["args"], "state_dict": sd}, broken)
        # same checkpoint but its embedding parquet is gone -> refused before generation
        gone_args = copy.deepcopy(ck["args"])
        gone_args.data_path = os.path.join(work, "nope.parquet")
        gone_ckpt = os.path.join(work, "gone_data.pth")
        torch.save({"args": gone_args, "state_dict": ck["state_dict"]}, gone_ckpt)
        fail_dir = tempfile.mkdtemp(prefix="exp_e8_fail_")
        try:
            check("a generator failure is raised as an E8 error, not swallowed",
                  raises(lambda: SR.refresh_sid("Toys", broken, fail_dir, device=DEVICE,
                                                timeout=GEN_TIMEOUT), SR.SidRefreshError))
            check("a failed round leaves no SID behind",
                  SR.list_versions(fail_dir, "Toys") == [])
            check("a failed round leaves no sidecar behind",
                  count(os.path.join(SR.version_dir(fail_dir), "**", "*.sid.json")) == 0)
            check("a failed round leaves no temp file behind",
                  count(os.path.join(SR.version_dir(fail_dir), "_tmp", "*")) == 0)
            check("a failed round does not clobber a good SID from an earlier round",
                  np.array_equal(SR.load_sid(rec["sid_path"]), codes))
        finally:
            shutil.rmtree(fail_dir, ignore_errors=True)
        check("a missing checkpoint is refused before anything is written",
              raises(lambda: SR.refresh_sid("Toys", os.path.join(work, "gone.pth"),
                                            work, device=DEVICE), SR.SidRefreshError))
        check("a checkpoint whose embedding parquet is gone is refused, not skipped",
              raises(lambda: SR.n_items_of(gone_ckpt), SR.SidRefreshError))

        good = SR.load_sid(sid)
        check("validation rejects a coverage shortfall (1)",
              not SR.validate_sid(good[:-1], n_items=good.shape[0], strict=False)["ok"])
        bad_range = good.copy()
        bad_range[0, 0] = 256
        check("validation rejects an out-of-range token (3)",
              not SR.validate_sid(bad_range, n_items=len(good), strict=False)["ok"])
        dup = good.copy()
        dup[1] = dup[0]
        dup_stats = SR.validate_sid(dup, n_items=len(good), strict=False)
        check("validation rejects surviving duplicates (2)",
              not dup_stats["ok"] and dup_stats["resolved_duplicate_rows"] > 0)
        check("validation rejects a non-integer array and a mis-shaped array",
              not SR.validate_sid(good.astype(float), strict=False)["ok"]
              and not SR.validate_sid(good[:, :3], strict=False)["ok"])

        # ========================================================= 7. publish
        print("\n[7] publish (atomic hand-off to the canonical path)")
        SR.publish_sid(sid, canon)
        check("publish points the canonical SID at the new version",
              np.array_equal(np.load(canon), codes))
        check("publishing is atomic: no staging file is left behind",
              count(canon + ".tmp-*") == 0)
        keep = np.load(canon).copy()
        check("publishing a missing SID raises instead of truncating the target",
              raises(lambda: SR.publish_sid(os.path.join(work, "gone.npy"), canon),
                     SR.SidRefreshError) and np.array_equal(np.load(canon), keep))
        check("publishing into a non-existent directory raises, target untouched",
              raises(lambda: SR.publish_sid(sid, os.path.join(work, "nodir", "x.npy")),
                     SR.SidRefreshError) and np.array_equal(np.load(canon), keep))

    finally:
        shutil.rmtree(work, ignore_errors=True)

    # ============================================================== 8. scope
    print("\n[8] scope: no new encoding mechanism, no coupling, frozen dirs untouched")
    src = open(SR.__file__).read()
    check("the module never writes codes itself: no np.save and no RQ-VAE API in it",
          "np.save" not in src
          and not any(tok in src for tok in ("RQVAE", "get_indices", "vq_layers",
                                             "VectorQuantizer")),
          "codes are produced only by rqvae/generate_code.py")
    grep = subprocess.run(
        ["grep", "-rn", "--include=*.py", "-E",
         "^[[:space:]]*(from|import)[[:space:]]+tiger_ext",
         os.path.join(REPO, "model"), os.path.join(REPO, "rqvae")],
        capture_output=True, text=True)
    check("nothing under model/ or rqvae/ imports tiger_ext (no runtime coupling)",
          grep.returncode == 1 and not grep.stdout.strip(), grep.stdout.strip()[:80])
    status = subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                            capture_output=True, text=True).stdout.splitlines()
    check("clean_tiger/ is untouched (E8 writes nothing into it)",
          not [l for l in status if "clean_tiger" in l])
    modified = {l[3:].strip() for l in status if l[:2].strip() == "M"}
    check("model/ and rqvae/ carry no E8 change (only the E1-E6 files are modified)",
          modified <= PRE_E8_MODIFIED, f"modified: {sorted(modified)}")

    # ============================================================ summary
    n_ok, n = sum(RESULTS), len(RESULTS)
    print("\n" + "=" * 70)
    print(f"EXP-E8: {n_ok}/{n} checks passed  (device={DEVICE})")
    print("=" * 70)
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
