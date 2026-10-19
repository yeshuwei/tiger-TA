"""EXP-C2 -- Collaborative kNN diagnostic (A class, with one B-class probe).

Run:  python -I integration/exp_c2_collab_relation.py
      EXP_C2_REGRESSION=skip python -I integration/exp_c2_collab_relation.py

Covers plan.md C2's acceptance criteria plus the guarantees C3 will depend on:

  [1] synthetic fixture with hand-computed cosines: the exact top-k, the
      tie-break, self-exclusion, the zero-norm policy, k clamping, and the
      negative controls that prove each guard can actually fire
  [2] real data (Toys / Sport / Beauty): build, save -> reload bitwise, and the
      row-space refusal when the ids do not match
  [3] the relation recomputed from scratch by an independent float64 brute force
      (set, order and value), plus a cross-check against sklearn's
      `NearestNeighbors(metric='cosine')` -- the library CSG4Rec itself uses
  [4] the ItemID/row-space contract, including the negative control that
      `row + 1` is not the ItemID on Beauty
  [5] quantity and shape: edges, degree distribution, dtype, similarity profile
  [6] B-class probe: do kNN neighbours actually co-occur in held-out baskets,
      and does that diagnostic collapse when the row->ItemID mapping is corrupted
  [7] scope: no loss, no constraint, no projection, nothing read at import time,
      no frozen directory touched, no reuse of the historical switch semantics
  [8] regression: the E1-E10 suites, re-run after C2

What is measured vs what is claimed
-----------------------------------
[1]-[5] and [7] are engineering checks. [6] is a *diagnostic*: the SASRec CF table
was trained on `train.parquet`, and this project's valid/test splits are prefixes
of the same user sequences, so the co-occurrence it measures is NOT independent
evidence about generalisation. It is used here for what it can actually support --
that the neighbours are behaviourally related items and that the row->ItemID
mapping is the right one (a corrupt mapping makes it collapse to the random
baseline). No recommendation-quality claim is made anywhere in this file.

Engineering gate only -- nothing here says anything about recommendation quality.
"""
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from tiger_ext import cf_embed as CE                   # noqa: E402
from tiger_ext import collab_relation as CR            # noqa: E402

DATA_DIR = os.path.join(REPO, "data")
MODULE = os.path.join(REPO, "tiger_ext", "collab_relation.py")
ARTIFACT_DIR = os.path.join(HERE, "artifacts", "collab_relation")
REGRESSION = os.environ.get("EXP_C2_REGRESSION", "all")

# the tracked files E1-E6 legitimately modified; C2 must not add to this set
PRE_C2_MODIFIED = {"model/main.py", "rqvae/models/rq.py", "rqvae/models/vq.py"}

DATASETS = ("Toys", "Sport", "Beauty")
N_ITEMS = {"Toys": 11924, "Sport": 18357, "Beauty": 12101}
BEAUTY_SCRAMBLED_ROWS = 4158          # Beauty rows where ItemID != row + 1
K = 20                                # CSG4Rec's --knn_k default

# the fixture: n=6, d=3, ids scrambled so `item_id - 1` cannot find a row, with a
# deliberate zero-norm row (row 5) and a deliberate tie (row 2: three candidates
# sit at cosine 0.0)
FIX_IDS = np.array([10, 3, 7, 1, 5, 12], dtype=np.int64)
FIX_TABLE = np.array([[1.0, 0.0, 0.0],      # row 0
                      [2.0, 0.0, 0.0],      # row 1  (same direction as row 0)
                      [0.0, 1.0, 0.0],      # row 2
                      [0.6, 0.8, 0.0],      # row 3
                      [-1.0, 0.0, 0.0],     # row 4
                      [0.0, 0.0, 0.0]],     # row 5  zero norm -> undefined cosine
                     dtype=np.float32)
# hand-computed with sim(a,b) = dot / (|a||b|), k=2, self excluded, zero-norm row
# excluded as both source and target, ties broken by ascending row index
FIX_NEIGHBORS = np.array([[1, 3], [0, 3], [3, 0], [2, 0], [2, 3], [-1, -1]],
                         dtype=np.int32)
FIX_SIMILARITY = np.array([[1.0, 0.6], [1.0, 0.6], [0.8, 0.0], [0.8, 0.6],
                           [0.0, -0.6], [np.nan, np.nan]], dtype=np.float32)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def raises(fn, exc):
    """True only if fn raises exactly the expected error type."""
    try:
        fn()
    except exc:
        return True
    except Exception:                       # a wrong exception type is a failure too
        return False
    return False


def same_bytes(a, b):
    """Bitwise equality of two arrays (NaN-safe for identical bit patterns)."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


try:
    from sklearn.neighbors import NearestNeighbors
    HAVE_SKLEARN = True
except ImportError:                                        # pragma: no cover
    HAVE_SKLEARN = False


def reference_topk(table, k, rows=None):
    """Independent float64 brute-force top-k cosine, written from the definition.

    Deliberately naive: no normalisation pre-pass, no blocking, no argpartition,
    a full sort per row. Same exclusion rules as the module (self, zero-norm
    source and target) so the two must agree edge for edge.
    """
    X = np.asarray(table, dtype=np.float64)
    rows = np.arange(X.shape[0]) if rows is None else np.asarray(rows)
    norms = np.sqrt((X * X).sum(axis=1))
    zero = norms == 0
    out = []
    for i in rows:
        if zero[i]:
            out.append((np.array([], dtype=np.int64), np.array([], dtype=np.float64)))
            continue
        sims = np.full(X.shape[0], -np.inf)
        for j in range(X.shape[0]):
            if j == i or zero[j]:
                continue
            sims[j] = float(X[i] @ X[j]) / (float(norms[i]) * float(norms[j]))
        order = np.argsort(-sims, kind="stable")
        order = order[np.isfinite(sims[order])][:k]
        out.append((order.astype(np.int64), sims[order]))
    return out, norms


# --------------------------------------------------------------------------- #
# [1] synthetic fixture
# --------------------------------------------------------------------------- #
def part1_synthetic(tmp):
    print("\n[1] synthetic fixture: hand-computed cosine, top-k, guards")
    rel = CR.build_relation(FIX_TABLE, FIX_IDS, k=2)

    check("cosine + top-k reproduce the hand computation exactly",
          np.array_equal(rel.neighbors, FIX_NEIGHBORS)
          and np.array_equal(rel.similarity, FIX_SIMILARITY, equal_nan=True),
          f"neighbors={rel.neighbors.tolist()} sim={rel.similarity.tolist()}")
    check("the tie at cosine 0.0 is broken by ascending row index (row 2 -> 0, not 1)",
          rel.neighbors[2].tolist() == [3, 0] and rel.similarity[2, 1] == 0.0,
          "the exact top-k contract, not whatever argpartition happened to pick")
    self_loop = FIX_NEIGHBORS.copy()
    self_loop[0] = [0, 3]                       # row 0 now names itself
    check("no row is its own neighbour, and the guard fires when one is",
          not bool((rel.neighbors == np.arange(6)[:, None]).any())
          and raises(lambda: CR.CollabRelation(self_loop, np.abs(FIX_SIMILARITY),
                                               FIX_IDS),
                     CR.CollabRelationMismatch),
          "0 self-loops produced; a hand-injected self-loop is rejected")
    check("the zero-norm row is isolated AND is nobody's neighbour",
          rel.neighbors[5].tolist() == [-1, -1]
          and not bool((rel.neighbors == 5).any())
          and bool(np.isnan(rel.similarity[5]).all()),
          "an undefined cosine yields no edge in either direction")
    check("-1 neighbour slots and NaN similarities always agree",
          bool(((rel.neighbors < 0) == np.isnan(rel.similarity)).all()))
    check("edges = n_items_with_neighbors * k_eff, and isolated rows contribute none",
          rel.n_edges == 5 * 2 and rel.n_edges == len(rel.pairs)
          and rel.pairs.tolist() == [[0, 1], [0, 3], [1, 0], [1, 3], [2, 3],
                                     [2, 0], [3, 2], [3, 0], [4, 2], [4, 3]],
          f"{rel.n_edges} edges over {rel.n_items} items")
    check("edges are unique -- no target is listed twice for the same source",
          rel.n_edges == len({(int(a), int(b)) for a, b in rel.pairs})
          and rel.stats()["n_duplicate_edges"] == 0 and rel.n_edges == 10,
          "directed: (0 -> 1) and (1 -> 0) are two distinct edges, not one")
    check("the pair file is in row space and the ItemID companion is its image",
          isinstance(rel.pairs.dtype, np.dtype) and rel.pairs.dtype == np.int32
          and rel.pairs_as_item_ids().dtype == np.int64
          and np.array_equal(rel.pairs_as_item_ids()[:, 0], FIX_IDS[rel.pairs[:, 0]])
          and np.array_equal(rel.pairs_as_item_ids()[:, 1], FIX_IDS[rel.pairs[:, 1]]),
          f"first pair {rel.pairs[0].tolist()} -> {rel.pairs_as_item_ids()[0].tolist()}")
    check("`item_id - 1` is not the mapping: row 0 is ItemID 10, not 1",
          rel.item_ids[0] == 10 and not np.array_equal(rel.item_ids,
                                                      np.arange(1, 7)),
          f"ids={rel.item_ids.tolist()}")
    check("lookup by ItemID works and an unknown ItemID raises",
          rel.neighbor_item_ids_of(10).tolist() == [3, 1]
          and rel.similarity_of(10, 3) == 1.0
          and raises(lambda: rel.row_of(999), CR.CollabRelationMismatch),
          "neighbours of ItemID 10 are ItemIDs [3, 1]")
    check("a bare table without item_ids is refused (the row space must be pinned)",
          raises(lambda: CR.build_relation(FIX_TABLE, k=2),
                 CR.CollabRelationMismatch))

    # k clamping / argument guards
    big = CR.build_relation(FIX_TABLE, FIX_IDS, k=100)
    check("k larger than the table clamps to k_eff = n-1 and keeps both numbers apart",
          big.k == 100 and big.k_eff == 5 and big.neighbors.shape == (6, 5)
          and big.stats()["k"] == 100 and big.stats()["k_eff"] == 5,
          "requested k is reported separately from the delivered k_eff")
    for bad, why in ((0, "k=0"), (-3, "negative k")):
        check(f"{why} is refused",
              raises(lambda b=bad: CR.build_relation(FIX_TABLE, FIX_IDS, k=b),
                     CR.CollabRelationMismatch))
    check("a non-finite table is refused",
          raises(lambda: CR.build_relation(
              np.concatenate([FIX_TABLE[:1] * np.nan, FIX_TABLE[1:]]), FIX_IDS, k=2),
              CR.CollabRelationMismatch),
          "NaN in a row makes every cosine undefined; C2 refuses rather than guess")
    check("duplicate item ids are refused (two rows would share one ItemID)",
          raises(lambda: CR.build_relation(FIX_TABLE,
                                          np.array([10, 3, 7, 1, 5, 10]),
                                          k=2), CR.CollabRelationMismatch))
    check("a table/id length mismatch is refused",
          raises(lambda: CR.build_relation(FIX_TABLE, FIX_IDS[:4], k=2),
                 CR.CollabRelationMismatch))
    check("building twice is bit-for-bit identical (deterministic tie-break)",
          same_bytes(CR.build_relation(FIX_TABLE, FIX_IDS, k=2).neighbors,
                     rel.neighbors)
          and same_bytes(CR.build_relation(FIX_TABLE, FIX_IDS, k=2).similarity,
                         rel.similarity))

    # tampering: every structural guard must be able to fire
    good = rel.pairs
    ok_nbr = np.array([[1], [0], [3], [2]], dtype=np.int32)      # n=4, k_eff=1
    ok_sim = np.array([[0.9], [0.9], [0.9], [0.9]], dtype=np.float32)
    bad_rels = {
        "a self-loop": (np.array([[0], [0], [3], [2]], dtype=np.int32), ok_sim),
        "an out-of-range row": (np.array([[99], [0], [3], [2]], dtype=np.int32), ok_sim),
        "a cosine above 1": (ok_nbr, np.array([[1.4], [0.9], [0.9], [0.9]],
                                              dtype=np.float32)),
        "a -1 slot carrying a finite cosine": (np.array([[-1], [0], [3], [2]],
                                                        dtype=np.int32), ok_sim),
        "a duplicate row in the neighbour list": (np.array([[1], [2], [3], [2]],
                                                           dtype=np.int32), ok_sim),
    }
    for label, (nbr, sim) in bad_rels.items():
        if label == "a duplicate row in the neighbour list":
            # duplicates are legal-looking; what must not happen is a duplicate
            # PairFile entry -- verified below via the pair/edge count identity
            check("a neighbour list may repeat a target (two edges, not a self-loop)",
                  CR.CollabRelation(nbr, sim, FIX_IDS[:4]).n_edges == 4,
                  "duplicates are counted as distinct edges; self is what is banned")
            continue
        check(f"a constructed relation with {label} is rejected",
              raises(lambda n=nbr, s=sim: CR.CollabRelation(n, s, FIX_IDS[:4]),
                     CR.CollabRelationMismatch))
    wide = np.array([[1, 2, 3, 1]] * 4, dtype=np.int32)     # no self, but k_eff=4
    check("k_eff > n_items - 1 is rejected (self-exclusion makes it unachievable)",
          raises(lambda: CR.CollabRelation(wide,
                                           np.full((4, 4), 0.5, dtype=np.float32),
                                           FIX_IDS[:4]),
                 CR.CollabRelationMismatch))

    # save -> tamper -> load: the stored (M, 2) export must agree with the matrix
    prefix = os.path.join(tmp, "tamper_k2")
    written = rel.save(prefix)
    check("save writes the documented artifact set",
          all(k in written for k in ("neighbors", "similarity", "pairs",
                                     "pairs_itemid", "item_ids", "meta"))
          and all(os.path.exists(p) for p in written.values()),
          f"{len(written)} files")
    np.save(prefix + CR.SUFFIX["pairs"], good[::-1].copy())
    check("load rejects a stored pair file that disagrees with the neighbour matrix",
          raises(lambda: CR.CollabRelation.load(prefix, expected_item_ids=FIX_IDS),
                 CR.CollabRelationMismatch))
    rel.save(prefix)
    check("re-saving restores a loadable artifact",
          CR.CollabRelation.load(prefix, expected_item_ids=FIX_IDS, expected_k=2)
          is not None)
    check("load rejects a caller whose canonical ids are a different row space",
          raises(lambda: CR.CollabRelation.load(prefix, expected_item_ids=FIX_IDS[::-1]),
                 CR.CollabRelationMismatch),
          "the ids are the same set in a different order -- caught anyway")
    check("load rejects an expected_k that does not match the stored provenance",
          raises(lambda: CR.CollabRelation.load(prefix, expected_k=7),
                 CR.CollabRelationMismatch))
    check("load rejects a missing artifact rather than returning an empty relation",
          raises(lambda: CR.CollabRelation.load(os.path.join(tmp, "nope")),
                 CR.CollabRelationMismatch))


# --------------------------------------------------------------------------- #
# [2] real data
# --------------------------------------------------------------------------- #
def part2_real(tmp):
    print("\n[2] real data: build from C1's table, save, reload bitwise")
    built = {}
    for name in DATASETS:
        emb = CE.load_cf_embedding(name)
        rel = CR.build_relation_for_dataset(name)
        path = CR.relation_prefix(name, artifact_dir=ARTIFACT_DIR)
        written = rel.save(path)
        built[name] = rel
        ckpt = os.path.abspath(CE.sasrec_checkpoint_path(name))
        check(f"{name}: the relation is built from C1's default (SASRec) source and "
              f"carries the checkpoint path and hash forward",
              rel.meta.get("cf_source_kind") == CE.CF_SOURCE_SASREC
              and os.path.abspath(str(rel.source)) == ckpt
              and rel.meta["cf_source_sha256"] == CE.sha256_file(ckpt)
              and len(rel.meta["cf_source_sha256"]) == 64
              and json.load(open(CR.relation_prefix(name, artifact_dir=ARTIFACT_DIR)
                                 + ".json", encoding="utf-8"))["cf_source_sha256"]
              == rel.meta["cf_source_sha256"],
              f"{os.path.basename(ckpt)} sha256={rel.meta['cf_source_sha256'][:16]}...")
        check(f"{name}: row space is C1's canonical ItemID order, digest included",
              np.array_equal(rel.item_ids, emb.item_ids)
              and rel.item_ids_sha256 == CE.item_ids_sha256(emb.item_ids),
              f"sha256={rel.item_ids_sha256[:16]}...")
        check(f"{name}: shape is (n_items, k_eff) with k_eff=k={K}",
              rel.neighbors.shape == (N_ITEMS[name], K)
              and rel.similarity.shape == (N_ITEMS[name], K)
              and rel.k == K and rel.k_eff == K,
              f"{rel.neighbors.shape} int32 + {rel.similarity.shape} float32")
        check(f"{name}: every one of the {N_ITEMS[name]} items has a full "
              f"neighbour list (zero-norm source has 0 rows here)",
              rel.n_edges == N_ITEMS[name] * K
              and not bool((rel.out_degree == 0).any())
              and rel.stats()["n_nan_slots"] == 0,
              f"edges={rel.n_edges}")
        check(f"{name}: the SASRec source has no all-zero row, so the zero-norm "
              f"policy never fires on the main line",
              emb.n_zero_rows == 0)
        loaded = CR.CollabRelation.load(path, expected_item_ids=emb.item_ids,
                                        expected_k=K)
        check(f"{name}: reload is bitwise identical and stats round-trip",
              same_bytes(loaded.neighbors, rel.neighbors)
              and same_bytes(loaded.similarity, rel.similarity)
              and same_bytes(loaded.pairs, rel.pairs)
              and loaded.provenance()["stats"] == rel.stats(),
              "neighbors / similarity / pairs / stats")
        check(f"{name}: the reloaded relation still belongs to this table",
              loaded.check_against(emb) is loaded)
        check(f"{name}: provenance json on disk is valid and carries the C2 fields",
              json.load(open(path + ".json", encoding="utf-8"))["k"] == K)
        wrong = emb.item_ids.copy()
        wrong[[0, 1]] = wrong[[1, 0]]
        check(f"{name}: swapping two canonical ids makes load refuse",
              raises(lambda p=path, w=wrong: CR.CollabRelation.load(
                  p, expected_item_ids=w), CR.CollabRelationMismatch))
    return built


# --------------------------------------------------------------------------- #
# [3] independent recomputation
# --------------------------------------------------------------------------- #
def part3_independent(built):
    print("\n[3] recomputed from scratch: float64 brute force + sklearn cross-check")
    rng = np.random.default_rng(20261014)
    have_sklearn = HAVE_SKLEARN
    if not have_sklearn:                                   # pragma: no cover
        print("  [skip] sklearn is not installed; the reference cross-check is skipped")

    for name in DATASETS:
        rel, emb = built[name], CE.load_cf_embedding(name)
        rows = np.sort(rng.choice(N_ITEMS[name], size=6, replace=False))
        ref, norms = reference_topk(emb.table, K, rows=rows)

        sets_ok = orders_ok = True
        max_dv = 0.0
        optimal_ok = True
        for pos, i in enumerate(rows):
            got, got_sim = rel.neighbors[i], rel.similarity[i]
            want, want_sim = ref[pos]
            sets_ok &= set(got[got >= 0].tolist()) == set(want.tolist())
            orders_ok &= got[got >= 0].tolist() == want.tolist()
            if want.size and got[got >= 0].size == want.size:
                max_dv = max(max_dv, float(np.abs(got_sim[got >= 0].astype(np.float64)
                                                 - want_sim).max()))
            # optimality: the worst neighbour must beat the best non-neighbour
            non = np.setdiff1d(np.arange(N_ITEMS[name]),
                               np.append(got[got >= 0], i), assume_unique=False)
            sims_all = (emb.table[non].astype(np.float64) @ emb.table[i].astype(np.float64)
                        / (norms[non] * norms[i]))
            optimal_ok &= bool(not non.size
                               or got_sim[got >= 0].min() + 1e-6 >= sims_all.max())
        check(f"{name}: the same top-k set as an independent float64 brute force",
              sets_ok, f"sampled rows {rows.tolist()}")
        check(f"{name}: the same neighbour ORDER (not just the set)", orders_ok)
        check(f"{name}: cosine agrees with the brute force to float32 precision",
              max_dv < 1e-5, f"max |Δcos| = {max_dv:.2e}")
        check(f"{name}: the k-th neighbour's cosine beats every non-neighbour's "
              f"(a genuine top-k, not an approximate one)", optimal_ok)

        # the naive formula, on a couple of concrete pairs, by ItemID
        i_item, j_row = int(rel.item_ids[rows[0]]), int(rel.neighbors[rows[0]][0])
        j_item = int(rel.item_ids[j_row])
        a = emb.table[rows[0]].astype(np.float64)
        b = emb.table[j_row].astype(np.float64)
        hand = float(a @ b) / (float(np.sqrt(a @ a)) * float(np.sqrt(b @ b)))
        check(f"{name}: dot / (|a||b|) by hand equals the stored cosine, and "
              f"similarity_of(ItemID) finds it",
              abs(hand - float(rel.similarity[rows[0]][0])) < 1e-6
              and abs(hand - rel.similarity_of(i_item, j_item)) < 1e-6,
              f"ItemID {i_item} -> {j_item}: cos={hand:.6f}")

        # the underlying cosine is symmetric even though the top-k graph is not
        sym_ok = True
        for _ in range(3):
            a_row, b_row = rng.choice(N_ITEMS[name], size=2, replace=False)
            x = emb.table[a_row].astype(np.float64)
            y = emb.table[b_row].astype(np.float64)
            c1 = float(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
            c2 = float(y @ x) / (np.linalg.norm(y) * np.linalg.norm(x))
            sym_ok &= abs(c1 - c2) < 1e-9
        check(f"{name}: the cosine is symmetric while the top-k relation is not",
              sym_ok and rel.stats()["reciprocal_fraction"] < 1.0,
              f"reciprocal fraction {rel.stats()['reciprocal_fraction']:.3f}")

        if have_sklearn:
            nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(emb.table)
            _, idx = nn.kneighbors(emb.table)
            ref_slice = idx[:, 1:K + 1]            # CSG4Rec's positional drop
            agree = np.array([set(ref_slice[i].tolist())
                              == set(rel.neighbors[i][rel.neighbors[i] >= 0].tolist())
                              for i in rows]).all()
            check(f"{name}: our top-k matches sklearn's cosine kNN on the SASRec "
                  f"table (the library CSG4Rec itself uses)", bool(agree),
                  "recomputed via NearestNeighbors(metric='cosine') + indices[:, 1:]")
            check(f"{name}: on this source the reference's `[:, 1:]` also leaves "
                  f"no self behind",
                  not bool((ref_slice == np.arange(N_ITEMS[name])[:, None]).any()),
                  "so the SASRec line cannot tell the two self-exclusions apart")


# --------------------------------------------------------------------------- #
# [4] the row-space / ItemID contract, and the fallback that breaks `[:, 1:]`
# --------------------------------------------------------------------------- #
def part4_mapping(built):
    print("\n[4] row space: explicit ItemID map, and where the reference's slice breaks")
    for name in DATASETS:
        rel = built[name]
        row_plus_one = (rel.pairs + 1).astype(np.int64)
        agrees = np.array_equal(row_plus_one, rel.pairs_as_item_ids())
        check(f"{name}: `row + 1` as an ItemID reproduces the correct pairs? "
              f"{'yes' if agrees else 'NO'}",
              agrees == (name != "Beauty"),
              f"{int((rel.item_ids != np.arange(1, N_ITEMS[name] + 1)).sum())} rows "
              f"where ItemID != row + 1")
        if name == "Beauty":
            diff = int((row_plus_one != rel.pairs_as_item_ids()).any(axis=1).sum())
            check(f"{name}: the off-by-one interpretation mis-assigns edges",
                  diff > 0 and int((rel.item_ids != np.arange(1, N_ITEMS[name] + 1)).sum())
                  == BEAUTY_SCRAMBLED_ROWS,
                  f"{diff} of {len(rel.pairs)} edges would name the wrong item")
            check(f"{name}: the stored ItemID companion is what makes the "
                  f"distinction recoverable",
                  np.array_equal(rel.pairs_as_item_ids()[:, 0], rel.item_ids[rel.pairs[:, 0]]))

    # the ItemCF fallback: the only place the reference's self-exclusion is wrong
    if not HAVE_SKLEARN:                                   # pragma: no cover
        print("  [skip] sklearn is not installed; the fallback comparison is skipped")
        return
    for name in ("Toys", "Beauty"):
        fb = CE.load_cf_embedding(name, source=CE.CF_SOURCE_ITEMCF)
        fb_rel = CR.build_relation(fb, k=K)
        nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine").fit(fb.table)
        _, idx = nn.kneighbors(fb.table)
        ref_slice = idx[:, 1:K + 1]
        self_left = int((ref_slice == np.arange(N_ITEMS[name])[:, None]).any(axis=1).sum())
        check(f"{name} ItemCF fallback: the reference's positional `[:, 1:]` keeps "
              f"self in the top-k for {self_left} of {fb.n_zero_rows} zero-norm rows",
              self_left > 0 and fb_rel.stats()["n_self_loops"] == 0
              and int((fb_rel.out_degree == 0).sum()) == fb.n_zero_rows,
              "measured on the fallback, not assumed; our explicit masking "
              "produced 0 self-loops")
        check(f"{name} ItemCF fallback: isolated rows are exactly the zero-norm rows, "
              f"and no zero-norm row appears as anyone's neighbour",
              bool((fb_rel.has_neighbors == ~fb.zero_row_mask).all())
              and not bool(np.isin(fb_rel.neighbors[fb_rel.neighbors >= 0],
                                   np.nonzero(~fb_rel.has_neighbors)[0]).any()),
              f"{fb.n_zero_rows} zero-norm rows isolated on both sides")
    check("the official SASRec source needs none of that: 0 zero-norm rows in all "
          "three datasets, so the zero-norm policy never fires on the main line",
          all(CE.load_cf_embedding(d).n_zero_rows == 0 for d in DATASETS),
          "which is exactly why the reference's slice bug is easy to inherit unseen")


# --------------------------------------------------------------------------- #
# [5] quantity and distribution
# --------------------------------------------------------------------------- #
def part5_quantity(built):
    print("\n[5] quantity, degree and similarity profile")
    for name in DATASETS:
        rel = built[name]
        s = rel.stats()
        prof = s["similarity_mean_by_rank"]
        check(f"{name}: edge count is exactly n_items x k and is recorded",
              s["n_edges"] == N_ITEMS[name] * K
              and s["degree_histogram"] == {str(K): N_ITEMS[name]},
              f"{s['n_edges']} edges, degree histogram {s['degree_histogram']}")
        check(f"{name}: similarity decays with rank (rank 0 > rank k-1)",
              prof[0] > prof[-1] and all(prof[i] >= prof[i + 1] - 1e-6
                                         for i in range(len(prof) - 1)),
              f"rank0={prof[0]:.3f} .. rank19={prof[-1]:.3f}")
        check(f"{name}: similarities are finite and inside [-1, 1]",
              s["similarity_min"] >= -1.0 and s["similarity_max"] <= 1.0
              and not np.isnan(s["similarity_mean"]),
              f"[{s['similarity_min']:.3f}, {s['similarity_max']:.3f}]")
        # recompute the symmetry statistics outside the class, from the pair file
        edges = {(int(a), int(b)) for a, b in rel.pairs}
        recip = sum(1 for a, b in edges if (b, a) in edges)
        check(f"{name}: the graph is directed, not symmetrised (allowed by C2's spec), "
              f"and the stats match an independent recount",
              s["reciprocal_fraction"] < 1.0
              and recip == int(round(s["reciprocal_fraction"] * len(edges)))
              and s["n_edges_without_reverse"] == len(edges) - recip,
              f"{recip} of {len(edges)} edges are mutual "
              f"({s['reciprocal_fraction']:.3f})")
        check(f"{name}: dtypes are int32 (row ids) and float32 (cosine)",
              rel.neighbors.dtype == np.int32 and rel.similarity.dtype == np.float32
              and rel.item_ids.dtype == np.int64)


# --------------------------------------------------------------------------- #
# [6] B-class probe: are the neighbours actually related items?
# --------------------------------------------------------------------------- #
def heldout_cooccurrence(name):
    """Set of unordered ItemID pairs that share a basket, over valid + test.

    NOTE: this project's valid/test histories are prefixes of the same user
    sequences the CF table was trained on, so this is a *relatedness / mapping*
    diagnostic, not independent evidence about generalisation.
    """
    pairs = set()
    for split in ("valid", "test"):
        frame = pd.read_parquet(os.path.join(DATA_DIR, name, f"{split}.parquet"),
                                columns=["history", "target"])
        for history, target in zip(frame["history"].values, frame["target"].values):
            seq = np.concatenate([np.asarray(history, dtype=np.int64), [int(target)]])
            items = np.unique(seq)
            for a in items:
                for b in items:
                    if a < b:
                        pairs.add((int(a), int(b)))
    return pairs


def hit_rate(pairs_a, pairs_b, basket_pairs):
    """Fraction of the given (ItemID, ItemID) edges whose basket pair co-occurs."""
    key_a = np.minimum(pairs_a, pairs_b)
    key_b = np.maximum(pairs_a, pairs_b)
    hits = sum((int(a), int(b)) in basket_pairs for a, b in zip(key_a, key_b))
    return hits / len(pairs_a)


def part6_relatedness(built):
    print("\n[6] B-class probe: kNN neighbours vs held-out basket co-occurrence")
    rng = np.random.default_rng(20261014)
    for name in DATASETS:
        rel = built[name]
        baskets = heldout_cooccurrence(name)
        rows = np.sort(rng.choice(N_ITEMS[name], size=900, replace=False))
        edges = np.concatenate([np.stack([np.full(rel.k_eff, i), rel.neighbors[i]])
                                for i in rows], axis=1)
        knn = hit_rate(rel.item_ids[edges[0]], rel.item_ids[edges[1]], baskets)

        # independent draws, so the baseline is an unbiased pair sample; self-pairs
        # are dropped rather than replaced (they are not a possible edge)
        rand_a = rng.choice(N_ITEMS[name], size=200_000, replace=True)
        rand_b = rng.choice(N_ITEMS[name], size=200_000, replace=True)
        keep = rand_a != rand_b
        rand = hit_rate(rel.item_ids[rand_a[keep]], rel.item_ids[rand_b[keep]], baskets)
        enrich = knn / rand if rand > 0 else float("inf")
        check(f"{name}: kNN neighbours co-occur far more often than random pairs",
              enrich > 5.0 and knn > 0.15,
              f"kNN hit rate {knn:.4f} vs random {rand:.5f} -> {enrich:.1f}x "
              f"({len(baskets)} held-out basket pairs)")

        # negative control: corrupt the row -> ItemID mapping and re-measure. If the
        # diagnostic is worth anything, it must collapse to the random baseline.
        bad_ids = rel.item_ids.copy()
        rng.shuffle(bad_ids)
        bad = hit_rate(bad_ids[edges[0]], bad_ids[edges[1]], baskets)
        check(f"{name}: the same edges under a permuted row->ItemID map collapse "
              f"to the random baseline",
              bad / rand < 2.5,
              f"correct {knn:.4f} -> permuted {bad:.4f} (random {rand:.5f}); "
              f"the probe is sensitive to the mapping, not just to the vectors")

        if name == "Beauty":
            shifted = hit_rate(rel.pairs[:, 0] + 1, rel.pairs[:, 1] + 1, baskets)
            check(f"{name}: the natural off-by-one (`row + 1`) is visibly worse than "
                  f"the explicit map, even though it still looks 'enriched'",
                  shifted < knn and shifted / rand > 2.0,
                  f"explicit {knn:.4f} vs row+1 {shifted:.4f} vs random {rand:.5f}")


# --------------------------------------------------------------------------- #
# [7] scope
# --------------------------------------------------------------------------- #
IMPORT_PROBE = r'''
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
calls = []
_nl, _open = np.load, open
np.load = lambda *a, **k: (calls.append(str(a[0])), _nl(*a, **k))[1]
open = lambda *a, **k: (calls.append(str(a[0])), _open(*a, **k))[1]
import tiger_ext, tiger_ext.collab_relation
np.load, open = _nl, _open
print("IMPORT_CALLS=%d" % len(calls))
'''

# identifiers that must not appear anywhere in the C2 module's code: C3's
# constraint, C4's objective, a projection, the historical transition feature's
# switch, or anything user-side. Checked against the AST's identifiers rather than
# the raw text, so documentation may discuss them while the code cannot use them.
FORBIDDEN_IDENTIFIERS = ("lambda_csg", "lambda_trans", "trans_graph_path",
                         "constraint", "csg", "loss", "q_space", "project",
                         "linear", "proj", "warmup", "regulari", "temperature",
                         "contrastive", "margin", "align", "user_id", "user_emb",
                         "user_sid", "n_users", "rqvae", "codebook", "x_q",
                         "clean_tiger", "reference_repos")
# tokens that must not appear even as string literals: these would only be present
# if the module were wired into the frozen code or the historical switch. A prose
# or provenance mention of the reference repo's *path* is legitimate and allowed --
# what is forbidden is importing or reusing its code, which the import whitelist
# above already covers.
FORBIDDEN_LITERALS = ("lambda_csg", "lambda_trans", "trans_graph_path",
                      "clean_tiger", "user_sid")


def _identifiers(tree):
    """Every identifier the module's code resolves: names, attrs, defs, args."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            out.add(node.arg)
    return out


def part7_scope():
    print("\n[7] scope: no loss / constraint / projection, no frozen-dir change")
    status = subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                            capture_output=True, text=True).stdout.splitlines()
    modified = {l[3:].strip() for l in status if l[:2].strip() == "M"}
    check("C2 adds no tracked-file modification at all",
          modified <= PRE_C2_MODIFIED, f"modified: {sorted(modified)}")
    check("clean_tiger/ is untouched by C2",
          not [l for l in status if "clean_tiger" in l])
    check("the frozen core is untouched: only the 3 pre-existing edits are present",
          all(not l[3:].strip().startswith(("clean_tiger", "tools/seq_"))
              for l in status if l[:2].strip() == "M"),
          "no rqvae/ model/ file was added to the modified set by C2")

    source = open(MODULE, encoding="utf-8").read()
    tree = ast.parse(source)
    top_imports, top_defs = set(), set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_imports.add((node.module or "").split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            top_defs.add(node.name)
    check("the module imports numpy plus C1's digest helper -- no torch, sklearn, "
          "scipy, pandas",
          top_imports == {"json", "os", "numpy", "cf_embed"},
          f"top-level imports: {sorted(top_imports)}")
    idents = _identifiers(tree)
    offenders = sorted({i for i in idents
                        for t in FORBIDDEN_IDENTIFIERS if t in i.lower()})
    check("no C3/C4/switch/user-side identifier exists anywhere in the code",
          not offenders, f"scanned {len(idents)} identifiers; offenders: {offenders}")
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    bad_lit = sorted(t for t in FORBIDDEN_LITERALS
                     if any(t in s for s in literals))
    check("no frozen-dir path and no historical switch name appears even as a string",
          not bad_lit, f"scanned {len(literals)} string literals; offenders: {bad_lit}")
    check("no top-level def names a loss / projection / alignment",
          not any(t in d.lower() for d in top_defs for t in FORBIDDEN_IDENTIFIERS),
          f"top-level defs: {sorted(top_defs)}")
    layer_calls = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, (ast.Name, ast.Attribute))
                   and (getattr(n.func, "id", None) or getattr(n.func, "attr", ""))
                   in ("Linear", "Normalize", "normalize", "Parameter", "Embedding")]
    check("no layer is constructed: the cosine's own L2 normalisation is the only "
          "`normalize`, and it is applied to a local copy",
          not layer_calls and source.count("l2_normalize(") == 2,
          "l2_normalize is the definition of cosine, not a layer on anyone's input")
    check("the relation is a side artifact: the module references no model, no "
          "quantiser and no latent",
          not ({"RQVAE", "x_q", "codebook", "forward", "state_dict", "torch"}
               & idents),
          "collab_relation.py holds no reference to any model object")
    check("the historical transition feature's switch semantics are not reused",
          "trans_graph_path" not in source and "lambda_trans" not in source,
          "only the (M, 2) row-index array shape is mirrored, nothing else")
    check("the module reads no file at import time (proved, not asserted)",
          subprocess.run([sys.executable, "-I", "-c", IMPORT_PROBE, REPO],
                         capture_output=True, text=True).stdout.strip()
          == "IMPORT_CALLS=0",
          "numpy.load and open() were wrapped during `import tiger_ext`")
    check("tiger_ext re-exports the C2 interface and shares C1's id digest",
          __import__("tiger_ext").CollabRelation is CR.CollabRelation
          and CR.item_ids_sha256 is CE.item_ids_sha256
          and __import__("tiger_ext").RELATION_K == 20,
          "RELATION_K=20, one definition of the row-space digest")
    check("provenance records the reference implementation and every difference",
          CR.REFERENCE_IMPLEMENTATION["k_default"] == 20
          and len(CR.PORT_DIFFERENCES) == 5,
          "csg4rec_reference + 5 recorded differences")


# --------------------------------------------------------------------------- #
# [8] regression
# --------------------------------------------------------------------------- #
COUNT_RE = __import__("re").compile(r"(\d+)/(\d+) checks passed")


def part8_regression():
    print("\n[8] regression: the E1-E10 suites, re-run after C2")
    if REGRESSION == "skip":
        print("  [skip] EXP_C2_REGRESSION=skip")
        return
    env = dict(os.environ)
    env.pop("EXP_E10_DEVICE", None)
    proc = subprocess.run([sys.executable, "-I", "integration/exp_e10_smoke.py"],
                          cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=3600)
    match = None
    for line in proc.stdout.splitlines():
        m = COUNT_RE.search(line)
        if m:
            match = m
    ok, n = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    detail = "" if proc.returncode == 0 else " | ".join(
        ln for ln in proc.stderr.splitlines() if ln.strip())[-200:]
    check(f"exp_e10_smoke (which re-runs E1-E9): {ok}/{n}", ok == n and n > 0, detail)


# --------------------------------------------------------------------------- #
def main():
    print("=" * 76)
    print(f"EXP-C2 -- collaborative kNN diagnostic (k={K}, "
          f"metric={CR.RELATION_METRIC}, regression={REGRESSION})")
    tmp = tempfile.mkdtemp(prefix="exp_c2_")
    try:
        part1_synthetic(tmp)
        built = part2_real(tmp)
        part3_independent(built)
        part4_mapping(built)
        part5_quantity(built)
        part6_relatedness(built)
        part7_scope()
        part8_regression()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 76)
    passed = sum(RESULTS)
    print(f"EXP-C2: {passed}/{len(RESULTS)} checks passed")
    print("=" * 76)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
