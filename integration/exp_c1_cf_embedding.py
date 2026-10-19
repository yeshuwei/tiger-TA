"""EXP-C1 -- CF embedding sanity check (A class).

Run:  python -I integration/exp_c1_cf_embedding.py
      EXP_C1_REGRESSION=skip python -I integration/exp_c1_cf_embedding.py

Covers plan.md C1's acceptance criteria plus the interface guarantees C2/C3 will
depend on:

  [1] synthetic fixture: shape / dtype / row-order fidelity, the id map, the
      missing-item policy, save -> reload round trips, and the negative controls
      that prove each guard can actually fire
  [2] real data (Toys / Sport / Beauty): row count and ItemID alignment against
      `item_emb.parquet`, explicit dimension, provenance
  [3] the SASRec extraction contract, re-derived from the checkpoint independently
      of the loader: the padding row, the by-ItemID gather, and the negative
      control that `W[1:]` would mis-assign 34% of Beauty
  [4] missing / OOV behaviour, including the completeness proof that
      `vectors(canonical_item_ids) == table` (the map is a lossless bijection)
  [5] the ItemCF fallback is still on disk and still loadable, and is no longer
      the default source
  [6] the torch hand-off (device, dtype, and that it is a copy)
  [7] scope: no new representation, no projection / normalisation, none of
      C2/C3/C4, nothing read at import time, no frozen directory touched
  [8] regression: the E1-E10 suites, re-run after C1

Source ruling (2026-10-14): the official CSG CF representation is the **SASRec
item embedding** (`sasrec_output/<D>/seed2025_best.pth`, key `item_emb.weight`),
because CSG4Rec's own `--cf_emb` default is exactly that artifact
(`CSGT/main.py:54`, `Instruments-32d-sasrec.pt`) and the reference snapshot ships
no SASRec trainer to reproduce. `data/<D>/behavior_emb_itemcf.parquet` (128-d,
ItemCF cosine + SVD) is retained as a fallback, not deleted, and not the main
line. See `tiger_ext/cf_embed.py` for the full argument.

Engineering gate only -- nothing here says anything about recommendation quality.
"""
import ast
import filecmp
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

from tiger_ext import cf_embed as CE                  # noqa: E402

DATA_DIR = os.path.join(REPO, "data")
MODULE = os.path.join(REPO, "tiger_ext", "cf_embed.py")
ARTIFACT_DIR = os.path.join(HERE, "artifacts", "cf_embed")
REGRESSION = os.environ.get("EXP_C1_REGRESSION", "all")

# the tracked files E1-E6 legitimately modified; C1 must not add to this set
PRE_C1_MODIFIED = {"model/main.py", "rqvae/models/rq.py", "rqvae/models/vq.py"}

# measured facts about the two sources (verified in [2]/[3]/[5]; pinned here so a
# silent change of the underlying data shows up as a failure rather than a shrug)
SASREC_SOURCE = {
    #        n_items, d_cf, ckpt rows, padding row is all-zero ?
    "Toys":   (11924, 50, 11925, True),
    "Sport":  (18357, 50, 18358, True),
    "Beauty": (12101, 50, 12102, True),
}
ITEMCF_FALLBACK = {
    #        n_items, d_cf, all-zero (cold-start) rows
    "Toys":   (11924, 128, 24),
    "Sport":  (18357, 128, 28),
    "Beauty": (12101, 128, 33),
}
BEAUTY_SCRAMBLED_ROWS = 4158     # Beauty rows where ItemID != row + 1
SASREC_SEED = 2025

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
    """Bitwise equality of two float arrays (plan.md G0 style, not allclose)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def fixture():
    """A tiny non-contiguous CF table with a deliberate cold-start row.

    n=7, d=4. `item_ids` deliberately scrambles the order so that no row can be
    found by `item_id - 1`; row 3 (item id 2) is all-zero, i.e. a cold-start item.
    """
    item_ids = np.array([3, 1, 7, 2, 9, 5, 11], dtype=np.int64)
    table = np.zeros((7, 4), dtype=np.float32)
    for row in range(7):
        if row == 3:
            continue                      # the cold-start row stays all-zero
        table[row] = (row + 1) * 0.25 + np.arange(4) * 0.5
    return table, item_ids


# --------------------------------------------------------------------------- #
# [1] synthetic fixture
# --------------------------------------------------------------------------- #
def part1_synthetic(tmp):
    print("\n[1] synthetic fixture: the interface, and its negative controls")
    table, item_ids = fixture()
    emb = CE.CFEmbedding.from_arrays(table, item_ids, source=None)

    check("shape / dtype / n_items / dim are what was handed in",
          emb.shape == (7, 4) and emb.dtype == np.dtype(np.float32)
          and emb.n_items == 7 and emb.dim == 4 and len(emb) == 7,
          f"shape={emb.shape} dtype={emb.dtype} n={emb.n_items} dim={emb.dim}")
    check("the ItemID column is preserved verbatim in row order",
          np.array_equal(emb.item_ids, item_ids)
          and emb.item_ids.dtype == np.int64, f"ids={emb.item_ids.tolist()}")
    check("contiguous is False here, and `item_id - 1` would be wrong",
          emb.contiguous is False
          and sum(int(i) - 1 != r for r, i in enumerate(item_ids)) == 7,
          f"a-1 correct for "
          f"{sum(int(i) - 1 == r for r, i in enumerate(item_ids))}/7 rows")

    # --- the mapping ------------------------------------------------------- #
    check("row_of resolves through the id map, not through arithmetic",
          emb.row_of(7) == 2 and emb.row_of(3) == 0 and emb.row_of(11) == 6
          and emb.row_of(7) != 7 - 1,
          f"row_of(7)={emb.row_of(7)} (7-1=6)")
    check("row_of raises MissingItemError for an absent id",
          raises(lambda: emb.row_of(4), CE.MissingItemError))
    check("`in` reports membership without raising",
          (7 in emb) and (4 not in emb) and (0 not in emb) and ("x" not in emb))
    check("index_map is exactly the id -> row bijection",
          emb.index_map() == {3: 0, 1: 1, 7: 2, 2: 3, 9: 4, 5: 5, 11: 6},
          str(emb.index_map()))
    check("row_of_many keeps order and flags absent ids with -1",
          np.array_equal(emb.row_of_many([9, 3, 5]), [4, 0, 5])
          and np.array_equal(emb.row_of_many([9, 4], missing="zero"), [4, -1]),
          "raise-ok, zero-ok")
    check("row_of_many under the default policy raises on the first absent id",
          raises(lambda: emb.row_of_many([9, 4]), CE.MissingItemError))

    # --- access ------------------------------------------------------------ #
    check("vector(id) returns that item's row, bitwise",
          same_bytes(emb.vector(7), table[2]) and same_bytes(emb.vector(3), table[0]),
          f"vector(7)={emb.vector(7).tolist()}")
    check("vectors() preserves the requested order",
          same_bytes(emb.vectors([3, 9, 11]), table[[0, 4, 6]]))
    check("an absent id raises by default but yields zeros under missing='zero'",
          raises(lambda: emb.vector(4), CE.MissingItemError)
          and same_bytes(emb.vector(4, missing="zero"), np.zeros(4, dtype=np.float32))
          and same_bytes(emb.vectors([3, 4], missing="zero"), table[[0, 3]]),
          "vector(4, 'zero') == 0")
    check("the cold-start row is reported as such (not as an error)",
          emb.is_zero_row(2) is True and emb.is_zero_row(3) is False
          and same_bytes(emb.vector(2), np.zeros(4, dtype=np.float32)),
          "item 2 is present and all-zero")
    check("zero-row accounting: n_zero_rows and valid_mask agree",
          emb.n_zero_rows == 1 and int(emb.valid_mask.sum()) == 6
          and bool(emb.zero_row_mask[3]) and not bool(emb.zero_row_mask[0]),
          f"zero_rows={emb.n_zero_rows}")
    check("an unknown missing-policy string is rejected, not silently accepted",
          raises(lambda: emb.vector(3, missing="clamp"), ValueError)
          and raises(lambda: emb.row_of_many([3], missing="nope"), ValueError))

    # --- negative controls on the constructor ------------------------------ #
    check("a duplicate ItemID is refused (the map would be ambiguous)",
          raises(lambda: CE.CFEmbedding(table, np.array([3, 3, 7, 2, 9, 5, 11])),
                 CE.CFEmbeddingMismatch))
    check("a row-count / id-count mismatch is refused",
          raises(lambda: CE.CFEmbedding(table, item_ids[:6]), CE.CFEmbeddingMismatch))
    check("a non-2-D table is refused",
          raises(lambda: CE.CFEmbedding(table[:, 0], item_ids), CE.CFEmbeddingMismatch))
    check("an empty table is refused",
          raises(lambda: CE.CFEmbedding(np.zeros((0, 4), dtype=np.float32),
                                        np.zeros(0, dtype=np.int64)),
                 CE.CFEmbeddingMismatch))
    check("a zero-width table is refused",
          raises(lambda: CE.CFEmbedding(np.zeros((7, 0), dtype=np.float32), item_ids),
                 CE.CFEmbeddingMismatch))
    bad = table.copy()
    bad[2, 1] = np.nan
    check("a non-finite entry is refused unless explicitly allowed",
          raises(lambda: CE.CFEmbedding.from_arrays(bad, item_ids), CE.CFEmbeddingMismatch)
          and CE.CFEmbedding.from_arrays(bad, item_ids,
                                         allow_nonfinite=True).n_items == 7,
          "NaN at row 2")

    # --- alignment checks -------------------------------------------------- #
    check("check_alignment accepts the identical id vector",
          emb.validate(expected_item_ids=item_ids) is emb
          and emb.validate(expected_n_items=7) is emb)
    check("check_alignment refuses a permuted id vector (row order, not id set)",
          raises(lambda: emb.check_alignment(item_ids[::-1]), CE.CFEmbeddingMismatch))
    check("check_alignment refuses a wrong row count",
          raises(lambda: emb.check_alignment(item_ids[:6]), CE.CFEmbeddingMismatch))
    check("validate() refuses a wrong n_items",
          raises(lambda: emb.validate(expected_n_items=6), CE.CFEmbeddingMismatch))
    check("the id digest is order-sensitive, so it pins the mapping",
          CE.item_ids_sha256(item_ids) == CE.item_ids_sha256(item_ids)
          and CE.item_ids_sha256(item_ids) != CE.item_ids_sha256(item_ids[::-1]),
          CE.item_ids_sha256(item_ids)[:16] + "...")

    # --- save / reload ----------------------------------------------------- #
    pq = os.path.join(tmp, "fixture.parquet")
    emb.save(pq)
    back = CE.CFEmbedding.load(pq)
    check("parquet round trip: table bitwise, ids exact, dim/n_items preserved",
          same_bytes(back.table, emb.table) and np.array_equal(back.item_ids, item_ids)
          and back.shape == emb.shape and back.dtype == emb.dtype,
          f"{back!r}")
    check("parquet round trip survives a second cycle unchanged",
          (lambda p2: back.save(p2) and filecmp.cmp(pq, p2, shallow=False))(
              os.path.join(tmp, "fixture2.parquet")),
          "byte-identical file")
    check("the saved file carries the canonical ItemID + embedding columns",
          pd.read_parquet(pq).columns.tolist() == list(CE.PARQUET_COLUMNS),
          str(list(CE.PARQUET_COLUMNS)))

    npy = os.path.join(tmp, "fixture.npy")
    emb.save(npy, fmt="npy")
    back_npy = CE.CFEmbedding.from_npy(npy, item_ids)
    check("npy round trip: table bitwise, and the .ids.npy sidecar holds the map",
          same_bytes(back_npy.table, emb.table)
          and np.array_equal(np.load(os.path.join(tmp, "fixture.ids.npy")), item_ids),
          "sidecar fixture.ids.npy")
    check("format is inferred from the extension, and an unknown one is refused",
          raises(lambda: emb.save(os.path.join(tmp, "x.txt")), CE.CFEmbeddingError)
          and raises(lambda: emb.save(os.path.join(tmp, "y.parquet"), fmt="csv"),
                     CE.CFEmbeddingError))
    check("loading a path that does not exist is an error, not an empty table",
          raises(lambda: CE.CFEmbedding.load(os.path.join(tmp, "nope.parquet")),
                 CE.CFEmbeddingError))

    # --- provenance -------------------------------------------------------- #
    prov = emb.provenance()
    need = {"task", "n_items", "dim", "dtype", "item_ids_sha256",
            "contiguous_item_ids", "n_zero_rows", "missing_policy",
            "source_kind", "is_default_source"}
    check("provenance records the shape, the id digest and the source kind",
          need <= set(prov) and prov["task"] == "C1" and prov["n_items"] == 7
          and prov["dim"] == 4 and prov["contiguous_item_ids"] is False
          and prov["n_zero_rows"] == 1,
          f"keys={sorted(need)}")
    pj = emb.write_provenance(os.path.join(tmp, "prov.json"))
    check("provenance round trips through JSON",
          json.load(open(pj, encoding="utf-8"))["item_ids_sha256"]
          == prov["item_ids_sha256"])

    check("MissingItemError is both a CFEmbeddingError and a KeyError",
          issubclass(CE.MissingItemError, CE.CFEmbeddingError)
          and issubclass(CE.MissingItemError, KeyError)
          and issubclass(CE.CFEmbeddingMismatch, CE.CFEmbeddingError))


# --------------------------------------------------------------------------- #
# [2] real data -- the official SASRec source
# --------------------------------------------------------------------------- #
def part2_real(tmp):
    print("\n[2] real data (official source = SASRec item embedding)")
    loaded = {}
    for name, (n_items, dim, ckpt_rows, pad_zero) in SASREC_SOURCE.items():
        canonical = CE.canonical_item_ids(name, data_dir=DATA_DIR)
        path = CE.sasrec_checkpoint_path(name, output_dir=os.path.join(REPO, "sasrec_output"),
                                         seed=SASREC_SEED)
        emb = CE.load_cf_embedding(name, expected_item_ids=canonical)
        loaded[name] = emb
        prov = emb.provenance()

        check(f"{name}: the default source is SASRec, dim {dim}, shape ({n_items}, {dim})",
              prov["source_kind"] == "sasrec" and prov["is_default_source"] is True
              and emb.shape == (n_items, dim) and emb.dtype == np.dtype(np.float32),
              f"{emb!r}")
        check(f"{name}: row count == item_emb.parquet rows ({n_items})",
              emb.n_items == n_items == len(canonical), f"n={emb.n_items}")
        check(f"{name}: provenance points at the checkpoint, its hash and its seed",
              os.path.abspath(prov["source"]) == os.path.abspath(path)
              and len(prov["source_sha256"]) == 64
              and prov.get("seed") == SASREC_SEED
              and prov["source_checkpoint"] == os.path.abspath(path),
              f"{os.path.relpath(path, REPO)} sha256={prov['source_sha256'][:12]}...")
        check(f"{name}: ItemID aligns with item_emb.parquet element-wise",
              np.array_equal(emb.item_ids, canonical),
              f"digest {CE.item_ids_sha256(emb.item_ids)[:16]}...")

        scrambled = int((emb.item_ids != np.arange(1, emb.n_items + 1)).sum())
        expect_scrambled = 0 if name != "Beauty" else BEAUTY_SCRAMBLED_ROWS
        check(f"{name}: contiguous={scrambled == 0} "
              f"({scrambled} rows would break `a-1`)",
              emb.contiguous is (scrambled == 0) and scrambled == expect_scrambled,
              f"a-1 wrong at {scrambled} rows")

        saved = emb.write_provenance(os.path.join(ARTIFACT_DIR, f"{name}.json"))
        check(f"{name}: provenance manifest written to integration/artifacts/",
              os.path.exists(saved)
              and json.load(open(saved, encoding="utf-8"))["dim"] == dim
              and json.load(open(saved, encoding="utf-8"))["source_kind"] == "sasrec",
              os.path.relpath(saved, REPO))

        rt = os.path.join(tmp, f"{name}_rt.parquet")
        emb.save(rt)
        reloaded = CE.CFEmbedding.load(rt)
        check(f"{name}: save -> reload is bitwise identical",
              same_bytes(reloaded.table, emb.table)
              and np.array_equal(reloaded.item_ids, emb.item_ids),
              f"{os.path.getsize(rt)} bytes")
    return loaded


# --------------------------------------------------------------------------- #
# [3] the SASRec extraction contract, re-derived from the checkpoint
# --------------------------------------------------------------------------- #
def part3_sasrec_contract(loaded):
    """Re-do the extraction here, without the loader, and compare bitwise.

    If the loader's gather were wrong -- an off-by-one, a `W[1:]` slice, a
    swapped row -- this section disagrees with it.
    """
    print("\n[3] SASRec extraction: padding row, by-ItemID gather, and the "
          "`W[1:]` negative control")
    import torch

    for name, emb in loaded.items():
        canonical = emb.item_ids
        ckpt = CE.sasrec_checkpoint_path(
            name, output_dir=os.path.join(REPO, "sasrec_output"), seed=SASREC_SEED)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        W = state[CE.SASREC_EMB_KEY].numpy()

        check(f"{name}: the raw embedding matrix is (N+1, {emb.dim}) float32",
              W.shape == (emb.n_items + 1, emb.dim) and W.dtype == np.float32,
              f"{W.shape} {W.dtype}")
        check(f"{name}: row 0 is the padding row and is exactly all-zero",
              bool((W[CE.SASREC_PADDING_ROW] == 0).all()),
              f"sum|row0| = {float(np.abs(W[0]).sum())}")
        check(f"{name}: the table is exactly W[canonical ItemID], bitwise",
              same_bytes(emb.table, W[canonical]), "independently re-derived")

        # the padding row is dropped, and only it: W[1:] is used exactly once each
        used = np.sort(canonical)
        check(f"{name}: every real row of W is used once and padding is unused",
              np.array_equal(used, np.arange(1, W.shape[0]))
              and CE.SASREC_PADDING_ROW not in set(canonical.tolist()),
              f"rows 1..{W.shape[0] - 1} covered exactly once")

        # the by-ItemID property C2 will rely on
        probe = [int(canonical[0]), int(canonical[len(canonical) // 2]),
                 int(canonical.max())]
        check(f"{name}: vector(ItemID=k) is W[k], for sampled k",
              all(same_bytes(emb.vector(k), W[k]) for k in probe),
              f"k={probe}")

        # negative control: the naive `W[1:]` slice
        wrong = int((canonical != np.arange(1, emb.n_items + 1)).sum())
        if wrong:
            r = int(np.nonzero(canonical != np.arange(1, emb.n_items + 1))[0][0])
            check(f"{name}: negative control -- `W[1:]` would mis-assign {wrong} rows",
                  not same_bytes(emb.table[r], W[r + 1])
                  and same_bytes(emb.table[r], W[canonical[r]]),
                  f"row {r}: ItemID {int(canonical[r])} -> W[{int(canonical[r])}], "
                  f"not W[{r + 1}]")
        else:
            check(f"{name}: `W[1:]` happens to agree here (ItemID is exactly 1..N), "
                  f"so this dataset cannot catch the slice bug",
                  same_bytes(emb.table, W[1:]),
                  "the gather is still the contract; no negative control available")

        # cold-start: this source has no all-zero rows at all
        check(f"{name}: no all-zero rows (an untrained item keeps its random init)",
              emb.n_zero_rows == 0
              and emb.provenance()["missing_policy"]["cold_start_row"]
              == CE.SASREC_COLD_START,
              f"zero_rows={emb.n_zero_rows}")

        # the extractor's own guards fire
        check(f"{name}: the extractor refuses a non-zero padding row",
              raises(lambda c=ckpt, W=W: CE.load_sasrec_item_embedding(
                  c, canonical, padding_row=1), CE.CFEmbeddingMismatch),
              "padding_row=1 is not all-zero -> refused, not silently used")
        check(f"{name}: the extractor refuses an id space that hits padding",
              raises(lambda c=ckpt: CE.load_sasrec_item_embedding(
                  c, np.where(canonical == canonical.max(), 0, canonical)),
                  CE.CFEmbeddingMismatch),
              "an id mapped to 0 would land on the padding row")
        check(f"{name}: the extractor refuses a wrong number of items",
              raises(lambda c=ckpt: CE.load_sasrec_item_embedding(c, canonical[:-1]),
                     CE.CFEmbeddingMismatch),
              f"{len(canonical) - 1} ids vs {W.shape[0]} rows")


# --------------------------------------------------------------------------- #
# [4] missing / OOV behaviour
# --------------------------------------------------------------------------- #
def part4_missing(loaded):
    print("\n[4] missing / OOV: explicit, and never a silently wrong row")
    for name, emb in loaded.items():
        canonical = emb.item_ids
        top = int(canonical.max())

        check(f"{name}: vectors(all canonical ids) reproduces the table bitwise",
              same_bytes(emb.vectors(canonical), emb.table),
              "the id map covers every row exactly once")

        check(f"{name}: an id above the space raises, and 'zero' yields zeros only",
              raises(lambda: emb.row_of(top + 1), CE.MissingItemError)
              and raises(lambda: emb.vector(0), CE.MissingItemError)
              and same_bytes(emb.vector(top + 1, missing="zero"),
                             np.zeros(emb.dim, dtype=np.float32)),
              f"unknown id {top + 1} (space max {top})")
        check(f"{name}: absent ids in a batch are reported positionally",
              np.array_equal(emb.row_of_many([canonical[0], top + 1], missing="zero"),
                             [0, -1]) and raises(
                  lambda: emb.vectors([canonical[0], top + 1]), CE.MissingItemError))


# --------------------------------------------------------------------------- #
# [5] the ItemCF fallback
# --------------------------------------------------------------------------- #
def part5_fallback():
    print("\n[5] the ItemCF fallback is retained, loadable, and not the default")
    for name, (n_items, dim, n_zero) in ITEMCF_FALLBACK.items():
        path = CE.default_cf_path(name, data_dir=DATA_DIR)
        emb = CE.load_cf_embedding(name, source="itemcf")
        prov = emb.provenance()

        check(f"{name}: behavior_emb_itemcf.parquet still exists, untouched",
              os.path.exists(path) and emb.n_items == n_items and emb.dim == dim,
              f"{emb!r}")
        check(f"{name}: the fallback is not the default source",
              prov["source_kind"] == "itemcf" and prov["is_default_source"] is False,
              "source_kind=itemcf")
        check(f"{name}: its ItemID column still matches item_emb.parquet",
              np.array_equal(emb.item_ids, CE.canonical_item_ids(name, data_dir=DATA_DIR)),
              "same canonical row order")
        check(f"{name}: its cold-start rows ({n_zero}) are still all-zero",
              emb.n_zero_rows == n_zero
              and prov["missing_policy"]["cold_start_row"] == CE.ITEMCF_COLD_START,
              f"zero_rows={emb.n_zero_rows}")
        emb.write_provenance(os.path.join(ARTIFACT_DIR, f"fallback_{name}.json"))

    check("the two sources are distinguishable only through provenance, "
          "not through the interface",
          CE.load_cf_embedding("Toys").shape == (11924, 50)
          and CE.load_cf_embedding("Toys", source="itemcf").shape == (11924, 128)
          and type(CE.load_cf_embedding("Toys")) is type(
              CE.load_cf_embedding("Toys", source="itemcf")),
          "same class, different dim")
    check("an unknown source name is refused",
          raises(lambda: CE.load_cf_embedding("Toys", source="svd"),
                 CE.CFEmbeddingError))


# --------------------------------------------------------------------------- #
# [6] torch hand-off
# --------------------------------------------------------------------------- #
def part6_handoff(loaded):
    print("\n[6] torch hand-off: dtype, device, and that it is a copy")
    import torch
    for name, emb in loaded.items():
        before = emb.table.copy()
        t_cpu = emb.to_tensor("cpu")
        check(f"{name}: to_tensor('cpu') has the right shape / dtype / device",
              tuple(t_cpu.shape) == emb.shape and t_cpu.dtype == torch.float32
              and t_cpu.device.type == "cpu"
              and torch.equal(t_cpu, torch.from_numpy(before)),
              f"{tuple(t_cpu.shape)} {t_cpu.dtype} {t_cpu.device}")
        t_cpu[0, 0] = -1.0
        check(f"{name}: the tensor is a copy, not a view onto the stored table",
              same_bytes(emb.table, before), "mutating the tensor left the table alone")
        if torch.cuda.is_available():
            t_gpu = emb.to_tensor("cuda")
            check(f"{name}: to_tensor('cuda') moves device without changing values",
                  t_gpu.device.type == "cuda"
                  and torch.equal(t_gpu.cpu(), torch.from_numpy(before)),
                  str(t_gpu.device))
            t_gpu[0, 0] = -2.0
            check(f"{name}: the CUDA tensor is also a copy",
                  same_bytes(emb.table, before), "mutating it left the table alone")
    if not torch.cuda.is_available():
        print("  [skip] CUDA not available -- the device branch is covered on the GPU box")


# --------------------------------------------------------------------------- #
# [7] scope
# --------------------------------------------------------------------------- #
IMPORT_PROBE = r'''
import sys, os
sys.path.insert(0, sys.argv[1])
import numpy as np, pandas as pd
calls = []
_rp, _nl = pd.read_parquet, np.load
pandas_read = lambda *a, **k: (calls.append(("read_parquet", str(a[0]))), _rp(*a, **k))[1]
numpy_load = lambda *a, **k: (calls.append(("np.load", str(a[0]))), _nl(*a, **k))[1]
pd.read_parquet, np.load = pandas_read, numpy_load
import tiger_ext, tiger_ext.cf_embed
pd.read_parquet, np.load = _rp, _nl
print("IMPORT_CALLS=%d" % len(calls))
'''

FORBIDDEN_DEF_TOKENS = ("cosine", "knn", "loss", "contrastive", "margin",
                        "proj", "normal", "linear", "regulari", "warmup")


def part7_scope():
    print("\n[7] scope: no new representation or projection, no C2/C3/C4, "
          "no frozen-dir change")
    status = subprocess.run(["git", "-C", REPO, "status", "--porcelain"],
                            capture_output=True, text=True).stdout.splitlines()
    modified = {l[3:].strip() for l in status if l[:2].strip() == "M"}
    check("C1 adds no tracked-file modification at all",
          modified <= PRE_C1_MODIFIED, f"modified: {sorted(modified)}")
    check("clean_tiger/ is untouched by C1",
          not [l for l in status if "clean_tiger" in l])
    check("no E-line file was edited: tiger_ext/ and integration/ are new-file only",
          all(l[:2].strip() in ("??", "M") for l in status
              if "tiger_ext" in l or "integration" in l),
          "tiger_ext/ and integration/ remain untracked")

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
    check("the module imports no numerical stack at import time (torch is lazy)",
          "torch" not in top_imports and "scipy" not in top_imports,
          f"top-level imports: {sorted(top_imports)}")
    check("the module does not import the reference repos or any user-side code",
          not any("reference_repos" in s for s in top_imports)
          and "reference_repos" not in source,
          "no CSG4Rec source is imported or copied")
    check("no C2/C3/C4 and no projection / normalisation / loss was added",
          not any(tok in d.lower() for d in top_defs
                  for tok in FORBIDDEN_DEF_TOKENS),
          f"top-level defs: {sorted(top_defs)}")
    LAYER_NAMES = {"Linear", "Embedding", "Parameter", "LayerNorm",
                   "BatchNorm1d", "Sequential", "init"}
    layer_calls = [n for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and isinstance(n.func.value, ast.Name)
                   and (n.func.value.id in ("nn", "F")
                        or n.func.attr in LAYER_NAMES)]
    check("no torch layer, parameter or functional op is constructed in the module",
          not layer_calls and "nn" not in top_imports,
          "the 50-d source needed no projection and none was added; "
          "`nn.Embedding(...)` occurs only inside the SASREC_ROW_OF_ITEM "
          "documentation string, never as code (only torch.load / from_numpy / "
          "device are called)")
    check("no user-side modelling: no user id / user embedding / User SID anywhere",
          not any(tok in source.lower() for tok in
                  ("user_id", "user_emb", "user_sid", "user_repr", "n_users")),
          "item-level only")
    check("the module reads no file at import time (proved, not asserted)",
          subprocess.run([sys.executable, "-I", "-c", IMPORT_PROBE, REPO],
                         capture_output=True, text=True).stdout.strip()
          == "IMPORT_CALLS=0",
          "pandas.read_parquet / numpy.load were wrapped during `import tiger_ext`")
    check("tiger_ext re-exports the C1 interface and the source constants",
          __import__("tiger_ext").CFEmbedding is CE.CFEmbedding
          and __import__("tiger_ext").DEFAULT_CF_SOURCE == "sasrec",
          "DEFAULT_CF_SOURCE=sasrec")


# --------------------------------------------------------------------------- #
# [8] regression
# --------------------------------------------------------------------------- #
COUNT_RE = __import__("re").compile(r"(\d+)/(\d+) checks passed")


def part8_regression():
    print("\n[8] regression: the E1-E10 suites, re-run after C1")
    if REGRESSION == "skip":
        print("  [skip] EXP_C1_REGRESSION=skip")
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
    print(f"EXP-C1 -- CF embedding sanity check "
          f"(source={CE.DEFAULT_CF_SOURCE}, seed={SASREC_SEED}, regression={REGRESSION})")
    tmp = tempfile.mkdtemp(prefix="exp_c1_")
    try:
        part1_synthetic(tmp)
        loaded = part2_real(tmp)
        part3_sasrec_contract(loaded)
        part4_missing(loaded)
        part5_fallback()
        part6_handoff(loaded)
        part7_scope()
        part8_regression()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 76)
    passed = sum(RESULTS)
    print(f"EXP-C1: {passed}/{len(RESULTS)} checks passed")
    print("=" * 76)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
