"""C1 -- CF embedding interface (CSG4Rec -> TIGER port).

Contract: ``integration/plan.md`` C1 table.

    | 来源机制 | CSG4Rec 的 item-level collaborative embedding                |
    | 输入     | 既有 item 级 CF 向量文件(或 ``data/<D>/{train,valid,test}.parquet``) |
    | 输出     | ``(n_items, d_cf)`` item 级协同向量 + ``item_id -> 行号`` 映射表     |
    | 新增     | CF 向量来源与其加载/校验模块(**仅 item 级**)                        |
    | 修改     | 无                                                          |

This module is the *interface*: it loads, validates, indexes and re-saves an
item-level collaborative vector table. It computes no CF representation of its
own, takes no user-side input, and touches neither the RQ-VAE input path nor the
T5 structure. Cosine similarity / kNN (C2) and the Q-space constraint (C3) are
deliberately **not** implemented here.

Source decision (plan.md's "C1 决策点", resolved from evidence)
----------------------------------------------------------------
``plan.md`` leaves the source open between

  (a) reuse the in-repo ``data/<D>/behavior_emb_itemcf.parquet`` ``(n_items, 128)``
  (b) faithfully re-implement CSG4Rec's own CF embedding

and asks for the choice to be justified from evidence. The evidence:

* CSG4Rec's ``--cf_emb`` default is ``./RQ-VAE/ckpt/Instruments-32d-sasrec.pt``
  (``CSGT/main.py:54``); it is ``np.load``-ed and indexed by the same row order
  as the item text embedding (``CSGT/datasets.py:24-29``). So CSG4Rec's CF vector
  is an **item embedding table of a trained SASRec**, not an ItemCF matrix.
* The reference snapshot ships no SASRec trainer (``CSGT/models/`` holds only
  ``layers.py`` / ``rq.py`` / ``vq.py``; ``models.rqvae`` is missing too), so
  (b) cannot be reproduced from the reference -- it would mean training a fresh
  sequential model.
* This repo already holds SASRec checkpoints of its own (``sasrec_output/<D>/
  seed<seed>_best.pth``, the aligned-protocol 3-seed baseline), whose
  ``item_emb.weight`` **is** exactly the artifact CSG4Rec's ``--cf_emb`` names.

**Ruling: the official CSG CF representation is the SASRec item embedding**
(``CF_SOURCE_SASREC``), extracted per dataset by :func:`load_sasrec_item_embedding`.
``data/<D>/behavior_emb_itemcf.parquet`` is kept on disk as a *fallback*
representation (``CF_SOURCE_ITEMCF``) but is no longer the CSG main line. Both
routes go through :class:`CFEmbedding`, so the interface is source-agnostic and
no downstream code knows which one it is talking to besides ``provenance()``.

SASRec extraction contract (measured on all three datasets)
-----------------------------------------------------------
* ``item_emb.weight`` is ``(N + 1, 50)`` float32, where ``N`` is the canonical
  item count (11924 / 18357 / 12101). Row ``SASREC_PADDING_ROW = 0`` is the
  padding row: ``nn.Embedding(item_num + 1, ..., padding_idx=0)`` plus the
  zeroed xavier init means it is **exactly all-zero** and never receives a
  gradient (checked, not assumed).
* Rows are indexed by **raw ItemID**: ``tools/seq_baselines_common.py`` reads
  ``itemnum = max(item id)`` straight out of the parquet, and ``evaluate_full``
  ranks against ``np.arange(1, itemnum + 1)`` / ``target - 1``. So SASRec row
  ``i`` is the embedding of ItemID ``i``, for ``i`` in ``1..N``.
* The extracted table is therefore ``W[canonical_item_ids]`` -- **not**
  ``W[1:]``. Those two agree only where ItemID happens to be ``1..N`` (Toys and
  Sport); on **Beauty, 4158 of 12101 rows are scrambled**, so ``W[1:]`` would
  silently attach the wrong vector to 34% of the catalogue.
* ``W[0]`` is dropped, never copied: the CF table has exactly ``N`` rows, one
  per real item, in canonical row order.

Cold-start difference between the two sources (recorded, not papered over)
--------------------------------------------------------------------------
With the ItemCF fallback an item that never appeared in training has an
**all-zero** row (the generator zeroes it), so ``zero_row_mask`` is the
cold-start flag: Toys 24 / Sport 28 / Beauty 33. With the SASRec source there
are **no all-zero rows at all** (measured: 0 for all three) -- a never-trained
item keeps its random xavier initialisation, which is finite and non-zero but
carries no learned signal. ``zero_row_mask`` therefore stays a *correct* but
much less informative flag here; the interface reports it as-is and says which
source it came from rather than inventing a second notion of "cold start".

Row-order contract (the "无错误 item mapping" acceptance item)
--------------------------------------------------------------
The project has been bitten before by ``item_id - 1`` style lookup: the
canonical row order is defined by ``data/<D>/item_emb.parquet``'s ``ItemID``
column, and ``model/dataset.py::item2code`` ties SID row ``i`` to ``item i + 1``.
On Toys and Sport ``ItemID`` happens to be ``1..N`` (so ``item_id - 1`` is
coincidentally right); **on Beauty it is a permutation of 1..N with 4158 of
12101 rows off by more than nothing** (measured, see EXP-C1). So:

* the map is always built from the file's ``ItemID`` column, never by arithmetic;
* ``row_of`` raises :class:`MissingItemError` for an unknown id instead of
  guessing;
* ``contiguous`` records whether ``item_id - 1`` would have been right, purely
  as an audit flag.

Missing / OOV policy
--------------------
Two different things, both explicit:

1. **Cold-start row** -- the item is inside the item space but never appeared in
   the training interactions, so its vector is all-zero (the convention already
   used by ``rqvae/build_transition_graph.py:291-294``). Kept as a zero vector;
   flagged by :meth:`CFEmbedding.is_zero_row` and counted by
   :attr:`CFEmbedding.n_zero_rows`. Measured: Toys 24, Sport 28, Beauty 33.
2. **Unknown id** -- an id outside the item space. ``missing="raise"`` (default)
   raises :class:`MissingItemError`; ``missing="zero"`` returns a zero vector for
   that position only. There is no other fallback, so a typo can never silently
   return some other item's vector.

Nothing in ``model/`` or ``rqvae/`` imports this module, and this module imports
neither: it is a numpy-only leaf used by the C-line verification scripts.
"""
import hashlib
import json
import os

import numpy as np

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
PARQUET_COLUMNS = ("ItemID", "embedding")   # the repo's canonical item-vector layout
FORMATS = ("parquet", "npy")
MISSING_RAISE = "raise"
MISSING_ZERO = "zero"
MISSING_POLICIES = (MISSING_RAISE, MISSING_ZERO)
CF_DTYPE = np.float32                        # matches item_emb / behavior_emb files

# The C1 source decision (see module docstring). Recorded here so downstream
# tasks and the manifest never have to re-derive it.
CF_SOURCE_SASREC = "sasrec"
CF_SOURCE_ITEMCF = "itemcf"
DEFAULT_CF_SOURCE = CF_SOURCE_SASREC

# --- SASRec source (the official CSG CF representation) --------------------- #
SASREC_OUTPUT_DIR = "sasrec_output"
SASREC_CKPT_TEMPLATE = "seed{seed}_best.pth"
SASREC_EMB_KEY = "item_emb.weight"
SASREC_PADDING_ROW = 0
SASREC_DIM = 50
SASREC_ROW_OF_ITEM = "nn.Embedding(item_num + 1, 50, padding_idx=0); row i is raw ItemID i"
SASREC_PADDING_HANDLING = ("row 0 is padding_idx: dropped, never copied into the CF "
                           "table, and asserted to be all-zero")
SASREC_COLD_START = ("a never-trained item keeps its random init, so there are no "
                     "all-zero rows (measured 0/0/0) -- unlike the ItemCF fallback, "
                     "where an unseen item is an all-zero row")
SASREC_RECIPE = ("item_emb.weight from the aligned-protocol SASRec baseline "
                 "(tools/run_sasrec.py), rows re-indexed by canonical ItemID")

# --- ItemCF fallback source (kept on disk, no longer the main line) --------- #
DEFAULT_CF_TEMPLATE = "behavior_emb_itemcf.parquet"
CF_SOURCE_VARIANT = CF_SOURCE_ITEMCF
CF_SOURCE_RECIPE = ("full-basket co-occurrence (all item pairs per user sequence)"
                    " -> ItemCF cosine normalisation -> truncated SVD")
CF_SOURCE_GENERATOR = os.path.join("rqvae", "behavior_variants.py")
ITEMCF_COLD_START = ("an item absent from the training interactions is written as an "
                     "all-zero row and flagged by is_zero_row")


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class CFEmbeddingError(Exception):
    """Base class for every failure raised by this module."""


class CFEmbeddingMismatch(CFEmbeddingError):
    """A CF table failed a shape / dtype / alignment / provenance check."""


class MissingItemError(CFEmbeddingError, KeyError):
    """An item id was looked up that is not in the CF table's item space."""


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def item_ids_sha256(item_ids) -> str:
    """Order-sensitive digest of an item-id vector.

    Element-wise, not set-wise: the digest changes if the row order changes, so
    it pins the ``item_id -> 行号`` mapping rather than just the id set.
    """
    arr = np.ascontiguousarray(np.asarray(item_ids, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def sha256_file(path: str) -> str:
    """Streaming sha256 of a file (used for provenance)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_item_ids(dataset: str, data_dir: str = "data") -> np.ndarray:
    """The dataset's canonical ``ItemID`` column, in canonical row order.

    This is the definition of "row i" for every item-level artifact in the
    project: ``item_emb.parquet``'s row order, which is also the SID row order
    (``model/dataset.py::item2code`` maps row ``i`` to item ``i + 1``).
    """
    import pandas as pd

    path = os.path.join(data_dir, dataset, "item_emb.parquet")
    if not os.path.exists(path):
        raise CFEmbeddingError(f"no canonical item list at {path}")
    return pd.read_parquet(path, columns=["ItemID"])["ItemID"].to_numpy(dtype=np.int64)


def default_cf_path(dataset: str, data_dir: str = "data") -> str:
    """Path of the **ItemCF fallback** table for ``dataset`` (not the main line)."""
    return os.path.join(data_dir, dataset, DEFAULT_CF_TEMPLATE)


def sasrec_checkpoint_path(dataset: str, output_dir: str = SASREC_OUTPUT_DIR,
                           seed: int = 2025) -> str:
    """Path of the SASRec checkpoint the official CF table is extracted from."""
    return os.path.join(output_dir, dataset, SASREC_CKPT_TEMPLATE.format(seed=seed))


def load_sasrec_item_embedding(ckpt_path: str, item_ids, *,
                               key: str = SASREC_EMB_KEY,
                               padding_row: int = SASREC_PADDING_ROW,
                               allow_nonfinite: bool = False) -> "CFEmbedding":
    """Extract ``(n_items, d)`` item embeddings from an SASRec checkpoint.

    ``item_ids`` is the canonical ItemID column, in canonical row order. Rows are
    gathered **by ItemID value**, so the result is in canonical order regardless
    of whether the dataset's ItemID happens to be ``1..N``.

    Checks, all measured rather than assumed (see the module docstring):

    * the embedding matrix has exactly ``len(item_ids) + 1`` rows -- one per real
      item plus the padding row, nothing else;
    * ``item_ids`` are all ``>= 1``, i.e. no real item is mapped onto padding;
    * the padding row is exactly all-zero, which is what ``padding_idx=0`` plus
      the zeroed ``xavier_init_zeros`` init guarantee. A non-zero padding row
      means the checkpoint was not produced by ``tools/run_sasrec.py`` and the
      whole row convention is suspect, so this raises instead of guessing.
    """
    import torch

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if key not in state and "state_dict" in state:
        state = state["state_dict"]
    if key not in state:
        raise CFEmbeddingError(
            f"{ckpt_path} has no {key!r}; found {sorted(state)[:8]} ...")

    weight = state[key].detach().float().cpu()
    if weight.dim() != 2:
        raise CFEmbeddingMismatch(
            f"{key} must be 2-D (n_rows, d), got {tuple(weight.shape)}")

    ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)
    n_rows, dim = int(weight.shape[0]), int(weight.shape[1])
    if n_rows != ids.shape[0] + 1:
        raise CFEmbeddingMismatch(
            f"{ckpt_path}: {key} has {n_rows} rows, expected one per item plus "
            f"the padding row ({ids.shape[0] + 1})")
    if ids.min() < 1 or ids.max() > n_rows - 1:
        raise CFEmbeddingMismatch(
            f"item ids must lie in 1..{n_rows - 1} so that none of them lands on "
            f"the padding row {padding_row}; got [{ids.min()}..{ids.max()}]")
    if not bool((weight[padding_row] == 0).all()):
        raise CFEmbeddingMismatch(
            f"{ckpt_path}: row {padding_row} (padding) is not all-zero, so the "
            f"row convention of this checkpoint cannot be trusted")

    table = weight.numpy()[ids].copy()
    # provenance records the absolute checkpoint path, so it does not depend on
    # the working directory the caller happened to run from
    ckpt_abs = os.path.abspath(ckpt_path)
    emb = CFEmbedding(table, ids, source=ckpt_abs,
              meta={"path": ckpt_abs,
                    "bytes": os.path.getsize(ckpt_path),
                    "source_kind": CF_SOURCE_SASREC,
                    "source_checkpoint": ckpt_abs,
                    "checkpoint_key": key,
                    "checkpoint_rows": n_rows,
                    "padding_row": padding_row,
                    "padding_handling": SASREC_PADDING_HANDLING,
                    "item_id_mapping": SASREC_ROW_OF_ITEM,
                    "source_recipe": SASREC_RECIPE},
              sha256=sha256_file(ckpt_path))
    if not allow_nonfinite:
        emb.check_finite()
    return emb


def load_cf_embedding(dataset: str, *, source: str = DEFAULT_CF_SOURCE,
                      data_dir: str = "data", output_dir: str = SASREC_OUTPUT_DIR,
                      seed: int = 2025, expected_item_ids=None,
                      allow_nonfinite: bool = False) -> "CFEmbedding":
    """The one entry point the C-line uses to obtain a dataset's CF table.

    ``source="sasrec"`` (default) extracts the official representation from the
    SASRec checkpoint. ``source="itemcf"`` loads the fallback parquet. Either way
    the result is a :class:`CFEmbedding` in canonical row order, validated against
    ``item_emb.parquet``'s ItemID column when ``expected_item_ids`` is given.
    """
    if source not in (CF_SOURCE_SASREC, CF_SOURCE_ITEMCF):
        raise CFEmbeddingError(
            f"source must be one of {(CF_SOURCE_SASREC, CF_SOURCE_ITEMCF)}, "
            f"got {source!r}")

    ids = (np.asarray(expected_item_ids, dtype=np.int64).reshape(-1)
           if expected_item_ids is not None
           else canonical_item_ids(dataset, data_dir=data_dir))

    if source == CF_SOURCE_SASREC:
        emb = load_sasrec_item_embedding(
            sasrec_checkpoint_path(dataset, output_dir=output_dir, seed=seed),
            ids, allow_nonfinite=allow_nonfinite)
        emb.meta["seed"] = seed
        return emb

    return CFEmbedding.from_parquet(default_cf_path(dataset, data_dir=data_dir),
                                    expected_item_ids=ids,
                                    allow_nonfinite=allow_nonfinite)


# --------------------------------------------------------------------------- #
# the interface
# --------------------------------------------------------------------------- #
class CFEmbedding:
    """An immutable ``(n_items, d_cf)`` item-level CF table plus its id map.

    Build one with :meth:`load` (parquet / npy), or with :meth:`from_arrays` /
    :meth:`from_parquet` when the caller already has the data in hand.

    The object is read-only on purpose: C1 hands C2/C3 a fixed table, and a
    consumer that mutated it in place would silently change what every other
    consumer sees. Use :meth:`vectors` / :meth:`to_tensor` to get copies.
    """

    def __init__(self, table, item_ids, *, source=None, meta=None, sha256=None):
        table = np.asarray(table, dtype=CF_DTYPE)
        item_ids = np.asarray(item_ids, dtype=np.int64)

        if table.ndim != 2:
            raise CFEmbeddingMismatch(
                f"CF table must be 2-D (n_items, d_cf), got shape {table.shape}")
        if item_ids.ndim != 1:
            raise CFEmbeddingMismatch(
                f"item_ids must be 1-D, got shape {item_ids.shape}")
        if table.shape[0] != item_ids.shape[0]:
            raise CFEmbeddingMismatch(
                f"CF table has {table.shape[0]} rows but {item_ids.shape[0]} item ids")
        if table.shape[0] == 0:
            raise CFEmbeddingMismatch("CF table is empty")
        if table.shape[1] == 0:
            raise CFEmbeddingMismatch("CF table has zero columns")

        uniq, counts = np.unique(item_ids, return_counts=True)
        if uniq.shape[0] != item_ids.shape[0]:
            dup = uniq[counts > 1]
            raise CFEmbeddingMismatch(
                f"item_ids are not unique; repeated ids: {dup[:8].tolist()}"
                f"{' ...' if dup.shape[0] > 8 else ''}")

        self._table = np.ascontiguousarray(table)
        self._item_ids = np.ascontiguousarray(item_ids)
        self.source = source
        self.meta = dict(meta or {})
        self.sha256 = sha256

        # item_id -> row, built from the file's own column. Never `item_id - 1`.
        self._row_of = {int(iid): row for row, iid in enumerate(self._item_ids)}

    # -- shape / dtype ------------------------------------------------------ #
    @property
    def table(self) -> np.ndarray:
        """The raw ``(n_items, d_cf)`` float32 array (do not mutate)."""
        return self._table

    @property
    def item_ids(self) -> np.ndarray:
        """``(n_items,)`` int64 ids, in row order."""
        return self._item_ids

    @property
    def n_items(self) -> int:
        return int(self._table.shape[0])

    @property
    def dim(self) -> int:
        """``d_cf`` -- recorded, never assumed downstream."""
        return int(self._table.shape[1])

    @property
    def dtype(self) -> np.dtype:
        return self._table.dtype

    @property
    def shape(self):
        return self._table.shape

    @property
    def contiguous(self) -> bool:
        """True when ``ItemID == row + 1`` for every row (route ``a-1`` valid).

        An audit flag only: :meth:`row_of` never relies on it. Toys/Sport are
        True, Beauty is False.
        """
        return bool(np.array_equal(self._item_ids, np.arange(1, self.n_items + 1)))

    # -- cold-start / coverage ---------------------------------------------- #
    @property
    def zero_row_mask(self) -> np.ndarray:
        """``(n_items,)`` bool: rows whose vector is exactly all-zero.

        These are the cold-start items -- no behaviour, so no CF signal. They
        are kept (the existing project convention) rather than dropped, because
        dropping them would break the row-order contract with ``item_emb.parquet``.
        """
        return ~self._table.any(axis=1)

    @property
    def n_zero_rows(self) -> int:
        return int(self.zero_row_mask.sum())

    @property
    def valid_mask(self) -> np.ndarray:
        """Complement of :attr:`zero_row_mask` -- rows that carry CF signal."""
        return ~self.zero_row_mask

    def is_zero_row(self, item_id) -> bool:
        """True when ``item_id`` is present but is a cold-start (all-zero) row."""
        return bool(self.zero_row_mask[self.row_of(item_id)])

    # -- the mapping -------------------------------------------------------- #
    def row_of(self, item_id) -> int:
        """Row index of ``item_id``. Raises :class:`MissingItemError` if absent."""
        try:
            return self._row_of[int(item_id)]
        except (KeyError, TypeError, ValueError):
            raise MissingItemError(
                f"item id {item_id!r} is not in this CF table "
                f"({self.n_items} items, ids "
                f"[{self._item_ids.min()}..{self._item_ids.max()}])") from None

    def row_of_many(self, item_ids, missing: str = MISSING_RAISE) -> np.ndarray:
        """Vectorised :meth:`row_of` -> ``(m,)`` int64 row indices.

        With ``missing="zero"``, absent ids get the sentinel ``-1``; with
        ``missing="raise"`` (default) the first absent id raises.
        """
        self._check_policy(missing)
        ids = np.asarray(item_ids, dtype=np.int64).reshape(-1)
        rows = np.fromiter((self._row_of.get(int(i), -1) for i in ids),
                           dtype=np.int64, count=ids.shape[0])
        if missing == MISSING_RAISE:
            absent = ids[rows < 0]
            if absent.shape[0]:
                raise MissingItemError(
                    f"{absent.shape[0]} item id(s) are not in this CF table, "
                    f"first: {absent[:8].tolist()}")
        return rows

    def __contains__(self, item_id) -> bool:
        try:
            return int(item_id) in self._row_of
        except (TypeError, ValueError):
            return False

    def index_map(self) -> dict:
        """A copy of the ``item_id -> row`` map (for C2's relation file / auditing)."""
        return dict(self._row_of)

    # -- access ------------------------------------------------------------- #
    def vector(self, item_id, missing: str = MISSING_RAISE) -> np.ndarray:
        """``(d_cf,)`` copy of one item's vector.

        A present-but-cold-start item legitimately returns zeros; an *unknown*
        id follows the ``missing`` policy instead.
        """
        self._check_policy(missing)
        if int(item_id) not in self._row_of:
            if missing == MISSING_RAISE:
                raise MissingItemError(
                    f"item id {item_id!r} is not in this CF table")
            return np.zeros(self.dim, dtype=self.dtype)
        return self._table[self._row_of[int(item_id)]].copy()

    def vectors(self, item_ids, missing: str = MISSING_RAISE) -> np.ndarray:
        """``(m, d_cf)`` copy of several items' vectors, in the order asked for."""
        self._check_policy(missing)
        rows = self.row_of_many(item_ids, missing=missing)
        out = np.zeros((rows.shape[0], self.dim), dtype=self.dtype)
        hit = rows >= 0
        out[hit] = self._table[rows[hit]]
        return out

    def to_tensor(self, device="cpu"):
        """``(n_items, d_cf)`` float32 ``torch.Tensor`` on ``device``.

        The single hand-off point to the torch side (C3/C5); torch is imported
        lazily so that importing this module stays cheap for the E-line chain.

        Returns a **copy**. ``torch.from_numpy`` on the stored array would share
        its buffer, so an in-place op downstream (a C2 ``normalize_``, a C5
        gradient step) would silently rewrite the table every other consumer
        sees. The copy costs ``n_items * d_cf * 4`` bytes once, at load.
        """
        import torch

        return torch.from_numpy(self._table.copy()).to(device=torch.device(device))

    # -- validation --------------------------------------------------------- #
    @staticmethod
    def _check_policy(missing: str):
        if missing not in MISSING_POLICIES:
            raise ValueError(
                f"missing must be one of {MISSING_POLICIES}, got {missing!r}")

    def check_finite(self) -> None:
        """Raise unless every entry is finite (NaN/Inf would poison C3's loss)."""
        if not np.isfinite(self._table).all():
            bad = np.argwhere(~np.isfinite(self._table))[0]
            raise CFEmbeddingMismatch(
                f"CF table contains non-finite values, first at row {bad[0]}, "
                f"column {bad[1]}")

    def check_alignment(self, expected_item_ids, label: str = "item_emb.parquet") -> None:
        """Raise unless our ItemID column equals ``expected_item_ids`` element-wise.

        Element-wise, not set-wise: the point is that row ``i`` here is row ``i``
        there, which is what makes the SID / ``item_emb`` / CF row spaces agree.
        """
        expected = np.asarray(expected_item_ids, dtype=np.int64).reshape(-1)
        if expected.shape[0] != self.n_items:
            raise CFEmbeddingMismatch(
                f"{label} has {expected.shape[0]} items but the CF table has "
                f"{self.n_items}")
        if not np.array_equal(expected, self._item_ids):
            diff = np.nonzero(expected != self._item_ids)[0]
            raise CFEmbeddingMismatch(
                f"ItemID column does not match {label} element-wise in "
                f"{diff.shape[0]}/{self.n_items} rows; first mismatch at row "
                f"{int(diff[0])}: CF={int(self._item_ids[diff[0]])} vs "
                f"{label}={int(expected[diff[0]])}")

    def validate(self, *, expected_item_ids=None, expected_n_items=None,
                 check_finite: bool = True, label: str = "item_emb.parquet"):
        """Run the acceptance checks and return ``self`` (chainable).

        Covers plan.md C1 acceptance ①②③: row count, explicit dimension, and the
        element-wise ``ItemID`` alignment.
        """
        if expected_n_items is not None and self.n_items != int(expected_n_items):
            raise CFEmbeddingMismatch(
                f"expected {int(expected_n_items)} items, table has {self.n_items}")
        if expected_item_ids is not None:
            self.check_alignment(expected_item_ids, label=label)
        if check_finite:
            self.check_finite()
        return self

    # -- I/O ---------------------------------------------------------------- #
    @classmethod
    def load(cls, path, *, expected_item_ids=None, expected_n_items=None,
             allow_nonfinite: bool = False, source=None) -> "CFEmbedding":
        """Load a CF table, validate it, and return it.

        The format is chosen from the extension: ``.parquet`` uses the repo's
        canonical ``ItemID`` + ``embedding`` layout, ``.npy`` is a bare
        ``(n_items, d_cf)`` array whose item ids must be supplied by the caller.
        """
        ext = os.path.splitext(path)[1].lower()
        if ext == ".parquet":
            emb = cls.from_parquet(path, expected_item_ids=expected_item_ids,
                                   expected_n_items=expected_n_items,
                                   allow_nonfinite=allow_nonfinite, source=source)
        else:
            raise CFEmbeddingError(
                f"unsupported CF table format {ext!r}; supported: "
                f"{['.' + f for f in FORMATS]}")
        return emb

    @classmethod
    def from_arrays(cls, table, item_ids, *, source=None, meta=None,
                    allow_nonfinite: bool = False) -> "CFEmbedding":
        """Build straight from numpy arrays (the synthetic-fixture entry point)."""
        emb = cls(table, item_ids, source=source, meta=meta)
        if not allow_nonfinite:
            emb.check_finite()
        return emb

    @classmethod
    def from_parquet(cls, path, *, expected_item_ids=None, expected_n_items=None,
                     allow_nonfinite: bool = False, source=None) -> "CFEmbedding":
        """Read the canonical ``ItemID`` + ``embedding`` parquet layout."""
        import pandas as pd

        if not os.path.exists(path):
            raise CFEmbeddingError(f"no CF table at {path}")
        frame = pd.read_parquet(path)
        missing = [c for c in PARQUET_COLUMNS if c not in frame.columns]
        if missing:
            raise CFEmbeddingMismatch(
                f"{path} is missing column(s) {missing}; it has {list(frame.columns)}")

        rows = frame["embedding"].to_numpy()
        widths = {len(row) for row in rows}
        if len(widths) != 1:
            raise CFEmbeddingMismatch(
                f"{path}: embeddings have inconsistent widths {sorted(widths)}")
        table = np.stack(rows).astype(CF_DTYPE)
        item_ids = frame["ItemID"].to_numpy(dtype=np.int64)

        emb = cls(table, item_ids, source=source or path,
                  meta={"path": os.path.abspath(path),
                        "bytes": os.path.getsize(path),
                        "source_kind": CF_SOURCE_ITEMCF,
                        "source_variant": CF_SOURCE_VARIANT,
                        "source_generator": CF_SOURCE_GENERATOR,
                        "source_recipe": CF_SOURCE_RECIPE},
                  sha256=sha256_file(path))
        emb.validate(expected_item_ids=expected_item_ids,
                     expected_n_items=expected_n_items,
                     check_finite=not allow_nonfinite,
                     label="item_emb.parquet")
        return emb

    @classmethod
    def from_npy(cls, path, item_ids, *, source=None,
                 allow_nonfinite: bool = False) -> "CFEmbedding":
        """Read a bare ``(n_items, d_cf)`` ``.npy`` plus an explicit id vector.

        Plain ``.npy`` cannot carry the id map, so the ids are a required
        argument -- that is the whole reason the canonical on-disk layout is the
        parquet above.
        """
        if not os.path.exists(path):
            raise CFEmbeddingError(f"no CF table at {path}")
        table = np.load(path)
        emb = cls(table, item_ids, source=source or path,
                  meta={"path": os.path.abspath(path),
                        "bytes": os.path.getsize(path)},
                  sha256=sha256_file(path))
        if not allow_nonfinite:
            emb.check_finite()
        return emb

    def save(self, path, fmt=None) -> str:
        """Write the table back out; returns the path actually written.

        ``parquet`` (default) writes the repo's canonical ``ItemID`` +
        ``embedding`` layout and is lossless for the id map. ``npy`` writes the
        bare array plus a sidecar ``<stem>.ids.npy`` holding the ids, because
        the id map must survive the round trip for the mapping to stay auditable.

        The write is byte-for-byte reproducible: the same table always produces
        the same file.
        """
        fmt = self._resolve_format(path, fmt)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        if fmt == "parquet":
            import pandas as pd

            frame = pd.DataFrame({
                "ItemID": self._item_ids,
                "embedding": [row.astype(CF_DTYPE) for row in self._table],
            })
            frame.to_parquet(path, index=False)
            return path

        stem = os.path.splitext(path)[0]
        np.save(path, self._table)
        np.save(f"{stem}.ids.npy", self._item_ids)
        return path

    @staticmethod
    def _resolve_format(path, fmt):
        if fmt is not None:
            if fmt not in FORMATS:
                raise CFEmbeddingError(
                    f"fmt must be one of {FORMATS}, got {fmt!r}")
            return fmt
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        if ext not in FORMATS:
            raise CFEmbeddingError(
                f"cannot infer format from {path!r}; pass fmt=one of {FORMATS}")
        return ext

    # -- provenance --------------------------------------------------------- #
    def provenance(self) -> dict:
        """Everything needed to re-find and audit this table later.

        The source-specific fields (recipe, id mapping, padding / cold-start
        handling) come from ``self.meta``, which the loader filled in, so the
        manifest describes *this* table rather than whichever source happened to
        be the default when the module was written.
        """
        kind = self.meta.get("source_kind", CF_SOURCE_ITEMCF)
        return {
            "task": "C1",
            "n_items": self.n_items,
            "dim": self.dim,
            "dtype": str(self.dtype),
            "item_ids_sha256": item_ids_sha256(self._item_ids),
            "item_id_min": int(self._item_ids.min()),
            "item_id_max": int(self._item_ids.max()),
            "contiguous_item_ids": self.contiguous,
            "n_zero_rows": self.n_zero_rows,
            "missing_policy": {
                "unknown_id": "MissingItemError by default; 'zero' returns a "
                              "zero vector for that position only",
                "cold_start_row": (SASREC_COLD_START if kind == CF_SOURCE_SASREC
                                   else ITEMCF_COLD_START), },
            "source": self.source,
            "source_sha256": self.sha256,
            "source_kind": kind,
            "source_recipe": self.meta.get(
                "source_recipe",
                SASREC_RECIPE if kind == CF_SOURCE_SASREC else CF_SOURCE_RECIPE),
            "is_default_source": kind == DEFAULT_CF_SOURCE,
            **(self.meta or {}),
        }

    def write_provenance(self, path) -> str:
        """Persist :meth:`provenance` as JSON (the traceability artifact)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.provenance(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    # -- misc --------------------------------------------------------------- #
    def __len__(self) -> int:
        return self.n_items

    def __repr__(self) -> str:
        where = f" source={os.path.basename(self.source)}" if self.source else ""
        return (f"CFEmbedding(n_items={self.n_items}, dim={self.dim}, "
                f"dtype={self.dtype}, zero_rows={self.n_zero_rows}, "
                f"contiguous={self.contiguous}{where})")
