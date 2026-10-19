"""C2 -- Collaborative relation (CSG4Rec -> TIGER port).

Contract: ``integration/plan.md`` C2 table.

    | 来源机制 | CSG4Rec 的协同相似度 / kNN 关系                                   |
    | 输入     | C1 的 ``(n_items, d_cf)``                                       |
    | 输出     | 关系对文件（``(N, 2)`` int）+ 关系统计（边数、度数分布、对称性）      |
    | 新增     | cosine 相似度 → top-k kNN → 关系对落盘 + 诊断产物                  |
    | 修改     | 无                                                            |

This module is a **numpy leaf**: it consumes a ``(n_items, d_cf)`` table (or a
:class:`tiger_ext.cf_embed.CFEmbedding`) and produces an item-to-item top-k
collaborative relation, nothing else. It builds no representation of its own, adds
no loss, no projection and no normalisation-module, reads no user-side data, and
touches neither the frozen RQ-VAE nor the T5 structure. The Q-space constraint
(C3) and the CSG objective (C4) are deliberately **not** implemented here.

The relation it produces is *not* an input transformation: nothing about it is
concatenated into the RQ-VAE input, and no alignment between the CF table and the
RQ-VAE latent space is performed. It is a standalone side artifact.

What CSG4Rec actually does
--------------------------
The reference builds its collaborative neighbours inside ``ItemDataset``::

    # CSGT/datasets.py:36-38
    nbrs = NearestNeighbors(n_neighbors=min(knn_k+1, self.N), metric='cosine').fit(cf_emb_np)
    _, indices = nbrs.kneighbors(cf_emb_np)
    self.global_knn = indices[:, 1:min(knn_k+1, self.N)]

with ``--knn_k`` defaulting to **20** (``CSGT/main.py:39``). Four things follow,
and each is reproduced or corrected here **explicitly** rather than inherited by
accident:

1. **Similarity = cosine**, computed by querying the table against itself.
   ``sklearn``'s ``metric='cosine'`` is cosine *distance* ``1 - cos``; only the
   ordering is used. Reproduced as a similarity (``cos``), which is the form this
   port wants.
2. **k = 20.** Inherited unchanged from the reference default. ``k`` is *not*
   re-tuned here: tuning it would be a design decision, and this Task only ports
   the mechanism. ``k_eff = min(k, n_items - 1)``.
3. **Directed, not symmetric.** Each row independently selects *its own* top-k.
   Cosine is symmetric but the top-k truncation is not, so the relation is a
   directed graph; the reciprocal fraction is measured and recorded instead of
   symmetrising it (symmetrising would be a design expansion the plan does not ask
   for).
4. **Self-exclusion is done by explicit masking, not by position.** The reference
   drops column 0 of the neighbour matrix, which assumes self is always the
   nearest neighbour. That assumption holds for a well-conditioned table but
   **fails on zero-norm rows**, where the cosine is undefined and the row's own
   ordering is arbitrary. Measured on this repo's ItemCF fallback table
   (128-d, which does contain all-zero cold-start rows): the reference's
   ``[:, 1:]`` leaves **self inside the top-k for 6/24 (Toys) and 11/33
   (Beauty)** rows -- i.e. it produces the self-loops C2 forbids. On the official
   SASRec source there are no zero-norm rows and the two agree exactly
   (0/0/0), which is precisely why the defect is easy to inherit unnoticed.
   This module masks the diagonal explicitly and then *asserts* no self-loop.

Zero-norm rows (policy, stated rather than guessed)
---------------------------------------------------
A row with ``||v|| == 0`` has no direction, so its cosine to anything is
undefined. Policy here: **such a row is excluded both as a source and as a
target** -- it gets no neighbours, and it never appears in anyone else's list.
The alternative convention (treat the cosine as 0) is a claim the data does not
support; excluding it is the choice that cannot silently invent an edge. Affected
positions are marked ``-1`` in the neighbour ids and ``NaN`` in the similarities,
so downstream code must handle them explicitly. Measured: the official SASRec
source has **0** such rows in all three datasets (a never-trained item keeps its
finite random init), so this is a guard that is exercised by the ItemCF fallback,
not a live constraint on the main line.

Row space and the id mapping (same contract as C1)
--------------------------------------------------
Neighbours are stored in **canonical row order**, the order of
``data/<D>/item_emb.parquet``'s ``ItemID`` column, exactly as C1 defines it, so a
row index here means the same thing as a row index there. The artifact always
carries that ``item_ids`` vector alongside, plus its order-sensitive digest, and
:meth:`CollabRelation.load` refuses to load if the caller's canonical ids differ.
Two consequences worth stating plainly:

* ``item_id - 1`` is never used, and a downstream consumer that treats a row index
  as an ItemID is caught by the digest rather than silently mis-mapping. On Beauty
  (4158 of 12101 rows are scrambled) that distinction is load-bearing.
* the ``(M, 2)`` pair file is in **row space**. It mirrors the shape of the
  historical transition-pair file so the existing loading skeleton can read it,
  but it deliberately does **not** reuse that feature's switch or its
  ``args``-driven semantics -- this relation is produced and consumed through its
  own interface only. ``pairs_as_item_ids()`` returns the same edges with the ids
  substituted, for human inspection and for cross-checks.

Engineering gate only -- nothing here says anything about recommendation quality.
"""
import json
import os

import numpy as np

# the digest helper is C1's, imported rather than re-written: the two modules must
# agree bit for bit on what a row space is, and one definition cannot drift from
# itself
from .cf_embed import canonical_item_ids, item_ids_sha256

# --------------------------------------------------------------------------- #
# pinned definition of the relation
# --------------------------------------------------------------------------- #
RELATION_K = 20                     # CSG4Rec `--knn_k` default (CSGT/main.py:39)
RELATION_METRIC = "cosine"
RELATION_DTYPE = np.int32           # row-index storage; -1 marks "no neighbour"
SIMILARITY_DTYPE = np.float32
RELATION_SPACE = "canonical_row_order"
ROW_OF_ITEM = ("row i of the relation is row i of C1's CF table, i.e. row i of "
               "data/<D>/item_emb.parquet's ItemID column; never `item_id - 1`")

SIM_FORMULA = ("cos(a, b) = dot(a, b) / (||a||_2 * ||b||_2), both vectors in "
               "float32; the table is L2-normalised once and the kNN is a blocked "
               "normalised matmul")
SELF_EXCLUSION = ("the diagonal of the similarity block is masked to -inf before "
                  "the top-k, then the result is asserted free of self-loops; the "
                  "reference's positional `indices[:, 1:]` drop is NOT used because "
                  "it keeps self on zero-norm rows")
ZERO_NORM_POLICY = ("a row with ||v|| == 0 has no defined cosine: it selects no "
                    "neighbours (marked -1 / NaN) and is never selected as one")
DIRECTEDNESS = ("directed: every row selects its own top-k, so (i -> j) does not "
                "imply (j -> i); the relation is not symmetrised")
TIE_BREAK = "similarity descending, then row index ascending (fully deterministic)"
PAIR_FILE_NOTE = ("(M, 2) int row-index edges; same array shape as the historical "
                  "transition-pair file, but produced and consumed only through "
                  "this interface")
REFERENCE_IMPLEMENTATION = {
    "file": "reference_repos/CSG4Rec_original/CSGT/datasets.py",
    "lines": "36-38",
    "code": "NearestNeighbors(n_neighbors=min(knn_k+1, N), metric='cosine'); "
            "global_knn = indices[:, 1:knn_k+1]",
    "k_default": 20,
    "metric": "cosine distance (1 - cos), ordering only",
    "self_exclusion": "positional drop of column 0",
    "k_setting": "CSGT/main.py:39 --knn_k",
}
PORT_DIFFERENCES = [
    "similarity is computed in numpy (blocked normalised matmul), not via sklearn "
    "-- no new dependency, and it makes the self-exclusion explicit",
    "self is excluded by masking the diagonal, not by dropping column 0: the "
    "positional drop leaves self in the top-k on zero-norm rows (measured on the "
    "ItemCF fallback: 6/24 Toys, 11/33 Beauty)",
    "zero-norm rows are excluded as source and target and marked -1 / NaN; the "
    "reference leaves their ordering to whatever the solver does",
    "edges are persisted with an explicit ItemID companion and an id digest; the "
    "reference keeps them in an in-memory array only",
    "the relation is a standalone artifact with its own loader -- the historical "
    "transition feature's switch and args semantics are not reused",
]

ARTIFACT_DIRNAME = os.path.join("integration", "artifacts", "collab_relation")
DEFAULT_BLOCK = 512
SUFFIX = {"neighbors": ".neighbors.npy", "similarity": ".similarity.npy",
          "pairs": ".pairs.npy", "pairs_itemid": ".pairs_itemid.npy",
          "item_ids": ".item_ids.npy", "meta": ".json"}


class CollabRelationError(Exception):
    """Base class for C2 relation errors."""


def _mean_or_nan(values):
    """Mean of the finite entries, or NaN when there are none (no warning)."""
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()) if values.size else float("nan")


class CollabRelationMismatch(CollabRelationError):
    """The relation does not match the table / ids it is supposed to belong to."""


# --------------------------------------------------------------------------- #
# cosine + top-k
# --------------------------------------------------------------------------- #
def l2_normalize(table):
    """``(unit_rows, norms)`` for a ``(n, d)`` table; zero-norm rows stay zero.

    The only place the word "normalise" appears in this module: it is part of the
    *definition of cosine*, not a normalisation layer applied to anyone's input.
    """
    table = np.ascontiguousarray(table, dtype=SIMILARITY_DTYPE)
    if table.ndim != 2:
        raise CollabRelationMismatch(f"table must be 2-D, got shape {table.shape}")
    if not np.isfinite(table).all():
        raise CollabRelationMismatch("table contains non-finite values")
    norms = np.linalg.norm(table.astype(np.float64), axis=1)
    safe = norms.copy()
    safe[safe == 0] = 1.0
    return (table / safe[:, None]).astype(SIMILARITY_DTYPE), norms


def topk_cosine_neighbors(table, k=RELATION_K, block_size=DEFAULT_BLOCK):
    """Top-k cosine neighbours of every row of ``table``, self excluded.

    Returns ``(neighbors, similarity)``, both ``(n, k_eff)`` with
    ``k_eff = min(k, n - 1)``:

    * ``neighbors`` -- ``int32`` row indices, ``-1`` where a row has no neighbour
      (a zero-norm source row, or fewer than ``k_eff`` admissible targets);
    * ``similarity`` -- ``float32`` cosine, ``NaN`` in those same positions.

    Zero-norm rows are excluded as targets as well as sources. The order within
    each row is similarity descending then index ascending, so the result is
    reproducible bit for bit.
    """
    unit, norms = l2_normalize(table)
    n = unit.shape[0]
    if n < 2:
        raise CollabRelationMismatch(
            f"a top-k relation needs at least 2 rows, got {n}")
    if k < 1:
        raise CollabRelationMismatch(f"k must be >= 1, got {k}")
    k_eff = min(int(k), n - 1)
    zero_norm = norms == 0

    neighbors = np.full((n, k_eff), -1, dtype=RELATION_DTYPE)
    similarity = np.full((n, k_eff), np.nan, dtype=SIMILARITY_DTYPE)

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        sims = unit[start:stop] @ unit.T                 # (b, n) float32
        sims[:, zero_norm] = -np.inf                     # no zero-norm target
        sims[zero_norm[start:stop]] = -np.inf            # no zero-norm source
        sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf  # no self

        # a stable argsort of -sims is exactly "similarity descending, then row
        # index ascending": equal similarities keep their original order. Using
        # argpartition instead would be faster but would let an arbitrary tied
        # element through at the k-th position, which contradicts the tie-break
        # the provenance advertises.
        order = np.argsort(-sims, axis=1, kind="stable")[:, :k_eff]
        cand_sim = np.take_along_axis(sims, order, axis=1)

        valid = np.isfinite(cand_sim)
        neighbors[start:stop] = np.where(valid, order, -1)
        similarity[start:stop] = np.where(valid, cand_sim, np.nan)
    return neighbors, similarity


def _item_ids_of(table_or_embedding, item_ids):
    """Resolve ``(table, item_ids, source, meta)`` from either accepted input."""
    if item_ids is None and hasattr(table_or_embedding, "table"):
        emb = table_or_embedding
        return (np.asarray(emb.table), np.asarray(emb.item_ids),
                getattr(emb, "source", None),
                # C1's provenance, carried forward verbatim: the relation must be
                # traceable to the exact CF artifact it was computed from
                {"cf_source_kind": emb.meta.get("source_kind"),
                 "cf_source_sha256": emb.sha256})
    table = np.asarray(table_or_embedding)
    if item_ids is None:
        raise CollabRelationMismatch(
            "item_ids are required when passing a bare table: the row space has "
            "to be pinned to C1's canonical order explicitly")
    return table, np.asarray(item_ids, dtype=np.int64), None, {}


# --------------------------------------------------------------------------- #
# the relation
# --------------------------------------------------------------------------- #
class CollabRelation:
    """A directed item-to-item top-k relation over C1's canonical row space.

    Build one with :func:`build_relation` / :func:`build_relation_for_dataset`.
    The object is read-only: C2 hands C3 a fixed relation, and a consumer that
    edited it in place would change what every other consumer sees.
    """

    def __init__(self, neighbors, similarity, item_ids, *, k=None, source=None,
                 meta=None):
        self._neighbors = np.ascontiguousarray(neighbors, dtype=RELATION_DTYPE)
        self._similarity = np.ascontiguousarray(similarity, dtype=SIMILARITY_DTYPE)
        self._item_ids = np.ascontiguousarray(item_ids, dtype=np.int64)
        if self._neighbors.ndim != 2 or self._similarity.shape != self._neighbors.shape:
            raise CollabRelationMismatch(
                f"neighbors/similarity must be equal-shape 2-D arrays, got "
                f"{self._neighbors.shape} and {self._similarity.shape}")
        if self._neighbors.shape[0] != self._item_ids.shape[0]:
            raise CollabRelationMismatch(
                f"relation has {self._neighbors.shape[0]} rows but "
                f"{self._item_ids.shape[0]} item ids")
        self.k = int(k if k is not None else self._neighbors.shape[1])
        self.source = source
        self.meta = dict(meta or {})
        self.validate()

    # -- core accessors ----------------------------------------------------- #
    @property
    def neighbors(self):
        """``(n_items, k_eff)`` int32 row indices; ``-1`` = no neighbour."""
        return self._neighbors

    @property
    def similarity(self):
        """``(n_items, k_eff)`` float32 cosine; ``NaN`` where ``-1``."""
        return self._similarity

    @property
    def item_ids(self):
        """``(n_items,)`` canonical ItemID per row (the row-space contract)."""
        return self._item_ids

    @property
    def n_items(self):
        return int(self._neighbors.shape[0])

    @property
    def k_eff(self):
        return int(self._neighbors.shape[1])

    @property
    def item_ids_sha256(self):
        return item_ids_sha256(self._item_ids)

    @property
    def pairs(self):
        """``(M, 2)`` int32 row-index edges ``(source_row, target_row)``.

        Rows without neighbours contribute nothing, so ``M == n_items * k_eff``
        exactly when every row has a full neighbour list.
        """
        src, dst = np.nonzero(self._neighbors >= 0)
        return np.stack([src.astype(RELATION_DTYPE),
                         self._neighbors[src, dst]], axis=1)

    def pairs_as_item_ids(self):
        """The same edges with ItemIDs substituted for row indices."""
        pairs = self.pairs
        return np.stack([self._item_ids[pairs[:, 0]],
                         self._item_ids[pairs[:, 1]]], axis=1).astype(np.int64)

    @property
    def n_edges(self):
        return int(self._neighbors.size - (self._neighbors < 0).sum())

    @property
    def out_degree(self):
        return (self._neighbors >= 0).sum(axis=1).astype(RELATION_DTYPE)

    @property
    def has_neighbors(self):
        return self.out_degree > 0

    # -- lookups ------------------------------------------------------------ #
    def row_of(self, item_id):
        """Row index of an ItemID; raises rather than guessing (C1's rule)."""
        item_id = int(item_id)
        hits = np.nonzero(self._item_ids == item_id)[0]
        if hits.size == 0:
            raise CollabRelationMismatch(
                f"item id {item_id} is not in this relation's row space")
        return int(hits[0])

    def neighbors_of(self, item_id, drop_missing=False):
        """Neighbour **row indices** of ``item_id`` (a copy)."""
        rows = self._neighbors[self.row_of(item_id)]
        return rows[rows >= 0].copy() if drop_missing else rows.copy()

    def neighbor_item_ids_of(self, item_id):
        """Neighbour **ItemIDs** of ``item_id`` (a copy)."""
        return self._item_ids[self.neighbors_of(item_id, drop_missing=True)]

    def similarity_of(self, item_id, neighbor_item_id):
        """Cosine between two ItemIDs, or ``NaN`` if they are not related."""
        row = self.row_of(item_id)
        target = self.row_of(neighbor_item_id)
        where = np.nonzero(self._neighbors[row] == target)[0]
        return float(self._similarity[row, where[0]]) if where.size else float("nan")

    # -- validation --------------------------------------------------------- #
    def validate(self):
        """Self-contained structural checks; raises on the first failure."""
        n, k_eff = self.n_items, self.k_eff
        uniq = np.unique(self._item_ids)
        if uniq.shape[0] != n:
            raise CollabRelationMismatch("relation item_ids are not unique")
        if self._neighbors.min() < -1 or self._neighbors.max() > n - 1:
            raise CollabRelationMismatch(
                f"neighbour rows must lie in 0..{n - 1} (or be -1), got "
                f"[{self._neighbors.min()}..{self._neighbors.max()}]")
        rows = np.arange(n)[:, None]
        if bool((self._neighbors == rows).any()):
            bad = np.nonzero((self._neighbors == rows).any(axis=1))[0][:5]
            raise CollabRelationMismatch(
                f"relation contains self-loops (rows {bad.tolist()}): C2 forbids them")
        missing = self._neighbors < 0
        if not bool((missing == np.isnan(self._similarity)).all()):
            raise CollabRelationMismatch(
                "-1 neighbour slots and NaN similarities disagree; a slot is "
                "either related (index + finite cosine) or absent (both sentinels)")
        sims = self._similarity
        real = sims[~missing]
        if real.size and (not np.isfinite(real).all() or (real < -1.0001).any()
                          or (real > 1.0001).any()):
            raise CollabRelationMismatch(
                "cosine similarities must be finite and within [-1, 1]")
        if k_eff > n - 1:
            raise CollabRelationMismatch(
                f"k_eff={k_eff} exceeds n_items-1={n - 1}: self-exclusion makes "
                f"that unachievable")
        # every row must read "rank 0 .. rank j-1 finite, descending, then the
        # holes", i.e. real neighbours first as a contiguous descending block.
        holes_before_real = np.diff(missing.astype(np.int8), axis=1) < 0
        both_real = np.isfinite(sims[:, :-1]) & np.isfinite(sims[:, 1:])
        not_descending = np.zeros_like(both_real)
        not_descending[both_real] = (
            (sims[:, :-1] - sims[:, 1:])[both_real] < -1e-6)
        breaks = holes_before_real | not_descending
        if bool(breaks.any()):
            row, col = np.argwhere(breaks)[0]
            raise CollabRelationMismatch(
                f"row {row}'s neighbour list breaks the contract between ranks "
                f"{col} and {col + 1}: real neighbours must come first, in "
                f"descending similarity order, with the empty slots last")
        return self

    def check_against(self, table_or_embedding, item_ids=None):
        """The relation belongs to *this* table: ids and row space must line up."""
        _, ids, _, _ = _item_ids_of(table_or_embedding, item_ids)
        if ids.shape[0] != self.n_items:
            raise CollabRelationMismatch(
                f"relation covers {self.n_items} items but the table has {ids.shape[0]}")
        if not np.array_equal(ids, self._item_ids):
            n_bad = int((ids != self._item_ids).sum())
            raise CollabRelationMismatch(
                f"the relation's row space differs from the table's ItemID order "
                f"in {n_bad} of {ids.shape[0]} rows; refusing to pair them up")
        return self

    # -- statistics --------------------------------------------------------- #
    def stats(self):
        """Edge count, degree distribution, symmetry and similarity profile.

        ``n_duplicate_edges`` is always 0 *by construction* (the relation is an
        ``(n, k_eff)`` matrix, so a row cannot name the same target twice in the
        same slot). It is computed rather than hard-coded so that the invariant is
        checked as the export is derived, not merely asserted.
        """
        deg = self.out_degree
        sims = self._similarity
        finite = np.isfinite(sims)
        pairs = self.pairs
        edges = {(int(a), int(b)) for a, b in pairs}
        reciprocal = sum(1 for a, b in edges if (b, a) in edges)
        dup = len(pairs) - len(edges)
        real = sims[finite]
        per_rank = [_mean_or_nan(sims[:, r]) for r in range(self.k_eff)]
        return {
            "n_items": self.n_items,
            "k": self.k,
            "k_eff": self.k_eff,
            "n_edges": int(len(pairs)),
            "n_items_with_neighbors": int((deg > 0).sum()),
            "n_items_without_neighbors": int((deg == 0).sum()),
            "out_degree_min": int(deg.min()),
            "out_degree_max": int(deg.max()),
            "out_degree_mean": float(deg.mean()),
            "degree_histogram": {str(int(d)): int(c) for d, c in
                                 zip(*np.unique(deg, return_counts=True))},
            "n_self_loops": 0,
            "n_duplicate_edges": int(dup),
            "n_edges_without_reverse": int(len(edges) - reciprocal),
            "reciprocal_fraction": float(reciprocal / len(edges)) if edges else 0.0,
            "similarity_mean": _mean_or_nan(real),
            "similarity_min": float(real.min()) if real.size else float("nan"),
            "similarity_max": float(real.max()) if real.size else float("nan"),
            "similarity_mean_by_rank": per_rank,
            "n_nonpositive_similarity_edges": int(((sims <= 0) & finite).sum()),
            "n_nan_slots": int((~finite).sum()),
        }

    # -- provenance --------------------------------------------------------- #
    def provenance(self):
        """Everything needed to re-find and audit this relation later."""
        return {
            "task": "C2",
            "method": RELATION_METRIC,
            "similarity_definition": SIM_FORMULA,
            "k": self.k,
            "k_eff": self.k_eff,
            "k_basis": "CSG4Rec `--knn_k` default (CSGT/main.py:39); not re-tuned here",
            "directed": True,
            "directedness": DIRECTEDNESS,
            "self_exclusion": SELF_EXCLUSION,
            "zero_norm_policy": ZERO_NORM_POLICY,
            "tie_break": TIE_BREAK,
            "row_space": RELATION_SPACE,
            "item_id_mapping": ROW_OF_ITEM,
            "n_items": self.n_items,
            "item_ids_sha256": self.item_ids_sha256,
            "item_id_min": int(self._item_ids.min()),
            "item_id_max": int(self._item_ids.max()),
            "pair_file": PAIR_FILE_NOTE,
            "csg4rec_reference": REFERENCE_IMPLEMENTATION,
            "differences_from_reference": list(PORT_DIFFERENCES),
            "cf_source": self.source,
            **(self.meta or {}),
            "stats": self.stats(),
        }

    def write_provenance(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.provenance(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    # -- persistence -------------------------------------------------------- #
    def save(self, prefix):
        """Write the relation under ``prefix``; returns the written paths.

        Layout (all byte-reproducible):

        ``<prefix>.neighbors.npy``     ``(n, k_eff)`` int32 row indices, ``-1`` holes
        ``<prefix>.similarity.npy``    ``(n, k_eff)`` float32 cosines, ``NaN`` holes
        ``<prefix>.pairs.npy``         ``(M, 2)`` int32 row-index edges
        ``<prefix>.pairs_itemid.npy``  ``(M, 2)`` int64 the same edges as ItemIDs
        ``<prefix>.item_ids.npy``      ``(n,)`` int64 canonical ItemIDs
        ``<prefix>.json``              provenance + statistics
        """
        directory = os.path.dirname(os.path.abspath(prefix))
        os.makedirs(directory, exist_ok=True)
        written = {}
        for key, arr in (("neighbors", self._neighbors),
                         ("similarity", self._similarity),
                         ("pairs", self.pairs),
                         ("pairs_itemid", self.pairs_as_item_ids()),
                         ("item_ids", self._item_ids)):
            path = prefix + SUFFIX[key]
            np.save(path, arr)
            written[key] = path
        written["meta"] = self.write_provenance(prefix + SUFFIX["meta"])
        return written

    @classmethod
    def load(cls, prefix, *, expected_item_ids=None, expected_k=None):
        """Load a saved relation, refusing to if it does not match expectations.

        ``expected_item_ids`` is the canonical ItemID vector the caller believes
        this row space means. If it is given and disagrees, this raises: a
        relation loaded onto the wrong row space would silently pair the wrong
        items, which is exactly the failure mode C1's mapping contract exists to
        prevent.
        """
        arrays = {}
        for key in ("neighbors", "similarity", "pairs", "pairs_itemid", "item_ids"):
            path = prefix + SUFFIX[key]
            if not os.path.exists(path):
                raise CollabRelationMismatch(f"missing relation file: {path}")
            arrays[key] = np.load(path)
        meta_path = prefix + SUFFIX["meta"]
        if not os.path.exists(meta_path):
            raise CollabRelationMismatch(f"missing relation provenance: {meta_path}")
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)

        if expected_item_ids is not None:
            expected = np.asarray(expected_item_ids, dtype=np.int64).reshape(-1)
            if not np.array_equal(expected, arrays["item_ids"]):
                raise CollabRelationMismatch(
                    "the saved relation belongs to a different row space than the "
                    "expected canonical ItemIDs; refusing to load")
        if expected_k is not None and int(meta["k"]) != int(expected_k):
            raise CollabRelationMismatch(
                f"relation was built with k={meta['k']}, expected k={expected_k}")

        rel = cls(arrays["neighbors"], arrays["similarity"], arrays["item_ids"],
                  k=meta["k"], source=meta.get("cf_source"),
                  meta={k: v for k, v in meta.items()
                        if k not in ("stats", "task", "method", "k")})
        if not np.array_equal(rel.pairs, arrays["pairs"]):
            raise CollabRelationMismatch(
                "the stored pair file does not agree with the stored neighbour "
                "matrix (the (M, 2) export and the (n, k) matrix disagree)")
        if not np.array_equal(rel.pairs_as_item_ids(), arrays["pairs_itemid"]):
            raise CollabRelationMismatch(
                "the stored ItemID pair file does not match the row-index pair "
                "file mapped through the stored item ids")
        return rel

    # -- misc --------------------------------------------------------------- #
    def __len__(self):
        return self.n_items

    def __repr__(self):
        return (f"CollabRelation(n_items={self.n_items}, k={self.k}, "
                f"k_eff={self.k_eff}, edges={self.n_edges}, "
                f"isolated={int((self.out_degree == 0).sum())})")


def build_relation(table_or_embedding, item_ids=None, *, k=RELATION_K,
                   block_size=DEFAULT_BLOCK, source=None, meta=None):
    """Build the C2 relation from a CF table (or a C1 ``CFEmbedding``)."""
    table, ids, src, extra = _item_ids_of(table_or_embedding, item_ids)
    if table.shape[0] != ids.shape[0]:
        raise CollabRelationMismatch(
            f"table has {table.shape[0]} rows but {ids.shape[0]} item ids")
    neighbors, similarity = topk_cosine_neighbors(table, k=k, block_size=block_size)
    merged = dict(extra)
    merged.update(meta or {})
    # k is the *requested* k; k_eff is what the table can deliver (<= k), and the
    # two are reported separately so a shrunk neighbourhood is never silent
    rel = CollabRelation(neighbors, similarity, ids, k=int(k),
                         source=source or src, meta=merged)
    # the only admissible reason to be dropped is an undefined cosine, and the
    # only admissible consequence of an undefined cosine is being dropped: both
    # directions are checked so a silently isolated row cannot slip through
    zero_norm = np.linalg.norm(table.astype(np.float64), axis=1) == 0
    wrong = np.nonzero(rel.has_neighbors == zero_norm)[0]
    if wrong.size:
        raise CollabRelationMismatch(
            f"rows {wrong[:5].tolist()} are isolated (or not) without the "
            f"corresponding zero-norm vector; isolation must mean exactly "
            f"'this row's cosine is undefined'")
    return rel


def build_relation_for_dataset(dataset, *, k=RELATION_K, source=None,
                               data_dir="data", **kwargs):
    """Convenience: C1's CF table for ``dataset`` -> its C2 relation.

    Loads the table through C1's single entry point (so the CF source ruling is
    applied in one place) and pins the row space to ``item_emb.parquet``.
    """
    from .cf_embed import load_cf_embedding

    ids = canonical_item_ids(dataset, data_dir=data_dir)
    emb = load_cf_embedding(dataset, expected_item_ids=ids, **kwargs)
    rel = build_relation(emb, k=k, source=emb.source)
    rel.meta["dataset"] = dataset
    return rel


def relation_prefix(dataset, k=RELATION_K, artifact_dir=ARTIFACT_DIRNAME):
    """Where :func:`build_relation_for_dataset` output lives for a dataset."""
    return os.path.join(artifact_dir, f"{dataset}_k{int(k)}")
