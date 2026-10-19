"""SIA loss -- Task E4 of the ETEGRec -> TIGER port.

Aligns the two 3-level codebook distributions produced by E3:

    sequence side:  encoder hidden -> SIAAdapter -> (B, 32) -> rq -> 3 x (B, 256)
    target side:    item content embedding -> RQVAE.encoder -> (B, 32) -> rq -> 3 x (B, 256)

The loss is the *symmetric* KL divergence of the two softmax distributions,
averaged over the three RQ levels (frozen definition, ``integration/plan.md``
E4). Nothing else is applied: no extra normalisation, no projection, no
stop-gradient, no warmup, no rescaled query, no second codebook.

``SIAMount`` is the wiring object the Recommender (T5) training loop calls. It
holds the adapter, the target-side embedding table and ``lambda_sia``; when
``lambda_sia == 0`` it is *disabled* and the caller must not run it at all, so
that Clean TIGER's step is untouched bit for bit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sia_codes import SIACodeExtractor

# TIGER's item->code offsets are `c + i * codebook_size + 1` (model/dataset.py
# `item2code`); the first three columns are the RQ levels, the fourth is the
# collision token. The collision token identifies the item but is never a
# quantizer level, so it only participates in the SID -> item lookup below.
CODEBOOK_SIZE = 256
NUM_LEVELS = 3


def _as_level_tuple(logits):
    """Accept either (B, L, n_e) or a tuple/list of L x (B, n_e) blocks."""
    if isinstance(logits, (tuple, list)):
        return tuple(logits)
    if logits.dim() != 3:
        raise ValueError(f"expected (B, levels, n_e) or a tuple of blocks, got {tuple(logits.shape)}")
    return tuple(logits[:, level] for level in range(logits.shape[1]))


def symmetric_kl(seq_logits: torch.Tensor, tgt_logits: torch.Tensor,
                 temperature: float = 1.0) -> torch.Tensor:
    """Symmetric KL between two logit blocks, per batch element.

    ``KL(p || q) + KL(q || p)`` with ``p = softmax(seq_logits / T)`` and
    ``q = softmax(tgt_logits / T)``; returns shape ``(B,)``. Zero iff the two
    distributions are identical.
    """
    log_p = F.log_softmax(seq_logits / temperature, dim=-1)
    log_q = F.log_softmax(tgt_logits / temperature, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    diff = log_p - log_q
    return (p * diff).sum(dim=-1) - (q * diff).sum(dim=-1)


class SIALoss(nn.Module):
    """Per-level symmetric KL, reduced over the RQ levels and the batch.

    Args:
        temperature: divides the logits before the softmax (T in the frozen
            definition). Default 1.0 keeps the raw ``squared_L2`` logits.
        level_reduction: ``"mean"`` (default) or ``"sum"`` over the 3 levels.
    """

    def __init__(self, temperature: float = 1.0, level_reduction: str = "mean"):
        super().__init__()
        if level_reduction not in ("mean", "sum"):
            raise ValueError(f"level_reduction must be 'mean' or 'sum', got {level_reduction!r}")
        self.temperature = temperature
        self.level_reduction = level_reduction

    def forward(self, seq_logits, tgt_logits) -> torch.Tensor:
        """3 x (B, 256) on each side -> scalar."""
        seq_levels = _as_level_tuple(seq_logits)
        tgt_levels = _as_level_tuple(tgt_logits)
        if len(seq_levels) != len(tgt_levels):
            raise ValueError(
                f"both sides must have the same number of RQ levels, "
                f"got {len(seq_levels)} and {len(tgt_levels)}")
        per_level = [symmetric_kl(s, t, self.temperature).mean()
                     for s, t in zip(seq_levels, tgt_levels)]
        stacked = torch.stack(per_level)
        return stacked.mean() if self.level_reduction == "mean" else stacked.sum()


class SIAMount(nn.Module):
    """Bundle the SIA path so the Recommender loop can call it in one line.

    Holds the E2 adapter, the E3 extractor and the target-side item content
    embedding table. ``lambda_sia`` defaults to 0, in which case ``enabled`` is
    False and the training loop must skip the whole path.

    The target item embeddings are the *same* table the RQ-VAE tokenizer was
    trained on (``item_emb.parquet``); they are registered as a plain buffer,
    never trained.

    Args:
        adapter: E2 ``SIAAdapter``.
        rqvae: TIGER ``RQVAE`` whose ``encoder``/``rq`` are shared by both sides.
        target_embeddings: (n_items, D_content) item content embeddings, row order
            matching the SID file (row i <-> item i+1 <-> SID row i).
        sid_tokens: (n_items, 4) int64 SID tokens in TIGER's offset layout, used
            to route a batch of target SID labels back to embedding rows.
        lambda_sia: weight; 0 disables SIA entirely.
        temperature: forwarded to ``SIALoss``.
    """

    def __init__(self, adapter: nn.Module, rqvae: nn.Module,
                 target_embeddings: torch.Tensor, sid_tokens: torch.Tensor,
                 lambda_sia: float = 0.0, temperature: float = 1.0,
                 level_reduction: str = "mean"):
        super().__init__()
        self.adapter = adapter
        self.rqvae = rqvae
        self.lambda_sia = float(lambda_sia)
        self.loss_fn = SIALoss(temperature=temperature, level_reduction=level_reduction)
        self.extractor = SIACodeExtractor(adapter, rqvae)
        self.register_buffer("target_embeddings", target_embeddings.detach().clone(), persistent=False)

        # the SID file stores raw codes; the batch carries TIGER's offset layout
        # (`model/dataset.py: item2code` -> code + level * codebook_size + 1)
        sid_tokens = sid_tokens.detach().cpu()
        level_offset = torch.arange(sid_tokens.shape[1]) * CODEBOOK_SIZE + 1
        offsets = sid_tokens + level_offset
        lookup = {}
        for row in range(sid_tokens.shape[0]):
            key = tuple(int(t) for t in offsets[row, :NUM_LEVELS + 1])
            lookup.setdefault(key, row)
        # dict -> index tensors so a batch is one gather instead of a python loop
        keys = torch.tensor(list(lookup.keys()), dtype=torch.long)
        rows = torch.tensor(list(lookup.values()), dtype=torch.long)
        self.register_buffer("_sid_keys", keys, persistent=False)
        self.register_buffer("_sid_rows", rows, persistent=False)
        self._lookup = lookup

    @property
    def enabled(self) -> bool:
        """SIA only runs when the weight is strictly positive."""
        return self.lambda_sia > 0.0

    def rows_for_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """(B, 4) target SID tokens -> (B,) rows into ``target_embeddings``."""
        missing = []
        rows = []
        for token_row in labels.detach().cpu():
            key = tuple(int(t) for t in token_row[:NUM_LEVELS + 1])
            row = self._lookup.get(key)
            if row is None:
                missing.append(key)
            rows.append(-1 if row is None else row)
        if missing:
            raise KeyError(
                f"{len(missing)} target SID(s) are not in the item table, "
                f"e.g. {missing[0]}; SIA cannot align an unknown target")
        return torch.as_tensor(rows, dtype=torch.long, device=labels.device)

    def target_side_logits(self, labels: torch.Tensor):
        """(B, 4) target SID tokens -> 3 x (B, 256) logits through the encoder."""
        rows = self.rows_for_labels(labels)
        return self.extractor.target_logits(self.target_embeddings[rows])

    def forward(self, encoder_hidden: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """Return ``lambda_sia * L_SIA`` (a scalar tensor).

        Raises if called while disabled -- the caller is expected to guard with
        ``enabled`` instead of relying on a zero return value.
        """
        if not self.enabled:
            raise RuntimeError("SIAMount.forward called with lambda_sia = 0; guard with `enabled`")
        seq_logits = self.extractor.sequence_logits(encoder_hidden, attention_mask)
        tgt_logits = self.target_side_logits(labels)
        return self.lambda_sia * self.loss_fn(seq_logits, tgt_logits)
