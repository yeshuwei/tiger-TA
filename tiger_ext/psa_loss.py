"""PSA loss -- Task E6 of the ETEGRec -> TIGER port.

Aligns the PSA representation with the target item's *tokenizer reconstruction*:

    sequence side:  T5 decoder hidden (last layer, position 0) -> PSAAdapter -> (B, D_target)
    target side:    item content embedding -> RQVAE.forward -> out = decoder(x_q) -> (B, D_target)

``plan.md`` E6 fixes the shape of this ("symmetric contrastive loss, in-batch
negatives", target = the RQ-VAE reconstruction -- explicitly *not* the raw
content embedding and *not* an RQ latent), and fixes the target side to
``RQVAE.forward``'s first return value. It does **not** pin the similarity, the
temperature or the reduction. Those now follow the ETEGRec official
implementation (``trainer.py::compute_contrastive_loss`` plus the
``dec_cl_loss`` call site in ``_train_epoch_rec``):

  * similarity: cosine -- ``F.normalize`` on both sides, then the matrix
    product. This is ETEGRec's ``sim='cos'`` branch
    (``config/scientific.yaml: sim: cos``).
  * temperature: 0.07 -- the static default at ETEGRec's call site, which never
    overrides it (``config: tau: 0.07``).
  * reduction: mean over the batch per direction (``F.cross_entropy``'s
    default) and a plain *sum* of the two directions. ETEGRec computes
    ``cl(a, b) + cl(b, a)`` -- no ``0.5`` factor.

No margin, no hard negatives, no memory bank, no stop-gradient.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .psa import PSAAdapter

# item->SID offsets are `code + level * codebook_size + 1` (model/dataset.py).
CODEBOOK_SIZE = 256
NUM_LEVELS = 3


def symmetric_contrastive_loss(seq_emb: torch.Tensor, tgt_emb: torch.Tensor,
                               temperature: float = 0.07) -> torch.Tensor:
    """In-batch symmetric InfoNCE over *cosine* similarities.

    ``seq_emb`` is (B, D), ``tgt_emb`` is (B, D) and row ``i`` of one pairs with
    row ``i`` of the other; every other row in the batch is a negative. Returns
    ``CE(cos(seq, tgt) / T, arange) + CE(cos(tgt, seq) / T, arange)``, matching
    ``trainer.py::compute_contrastive_loss(..., sim='cos')`` with the default
    ``temperature=0.07``, summed over both directions.
    """
    if seq_emb.dim() != 2 or tgt_emb.dim() != 2:
        raise ValueError(f"expected (B, D) on both sides, got {tuple(seq_emb.shape)} "
                         f"and {tuple(tgt_emb.shape)}")
    if seq_emb.shape[0] != tgt_emb.shape[0]:
        raise ValueError(f"batch sizes differ: {seq_emb.shape[0]} vs {tgt_emb.shape[0]}")
    seq_emb = F.normalize(seq_emb, dim=-1)
    tgt_emb = F.normalize(tgt_emb, dim=-1)
    logits = seq_emb @ tgt_emb.t() / temperature
    positives = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, positives) + F.cross_entropy(logits.t(), positives)


class PSALoss(nn.Module):
    """The frozen E6 loss: symmetric in-batch contrastive loss."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, psa_representation: torch.Tensor,
                target_representation: torch.Tensor) -> torch.Tensor:
        """(B, D_target) and (B, D_target) -> scalar."""
        return symmetric_contrastive_loss(psa_representation, target_representation,
                                          self.temperature)


class PSAMount(nn.Module):
    """Bundle the PSA path so the Recommender loop can call it in one line.

    Holds the E5 adapter and the target-side item table. The target-side
    representation is the RQ-VAE *reconstruction* of the target item, matching
    E6 -- the raw content embedding is only the input to that reconstruction.

    Args:
        adapter: E5 ``PSAAdapter``.
        rqvae: TIGER ``RQVAE``; its ``forward`` first return value is the
            reconstruction ``decoder(x_q)`` and is used unmodified.
        target_embeddings: (n_items, D_target) item content embeddings, row order
            matching the SID file (row i <-> item i+1 <-> SID row i).
        sid_tokens: (n_items, 4) int64 SID tokens in TIGER's offset layout.
        lambda_psa: weight; 0 disables PSA entirely.
        temperature: forwarded to ``PSALoss``.
    """

    def __init__(self, adapter: nn.Module, rqvae: nn.Module,
                 target_embeddings: torch.Tensor, sid_tokens: torch.Tensor,
                 lambda_psa: float = 0.0, temperature: float = 0.07):
        super().__init__()
        self.adapter = adapter
        self.rqvae = rqvae
        self.lambda_psa = float(lambda_psa)
        self.loss_fn = PSALoss(temperature=temperature)
        self.register_buffer("target_embeddings", target_embeddings.detach().clone(),
                             persistent=False)

        # same SID -> item row routing as the SIA mount (tiger_ext/sia_loss.py):
        # the batch carries offset-layout SID tokens, the table is raw codes
        sid_tokens = sid_tokens.detach().cpu()
        offsets = sid_tokens + torch.arange(sid_tokens.shape[1]) * CODEBOOK_SIZE + 1
        lookup = {}
        for row in range(offsets.shape[0]):
            lookup.setdefault(tuple(int(t) for t in offsets[row, :NUM_LEVELS + 1]), row)
        self._lookup = lookup

    @property
    def enabled(self) -> bool:
        """PSA only runs when the weight is strictly positive."""
        return self.lambda_psa > 0.0

    def rows_for_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """(B, 4) target SID tokens -> (B,) rows into ``target_embeddings``."""
        rows = []
        for token_row in labels.detach().cpu():
            key = tuple(int(t) for t in token_row[:NUM_LEVELS + 1])
            row = self._lookup.get(key)
            if row is None:
                raise KeyError(
                    f"target SID {key} is not in the item table; PSA cannot build "
                    f"a reconstruction target for it")
            rows.append(row)
        return torch.as_tensor(rows, dtype=torch.long, device=labels.device)

    def target_representation(self, labels: torch.Tensor) -> torch.Tensor:
        """(B, 4) target SID -> (B, D_target) RQ-VAE reconstruction."""
        x = self.target_embeddings[self.rows_for_labels(labels)]
        out, _, _, _ = self.rqvae(x)      # out = decoder(x_q), the E6 target
        return out

    def forward(self, decoder_hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return ``lambda_psa * L_PSA`` (a scalar tensor).

        Raises if called while disabled -- the caller is expected to guard with
        ``enabled`` instead of relying on a zero return value.
        """
        if not self.enabled:
            raise RuntimeError("PSAMount.forward called with lambda_psa = 0; guard with `enabled`")
        psa_rep = self.adapter(decoder_hidden)
        return self.lambda_psa * self.loss_fn(psa_rep, self.target_representation(labels))


__all__ = ["PSAAdapter", "PSAMount", "PSALoss", "symmetric_contrastive_loss"]
