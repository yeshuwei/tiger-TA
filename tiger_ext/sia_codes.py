"""SIA code-distribution extraction -- Task E3 of the ETEGRec -> TIGER port.

Produces, for the same TIGER RQ-VAE quantizer, a per-RQ-level distribution over
the codebook on each side of the SIA alignment:

    sequence side:  encoder_hidden -> SIAAdapter -> (B, 32) query -> rq
    target side:    target item content embedding -> RQVAE.encoder -> (B, 32) -> rq
                                                                |
                                          -> 3 x (B, 256) code logits each

The two sides share one codebook set: the sequence-side query goes straight into
``RQVAE.rq``, the same module the target side uses. TIGER's own quantizer is
reused as-is -- ETEGRec's RQ-VAE is deliberately not ported.

Only the three RQ levels produce logits. The 4th SID column (the collision
token) is not a quantizer level: it is produced later by collision resolution in
``rqvae/generate_code.py``, so it cannot appear here.
"""
import torch
import torch.nn as nn


class SIACodeExtractor(nn.Module):
    """Adapter + TIGER quantizer, returning per-level code logits for both sides.

    Args:
        adapter: the E2 ``SIAAdapter``.
        rqvae: a TIGER ``RQVAE`` -- its ``encoder`` and ``rq`` are used directly.
        use_sk: forwarded to the quantizer; matches how TIGER assigns codes.
        logit_temperature: divides the logits. Default 1.0 exposes the raw
            ``squared_L2_distance``; E4 applies its own temperature at softmax.
    """

    def __init__(self, adapter: nn.Module, rqvae: nn.Module,
                 use_sk: bool = True, logit_temperature: float = 1.0):
        super().__init__()
        self.adapter = adapter
        self.rqvae = rqvae
        self.use_sk = use_sk
        self.logit_temperature = logit_temperature

    @property
    def num_levels(self) -> int:
        """Number of RQ levels with a codebook (3 for TIGER)."""
        return self.rqvae.rq.num_quantizers

    def to_query(self, encoder_hidden: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) encoder hidden -> (B, 32) sequence-side latent.

        This is the E2 adapter and nothing else: the sequence side never passes
        through ``RQVAE.encoder``.
        """
        return self.adapter(encoder_hidden, attention_mask)

    def codes_from_query(self, query: torch.Tensor):
        """(B, 32) latent -> a tuple of ``num_levels`` (B, 256) code logits.

        The query is handed to the quantizer unchanged, so its value at that
        point is the tensor the caller passed in.
        """
        _, _, _, logits = self.rqvae.rq(
            query, use_sk=self.use_sk, return_logits=True,
            logit_temperature=self.logit_temperature)
        return tuple(logits[:, level] for level in range(logits.shape[1]))

    def sequence_logits(self, encoder_hidden: torch.Tensor,
                        attention_mask: torch.Tensor):
        """Sequence side: encoder hidden -> (B, 32) -> 3 x (B, 256) logits."""
        return self.codes_from_query(self.to_query(encoder_hidden, attention_mask))

    def target_logits(self, target_item_embedding: torch.Tensor):
        """Target side: content embedding -> ``RQVAE.encoder`` -> 3 x (B, 256).

        Unlike the sequence side, this path *does* go through the RQ-VAE
        encoder, because that is where TIGER's codebooks live.
        """
        return self.codes_from_query(self.rqvae.encoder(target_item_embedding))
