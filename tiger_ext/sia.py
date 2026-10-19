"""SIA adapter -- Task E2 of the ETEGRec -> TIGER port.

Turns the T5 encoder hidden sequence into a single 32-d sequence-side latent:

    encoder_hidden [B, L, d_model]  +  attention_mask [B, L]
            -> masked mean pooling      -> [B, d_model]
            -> Linear(d_model, 32)      -> SIA query [B, 32]

The 32 is not a free parameter: it is TIGER's RQ-VAE latent width
(``RQVAE.e_dim = 32``), so the sequence-side query lands in the same space the
codebooks were trained on. ETEGRec's own RQ-VAE is deliberately NOT ported.

The pooled mean is taken over the *valid* positions only. Encoder hidden states
at padded positions are not zero (the encoder runs over the full padded length),
so an unmasked ``mean(dim=1)`` would dilute short sequences -- on a real TIGER
batch most positions are padding.
"""
import torch
import torch.nn as nn

SIA_LATENT_DIM = 32


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean over the positions where ``attention_mask`` is non-zero.

    Args:
        hidden_states: (B, L, d_model) encoder hidden states.
        attention_mask: (B, L) 1 for a real token, 0 for padding.

    Returns:
        (B, d_model) masked mean. Padded positions contribute nothing, so the
        result does not change when extra padding is appended. A row with no
        valid position (empty history) yields zeros rather than NaN.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be (B, L, d_model), got {tuple(hidden_states.shape)}")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError(
            f"attention_mask {tuple(attention_mask.shape)} does not match "
            f"hidden_states {tuple(hidden_states.shape[:2])}")

    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)      # (B, L, 1)
    summed = (hidden_states * mask).sum(dim=1)                       # (B, d_model)
    counts = mask.sum(dim=1).clamp(min=1.0)                          # (B, 1)
    return summed / counts


class SIAAdapter(nn.Module):
    """Masked-mean pooling followed by a linear projection to the RQ latent space.

    Args:
        d_model: width of the incoming T5 encoder hidden states (128 in TIGER).
        latent_dim: TIGER RQ-VAE latent width; fixed at 32 by design.
    """

    def __init__(self, d_model: int, latent_dim: int = SIA_LATENT_DIM):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.proj = nn.Linear(d_model, latent_dim)

    def forward(self, encoder_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) encoder hidden -> (B, latent_dim) SIA query."""
        return self.proj(masked_mean_pool(encoder_hidden, attention_mask))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, latent_dim={self.latent_dim}"
