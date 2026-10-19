"""PSA decoder interface -- Task E5 of the ETEGRec -> TIGER port.

Builds the sequence-side representation PSA aligns:

    T5 decoder_hidden_states[-1][:, 0, :]  ->  Linear(d_model, D_target)

Position ``0`` is the *sequence* position (the decoder's first input token,
``decoder_start_token_id``); the layer is the **last** one, which is what E1's
``return_hidden=True`` already hands over. Both are fixed by
``integration/plan.md`` E5 and are not configurable here.

``D_target`` is the RQ-VAE reconstruction dimension = the item content
embedding dimension (plan.md §1.3): 4096 for Toys/Sport/Beauty, 768 for the
other datasets. It is passed in rather than guessed.
"""
import torch
import torch.nn as nn

# Sequence position of the decoder's first input token (plan.md E5).
PSA_SEQ_POS = 0


def psa_position0(decoder_hidden: torch.Tensor) -> torch.Tensor:
    """(B, L_dec, d_model) -> (B, d_model), the last layer's sequence position 0.

    The layer choice happens upstream: E1 exposes ``decoder_hidden_states[-1]``,
    so this only selects the position.
    """
    if decoder_hidden.dim() != 3:
        raise ValueError(
            f"expected the last decoder hidden layer (B, L_dec, d_model), "
            f"got a {decoder_hidden.dim()}-D tensor of shape {tuple(decoder_hidden.shape)}")
    if decoder_hidden.shape[1] <= PSA_SEQ_POS:
        raise ValueError(
            f"decoder hidden has {decoder_hidden.shape[1]} positions, "
            f"cannot take position {PSA_SEQ_POS}")
    return decoder_hidden[:, PSA_SEQ_POS, :]


class PSAAdapter(nn.Module):
    """d_model -> D_target projection of the decoder's position-0 hidden state."""

    def __init__(self, d_model: int, d_target: int):
        super().__init__()
        self.d_model = d_model
        self.d_target = d_target
        self.proj = nn.Linear(d_model, d_target)

    def forward(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        """(B, L_dec, d_model) -> (B, D_target)."""
        return self.proj(psa_position0(decoder_hidden))
