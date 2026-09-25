"""
trans_loss.py — Track B: Transition-aware Contrastive Loss (L_trans) + Balance Loss (L_balance)

Track B = Track A + L_trans + L_balance
- L_trans:  InfoNCE loss that pulls behaviorally-related items together in latent space
- L_balance: KL-divergence that encourages uniform codebook usage (anti-collapse)

Usage:
    from trans_loss import ContrastiveLoss, BalanceLoss

    contrastive = ContrastiveLoss(tau=0.1)
    balance = BalanceLoss()

    # L_trans
    loss_ct = contrastive(z=encoder_output,          # (B, D)
                          anchor_indices=anchor_idxs, # (P,)
                          positive_indices=pos_idxs)  # (P,)

    # L_balance
    loss_bal = balance(indices=rq_indices,            # list of (B,) tensors
                       num_embeddings_list=[256,256,256])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for transition-aware representation learning.

    Intuition:
        If item_i -> item_j is a behavioral transition (user clicked i then j),
        we want their latent representations z_i and z_j to be similar.

        For each anchor i, the loss is:
            -log( exp(sim(z_i, z_j)/τ) / Σ_{k∈B} exp(sim(z_i, z_k)/τ) )

    Args:
        tau: temperature scaling factor (default 0.1)
    """
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        z: torch.Tensor,
        anchor_indices: torch.Tensor,
        positive_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z: (B, D) encoder output vectors (unnormalized)
            anchor_indices: (P,)  indices in batch for anchor items
            positive_indices: (P,) indices in batch for positive items
                              (positive_j for anchor_i where (i,j) is a transition)

        Returns:
            scalar loss (0 if no pairs)
        """
        B = z.shape[0]
        P = anchor_indices.shape[0]

        if P == 0 or B < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        # L2 normalize along feature dimension
        z_norm = F.normalize(z, dim=-1)  # (B, D)

        # Similarity matrix: sim(i, j) = z_i · z_j / τ
        logits = torch.mm(z_norm, z_norm.t()) / self.tau  # (B, B)

        # Mask out self-similarity (diagonal) — set to -inf so they don't contribute
        diag_mask = torch.eye(B, device=z.device, dtype=torch.bool)
        logits = logits.masked_fill(diag_mask, float('-inf'))

        # Cross-entropy: for each anchor, positive = positive_indices[k]
        # CE loss = -log(exp(logit[pos]) / Σ exp(logits))
        loss = F.cross_entropy(logits[anchor_indices], positive_indices)

        return loss


class BalanceLoss(nn.Module):
    """
    Codebook balance loss — prevents codebook collapse by encouraging
    uniform usage of all codebook entries.

    For each VQ layer l:
        L_bal_l = KL(q_l(c) || U(c))
        where q_l(c) = empirical codebook usage distribution
              U(c)  = uniform distribution over codebook entries

    Args:
        eps: small epsilon to avoid log(0) in KL computation
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        indices: torch.Tensor,
        num_embeddings_list: list,
    ) -> torch.Tensor:
        """
        Args:
            indices: (B, num_layers) tensor of codebook assignments
                     each column = one VQ layer's indices
            num_embeddings_list: list of ints, codebook sizes per layer

        Returns:
            scalar loss (0 if no batch)
        """
        B, num_layers = indices.shape
        if B == 0 or num_layers == 0:
            return torch.tensor(0.0)

        total_loss = 0.0

        for layer_idx in range(num_layers):
            num_emb = num_embeddings_list[layer_idx]
            idx = indices[:, layer_idx]  # (B,)

            # Count usage per codebook entry
            counts = torch.bincount(idx, minlength=num_emb).float()  # (num_emb,)

            # Empirical distribution
            prob = counts / (counts.sum() + self.eps)  # (num_emb,)

            # Target: uniform distribution
            uniform = torch.full_like(prob, fill_value=1.0 / num_emb)

            # KL(prob || uniform) = Σ prob_k * (log(prob_k) - log(uniform_k))
            kl = (prob * ((prob + self.eps).log() - (uniform + self.eps).log())).sum()
            total_loss += kl

        return total_loss / num_layers
