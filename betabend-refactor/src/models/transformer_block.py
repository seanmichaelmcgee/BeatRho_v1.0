from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    Transformer block for RNA folding model with residue and pair updates.

    This block processes both residue-level and pair-level representations
    and updates them through attention and feed-forward networks.
    """

    def __init__(self, config: Dict):
        """
        Initialize transformer block.

        Args:
            config: Dictionary containing model parameters:
                - residue_embed_dim: Dimension of residue embeddings
                - pair_embed_dim: Dimension of pair embeddings
                - num_attention_heads: Number of attention heads
                - dropout: Dropout probability
                - ffn_dim: Hidden dimension for feed-forward networks
        """
        super().__init__()

        # Extract parameters from config
        self.residue_dim = config.get("residue_embed_dim", 128)
        self.pair_dim = config.get("pair_embed_dim", 64)
        self.num_heads = config.get("num_attention_heads", 4)
        self.dropout_rate = config.get("dropout", 0.1)
        self.ffn_dim = config.get("ffn_dim", self.residue_dim * 4)

        # Validate dimensions
        if self.residue_dim % self.num_heads != 0:
            raise ValueError(
                f"residue_embed_dim ({self.residue_dim}) must be divisible by "
                f"num_attention_heads ({self.num_heads})"
            )

        # Initialize components for residue update
        self._init_residue_update_components()

        # Initialize components for pair update
        self._init_pair_update_components()

    def _init_residue_update_components(self):
        """Initialize layers for residue representation update."""
        # Pre-normalization for attention
        self.residue_attn_norm = nn.LayerNorm(self.residue_dim)

        # Multi-head attention
        self.residue_attention = nn.MultiheadAttention(
            embed_dim=self.residue_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True,
        )

        # Dropout after attention
        self.residue_attn_dropout = nn.Dropout(self.dropout_rate)

        # Pre-normalization for feed-forward
        self.residue_ffn_norm = nn.LayerNorm(self.residue_dim)

        # Feed-forward network for residue update
        self.residue_ffn = nn.Sequential(
            nn.Linear(self.residue_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.ffn_dim, self.residue_dim),
        )

        # Dropout after feed-forward
        self.residue_ffn_dropout = nn.Dropout(self.dropout_rate)

    def _init_pair_update_components(self):
        """Initialize layers for pair representation update."""
        # Pre-normalization for pair update
        self.pair_norm = nn.LayerNorm(self.pair_dim)

        # Calculate input dimension for pair MLP
        # Concatenate: h_i, h_j, and pair_repr
        pair_input_dim = 2 * self.residue_dim + self.pair_dim

        # MLP for pair update
        self.pair_update_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, self.pair_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.pair_dim * 2, self.pair_dim),
        )

        # Dropout for pair update
        self.pair_dropout = nn.Dropout(self.dropout_rate)

    def forward(
        self,
        residue_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.

        Args:
            residue_repr: Residue representations of shape (batch_size, seq_len, residue_dim)
            pair_repr: Pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
            mask: Boolean mask of shape (batch_size, seq_len) where True indicates valid positions

        Returns:
            Tuple of:
            - Updated residue representations of shape (batch_size, seq_len, residue_dim)
            - Updated pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
        """
        # Get device
        device = residue_repr.device

        # Update residue representations using self-attention
        residue_repr = self._update_residue_repr(residue_repr, mask)

        # Update pair representations
        pair_repr = self._update_pair_repr(residue_repr, pair_repr, mask)

        return residue_repr, pair_repr

    def _update_residue_repr(
        self, residue_repr: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Update residue representations through self-attention and feed-forward network.

        Args:
            residue_repr: Residue representations of shape (batch_size, seq_len, residue_dim)
            mask: Boolean mask of shape (batch_size, seq_len)

        Returns:
            Updated residue representations of shape (batch_size, seq_len, residue_dim)
        """
        # Prepare attention mask from boolean mask
        attn_mask = None
        key_padding_mask = None

        if mask is not None:
            # Invert mask for PyTorch attention (False = keep, True = mask)
            key_padding_mask = ~mask

        # Pre-normalization
        res_norm = self.residue_attn_norm(residue_repr)

        # Self-attention
        attn_output, _ = self.residue_attention(
            query=res_norm,
            key=res_norm,
            value=res_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Residual connection with dropout
        residue_repr = residue_repr + self.residue_attn_dropout(attn_output)

        # Feed-forward network with pre-normalization
        res_norm = self.residue_ffn_norm(residue_repr)
        ffn_output = self.residue_ffn(res_norm)

        # Residual connection with dropout
        residue_repr = residue_repr + self.residue_ffn_dropout(ffn_output)

        # Apply mask to ensure padded positions remain zero
        if mask is not None:
            residue_repr = residue_repr * mask.unsqueeze(-1).float()

        return residue_repr

    def _update_pair_repr(
        self,
        residue_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update pair representations using residue representations.

        Args:
            residue_repr: Residue representations of shape (batch_size, seq_len, residue_dim)
            pair_repr: Pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
            mask: Boolean mask of shape (batch_size, seq_len)

        Returns:
            Updated pair representations of shape (batch_size, seq_len, seq_len, pair_dim)
        """
        batch_size, seq_len, _ = residue_repr.shape

        # Pre-normalization
        pair_norm = self.pair_norm(pair_repr)

        # Create outer product of residue representations
        # For each pair (i,j), we'll concatenate h_i, h_j, and pair_repr[i,j]

        # Expand residue representations for broadcasting
        h_i = residue_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (B, L, L, D_res)
        h_j = residue_repr.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (B, L, L, D_res)

        # Concatenate along the feature dimension
        pair_inputs = torch.cat(
            [h_i, h_j, pair_norm], dim=-1
        )  # (B, L, L, 2*D_res + D_pair)

        # Apply MLP to update pair representations
        pair_update = self.pair_update_mlp(pair_inputs)

        # Residual connection with dropout
        pair_repr = pair_repr + self.pair_dropout(pair_update)

        # Apply mask to ensure masked positions remain zero
        if mask is not None:
            # Create 2D mask for pairs
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, L, L)
            pair_mask = pair_mask.unsqueeze(-1).float()  # (B, L, L, 1)
            pair_repr = pair_repr * pair_mask

        return pair_repr
