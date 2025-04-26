import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class SequenceEmbedding(nn.Module):
    """
    Embedding layer for RNA nucleotide sequences.
    Maps integer-encoded nucleotides to learned embeddings.
    """

    def __init__(self, config: Dict):
        """
        Initialize sequence embedding layer.

        Args:
            config: Dictionary containing model parameters:
                - num_embeddings: Number of distinct nucleotides (5 for A,C,G,U,N)
                - seq_embed_dim: Dimension of embedding vectors
        """
        super().__init__()

        # Extract parameters from config
        self.num_embeddings = config.get("num_embeddings", 5)  # A, C, G, U, N/padding
        self.embedding_dim = config.get("seq_embed_dim", 32)
        self.padding_idx = config.get("padding_idx", 4)  # Usually N or padding

        # Initialize embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx,
        )

    def forward(self, sequence_int: torch.Tensor) -> torch.Tensor:
        """
        Convert integer-encoded sequences to embeddings.

        Args:
            sequence_int: Integer tensor of shape (batch_size, seq_len)

        Returns:
            Embedded sequences of shape (batch_size, seq_len, embedding_dim)
        """
        return self.embedding(sequence_int)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for RNA sequences.
    Provides position information to the model.
    """

    def __init__(self, config: Dict):
        """
        Initialize positional encoding.

        Args:
            config: Dictionary containing model parameters:
                - residue_embed_dim: Embedding dimension
                - max_len: Maximum sequence length to pre-compute (default: 500)
        """
        super().__init__()

        # Extract parameters from config
        self.embed_dim = config.get("residue_embed_dim", 128)
        self.max_len = config.get("max_len", 500)

        # Create constant positional encoding matrix
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim)
        )

        pe = torch.zeros(self.max_len, self.embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register buffer (not a parameter, but part of state)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, embed_dim)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encodings for sequences of length seq_len.

        Args:
            seq_len: Sequence length to retrieve encodings for

        Returns:
            Positional encodings of shape (1, seq_len, embed_dim)
        """
        return self.pe[:, :seq_len]


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding for pairs of nucleotides.
    Encodes the distance between positions in the sequence.
    """

    def __init__(self, config: Dict):
        """
        Initialize relative positional encoding.

        Args:
            config: Dictionary containing model parameters:
                - max_relative_position: Maximum relative distance to consider
                - rel_pos_dim: Dimension of the relative position embedding
        """
        super().__init__()

        # Extract parameters from config
        self.max_relative_position = config.get("max_relative_position", 32)
        self.rel_pos_dim = config.get("rel_pos_dim", 32)

        # Create embedding for relative positions
        # Total embeddings: 2*max_rel_pos + 1 (to account for -max to +max)
        num_embeddings = 2 * self.max_relative_position + 1
        self.embeddings = nn.Embedding(num_embeddings, self.rel_pos_dim)

        # Initialize with sinusoidal pattern
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding weights with sinusoidal pattern."""
        position = torch.arange(
            -self.max_relative_position, self.max_relative_position + 1
        ).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, self.rel_pos_dim, 2).float() * (-math.log(10000.0) / self.rel_pos_dim)
        )

        # Create sinusoidal pattern
        pe = torch.zeros(2 * self.max_relative_position + 1, self.rel_pos_dim)
        pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].shape[1]])
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        # Set embedding weights
        with torch.no_grad():
            self.embeddings.weight.copy_(pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute relative positional encodings for all position pairs.

        Args:
            seq_len: Sequence length

        Returns:
            Tensor of shape (seq_len, seq_len, rel_pos_dim) with relative
            position embeddings for each position pair
        """
        # Create position indices
        positions = torch.arange(seq_len, device=self.embeddings.weight.device)

        # Compute relative positions between all position pairs
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Shift and clip relative positions to be in the range [0, 2*max_relative_position]
        relative_positions = torch.clamp(
            relative_positions + self.max_relative_position, 0, 2 * self.max_relative_position
        )

        # Get embeddings for all position pairs
        return self.embeddings(relative_positions)


class EmbeddingModule(nn.Module):
    """
    Complete embedding module that initializes both residue and pair representations.
    """

    def __init__(self, config: Dict):
        """
        Initialize the embedding module.

        Args:
            config: Dictionary containing model parameters
        """
        super().__init__()

        # Extract dimensions from config
        self.seq_embed_dim = config.get("seq_embed_dim", 32)
        self.residue_dim = config.get("residue_embed_dim", 128)
        self.pair_dim = config.get("pair_embed_dim", 64)
        self.use_conservation = config.get("use_conservation", True)

        # Initialize embedding components
        self.sequence_embedding = SequenceEmbedding(config)
        self.positional_encoding = PositionalEncoding(config)
        self.relative_pos_encoding = RelativePositionalEncoding(config)

        # Calculate input dimension for residue projection:
        # sequence_embed + dihedral(4) + positional_entropy(1) + accessibility(1) + conservation(1) [if available]
        self.residue_in_dim = self.seq_embed_dim + 4 + 1 + 1
        if self.use_conservation:
            self.residue_in_dim += 1

        # Calculate input dimension for pair projection:
        # pairing_probs(1) + coupling_matrix(1) + relative_pos_encoding
        pair_in_dim = 1 + 1 + self.relative_pos_encoding.rel_pos_dim

        # Create projection layers
        self.residue_projection = nn.Linear(self.residue_in_dim, self.residue_dim)
        self.pair_projection = nn.Linear(pair_in_dim, self.pair_dim)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process input features to create initial residue and pair representations.

        Args:
            batch: Dictionary of input tensors from the data loader

        Returns:
            Tuple of (residue_repr, pair_repr, mask):
                - residue_repr: Tensor of shape (batch_size, seq_len, residue_dim)
                - pair_repr: Tensor of shape (batch_size, seq_len, seq_len, pair_dim)
                - mask: Boolean mask of shape (batch_size, seq_len)
        """
        # Extract tensors from batch with safety checks
        sequence_int = batch["sequence_int"]  # (batch_size, seq_len) or (batch_size, seq_len, dim)
        mask = batch["mask"]  # (batch_size, seq_len)
        device = sequence_int.device
        
        # Get batch size and sequence length
        if len(sequence_int.shape) == 3:
            # Take first channel if it's a 3D tensor (batch_size, seq_len, dim)
            sequence_int = sequence_int[:, :, 0].long()
        batch_size, seq_len = sequence_int.shape
        
        # Safely extract optional features with fallbacks
        # 1. Dihedral features (zeros in test mode)
        if "dihedral_features" in batch:
            dihedral_features = batch["dihedral_features"]  # (batch_size, seq_len, 4)
        else:
            logging.info("Dihedral features not found, using zeros (test mode)")
            dihedral_features = torch.zeros((batch_size, seq_len, 4), device=device)
            
        # 2. Pairing probabilities
        if "pairing_probs" in batch:
            pairing_probs = batch["pairing_probs"]  # (batch_size, seq_len, seq_len)
        else:
            logging.warning("Pairing probabilities not found, using zeros")
            pairing_probs = torch.zeros((batch_size, seq_len, seq_len), device=device)
            
        # 3. Positional entropy
        if "positional_entropy" in batch:
            positional_entropy = batch["positional_entropy"]  # (batch_size, seq_len)
        else:
            logging.warning("Positional entropy not found, using zeros")
            positional_entropy = torch.zeros((batch_size, seq_len), device=device)
            
        # 4. Coupling matrix
        if "coupling_matrix" in batch:
            coupling_matrix = batch["coupling_matrix"]  # (batch_size, seq_len, seq_len)
        else:
            logging.warning("Coupling matrix not found, using zeros")
            coupling_matrix = torch.zeros((batch_size, seq_len, seq_len), device=device)
            
        # 5. Accessibility
        if "accessibility" in batch:
            accessibility = batch["accessibility"]  # (batch_size, seq_len)
        else:
            logging.warning("Accessibility not found, using zeros")
            accessibility = torch.zeros((batch_size, seq_len), device=device)

        # We already processed sequence_int shape and got batch_size, seq_len above

        # Generate embeddings
        seq_embedding = self.sequence_embedding(sequence_int)  # (batch_size, seq_len, seq_embed_dim)
        pos_encoding = self.positional_encoding(seq_len)  # (1, seq_len, residue_dim)
        rel_pos_encoding = self.relative_pos_encoding(seq_len)  # (seq_len, seq_len, rel_pos_dim)

        # Prepare residue features for projection
        residue_features_list: List[torch.Tensor] = [
            seq_embedding,  # (batch_size, seq_len, seq_embed_dim)
            dihedral_features,  # (batch_size, seq_len, 4)
            positional_entropy.unsqueeze(-1),  # (batch_size, seq_len, 1)
            accessibility.unsqueeze(-1),  # (batch_size, seq_len, 1)
        ]

        # Add conservation if available and requested
        if self.use_conservation and "conservation" in batch:
            residue_features_list.append(batch["conservation"].unsqueeze(-1))  # (batch_size, seq_len, 1)

        # Concatenate all residue features
        residue_features = torch.cat(residue_features_list, dim=-1)  # (batch_size, seq_len, sum_of_dims)
        
        # Ensure the residue_features tensor has the expected dimension
        if residue_features.shape[-1] != self.residue_in_dim:
            # Print debugging info if there's a mismatch
            expected_dim = self.residue_in_dim
            actual_dim = residue_features.shape[-1]
            logging.warning(f"Expected residue_in_dim={expected_dim}, got shape[-1]={actual_dim}")
            
            # Adjust the projection layer to match the actual input dimension
            if not hasattr(self, 'adjusted_residue_projection') or self.adjusted_residue_projection.in_features != actual_dim:
                logging.info(f"Creating adjusted projection layer with in_features={actual_dim}")
                self.adjusted_residue_projection = nn.Linear(actual_dim, self.residue_dim, 
                                                         device=residue_features.device)
                # Initialize with similar statistics to the original projection
                if hasattr(self, 'residue_projection'):
                    with torch.no_grad():
                        # Copy weights for dimensions that match
                        min_dim = min(actual_dim, self.residue_projection.weight.shape[1])
                        self.adjusted_residue_projection.weight[:, :min_dim].copy_(
                            self.residue_projection.weight[:, :min_dim])
                        # Initialize bias the same 
                        self.adjusted_residue_projection.bias.copy_(self.residue_projection.bias)
            
            # Use the adjusted projection
            residue_repr = self.adjusted_residue_projection(residue_features)
        else:
            # Use the standard projection if dimensions match
            residue_repr = self.residue_projection(residue_features)

        # Add positional encodings
        residue_repr = residue_repr + pos_encoding  # (batch_size, seq_len, residue_dim)

        # Prepare pair features for projection
        # Expand rel_pos_encoding to batch dimension
        rel_pos_batch = rel_pos_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Debug the tensor shapes
        logging.info(f"pairing_probs shape: {pairing_probs.shape}")
        logging.info(f"coupling_matrix shape: {coupling_matrix.shape}")
        logging.info(f"rel_pos_batch shape: {rel_pos_batch.shape}")
        
        # Ensure all tensors have the same dimensions before concatenation
        if len(pairing_probs.shape) != 4:
            pairing_probs = pairing_probs.unsqueeze(-1)  # Add dimension if needed
        if len(coupling_matrix.shape) != 4:
            coupling_matrix = coupling_matrix.unsqueeze(-1)  # Add dimension if needed
            
        pair_features_list: List[torch.Tensor] = [
            pairing_probs,  # Should be (batch_size, seq_len, seq_len, 1)
            coupling_matrix,  # Should be (batch_size, seq_len, seq_len, 1)
            rel_pos_batch,  # (batch_size, seq_len, seq_len, rel_pos_dim)
        ]
        
        # Debug the adjusted tensor shapes
        for i, tensor in enumerate(pair_features_list):
            logging.info(f"pair_features_list[{i}] shape: {tensor.shape}")

        # Concatenate all pair features
        pair_features = torch.cat(pair_features_list, dim=-1)  # (batch_size, seq_len, seq_len, pair_in_dim)
        
        # Ensure the pair_features tensor has the expected dimension
        actual_pair_dim = pair_features.shape[-1]
        expected_pair_dim = self.pair_projection.in_features
        
        if actual_pair_dim != expected_pair_dim:
            logging.warning(f"Pair dimension mismatch: expected {expected_pair_dim}, got {actual_pair_dim}")
            
            # Create or update an adjusted projection layer
            if not hasattr(self, 'adjusted_pair_projection') or self.adjusted_pair_projection.in_features != actual_pair_dim:
                logging.info(f"Creating adjusted pair projection with in_features={actual_pair_dim}")
                self.adjusted_pair_projection = nn.Linear(
                    actual_pair_dim, self.pair_dim, device=pair_features.device
                )
                
                # Initialize with similar statistics if possible
                if hasattr(self, 'pair_projection'):
                    with torch.no_grad():
                        min_dim = min(actual_pair_dim, self.pair_projection.weight.shape[1])
                        if min_dim > 0:
                            self.adjusted_pair_projection.weight[:, :min_dim].copy_(
                                self.pair_projection.weight[:, :min_dim]
                            )
                        self.adjusted_pair_projection.bias.copy_(self.pair_projection.bias)
            
            # Reshape for linear layer
            orig_shape = pair_features.shape
            reshaped = pair_features.view(-1, actual_pair_dim)
            
            # Use the adjusted projection
            pair_repr_flat = self.adjusted_pair_projection(reshaped)
            
            # Reshape back
            pair_repr = pair_repr_flat.view(orig_shape[0], orig_shape[1], orig_shape[2], self.pair_dim)
        else:
            # Reshape for linear layer
            orig_shape = pair_features.shape
            reshaped = pair_features.view(-1, expected_pair_dim)
            
            # Use the standard projection
            pair_repr_flat = self.pair_projection(reshaped)
            
            # Reshape back
            pair_repr = pair_repr_flat.view(orig_shape[0], orig_shape[1], orig_shape[2], self.pair_dim)

        # Apply mask to both representations to ensure padded positions are zeros
        if mask is not None:
            # Create 1D mask for residue representations (B, L, 1)
            residue_mask = mask.unsqueeze(-1).float()
            residue_repr = residue_repr * residue_mask

            # Create 2D mask for pair representations (B, L, L, 1)
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (batch_size, seq_len, seq_len)
            pair_mask = pair_mask.unsqueeze(-1).float()  # (batch_size, seq_len, seq_len, 1)
            pair_repr = pair_repr * pair_mask

        return residue_repr, pair_repr, mask