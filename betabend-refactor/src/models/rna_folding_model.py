"""
RNA 3D Structure Prediction Model (V1)

This module implements the main model architecture for RNA 3D structure prediction,
integrating embeddings, transformer blocks, and the IPA module for coordinate prediction.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import EmbeddingModule
from .transformer_block import TransformerBlock
from .ipa_module import IPAModule


class RNAFoldingModel(nn.Module):
    """
    End-to-end RNA 3D folding model that combines embeddings, transformer blocks,
    and a coordinate prediction module.

    The model takes RNA sequence and feature data and predicts 3D coordinates,
    confidence scores, and torsion angles.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the RNA folding model.

        Args:
            Can be initialized in two ways:
            1. With a config dictionary as the first positional argument
            2. With keyword arguments that are passed to create a config dictionary

            Parameters:
                - num_blocks: Number of transformer blocks (default: 4)
                - residue_embed_dim: Dimension of residue embeddings (default: 128)
                - pair_embed_dim: Dimension of pair embeddings (default: 64)
                - num_attention_heads: Number of attention heads in transformer (default: 4)
                - dropout: Dropout probability (default: 0.1)
                - confidence_output_dim: Dimension of confidence output (default: 1)
                - angles_output_dim: Dimension of angle output (default: 4)
                - ffn_dim: Hidden dimension for feed-forward networks (default: 4*residue_embed_dim)
                - max_relative_position: Maximum relative distance for positional encoding (default: 32)
                - embed_dim: (Alias for residue_embed_dim)
                - num_encoder_layers: (Alias for num_blocks)
                - num_heads: (Alias for num_attention_heads)
                - num_ipa_layers: Number of IPA iterations (default: 1, for future use)
        """
        super().__init__()

        # Create config dictionary from arguments
        if len(args) > 0 and isinstance(args[0], dict):
            config = args[0]
        else:
            config = kwargs

        # Handle parameter aliases for compatibility with tests
        if "embed_dim" in config and "residue_embed_dim" not in config:
            config["residue_embed_dim"] = config["embed_dim"]
        
        if "num_encoder_layers" in config and "num_blocks" not in config:
            config["num_blocks"] = config["num_encoder_layers"]
            
        if "num_heads" in config and "num_attention_heads" not in config:
            config["num_attention_heads"] = config["num_heads"]

        # Extract parameters from config
        self.num_blocks = config.get("num_blocks", 4)
        self.residue_dim = config.get("residue_embed_dim", 128)
        self.pair_dim = config.get("pair_embed_dim", 64)
        self.confidence_output_dim = config.get("confidence_output_dim", 1)
        self.angles_output_dim = config.get("angles_output_dim", 4)

        # Initialize the embedding module
        self.embedding_module = EmbeddingModule(config)

        # Initialize transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(self.num_blocks)]
        )

        # Initialize IPA module for coordinate prediction
        self.ipa_module = IPAModule(config)

        # Initialize confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(self.residue_dim, self.residue_dim // 2),
            nn.ReLU(),
            nn.Linear(self.residue_dim // 2, self.confidence_output_dim),
        )

        # Initialize angle prediction head
        self.angle_head = nn.Sequential(
            nn.Linear(self.residue_dim, self.residue_dim // 2),
            nn.ReLU(),
            nn.Linear(self.residue_dim // 2, self.angles_output_dim),
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the RNA folding model.

        Args:
            batch: Dictionary of input tensors from the data loader
                - sequence_int: Integer-encoded RNA sequence (batch_size, seq_len)
                - dihedral_features: Dihedral angle features (batch_size, seq_len, 4)
                - pairing_probs: Pairing probabilities (batch_size, seq_len, seq_len)
                - positional_entropy: Positional entropy (batch_size, seq_len)
                - accessibility: Solvent accessibility (batch_size, seq_len)
                - coupling_matrix: Evolutionary coupling (batch_size, seq_len, seq_len)
                - mask: Boolean mask (batch_size, seq_len)

        Returns:
            Dictionary of output tensors:
                - pred_coords: Predicted coordinates (batch_size, seq_len, 3)
                - pred_confidence: Predicted confidence scores (batch_size, seq_len)
                - pred_angles: Predicted torsion angles (batch_size, seq_len, 4)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Check if we have lists instead of tensors (due to collate issues)
        # If so, process samples individually and combine results
        has_list_inputs = False
        for key, value in batch.items():
            if isinstance(value, list) and key not in ["target_ids"]:
                has_list_inputs = True
                break
                
        if has_list_inputs:
            logger.warning("Processing batch with list inputs individually")
            
            # Process each sample individually and collect results
            all_outputs = []
            for i in range(len(batch["lengths"])):
                # Create a sample batch with just this example
                sample_batch = {}
                for key, value in batch.items():
                    if isinstance(value, list):
                        if i < len(value):
                            if isinstance(value[i], torch.Tensor):
                                sample_batch[key] = value[i].unsqueeze(0)  # Add batch dimension
                            else:
                                # For non-tensor list items
                                sample_batch[key] = value[i]
                    elif isinstance(value, torch.Tensor):
                        sample_batch[key] = value[i:i+1]  # Select single example
                    else:
                        # Handle non-tensor values
                        if key == "target_ids":
                            sample_batch[key] = [value[i]]
                        elif isinstance(value, (str, int, float, bool)):
                            sample_batch[key] = value  # Keep scalar values as is
                
                # Process this single example
                try:
                    sample_output = self._forward_single_batch(sample_batch)
                    all_outputs.append(sample_output)
                except Exception as e:
                    logger.error(f"Error processing sample {i}: {e}")
                    # Create empty outputs
                    seq_len = sample_batch["lengths"].item()
                    device = sample_batch["lengths"].device
                    all_outputs.append({
                        "pred_coords": torch.zeros((1, seq_len, 3), device=device),
                        "pred_confidence": torch.zeros((1, seq_len), device=device),
                        "pred_angles": torch.zeros((1, seq_len, 4), device=device),
                    })
            
            # Combine individual results
            batch_size = len(all_outputs)
            seq_len = max(out["pred_coords"].shape[1] for out in all_outputs)
            
            # Initialize combined outputs
            device = all_outputs[0]["pred_coords"].device
            combined_coords = torch.zeros((batch_size, seq_len, 3), device=device)
            combined_conf = torch.zeros((batch_size, seq_len), device=device)
            combined_angles = torch.zeros((batch_size, seq_len, 4), device=device)
            
            # Copy each output to the combined tensor
            for i, out in enumerate(all_outputs):
                sample_len = out["pred_coords"].shape[1]
                combined_coords[i, :sample_len] = out["pred_coords"].squeeze(0)
                combined_conf[i, :sample_len] = out["pred_confidence"].squeeze(0)
                combined_angles[i, :sample_len] = out["pred_angles"].squeeze(0)
            
            return {
                "pred_coords": combined_coords,
                "pred_confidence": combined_conf,
                "pred_angles": combined_angles
            }
            
        else:
            # Standard processing for properly batched inputs
            return self._forward_single_batch(batch)
    
    def _forward_single_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process a single batch with all tensor inputs (no lists).
        This is the original forward implementation.
        """
        # Validate required inputs
        required_keys = [
            "sequence_int",
            "pairing_probs",
            "positional_entropy",
            "accessibility",
            "coupling_matrix",
            "mask",
        ]

        # Optional keys (for test vs train mode compatibility)
        optional_keys = [
            "dihedral_features",
        ]

        # Check for required keys
        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Input batch missing required key: {key}")
                
        # Add placeholder for dihedral features if missing (test mode)
        if "dihedral_features" not in batch and "sequence_int" in batch:
            # Create zeros tensor with appropriate shape
            seq_len = batch["sequence_int"].shape[1]
            batch_size = batch["sequence_int"].shape[0]
            batch["dihedral_features"] = torch.zeros(
                (batch_size, seq_len, 4), 
                device=batch["sequence_int"].device
            )

        # Extract mask for convenience
        mask = batch["mask"]  # (batch_size, seq_len)

        # Create initial representations using embedding module
        residue_repr, pair_repr, mask = self.embedding_module(batch)
        # residue_repr: (batch_size, seq_len, residue_dim)
        # pair_repr: (batch_size, seq_len, seq_len, pair_dim)

        # Process through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # Update residue and pair representations
            residue_repr, pair_repr = block(residue_repr, pair_repr, mask)

        # Generate 3D coordinates using IPA module
        pred_coords = self.ipa_module(
            residue_repr, pair_repr, mask
        )  # (batch_size, seq_len, 3)

        # Predict per-residue confidence
        confidence_logits = self.confidence_head(
            residue_repr
        )  # (batch_size, seq_len, 1)
        pred_confidence = confidence_logits.squeeze(-1)  # (batch_size, seq_len)

        # Predict angles
        pred_angles = self.angle_head(residue_repr)  # (batch_size, seq_len, 4)

        # Apply mask to outputs
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            pred_coords = pred_coords * mask_float
            pred_confidence = pred_confidence * mask.float()
            pred_angles = pred_angles * mask_float

        # Return all predictions
        outputs = {
            "pred_coords": pred_coords,
            "pred_confidence": pred_confidence,
            "pred_angles": pred_angles,
        }

        return outputs
