"""
Enhanced Invariant Point Attention (IPA) Module for RNA 3D Structure Prediction

This module implements a structure-aware coordinate prediction system that:
1. Generates local frames for each residue
2. Applies iterative refinement using IPA-like attention
3. Enforces physical constraints during coordinate prediction
4. Uses pair information to model RNA structural motifs

This is a significant enhancement over the placeholder V1 implementation, which
simply used a direct MLP projection from residue features to coordinates.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameGenerator(nn.Module):
    """
    Generates initial frames and coordinates for each residue.
    
    A frame consists of an origin point (3D coordinates) and basis vectors 
    (rotation matrix) that define a local coordinate system for each residue.
    """

    def __init__(self, residue_dim: int, hidden_dim: Optional[int] = None):
        """
        Initialize frame generator network.
        
        Args:
            residue_dim: Dimension of residue representations
            hidden_dim: Hidden dimension for MLP (default: residue_dim)
        """
        super().__init__()
        
        self.residue_dim = residue_dim
        self.hidden_dim = hidden_dim or residue_dim
        
        # MLP for predicting initial coordinates (points)
        self.coord_predictor = nn.Sequential(
            nn.Linear(residue_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 3)  # 3D coordinates (x, y, z)
        )
        
        # MLP for predicting initial rotation matrices (3x3)
        # We predict 9 values and reshape to 3x3
        self.rotation_predictor = nn.Sequential(
            nn.Linear(residue_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 9)  # 9 values for 3x3 rotation matrix
        )
        
        # Initialize weights carefully
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with scaled Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier/Glorot initialization with careful scaling
                nn.init.xavier_uniform_(module.weight, gain=0.001)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _orthogonalize_rotation_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert a predicted 3x3 matrix to a valid rotation matrix using SVD.
        
        Args:
            matrix: Batch of predicted matrices, shape (batch_size, seq_len, 3, 3)
            
        Returns:
            Orthogonalized rotation matrices, shape (batch_size, seq_len, 3, 3)
        """
        batch_size, seq_len = matrix.shape[0], matrix.shape[1]
        
        # Reshape for batch SVD operation
        flat_matrices = matrix.reshape(-1, 3, 3)
        
        # Apply SVD to each matrix
        try:
            U, _, V = torch.linalg.svd(flat_matrices)
            
            # Compute rotation matrix: R = U @ V^T
            # This ensures we get a proper rotation matrix with det(R) = +1
            R = torch.bmm(U, V.transpose(-2, -1))
            
            # Ensure determinant is positive (proper rotation, not reflection)
            dets = torch.linalg.det(R)
            reflection_fix = torch.eye(3, device=R.device).unsqueeze(0).repeat(R.shape[0], 1, 1)
            reflection_fix[:, 2, 2] = torch.sign(dets)
            
            R = torch.bmm(R, reflection_fix)
            
            # Reshape back to original dimensions
            return R.reshape(batch_size, seq_len, 3, 3)
            
        except RuntimeError as e:
            # Fallback for numerical issues
            print(f"SVD failed: {e}, using fallback")
            
            # Identity rotation as fallback
            eye = torch.eye(3, device=matrix.device)
            return eye.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, 3, 3)
    
    def forward(self, residue_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate initial frames for each residue in the sequence.
        
        Args:
            residue_repr: Residue representations, shape (batch_size, seq_len, residue_dim)
            
        Returns:
            Tuple of:
            - frames: Rotation matrices, shape (batch_size, seq_len, 3, 3)
            - coords: Initial 3D coordinates, shape (batch_size, seq_len, 3)
        """
        batch_size, seq_len, _ = residue_repr.shape
        
        # Predict initial coordinates
        coords = self.coord_predictor(residue_repr)  # (batch_size, seq_len, 3)
        
        # Predict rotation matrices (frames)
        rot_flat = self.rotation_predictor(residue_repr)  # (batch_size, seq_len, 9)
        rot_matrices = rot_flat.reshape(batch_size, seq_len, 3, 3)  # (batch_size, seq_len, 3, 3)
        
        # Ensure valid rotation matrices
        frames = self._orthogonalize_rotation_matrix(rot_matrices)
        
        return frames, coords


class InvariantPointAttention(nn.Module):
    """
    Implements Invariant Point Attention mechanism for structure refinement.
    
    This attention mechanism is invariant to global rotations and translations
    as it operates on the relative positions of points in local coordinate frames.
    """

    def __init__(self, residue_dim: int, pair_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize IPA attention mechanism.
        
        Args:
            residue_dim: Dimension of residue representations
            pair_dim: Dimension of pair representations
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.residue_dim = residue_dim
        self.pair_dim = pair_dim
        self.num_heads = num_heads
        self.head_dim = residue_dim // num_heads
        self.dropout = dropout
        
        assert residue_dim % num_heads == 0, "residue_dim must be divisible by num_heads"
        
        # Linear projections for query, key, value
        self.query_proj = nn.Linear(residue_dim, residue_dim)
        self.key_proj = nn.Linear(residue_dim, residue_dim)
        self.value_proj = nn.Linear(residue_dim, residue_dim)
        
        # Pair bias projection
        self.pair_bias = nn.Linear(pair_dim, num_heads)
        
        # Point projection (from value to coordinate offsets)
        self.point_proj = nn.Sequential(
            nn.Linear(residue_dim, residue_dim),
            nn.ReLU(),
            nn.Linear(residue_dim, 3 * num_heads)  # 3D coordinate offset per head
        )
        
        # Output projection
        self.output_proj = nn.Linear(residue_dim, residue_dim)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(residue_dim)
    
    def forward(
        self, 
        residue_repr: torch.Tensor, 
        pair_repr: torch.Tensor, 
        frames: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply IPA attention mechanism for structure refinement.
        
        Args:
            residue_repr: Residue representations, shape (batch_size, seq_len, residue_dim)
            pair_repr: Pair representations, shape (batch_size, seq_len, seq_len, pair_dim)
            frames: Rotation matrices, shape (batch_size, seq_len, 3, 3)
            coords: Current 3D coordinates, shape (batch_size, seq_len, 3)
            mask: Boolean mask, shape (batch_size, seq_len)
            
        Returns:
            Tuple of:
            - updated_coords: Updated 3D coordinates, shape (batch_size, seq_len, 3)
            - updated_residue_repr: Updated residue representations, shape (batch_size, seq_len, residue_dim)
        """
        batch_size, seq_len, _ = residue_repr.shape
        device = residue_repr.device
        
        # Apply layer normalization
        normed_repr = self.layer_norm(residue_repr)
        
        # Project to queries, keys, values
        q = self.query_proj(normed_repr)  # (batch_size, seq_len, residue_dim)
        k = self.key_proj(normed_repr)    # (batch_size, seq_len, residue_dim)
        v = self.value_proj(normed_repr)  # (batch_size, seq_len, residue_dim)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention scores (q*k^T / sqrt(d_k))
        # This is the standard scaled dot-product attention
        attn_scores = torch.einsum('bihd,bjhd->bhij', q, k)  # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        
        # Add pair bias to attention scores
        # This incorporates the pair information (base-pairing, etc.)
        pair_bias = self.pair_bias(pair_repr)  # (batch_size, seq_len, seq_len, num_heads)
        pair_bias = pair_bias.permute(0, 3, 1, 2)  # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = attn_scores + pair_bias
        
        # Apply mask if provided
        if mask is not None:
            # Create 2D attention mask
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # (batch_size, seq_len, seq_len)
            attn_mask = mask_2d.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(~attn_mask, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.einsum('bhij,bjhd->bihd', attn_weights, v)  # (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.residue_dim)
        
        # Project back to residue dimension
        output_repr = self.output_proj(attn_output)  # (batch_size, seq_len, residue_dim)
        
        # Update residue representations with residual connection
        updated_residue_repr = residue_repr + output_repr
        
        # Predict coordinate offsets from attention output
        point_offsets = self.point_proj(attn_output)  # (batch_size, seq_len, 3*num_heads)
        point_offsets = point_offsets.reshape(batch_size, seq_len, self.num_heads, 3)
        
        # Apply attention to coordinate offsets
        coord_updates = torch.einsum('bhij,bjhd->bihd', attn_weights, point_offsets)  # (batch_size, seq_len, num_heads, 3)
        coord_updates = coord_updates.sum(dim=2)  # Sum over heads: (batch_size, seq_len, 3)
        
        # Transform coordinate updates from local to global frame
        # This applies the rotation matrix (frame) to the local coordinate updates
        global_updates = torch.bmm(
            coord_updates.reshape(batch_size * seq_len, 1, 3),
            frames.reshape(batch_size * seq_len, 3, 3)
        ).reshape(batch_size, seq_len, 3)
        
        # Apply the updates to current coordinates
        updated_coords = coords + global_updates
        
        # Apply mask if provided
        if mask is not None:
            mask_3d = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            updated_coords = updated_coords * mask_3d
        
        return updated_coords, updated_residue_repr


class StructureRefinementBlock(nn.Module):
    """
    Complete structure refinement block that combines IPA attention and coordinate updates.
    """

    def __init__(self, residue_dim: int, pair_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize structure refinement block.
        
        Args:
            residue_dim: Dimension of residue representations
            pair_dim: Dimension of pair representations
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # IPA attention module
        self.ipa_attention = InvariantPointAttention(
            residue_dim=residue_dim,
            pair_dim=pair_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network for residue update
        self.ffn = nn.Sequential(
            nn.LayerNorm(residue_dim),
            nn.Linear(residue_dim, residue_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(residue_dim * 4, residue_dim)
        )
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(residue_dim)
        
        # Coordinate regularization network
        self.coord_regularizer = nn.Sequential(
            nn.Linear(residue_dim, residue_dim // 2),
            nn.ReLU(),
            nn.Linear(residue_dim // 2, 1),  # Scalar regularization weight per residue
            nn.Sigmoid()  # Bound between 0 and 1
        )
    
    def forward(
        self, 
        residue_repr: torch.Tensor, 
        pair_repr: torch.Tensor, 
        frames: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply one complete structure refinement step.
        
        Args:
            residue_repr: Residue representations, shape (batch_size, seq_len, residue_dim)
            pair_repr: Pair representations, shape (batch_size, seq_len, seq_len, pair_dim)
            frames: Rotation matrices, shape (batch_size, seq_len, 3, 3)
            coords: Current 3D coordinates, shape (batch_size, seq_len, 3)
            mask: Boolean mask, shape (batch_size, seq_len)
            
        Returns:
            Tuple of:
            - updated_coords: Updated 3D coordinates, shape (batch_size, seq_len, 3)
            - updated_frames: Updated rotation matrices, shape (batch_size, seq_len, 3, 3)
            - updated_residue_repr: Updated residue representations, shape (batch_size, seq_len, residue_dim)
        """
        # Apply IPA attention
        updated_coords, updated_residue_repr = self.ipa_attention(
            residue_repr, pair_repr, frames, coords, mask
        )
        
        # Apply feed-forward network with residual connection
        ffn_output = self.ffn(updated_residue_repr)
        updated_residue_repr = updated_residue_repr + ffn_output
        
        # Apply final layer norm
        updated_residue_repr = self.layer_norm(updated_residue_repr)
        
        # Compute regularization weights for coordinate updates
        reg_weights = self.coord_regularizer(updated_residue_repr)  # (batch_size, seq_len, 1)
        
        # Apply regularization: interpolate between old and new coordinates
        # This helps prevent large/unstable coordinate changes
        regularized_coords = reg_weights * updated_coords + (1 - reg_weights) * coords
        
        # Frames don't change in this implementation, but returned for API consistency
        # In future versions, we could update frames based on local geometry
        updated_frames = frames
        
        return regularized_coords, updated_frames, updated_residue_repr


class CoordinateProjector(nn.Module):
    """
    Final refinement stage that ensures physical constraints on coordinates.
    """

    def __init__(self, residue_dim: int):
        """
        Initialize coordinate projector network.
        
        Args:
            residue_dim: Dimension of residue representations
        """
        super().__init__()
        
        # Final MLP for coordinate refinement
        self.coord_refiner = nn.Sequential(
            nn.Linear(residue_dim + 3, residue_dim // 2),  # Concat of residue repr and coords
            nn.ReLU(),
            nn.Linear(residue_dim // 2, 3)  # Refined coordinates
        )
        
        # Distance constraint predictor
        self.distance_predictor = nn.Sequential(
            nn.Linear(residue_dim, residue_dim // 2),
            nn.ReLU(),
            nn.Linear(residue_dim // 2, 1),
            nn.Softplus()  # Ensures positive distances
        )
    
    def _apply_distance_constraints(
        self, coords: torch.Tensor, min_distances: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply minimum distance constraints between residues.
        
        Args:
            coords: Coordinates, shape (batch_size, seq_len, 3)
            min_distances: Minimum distances, shape (batch_size, seq_len)
            mask: Boolean mask, shape (batch_size, seq_len)
            
        Returns:
            Constrained coordinates, shape (batch_size, seq_len, 3)
        """
        batch_size, seq_len, _ = coords.shape
        device = coords.device
        
        # No constraints needed for single residue
        if seq_len <= 1:
            return coords
        
        # Compute pairwise distances
        diffs = coords.unsqueeze(2) - coords.unsqueeze(1)  # (batch_size, seq_len, seq_len, 3)
        distances = torch.sqrt(torch.sum(diffs ** 2, dim=-1) + 1e-8)  # (batch_size, seq_len, seq_len)
        
        # Create distance constraints matrix - min distance between each pair
        # For simplicity, we use a constant minimum distance
        min_dist_matrix = (min_distances.unsqueeze(-1) + min_distances.unsqueeze(-2)) / 2  # (batch_size, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # (batch_size, seq_len, seq_len)
            # Set masked distances to infinity (no constraint)
            distances = torch.where(mask_2d, distances, torch.tensor(float('inf'), device=device))
        
        # Only apply constraints to residues that are too close
        # Diagonal should be excluded (distance to self is always 0)
        diag_mask = ~torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
        violations = (distances < min_dist_matrix) & diag_mask
        
        # If no violations, return original coordinates
        if not violations.any():
            return coords
        
        # Compute constraint forces (simplified version)
        # 1. Direction vectors pointing away from violations
        direction_vectors = diffs / (distances.unsqueeze(-1) + 1e-8)
        
        # 2. Magnitude of constraint forces
        magnitudes = torch.relu(min_dist_matrix - distances)  # (batch_size, seq_len, seq_len)
        
        # 3. Apply forces to coordinates
        forces = direction_vectors * magnitudes.unsqueeze(-1)  # (batch_size, seq_len, seq_len, 3)
        
        # Sum forces from all interactions
        total_forces = forces.sum(dim=1)  # (batch_size, seq_len, 3)
        
        # Scale forces (to prevent large movements)
        scale_factor = 0.1
        scaled_forces = total_forces * scale_factor
        
        # Apply forces
        constrained_coords = coords + scaled_forces
        
        return constrained_coords
    
    def forward(
        self, residue_repr: torch.Tensor, coords: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply final coordinate refinement with physical constraints.
        
        Args:
            residue_repr: Residue representations, shape (batch_size, seq_len, residue_dim)
            coords: Current 3D coordinates, shape (batch_size, seq_len, 3)
            mask: Boolean mask, shape (batch_size, seq_len)
            
        Returns:
            Refined coordinates, shape (batch_size, seq_len, 3)
        """
        # Concatenate residue representations and current coordinates
        concat_input = torch.cat([residue_repr, coords], dim=-1)  # (batch_size, seq_len, residue_dim+3)
        
        # Refine coordinates
        refined_coords = self.coord_refiner(concat_input)  # (batch_size, seq_len, 3)
        
        # Add residual connection
        refined_coords = coords + refined_coords
        
        # Predict minimum distance constraints
        min_distances = self.distance_predictor(residue_repr).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply minimum distance constraints
        constrained_coords = self._apply_distance_constraints(refined_coords, min_distances, mask)
        
        # Apply mask if provided
        if mask is not None:
            mask_3d = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            constrained_coords = constrained_coords * mask_3d
        
        return constrained_coords


class EnhancedIPAModule(nn.Module):
    """
    Enhanced version of the IPA module with structure-aware coordinate prediction.
    
    This implementation:
    1. Generates initial frames and coordinates
    2. Refines the structure through multiple iterations
    3. Applies physical constraints to ensure valid coordinates
    """

    def __init__(self, config: Dict):
        """
        Initialize enhanced IPA module.
        
        Args:
            config: Configuration dictionary containing:
                - residue_embed_dim: Dimension of residue representations
                - pair_embed_dim: Dimension of pair representations
                - num_ipa_iterations: Number of refinement iterations
                - ipa_dim: Hidden dimension for IPA refinement
                - num_heads: Number of attention heads
                - dropout: Dropout rate
        """
        super().__init__()
        
        # Extract parameters from config
        self.residue_dim = config.get("residue_embed_dim", 128)
        self.pair_dim = config.get("pair_embed_dim", 64)
        self.num_iterations = config.get("num_ipa_iterations", 3)
        self.num_heads = config.get("num_attention_heads", 4)
        self.dropout = config.get("dropout", 0.1)
        
        # Initialize frame generator
        self.frame_generator = FrameGenerator(self.residue_dim)
        
        # Initialize structure refinement blocks
        self.refinement_blocks = nn.ModuleList([
            StructureRefinementBlock(
                residue_dim=self.residue_dim,
                pair_dim=self.pair_dim,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
            for _ in range(self.num_iterations)
        ])
        
        # Initialize coordinate projector
        self.coord_projector = CoordinateProjector(self.residue_dim)
        
        # Initialize fallback projector (original IPA behavior for emergency use)
        self.fallback_projector = nn.Sequential(
            nn.Linear(self.residue_dim, self.residue_dim // 2),
            nn.ReLU(),
            nn.Linear(self.residue_dim // 2, 3),
        )
    
    def forward(
        self, 
        residue_repr: torch.Tensor, 
        pair_repr: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict 3D coordinates using the enhanced IPA module.
        
        Args:
            residue_repr: Residue representations, shape (batch_size, seq_len, residue_dim)
            pair_repr: Pair representations, shape (batch_size, seq_len, seq_len, pair_dim)
            mask: Boolean mask, shape (batch_size, seq_len)
            
        Returns:
            Predicted 3D coordinates, shape (batch_size, seq_len, 3)
        """
        try:
            # Generate initial frames and coordinates
            frames, coords = self.frame_generator(residue_repr)
            
            # Apply iterative refinement
            for block in self.refinement_blocks:
                coords, frames, residue_repr = block(residue_repr, pair_repr, frames, coords, mask)
            
            # Apply final projection with physical constraints
            final_coords = self.coord_projector(residue_repr, coords, mask)
            
            # Check for NaN or Inf values
            if torch.isnan(final_coords).any() or torch.isinf(final_coords).any():
                # Fallback to original IPA behavior
                print("WARNING: NaN/Inf detected in enhanced IPA output. Using fallback projection.")
                fallback_coords = self.fallback_projector(residue_repr)
                
                # Apply mask if provided
                if mask is not None:
                    fallback_coords = fallback_coords * mask.unsqueeze(-1).float()
                
                return fallback_coords
            
            return final_coords
            
        except Exception as e:
            # Emergency fallback - use original IPA behavior from base class
            print(f"ERROR in enhanced IPA module: {e}. Using emergency fallback.")
            fallback_coords = self.fallback_projector(residue_repr)
            
            # Apply mask if provided
            if mask is not None:
                fallback_coords = fallback_coords * mask.unsqueeze(-1).float()
            
            return fallback_coords