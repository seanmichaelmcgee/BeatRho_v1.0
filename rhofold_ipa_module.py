"""
RhoFold+ IPA Structure Module Integration

This module implements the integration of the RhoFold+ Invariant-Point-Attention (IPA)
structure module for RNA 3D structure prediction with the existing Betabend RNA Feature-Embedding model.

The integration follows the design outlined in technical_guide.md, incorporating rigid frame
initialization, IPA for coordinate refinement, and angle prediction components.
"""

import sys
import os
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add paths for importing from different repos
current_dir = os.path.dirname(os.path.abspath(__file__))
betabend_dir = os.path.join(current_dir, "betabend-refactor")
rhofold_dir = os.path.join(current_dir, "RhoFold-refactor")

sys.path.append(betabend_dir)
sys.path.append(os.path.join(betabend_dir, "src"))
sys.path.append(rhofold_dir)
sys.path.append(os.path.join(rhofold_dir, "src"))

# Import RhoFold+ components
from model.rhofold_components.primitives import Linear, LayerNorm
from model.rhofold_components.structure_module import InvariantPointAttention, AngleResnet, BackboneUpdate
from utils.rhofold_utils.rigid_utils import Rigid
from utils.rhofold_utils.tensor_utils import dict_multimap

# Import RNA conversion utilities if available
try:
    from utils.converter import RNAConverter
except ImportError:
    # Create a placeholder converter class
    class RNAConverter:
        """
        Placeholder RNA coordinate converter class.
        
        This is a temporary implementation that will be replaced by the actual
        RNAConverter from RhoFold when available.
        """
        
        def __init__(self):
            """Initialize the RNA converter."""
            pass
        
        def build_cords(self, sequences, frames, angles, rtn_cmsk=False):
            """
            Build RNA coordinates from sequences, frames, and angles.
            
            Args:
                sequences: RNA sequences (integer encoding)
                frames: Rigid frames
                angles: Predicted torsion angles
                rtn_cmsk: Whether to return atom mask
                
            Returns:
                Tuple of (coordinates, atom_mask)
            """
            batch_size = sequences.shape[0]
            seq_len = sequences.shape[1]
            device = sequences.device
            
            # Placeholder implementation - in reality, this would use frames and angles
            # to compute actual RNA atom coordinates
            
            # Use translations from frames as C1' coordinates
            if isinstance(frames, torch.Tensor):
                # Assuming frames is a tensor of shape [batch_size, seq_len, 7]
                # where the last 3 dimensions are translations
                c1_coords = frames[..., 4:7]
            else:
                # Assuming frames is a Rigid object
                c1_coords = frames.get_trans()
            
            # Create dummy atom mask
            atom_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
            
            if rtn_cmsk:
                return c1_coords, atom_mask
            else:
                return c1_coords


class RigidFrameInitializer(nn.Module):
    """
    Initialize rigid frames for RNA residues based on embedding representations.
    
    This module converts embedding vectors to initial 3D frames (rotation + translation)
    for the Invariant Point Attention module, using a learnable initialization approach.
    """
    
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None):
        """
        Initialize the frame initializer.
        
        Args:
            embed_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for MLP (defaults to embed_dim//2)
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = embed_dim // 2
            
        # MLP for initial coordinates prediction
        self.coord_initializer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Output (x, y, z) coordinates
        )
        
        # MLP for frame orientation (outputs 6D rotation representation)
        self.orientation_initializer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 6D rotation representation
        )
    
    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[Rigid, torch.Tensor]:
        """
        Convert embeddings to initial rigid frames and coordinates.
        
        Args:
            embeddings: Residue embeddings of shape [batch_size, seq_len, embed_dim]
            mask: Boolean mask of shape [batch_size, seq_len], True for valid positions
            
        Returns:
            Tuple of (rigid_frames, coordinates):
                - rigid_frames: Rigid object containing rotations and translations
                - coordinates: Initial coordinates of shape [batch_size, seq_len, 3]
        """
        batch_size, seq_len, _ = embeddings.shape
        device = embeddings.device
        
        # Default mask if none provided
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        
        # Generate initial coordinates [batch_size, seq_len, 3]
        initial_coords = self.coord_initializer(embeddings)
        
        # Apply mask to coordinates
        mask_3d = mask.unsqueeze(-1).expand(-1, -1, 3)
        initial_coords = initial_coords * mask_3d.float()
        
        # Create identity rigid frames for all positions
        rigid_frames = Rigid.identity(
            (batch_size, seq_len),
            dtype=embeddings.dtype,
            device=device,
            requires_grad=True,
            fmt="quat"
        )
        
        # We use initial_coords as translations for the rigid frames
        # This ensures that the rigid transforms start at the predicted positions
        trans = initial_coords
        
        # Generate 6D rotation representation
        rot_6d = self.orientation_initializer(embeddings)
        
        # Convert 6D representation to rotation matrices
        # We use a simplified 6D→3×3 algorithm that ensures orthogonality
        rot_6d = rot_6d * mask.unsqueeze(-1).expand(-1, -1, 6).float()
        
        # Split into two 3D vectors
        x_basis = rot_6d[..., :3]
        y_basis_raw = rot_6d[..., 3:]
        
        # Normalize x_basis
        x_basis_norm = F.normalize(x_basis, dim=-1)
        
        # Make y_basis orthogonal to x_basis
        y_basis_ortho = y_basis_raw - torch.sum(y_basis_raw * x_basis_norm, dim=-1, keepdim=True) * x_basis_norm
        y_basis = F.normalize(y_basis_ortho, dim=-1)
        
        # Compute z_basis as cross product
        z_basis = torch.cross(x_basis_norm, y_basis, dim=-1)
        
        # Stack to form rotation matrices [batch_size, seq_len, 3, 3]
        rot_mats = torch.stack([x_basis_norm, y_basis, z_basis], dim=-2)
        
        # Create new rigid frames with the computed rotations and translations
        rigid_frames = Rigid(
            rots=rot_mats,
            trans=trans,
        )
        
        return rigid_frames, initial_coords


class RhoFoldIPAModule(nn.Module):
    """
    RhoFold+ IPA structure module for RNA 3D structure prediction.
    
    This module integrates the Invariant Point Attention (IPA) module from RhoFold+ 
    with the Betabend RNA feature-embedding model, allowing for accurate prediction
    of RNA 3D structures using rotation and translation of rigid frames.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RhoFold IPA module.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        
        # Extract parameters from config with defaults
        self.c_s = config.get("residue_embed_dim", 128)  # Single (residue) representation dimension
        self.c_z = config.get("pair_embed_dim", 64)      # Pair representation dimension
        self.c_ipa = config.get("c_ipa", 16)             # IPA hidden channel dimension
        self.c_resnet = config.get("c_resnet", 128)      # Angle resnet hidden dimension
        self.no_heads = config.get("no_heads", 4)        # Number of IPA attention heads
        self.no_qk_points = config.get("no_qk_points", 4) # Number of query/key points
        self.no_v_points = config.get("no_v_points", 8)   # Number of value points
        self.no_blocks = config.get("num_ipa_blocks", 4)  # Number of IPA iterations
        self.no_angles = config.get("no_angles", 7)       # Number of RNA torsion angles
        self.trans_scale_factor = config.get("trans_scale_factor", 10.0)  # Translation scale factor
        self.epsilon = config.get("epsilon", 1e-8)        # Small number for numerical stability
        
        # Dimension adaptation (residue embeddings → IPA input format)
        self.residue_adapter = nn.Linear(self.c_s, 384)  # RhoFold uses 384-dim embeddings
        self.pair_adapter = nn.Linear(self.c_z, 128)     # RhoFold uses 128-dim pair embeddings
        
        # Layer normalization
        self.layer_norm_s = LayerNorm(384)
        self.layer_norm_z = LayerNorm(128)
        
        # Rigid frame initialization
        self.frame_initializer = RigidFrameInitializer(embed_dim=384)
        
        # IPA components
        self.ipa = InvariantPointAttention(
            c_s=384,             # RhoFold's single representation dimension
            c_z=128,             # RhoFold's pair representation dimension
            c_hidden=self.c_ipa,
            no_heads=self.no_heads,
            no_qk_points=self.no_qk_points,
            no_v_points=self.no_v_points,
        )
        
        # Layer norm after IPA
        self.layer_norm_ipa = LayerNorm(384)
        
        # Backbone update module (for updating rigid frames)
        self.bb_update = BackboneUpdate(384)
        
        # Angle resnet for torsion angle prediction
        self.angle_resnet = AngleResnet(
            c_in=384,
            c_hidden=self.c_resnet,
            no_blocks=3,
            no_angles=self.no_angles,
            epsilon=self.epsilon,
        )
        
        # Confidence predictor
        self.confidence_predictor = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # RNA structure converter
        self.converter = RNAConverter()
    
    def forward(
        self, 
        residue_repr: torch.Tensor, 
        pair_repr: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        sequences_int: Optional[torch.Tensor] = None  # Added for sequence info
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the RhoFold IPA module.
        
        Args:
            residue_repr: Per-residue representation [batch_size, seq_len, c_s]
            pair_repr: Pairwise representation [batch_size, seq_len, seq_len, c_z]
            mask: Boolean mask [batch_size, seq_len], True for valid positions
            sequences_int: Integer-encoded RNA sequences [batch_size, seq_len]
            
        Returns:
            Dictionary containing:
                - pred_coords: Predicted C1' coordinates [batch_size, seq_len, 3]
                - pred_angles: Predicted torsion angles [batch_size, seq_len, no_angles, 2]
                - pred_confidence: Predicted confidence scores [batch_size, seq_len]
        """
        batch_size, seq_len, _ = residue_repr.shape
        device = residue_repr.device
        
        # Default mask if none provided
        if mask is None:
            mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
        
        # Dimension adaptation
        s = self.residue_adapter(residue_repr)  # [batch_size, seq_len, 384]
        z = self.pair_adapter(pair_repr)        # [batch_size, seq_len, seq_len, 128]
        
        # Layer normalization
        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        
        # Store initial representation for angle prediction
        s_initial = s
        
        # Initialize rigid frames
        rigids, initial_coords = self.frame_initializer(s, mask)
        
        # Iterative refinement using IPA
        outputs = []
        
        for i in range(self.no_blocks):
            # Update single representation with IPA
            s = s + self.ipa(s, z, rigids, mask)
            s = self.layer_norm_ipa(s)
            
            # Update rigid frames with backbone update
            rigids = rigids.compose_q_update_vec(self.bb_update(s))
            
            # Predict angles
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)
            
            # Store outputs for this iteration
            outputs.append({
                "frames": rigids.to_tensor_7(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "single": s
            })
            
            # Stop gradient flow in rotations for stability
            if i != self.no_blocks - 1:
                rigids = rigids.apply_rot_fn(lambda x: x.detach())
        
        # Stack outputs across iterations
        stacked_outputs = dict_multimap(torch.stack, outputs)
        
        # Generate RNA structure using converter if sequence info is available
        if sequences_int is not None:
            # Convert sequence integers to one-hot encoding for converter
            # Note: The actual implementation would depend on the expected format
            # of the converter; this is a placeholder
            final_coords, atom_mask = self.converter.build_cords(
                sequences_int, 
                stacked_outputs['frames'][-1], 
                stacked_outputs['angles'][-1],
                rtn_cmsk=True
            )
            
            # Extract C1' atoms (assuming index 1 is C1')
            c1_coords = final_coords
        else:
            # If no sequence info, use translations from rigid frames
            # Scale translations appropriately
            scaled_rigids = rigids.scale_translation(self.trans_scale_factor)
            c1_coords = scaled_rigids.get_trans()
        
        # Predict confidence scores
        confidence = torch.sigmoid(self.confidence_predictor(s).squeeze(-1))
        
        # Apply mask to outputs
        mask_float = mask.float()
        mask_3d = mask_float.unsqueeze(-1).expand(-1, -1, 3)
        
        c1_coords = c1_coords * mask_3d
        confidence = confidence * mask_float
        
        # Get angles from last iteration
        final_angles = stacked_outputs['angles'][-1]
        
        return {
            "pred_coords": c1_coords,
            "pred_angles": final_angles,
            "pred_confidence": confidence,
            "all_outputs": stacked_outputs,  # Include all iterations for analysis
        }


class RhoFoldIPAModel(nn.Module):
    """
    Complete RNA structure prediction model integrating the RhoFold+ IPA module
    with the Betabend RNA Feature Embedding model.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the complete model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()
        
        # Extract parameters from config
        self.residue_dim = config.get("residue_embed_dim", 128)
        self.pair_dim = config.get("pair_embed_dim", 64)
        self.num_blocks = config.get("num_blocks", 4)
        
        # Import embedding and transformer modules from betabend
        from src.models.embeddings import EmbeddingModule
        from src.models.transformer_block import TransformerBlock
        
        # Initialize the embedding module
        self.embedding_module = EmbeddingModule(config)
        
        # Initialize transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(self.num_blocks)]
        )
        
        # Initialize RhoFold IPA module
        self.ipa_module = RhoFoldIPAModule(config)
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            batch: Dictionary of input tensors from the data loader
            
        Returns:
            Dictionary of output tensors with predicted coordinates,
            confidence scores, and angles
        """
        # Create initial representations using embedding module
        residue_repr, pair_repr, mask = self.embedding_module(batch)
        # residue_repr: (batch_size, seq_len, residue_dim)
        # pair_repr: (batch_size, seq_len, seq_len, pair_dim)
        
        # Process through transformer blocks
        for block in self.transformer_blocks:
            residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
        
        # Process through RhoFold IPA module
        outputs = self.ipa_module(
            residue_repr=residue_repr,
            pair_repr=pair_repr,
            mask=mask,
            sequences_int=batch.get("sequence_int")
        )
        
        return outputs
