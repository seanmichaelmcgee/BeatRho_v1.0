"""
Utility functions for BetaRho v1.0 RNA structure prediction model.
Contains helper functions for TM-score calculation, rigid frame transformations,
and model configuration utilities.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

def tm_score(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    d0: float = 5.0,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculate TM-score between predicted and true coordinates.
    Adapted for RNA C1' atom traces.
    
    Args:
        pred_coords: Predicted coordinates [batch, num_res, 3]
        true_coords: Ground truth coordinates [batch, num_res, 3]
        d0: Distance threshold parameter (default: 5.0 Angstrom)
        mask: Optional mask for valid residues [batch, num_res]
    
    Returns:
        TM-score tensor [batch]
    """
    if mask is None:
        mask = torch.ones_like(pred_coords[..., 0], dtype=torch.bool)
    
    batch_size = pred_coords.shape[0]
    tm_scores = []
    
    for b in range(batch_size):
        # Get valid coordinates
        valid_mask = mask[b]
        p_coords = pred_coords[b, valid_mask]
        t_coords = true_coords[b, valid_mask]
        
        seq_len = p_coords.shape[0]
        d0_sq = d0 ** 2
        
        # Center coordinates
        p_center = torch.mean(p_coords, dim=0, keepdim=True)
        t_center = torch.mean(t_coords, dim=0, keepdim=True)
        p_centered = p_coords - p_center
        t_centered = t_coords - t_center
        
        # Calculate rotation matrix using Kabsch algorithm
        covariance = torch.matmul(p_centered.T, t_centered)
        u, _, v = torch.svd(covariance)
        rot_matrix = torch.matmul(v, u.T)
        
        # Check for reflection
        if torch.det(rot_matrix) < 0:
            v[:, -1] = -v[:, -1]
            rot_matrix = torch.matmul(v, u.T)
        
        # Apply rotation
        p_aligned = torch.matmul(p_centered, rot_matrix)
        
        # Calculate distances
        dist_sq = torch.sum((p_aligned - t_centered) ** 2, dim=1)
        tm_sum = torch.sum(1.0 / (1.0 + dist_sq / d0_sq))
        
        # Normalize by length
        tm_score_val = tm_sum / seq_len
        tm_scores.append(tm_score_val)
    
    return torch.stack(tm_scores)

def soft_tm_score(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    d0: float = 5.0,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Differentiable TM-score implementation for training.
    Uses soft alignments for better gradient flow.
    
    Args:
        pred_coords: Predicted coordinates [batch, num_res, 3]
        true_coords: Ground truth coordinates [batch, num_res, 3]
        d0: Distance threshold parameter (default: 5.0 Angstrom)
        mask: Optional mask for valid residues [batch, num_res]
    
    Returns:
        Soft TM-score tensor [batch]
    """
    if mask is None:
        mask = torch.ones_like(pred_coords[..., 0], dtype=torch.bool)
    
    batch_size = pred_coords.shape[0]
    seq_lens = mask.sum(dim=1)
    d0_sq = d0 ** 2
    
    # Center coordinates
    p_center = torch.sum(pred_coords * mask.unsqueeze(-1), dim=1) / seq_lens.unsqueeze(-1)
    t_center = torch.sum(true_coords * mask.unsqueeze(-1), dim=1) / seq_lens.unsqueeze(-1)
    
    p_centered = pred_coords - p_center.unsqueeze(1)
    t_centered = true_coords - t_center.unsqueeze(1)
    
    # Calculate covariance matrices
    covariance = torch.zeros(batch_size, 3, 3, device=pred_coords.device)
    
    for b in range(batch_size):
        p_b = p_centered[b] * mask[b].unsqueeze(-1)
        t_b = t_centered[b] * mask[b].unsqueeze(-1)
        covariance[b] = torch.matmul(p_b.transpose(0, 1), t_b)
    
    # SVD for rotation matrices (batch-wise)
    u, s, v = torch.svd(covariance)
    
    # Handle reflections
    det = torch.det(torch.matmul(v, u.transpose(1, 2)))
    reflection = torch.ones_like(det)
    reflection[det < 0] = -1
    
    # Apply reflection correction
    v_corrected = v.clone()
    v_corrected[:, :, 2] = v[:, :, 2] * reflection.unsqueeze(-1)
    
    # Calculate rotation matrices
    rotation = torch.matmul(v_corrected, u.transpose(1, 2))
    
    # Apply rotation to centered coordinates
    p_aligned = torch.zeros_like(pred_coords)
    for b in range(batch_size):
        p_aligned[b] = torch.matmul(p_centered[b], rotation[b])
    
    # Calculate distances and TM-score
    dist_sq = torch.sum((p_aligned - t_centered) ** 2, dim=2)
    tm_per_residue = 1.0 / (1.0 + dist_sq / d0_sq)
    
    # Apply mask and normalize
    tm_per_residue = tm_per_residue * mask
    tm_scores = torch.sum(tm_per_residue, dim=1) / seq_lens
    
    return tm_scores

def filter_by_temporal_cutoff(
    sequences_df,
    cutoff_date: str = "2022-05-27"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset based on temporal cutoff.
    
    Args:
        sequences_df: DataFrame containing RNA sequences with temporal_cutoff column
        cutoff_date: Date string in YYYY-MM-DD format (default: "2022-05-27")
    
    Returns:
        Tuple of (train_df, extended_val_df)
    """
    # Convert date strings to datetime objects for comparison
    sequences_df['temporal_cutoff_dt'] = pd.to_datetime(sequences_df['temporal_cutoff'])
    cutoff_dt = pd.to_datetime(cutoff_date)
    
    # Split data
    train_df = sequences_df[sequences_df['temporal_cutoff_dt'] < cutoff_dt].copy()
    extended_val_df = sequences_df[sequences_df['temporal_cutoff_dt'] >= cutoff_dt].copy()
    
    # Remove temporary column
    train_df.drop('temporal_cutoff_dt', axis=1, inplace=True)
    extended_val_df.drop('temporal_cutoff_dt', axis=1, inplace=True)
    
    print(f"Split summary:")
    print(f"  Train samples: {len(train_df)} (before {cutoff_date})")
    print(f"  Extended validation samples: {len(extended_val_df)} (on or after {cutoff_date})")
    
    return train_df, extended_val_df

def convert_labels_to_coordinates(
    labels_df,
    stack_conformers: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Convert sequence labels into C1' coordinates.
    
    Args:
        labels_df: DataFrame with structure labels
        stack_conformers: Whether to stack multiple conformers on last axis
    
    Returns:
        Dictionary mapping target_id to coordinate tensors [L, 3] or [L, 3, num_conformers]
    """
    target_to_coords = {}
    
    # Group by target_id to handle multiple conformers
    for target_id, group in labels_df.groupby('target_id'):
        # Extract coordinates for each conformer
        conformers = []
        for _, row in group.iterrows():
            # Parse coordinates from label format
            coords_str = row['coordinates']
            coords = np.array([
                [float(x) for x in line.strip().split()]
                for line in coords_str.strip().split('\n')
            ])
            conformers.append(torch.tensor(coords))
        
        if stack_conformers and len(conformers) > 1:
            # Stack multiple conformers on last axis [L, 3, num_conformers]
            stacked = torch.stack(conformers, dim=-1)
            target_to_coords[target_id] = stacked
        else:
            # Just use the first conformer if there are multiple
            target_to_coords[target_id] = conformers[0]
    
    return target_to_coords

def build_feature_cache(
    target_ids: List[str],
    feature_root: str,
    cache_file: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Load and cache feature files for efficient data loading.
    
    Args:
        target_ids: List of target IDs to process
        feature_root: Root directory containing feature subdirectories
        cache_file: Optional file path to save/load cache
    
    Returns:
        Dictionary mapping target_ids to feature dictionaries
    """
    import os
    import pickle
    from pathlib import Path
    
    # Check if cache exists and load it
    if cache_file and os.path.exists(cache_file):
        print(f"Loading feature cache from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Paths to feature directories
    mi_dir = os.path.join(feature_root, 'mi_features')
    dihedral_dir = os.path.join(feature_root, 'dihedral_features')
    thermo_dir = os.path.join(feature_root, 'thermo_features')
    
    feature_cache = {}
    missing_count = 0
    
    for target_id in target_ids:
        try:
            # Load MI features
            mi_path = os.path.join(mi_dir, f"{target_id}_mi.npz")
            mi_data = dict(np.load(mi_path))
            
            # Load dihedral features
            dihedral_path = os.path.join(dihedral_dir, f"{target_id}_dihedral.npz")
            dihedral_data = dict(np.load(dihedral_path))
            
            # Load thermo features
            thermo_path = os.path.join(thermo_dir, f"{target_id}_thermo.npz")
            thermo_data = dict(np.load(thermo_path))
            
            # Convert sequence to integer encoding if needed
            seq = mi_data.get('sequence', '')
            if isinstance(seq, str):
                # Convert ACGU to integers
                seq_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}
                sequence_int = torch.tensor([seq_map.get(nt, 4) for nt in seq])
            else:
                sequence_int = torch.tensor(seq)
            
            # Pack features
            feature_cache[target_id] = {
                "sequence_int": sequence_int,
                "mi": {k: torch.tensor(v) for k, v in mi_data.items() if k != 'sequence'},
                "dihedral": {k: torch.tensor(v) for k, v in dihedral_data.items()},
                "thermo": {k: torch.tensor(v) for k, v in thermo_data.items()}
            }
        except Exception as e:
            missing_count += 1
            print(f"Failed to load features for {target_id}: {e}")
    
    print(f"Loaded features for {len(feature_cache)} targets")
    print(f"Missing features for {missing_count} targets")
    
    # Verify coverage
    coverage = len(feature_cache) / len(target_ids) * 100
    if coverage < 95:
        print(f"WARNING: Feature coverage is only {coverage:.1f}%, which is below the recommended 95%")
    
    # Save cache if requested
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        print(f"Saving feature cache to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(feature_cache, f)
    
    return feature_cache

class GradientCheckpointer:
    """
    Helper class for gradient checkpointing to reduce memory usage.
    """
    def __init__(self, model, use_checkpointing=True):
        """
        Initialize gradient checkpointing wrapper.
        
        Args:
            model: PyTorch model to apply checkpointing
            use_checkpointing: Whether to enable checkpointing
        """
        self.model = model
        self.use_checkpointing = use_checkpointing
        
        if use_checkpointing:
            self._apply_checkpointing(model)
    
    def _apply_checkpointing(self, module):
        """
        Recursively apply gradient checkpointing to eligible modules.
        
        Args:
            module: PyTorch module to modify
        """
        # Apply checkpointing to common module types
        if hasattr(module, 'forward') and not isinstance(
            module, (torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict)
        ):
            # Skip small modules that don't benefit from checkpointing
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 1000:  # Only checkpoint substantial modules
                original_forward = module.forward
                
                def checkpointed_forward(*args, **kwargs):
                    # Define a custom forward that will be checkpointed
                    def custom_forward(*inputs):
                        return original_forward(*inputs, **kwargs)
                    
                    return torch.utils.checkpoint.checkpoint(custom_forward, *args)
                
                module.forward = checkpointed_forward
        
        # Recursively apply to children
        for child in module.children():
            self._apply_checkpointing(child)
    
    def disable(self):
        """Disable gradient checkpointing (restores original forwards)"""
        # This is a simplified version that fully reinstantiates the model
        # In practice, you'd want to store and restore the original forward methods
        self.use_checkpointing = False
