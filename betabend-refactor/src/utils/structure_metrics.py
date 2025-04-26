"""
Structure Evaluation Metrics for RNA 3D Structure Prediction

This module implements metrics for evaluating predicted RNA 3D structures against 
reference structures, including RMSD (Root-Mean-Square Deviation) and TM-score
(Template-Modeling score).
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def compute_rmsd(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    aligned: bool = True,
    epsilon: float = 1e-8,
    max_rmsd: float = 100.0,  # Maximum RMSD value (Å) to return
) -> torch.Tensor:
    """
    Calculate the Root Mean Square Deviation (RMSD) between predicted and true coordinates.
    
    RMSD measures the average distance between atoms in the predicted structure and the
    corresponding atoms in the reference structure after optimal superposition.
    
    This implementation uses MDAnalysis's QCP algorithm for optimal alignment and RMSD
    calculation, which is more robust and efficient than Kabsch, especially for
    handling rotations and reflections.
    
    Args:
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3) or (seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3) or (seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len) or (seq_len), True for valid positions
        aligned: Whether to optimally align the structures before RMSD calculation
        epsilon: Small constant for numerical stability
        max_rmsd: Maximum RMSD value to return (prevents extreme outliers)
        
    Returns:
        RMSD value(s), shape (batch_size) or scalar
    """
    try:
        # Import MDAnalysis for QCP algorithm-based RMSD calculation
        from MDAnalysis.analysis import rms
    except ImportError:
        logging.error("MDAnalysis is not installed. Falling back to PyTorch implementation.")
        # Use the legacy Kabsch-based RMSD implementation as fallback
        return _compute_rmsd_legacy(pred_coords, true_coords, mask, aligned, epsilon, max_rmsd)
    
    # Add batch dimension if not present
    single_input = False
    if len(pred_coords.shape) == 2:
        single_input = True
        pred_coords = pred_coords.unsqueeze(0)
        true_coords = true_coords.unsqueeze(0)
        if mask is not None and len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
    
    device = pred_coords.device
    dtype = pred_coords.dtype
    batch_size, seq_len, _ = pred_coords.shape
    
    # Handle shape mismatches
    if true_coords.shape[1] != seq_len:
        min_len = min(seq_len, true_coords.shape[1])
        logging.warning(
            f"Sequence length mismatch in RMSD: pred={seq_len}, true={true_coords.shape[1]}, using {min_len}"
        )
        pred_coords = pred_coords[:, :min_len, :]
        true_coords = true_coords[:, :min_len, :]
        if mask is not None:
            mask = mask[:, :min_len]
            
    # Handle 4D tensor for true_coords (batch_size, seq_len, seq_len, 3)
    if len(true_coords.shape) == 4:
        # Extract diagonal entries (i==j) to get (batch_size, seq_len, 3)
        batch_size, seq_len1, seq_len2, coords_dim = true_coords.shape
        if seq_len1 != seq_len2:
            logging.warning(f"Unexpected true_coords shape in RMSD: {true_coords.shape}")
        
        # Create indices for the diagonal
        diag_indices = torch.arange(min(seq_len1, seq_len2), device=device)
        # Extract diagonal for each batch
        true_coords_diag = true_coords[:, diag_indices, diag_indices, :]
        true_coords = true_coords_diag
        
        # Update seq_len based on extracted diagonal
        seq_len = min(seq_len1, seq_len2)
        pred_coords = pred_coords[:, :seq_len, :]
    
    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Handle NaN inputs in coordinates
    if torch.isnan(pred_coords).any() or torch.isnan(true_coords).any():
        logging.warning("NaN values detected in input coordinates for RMSD calculation")
        # Replace NaNs with zeros only for RMSD calculation
        pred_coords = torch.nan_to_num(pred_coords, nan=0.0)
        true_coords = torch.nan_to_num(true_coords, nan=0.0)
        # Update mask to exclude positions with NaNs
        if torch.isnan(pred_coords).any():
            has_nan_pred = torch.isnan(pred_coords).any(dim=-1)
            mask = mask & (~has_nan_pred)
        if torch.isnan(true_coords).any():
            has_nan_true = torch.isnan(true_coords).any(dim=-1)
            mask = mask & (~has_nan_true)
    
    # Initialize results tensor
    rmsd_values = torch.zeros(batch_size, dtype=dtype, device=device)
    
    for b in range(batch_size):
        valid_mask = mask[b]
        valid_count = valid_mask.sum().item()
        
        if valid_count < 3:  # Need at least 3 points for meaningful RMSD
            logging.warning(f"Batch {b} has fewer than 3 valid points for RMSD calculation")
            rmsd_values[b] = float('nan')
            continue
            
        # Extract valid coordinates
        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]
        
        # Special case for identical structures - fast path
        if torch.allclose(p_valid, t_valid, atol=1e-8):
            rmsd_values[b] = torch.tensor(0.0, dtype=dtype, device=device)
            continue
        
        # Special case handling for degenerate/coincident points
        p_centered = p_valid - p_valid.mean(dim=0, keepdim=True)
        t_centered = t_valid - t_valid.mean(dim=0, keepdim=True)
        
        is_degenerate_pred = torch.allclose(p_centered, torch.zeros_like(p_centered), atol=1e-5)
        is_degenerate_true = torch.allclose(t_centered, torch.zeros_like(t_centered), atol=1e-5)
        
        # If both are degenerate (all points coincident), just compare centers
        if is_degenerate_pred and is_degenerate_true:
            center_distance = torch.norm(p_valid.mean(dim=0) - t_valid.mean(dim=0))
            rmsd_values[b] = center_distance
            continue
        
        # If only one is degenerate, this is not a meaningful comparison
        # Return a high but finite RMSD
        if (is_degenerate_pred and not is_degenerate_true) or (not is_degenerate_pred and is_degenerate_true):
            logging.warning(f"Batch {b} has degenerate points in only one structure - poor alignment expected")
            rmsd_values[b] = torch.tensor(max_rmsd * 0.75, dtype=dtype, device=device)
            continue
            
        try:
            # Convert to numpy for MDAnalysis
            p_valid_np = p_valid.detach().cpu().numpy()
            t_valid_np = t_valid.detach().cpu().numpy()
            
            # Calculate RMSD using MDAnalysis QCP algorithm
            if aligned:
                # With alignment (centered and rotated to minimize RMSD)
                rmsd = rms.rmsd(p_valid_np, t_valid_np, center=True, superposition=True)
            else:
                # Without alignment (just centered)
                rmsd = rms.rmsd(p_valid_np, t_valid_np, center=True, superposition=False)
            
            # Convert result back to torch tensor and right device/dtype
            rmsd = torch.tensor(rmsd, dtype=dtype, device=device)
            
            # Clamp to maximum reasonable value
            rmsd = torch.clamp(rmsd, max=max_rmsd)
            
            rmsd_values[b] = rmsd
            
        except Exception as e:
            logging.error(f"Error in RMSD calculation with MDAnalysis for batch {b}: {e}")
            # Fall back to legacy implementation for this batch
            try:
                from src.losses import stable_kabsch_align, robust_distance_calculation
                
                if aligned:
                    p_aligned = stable_kabsch_align(p_valid, t_valid, epsilon=epsilon)
                else:
                    p_aligned = p_valid
                
                sq_distances = robust_distance_calculation(p_aligned, t_valid, epsilon=epsilon) ** 2
                mean_sq_dist = torch.mean(sq_distances)
                rmsd = torch.sqrt(mean_sq_dist)
                rmsd = torch.clamp(rmsd, max=max_rmsd)
                
                if torch.isnan(rmsd) or torch.isinf(rmsd):
                    rmsd = torch.tensor(max_rmsd, dtype=dtype, device=device)
                
                rmsd_values[b] = rmsd
                logging.warning(f"Used legacy fallback for batch {b}, result: {rmsd.item():.4f}")
                
            except Exception as inner_e:
                logging.error(f"Both RMSD methods failed for batch {b}: {inner_e}")
                rmsd_values[b] = torch.tensor(max_rmsd, dtype=dtype, device=device)
    
    # Return scalar if input was not batched
    if single_input:
        return rmsd_values[0]
    else:
        return rmsd_values


def _compute_rmsd_legacy(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    aligned: bool = True,
    epsilon: float = 1e-8,
    max_rmsd: float = 100.0,
) -> torch.Tensor:
    """
    Legacy implementation of RMSD calculation using PyTorch and Kabsch algorithm.
    
    This is kept as a fallback in case MDAnalysis is not installed or fails.
    """
    # Add batch dimension if not present
    if len(pred_coords.shape) == 2:
        pred_coords = pred_coords.unsqueeze(0)
        true_coords = true_coords.unsqueeze(0)
        if mask is not None and len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
    
    device = pred_coords.device
    dtype = pred_coords.dtype
    batch_size, seq_len, _ = pred_coords.shape
    
    # Handle shape mismatches
    if true_coords.shape[1] != seq_len:
        min_len = min(seq_len, true_coords.shape[1])
        logging.warning(
            f"Sequence length mismatch in RMSD: pred={seq_len}, true={true_coords.shape[1]}, using {min_len}"
        )
        pred_coords = pred_coords[:, :min_len, :]
        true_coords = true_coords[:, :min_len, :]
        if mask is not None:
            mask = mask[:, :min_len]
            
    # Handle 4D tensor for true_coords (batch_size, seq_len, seq_len, 3)
    if len(true_coords.shape) == 4:
        # Extract diagonal entries (i==j) to get (batch_size, seq_len, 3)
        batch_size, seq_len1, seq_len2, coords_dim = true_coords.shape
        if seq_len1 != seq_len2:
            logging.warning(f"Unexpected true_coords shape in RMSD: {true_coords.shape}")
        
        # Create indices for the diagonal
        diag_indices = torch.arange(min(seq_len1, seq_len2), device=device)
        # Extract diagonal for each batch
        true_coords_diag = true_coords[:, diag_indices, diag_indices, :]
        true_coords = true_coords_diag
        
        # Update seq_len based on extracted diagonal
        seq_len = min(seq_len1, seq_len2)
        pred_coords = pred_coords[:, :seq_len, :]
    
    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Handle NaN inputs in coordinates
    if torch.isnan(pred_coords).any() or torch.isnan(true_coords).any():
        logging.warning("NaN values detected in input coordinates for RMSD calculation")
        # Replace NaNs with zeros only for RMSD calculation
        pred_coords = torch.nan_to_num(pred_coords, nan=0.0)
        true_coords = torch.nan_to_num(true_coords, nan=0.0)
        # Update mask to exclude positions with NaNs
        if torch.isnan(pred_coords).any():
            has_nan_pred = torch.isnan(pred_coords).any(dim=-1)
            mask = mask & (~has_nan_pred)
        if torch.isnan(true_coords).any():
            has_nan_true = torch.isnan(true_coords).any(dim=-1)
            mask = mask & (~has_nan_true)
    
    # Initialize results tensor
    rmsd_values = torch.zeros(batch_size, dtype=dtype, device=device)
    
    for b in range(batch_size):
        valid_mask = mask[b]
        valid_count = valid_mask.sum().item()
        
        if valid_count < 3:  # Need at least 3 points for meaningful RMSD
            logging.warning(f"Batch {b} has fewer than 3 valid points for RMSD calculation")
            rmsd_values[b] = float('nan')
            continue
            
        # Extract valid coordinates
        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]
        
        # Special case for identical structures - fast path
        if torch.allclose(p_valid, t_valid, atol=1e-8):
            rmsd_values[b] = torch.tensor(0.0, dtype=dtype, device=device)
            continue
        
        # Special case handling for degenerate/coincident points
        p_centered = p_valid - p_valid.mean(dim=0, keepdim=True)
        t_centered = t_valid - t_valid.mean(dim=0, keepdim=True)
        
        is_degenerate_pred = torch.allclose(p_centered, torch.zeros_like(p_centered), atol=1e-5)
        is_degenerate_true = torch.allclose(t_centered, torch.zeros_like(t_centered), atol=1e-5)
        
        # If both are degenerate (all points coincident), just compare centers
        if is_degenerate_pred and is_degenerate_true:
            center_distance = torch.norm(p_valid.mean(dim=0) - t_valid.mean(dim=0))
            rmsd_values[b] = center_distance
            continue
        
        # If only one is degenerate, this is not a meaningful comparison
        # Return a high but finite RMSD
        if (is_degenerate_pred and not is_degenerate_true) or (not is_degenerate_pred and is_degenerate_true):
            logging.warning(f"Batch {b} has degenerate points in only one structure - poor alignment expected")
            rmsd_values[b] = torch.tensor(max_rmsd * 0.75, dtype=dtype, device=device)
            continue
            
        try:
            if aligned:
                # Use the stable Kabsch alignment from losses.py
                from src.losses import stable_kabsch_align
                p_aligned = stable_kabsch_align(p_valid, t_valid, epsilon=epsilon)
            else:
                # Use coordinates as is
                p_aligned = p_valid
                
            # Use robust distance calculation for squared differences
            from src.losses import robust_distance_calculation
            sq_distances = robust_distance_calculation(p_aligned, t_valid, epsilon=epsilon) ** 2
            
            # Mean of squared distances
            mean_sq_dist = torch.mean(sq_distances)
            
            # Take sqrt of mean squared distances
            rmsd = torch.sqrt(mean_sq_dist)
            
            # Clamp to maximum reasonable value
            rmsd = torch.clamp(rmsd, max=max_rmsd)
            
            # Final validity check
            if torch.isnan(rmsd) or torch.isinf(rmsd):
                logging.error(f"NaN/Inf in RMSD calculation for batch {b} despite safeguards")
                rmsd = torch.tensor(max_rmsd, dtype=dtype, device=device)
            
            rmsd_values[b] = rmsd
            
        except Exception as e:
            logging.error(f"Error in RMSD calculation for batch {b}: {e}")
            rmsd_values[b] = torch.tensor(max_rmsd, dtype=dtype, device=device)
    
    # Return scalar if batch size is 1
    if batch_size == 1:
        return rmsd_values[0]
    else:
        return rmsd_values


def compute_tm_score(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    d0_scale: float = 1.24,
    d0_offset: float = -1.8,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Calculate the Template-Modeling score (TM-score) between predicted and true coordinates.
    
    TM-score is a measure of similarity between two protein/RNA structures with values in (0,1].
    Higher values indicate better alignment. The score is normalized to avoid length dependency.
    
    Args:
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3) or (seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3) or (seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len) or (seq_len), True for valid positions
        d0_scale: Scale factor for normalization parameter d0
        d0_offset: Offset for normalization parameter d0
        epsilon: Small constant for numerical stability
        
    Returns:
        TM-score value(s), shape (batch_size) or scalar
    """
    # Add batch dimension if not present
    if len(pred_coords.shape) == 2:
        pred_coords = pred_coords.unsqueeze(0)
        true_coords = true_coords.unsqueeze(0)
        if mask is not None and len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
    
    device = pred_coords.device
    dtype = pred_coords.dtype
    batch_size, seq_len, _ = pred_coords.shape
    
    # Handle shape mismatches
    if true_coords.shape[1] != seq_len:
        min_len = min(seq_len, true_coords.shape[1])
        logging.warning(
            f"Sequence length mismatch in TM-score: pred={seq_len}, true={true_coords.shape[1]}, using {min_len}"
        )
        pred_coords = pred_coords[:, :min_len, :]
        true_coords = true_coords[:, :min_len, :]
        if mask is not None:
            mask = mask[:, :min_len]
            
    # Handle 4D tensor for true_coords (batch_size, seq_len, seq_len, 3)
    if len(true_coords.shape) == 4:
        # Extract diagonal entries (i==j) to get (batch_size, seq_len, 3)
        batch_size, seq_len1, seq_len2, coords_dim = true_coords.shape
        if seq_len1 != seq_len2:
            logging.warning(f"Unexpected true_coords shape in TM-score: {true_coords.shape}")
        
        # Create indices for the diagonal
        diag_indices = torch.arange(min(seq_len1, seq_len2), device=device)
        # Extract diagonal for each batch
        true_coords_diag = true_coords[:, diag_indices, diag_indices, :]
        true_coords = true_coords_diag
        
        # Update seq_len based on extracted diagonal
        seq_len = min(seq_len1, seq_len2)
        pred_coords = pred_coords[:, :seq_len, :]
    
    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Initialize results tensor
    tm_scores = torch.zeros(batch_size, dtype=dtype, device=device)
    
    for b in range(batch_size):
        valid_mask = mask[b]
        valid_count = valid_mask.sum().item()
        
        if valid_count < 3:  # Need at least 3 points for meaningful TM-score
            logging.warning(f"Batch {b} has fewer than 3 valid points for TM-score calculation")
            tm_scores[b] = float('nan')
            continue
            
        # Extract valid coordinates
        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]
        
        # Calculate normalization parameter d0 based on sequence length
        # Formula from Yang & Skolnick (2004)
        L = valid_count
        d0 = d0_scale * (L ** (1.0/3.0)) + d0_offset
        d0 = max(d0, 0.5)  # Lower bound
        
        # Use the stable Kabsch alignment from losses.py
        from src.losses import stable_kabsch_align
        p_aligned = stable_kabsch_align(p_valid, t_valid, epsilon=epsilon)
        
        # Handle direct checks for identical tensors
        if torch.allclose(p_valid, t_valid, atol=1e-8):
            tm_scores[b] = 1.0  # Perfect score for identical structures
            continue
            
        # Compute distance squared for each atom pair
        squared_diff = torch.sum((p_aligned - t_valid) ** 2, dim=-1)
        
        # Calculate the TM-score term for each atom
        tm_terms = 1.0 / (1.0 + squared_diff / (d0 ** 2))
        
        # Calculate TM-score as the mean over all atoms, normalized by length
        tm_score = torch.sum(tm_terms) / L
        tm_scores[b] = tm_score
    
    # Return scalar if batch size is 1
    if batch_size == 1:
        return tm_scores[0]
    else:
        return tm_scores


def compute_structure_metrics(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    metrics: List[str] = ["rmsd", "tm_score"],
) -> Dict[str, torch.Tensor]:
    """
    Compute multiple structure evaluation metrics in a single pass.
    
    Args:
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3) or (seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3) or (seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len) or (seq_len), True for valid positions
        metrics: List of metrics to compute, options: ["rmsd", "tm_score"]
        
    Returns:
        Dictionary of computed metrics
    """
    results = {}
    
    if "rmsd" in metrics:
        results["rmsd"] = compute_rmsd(pred_coords, true_coords, mask)
    
    if "tm_score" in metrics:
        results["tm_score"] = compute_tm_score(pred_coords, true_coords, mask)
    
    return results


def compute_per_residue_rmsd(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    aligned: bool = True,
    window_size: int = 1,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Calculate per-residue RMSD between predicted and true coordinates.
    
    This can be used to identify regions with high deviation.
    
    Args:
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3) or (seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3) or (seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len) or (seq_len), True for valid positions
        aligned: Whether to optimally align the structures before RMSD calculation
        window_size: Size of window for local RMSD calculation (1 for per-residue)
        epsilon: Small constant for numerical stability
        
    Returns:
        Per-residue RMSD values, shape (batch_size, seq_len) or (seq_len)
    """
    # Add batch dimension if not present
    if len(pred_coords.shape) == 2:
        pred_coords = pred_coords.unsqueeze(0)
        true_coords = true_coords.unsqueeze(0)
        if mask is not None and len(mask.shape) == 1:
            mask = mask.unsqueeze(0)
    
    device = pred_coords.device
    dtype = pred_coords.dtype
    batch_size, seq_len, _ = pred_coords.shape
    
    # Handle shape mismatches
    if true_coords.shape[1] != seq_len:
        min_len = min(seq_len, true_coords.shape[1])
        logging.warning(
            f"Sequence length mismatch in per-residue RMSD: pred={seq_len}, true={true_coords.shape[1]}, using {min_len}"
        )
        pred_coords = pred_coords[:, :min_len, :]
        true_coords = true_coords[:, :min_len, :]
        if mask is not None:
            mask = mask[:, :min_len]
            
    # Handle 4D tensor for true_coords (batch_size, seq_len, seq_len, 3)
    if len(true_coords.shape) == 4:
        # Extract diagonal entries (i==j) to get (batch_size, seq_len, 3)
        batch_size, seq_len1, seq_len2, coords_dim = true_coords.shape
        if seq_len1 != seq_len2:
            logging.warning(f"Unexpected true_coords shape in per-residue RMSD: {true_coords.shape}")
        
        # Create indices for the diagonal
        diag_indices = torch.arange(min(seq_len1, seq_len2), device=device)
        # Extract diagonal for each batch
        true_coords_diag = true_coords[:, diag_indices, diag_indices, :]
        true_coords = true_coords_diag
        
        # Update seq_len based on extracted diagonal
        seq_len = min(seq_len1, seq_len2)
        pred_coords = pred_coords[:, :seq_len, :]
    
    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Initialize results tensor
    per_residue_rmsd = torch.full((batch_size, seq_len), float('nan'), dtype=dtype, device=device)
    
    for b in range(batch_size):
        valid_mask = mask[b]
        valid_count = valid_mask.sum().item()
        
        if valid_count < 3:  # Need at least 3 points for meaningful RMSD
            logging.warning(f"Batch {b} has fewer than 3 valid points for per-residue RMSD calculation")
            continue
            
        # Extract valid coordinates for global alignment
        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]
        
        # Special case for identical structures
        if torch.allclose(p_valid, t_valid, atol=1e-8):
            per_residue_rmsd[b, valid_mask] = torch.zeros(valid_mask.sum(), dtype=dtype, device=device)
            continue
        
        if aligned:
            # Use the stable Kabsch alignment from losses.py
            from src.losses import stable_kabsch_align
            p_aligned_all = stable_kabsch_align(p_valid, t_valid, epsilon=epsilon)
            
            # Map aligned coordinates back to full tensor 
            p_aligned = torch.zeros_like(pred_coords[b])
            p_aligned[valid_mask] = p_aligned_all
        else:
            # Use coordinates as is
            p_aligned = pred_coords[b]
        
        # For each residue with valid mask
        for i in range(seq_len):
            if not valid_mask[i]:
                continue
                
            # For window_size=1, just calculate single residue deviation
            if window_size == 1:
                squared_diff = torch.sum((p_aligned[i] - true_coords[b, i]) ** 2)
                per_residue_rmsd[b, i] = torch.sqrt(squared_diff + epsilon)
            else:
                # Calculate local RMSD within a window
                start_idx = max(0, i - window_size // 2)
                end_idx = min(seq_len, i + window_size // 2 + 1)
                
                # Filter by valid mask within window
                window_mask = valid_mask[start_idx:end_idx]
                if window_mask.sum() < 1:
                    continue
                    
                # Calculate RMSD for window
                p_window = p_aligned[start_idx:end_idx][window_mask]
                t_window = true_coords[b, start_idx:end_idx][window_mask]
                
                squared_diff = torch.sum((p_window - t_window) ** 2, dim=1)
                window_rmsd = torch.sqrt(torch.mean(squared_diff) + epsilon)
                per_residue_rmsd[b, i] = window_rmsd
    
    # Return without batch dimension if input didn't have it
    if pred_coords.shape[0] == 1 and len(pred_coords.shape) == 3:
        return per_residue_rmsd[0]
    else:
        return per_residue_rmsd