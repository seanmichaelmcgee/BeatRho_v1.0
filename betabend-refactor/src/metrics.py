"""
RNA Structure Quality Assessment Metrics

This module provides metrics for evaluating the quality of predicted RNA structures,
including TM-score (Template Modeling score) adapted for RNA C1' atoms, RMSD,
and related structural quality assessments.
"""

import math
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import stable_kabsch_align, robust_distance_calculation

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_tm_score(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    d0_scale: float = 1.5,  # Scale factor for d0 parameter
    epsilon: float = 1e-8,
    return_components: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Compute TM-score for RNA structures based on C1' atoms.
    
    TM-score is a measure of global structural similarity between two structures,
    ranging from 0 to 1, where 1 indicates a perfect match. It's less sensitive to
    local structural differences than RMSD.
    
    Formula: TM-score = 1/L * sum_i [ 1 / (1 + (d_i/d0)²) ]
    
    Where:
        L is the length of the target structure
        d_i is the distance between the i-th pair of residues after alignment
        d0 = 1.24 * (L - 15)^(1/3) - 1.8 is a scale factor (adapted for RNA)
    
    Args:
        pred_coords: Predicted C1' coordinates of shape (batch_size, seq_len, 3)
        true_coords: Target C1' coordinates of shape (batch_size, seq_len, 3)
        mask: Boolean mask of shape (batch_size, seq_len), True for valid positions
        d0_scale: Scale factor for d0 parameter calculation (default: 1.5)
        epsilon: Small constant for numerical stability
        return_components: If True, return individual terms for analysis
        
    Returns:
        TM-scores of shape (batch_size,) if return_components is False
        Tuple of (TM-scores, components dict) if return_components is True
    """
    # Ensure tensors are on the same device
    device = pred_coords.device
    true_coords = true_coords.to(device)
    if mask is not None:
        mask = mask.to(device)
    
    # Extract batch size and sequence length
    batch_size, seq_len, _ = pred_coords.shape
    
    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
        
    # For empty sequences, return 0.0
    if seq_len == 0:
        if return_components:
            return torch.zeros(batch_size, device=device), {}
        return torch.zeros(batch_size, device=device)
    
    # Initialize TM-score and components for each sequence
    tm_scores = []
    components = {
        "d0_values": [],
        "distances": [],
        "tm_terms": [],
        "normalization": []
    }
    
    # Process each sequence in the batch
    for b in range(batch_size):
        # Extract valid positions using mask
        valid_mask = mask[b]
        n_valid = valid_mask.sum().item()
        
        # Skip if there are too few valid positions
        if n_valid < 3:  # Need at least 3 points for reasonable alignment
            tm_scores.append(torch.tensor(0.0, device=device))
            if return_components:
                for key in components:
                    components[key].append(torch.tensor(0.0, device=device))
            continue
            
        # Extract coordinates for valid positions
        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]
        
        # Calculate d0 parameter - RNA-specific formula
        # Adapted from protein TM-score with parameters tuned for RNA
        # d0 = 1.24 * (L - 15)^(1/3) - 1.8, with minimum value of 0.5
        # Apply scaling factor for adjustment
        L = n_valid
        d0_base = max(0.5, 1.24 * ((L - 15) ** (1/3)) - 1.8)
        d0 = d0_base * d0_scale
        
        # Apply Kabsch alignment to align predicted coordinates to target
        try:
            p_aligned = stable_kabsch_align(p_valid, t_valid, epsilon=epsilon)
            
            # Compute distances between aligned pairs
            distances = robust_distance_calculation(p_aligned, t_valid, epsilon=epsilon)
            
            # Compute TM-score terms: 1 / (1 + (d_i/d0)²)
            d_norm = distances / d0
            tm_terms = 1.0 / (1.0 + d_norm * d_norm)
            
            # Compute TM-score: average of terms normalized by target length
            tm_score = tm_terms.mean()
            
            tm_scores.append(tm_score)
            
            # Store components for analysis if requested
            if return_components:
                components["d0_values"].append(torch.tensor(d0, device=device))
                components["distances"].append(distances)
                components["tm_terms"].append(tm_terms)
                components["normalization"].append(torch.tensor(1.0 / L, device=device))
                
        except Exception as e:
            logger.error(f"Error computing TM-score for batch item {b}: {e}")
            tm_scores.append(torch.tensor(0.0, device=device))
            if return_components:
                for key in components:
                    components[key].append(torch.tensor(0.0, device=device))
    
    # Stack scores into a tensor
    tm_scores_tensor = torch.stack(tm_scores)
    
    if return_components:
        # Process components into tensors
        components_tensors = {}
        for key, values in components.items():
            # Handle different shapes of components
            if all(isinstance(v, torch.Tensor) and v.dim() == 0 for v in values):
                # Scalar components
                components_tensors[key] = torch.stack(values)
            else:
                # Keep as list if tensors have different shapes
                components_tensors[key] = values
                
        return tm_scores_tensor, components_tensors
    
    return tm_scores_tensor


def compute_tm_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    d0_scale: float = 1.5,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute a loss based on 1 - TM-score for optimizing RNA structural predictions.
    
    This loss encourages maximizing the TM-score between predicted and target structures.
    
    Args:
        pred_coords: Predicted C1' coordinates of shape (batch_size, seq_len, 3)
        true_coords: Target C1' coordinates of shape (batch_size, seq_len, 3)
        mask: Boolean mask of shape (batch_size, seq_len), True for valid positions
        d0_scale: Scale factor for d0 parameter calculation (default: 1.5)
        epsilon: Small constant for numerical stability
        
    Returns:
        Loss tensor (batch mean of 1 - TM-score)
    """
    # Compute TM-scores
    tm_scores = compute_tm_score(
        pred_coords, true_coords, mask, d0_scale, epsilon, return_components=False
    )
    
    # Compute loss as 1 - TM-score (average over batch)
    # This converts the score into a loss to be minimized
    loss = 1.0 - tm_scores
    
    return loss.mean()


def compute_rmsd(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    aligned: bool = True,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Root Mean Square Deviation (RMSD) between predicted and true coordinates.
    
    RMSD measures the average distance between atoms of superimposed structures.
    
    Args:
        pred_coords: Predicted coordinates of shape (batch_size, seq_len, 3)
        true_coords: Target coordinates of shape (batch_size, seq_len, 3)
        mask: Boolean mask of shape (batch_size, seq_len), True for valid positions
        aligned: Whether to align structures using Kabsch algorithm before RMSD calculation
        epsilon: Small constant for numerical stability
        
    Returns:
        RMSD values of shape (batch_size,)
    """
    # Ensure tensors are on the same device
    device = pred_coords.device
    true_coords = true_coords.to(device)
    if mask is not None:
        mask = mask.to(device)
    
    # Extract batch size and sequence length
    batch_size, seq_len, _ = pred_coords.shape
    
    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Initialize RMSD for each sequence
    rmsd_values = []
    
    # Process each sequence in the batch
    for b in range(batch_size):
        # Extract valid positions using mask
        valid_mask = mask[b]
        n_valid = valid_mask.sum().item()
        
        # Skip if there are too few valid positions
        if n_valid < 1:
            rmsd_values.append(torch.tensor(float('nan'), device=device))
            continue
            
        # Extract coordinates for valid positions
        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]
        
        # Align structures if requested
        if aligned and n_valid >= 3:
            try:
                p_valid = stable_kabsch_align(p_valid, t_valid, epsilon=epsilon)
            except Exception as e:
                logger.error(f"Error in Kabsch alignment for batch {b}: {e}")
                # Continue with unaligned coordinates
        
        # Compute squared distances
        squared_diff = (p_valid - t_valid) ** 2
        msd = squared_diff.sum() / (n_valid * 3)  # Mean square deviation
        rmsd = torch.sqrt(msd + epsilon)  # Root mean square deviation
        
        rmsd_values.append(rmsd)
    
    # Stack values into a tensor
    return torch.stack(rmsd_values)


def compute_contact_accuracy(
    pred_contacts: torch.Tensor,
    true_contacts: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute base-pair contact prediction accuracy metrics.
    
    Args:
        pred_contacts: Predicted contact probabilities (batch_size, seq_len, seq_len)
        true_contacts: True contact binary matrix (batch_size, seq_len, seq_len)
        mask: Boolean mask of shape (batch_size, seq_len), True for valid positions
        threshold: Probability threshold for positive prediction
        
    Returns:
        Tuple of (precision, recall, F1 score) each of shape (batch_size,)
    """
    # Ensure tensors are on the same device
    device = pred_contacts.device
    true_contacts = true_contacts.to(device)
    if mask is not None:
        mask = mask.to(device)
    
    # Extract batch size and sequence length
    batch_size, seq_len, _ = pred_contacts.shape
    
    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    
    # Create 2D mask for contact maps
    contact_mask = torch.einsum('bi,bj->bij', mask, mask)
    
    # Binarize predicted contacts using threshold
    pred_binary = (pred_contacts > threshold).float()
    
    # Initialize metrics for each sequence
    precision = torch.zeros(batch_size, device=device)
    recall = torch.zeros(batch_size, device=device)
    f1_score = torch.zeros(batch_size, device=device)
    
    # Compute metrics for each sequence
    for b in range(batch_size):
        # Get binary predictions and targets
        pred_b = pred_binary[b]
        true_b = true_contacts[b]
        mask_b = contact_mask[b]
        
        # Apply mask
        pred_masked = pred_b * mask_b
        true_masked = true_b * mask_b
        
        # Count true positives, false positives, and false negatives
        true_positives = (pred_masked * true_masked).sum()
        false_positives = pred_masked.sum() - true_positives
        false_negatives = true_masked.sum() - true_positives
        
        # Compute precision
        if true_positives + false_positives > 0:
            precision[b] = true_positives / (true_positives + false_positives)
        else:
            precision[b] = torch.tensor(0.0, device=device)
        
        # Compute recall
        if true_positives + false_negatives > 0:
            recall[b] = true_positives / (true_positives + false_negatives)
        else:
            recall[b] = torch.tensor(0.0, device=device)
        
        # Compute F1 score
        if precision[b] + recall[b] > 0:
            f1_score[b] = 2 * precision[b] * recall[b] / (precision[b] + recall[b])
        else:
            f1_score[b] = torch.tensor(0.0, device=device)
    
    return precision, recall, f1_score
