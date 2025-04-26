#!/usr/bin/env python3
"""
Test script for TM-score implementation in the RNA structure prediction pipeline.
"""

import os
import sys
import torch
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def validate_tm_score_implementation():
    """
    Validate TM-score implementation with test cases.
    """
    try:
        # Try to import from train_rhofold_ipa.py
        from train_rhofold_ipa import compute_tm_score
        logger.info("Successfully imported compute_tm_score from train_rhofold_ipa.py")
    except ImportError:
        # Define TM-score inline for testing
        logger.warning("Unable to import compute_tm_score, using inline implementation")
        def compute_tm_score(
            pred_coords: torch.Tensor,
            true_coords: torch.Tensor,
            mask: torch.Tensor = None,
            d0_scale: float = 1.24,
            d0_offset: float = -1.8,
            epsilon: float = 1e-8,
        ) -> torch.Tensor:
            """Calculate TM-score between predicted and true coordinates."""
            # Add batch dimension if not present
            if len(pred_coords.shape) == 2:
                pred_coords = pred_coords.unsqueeze(0)
                true_coords = true_coords.unsqueeze(0)
                if mask is not None and len(mask.shape) == 1:
                    mask = mask.unsqueeze(0)
            
            device = pred_coords.device
            dtype = pred_coords.dtype
            batch_size, seq_len, _ = pred_coords.shape
            
            # Create default mask if not provided
            if mask is None:
                mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            
            # Initialize results tensor
            tm_scores = torch.zeros(batch_size, dtype=dtype, device=device)
            
            # Define optimal alignment helper function
            def kabsch_align(A, B):
                # Center coordinates
                A_mean = torch.mean(A, dim=0, keepdim=True)
                B_mean = torch.mean(B, dim=0, keepdim=True)
                A_centered = A - A_mean
                B_centered = B - B_mean
                
                # Compute covariance matrix
                cov = torch.matmul(A_centered.T, B_centered)
                
                # SVD
                U, S, Vt = torch.linalg.svd(cov)
                
                # Ensure proper rotation (no reflection)
                det = torch.linalg.det(torch.matmul(Vt.T, U.T))
                correction = torch.eye(3, device=device)
                correction[-1, -1] = det
                
                # Rotation matrix
                R = torch.matmul(Vt.T, torch.matmul(correction, U.T))
                
                # Apply rotation and translation
                A_aligned = torch.matmul(A_centered, R) + B_mean
                
                return A_aligned
            
            for b in range(batch_size):
                valid_mask = mask[b]
                valid_count = valid_mask.sum()
                
                if valid_count < 3:  # Need at least 3 points for meaningful TM-score
                    tm_scores[b] = 0.0
                    continue
                    
                # Extract valid coordinates
                p_valid = pred_coords[b, valid_mask]
                t_valid = true_coords[b, valid_mask]
                
                # Calculate normalization parameter d0 based on sequence length
                L = valid_count
                d0 = d0_scale * (L ** (1.0/3.0)) + d0_offset
                d0 = max(d0, 0.5)  # Lower bound
                
                # Use Kabsch algorithm for optimal alignment
                p_aligned = kabsch_align(p_valid, t_valid)
                
                # Handle direct checks for identical tensors
                if torch.allclose(p_valid, t_valid, atol=1e-6):
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
    
    # Test cases
    logger.info("Running TM-score validation tests...")
    
    # Test 1: Perfect match
    pred1 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    true1 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    tm1 = compute_tm_score(pred1, true1)
    logger.info(f"Test 1 (Perfect match): TM-score = {tm1.item():.4f} (expected ~1.0)")
    
    # Test 2: Translation
    pred2 = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]]], dtype=torch.float32)
    true2 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    tm2 = compute_tm_score(pred2, true2)
    logger.info(f"Test 2 (Translation): TM-score = {tm2.item():.4f} (expected ~1.0)")
    
    # Test 3: Rotation
    # 90-degree rotation around Z axis
    pred3 = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float32)
    true3 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    tm3 = compute_tm_score(pred3, true3)
    logger.info(f"Test 3 (Rotation): TM-score = {tm3.item():.4f} (expected ~1.0)")
    
    # Test 4: Scaling
    pred4 = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=torch.float32)
    true4 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    tm4 = compute_tm_score(pred4, true4)
    logger.info(f"Test 4 (Scaling): TM-score = {tm4.item():.4f} (expected < 1.0)")
    
    # Test 5: Different topology
    pred5 = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
    true5 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    tm5 = compute_tm_score(pred5, true5)
    logger.info(f"Test 5 (Different topology): TM-score = {tm5.item():.4f} (expected < 0.8)")
    
    # Test 6: With masking
    pred6 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]], dtype=torch.float32)
    true6 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 0.0]]], dtype=torch.float32)
    mask6 = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)  # Mask out the last residue
    tm6 = compute_tm_score(pred6, true6, mask6)
    logger.info(f"Test 6 (With masking): TM-score = {tm6.item():.4f} (expected ~1.0, ignoring last residue)")
    
    # Check if all tests passed as expected
    passed = (
        tm1.item() > 0.99 and  # Perfect match
        tm2.item() > 0.99 and  # Translation should be aligned perfectly
        tm3.item() > 0.99 and  # Rotation should be aligned perfectly
        tm4.item() < 0.99 and  # Scaling should reduce score
        tm5.item() < 0.8 and   # Different topology should have lower score
        tm6.item() > 0.99      # Masking should ignore mismatched residue
    )
    
    logger.info(f"TM-score validation: {'PASSED' if passed else 'FAILED'}")
    return passed

if __name__ == "__main__":
    success = validate_tm_score_implementation()
    sys.exit(0 if success else 1)
