#!/usr/bin/env python3
"""
Test utility functions for the BetaRho RNA structure prediction pipeline.

This script tests the utility functions implemented in utils/model_utils.py,
particularly focusing on the TM-score calculation and temporal cutoff filtering.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
from utils.model_utils import tm_score, soft_tm_score, filter_by_temporal_cutoff

def test_tm_score():
    """Test TM-score calculation with various test cases."""
    logger.info("Testing TM-score calculation...")
    
    # Test case 1: Perfect match
    pred1 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    true1 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    tm1 = tm_score(pred1, true1)
    logger.info(f"Test 1 (Perfect match): TM-score = {tm1.item():.4f} (expected ~1.0)")
    assert tm1 > 0.99, "Perfect match should have TM-score close to 1"
    
    # Test case 2: Translation
    pred2 = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [1.0, 2.0, 1.0]]], dtype=torch.float32)
    true2 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    tm2 = tm_score(pred2, true2)
    logger.info(f"Test 2 (Translation): TM-score = {tm2.item():.4f} (expected ~1.0)")
    assert tm2 > 0.99, "Translation should not affect TM-score"
    
    # Test case 3: Rotation
    # 90-degree rotation around Z axis
    pred3 = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float32)
    true3 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    tm3 = tm_score(pred3, true3)
    logger.info(f"Test 3 (Rotation): TM-score = {tm3.item():.4f} (expected ~1.0)")
    assert tm3 > 0.99, "Rotation should not affect TM-score"
    
    # Test case 4: Scaling
    pred4 = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]], dtype=torch.float32)
    true4 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    tm4 = tm_score(pred4, true4)
    logger.info(f"Test 4 (Scaling): TM-score = {tm4.item():.4f} (expected < 1.0)")
    assert tm4 < 0.99, "Scaling should affect TM-score"
    
    # Test case 5: Different topology
    pred5 = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
    true5 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    tm5 = tm_score(pred5, true5)
    logger.info(f"Test 5 (Different topology): TM-score = {tm5.item():.4f} (expected < 0.8)")
    assert tm5 < 0.8, "Different topology should have low TM-score"
    
    # Test case 6: With masking
    pred6 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]], dtype=torch.float32)
    true6 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 2.0, 0.0]]], dtype=torch.float32)
    mask6 = torch.tensor([[True, True, True, False]], dtype=torch.bool)  # Mask out the last residue
    tm6 = tm_score(pred6, true6, mask6)
    logger.info(f"Test 6 (With masking): TM-score = {tm6.item():.4f} (expected ~1.0)")
    assert tm6 > 0.99, "Masking should ignore mismatched residue"
    
    logger.info("All TM-score tests passed!")

def test_soft_tm_score():
    """Test differentiable soft TM-score calculation."""
    logger.info("Testing soft TM-score calculation...")
    
    # Test with simple case
    pred = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32, requires_grad=True)
    true = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
    
    # Calculate soft TM-score
    score = soft_tm_score(pred, true)
    logger.info(f"Soft TM-score: {score.item():.4f}")
    
    # Test gradient flow
    score.backward()
    assert pred.grad is not None, "Gradient should flow through soft_tm_score"
    logger.info("Gradient flow test passed")
    
    # Compare with regular TM-score
    with torch.no_grad():
        regular_score = tm_score(pred.detach(), true)
    
    logger.info(f"Regular TM-score: {regular_score.item():.4f}")
    assert abs(score.item() - regular_score.item()) < 0.05, "Soft and regular TM-scores should be similar"
    
    logger.info("Soft TM-score tests passed!")

def test_temporal_cutoff():
    """Test temporal cutoff filtering."""
    logger.info("Testing temporal cutoff filtering...")
    
    # Create a test DataFrame
    data = {
        'target_id': ['RNA001', 'RNA002', 'RNA003', 'RNA004', 'RNA005'],
        'sequence': ['ACGU', 'CGUA', 'GUAC', 'UACG', 'ACGU'],
        'temporal_cutoff': ['2022-01-01', '2022-03-15', '2022-05-26', '2022-05-27', '2022-06-30']
    }
    df = pd.DataFrame(data)
    
    # Test with default cutoff (2022-05-27)
    train_df, val_df = filter_by_temporal_cutoff(df)
    
    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    
    assert len(train_df) == 3, "Train set should have 3 samples (before 2022-05-27)"
    assert len(val_df) == 2, "Validation set should have 2 samples (on or after 2022-05-27)"
    
    # Test with custom cutoff
    train_df, val_df = filter_by_temporal_cutoff(df, cutoff_date='2022-03-01')
    
    logger.info(f"Train set size (custom cutoff): {len(train_df)}")
    logger.info(f"Validation set size (custom cutoff): {len(val_df)}")
    
    assert len(train_df) == 1, "Train set should have 1 sample (before 2022-03-01)"
    assert len(val_df) == 4, "Validation set should have 4 samples (on or after 2022-03-01)"
    
    logger.info("Temporal cutoff tests passed!")

def main():
    """Run all tests."""
    logger.info("Running BetaRho utility tests...")
    
    # Run TM-score tests
    test_tm_score()
    
    # Run soft TM-score tests
    test_soft_tm_score()
    
    # Run temporal cutoff tests
    test_temporal_cutoff()
    
    logger.info("All tests passed!")

if __name__ == "__main__":
    main()
