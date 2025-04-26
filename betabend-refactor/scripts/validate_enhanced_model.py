#!/usr/bin/env python3
"""
Standalone validation script for RNA structure prediction models.

This script provides comprehensive validation of model checkpoints using the
ValidationRunner to run dual-mode validation (test-equivalent and training-equivalent).
It can be used to validate checkpoints during or after training.
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np

# Add project root to path for importing project modules
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import model and validation modules
from models.rna_folding_model import RNAFoldingModel
from validation.validation_runner import ValidationRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_checkpoint(
    checkpoint_path: str,
    subset: str,
    batch_size: int,
    device: str,
    output_dir: str,
    run_both_modes: bool = True,
    test_equivalent_only: bool = False,
    training_equivalent_only: bool = False
) -> Dict[str, Any]:
    """
    Validate a model checkpoint using the comprehensive validation system.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        subset: Validation subset name to use
        batch_size: Batch size for validation
        device: Device to run validation on
        output_dir: Directory to save validation results
        run_both_modes: Whether to run both test-equivalent and training-equivalent modes
        test_equivalent_only: Whether to run only test-equivalent mode
        training_equivalent_only: Whether to run only training-equivalent mode
        
    Returns:
        Dictionary of validation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model from checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict based on checkpoint format
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        # Assume the checkpoint is just the model state dict
        model_state_dict = checkpoint
    
    # Get model configuration if available in checkpoint
    if 'config' in checkpoint:
        model_config = checkpoint['config']
    else:
        # Use default configuration
        logger.warning("No model config found in checkpoint. Using default values.")
        model_config = {
            'num_blocks': 8,
            'residue_embed_dim': 256,
            'pair_embed_dim': 128,
            'num_heads': 8,
            'ff_dim': 1024,
            'dropout': 0.1,
        }
    
    # Create model
    logger.info(f"Initializing model with config: {model_config}")
    model = RNAFoldingModel(**model_config).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Configure validation
    validation_config = {
        "batch_size": batch_size,
        "results_dir": output_dir,
        "save_results": True,
        "verbose": True,
        "generate_visualizations": True,
        "max_samples": 10,  # Limit samples for quicker validation
    }
    
    # Create validation runner
    validator = ValidationRunner(
        model=model,
        config=validation_config,
        device=device
    )
    
    # Determine which modes to run
    if test_equivalent_only:
        run_both_modes = False
        logger.info(f"Running validation in test-equivalent mode only")
    elif training_equivalent_only:
        run_both_modes = False
        logger.info(f"Running validation in training-equivalent mode only")
    else:
        logger.info(f"Running validation in {'both modes' if run_both_modes else 'default mode'}")
    
    # Run validation
    results = validator.run_validation(
        subset_name=subset,
        run_both_modes=run_both_modes,
        test_equivalent_only=test_equivalent_only,
        training_equivalent_only=training_equivalent_only
    )
    
    # Save results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(output_dir, f"validation_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        serializable_results = validator._make_serializable(results)
        json.dump(serializable_results, f, indent=2)
    
    # Print summary
    print("\n=== Validation Summary ===")
    for mode in ["test_equivalent", "training_equivalent"]:
        if mode in results:
            print(f"\n{mode.upper()} MODE:")
            print(f"Mean RMSD: {results[mode].get('mean_rmsd', 'N/A'):.4f}")
            print(f"Mean TM-score: {results[mode].get('mean_tm_score', 'N/A'):.4f}")
    
    return results


def main():
    """
    Parse command line arguments and run validation.
    """
    parser = argparse.ArgumentParser(description="Validate RNA folding model checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--subset", type=str, default="validation_medium",
                        choices=["validation_small", "validation_medium", "validation"],
                        help="Validation subset to use")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for validation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run validation on")
    parser.add_argument("--output-dir", type=str, default="validation_results",
                        help="Directory to save validation results")
    
    # Mode selection arguments (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--run-both-modes", action="store_true", default=True,
                           help="Run both test-equivalent and training-equivalent modes")
    mode_group.add_argument("--test-equivalent-only", action="store_true",
                           help="Run only test-equivalent mode")
    mode_group.add_argument("--training-equivalent-only", action="store_true",
                           help="Run only training-equivalent mode")
    
    args = parser.parse_args()
    
    # Run validation
    validate_checkpoint(
        checkpoint_path=args.checkpoint,
        subset=args.subset,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        run_both_modes=args.run_both_modes,
        test_equivalent_only=args.test_equivalent_only,
        training_equivalent_only=args.training_equivalent_only
    )


if __name__ == "__main__":
    main()