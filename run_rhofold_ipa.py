#!/usr/bin/env python3
"""
RhoFold+ IPA Pipeline Runner

This script provides a convenient wrapper to run the RhoFold+ IPA RNA structure
prediction pipeline. It handles command-line arguments and executes the training
or testing process with the appropriate parameters.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run RhoFold+ IPA RNA structure prediction pipeline")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["train", "test", "validate"], 
                       default="train", help="Pipeline mode: train, test, or validate")
    
    # Data parameters
    parser.add_argument("--train_csv", type=str, default="data/train_sequences.csv",
                       help="Path to training sequences CSV")
    parser.add_argument("--label_csv", type=str, default="data/train_labels.csv",
                       help="Path to training labels CSV")
    parser.add_argument("--feature_root", type=str, default="data/processed",
                       help="Path to processed features directory")
    parser.add_argument("--val_csv", type=str, default="data/validation_sequences.csv",
                       help="Path to validation sequences CSV")
    parser.add_argument("--temporal_cutoff", type=str, default="2022-05-27",
                       help="Temporal cutoff date (YYYY-MM-DD)")
    
    # Model parameters
    parser.add_argument("--residue_embed_dim", type=int, default=128,
                       help="Residue embedding dimension")
    parser.add_argument("--pair_embed_dim", type=int, default=64,
                       help="Pair embedding dimension")
    parser.add_argument("--num_blocks", type=int, default=4,
                       help="Number of transformer blocks")
    parser.add_argument("--num_ipa_blocks", type=int, default=4,
                       help="Number of IPA iterations")
    parser.add_argument("--no_heads", type=int, default=4,
                       help="Number of attention heads in IPA")
    parser.add_argument("--no_qk_points", type=int, default=4,
                       help="Number of query/key points in IPA")
    parser.add_argument("--no_v_points", type=int, default=8,
                       help="Number of value points in IPA")
    
    # Training parameters
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--grad_checkpoint", action="store_true",
                       help="Use gradient checkpointing")
    
    # Loss weights
    parser.add_argument("--tm_weight", type=float, default=1.0,
                       help="Weight for TM-score loss")
    parser.add_argument("--fape_weight", type=float, default=0.5,
                       help="Weight for FAPE loss")
    parser.add_argument("--contact_bce_weight", type=float, default=0.1,
                       help="Weight for contact BCE loss")
    
    # Output parameters
    parser.add_argument("--ckpt_out", type=str, default="checkpoints/rhofold_ipa_final.pt",
                       help="Path to save checkpoint")
    parser.add_argument("--run_tests", action="store_true",
                       help="Run tests before training")
    parser.add_argument("--wandb", action="store_true",
                       help="Log to Weights & Biases")
    
    # Testing/validation parameters
    parser.add_argument("--ckpt_path", type=str,
                       help="Path to checkpoint for testing/validation")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save test/validation results")
    
    return parser.parse_args()

def run_training(args):
    """Run the training pipeline with the specified arguments."""
    from train_rhofold_ipa import main
    
    # Create clean arguments namespace for train_rhofold_ipa.py
    train_args = argparse.Namespace()
    
    # Copy relevant arguments
    for arg_name, arg_value in vars(args).items():
        if arg_name not in ['mode', 'ckpt_path', 'output_dir']:
            setattr(train_args, arg_name, arg_value)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    
    # Run training
    logger.info("Starting RhoFold+ IPA training pipeline")
    sys.argv = [sys.argv[0]]  # Clear sys.argv to avoid interference with main's argparse
    
    # Call the main function with our args
    main(train_args)
    
    logger.info(f"Training completed. Checkpoint saved to {args.ckpt_out}")

def run_validation(args):
    """Run the validation pipeline with the specified arguments."""
    
    # Ensure checkpoint path is provided
    if not args.ckpt_path:
        logger.error("Checkpoint path (--ckpt_path) is required for validation mode")
        sys.exit(1)
    
    # Import validation function
    try:
        from validate_enhanced_model import validate_model
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run validation
        logger.info(f"Starting validation with checkpoint: {args.ckpt_path}")
        
        validate_model(
            checkpoint_path=args.ckpt_path,
            val_csv=args.val_csv,
            label_csv=args.label_csv,
            feature_root=args.feature_root,
            output_dir=args.output_dir,
            batch_size=args.batch
        )
        
        logger.info(f"Validation completed. Results saved to {args.output_dir}")
        
    except ImportError:
        logger.error("Could not import validate_enhanced_model.py. Please ensure the file exists.")
        sys.exit(1)

def run_unit_tests():
    """Run the TM-score unit tests."""
    import test_tm_score
    
    logger.info("Running TM-score unit tests...")
    success = test_tm_score.validate_tm_score_implementation()
    
    if success:
        logger.info("TM-score unit tests passed.")
    else:
        logger.error("TM-score unit tests failed.")
        sys.exit(1)

def main():
    """Main function to parse arguments and run the appropriate pipeline."""
    args = parse_args()
    
    # Output execution parameters
    logger.info("RhoFold+ IPA Pipeline")
    logger.info("=====================")
    logger.info(f"Mode: {args.mode}")
    
    # Run unit tests if requested
    if args.run_tests:
        run_unit_tests()
    
    # Run the appropriate pipeline based on mode
    if args.mode == "train":
        run_training(args)
    elif args.mode == "validate":
        run_validation(args)
    elif args.mode == "test":
        # For now, test mode is the same as validate
        run_validation(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
    logger.info("RhoFold+ IPA Pipeline completed successfully.")

if __name__ == "__main__":
    main()
