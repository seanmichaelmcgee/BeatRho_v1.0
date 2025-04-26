#!/usr/bin/env python3
"""
Validation script for the RhoFold+ IPA RNA structure prediction model.

This script loads a trained model checkpoint and evaluates its performance
on validation data, calculating TM-scores and generating visualizations.
"""

import os
import sys
import time
import argparse
import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import model and data loading
from rhofold_ipa_module import RhoFoldIPAModel
from train_rhofold_ipa import RNAFeatureDataset, collate_fn
from utils.model_utils import tm_score


def load_model(checkpoint_path: str, device: torch.device) -> RhoFoldIPAModel:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model onto
        
    Returns:
        Loaded model
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model config from checkpoint
        if "config" in checkpoint:
            model_config = checkpoint["config"]
        else:
            # Default config if not found in checkpoint
            model_config = {
                "residue_embed_dim": 128,
                "pair_embed_dim": 64,
                "num_blocks": 4,
                "num_ipa_blocks": 4,
                "no_heads": 4,
                "no_qk_points": 4,
                "no_v_points": 8,
            }
            logger.warning(f"Config not found in checkpoint, using default: {model_config}")
        
        # Create model
        model = RhoFoldIPAModel(model_config)
        
        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Try loading directly
            model.load_state_dict(checkpoint)
        
        logger.info(f"Model loaded from {checkpoint_path}")
        
        # Print model parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model has {num_params:,} parameters")
        
        return model.to(device)
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def validate_model(
    checkpoint_path: str,
    val_csv: str,
    label_csv: str,
    feature_root: str,
    output_dir: str,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Validate model on validation data.
    
    Args:
        checkpoint_path: Path to model checkpoint
        val_csv: Path to validation sequences CSV
        label_csv: Path to labels CSV
        feature_root: Path to feature directory
        output_dir: Directory to save results
        batch_size: Batch size for validation
        device: Device to run validation on
        
    Returns:
        Dictionary of validation metrics
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(checkpoint_path, device)
    model.eval()
    
    # Create validation dataset
    val_dataset = RNAFeatureDataset(
        sequences_csv=val_csv,
        labels_csv=label_csv,
        feature_root=feature_root,
        mode="val",
    )
    
    # Create data loader
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    logger.info(f"Validating on {len(val_dataset)} samples")
    
    # Initialize metrics
    all_tm_scores = []
    all_predictions = {}
    total_time = 0
    
    # Run validation
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Skip empty batches
            if not batch:
                continue
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Time forward pass
            start_time = time.time()
            outputs = model(batch)
            batch_time = time.time() - start_time
            total_time += batch_time
            
            # Calculate TM-score for each sample
            for i in range(len(batch["target_ids"])):
                target_id = batch["target_ids"][i]
                
                # Calculate TM-score
                sample_tm = tm_score(
                    pred_coords=outputs["pred_coords"][i].unsqueeze(0),
                    true_coords=batch["coordinates"][i].unsqueeze(0),
                    mask=batch["mask"][i].unsqueeze(0),
                )
                
                # Store results
                all_tm_scores.append((target_id, sample_tm.item()))
                
                # Store predictions
                all_predictions[target_id] = {
                    "pred_coords": outputs["pred_coords"][i].cpu().numpy(),
                    "true_coords": batch["coordinates"][i].cpu().numpy(),
                    "mask": batch["mask"][i].cpu().numpy(),
                    "sequence": val_dataset.sequences.get(target_id, ""),
                    "tm_score": sample_tm.item(),
                }
            
            # Log progress
            avg_time_per_sample = batch_time / len(batch["target_ids"])
            logger.info(
                f"Batch {batch_idx+1}/{len(val_loader)}: "
                f"processed {len(batch['target_ids'])} samples in {batch_time:.2f}s "
                f"({avg_time_per_sample:.4f}s per sample)"
            )
    
    # Calculate average metrics
    tm_scores = [score for _, score in all_tm_scores]
    avg_tm_score = np.mean(tm_scores)
    
    logger.info(f"Validation completed in {total_time:.2f}s")
    logger.info(f"Average TM-score: {avg_tm_score:.4f}")
    
    # Save results
    save_validation_report(all_tm_scores, os.path.join(output_dir, "validation_report.csv"))
    
    # Save predictions
    save_predictions(all_predictions, os.path.join(output_dir, "predictions"))
    
    return {
        "tm_score": avg_tm_score,
        "total_time": total_time,
        "num_samples": len(val_dataset),
        "time_per_sample": total_time / len(val_dataset),
    }


def save_validation_report(tm_scores: List[Tuple[str, float]], output_path: str) -> None:
    """
    Save validation report as CSV.
    
    Args:
        tm_scores: List of (target_id, tm_score) tuples
        output_path: Path to save CSV
    """
    # Sort by TM-score (descending)
    sorted_scores = sorted(tm_scores, key=lambda x: x[1], reverse=True)
    
    # Calculate statistics
    scores = [score for _, score in sorted_scores]
    stats = {
        "mean": np.mean(scores),
        "median": np.median(scores),
        "min": np.min(scores),
        "max": np.max(scores),
        "std": np.std(scores),
    }
    
    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_id", "tm_score"])
        for target_id, score in sorted_scores:
            writer.writerow([target_id, f"{score:.4f}"])
        
        # Add statistics
        writer.writerow([])
        writer.writerow(["Statistics", ""])
        for stat_name, stat_value in stats.items():
            writer.writerow([stat_name, f"{stat_value:.4f}"])
    
    logger.info(f"Validation report saved to {output_path}")
    
    # Print summary
    logger.info("TM-score statistics:")
    for stat_name, stat_value in stats.items():
        logger.info(f"  {stat_name}: {stat_value:.4f}")


def save_predictions(predictions: Dict[str, Dict], output_dir: str) -> None:
    """
    Save predictions as NPZ files.
    
    Args:
        predictions: Dictionary of predictions
        output_dir: Directory to save predictions
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each prediction
    for target_id, pred_data in predictions.items():
        output_path = os.path.join(output_dir, f"{target_id}_prediction.npz")
        
        # Save as NPZ
        np.savez(
            output_path,
            pred_coords=pred_data["pred_coords"],
            true_coords=pred_data["true_coords"],
            mask=pred_data["mask"],
            tm_score=pred_data["tm_score"],
            sequence=pred_data["sequence"],
        )
    
    logger.info(f"Predictions saved to {output_dir}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate RhoFold IPA model")
    
    # Required arguments
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation sequences CSV")
    parser.add_argument("--label_csv", type=str, required=True, help="Path to labels CSV")
    parser.add_argument("--feature_root", type=str, required=True, help="Path to feature directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    
    # Optional arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU even if GPU is available")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Run validation
    validate_model(
        checkpoint_path=args.ckpt_path,
        val_csv=args.val_csv,
        label_csv=args.label_csv,
        feature_root=args.feature_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
