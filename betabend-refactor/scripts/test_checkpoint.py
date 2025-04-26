#!/usr/bin/env python3
"""
Checkpoint Testing Utility for RNA 3D Structure Prediction

This script tests a checkpoint by:
1. Loading the model from the checkpoint
2. Running inference on a small validation set
3. Calculating metrics
4. Saving results and visualizations
"""

import os
import sys
import argparse
import logging
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to path for importing project modules
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
try:
    from src.models.rna_folding_model import RNAFoldingModel
    from src.data_loading import RNADataset, collate_fn
    from src.losses import compute_stable_fape_loss
except ImportError:
    sys.path.append(str(project_root / 'src'))
    sys.path.append(str(project_root / 'src/models'))
    from models.rna_folding_model import RNAFoldingModel
    from data_loading_fixed import RNADataset, collate_fn
    from losses import compute_stable_fape_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test an RNA folding model checkpoint')
    
    # Required parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file to test')
    
    # Data parameters
    parser.add_argument('--train_csv', type=str, default='data/raw/train_sequences.csv',
                       help='Path to training sequences CSV')
    parser.add_argument('--labels_csv', type=str, default='data/raw/train_labels.csv',
                       help='Path to training labels CSV with 3D coordinates')
    parser.add_argument('--features_dir', type=str, default='data/processed/',
                       help='Path to processed features directory')
    
    # Test parameters
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to test')
    parser.add_argument('--min_seq_length', type=int, default=50,
                       help='Minimum sequence length to test')
    parser.add_argument('--max_seq_length', type=int, default=300,
                       help='Maximum sequence length to test')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for testing')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='validation_results',
                       help='Directory to save test results')
    parser.add_argument('--save_structures', action='store_true',
                       help='Save predicted structures as PDB files')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save visualization plots')
    
    return parser.parse_args()

def load_checkpoint(checkpoint_path, device):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model onto
        
    Returns:
        Loaded model and checkpoint data
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Check if this is a full checkpoint or just model weights
        if 'model_state_dict' in checkpoint:
            # Extract model configuration if available
            model_config = checkpoint.get('model_config', {})
            if not model_config and 'args' in checkpoint:
                # Try to extract from args
                args = checkpoint['args']
                model_config = {
                    'num_blocks': args.get('num_blocks', 6),
                    'residue_embed_dim': args.get('residue_embed_dim', 192),
                    'pair_embed_dim': args.get('pair_embed_dim', 64),
                    'num_attention_heads': args.get('num_heads', 8),
                    'ff_dim': args.get('ff_dim', 512),
                    'dropout': args.get('dropout', 0.1),
                }
            
            # Use default config if nothing found
            if not model_config:
                logger.warning("No model configuration found in checkpoint. Using default values.")
                model_config = {
                    'num_blocks': 6,
                    'residue_embed_dim': 192,
                    'pair_embed_dim': 64,
                    'num_attention_heads': 8,
                    'ff_dim': 512,
                    'dropout': 0.1,
                }
            
            # Create model with extracted config
            model = RNAFoldingModel(model_config)
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Loaded model with config: {model_config}")
            
        else:
            # Assume the checkpoint contains the full model
            logger.warning("Checkpoint does not contain model_state_dict. Attempting to load full model.")
            model = checkpoint
            
        # Move model to device
        model = model.to(device)
        model.eval()
        
        # Return model and full checkpoint data
        return model, checkpoint
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise

def create_test_dataset(args):
    """
    Create a small test dataset from the training data.
    
    Args:
        args: Command line arguments
        
    Returns:
        Test dataset
    """
    # Create full dataset
    try:
        dataset = RNADataset(
            sequences_csv_path=args.train_csv,
            labels_csv_path=args.labels_csv,
            features_dir=args.features_dir,
        )
        
        logger.info(f"Loaded dataset with {len(dataset)} samples")
        
        # Filter for sequence length
        valid_indices = []
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                if 'length' in sample:
                    length = sample['length']
                elif 'sequence_int' in sample:
                    length = len(sample['sequence_int'])
                else:
                    continue
                
                if args.min_seq_length <= length <= args.max_seq_length:
                    valid_indices.append(i)
                    
                if len(valid_indices) >= args.num_samples:
                    break
            except Exception as e:
                logger.warning(f"Error accessing sample {i}: {e}")
                
        # Create subset of appropriate size
        if len(valid_indices) < args.num_samples:
            logger.warning(f"Only found {len(valid_indices)} valid samples. Using all of them.")
            
        if not valid_indices:
            raise ValueError("No valid samples found. Please check sequence length criteria.")
            
        # Create subset
        from torch.utils.data import Subset
        test_dataset = Subset(dataset, valid_indices)
        
        logger.info(f"Created test dataset with {len(test_dataset)} samples")
        return test_dataset
        
    except Exception as e:
        logger.error(f"Error creating test dataset: {e}")
        raise

def test_model(model, dataset, args, device):
    """
    Test model on dataset and calculate metrics.
    
    Args:
        model: Model to test
        dataset: Dataset to test on
        args: Command line arguments
        device: Device to run inference on
        
    Returns:
        Dictionary of test results
    """
    # Create dataloader
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1,
    )
    
    # Run inference
    all_rmsd = []
    predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(batch)
                
                # Calculate RMSD
                for i in range(len(batch["lengths"])):
                    seq_len = batch["lengths"][i].item()
                    pred_coords_i = outputs["pred_coords"][i, :seq_len].cpu().numpy()
                    true_coords_i = batch["coordinates"][i, :seq_len].cpu().numpy()
                    
                    # Basic RMSD calculation
                    rmsd = np.sqrt(np.mean(np.sum((pred_coords_i - true_coords_i) ** 2, axis=1)))
                    all_rmsd.append(rmsd)
                    
                    # Save predictions
                    pred_info = {
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'sequence_id': batch.get('id', [f"sample_{batch_idx}_{i}"])[i],
                        'length': seq_len,
                        'rmsd': float(rmsd),
                        'pred_coords': pred_coords_i,
                        'true_coords': true_coords_i,
                    }
                    predictions.append(pred_info)
                    
            except Exception as e:
                logger.error(f"Error during inference on batch {batch_idx}: {e}")
    
    # Calculate metrics
    results = {
        'all_rmsd': all_rmsd,
        'mean_rmsd': float(np.mean(all_rmsd)) if all_rmsd else float('nan'),
        'median_rmsd': float(np.median(all_rmsd)) if all_rmsd else float('nan'),
        'min_rmsd': float(np.min(all_rmsd)) if all_rmsd else float('nan'),
        'max_rmsd': float(np.max(all_rmsd)) if all_rmsd else float('nan'),
        'std_rmsd': float(np.std(all_rmsd)) if all_rmsd else float('nan'),
        'num_samples': len(all_rmsd),
        'predictions': predictions
    }
    
    return results

def save_results(results, args, checkpoint_info=None):
    """
    Save test results to output directory.
    
    Args:
        results: Test results dictionary
        args: Command line arguments
        checkpoint_info: Information about the checkpoint
        
    Returns:
        None
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results summary
    results_summary = {
        'checkpoint': args.checkpoint,
        'test_time': datetime.now().isoformat(),
        'mean_rmsd': results['mean_rmsd'],
        'median_rmsd': results['median_rmsd'],
        'min_rmsd': results['min_rmsd'],
        'max_rmsd': results['max_rmsd'],
        'std_rmsd': results['std_rmsd'],
        'num_samples': results['num_samples'],
    }
    
    # Add checkpoint info if available
    if checkpoint_info:
        if isinstance(checkpoint_info, dict):
            # Extract relevant info
            cp_info = {}
            for key in ['epoch', 'timestamp', 'val_loss', 'val_rmsd']:
                if key in checkpoint_info:
                    cp_info[key] = checkpoint_info[key]
            results_summary['checkpoint_info'] = cp_info
    
    # Save summary
    with open(os.path.join(args.output_dir, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save all RMSDs
    with open(os.path.join(args.output_dir, 'all_rmsd.json'), 'w') as f:
        json.dump({
            'all_rmsd': [float(x) for x in results['all_rmsd']]
        }, f, indent=2)
    
    # Save plots if requested
    if args.save_plots:
        # Plot RMSD distribution
        plt.figure(figsize=(10, 6))
        plt.hist(results['all_rmsd'], bins=10, alpha=0.7)
        plt.axvline(results['mean_rmsd'], color='r', linestyle='--', label=f'Mean: {results["mean_rmsd"]:.2f}Å')
        plt.axvline(results['median_rmsd'], color='g', linestyle='--', label=f'Median: {results["median_rmsd"]:.2f}Å')
        plt.xlabel('RMSD (Å)')
        plt.ylabel('Count')
        plt.title('RMSD Distribution')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'rmsd_distribution.png'), dpi=300)
        plt.close()
        
        # Plot RMSD vs length
        lengths = [p['length'] for p in results['predictions']]
        rmsds = [p['rmsd'] for p in results['predictions']]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(lengths, rmsds, alpha=0.7)
        plt.xlabel('Sequence Length')
        plt.ylabel('RMSD (Å)')
        plt.title('RMSD vs Sequence Length')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(args.output_dir, 'rmsd_vs_length.png'), dpi=300)
        plt.close()
    
    # Save structures if requested
    if args.save_structures:
        structures_dir = os.path.join(args.output_dir, 'structures')
        os.makedirs(structures_dir, exist_ok=True)
        
        for pred in results['predictions']:
            sample_id = pred['sequence_id']
            pred_coords = pred['pred_coords']
            true_coords = pred['true_coords']
            
            # Save as PDB or numpy files
            np.save(os.path.join(structures_dir, f'{sample_id}_pred_coords.npy'), pred_coords)
            np.save(os.path.join(structures_dir, f'{sample_id}_true_coords.npy'), true_coords)
            
            # TODO: Add PDB file conversion if needed
    
    logger.info(f"Results saved to {args.output_dir}")
    
    return results_summary

def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load model from checkpoint
        model, checkpoint_data = load_checkpoint(args.checkpoint, device)
        
        # Create test dataset
        test_dataset = create_test_dataset(args)
        
        # Test model
        logger.info("Testing model...")
        results = test_model(model, test_dataset, args, device)
        
        # Save results
        results_summary = save_results(results, args, checkpoint_data)
        
        # Log summary
        logger.info(f"Test completed successfully.")
        logger.info(f"Mean RMSD: {results_summary['mean_rmsd']:.2f}Å")
        logger.info(f"Median RMSD: {results_summary['median_rmsd']:.2f}Å")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in checkpoint testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())