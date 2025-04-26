#!/usr/bin/env python3
"""
Checkpoint Converter for Kaggle Submission

This script converts a training checkpoint to the format expected by Kaggle:
1. Removes optimizer state and other training-specific information
2. Standardizes model configuration for compatibility
3. Creates metadata with model information
4. Saves the simplified checkpoint to the output location
"""

import os
import sys
import argparse
import logging
import json
import time
import copy
from datetime import datetime
from pathlib import Path

import torch
import numpy as np

# Add project root to path for importing project modules
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert RNA model checkpoint to Kaggle format')
    
    # Required parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file to convert')
    
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save converted checkpoint')
    
    # Optional parameters
    parser.add_argument('--save-original-config', action='store_true',
                       help='Save the original model configuration as a JSON file')
    parser.add_argument('--create-metadata', action='store_true',
                       help='Create metadata JSON file for the model')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Name for the model in metadata (defaults to output filename)')
    parser.add_argument('--description', type=str, default=None,
                       help='Description of the model for metadata')
    parser.add_argument('--model-version', type=str, default='1.0.0',
                       help='Version of the model for metadata')
    
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    """
    Load checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Loaded checkpoint
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            
        return checkpoint
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise

def convert_checkpoint(checkpoint):
    """
    Convert checkpoint to Kaggle format.
    
    Args:
        checkpoint: Loaded checkpoint data
        
    Returns:
        Converted checkpoint and original configuration
    """
    # Keep a copy of the original configuration
    original_config = {}
    
    if 'model_config' in checkpoint:
        original_config['model_config'] = checkpoint['model_config']
    elif 'args' in checkpoint:
        # Extract model configuration from args
        args = checkpoint['args']
        model_config = {}
        
        for key in ['num_blocks', 'residue_embed_dim', 'pair_embed_dim', 
                   'num_heads', 'ff_dim', 'dropout']:
            if key in args:
                model_config[key] = args[key]
                
        if model_config:
            original_config['model_config'] = model_config
            
    # Copy model_state_dict
    if 'model_state_dict' in checkpoint:
        converted_checkpoint = {
            'model_state_dict': checkpoint['model_state_dict'],
        }
        
        # Add minimal metadata
        if 'model_config' in checkpoint:
            converted_checkpoint['model_config'] = checkpoint['model_config']
        
        # Add epoch information if available
        if 'epoch' in checkpoint:
            converted_checkpoint['epoch'] = checkpoint['epoch']
            
        # Add metrics if available
        for key in ['val_loss', 'val_rmsd']:
            if key in checkpoint:
                converted_checkpoint[key] = checkpoint[key]
                
    else:
        # If no model_state_dict, assume it's already in the right format
        logger.warning("Checkpoint has no model_state_dict. It may already be in Kaggle format.")
        converted_checkpoint = checkpoint
        
    # Add compatibility configuration required for Kaggle
    kaggle_config = {
        'model_type': 'rna_fold_v2',
        'kaggle_compatible': True,
        'version': '2.0.0',
    }
    
    converted_checkpoint.update(kaggle_config)
    
    return converted_checkpoint, original_config

def create_model_metadata(args, checkpoint, original_config):
    """
    Create metadata JSON file for the model.
    
    Args:
        args: Command line arguments
        checkpoint: Original checkpoint data
        original_config: Original model configuration
        
    Returns:
        Dictionary with metadata
    """
    # Determine model name
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = os.path.basename(args.output).replace('.pt', '')
        
    # Determine description
    if args.description:
        description = args.description
    else:
        # Generate default description
        if 'epoch' in checkpoint:
            description = f"RNA 3D structure prediction model trained for {checkpoint['epoch']} epochs"
        else:
            description = "RNA 3D structure prediction model for Kaggle submission"
            
    # Extract performance metrics
    metrics = {}
    for key in ['val_loss', 'val_rmsd']:
        if key in checkpoint:
            metrics[key] = checkpoint[key]
            
    # Create metadata
    metadata = {
        'model_name': model_name,
        'model_version': args.model_version,
        'description': description,
        'date_created': datetime.now().isoformat(),
        'architecture': 'transformer_ipa_hybrid',
        'performance_metrics': metrics,
    }
    
    # Add model configuration if available
    if 'model_config' in original_config:
        metadata['model_config'] = original_config['model_config']
        
    # Add input and output specifications
    metadata['input_format'] = {
        'sequence_int': 'Int tensor with RNA sequence (AUCG encoded as 0,1,2,3)',
        'mask': 'Boolean tensor with sequence mask',
        'pairing_probs': '2D tensor with base pairing probabilities (optional)',
        'dihedral_features': 'Dihedral angle features (optional)',
        'thermo_features': 'Thermodynamic features (optional)'
    }
    
    metadata['output_format'] = {
        'pred_coords': '3D coordinates (N x 3) for each residue',
        'pred_confidence': 'Per-residue confidence scores',
        'pred_angles': 'Predicted dihedral angles'
    }
    
    return metadata

def save_checkpoint(converted_checkpoint, output_path):
    """
    Save converted checkpoint.
    
    Args:
        converted_checkpoint: Converted checkpoint data
        output_path: Path to save checkpoint
        
    Returns:
        None
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Save checkpoint
    torch.save(converted_checkpoint, output_path)
    logger.info(f"Converted checkpoint saved to {output_path}")

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Load checkpoint
        checkpoint = load_checkpoint(args.checkpoint)
        
        # Convert checkpoint
        converted_checkpoint, original_config = convert_checkpoint(checkpoint)
        
        # Save original config if requested
        if args.save_original_config:
            config_path = args.output.replace('.pt', '_original_config.json')
            with open(config_path, 'w') as f:
                json.dump(original_config, f, indent=2)
            logger.info(f"Original config saved to {config_path}")
        
        # Save converted checkpoint
        save_checkpoint(converted_checkpoint, args.output)
        
        # Create and save metadata if requested
        if args.create_metadata:
            metadata = create_model_metadata(args, checkpoint, original_config)
            
            # Save metadata
            metadata_path = args.output.replace('.pt', '_metadata.json')
            if metadata_path == args.output:  # Avoid overwriting checkpoint
                metadata_path = os.path.join(os.path.dirname(args.output), 
                                          f"metadata_{os.path.basename(args.output).replace('.pt', '')}.json")
                
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Model metadata saved to {metadata_path}")
            
            # Also save in a standard location for Kaggle submission
            kaggle_dir = os.path.dirname(args.output)
            submission_metadata_path = os.path.join(kaggle_dir, 'submission_metadata.json')
            
            # Add submission specific fields
            submission_metadata = copy.deepcopy(metadata)
            submission_metadata['submission_timestamp'] = datetime.now().strftime("%Y%m%d-%H%M%S")
            submission_metadata['model_file'] = os.path.basename(args.output)
            
            with open(submission_metadata_path, 'w') as f:
                json.dump(submission_metadata, f, indent=2)
            logger.info(f"Submission metadata saved to {submission_metadata_path}")
        
        logger.info("Checkpoint conversion completed successfully.")
        return 0
        
    except Exception as e:
        logger.error(f"Error in checkpoint conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())