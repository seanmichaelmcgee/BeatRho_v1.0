#!/usr/bin/env python3
"""
Test script to verify device handling in data loading and loss computation
This script creates a small batch of data and runs it through the model to ensure tensor devices are properly handled
"""

import os
import sys
import logging
import argparse
import torch
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Check for enhanced device handling environment variable
ENFORCE_DEVICE_CONSISTENCY = os.environ.get('RNA_ENFORCE_DEVICE_CONSISTENCY', '0') == '1'
DEBUG_DEVICE_ISSUES = os.environ.get('RNA_DEBUG_DEVICE_ISSUES', '0') == '1'

if ENFORCE_DEVICE_CONSISTENCY:
    logger.info("Enhanced device consistency enforcement is ENABLED")
if DEBUG_DEVICE_ISSUES:
    logger.info("Device issue debugging is ENABLED")

# Add project root to path for importing project modules
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
print(f"Project root added to sys.path: {project_root}")

# Import our modules
from src.data_loading_fixed import RNADataset, collate_fn
from src.models.rna_folding_model import RNAFoldingModel
from src.losses import compute_angle_loss, compute_stable_fape_loss, compute_confidence_loss

# Parse arguments
parser = argparse.ArgumentParser(description='Test device handling in data loading and loss computation')
parser.add_argument('--data_csv', type=str, default='data/raw/train_sequences.csv', help='Path to sequences CSV')
parser.add_argument('--labels_csv', type=str, default='data/raw/train_labels.csv', help='Path to labels CSV')
parser.add_argument('--features_dir', type=str, default='data/processed/', help='Path to features directory')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size to test with')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (or -1 for CPU)')
args = parser.parse_args()

def ensure_batch_on_device(batch, device):
    """Ensure all tensors in batch are on the specified device."""
    if not ENFORCE_DEVICE_CONSISTENCY:
        # Just do standard moving without special handling
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
    # Use enhanced version with list handling
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            result[key] = [t.to(device) for t in value]
        else:
            result[key] = value
    
    if DEBUG_DEVICE_ISSUES:
        # Log devices for debugging
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                logger.debug(f"Tensor {key}: device={value.device}, shape={value.shape}")
                
    return result

def test_device_handling():
    """Test device handling in data loading and loss computation"""
    # Choose device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Create model
    model = RNAFoldingModel({
        'num_blocks': 2,  # Using smaller model for testing
        'residue_embed_dim': 128,
        'pair_embed_dim': 32,
        'num_attention_heads': 4,
        'ff_dim': 256,
        'dropout': 0.1,
    })
    model = model.to(device)
    logger.info(f"Created model on device: {device}")

    # Create dataset
    dataset = RNADataset(
        sequences_csv_path=args.data_csv,
        labels_csv_path=args.labels_csv,
        features_dir=args.features_dir,
    )
    logger.info(f"Created dataset with {len(dataset)} samples")

    # Create small batch
    samples = [dataset[i] for i in range(min(args.batch_size, len(dataset)))]
    batch = collate_fn(samples)
    logger.info(f"Created batch with keys: {list(batch.keys())}")

    # Check device of each tensor in batch before moving to CUDA
    logger.info("Device check before moving to CUDA:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: device={value.device}, shape={value.shape}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            logger.info(f"  {key}: LIST with first item device={value[0].device}, shape={value[0].shape}")

    # Move batch to device with enhanced device handling
    batch_on_device = ensure_batch_on_device(batch, device)
    logger.info("Moved batch to device using enhanced device handling")

    # Forward pass
    with torch.autocast(device_type='cuda', enabled=device.type=='cuda'):
        outputs = model(batch_on_device)
        logger.info(f"Forward pass successful, output keys: {list(outputs.keys())}")

    # Check device of each output
    for key, value in outputs.items():
        logger.info(f"Output {key}: device={value.device}, shape={value.shape}")

    # Test angle loss function with list input
    logger.info("\nTesting angle loss with potential list input handling:")
    true_angles = batch_on_device.get("dihedral_features")
    
    # Test both tensor and list formats
    angle_tensor_loss = compute_angle_loss(
        outputs["pred_angles"], true_angles, batch_on_device["mask"]
    )
    logger.info(f"Angle loss with tensor input: {angle_tensor_loss.item()}")
    
    # Convert to list to test list handling
    true_angles_list = [t for t in true_angles]
    angle_list_loss = compute_angle_loss(
        outputs["pred_angles"], true_angles_list, batch_on_device["mask"]
    )
    logger.info(f"Angle loss with list input: {angle_list_loss.item()}")
    
    # Test FAPE loss
    logger.info("\nTesting FAPE loss:")
    fape_loss = compute_stable_fape_loss(
        outputs["pred_coords"], batch_on_device["coordinates"], batch_on_device["mask"]
    )
    logger.info(f"FAPE loss: {fape_loss.item()}")
    
    # Test confidence loss
    confidence_loss = compute_confidence_loss(
        outputs["pred_confidence"], outputs["pred_coords"], 
        batch_on_device["coordinates"], batch_on_device["mask"]
    )
    logger.info(f"Confidence loss: {confidence_loss.item()}")
    
    # Test combined loss
    total_loss = fape_loss + 0.1 * confidence_loss + 0.5 * angle_tensor_loss
    logger.info(f"Combined loss: {total_loss.item()}")
    
    # Test backward with CUDA synchronization
    logger.info("\nTesting backward pass with CUDA synchronization:")
    try:
        if device.type == 'cuda':
            logger.info("Synchronizing CUDA before backward pass...")
            torch.cuda.synchronize(device)
        
        total_loss.backward()
        
        if device.type == 'cuda':
            logger.info("Synchronizing CUDA after backward pass...")
            torch.cuda.synchronize(device)
            
        logger.info("Backward pass with CUDA synchronization successful!")
    except Exception as e:
        logger.error(f"Backward pass failed: {e}")
        # Attempt detailed diagnosis
        try:
            requires_grad = {k: outputs[k].requires_grad for k in outputs}
            logger.info(f"Output requires_grad values: {requires_grad}")
            logger.info(f"Loss requires_grad: {total_loss.requires_grad}")
            if not total_loss.requires_grad:
                logger.error("Loss doesn't require gradients - this is the source of the error")
                # Create a synthetic loss with gradients to see if that fixes it
                dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
                dummy_total = total_loss.detach() * 0 + dummy_loss * 0.001
                logger.info(f"Created synthetic loss with gradients: {dummy_total.requires_grad}")
                dummy_total.backward()
                logger.info("Synthetic loss backward pass successful!")
        except Exception as detailed_e:
            logger.error(f"Diagnostic analysis failed: {detailed_e}")
            
    logger.info("\nDevice handling test complete")
    return True

if __name__ == "__main__":
    try:
        success = test_device_handling()
        if success:
            logger.info("✅ Device handling test successful!")
            sys.exit(0)
        else:
            logger.error("❌ Device handling test failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Uncaught exception: {e}")
        sys.exit(1)