#!/usr/bin/env python3
# FIXED VERSION - Uses direct imports instead of package imports
"""
Enhanced Training Script for RNA 3D Structure Prediction (V2)

This script extends the enhanced training pipeline with:
1. Mixed precision training (FP16) for memory efficiency
2. Curriculum learning that gradually increases sequence length
3. Memory monitoring and optimization
4. Advanced checkpointing with crash recovery
5. Gradient checkpointing option for transformers and IPA module
6. Comprehensive validation hook for dual-mode validation
7. TM-score validation metrics for structure quality assessment
8. Test vs. training mode performance analysis
"""

import os
import sys
import argparse
import json
import time
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Add project root to path for importing project modules
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
print(f"Project root added to sys.path: {project_root}")

# Print sys.path for debugging
print("Python sys.path:")
for p in sys.path:
    print(f"  - {p}")

# Try direct imports
try:
    import src
    print(f"Successfully imported src module: {src.__file__}")
except ImportError as e:
    print(f"Error importing src module: {e}")

# Add pipeline directory to path
pipeline_dir = project_root / "pipeline"
if pipeline_dir.exists():
    sys.path.insert(0, str(pipeline_dir))
    print(f"Pipeline directory added to sys.path: {pipeline_dir}")

# Import project modules
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'src/models'))
from models.rna_folding_model import RNAFoldingModel
from data_loading_fixed import RNADataset, collate_fn, create_data_loader  # Use fixed data loading module
from losses import compute_stable_fape_loss, compute_confidence_loss, compute_angle_loss

# Import pipeline utilities if available
try:
    from pipeline.src.utils.memory_tracker import MemoryTracker, setup_memory_optimizations
    from pipeline.src.utils.curriculum import (
        CurriculumManager, create_length_limited_collate_fn, analyze_dataset_lengths
    )
    from pipeline.src.utils.checkpoint_manager import CheckpointManager
    from pipeline.src.utils.gradient_checkpointing import (
        apply_gradient_checkpointing_to_transformer,
        apply_checkpointing_to_ipa
    )
    PIPELINE_UTILS_AVAILABLE = True
    print("Successfully imported pipeline utilities")
except ImportError as e:
    print(f"Pipeline utilities import error: {e}")
    PIPELINE_UTILS_AVAILABLE = False

# Configure logging - default level will be overridden by command line args
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Check for enhanced device handling environment variable
ENFORCE_DEVICE_CONSISTENCY = os.environ.get('RNA_ENFORCE_DEVICE_CONSISTENCY', '0') == '1'
DEBUG_DEVICE_ISSUES = os.environ.get('RNA_DEBUG_DEVICE_ISSUES', '0') == '1'

if ENFORCE_DEVICE_CONSISTENCY:
    logger.info("Enhanced device consistency enforcement is ENABLED")
if DEBUG_DEVICE_ISSUES:
    logger.info("Device issue debugging is ENABLED")


def parse_args():
    """Parse command line arguments with enhanced options."""
    parser = argparse.ArgumentParser(description='Train an enhanced RNA folding model (V2)')
    
    # Data parameters
    parser.add_argument('--train_csv', type=str, default='data/raw/train_sequences.csv',
                       help='Path to training sequences CSV')
    parser.add_argument('--labels_csv', type=str, default='data/raw/train_labels.csv',
                       help='Path to training labels CSV with 3D coordinates')
    parser.add_argument('--features_dir', type=str, default='data/processed/',
                       help='Path to processed features directory')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--temporal_cutoff', type=str, default='2022-05-01',
                       help='Temporal cutoff date for features')
    parser.add_argument('--max_seq_len', type=int, default=300,
                       help='Maximum sequence length for training')
    
    # Model architecture parameters
    parser.add_argument('--num_blocks', type=int, default=6,
                       help='Number of transformer blocks')
    parser.add_argument('--residue_embed_dim', type=int, default=192,
                       help='Residue embedding dimension')
    parser.add_argument('--pair_embed_dim', type=int, default=64,
                       help='Pair embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=512,
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=2,
                       help='Number of gradient accumulation steps (effective batch size = batch_size * grad_accum_steps)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['plateau', 'cosine', 'none'],
                       help='LR scheduler type')
    
    # Loss weights
    parser.add_argument('--fape_weight', type=float, default=1.0,
                       help='Weight for FAPE loss')
    parser.add_argument('--confidence_weight', type=float, default=0.1,
                       help='Weight for confidence loss')
    parser.add_argument('--angle_weight', type=float, default=0.5,
                       help='Weight for angle loss')
    
    # Memory and performance optimization
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Enable mixed precision training with autocast')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing to reduce memory usage')
    parser.add_argument('--memory_fraction_warning', type=float, default=0.85,
                       help='Fraction of GPU memory usage to trigger warnings')
    parser.add_argument('--memory_fraction_critical', type=float, default=0.92,
                       help='Fraction of GPU memory usage to trigger emergency actions')
    
    # Curriculum learning
    parser.add_argument('--curriculum_learning', action='store_true',
                       help='Enable curriculum learning by sequence length')
    parser.add_argument('--curriculum_stages', type=int, nargs='+', 
                       default=[100, 150, 200, 250, 300],
                       help='Sequence length stages for curriculum learning')
    parser.add_argument('--epochs_per_stage', type=int, default=5,
                       help='Minimum epochs per curriculum stage')
    parser.add_argument('--batch_adaptive', action='store_true',
                       help='Dynamically adapt batch size based on sequence length')
    
    # Checkpointing and save options
    parser.add_argument('--save_interval_epochs', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--save_interval_steps', type=int, default=None,
                       help='Save checkpoint every N steps (if specified)')
    parser.add_argument('--max_checkpoints', type=int, default=3,
                       help='Maximum number of checkpoints to keep (0 for unlimited)')
    
    # Training options
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')
    parser.add_argument('--output_dir', type=str, default='results/enhanced_model',
                       help='Output directory for saving models and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--resume_reset_optimizer', action='store_true',
                       help='Reset optimizer when resuming training')
    parser.add_argument('--resume_reset_scheduler', action='store_true',
                       help='Reset learning rate scheduler when resuming training')
    parser.add_argument('--resume_reset_curriculum', action='store_true',
                       help='Reset curriculum stage when resuming training')
    parser.add_argument('--validate_checkpoints', action='store_true',
                       help='Run full validation on best checkpoints')
    
    # Comprehensive validation options
    parser.add_argument('--comprehensive_val', action='store_true',
                       help='Enable comprehensive validation with ValidationRunner')
    parser.add_argument('--comprehensive_val_freq', type=int, default=5,
                       help='Run comprehensive validation every N epochs')
    parser.add_argument('--comprehensive_val_subset', type=str, default='technical',
                       choices=['technical', 'scientific', 'comprehensive'],
                       help='Default validation subset to use for comprehensive validation')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with small dataset')
    parser.add_argument('--debug_samples', type=int, default=20,
                       help='Number of samples to use in debug mode')
    parser.add_argument('--profile', action='store_true',
                       help='Profile one training step and exit')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_output_dirs(args):
    """Setup output directories for logs and model checkpoints."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    # Setup file logging
    file_handler = logging.FileHandler(os.path.join(run_dir, "logs", "training.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    # Save config
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return run_dir


def create_model(args):
    """Create model with specified architecture."""
    model_config = {
        'num_blocks': args.num_blocks,
        'residue_embed_dim': args.residue_embed_dim,
        'pair_embed_dim': args.pair_embed_dim,
        'num_attention_heads': args.num_heads,
        'ff_dim': args.ff_dim,
        'dropout': args.dropout,
    }
    
    model = RNAFoldingModel(model_config)
    
    # Apply gradient checkpointing if enabled
    if args.gradient_checkpointing and PIPELINE_UTILS_AVAILABLE:
        logger.info("Applying gradient checkpointing to transformer blocks")
        apply_gradient_checkpointing_to_transformer(model)
        logger.info("Applying gradient checkpointing to IPA module")
        apply_checkpointing_to_ipa(model)
    
    return model, model_config


def create_datasets(args, curriculum_manager=None, debug=False):
    """Create training and validation datasets."""
    # Create full dataset
    dataset = RNADataset(
        sequences_csv_path=args.train_csv,
        labels_csv_path=args.labels_csv,
        features_dir=args.features_dir,
        temporal_cutoff=args.temporal_cutoff,
        # Note: max_seq_len is handled during collation, not in dataset creation
    )
    
    # For debug mode, use a small subset
    if debug:
        logger.info(f"Debug mode: using {args.debug_samples} samples")
        indices = list(range(min(args.debug_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Split into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"Dataset split: {train_size} training, {val_size} validation")
    
    # Apply curriculum filtering if enabled
    if curriculum_manager is not None and PIPELINE_UTILS_AVAILABLE:
        try:
            # Import the safer dataset analyzer
            from scripts.fix_dataset_analyzer import analyze_rna_dataset_lengths
            
            # Analyze sequence lengths with our safer function
            logger.info("Analyzing dataset lengths with enhanced analyzer...")
            train_stats = analyze_rna_dataset_lengths(train_dataset)
            val_stats = analyze_rna_dataset_lengths(val_dataset)
            
            # Log stats if available
            if train_stats["count"] > 0:
                logger.info(f"Training dataset length stats: min={train_stats['min']}, "
                          f"max={train_stats['max']}, mean={train_stats['mean']:.1f}")
            else:
                logger.warning("No valid lengths found in training dataset. Skipping curriculum filtering.")
                
            if val_stats["count"] > 0:
                logger.info(f"Validation dataset length stats: min={val_stats['min']}, "
                          f"max={val_stats['max']}, mean={val_stats['mean']:.1f}")
            
            # Only apply curriculum filtering if we found valid lengths
            if train_stats["count"] > 0:
                try:
                    # Get current max length from curriculum manager
                    curr_max_len = curriculum_manager.get_current_max_length()
                    logger.info(f"Applying curriculum filtering with max length {curr_max_len}...")
                    
                    # Define a length getter function for the dataset
                    def get_length(sample):
                        if isinstance(sample, dict) and "length" in sample:
                            return sample["length"]
                        elif isinstance(sample, dict) and "sequence_int" in sample:
                            return len(sample["sequence_int"])
                        else:
                            return 0
                    
                    # Filter datasets based on current curriculum stage
                    filtered_dataset = curriculum_manager.get_filtered_dataset(
                        train_dataset, length_key=get_length
                    )
                    
                    # Only use filtered dataset if it's not empty
                    if len(filtered_dataset) > 0:
                        logger.info(f"Curriculum filtering applied: {len(train_dataset)} -> {len(filtered_dataset)} samples")
                        train_dataset = filtered_dataset
                    else:
                        logger.warning("Curriculum filtering resulted in empty dataset. Using original dataset.")
                except Exception as e:
                    logger.warning(f"Error during curriculum filtering: {e}")
                    logger.warning("Using original dataset without curriculum filtering.")
        except Exception as e:
            logger.warning(f"Error during dataset length analysis: {e}")
            logger.warning("Proceeding without curriculum filtering.")
    
    return train_dataset, val_dataset


def create_dataloaders(args, train_dataset, val_dataset, curriculum_manager=None):
    """Create training and validation data loaders with curriculum-aware batching."""
    # Determine batch size and sequence length limit
    batch_size = args.batch_size
    max_seq_len = args.max_seq_len
    
    # If using curriculum learning, override with current stage params
    if curriculum_manager is not None and PIPELINE_UTILS_AVAILABLE:
        max_seq_len = curriculum_manager.get_current_max_length()
        
        if curriculum_manager.batch_adaptive:
            batch_size, _ = curriculum_manager.get_batch_params()
    
    # Define custom collate function that respects max_seq_len
    if PIPELINE_UTILS_AVAILABLE:
        length_limited_collate = create_length_limited_collate_fn(max_seq_len, collate_fn)
    else:
        # Fallback to original collate function and custom implementation
        def length_limited_collate(batch):
            # Safety check for empty batch
            if not batch:
                return {}
                
            # Get batch size
            batch_size = len(batch)
            
            # Safe function to get sample length
            def get_sample_length(sample):
                if isinstance(sample, dict) and "length" in sample:
                    return sample["length"]
                elif isinstance(sample, dict) and "sequence_int" in sample:
                    return len(sample["sequence_int"])
                elif hasattr(sample, "length"):
                    return sample.length
                elif hasattr(sample, "__len__"):
                    return len(sample)
                else:
                    logger.warning(f"Cannot determine length for sample: {type(sample)}")
                    return 0
            
            # Get max length in batch (safely)
            try:
                sample_lengths = [get_sample_length(sample) for sample in batch]
                if not sample_lengths or max(sample_lengths) == 0:
                    logger.warning("No valid lengths found in batch. Using default max_seq_len.")
                    max_len = max_seq_len
                else:
                    max_len = min(max(sample_lengths), max_seq_len)
            except Exception as e:
                logger.warning(f"Error calculating max length: {e}")
                max_len = max_seq_len
            
            # Apply sequence length limit to each sample if needed
            for sample in batch:
                sample_len = get_sample_length(sample)
                
                if sample_len > max_seq_len:
                    # Trim all sequence-related tensors
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            # Trim 1D tensors
                            if len(value.shape) == 1 and value.shape[0] > max_seq_len:
                                sample[key] = value[:max_seq_len]
                            # Trim 2D tensors
                            elif len(value.shape) == 2 and value.shape[0] > max_seq_len:
                                sample[key] = value[:max_seq_len, :min(value.shape[1], max_seq_len)]
                    
                    # Update the length
                    if isinstance(sample, dict):
                        sample["length"] = max_seq_len
            
            # Use the original collate function now that lengths are restricted
            return collate_fn(batch)
    
    # Create data loaders with the custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=length_limited_collate,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=length_limited_collate,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device, loss_weights, 
               grad_accum_steps=1, memory_tracker=None, scaler=None, 
               step_offset=0, checkpoint_manager=None, curriculum_manager=None):
    """Train for one epoch with gradient accumulation and mixed precision."""
    model.train()
    total_loss = 0
    fape_losses = 0
    conf_losses = 0
    angle_losses = 0
    
    # Initialize counters for gradient accumulation
    batch_count = 0
    accumulated_loss = 0
    global_step = step_offset
    
    # Wrap train_loader with tqdm for progress tracking
    from tqdm import tqdm
    progress_bar = tqdm(train_loader, desc="Training")
    
    # Zero gradients at the beginning
    optimizer.zero_grad()
    
    for batch in progress_bar:
        # Update memory tracking if available
        if memory_tracker:
            memory_tracker.update(f"start of batch {batch_count+1}")
            
            # Check for memory pressure
            if memory_tracker.should_checkpoint() and checkpoint_manager:
                # Create emergency checkpoint
                logger.warning("Memory pressure detected! Creating emergency checkpoint")
                checkpoint_manager.save_checkpoint(
                    model=model,
                    epoch=global_step // len(train_loader),
                    step=global_step,
                    optimizer=optimizer,
                    metrics={'in_progress': True},
                    emergency=True
                )
        
        # Move batch to device with enhanced device handling
        batch = ensure_batch_on_device(batch, device)
        
        # Forward pass with mixed precision if enabled
        if scaler:
            with autocast():
                # Forward pass
                outputs = model(batch)
                
                # Compute losses
                fape_loss = compute_stable_fape_loss(
                    pred_coords=outputs["pred_coords"],
                    true_coords=batch["coordinates"],
                    mask=batch["mask"],
                )
                
                confidence_loss = compute_confidence_loss(
                    pred_confidence=outputs["pred_confidence"],
                    true_coords=batch["coordinates"],
                    pred_coords=outputs["pred_coords"],
                    mask=batch["mask"],
                )
                
                angle_loss = compute_angle_loss(
                    pred_angles=outputs["pred_angles"],
                    true_angles=batch["dihedral_features"],
                    mask=batch["mask"],
                )
                
                # Combine losses
                loss = (
                    loss_weights["fape"] * fape_loss
                    + loss_weights["confidence"] * confidence_loss
                    + loss_weights["angle"] * angle_loss
                )
                
                # Scale loss by gradient accumulation steps
                loss = loss / grad_accum_steps
            
            # Check if loss is detached/doesn't require gradients (happens with invalid inputs)
            if not loss.requires_grad:
                logger.warning(f"Loss doesn't require gradients. Creating synthetic loss with gradients.")
                # Create a synthetic loss with gradients to avoid backward issues
                # This allows training to continue by essentially skipping problematic batches
                dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
                loss = loss.detach() * 0 + dummy_loss * 0.001
            
            # Backward pass with scaler, with CUDA synchronization
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            scaler.scale(loss).backward()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            
        else:
            # Standard forward pass without mixed precision
            outputs = model(batch)
            
            # Compute losses
            fape_loss = compute_stable_fape_loss(
                pred_coords=outputs["pred_coords"],
                true_coords=batch["coordinates"],
                mask=batch["mask"],
            )
            
            confidence_loss = compute_confidence_loss(
                pred_confidence=outputs["pred_confidence"],
                true_coords=batch["coordinates"],
                pred_coords=outputs["pred_coords"],
                mask=batch["mask"],
            )
            
            angle_loss = compute_angle_loss(
                pred_angles=outputs["pred_angles"],
                true_angles=batch["dihedral_features"],
                mask=batch["mask"],
            )
            
            # Combine losses
            loss = (
                loss_weights["fape"] * fape_loss
                + loss_weights["confidence"] * confidence_loss
                + loss_weights["angle"] * angle_loss
            )
            
            # Scale loss by gradient accumulation steps
            loss = loss / grad_accum_steps
            
            # Check if loss is detached/doesn't require gradients (happens with invalid inputs)
            if not loss.requires_grad:
                logger.warning(f"Loss doesn't require gradients. Creating synthetic loss with gradients.")
                # Create a synthetic loss with gradients to avoid backward issues
                # This allows training to continue by essentially skipping problematic batches
                dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
                loss = loss.detach() * 0 + dummy_loss * 0.001
            
            # Backward pass with CUDA synchronization
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            loss.backward()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        
        # Track metrics (use unscaled loss for logging)
        unscaled_loss = loss.item() * grad_accum_steps
        total_loss += unscaled_loss
        fape_losses += fape_loss.item()
        conf_losses += confidence_loss.item()
        angle_losses += angle_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': unscaled_loss, 
            'fape': fape_loss.item(),
            'conf': confidence_loss.item(),
            'angle': angle_loss.item()
        })
        
        # Increment batch counter
        batch_count += 1
        global_step += 1
        
        # Optimize after accumulating gradients for specified number of steps
        if batch_count % grad_accum_steps == 0:
            # Add gradient clipping for stability (even without mixed precision)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if scaler:
                # Step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard step
                optimizer.step()
                
            optimizer.zero_grad()
            accumulated_loss = 0
            
            # Check if we should save intermediate checkpoint
            if checkpoint_manager and checkpoint_manager.save_interval_steps is not None:
                if global_step % checkpoint_manager.save_interval_steps == 0:
                    checkpoint_manager.save_checkpoint(
                        model=model,
                        epoch=global_step // len(train_loader),
                        step=global_step,
                        optimizer=optimizer,
                        metrics={'in_progress': True}
                    )
        
        # Update memory tracking after batch
        if memory_tracker:
            memory_tracker.update(f"end of batch {batch_count}")
    
    # Perform final optimization step if there are any remaining gradients
    if batch_count % grad_accum_steps != 0:
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if scaler:
            # Step with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard step
            optimizer.step()
            
        optimizer.zero_grad()
    
    # Calculate averages
    avg_loss = total_loss / len(train_loader)
    avg_fape = fape_losses / len(train_loader)
    avg_conf = conf_losses / len(train_loader)
    avg_angle = angle_losses / len(train_loader)
    
    return {
        "loss": avg_loss,
        "fape_loss": avg_fape,
        "confidence_loss": avg_conf,
        "angle_loss": avg_angle,
    }, global_step


def validate(model, val_loader, device, loss_weights, scaler=None, memory_tracker=None):
    """Validate the model with mixed precision support and enhanced structure metrics."""
    from src.utils.structure_metrics import compute_rmsd, compute_tm_score
    
    model.eval()
    total_loss = 0
    fape_losses = 0
    conf_losses = 0
    angle_losses = 0
    all_rmsd = []
    all_tm_scores = []
    
    # Set validation batch size to 1 to avoid shape mismatch issues
    val_batch_size = 1
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                # Update memory tracking if available
                if memory_tracker:
                    memory_tracker.update("validation batch")
                
                # Log tensor shapes for debugging
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"{key} shape: {value.shape}")
                
                # Move batch to device with enhanced device handling
                batch = ensure_batch_on_device(batch, device)
                
                # Forward pass with mixed precision if enabled
                if scaler:
                    with autocast():
                        # Forward pass
                        outputs = model(batch)
                        
                        # Compute losses
                        fape_loss = compute_stable_fape_loss(
                            pred_coords=outputs["pred_coords"],
                            true_coords=batch["coordinates"],
                            mask=batch["mask"],
                        )
                        
                        confidence_loss = compute_confidence_loss(
                            pred_confidence=outputs["pred_confidence"],
                            true_coords=batch["coordinates"],
                            pred_coords=outputs["pred_coords"],
                            mask=batch["mask"],
                        )
                        
                        angle_loss = compute_angle_loss(
                            pred_angles=outputs["pred_angles"],
                            true_angles=batch["dihedral_features"],
                            mask=batch["mask"],
                        )
                        
                        # Combine losses
                        loss = (
                            loss_weights["fape"] * fape_loss
                            + loss_weights["confidence"] * confidence_loss
                            + loss_weights["angle"] * angle_loss
                        )
                else:
                    # Standard forward pass without mixed precision
                    outputs = model(batch)
                    
                    # Compute losses
                    fape_loss = compute_stable_fape_loss(
                        pred_coords=outputs["pred_coords"],
                        true_coords=batch["coordinates"],
                        mask=batch["mask"],
                    )
                    
                    confidence_loss = compute_confidence_loss(
                        pred_confidence=outputs["pred_confidence"],
                        true_coords=batch["coordinates"],
                        pred_coords=outputs["pred_coords"],
                        mask=batch["mask"],
                    )
                    
                    angle_loss = compute_angle_loss(
                        pred_angles=outputs["pred_angles"],
                        true_angles=batch["dihedral_features"],
                        mask=batch["mask"],
                    )
                    
                    # Combine losses
                    loss = (
                        loss_weights["fape"] * fape_loss
                        + loss_weights["confidence"] * confidence_loss
                        + loss_weights["angle"] * angle_loss
                    )
                
                # Track metrics
                total_loss += loss.item()
                fape_losses += fape_loss.item()
                conf_losses += confidence_loss.item()
                angle_losses += angle_loss.item()
                
                # Calculate enhanced structure metrics for each sequence in batch
                for i in range(len(batch["lengths"])):
                    try:
                        # Get sequence length for this sample
                        seq_len = batch["lengths"][i].item()
                        
                        # Extract and prepare coordinates, keeping them on device for structure metrics
                        pred_coords_i = outputs["pred_coords"][i, :seq_len]
                        
                        # Check for NaN/Inf in true coordinates
                        if torch.isnan(batch["coordinates"][i]).any() or torch.isinf(batch["coordinates"][i]).any():
                            logger.warning("NaN/Inf detected in true coordinates. Skipping structure metrics.")
                            all_rmsd.append(float('nan'))
                            all_tm_scores.append(0.0)
                            continue
                        
                        true_coords_i = batch["coordinates"][i, :seq_len]
                        
                        # Create mask for this sequence (all True up to seq_len)
                        mask_i = batch["mask"][i, :seq_len] if "mask" in batch else None
                        
                        # Compute proper RMSD with alignment
                        rmsd = compute_rmsd(
                            pred_coords=pred_coords_i.unsqueeze(0),  # Add batch dimension
                            true_coords=true_coords_i.unsqueeze(0),
                            mask=mask_i.unsqueeze(0) if mask_i is not None else None,
                            aligned=True  # Apply optimal alignment before RMSD calculation
                        )
                        all_rmsd.append(rmsd.item())  # Extract scalar value
                        
                        # Add TM-score calculation
                        tm_score = compute_tm_score(
                            pred_coords=pred_coords_i.unsqueeze(0),
                            true_coords=true_coords_i.unsqueeze(0),
                            mask=mask_i.unsqueeze(0) if mask_i is not None else None
                        )
                        all_tm_scores.append(tm_score.item())
                        
                    except Exception as e:
                        logger.error(f"Error calculating structure metrics for sample {i} in batch {batch_idx}: {e}")
                        all_rmsd.append(float('nan'))
                        all_tm_scores.append(0.0)
            except RuntimeError as e:
                # Log error but continue with next batch
                logger.error(f"Error during validation batch {batch_idx}: {e}")
                # If we encounter an error, add a placeholder value to maintain counting
                total_loss += 0.0
                fape_losses += 0.0
                conf_losses += 0.0
                angle_losses += 0.0
    
    # Calculate averages
    avg_loss = total_loss / len(val_loader)
    avg_fape = fape_losses / len(val_loader)
    avg_conf = conf_losses / len(val_loader)
    avg_angle = angle_losses / len(val_loader)
    
    # Filter out NaN values for structure metrics
    valid_rmsd = [r for r in all_rmsd if not (np.isnan(r) or np.isinf(r))]
    valid_tm_scores = [t for t in all_tm_scores if not (np.isnan(t) or np.isinf(t))]
    
    avg_rmsd = np.mean(valid_rmsd) if valid_rmsd else float('nan')
    avg_tm_score = np.mean(valid_tm_scores) if valid_tm_scores else 0.0
    
    return {
        "loss": avg_loss,
        "fape_loss": avg_fape,
        "confidence_loss": avg_conf,
        "angle_loss": avg_angle,
        "rmsd": avg_rmsd,
        "tm_score": avg_tm_score,
    }


def run_comprehensive_validation(model, epoch, run_dir, device, args, is_final=False):
    """
    Run comprehensive validation using ValidationRunner for both test and training modes.
    
    This function implements the validation hook needed for Day 2 tasks, providing
    detailed structural validation metrics beyond the basic validation done during training.
    
    Args:
        model: The model to validate
        epoch: Current training epoch
        run_dir: Directory for run outputs
        device: Device to run validation on
        args: Training arguments
        is_final: Whether this is the final validation after training
        
    Returns:
        Dictionary of validation results with key metrics
    """
    try:
        # Import ValidationRunner
        from validation.validation_runner import ValidationRunner

        # Create validation results directory within the run directory
        validation_dir = os.path.join(run_dir, "validation_results")
        os.makedirs(validation_dir, exist_ok=True)
        logger.info(f"Running comprehensive validation (epoch {epoch}) - both test and training modes")
        
        # Configure validation to use a specific output directory
        validation_config = {
            "batch_size": min(4, args.batch_size),  # Use smaller batch size for validation
            "results_dir": validation_dir,           # Store results in the run directory
            "save_results": True,                    # Save visualization results
            "verbose": False,                        # Limit output unless debugging
            "max_sequence_length": args.max_seq_len  # Use same max length as training
        }
        
        # Create validation runner with current model state
        validator = ValidationRunner(
            model=model,
            config=validation_config,
            device=device
        )
        
        # Determine validation subset based on epoch
        # Use technical validation for regular epochs, comprehensive for final validation
        subset_name = "technical"
        run_both_modes = True  # Always run both test and training modes
        
        if is_final or (epoch % 10 == 0 and epoch > 0):  # More comprehensive validation periodically
            subset_name = "scientific"
            
        # Execute validation
        validation_results = validator.run_validation(subset_name, run_both_modes=run_both_modes)
        
        # Generate timestamp for filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save validation results in JSON format
        results_file = os.path.join(validation_dir, f"validation_results_epoch_{epoch}_{timestamp}.json")
        with open(results_file, 'w') as f:
            # Convert results to JSON-serializable format
            serializable_results = validator._make_serializable(validation_results)
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Comprehensive validation results saved to {results_file}")
        
        # Log key metrics for easy tracking
        test_rmsd = validation_results["test_mode"].get("mean_rmsd", float('nan'))
        train_rmsd = validation_results["train_mode"].get("mean_rmsd", float('nan')) if "train_mode" in validation_results else float('nan')
        test_tm = validation_results["test_mode"].get("mean_tm_score", float('nan'))
        train_tm = validation_results["train_mode"].get("mean_tm_score", float('nan')) if "train_mode" in validation_results else float('nan')
        
        logger.info(f"Validation at epoch {epoch}:")
        logger.info(f"  RMSD: Test={test_rmsd:.4f}Å, Train={train_rmsd:.4f}Å, Diff={(test_rmsd-train_rmsd):.4f}Å")
        logger.info(f"  TM-score: Test={test_tm:.4f}, Train={train_tm:.4f}, Diff={(train_tm-test_tm):.4f}")
        
        # Create summary of results for later reference
        summary = {
            "epoch": epoch,
            "timestamp": timestamp,
            "test_mode": {
                "rmsd": test_rmsd,
                "tm_score": test_tm
            },
            "train_mode": {
                "rmsd": train_rmsd,
                "tm_score": train_tm
            },
            "subset": subset_name
        }
        
        # Add analysis summary if available
        if "analysis" in validation_results and "conclusion" in validation_results["analysis"]:
            conclusion = validation_results["analysis"]["conclusion"]
            summary["analysis"] = {
                "impact": conclusion.get("overall_impact", "UNKNOWN"),
                "severity": conclusion.get("severity", "UNKNOWN")
            }
        
        return summary
    
    except Exception as e:
        logger.error(f"Error during comprehensive validation: {e}")
        logger.exception("Validation error details:")
        # Return minimal results to avoid training pipeline disruption
        return {
            "epoch": epoch,
            "error": str(e),
            "test_mode": {"rmsd": float('nan'), "tm_score": 0.0},
            "train_mode": {"rmsd": float('nan'), "tm_score": 0.0}
        }


def create_optimizer(model, args):
    """Create optimizer and scheduler."""
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Create scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def validate_training_data(train_loader, val_loader, model, device, loss_weights):
    """Pre-validate data before starting training to catch issues early."""
    logger.info("Pre-validating training data...")
    
    # Check train loader
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty! Check dataset and curriculum settings.")
    
    # Check batch shapes and contents
    try:
        sample_batch = next(iter(train_loader))
        logger.info(f"Sample batch keys: {list(sample_batch.keys())}")
        
        # Check for required keys
        required_keys = ['sequence_int', 'pairing_probs', 'mask']
        missing_keys = [k for k in required_keys if k not in sample_batch]
        if missing_keys:
            raise ValueError(f"Sample batch missing required keys: {missing_keys}")
            
        # Check tensor shapes
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"  {k}: {v.shape}, {v.dtype}")
                
        # Validate sequence lengths
        if 'mask' in sample_batch:
            lengths = sample_batch['mask'].sum(dim=1)
            min_len = lengths.min().item()
            if min_len < 3:
                raise ValueError(f"Sequences too short for Kabsch alignment! Min length: {min_len}, need at least 3.")
                
        # Try a forward pass
        logger.info("Testing forward pass...")
        model.eval()
        with torch.no_grad():
            batch_device = ensure_batch_on_device(sample_batch, device)
            outputs = model(batch_device)
            
            # Check outputs
            logger.info(f"Model output keys: {list(outputs.keys())}")
            
            # Try loss computation
            fape_loss = compute_stable_fape_loss(
                outputs["pred_coords"], batch_device["coordinates"], batch_device["mask"]
            )
            logger.info(f"Sample FAPE loss: {fape_loss.item()}")
            
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise ValueError(f"Training data validation failed! {str(e)}")
        
    logger.info("Data validation successful! Training can proceed.")

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

def log_metrics(metrics, epoch, prefix="train"):
    """Log metrics to console."""
    metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
    logger.info(f"{prefix.capitalize()} Epoch {epoch}: {metrics_str}")


def save_training_log(train_log, val_log, run_dir):
    """Save training logs to CSV file with enhanced metrics."""
    # Check if TM-score is available in validation logs
    has_tm_score = all("tm_score" in log for log in val_log)
    
    # Create base dataframe with common metrics
    log_data = {
        "epoch": range(1, len(train_log) + 1),
        "train_loss": [log["loss"] for log in train_log],
        "train_fape_loss": [log["fape_loss"] for log in train_log],
        "train_confidence_loss": [log["confidence_loss"] for log in train_log],
        "train_angle_loss": [log["angle_loss"] for log in train_log],
        "val_loss": [log["loss"] for log in val_log],
        "val_fape_loss": [log["fape_loss"] for log in val_log],
        "val_confidence_loss": [log["confidence_loss"] for log in val_log],
        "val_angle_loss": [log["angle_loss"] for log in val_log],
        "val_rmsd": [log["rmsd"] for log in val_log],
    }
    
    # Add TM-score if available
    if has_tm_score:
        log_data["val_tm_score"] = [log["tm_score"] for log in val_log]
    
    # Create DataFrame and save
    log_df = pd.DataFrame(log_data)
    log_path = os.path.join(run_dir, "training_log.csv")
    log_df.to_csv(log_path, index=False)
    logger.info(f"Training log saved to {log_path}")
    
    # Also save validation results in a separate file for easier analysis
    val_metrics_df = pd.DataFrame({
        "epoch": range(1, len(val_log) + 1),
        "loss": [log["loss"] for log in val_log],
        "rmsd": [log["rmsd"] for log in val_log],
        **({"tm_score": [log["tm_score"] for log in val_log]} if has_tm_score else {})
    })
    val_path = os.path.join(run_dir, "validation_results.csv")
    val_metrics_df.to_csv(val_path, index=False)
    logger.info(f"Validation results saved to {val_path}")


def plot_metrics(train_log, val_log, run_dir):
    """Plot training and validation metrics including TM-score."""
    epochs = range(1, len(train_log) + 1)
    
    # Check if TM-score is available in log
    has_tm_score = all("tm_score" in log for log in val_log)
    
    # Plot losses
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(epochs, [log["loss"] for log in train_log], 'b-', label='Training Loss')
    plt.plot(epochs, [log["loss"] for log in val_log], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # RMSD plot
    plt.subplot(2, 3, 2)
    plt.plot(epochs, [log["rmsd"] for log in val_log], 'g-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('RMSD (Å)')
    plt.title('Validation RMSD')
    plt.grid(alpha=0.3)
    
    # TM-score plot (if available)
    if has_tm_score:
        plt.subplot(2, 3, 3)
        plt.plot(epochs, [log["tm_score"] for log in val_log], 'm-', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('TM-score')
        plt.title('Validation TM-score')
        plt.grid(alpha=0.3)
        # TM-score ranges from 0 to 1, with 1 being perfect
        plt.ylim(0, 1)
    
    # Component losses
    plt.subplot(2, 3, 4)
    plt.plot(epochs, [log["fape_loss"] for log in train_log], 'b-', label='Train FAPE')
    plt.plot(epochs, [log["fape_loss"] for log in val_log], 'r-', label='Val FAPE')
    plt.xlabel('Epoch')
    plt.ylabel('FAPE Loss')
    plt.title('FAPE Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Angle losses
    plt.subplot(2, 3, 5)
    plt.plot(epochs, [log["angle_loss"] for log in train_log], 'b-', label='Train Angle')
    plt.plot(epochs, [log["angle_loss"] for log in val_log], 'r-', label='Val Angle')
    plt.xlabel('Epoch')
    plt.ylabel('Angle Loss')
    plt.title('Angle Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Confidence losses
    plt.subplot(2, 3, 6)
    plt.plot(epochs, [log["confidence_loss"] for log in train_log], 'b-', label='Train Conf')
    plt.plot(epochs, [log["confidence_loss"] for log in val_log], 'r-', label='Val Conf')
    plt.xlabel('Epoch')
    plt.ylabel('Confidence Loss')
    plt.title('Confidence Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_metrics.png"), dpi=300)
    
    # Also save as separate loss curves file
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [log["loss"] for log in train_log], 'b-', label='Training Loss')
    plt.plot(epochs, [log["loss"] for log in val_log], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=300)
    
    # RMSD plot in separate file
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [log["rmsd"] for log in val_log], 'g-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('RMSD (Å)')
    plt.title('RMSD over Training')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "rmsd_over_training.png"), dpi=300)
    
    # TM-score plot in separate file (if available)
    if has_tm_score:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, [log["tm_score"] for log in val_log], 'm-', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('TM-score')
        plt.title('TM-score over Training')
        plt.grid(alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "tm_score_over_training.png"), dpi=300)
    
    # Generate distribution plots for the final epoch metrics
    # We don't have access to all_rmsd and all_tm_scores here, 
    # but we can plot the distribution from previous epochs
    if len(val_log) > 0:
        try:
            # Get all valid RMSD values from previous epochs
            rmsd_values = [log.get("rmsd", float('nan')) for log in val_log]
            valid_rmsd = [r for r in rmsd_values if not (np.isnan(r) or np.isinf(r))]
            
            if valid_rmsd:
                plt.figure(figsize=(10, 6))
                plt.hist(valid_rmsd, bins=20, alpha=0.7, color='g')
                plt.xlabel('RMSD (Å)')
                plt.ylabel('Frequency')
                plt.title('RMSD Distribution Across Epochs')
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, "rmsd_distribution.png"), dpi=300)
            
            # Create TM-score distribution plot if available
            if has_tm_score:
                tm_values = [log.get("tm_score", float('nan')) for log in val_log]
                valid_tm = [t for t in tm_values if not (np.isnan(t) or np.isinf(t))]
                
                if valid_tm:
                    plt.figure(figsize=(10, 6))
                    plt.hist(valid_tm, bins=20, alpha=0.7, color='m')
                    plt.xlabel('TM-score')
                    plt.ylabel('Frequency')
                    plt.title('TM-score Distribution Across Epochs')
                    plt.grid(alpha=0.3)
                    plt.xlim(0, 1)
                    plt.tight_layout()
                    plt.savefig(os.path.join(run_dir, "tm_score_distribution.png"), dpi=300)
        except Exception as e:
            logger.error(f"Error generating distribution plots: {e}")
    
    plt.close('all')


def plot_curriculum_progress(train_log, val_log, curriculum_manager, run_dir):
    """Plot training progress with curriculum stage transitions."""
    if curriculum_manager is None or not curriculum_manager.stage_history:
        return
    
    epochs = range(1, len(train_log) + 1)
    
    # Plot losses with curriculum stages
    plt.figure(figsize=(12, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [log["loss"] for log in train_log], 'b-', label='Training Loss')
    plt.plot(epochs, [log["loss"] for log in val_log], 'r-', label='Validation Loss')
    
    # Add vertical lines for curriculum transitions
    for transition in curriculum_manager.stage_history:
        plt.axvline(x=transition['epoch'], color='g', linestyle='--', 
                   alpha=0.7, label=f"Stage {transition['to_stage']+1}" if transition == curriculum_manager.stage_history[0] else "")
        plt.text(transition['epoch'], plt.ylim()[1]*0.95, 
                f"→{transition['to_length']}", rotation=90, verticalalignment='top')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss with Curriculum Stages')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # RMSD plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [log["rmsd"] for log in val_log], 'g-', marker='o')
    
    # Add vertical lines for curriculum transitions
    for transition in curriculum_manager.stage_history:
        plt.axvline(x=transition['epoch'], color='g', linestyle='--', alpha=0.7)
        plt.text(transition['epoch'], plt.ylim()[1]*0.95, 
                f"→{transition['to_length']}", rotation=90, verticalalignment='top')
    
    plt.xlabel('Epoch')
    plt.ylabel('RMSD (Å)')
    plt.title('Validation RMSD with Curriculum Stages')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "curriculum_progress.png"), dpi=300)
    plt.close()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set logging level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level)
    logger.info(f"Set logging level to {args.log_level}")
    
    # Silence other loggers for cleaner output
    if log_level >= logging.WARNING:
        for lib_logger in ['matplotlib', 'PIL', 'torch.distributed', 'urllib3', 'dataloader', 'torch._C', 'torch.utils.data']:
            logging.getLogger(lib_logger).setLevel(logging.WARNING)
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup output directories
    run_dir = setup_output_dirs(args)
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Initialize memory tracking if GPU is available
    memory_tracker = None
    if torch.cuda.is_available() and PIPELINE_UTILS_AVAILABLE:
        memory_tracker = MemoryTracker(
            device=device,
            log_interval=20,
            memory_fraction_warn=args.memory_fraction_warning,
            memory_fraction_critical=args.memory_fraction_critical,
            track_history=True
        )
        logger.info("Initialized GPU memory tracking")
        
        # Configure memory optimizations
        setup_memory_optimizations(amp_enabled=args.mixed_precision)
    
    # Initialize curriculum manager if enabled
    curriculum_manager = None
    if args.curriculum_learning and PIPELINE_UTILS_AVAILABLE:
        curriculum_manager = CurriculumManager(
            sequence_stages=args.curriculum_stages,
            epochs_per_stage=args.epochs_per_stage,
            base_batch_size=args.batch_size,
            base_grad_accum_steps=args.grad_accum_steps,
            batch_adaptive=args.batch_adaptive
        )
        logger.info(f"Initialized curriculum learning with stages: {args.curriculum_stages}")
    
    # Initialize checkpoint manager
    checkpoint_manager = None
    if PIPELINE_UTILS_AVAILABLE:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(run_dir, "checkpoints"),
            metric_name='val_loss',
            metric_mode='min',
            save_interval_epochs=args.save_interval_epochs,
            save_interval_steps=args.save_interval_steps,
            max_checkpoints=args.max_checkpoints
        )
        logger.info("Initialized checkpoint manager")
    
    # Initialize mixed precision scaler if enabled
    scaler = None
    if args.mixed_precision and torch.cuda.is_available():
        scaler = GradScaler()
        logger.info("Initialized mixed precision training with autocast")
    
    # Create model
    model, model_config = create_model(args)
    model = model.to(device)
    logger.info(f"Created model with config: {model_config}")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, args)
    
    # Set loss weights
    loss_weights = {
        "fape": args.fape_weight,
        "confidence": args.confidence_weight,
        "angle": args.angle_weight,
    }
    logger.info(f"Using loss weights: {loss_weights}")
    
    # Create datasets with curriculum manager if available
    train_dataset, val_dataset = create_datasets(args, curriculum_manager, debug=args.debug)
    
    # Create dataloaders with curriculum manager if available
    train_loader, val_loader = create_dataloaders(args, train_dataset, val_dataset, curriculum_manager)
    
    logger.info(f"Created dataloaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    # Optionally resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume is not None and os.path.exists(args.resume):
        if checkpoint_manager:
            # Use advanced checkpoint loading
            epoch, step, metrics = checkpoint_manager.load_checkpoint(
                model=model,
                checkpoint_path=args.resume,
                optimizer=optimizer,
                scheduler=scheduler,
                map_location=device,
                reset_optimizer=args.resume_reset_optimizer,
                reset_scheduler=args.resume_reset_scheduler
            )
            
            # Update start epoch and global step
            start_epoch = epoch + 1
            global_step = step if step is not None else epoch * len(train_loader)
            
            # Load curriculum state if available
            if curriculum_manager and not args.resume_reset_curriculum:
                curriculum_state = checkpoint_manager.load_curriculum_state(args.resume)
                if curriculum_state:
                    curriculum_manager.load_state_dict(curriculum_state)
                    
                    # Reload dataloaders with updated curriculum state
                    train_loader, val_loader = create_dataloaders(
                        args, train_dataset, val_dataset, curriculum_manager
                    )
                
        else:
            # Use basic checkpoint loading
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            if not args.resume_reset_optimizer:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if not args.resume_reset_scheduler and scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            start_epoch = checkpoint["epoch"] + 1
        
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    # Pre-validate training data to catch issues early
    try:
        validate_training_data(train_loader, val_loader, model, device, loss_weights)
    except ValueError as e:
        logger.error(f"Training validation failed: {e}")
        logger.error("Please check your dataset and configuration settings.")
        logger.error("Attempting to continue anyway, but training may fail.")
        
    # Initialize tracking variables
    train_log = []
    val_log = []
    best_val_loss = float('inf')
    best_val_rmsd = float('inf')
    patience_counter = 0
    
    # For profiling
    if args.profile:
        logger.info("Running in profile mode - will profile one training step and exit")
        # Run one epoch with profiling
        try:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(run_dir, "profile")),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                # Profile one step
                for batch in train_loader:
                    # Move batch to device with enhanced device handling
                    batch = ensure_batch_on_device(batch, device)
                    
                    # Forward pass
                    outputs = model(batch)
                    
                    # Compute losses
                    fape_loss = compute_stable_fape_loss(
                        pred_coords=outputs["pred_coords"],
                        true_coords=batch["coordinates"],
                        mask=batch["mask"],
                    )
                    
                    confidence_loss = compute_confidence_loss(
                        pred_confidence=outputs["pred_confidence"],
                        true_coords=batch["coordinates"],
                        pred_coords=outputs["pred_coords"],
                        mask=batch["mask"],
                    )
                    
                    angle_loss = compute_angle_loss(
                        pred_angles=outputs["pred_angles"],
                        true_angles=batch["dihedral_features"],
                        mask=batch["mask"],
                    )
                    
                    # Combine losses
                    loss = (
                        loss_weights["fape"] * fape_loss
                        + loss_weights["confidence"] * confidence_loss
                        + loss_weights["angle"] * angle_loss
                    )
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Record profiling step
                    prof.step()
                    
                    # Only profile one batch
                    break
            
            # Print profiling results
            logger.info("Profiling results summary:")
            logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            logger.info(f"Profile data saved to {os.path.join(run_dir, 'profile')}")
            
            # Exit after profiling
            return 0
            
        except Exception as e:
            logger.error(f"Error during profiling: {e}")
            return 1
    
    # Main training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Train for one epoch with gradient accumulation
        train_metrics, global_step = train_epoch(
            model=model,
            train_loader=train_loader, 
            optimizer=optimizer, 
            device=device, 
            loss_weights=loss_weights, 
            grad_accum_steps=args.grad_accum_steps,
            memory_tracker=memory_tracker,
            scaler=scaler,
            step_offset=global_step,
            checkpoint_manager=checkpoint_manager,
            curriculum_manager=curriculum_manager
        )
        train_log.append(train_metrics)
        log_metrics(train_metrics, epoch + 1, prefix="train")
        
        # Validate
        if (epoch + 1) % args.eval_every == 0:
            # Skip validation if val_loader is empty
            if len(val_loader) > 0:
                val_metrics = validate(
                    model=model, 
                    val_loader=val_loader, 
                    device=device, 
                    loss_weights=loss_weights,
                    scaler=scaler,
                    memory_tracker=memory_tracker
                )
                val_log.append(val_metrics)
                log_metrics(val_metrics, epoch + 1, prefix="val")
            else:
                # Use dummy validation metrics if no validation data
                logger.info("No validation data available. Using dummy validation metrics.")
                dummy_val_metrics = {
                    "loss": train_metrics["loss"],  # Use training loss
                    "fape_loss": train_metrics["fape_loss"],
                    "confidence_loss": train_metrics["confidence_loss"],
                    "angle_loss": train_metrics["angle_loss"],
                    "rmsd": 10.0,  # Dummy high RMSD
                }
                val_log.append(dummy_val_metrics)
            
            # Update learning rate if using ReduceLROnPlateau
            if args.scheduler == 'plateau' and scheduler is not None:
                if len(val_loader) > 0:
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step(dummy_val_metrics["loss"])
            
            # Run comprehensive validation if enabled and at appropriate frequency
            comp_val_results = None
            if args.comprehensive_val and (epoch + 1) % args.comprehensive_val_freq == 0:
                logger.info(f"Running comprehensive validation at epoch {epoch + 1}...")
                try:
                    # Track current validation metrics (either from val or dummy)
                    current_metrics = val_metrics if len(val_loader) > 0 else dummy_val_metrics
                    
                    # Run comprehensive validation
                    comp_val_results = run_comprehensive_validation(
                        model=model,
                        epoch=epoch + 1,
                        run_dir=run_dir,
                        device=device,
                        args=args,
                        is_final=False
                    )
                    
                    # Log key metrics from comprehensive validation
                    if "test_mode" in comp_val_results and "rmsd" in comp_val_results["test_mode"]:
                        test_rmsd = comp_val_results["test_mode"]["rmsd"]
                        logger.info(f"Comprehensive validation test-mode RMSD: {test_rmsd:.4f}Å")
                        
                        # If test-mode RMSD is better than validation RMSD, consider it for best model
                        if test_rmsd < best_val_rmsd:
                            logger.info(f"New best test-mode RMSD from comprehensive validation: {test_rmsd:.4f}Å")
                            best_val_rmsd = test_rmsd
                            is_best = True
                            patience_counter = 0
                except Exception as e:
                    logger.error(f"Error during comprehensive validation: {e}")
                    # Continue training process even if comprehensive validation fails
                    logger.error("Continuing training despite comprehensive validation error")
            
            # Save checkpoint if best so far (initialize the flag, don't reset it)
            if not 'is_best' in locals():
                is_best = False
            
            # Get the right metrics, either from validation or dummy
            current_metrics = val_metrics if len(val_loader) > 0 else dummy_val_metrics
            
            # Track best TM-score if available
            if "tm_score" in current_metrics:
                # Initialize best_tm_score if first time seeing it
                if not hasattr(train_epoch, "best_tm_score"):
                    train_epoch.best_tm_score = 0.0
                
                # TM-score is better when higher (range 0-1)
                if current_metrics["tm_score"] > train_epoch.best_tm_score:
                    logger.info(f"New best TM-score: {current_metrics['tm_score']:.4f} (previous: {train_epoch.best_tm_score:.4f})")
                    train_epoch.best_tm_score = current_metrics["tm_score"]
                    is_best = True
                    patience_counter = 0
            
            # Also consider loss and RMSD improvements
            if current_metrics["loss"] < best_val_loss:
                best_val_loss = current_metrics["loss"]
                is_best = True
                patience_counter = 0
                logger.info(f"New best validation loss: {best_val_loss:.6f}")
            elif current_metrics["rmsd"] < best_val_rmsd:
                best_val_rmsd = current_metrics["rmsd"]
                is_best = True
                patience_counter = 0
                logger.info(f"New best RMSD: {best_val_rmsd:.4f} Å")
            elif not is_best:  # Only increment patience if no metric improved
                patience_counter += 1
            
            # Save checkpoint with manager if available
            if checkpoint_manager:
                # Prepare metrics dictionary, including comprehensive validation results if available
                metrics_dict = current_metrics.copy()
                
                # Add comprehensive validation results if available
                if comp_val_results:
                    metrics_dict["comprehensive_validation"] = comp_val_results
                
                checkpoint_manager.save_checkpoint(
                    model=model,
                    epoch=epoch,
                    step=global_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metric_value=current_metrics["loss"],
                    metrics=metrics_dict,
                    is_best=is_best,
                    curriculum_state=curriculum_manager.get_state_dict() if curriculum_manager else None,
                    args=vars(args)
                )
            else:
                # Use original checkpoint saving
                checkpoint_dir = os.path.join(run_dir, "checkpoints")
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": current_metrics["loss"],
                    "val_rmsd": current_metrics["rmsd"],
                }
                
                # Add standard validation metrics
                for key, value in current_metrics.items():
                    if key not in checkpoint:
                        checkpoint[key] = value
                
                # Add comprehensive validation results if available
                if comp_val_results:
                    checkpoint["comprehensive_validation"] = comp_val_results
                
                if scheduler is not None:
                    checkpoint["scheduler_state_dict"] = scheduler.state_dict()
                
                if curriculum_manager:
                    checkpoint["curriculum_state"] = curriculum_manager.get_state_dict()
                
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")
                
                # Save best model separately
                if is_best:
                    best_path = os.path.join(checkpoint_dir, "best_model.pt")
                    torch.save(checkpoint, best_path)
                    logger.info(f"Best model saved to {best_path}")
            
            # Update curriculum stage if enabled
            if curriculum_manager:
                try:
                    # Import the safer dataset analyzer
                    from scripts.fix_dataset_analyzer import analyze_rna_dataset_lengths
                    
                    # Get the dummy or actual validation metrics depending on whether validation was run
                    current_metrics = dummy_val_metrics if len(val_loader) == 0 else val_metrics
                    
                    # Update curriculum stage based on validation loss
                    next_stage_sequences = None
                    
                    # If we're not at the final stage, analyze available sequences for next stage
                    if curriculum_manager.current_stage < len(curriculum_manager.sequence_stages) - 1:
                        next_max_len = curriculum_manager.sequence_stages[curriculum_manager.current_stage + 1]
                        
                        # Analyze dataset to get count of available sequences
                        try:
                            seq_stats = analyze_rna_dataset_lengths(train_dataset)
                            if 'sequences_available' in seq_stats and next_max_len in seq_stats['sequences_available']:
                                next_stage_sequences = seq_stats['sequences_available'][next_max_len]
                                logger.info(f"Found {next_stage_sequences} sequences available for next stage (max_len={next_max_len})")
                        except Exception as e:
                            logger.warning(f"Error analyzing dataset for curriculum stage: {e}")
                    
                    # Check if curriculum should advance
                    stage_changed = curriculum_manager.update_stage(
                        epoch=epoch,
                        epoch_loss=current_metrics["loss"],
                        num_sequences_at_next_stage=next_stage_sequences
                    )
                except Exception as e:
                    logger.warning(f"Error updating curriculum stage: {e}")
                    stage_changed = False
                
                # Update dataloaders if stage changed
                if stage_changed:
                    # Update batch size if adaptive batching is enabled
                    if curriculum_manager.batch_adaptive:
                        # Estimate new batch size
                        estimated_batch_size = curriculum_manager.get_estimated_batch_size()
                        
                        # Calculate grad accumulation steps to maintain effective batch size
                        estimated_grad_accum = curriculum_manager.get_estimated_grad_accum(estimated_batch_size)
                        
                        # Update curriculum with new batch parameters
                        curriculum_manager.update_batch_params(estimated_batch_size, estimated_grad_accum)
                        logger.info(f"Updated batch parameters for new curriculum stage: "
                                   f"batch_size={estimated_batch_size}, "
                                   f"grad_accum={estimated_grad_accum}")
                    
                    # Create filtered dataset
                    train_dataset_filtered = curriculum_manager.get_filtered_dataset(train_dataset)
                    
                    # Create new dataloaders with updated parameters
                    train_loader, val_loader = create_dataloaders(
                        args, train_dataset_filtered, val_dataset, curriculum_manager
                    )
                    
                    logger.info(f"Updated dataloaders for curriculum stage {curriculum_manager.current_stage+1}")
        else:
            # Keep validation log in sync with training log
            if val_log:
                val_log.append(val_log[-1])
            else:
                # Initialize with empty validation
                val_log.append({
                    "loss": float('inf'),
                    "fape_loss": float('inf'),
                    "confidence_loss": float('inf'),
                    "angle_loss": float('inf'),
                    "rmsd": float('inf'),
                })
        
        # Update learning rate if using CosineAnnealingLR
        if args.scheduler == 'cosine' and scheduler is not None:
            scheduler.step()
        
        # Check early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Log epoch time
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
        
        # Plot learning curve after each epoch
        plot_metrics(train_log, val_log, run_dir)
        
        # Plot curriculum progress if applicable
        if curriculum_manager and curriculum_manager.stage_history:
            plot_curriculum_progress(train_log, val_log, curriculum_manager, run_dir)
    
    # Save final model
    if checkpoint_manager:
        # Use the last validation metrics
        final_metrics = val_log[-1] if val_log else {"loss": float('inf'), "rmsd": float('inf')}
        
        checkpoint_manager.save_checkpoint(
            model=model,
            epoch=args.epochs - 1,
            step=global_step,
            optimizer=optimizer,
            scheduler=scheduler,
            metric_value=final_metrics["loss"],
            metrics=final_metrics,
            curriculum_state=curriculum_manager.get_state_dict() if curriculum_manager else None,
            args=vars(args)
        )
    else:
        # Use original checkpoint saving
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{args.epochs-1}.pt")
        
        # Use the last validation metrics
        final_metrics = val_log[-1] if val_log else {"loss": float('inf'), "rmsd": float('inf')}
        
        checkpoint = {
            "epoch": args.epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": final_metrics["loss"],
            "val_rmsd": final_metrics["rmsd"],
        }
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Final checkpoint saved to {checkpoint_path}")
    
    # Save training log and plots
    save_training_log(train_log, val_log, run_dir)
    plot_metrics(train_log, val_log, run_dir)
    
    # Save memory usage plot if available
    if memory_tracker:
        memory_tracker.plot_history(os.path.join(run_dir, "gpu_memory_usage.png"))
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}, Best RMSD: {best_val_rmsd:.6f}")
    logger.info(f"All outputs saved to {run_dir}")
    
    # Run final comprehensive validation
    if args.comprehensive_val:
        logger.info("Running final comprehensive validation...")
        try:
            final_validation_results = run_comprehensive_validation(
                model=model,
                epoch=args.epochs,
                run_dir=run_dir,
                device=device,
                args=args,
                is_final=True
            )
            
            # Save final validation results to a special file
            final_val_file = os.path.join(run_dir, "final_validation_results.json")
            with open(final_val_file, 'w') as f:
                json.dump(final_validation_results, f, indent=2)
            
            logger.info(f"Final comprehensive validation completed and saved to {final_val_file}")
            
            # Log final metrics
            if "test_mode" in final_validation_results and "rmsd" in final_validation_results["test_mode"]:
                logger.info(f"Final test-mode RMSD: {final_validation_results['test_mode']['rmsd']:.4f}Å")
                if "tm_score" in final_validation_results["test_mode"]:
                    logger.info(f"Final test-mode TM-score: {final_validation_results['test_mode']['tm_score']:.4f}")
        except Exception as e:
            logger.error(f"Error during final comprehensive validation: {e}")
            logger.exception("Final validation error details:")
    
    # Optionally run full validation on best checkpoint
    if args.validate_checkpoints:
        best_checkpoint_path = os.path.join(run_dir, "checkpoints", "best_model.pt")
        if os.path.exists(best_checkpoint_path):
            logger.info("Running full validation on best checkpoint...")
            
            if args.comprehensive_val:
                # Load best model
                logger.info(f"Loading best model from {best_checkpoint_path}")
                
                try:
                    # Load best checkpoint
                    checkpoint = torch.load(best_checkpoint_path, map_location=device)
                    
                    # Create a temporary model to avoid modifying current model
                    best_model, _ = create_model(args)
                    best_model.to(device)
                    
                    # Load state dict
                    if "model_state_dict" in checkpoint:
                        best_model.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        best_model.load_state_dict(checkpoint)
                    
                    # Run comprehensive validation on best model
                    best_validation_results = run_comprehensive_validation(
                        model=best_model,
                        epoch="best",
                        run_dir=run_dir,
                        device=device,
                        args=args,
                        is_final=True
                    )
                    
                    # Save best checkpoint validation results
                    best_val_file = os.path.join(run_dir, "best_checkpoint_validation_results.json")
                    with open(best_val_file, 'w') as f:
                        json.dump(best_validation_results, f, indent=2)
                    
                    logger.info(f"Best checkpoint validation completed and saved to {best_val_file}")
                except Exception as e:
                    logger.error(f"Error validating best checkpoint: {e}")
            else:
                # Could call a simpler validation function here if comprehensive validation is disabled
                pass
    
    return 0


if __name__ == "__main__":
    sys.exit(main())