"""
Curriculum Learning for RNA Structure Prediction

This module implements curriculum learning strategies for RNA structure prediction,
particularly focused on gradually increasing sequence length during training.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)

class CurriculumManager:
    """Manages curriculum learning for RNA 3D structure prediction training."""
    
    def __init__(self, 
                dataset: Dataset,
                initial_max_length: int = 100,
                final_max_length: Optional[int] = None,
                stages: int = 5,
                epochs_per_stage: int = 5,
                length_increase_strategy: str = 'linear',
                plateau_patience: int = 3,
                min_improvement: float = 0.005):
        """
        Initialize curriculum learning manager.
        
        Args:
            dataset: Dataset to filter by sequence length
            initial_max_length: Maximum sequence length for initial stage
            final_max_length: Maximum sequence length for final stage (None for dataset max)
            stages: Number of curriculum stages
            epochs_per_stage: Minimum epochs before advancing to next stage
            length_increase_strategy: Strategy for sequence length increase ('linear', 'exponential')
            plateau_patience: Epochs without improvement to advance stage
            min_improvement: Minimum relative improvement to not trigger plateau
        """
        self.dataset = dataset
        self.initial_max_length = initial_max_length
        self.stages = stages
        self.epochs_per_stage = epochs_per_stage
        self.plateau_patience = plateau_patience
        self.min_improvement = min_improvement
        
        # Initialize tracking variables
        self.current_stage = 0
        self.epochs_at_current_stage = 0
        self.best_loss_at_stage = float('inf')
        self.plateau_counter = 0
        self.stage_history = []
        
        # Calculate max lengths for each stage
        self._compute_stage_lengths(length_increase_strategy, final_max_length)
        
        # Initialize stage sequence counts (for logging)
        self._count_sequences_per_stage()
        
        logger.info(f"Initialized curriculum learning with {stages} stages:")
        for i, max_len in enumerate(self.stage_max_lengths):
            count = self.sequences_per_stage.get(i, 0)
            logger.info(f"  Stage {i}: max length {max_len}, ~{count} sequences")
    
    def _compute_stage_lengths(self, strategy: str, final_max_length: Optional[int]):
        """Compute maximum sequence lengths for each stage."""
        # Get the maximum sequence length in the dataset if not specified
        if final_max_length is None:
            try:
                # Try to get sequence lengths from dataset
                sequence_lengths = []
                for i in range(len(self.dataset)):
                    # Try different attribute names
                    sample = self.dataset[i]
                    if hasattr(sample, 'get'):
                        if 'length' in sample:
                            sequence_lengths.append(sample['length'])
                        elif 'sequence_int' in sample and isinstance(sample['sequence_int'], torch.Tensor):
                            sequence_lengths.append(len(sample['sequence_int']))
                    
                    # Limit to sampling a few examples for efficiency
                    if len(sequence_lengths) >= 100:
                        break
                
                if sequence_lengths:
                    final_max_length = max(sequence_lengths)
                else:
                    # Default if we can't determine
                    final_max_length = 500
                    logger.warning(f"Could not determine max sequence length, using default {final_max_length}")
            except Exception as e:
                logger.warning(f"Error determining dataset max length: {e}. Using default 500.")
                final_max_length = 500
        
        # Set max lengths for each stage
        if strategy == 'linear':
            # Linear increase in sequence length
            self.stage_max_lengths = [
                self.initial_max_length + i * (final_max_length - self.initial_max_length) / max(1, self.stages - 1)
                for i in range(self.stages)
            ]
            # Convert to integers
            self.stage_max_lengths = [int(length) for length in self.stage_max_lengths]
        
        elif strategy == 'exponential':
            # Exponential increase (faster ramp-up)
            growth_factor = (final_max_length / self.initial_max_length) ** (1 / max(1, self.stages - 1))
            self.stage_max_lengths = [
                int(self.initial_max_length * (growth_factor ** i))
                for i in range(self.stages)
            ]
            
        else:
            raise ValueError(f"Unknown length increase strategy: {strategy}")
    
    def _count_sequences_per_stage(self):
        """Count how many sequences are available for each stage."""
        self.sequences_per_stage = {}
        
        try:
            # Sample the dataset to get length distribution
            sequence_lengths = []
            max_samples = min(1000, len(self.dataset))
            
            for i in range(max_samples):
                # Get sample length
                sample = self.dataset[i]
                if hasattr(sample, 'get') and 'length' in sample:
                    sequence_lengths.append(sample['length'])
                elif hasattr(sample, 'get') and 'sequence_int' in sample:
                    sequence_lengths.append(len(sample['sequence_int']))
            
            # Count sequences in each stage
            for stage, max_len in enumerate(self.stage_max_lengths):
                self.sequences_per_stage[stage] = sum(1 for length in sequence_lengths if length <= max_len)
                
            # Scale counts to full dataset size if we sampled
            if max_samples < len(self.dataset):
                scale_factor = len(self.dataset) / max_samples
                for stage in self.sequences_per_stage:
                    self.sequences_per_stage[stage] = int(self.sequences_per_stage[stage] * scale_factor)
                    
        except Exception as e:
            logger.warning(f"Error counting sequences per stage: {e}")
            # Default - just estimate linearly
            for stage in range(self.stages):
                self.sequences_per_stage[stage] = int(len(self.dataset) * (stage + 1) / self.stages)
    
    def get_current_max_length(self) -> int:
        """Get the maximum sequence length for the current stage."""
        return self.stage_max_lengths[self.current_stage]
    
    def get_filtered_dataset(self) -> Dataset:
        """Get dataset filtered to current curriculum stage maximum length."""
        # Get maximum length for current stage
        max_length = self.get_current_max_length()
        
        # Create indices for sequences with length <= max_length
        indices = []
        
        for i in range(len(self.dataset)):
            try:
                # Get sample length
                sample = self.dataset[i]
                length = None
                
                if hasattr(sample, 'get'):
                    if 'length' in sample:
                        length = sample['length']
                    elif 'sequence_int' in sample and isinstance(sample['sequence_int'], torch.Tensor):
                        length = len(sample['sequence_int'])
                
                # Add if within current maximum length
                if length is not None and length <= max_length:
                    indices.append(i)
                    
                # Only check a subset of dataset for large datasets
                if len(indices) >= 5000:
                    logger.info(f"Dataset filtering limited to first 5000 matching sequences")
                    break
                    
            except Exception as e:
                logger.warning(f"Error accessing sample {i}: {e}")
        
        if not indices:
            logger.warning(f"No sequences found with length <= {max_length}, using full dataset")
            indices = list(range(len(self.dataset)))
        
        # Create and return subset dataset
        filtered_dataset = Subset(self.dataset, indices)
        logger.info(f"Stage {self.current_stage}: filtered dataset contains {len(filtered_dataset)} sequences (max length {max_length})")
        
        return filtered_dataset
    
    def update_stage(self, epoch: int, epoch_loss: float, 
                    num_sequences_at_next_stage: Optional[int] = None) -> bool:
        """Update curriculum stage based on epochs and loss.
        
        Args:
            epoch: Current epoch number
            epoch_loss: Validation loss for current epoch
            num_sequences_at_next_stage: Number of sequences that would be available at next stage
                                         (provide to prevent advancing if too few sequences)
        
        Returns:
            True if stage was updated, False otherwise
        """
        # Increment epochs at current stage
        self.epochs_at_current_stage += 1
        
        # Check if we've already reached the final stage
        if self.current_stage >= self.stages - 1:
            logger.info(f"Already at final curriculum stage ({self.current_stage})")
            return False
        
        # Check if we have the minimum epochs at current stage
        minimum_epochs_met = self.epochs_at_current_stage >= self.epochs_per_stage
        
        # Check for loss plateau
        if epoch_loss < self.best_loss_at_stage * (1.0 - self.min_improvement):
            # Loss improved
            self.best_loss_at_stage = epoch_loss
            self.plateau_counter = 0
        else:
            # No significant improvement
            self.plateau_counter += 1
        
        # Determine if plateau criterion is met
        plateau_triggered = self.plateau_counter >= self.plateau_patience
        
        # Check if we should advance to the next stage
        # (minimum epochs met AND either plateau triggered or 2x minimum epochs)
        should_advance = minimum_epochs_met and (plateau_triggered or 
                                               self.epochs_at_current_stage >= 2*self.epochs_per_stage)
        
        # Additional check: ensure we have enough sequences at the next stage
        data_available = True
        if should_advance and num_sequences_at_next_stage is not None:
            if num_sequences_at_next_stage < 50:  # Minimum threshold
                logger.warning(f"Not advancing to stage {self.current_stage + 1} due to insufficient data ({num_sequences_at_next_stage} sequences)")
                data_available = False
        
        # Advance stage if all criteria are met
        if should_advance and data_available:
            # Record stats from previous stage
            self.stage_history.append({
                'stage': self.current_stage,
                'epochs': self.epochs_at_current_stage,
                'max_length': self.get_current_max_length(),
                'final_loss': epoch_loss,
                'best_loss': self.best_loss_at_stage,
                'global_epoch': epoch
            })
            
            # Advance to next stage
            self.current_stage += 1
            self.epochs_at_current_stage = 0
            self.plateau_counter = 0
            self.best_loss_at_stage = float('inf')
            
            # Log the stage advancement
            next_max_len = self.get_current_max_length()
            logger.info(f"Advanced to curriculum stage {self.current_stage}, max sequence length now {next_max_len}")
            
            return True
        
        if should_advance and not data_available:
            logger.info(f"Criteria for advancement to stage {self.current_stage + 1} met, but insufficient data available")
        elif self.epochs_at_current_stage < self.epochs_per_stage:
            logger.debug(f"Need {self.epochs_per_stage - self.epochs_at_current_stage} more epochs at stage {self.current_stage}")
        elif not plateau_triggered:
            logger.debug(f"Loss still improving at stage {self.current_stage}, plateau counter: {self.plateau_counter}/{self.plateau_patience}")
        
        return False
    
    def create_curriculum_dataloader(self, batch_size: int, shuffle: bool = True, 
                                   collate_fn: Optional[Callable] = None,
                                   num_workers: int = 4) -> DataLoader:
        """Create a DataLoader with curriculum-filtered data for the current stage.
        
        Args:
            batch_size: Batch size for the data loader
            shuffle: Whether to shuffle the data
            collate_fn: Function for collating samples into batches
            num_workers: Number of worker processes
            
        Returns:
            DataLoader configured for current curriculum stage
        """
        # Get filtered dataset for current stage
        filtered_dataset = self.get_filtered_dataset()
        
        # Create and return data loader
        return DataLoader(
            filtered_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_stage_summary(self) -> Dict:
        """Get summary information about curriculum stages and progress."""
        return {
            'current_stage': self.current_stage,
            'total_stages': self.stages,
            'stage_max_lengths': self.stage_max_lengths,
            'epochs_at_current_stage': self.epochs_at_current_stage,
            'current_max_length': self.get_current_max_length(),
            'epoch_history': self.stage_history,
            'sequences_per_stage': self.sequences_per_stage
        }

class DynamicBatchSizer:
    """Dynamically adjusts batch size based on sequence length to optimize memory usage."""
    
    def __init__(self, 
                base_batch_size: int = 8,
                base_sequence_length: int = 100,
                scaling_factor: float = 2.0,
                min_batch_size: int = 1,
                max_batch_size: Optional[int] = None):
        """
        Initialize dynamic batch sizer.
        
        Args:
            base_batch_size: Batch size for sequences of base_sequence_length
            base_sequence_length: Reference sequence length for base_batch_size
            scaling_factor: Power to which relative sequence length is raised (quadratic = 2.0)
            min_batch_size: Minimum batch size regardless of sequence length
            max_batch_size: Maximum batch size regardless of sequence length
        """
        self.base_batch_size = base_batch_size
        self.base_sequence_length = base_sequence_length
        self.scaling_factor = scaling_factor
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size if max_batch_size is not None else base_batch_size * 4
        
        # Track batch size history
        self.history = []
    
    def get_batch_size(self, sequence_length: int) -> int:
        """Calculate optimal batch size for given sequence length.
        
        Args:
            sequence_length: Current sequence length
            
        Returns:
            Recommended batch size
        """
        # Calculate memory scaling factor
        length_ratio = self.base_sequence_length / max(1, sequence_length)
        
        # Calculate raw batch size - scales with (base_length/length)^scaling_factor
        # For scaling_factor=2, this gives quadratic memory growth with sequence length
        raw_batch_size = self.base_batch_size * (length_ratio ** self.scaling_factor)
        
        # Apply floor to get integer batch size
        batch_size = max(self.min_batch_size, min(self.max_batch_size, int(raw_batch_size)))
        
        # Record in history
        self.history.append((sequence_length, batch_size))
        
        return batch_size
    
    def get_batch_size_for_loader(self, data_loader: DataLoader) -> int:
        """Calculate optimal batch size for a DataLoader based on its dataset.
        
        Args:
            data_loader: DataLoader to analyze
            
        Returns:
            Recommended batch size or None if couldn't determine
        """
        try:
            # Sample a few sequences to get average length
            lengths = []
            dataset = data_loader.dataset
            
            # Maximum sequences to sample
            max_samples = min(100, len(dataset))
            
            for i in range(max_samples):
                sample = dataset[i]
                if hasattr(sample, 'get') and 'length' in sample:
                    lengths.append(sample['length'])
                elif hasattr(sample, 'get') and 'sequence_int' in sample:
                    lengths.append(len(sample['sequence_int']))
            
            # Calculate average length if we got any
            if lengths:
                avg_length = sum(lengths) / len(lengths)
                max_length = max(lengths)
                
                # Use maximum length for batch size calculation
                # for more conservative memory usage
                return self.get_batch_size(max_length)
            
        except Exception as e:
            logger.warning(f"Error calculating batch size for loader: {e}")
            
        # Return base batch size if we couldn't determine
        return self.base_batch_size