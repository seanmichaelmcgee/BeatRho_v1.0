"""
Memory Tracking and Optimization Utilities

This module provides tools for tracking and optimizing GPU memory usage
during training, particularly helpful for handling long RNA sequences.
"""

import os
import gc
import time
import logging
from typing import Dict, Optional, Tuple, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)

class MemoryTracker:
    """Class for tracking GPU memory usage during training."""
    
    def __init__(self, 
                device: torch.device, 
                log_interval: int = 100,
                memory_fraction_warn: float = 0.85,
                memory_fraction_critical: float = 0.92,
                track_history: bool = True,
                history_size: int = 1000):
        """
        Initialize memory tracker.
        
        Args:
            device: PyTorch device to track
            log_interval: How often to log memory stats (in updates)
            memory_fraction_warn: Fraction of memory that triggers warnings
            memory_fraction_critical: Fraction that triggers emergency checkpointing
            track_history: Whether to maintain a history of memory usage
            history_size: Maximum number of history entries to keep
        """
        self.device = device
        self.log_interval = log_interval
        self.memory_fraction_warn = memory_fraction_warn
        self.memory_fraction_critical = memory_fraction_critical
        self.update_count = 0
        self.track_history = track_history
        
        # Initialize memory history tracking
        if track_history:
            self.history_size = history_size
            self.history = {
                'allocated': [],
                'reserved': [],
                'free': [],
                'total': [],
                'fraction': [],
                'timestamp': [],
                'step_name': []
            }
        
        # Set CUDA memory allocator for better fragmentation handling
        if torch.cuda.is_available():
            # Configure PyTorch memory allocator for better handling
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        if not torch.cuda.is_available():
            return {
                'allocated': 0.0,
                'reserved': 0.0,
                'free': 0.0,
                'total': 0.0,
                'fraction': 0.0
            }
            
        # Get memory stats in MB
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        
        # Get free and total memory if available
        if hasattr(torch.cuda, 'mem_get_info'):
            free, total = torch.cuda.mem_get_info(self.device)
            free = free / (1024 ** 2)
            total = total / (1024 ** 2)
        else:
            # Fallback for older PyTorch versions
            free = 0
            total = reserved
            
        memory_fraction = 1.0 - (free / total) if total > 0 else 0.0
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total,
            'fraction': memory_fraction
        }
    
    def update(self, step_name: str = ""):
        """Update memory tracking stats and log if interval is reached."""
        stats = self.get_memory_stats()
        self.update_count += 1
        
        # Store history if enabled
        if self.track_history:
            self.history['allocated'].append(stats['allocated'])
            self.history['reserved'].append(stats['reserved'])
            self.history['free'].append(stats['free'])
            self.history['total'].append(stats['total'])
            self.history['fraction'].append(stats['fraction'])
            self.history['timestamp'].append(time.time())
            self.history['step_name'].append(step_name)
            
            # Trim history if needed
            if len(self.history['allocated']) > self.history_size:
                for key in self.history:
                    self.history[key] = self.history[key][-self.history_size:]
        
        # Log memory stats periodically
        if self.update_count % self.log_interval == 0:
            mem_frac = stats['fraction']
            log_msg = (f"Memory usage {step_name}: {stats['allocated']:.1f}MB allocated, "
                      f"{stats['reserved']:.1f}MB reserved, {mem_frac:.1%} used")
            
            # Adjust log level based on memory pressure
            if mem_frac > self.memory_fraction_critical:
                logger.critical(log_msg)
            elif mem_frac > self.memory_fraction_warn:
                logger.warning(log_msg)
            else:
                logger.info(log_msg)
        
        return stats
    
    def should_checkpoint(self) -> bool:
        """Determine if memory pressure is high enough to trigger an emergency checkpoint."""
        stats = self.get_memory_stats()
        return stats['fraction'] > self.memory_fraction_critical
    
    def cleanup(self) -> Dict[str, float]:
        """Force garbage collection and CUDA cache clearing.
        
        Returns:
            Memory stats after cleanup
        """
        # Run Python garbage collection
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Return memory stats after cleanup
        return self.get_memory_stats()
    
    def get_history_dataframe(self):
        """Get memory history as a pandas DataFrame."""
        if not self.track_history:
            raise ValueError("History tracking is disabled")
            
        try:
            import pandas as pd
            df = pd.DataFrame(self.history)
            
            # Add relative timestamps
            if len(df) > 0:
                df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
            
            return df
        except ImportError:
            logger.warning("pandas not available, returning history dict")
            return self.history
    
    def plot_memory_usage(self, save_path: Optional[str] = None):
        """Plot memory usage history and optionally save to file."""
        if not self.track_history:
            raise ValueError("History tracking is disabled")
        
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            df = self.get_history_dataframe()
            if len(df) == 0:
                return
                
            # Create plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot allocated and reserved memory
            ax1.plot(df['relative_time'], df['allocated'], 'b-', label='Allocated (MB)')
            ax1.plot(df['relative_time'], df['reserved'], 'g--', label='Reserved (MB)')
            
            # Add labels and legend
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Memory (MB)')
            ax1.tick_params(axis='y')
            
            # Create second y-axis for memory fraction
            ax2 = ax1.twinx()
            ax2.plot(df['relative_time'], df['fraction']*100, 'r-.', label='Usage (%)')
            ax2.set_ylabel('Usage (%)')
            ax2.tick_params(axis='y')
            
            # Add warning and critical thresholds
            ax2.axhline(self.memory_fraction_warn*100, color='orange', linestyle=':', 
                        alpha=0.5, label=f'Warning ({self.memory_fraction_warn*100:.0f}%)')
            ax2.axhline(self.memory_fraction_critical*100, color='red', linestyle=':', 
                        alpha=0.5, label=f'Critical ({self.memory_fraction_critical*100:.0f}%)')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Add title
            plt.title('GPU Memory Usage During Training')
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=100)
                logger.info(f"Memory usage plot saved to {save_path}")
            
            return fig
            
        except ImportError:
            logger.warning("matplotlib or pandas not available, cannot plot")

def calculate_optimal_batch_size(sequence_length: int, 
                               model_params: int, 
                               model_activations_factor: float = 20.0,
                               target_memory_usage: float = 0.8) -> int:
    """Calculate optimal batch size based on sequence length and model size.
    
    Args:
        sequence_length: Length of RNA sequence
        model_params: Number of parameters in the model
        model_activations_factor: Factor for estimating per-sequence activation memory
        target_memory_usage: Target fraction of GPU memory to use
        
    Returns:
        Recommended batch size
    """
    if not torch.cuda.is_available():
        # Default for CPU
        return 8
    
    # Get available GPU memory in bytes
    free, total = torch.cuda.mem_get_info()
    
    # Model parameters memory (32-bit floats)
    model_param_bytes = model_params * 4
    
    # Optimizer states (Adam has 2 states per parameter)
    optim_bytes = model_params * 4 * 2
    
    # Memory for activations scales with sequence length and batch size
    bytes_per_seq = sequence_length**2 * model_activations_factor
    
    # Available memory for batch data
    available_batch_memory = total * target_memory_usage - model_param_bytes - optim_bytes
    
    # Recommended batch size
    batch_size = int(available_batch_memory / bytes_per_seq)
    
    # Ensure at least 1
    batch_size = max(1, batch_size)
    
    return batch_size

def setup_memory_optimizations(mixed_precision: bool = True,
                             set_cuda_allocator: bool = True) -> None:
    """Setup memory optimizations for PyTorch training.
    
    Args:
        mixed_precision: Whether to use mixed precision settings
        set_cuda_allocator: Whether to set CUDA memory allocator configuration
    """
    if set_cuda_allocator and torch.cuda.is_available():
        # Configure PyTorch memory allocator for better fragmentation handling
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        logger.info("Set CUDA memory allocator configuration")
    
    # For mixed precision, set PyTorch autocast settings
    if mixed_precision:
        try:
            # This works on PyTorch 1.10+
            torch.set_float32_matmul_precision('high')
            logger.info("Set float32 matmul precision to 'high'")
        except AttributeError:
            logger.info("Could not set float32 matmul precision (requires PyTorch 1.10+)")

def apply_gradient_checkpointing(model: torch.nn.Module, enable: bool = True) -> bool:
    """Apply gradient checkpointing to transformer blocks if the model supports it.
    
    Args:
        model: PyTorch model
        enable: Whether to enable gradient checkpointing
        
    Returns:
        True if gradient checkpointing was successfully applied
    """
    # Try to apply to transformer blocks
    if hasattr(model, 'transformer_blocks'):
        if enable:
            for block in model.transformer_blocks:
                block.use_checkpointing = True
            logger.info(f"Enabled gradient checkpointing on {len(model.transformer_blocks)} transformer blocks")
        else:
            for block in model.transformer_blocks:
                block.use_checkpointing = False
            logger.info(f"Disabled gradient checkpointing on {len(model.transformer_blocks)} transformer blocks")
        return True
    
    # No compatible model components found
    logger.warning("Model does not have compatible components for gradient checkpointing")
    return False