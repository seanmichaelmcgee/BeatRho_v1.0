"""
Enhanced Checkpoint Management for RNA Structure Prediction

This module provides comprehensive checkpoint management including:
- Regular interval checkpoints
- Best model tracking
- Emergency checkpoints (e.g., when memory pressure is high)
- Crash recovery
"""

import os
import json
import time
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages model checkpoints during training with advanced features."""
    
    def __init__(self,
                checkpoint_dir: str,
                keep_best: bool = True,
                keep_last: bool = True,
                keep_interval: Optional[int] = 10,
                max_checkpoints: int = 5,
                backup_dir: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            keep_best: Whether to keep best performing checkpoint
            keep_last: Whether to keep last checkpoint
            keep_interval: Save checkpoints at this epoch interval (None to disable)
            max_checkpoints: Maximum number of non-best checkpoints to keep
            backup_dir: Optional directory for checkpoint backups
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_best = keep_best
        self.keep_last = keep_last
        self.keep_interval = keep_interval
        self.max_checkpoints = max_checkpoints
        self.backup_dir = Path(backup_dir) if backup_dir else None
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        self.checkpoint_history = []
        self.best_metric = float('inf')
        self.best_checkpoint_path = None
        self.last_checkpoint_path = None
        self.emergency_checkpoint_path = None
        
        # Load existing checkpoint info if available
        self._load_checkpoint_info()
        
        logger.info(f"Checkpoint manager initialized with directory: {checkpoint_dir}")
        if self.backup_dir:
            logger.info(f"Backup directory: {backup_dir}")
    
    def _load_checkpoint_info(self):
        """Load existing checkpoint information from checkpoint directory."""
        info_path = self.checkpoint_dir / "checkpoint_info.json"
        
        if info_path.exists():
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                # Load checkpoint history
                if 'checkpoint_history' in info:
                    self.checkpoint_history = info['checkpoint_history']
                
                # Load best metric and best checkpoint path
                if 'best_metric' in info:
                    self.best_metric = info['best_metric']
                
                if 'best_checkpoint_path' in info and os.path.exists(info['best_checkpoint_path']):
                    self.best_checkpoint_path = info['best_checkpoint_path']
                
                if 'last_checkpoint_path' in info and os.path.exists(info['last_checkpoint_path']):
                    self.last_checkpoint_path = info['last_checkpoint_path']
                
                logger.info(f"Loaded checkpoint info from {info_path}")
                if self.best_checkpoint_path:
                    logger.info(f"Best checkpoint: {self.best_checkpoint_path} with metric {self.best_metric}")
                if self.last_checkpoint_path:
                    logger.info(f"Last checkpoint: {self.last_checkpoint_path}")
                    
            except Exception as e:
                logger.warning(f"Error loading checkpoint info: {e}")
    
    def _save_checkpoint_info(self):
        """Save checkpoint information to checkpoint directory."""
        info_path = self.checkpoint_dir / "checkpoint_info.json"
        
        info = {
            'checkpoint_history': self.checkpoint_history,
            'best_metric': self.best_metric,
            'best_checkpoint_path': self.best_checkpoint_path,
            'last_checkpoint_path': self.last_checkpoint_path,
            'emergency_checkpoint_path': self.emergency_checkpoint_path,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving checkpoint info: {e}")
    
    def should_checkpoint_at_epoch(self, epoch: int) -> bool:
        """Determine if a checkpoint should be saved at this epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            True if checkpoint should be saved
        """
        # Always checkpoint at first epoch
        if epoch == 0:
            return True
        
        # Always checkpoint at specified intervals
        if self.keep_interval is not None and epoch % self.keep_interval == 0:
            return True
        
        return False
    
    def save_checkpoint(self, model: torch.nn.Module, epoch: int, step: Optional[int] = None,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       metric: Optional[float] = None,
                       metric_name: str = "val_loss",
                       is_lower_better: bool = True,
                       emergency: bool = False,
                       is_best: Optional[bool] = None,
                       additional_state: Optional[Dict[str, Any]] = None) -> str:
        """Save checkpoint with complete training state.
        
        Args:
            model: PyTorch model to save
            epoch: Current epoch number
            step: Optional step within epoch
            optimizer: Optional optimizer state to save
            scheduler: Optional scheduler state to save
            metric: Optional validation metric
            metric_name: Name of validation metric
            is_lower_better: Whether lower metric is better
            emergency: Whether this is an emergency checkpoint
            is_best: Manually specify if this is the best model
            additional_state: Additional data to save in the checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint filename
        if emergency:
            checkpoint_filename = f"emergency_epoch_{epoch}.pt"
        elif is_best:
            checkpoint_filename = "best_model.pt"
        else:
            checkpoint_filename = f"checkpoint_epoch_{epoch}.pt"
            
        checkpoint_path = str(self.checkpoint_dir / checkpoint_filename)
        
        # Create checkpoint state
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_blocks': model.num_blocks if hasattr(model, 'num_blocks') else None,
                'residue_embed_dim': model.residue_dim if hasattr(model, 'residue_dim') else None,
                'pair_embed_dim': model.pair_dim if hasattr(model, 'pair_dim') else None,
            }
        }
        
        # Add optimizer state
        if optimizer is not None:
            checkpoint_state['optimizer_state_dict'] = optimizer.state_dict()
        
        # Add scheduler state
        if scheduler is not None:
            checkpoint_state['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add metric
        if metric is not None:
            checkpoint_state[metric_name] = metric
        
        # Add step if provided
        if step is not None:
            checkpoint_state['step'] = step
        
        # Add timestamp
        checkpoint_state['timestamp'] = datetime.now().isoformat()
        
        # Add emergency flag
        if emergency:
            checkpoint_state['emergency'] = True
        
        # Add additional state
        if additional_state is not None:
            for key, value in additional_state.items():
                checkpoint_state[key] = value
        
        # Save checkpoint
        try:
            torch.save(checkpoint_state, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Save backup if enabled
            if self.backup_dir:
                backup_path = str(self.backup_dir / checkpoint_filename)
                shutil.copy2(checkpoint_path, backup_path)
                logger.info(f"Saved backup to {backup_path}")
            
            # Update checkpoint history
            checkpoint_info = {
                'path': checkpoint_path,
                'epoch': epoch,
                'step': step,
                'metric': metric,
                'metric_name': metric_name,
                'timestamp': datetime.now().isoformat(),
                'emergency': emergency,
                'is_best': bool(is_best) if is_best is not None else False
            }
            
            self.checkpoint_history.append(checkpoint_info)
            
            # Update best checkpoint if applicable
            if is_best or (metric is not None and 
                         ((is_lower_better and metric < self.best_metric) or
                          (not is_lower_better and metric > self.best_metric))):
                
                self.best_metric = metric if metric is not None else self.best_metric
                
                # Create best checkpoint if not already saved as best
                if not is_best:
                    best_path = str(self.checkpoint_dir / "best_model.pt")
                    shutil.copy2(checkpoint_path, best_path)
                    logger.info(f"Copied {checkpoint_path} to best model at {best_path}")
                    
                    # Also copy to backup if enabled
                    if self.backup_dir:
                        backup_best_path = str(self.backup_dir / "best_model.pt")
                        shutil.copy2(checkpoint_path, backup_best_path)
                
                self.best_checkpoint_path = str(self.checkpoint_dir / "best_model.pt")
                logger.info(f"Updated best checkpoint with metric {self.best_metric}")
            
            # Update last checkpoint
            self.last_checkpoint_path = checkpoint_path
            
            # Update emergency checkpoint if applicable
            if emergency:
                self.emergency_checkpoint_path = checkpoint_path
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            # Save checkpoint info
            self._save_checkpoint_info()
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            
            # Try saving with minimal state in case of out of memory
            if not emergency:
                try:
                    # Create minimal checkpoint with just a dummy state
                    minimal_state = {
                        'epoch': epoch,
                        'model_state_dict': {'dummy': torch.tensor([0.0])},
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'minimal_emergency': True
                    }
                    
                    minimal_path = str(self.checkpoint_dir / f"minimal_emergency_epoch_{epoch}.pt")
                    torch.save(minimal_state, minimal_path)
                    logger.warning(f"Saved minimal emergency checkpoint to {minimal_path}")
                    return minimal_path
                    
                except Exception as e2:
                    logger.error(f"Also failed to save minimal checkpoint: {e2}")
            
            return ""
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints according to keep policy."""
        # Don't remove best or last if we're keeping them
        exclude_paths = []
        if self.keep_best and self.best_checkpoint_path:
            exclude_paths.append(self.best_checkpoint_path)
        
        if self.keep_last and self.last_checkpoint_path:
            exclude_paths.append(self.last_checkpoint_path)
        
        # Only consider regular checkpoints (not best/emergency)
        regular_checkpoints = [
            info for info in self.checkpoint_history 
            if not info['is_best'] and not info['emergency'] and 
            info['path'] not in exclude_paths and
            Path(info['path']).exists()
        ]
        
        # Sort by epoch
        regular_checkpoints.sort(key=lambda x: x['epoch'])
        
        # Keep interval checkpoints if specified
        if self.keep_interval is not None:
            keep_epochs = set(
                info['epoch'] for info in regular_checkpoints
                if info['epoch'] % self.keep_interval == 0
            )
            
            # Filter to interval checkpoints and non-interval
            interval_checkpoints = [
                info for info in regular_checkpoints
                if info['epoch'] in keep_epochs
            ]
            
            non_interval_checkpoints = [
                info for info in regular_checkpoints
                if info['epoch'] not in keep_epochs
            ]
            
            # Only remove non-interval checkpoints if needed
            if len(interval_checkpoints) > self.max_checkpoints:
                # Only keep the most recent max_checkpoints
                to_remove = interval_checkpoints[:-self.max_checkpoints]
                for info in to_remove:
                    try:
                        os.remove(info['path'])
                        logger.debug(f"Removed old interval checkpoint: {info['path']}")
                    except Exception as e:
                        logger.warning(f"Error removing checkpoint {info['path']}: {e}")
            
            # Remove all non-interval checkpoints
            for info in non_interval_checkpoints:
                try:
                    os.remove(info['path'])
                    logger.debug(f"Removed non-interval checkpoint: {info['path']}")
                except Exception as e:
                    logger.warning(f"Error removing checkpoint {info['path']}: {e}")
        else:
            # Keep only max_checkpoints
            if len(regular_checkpoints) > self.max_checkpoints:
                to_remove = regular_checkpoints[:-self.max_checkpoints]
                for info in to_remove:
                    try:
                        os.remove(info['path'])
                        logger.debug(f"Removed old checkpoint: {info['path']}")
                    except Exception as e:
                        logger.warning(f"Error removing checkpoint {info['path']}: {e}")
    
    def load_checkpoint(self, model: torch.nn.Module, 
                      optimizer: Optional[torch.optim.Optimizer] = None,
                      scheduler: Optional[Any] = None,
                      checkpoint_path: Optional[str] = None,
                      load_best: bool = False,
                      strict: bool = True,
                      device: Optional[torch.device] = None) -> Tuple[int, Optional[int], Optional[float]]:
        """Load model and training state from checkpoint.
        
        Args:
            model: PyTorch model to load weights into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            checkpoint_path: Path to specific checkpoint to load
            load_best: Whether to load best checkpoint
            strict: Whether to strictly enforce that the keys in state_dict match
            device: Device to load tensors onto
            
        Returns:
            Tuple of (epoch, step, metric)
        """
        # Determine which checkpoint to load
        if checkpoint_path is not None:
            load_path = checkpoint_path
        elif load_best and self.best_checkpoint_path:
            load_path = self.best_checkpoint_path
        elif self.last_checkpoint_path:
            load_path = self.last_checkpoint_path
        else:
            logger.warning("No checkpoint found to load")
            return 0, None, None
        
        # Check if checkpoint exists
        if not os.path.exists(load_path):
            logger.warning(f"Checkpoint {load_path} does not exist")
            return 0, None, None
        
        try:
            # Load checkpoint
            logger.info(f"Loading checkpoint from {load_path}")
            
            # Handle device mapping
            map_location = device if device is not None else torch.device('cpu')
            checkpoint = torch.load(load_path, map_location=map_location)
            
            # Check if this is a minimal emergency checkpoint
            if 'minimal_emergency' in checkpoint and checkpoint['minimal_emergency']:
                logger.warning(f"This is a minimal emergency checkpoint from a previous error: {checkpoint.get('error', 'unknown error')}")
                logger.warning(f"Only epoch information will be loaded, not model weights")
                return checkpoint.get('epoch', 0), checkpoint.get('step', None), None
            
            # Load model
            try:
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
                    logger.info("Loaded model state")
                else:
                    logger.warning("Checkpoint does not contain model state")
            except Exception as e:
                logger.error(f"Error loading model state: {e}")
                if strict:
                    raise
            
            # Load optimizer
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Loaded optimizer state")
                except Exception as e:
                    logger.warning(f"Error loading optimizer state: {e}")
            
            # Load scheduler
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    logger.info("Loaded scheduler state")
                except Exception as e:
                    logger.warning(f"Error loading scheduler state: {e}")
            
            # Get epoch and step
            epoch = checkpoint.get('epoch', 0)
            step = checkpoint.get('step', None)
            
            # Get metric
            metric = None
            metric_name = None
            
            # Try common metric names
            metric_names = ['val_loss', 'val_rmsd', 'validation_loss', 'validation_rmsd']
            for name in metric_names:
                if name in checkpoint:
                    metric = checkpoint[name]
                    metric_name = name
                    break
            
            logger.info(f"Loaded checkpoint from epoch {epoch}" + 
                      (f", step {step}" if step is not None else "") +
                      (f", {metric_name}={metric:.6f}" if metric is not None else ""))
            
            return epoch, step, metric
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            
            return 0, None, None
    
    def list_available_checkpoints(self) -> Dict[str, List[Dict[str, Any]]]:
        """List available checkpoints grouped by type.
        
        Returns:
            Dictionary with checkpoint types as keys and lists of checkpoint info as values
        """
        checkpoints = {
            'best': [],
            'last': [],
            'regular': [],
            'emergency': []
        }
        
        # Check all checkpoints
        for info in self.checkpoint_history:
            # Skip if file doesn't exist
            if not os.path.exists(info['path']):
                continue
                
            # Categorize checkpoint
            if info.get('is_best', False):
                checkpoints['best'].append(info)
            elif info.get('emergency', False):
                checkpoints['emergency'].append(info)
            else:
                checkpoints['regular'].append(info)
                
            # Mark as last if it's the last checkpoint
            if info['path'] == self.last_checkpoint_path:
                checkpoints['last'].append(info)
        
        return checkpoints
    
    def find_checkpoint_by_epoch(self, epoch: int) -> Optional[str]:
        """Find checkpoint path for specific epoch.
        
        Args:
            epoch: Epoch number to find
            
        Returns:
            Path to checkpoint or None if not found
        """
        for info in self.checkpoint_history:
            if info['epoch'] == epoch and os.path.exists(info['path']):
                return info['path']
        
        return None
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint by epoch.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints found
        """
        if not self.checkpoint_history:
            return None
            
        # Sort by epoch and step
        sorted_checkpoints = sorted(
            [info for info in self.checkpoint_history if os.path.exists(info['path'])],
            key=lambda x: (x['epoch'], x.get('step', 0)),
            reverse=True
        )
        
        if sorted_checkpoints:
            return sorted_checkpoints[0]['path']
            
        return None
    
    def find_best_checkpoint(self) -> Optional[str]:
        """Find the best checkpoint based on metrics.
        
        Returns:
            Path to best checkpoint or None if no checkpoints found
        """
        return self.best_checkpoint_path if self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path) else None