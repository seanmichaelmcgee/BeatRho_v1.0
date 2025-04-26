"""
Gradient Checkpointing for Memory Optimization

This module implements gradient checkpointing for transformer and IPA modules
to reduce memory usage during training by trading computation for memory.
"""

import logging
import types
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

logger = logging.getLogger(__name__)

def apply_gradient_checkpointing_to_transformer(model: nn.Module,
                                              use_reentrant: bool = True) -> bool:
    """Apply gradient checkpointing to transformer blocks in the model.
    
    Args:
        model: RNA folding model
        use_reentrant: Whether to use reentrant checkpointing (PyTorch 1.10+)
        
    Returns:
        True if successful, False otherwise
    """
    if not hasattr(model, 'transformer_blocks'):
        logger.warning("Model does not have transformer_blocks attribute")
        return False
    
    # Count transformer blocks that had checkpointing applied
    blocks_modified = 0
    
    # Apply to each transformer block
    for i, block in enumerate(model.transformer_blocks):
        if _checkpoint_transformer_block(block, i, use_reentrant):
            blocks_modified += 1
    
    if blocks_modified > 0:
        logger.info(f"Applied gradient checkpointing to {blocks_modified} transformer blocks")
        return True
    else:
        logger.warning("Could not apply gradient checkpointing to any transformer blocks")
        return False

def _checkpoint_transformer_block(block: nn.Module, block_idx: int, use_reentrant: bool = True) -> bool:
    """Apply gradient checkpointing to a single transformer block.
    
    Args:
        block: Transformer block
        block_idx: Index of block (for logging)
        use_reentrant: Whether to use reentrant checkpointing (PyTorch 1.10+)
        
    Returns:
        True if successful, False otherwise
    """
    # Store original forward function
    if not hasattr(block, 'forward'):
        logger.warning(f"Block {block_idx} has no forward method")
        return False
    
    # Check if already checkpointed
    if hasattr(block, '_original_forward'):
        logger.info(f"Block {block_idx} already has checkpointing applied")
        return True
    
    # Save original forward method
    block._original_forward = block.forward
    
    # Create checkpointed forward method
    def checkpointed_forward(self, residue_repr, pair_repr, mask=None):
        def custom_forward(residue_repr, pair_repr, mask):
            return self._original_forward(residue_repr, pair_repr, mask)
        
        return checkpoint.checkpoint(
            custom_forward, 
            residue_repr, 
            pair_repr, 
            mask, 
            use_reentrant=use_reentrant
        )
    
    # Replace forward method with checkpointed version
    block.forward = types.MethodType(checkpointed_forward, block)
    
    # Flag for bookkeeping
    block._is_checkpointed = True
    
    return True

def apply_checkpointing_to_ipa(model: nn.Module, use_reentrant: bool = True) -> bool:
    """Apply gradient checkpointing to IPA module in the model.
    
    Args:
        model: RNA folding model
        use_reentrant: Whether to use reentrant checkpointing (PyTorch 1.10+)
        
    Returns:
        True if successful, False otherwise
    """
    if not hasattr(model, 'ipa_module'):
        logger.warning("Model does not have ipa_module attribute")
        return False
    
    ipa_module = model.ipa_module
    
    # Check if already checkpointed
    if hasattr(ipa_module, '_original_forward'):
        logger.info("IPA module already has checkpointing applied")
        return True
    
    # Save original forward method
    ipa_module._original_forward = ipa_module.forward
    
    # Create checkpointed forward method
    def checkpointed_forward(self, residue_repr, pair_repr, mask=None):
        def custom_forward(residue_repr, pair_repr, mask):
            return self._original_forward(residue_repr, pair_repr, mask)
        
        return checkpoint.checkpoint(
            custom_forward, 
            residue_repr, 
            pair_repr, 
            mask, 
            use_reentrant=use_reentrant
        )
    
    # Replace forward method with checkpointed version
    ipa_module.forward = types.MethodType(checkpointed_forward, ipa_module)
    
    # Flag for bookkeeping
    ipa_module._is_checkpointed = True
    
    logger.info("Applied gradient checkpointing to IPA module")
    return True

def remove_checkpointing(model: nn.Module) -> int:
    """Remove gradient checkpointing from model components.
    
    Args:
        model: RNA folding model
        
    Returns:
        Number of components restored
    """
    count = 0
    
    # Restore transformer blocks
    if hasattr(model, 'transformer_blocks'):
        for block in model.transformer_blocks:
            if hasattr(block, '_original_forward'):
                block.forward = block._original_forward
                delattr(block, '_original_forward')
                delattr(block, '_is_checkpointed')
                count += 1
    
    # Restore IPA module
    if hasattr(model, 'ipa_module'):
        ipa_module = model.ipa_module
        if hasattr(ipa_module, '_original_forward'):
            ipa_module.forward = ipa_module._original_forward
            delattr(ipa_module, '_original_forward')
            delattr(ipa_module, '_is_checkpointed')
            count += 1
    
    if count > 0:
        logger.info(f"Removed gradient checkpointing from {count} components")
    
    return count

def apply_checkpointing_to_model(model: nn.Module, 
                               transformer_blocks: bool = True, 
                               ipa_module: bool = True,
                               use_reentrant: bool = True) -> Dict[str, bool]:
    """Apply gradient checkpointing to model components.
    
    Args:
        model: RNA folding model
        transformer_blocks: Whether to apply to transformer blocks
        ipa_module: Whether to apply to IPA module
        use_reentrant: Whether to use reentrant checkpointing
        
    Returns:
        Dictionary with success status for each component type
    """
    result = {
        'transformer_blocks': False,
        'ipa_module': False
    }
    
    # Apply to transformer blocks if requested
    if transformer_blocks:
        result['transformer_blocks'] = apply_gradient_checkpointing_to_transformer(
            model, use_reentrant=use_reentrant
        )
    
    # Apply to IPA module if requested
    if ipa_module:
        result['ipa_module'] = apply_checkpointing_to_ipa(
            model, use_reentrant=use_reentrant
        )
    
    return result

def estimate_memory_savings(model: nn.Module) -> Dict[str, Any]:
    """Estimate memory savings from gradient checkpointing.
    
    Args:
        model: RNA folding model
        
    Returns:
        Dictionary with activation memory savings estimates
    """
    activation_savings = 0
    transformer_layers = 0
    
    # Count transformer blocks
    if hasattr(model, 'transformer_blocks'):
        transformer_layers = len(model.transformer_blocks)
        
        # Estimate memory savings (rough heuristic)
        # Assume we save ~60% of activation memory in transformer blocks
        transformer_savings_fraction = 0.6 * transformer_layers
        
        # Scale by model dimensions
        if hasattr(model, 'residue_dim'):
            residue_dim = model.residue_dim
            activation_savings = transformer_savings_fraction * residue_dim
    
    # More complex estimates could account for sequence length, etc.
    
    return {
        'transformer_layers': transformer_layers,
        'has_ipa': hasattr(model, 'ipa_module'),
        'estimated_saving_factor': transformer_savings_fraction if 'transformer_savings_fraction' in locals() else 0,
        'transformer_savings_mb': activation_savings if activation_savings else 0
    }