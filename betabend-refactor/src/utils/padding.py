import torch
from typing import Tuple, Union


def pad_1d(x: torch.Tensor, max_len: int, pad_value: float = 0) -> torch.Tensor:
    """Pad a 1D tensor to a specified length.
    
    Args:
        x: Input tensor of shape (L,)
        max_len: Target length for padding
        pad_value: Value to use for padding
        
    Returns:
        Padded tensor of shape (max_len,)
    """
    if x.shape[0] == 0:
        return torch.full((max_len,), pad_value, dtype=x.dtype, device=x.device)
        
    if x.shape[0] >= max_len:
        return x[:max_len]
        
    # Create padding
    padding = torch.full((max_len - x.shape[0],), pad_value, dtype=x.dtype, device=x.device)
    
    # Concatenate input with padding
    return torch.cat([x, padding], dim=0)


def pad_2d(x: torch.Tensor, max_len: int, pad_value: float = 0) -> torch.Tensor:
    """Pad a 2D tensor to a specified length in both dimensions.
    
    Args:
        x: Input tensor of shape (L, L) or (L, D)
        max_len: Target length for padding
        pad_value: Value to use for padding
        
    Returns:
        Padded tensor of shape (max_len, max_len) or (max_len, D)
    """
    if x.shape[0] == 0:
        if len(x.shape) > 1 and x.shape[1] != x.shape[0]:
            # For tensors like (L, D)
            return torch.full((max_len, x.shape[1]), pad_value, dtype=x.dtype, device=x.device)
        else:
            # For square tensors (L, L)
            return torch.full((max_len, max_len), pad_value, dtype=x.dtype, device=x.device)
    
    # Handle tensors of shape (L, D) where D is a feature dimension
    if len(x.shape) > 1 and x.shape[1] != x.shape[0]:
        # Truncate if longer than max_len
        if x.shape[0] > max_len:
            return x[:max_len]
            
        # Create result tensor with target shape
        result = torch.full((max_len, x.shape[1]), pad_value, dtype=x.dtype, device=x.device)
        
        # Copy data
        result[:x.shape[0]] = x
        return result
    
    # Handle square tensors of shape (L, L)
    else:
        # Pre-allocate full tensor (memory efficient)
        result = torch.full((max_len, max_len), pad_value, dtype=x.dtype, device=x.device)
        
        # Copy only what's needed
        src_len = min(x.shape[0], max_len)
        result[:src_len, :src_len] = x[:src_len, :src_len]
        
        return result


def pad_tensor(x: torch.Tensor, target_shape: Tuple[int, ...], pad_value: float = 0) -> torch.Tensor:
    """Pad a tensor to a specified shape.
    
    Args:
        x: Input tensor of any shape
        target_shape: Target shape for padding
        pad_value: Value to use for padding
        
    Returns:
        Padded tensor of shape target_shape
    """
    # Check if tensor is already larger than target in any dimension
    for i, (src, tgt) in enumerate(zip(x.shape, target_shape)):
        if src > tgt:
            # Truncate along this dimension
            slices = [slice(None)] * len(x.shape)
            slices[i] = slice(0, tgt)
            x = x[tuple(slices)]
    
    # Calculate padding for each dimension
    padding = []
    for src, tgt in zip(x.shape, target_shape):
        padding.append((0, tgt - src))
    
    # Reverse padding for torch.nn.functional.pad (which takes padding from last to first dim)
    padding = [p for pair in reversed(padding) for p in pair]
    
    # Pad tensor
    return torch.nn.functional.pad(x, padding, mode='constant', value=pad_value)