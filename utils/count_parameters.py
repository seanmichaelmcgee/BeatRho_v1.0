"""
Utility functions for counting model parameters and analyzing model architecture.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional

def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate the model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    # Get model state dict
    state_dict = model.state_dict()
    
    # Calculate size in bytes
    param_size = 0
    for key, param in state_dict.items():
        param_size += param.nelement() * param.element_size()
    
    # Convert to megabytes
    return param_size / (1024 ** 2)

def get_parameter_counts_by_module(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters by module.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping module names to parameter counts
    """
    counts = {}
    
    for name, module in model.named_modules():
        if name == "":
            continue  # Skip the model itself
        
        # Count parameters in this module (excluding sub-modules)
        params = 0
        for param_name, param in module.named_parameters(recurse=False):
            if param.requires_grad:
                params += param.numel()
        
        counts[name] = params
    
    return counts

def print_model_summary(model: nn.Module) -> None:
    """
    Print a summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model
    """
    # Get parameter counts
    total_params = count_parameters(model)
    module_counts = get_parameter_counts_by_module(model)
    
    # Sort modules by parameter count
    sorted_modules = sorted(module_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Print summary
    print(f"Model Summary:")
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"Model Size: {get_model_size_mb(model):.2f} MB")
    
    print("\nParameters by Module:")
    for name, count in sorted_modules:
        if count > 0:
            print(f"  {name}: {count:,} ({count / total_params * 100:.1f}%)")
    
    print("\nModule Hierarchy:")
    for name, module in model.named_modules():
        if name == "":
            continue
        indent = "  " * name.count(".")
        print(f"{indent}{name.split('.')[-1]}: {module.__class__.__name__}")
