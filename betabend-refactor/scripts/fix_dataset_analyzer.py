#!/usr/bin/env python3
"""
Fix dataset analyzer to correctly handle sequence lengths from RNA datasets.
"""
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
from torch.utils.data import Dataset

# Get the project root
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_rna_dataset_lengths(dataset: Dataset) -> Dict[str, Any]:
    """
    Analyze sequence length distribution in RNA dataset.
    
    Args:
        dataset: PyTorch Dataset (RNADataset or Subset containing RNADataset)
        
    Returns:
        Dictionary with sequence length statistics
    """
    # Extract sequence lengths from each dataset item
    lengths = []
    total_samples = len(dataset)
    max_to_try = min(total_samples, 1000)  # Try at most 1000 samples for performance
    
    # Debugging output
    logger.info(f"Analyzing dataset with {total_samples} samples (up to {max_to_try})")
    
    # Try different ways to get lengths
    attempts = ["length", "sequence_int", "sequence", "__len__"]
    success_method = None
    
    for i in range(min(total_samples, max_to_try)):
        try:
            sample = dataset[i]
            
            # Try known methods to determine length
            length = None
            
            # Method 1: Try getting 'length' directly
            if isinstance(sample, dict) and "length" in sample:
                length = sample["length"]
                if success_method is None:
                    success_method = "length key"
                    logger.info(f"Using 'length' key from dictionary to determine lengths")
                
            # Method 2: Try sequence_int
            elif isinstance(sample, dict) and "sequence_int" in sample and isinstance(sample["sequence_int"], torch.Tensor):
                length = len(sample["sequence_int"])
                if success_method is None:
                    success_method = "sequence_int length"
                    logger.info(f"Using sequence_int tensor length to determine lengths")
                
            # Method 3: Try sequence
            elif isinstance(sample, dict) and "sequence" in sample:
                length = len(sample["sequence"])
                if success_method is None:
                    success_method = "sequence length"
                    logger.info(f"Using sequence string length to determine lengths")
            
            # Method 4: Try __len__ or length attribute
            elif hasattr(sample, "__len__") and callable(getattr(sample, "__len__")):
                length = len(sample)
                if success_method is None:
                    success_method = "sample __len__"
                    logger.info(f"Using sample __len__ to determine lengths")
            
            # Method 5: Last try direct length attribute
            elif hasattr(sample, "length"):
                length = sample.length
                if success_method is None:
                    success_method = "length attribute"
                    logger.info(f"Using length attribute to determine lengths")
            
            # Add the length if found
            if length is not None:
                lengths.append(length)
            
        except Exception as e:
            # Log the error but continue with other samples
            logger.warning(f"Error getting length for sample {i}: {e}")
    
    # If no valid lengths found, return a safe dictionary
    if not lengths:
        logger.warning("No valid lengths found in dataset")
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0,
            "std": 0.0,
            "histogram": {"bins": [50, 100, 150, 200, 250, 300], "counts": [0, 0, 0, 0, 0, 0]},
            "sequences_available": {50: 0, 100: 0, 150: 0, 200: 0, 250: 0, 300: 0}
        }
    
    # Calculate basic statistics
    lengths = np.array(lengths)
    stats = {
        "count": len(lengths),
        "min": int(np.min(lengths)) if lengths.size > 0 else 0,
        "max": int(np.max(lengths)) if lengths.size > 0 else 0,
        "mean": float(np.mean(lengths)) if lengths.size > 0 else 0.0,
        "median": int(np.median(lengths)) if lengths.size > 0 else 0,
        "std": float(np.std(lengths)) if lengths.size > 0 else 0.0,
    }
    
    # Define length bins
    length_bins = [50, 100, 150, 200, 250, 300, 350, 400, 500, 1000]
    
    # Calculate histogram
    hist, _ = np.histogram(lengths, bins=length_bins + [np.inf])
    stats["histogram"] = {
        "bins": length_bins,
        "counts": hist.tolist()
    }
    
    # Calculate sequences available at each length limit
    stats["sequences_available"] = {
        limit: int(np.sum(lengths <= limit)) for limit in length_bins
    }
    
    # Log the results
    logger.info(f"Length analysis complete: found {stats['count']} valid lengths")
    logger.info(f"Min: {stats['min']}, Max: {stats['max']}, Mean: {stats['mean']:.1f}")
    
    return stats

def main():
    logger.info("Fixed RNA dataset analyzer created successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())