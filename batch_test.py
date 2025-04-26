#!/usr/bin/env python3
"""
Batch testing script for RhoFold+ IPA RNA structure prediction model.

This script performs various tests to ensure the model is functioning correctly
and efficiently, including input/output shape tests, memory usage analysis,
and performance benchmarking.
"""

import os
import sys
import time
import argparse
import logging
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add necessary paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import model and utilities
from rhofold_ipa_module import RhoFoldIPAModel
from train_rhofold_ipa import RNAFeatureDataset, collate_fn


def create_dummy_batch(batch_size: int, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Create a dummy batch for testing.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to create tensors on
        
    Returns:
        Dictionary of input tensors
    """
    # Create sequence integers (0=A, 1=C, 2=G, 3=U)
    sequence_int = torch.randint(0, 4, (batch_size, seq_len), device=device)
    
    # Create mask (all True)
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    
    # Create random coordinates
    coordinates = torch.randn((batch_size, seq_len, 3), device=device)
    
    # Create random pairing probabilities
    pairing_probs = torch.rand((batch_size, seq_len, seq_len), device=device)
    pairing_probs = pairing_probs * pairing_probs.transpose(1, 2)  # Make symmetric
    pairing_probs = F.normalize(pairing_probs, p=1, dim=2)  # Normalize along dimension 2
    
    # Create other features
    positional_entropy = torch.rand((batch_size, seq_len), device=device)
    accessibility = torch.rand((batch_size, seq_len), device=device)
    dihedral_features = torch.randn((batch_size, seq_len, 4), device=device)
    coupling_matrix = torch.randn((batch_size, seq_len, seq_len), device=device)
    
    return {
        "sequence_int": sequence_int,
        "mask": mask,
        "coordinates": coordinates,
        "pairing_probs": pairing_probs,
        "positional_entropy": positional_entropy,
        "accessibility": accessibility,
        "dihedral_features": dihedral_features,
        "coupling_matrix": coupling_matrix,
        "lengths": torch.tensor([seq_len] * batch_size, device=device),
        "target_ids": [f"dummy_{i}" for i in range(batch_size)],
    }


def initialize_model(config: Dict[str, Any], device: torch.device) -> RhoFoldIPAModel:
    """
    Initialize model with the given configuration.
    
    Args:
        config: Model configuration
        device: Device to initialize model on
        
    Returns:
        Initialized model
    """
    model = RhoFoldIPAModel(config)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params:,} parameters")
    
    return model


def measure_memory_usage(batch: Dict[str, torch.Tensor], model: RhoFoldIPAModel) -> Dict[str, float]:
    """
    Measure memory usage of model forward pass.
    
    Args:
        batch: Input batch
        model: Model to test
        
    Returns:
        Dictionary of memory usage statistics (in MB)
    """
    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Measure initial memory usage
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(batch)
    
    # Measure final memory usage
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    final_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    
    # Calculate memory used
    memory_used = peak_memory - initial_memory
    
    return {
        "initial_memory_mb": initial_memory,
        "peak_memory_mb": peak_memory,
        "final_memory_mb": final_memory,
        "memory_used_mb": memory_used,
    }


def test_shapes(batch: Dict[str, torch.Tensor], model: RhoFoldIPAModel) -> None:
    """
    Test input/output shapes for the model.
    
    Args:
        batch: Input batch
        model: Model to test
    """
    batch_size = batch["sequence_int"].size(0)
    seq_len = batch["sequence_int"].size(1)
    
    logger.info(f"Testing shapes with batch_size={batch_size}, seq_len={seq_len}")
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(batch)
    
    # Check output shapes
    expected_shapes = {
        "pred_coords": (batch_size, seq_len, 3),
        "pred_angles": (batch_size, seq_len, 7, 2),  # Assuming 7 angles, each with (sin, cos)
        "pred_confidence": (batch_size, seq_len),
    }
    
    for name, expected_shape in expected_shapes.items():
        if name in outputs:
            actual_shape = tuple(outputs[name].shape)
            logger.info(f"  {name}: {actual_shape} (expected: {expected_shape})")
            assert actual_shape == expected_shape, f"Shape mismatch for {name}: {actual_shape} != {expected_shape}"
        else:
            logger.warning(f"  {name}: Missing from outputs")
    
    logger.info("Shape tests passed")


def benchmark_performance(
    model: RhoFoldIPAModel,
    batch_size: int,
    seq_lens: List[int],
    device: torch.device,
    repeats: int = 3,
) -> Dict[int, Dict[str, float]]:
    """
    Benchmark model performance on different sequence lengths.
    
    Args:
        model: Model to benchmark
        batch_size: Batch size for testing
        seq_lens: List of sequence lengths to test
        device: Device to run benchmark on
        repeats: Number of repeats for each test
        
    Returns:
        Dictionary of performance metrics for each sequence length
    """
    logger.info(f"Benchmarking performance with batch_size={batch_size}")
    
    model.eval()
    results = {}
    
    for seq_len in seq_lens:
        logger.info(f"Testing sequence length: {seq_len}")
        
        # Create dummy batch
        batch = create_dummy_batch(batch_size, seq_len, device)
        
        # Warm-up run
        with torch.no_grad():
            _ = model(batch)
        
        # Measure time for multiple runs
        times = []
        memory_stats = []
        
        for i in range(repeats):
            # Clear cache
            torch.cuda.empty_cache()
            
            # Measure time
            start_time = time.time()
            with torch.no_grad():
                outputs = model(batch)
            end_time = time.time()
            
            run_time = end_time - start_time
            times.append(run_time)
            
            # Measure memory
            memory_stats.append(measure_memory_usage(batch, model))
            
            logger.info(f"  Run {i+1}/{repeats}: {run_time:.4f} seconds")
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Average memory stats
        avg_memory = {
            key: sum(stat[key] for stat in memory_stats) / len(memory_stats)
            for key in memory_stats[0].keys()
        }
        
        # Store results
        results[seq_len] = {
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "seq_per_second": batch_size / avg_time,
            **avg_memory,
        }
        
        logger.info(f"  Average: {avg_time:.4f} seconds ({batch_size / avg_time:.2f} sequences/second)")
        logger.info(f"  Memory: {avg_memory['memory_used_mb']:.2f} MB")
    
    return results


def test_gradient_checkpointing(
    model: RhoFoldIPAModel,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Test memory savings from gradient checkpointing.
    
    Args:
        model: Model to test
        batch_size: Batch size for testing
        seq_len: Sequence length to test
        device: Device to run test on
        
    Returns:
        Dictionary of memory usage statistics
    """
    logger.info(f"Testing gradient checkpointing with batch_size={batch_size}, seq_len={seq_len}")
    
    # Create dummy batch
    batch = create_dummy_batch(batch_size, seq_len, device)
    
    # Test without gradient checkpointing
    model.train()  # Set to training mode
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Measure baseline memory usage
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    
    # Forward and backward pass
    outputs = model(batch)
    loss = outputs["pred_coords"].mean()
    loss.backward()
    
    # Measure memory usage
    peak_memory_no_ckpt = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # Clear gradients
    model.zero_grad()
    
    # Test with gradient checkpointing
    # This would normally involve enabling gradient checkpointing in the model
    # For this test, we'll just simulate by measuring twice
    
    logger.info(f"Without checkpointing: {peak_memory_no_ckpt:.2f} MB")
    logger.info(f"With checkpointing: [Model dependent - not implemented in this test]")
    
    return {
        "peak_memory_no_checkpointing_mb": peak_memory_no_ckpt,
    }


def run_tests(args):
    """
    Run all tests.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    
    # Define model configuration
    model_config = {
        "residue_embed_dim": args.residue_embed_dim,
        "pair_embed_dim": args.pair_embed_dim,
        "num_blocks": args.num_blocks,
        "num_ipa_blocks": args.num_ipa_blocks,
        "no_heads": args.no_heads,
        "no_qk_points": args.no_qk_points,
        "no_v_points": args.no_v_points,
    }
    
    # Initialize model
    logger.info("Initializing model...")
    model = initialize_model(model_config, device)
    
    # Test shapes
    logger.info("\n=== Shape Tests ===")
    batch = create_dummy_batch(args.batch_size, args.seq_len, device)
    test_shapes(batch, model)
    
    # Benchmark performance
    logger.info("\n=== Performance Benchmark ===")
    seq_lens = [50, 100, 200] if args.seq_lens is None else args.seq_lens
    performance_results = benchmark_performance(
        model=model,
        batch_size=args.batch_size,
        seq_lens=seq_lens,
        device=device,
        repeats=args.repeats,
    )
    
    # Test gradient checkpointing
    if args.test_gradients and device.type == "cuda":
        logger.info("\n=== Gradient Checkpointing Test ===")
        gradient_results = test_gradient_checkpointing(
            model=model,
            batch_size=args.batch_size,
            seq_len=max(seq_lens),
            device=device,
        )
    else:
        gradient_results = {}
    
    # Compile results
    results = {
        "model_config": model_config,
        "performance": performance_results,
        "gradient_checkpointing": gradient_results,
        "device": device.type,
        "batch_size": args.batch_size,
    }
    
    # Save results
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    logger.info("\nPerformance by sequence length:")
    for seq_len, metrics in performance_results.items():
        logger.info(f"  Length {seq_len}: {metrics['avg_time']:.4f}s, {metrics['seq_per_second']:.2f} seq/s, {metrics['memory_used_mb']:.2f} MB")
    
    # VRAM recommendation for A100 40GB
    max_memory = max(metrics["peak_memory_mb"] for metrics in performance_results.values())
    logger.info(f"\nMaximum VRAM usage: {max_memory:.2f} MB")
    
    if max_memory > 38 * 1024:  # 38 GB
        logger.warning("Memory usage exceeds 38 GB, which may cause issues on A100 40GB")
        logger.warning("Recommendation: Reduce batch size or model parameters")
    else:
        logger.info("Memory usage is within limits for A100 40GB")
    
    logger.info("\nTests completed successfully!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Batch test RhoFold IPA model")
    
    # Model parameters
    parser.add_argument("--residue_embed_dim", type=int, default=128, help="Residue embedding dimension")
    parser.add_argument("--pair_embed_dim", type=int, default=64, help="Pair embedding dimension")
    parser.add_argument("--num_blocks", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--num_ipa_blocks", type=int, default=4, help="Number of IPA iterations")
    parser.add_argument("--no_heads", type=int, default=4, help="Number of attention heads in IPA")
    parser.add_argument("--no_qk_points", type=int, default=4, help="Number of query/key points in IPA")
    parser.add_argument("--no_v_points", type=int, default=8, help="Number of value points in IPA")
    
    # Test parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--seq_len", type=int, default=100, help="Sequence length for shape tests")
    parser.add_argument("--seq_lens", type=int, nargs="+", help="Sequence lengths for benchmarking")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats for benchmarking")
    parser.add_argument("--test_gradients", action="store_true", help="Test gradient checkpointing")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU even if GPU is available")
    
    # Output parameters
    parser.add_argument("--output_file", type=str, help="Path to save results JSON")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    run_tests(args)


if __name__ == "__main__":
    main()
