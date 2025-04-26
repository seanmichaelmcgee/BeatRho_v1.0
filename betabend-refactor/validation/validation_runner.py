"""
Validation Runner for RNA 3D folding model.

This module implements the ValidationRunner class that executes validation
in dual-mode (test-equivalent and training-equivalent) to quantify the 
impact of feature availability differences.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our validation dataset
from validation.validation_dataset import ValidationDataset

# Import core model components for evaluation
from src.models.rna_folding_model import RNAFoldingModel
from src.losses import compute_stable_fape_loss, stable_kabsch_align, robust_distance_calculation
from src.utils.structure_metrics import compute_rmsd, compute_tm_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ValidationRunner")


class ValidationRunner:
    """
    Runs validation in dual mode: test-equivalent and training-equivalent.
    
    Provides comprehensive analysis of model performance under both conditions
    to quantify the impact of missing features at test time.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 data_dir: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None):
        """
        Initialize validation runner.
        
        Args:
            model: Model to validate
            data_dir: Path to data directory. If None, will try to find it.
            config: Configuration dictionary. If None, uses default configuration.
            device: Device to run validation on (e.g., 'cuda', 'cpu')
        """
        self.model = model
        
        # Find data directory if not provided
        if data_dir is None:
            data_dir = self._find_data_dir()
        self.data_dir = data_dir
        
        # Initialize configuration with defaults
        self.config = {
            "batch_size": 4,  # Default batch size
            "num_workers": 2, # Workers for data loading
            "max_targets": None, # Maximum number of targets to use (None = use tier default)
            "seed": 42,       # Random seed for reproducibility
            "verbose": True,  # Whether to show progress bars
            "max_sequence_length": 512, # Maximum sequence length for memory management
            "metrics": ["rmsd", "tm_score"], # Metrics to compute
            "save_results": True, # Whether to save results
            "results_dir": None, # Directory for saving results
            "image_format": "png", # Format for saving images
        }
        
        # Update with provided config
        if config is not None:
            self.config.update(config)
            
        # Set results directory if needed using Path for better portability
        if self.config["results_dir"] is None:
            # Get the validation directory (where this script is located) and create results directory inside it
            script_dir = Path(__file__).resolve().parent
            self.config["results_dir"] = str(script_dir / "results")
            
        # Ensure results directory exists
        os.makedirs(self.config["results_dir"], exist_ok=True)
            
        # Set device for model
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model.to(self.device)
        logger.info(f"Initialized ValidationRunner with model on {self.device}")
    
    def _find_data_dir(self) -> str:
        """Find the data directory using multiple strategies with path objects for better portability."""
        # Get script directory and convert to Path for better manipulation
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        
        # Try common locations relative to script path and working directory
        possible_dirs = [
            project_root / "data",             # Most likely location (project_root/data)
            script_dir / "data",               # In validation directory
            Path.cwd() / "data",               # Current working directory
            Path.cwd().parent / "data",        # Parent of current directory
            Path.cwd().parent.parent / "data", # Grandparent of current directory
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists():
                logger.info(f"Found data directory: {dir_path}")
                return str(dir_path)
        
        # Use the project root's data directory as a last resort
        fallback_dir = project_root / "data"
        logger.warning(f"No data directory found, using fallback: {fallback_dir}")
        return str(fallback_dir)
        
    def run_validation(self, 
                       subset_name: str = "technical", 
                       run_both_modes: bool = True) -> Dict[str, Any]:
        """
        Run validation in one or both modes.
        
        Args:
            subset_name: Validation subset ("technical", "scientific", "comprehensive")
            run_both_modes: Whether to run both test and train modes
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "subset_name": subset_name,
            "configuration": self.config,
        }
        
        # Always run test-equivalent mode (for Kaggle performance estimation)
        logger.info(f"\n{'='*50}\nRunning TEST-EQUIVALENT mode validation\n{'='*50}\n")
        test_results = self.run_test_equivalent_mode(subset_name)
        results["test_mode"] = test_results
        
        # Optionally run training-equivalent mode (with all features)
        if run_both_modes:
            logger.info(f"\n{'='*50}\nRunning TRAINING-EQUIVALENT mode validation\n{'='*50}\n")
            train_results = self.run_training_equivalent_mode(subset_name)
            results["train_mode"] = train_results
            
            # Compare modes and analyze difference
            logger.info(f"\n{'='*50}\nAnalyzing performance differences\n{'='*50}\n")
            analysis = self.analyze_mode_differences(test_results, train_results)
            results["analysis"] = analysis
            
            # Generate visualization comparing the modes
            if self.config["save_results"]:
                self._generate_comparison_visualization(test_results, train_results, subset_name)
        
        # Save overall results if configured
        if self.config["save_results"]:
            self._save_results(results, subset_name)
        
        return results
    
    def run_test_equivalent_mode(self, subset_name: str) -> Dict[str, Any]:
        """
        Run validation using only test-available features.
        
        Args:
            subset_name: Validation subset name
            
        Returns:
            Dictionary with test-mode validation results
        """
        # Create test-mode dataset (no pseudo-dihedrals)
        dataset = ValidationDataset(
            data_dir=self.data_dir,
            subset_name=subset_name,
            test_mode=True,
            max_targets=self.config["max_targets"],
            seed=self.config["seed"],
            target_ids=self.config.get("target_ids")
        )
        
        # Check if we have any data
        if len(dataset) == 0:
            logger.warning(f"No validation targets available for {subset_name} subset in test mode")
            return {"error": "No validation targets available"}
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self._calculate_batch_size(dataset),
            collate_fn=dataset.collate_fn,
            shuffle=False,
            num_workers=self.config["num_workers"]
        )
        
        # Run evaluation
        return self._evaluate_model(dataloader, "test_equivalent")
    
    def run_training_equivalent_mode(self, subset_name: str) -> Dict[str, Any]:
        """
        Run validation using all training features (including pseudo-dihedrals).
        
        Args:
            subset_name: Validation subset name
            
        Returns:
            Dictionary with train-mode validation results
        """
        # Create train-mode dataset (with pseudo-dihedrals)
        dataset = ValidationDataset(
            data_dir=self.data_dir,
            subset_name=subset_name,
            test_mode=False,
            max_targets=self.config["max_targets"],
            seed=self.config["seed"],
            target_ids=self.config.get("target_ids")
        )
        
        # Check if we have any data
        if len(dataset) == 0:
            logger.warning(f"No validation targets available for {subset_name} subset in train mode")
            return {"error": "No validation targets available"}
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self._calculate_batch_size(dataset),
            collate_fn=dataset.collate_fn,
            shuffle=False,
            num_workers=self.config["num_workers"]
        )
        
        # Run evaluation
        return self._evaluate_model(dataloader, "training_equivalent")
    
    def _calculate_batch_size(self, dataset: ValidationDataset) -> int:
        """
        Calculate appropriate batch size based on sequence lengths.
        
        Adjusts batch size based on average sequence length to prevent OOM errors.
        
        Args:
            dataset: Validation dataset
            
        Returns:
            Adjusted batch size
        """
        # Get average sequence length in dataset
        avg_len = dataset.get_average_sequence_length()
        
        # Get base batch size from config
        base_batch_size = self.config.get("batch_size", 4)
        
        # Adjust based on sequence length (longer sequences need smaller batches)
        if avg_len > 300:
            return max(1, base_batch_size // 4)
        elif avg_len > 150:
            return max(1, base_batch_size // 2)
        else:
            return base_batch_size
    
    def _evaluate_model(self, dataloader: DataLoader, mode_name: str) -> Dict[str, Any]:
        """
        Evaluate model on provided dataloader.
        
        Args:
            dataloader: DataLoader with validation samples
            mode_name: Mode name for logging ("test_equivalent" or "training_equivalent")
        
        Returns:
            Dictionary with evaluation metrics including:
            - Basic metrics (RMSD, TM-score)
            - Per-sample statistics
            - Problematic sample diagnostics
        """
        device = self.device
        model = self.model
        
        # Set model to evaluation mode
        model.eval()
        
        # Initialize result containers
        all_target_ids = []
        all_pred_coords = []
        all_true_coords = []
        all_rmsd_values = []
        all_per_residue_errors = []
        all_sequence_lengths = []
        
        # Track problematic samples
        problematic_samples = []
        
        # Track validation time
        start_time = time.time()
        
        # Use tqdm for progress bar if verbose
        dataloader_iter = tqdm(dataloader) if self.config["verbose"] else dataloader
        
        # Iterate through validation batches
        with torch.no_grad():
            for batch in dataloader_iter:
                # Move data to device
                batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                               for k, v in batch.items()}
                
                # Store target IDs
                all_target_ids.extend(batch["ids"])
                
                # Forward pass - retry with smaller batch if OOM error
                try:
                    # Get model predictions
                    outputs = model(batch_device)
                    
                    # Process each sample in batch separately
                    for i in range(len(batch["ids"])):
                        # Extract data for this sample
                        target_id = batch["ids"][i]
                        seq_len = batch["lengths"][i].item()
                        
                        # Extract ground truth and predicted coordinates
                        true_coords = batch["atom_positions"][i, :seq_len].cpu()  # (L, 3)
                        pred_coords = outputs["pred_coords"][i, :seq_len].cpu()   # (L, 3)
                        mask = batch["mask"][i, :seq_len].cpu().bool()            # (L,)
                        
                        # Skip if we have no valid coordinates
                        if mask.sum() < 3:
                            logger.warning(f"Sample {target_id} has fewer than 3 valid positions, skipping")
                            continue
                        
                        # Store coordinates for later analysis
                        all_pred_coords.append(pred_coords[mask])
                        all_true_coords.append(true_coords[mask])
                        all_sequence_lengths.append(seq_len)
                        
                        # Calculate RMSD for this sample
                        try:
                            # Check for extreme values in coordinates first
                            pred_min = pred_coords[mask].min().item()
                            pred_max = pred_coords[mask].max().item()
                            true_min = true_coords[mask].min().item()
                            true_max = true_coords[mask].max().item()
                            
                            # Define reasonable coordinate limits
                            MAX_COORD_VALUE = 500.0  # Reasonable upper limit for RNA coordinates
                            
                            # Log warning if extreme values are found
                            if (abs(pred_min) > MAX_COORD_VALUE or abs(pred_max) > MAX_COORD_VALUE or
                                abs(true_min) > MAX_COORD_VALUE or abs(true_max) > MAX_COORD_VALUE):
                                error_msg = f"Extreme coordinate values detected: " + \
                                          f"pred=[{pred_min:.2f}, {pred_max:.2f}], " + \
                                          f"true=[{true_min:.2f}, {true_max:.2f}]"
                                logger.warning(f"{error_msg} for {target_id}")
                                
                                # Record problematic sample
                                problematic_samples.append({
                                    "id": target_id,
                                    "issue": "extreme_coordinates",
                                    "details": error_msg,
                                    "seq_len": seq_len,
                                    "pred_min": float(pred_min),
                                    "pred_max": float(pred_max),
                                    "true_min": float(true_min),
                                    "true_max": float(true_max)
                                })
                                
                                # Skip this sample if it has extreme values
                                logger.warning(f"Skipping sample {target_id} due to extreme coordinate values")
                                continue
                            
                            # Use stable compute_rmsd function from structure_metrics
                            rmsd = compute_rmsd(
                                pred_coords[mask].unsqueeze(0), 
                                true_coords[mask].unsqueeze(0)
                            ).item()
                            
                            # Sanity check on RMSD value
                            if rmsd > 100.0:  # RMSD shouldn't be this large for valid RNA structures
                                logger.warning(f"Unusually large RMSD detected for {target_id}: {rmsd:.4f} Å")
                                
                                # Record problematic sample
                                problematic_samples.append({
                                    "id": target_id,
                                    "issue": "extreme_rmsd",
                                    "details": f"RMSD value: {rmsd:.4f} Å",
                                    "seq_len": seq_len,
                                    "original_rmsd": float(rmsd)
                                })
                                
                                # Limit RMSD to a reasonable value for reporting purposes
                                rmsd = 100.0
                                
                            all_rmsd_values.append(rmsd)
                            
                            # Apply Kabsch alignment for per-residue error calculation
                            pred_aligned = stable_kabsch_align(
                                pred_coords[mask], true_coords[mask]
                            )
                            
                            # Compute per-residue distances for detailed analysis
                            distances = robust_distance_calculation(
                                pred_aligned, true_coords[mask]
                            )
                            
                            # Cap distances at a reasonable value
                            distances = torch.clamp(distances, max=50.0)
                            
                            # Store per-residue errors for detailed analysis
                            per_residue_error = torch.zeros(seq_len)
                            per_residue_error[mask] = distances
                            all_per_residue_errors.append(per_residue_error)
                            
                        except Exception as e:
                            error_msg = f"Error calculating RMSD: {str(e)}"
                            logger.error(f"{error_msg} for {target_id}")
                            
                            # Record problematic sample
                            problematic_samples.append({
                                "id": target_id,
                                "issue": "calculation_error",
                                "details": error_msg,
                                "seq_len": seq_len
                            })
                            continue
                
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        logger.warning(f"OOM error with batch size {len(batch['ids'])}. Consider reducing batch size.")
                        # Could implement batch splitting here if needed
                    else:
                        logger.error(f"Error during model evaluation: {e}")
                    continue
        
        # Get total time
        end_time = time.time()
        eval_time = end_time - start_time
        
        # Create summary of results
        num_samples = len(all_rmsd_values)
        if num_samples == 0:
            logger.error(f"No valid samples processed in {mode_name} mode")
            return {"error": "No valid samples processed"}
        
        # Compute summary statistics
        mean_rmsd = np.mean(all_rmsd_values)
        median_rmsd = np.median(all_rmsd_values)
        min_rmsd = np.min(all_rmsd_values)
        max_rmsd = np.max(all_rmsd_values)
        
        # Compute average per-residue error for visualization
        avg_per_residue_error = self._compute_avg_per_residue_error(all_per_residue_errors)
        
        # Compute TM-score if in metrics list
        tm_scores = []
        if "tm_score" in self.config["metrics"]:
            for i, (pred, true) in enumerate(zip(all_pred_coords, all_true_coords)):
                try:
                    # Check for extreme values first
                    pred_min = pred.min().item()
                    pred_max = pred.max().item()
                    true_min = true.min().item()
                    true_max = true.max().item()
                    
                    # Define reasonable coordinate limits
                    MAX_COORD_VALUE = 500.0  # Reasonable upper limit for RNA coordinates
                    
                    # Skip if extreme values are found
                    if (abs(pred_min) > MAX_COORD_VALUE or abs(pred_max) > MAX_COORD_VALUE or
                        abs(true_min) > MAX_COORD_VALUE or abs(true_max) > MAX_COORD_VALUE):
                        logger.warning(f"Skipping TM-score calculation due to extreme coordinate values: " +
                                    f"pred=[{pred_min:.2f}, {pred_max:.2f}], " +
                                    f"true=[{true_min:.2f}, {true_max:.2f}]")
                        continue
                    
                    tm_score = self._calculate_tm_score(pred, true)
                    
                    # Sanity check on TM-score value
                    if tm_score < 0.0 or tm_score > 1.0 or torch.isnan(torch.tensor(tm_score)):
                        logger.warning(f"Invalid TM-score value: {tm_score}. Setting to 0.0")
                        tm_score = 0.0
                        
                    tm_scores.append(tm_score)
                except Exception as e:
                    logger.warning(f"Error calculating TM-score: {e}")
        
        # Create results dictionary
        results = {
            "mode": mode_name,
            "num_samples": num_samples,
            "target_ids": all_target_ids[:num_samples],  # Only keep successful ones
            "mean_rmsd": mean_rmsd,
            "median_rmsd": median_rmsd,
            "min_rmsd": min_rmsd, 
            "max_rmsd": max_rmsd,
            "rmsd_values": all_rmsd_values,
            "avg_per_residue_error": avg_per_residue_error.tolist() if isinstance(avg_per_residue_error, np.ndarray) else None,
            "evaluation_time_seconds": eval_time,
            "mean_sequence_length": np.mean(all_sequence_lengths),
            "problematic_samples": problematic_samples
        }
        
        # Add TM-score if available
        if tm_scores:
            results["mean_tm_score"] = np.mean(tm_scores)
            results["median_tm_score"] = np.median(tm_scores)
            results["tm_scores"] = tm_scores
        
        # Log summary
        logger.info(f"{mode_name.upper()} Validation Results:")
        logger.info(f"  Samples processed: {num_samples}")
        logger.info(f"  Mean RMSD: {mean_rmsd:.4f} Å")
        logger.info(f"  Median RMSD: {median_rmsd:.4f} Å")
        if tm_scores:
            logger.info(f"  Mean TM-score: {results['mean_tm_score']:.4f}")
        logger.info(f"  Evaluation time: {eval_time:.2f} seconds")
        
        # Log issues
        if problematic_samples:
            logger.warning(f"  Problematic samples detected: {len(problematic_samples)}")
            for i, sample in enumerate(problematic_samples[:3]):  # Show first 3 only to avoid log spam
                logger.warning(f"    {i+1}. {sample['id']}: {sample['issue']} - {sample['details']}")
            if len(problematic_samples) > 3:
                logger.warning(f"    ... and {len(problematic_samples) - 3} more issues")
        
        # Create and save mode-specific visualizations
        if self.config["save_results"]:
            self._generate_mode_visualization(results, mode_name)
        
        return results
    
    def _compute_avg_per_residue_error(self, per_residue_errors: List[torch.Tensor]) -> np.ndarray:
        """
        Compute average per-residue error across all samples.
        
        Args:
            per_residue_errors: List of per-residue error tensors
            
        Returns:
            Average error at each position normalized by sequence length
        """
        # Convert all to numpy and normalize by sequence length
        normalized_errors = []
        
        for error_tensor in per_residue_errors:
            seq_len = len(error_tensor)
            # Create normalized positions (0 to 1)
            positions = np.linspace(0, 1, seq_len)
            # Convert to numpy
            errors = error_tensor.numpy() if hasattr(error_tensor, 'numpy') else np.array(error_tensor)
            # Store as (position, error) pairs
            for pos, err in zip(positions, errors):
                if not np.isnan(err) and not np.isinf(err):
                    normalized_errors.append((pos, err))
        
        if not normalized_errors:
            return np.array([])
            
        # Convert to array
        normalized_errors = np.array(normalized_errors)
        
        # Bin into 20 equal-sized bins
        num_bins = 20
        bin_edges = np.linspace(0, 1, num_bins + 1)
        binned_errors = [[] for _ in range(num_bins)]
        
        # Assign errors to bins
        for pos, err in normalized_errors:
            bin_idx = min(int(pos * num_bins), num_bins - 1)
            binned_errors[bin_idx].append(err)
        
        # Compute average for each bin
        avg_errors = np.array([np.mean(errors) if errors else np.nan for errors in binned_errors])
        
        # Get bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return np.column_stack((bin_centers, avg_errors))
    
    def _calculate_tm_score(self, pred_coords: torch.Tensor, true_coords: torch.Tensor) -> float:
        """
        Calculate TM-score for predicted structure.
        
        TM-score measures the similarity of two protein structures with different
        lengths. It is a length-independent metric, with values in (0,1].
        A TM-score ≤ 0.17 corresponds to random similarity,
        while TM-score > 0.5 typically corresponds to structures with same fold.
        
        Args:
            pred_coords: Predicted coordinates (L, 3)
            true_coords: True coordinates (L, 3)
            
        Returns:
            TM-score value
        """
        # Ensure we have matching sizes
        assert pred_coords.shape == true_coords.shape, "Coordinate shapes must match"
        
        # Use the compute_tm_score function from structure_metrics
        tm_score = compute_tm_score(
            pred_coords.unsqueeze(0), 
            true_coords.unsqueeze(0)
        ).item()
        
        return tm_score
    
    def analyze_mode_differences(self, 
                                test_results: Dict[str, Any], 
                                train_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze differences between test and training modes.
        
        Calculates absolute and relative improvements, significance of differences,
        and provides scientific insights about feature importance.
        
        Args:
            test_results: Results from test-equivalent mode
            train_results: Results from training-equivalent mode
            
        Returns:
            Dictionary with analysis metrics
        """
        # Check for errors in either mode
        if "error" in test_results or "error" in train_results:
            logger.warning("Cannot analyze mode differences - errors in validation results")
            return {"error": "Cannot analyze mode differences - errors in validation results"}
        
        # Initialize analysis dictionary
        analysis = {}
        
        # Calculate RMSD differences (lower is better for RMSD)
        if "mean_rmsd" in test_results and "mean_rmsd" in train_results:
            test_rmsd = test_results["mean_rmsd"]
            train_rmsd = train_results["mean_rmsd"]
            
            # Absolute difference
            rmsd_abs_diff = test_rmsd - train_rmsd
            
            # Relative difference (as percentage)
            if train_rmsd > 0:
                rmsd_rel_diff = (rmsd_abs_diff / train_rmsd) * 100
            else:
                rmsd_rel_diff = 0
            
            # Interpret the difference
            if rmsd_abs_diff > 0:
                rmsd_impact = "NEGATIVE"  # Test mode performs worse
                rmsd_interpretation = "Missing dihedral features degrades performance"
            elif rmsd_abs_diff < 0:
                rmsd_impact = "POSITIVE"  # Test mode performs better
                rmsd_interpretation = "Model performs better without dihedral features (unexpected)"
            else:
                rmsd_impact = "NEUTRAL"
                rmsd_interpretation = "Dihedral features have no impact on performance"
            
            # Add to analysis
            analysis["rmsd"] = {
                "test_value": test_rmsd,
                "train_value": train_rmsd,
                "absolute_difference": rmsd_abs_diff,
                "relative_difference_percent": rmsd_rel_diff,
                "impact": rmsd_impact,
                "interpretation": rmsd_interpretation
            }
        
        # Calculate TM-score differences (higher is better for TM-score)
        if "mean_tm_score" in test_results and "mean_tm_score" in train_results:
            test_tm = test_results["mean_tm_score"]
            train_tm = train_results["mean_tm_score"]
            
            # Absolute difference
            tm_abs_diff = train_tm - test_tm
            
            # Relative difference (as percentage)
            if test_tm > 0:
                tm_rel_diff = (tm_abs_diff / test_tm) * 100
            else:
                tm_rel_diff = 0
            
            # Interpret the difference
            if tm_abs_diff > 0:
                tm_impact = "NEGATIVE"  # Train mode performs better
                tm_interpretation = "Missing dihedral features degrades performance"
            elif tm_abs_diff < 0:
                tm_impact = "POSITIVE"  # Test mode performs better
                tm_interpretation = "Model performs better without dihedral features (unexpected)"
            else:
                tm_impact = "NEUTRAL"
                tm_interpretation = "Dihedral features have no impact on performance"
            
            # Add to analysis
            analysis["tm_score"] = {
                "test_value": test_tm,
                "train_value": train_tm,
                "absolute_difference": tm_abs_diff,
                "relative_difference_percent": tm_rel_diff,
                "impact": tm_impact,
                "interpretation": tm_interpretation
            }
        
        # Per-residue error analysis
        if "avg_per_residue_error" in test_results and "avg_per_residue_error" in train_results:
            test_errors = np.array(test_results["avg_per_residue_error"])
            train_errors = np.array(train_results["avg_per_residue_error"])
            
            if len(test_errors) > 0 and len(train_errors) > 0:
                # Calculate where the largest differences occur
                try:
                    # Extract position and error columns
                    test_pos, test_err = test_errors[:, 0], test_errors[:, 1]
                    train_pos, train_err = train_errors[:, 0], train_errors[:, 1]
                    
                    # Verify matching positions
                    if np.allclose(test_pos, train_pos):
                        # Calculate error differences (test - train)
                        error_diffs = test_err - train_err
                        
                        # Find the positions with the largest differences
                        valid_diffs = ~np.isnan(error_diffs)
                        if np.any(valid_diffs):
                            max_diff_idx = np.nanargmax(np.abs(error_diffs))
                            max_diff_pos = test_pos[max_diff_idx]
                            max_diff_value = error_diffs[max_diff_idx]
                            
                            # Add to analysis
                            analysis["per_residue"] = {
                                "max_difference_position": float(max_diff_pos),
                                "max_difference_value": float(max_diff_value),
                                "position_category": "end" if max_diff_pos < 0.1 or max_diff_pos > 0.9 else "middle"
                            }
                except Exception as e:
                    logger.warning(f"Error in per-residue analysis: {e}")
        
        # Overall conclusion
        rmsd_impact = analysis.get("rmsd", {}).get("impact", "UNKNOWN")
        tm_impact = analysis.get("tm_score", {}).get("impact", "UNKNOWN")
        
        # Get the magnitude of the impact
        rmsd_rel_diff = abs(analysis.get("rmsd", {}).get("relative_difference_percent", 0))
        tm_rel_diff = abs(analysis.get("tm_score", {}).get("relative_difference_percent", 0))
        
        if "NEGATIVE" in [rmsd_impact, tm_impact]:
            if rmsd_rel_diff > 20 or tm_rel_diff > 20:
                severity = "MAJOR"
            elif rmsd_rel_diff > 5 or tm_rel_diff > 5:
                severity = "MODERATE"
            else:
                severity = "MINOR"
        else:
            severity = "NEGLIGIBLE"
                
        analysis["conclusion"] = {
            "overall_impact": "NEGATIVE" if "NEGATIVE" in [rmsd_impact, tm_impact] else "NEUTRAL",
            "severity": severity,
            "recommendation": self._generate_recommendation(severity)
        }
        
        return analysis
    
    def _generate_recommendation(self, severity: str) -> str:
        """Generate recommendation based on impact severity."""
        if severity == "MAJOR":
            return ("The absence of dihedral features at test time has a major negative impact on model performance. "
                   "Consider redesigning the model architecture to better handle the feature availability mismatch, "
                   "or implement a strategy to predict dihedral features from the available test data.")
        elif severity == "MODERATE":
            return ("The absence of dihedral features has a moderate negative impact. "
                   "Consider adding a self-supervised auxiliary task to learn to predict dihedral features "
                   "internally from the available test features.")
        elif severity == "MINOR":
            return ("The absence of dihedral features has only a minor negative impact. "
                   "The current model is relatively robust to this feature availability mismatch, "
                   "but minor improvements could still be made.")
        else:
            return ("The model appears robust to the absence of dihedral features at test time. "
                   "No significant changes to the architecture are needed to address this specific issue.")
    
    def _generate_mode_visualization(self, results: Dict[str, Any], mode_name: str) -> None:
        """
        Generate and save visualizations for a specific validation mode.
        
        Args:
            results: Results dictionary for the mode
            mode_name: Name of the validation mode
        """
        if "error" in results or "rmsd_values" not in results:
            return
        
        # Create figure for RMSD distribution
        plt.figure(figsize=(10, 6))
        
        # Extract RMSD values
        rmsd_values = results["rmsd_values"]
        
        # Create histogram
        plt.hist(rmsd_values, bins=15, alpha=0.7, color='skyblue')
        plt.axvline(results["mean_rmsd"], color='red', linestyle='--', 
                   label=f'Mean: {results["mean_rmsd"]:.2f} Å')
        plt.axvline(results["median_rmsd"], color='green', linestyle='--', 
                   label=f'Median: {results["median_rmsd"]:.2f} Å')
        
        # Add title and labels
        plt.title(f'RMSD Distribution - {mode_name.upper()} Mode')
        plt.xlabel('RMSD (Å)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save figure
        filename = os.path.join(self.config["results_dir"], f"rmsd_dist_{mode_name}.{self.config['image_format']}")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create per-residue error plot if available
        if "avg_per_residue_error" in results and results["avg_per_residue_error"]:
            try:
                # Extract data
                error_data = np.array(results["avg_per_residue_error"])
                positions = error_data[:, 0]
                errors = error_data[:, 1]
                
                # Remove NaN values
                valid_mask = ~np.isnan(errors)
                positions = positions[valid_mask]
                errors = errors[valid_mask]
                
                if len(positions) > 0:
                    # Create figure
                    plt.figure(figsize=(10, 6))
                    
                    # Plot per-residue error
                    plt.plot(positions, errors, 'o-', color='blue', alpha=0.7)
                    
                    # Add title and labels
                    plt.title(f'Per-Residue Error - {mode_name.upper()} Mode')
                    plt.xlabel('Normalized Position (0-1)')
                    plt.ylabel('Average Error (Å)')
                    plt.grid(alpha=0.3)
                    
                    # Add horizontal line for mean RMSD
                    plt.axhline(results["mean_rmsd"], color='red', linestyle='--', 
                               label=f'Mean RMSD: {results["mean_rmsd"]:.2f} Å')
                    plt.legend()
                    
                    # Save figure
                    filename = os.path.join(self.config["results_dir"], 
                                          f"per_residue_error_{mode_name}.{self.config['image_format']}")
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                logger.warning(f"Error generating per-residue error plot: {e}")
    
    def _generate_comparison_visualization(self, 
                                          test_results: Dict[str, Any], 
                                          train_results: Dict[str, Any],
                                          subset_name: str) -> None:
        """
        Generate and save visualizations comparing test and train modes.
        
        Args:
            test_results: Results from test-equivalent mode
            train_results: Results from training-equivalent mode
            subset_name: Name of the validation subset
        """
        if "error" in test_results or "error" in train_results:
            return
        
        # Check for required data
        if ("avg_per_residue_error" not in test_results or 
            "avg_per_residue_error" not in train_results or
            not test_results["avg_per_residue_error"] or
            not train_results["avg_per_residue_error"]):
            return
        
        try:
            # Extract data
            test_error_data = np.array(test_results["avg_per_residue_error"])
            train_error_data = np.array(train_results["avg_per_residue_error"])
            
            test_positions = test_error_data[:, 0]
            test_errors = test_error_data[:, 1]
            
            train_positions = train_error_data[:, 0]
            train_errors = train_error_data[:, 1]
            
            # Verify positions match
            if len(test_positions) != len(train_positions) or not np.allclose(test_positions, train_positions):
                logger.warning("Position mismatch in per-residue error data, skipping comparison visualization")
                return
            
            # Create figure
            plt.figure(figsize=(12, 7))
            
            # Plot both error profiles
            plt.plot(test_positions, test_errors, 'o-', color='red', alpha=0.7, 
                    label='Test-Equivalent Mode (No Dihedrals)')
            plt.plot(train_positions, train_errors, 'o-', color='blue', alpha=0.7,
                    label='Training-Equivalent Mode (With Dihedrals)')
            
            # Add shaded region to highlight difference
            plt.fill_between(test_positions, test_errors, train_errors, 
                            color='grey', alpha=0.3, label='Performance Gap')
            
            # Add title and labels
            plt.title(f'Feature Availability Impact - {subset_name.capitalize()} Validation')
            plt.xlabel('Normalized Position (0-1)')
            plt.ylabel('Average Error (Å)')
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Add mean RMSD values
            plt.axhline(test_results["mean_rmsd"], color='red', linestyle='--', 
                       label=f'Mean RMSD (Test): {test_results["mean_rmsd"]:.2f} Å')
            plt.axhline(train_results["mean_rmsd"], color='blue', linestyle='--',
                       label=f'Mean RMSD (Train): {train_results["mean_rmsd"]:.2f} Å')
            
            # Annotate difference percentage
            if "rmsd" in test_results and "rmsd" in train_results:
                test_rmsd = test_results["mean_rmsd"]
                train_rmsd = train_results["mean_rmsd"]
                diff_pct = ((test_rmsd - train_rmsd) / train_rmsd) * 100 if train_rmsd > 0 else 0
                
                plt.annotate(f'RMSD Difference: {diff_pct:.1f}%', 
                            xy=(0.75, 0.9), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
            
            # Save figure
            filename = os.path.join(self.config["results_dir"], 
                                  f"mode_comparison_{subset_name}.{self.config['image_format']}")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also create a summary bar chart
            plt.figure(figsize=(8, 6))
            metrics = []
            test_values = []
            train_values = []
            
            # Add RMSD
            if "mean_rmsd" in test_results and "mean_rmsd" in train_results:
                metrics.append("Mean RMSD (Å)")
                test_values.append(test_results["mean_rmsd"])
                train_values.append(train_results["mean_rmsd"])
            
            # Add TM-score if available
            if "mean_tm_score" in test_results and "mean_tm_score" in train_results:
                metrics.append("Mean TM-score")
                test_values.append(test_results["mean_tm_score"])
                train_values.append(train_results["mean_tm_score"])
            
            # Create bar chart if we have metrics
            if metrics:
                # Set up bar positions
                x = np.arange(len(metrics))
                width = 0.35
                
                # Create bars
                plt.bar(x - width/2, test_values, width, label='Test-Equivalent Mode')
                plt.bar(x + width/2, train_values, width, label='Training-Equivalent Mode')
                
                # Add chart elements
                plt.xlabel('Metric')
                plt.ylabel('Value')
                plt.title(f'Performance Comparison - {subset_name.capitalize()} Validation')
                plt.xticks(x, metrics)
                plt.legend()
                plt.grid(axis='y', alpha=0.3)
                
                # Save figure
                filename = os.path.join(self.config["results_dir"], 
                                     f"metrics_comparison_{subset_name}.{self.config['image_format']}")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
        
        except Exception as e:
            logger.warning(f"Error generating comparison visualization: {e}")
    
    def _save_results(self, results: Dict[str, Any], subset_name: str) -> None:
        """
        Save validation results to file.
        
        Args:
            results: Results dictionary
            subset_name: Name of the validation subset
        """
        if not self.config["save_results"]:
            return
        
        try:
            # Create JSON-serializable results (convert numpy values, etc.)
            serializable_results = self._make_serializable(results)
            
            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.config["results_dir"], 
                                   f"validation_results_{subset_name}_{timestamp}.json")
            
            # Save JSON file
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved validation results to {filename}")
            
            # Also save a readable markdown report
            self._generate_markdown_report(serializable_results, subset_name)
            
        except Exception as e:
            logger.error(f"Error saving validation results: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            if obj.size == 1:  # Single value
                return obj.item()
            elif obj.ndim == 1:  # 1D array
                return obj.tolist()
            else:  # Multi-dimensional array
                return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, 'item'):  # For torch tensors, etc.
            return obj.item()
        else:
            # Try to convert to string if not directly serializable
            try:
                return str(obj)
            except:
                return None
    
    def _format_metric_value(self, value, precision=4):
        """Format a metric value safely with proper type checking."""
        if value is None:
            return "N/A"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                return "N/A"
            return f"{value:.{precision}f}"
        return str(value)
    
    def _generate_markdown_report(self, results: Dict[str, Any], subset_name: str) -> None:
        """
        Generate a markdown report of validation results.
        
        Args:
            results: Results dictionary (serializable)
            subset_name: Name of the validation subset
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.config["results_dir"], 
                               f"validation_report_{subset_name}_{timestamp}.md")
        
        try:
            with open(filename, 'w') as f:
                f.write(f"# RNA 3D Folding Model Validation Report\n\n")
                f.write(f"## Overview\n\n")
                f.write(f"- **Validation Subset:** {subset_name.capitalize()}\n")
                f.write(f"- **Date:** {results['timestamp']}\n")
                f.write(f"- **Configuration:** {json.dumps(results['configuration'], indent=2)}\n\n")
                
                # Summary section
                f.write(f"## Summary\n\n")
                
                # Check if we have both modes
                if "test_mode" in results and "train_mode" in results:
                    test_rmsd = results["test_mode"].get("mean_rmsd", "N/A")
                    train_rmsd = results["train_mode"].get("mean_rmsd", "N/A")
                    
                    f.write("| Metric | Test-Equivalent Mode | Training-Equivalent Mode | Difference |\n")
                    f.write("|--------|---------------------|--------------------------|------------|\n")
                    
                    # Use the safe formatting method
                    test_rmsd_str = self._format_metric_value(test_rmsd)
                    train_rmsd_str = self._format_metric_value(train_rmsd)
                    
                    if isinstance(test_rmsd, (int, float)) and isinstance(train_rmsd, (int, float)) and train_rmsd > 0:
                        diff = test_rmsd - train_rmsd
                        diff_pct = (diff / train_rmsd) * 100
                        diff_str = f"{self._format_metric_value(diff)} ({self._format_metric_value(diff_pct, precision=1)}%)"
                    else:
                        diff_str = "N/A"
                        
                    f.write(f"| Mean RMSD (Å) | {test_rmsd_str} | {train_rmsd_str} | {diff_str} |\n")
                    
                    # Add TM-score if available
                    if "mean_tm_score" in results["test_mode"] and "mean_tm_score" in results["train_mode"]:
                        test_tm = results["test_mode"]["mean_tm_score"]
                        train_tm = results["train_mode"]["mean_tm_score"]
                        
                        # Use the safe formatting method
                        test_tm_str = self._format_metric_value(test_tm)
                        train_tm_str = self._format_metric_value(train_tm)
                        
                        if isinstance(test_tm, (int, float)) and isinstance(train_tm, (int, float)) and test_tm > 0:
                            tm_diff = train_tm - test_tm
                            tm_diff_pct = (tm_diff / test_tm) * 100
                            tm_diff_str = f"{self._format_metric_value(tm_diff)} ({self._format_metric_value(tm_diff_pct, precision=1)}%)"
                        else:
                            tm_diff_str = "N/A"
                            
                        f.write(f"| Mean TM-score | {test_tm_str} | {train_tm_str} | {tm_diff_str} |\n")
                    
                    # Add analysis conclusion if available
                    if "analysis" in results and "conclusion" in results["analysis"]:
                        conclusion = results["analysis"]["conclusion"]
                        f.write(f"\n### Impact Analysis\n\n")
                        f.write(f"- **Overall Impact:** {conclusion['overall_impact']}\n")
                        f.write(f"- **Severity:** {conclusion['severity']}\n")
                        f.write(f"- **Recommendation:**\n  {conclusion['recommendation']}\n")
                else:
                    # Single mode report
                    mode_key = "test_mode" if "test_mode" in results else "train_mode"
                    if mode_key in results and "mean_rmsd" in results[mode_key]:
                        mode_results = results[mode_key]
                        f.write("| Metric | Value |\n")
                        f.write("|--------|-------|\n")
                        f.write(f"| Mean RMSD (Å) | {self._format_metric_value(mode_results.get('mean_rmsd'))} |\n")
                        f.write(f"| Median RMSD (Å) | {self._format_metric_value(mode_results.get('median_rmsd'))} |\n")
                        if "mean_tm_score" in mode_results:
                            f.write(f"| Mean TM-score | {self._format_metric_value(mode_results.get('mean_tm_score'))} |\n")
                
                # Add figures section
                image_format = results["configuration"]["image_format"]
                f.write(f"\n## Visualizations\n\n")
                
                if "test_mode" in results and "train_mode" in results:
                    f.write(f"### Mode Comparison\n\n")
                    f.write(f"![Mode Comparison](mode_comparison_{subset_name}.{image_format})\n\n")
                    f.write(f"![Metrics Comparison](metrics_comparison_{subset_name}.{image_format})\n\n")
                
                f.write(f"### Error Distributions\n\n")
                
                if "test_mode" in results:
                    f.write(f"#### Test-Equivalent Mode\n\n")
                    f.write(f"![RMSD Distribution](rmsd_dist_test_equivalent.{image_format})\n\n")
                    f.write(f"![Per-Residue Error](per_residue_error_test_equivalent.{image_format})\n\n")
                
                if "train_mode" in results:
                    f.write(f"#### Training-Equivalent Mode\n\n")
                    f.write(f"![RMSD Distribution](rmsd_dist_training_equivalent.{image_format})\n\n")
                    f.write(f"![Per-Residue Error](per_residue_error_training_equivalent.{image_format})\n\n")
                
                # Add detailed results section
                f.write(f"\n## Detailed Results\n\n")
                
                # Test mode
                if "test_mode" in results:
                    f.write(f"### Test-Equivalent Mode\n\n")
                    test_mode = results["test_mode"]
                    f.write(f"- Samples processed: {test_mode.get('num_samples', 'N/A')}\n")
                    f.write(f"- Mean sequence length: {self._format_metric_value(test_mode.get('mean_sequence_length'), precision=1)}\n")
                    f.write(f"- Evaluation time: {self._format_metric_value(test_mode.get('evaluation_time_seconds'), precision=2)} seconds\n")
                    
                    # Add problematic samples info
                    if "problematic_samples" in test_mode and test_mode["problematic_samples"]:
                        f.write(f"\n#### Problematic Samples ({len(test_mode['problematic_samples'])} found)\n\n")
                        f.write("| Sample ID | Issue | Details |\n")
                        f.write("|-----------|-------|--------|\n")
                        for sample in test_mode["problematic_samples"]:
                            f.write(f"| {sample['id']} | {sample['issue']} | {sample['details']} |\n")
                
                # Train mode
                if "train_mode" in results:
                    f.write(f"\n### Training-Equivalent Mode\n\n")
                    train_mode = results["train_mode"]
                    f.write(f"- Samples processed: {train_mode.get('num_samples', 'N/A')}\n")
                    f.write(f"- Mean sequence length: {self._format_metric_value(train_mode.get('mean_sequence_length'), precision=1)}\n")
                    f.write(f"- Evaluation time: {self._format_metric_value(train_mode.get('evaluation_time_seconds'), precision=2)} seconds\n")
                    
                    # Add problematic samples info
                    if "problematic_samples" in train_mode and train_mode["problematic_samples"]:
                        f.write(f"\n#### Problematic Samples ({len(train_mode['problematic_samples'])} found)\n\n")
                        f.write("| Sample ID | Issue | Details |\n")
                        f.write("|-----------|-------|--------|\n")
                        for sample in train_mode["problematic_samples"]:
                            f.write(f"| {sample['id']} | {sample['issue']} | {sample['details']} |\n")
            
            logger.info(f"Generated markdown report: {filename}")
        
        except Exception as e:
            logger.error(f"Error generating markdown report: {e}")


# Example usage
if __name__ == "__main__":
    import argparse
    from src.models.rna_folding_model import RNAFoldingModel
    
    parser = argparse.ArgumentParser(description="Run dual-mode validation for RNA 3D folding model")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint", default=None)
    parser.add_argument("--subset", type=str, default="technical", 
                        choices=["technical", "scientific", "comprehensive"],
                        help="Validation subset to use")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to data directory")
    parser.add_argument("--mode", type=str, default="both", 
                        choices=["both", "test", "train"],
                        help="Validation mode(s) to run")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for output files")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage (not recommended)")
    args = parser.parse_args()
    
    # Determine device
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create configuration
    config = {
        "batch_size": args.batch_size,
        "results_dir": args.output_dir,
        "verbose": True,
    }
    
    try:
        # Create default model
        if args.checkpoint is None:
            logger.info("No checkpoint provided, initializing default model")
            model = RNAFoldingModel()
        else:
            # Load model from checkpoint
            logger.info(f"Loading model from checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if "model_state_dict" in checkpoint:
                model = RNAFoldingModel()
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                logger.warning("Checkpoint doesn't contain model_state_dict, trying direct load")
                model = RNAFoldingModel()
                model.load_state_dict(checkpoint)
        
        # Run validation
        runner = ValidationRunner(model, args.data_dir, config, device)
        
        # Run validation according to selected mode
        if args.mode == "both":
            results = runner.run_validation(args.subset, run_both_modes=True)
        elif args.mode == "test":
            results = runner.run_validation(args.subset, run_both_modes=False)
        else:  # train mode
            logger.info(f"\n{'='*50}\nRunning TRAINING-EQUIVALENT mode only\n{'='*50}\n")
            results = {"train_mode": runner.run_training_equivalent_mode(args.subset)}
        
        # Log summary of results
        logger.info(f"\n{'='*50}\nValidation Complete\n{'='*50}\n")
        
        # Print key metrics
        if "test_mode" in results:
            logger.info(f"Test-Equivalent Mode - Mean RMSD: {results['test_mode'].get('mean_rmsd', 'N/A')}")
        if "train_mode" in results:
            logger.info(f"Training-Equivalent Mode - Mean RMSD: {results['train_mode'].get('mean_rmsd', 'N/A')}")
        if "analysis" in results and "conclusion" in results["analysis"]:
            conclusion = results["analysis"]["conclusion"]
            logger.info(f"Impact Analysis: {conclusion['overall_impact']} - {conclusion['severity']}")
            
    except Exception as e:
        logger.error(f"Error running validation: {e}")