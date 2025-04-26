#!/usr/bin/env python3
"""
Training Report Generator for RNA 3D Structure Prediction

This script analyzes training logs, validation metrics, and GPU usage data
to generate a comprehensive report on model training progress and performance.
"""

import os
import sys
import argparse
import logging
import json
import re
import glob
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# Add project root to Python path so we can import modules
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.utils.structure_metrics import compute_rmsd, compute_tm_score
from src.models.rna_folding_model import RNAFoldingModel
from src.data_loading import create_data_loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate training report')
    
    parser.add_argument('--training_dir', type=str, required=True,
                       help='Directory containing training results')
    parser.add_argument('--output_format', type=str, default='pdf',
                       choices=['pdf', 'html', 'md'],
                       help='Output format for the report')
    parser.add_argument('--validation_samples', type=int, default=5,
                       help='Number of validation samples to visualize')
    
    return parser.parse_args()

def parse_training_logs(training_dir: str) -> pd.DataFrame:
    """
    Parse training log files into a structured DataFrame.
    
    Args:
        training_dir: Directory containing training logs
        
    Returns:
        DataFrame with parsed training metrics
    """
    # First try to load the CSV log if it exists
    csv_log_path = os.path.join(training_dir, "training_log.csv")
    if os.path.exists(csv_log_path):
        try:
            logger.info(f"Found training_log.csv at {csv_log_path}")
            df = pd.read_csv(csv_log_path)
            return df
        except Exception as e:
            logger.warning(f"Error reading training_log.csv: {e}")
    
    # Look for log files in the training directory and logs subdirectory
    log_files = glob.glob(os.path.join(training_dir, "*.log"))
    log_files += glob.glob(os.path.join(training_dir, "logs", "*.log"))
    
    # Specifically look for training log
    training_log = os.path.join(training_dir, "logs", "training.log")
    if os.path.exists(training_log) and training_log not in log_files:
        log_files.append(training_log)
    
    if not log_files:
        logger.warning(f"No log files found in {training_dir} or {os.path.join(training_dir, 'logs')}")
        return pd.DataFrame()
    
    # Regular expressions for extracting metrics
    epoch_pattern = r"Epoch (\d+)/(\d+)"
    train_loss_pattern = r"Train loss: ([\d\.]+)"
    val_loss_pattern = r"Val loss: ([\d\.]+)"
    val_rmsd_pattern = r"Val RMSD: ([\d\.]+)"
    
    # Lists to store extracted data
    data = []
    
    # Process each log file
    for log_file in log_files:
        logger.info(f"Processing log file: {log_file}")
        try:
            with open(log_file, 'r') as f:
                log_text = f.read()
                
                # Extract all epoch entries
                entries = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - INFO - Epoch \d+/\d+", log_text)
                
                for i, timestamp in enumerate(entries):
                    # Extract the log section for this epoch
                    if i < len(entries) - 1:
                        next_timestamp = entries[i+1]
                        section = log_text[log_text.find(timestamp):log_text.find(next_timestamp)]
                    else:
                        section = log_text[log_text.find(timestamp):]
                    
                    # Extract metrics from the section
                    epoch_match = re.search(epoch_pattern, section)
                    train_loss_match = re.search(train_loss_pattern, section)
                    val_loss_match = re.search(val_loss_pattern, section)
                    val_rmsd_match = re.search(val_rmsd_pattern, section)
                    
                    if epoch_match:
                        epoch = int(epoch_match.group(1))
                        max_epochs = int(epoch_match.group(2))
                        
                        entry = {
                            'timestamp': timestamp,
                            'epoch': epoch,
                            'max_epochs': max_epochs,
                        }
                        
                        if train_loss_match:
                            entry['train_loss'] = float(train_loss_match.group(1))
                        
                        if val_loss_match:
                            entry['val_loss'] = float(val_loss_match.group(1))
                        
                        if val_rmsd_match:
                            entry['val_rmsd'] = float(val_rmsd_match.group(1))
                        
                        data.append(entry)
        except Exception as e:
            logger.warning(f"Error processing log file {log_file}: {e}")
    
    # Convert to DataFrame and sort by epoch
    if data:
        df = pd.DataFrame(data)
        df = df.sort_values('epoch')
        return df
    else:
        logger.warning("No training data extracted from logs")
        return pd.DataFrame()

def parse_gpu_metrics(training_dir: str) -> pd.DataFrame:
    """
    Parse GPU monitoring logs into a structured DataFrame.
    
    Args:
        training_dir: Directory containing training results
        
    Returns:
        DataFrame with parsed GPU metrics
    """
    # Look for GPU metrics CSV files
    metrics_dir = os.path.join(training_dir, "metrics")
    if not os.path.exists(metrics_dir):
        logger.warning(f"Metrics directory not found: {metrics_dir}")
        return pd.DataFrame()
    
    gpu_csv_files = glob.glob(os.path.join(metrics_dir, "gpu_metrics_*.csv"))
    
    if not gpu_csv_files:
        logger.warning(f"No GPU metrics files found in {metrics_dir}")
        return pd.DataFrame()
    
    # Combine all metrics files
    dfs = []
    for csv_file in gpu_csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert timestamp to datetime
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp')
    
    return combined_df

def load_validation_results(training_dir: str) -> Dict:
    """
    Load validation results from JSON or CSV files.
    
    Args:
        training_dir: Directory containing training results
        
    Returns:
        Dictionary with validation results
    """
    # First check for CSV format
    csv_validation_path = os.path.join(training_dir, "validation_results.csv")
    if os.path.exists(csv_validation_path):
        try:
            logger.info(f"Found validation_results.csv at {csv_validation_path}")
            val_df = pd.read_csv(csv_validation_path)
            
            # Convert CSV data to the expected dictionary format
            validation_results = {
                'num_samples': len(val_df),
            }
            
            # Extract RMSD values if available
            if 'rmsd' in val_df.columns:
                rmsd_values = val_df['rmsd'].dropna().tolist()
                validation_results['rmsd_values'] = rmsd_values
                validation_results['mean_rmsd'] = np.mean(rmsd_values)
                validation_results['median_rmsd'] = np.median(rmsd_values)
                validation_results['min_rmsd'] = min(rmsd_values)
                validation_results['max_rmsd'] = max(rmsd_values)
            
            # Extract TM scores if available
            if 'tm_score' in val_df.columns:
                tm_scores = val_df['tm_score'].dropna().tolist()
                validation_results['tm_scores'] = tm_scores
                validation_results['mean_tm_score'] = np.mean(tm_scores)
            
            # Extract per-target results if target_id is available
            if 'target_id' in val_df.columns:
                per_target_results = {}
                for _, row in val_df.iterrows():
                    target_id = row['target_id']
                    target_data = {}
                    
                    if 'length' in row:
                        target_data['length'] = row['length']
                    
                    if 'rmsd' in row:
                        target_data['rmsd'] = row['rmsd']
                    
                    if 'tm_score' in row:
                        target_data['tm_score'] = row['tm_score']
                    
                    per_target_results[target_id] = target_data
                
                validation_results['per_target_results'] = per_target_results
            
            return validation_results
        except Exception as e:
            logger.warning(f"Error reading validation_results.csv: {e}")
    
    # Fallback to JSON format
    validation_files = glob.glob(os.path.join(training_dir, "**/validation_results.json"), recursive=True)
    
    if not validation_files:
        logger.warning(f"No validation results found in {training_dir}")
        return {}
    
    # Use the most recent validation file
    latest_validation_file = max(validation_files, key=os.path.getmtime)
    
    try:
        with open(latest_validation_file, 'r') as f:
            validation_results = json.load(f)
        return validation_results
    except Exception as e:
        logger.error(f"Error loading validation results: {e}")
        return {}

def load_model_checkpoint(training_dir: str) -> Optional[Dict]:
    """
    Load the best model checkpoint from training results.
    
    Args:
        training_dir: Directory containing training results
        
    Returns:
        Checkpoint dictionary or None if not found
    """
    # Check for checkpoints directory
    checkpoints_dir = os.path.join(training_dir, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        logger.warning(f"Checkpoints directory not found: {checkpoints_dir}")
        return None
    
    # Look for best model or use latest if not found
    best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model_path = best_model_path
    else:
        # Find most recent checkpoint
        checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "checkpoint_epoch_*.pt"))
        if not checkpoint_files:
            logger.warning(f"No checkpoint files found in {checkpoints_dir}")
            return None
        
        model_path = max(checkpoint_files, key=os.path.getmtime)
    
    try:
        import torch
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading model checkpoint: {e}")
        return None

def generate_loss_plots(training_data: pd.DataFrame, output_dir: str):
    """
    Generate plots for training and validation losses.
    
    Args:
        training_data: DataFrame containing training metrics
        output_dir: Directory to save plots
    """
    if training_data.empty:
        logger.warning("No training data available for plotting")
        return
    
    # Create figure for loss plots
    plt.figure(figsize=(12, 6))
    
    # Plot training loss
    if 'train_loss' in training_data.columns:
        plt.plot(training_data['epoch'], training_data['train_loss'], 'b-', label='Training Loss')
    
    # Plot validation loss
    if 'val_loss' in training_data.columns:
        plt.plot(training_data['epoch'], training_data['val_loss'], 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save figure
    loss_plot_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Loss plot saved to {loss_plot_path}")
    
    # Create figure for RMSD plot
    if 'val_rmsd' in training_data.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(training_data['epoch'], training_data['val_rmsd'], 'g-', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('RMSD (Å)')
        plt.title('Validation RMSD Over Training')
        plt.grid(alpha=0.3)
        
        # Add horizontal line at 15Å (threshold for reasonable structures)
        if min(training_data['val_rmsd']) < 30:  # Only if we have reasonable values
            plt.axhline(y=15.0, color='r', linestyle='--', alpha=0.7, 
                       label='15Å threshold (reasonable structures)')
            plt.legend()
        
        # Save figure
        rmsd_plot_path = os.path.join(output_dir, "rmsd_over_training.png")
        plt.savefig(rmsd_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"RMSD plot saved to {rmsd_plot_path}")

def generate_gpu_metrics_plots(gpu_data: pd.DataFrame, output_dir: str):
    """
    Generate plots for GPU monitoring metrics.
    
    Args:
        gpu_data: DataFrame containing GPU metrics
        output_dir: Directory to save plots
    """
    if gpu_data.empty:
        logger.warning("No GPU data available for plotting")
        return
    
    # Convert elapsed time to hours for x-axis
    if 'elapsed_time_s' in gpu_data.columns:
        gpu_data['elapsed_time_h'] = gpu_data['elapsed_time_s'].astype(float) / 3600.0
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    # Get unique GPU IDs
    gpu_ids = gpu_data['gpu_id'].unique()
    
    # 1. GPU Utilization
    ax1 = fig.add_subplot(gs[0, 0])
    for gpu_id in gpu_ids:
        gpu_subset = gpu_data[gpu_data['gpu_id'] == gpu_id]
        ax1.plot(gpu_subset['elapsed_time_h'], gpu_subset['gpu_utilization_percent'], 
                label=f'GPU {gpu_id}')
    
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('GPU Utilization %')
    ax1.set_title('GPU Utilization During Training')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # 2. Memory Usage
    ax2 = fig.add_subplot(gs[0, 1])
    for gpu_id in gpu_ids:
        gpu_subset = gpu_data[gpu_data['gpu_id'] == gpu_id]
        ax2.plot(gpu_subset['elapsed_time_h'], gpu_subset['memory_used_percent'], 
                label=f'GPU {gpu_id}')
    
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Memory Usage %')
    ax2.set_title('GPU Memory Usage During Training')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    # 3. Temperature
    ax3 = fig.add_subplot(gs[1, 0])
    for gpu_id in gpu_ids:
        gpu_subset = gpu_data[gpu_data['gpu_id'] == gpu_id]
        ax3.plot(gpu_subset['elapsed_time_h'], gpu_subset['temperature_c'], 
                label=f'GPU {gpu_id}')
    
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_title('GPU Temperature During Training')
    ax3.grid(alpha=0.3)
    ax3.legend()
    
    # 4. Memory Usage (MB)
    ax4 = fig.add_subplot(gs[1, 1])
    for gpu_id in gpu_ids:
        gpu_subset = gpu_data[gpu_data['gpu_id'] == gpu_id]
        ax4.plot(gpu_subset['elapsed_time_h'], gpu_subset['memory_used_mb'], 
                label=f'GPU {gpu_id}')
    
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Memory Used (MB)')
    ax4.set_title('GPU Memory Usage (MB) During Training')
    ax4.grid(alpha=0.3)
    ax4.legend()
    
    # 5. Power Usage
    if 'power_draw_w' in gpu_data.columns:
        ax5 = fig.add_subplot(gs[2, 0])
        for gpu_id in gpu_ids:
            gpu_subset = gpu_data[gpu_data['gpu_id'] == gpu_id]
            # Filter out non-numeric values
            power_subset = gpu_subset[gpu_subset['power_draw_w'] != 'N/A']
            if not power_subset.empty:
                ax5.plot(power_subset['elapsed_time_h'], 
                        power_subset['power_draw_w'].astype(float), 
                        label=f'GPU {gpu_id}')
        
        ax5.set_xlabel('Time (hours)')
        ax5.set_ylabel('Power Draw (W)')
        ax5.set_title('GPU Power Consumption During Training')
        ax5.grid(alpha=0.3)
        ax5.legend()
    
    # 6. Utilization Histogram
    ax6 = fig.add_subplot(gs[2, 1])
    for gpu_id in gpu_ids:
        gpu_subset = gpu_data[gpu_data['gpu_id'] == gpu_id]
        sns.histplot(gpu_subset['gpu_utilization_percent'], 
                    bins=20, alpha=0.5, label=f'GPU {gpu_id}', ax=ax6)
    
    ax6.set_xlabel('GPU Utilization %')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distribution of GPU Utilization')
    ax6.legend()
    
    plt.tight_layout()
    
    # Save figure
    gpu_plot_path = os.path.join(output_dir, "gpu_metrics.png")
    plt.savefig(gpu_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"GPU metrics plots saved to {gpu_plot_path}")

def generate_validation_plots(validation_results: Dict, output_dir: str):
    """
    Generate plots from validation results.
    
    Args:
        validation_results: Dictionary with validation metrics
        output_dir: Directory to save plots
    """
    if not validation_results:
        logger.warning("No validation results available for plotting")
        return
    
    # 1. RMSD Distribution
    if 'rmsd_values' in validation_results:
        plt.figure(figsize=(10, 6))
        rmsd_values = validation_results['rmsd_values']
        
        if rmsd_values:
            plt.hist(rmsd_values, bins=min(20, len(rmsd_values)), alpha=0.7)
            plt.axvline(validation_results.get('mean_rmsd', 0), color='r', linestyle='--',
                      label=f"Mean: {validation_results.get('mean_rmsd', 0):.2f} Å")
            plt.axvline(validation_results.get('median_rmsd', 0), color='g', linestyle='--',
                      label=f"Median: {validation_results.get('median_rmsd', 0):.2f} Å")
        
        plt.xlabel('RMSD (Å)')
        plt.ylabel('Count')
        plt.title('Distribution of RMSD Values')
        plt.legend()
        plt.grid(alpha=0.3)
        
        rmsd_dist_path = os.path.join(output_dir, "rmsd_distribution.png")
        plt.savefig(rmsd_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"RMSD distribution plot saved to {rmsd_dist_path}")
    
    # 2. TM-Score Distribution
    if 'tm_scores' in validation_results:
        plt.figure(figsize=(10, 6))
        tm_scores = validation_results['tm_scores']
        
        if tm_scores:
            # Filter out NaN values
            tm_scores = [score for score in tm_scores if not np.isnan(score)]
            if tm_scores:
                plt.hist(tm_scores, bins=min(20, len(tm_scores)), alpha=0.7)
                plt.axvline(np.mean(tm_scores), color='r', linestyle='--',
                          label=f"Mean: {np.mean(tm_scores):.4f}")
        
        plt.xlabel('TM-Score')
        plt.ylabel('Count')
        plt.title('Distribution of TM-Scores')
        plt.legend()
        plt.grid(alpha=0.3)
        
        tm_dist_path = os.path.join(output_dir, "tm_score_distribution.png")
        plt.savefig(tm_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"TM-score distribution plot saved to {tm_dist_path}")
    
    # 3. RMSD vs Sequence Length
    if 'per_target_results' in validation_results:
        plt.figure(figsize=(10, 6))
        
        per_target = validation_results['per_target_results']
        lengths = []
        rmsds = []
        
        for target_id, result in per_target.items():
            if 'length' in result and 'rmsd' in result:
                lengths.append(result['length'])
                rmsds.append(result['rmsd'])
        
        if lengths and rmsds:
            plt.scatter(lengths, rmsds, alpha=0.7)
            
            # Try to fit a trend line
            if len(lengths) >= 2:
                try:
                    z = np.polyfit(lengths, rmsds, 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(lengths), p(sorted(lengths)), "r--", alpha=0.7)
                except Exception as e:
                    logger.warning(f"Could not fit trend line: {e}")
        
        plt.xlabel('Sequence Length')
        plt.ylabel('RMSD (Å)')
        plt.title('RMSD vs Sequence Length')
        plt.grid(alpha=0.3)
        
        length_plot_path = os.path.join(output_dir, "rmsd_vs_length.png")
        plt.savefig(length_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"RMSD vs Length plot saved to {length_plot_path}")

def generate_markdown_report(
    training_dir: str,
    training_data: pd.DataFrame,
    gpu_data: pd.DataFrame,
    validation_results: Dict,
    checkpoint: Optional[Dict]
) -> str:
    """
    Generate a comprehensive training report in Markdown format.
    
    Args:
        training_dir: Directory containing training results
        training_data: DataFrame with training metrics
        gpu_data: DataFrame with GPU metrics
        validation_results: Dictionary with validation metrics
        checkpoint: Model checkpoint dictionary
        
    Returns:
        Markdown report as a string
    """
    # Initialize report
    report = []
    
    # Add title and metadata
    report.append("# RNA 3D Structure Prediction Model Training Report")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Training directory**: `{training_dir}`")
    report.append("")
    
    # Add training summary
    report.append("## Training Summary")
    
    if checkpoint and 'epoch' in checkpoint:
        report.append(f"**Training duration**: {checkpoint['epoch'] + 1} epochs")
    elif not training_data.empty:
        report.append(f"**Training duration**: {training_data['epoch'].max()} epochs")
    
    # Add training parameters if available
    if checkpoint and 'model_config' in checkpoint:
        report.append("")
        report.append("### Model Configuration")
        report.append("```json")
        report.append(json.dumps(checkpoint['model_config'], indent=2))
        report.append("```")
    
    if checkpoint and 'loss_weights' in checkpoint:
        report.append("")
        report.append("### Loss Weights")
        report.append("```json")
        report.append(json.dumps(checkpoint['loss_weights'], indent=2))
        report.append("```")
    
    # Add performance metrics
    report.append("")
    report.append("## Performance Metrics")
    
    if not training_data.empty:
        # Final training loss
        report.append("")
        report.append("### Training Metrics")
        report.append("")
        report.append("| Metric | Initial | Final | Best |")
        report.append("|--------|---------|-------|------|")
        
        # Training loss
        if 'train_loss' in training_data.columns:
            initial_loss = training_data['train_loss'].iloc[0]
            final_loss = training_data['train_loss'].iloc[-1]
            best_loss = training_data['train_loss'].min()
            report.append(f"| Training Loss | {initial_loss:.4f} | {final_loss:.4f} | {best_loss:.4f} |")
        
        # Validation loss
        if 'val_loss' in training_data.columns:
            initial_val_loss = training_data['val_loss'].iloc[0] if not np.isnan(training_data['val_loss'].iloc[0]) else "N/A"
            final_val_loss = training_data['val_loss'].iloc[-1] if not np.isnan(training_data['val_loss'].iloc[-1]) else "N/A"
            
            if np.isnan(training_data['val_loss']).all():
                best_val_loss = "N/A"
            else:
                best_val_loss = training_data['val_loss'].min()
                
            report.append(f"| Validation Loss | {initial_val_loss} | {final_val_loss} | {best_val_loss} |")
        
        # Validation RMSD
        if 'val_rmsd' in training_data.columns:
            initial_rmsd = training_data['val_rmsd'].iloc[0] if not np.isnan(training_data['val_rmsd'].iloc[0]) else "N/A"
            final_rmsd = training_data['val_rmsd'].iloc[-1] if not np.isnan(training_data['val_rmsd'].iloc[-1]) else "N/A"
            
            if np.isnan(training_data['val_rmsd']).all():
                best_rmsd = "N/A"
            else:
                best_rmsd = training_data['val_rmsd'].min()
                
            report.append(f"| Validation RMSD (Å) | {initial_rmsd} | {final_rmsd} | {best_rmsd} |")
    
    # Add validation results summary
    if validation_results:
        report.append("")
        report.append("### Validation Results Summary")
        report.append("")
        
        if 'num_samples' in validation_results:
            report.append(f"**Number of validation samples**: {validation_results['num_samples']}")
        
        if 'mean_rmsd' in validation_results:
            report.append(f"**Mean RMSD**: {validation_results['mean_rmsd']:.4f} Å")
        
        if 'median_rmsd' in validation_results:
            report.append(f"**Median RMSD**: {validation_results['median_rmsd']:.4f} Å")
        
        if 'min_rmsd' in validation_results and 'max_rmsd' in validation_results:
            report.append(f"**RMSD Range**: {validation_results['min_rmsd']:.4f} - {validation_results['max_rmsd']:.4f} Å")
        
        if 'mean_tm_score' in validation_results:
            report.append(f"**Mean TM-Score**: {validation_results['mean_tm_score']:.4f}")
    
    # Add visualization references
    report.append("")
    report.append("## Visualizations")
    
    # Check if plots exist
    loss_plot = os.path.join(training_dir, "loss_curves.png")
    rmsd_plot = os.path.join(training_dir, "rmsd_over_training.png")
    gpu_plot = os.path.join(training_dir, "gpu_metrics.png")
    rmsd_dist_plot = os.path.join(training_dir, "rmsd_distribution.png")
    
    if os.path.exists(loss_plot):
        report.append("")
        report.append("### Loss Curves")
        report.append("")
        report.append(f"![Loss Curves](loss_curves.png)")
    
    if os.path.exists(rmsd_plot):
        report.append("")
        report.append("### RMSD Over Training")
        report.append("")
        report.append(f"![RMSD Over Training](rmsd_over_training.png)")
    
    if os.path.exists(gpu_plot):
        report.append("")
        report.append("### GPU Metrics")
        report.append("")
        report.append(f"![GPU Metrics](gpu_metrics.png)")
    
    if os.path.exists(rmsd_dist_plot):
        report.append("")
        report.append("### RMSD Distribution")
        report.append("")
        report.append(f"![RMSD Distribution](rmsd_distribution.png)")
    
    # Add resource utilization summary
    if not gpu_data.empty:
        report.append("")
        report.append("## Resource Utilization")
        report.append("")
        
        # Calculate training duration
        if 'elapsed_time_s' in gpu_data.columns:
            max_elapsed_time = gpu_data['elapsed_time_s'].astype(float).max()
            hours, remainder = divmod(max_elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            report.append(f"**Total training time**: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # GPU utilization stats
        if 'gpu_utilization_percent' in gpu_data.columns:
            report.append("")
            report.append("### GPU Utilization")
            report.append("")
            report.append("| GPU ID | Mean Utilization | Peak Utilization | Mean Memory Usage | Peak Memory Usage | Mean Temperature |")
            report.append("|--------|-----------------|------------------|-------------------|------------------|------------------|")
            
            for gpu_id in gpu_data['gpu_id'].unique():
                gpu_subset = gpu_data[gpu_data['gpu_id'] == gpu_id]
                
                mean_util = gpu_subset['gpu_utilization_percent'].mean()
                peak_util = gpu_subset['gpu_utilization_percent'].max()
                mean_mem = gpu_subset['memory_used_percent'].mean() if 'memory_used_percent' in gpu_subset.columns else np.nan
                peak_mem = gpu_subset['memory_used_percent'].max() if 'memory_used_percent' in gpu_subset.columns else np.nan
                mean_temp = gpu_subset['temperature_c'].mean() if 'temperature_c' in gpu_subset.columns else np.nan
                
                report.append(f"| {gpu_id} | {mean_util:.2f}% | {peak_util:.2f}% | {mean_mem:.2f}% | {peak_mem:.2f}% | {mean_temp:.1f}°C |")
    
    # Add conclusions and recommendations
    report.append("")
    report.append("## Analysis and Recommendations")
    
    # Training convergence analysis
    if not training_data.empty and 'train_loss' in training_data.columns:
        report.append("")
        report.append("### Training Convergence")
        
        # Check if loss is still decreasing
        if len(training_data) >= 5:
            recent_train_loss = training_data['train_loss'].iloc[-5:].values
            is_decreasing = all(recent_train_loss[i] > recent_train_loss[i+1] for i in range(len(recent_train_loss)-1))
            
            if is_decreasing:
                report.append("✅ **Training loss is still decreasing**: The model may benefit from additional training epochs.")
            else:
                report.append("⚠️ **Training loss has plateaued**: The model has likely reached convergence.")
        
        # Check for overfitting
        if 'val_loss' in training_data.columns:
            train_losses = training_data['train_loss'].values
            val_losses = training_data['val_loss'].values
            
            # Remove NaN values
            valid_indices = ~np.isnan(val_losses)
            if valid_indices.any():
                train_losses = train_losses[valid_indices]
                val_losses = val_losses[valid_indices]
                
                train_val_gap = (val_losses - train_losses) / train_losses
                if len(train_val_gap) >= 5:
                    recent_gap = train_val_gap[-5:]
                    if np.mean(recent_gap) > 0.3 and np.all(np.diff(recent_gap) > 0):
                        report.append("⚠️ **Possible overfitting detected**: Validation loss is significantly higher than training loss and the gap is increasing.")
    
    # RMSD analysis
    if validation_results and 'mean_rmsd' in validation_results:
        report.append("")
        report.append("### Structure Quality Analysis")
        
        mean_rmsd = validation_results['mean_rmsd']
        if mean_rmsd < 10.0:
            report.append("✅ **Good structure quality**: Mean RMSD below 10Å indicates good structure predictions.")
        elif mean_rmsd < 15.0:
            report.append("✓ **Reasonable structure quality**: Mean RMSD below 15Å indicates the model is learning meaningful structure features.")
        else:
            report.append("⚠️ **Poor structure quality**: Mean RMSD above 15Å indicates the model needs significant improvement.")
    
    # Recommendations
    report.append("")
    report.append("### Recommendations for Improvement")
    report.append("")
    
    # Add generic recommendations
    recommendations = [
        "**Increase training data**: If possible, expand the dataset with more diverse RNA structures.",
        "**Experiment with learning rate**: Try different learning rate schedules to improve convergence.",
        "**Adjust loss weights**: Fine-tune the balance between FAPE, confidence, and angle losses based on validation metrics.",
        "**Model architecture tuning**: Adjust the number of transformer blocks, embedding dimensions, or attention heads."
    ]
    
    for rec in recommendations:
        report.append(f"- {rec}")
    
    # Join all report lines
    return "\n".join(report)

def export_report(report_text: str, training_dir: str, output_format: str = 'pdf'):
    """
    Export the training report in the specified format.
    
    Args:
        report_text: Markdown report text
        training_dir: Directory containing training results
        output_format: Output format ('pdf', 'html', or 'md')
    """
    # Save markdown report
    report_md_path = os.path.join(training_dir, "training_report.md")
    with open(report_md_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Markdown report saved to {report_md_path}")
    
    # Export to other formats if requested
    if output_format == 'pdf':
        try:
            # Check if pandoc is available
            subprocess.run(["pandoc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Convert markdown to PDF
            pdf_path = os.path.join(training_dir, "training_report.pdf")
            subprocess.run([
                "pandoc", 
                report_md_path, 
                "-o", pdf_path,
                "--pdf-engine=xelatex",
                "-V", "geometry:margin=1in"
            ], check=True)
            
            logger.info(f"PDF report saved to {pdf_path}")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Pandoc not available. Skipping PDF export.")
    
    elif output_format == 'html':
        try:
            # Check if pandoc is available
            subprocess.run(["pandoc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Convert markdown to HTML
            html_path = os.path.join(training_dir, "training_report.html")
            subprocess.run([
                "pandoc", 
                report_md_path, 
                "-o", html_path,
                "--self-contained",
                "--css=https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
            ], check=True)
            
            logger.info(f"HTML report saved to {html_path}")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("Pandoc not available. Skipping HTML export.")

def main():
    """Main function to generate the training report."""
    args = parse_args()
    
    # Parse training data from logs
    logger.info(f"Analyzing training logs from {args.training_dir}")
    training_data = parse_training_logs(args.training_dir)
    
    # Parse GPU metrics
    logger.info("Analyzing GPU metrics")
    gpu_data = parse_gpu_metrics(args.training_dir)
    
    # Load validation results
    logger.info("Loading validation results")
    validation_results = load_validation_results(args.training_dir)
    
    # Load model checkpoint
    logger.info("Loading model checkpoint")
    checkpoint = load_model_checkpoint(args.training_dir)
    
    # Generate plots
    logger.info("Generating visualizations")
    generate_loss_plots(training_data, args.training_dir)
    generate_gpu_metrics_plots(gpu_data, args.training_dir)
    generate_validation_plots(validation_results, args.training_dir)
    
    # Generate markdown report
    logger.info("Generating training report")
    report_text = generate_markdown_report(
        args.training_dir,
        training_data,
        gpu_data,
        validation_results,
        checkpoint
    )
    
    # Export report
    logger.info(f"Exporting report in {args.output_format} format")
    export_report(report_text, args.training_dir, args.output_format)
    
    logger.info("Report generation complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())