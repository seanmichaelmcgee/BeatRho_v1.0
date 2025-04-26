#!/usr/bin/env python3
"""
Validation Comparison Report Generator

This script compares validation results between different training runs to help track
improvements across model iterations. It creates a comprehensive summary report showing
metrics side-by-side and generates visualizations for easy comparison.
"""

import os
import sys
import argparse
import logging
import json
import glob
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate validation comparison report')
    
    parser.add_argument('--run_dirs', type=str, nargs='+', required=True,
                       help='List of run directories to compare')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='Labels for each run (defaults to directory names)')
    parser.add_argument('--output_dir', type=str, default='notebook_reports',
                       help='Output directory for the comparison report')
    parser.add_argument('--report_name', type=str, default='validation_comparison',
                       help='Name of the report file (without extension)')
    
    return parser.parse_args()

def load_validation_results(run_dir: str) -> Dict:
    """
    Load validation results from a run directory.
    
    Args:
        run_dir: Training run directory path
        
    Returns:
        Dictionary with validation metrics
    """
    # First check for validation_results.csv
    csv_path = os.path.join(run_dir, "validation_results.csv")
    
    # Also check subdirectories like run_YYYYMMDD-HHMMSS/
    if not os.path.exists(csv_path):
        run_subdirs = glob.glob(os.path.join(run_dir, "run_*"))
        if run_subdirs:
            latest_run = max(run_subdirs, key=os.path.getmtime)
            csv_path = os.path.join(latest_run, "validation_results.csv")
    
    if os.path.exists(csv_path):
        try:
            logger.info(f"Found validation_results.csv at {csv_path}")
            val_df = pd.read_csv(csv_path)
            
            # Calculate metrics
            results = {
                'csv_path': csv_path,
                'num_epochs': len(val_df),
                'final_epoch': val_df['epoch'].max() if 'epoch' in val_df.columns else None,
            }
            
            # Get the last epoch's metrics as the final results
            last_row = val_df.iloc[-1]
            
            # Add all metrics from the last row
            for col in val_df.columns:
                if col != 'epoch':
                    results[f'final_{col}'] = last_row[col]
            
            # Calculate improvements over time
            if len(val_df) > 1:
                first_row = val_df.iloc[0]
                
                for col in val_df.columns:
                    if col != 'epoch':
                        # Ensure values are numeric
                        try:
                            first_val = float(first_row[col])
                            last_val = float(last_row[col])
                            
                            # Calculate absolute change
                            abs_change = last_val - first_val
                            results[f'{col}_abs_change'] = abs_change
                            
                            # Calculate percentage change
                            if first_val != 0:
                                pct_change = (abs_change / first_val) * 100
                                results[f'{col}_pct_change'] = pct_change
                        except (ValueError, TypeError):
                            # Skip non-numeric columns
                            logger.warning(f"Skipping non-numeric column: {col}")
            
            # Store the full dataframe
            results['data'] = val_df
            
            return results
        except Exception as e:
            logger.warning(f"Error reading validation results from {csv_path}: {e}")
    
    # Fallback to looking for validation_results.json
    json_files = glob.glob(os.path.join(run_dir, "**/validation_results.json"), recursive=True)
    
    if json_files:
        try:
            # Use the most recent file
            latest_json = max(json_files, key=os.path.getmtime)
            logger.info(f"Found validation_results.json at {latest_json}")
            
            with open(latest_json, 'r') as f:
                results = json.load(f)
            
            # Add file path for reference
            results['json_path'] = latest_json
            
            return results
        except Exception as e:
            logger.warning(f"Error reading validation results from {json_files}: {e}")
    
    logger.warning(f"No validation results found in {run_dir}")
    return {}

def extract_run_info(run_dir: str) -> Dict:
    """
    Extract metadata about a training run.
    
    Args:
        run_dir: Training run directory path
        
    Returns:
        Dictionary with run metadata
    """
    run_info = {
        'dir': run_dir,
        'name': os.path.basename(run_dir),
    }
    
    # Extract timestamp from directory name if available
    timestamp_match = re.search(r'(\d{8}-\d{6})', run_dir)
    if timestamp_match:
        timestamp_str = timestamp_match.group(1)
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d-%H%M%S')
            run_info['timestamp'] = timestamp
        except ValueError:
            pass
    
    # Look for config.json
    config_path = os.path.join(run_dir, "config.json")
    
    # Also check subdirectories like run_YYYYMMDD-HHMMSS/
    if not os.path.exists(config_path):
        run_subdirs = glob.glob(os.path.join(run_dir, "run_*"))
        if run_subdirs:
            latest_run = max(run_subdirs, key=os.path.getmtime)
            config_path = os.path.join(latest_run, "config.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract relevant configuration parameters
            run_info['config'] = config
            
            if 'num_blocks' in config:
                run_info['num_blocks'] = config['num_blocks']
            
            if 'residue_embed_dim' in config:
                run_info['embed_dim'] = config['residue_embed_dim']
            
            if 'batch_size' in config and 'grad_accum_steps' in config:
                run_info['effective_batch_size'] = config['batch_size'] * config['grad_accum_steps']
            
            if 'lr' in config:
                run_info['learning_rate'] = config['lr']
        except Exception as e:
            logger.warning(f"Error reading config.json from {config_path}: {e}")
    
    return run_info

def get_run_label(run_info: Dict, provided_label: Optional[str] = None) -> str:
    """
    Generate a descriptive label for a run.
    
    Args:
        run_info: Dictionary with run metadata
        provided_label: User-provided label (if any)
        
    Returns:
        Run label string
    """
    if provided_label:
        return provided_label
    
    # Try to create an informative default label
    label_parts = []
    
    # Add run name or directory
    label_parts.append(run_info['name'])
    
    # Add model architecture info if available
    if 'num_blocks' in run_info and 'embed_dim' in run_info:
        label_parts.append(f"{run_info['num_blocks']}b-{run_info['embed_dim']}d")
    
    # Add batch size if available
    if 'effective_batch_size' in run_info:
        label_parts.append(f"bs{run_info['effective_batch_size']}")
    
    # Add learning rate if available
    if 'learning_rate' in run_info:
        label_parts.append(f"lr{run_info['learning_rate']}")
    
    return " ".join(label_parts)

def generate_metrics_table(run_results: List[Dict], run_labels: List[str]) -> pd.DataFrame:
    """
    Generate a metrics comparison table.
    
    Args:
        run_results: List of validation results dictionaries
        run_labels: List of labels for each run
        
    Returns:
        DataFrame with metrics comparison
    """
    # Define the metrics to include
    metrics = [
        'final_loss', 
        'final_rmsd', 
        'final_tm_score',
        'loss_abs_change',
        'rmsd_abs_change',
        'tm_score_abs_change',
        'loss_pct_change',
        'rmsd_pct_change',
        'tm_score_pct_change',
    ]
    
    # Create empty dataframe
    df = pd.DataFrame(index=run_labels)
    
    # Fill in metrics for each run
    for i, (results, label) in enumerate(zip(run_results, run_labels)):
        for metric in metrics:
            if metric in results:
                df.loc[label, metric] = results[metric]
    
    # Add units and friendlier names
    df.columns = [
        'Final Loss',
        'Final RMSD (Å)',
        'Final TM-score',
        'Loss Change',
        'RMSD Change (Å)',
        'TM-score Change',
        'Loss Change (%)',
        'RMSD Change (%)',
        'TM-score Change (%)',
    ]
    
    return df

def generate_comparison_plots(run_results: List[Dict], run_labels: List[str], output_dir: str):
    """
    Generate comparison plots for validation metrics.
    
    Args:
        run_results: List of validation results dictionaries
        run_labels: List of labels for each run
        output_dir: Directory to save plots
    """
    # 1. RMSD comparison bar chart
    plt.figure(figsize=(10, 6))
    
    rmsd_values = []
    for results in run_results:
        if 'final_rmsd' in results:
            rmsd_values.append(results['final_rmsd'])
        else:
            rmsd_values.append(np.nan)
    
    bars = plt.bar(run_labels, rmsd_values, alpha=0.7)
    
    # Add value labels on top of bars
    for i, v in enumerate(rmsd_values):
        if not np.isnan(v):
            plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.xlabel('Training Run')
    plt.ylabel('Final RMSD (Å)')
    plt.title('RMSD Comparison Across Runs')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    rmsd_plot_path = os.path.join(output_dir, "rmsd_comparison.png")
    plt.savefig(rmsd_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. TM-score comparison bar chart
    plt.figure(figsize=(10, 6))
    
    tm_values = []
    for results in run_results:
        if 'final_tm_score' in results:
            tm_values.append(results['final_tm_score'])
        else:
            tm_values.append(np.nan)
    
    bars = plt.bar(run_labels, tm_values, alpha=0.7, color='green')
    
    # Add value labels on top of bars
    for i, v in enumerate(tm_values):
        if not np.isnan(v):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.xlabel('Training Run')
    plt.ylabel('Final TM-score')
    plt.title('TM-score Comparison Across Runs')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(1.0, max([x for x in tm_values if not np.isnan(x)]) * 1.2))
    plt.tight_layout()
    
    # Save figure
    tm_plot_path = os.path.join(output_dir, "tm_score_comparison.png")
    plt.savefig(tm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Loss trajectory comparison
    plt.figure(figsize=(12, 6))
    
    for i, (results, label) in enumerate(zip(run_results, run_labels)):
        if 'data' in results and 'epoch' in results['data'].columns and 'loss' in results['data'].columns:
            data = results['data']
            plt.plot(data['epoch'], data['loss'], marker='o', alpha=0.7, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Progression Across Runs')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    loss_plot_path = os.path.join(output_dir, "loss_progression_comparison.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. RMSD trajectory comparison
    plt.figure(figsize=(12, 6))
    
    for i, (results, label) in enumerate(zip(run_results, run_labels)):
        if 'data' in results and 'epoch' in results['data'].columns and 'rmsd' in results['data'].columns:
            data = results['data']
            plt.plot(data['epoch'], data['rmsd'], marker='o', alpha=0.7, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('RMSD (Å)')
    plt.title('RMSD Progression Across Runs')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    rmsd_traj_path = os.path.join(output_dir, "rmsd_progression_comparison.png")
    plt.savefig(rmsd_traj_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. TM-score trajectory comparison
    plt.figure(figsize=(12, 6))
    
    for i, (results, label) in enumerate(zip(run_results, run_labels)):
        if 'data' in results and 'epoch' in results['data'].columns and 'tm_score' in results['data'].columns:
            data = results['data']
            plt.plot(data['epoch'], data['tm_score'], marker='o', alpha=0.7, label=label)
    
    plt.xlabel('Epoch')
    plt.ylabel('TM-score')
    plt.title('TM-score Progression Across Runs')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    tm_traj_path = os.path.join(output_dir, "tm_score_progression_comparison.png")
    plt.savefig(tm_traj_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_markdown_report(
    run_info_list: List[Dict],
    run_results: List[Dict],
    run_labels: List[str],
    metrics_table: pd.DataFrame,
    output_dir: str,
    report_name: str,
) -> str:
    """
    Generate a markdown report comparing validation results.
    
    Args:
        run_info_list: List of run metadata dictionaries
        run_results: List of validation results dictionaries
        run_labels: List of labels for each run
        metrics_table: DataFrame with metrics comparison
        output_dir: Directory to save the report
        report_name: Base name for the report file
        
    Returns:
        Path to the saved report
    """
    # Format the current date and time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Initialize report
    report = []
    
    # Add title and metadata
    report.append("# RNA 3D Structure Prediction: Validation Comparison Report")
    report.append("")
    report.append(f"**Generated**: {now}")
    report.append(f"**Comparing**: {len(run_labels)} training runs")
    report.append("")
    
    # Add summary of runs
    report.append("## Training Runs Overview")
    report.append("")
    report.append("| Run Label | Directory | Blocks | Embedding Dim | Batch Size | Learning Rate |")
    report.append("|-----------|-----------|--------|---------------|------------|---------------|")
    
    for info, label in zip(run_info_list, run_labels):
        blocks = info.get('num_blocks', 'N/A')
        embed_dim = info.get('embed_dim', 'N/A')
        batch_size = info.get('effective_batch_size', 'N/A')
        lr = info.get('learning_rate', 'N/A')
        
        report.append(f"| {label} | {info['dir']} | {blocks} | {embed_dim} | {batch_size} | {lr} |")
    
    report.append("")
    
    # Add metrics comparison
    report.append("## Validation Metrics Comparison")
    report.append("")
    
    # Convert DataFrame to markdown table manually (no dependency on tabulate)
    df = metrics_table.reset_index()
    
    # Add header row
    header = "| " + " | ".join(df.columns) + " |"
    report.append(header)
    
    # Add separator row
    separator = "| " + " | ".join(["---" for _ in df.columns]) + " |"
    report.append(separator)
    
    # Add data rows
    for _, row in df.iterrows():
        # Format numbers with proper precision
        formatted_row = []
        for i, val in enumerate(row):
            if pd.isna(val):
                formatted_row.append("N/A")
            elif isinstance(val, (int, float)):
                # Format based on column type
                col_name = df.columns[i].lower()
                if "tm_score" in col_name:
                    formatted_row.append(f"{val:.4f}")
                elif "rmsd" in col_name:
                    formatted_row.append(f"{val:.4f}")
                elif "loss" in col_name:
                    formatted_row.append(f"{val:.4f}")
                elif "pct" in col_name or "percent" in col_name:
                    formatted_row.append(f"{val:.2f}%")
                else:
                    formatted_row.append(f"{val}")
            else:
                formatted_row.append(str(val))
                
        data_row = "| " + " | ".join(formatted_row) + " |"
        report.append(data_row)
    report.append("")
    
    # Add visualizations
    report.append("## Visual Comparisons")
    
    report.append("")
    report.append("### Final RMSD Comparison")
    report.append("")
    report.append("![RMSD Comparison](rmsd_comparison.png)")
    
    report.append("")
    report.append("### Final TM-score Comparison")
    report.append("")
    report.append("![TM-score Comparison](tm_score_comparison.png)")
    
    report.append("")
    report.append("### Loss Progression")
    report.append("")
    report.append("![Loss Progression](loss_progression_comparison.png)")
    
    report.append("")
    report.append("### RMSD Progression")
    report.append("")
    report.append("![RMSD Progression](rmsd_progression_comparison.png)")
    
    report.append("")
    report.append("### TM-score Progression")
    report.append("")
    report.append("![TM-score Progression](tm_score_progression_comparison.png)")
    
    # Add analysis and findings
    report.append("")
    report.append("## Analysis and Findings")
    report.append("")
    
    # Find the best run for each metric
    best_loss_idx = metrics_table['Final Loss'].idxmin()
    best_rmsd_idx = metrics_table['Final RMSD (Å)'].idxmin()
    best_tm_idx = metrics_table['Final TM-score'].idxmax()
    
    # Add best runs summary
    if not pd.isna(best_loss_idx):
        report.append(f"- **Best Loss**: {best_loss_idx} with {metrics_table.loc[best_loss_idx, 'Final Loss']:.4f}")
    
    if not pd.isna(best_rmsd_idx):
        report.append(f"- **Best RMSD**: {best_rmsd_idx} with {metrics_table.loc[best_rmsd_idx, 'Final RMSD (Å)']:.4f} Å")
    
    if not pd.isna(best_tm_idx):
        report.append(f"- **Best TM-score**: {best_tm_idx} with {metrics_table.loc[best_tm_idx, 'Final TM-score']:.4f}")
    
    report.append("")
    
    # Analyze improvement trends
    report.append("### Improvement Trends")
    report.append("")
    
    if 'RMSD Change (%)' in metrics_table.columns and not metrics_table['RMSD Change (%)'].isna().all():
        best_rmsd_improvement_idx = metrics_table['RMSD Change (%)'].idxmin()
        best_rmsd_improvement = metrics_table.loc[best_rmsd_improvement_idx, 'RMSD Change (%)']
        
        if best_rmsd_improvement < 0:
            report.append(f"- **Best RMSD Improvement**: {best_rmsd_improvement_idx} with {abs(best_rmsd_improvement):.2f}% reduction")
    
    if 'TM-score Change (%)' in metrics_table.columns and not metrics_table['TM-score Change (%)'].isna().all():
        best_tm_improvement_idx = metrics_table['TM-score Change (%)'].idxmax()
        best_tm_improvement = metrics_table.loc[best_tm_improvement_idx, 'TM-score Change (%)']
        
        if best_tm_improvement > 0:
            report.append(f"- **Best TM-score Improvement**: {best_tm_improvement_idx} with {best_tm_improvement:.2f}% increase")
    
    report.append("")
    
    # Add recommendations
    report.append("### Recommendations")
    report.append("")
    report.append("Based on this comparison:")
    report.append("")
    
    # Add generic recommendations
    recommendations = [
        "- Look at both TM-score and RMSD to evaluate structure quality",
        "- Consider using the best-performing model configuration as a baseline for further improvements",
        "- For models with the same final metrics, prefer the one that converges faster",
        "- Check for overfitting by comparing early vs. late epochs"
    ]
    
    for rec in recommendations:
        report.append(rec)
    
    # Generate detailed observations
    mean_rmsd = metrics_table['Final RMSD (Å)'].mean()
    mean_tm = metrics_table['Final TM-score'].mean()
    
    if not pd.isna(mean_rmsd) and not pd.isna(mean_tm):
        report.append("")
        report.append(f"The average RMSD across all runs is {mean_rmsd:.4f} Å, and the average TM-score is {mean_tm:.4f}.")
    
    # Join all report lines
    report_text = "\n".join(report)
    
    # Save the report
    report_path = os.path.join(output_dir, f"{report_name}.md")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Comparison report saved to {report_path}")
    
    return report_path

def main():
    """Main function."""
    args = parse_args()
    
    # Get runs to compare
    run_dirs = args.run_dirs
    logger.info(f"Comparing {len(run_dirs)} runs")
    
    # Process each run
    run_info_list = []
    run_results = []
    
    for run_dir in run_dirs:
        # Extract run info
        run_info = extract_run_info(run_dir)
        run_info_list.append(run_info)
        
        # Load validation results
        results = load_validation_results(run_dir)
        run_results.append(results)
    
    # Generate run labels
    if args.labels and len(args.labels) == len(run_dirs):
        run_labels = args.labels
    else:
        run_labels = [get_run_label(info) for info in run_info_list]
    
    # Generate metrics comparison table
    metrics_table = generate_metrics_table(run_results, run_labels)
    
    # Generate comparison plots
    generate_comparison_plots(run_results, run_labels, args.output_dir)
    
    # Generate markdown report
    report_path = generate_markdown_report(
        run_info_list,
        run_results,
        run_labels,
        metrics_table,
        args.output_dir,
        args.report_name
    )
    
    print(f"\nComparison report generated: {report_path}")
    print("\nTo compare other runs, use:")
    print(f"python scripts/generate_validation_comparison.py --run_dirs [dir1] [dir2] --labels [label1] [label2]")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())