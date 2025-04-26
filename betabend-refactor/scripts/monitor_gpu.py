#!/usr/bin/env python3
"""
GPU Monitoring Tool for RNA 3D Structure Prediction Training

This script monitors GPU utilization, memory usage, and temperature during training.
It runs as a background process and logs metrics to a CSV file at specified intervals.
"""

import os
import sys
import time
import argparse
import logging
import csv
import subprocess
import signal
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Monitor GPU metrics during training')
    
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save monitoring logs')
    parser.add_argument('--interval', type=int, default=10,
                       help='Sampling interval in seconds')
    parser.add_argument('--gpu_ids', type=str, default=None,
                       help='Comma-separated list of GPU IDs to monitor (default: all)')
    
    return parser.parse_args()

def get_nvidia_smi_metrics():
    """
    Query GPU metrics using nvidia-smi command.
    
    Returns:
        List of dictionaries with metrics for each GPU
    """
    try:
        # Run nvidia-smi with CSV format output
        cmd = [
            "nvidia-smi", 
            "--query-gpu=index,timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw", 
            "--format=csv,noheader,nounits"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the output
        metrics = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
                
            # Split the CSV line and parse values
            values = [v.strip() for v in line.split(',')]
            
            gpu_metrics = {
                'gpu_id': int(values[0]),
                'timestamp': values[1],
                'gpu_utilization': float(values[2]),
                'memory_utilization': float(values[3]),
                'memory_used_mb': float(values[4]),
                'memory_total_mb': float(values[5]),
                'temperature_c': float(values[6]),
                'power_draw_w': float(values[7]) if values[7] != "N/A" else None
            }
            metrics.append(gpu_metrics)
            
        return metrics
        
    except (subprocess.SubprocessError, FileNotFoundError, IndexError, ValueError) as e:
        logger.error(f"Error getting GPU metrics: {e}")
        return []

def monitor_loop(output_dir, interval, gpu_ids=None):
    """
    Main monitoring loop that logs GPU metrics at specified intervals.
    
    Args:
        output_dir: Directory to save monitoring logs
        interval: Sampling interval in seconds
        gpu_ids: List of GPU IDs to monitor (or None for all GPUs)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metrics subdirectory
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Prepare log file path
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(metrics_dir, f"gpu_metrics_{timestamp}.csv")
    
    # Convert gpu_ids string to list of integers if provided
    gpu_id_list = None
    if gpu_ids is not None:
        try:
            gpu_id_list = [int(gpu_id.strip()) for gpu_id in gpu_ids.split(',')]
        except ValueError:
            logger.error(f"Invalid GPU IDs format: {gpu_ids}, monitoring all GPUs")
    
    # Initialize CSV file with headers
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'elapsed_time_s', 'gpu_id', 
            'gpu_utilization_percent', 'memory_utilization_percent',
            'memory_used_mb', 'memory_total_mb', 'memory_used_percent',
            'temperature_c', 'power_draw_w'
        ])
    
    # Log start time
    start_time = time.time()
    logger.info(f"GPU monitoring started. Logging to {log_file}")
    logger.info(f"Sampling interval: {interval} seconds")
    if gpu_id_list:
        logger.info(f"Monitoring GPUs: {gpu_id_list}")
    else:
        logger.info(f"Monitoring all available GPUs")
    
    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received termination signal. Shutting down monitoring...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Main monitoring loop
        while True:
            # Get current timestamp and elapsed time
            current_time = time.time()
            elapsed_time = current_time - start_time
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get GPU metrics
            metrics = get_nvidia_smi_metrics()
            
            # Filter metrics if gpu_id_list is specified
            if gpu_id_list:
                metrics = [m for m in metrics if m['gpu_id'] in gpu_id_list]
            
            # Log metrics to CSV file
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for gpu_metrics in metrics:
                    # Calculate memory used percent
                    memory_used_percent = (
                        gpu_metrics['memory_used_mb'] / gpu_metrics['memory_total_mb'] * 100
                        if gpu_metrics['memory_total_mb'] > 0 else 0
                    )
                    
                    writer.writerow([
                        timestamp,
                        f"{elapsed_time:.2f}",
                        gpu_metrics['gpu_id'],
                        gpu_metrics['gpu_utilization'],
                        gpu_metrics['memory_utilization'],
                        gpu_metrics['memory_used_mb'],
                        gpu_metrics['memory_total_mb'],
                        f"{memory_used_percent:.2f}",
                        gpu_metrics['temperature_c'],
                        gpu_metrics['power_draw_w'] if gpu_metrics['power_draw_w'] is not None else 'N/A'
                    ])
            
            # Wait for next sampling interval
            time.sleep(interval)
            
    except Exception as e:
        logger.error(f"Error in monitoring loop: {e}")
        sys.exit(1)
    finally:
        logger.info("GPU monitoring stopped")

def main():
    """Main entry point"""
    args = parse_args()
    monitor_loop(args.output_dir, args.interval, args.gpu_ids)
    return 0

if __name__ == "__main__":
    sys.exit(main())