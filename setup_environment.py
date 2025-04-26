#!/usr/bin/env python3
"""
Setup script for BetaRho v1.0 RNA structure prediction pipeline.

This script creates the necessary directory structure for the pipeline and
verifies the environment setup.
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def create_directories(base_dir: str = ".") -> None:
    """
    Create necessary directories for the pipeline.
    
    Args:
        base_dir: Base directory for the pipeline
    """
    # Create directory structure
    directories = [
        "checkpoints",
        "data/processed/mi_features",
        "data/processed/dihedral_features",
        "data/processed/thermo_features",
        "results/validation",
        "results/benchmark",
        "logs",
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def check_permissions() -> bool:
    """
    Check write permissions in the current directory.
    
    Returns:
        True if permissions are valid, False otherwise
    """
    try:
        # Create a temporary file to test permissions
        test_file = "write_test.tmp"
        with open(test_file, "w") as f:
            f.write("test")
        
        # Clean up
        os.remove(test_file)
        
        return True
    except Exception as e:
        logger.error(f"Permission error: {e}")
        return False

def verify_utility_files() -> None:
    """
    Verify that necessary utility files exist.
    """
    required_files = [
        "utils/model_utils.py",
        "rhofold_ipa_module.py",
        "train_rhofold_ipa.py",
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.warning(f"Required file missing: {file_path}")
        else:
            logger.info(f"Found required file: {file_path}")

def make_executable(file_path: str) -> None:
    """
    Make a Python file executable.
    
    Args:
        file_path: Path to the Python file
    """
    if os.path.exists(file_path):
        current_mode = os.stat(file_path).st_mode
        os.chmod(file_path, current_mode | 0o111)  # Add execute permission
        logger.info(f"Made executable: {file_path}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Setup BetaRho environment")
    
    parser.add_argument("--base_dir", type=str, default=".",
                      help="Base directory for the pipeline")
    parser.add_argument("--make_executable", action="store_true",
                      help="Make Python scripts executable")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    logger.info("Setting up BetaRho v1.0 environment...")
    
    # Check permissions
    if not check_permissions():
        logger.error("Insufficient permissions. Please check directory access.")
        sys.exit(1)
    
    # Create directories
    create_directories(args.base_dir)
    
    # Verify utility files
    verify_utility_files()
    
    # Make scripts executable if requested
    if args.make_executable:
        executable_scripts = [
            "run_rhofold_ipa.py",
            "train_rhofold_ipa.py",
            "validate_rhofold_ipa.py",
            "batch_test.py",
            "test_utils.py",
        ]
        
        for script in executable_scripts:
            make_executable(script)
    
    logger.info("Environment setup complete!")

if __name__ == "__main__":
    main()
