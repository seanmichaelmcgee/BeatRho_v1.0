#!/usr/bin/env python3
"""
Script to make all BetaRho Python files executable.
"""

import os
import sys

def make_executable(file_path):
    """Make a file executable."""
    if os.path.exists(file_path):
        current_mode = os.stat(file_path).st_mode
        os.chmod(file_path, current_mode | 0o111)  # Add execute permission
        print(f"Made executable: {file_path}")
    else:
        print(f"File not found: {file_path}")

def main():
    """Make all main scripts executable."""
    # Files to make executable
    script_files = [
        "run_rhofold_ipa.py",
        "train_rhofold_ipa.py",
        "validate_rhofold_ipa.py",
        "batch_test.py",
        "test_utils.py",
        "setup_environment.py",
        "make_executable.py"  # Include this script itself
    ]
    
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Make each file executable
    for script in script_files:
        file_path = os.path.join(base_dir, script)
        make_executable(file_path)
    
    print("Done! All scripts are now executable.")

if __name__ == "__main__":
    main()
