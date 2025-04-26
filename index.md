# BetaRho v1.0 Codebase Index

This document provides a comprehensive index of all files and directories in the BetaRho v1.0 RNA structure prediction pipeline.

## Top-Level Files

- `/README.md` - Main project README file
- `/batch_test.py` - Batch testing script for model performance
- `/debugging_guide.md` - Guide for debugging common issues
- `/hardware_adaptation.md` - Guide for adapting to different hardware
- `/implementation_timeline.md` - Timeline for implementation phases
- `/index.md` - This index file
- `/make_executable.py` - Script to make files executable
- `/rhofold_ipa_module.py` - Core IPA module implementation
- `/run_pipeline.sh` - Pipeline execution script
- `/run_rhofold_ipa.py` - Script to run the RhoFold IPA pipeline
- `/setup_environment.py` - Environment setup script
- `/technical_guide.md` - Technical reference guide
- `/technical_summary.md` - Technical summary of the project
- `/test_tm_score.py` - Script to test TM-score calculations
- `/test_utils.py` - Utility test functions
- `/train_rhofold_ipa.py` - Training script for RhoFold IPA
- `/validate_rhofold_ipa.py` - Validation script for RhoFold IPA
- `/writetest.txt` - Information about Robert Frost
- `/writetest2.txt` - Biography of Emily Dickinson

## BetaRhoPlan Directory

- `/BetaRhoPlan/ai-agent-prompt.md` - AI agent prompt for the project
- `/BetaRhoPlan/debugging-guide.md` - Debugging guide for implementation
- `/BetaRhoPlan/implementation-guide.md` - Implementation guide with technical details
- `/BetaRhoPlan/implementation-timeline.md` - Timeline for implementation phases

## RhoFold-refactor Directory

### Model Components
- `/RhoFold-refactor/src/model/rhofold_components/primitives.py` - Basic model components 
- `/RhoFold-refactor/src/model/rhofold_components/structure_module.py` - Structure module with IPA implementation

### Utilities
- `/RhoFold-refactor/src/utils/rhofold_utils/rigid_utils.py` - Utilities for rigid transformations
- `/RhoFold-refactor/src/utils/rhofold_utils/tensor_utils.py` - Tensor manipulation utilities

## Utils Directory

- `/utils/count_parameters.py` - Script to count model parameters
- `/utils/model_utils.py` - Model utility functions

## betabend-refactor Directory

### Documentation
- `/betabend-refactor/CLAUDE.md` - Documentation for Claude AI assistant
- `/betabend-refactor/docs/RNA_Architecture_Detailed_20250424.md` - Detailed architecture documentation
- `/betabend-refactor/docs/RNA_Architecture_Elevator_Pitches_20250424.md` - High-level architecture summaries
- `/betabend-refactor/docs/RNA_Model_Performance_Analysis_20250424.md` - Performance analysis
- `/betabend-refactor/docs/training_validation_enhancements.md` - Training and validation improvements

#### Legacy Documentation
- `/betabend-refactor/docs/old/1_Context_and_Setup.md` - Initial context and setup information
- `/betabend-refactor/docs/old/2_Feature_Specification.md` - Feature specifications
- `/betabend-refactor/docs/old/3_Architecture_Specification.md` - Architecture specifications
- `/betabend-refactor/docs/old/4_Product_Requirements_V1.md` - Product requirements
- `/betabend-refactor/docs/old/5_Roadmap_V1.md` - Initial roadmap
- `/betabend-refactor/docs/old/6_Tactical_Plan_V1.md` - Tactical plan
- `/betabend-refactor/docs/old/7_AI_Agent_Rules.md` - Rules for AI agents
- `/betabend-refactor/docs/old/10_Validation_and_iteration_stragety` - Validation and iteration strategy

### Source Code
- `/betabend-refactor/src/__init__.py` - Package initialization
- `/betabend-refactor/src/data_loading.py` - Data loading utilities
- `/betabend-refactor/src/data_loading_fixed.py` - Fixed version of data loading
- `/betabend-refactor/src/losses.py` - Loss functions for training

#### Models
- `/betabend-refactor/src/models/__init__.py` - Models package initialization
- `/betabend-refactor/src/models/embeddings.py` - RNA sequence and feature embedding module
- `/betabend-refactor/src/models/enhanced_ipa_module.py` - Enhanced Invariant Point Attention module
- `/betabend-refactor/src/models/ipa_module.py` - Basic Invariant Point Attention module
- `/betabend-refactor/src/models/rna_folding_model.py` - Main RNA folding model implementation
- `/betabend-refactor/src/models/transformer_block.py` - Transformer block for feature refinement

### Scripts
- `/betabend-refactor/scripts/checkpoint_converter.py` - Converts model checkpoints between formats
- `/betabend-refactor/scripts/fix_dataset_analyzer.py` - Analyzes and fixes dataset issues
- `/betabend-refactor/scripts/generate_training_report.py` - Generates training reports
- `/betabend-refactor/scripts/generate_validation_comparison.py` - Compares validation results
- `/betabend-refactor/scripts/monitor_gpu.py` - Monitors GPU usage during training
- `/betabend-refactor/scripts/run_production_training_fixed_v4.sh` - Production training script (v4)
- `/betabend-refactor/scripts/run_production_training_fixed_v5.sh` - Production training script (v5)
- `/betabend-refactor/scripts/run_production_training_fixed_v6.sh` - Production training script (v6)
- `/betabend-refactor/scripts/test_checkpoint.py` - Tests model checkpoints
- `/betabend-refactor/scripts/test_device_handling.py` - Tests device handling (CPU/GPU)
- `/betabend-refactor/scripts/train_enhanced_model_fixed.py` - Training script for enhanced model
- `/betabend-refactor/scripts/validate_enhanced_model.py` - Validates enhanced model performance

### Data Directory
- `/betabend-refactor/data/processed/README.md` - README for processed data directory
- `/betabend-refactor/data/processed/dihedral_features/` - Directory containing dihedral angle features extracted from PDB structures (700+ NPZ files)

### Validation Directory
- `/betabend-refactor/validation/validation_runner.py` - Runs validation tests on models

### Results Directory
- `/betabend-refactor/results/production_run_intensive_v4/` - Results from production run v4
  - `/betabend-refactor/results/production_run_intensive_v4/active_logs/` - Active log files
  - `/betabend-refactor/results/production_run_intensive_v4/debug/` - Debug logs
  - `/betabend-refactor/results/production_run_intensive_v4/metrics/` - Performance metrics
  - `/betabend-refactor/results/production_run_intensive_v4/notifications/` - Notification logs
  - `/betabend-refactor/results/production_run_intensive_v4/run_20250426-014319/` - Specific run results
    - `/betabend-refactor/results/production_run_intensive_v4/run_20250426-014319/checkpoints/` - Model checkpoints
    - `/betabend-refactor/results/production_run_intensive_v4/run_20250426-014319/config.json` - Run configuration
    - `/betabend-refactor/results/production_run_intensive_v4/run_20250426-014319/logs/` - Run-specific logs
- `/betabend-refactor/results/production_run_validation_v6/` - Results from validation run v6
  - `/betabend-refactor/results/production_run_validation_v6/active_logs/` - Active log files
  - `/betabend-refactor/results/production_run_validation_v6/debug/` - Debug logs
  - `/betabend-refactor/results/production_run_validation_v6/metrics/` - Performance metrics
  - `/betabend-refactor/results/production_run_validation_v6/notifications/` - Notification logs

## Feature Files

The `/betabend-refactor/data/processed/dihedral_features/` directory contains over 700 NPZ feature files, each following the naming pattern `{PDB_ID}_{Chain_ID}_dihedral_features.npz`. These files contain pre-computed dihedral angle features extracted from RNA structures. Due to the large number of files (over 700), they are not individually listed here.
