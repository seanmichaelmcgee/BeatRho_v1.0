# Training Validation Enhancements

This document describes the comprehensive validation framework integrated into the training pipeline.

## Overview

The enhanced validation system provides detailed structural validation metrics during training by:

1. Adding a validation hook that performs dual-mode validation (test-equivalent and training-equivalent)
2. Configuring periodic validation calls at specified intervals  
3. Structured logging of comprehensive metrics for both modes
4. HTML report generation for easy comparison across training runs

## Implementation

### Core Components

- **ValidationRunner Class**: Implements dual-mode validation capability from `validation/validation_runner.py`
- **Validation Hook**: Integration into the training loop via the `run_comprehensive_validation` function
- **Periodic Execution**: Configurable frequency for comprehensive validation during training
- **Comparison Visualization**: Generates HTML reports with interactive visualizations

### Command Line Options

The training script now supports these additional validation parameters:

```
--comprehensive_val           Enable comprehensive validation with ValidationRunner
--comprehensive_val_freq N    Run comprehensive validation every N epochs (default: 5)
--comprehensive_val_subset    Default validation subset (technical/scientific/comprehensive)
```

### Validation Modes

1. **Test-Equivalent Mode**: Uses only features available at test time to simulate Kaggle inference
2. **Training-Equivalent Mode**: Uses all features available during training
3. **Mode Comparison**: Analysis of the impact of feature availability differences

### Metrics

- **RMSD (Root Mean Square Deviation)**: After Kabsch alignment
- **TM-score (Template Modeling score)**: Structural similarity (0-1 scale)
- **Per-residue error**: Position-specific analysis of prediction accuracy
- **Feature impact analysis**: Quantifies the effect of missing features at test time

## Usage

### During Training

Add comprehensive validation to your training run:

```bash
python scripts/train_enhanced_model_fixed.py \
  --comprehensive_val \
  --comprehensive_val_freq 10 \
  --comprehensive_val_subset technical \
  [other training options...]
```

### Standalone Validation

For validation of a saved model checkpoint:

```bash
./scripts/run_enhanced_validation.sh \
  --run-dir /path/to/run_directory \
  --model-path /path/to/model/checkpoint.pt \
  --subset technical
```

### Report Generation

Generate a comparison report across multiple validation points:

```bash
python scripts/generate_validation_comparison.py \
  --run_dir /path/to/run_directory \
  --title "Model Name Validation Report"
```

## Output Files

Each validation run produces:

- `validation_results_epoch_{N}_{timestamp}.json`: Detailed validation metrics
- Visualization plots:
  - RMSD distribution
  - TM-score distribution
  - Per-residue error analysis
  - Mode comparison charts

The comparison report generates an HTML file with:

- Interactive tables of metrics across epochs
- Trend visualizations
- Automatically generated observations and recommendations

## Implementation Notes

1. Validation timing:
   - Regular validation still runs at `--eval_every` interval
   - Comprehensive validation runs at `--comprehensive_val_freq` interval (typically less frequent)
   - Final validation runs at the end of training

2. Memory considerations:
   - Comprehensive validation uses a smaller batch size to avoid OOM errors
   - Multiple error handling strategies ensure training continues even if validation fails

3. Result integration:
   - Validation results are included in checkpoints for reproducibility
   - RMSD from comprehensive validation is considered when tracking best model
   - TM-score tracking provides an orthogonal quality metric

## Future Enhancements

Potential future improvements:

1. Additional structural metrics (GDT-TS, contact map accuracy)
2. Visual 3D structure comparison
3. Feature attribution analysis (what features contribute most to performance differences)
4. Sample difficulty stratification
5. Incorporation of RNA family analysis for domain-specific performance