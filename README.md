# BetaRho v1.0: RNA Structure Prediction Pipeline

This repository contains the implementation of the BetaRho v1.0 RNA structure prediction pipeline, which integrates the RhoFold+ Invariant-Point-Attention (IPA) structure module into the existing Betabend RNA Feature-Embedding model for improved RNA 3D structure prediction.

## Overview

BetaRho v1.0 is designed to maximize TM-score for RNA structure prediction by combining:

1. **Feature Embedding**: Advanced RNA sequence and feature embedding from Betabend
2. **Transformer Refinement**: Pair and residue representation refinement through transformer blocks
3. **Invariant Point Attention**: Sophisticated structure prediction using RhoFold+'s IPA module
4. **Frame-based Coordinate Generation**: Accurate 3D coordinate prediction using rigid frames

The pipeline is optimized for Kaggle competitions and designed to run on a single A100 40GB GPU.

## Directory Structure

- `rhofold_ipa_module.py`: Core implementation of the RhoFold+ IPA module
- `train_rhofold_ipa.py`: Training pipeline script
- `run_rhofold_ipa.py`: Wrapper script for running training and validation
- `validate_rhofold_ipa.py`: Validation script for evaluating model performance
- `batch_test.py`: Benchmarking and testing script for model performance
- `test_utils.py`: Utility tests for TM-score and other functions
- `utils/`: Helper functions and utilities
  - `model_utils.py`: Model utility functions (TM-score, FAPE, etc.)
  - `count_parameters.py`: Parameter counting and model analysis utilities

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- (Optional) Weights & Biases for logging

## Data

The pipeline expects data in the following structure:

```
data/
├── train_sequences.csv       # RNA sequences with temporal_cutoff column
├── train_labels.csv          # 3D coordinates for training structures
├── validation_sequences.csv  # Validation set sequences
├── validation_labels.csv     # Validation set 3D coordinates
└── processed/                # Processed feature files
    ├── mi_features/          # Mutual information features
    ├── dihedral_features/    # Dihedral angle features
    └── thermo_features/      # Thermodynamic features
```

## Usage

### Training

To train the model:

```bash
python run_rhofold_ipa.py --mode train \
  --train_csv data/train_sequences.csv \
  --label_csv data/train_labels.csv \
  --feature_root data/processed \
  --val_csv data/validation_sequences.csv \
  --epochs 30 --batch 4 --lr 3e-4 \
  --ckpt_out checkpoints/rhofold_ipa_final.pt
```

### Validation

To validate a trained model:

```bash
python run_rhofold_ipa.py --mode validate \
  --ckpt_path checkpoints/rhofold_ipa_final.pt \
  --val_csv data/validation_sequences.csv \
  --label_csv data/validation_labels.csv \
  --feature_root data/processed \
  --output_dir results/validation
```

### Benchmark Testing

To benchmark model performance:

```bash
python batch_test.py \
  --batch_size 4 \
  --seq_lens 50 100 200 \
  --output_file results/benchmark/benchmark_results.json
```

## Model Architecture

The model architecture consists of the following components:

1. **EmbeddingModule**: Processes RNA sequences and features into initial representations
2. **TransformerBlocks**: Refine representations through self-attention and feed-forward networks
3. **RhoFoldIPAModule**: 
   - Adapts representations for IPA
   - Initializes rigid frames
   - Applies IPA for 3D structure prediction
   - Predicts RNA torsion angles
   - Generates final C1' coordinates

## Training Strategy

The training pipeline uses:

- **Primary loss**: 1 - TM-score for global structure accuracy
- **Auxiliary losses**: 
  - Frame-Aligned Point Error (FAPE) for local coordinate accuracy
  - Base-pair contact BCE using pairing probabilities
- **Optimization**: Adam optimizer with mixed precision and gradient checkpointing
- **Early stopping**: Based on moving average validation TM-score
- **Temporal validation split**: Train on sequences with temporal_cutoff < 2022-05-27
- **Hardware target**: Single A100 40GB GPU

## Performance

The model is optimized for:

- **Accuracy**: Maximizing TM-score on CASP15-style RNA structure prediction
- **Efficiency**: Training within 40GB VRAM constraints
- **Speed**: Gradient checkpointing for larger batch sizes

## References

- RhoFold+: Advanced RNA 3D structure prediction system
- Betabend: RNA feature embedding model
- Invariant Point Attention: Geometric deep learning for 3D structure prediction

## License

This project is licensed under [License information].
