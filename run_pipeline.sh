#!/bin/bash
# Complete BetaRho v1.0 RNA structure prediction pipeline runner
# This script runs the entire pipeline from setup to validation

set -e  # Exit on error

# ============================================
# Configuration (modify as needed)
# ============================================
TRAIN_CSV="data/train_sequences.csv"
LABEL_CSV="data/train_labels.csv"
FEATURE_ROOT="data/processed"
VAL_CSV="data/validation_sequences.csv"
BATCH_SIZE=4
EPOCHS=30
LEARNING_RATE=3e-4
CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results"
IPA_HEADS=4
IPA_BLOCKS=4
USE_MIXED_PRECISION=true
USE_GRAD_CHECKPOINT=true

# ============================================
# Print header
# ============================================
echo "====================================================="
echo " BetaRho v1.0 RNA Structure Prediction Pipeline"
echo "====================================================="
echo "Starting pipeline run at $(date)"
echo ""

# ============================================
# Setup environment
# ============================================
echo "Setting up environment..."
python setup_environment.py --make_executable
echo ""

# Make this script executable too
chmod +x "$0"
python make_executable.py
echo ""

# ============================================
# Run model tests
# ============================================
echo "Running utility tests..."
python test_utils.py
echo ""

# ============================================
# Run batch tests and benchmarks
# ============================================
echo "Running batch tests and benchmarks..."
python batch_test.py \
    --batch_size "${BATCH_SIZE}" \
    --seq_lens 50 100 200 \
    --output_file "${RESULTS_DIR}/benchmark/batch_test_results.json" \
    --no_heads "${IPA_HEADS}" \
    --num_ipa_blocks "${IPA_BLOCKS}"
echo ""

# ============================================
# Train the model
# ============================================
echo "Training RhoFold+ IPA model..."
if [ "${USE_MIXED_PRECISION}" = true ]; then
    MIXED_PRECISION_ARG="--mixed_precision"
else
    MIXED_PRECISION_ARG=""
fi

if [ "${USE_GRAD_CHECKPOINT}" = true ]; then
    GRAD_CHECKPOINT_ARG="--grad_checkpoint"
else
    GRAD_CHECKPOINT_ARG=""
fi

python run_rhofold_ipa.py --mode train \
    --train_csv "${TRAIN_CSV}" \
    --label_csv "${LABEL_CSV}" \
    --feature_root "${FEATURE_ROOT}" \
    --val_csv "${VAL_CSV}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH_SIZE}" \
    --lr "${LEARNING_RATE}" \
    --no_heads "${IPA_HEADS}" \
    --num_ipa_blocks "${IPA_BLOCKS}" \
    --ckpt_out "${CHECKPOINT_DIR}/rhofold_ipa_final.pt" \
    ${MIXED_PRECISION_ARG} \
    ${GRAD_CHECKPOINT_ARG} \
    --run_tests
echo ""

# ============================================
# Validate the model
# ============================================
echo "Validating trained model..."
python run_rhofold_ipa.py --mode validate \
    --ckpt_path "${CHECKPOINT_DIR}/rhofold_ipa_final.pt" \
    --val_csv "${VAL_CSV}" \
    --label_csv "${LABEL_CSV}" \
    --feature_root "${FEATURE_ROOT}" \
    --output_dir "${RESULTS_DIR}/validation"
echo ""

# ============================================
# Print completion message
# ============================================
echo "====================================================="
echo " Pipeline completed successfully!"
echo "====================================================="
echo "Completed at $(date)"
echo ""
echo "Results available at:"
echo "  - Model checkpoint: ${CHECKPOINT_DIR}/rhofold_ipa_final.pt"
echo "  - Validation report: ${RESULTS_DIR}/validation/validation_report.csv"
echo "  - Benchmark results: ${RESULTS_DIR}/benchmark/batch_test_results.json"
echo ""
