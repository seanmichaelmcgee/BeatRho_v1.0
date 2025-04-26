#!/bin/bash
# HIGH-PERFORMANCE PRODUCTION TRAINING - V5
# MAXIMUM INTENSITY: ~3x resources of v4, ~15x of v3
# WARNING: This will use substantial GPU resources (target: ~50% of RTX 4070)
# Builds on the successful device handling fixes and structure of v4

set -e

# Make script directory the working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR/..

# Create output directory
OUTPUT_DIR="results/production_run_maximum_v5"
mkdir -p $OUTPUT_DIR

# Activate conda environment
eval "$(mamba shell hook --shell bash)"
mamba activate rna-3d-folding

# Set explicit CUDA environment variables for better device management
export CUDA_VISIBLE_DEVICES=0  # Restrict to only GPU 0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.7

# Optimize GPU for compute performance
export CUDA_AUTO_BOOST=1
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB

# Set OMP for CPU operations optimization
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

# Log device settings
echo "Set CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Set PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "Set OMP_NUM_THREADS=${OMP_NUM_THREADS}, MKL_NUM_THREADS=${MKL_NUM_THREADS}"

# Create a directory for saving training logs that will be tailed
LOG_DIR="${OUTPUT_DIR}/active_logs"
mkdir -p $LOG_DIR

# Set up GPU monitoring in background - explicitly monitoring only GPU 0
python scripts/monitor_gpu.py --output_dir "$OUTPUT_DIR" --interval 5 --gpu_ids 0 &
MONITOR_PID=$!

# Create notifications directory
NOTIFICATIONS_DIR="${OUTPUT_DIR}/notifications"
mkdir -p $NOTIFICATIONS_DIR

# Create directory for debugging information
DEBUG_DIR="${OUTPUT_DIR}/debug"
mkdir -p $DEBUG_DIR

# Set up status check interval (in seconds)
STATUS_CHECK_INTERVAL=900  # 15 minutes

# Function to send notification
notify() {
    local message="$1"
    local status="$2"  # success, warning, error
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    local notification_file="${NOTIFICATIONS_DIR}/notification_${status}_$(date +%Y%m%d-%H%M%S).txt"
    
    echo "[${timestamp}] ${status^^}: ${message}" | tee -a "${OUTPUT_DIR}/notifications.log"
    echo "${message}" > "$notification_file"
}

# Function to check training status
check_training_status() {
    local pid=$1
    local training_start_time=$2
    local current_time=$(date +%s)
    local training_elapsed_time=$((current_time - training_start_time))
    
    # Skip checks if training hasn't been running long enough
    if [ $training_elapsed_time -lt 300 ]; then  # Less than 5 minutes
        echo "Training has been running for less than 5 minutes. Skipping status checks."
        return 0
    fi
    
    # Initialize with default values
    local last_checkpoint_time=$training_start_time
    local last_log_time=$training_start_time
    local found_checkpoint=false
    local found_log=false
    
    # Find the newest checkpoint file
    local newest_checkpoint=""
    local checkpoint_pattern="${OUTPUT_DIR}/run_*/checkpoints/*.pt"
    
    if ls $checkpoint_pattern 2>/dev/null >/dev/null; then
        newest_checkpoint=$(find ${OUTPUT_DIR}/run_*/checkpoints -name "*.pt" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
        if [ -n "$newest_checkpoint" ]; then
            last_checkpoint_time=$(stat -c %Y "$newest_checkpoint")
            found_checkpoint=true
        fi
    fi
    
    # Find the latest log update time
    local latest_log=""
    local log_pattern="${OUTPUT_DIR}/run_*/logs/training.log"
    
    if ls $log_pattern 2>/dev/null >/dev/null; then
        latest_log=$(find ${OUTPUT_DIR}/run_*/logs -name "training.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
        if [ -n "$latest_log" ]; then
            last_log_time=$(stat -c %Y "$latest_log")
            found_log=true
        fi
    fi
    
    # Only check for stalled logs if we have actually found logs
    if [ "$found_log" = true ]; then
        # Calculate time since last log update
        local time_since_log=$((current_time - last_log_time))
        
        # Check for stalled training
        if [ $time_since_log -gt 3600 ]; then  # No log updates for 1 hour
            notify "Training appears to be stalled. No log updates for $((time_since_log / 60)) minutes." "warning"
        fi
    else
        # No logs found yet but training has been running for a while
        if [ $training_elapsed_time -gt 900 ]; then  # 15 minutes
            notify "No training logs found after 15 minutes of training. Check for configuration issues." "warning"
        fi
    fi
    
    # Only check for missing checkpoints if we're far enough into training
    if [ $training_elapsed_time -gt 3600 ]; then  # More than 1 hour
        if [ "$found_checkpoint" = true ]; then
            # Calculate time since last checkpoint
            local time_since_checkpoint=$((current_time - last_checkpoint_time))
            
            # Check for checkpoint creation
            if [ $time_since_checkpoint -gt 7200 ]; then  # No checkpoint for 2 hours
                notify "No new checkpoints for $((time_since_checkpoint / 60)) minutes. Training may be stuck." "warning"
            fi
        else
            # No checkpoints found after an hour
            notify "No checkpoints found after 1 hour of training. Check training configuration." "warning"
        fi
    fi
    
    # Check if training is still active
    if ! ps -p $pid > /dev/null; then
        notify "Training process (PID: $pid) is no longer running. Please check logs for details." "error"
        return 1
    fi
    
    return 0
}

# Run periodic checkpoint validation
validate_checkpoint() {
    local checkpoint_path="$1"
    local validation_log="${OUTPUT_DIR}/checkpoint_validation.log"
    
    echo "Validating checkpoint: $checkpoint_path" >> "$validation_log"
    
    # Run a simple validation test on the checkpoint
    python scripts/test_checkpoint.py \
        --checkpoint "$checkpoint_path" \
        --num_samples 5 \
        --output_dir "${OUTPUT_DIR}/validation" >> "$validation_log" 2>&1
    
    if [ $? -eq 0 ]; then
        notify "Checkpoint validation successful: $checkpoint_path" "success"
    else:
        notify "Checkpoint validation failed: $checkpoint_path. Check logs for details." "error"
    fi
}

echo "Starting full production training run with enhanced device handling and fixes..."
echo "Output directory: $OUTPUT_DIR"
echo "Curriculum learning is enabled with sequences gradually increasing in length"

# First verify device handling with a test script
echo "Verifying device handling with test script..."
python scripts/test_device_handling.py \
  --data_csv data/raw/train_sequences.csv \
  --labels_csv data/raw/train_labels.csv \
  --features_dir data/processed/ \
  --batch_size 2 \
  --gpu 0 > "${DEBUG_DIR}/device_test.log" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Device handling verification successful"
else
    echo "❌ Device handling verification failed - check ${DEBUG_DIR}/device_test.log"
    # Continue anyway but at higher risk of failure
fi

# Create new environment variables to enable enhanced device handling
export RNA_ENFORCE_DEVICE_CONSISTENCY=1
export RNA_DEBUG_DEVICE_ISSUES=1
echo "Enhanced device consistency enforcement is ENABLED"
echo "Device issue debugging is ENABLED"

# Run training with MAXIMUM model and high computational load
# 1. SUBSTANTIALLY larger model (now ~15x bigger than original v3)
# 2. Higher batch size and longer sequences
# 3. Extended curriculum with more stages and longer sequences
# 4. Optimized learning rate and scheduling

# First create a file that will be actively tailed for monitoring
touch "${LOG_DIR}/live_training.log"

# Create special GPU log
touch "${LOG_DIR}/gpu_stats.log"

# Begin MAXIMUM training - parameters increased ~15x from v3, ~3x from v4
python scripts/train_enhanced_model_fixed.py \
  --train_csv data/raw/train_sequences.csv \
  --labels_csv data/raw/train_labels.csv \
  --features_dir data/processed/ \
  --output_dir $OUTPUT_DIR \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --epochs 750 \
  --lr 0.0002 \
  --num_blocks 24 \
  --residue_embed_dim 384 \
  --pair_embed_dim 192 \
  --num_heads 24 \
  --ff_dim 2048 \
  --dropout 0.2 \
  --curriculum_learning \
  --curriculum_stages 50 100 150 200 250 300 350 400 450 500 550 600 650 \
  --epochs_per_stage 5 \
  --batch_adaptive \
  --gradient_checkpointing \
  --mixed_precision \
  --memory_fraction_warning 0.8 \
  --memory_fraction_critical 0.95 \
  --save_interval_epochs 10 \
  --save_interval_steps 1000 \
  --patience 30 \
  --log_level INFO | tee -a "${LOG_DIR}/live_training.log" "${OUTPUT_DIR}/training.log" 2>&1 &

TRAINING_PID=$!
echo "Training started with PID: $TRAINING_PID"
notify "Started enhanced production training run with PID: $TRAINING_PID" "success"

# Record training start time
TRAINING_START_TIME=$(date +%s)
echo "Training start timestamp: $(date -d @$TRAINING_START_TIME)"

# Start background process to log GPU stats every 30 seconds
(
  while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv)" >> "${LOG_DIR}/gpu_stats.log"
    sleep 30
  done
) &
GPU_MONITOR_PID=$!
echo "Started GPU stat logging with PID: $GPU_MONITOR_PID"

# Display monitoring commands for user convenience
echo ""
echo "====================== MONITORING COMMANDS ======================"
echo "To watch live training progress in real-time, run:"
echo "  tail -f ${LOG_DIR}/live_training.log"
echo ""
echo "To monitor GPU usage in real-time:"
echo "  tail -f ${LOG_DIR}/gpu_stats.log"
echo ""
echo "For detailed GPU monitoring:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "To see recent metrics:"
echo "  tail -n 50 ${OUTPUT_DIR}/training.log"
echo ""
echo "To see list of saved checkpoints:"
echo "  ls -la \$(find ${OUTPUT_DIR} -name 'checkpoints' -type d)"
echo ""
echo "To stop the training and clean up background processes:"
echo "  kill $TRAINING_PID $MONITOR_PID $GPU_MONITOR_PID"
echo "==============================================================="
echo ""

# Create a flag file to indicate first status check
FIRST_CHECK_FLAG="${OUTPUT_DIR}/first_check_done"

# Monitor the training process
declare -i checks=0
while true; do
    # Check if training process is still running
    if ! ps -p $TRAINING_PID > /dev/null; then
        echo "Training process has completed or terminated."
        break
    fi
    
    # Extract recent GPU memory info for immediate feedback
    if [ -f "${OUTPUT_DIR}/metrics/gpu_metrics_"*".csv" ]; then
        RECENT_MEM=$(tail -n 1 ${OUTPUT_DIR}/metrics/gpu_metrics_*.csv 2>/dev/null | awk -F',' '{print $4}')
        if [ -n "$RECENT_MEM" ]; then
            echo "Current GPU memory usage: ${RECENT_MEM} MB"
        fi
    fi
    
    # Perform status check every interval, with special handling for first check
    if ((checks == 30)); then  # First real check after 5 minutes (30 x 10s)
        echo "Performing first training status check after 5 minutes..."
        check_training_status $TRAINING_PID $TRAINING_START_TIME
        touch "$FIRST_CHECK_FLAG"
    elif ((checks > 30)) && ((checks % (STATUS_CHECK_INTERVAL / 10) == 0)); then
        echo "Performing regular training status check..."
        check_training_status $TRAINING_PID $TRAINING_START_TIME
    fi
    
    # Validate the latest checkpoint periodically (only after at least 30 minutes)
    if ((checks >= 180)) && ((checks % (STATUS_CHECK_INTERVAL * 6 / 10) == 0)); then
        echo "Checking for checkpoints to validate..."
        # Find the newest checkpoint
        checkpoint_pattern="${OUTPUT_DIR}/run_*/checkpoints/best_model.pt"
        
        if ls $checkpoint_pattern 2>/dev/null >/dev/null; then
            newest_checkpoint=$(find ${OUTPUT_DIR}/run_*/checkpoints -name "best_model.pt" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -f2- -d" ")
            if [ -n "$newest_checkpoint" ]; then
                # Check if checkpoint has been validated already
                checkpoint_hash=$(md5sum "$newest_checkpoint" | cut -d' ' -f1)
                validation_flag="${OUTPUT_DIR}/validated_${checkpoint_hash}"
                
                if [ ! -f "$validation_flag" ]; then
                    echo "Validating checkpoint: $newest_checkpoint"
                    validate_checkpoint "$newest_checkpoint"
                    touch "$validation_flag"
                else
                    echo "Checkpoint already validated, skipping: $newest_checkpoint"
                fi
            else
                echo "No best_model.pt checkpoint found yet."
            fi
        else
            echo "No checkpoint directories found yet."
        fi
    fi
    
    # Wait 10 seconds before next check
    sleep 10
    ((checks++))
done

# Wait for training to complete
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

# Kill all monitoring processes
if [ -n "$MONITOR_PID" ]; then
    echo "Stopping primary GPU monitoring (PID: ${MONITOR_PID})..."
    kill $MONITOR_PID 2>/dev/null || true
fi

if [ -n "$GPU_MONITOR_PID" ]; then
    echo "Stopping GPU stats logging (PID: ${GPU_MONITOR_PID})..."
    kill $GPU_MONITOR_PID 2>/dev/null || true
fi

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    notify "Training completed successfully!" "success"
    
    # Generate training report
    echo "Generating training report..."
    python scripts/generate_training_report.py --input_dir "$OUTPUT_DIR/run_"* > "${OUTPUT_DIR}/report_generation.log" 2>&1
    
    # Convert checkpoint to Kaggle format
    echo "Converting best checkpoint to Kaggle format..."
    python scripts/checkpoint_converter.py \
        --checkpoint "${OUTPUT_DIR}/run_*/checkpoints/best_model.pt" \
        --output "kaggle_submission/rna-3d-models/best_model.pt" \
        --create-metadata > "${OUTPUT_DIR}/checkpoint_conversion.log" 2>&1
        
    if [ $? -eq 0 ]; then
        notify "Checkpoint successfully converted to Kaggle format" "success"
    else
        notify "Checkpoint conversion failed. Check logs for details." "error"
    fi
    
    echo "Training complete! Results saved to $OUTPUT_DIR"
else
    notify "Training failed with exit code $TRAINING_EXIT_CODE" "error"
    echo "Training failed with exit code $TRAINING_EXIT_CODE"
    
    # Create debugging package to help with future debugging
    DEBUG_PACKAGE="${OUTPUT_DIR}/debug_package_$(date +%Y%m%d-%H%M%S).tar.gz"
    echo "Creating debugging package at $DEBUG_PACKAGE"
    
    # Collect relevant logs and files
    mkdir -p "${DEBUG_DIR}/logs"
    cp "${OUTPUT_DIR}/training.log" "${DEBUG_DIR}/logs/" 2>/dev/null || true
    cp "${OUTPUT_DIR}/run_"*/logs/training.log "${DEBUG_DIR}/logs/detailed_training.log" 2>/dev/null || true
    cp "${OUTPUT_DIR}/metrics/gpu_metrics_"*.csv "${DEBUG_DIR}/" 2>/dev/null || true
    cp "${DEBUG_DIR}/device_test.log" "${DEBUG_DIR}/logs/" 2>/dev/null || true
    
    # Create the debug package
    tar -czf $DEBUG_PACKAGE -C "${DEBUG_DIR}" .
    echo "Debug package created. Please provide this file for analysis."
fi
