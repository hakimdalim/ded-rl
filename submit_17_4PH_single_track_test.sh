#!/bin/bash
#SBATCH --job-name=ded_17-4PH_single_track_test
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --partition=epyc-256
#SBATCH --mem=0
#SBATCH --array=0

# Test submission script for 17-4PH single-track experiments
# Reduced parameter set: 32 total experiments
# 16 parallel experiments × 1 CPU each
# Uses SLURM steps instead of GNU Parallel

echo "=========================================="
echo "DED 17-4PH Single-Track TEST Run"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Total Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module purge  # Clean environment to prevent conflicts
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate tft_3_9

# Verify environment
echo ""
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================
PARALLEL_JOBS=16  # Number of experiments to run simultaneously
MAX_CPU_CORES=$SLURM_CPUS_PER_TASK  # CPUs per experiment (1)
# Note: No explicit memory limit per experiment - let SLURM divide fairly

echo "=========================================="
echo "Resource Allocation"
echo "=========================================="
echo "Total CPUs available: 16"
echo "Parallel experiments: $PARALLEL_JOBS"
echo "CPUs per experiment: $MAX_CPU_CORES"
echo "Memory: Shared from node allocation (no per-step limit)"
echo "=========================================="
echo ""

# Configuration
MATERIAL="17-4PH"
BATCH_ID=$SLURM_ARRAY_TASK_ID
EXPERIMENTS_PER_BATCH=32  # Test with only 32 experiments
TOTAL_EXPERIMENTS=32

# Shared directories
POWDER_STREAM_DIR="/scratch/schuermm-shared/sim_powder_stream_arrays"
OUTPUT_BASE_DIR="/scratch/schuermm-shared/ded_17_4PH_single_track_TEST"  # Separate test directory

# Calculate start and end indices for this batch
START_IDX=$((BATCH_ID * EXPERIMENTS_PER_BATCH))
END_IDX=$(((BATCH_ID + 1) * EXPERIMENTS_PER_BATCH - 1))

# Don't exceed total
if [ $END_IDX -ge $TOTAL_EXPERIMENTS ]; then
    END_IDX=$((TOTAL_EXPERIMENTS - 1))
fi

BATCH_SIZE=$((END_IDX - START_IDX + 1))

echo "Batch $BATCH_ID: Experiments $START_IDX to $END_IDX ($BATCH_SIZE experiments)"
echo ""

# Create output directory for this batch
BATCH_OUTPUT_DIR="${OUTPUT_BASE_DIR}/batch_${BATCH_ID}_job_${SLURM_JOB_ID}"
mkdir -p "$BATCH_OUTPUT_DIR"

echo "Batch output directory: $BATCH_OUTPUT_DIR"
echo ""

# Generate parameter combinations for this batch
# Test design: 4 powers × 2 diameters × 2 feeds × 2 speeds = 32 experiments

PARAM_FILE="$BATCH_OUTPUT_DIR/parameters.txt"
> "$PARAM_FILE"  # Clear file

# Reduced parameter ranges for testing
POWERS=(600 900 1200 1500)           # 4 values (was 11)
DIAMETERS=(0.5 1.8)                  # 2 values (was 5)
FEEDS=(2.0 4.0)                      # 2 values (was 5)
SPEEDS=(2 20)                        # 2 values (was 10)

echo "Generating parameter combinations..."

experiment_idx=0
for power in "${POWERS[@]}"; do
    for diameter in "${DIAMETERS[@]}"; do
        for feed in "${FEEDS[@]}"; do
            for speed in "${SPEEDS[@]}"; do
                # Check if this experiment belongs to current batch
                if [ $experiment_idx -ge $START_IDX ] && [ $experiment_idx -le $END_IDX ]; then
                    echo "$MATERIAL	$power	$diameter	$feed	$speed" >> "$PARAM_FILE"
                fi
                ((experiment_idx++))
            done
        done
    done
done

ACTUAL_COUNT=$(wc -l < "$PARAM_FILE")
echo "Generated $ACTUAL_COUNT parameter combinations for this batch"
echo ""

if [ $ACTUAL_COUNT -ne $BATCH_SIZE ]; then
    echo "WARNING: Expected $BATCH_SIZE experiments but generated $ACTUAL_COUNT"
fi

# Run experiments with SLURM steps
echo "=========================================="
mkdir -p "$BATCH_OUTPUT_DIR/experiments" "$BATCH_OUTPUT_DIR/results"
echo "Starting SLURM step execution"
echo "Batch size: $ACTUAL_COUNT experiments"
echo "=========================================="
echo ""

START_TIME=$(date +%s)

# ============================================================================
# RUN EXPERIMENTS WITH SLURM STEPS
# Read parameters into memory first to avoid file descriptor issues
# ============================================================================

# Read all parameter lines into array BEFORE launching jobs
mapfile -t PARAM_LINES < "$PARAM_FILE"

echo "Read ${#PARAM_LINES[@]} parameter combinations into memory"

# Create detailed launch log
LAUNCH_LOG="$BATCH_OUTPUT_DIR/launch.log"
echo "Timestamp	ExpNum	Material	Power	Diameter	Feed	Speed" > "$LAUNCH_LOG"

exp_num=0
for line in "${PARAM_LINES[@]}"; do
    
    # Parse the tab-separated line
    IFS=$'\t' read -r MATERIAL POWER DIAMETER FEED SPEED <<< "$line"
    
    # Create meaningful log names (including experiment number for easy correlation)
    EXP_NAME="exp_${exp_num}_job_${SLURM_ARRAY_JOB_ID}_task_${BATCH_ID}_${MATERIAL}_P${POWER}W_D${DIAMETER}mm_F${FEED}gmin_V${SPEED}mms"
    LOG_OUT="$BATCH_OUTPUT_DIR/results/${EXP_NAME}.out"
    LOG_ERR="$BATCH_OUTPUT_DIR/results/${EXP_NAME}.err"
    
    # Run on local storage
    LOCAL_DIR="${TMPDIR:-/tmp}/ded_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_exp_${exp_num}"
    
    # Log launch with timestamp
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$TIMESTAMP	$exp_num	$MATERIAL	$POWER	$DIAMETER	$FEED	$SPEED" >> "$LAUNCH_LOG"
    echo "[$TIMESTAMP] Launching experiment $((exp_num + 1))/$ACTUAL_COUNT: ${MATERIAL} P=${POWER}W D=${DIAMETER}mm F=${FEED}g/min V=${SPEED}mm/s"
    
    # Use srun to create SLURM job step
    srun --ntasks=1 \
         --cpus-per-task=$MAX_CPU_CORES \
         --exclusive \
         bash -c "
           set -uo pipefail
           mkdir -p '$LOCAL_DIR'
           
           # Run Python and capture exit code
           python run_single_track_experiment.py \
             -m $MATERIAL -p $POWER -d $DIAMETER -f $FEED -v $SPEED \
             --powder-stream-dir '$POWDER_STREAM_DIR' \
             --output-dir '$LOCAL_DIR' \
             --max-cpu-cores $MAX_CPU_CORES \
             --experiment-id exp_${exp_num} \
             > '$LOG_OUT' 2> '$LOG_ERR'
           PYTHON_EXIT=\$?
           
           # Mark failed experiments
           if [ \$PYTHON_EXIT -ne 0 ]; then
               echo \"ERROR: Python exited with code \$PYTHON_EXIT\" >> '$LOG_ERR'
               touch '$LOCAL_DIR/FAILED'
           fi
           
           # ALWAYS sync results back to shared storage (even on failure)
           rsync -a --partial --inplace '$LOCAL_DIR'/ '$BATCH_OUTPUT_DIR/experiments/' || true

           # Rename experiment directory to include exp number
           EXP_DIR=$(find '$BATCH_OUTPUT_DIR/experiments/' -maxdepth 1 -type d -name '${MATERIAL}_P*' -newer '$LOCAL_DIR' 2>/dev/null | head -1)
           if [ -n "$EXP_DIR" ]; then
               BASENAME=$(basename "$EXP_DIR")
               mv "$EXP_DIR" "$(dirname "$EXP_DIR")/exp_${exp_num}_$BASENAME"
           fi
           
           # ALWAYS cleanup temp directory
           rm -rf '$LOCAL_DIR'
           
           # Exit with original Python exit code (SLURM records this)
           exit \$PYTHON_EXIT
         " &
    
    exp_num=$((exp_num + 1))
    
done  # End of for loop through PARAM_LINES array

# Check if loop completed all experiments
LAUNCHED_COUNT=$exp_num
echo ""
echo "Loop completed. Launched $LAUNCHED_COUNT experiments (expected $ACTUAL_COUNT)"
if [ $LAUNCHED_COUNT -lt $ACTUAL_COUNT ]; then
    echo "WARNING: Loop stopped early! Only launched $LAUNCHED_COUNT/$ACTUAL_COUNT"
fi

echo ""
echo "Waiting for all experiments to complete..."
wait

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED / 3600))
ELAPSED_MINS=$(((ELAPSED % 3600) / 60))

echo ""
echo "=========================================="
echo "Execution complete"
echo "Exit code: $EXIT_CODE"
echo "Total time: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"
echo "=========================================="

# Analyze results
echo ""
echo "Analyzing results..."
echo ""

# Check output directories
OUTPUT_DIRS=$(find "$BATCH_OUTPUT_DIR/experiments" -name "17-4PH_P*" -type d 2>/dev/null | wc -l)
echo "Output directories created: $OUTPUT_DIRS"

# Check for key output files
CSV_FILES=$(find "$BATCH_OUTPUT_DIR/experiments" -name "simulation_data.csv" 2>/dev/null | wc -l)
H5_THERMAL=$(find "$BATCH_OUTPUT_DIR/experiments" -name "thermal_fields.h5" 2>/dev/null | wc -l)
H5_ACTIVATION=$(find "$BATCH_OUTPUT_DIR/experiments" -name "activation_volumes.h5" 2>/dev/null | wc -l)

echo "CSV files created: $CSV_FILES"
echo "Thermal HDF5 files: $H5_THERMAL"
echo "Activation HDF5 files: $H5_ACTIVATION"

# Count successes
SUCCESS_COUNT=0
FAIL_COUNT=0
for err_file in "$BATCH_OUTPUT_DIR/results"/*.err; do
    if [ -f "$err_file" ]; then
        # Ignore srun retry messages, only count actual errors
        if grep -q "ERROR\|Traceback\|Exception" "$err_file" 2>/dev/null; then
            FAIL_COUNT=$((FAIL_COUNT + 1))
        else
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    fi
done

echo ""
echo "Successful jobs: $SUCCESS_COUNT / $ACTUAL_COUNT"
echo "Failed jobs: $FAIL_COUNT / $ACTUAL_COUNT"

echo ""
echo "=========================================="
echo "Batch $BATCH_ID Summary"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ] && [ "$SUCCESS_COUNT" -eq "$ACTUAL_COUNT" ]; then
    echo "✓ All experiments completed successfully!"
    echo ""
    echo "Output location: $BATCH_OUTPUT_DIR/experiments"
    echo "Launch log: $BATCH_OUTPUT_DIR/launch.log"
else
    echo "✗ Some experiments failed or incomplete"
    echo ""
    echo "Check logs:"
    echo "  Launch log: $BATCH_OUTPUT_DIR/launch.log"
    echo "  Individual results: $BATCH_OUTPUT_DIR/results/"
    echo ""
    if [ $FAIL_COUNT -gt 0 ]; then
        echo "Failed experiments: $FAIL_COUNT"
        echo "Check .err files in $BATCH_OUTPUT_DIR/results/ for details"
    fi
fi

echo ""
echo "Note: .err files may contain 'nodes busy' messages during queueing."
echo "      These are normal SLURM scheduling messages, not errors."
echo ""
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
