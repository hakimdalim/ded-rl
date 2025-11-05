#!/bin/bash
#SBATCH --job-name=ded_316L
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --mem=0
#SBATCH --array=0-9

# Production submission script for 316L single-track experiments
# Full factorial design: 11×5×5×10 = 2,750 experiments
# Split across 10 job arrays, each handling 275 experiments
# Each job uses 1 node with 24 cores running experiments in parallel

echo "=========================================="
echo "DED 316L Single-Track Production Run"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh
conda activate tft_3_9

# Verify environment
echo ""
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# Check if GNU parallel is available
if ! command -v parallel &> /dev/null; then
    echo "ERROR: GNU Parallel not found!"
    echo "Try: module load parallel"
    exit 1
fi

echo "GNU Parallel found: $(which parallel)"
echo ""

# ============================================================================
# CONFIGURATION: Adjust parallel jobs here
# ============================================================================
PARALLEL_JOBS=12  # Number of experiments to run simultaneously

# Validate configuration
TOTAL_CPUS=$SLURM_CPUS_PER_TASK
if [ $PARALLEL_JOBS -gt $TOTAL_CPUS ]; then
    echo "ERROR: PARALLEL_JOBS ($PARALLEL_JOBS) exceeds available CPUs ($TOTAL_CPUS)"
    echo "Either reduce PARALLEL_JOBS or increase --cpus-per-task in SBATCH directives"
    exit 1
fi

# Calculate CPUs per experiment
MAX_CPU_CORES=$((TOTAL_CPUS / PARALLEL_JOBS))

echo "=========================================="
echo "Resource Allocation"
echo "=========================================="
echo "Total CPUs available: $TOTAL_CPUS"
echo "Parallel jobs: $PARALLEL_JOBS"
echo "CPUs per experiment: $MAX_CPU_CORES"
echo "=========================================="
echo ""

# Configuration
MATERIAL="316L"
BATCH_ID=$SLURM_ARRAY_TASK_ID
EXPERIMENTS_PER_BATCH=275
TOTAL_EXPERIMENTS=2750

# Shared directories
POWDER_STREAM_DIR="/scratch/schuermm-shared/sim_powder_stream_arrays"
OUTPUT_BASE_DIR="/scratch/schuermm-shared/ded_316L_experiments"

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
# Full factorial: Power(11) × Diameter(5) × Feed(5) × Speed(10) = 2,750

PARAM_FILE="$BATCH_OUTPUT_DIR/parameters.txt"
> "$PARAM_FILE"  # Clear file

# Parameter ranges
POWERS=(600 700 800 900 1000 1100 1200 1300 1400 1500 1600)
DIAMETERS=(0.5 0.825 1.15 1.475 1.8)
FEEDS=(2.0 2.5 3.0 3.5 4.0)
SPEEDS=(2 4 6 8 10 12 14 16 18 20)

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

# Run experiments with GNU Parallel
echo "=========================================="
mkdir -p "$BATCH_OUTPUT_DIR/experiments" "$BATCH_OUTPUT_DIR/results"
echo "Starting parallel execution"
echo "Batch size: $ACTUAL_COUNT experiments"
echo "Estimated time: ~$(((ACTUAL_COUNT * 30 / PARALLEL_JOBS / 60))) hours"
echo "=========================================="
echo ""

START_TIME=$(date +%s)

parallel -j $PARALLEL_JOBS --delay 0.2 --colsep '\t' \
    --joblog "$BATCH_OUTPUT_DIR/parallel.log" \
    '
      set -euo pipefail

      # Create meaningful log names with job and experiment info
      EXP_NAME="job_'"$SLURM_ARRAY_JOB_ID"'_task_'"$BATCH_ID"'_{1}_P{2}W_D{3}mm_F{4}gmin_V{5}mms"
      LOG_OUT="'"$BATCH_OUTPUT_DIR"'/results/${EXP_NAME}.out"
      LOG_ERR="'"$BATCH_OUTPUT_DIR"'/results/${EXP_NAME}.err"

      # Run on local storage
      LOCAL_DIR="${TMPDIR:-/tmp}/ded_'"$SLURM_JOB_ID"'_'"$SLURM_ARRAY_TASK_ID"'_${PARALLEL_SEQ}"
      mkdir -p "$LOCAL_DIR"

      python run_single_track_experiment.py -m {1} -p {2} -d {3} -f {4} -v {5} \
        --powder-stream-dir "'"$POWDER_STREAM_DIR"'" \
        --output-dir "$LOCAL_DIR" \
        --max-cpu-cores '"$MAX_CPU_CORES"' \
        > "$LOG_OUT" 2> "$LOG_ERR"

      # Sync results back to shared storage
      rsync -a --partial --inplace "$LOCAL_DIR"/ "'"$BATCH_OUTPUT_DIR"'/experiments/"
      rm -rf "$LOCAL_DIR"
    ' :::: "$PARAM_FILE"

EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED / 3600))
ELAPSED_MINS=$(((ELAPSED % 3600) / 60))

echo ""
echo "=========================================="
echo "Parallel execution complete"
echo "Exit code: $EXIT_CODE"
echo "Total time: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"
echo "=========================================="

# Analyze results
echo ""
echo "Analyzing results..."
echo ""

SUCCESS_COUNT=$(awk 'NR > 1 && $7 == 0' "$BATCH_OUTPUT_DIR/parallel.log" 2>/dev/null | wc -l || echo 0)
echo "Successful jobs: $SUCCESS_COUNT / $ACTUAL_COUNT"

# Check output directories
OUTPUT_DIRS=$(find "$BATCH_OUTPUT_DIR/experiments" -name "316L_P*" -type d 2>/dev/null | wc -l)
echo "Output directories created: $OUTPUT_DIRS"

# Check for key output files
CSV_FILES=$(find "$BATCH_OUTPUT_DIR/experiments" -name "simulation_data.csv" 2>/dev/null | wc -l)
H5_THERMAL=$(find "$BATCH_OUTPUT_DIR/experiments" -name "thermal_fields.h5" 2>/dev/null | wc -l)
H5_ACTIVATION=$(find "$BATCH_OUTPUT_DIR/experiments" -name "activation_volumes.h5" 2>/dev/null | wc -l)

echo "CSV files created: $CSV_FILES"
echo "Thermal HDF5 files: $H5_THERMAL"
echo "Activation HDF5 files: $H5_ACTIVATION"

echo ""
echo "=========================================="
echo "Batch $BATCH_ID Summary"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ] && [ "$SUCCESS_COUNT" -eq "$ACTUAL_COUNT" ]; then
    echo "✓ All experiments completed successfully!"
    echo ""
    echo "Output location: $BATCH_OUTPUT_DIR/experiments"
    echo "Parallel log: $BATCH_OUTPUT_DIR/parallel.log"
else
    echo "✗ Some experiments failed or incomplete"
    echo ""
    echo "Check logs:"
    echo "  Main log: $BATCH_OUTPUT_DIR/parallel.log"
    echo "  Individual results: $BATCH_OUTPUT_DIR/results/"
    echo ""
    echo "Failed experiments:"
    awk 'NR > 1 && $7 != 0' "$BATCH_OUTPUT_DIR/parallel.log"
fi

echo ""
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE