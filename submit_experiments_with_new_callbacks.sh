#!/bin/bash

# ==============================================================================
# SLURM Job Submission Script - WITH NEW CALLBACKS
# ==============================================================================
# This script submits parameter sweep jobs that include:
#   - Original callbacks (CSV, plots, etc.)
#   - NEW HDF5ThermalSaver (compressed thermal history)
#   - NEW HDF5ActivationSaver (compressed activation history)
#   - NEW PerspectiveCameraCallback (following camera with overlay)
#
# Usage: bash submit_experiments_with_new_callbacks.sh
# ==============================================================================

# === Basic Configuration ===
experiment_label=ded_doe_v6
memory=3G
partition=normal
cpus=1
run_time=10:00:00
jobs_per_array=24

# Define parameter ranges and steps
laser_power_min=600
laser_power_max=1600
laser_power_step=100

scan_speed_min=2.0
scan_speed_max=20.0
scan_speed_step=2.0

powder_feed_min=2.0
powder_feed_max=4.0
powder_feed_step=0.5

# Constants
voxel_size=200.0  # micrometers
delta_t=200.0     # milliseconds
build_x=20.0      # mm
build_y=20.0      # mm
build_z=15.0      # mm

# Callback configuration
save_interval=5   # How often to save (in steps)

# Function to create directory name matching Python script
create_safe_dirname() {
    local build_x=$1
    local build_y=$2
    local build_z=$3
    local part_x=$4
    local part_y=$5
    local part_z=$6
    local voxel_size=$7
    local delta_t=$8
    local scan_speed=$9
    local laser_power=${10}
    local powder_feed=${11}

    # Format build and part dimensions with one decimal point
    local build_str="$(printf "%.1fx%.1fx%.1f" $build_x $build_y $build_z)mm"
    local part_str="$(printf "%.1fx%.1fx%.1f" $part_x $part_y $part_z)mm"

    # Format process parameters with one decimal point
    local voxel_str=$(printf "%.1f" $voxel_size)
    local dt_str=$(printf "%.1f" $delta_t)
    local speed_str=$(printf "%.1f" $scan_speed)
    local power_str=$(printf "%.1f" $laser_power)
    local feed_str=$(printf "%.1f" $powder_feed)

    echo "build${build_str}_part${part_str}_vox${voxel_str}um_dt${dt_str}ms_v${speed_str}mms_p${power_str}W_f${feed_str}gmin"
}

# Function to create and submit SLURM job array
submit_job_array() {
    local param_array=("${!1}")    # First argument: parameter array
    local dirname_array=("${!2}")  # Second argument: directory names array
    local array_size=$3            # Third argument: size of the array

    cat << EOF > submit_array.sh
#!/bin/bash
#SBATCH -J ded_array_enhanced
#SBATCH -t ${run_time}
#SBATCH --mail-type=NONE
#SBATCH --mem=${memory}
#SBATCH -p ${partition}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c ${cpus}

# ===========================================================================
# Load required modules
# ===========================================================================
module load anaconda3/latest
. \$ANACONDA_HOME/etc/profile.d/conda.sh
module load cudnn
module load nvidia
module load ffmpeg

# ===========================================================================
# CRITICAL: Clear PYTHONPATH to avoid conflicts
# ===========================================================================
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# ===========================================================================
# Activate conda environment from scratch filesystem
# ===========================================================================
conda activate /scratch/dalim/miniconda3/envs/ded_scratch_py_3_10

# ===========================================================================
# Parse job parameters
# ===========================================================================
# Arrays of parameters and directories
declare -a param_sets=(${param_array[@]@Q})
declare -a dir_names=(${dirname_array[@]@Q})

# Get parameters for this task (from SLURM_ARRAY_TASK_ID)
read part_x part_y part_z laser_power scan_speed powder_feed <<< "\${param_sets[\$SLURM_ARRAY_TASK_ID]}"
dirname="\${dir_names[\$SLURM_ARRAY_TASK_ID]}"

# ===========================================================================
# Setup output directory
# ===========================================================================
output_dir="\${SCRATCH}/ded_sim_experiments/${experiment_label}/job\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_\${dirname}"
mkdir -p "\$output_dir"

# Redirect stdout/stderr to files
exec 1> "\$output_dir/\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.out"
exec 2> "\$output_dir/\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}.err"

# ===========================================================================
# Print job information
# ===========================================================================
echo "=========================================="
echo "DED-LB Simulation - WITH NEW CALLBACKS"
echo "=========================================="
echo "Job ID: \${SLURM_ARRAY_JOB_ID}"
echo "Task ID: \${SLURM_ARRAY_TASK_ID}"
echo "Node: \${SLURM_NODELIST}"
echo "Output: \$output_dir"
echo ""
echo "Parameters:"
echo "  Part: \${part_x}x\${part_y}x\${part_z} mm"
echo "  Laser power: \${laser_power} W"
echo "  Scan speed: \${scan_speed} mm/s"
echo "  Powder feed: \${powder_feed} g/min"
echo ""
echo "Callbacks enabled:"
echo "  ✓ Original callbacks (CSV, plots, etc.)"
echo "  ✓ HDF5ThermalSaver (compressed thermal)"
echo "  ✓ HDF5ActivationSaver (compressed activation)"
echo "  ✓ PerspectiveCameraCallback (camera + overlay)"
echo "=========================================="
echo ""

# ===========================================================================
# Change to simulation directory
# ===========================================================================
cd /home/dalim/masterarbeit/git/hypo-simulations || exit 1

# ===========================================================================
# Build command line arguments
# ===========================================================================
args="--part-x \$part_x --part-y \$part_y --part-z \$part_z"
args+=" --laser-power \$laser_power"
args+=" --scan-speed \$scan_speed"
args+=" --powder-feed \$powder_feed"
args+=" --exp-label ${experiment_label}"
args+=" --save-interval ${save_interval}"

# ===========================================================================
# Run simulation with NEW callbacks
# ===========================================================================
# OPTION 1: Use testing/simulate_with_all_callbacks.py (recommended for SLURM)
python -u testing/simulate_with_all_callbacks.py \${args}

# OPTION 2: Use _simulate.py (alternative, uncomment to use)
# python -u _simulate.py \${args}

exit_code=\$?

# ===========================================================================
# Verify outputs
# ===========================================================================
echo ""
echo "=========================================="
if [ \$exit_code -eq 0 ]; then
    echo "Simulation completed successfully"
    echo ""
    echo "Checking for expected outputs..."

    # Original outputs
    [ -f "\$output_dir/simulation_data.csv" ] && echo "  ✓ CSV data" || echo "  ✗ CSV data missing"
    [ -d "\$output_dir/thermal_plots" ] && echo "  ✓ Thermal plots" || echo "  ✗ Thermal plots missing"
    [ -f "\$output_dir/clad_manager.pkl" ] && echo "  ✓ Pickle" || echo "  ✗ Pickle missing"

    # NEW outputs
    [ -f "\$output_dir/thermal_fields.h5" ] && echo "  ✓ HDF5 thermal" || echo "  ✗ HDF5 thermal missing"
    [ -f "\$output_dir/activation_volumes.h5" ] && echo "  ✓ HDF5 activation" || echo "  ✗ HDF5 activation missing"
    [ -d "\$output_dir/cam" ] && echo "  ✓ Camera images" || echo "  ✗ Camera images missing"

else
    echo "Simulation failed with exit code: \$exit_code"
fi
echo "=========================================="

exit \$exit_code
EOF

    # ===========================================================================
    # Submit the job array to SLURM
    # ===========================================================================
    sbatch --array=0-$((array_size-1)) submit_array.sh
}

# ==============================================================================
# Main execution: Generate and submit job arrays
# ==============================================================================

echo "=========================================="
echo "DED-LB DoE Parameter Sweep Submission"
echo "WITH NEW CALLBACKS ENABLED"
echo "=========================================="
echo "Experiment label: $experiment_label"
echo "Output directory: \$SCRATCH/ded_sim_experiments/$experiment_label/"
echo ""
echo "Configuration:"
echo "  Memory per job: $memory"
echo "  CPUs per job: $cpus"
echo "  Time limit: $run_time"
echo "  Jobs per array: $jobs_per_array"
echo "  Save interval: $save_interval steps"
echo ""
echo "Parameter ranges:"
echo "  Laser power: $laser_power_min-$laser_power_max W (step: $laser_power_step)"
echo "  Scan speed: $scan_speed_min-$scan_speed_max mm/s (step: $scan_speed_step)"
echo "  Powder feed: $powder_feed_min-$powder_feed_max g/min (step: $powder_feed_step)"
echo ""
echo "NEW callbacks included:"
echo "  ✓ HDF5ThermalSaver (compressed thermal field history)"
echo "  ✓ HDF5ActivationSaver (compressed activation volume history)"
echo "  ✓ PerspectiveCameraCallback (camera with nozzle overlay)"
echo "=========================================="
echo ""

# Define test case dimensions (in mm)
declare -A test_cases=(
    ["train"]="x=5.0 y=5.0 z=5.0"
    ["validation"]="x=10.0 y=5.0 z=5.0"
    ["test"]="x=10.0 y=5.0 z=8.0"
)

# For each test case
for test_case in "${!test_cases[@]}"; do
    echo "Processing test case: $test_case"
    eval "${test_cases[$test_case]}"
    echo "  Part dimensions: ${x}x${y}x${z} mm"

    # Create arrays to store parameters and directories for this test case
    declare -a param_sets=()
    declare -a dir_names=()
    index=0

    # Generate all combinations
    for laser_power in $(seq $laser_power_min $laser_power_step $laser_power_max); do
        for scan_speed in $(seq $scan_speed_min $scan_speed_step $scan_speed_max); do
            for powder_feed in $(seq $powder_feed_min $powder_feed_step $powder_feed_max); do
                # Store parameters as a space-separated string
                param_sets[$index]="$x $y $z $laser_power $scan_speed $powder_feed"

                # Generate and store directory name
                dir_names[$index]=$(create_safe_dirname $build_x $build_y $build_z $x $y $z \
                    $voxel_size $delta_t $scan_speed $laser_power $powder_feed)

                ((index++))

                # If we have enough jobs for an array, submit it
                if [ $index -eq $jobs_per_array ]; then
                    submit_job_array param_sets[@] dir_names[@] $index

                    # Reset arrays and index for next batch
                    param_sets=()
                    dir_names=()
                    index=0
                    echo "  ✓ Submitted batch of $jobs_per_array jobs"
                    sleep 1  # Avoid overwhelming the scheduler
                fi
            done
        done
    done

    # Submit remaining jobs if any
    if [ $index -gt 0 ]; then
        submit_job_array param_sets[@] dir_names[@] $index
        echo "  ✓ Submitted final batch of $index jobs"
    fi

    echo ""
done

# Clean up temporary script
rm -f submit_array.sh

echo "=========================================="
echo "All job arrays submitted successfully!"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check results in: \$SCRATCH/ded_sim_experiments/$experiment_label/"
echo ""
echo "Expected outputs per job:"
echo "  - simulation_data.csv"
echo "  - thermal_plots/"
echo "  - clad_manager.pkl"
echo "  - thermal_fields.h5 (NEW)"
echo "  - activation_volumes.h5 (NEW)"
echo "  - cam/ (NEW)"
echo ""
