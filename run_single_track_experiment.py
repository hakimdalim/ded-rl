"""
Single-track DED-LB experimental setup with parameter variations.

Supports systematic variation of:
- Base materials: 316L, 17-4PH
- Laser power: 600-1600 W (100 W steps) - 11 variations
- Laser spot diameter: 0.5-1.8 mm (0.325 mm steps) - 5 variations
- Powder mass flow rate: 2.0-4.0 g/min (0.5 g/min steps) - 5 variations
- Scan speed: 2-20 mm/s (2 mm/s steps) - 10 variations
"""

import argparse
import os
import sys


def setup_hpc_environment(max_cpu_cores=None):
    """Configure environment for HPC usage - MUST be called before importing numpy/scipy/etc.

    Args:
        max_cpu_cores: Maximum number of CPU cores to use. If None, uses single-threaded mode.
                      If >0, limits threading to this value.
    """
    force_single_threaded = (max_cpu_cores is None or max_cpu_cores == 1)

    if force_single_threaded:
        thread_value = "1"
    else:
        thread_value = str(max_cpu_cores)

    # Set threading environment variables BEFORE importing numpy/scipy/torch
    thread_limit_vars = {
        "MKL_NUM_THREADS": thread_value,
        "NUMEXPR_NUM_THREADS": thread_value,
        "OMP_NUM_THREADS": thread_value,
        "OPENBLAS_NUM_THREADS": thread_value,
        "VECLIB_MAXIMUM_THREADS": thread_value,
        "NUMBA_NUM_THREADS": thread_value
    }

    for var, value in thread_limit_vars.items():
        os.environ[var] = value

    print(f"\n{'='*80}")
    print("HPC Environment Configuration (set before library imports)")
    print(f"{'='*80}")
    if force_single_threaded:
        print("Threading mode: SINGLE-THREADED (all libraries limited to 1 thread)")
    else:
        print(f"Threading mode: MULTI-THREADED (limited to {max_cpu_cores} threads)")
    for var, value in thread_limit_vars.items():
        print(f"{var}: {value}")
    print(f"{'='*80}\n")


# Parse args early to get max_cpu_cores before importing heavy libraries
if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--max-cpu-cores', type=int, default=None)
    early_args, _ = parser.parse_known_args()
    setup_hpc_environment(max_cpu_cores=early_args.max_cpu_cores)


# NOW import the heavy libraries that use threading
import multiprocessing
from pathlib import Path
from datetime import datetime
from simulate import SimulationRunner
from configuration.material_manager import MaterialManager
from callbacks.completion_callbacks import TrackCountCompletionCallback
from callbacks.step_data_collector import StepDataCollector
from callbacks.callback_collection import ProgressPrinter, FinalStateSaver
from callbacks.hdf5_thermal_saver import HDF5ThermalSaver
from callbacks.hdf5_activation_saver import HDF5ActivationSaver
from powder.powder_stream_new import VoxelPowderStream


# Try to configure torch after import if available
try:
    import torch
    max_cpu = int(os.environ.get('OMP_NUM_THREADS', '1'))
    if torch.get_num_threads() != max_cpu:
        torch.set_num_threads(max_cpu)
    if torch.get_num_interop_threads() != max_cpu:
        torch.set_num_interop_threads(max_cpu)
except ImportError:
    pass  # PyTorch not available


def print_cpu_diagnostics():
    """Print actual CPU usage configuration from libraries"""
    print(f"\n{'='*80}")
    print("CPU Configuration Diagnostics")
    print(f"{'='*80}")

    # Check NumPy
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        # Try to get actual thread count being used
        try:
            import threadpoolctl
            with threadpoolctl.threadpool_info() as info:
                for pool in info:
                    print(f"  {pool.get('user_api', 'unknown')}: {pool.get('num_threads', 'unknown')} threads")
        except ImportError:
            print("  (install threadpoolctl for detailed thread info)")
    except Exception as e:
        print(f"NumPy check failed: {e}")

    # Check Numba
    try:
        import numba
        print(f"Numba version: {numba.__version__}")
        print(f"  NUMBA_NUM_THREADS: {os.environ.get('NUMBA_NUM_THREADS', 'not set')}")
        print(f"  Numba config num_threads: {numba.config.NUMBA_NUM_THREADS}")
    except Exception as e:
        print(f"Numba check failed: {e}")

    # Check process affinity
    try:
        import psutil
        p = psutil.Process()
        affinity = p.cpu_affinity()
        print(f"Process CPU affinity: {affinity} ({len(affinity)} CPUs)")
    except ImportError:
        print("(install psutil to check CPU affinity)")
    except Exception as e:
        print(f"Affinity check failed: {e}")

    print(f"{'='*80}\n")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Single-track DED-LB experiment with parameter variations'
    )

    # Material
    parser.add_argument('--material', '-m', default='316L',
                       choices=['316L', '17-4PH'],
                       help='Base material')

    # Process parameters
    parser.add_argument('--laser-power', '-p', type=float, default=1000.0,
                       help='Laser power in W (600-1600, step 100)')
    parser.add_argument('--spot-diameter', '-d', type=float, default=1.15,
                       help='Laser spot diameter in mm (0.5-1.8, step 0.325)')
    parser.add_argument('--powder-feed', '-f', type=float, default=3.0,
                       help='Powder feed rate in g/min (2.0-4.0, step 0.5)')
    parser.add_argument('--scan-speed', '-v', type=float, default=20.0,
                       help='Scan speed in mm/s (2-20, step 2)')

    # Build configuration
    parser.add_argument('--build-volume-mm', type=float, nargs=3,
                       default=[10.0, 80.0, 10.0],
                       help='Build volume (x y z) in mm')
    parser.add_argument('--voxel-size-um', type=float, default=200.0,
                       help='Voxel size in micrometers')
    parser.add_argument('--delta-t-ms', type=float, default=200.0,
                       help='Time step in milliseconds')

    # Track configuration
    parser.add_argument('--track-length-mm', type=float, default=70.0,
                       help='Track length in mm')
    parser.add_argument('--hatch-spacing-um', type=float, default=700.0,
                       help='Hatch spacing in micrometers')
    parser.add_argument('--layer-spacing-um', type=float, default=350.0,
                       help='Layer spacing in micrometers')
    parser.add_argument('--substrate-height-mm', type=float, default=5.0,
                       help='Substrate height in mm')

    # Gas flows
    parser.add_argument('--carrier-gas-L-min', type=float, default=6.0,
                       help='Carrier gas flow in L/min')
    parser.add_argument('--shield-gas-L-min', type=float, default=8.0,
                       help='Shield gas flow in L/min')

    # Powder stream
    parser.add_argument('--powder-stream-dir', type=str, default='powder_stream_arrays',
                       help='Directory containing powder stream npz files')

    # Output
    parser.add_argument('--output-dir', type=str, default='_single_track_experiments',
                       help='Base output directory')

    parser.add_argument('--experiment-id', type=str, default=None,
                       help='Experiment ID prefix (e.g., exp_0) to prepend to directory name')

    # HPC configuration
    parser.add_argument('--max-cpu-cores', type=int, default=None,
                       help='Maximum CPU cores to use (None=single-threaded, >0=multi-threaded with limit)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Print CPU diagnostics
    print_cpu_diagnostics()

    # Create experiment label with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_label = (f"{args.material}_P{int(args.laser_power)}W_"
                f"D{args.spot_diameter:.2f}mm_"
                f"F{args.powder_feed:.1f}gmin_"
                f"V{args.scan_speed:.0f}mms_"
                f"{timestamp}")

    # Prepend experiment ID if provided
    if args.experiment_id is not None:
        exp_label = f"{args.experiment_id}_{exp_label}"

    print(f"\n{'='*80}")
    print(f"Single-Track Experiment: {exp_label}")
    print(f"{'='*80}")
    print(f"Material: {args.material}")
    print(f"Laser Power: {args.laser_power} W")
    print(f"Spot Diameter: {args.spot_diameter} mm")
    print(f"Powder Feed: {args.powder_feed} g/min")
    print(f"Scan Speed: {args.scan_speed} mm/s")
    print(f"{'='*80}\n")

    # Initialize powder stream from npz file FIRST to get detected nozzle height
    powder_stream_path = Path(args.powder_stream_dir) / f"{args.material}_only_{args.powder_feed}gmin_stream.npz"
    metadata_path = powder_stream_path.parent / f"{powder_stream_path.stem}_cropped.json"
    pickle_path = powder_stream_path.parent / f"{powder_stream_path.stem}_voxel_stream.pkl"

    if not powder_stream_path.exists():
        raise FileNotFoundError(
            f"Powder stream file not found: {powder_stream_path}\n"
            f"Expected format: {{material}}_only_{{feed_rate}}gmin_stream.npz"
        )

    print(f"Loading powder stream: {powder_stream_path}")

    # Start timing powder stream processing
    import time
    import pickle
    powder_start_time = time.time()

    # Try to load from pickle first
    voxel_powder_stream = None
    if pickle_path.exists():
        try:
            print(f"Found pickled VoxelPowderStream: {pickle_path}")
            with open(pickle_path, 'rb') as f:
                voxel_powder_stream = pickle.load(f)
            print("Successfully loaded VoxelPowderStream from pickle")
        except Exception as e:
            print(f"Warning: Could not load pickle: {e}")
            print("Will create new VoxelPowderStream instance")
            voxel_powder_stream = None

    # Create new instance if pickle load failed or doesn't exist
    if voxel_powder_stream is None:
        print("Creating new VoxelPowderStream instance...")

        # Check for cached metadata
        cached_metadata = None
        if metadata_path.exists():
            try:
                import json
                with open(metadata_path, 'r') as f:
                    cached_metadata = json.load(f)
                print(f"Found cached metadata: {metadata_path}")
            except Exception as e:
                print(f"Warning: Could not load metadata cache: {e}")
                cached_metadata = None

        voxel_powder_stream = VoxelPowderStream(
            voxel_path=str(powder_stream_path),
            nozzle_offset_slices=50,  # Distance from array top to nozzle in slices
            auto_detect_working_distance=True,
            cached_metadata=cached_metadata,  # Pass cached data if available
            visualize=False
        )

        # Save metadata if it was just calculated
        if cached_metadata is None:
            try:
                import json
                metadata = voxel_powder_stream.get_metadata()
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Saved metadata cache: {metadata_path}")
            except Exception as e:
                print(f"Warning: Could not save metadata cache: {e}")

        # Save the VoxelPowderStream object as pickle
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(voxel_powder_stream, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved VoxelPowderStream pickle: {pickle_path}")
        except Exception as e:
            print(f"Warning: Could not save pickle: {e}")

    powder_end_time = time.time()
    powder_processing_time = powder_end_time - powder_start_time
    print(f"\n{'=' * 80}")
    print(f"Powder stream processing time: {powder_processing_time:.2f} seconds")
    print(f"{'=' * 80}\n")

    # Use the detected nozzle height from the powder stream
    detected_nozzle_height = voxel_powder_stream.get_detected_nozzle_height()
    print(f"Using detected nozzle height: {detected_nozzle_height * 1000:.3f} mm\n")

    # Setup callbacks
    callbacks = [
        TrackCountCompletionCallback(max_tracks=1),
        StepDataCollector(save_path=None),
        HDF5ThermalSaver(
            filename="thermal_fields.h5",
            interval=1,
            compression='gzip',
            compression_opts=4
        ),
        HDF5ActivationSaver(
            filename="activation_volumes.h5",
            interval=1,
            compression='gzip',
            compression_opts=9
        ),
        StepDataCollector(tracked_fields=None, save_path="simulation_data.csv"),
        ProgressPrinter(),
        FinalStateSaver(),
    ]

    # Create MaterialManager with material and process parameters
    feeder_percent = args.powder_feed / MaterialManager.FEED_SLOPE_G_PER_MIN_PER_PERCENT

    params = MaterialManager.from_defaults()
    params.request_change(
        material=args.material,
        feeder_percent=feeder_percent,
        carrier_gas_L_min=args.carrier_gas_L_min,
        shield_gas_L_min=args.shield_gas_L_min,
        laser_power=args.laser_power,
        laser_radius=args.spot_diameter / 2000.0,  # mm to m, diameter to radius
        beam_waist_radius=args.spot_diameter / 2000.0,  # same as laser_radius
        beam_waist_position=0.0,  # focal plane at substrate (makes calc simple)
        scan_speed=args.scan_speed / 1000.0,  # mm/s to m/s
        nozzle_height=detected_nozzle_height,  # Use detected value from powder stream
    )

    # Part dimensions for single track (width × length × height)
    part_volume_mm = (
        args.hatch_spacing_um / 1000.0,  # One track width
        args.track_length_mm,
        args.layer_spacing_um / 1000.0   # One layer
    )

    # Create runner using from_si_units (more control over params)
    from configuration.simulation_config import SimulationConfig
    from core.multi_track_multi_layer import MultiTrackMultiLayerSimulation

    # Convert to SI
    build_volume = tuple(x / 1000 for x in args.build_volume_mm)
    part_volume = tuple(x / 1000 for x in part_volume_mm)
    voxel_size = args.voxel_size_um / 1e6
    delta_t = args.delta_t_ms / 1000
    hatch_spacing = args.hatch_spacing_um / 1e6
    layer_spacing = args.layer_spacing_um / 1e6
    substrate_height = args.substrate_height_mm / 1000

    # Create output directory
    output_dir = Path(args.output_dir) / exp_label
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create simulation configuration
    config = SimulationConfig(
        build_volume_size=build_volume,
        voxel_size=voxel_size,
        part_width=part_volume[0],
        part_length=part_volume[1],
        part_height=part_volume[2],
        hatch_spacing=hatch_spacing,
        layer_spacing=layer_spacing,
        substrate_height=substrate_height,
        bidirectional_tracks=True,
        bidirectional_layers=True,
        switch_scan_direction_between_layers=True,
        turnaround_time=0.0,
    )

    # Create simulation instance
    simulation = MultiTrackMultiLayerSimulation(
        config=config.get_simulation_config(),
        delta_t=delta_t,
        callbacks=callbacks,
        output_dir=output_dir,
        powder_concentration_func=voxel_powder_stream.powder_concentration,
    )
    simulation.reset()

    # Create runner with MaterialManager
    runner = SimulationRunner(simulation, config, params)

    # Run simulation with timing
    print("Starting simulation...\n")
    simulation_start_time = time.time()

    runner.run()

    simulation_end_time = time.time()
    simulation_time = simulation_end_time - simulation_start_time

    print(f"\n{'='*80}")
    print(f"Simulation execution time: {simulation_time:.2f} seconds")
    print(f"{'='*80}")
    print(f"\n{'='*80}")
    print("Experiment completed")
    print(f"{'='*80}")
    print(f"Total powder stream processing time: {powder_processing_time:.2f} seconds")
    print(f"Total simulation execution time: {simulation_time:.2f} seconds")
    print(f"Total experiment time: {powder_processing_time + simulation_time:.2f} seconds")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()