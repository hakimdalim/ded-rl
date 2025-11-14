#!/usr/bin/env python
"""
Simulation script with ALL callbacks enabled (for SLURM batch jobs).

This script is specifically designed for the SLURM job submission system.
It includes:
- Original callbacks (HeightCompletionCallback, ProgressPrinter, etc.)
- NEW HDF5ThermalSaver (compressed thermal field storage)
- NEW HDF5ActivationSaver (compressed activation volume storage)
- NEW PerspectiveCameraCallback (following camera with nozzle + powder overlay)

USAGE (on SLURM):
    python testing/simulate_with_all_callbacks.py \\
        --part-x 5.0 --part-y 5.0 --part-z 2.0 \\
        --laser-power 800 --scan-speed 5.0 \\
        --exp-label ded_doe_v6

This script is called by the SLURM submission scripts to run parameter sweeps.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import SimulationRunner
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulate import SimulationRunner

# Import callbacks
from callbacks.callback_collection import (
    get_default_callbacks,
    ProgressPrinter,
    FinalStateSaver,
    ThermalPlotSaver,
)
from callbacks.completion_callbacks import HeightCompletionCallback
from callbacks.step_data_collector import StepDataCollector

# Import NEW callbacks
from callbacks.hdf5_thermal_saver import HDF5ThermalSaver
from callbacks.hdf5_activation_saver import HDF5ActivationSaver
from callbacks.perspective_camera_callback import PerspectiveCameraCallback


def create_all_callbacks(save_interval: int = 5):
    """
    Create callback collection with ALL callbacks (original + new).

    Args:
        save_interval: How often to save (in steps). Default: 5

    Returns:
        List of callback instances
    """
    # Start with a minimal set of original callbacks
    # (We don't use get_default_callbacks() to have more control)
    callbacks = [
        # Completion callback (must have one)
        HeightCompletionCallback(),

        # Progress monitoring
        ProgressPrinter(),

        # Data collection
        StepDataCollector(
            tracked_fields=['position', 'melt_pool', 'build', 'clad'],
            save_path="simulation_data.csv"
        ),

        # Final state saving
        FinalStateSaver(),

        # Thermal plots (original visualization)
        ThermalPlotSaver(
            save_dir="thermal_plots",
            interval=save_interval
        ),
    ]

    # Add NEW HDF5 callbacks
    callbacks.extend([
        # HDF5 Thermal Field Saver (complete 3D thermal history)
        HDF5ThermalSaver(
            filename="thermal_fields.h5",
            interval=save_interval,
            compression='gzip',
            compression_opts=4  # Moderate compression for speed
        ),

        # HDF5 Activation Volume Saver (complete 3D activation history)
        HDF5ActivationSaver(
            filename="activation_volumes.h5",
            interval=save_interval,
            compression='gzip',
            compression_opts=9  # High compression (boolean data compresses well)
        ),

        # Perspective Camera Callback (following camera with overlay)
        PerspectiveCameraCallback(
            rel_offset_local=(0.0, -0.12, 0.04),  # Camera position relative to nozzle
            floor_angle_deg=30.0,                  # Look down at 30 degrees
            save_images=True,
            save_dir="cam",                        # IMPORTANT: Will create this directory
            interval=save_interval,
            resolution_wh=(800, 600),
            enable_overlay=True,
            overlay_config={
                # V-shaped powder stream configuration
                'stream_height_mm': 15.0,          # 13-16mm typical range
                'v_angle_deg': 15.0,               # V-opening angle
                'num_particles': 600,              # Particle count
                'gaussian_sigma_ratio': 0.25,      # Gaussian distribution

                # Nozzle geometry
                'nozzle_outlet_radius_mm': 4.0,
                'nozzle_top_radius_mm': 10.0,
                'nozzle_height_mm': 40.0,

                # Visual settings (optimized for visibility)
                'render_mode': 'schematic',
                'schematic_bg_color': (200, 200, 200),
                'show_substrate_line': True,
                'substrate_line_color': (255, 0, 0),
                'particle_color': (50, 50, 50),
                'nozzle_fill_color': (60, 100, 160),
                'show_v_cone': True,
            }
        ),
    ])

    return callbacks


def main():
    """Main function for SLURM execution."""
    parser = argparse.ArgumentParser(
        description='Run DED simulation with ALL callbacks (for SLURM parameter sweeps)'
    )

    # Build volume parameters
    parser.add_argument('--build-x', type=float, default=20.0,
                       help='Build volume X dimension in mm (default: 20.0)')
    parser.add_argument('--build-y', type=float, default=20.0,
                       help='Build volume Y dimension in mm (default: 20.0)')
    parser.add_argument('--build-z', type=float, default=15.0,
                       help='Build volume Z dimension in mm (default: 15.0)')

    # Part volume parameters
    parser.add_argument('--part-x', type=float, default=5.0,
                       help='Part X dimension in mm (default: 5.0)')
    parser.add_argument('--part-y', type=float, default=5.0,
                       help='Part Y dimension in mm (default: 5.0)')
    parser.add_argument('--part-z', type=float, default=5.0,
                       help='Part Z dimension in mm (default: 5.0)')

    # Process parameters
    parser.add_argument('--voxel-size', type=float, default=200.0,
                       help='Voxel size in micrometers (default: 200.0)')
    parser.add_argument('--delta-t', type=float, default=200.0,
                       help='Time step in milliseconds (default: 200.0)')
    parser.add_argument('--scan-speed', type=float, default=3.0,
                       help='Scan speed in mm/s (default: 3.0)')
    parser.add_argument('--laser-power', type=float, default=600.0,
                       help='Laser power in W (default: 600.0)')
    parser.add_argument('--powder-feed', type=float, default=2.0,
                       help='Powder feed rate in g/min (default: 2.0)')

    # Callback parameters
    parser.add_argument('--save-interval', type=int, default=5,
                       help='Save interval for callbacks in steps (default: 5)')

    # Output control
    parser.add_argument('--exp-label', type=str, default="unlabeled",
                       help='Experiment label for output directory (default: unlabeled)')

    args = parser.parse_args()

    # Print configuration
    print("\n" + "="*80)
    print("DED SIMULATION WITH ALL CALLBACKS")
    print("="*80)
    print("\nParameters:")
    print(f"  Build volume: {args.build_x}x{args.build_y}x{args.build_z} mm")
    print(f"  Part volume: {args.part_x}x{args.part_y}x{args.part_z} mm")
    print(f"  Voxel size: {args.voxel_size} μm")
    print(f"  Time step: {args.delta_t} ms")
    print(f"  Scan speed: {args.scan_speed} mm/s")
    print(f"  Laser power: {args.laser_power} W")
    print(f"  Powder feed: {args.powder_feed} g/min")
    print(f"  Save interval: {args.save_interval} steps")
    print(f"  Experiment label: {args.exp_label}")
    print("\nCallbacks enabled:")
    print("  ✓ HeightCompletionCallback")
    print("  ✓ ProgressPrinter")
    print("  ✓ StepDataCollector")
    print("  ✓ FinalStateSaver")
    print("  ✓ ThermalPlotSaver (original)")
    print("  ✓ HDF5ThermalSaver (NEW)")
    print("  ✓ HDF5ActivationSaver (NEW)")
    print("  ✓ PerspectiveCameraCallback (NEW)")
    print("="*80 + "\n")

    # Create callbacks with specified interval
    callbacks = create_all_callbacks(save_interval=args.save_interval)

    # Create runner with human units
    runner = SimulationRunner.from_human_units(
        build_volume_mm=(args.build_x, args.build_y, args.build_z),
        part_volume_mm=(args.part_x, args.part_y, args.part_z),
        voxel_size_um=args.voxel_size,
        delta_t_ms=args.delta_t,
        scan_speed_mm_s=args.scan_speed,
        laser_power_W=args.laser_power,
        powder_feed_g_min=args.powder_feed,
        experiment_label=args.exp_label,
        callbacks=callbacks
    )

    # Run simulation
    print("Starting simulation...")
    print("-"*80)
    runner.run()

    # Print completion information
    print("\n" + "="*80)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results saved to: {runner.simulation.output_dir}")
    print("\nExpected outputs:")
    print("  ✓ simulation_data.csv")
    print("  ✓ thermal_plots/ (PNG images)")
    print("  ✓ thermal_fields.h5 (compressed thermal history)")
    print("  ✓ activation_volumes.h5 (compressed activation history)")
    print("  ✓ cam/ (perspective camera images)")
    print("  ✓ final_*.npy (final state files)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
