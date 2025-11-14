"""
DED Simulation Live Plotter Demo
Choose between Simple or Advanced live visualization.
"""

import argparse
import matplotlib
matplotlib.use('TkAgg')  # Ensure interactive backend

from simulate import SimulationRunner
from callbacks.live_plotter_callback import AdvancedLivePlotter
from callbacks.callback_collection import (
    HeightCompletionCallback,
    StepDataCollector,
    ProgressPrinter,
    LivePlotter
)


def run_with_simple_plotter():
    """Run simulation with simple live plotter (build height only)."""
    print("\n" + "="*60)
    print("  SIMPLE Live Plotter - Build Height Progress")
    print("="*60)

    callbacks = [
        HeightCompletionCallback(),
        StepDataCollector(save_path="simulation_data.csv"),
        ProgressPrinter(),
        LivePlotter(interval=5)
    ]

    runner = SimulationRunner.from_human_units(
        part_volume_mm=(3.0, 3.0, 1.5),
        voxel_size_um=200.0,
        laser_power_W=600.0,
        scan_speed_mm_s=3.0,
        experiment_label="simple_live_demo",
        callbacks=callbacks
    )

    print("\nShowing: Build height vs time step")
    print("Update frequency: Every 5 steps")
    print("Overhead: <1%\n")

    runner.run()
    print(f"\nResults: {runner.simulation.output_dir}")


def run_with_advanced_plotter():
    """Run simulation with advanced live plotter (full visualization)."""
    print("\n" + "="*60)
    print("  ADVANCED Live Plotter - Full Thermal Visualization")
    print("="*60)

    callbacks = [
        HeightCompletionCallback(),
        StepDataCollector(save_path="simulation_data.csv"),
        ProgressPrinter(),
        AdvancedLivePlotter(
            interval=5,
            temp_range=(300, 2500),
            figsize=(20, 12),
            enabled=True
        )
    ]

    runner = SimulationRunner.from_human_units(
        part_volume_mm=(3.0, 3.0, 1.5),
        voxel_size_um=200.0,
        laser_power_W=600.0,
        scan_speed_mm_s=3.0,
        experiment_label="advanced_live_demo",
        callbacks=callbacks
    )

    print("\nShowing: 4 plots")
    print("  - Top view (XY): Temperature at current height")
    print("  - Front view (XZ): Vertical temperature slice")
    print("  - Side view (YZ): Longitudinal temperature slice")
    print("  - Cross-section: All layer profiles")
    print("\nUpdate frequency: Every 5 steps")
    print("Overhead: ~5-10%")
    print("\nLegend:")
    print("  White dashed = Activated volume")
    print("  Cyan = Melt pool boundary\n")

    runner.run()
    print(f"\nResults: {runner.simulation.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Run DED simulation with live plotting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple live plotter (build height only)
  python run_with_live_plot_demo.py --mode simple

  # Advanced live plotter (full thermal visualization)
  python run_with_live_plot_demo.py --mode advanced

  # No live plotting (fastest)
  python run_with_live_plot_demo.py --mode none
        """
    )

    parser.add_argument(
        '--mode',
        choices=['simple', 'advanced', 'none'],
        default='simple',
        help='Live plotter mode (default: simple)'
    )

    args = parser.parse_args()

    if args.mode == 'simple':
        run_with_simple_plotter()
    elif args.mode == 'advanced':
        run_with_advanced_plotter()
    else:
        print("\n" + "="*60)
        print("  No Live Plotter - Maximum Performance")
        print("="*60)

        callbacks = [
            HeightCompletionCallback(),
            StepDataCollector(save_path="simulation_data.csv"),
            ProgressPrinter()
        ]

        runner = SimulationRunner.from_human_units(
            part_volume_mm=(3.0, 3.0, 1.5),
            voxel_size_um=200.0,
            laser_power_W=600.0,
            scan_speed_mm_s=3.0,
            experiment_label="no_live_plot",
            callbacks=callbacks
        )

        print("\nNo live visualization - fastest runtime\n")
        runner.run()
        print(f"\nResults: {runner.simulation.output_dir}")


if __name__ == "__main__":
    main()
