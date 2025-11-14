"""
DED Simulation with Simple Live Plotter
Shows real-time build height progress.
"""

from simulate import SimulationRunner
from callbacks.callback_collection import (
    HeightCompletionCallback,
    StepDataCollector,
    ProgressPrinter,
    LivePlotter  # Simple live plotter
)

def main():
    print("\n" + "="*60)
    print("  DED Simulation with SIMPLE Live Progress Plot")
    print("="*60)

    # Configure callbacks
    callbacks = [
        HeightCompletionCallback(),
        StepDataCollector(save_path="simulation_data.csv"),
        ProgressPrinter(),

        # SIMPLE LIVE PLOTTER - shows build height vs time
        LivePlotter(interval=5)  # Update every 5 steps
    ]

    # Create simulation runner with quick test parameters
    runner = SimulationRunner.from_human_units(
        # Small part for quick demo
        part_volume_mm=(3.0, 3.0, 1.5),

        # Medium resolution
        voxel_size_um=200.0,
        delta_t_ms=200.0,

        # Process parameters
        laser_power_W=600.0,
        scan_speed_mm_s=3.0,
        powder_feed_g_min=2.0,

        # Output
        experiment_label="simple_live_plot_demo",

        # Attach callbacks
        callbacks=callbacks
    )

    print("\nSimulation will start shortly...")
    print("A plot window will appear showing build height progress.")
    print("\nRunning...\n")

    # Run simulation
    runner.run()

    print("\n" + "="*60)
    print("  Simulation Complete!")
    print("="*60)
    print(f"Results saved to: {runner.simulation.output_dir}\n")

if __name__ == "__main__":
    main()
