"""
Track calibration callback for DED simulation.
Runs a calibration track to measure actual clad dimensions and adjusts spacings.
"""

import numpy as np
from callbacks._base_callbacks import BaseCallback, SimulationEvent


class TrackCalibrationCallback(BaseCallback):
    """
    Runs a calibration track to measure actual clad dimensions,
    then reconfigures hatch and layer spacing based on measurements.
    """

    def __init__(
        self,
        calibration_steps: int = 10,
        hatch_overlap_percent: float = 30.0,
        layer_overlap_percent: float = 50.0,
        **kwargs
    ):
        super().__init__(events=SimulationEvent.CONFIG_LOADED, **kwargs)
        self.calibration_steps = calibration_steps
        self.hatch_overlap_percent = hatch_overlap_percent
        self.layer_overlap_percent = layer_overlap_percent

    def _execute(self, context: dict) -> None:
        sim = context['simulation']
        params = context['params']

        print("\n" + "="*60)
        print("TRACK CALIBRATION")
        print("="*60)

        original_hatch = sim.config['hatch_spacing']
        original_layer = sim.config.get('layer_spacing', 0.00035)

        # Create new temporary simulation for calibration
        from core.multi_track_multi_layer import MultiTrackMultiLayerSimulation

        calibration_sim = MultiTrackMultiLayerSimulation(
            config=sim.config,
            delta_t=sim.delta_t,
            powder_concentration_func=sim.powder_concentration_func,
            callbacks=[],  # No callbacks for calibration
            output_dir=None
        )

        widths = []
        heights = []

        for i in range(self.calibration_steps):
            melt_pool, clad_dims, _, _, _, _ = calibration_sim.step(params)
            if i >= 2:  # Skip first steps for stability
                widths.append(clad_dims['width'])
                heights.append(clad_dims['height'])

        # Delete calibration simulation
        del calibration_sim

        # Calculate recommendations
        measured_width = np.mean(widths)
        measured_height = np.mean(heights)

        recommended_hatch = measured_width * (1 - self.hatch_overlap_percent/100)
        recommended_layer = measured_height * (1 - self.layer_overlap_percent/100)

        print(f"\nMeasured: {measured_width*1000:.3f}mm wide, {measured_height*1000:.3f}mm high")
        print(f"Original: {original_hatch*1000:.3f}mm hatch, {original_layer*1000:.3f}mm layer")
        print(f"Recommended: {recommended_hatch*1000:.3f}mm hatch, {recommended_layer*1000:.3f}mm layer")

        # Apply changes to main simulation config
        sim.config['hatch_spacing'] = recommended_hatch
        sim.config['layer_spacing'] = recommended_layer
        sim.config['num_tracks'] = int(np.ceil(sim.config['part_width'] / recommended_hatch))
        sim.config['actual_width'] = sim.config['num_tracks'] * recommended_hatch

        print("="*60 + "\n")