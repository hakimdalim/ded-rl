"""
Advanced live plotter callback with efficient updates for DED simulation.
Replaces the simple LivePlotter with full thermal and cross-section visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from callbacks._base_callbacks import IntervalCallback, SimulationEvent


class AdvancedLivePlotter(IntervalCallback):
    """
    Creates real-time visualization with temperature fields and cross-sections.

    Efficiently updates a single persistent figure window with:
    - Three orthogonal temperature slice views (XY, XZ, YZ)
    - Cross-section plot showing all layers
    - Activated volume contours
    - Current position indicators

    Optimized for speed with in-place data updates.
    """

    def __init__(
            self,
            interval: int = 1,
            temp_range: tuple = (300, 2500),
            figsize: tuple = (20, 12),
            enabled: bool = True,
            **kwargs
    ):
        """
        Initialize the advanced live plotter.

        Args:
            interval: Update every N steps (default: 1 for every step)
            temp_range: Temperature range for colormaps in Kelvin
            figsize: Figure size in inches (width, height)
            enabled: Whether plotting is enabled
        """
        super().__init__(
            events=SimulationEvent.STEP_COMPLETE,
            interval=interval,
            enabled=enabled,
            **kwargs
        )

        self.temp_range = temp_range
        self.figsize = figsize

        # Figure and axes storage
        self.fig = None
        self.ax_xy = None  # Top view
        self.ax_xz = None  # Front view
        self.ax_yz = None  # Side view
        self.ax_cross = None  # Cross-section

        # Plot artists for efficient updates
        self.im_xy = None  # Temperature image for XY plane
        self.im_xz = None  # Temperature image for XZ plane
        self.im_yz = None  # Temperature image for YZ plane
        self.contour_xy = None  # Activation contour for XY
        self.contour_xz = None  # Activation contour for XZ
        self.contour_yz = None  # Activation contour for YZ
        self.melt_contour_xy = None  # Melt pool contour XY
        self.melt_contour_xz = None  # Melt pool contour XZ
        self.melt_contour_yz = None  # Melt pool contour YZ
        self.cross_lines = []  # Lines for each layer in cross-section

        # Data extents (calculated once)
        self.extent_xy = None
        self.extent_xz = None
        self.extent_yz = None

    def _initialize_plot(self, sim):
        """Initialize the persistent figure and axes."""
        plt.ion()  # Interactive mode

        # Create figure with GridSpec
        self.fig = plt.figure(figsize=self.figsize)
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 0.5])

        # Create subplots
        self.ax_xy = self.fig.add_subplot(gs[0, 0])
        self.ax_xz = self.fig.add_subplot(gs[0, 1])
        self.ax_yz = self.fig.add_subplot(gs[0, 2])
        self.ax_cross = self.fig.add_subplot(gs[1, :])

        # Calculate data extents once (in mm)
        voxel_size = sim.config['voxel_size']
        volume_shape = sim.config['volume_shape']

        self.extent_xy = [
            0, volume_shape[0] * voxel_size[0] * 1000,
            0, volume_shape[1] * voxel_size[1] * 1000
        ]
        self.extent_xz = [
            0, volume_shape[0] * voxel_size[0] * 1000,
            0, volume_shape[2] * voxel_size[2] * 1000
        ]
        self.extent_yz = [
            0, volume_shape[1] * voxel_size[1] * 1000,
            0, volume_shape[2] * voxel_size[2] * 1000
        ]

        # Initialize empty temperature images
        dummy_data_xy = np.zeros((volume_shape[1], volume_shape[0]))
        dummy_data_xz = np.zeros((volume_shape[2], volume_shape[0]))
        dummy_data_yz = np.zeros((volume_shape[2], volume_shape[1]))

        self.im_xy = self.ax_xy.imshow(
            dummy_data_xy, extent=self.extent_xy,
            origin='lower', cmap='hot', vmin=self.temp_range[0], vmax=self.temp_range[1]
        )
        self.im_xz = self.ax_xz.imshow(
            dummy_data_xz, extent=self.extent_xz,
            origin='lower', cmap='hot', vmin=self.temp_range[0], vmax=self.temp_range[1]
        )
        self.im_yz = self.ax_yz.imshow(
            dummy_data_yz, extent=self.extent_yz,
            origin='lower', cmap='hot', vmin=self.temp_range[0], vmax=self.temp_range[1]
        )

        # Add colorbars once
        self.fig.colorbar(self.im_xy, ax=self.ax_xy, fraction=0.046, pad=0.04)
        self.fig.colorbar(self.im_xz, ax=self.ax_xz, fraction=0.046, pad=0.04)
        self.fig.colorbar(self.im_yz, ax=self.ax_yz, fraction=0.046, pad=0.04)

        # Configure axes
        self.ax_xy.set_xlabel('X (mm)')
        self.ax_xy.set_ylabel('Y (mm)')
        self.ax_xy.set_aspect('equal')

        self.ax_xz.set_xlabel('X (mm)')
        self.ax_xz.set_ylabel('Z (mm)')
        self.ax_xz.set_aspect('equal')

        self.ax_yz.set_xlabel('Y (mm)')
        self.ax_yz.set_ylabel('Z (mm)')
        self.ax_yz.set_aspect('equal')

        self.ax_cross.set_xlabel('X (mm)')
        self.ax_cross.set_ylabel('Z (mm)')
        self.ax_cross.grid(True, alpha=0.3)

        self.fig.tight_layout()

    def _execute(self, context: dict) -> None:
        """Update the live plot with current simulation state."""
        sim = context['simulation']

        # Initialize on first call
        if self.fig is None:
            self._initialize_plot(sim)

        # Get current state
        step_ctx = sim.step_context
        if not step_ctx:
            return

        pos = step_ctx['position']
        voxel = step_ctx['voxel']
        build = step_ctx['build']
        params = step_ctx['params']

        # Get temperature field and activation mask
        temp_field = sim.temperature_tracker.temperature
        activated_mask = sim.volume_tracker.activated
        melting_temp = params.get('melting_temp', 1700)

        # Update temperature slices
        z_idx = min(voxel['z'], temp_field.shape[2] - 1)
        y_idx = min(voxel['y'], temp_field.shape[1] - 1)
        x_idx = min(voxel['x'], temp_field.shape[0] - 1)

        # Update image data in-place
        self.im_xy.set_data(temp_field[:, :, z_idx].T)
        self.im_xz.set_data(temp_field[:, y_idx, :].T)
        self.im_yz.set_data(temp_field[x_idx, :, :].T)

        # Clear old contours
        for contour in [self.contour_xy, self.contour_xz, self.contour_yz,
                        self.melt_contour_xy, self.melt_contour_xz, self.melt_contour_yz]:
            if contour is not None:
                for coll in contour.collections:
                    coll.remove()

        # Update activation contours
        x_coords_mm = np.linspace(0, self.extent_xy[1], activated_mask.shape[0])
        y_coords_mm = np.linspace(0, self.extent_xy[3], activated_mask.shape[1])
        z_coords_mm = np.linspace(0, self.extent_xz[3], activated_mask.shape[2])

        self.contour_xy = self.ax_xy.contour(
            x_coords_mm, y_coords_mm, activated_mask[:, :, z_idx].T,
            levels=[0.5], colors='white', linestyles='dashed', linewidths=0.5
        )

        self.contour_xz = self.ax_xz.contour(
            x_coords_mm, z_coords_mm, activated_mask[:, y_idx, :].T,
            levels=[0.5], colors='white', linestyles='dashed', linewidths=0.5
        )

        self.contour_yz = self.ax_yz.contour(
            y_coords_mm, z_coords_mm, activated_mask[x_idx, :, :].T,
            levels=[0.5], colors='white', linestyles='dashed', linewidths=0.5
        )

        # Update melt pool contours
        self.melt_contour_xy = self.ax_xy.contour(
            x_coords_mm, y_coords_mm, temp_field[:, :, z_idx].T,
            levels=[melting_temp], colors='cyan', linewidths=1.0
        )

        self.melt_contour_xz = self.ax_xz.contour(
            x_coords_mm, z_coords_mm, temp_field[:, y_idx, :].T,
            levels=[melting_temp], colors='cyan', linewidths=1.0
        )

        self.melt_contour_yz = self.ax_yz.contour(
            y_coords_mm, z_coords_mm, temp_field[x_idx, :, :].T,
            levels=[melting_temp], colors='cyan', linewidths=1.0
        )

        # Update titles with current position
        self.ax_xy.set_title(f'Top View (z={pos["z"] * 1000:.3f}mm) - Step {sim.progress_tracker.step_count}')
        self.ax_xz.set_title(f'Front View (y={pos["y"] * 1000:.3f}mm)')
        self.ax_yz.set_title(f'Side View (x={pos["x"] * 1000:.3f}mm)')

        # Update cross-section plot
        self._update_cross_section(sim, pos['y'])

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_cross_section(self, sim, y_pos):
        """Update the cross-section plot efficiently."""
        # Clear old lines
        for line in self.cross_lines:
            line.remove()
        self.cross_lines.clear()

        # Get current layer from build state
        current_layer = sim.progress_tracker.current_layer
        if current_layer is None:
            return

        # X coordinates for plotting
        config = sim.config
        x_start = -0.005 + config.get('x_offset', 0)
        x_end = config.get('x_offset', 0) + config['num_tracks'] * config['hatch_spacing'] + 0.005
        x = np.linspace(x_start, x_end, 500)

        # Plot each layer
        num_layers_to_plot = current_layer + 1
        for layer_idx in range(num_layers_to_plot):
            # Get cross section at current y position
            cross_section = sim.clad_manager.get_layer_cross_section(layer_idx, y_pos)
            z = np.array([cross_section(x_val) for x_val in x])

            # Color based on layer
            color = plt.cm.viridis(layer_idx / max(num_layers_to_plot - 1, 1))
            line, = self.ax_cross.plot(
                x * 1000, z * 1000,
                color=color, label=f'Layer {layer_idx}'
            )
            self.cross_lines.append(line)

        # Update legend efficiently (only if layer count changed)
        if len(self.ax_cross.get_legend_handles_labels()[0]) != num_layers_to_plot:
            self.ax_cross.legend(loc='upper left', fontsize='small')

        self.ax_cross.set_title(f'Layer Cross Sections (y={y_pos * 1000:.3f}mm)')
        self.ax_cross.relim()
        self.ax_cross.autoscale_view()

    def __del__(self):
        """Cleanup when callback is destroyed."""
        if self.fig is not None:
            plt.close(self.fig)