import numpy as np
from typing import Dict, Tuple
from utils.vectorize_inputs import prepare_points


class VoxelPowderStream:
    """
    Voxel-based powder stream with automatic alignment to laser position.
    Replaces analytical model with measured intensity field.
    """

    def __init__(
            self,
            voxel_path: str,
            outlet_offset: float = 1.1635e-3,
            nozzle_height: float = 0.015,
            visualize: bool = True
    ):
        """
        Initialize voxel-based powder stream.
        Automatically aligns powder stream center with laser at (0,0).

        Args:
            voxel_path: Path to .npz file containing intensity data
            outlet_offset: Distance from outlet to top of voxel array (1.1635mm)
            nozzle_height: Height of nozzle above substrate (15mm)
            visualize: Whether to show verification plots
        """
        # Load voxel data
        print("=" * 60)
        print("LOADING VOXEL POWDER STREAM")
        print("=" * 60)

        data = np.load(voxel_path)
        raw_field = data['data']
        print(f"Raw array shape: {raw_field.shape}")

        # Transpose from (y, x, z) to (z, y, x) if needed
        if raw_field.shape[2] > max(raw_field.shape[0], raw_field.shape[1]):
            self.field = np.transpose(raw_field, (2, 0, 1))
            print(f"Transposed to (z, y, x): {self.field.shape}")
        else:
            self.field = raw_field

        # Store dimensions
        self.shape = self.field.shape
        nz, ny, nx = self.shape

        # Store parameters
        self.outlet_offset = outlet_offset
        self.nozzle_height = nozzle_height

        # Define voxel spacing in meters
        mm_to_m = 1e-3
        self.spacing = (
            (1209 / 1024) * 0.02327 * mm_to_m,  # Y spacing: 2.75e-5 m
            (1000 / 1024) * 0.02327 * mm_to_m,  # X spacing: 2.27e-5 m
            0.02327 * mm_to_m  # Z spacing: 2.327e-5 m
        )

        # Set up Z coordinates
        # The nozzle is at nozzle_height (15mm above substrate)
        # The outlet is outlet_offset below the nozzle
        # The top of the voxel array is outlet_offset below the outlet
        self.z_top = self.nozzle_height - 2 * self.outlet_offset  # Top of voxel array
        self.z_bottom = self.z_top - nz * self.spacing[2]  # Bottom of voxel array

        # Find which slice index corresponds to z=0 (substrate)
        # We need to solve: z_coordinate = z_top - iz * spacing[2] = 0
        # Therefore: iz = z_top / spacing[2]
        self.iz_substrate = int(round(self.z_top / self.spacing[2]))
        self.iz_substrate = np.clip(self.iz_substrate, 0, nz - 1)

        actual_z_substrate = self.z_top - self.iz_substrate * self.spacing[2]
        print(f"Substrate slice: index {self.iz_substrate}, z={actual_z_substrate * 1000:.3f} mm")

        # Find powder stream center at substrate
        self._find_and_align_center()

        # Set up X,Y coordinates with powder aligned to laser at (0,0)
        # The powder center pixel coordinates are transformed to world coordinates
        # such that the powder center is at (0,0)
        self.x_min = -self.powder_center_px_x * self.spacing[1]
        self.x_max = self.x_min + nx * self.spacing[1]
        self.y_min = -self.powder_center_px_y * self.spacing[0]
        self.y_max = self.y_min + ny * self.spacing[0]

        # Precompute values
        self.max_intensity = np.max(self.field)
        self.boundary_threshold = 0.135 * self.max_intensity
        self._calibration_cache = {}

        # Print summary
        print("=" * 60)
        print(f"Array shape: {self.shape} (z, y, x)")
        print(
            f"Voxel spacing: ({self.spacing[0] * 1e6:.1f}, {self.spacing[1] * 1e6:.1f}, {self.spacing[2] * 1e6:.1f}) μm")
        print(f"X range: [{self.x_min * 1000:.2f}, {self.x_max * 1000:.2f}] mm")
        print(f"Y range: [{self.y_min * 1000:.2f}, {self.y_max * 1000:.2f}] mm")
        print(f"Z range: [{self.z_bottom * 1000:.2f}, {self.z_top * 1000:.2f}] mm")
        print(f"Powder center aligned to: (0.00, 0.00) mm at substrate")
        print(f"Max intensity: {self.max_intensity:.2f}")
        print("=" * 60)

        if visualize:
            self._visualize_alignment()

    def _find_and_align_center(self):
        """Find powder center at substrate and set up coordinates to align it with (0,0)"""

        # Get dimensions
        nz, ny, nx = self.shape

        # Get substrate slice
        slice_data = self.field[self.iz_substrate, :, :]

        print(f"\nFinding powder center at substrate (slice {self.iz_substrate})...")

        # Check if there's any intensity at substrate
        total = np.sum(slice_data)

        if total > 0:
            # For finding the center, we should use the peak intensity location
            # as it's more reliable than center of mass for asymmetric distributions

            # Find max intensity location - this is the true center of the powder stream
            max_iy, max_ix = np.unravel_index(np.argmax(slice_data), slice_data.shape)
            print(f"Max intensity (powder center) at pixel: (x={max_ix}, y={max_iy})")

            # Also calculate center of mass for comparison
            ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
            com_x = np.sum(ix * slice_data) / total
            com_y = np.sum(iy * slice_data) / total
            print(f"Center of mass at pixel: (x={com_x:.1f}, y={com_y:.1f})")

            # Use max intensity location as the powder center
            # This is more accurate for laser-powder alignment
            self.powder_center_px_x = float(max_ix)
            self.powder_center_px_y = float(max_iy)

            # Calculate offset for diagnostics
            dx = com_x - max_ix
            dy = com_y - max_iy
            offset_mm = np.sqrt((dx * self.spacing[1]) ** 2 + (dy * self.spacing[0]) ** 2) * 1000
            print(f"COM vs peak offset: ({dx:.1f}, {dy:.1f}) pixels = {offset_mm:.3f} mm")

            # Optionally use a refined center by fitting a 2D Gaussian around the peak
            # This can give sub-pixel accuracy
            if self._refine_center_with_gaussian_fit(slice_data, max_ix, max_iy):
                print(f"Refined center at pixel: (x={self.powder_center_px_x:.2f}, y={self.powder_center_px_y:.2f})")

        else:
            # No intensity at substrate - try to find the first slice with intensity
            print("Warning: No intensity at substrate slice, searching for powder stream...")

            found_powder = False
            for iz in range(self.shape[0]):
                slice_test = self.field[iz, :, :]
                if np.sum(slice_test) > 0.01 * self.max_intensity:  # Found significant intensity
                    z_coord = self.z_top - iz * self.spacing[2]
                    print(f"Found powder stream at slice {iz} (z={z_coord * 1000:.2f} mm)")

                    # Use max intensity for this slice
                    max_iy, max_ix = np.unravel_index(np.argmax(slice_test), slice_test.shape)
                    self.powder_center_px_x = float(max_ix)
                    self.powder_center_px_y = float(max_iy)
                    found_powder = True
                    break

            if not found_powder:
                # Fallback to array center
                self.powder_center_px_x = self.shape[2] // 2
                self.powder_center_px_y = self.shape[1] // 2
                print(f"Warning: No significant intensity found, using array center")

    def _refine_center_with_gaussian_fit(self, slice_data, ix_center, iy_center, window_size=11):
        """
        Refine center position using 2D Gaussian fit around the peak.
        Returns True if successful, False otherwise.
        """
        try:
            # Extract a window around the peak
            half_window = window_size // 2
            x_start = max(0, ix_center - half_window)
            x_end = min(slice_data.shape[1], ix_center + half_window + 1)
            y_start = max(0, iy_center - half_window)
            y_end = min(slice_data.shape[0], iy_center + half_window + 1)

            window = slice_data[y_start:y_end, x_start:x_end]

            # Calculate weighted centroid within the window
            if np.sum(window) > 0:
                ix_local, iy_local = np.meshgrid(
                    np.arange(window.shape[1]),
                    np.arange(window.shape[0])
                )

                refined_x_local = np.sum(ix_local * window) / np.sum(window)
                refined_y_local = np.sum(iy_local * window) / np.sum(window)

                # Convert back to global coordinates
                self.powder_center_px_x = x_start + refined_x_local
                self.powder_center_px_y = y_start + refined_y_local

                return True
        except:
            pass

        return False

    def _visualize_alignment(self):
        """Show verification plots"""
        import matplotlib.pyplot as plt

        # Create figure with multiple views
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Select z-slices to show
        z_slices = [
            (self.iz_substrate, "Substrate (z=0)"),
            (int(self.shape[0] * 0.6), "60% down"),
            (int(self.shape[0] * 0.3), "30% down (near top)")
        ]

        for i, (iz, label) in enumerate(z_slices):
            ax = axes[0, i]

            # Get slice and z-coordinate
            slice_data = self.field[iz, :, :]
            z_height = self.z_top - iz * self.spacing[2]

            # Create extent for plot in mm
            extent = [self.x_min * 1000, self.x_max * 1000,
                      self.y_min * 1000, self.y_max * 1000]

            im = ax.imshow(slice_data, origin='lower', extent=extent,
                           cmap='hot', aspect='equal')
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Mark origin (laser position)
            ax.plot(0, 0, 'g+', markersize=15, markeredgewidth=2, label='Laser')
            ax.axhline(y=0, color='green', linestyle=':', alpha=0.3)
            ax.axvline(x=0, color='green', linestyle=':', alpha=0.3)

            # Add circle to show typical laser spot size (if needed)
            circle = plt.Circle((0, 0), 0.5, fill=False, edgecolor='green',
                                linestyle='--', alpha=0.5, label='Laser spot')
            ax.add_patch(circle)

            ax.set_title(f'{label}, z = {z_height * 1000:.1f} mm')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

        # Show XZ cross-section at Y=0
        ax = axes[1, 0]
        # Find the y-index closest to y=0
        y_world_0_idx = int(round(self.powder_center_px_y))
        xz_slice = self.field[:, y_world_0_idx, :]

        extent = [self.x_min * 1000, self.x_max * 1000,
                  self.z_bottom * 1000, self.z_top * 1000]

        im = ax.imshow(xz_slice, origin='lower', extent=extent,
                       aspect='auto', cmap='hot')
        plt.colorbar(im, ax=ax)
        ax.axhline(y=0, color='red', linestyle='--', label='Substrate')
        ax.axvline(x=0, color='green', linestyle=':', alpha=0.5, label='Laser axis')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title('XZ Cross-section at Y=0')
        ax.legend()

        # Show YZ cross-section at X=0
        ax = axes[1, 1]
        # Find the x-index closest to x=0
        x_world_0_idx = int(round(self.powder_center_px_x))
        yz_slice = self.field[:, :, x_world_0_idx]

        extent = [self.y_min * 1000, self.y_max * 1000,
                  self.z_bottom * 1000, self.z_top * 1000]

        im = ax.imshow(yz_slice, origin='lower', extent=extent,
                       aspect='auto', cmap='hot')
        plt.colorbar(im, ax=ax)
        ax.axhline(y=0, color='red', linestyle='--', label='Substrate')
        ax.axvline(x=0, color='green', linestyle=':', alpha=0.5, label='Laser axis')
        ax.set_xlabel('Y (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title('YZ Cross-section at X=0')
        ax.legend()

        # Show intensity profile at center
        ax = axes[1, 2]
        # Get vertical profile at (0,0) in world coordinates
        center_ix = int(round(self.powder_center_px_x))
        center_iy = int(round(self.powder_center_px_y))
        profile = self.field[:, center_iy, center_ix]
        z_coords = self.z_top - np.arange(self.shape[0]) * self.spacing[2]

        ax.plot(profile, z_coords * 1000, 'b-', linewidth=2)
        ax.axhline(y=0, color='red', linestyle='--', label='Substrate')
        ax.axhline(y=self.nozzle_height * 1000, color='orange', linestyle='--', label='Nozzle')
        ax.set_xlabel('Intensity')
        ax.set_ylabel('Z (mm)')
        ax.set_title('Vertical Profile at (0,0)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Powder Stream Alignment Verification', fontsize=14)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def transform_coordinates(points: np.ndarray, params: Dict) -> np.ndarray:
        """For compatibility - no transformation needed"""
        return prepare_points(points)

    def _point_to_voxel_indices(self, points):
        """Convert world coordinates to voxel indices"""
        # Calculate indices
        ix = np.round((points[..., 0] - self.x_min) / self.spacing[1]).astype(np.int32)
        iy = np.round((points[..., 1] - self.y_min) / self.spacing[0]).astype(np.int32)
        iz = np.round((self.z_top - points[..., 2]) / self.spacing[2]).astype(np.int32)

        # Clip to valid range
        ix = np.clip(ix, 0, self.shape[2] - 1)
        iy = np.clip(iy, 0, self.shape[1] - 1)
        iz = np.clip(iz, 0, self.shape[0] - 1)

        return iz, iy, ix

    def _get_calibration_wrong(self, params):
        """Get calibration factor to convert intensity to kg/m³"""
        key = (params['powder_feed_rate'], params['particle_velocity'])

        if key not in self._calibration_cache:
            m_dot = params['powder_feed_rate']
            v_p = params['particle_velocity']

            # Use substrate cross-section for calibration
            cross_section = self.field[self.iz_substrate, :, :]

            # Calculate total intensity flux through substrate
            total_intensity = np.sum(cross_section) * self.spacing[0] * self.spacing[1]

            if total_intensity > 0:
                # Mass flux = feed rate, Volume flux = velocity * area
                # Concentration = mass flux / volume flux
                self._calibration_cache[key] = m_dot / (v_p * total_intensity)
            else:
                self._calibration_cache[key] = 0
                print("Warning: No intensity at substrate for calibration")

            print(f"Calibration factor: {self._calibration_cache[key]:.3e} kg/m⁴")

        return self._calibration_cache[key]

    def powder_concentration_wrong(self, points: np.ndarray, params: Dict) -> np.ndarray:
        """Calculate powder concentration at points"""
        points = prepare_points(points)
        iz, iy, ix = self._point_to_voxel_indices(points)
        intensity = self.field[iz, iy, ix]
        return self._get_calibration(params) * intensity

    def _get_calibration_flux_based(self, params):
        """
        Alternative calibration using flux conservation.
        This method is more accurate for varying velocity fields.
        """
        key = (params['powder_feed_rate'], params['particle_velocity'])

        if key not in self._calibration_cache:
            m_dot = params['powder_feed_rate']  # kg/s
            v_p = params['particle_velocity']  # m/s

            # Get substrate cross-section
            cross_section = self.field[self.iz_substrate, :, :]

            # Calculate the integral ∫∫ I(x,y) dA
            intensity_flux = np.sum(cross_section) * self.spacing[0] * self.spacing[1]  # m²

            if intensity_flux > 0:
                # The calibration converts intensity flux to mass flux
                # ṁ = K * v_p * ∫∫ I dA
                # Therefore: K = ṁ / (v_p * intensity_flux)
                self._calibration_cache[key] = m_dot / (v_p * intensity_flux)  # kg/m³

                print(f"Flux-based calibration: {self._calibration_cache[key]:.3e} kg/m³")
                print(f"  Intensity flux: {intensity_flux * 1e6:.2f} mm²")
                print(f"  Implied avg concentration: {self._calibration_cache[key] * np.mean(cross_section):.3f} kg/m³")

            else:
                self._calibration_cache[key] = 0
                print("Warning: No intensity at substrate for calibration")

        return self._calibration_cache[key]

    def _get_calibration(self, params):
        """
        Get calibration factor to convert intensity to kg/m³

        Physical model:
        - The intensity field I(x,y,z) represents relative powder distribution (dimensionless, 0-1)
        - Actual concentration C(x,y,z) = K * I(x,y,z) where K is the calibration factor
        - K has units of [kg/m³] to convert dimensionless intensity to concentration

        Mass conservation at substrate:
        ṁ = ∫∫ C(x,y) * v_p * dA = ∫∫ K * I(x,y) * v_p * dA

        For pixels above threshold (in the actual stream):
        ṁ = K * v_p * A_eff * I_avg

        Where:
        - A_eff = effective area of stream (pixels above threshold)
        - I_avg = average intensity within stream

        Therefore: K = ṁ / (v_p * A_eff * I_avg) = C_avg / I_avg

        This gives the average concentration in the stream divided by average intensity.
        """
        key = (params['powder_feed_rate'], params['particle_velocity'])

        if key not in self._calibration_cache:
            m_dot = params['powder_feed_rate']  # kg/s
            v_p = params['particle_velocity']  # m/s

            # Get substrate cross-section
            cross_section = self.field[self.iz_substrate, :, :]

            # Identify pixels actually in the powder stream (above threshold)
            in_stream_mask = cross_section > self.boundary_threshold
            pixels_in_stream = np.sum(in_stream_mask)

            if pixels_in_stream > 0:
                # Calculate effective area of powder stream at substrate
                pixel_area = self.spacing[0] * self.spacing[1]  # m² per pixel
                effective_area = pixels_in_stream * pixel_area  # m²

                # Average intensity within the stream (not including zeros outside)
                avg_intensity = np.mean(cross_section[in_stream_mask])

                # Average concentration in stream from mass conservation
                # ṁ = ρ_avg * v_p * A_eff
                avg_concentration = m_dot / (v_p * effective_area)  # kg/m³

                # Calibration factor: converts intensity to concentration
                # Since C = K * I, and we know C_avg and I_avg in the stream:
                self._calibration_cache[key] = avg_concentration / avg_intensity  # kg/m³ per intensity unit

                # Validation check: verify units
                # [kg/m³] = [kg/s] / ([m/s] * [m²]) ✓
                # [kg/m³ per intensity] = [kg/m³] / [dimensionless] ✓

                print(f"Calibration factor: {self._calibration_cache[key]:.3e} kg/m³ per unit intensity")
                print(f"  Effective area: {effective_area * 1e6:.2f} mm²")
                print(f"  Pixels in stream: {pixels_in_stream}")
                print(f"  Avg intensity in stream: {avg_intensity:.3f}")
                print(f"  Avg concentration in stream: {avg_concentration:.3f} kg/m³")

                # Sanity check - total mass flow
                total_mass_flow_check = avg_concentration * v_p * effective_area
                print(f"  Mass flow check: {total_mass_flow_check * 1e6:.3f} mg/s (input: {m_dot * 1e6:.3f} mg/s)")

            else:
                self._calibration_cache[key] = 0
                print("Warning: No intensity at substrate for calibration")

        return self._calibration_cache[key]

    def powder_concentration(self, points: np.ndarray, params: Dict) -> np.ndarray:
        """
        Calculate powder concentration at points in kg/m³

        Model: C(x,y,z) = K * I(x,y,z)

        Where:
        - C(x,y,z) = powder concentration [kg/m³]
        - K = calibration factor [kg/m³ per intensity unit]
        - I(x,y,z) = intensity from voxel field [dimensionless, 0-1]

        This assumes intensity linearly relates to concentration, which is valid
        for single-scattering regime but may break down at very high concentrations
        due to multiple scattering and shadowing effects.
        """
        points = prepare_points(points)
        iz, iy, ix = self._point_to_voxel_indices(points)
        intensity = self.field[iz, iy, ix]  # dimensionless (0-1)

        # Apply calibration to get concentration
        concentration = self._get_calibration(params) * intensity  # kg/m³

        # Validation: concentration should be positive and physically reasonable
        # Typical powder-gas streams: 0.1-10 kg/m³
        # If seeing >100 kg/m³, likely calibration error or saturation

        return concentration

    def number_concentration(self, points: np.ndarray, params: Dict) -> np.ndarray:
        """Calculate particle number concentration"""
        return self.powder_concentration(points, params) / params['particle_mass']

    def is_within_powder_stream_radius(self, points: np.ndarray, params: Dict, t: float = 0.0) -> np.ndarray:
        """Check if points are within powder stream"""
        points = prepare_points(points)
        if t != 0:
            points = points.copy()
            points[..., 1] -= params['scan_speed'] * t
        iz, iy, ix = self._point_to_voxel_indices(points)
        intensity = self.field[iz, iy, ix]
        return intensity > self.boundary_threshold

    def powder_stream_boundary_at_z(self, points: np.ndarray, params: Dict) -> np.ndarray:
        """Find boundary height for given (x,y) positions"""
        points = prepare_points(points)
        x, y = points[..., 0], points[..., 1]

        ix = np.round((x - self.x_min) / self.spacing[1]).astype(np.int32)
        iy = np.round((y - self.y_min) / self.spacing[0]).astype(np.int32)
        ix = np.clip(ix, 0, self.shape[2] - 1)
        iy = np.clip(iy, 0, self.shape[1] - 1)

        boundary_z = np.zeros_like(x)

        for i in range(x.size):
            column = self.field[:, iy.flat[i], ix.flat[i]]
            above = np.where(column > self.boundary_threshold)[0]
            if len(above) > 0:
                iz_top = above[0]
                boundary_z.flat[i] = self.z_top - iz_top * self.spacing[2]

        return boundary_z


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from field_visualization import ThermalPlotter
    from process_parameters import set_params
    import time

    # Setup parameters
    params = set_params()
    params.update({
        'particle_velocity': (2.5 / 60000) / (np.pi * (0.7e-3) ** 2),
        'scan_speed': 0.003,
        'powder_feed_rate': 2 * (1 / 1000) * (1 / 60),
        'particle_mass': 4.5e-8,
    })

    # Test three different nozzle heights
    nozzle_heights = [0.014, 0.015, 0.016]  # 14mm, 15mm, 16mm
    voxel_streams = []

    for nozzle_height in nozzle_heights:
        print(f"\n{'=' * 60}")
        print(f"TESTING NOZZLE HEIGHT: {nozzle_height * 1000:.0f} mm")
        print(f"{'=' * 60}")

        # Initialize with auto-alignment for this nozzle height
        voxel_stream = VoxelPowderStream(
            "_arrays/250422_10%3A36%3A17/target_stack.npz",
            outlet_offset=1.1635e-3,
            nozzle_height=nozzle_height,
            visualize=False  # Turn off individual visualizations
        )
        voxel_streams.append(voxel_stream)

        print(f"\nTesting concentration at key points (nozzle={nozzle_height * 1000:.0f}mm)...")
        test_points = np.array([
            [0, 0, 0],  # Laser position = powder center
            [0.001, 0, 0],  # 1mm offset in X
            [0, 0.001, 0],  # 1mm offset in Y
            [0, 0, 0.001],  # 1mm above substrate
        ])

        concentrations = voxel_stream.powder_concentration(test_points, params)
        for pt, conc in zip(test_points, concentrations):
            print(f"  ({pt[0] * 1000:.1f}, {pt[1] * 1000:.1f}, {pt[2] * 1000:.1f}) mm: {conc:.3f} kg/m³")

    # Create comparison visualization
    print("\n" + "=" * 60)
    print("CREATING COMPARISON VISUALIZATION")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Evaluate concentration field at substrate for each nozzle height
    x = np.linspace(-5e-3, 5e-3, 200)
    y = np.linspace(-5e-3, 5e-3, 200)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X, Y, np.zeros_like(X)], axis=-1)

    for i, (nozzle_height, voxel_stream) in enumerate(zip(nozzle_heights, voxel_streams)):
        # Top row: concentration at substrate
        ax = axes[0, i]
        concentration = voxel_stream.powder_concentration(points, params)

        # Create the image directly without using ThermalPlotter
        im = ax.imshow(concentration.reshape(X.shape),
                       extent=[X.min() * 1000, X.max() * 1000, Y.min() * 1000, Y.max() * 1000],
                       origin='lower', cmap='hot', aspect='equal')

        # Mark center
        ax.plot(0, 0, 'g+', markersize=15, markeredgewidth=2, label='Laser/Powder Center')
        ax.set_title(f'Nozzle Height: {nozzle_height * 1000:.0f} mm\nConcentration at Substrate')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Concentration (kg/m³)', rotation=270, labelpad=15)

        # Bottom row: XZ cross-section at Y=0
        ax = axes[1, i]

        # Get the y-index closest to y=0 (powder center)
        y_world_0_idx = int(round(voxel_stream.powder_center_px_y))
        xz_slice = voxel_stream.field[:, y_world_0_idx, :]

        extent = [voxel_stream.x_min * 1000, voxel_stream.x_max * 1000,
                  voxel_stream.z_bottom * 1000, voxel_stream.z_top * 1000]

        im = ax.imshow(xz_slice, origin='lower', extent=extent,
                       aspect='auto', cmap='hot')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Substrate')
        ax.axvline(x=0, color='green', linestyle=':', alpha=0.7, label='Laser axis')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Z (mm)')
        ax.set_title(f'XZ Cross-section at Y=0\nWorking Distance: {nozzle_height * 1000:.0f} mm')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Print peak concentration for comparison
        peak_conc = np.max(concentration)
        print(f"Nozzle {nozzle_height * 1000:.0f}mm - Peak concentration at substrate: {peak_conc:.3f} kg/m³")

    plt.suptitle('Powder Stream Analysis: Effect of Nozzle Height on Alignment', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Show individual visualization for the middle nozzle height (15mm)
    print("\nShowing detailed visualization for 15mm nozzle height...")
    voxel_streams[1]._visualize_alignment()