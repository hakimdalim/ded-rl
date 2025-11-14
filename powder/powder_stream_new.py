import numpy as np
from typing import Dict, Tuple
from utils.vectorize_inputs import prepare_points
import gc


class VoxelPowderStream:
    """
    Voxel-based powder stream with automatic alignment to laser position.
    Replaces analytical model with measured intensity field.
    """

    def __init__(
            self,
            voxel_path: str,
            nozzle_offset_slices: int = 50,
            auto_detect_working_distance: bool = True,
            cached_metadata: dict = None,
            visualize: bool = True
    ):
        """
        Initialize voxel-based powder stream.
        Automatically aligns powder stream center with laser at (0,0).

        Args:
            voxel_path: Path to .npz file containing intensity data
            nozzle_offset_slices: Number of slices between array top and nozzle (default: 50)
            auto_detect_working_distance: If True, detect working distance from focal point (recommended)
            cached_metadata: Pre-calculated metadata (focal point, working distance) to avoid recalculation
            visualize: Whether to show verification plots
        """
        # Load voxel data
        print("=" * 60)
        print("LOADING VOXEL POWDER STREAM")
        print("=" * 60)

        data = np.load(voxel_path)
        raw_field = data['stream']
        print(f"Raw array shape: {raw_field.shape}")

        # Transpose from (y, x, z) to (z, y, x) if needed
        if raw_field.shape[2] > max(raw_field.shape[0], raw_field.shape[1]):
            transposed_field = np.transpose(raw_field, (2, 0, 1))
            print(f"Transposed to (z, y, x): {transposed_field.shape}")
        else:
            transposed_field = raw_field

        # FIRST: Detect focal point in full array (or use cached if available)
        if cached_metadata is not None and 'iz_substrate_original' in cached_metadata:
            iz_focal_full = cached_metadata['iz_substrate_original']
            print(f"Using cached focal point (original coordinates): slice {iz_focal_full}")
        else:
            # Detect focal point in full transposed array
            nz_full = transposed_field.shape[0]
            max_intensities_per_slice = np.array([np.max(transposed_field[iz, :, :]) for iz in range(nz_full)])
            iz_focal_full = np.argmax(max_intensities_per_slice)
            print(f"Detected focal point (original coordinates): slice {iz_focal_full}")

        # THEN: Crop centered on focal point (keep ±quartile range around focal point)
        self.field, self.crop_offsets, iz_focal_cropped = self._crop_around_focal_point(transposed_field, iz_focal_full)

        # Calculate memory savings BEFORE deleting
        original_size = transposed_field.nbytes / (1024 ** 2)  # MB
        cropped_size = self.field.nbytes / (1024 ** 2)  # MB
        savings_pct = (1 - cropped_size / original_size) * 100
        nz_original = transposed_field.shape[0]

        # Explicitly delete the original arrays to free memory immediately
        del raw_field
        del transposed_field
        del data
        gc.collect()  # Force garbage collection

        print(f"Cropped to: {self.field.shape} (z, y, x)")
        print(f"Crop offsets (z, y, x): {self.crop_offsets}")

        # Calculate memory savings
        print(f"Memory: {cropped_size:.1f} MB (was {original_size:.1f} MB, saved {savings_pct:.1f}%)")

        # Store dimensions
        self.shape = self.field.shape
        nz, ny, nx = self.shape

        # Define voxel spacing in meters
        mm_to_m = 1e-3
        self.spacing = (
            (1209 / 1024) * 0.02327 * mm_to_m,  # Y spacing: 2.75e-5 m
            (1000 / 1024) * 0.02327 * mm_to_m,  # X spacing: 2.27e-5 m
            0.02327 * mm_to_m  # Z spacing: 2.327e-5 m
        )

        # Store parameters
        self.nozzle_offset_slices = nozzle_offset_slices
        self.nozzle_offset_distance = nozzle_offset_slices * self.spacing[2]  # Convert to meters

        # Focal point in cropped coordinates (for array operations)
        self.iz_substrate = iz_focal_cropped

        # Store original focal point index for caching (before cropping)
        self.iz_substrate_original = iz_focal_full

        # Calculate working distance
        # CRITICAL: Need to account for slices cropped from the top of the array
        #
        # In original array (before crop):
        #   - Top slice index: nz_original - 1
        #   - Focal point: iz_focal_original
        #   - Distance substrate to original top: (nz_original - 1 - iz_focal_original) slices
        #   - Nozzle is nozzle_offset_slices above original top
        #   - Working distance = distance_to_top + nozzle_offset
        #
        # After cropping:
        #   - We cropped from z_min to z_max of original array
        #   - Original array had nz_original slices (indexed 0 to nz_original-1)
        #   - Slices removed from top = (nz_original - 1) - (z_min + nz - 1) = nz_original - z_min - nz
        #   - The nozzle is still at the same physical location
        #   - So nozzle is now (slices_removed_from_top + nozzle_offset_slices) above cropped top

        # Get original array size from transposed field

        z_min = self.crop_offsets[0]

        # Slices between cropped top and original top
        slices_removed_from_top = (nz_original - 1) - (z_min + nz - 1)

        # Effective nozzle offset relative to cropped array
        effective_nozzle_offset = slices_removed_from_top + nozzle_offset_slices

        # Working distance calculation
        distance_substrate_to_array_top = ((nz - 1) - self.iz_substrate) * self.spacing[2]
        self.z_top = distance_substrate_to_array_top
        self.z_bottom = self.z_top - nz * self.spacing[2]
        working_distance_slices = ((nz - 1) - self.iz_substrate) + effective_nozzle_offset
        self.detected_nozzle_height = working_distance_slices * self.spacing[2]

        print(f"Substrate at slice index: {self.iz_substrate} (cropped array), z=0.000 mm (by definition)")
        print(f"Distance from substrate to cropped array top: {distance_substrate_to_array_top * 1000:.3f} mm")
        print(f"Slices removed from original top: {slices_removed_from_top}")
        print(f"Effective nozzle offset (relative to cropped top): {effective_nozzle_offset} slices")
        print(
            f"Nozzle offset (array top to nozzle): {self.nozzle_offset_distance * 1000:.3f} mm ({nozzle_offset_slices} slices)")
        print(
            f"Detected nozzle height (working distance): {self.detected_nozzle_height * 1000:.3f} mm ({working_distance_slices} slices)")

        # DEBUG: Check if there's actually powder at this slice
        substrate_intensity = np.sum(self.field[self.iz_substrate, :, :])
        max_intensity_array = np.max(self.field)
        print(f"DEBUG: Intensity at substrate slice: {substrate_intensity:.6f}")
        print(f"DEBUG: Max intensity in entire array: {max_intensity_array:.6f}")
        if substrate_intensity < 0.01 * max_intensity_array:
            print(f"WARNING: Very low intensity at calculated substrate slice!")

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

        # Check if array is empty
        if self.max_intensity == 0:
            raise ValueError(
                f"Powder stream array is completely empty (all zeros)! "
                f"Array shape: {self.shape}. "
                f"File may be corrupted or incorrectly generated."
            )

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

                    # FIX: Update substrate index to where powder actually is
                    self.iz_substrate = iz
                    print(f"Updated iz_substrate from calculated value to actual powder location: {iz}")

                    # Use max intensity for this slice
                    max_iy, max_ix = np.unravel_index(np.argmax(slice_test), slice_test.shape)
                    self.powder_center_px_x = float(max_ix)
                    self.powder_center_px_y = float(max_iy)
                    found_powder = True
                    break

            if not found_powder:
                # No powder found anywhere - this is a critical error
                raise ValueError(
                    f"No powder stream found in array! "
                    f"Max intensity in array: {self.max_intensity:.6f}. "
                    f"Expected powder stream file may be empty or corrupted. "
                    f"Check powder stream generation."
                )

    def _detect_focal_point_z(self) -> Tuple[int, float]:
        """
        Detect the focal point (maximum intensity) along Z-axis using loaded field.

        Returns:
            (iz_focal, max_intensity_at_focal)
            - iz_focal: Index of slice with maximum intensity
            - max_intensity_at_focal: Maximum intensity value at that slice
        """
        nz = self.shape[0]

        # Calculate maximum intensity in each Z-slice
        max_intensities_per_slice = np.array([np.max(self.field[iz, :, :]) for iz in range(nz)])

        # Find slice with overall maximum
        iz_focal = np.argmax(max_intensities_per_slice)
        max_intensity = max_intensities_per_slice[iz_focal]

        print(f"\n{'=' * 60}")
        print("AUTO-DETECTED FOCAL POINT")
        print(f"{'=' * 60}")
        print(f"Focal point at slice index: {iz_focal}")
        print(f"Maximum intensity at focal point: {max_intensity:.6f}")
        print(f"{'=' * 60}\n")

        return iz_focal, max_intensity

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
        except Exception as e:
            # Gaussian refinement failed - this is non-critical, we can use the peak location
            print(f"DEBUG: Gaussian refinement failed (non-critical): {e}")
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
        ax.axhline(y=self.detected_nozzle_height * 1000, color='orange', linestyle='--', label='Nozzle')
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

            # DEBUG: Check what we're getting
            total_intensity = np.sum(cross_section)
            print(
                f"DEBUG calibration: iz_substrate={self.iz_substrate}, total intensity at slice={total_intensity:.6f}")

            # Identify pixels actually in the powder stream (above threshold)
            in_stream_mask = cross_section > self.boundary_threshold
            pixels_in_stream = np.sum(in_stream_mask)

            # DEBUG: Detailed slice analysis
            print(f"DEBUG: Slice shape: {cross_section.shape}")
            print(
                f"DEBUG: Slice min/max/mean: {np.min(cross_section):.6f} / {np.max(cross_section):.6f} / {np.mean(cross_section):.6f}")
            print(f"DEBUG: Pixels > 0: {np.sum(cross_section > 0)}")
            print(f"DEBUG: Pixels > threshold ({self.boundary_threshold:.6f}): {pixels_in_stream}")
            if hasattr(self, 'powder_center_px_x') and hasattr(self, 'powder_center_px_y'):
                center_y = int(self.powder_center_px_y)
                center_x = int(self.powder_center_px_x)
                print(
                    f"DEBUG: Value at powder center pixel [{center_y}, {center_x}]: {cross_section[center_y, center_x]:.6f}")

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
                # No pixels in stream - critical error
                raise ValueError(
                    f"No powder stream pixels found at substrate (iz={self.iz_substrate})! "
                    f"Total intensity at slice: {total_intensity:.6f}, "
                    f"Boundary threshold: {self.boundary_threshold:.6f}. "
                    f"Powder stream may not reach substrate or array geometry is incorrect."
                )

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

    def get_detected_nozzle_height(self) -> float:
        """
        Get the detected or configured nozzle height.

        Returns:
            Nozzle height in meters
        """
        return self.detected_nozzle_height

    def get_metadata(self) -> dict:
        """
        Get metadata about the powder stream for caching.

        Returns:
            Dictionary with focal point and working distance information
        """
        return {
            'iz_substrate': int(self.iz_substrate),  # Cropped coordinates
            'iz_substrate_original': int(self.iz_substrate_original),  # Original coordinates for cache reuse
            'detected_nozzle_height': float(self.detected_nozzle_height),
            'z_top': float(self.z_top),
            'z_bottom': float(self.z_bottom),
            'nozzle_offset_slices': int(self.nozzle_offset_slices),
            'max_intensity': float(self.max_intensity)
        }

    def _crop_around_focal_point(self, field: np.ndarray, iz_focal: int) -> tuple:
        """
        Crop 3D array centered on focal point, keeping central 50% in each dimension.
        For Z-axis: crops centered on focal point
        For X,Y: crops centered on array center

        Args:
            field: 3D array (z, y, x) to crop
            iz_focal: Index of focal point in full array

        Returns:
            cropped_field: Cropped array
            offsets: Tuple of (z_offset, y_offset, x_offset) indicating crop start positions
            iz_focal_updated: Updated focal point index (should always be at center of cropped Z)
        """
        nz, ny, nx = field.shape

        # Z-axis: crop centered on focal point (keep 50% = ±25% around focal point)
        z_half_range = nz // 4  # Keep 50% total = nz/2 slices
        z_min = max(0, iz_focal - z_half_range)
        z_max = min(nz, iz_focal + z_half_range)

        # Ensure we keep exactly nz//2 slices if possible
        if z_max - z_min < nz // 2:
            if z_min == 0:
                z_max = min(nz, z_min + nz // 2)
            elif z_max == nz:
                z_min = max(0, z_max - nz // 2)

        # X,Y axes: crop from center (keep central 50%)
        y_min = ny // 4
        y_max = ny - (ny // 4)

        x_min = nx // 4
        x_max = nx - (nx // 4)

        # Crop WITHOUT copy - just create a view first
        cropped = field[z_min:z_max, y_min:y_max, x_min:x_max]

        # Update focal point index to cropped coordinates
        iz_focal_cropped = iz_focal - z_min

        return cropped, (z_min, y_min, x_min), iz_focal_cropped

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

    # Test auto-detection
    print(f"\n{'=' * 60}")
    print(f"TESTING AUTO-DETECTION")
    print(f"{'=' * 60}")

    # Initialize with auto-alignment
    voxel_stream = VoxelPowderStream(
        "_arrays/250422_10%3A36%3A17/target_stack.npz",
        nozzle_offset_slices=50,
        auto_detect_working_distance=True,
        visualize=False
    )

    print(f"\nTesting concentration at key points...")
    test_points = np.array([
        [0, 0, 0],  # Laser position = powder center
        [0.001, 0, 0],  # 1mm offset in X
        [0, 0.001, 0],  # 1mm offset in Y
        [0, 0, 0.001],  # 1mm above substrate
    ])

    concentrations = voxel_stream.powder_concentration(test_points, params)
    for pt, conc in zip(test_points, concentrations):
        print(f"  ({pt[0] * 1000:.1f}, {pt[1] * 1000:.1f}, {pt[2] * 1000:.1f}) mm: {conc:.3f} kg/m³")

    # Create visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Evaluate concentration field at substrate
    x = np.linspace(-5e-3, 5e-3, 200)
    y = np.linspace(-5e-3, 5e-3, 200)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X, Y, np.zeros_like(X)], axis=-1)

    # Left: concentration at substrate
    ax = axes[0]
    concentration = voxel_stream.powder_concentration(points, params)

    im = ax.imshow(concentration.reshape(X.shape),
                   extent=[X.min() * 1000, X.max() * 1000, Y.min() * 1000, Y.max() * 1000],
                   origin='lower', cmap='hot', aspect='equal')

    ax.plot(0, 0, 'g+', markersize=15, markeredgewidth=2, label='Laser/Powder Center')
    ax.set_title(f'Concentration at Substrate\nWorking Distance: {voxel_stream.detected_nozzle_height * 1000:.1f} mm')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Concentration (kg/m³)', rotation=270, labelpad=15)

    # Right: XZ cross-section at Y=0
    ax = axes[1]
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
    ax.set_title(f'XZ Cross-section at Y=0')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    peak_conc = np.max(concentration)
    print(f"Peak concentration at substrate: {peak_conc:.3f} kg/m³")

    plt.suptitle('Powder Stream Analysis with Auto-Detection', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Show detailed visualization
    print("\nShowing detailed visualization...")
    voxel_stream._visualize_alignment()