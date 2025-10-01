from typing import Callable, Tuple

from scipy import integrate
import numpy as np


class ParabolicCladProfile:
    """
    A callable class representing a parabolic track profile function.
    Implements the quadratic function: f(x) = ax² + bx + c
    """

    def __init__(self, a: float, b: float, c: float, start_x: float, end_x: float, baseline: float = 0.0,
                 metadata: dict = None, offset: float = 0.0, required_area=None):
        """
        Initialize parabolic track with quadratic coefficients, bounds and optional metadata.

        Args:
            a: Coefficient of x²
            b: Coefficient of x
            c: Constant term
            start_x: Starting x-coordinate of track validity
            end_x: Ending x-coordinate of track validity
            baseline: Constant height added to the parabolic profile
            metadata: Optional dictionary for storing track-related information
            offset: Horizontal translation of entire profile
        """
        self.a = a
        self.b = b
        self.c = c
        self.baseline = baseline
        self.start_x = start_x
        self.end_x = end_x
        self.metadata = metadata if metadata is not None else {}
        self.offset = offset
        self.required_area = required_area

        # Dictionary of properties and their validation rules
        validations = {
            'required_area': {
                'value': self.required_area,
                'error_msg': "Required area cannot be negative",
                'condition': lambda x: x >= 0 if x is not None else True
            },
            'max_profile_height': {
                'value': self.get_max_profile_height(),
                'error_msg': "Maximum profile height cannot be negative",
                'condition': lambda x: x >= 0 if x is not None else True
            },
            'max_f': {
                'value': self.get_max_f(),
                'error_msg': "Maximum f value cannot be less than baseline",
                'condition': lambda x: x >= self.baseline if x is not None else True
            },
            'bounds': {
                'value': (self.start_x, self.end_x),
                'error_msg': "Bounds must be defined and start_x must be less than end_x",
                'condition': lambda x: None not in x and x[0] < x[1]
            },
            'integration_result': {
                'value': self.integrate(self.start_x, self.end_x),
                'error_msg': "Integration result cannot be negative",
                'condition': lambda x: x >= 0 if x is not None else True
            }
        }

        # Validate each property
        for prop_name, validation in validations.items():
            value = validation['value']
            if not validation['condition'](value):
                raise ValueError(f"{validation['error_msg']}: {prop_name}: {value}")
                #print(f"{validation['error_msg']}: {prop_name}: {value}")


    def __call__(self, x: float) -> float:
        """
        Evaluate the parabolic function at point x.
        Returns baseline for x outside the track's valid range.
        """
        if self.start_x <= x <= self.end_x:
            x -= self.offset
            return self.a * x ** 2 + self.b * x + self.c + self.baseline
        return 0.0

    @property
    def x_center(self):
        return (self.start_x + self.end_x) / 2

    def vectorized(self, x: np.ndarray) -> np.ndarray:
        mask = (self.start_x <= x) & (x <= self.end_x)
        result = np.zeros_like(x)
        x_valid = x[mask] - self.offset
        result[mask] = self.a * x_valid ** 2 + self.b * x_valid + self.c + self.baseline
        return result

    def get_coefficients(self) -> Tuple[float, float, float]:
        """Return the quadratic coefficients (a, b, c)"""
        return self.a, self.b, self.c

    def get_bounds(self) -> Tuple[float, float]:
        """Return the track's valid x-range (start_x, end_x)"""
        return self.start_x, self.end_x

    def get_baseline(self) -> float:
        """Return the track's baseline height"""
        return self.baseline

    def get_max_x(self) -> float:
        """
        Calculate the x-coordinate of the profile's maximum point.

        For a parabola ax² + bx + c, the maximum occurs at x = -b/(2a).
        When including the track_center transformation, we add the track_center to this value.
        If the theoretical maximum is outside the valid range [start_x, end_x],
        returns the boundary point (start_x or end_x) with the higher function value.

        Returns:
            float: x-coordinate of the maximum point
        """
        theoretical_max_x = -self.b / (2 * self.a) + self.offset

        # Check if theoretical maximum is within bounds
        if self.start_x <= theoretical_max_x <= self.end_x:
            return theoretical_max_x

        # If outside bounds, evaluate both endpoints and return x with higher value
        start_val = self(self.start_x)
        end_val = self(self.end_x)

        return self.start_x if start_val > end_val else self.end_x

    def get_max_f(self) -> float:
        """
        Calculate the y-coordinate of the profile's maximum point.

        For a parabola ax² + bx + c, the maximum y-value is -b²/(4a) + c.
        When including the baseline, we add it to this value.
        If the theoretical maximum is outside the valid range [start_x, end_x],
        returns the highest function value at the boundaries.

        Returns:
            float: y-coordinate of the maximum point
        """
        # Get x-coordinate of the actual maximum (considering bounds)
        max_x = self.get_max_x()

        # Return the function evaluation at this point
        return self(max_x)

    def get_max_point(self) -> Tuple[float, float]:
        """
        Get both coordinates of the profile's maximum point.

        Returns:
            tuple[float, float]: (x, y) coordinates of the maximum point
        """
        return self.get_max_x(), self.get_max_f()

    def get_max_profile_height(self) -> float:
        """
        Calculate the maximum height of just the profile without baseline.

        For a parabola ax² + bx + c, the maximum y-value is -b²/(4a) + c.
        This represents the intrinsic height of the track profile before any baseline track_center.
        If the theoretical maximum is outside the valid range [start_x, end_x],
        returns the highest value at the boundaries (excluding baseline).

        Returns:
            float: Maximum height of the profile curve without baseline
        """
        # Get x-coordinate of the actual maximum (considering bounds)
        max_x = self.get_max_x()

        # Evaluate at max_x but subtract the baseline from result
        return self(max_x) - self.baseline

    def find_baseline_intersections(self):
        """
        Find x coordinates where the track profile intersects with the baseline.
        Returns tuple of (lower_x, upper_x). Returns None for any non-existent intersection.
        """
        # Quadratic formula: (-b ± √(b² - 4ac))/(2a)
        a = self.a
        b = self.b
        c = self.c - self.baseline

        # Handle degenerate cases
        if abs(a) < 1e-10:  # Linear case
            if abs(b) < 1e-10:  # Constant case
                return (None, None)
            x = -c / b
            return (x, x)

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return (None, None)

        x1 = (-b - (discriminant ** 0.5)) / (2 * a)
        x2 = (-b + (discriminant ** 0.5)) / (2 * a)

        # Account for track_center offset
        x1 += self.offset
        x2 += self.offset

        return (min(x1, x2), max(x1, x2))

    def integrate(self, x1: float, x2: float, remove_baseline=False) -> float:
        """
        Analytically compute the integral of the track profile between two x coordinates.
        Takes into account the track_center and baseline.

        Args:
            x1: Lower bound of integration
            x2: Upper bound of integration

        Returns:
            Area under the curve between x1 and x2
        """

        def F(x: float) -> float:
            """Indefinite integral of ax² + bx + c"""
            x = x - self.offset  # Apply track_center transformation
            return (self.a * x ** 3) / 3 + (self.b * x ** 2) / 2 + self.c * x + self.baseline * x

        # Clamp integration bounds to track validity region
        if x1 is None or x2 is None:
            raise ValueError(f"Integration bounds must be defined: x1={x1}, x2={x2}")
        baseline_x1, baseline_x2 = self.find_baseline_intersections()
        if baseline_x1 is not None:
            x1 = max(x1, baseline_x1)
        if baseline_x2 is not None:
            x2 = min(x2, baseline_x2)
        x1 = max(x1, self.start_x)
        x2 = min(x2, self.end_x)

        if x2 <= x1:
            return 0.0

        result = F(x2) - F(x1)

        if remove_baseline:
            result -= self.baseline * (x2 - x1)
        return result

    def plot(self, title: str = None, laser_position: float = None):
        """
        Plot the parabolic profile with optional laser position, intersection point,
        initial profile, and pre-track if available in metadata.

        Args:
            title: Optional title for the plot
            laser_position: Optional x-coordinate of laser position to mark on plot

        Returns:
            ParabolicCladProfile: The initial profile object if created
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        # Generate x values for plotting
        x = np.linspace(self.start_x, self.end_x, 200)
        y = self.vectorized(x)

        # Convert to mm for plotting
        x_mm = x * 1000
        y_mm = y * 1000

        # Plot pre-track if it exists in metadata
        if self.metadata is not None and 'prev_profile' in self.metadata:
            pre_track = self.metadata['prev_profile']
            x_pre = np.linspace(pre_track.start_x, pre_track.end_x, 200)
            y_pre = pre_track.vectorized(x_pre)
            plt.plot(x_pre * 1000, y_pre * 1000, 'g--', linewidth=2, label='Previous Track')

        '''# Plot initial profile if width and height are in metadata
        initial_profile = None
        if self.metadata is not None:
            if 'init_profile' in self.metadata:
                initial_profile = self.metadata['init_profile']

            elif 'width' in self.metadata and 'height' in self.metadata:
                width = self.metadata['width']
                height = self.metadata['height']

                # Get initial profile using the static method
                initial_profile = GenerateParabolicCladProfile.generate_initial_profile_function(
                    width=width,
                    height=height,
                    track_center=self.offset + self.metadata.get('dummy_offset', 0.0),
                    baseline=self.baseline,
                    metadata=self.metadata,
                    force_additional_material_balance=self.metadata.get('force_additional_material_balance', 0.0)
                )

        if initial_profile is not None:
            x_init = np.linspace(initial_profile.start_x, initial_profile.end_x, 200)
            y_init = initial_profile.vectorized(x_init)

            # Convert to mm and plot
            plt.plot(x_init * 1000, y_init * 1000, 'r--', linewidth=2, label='Initial ProfileGenerator')'''

        # Plot current profile
        plt.plot(x_mm, y_mm, 'b-', linewidth=2, label='Current ProfileGenerator')

        # Plot maximum point
        max_x, max_y = self.get_max_point()
        plt.plot(max_x * 1000, max_y * 1000, 'ro', markersize=8, label='Maximum Point')

        # Plot laser position if provided
        if laser_position is not None:
            laser_y = self(laser_position)
            plt.plot(laser_position * 1000, laser_y * 1000, 'g*', markersize=12, label='Laser Position')

        # Plot intersection point if it exists in metadata
        if self.metadata is not None and 'intersection_point' in self.metadata:
            int_x, int_y = self.metadata['intersection_point']
            plt.plot(int_x * 1000, int_y * 1000, 'mx', markersize=10, label='Intersection Point')

        if self.metadata is not None:
            title = title + ' ' if title else ''
            if 'layer' in self.metadata:
                title += f'Layer {self.metadata["layer"]}'
            if 'track' in self.metadata:
                title += f', Track {self.metadata["track"]}'
            if 'y_pos' in self.metadata:
                title += f' (Y-Position: {self.metadata["y_pos"] * 1000:.3f}mm)'

        # Add labels and title
        plt.xlabel('Position (mm)')
        plt.ylabel('Height (mm)')
        if title:
            plt.title(title)
        plt.grid(True)
        plt.legend()

        plt.show()

        return #initial_profile


def find_intersection_points(p1: ParabolicCladProfile, p2: ParabolicCladProfile, tol: float = 1e-10) -> Tuple[
    float, float]:
    """
    Find intersection points between two parabolic profiles by solving quadratic equation.

    Derivation:
    1. ProfileGenerator equations (including track_center and baseline):
       p1: a₁(x - track_center₁)² + b₁(x - track_center₁) + c₁ + baseline₁
       p2: a₂(x - track_center₂)² + b₂(x - track_center₂) + c₂ + baseline₂

    2. Expand squares:
       p1: a₁x² - 2a₁x·track_center₁ + a₁track_center₁² + b₁x - b₁track_center₁ + c₁ + baseline₁
       p2: a₂x² - 2a₂x·track_center₂ + a₂track_center₂² + b₂x - b₂track_center₂ + c₂ + baseline₂

    3. Set equal and rearrange to standard form (Ax² + Bx + C = 0):
       (a₁ - a₂)x² + (-2a₁track_center₁ + 2a₂track_center₂ + b₁ - b₂)x +
       (a₁track_center₁² - b₁track_center₁ + c₁ + baseline₁ - a₂track_center₂² + b₂track_center₂ - c₂ - baseline₂) = 0

    Args:
        p1: First parabolic profile
        p2: Second parabolic profile
        tol: Tolerance for numerical comparisons and range checking

    Returns:
        Tuple of (x₁, x₂) intersection points. Points may be None if no valid intersection exists.
    """

    def is_in_range(x: float, start: float, end: float) -> bool:
        """Check if x is within [start, end] with tolerance."""
        return (start - tol) <= x <= (end + tol)

    # Extract parameters from profiles
    a1, b1, c1 = p1.get_coefficients()
    a2, b2, c2 = p2.get_coefficients()
    offset1 = p1.offset
    offset2 = p2.offset
    baseline1 = p1.get_baseline()
    baseline2 = p2.get_baseline()

    # Calculate quadratic coefficients A, B, C
    A = a1 - a2
    B = -2 * a1 * offset1 + 2 * a2 * offset2 + b1 - b2
    C = (a1 * offset1 ** 2 - b1 * offset1 + c1 + baseline1 -
         a2 * offset2 ** 2 + b2 * offset2 - c2 - baseline2)

    # Handle case where profiles are nearly identical or linear
    if abs(A) < tol:
        if abs(B) < tol:
            return None, None  # Profiles are identical or parallel
        # Linear case: -Bx - C = 0
        x = -C / B
        if (is_in_range(x, p1.start_x, p1.end_x) and
                is_in_range(x, p2.start_x, p2.end_x)):
            return x, None
        return None, None

    # Calculate discriminant
    discriminant = B ** 2 - 4 * A * C

    if discriminant < -tol:  # Allow small negative values due to numerical error
        return None, None  # No real intersection points

    # Handle near-zero discriminant (tangent case)
    if abs(discriminant) < tol:
        x = -B / (2 * A)
        if (is_in_range(x, p1.start_x, p1.end_x) and
                is_in_range(x, p2.start_x, p2.end_x)):
            return x, None
        return None, None

    # Calculate both roots
    sqrt_discriminant = np.sqrt(max(0, discriminant))  # Ensure non-negative
    x1 = (-B - sqrt_discriminant) / (2 * A)
    x2 = (-B + sqrt_discriminant) / (2 * A)

    # Sort roots for convenience
    if x1 > x2:
        x1, x2 = x2, x1

    # Check if roots are within combined valid range of both profiles
    min_x = min(p1.start_x, p2.start_x)
    max_x = max(p1.end_x, p2.end_x)
    x1 = x1 if is_in_range(x1, min_x, max_x) else None
    x2 = x2 if is_in_range(x2, min_x, max_x) else None

    return x1, x2


class GenerateParabolicCladProfile:
    """
    Implements multi-track profile model from:
    'A new physics-based model for laser directed energy deposition
    (powder-fed additive manufacturing): From single-track to multi-track and multi-layer'
    - Huang et al. (2019)

    This class provides static methods to generate and analyze overlapping parabolic
    track profiles in directed energy deposition processes. Each track maintains geometric
    continuity with previous tracks while preserving material volume conservation.
    """

    @staticmethod
    def calculate_required_area(
            width: float,
            height: float,
            integrate_start: float,
            integrate_end: float,
            cross_section: Callable[[float], float] = None,
            force_additional_material_balance: float = 0.0,
    ) -> float:
        """
        Computes required area for new track based on material conservation, considering
        an arbitrary cross-section profile beneath.

        Args:
            width: Width of the new track
            height: Height of the new track
            integrate_start: Start x-coordinate for integration
            integrate_end: End x-coordinate for integration
            cross_section: Callable function that returns the height of the complete profile
                          at any given x-coordinate. If None, assumes zero height.
            force_additional_material_balance: Additional area to consider

        Returns:
            float: Required cross-sectional area for new track

        Notes:
            - The new material contribution follows the standard parabolic profile area formula (2/3 * width * height)
            - If a cross_section function is provided, integrates the cross-section height
              over the specified range
        """
        # Validate inputs
        if integrate_end < integrate_start:
            raise ValueError("Integration end point must be greater than start point")

        # Calculate base material contribution (parabolic profile area = 2/3 * width * height)
        required_area = (2 * width * height) / 3

        # If there's a cross-section to consider
        if cross_section is not None:
            # Add the integrated area of the cross-section
            additional_area = integrate.quad(
                cross_section,
                a=integrate_start,
                b=integrate_end,
                limit=100
            )[0]

            required_area += additional_area

        # Add any forced additional material balance
        return required_area + force_additional_material_balance

    @staticmethod
    def solve_profile_parameters(
            start_x: float,
            end_x: float,
            cross_section: Callable[[float], float],
            required_area: float,
    ) -> np.ndarray:
        """
        Solves system of equations for track profile parameters using three constraints:
        1. Profile intersects with cross section at start_x
        2. Profile intersects with cross section at end_x
        3. Profile satisfies required area condition for material conservation

        Args:
            start_x: Start x-coordinate of the track
            end_x: End x-coordinate of the track
            cross_section: Callable function that returns the height at any x-coordinate
            required_area: Required cross-sectional area to satisfy material conservation

        Returns:
            Array of quadratic coefficients [a, b, c]

        Notes:
            Solves the system of equations:
            [a*start_x² + b*start_x + c = cross_section(start_x)]  # Intersection at start
            [a*end_x² + b*end_x + c = cross_section(end_x)]       # Intersection at end
            [∫(ax² + bx + c)dx from start_x to end_x = required_area] # Area constraint
        """
        # Get intersection heights from cross section
        start_height = cross_section(start_x)
        end_height = cross_section(end_x)

        # Set up system of equations
        A = np.array([
            [start_x ** 2, start_x, 1],  # Intersection at start_x
            [end_x ** 2, end_x, 1],  # Intersection at end_x
            [  # Area constraint
                (end_x ** 3 - start_x ** 3) / 3,  # Integral of ax²
                (end_x ** 2 - start_x ** 2) / 2,  # Integral of bx
                end_x - start_x  # Integral of c
            ]
        ])

        # Right hand side of equations
        b = np.array([
            start_height,  # Height at start_x
            end_height,  # Height at end_x
            required_area  # Required area
        ])

        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError as e:
            raise Exception(f"Failed to solve profile parameters. Matrix A: {A}, vector b: {b}") from e

    @staticmethod
    def generate_profile_function(
            width: float,
            height: float,
            track_center: float,
            cross_section: Callable[[float], float] = None,
            metadata: dict = None,
            force_additional_material_balance: float = 0.0,
    ) -> ParabolicCladProfile:
        """
        Generates a parabolic profile function based on geometric parameters and an optional
        existing cross section profile.

        Args:
            width: Width of the track
            height: Height of the track
            track_center: Center position of the track
            cross_section: Optional callable function that returns the height at any x-coordinate.
                          If None, assumes zero height everywhere.
            metadata: Optional metadata dictionary for the track
            force_additional_material_balance: Additional area consideration

        Returns:
            ParabolicCladProfile object representing the track profile

        Notes:
            - If no cross_section is provided, generates an initial profile
            - If cross_section is provided, generates a profile that properly intersects with it
            - Profile is determined by solving a system of equations ensuring:
              1. Intersection with cross_section at boundaries
              2. Required area for material conservation
        """
        # Initialize metadata
        metadata = metadata if metadata is not None else {}
        metadata['width'] = width
        metadata['height'] = height
        metadata['force_additional_material_balance'] = force_additional_material_balance

        # Calculate profile bounds
        start_x = track_center - width / 2
        end_x = track_center + width / 2

        # If no cross section provided, use zero height function
        if cross_section is None:
            cross_section = lambda x: 0.0

        # Calculate required area
        required_area = GenerateParabolicCladProfile.calculate_required_area(
            width=width,
            height=height,
            integrate_start=start_x,
            integrate_end=end_x,
            cross_section=cross_section,
            force_additional_material_balance=force_additional_material_balance
        )

        # Solve for profile parameters
        a, b, c = GenerateParabolicCladProfile.solve_profile_parameters(
            start_x=start_x,
            end_x=end_x,
            cross_section=cross_section,
            required_area=required_area
        )

        # Create and return the profile
        profile = ParabolicCladProfile(
            a=a,
            b=b,
            c=c,
            start_x=start_x,
            end_x=end_x,
            baseline=0.0,  # No baseline needed in new approach
            metadata=metadata,
            required_area=required_area
        )

        return profile


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from typing import List, Callable, Optional, NamedTuple
    from dataclasses import dataclass


    @dataclass
    class ProfileParameters:
        """Parameters for profile generation"""
        width: float  # Track width in mm
        height: float  # Track height in mm
        hatch_distance: float  # Distance between track centers in mm


    class ProfileTester:
        def __init__(self, params: ProfileParameters):
            self.params = params

        @staticmethod
        def visualize_tracks(
                tracks: List[ParabolicCladProfile],
                title: str = "Track Profiles",
                show_bounds: bool = True,
                show_baseline: Optional[float] = None
        ) -> None:
            """Visualizes multiple track profiles with optional features"""
            plt.figure(figsize=(12, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, len(tracks)))

            # Plot each track
            for i, track in enumerate(tracks):
                start_x, end_x = track.get_bounds()
                x = np.linspace(start_x, end_x, 100)
                y = [track(xi) for xi in x]
                plt.plot(x, y, c=colors[i], label=f'Track {i}')

                if show_bounds:
                    plt.plot([start_x, end_x],
                             [track(start_x), track(end_x)],
                             'ko', markersize=4)

            # Show baseline if specified
            if show_baseline is not None:
                min_x = min(t.start_x for t in tracks)
                max_x = max(t.end_x for t in tracks)
                plt.hlines(show_baseline, min_x, max_x,
                           'k', linestyles='--', alpha=0.3, label='Baseline')

            plt.grid(True)
            plt.xlabel('Position (mm)')
            plt.ylabel('Height (mm)')
            plt.title(title)
            plt.legend()
            plt.axis('equal')
            plt.show()

        @staticmethod
        def combine_profiles(profiles: List[ParabolicCladProfile]) -> Callable[[float], float]:
            """Creates a combined height function from multiple profiles"""

            def combined_height(x: float) -> float:
                return max(profile(x) for profile in profiles) if profiles else 0.0

            return combined_height

        @staticmethod
        def create_baseline_function(height: float) -> Callable[[float], float]:
            """Creates a constant height function"""
            return lambda x: height

        def generate_track_sequence(
                self,
                num_tracks: int,
                start_center: float,
                direction: int = 1,  # 1 for left-to-right, -1 for right-to-left
                baseline_increment: float = 0.0,
                metadata_prefix: str = ""
        ) -> List[ParabolicCladProfile]:
            """Generates a sequence of tracks with optional baseline increment"""
            tracks = []

            # Initialize first track
            f0 = GenerateParabolicCladProfile.generate_profile_function(
                width=self.params.width,
                height=self.params.height,
                track_center=start_center,
                metadata={'track_number': f"{metadata_prefix}0", 'layer': 1}
            )
            tracks.append(f0)

            # Generate subsequent tracks
            for i in range(num_tracks - 1):
                baseline_height = (i + 1) * baseline_increment

                # Create cross section combining previous tracks and baseline
                prev_cross_section = self.combine_profiles(tracks)
                if baseline_increment > 0:
                    baseline_func = self.create_baseline_function(baseline_height)

                    def cross_section(x: float, p=prev_cross_section, b=baseline_func):
                        return max(p(x), b(x))
                else:
                    cross_section = prev_cross_section

                # Generate next track
                next_track = GenerateParabolicCladProfile.generate_profile_function(
                    width=self.params.width,
                    height=self.params.height,
                    track_center=start_center + direction * self.params.hatch_distance * (i + 1),
                    cross_section=cross_section,
                    metadata={'track_number': f"{metadata_prefix}{i + 1}", 'layer': 1}
                )
                tracks.append(next_track)

            return tracks

        def test_overlap_ratios(self, overlap_ratios: List[float] = None) -> None:
            """
            Tests and visualizes track profiles with different overlap ratios in a single figure
            for easy comparison. Keeps hatch distance constant while varying width.

            Args:
                overlap_ratios: List of overlap ratios as percentages (0-200).
                               If None, uses range from 0% to 200% in 10% steps.
            """
            if overlap_ratios is None:
                overlap_ratios = list(range(0, 101, 10))  # 0% to 200% in 10% steps

            # Calculate number of rows and columns for subplots
            n_plots = len(overlap_ratios)
            n_cols = 2
            n_rows = (n_plots + n_cols - 1) // n_cols

            # Create a single figure for all plots
            fig = plt.figure(figsize=(15, 2 * n_rows))  # Adjusted figure height

            # Set consistent y limits
            y_min = 0.0
            y_max = 1.0

            # Create all plots
            for i, overlap_ratio in enumerate(overlap_ratios):
                width = self.params.width
                hatch_distance = width - (overlap_ratio / 100) * width
                print(overlap_ratio, (width - hatch_distance) / width)

                test_params = ProfileParameters(
                    width=self.params.width,
                    height=self.params.height,
                    hatch_distance=hatch_distance
                )

                tester = ProfileTester(test_params)
                tracks = tester.generate_track_sequence(
                    num_tracks=10,
                    start_center=0,
                    metadata_prefix=f"OVERLAP_{overlap_ratio}_"
                )

                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                colors = plt.cm.tab10(np.linspace(0, 1, len(tracks)))

                for j, track in enumerate(tracks):
                    start_x, end_x = track.get_bounds()
                    x = np.linspace(start_x, end_x, 100)
                    y = [track(xi) for xi in x]
                    ax.plot(x, y, c=colors[j % 10], linewidth=1)

                    # Plot track boundaries
                    ax.plot([start_x, end_x],
                            [track(start_x), track(end_x)],
                            'ko', markersize=2)

                # Set axis limits and formatting
                ax.set_ylim(y_min, y_max)
                ax.set_xlim(-1, 7)  # Fixed x-axis range for all plots
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Overlap: {overlap_ratio}%')

                # Set tick formatting
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
                ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                ax.xaxis.set_major_locator(plt.MultipleLocator(2))
                ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

                # Show y-axis labels only for leftmost plots
                if i % n_cols != 0:
                    ax.set_yticklabels([])

            # Add common labels
            fig.text(0.5, 0.02, 'Position (mm)', ha='center', va='center')
            fig.text(0.02, 0.5, 'Height (mm)', ha='center', va='center', rotation='vertical')

            # Adjust layout
            plt.tight_layout()
            plt.show()

        def test_overlap_ratios_multi_layer(self, overlap_ratios: List[float] = None, num_layers: int = 3) -> None:
            """
            Tests and visualizes track profiles with different overlap ratios in a single figure
            for easy comparison, showing multiple stacked layers with proper cross sections.

            Args:
                overlap_ratios: List of overlap ratios as percentages (0-100).
                               If None, uses range from 0% to 100% in 10% steps.
                num_layers: Number of layers to stack
            """
            if overlap_ratios is None:
                overlap_ratios = list(range(0, 101, 10))

            n_plots = len(overlap_ratios)
            n_cols = 2
            n_rows = (n_plots + n_cols - 1) // n_cols

            fig = plt.figure(figsize=(15, 2 * n_rows))
            y_min = 0.0
            y_max = 1.0 * num_layers

            for i, overlap_ratio in enumerate(overlap_ratios):
                width = self.params.width
                hatch_distance = width - (overlap_ratio / 100) * width
                print(overlap_ratio, (width - hatch_distance) / width)

                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                colors = plt.cm.tab10(np.linspace(0, 1, 10))

                # Track all profiles for cross sections
                all_profiles = []

                # Generate and plot each layer
                for layer in range(num_layers):
                    test_params = ProfileParameters(
                        width=self.params.width,
                        height=self.params.height,
                        hatch_distance=hatch_distance
                    )

                    tester = ProfileTester(test_params)

                    # Create cross section function from all previous tracks
                    if all_profiles:
                        def cross_section(x):
                            return max(profile(x) for profile in all_profiles)
                    else:
                        cross_section = lambda x: 0.0

                    # Generate tracks for this layer
                    tracks = []
                    for track_idx in range(10):  # 10 tracks per layer
                        if track_idx == 0:
                            # First track in layer
                            track = GenerateParabolicCladProfile.generate_profile_function(
                                width=width,
                                height=self.params.height,
                                track_center=0,
                                cross_section=cross_section,
                                metadata={'track_number': f"OVERLAP_{overlap_ratio}_LAYER_{layer}_TRACK_0"}
                            )
                        else:
                            # Combine previous tracks in this layer with previous layer cross section
                            def combined_cross_section(x):
                                layer_max = max((t(x) for t in tracks), default=0.0)
                                prev_layers_max = cross_section(x)
                                return max(layer_max, prev_layers_max)

                            track = GenerateParabolicCladProfile.generate_profile_function(
                                width=width,
                                height=self.params.height,
                                track_center=track_idx * hatch_distance,
                                cross_section=combined_cross_section,
                                metadata={'track_number': f"OVERLAP_{overlap_ratio}_LAYER_{layer}_TRACK_{track_idx}"}
                            )

                        tracks.append(track)

                        # Plot the track
                        start_x, end_x = track.get_bounds()
                        x = np.linspace(start_x, end_x, 100)
                        y = [track(xi) for xi in x]
                        ax.plot(x, y, c=colors[track_idx % 10], linewidth=1)
                        ax.plot([start_x, end_x],
                                [track(start_x), track(end_x)],
                                'ko', markersize=2)

                    # Add all tracks from this layer to the profile list
                    all_profiles.extend(tracks)

                # Set axis limits and formatting
                ax.set_ylim(y_min, y_max)
                ax.set_xlim(-1, 7)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Overlap: {overlap_ratio}%')

                # Set tick formatting
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
                ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
                ax.xaxis.set_major_locator(plt.MultipleLocator(2))
                ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

                if i % n_cols != 0:
                    ax.set_yticklabels([])

            fig.text(0.5, 0.02, 'Position (mm)', ha='center', va='center')
            fig.text(0.02, 0.5, 'Height (mm)', ha='center', va='center', rotation='vertical')

            plt.tight_layout()
            plt.show()

    def run_test_cases(params: ProfileParameters):
        tester = ProfileTester(params)

        # Test Case 1: Basic left-to-right sequence
        print("\nTesting left-to-right track sequence...")
        ltr_tracks = tester.generate_track_sequence(
            num_tracks=4,
            start_center=2,
            metadata_prefix="LTR_"
        )
        tester.visualize_tracks(ltr_tracks, "Left to Right Track Sequence")

        # Test Case 2: Right-to-left sequence
        print("\nTesting right-to-left track sequence...")
        rtl_tracks = tester.generate_track_sequence(
            num_tracks=4,
            start_center=0,
            direction=-1,
            metadata_prefix="RTL_"
        )
        tester.visualize_tracks(rtl_tracks, "Right to Left Track Sequence")

        # Test Case 3: Sequence with increasing baselines (left-to-right)
        print("\nTesting track sequence with increasing baselines...")
        baseline_tracks_ltr = tester.generate_track_sequence(
            num_tracks=4,
            start_center=2,
            baseline_increment=0.2,
            metadata_prefix="BASE_LTR_"
        )
        tester.visualize_tracks(
            baseline_tracks_ltr,
            "Track Sequence with Increasing Baselines (Left to Right)",
            show_baseline=0.6
        )

        # Test Case 4: Sequence with increasing baselines (right-to-left)
        print("\nTesting track sequence with increasing baselines (right-to-left)...")
        baseline_tracks_rtl = tester.generate_track_sequence(
            num_tracks=4,
            start_center=0,
            direction=-1,
            baseline_increment=0.2,
            metadata_prefix="BASE_RTL_"
        )
        tester.visualize_tracks(
            baseline_tracks_rtl,
            "Track Sequence with Increasing Baselines (Right to Left)",
            show_baseline=0.6
        )

        # Test Case 5: Single track demonstrations
        print("\nDemonstrating single track profiles...")
        # Normal track
        single_track = tester.generate_track_sequence(
            num_tracks=1,
            start_center=0,
            metadata_prefix="SINGLE_"
        )[0]

        # Track with baseline
        baseline_height = 0.1
        single_track_baseline = GenerateParabolicCladProfile.generate_profile_function(
            width=params.width,
            height=params.height,
            track_center=0,
            cross_section=tester.create_baseline_function(baseline_height),
            metadata={'track_number': "SINGLE_BASELINE", 'layer': 1}
        )

        tester.visualize_tracks([single_track], "Single Track Profile")
        tester.visualize_tracks(
            [single_track_baseline],
            "Single Track Profile with Baseline",
            show_baseline=baseline_height
        )


    # Example parameter sets
    default_params = ProfileParameters(
        width=0.0011597105709616968 * 1000,
        height=0.00025049621353903504 * 1000,
        hatch_distance=0.7
    )

    huang_params = ProfileParameters(
        width=1.0,
        height=0.3,
        hatch_distance=0.7
    )

    #run_test_cases(default_params)
    #run_test_cases(huang_params)

    tester = ProfileTester(default_params)
    # Add overlap ratio test
    print("\nTesting different overlap ratios...")
    #tester.test_overlap_ratios()
    tester.test_overlap_ratios_multi_layer()
