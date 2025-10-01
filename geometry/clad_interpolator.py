from typing import Protocol
import numpy as np
from dataclasses import dataclass

from geometry.clad_profile_function import ParabolicCladProfile

@dataclass
class interpolate_track_old:
    """
    Represents a track function that interpolates between two ParabolicCladProfile functions
    based on distance from a reference point.
    """
    track1: ParabolicCladProfile
    track2: ParabolicCladProfile
    distance1: float
    distance2: float

    def __post_init__(self):
        """Validate distance signs after initialization."""
        if not (self.distance1 * self.distance2 <= 0):
            raise ValueError(
                "Distances must have opposite signs for proper interpolation. "
                f"Got distance1={self.distance1}, distance2={self.distance2}"
            )

    def __call__(self, x: float) -> float:
        """
        Interpolate between two track functions using direct coefficient interpolation.

        Args:
            x: Position to evaluate track height

        Returns:
            Height at position x using interpolated parameters
        """
        # Handle case where tracks are at same position
        if abs(self.distance2 - self.distance1) < 1e-10:
            return max(0, self.track1(x))

        # Calculate interpolation weight
        total_dist = self.distance2 - self.distance1
        weight1 = self.distance2 / total_dist
        weight2 = -self.distance1 / total_dist

        # Get coefficients of both parabolas
        a1, b1, c1 = self.track1.get_coefficients()
        a2, b2, c2 = self.track2.get_coefficients()

        # Interpolate coefficients
        a = weight1 * a1 + weight2 * a2
        b = weight1 * b1 + weight2 * b2
        c = weight1 * c1 + weight2 * c2

        # Get bounds of both tracks
        start1, end1 = self.track1.get_bounds()
        start2, end2 = self.track2.get_bounds()

        # Interpolate bounds
        start_x = weight1 * start1 + weight2 * start2
        end_x = weight1 * end1 + weight2 * end2

        # Return 0 if outside interpolated bounds
        if x < start_x or x > end_x:
            return 0.0

        # Evaluate interpolated parabola
        return max(0, a * x**2 + b * x + c)


def interpolate_track(
        track1: ParabolicCladProfile,
        track2: ParabolicCladProfile,
        distance1: float,
        distance2: float
) -> ParabolicCladProfile:
    """
    Interpolates between two track functions based on distances from a reference point.

    Args:
        track1: First track function
        track2: Second track function
        distance1: Distance from reference point to track1
        distance2: Distance from reference point to track2

    Returns:
        New ParabolicCladProfile with interpolated parameters and clipped bounds

    Raises:
        ValueError: If distances don't have opposite signs for proper interpolation
    """
    if not (distance1 * distance2 <= 0):
        raise ValueError(
            "Distances must have opposite signs for proper interpolation. "
            f"Got distance1={distance1}, distance2={distance2}"
        )

    # Handle case where tracks are at same position
    if abs(distance2 - distance1) < 1e-10:
        return track1

    # Calculate interpolation weights
    total_dist = distance2 - distance1
    weight1 = distance2 / total_dist
    weight2 = -distance1 / total_dist

    # Get coefficients and bounds of both tracks
    a1, b1, c1 = track1.get_coefficients()
    a2, b2, c2 = track2.get_coefficients()
    start1, end1 = track1.get_bounds()
    start2, end2 = track2.get_bounds()

    # Interpolate all parameters
    a = weight1 * a1 + weight2 * a2
    b = weight1 * b1 + weight2 * b2
    c = weight1 * c1 + weight2 * c2
    baseline = weight1 * track1.get_baseline() + weight2 * track2.get_baseline()
    offset = weight1 * track1.offset + weight2 * track2.offset
    start_x = weight1 * start1 + weight2 * start2
    end_x = weight1 * end1 + weight2 * end2

    '''# Find x values where y = 0 using quadratic formula
    roots = []
    if a != 0:
        # New coefficients after expanding (ax² + Bx + C)
        B = b - 2 * a * offset
        C = a * offset ** 2 - b * offset + c + baseline

        discriminant = B * B - 4 * a * C
        if discriminant >= 0:
            x1 = (-B - np.sqrt(discriminant)) / (2 * a)
            x2 = (-B + np.sqrt(discriminant)) / (2 * a)
            roots = sorted([x1, x2])
    elif b != 0:
        # Linear case: bx + C = 0
        C = -b * offset + c + baseline
        roots = [-C / b]

    # Rest of the clipping logic remains the same
    if not roots:
        test_x = (start_x + end_x) / 2
        # Need to evaluate the full function including track_center and baseline
        test_y = a * (test_x - offset) ** 2 + b * (test_x - offset) + c + baseline
        valid_start = start_x if test_y > 0 else 0
        valid_end = end_x if valid_start > 0 else 0
    else:
        # Check if parabola is positive between or outside roots
        test_x = (roots[0] + (roots[-1] if len(roots) > 1 else end_x)) / 2
        test_y = a * (test_x - offset) ** 2 + b * (test_x - offset) + c + baseline

        if test_y > 0:
            # Positive between roots
            valid_start = max(start_x, roots[0] if len(roots) > 1 else roots[0])
            valid_end = min(end_x, roots[-1] if len(roots) > 1 else end_x)
        else:
            # Positive outside roots
            valid_start = max(start_x, roots[-1])
            valid_end = min(end_x, roots[0])'''

    return ParabolicCladProfile(
        a=a,
        b=b,
        c=c,
        start_x=start_x,
        #start_x=valid_start,
        end_x=end_x,
        #end_x=valid_end,
        baseline=baseline,
        offset=offset
    )

def interpolated_track_surface(points, start_profile: ParabolicCladProfile, end_profile: ParabolicCladProfile, length_between, y_position):

    local_x_min = min(start_profile.start_x, end_profile.start_x)
    local_x_max = max(start_profile.end_x, end_profile.end_x)

    # Calculate how far along the track each point is (0 at start, 1 at end)
    # This serves as our interpolation parameter between the two profiles
    y_param = (points[..., 1] - y_position) / length_between

    #print('y_coords range:', points[..., 1].min(), points[..., 1].max())
    #print('length_between:', length_between)
    #print('y_param range before masking:', y_param.min(), y_param.max())



    # Ensure that the interpolation parameter y_param is only valid within the range [0, 1].
    # If y_param is outside this range, it is set to NaN. This is done to handle points
    # that lie outside the track section's length_between, ensuring that only valid interpolation
    # parameters are used for further calculations.
    y_param = np.where((y_param >= 0) & (y_param <= 1), y_param, np.nan)

    # Get heights from both profiles
    x_coords = points[..., 0]
    x_valid = (x_coords >= local_x_min) & (x_coords <= local_x_max)

    #print('x_coords range:', x_coords.min(), x_coords.max())
    #print('local bounds:', local_x_min, local_x_max)

    # Get heights from both profiles
    start_z = start_profile.vectorized(x_coords)
    end_z = end_profile.vectorized(x_coords)

    #print('start_z range:', np.nanmin(start_z), np.nanmax(start_z))
    #print('end_z range:', np.nanmin(end_z), np.nanmax(end_z))

    # Interpolate between profiles and mask invalid x coordinates with NaN
    interpolated_z = (1 - y_param) * start_z + y_param * end_z
    return np.where(x_valid, interpolated_z, np.nan)

if __name__ == "__main__":
    from clad_profile_function import GenerateParabolicCladProfile as Profile, ParabolicCladProfile
    import matplotlib.pyplot as plt

    # Create figure with two subplots
    plt.figure(figsize=(12, 12))

    # Test case 1: Original comparison of initial tracks with different sizes
    width = 1.0
    height = 0.3
    wetting_angle = 0.7125633369575821
    f1 = Profile.generate_profile_function(width, height, track_center=0)
    f2 = Profile.generate_profile_function(width * 2, height * 2, track_center=0)

    plt.subplot(2, 1, 1)
    x = np.linspace(-0.2, 2.2, 200)

    # Plot original tracks
    plt.plot(x, [f1(xi) for xi in x], 'b--', label='Track 1 (Initial)')
    plt.plot(x, [f2(xi) for xi in x], 'r--', label='Track 2 (Initial, 2x size)')

    # Plot interpolated profiles at different positions
    positions = [0.2, 0.35, 0.5]
    colors = ['g', 'c', 'm']

    for pos, color in zip(positions, colors):
        distance1 = pos - 0.0
        distance2 = pos - 0.7

        interpolated_track = interpolate_track(
            track1=f1,
            track2=f2,
            distance1=distance1,
            distance2=distance2
        )

        y = [interpolated_track(xi) for xi in x]
        plt.plot(x, y, color=color, label=f'Interpolated at {pos}')

    plt.grid(True)
    plt.xlabel('Position (mm)')
    plt.ylabel('Height (mm)')
    plt.title('Track ProfileGenerator Interpolation (Between Initial Tracks of Different Sizes)')
    plt.legend()
    plt.axis('equal')

    # Test case 2: Comparison of second tracks with different sizes
    # Create two sets of tracks - one normal sized, one larger
    plt.subplot(2, 1, 2)

    # First set - normal size
    initial1 = Profile.generate_profile_function(width, height, track_center=0)
    second1 = Profile.generate_profile_function(0.7, width, height)

    # Second set - larger size
    initial2 = Profile.generate_profile_function(width * 1.5, height * 1.5, track_center=0)
    second2 = Profile.generate_profile_function(0.7, width * 1.5, height * 1.5)

    # Plot the second tracks from each set
    plt.plot(x, [second1(xi) for xi in x], 'b--', label='Second Track (Normal)')
    plt.plot(x, [second2(xi) for xi in x], 'r--', label='Second Track (1.5x size)')

    # Plot interpolated profiles between the second tracks
    for pos, color in zip(positions, colors):
        distance1 = pos - 0.0
        distance2 = pos - 0.7

        interpolated_track = interpolate_track(
            track1=second1,
            track2=second2,
            distance1=distance1,
            distance2=distance2
        )

        y = [interpolated_track(xi) for xi in x]
        plt.plot(x, y, color=color, label=f'Interpolated at {pos}')

    plt.grid(True)
    plt.xlabel('Position (mm)')
    plt.ylabel('Height (mm)')
    plt.title('Track ProfileGenerator Interpolation (Between Second Tracks of Different Sizes)')
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    # Parameters
    width = 1.0
    height = 0.3
    hatch_distance = 0.7

    # Create track sequences
    initial1 = Profile.generate_initial_profile_function(width, height, track_center=0)
    second1 = Profile.generate_next_profile_function(initial1, hatch_distance, width, height, wetting_angle=wetting_angle)

    initial2 = Profile.generate_initial_profile_function(width * 1.5, height * 1.5, track_center=0)
    second2 = Profile.generate_next_profile_function(initial2, hatch_distance, width * 1.5, height * 1.5, wetting_angle=wetting_angle, baseline=0.1)

    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Track Pair Interpolations at Various Distances', fontsize=16)

    # Test different distance combinations
    distance_pairs = [
        (-0.0, 1.0),  # Original positions
        (-0.2, 0.8),  # 20% interpolation
        (-0.4, 0.6),  # 40% interpolation
        (0.5, -0.5),  # Middle point
        (0.6, -0.4),  # 60% interpolation
        (0.8, -0.2),  # 80% interpolation
    ]

    x = np.linspace(-0.2, 2.2, 200)

    for idx, (d1, d2) in enumerate(distance_pairs):
        ax = axs[idx // 2, idx % 2]

        # Plot original first tracks
        ax.plot(x, [initial1(xi) for xi in x], 'b--', label='First Track 1', alpha=0.3)
        ax.plot(x, [initial2(xi) for xi in x], 'r--', label='First Track 2', alpha=0.3)

        # Interpolate and plot first tracks
        interp_first = interpolate_track(initial1, initial2, d1, d2)
        y_first = [interp_first(xi) for xi in x]
        ax.plot(x, y_first, 'g-', label='Interpolated First Track')

        # Plot original second tracks
        ax.plot(x, [second1(xi) for xi in x], 'b:', label='Second Track 1', alpha=0.3)
        ax.plot(x, [second2(xi) for xi in x], 'r:', label='Second Track 2', alpha=0.3)

        # Interpolate and plot second tracks
        interp_second = interpolate_track(second1, second2, d1, d2)
        y_second = [interp_second(xi) for xi in x]
        ax.plot(x, y_second, 'm-', label='Interpolated Second Track')

        # Formatting
        ax.grid(True)
        ax.set_xlabel('Position (mm)')
        ax.set_ylabel('Height (mm)')
        ax.set_title(f'Interpolation weights: {d1:.1f}, {d2:.1f}')
        if idx == 0:  # Only show legend for first subplot
            ax.legend()
        ax.axis('equal')
        ax.set_ylim(-0.1, 0.8)

    plt.tight_layout()
    plt.show()