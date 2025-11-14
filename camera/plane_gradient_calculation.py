import numpy as np

# Given dimensions
small_plane_width = 46.13  # mm (small side)
small_plane_length = 61.51  # mm (large side)

big_plane_width = 89.65  # mm (small side)
big_plane_length = 119.53  # mm (large side)

perpendicular_distance = 100  # mm (distance between planes)

# Calculate the differences (half on each side since centered)
width_diff_half = (big_plane_width - small_plane_width) / 2
length_diff_half = (big_plane_length - small_plane_length) / 2

print("=" * 60)
print("PLANE GRADIENT ANALYSIS")
print("=" * 60)
print(f"\nSmall plane: {small_plane_width} × {small_plane_length} mm")
print(f"Big plane: {big_plane_width} × {big_plane_length} mm")
print(f"Perpendicular distance between planes: {perpendicular_distance} mm")

print("\n" + "-" * 60)
print("DIMENSION DIFFERENCES (half, per side)")
print("-" * 60)
print(f"Small side (width) difference per side: {width_diff_half:.4f} mm")
print(f"Large side (length) difference per side: {length_diff_half:.4f} mm")

print("\n" + "-" * 60)
print("GRADIENTS (Steigung)")
print("-" * 60)
gradient_small_side = width_diff_half / perpendicular_distance
gradient_large_side = length_diff_half / perpendicular_distance

print(f"Small side gradient: {gradient_small_side:.6f}")
print(f"Large side gradient: {gradient_large_side:.6f}")

# Calculate angles
angle_small_side_rad = np.arctan(gradient_small_side)
angle_large_side_rad = np.arctan(gradient_large_side)
angle_small_side_deg = np.degrees(angle_small_side_rad)
angle_large_side_deg = np.degrees(angle_large_side_rad)

print(f"\nSmall side angle: {angle_small_side_deg:.4f}°")
print(f"Large side angle: {angle_large_side_deg:.4f}°")

print("\n" + "-" * 60)
print("TRAVEL ALONG SMALL SIDE GRADIENT")
print("-" * 60)
travel_distance_along_gradient = 35  # mm
print(f"Distance traveled along gradient: {travel_distance_along_gradient} mm")

# Method 1: Using trigonometry with angle
perpendicular_component_method1 = travel_distance_along_gradient * np.cos(angle_small_side_rad)
horizontal_component_method1 = travel_distance_along_gradient * np.sin(angle_small_side_rad)

print(f"\nMethod 1 (using angle):")
print(f"  Perpendicular component: {perpendicular_component_method1:.4f} mm")
print(f"  Horizontal component: {horizontal_component_method1:.4f} mm")

# Method 2: Using gradient ratio directly
# The slant distance is sqrt(100^2 + width_diff_half^2) for 100mm perpendicular
slant_distance_for_100mm = np.sqrt(perpendicular_distance**2 + width_diff_half**2)
ratio_perpendicular_to_slant = perpendicular_distance / slant_distance_for_100mm

perpendicular_component_method2 = travel_distance_along_gradient * ratio_perpendicular_to_slant
horizontal_component_method2 = travel_distance_along_gradient * (width_diff_half / slant_distance_for_100mm)

print(f"\nMethod 2 (using ratio):")
print(f"  Slant distance for 100mm perpendicular: {slant_distance_for_100mm:.4f} mm")
print(f"  Ratio (perpendicular/slant): {ratio_perpendicular_to_slant:.6f}")
print(f"  Perpendicular component: {perpendicular_component_method2:.4f} mm")
print(f"  Horizontal component: {horizontal_component_method2:.4f} mm")

# Verification: check that components squared add up to travel distance squared
verification = np.sqrt(perpendicular_component_method1**2 + horizontal_component_method1**2)
print(f"\nVerification (should equal {travel_distance_along_gradient}):")
print(f"  sqrt(perp² + horiz²) = {verification:.4f} mm ✓")

print("\n" + "=" * 60)
print(f"ANSWER: {perpendicular_component_method1:.2f} mm perpendicular")
print("=" * 60)#


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Given dimensions
small_plane_width = 46.13  # mm (small side)
small_plane_length = 61.51  # mm (large side)

big_plane_width = 89.65  # mm (small side)
big_plane_length = 119.53  # mm (large side)

perpendicular_distance_between_planes = 100  # mm (distance between planes along their normal)

# Travel distance from small plane where intersection happens
travel_distance_along_gradient = 35  # mm (along the gradient)
perpendicular_travel = 34.20  # mm (calculated before, perpendicular between the two planes)

# Angle of the plane system to the substrate
tilt_angle_deg = 40  # degrees
tilt_angle_rad = np.radians(tilt_angle_deg)

print("=" * 70)
print("3D VECTOR CALCULATION: INTERSECTION TO SMALL PLANE CENTER")
print("=" * 70)

# Set up coordinate system:
# Origin at intersection point on substrate plane
# X-axis: along substrate plane (horizontal)
# Y-axis: perpendicular to X in substrate plane
# Z-axis: vertical (up)

print("\n" + "-" * 70)
print("COORDINATE SYSTEM")
print("-" * 70)
print("Origin: Intersection point on substrate plane")
print("X-axis: Horizontal along substrate (direction of plane tilt)")
print("Y-axis: Horizontal perpendicular to X")
print("Z-axis: Vertical (up)")

# The center line between the two planes is tilted at 40° to the substrate
# At the intersection point, we are 34.20 mm from the small plane (perpendicular)
# We need to go back along the tilted center line

print("\n" + "-" * 70)
print("GEOMETRY SETUP")
print("-" * 70)
print(f"Tilt angle: {tilt_angle_deg}°")
print(f"Distance from intersection to small plane (along center line): {perpendicular_travel:.4f} mm")
print("Small plane is ABOVE substrate, big plane is BELOW")

# Vector along the tilted center line (pointing from big plane to small plane)
# This vector is in the X-Z plane, tilted 40° from horizontal
# Direction: going backwards (negative X) and upwards (positive Z)

# Components of the center line vector
# Moving perpendicular_travel mm along the center line at 40° angle
dx = -perpendicular_travel * np.cos(tilt_angle_rad)  # negative because going back
dz = perpendicular_travel * np.sin(tilt_angle_rad)   # positive because going up
dy = 0  # no movement in Y direction (perpendicular to tilt plane)

vector_to_small_plane = np.array([dx, dy, dz])

print("\n" + "-" * 70)
print("VECTOR FROM INTERSECTION TO SMALL PLANE CENTER")
print("-" * 70)
print(f"Vector components:")
print(f"  X (horizontal along tilt): {dx:.4f} mm")
print(f"  Y (perpendicular to tilt): {dy:.4f} mm")
print(f"  Z (vertical):              {dz:.4f} mm")
print(f"\nVector magnitude: {np.linalg.norm(vector_to_small_plane):.4f} mm")

# Verify the magnitude matches our perpendicular travel
print(f"Expected magnitude: {perpendicular_travel:.4f} mm ✓")

# Calculate positions for visualization
intersection_point = np.array([0, 0, 0])
small_plane_center = intersection_point + vector_to_small_plane

print("\n" + "-" * 70)
print("COORDINATES")
print("-" * 70)
print(f"Intersection point: {intersection_point}")
print(f"Small plane center: {small_plane_center}")

# Additional information: where would the big plane center be?
# We need to travel the remaining distance (100 - 34.20 = 65.80 mm) forward
remaining_distance = perpendicular_distance_between_planes - perpendicular_travel
dx_big = remaining_distance * np.cos(tilt_angle_rad)
dz_big = -remaining_distance * np.sin(tilt_angle_rad)  # negative, going down
vector_to_big_plane = np.array([dx_big, 0, dz_big])
big_plane_center = intersection_point + vector_to_big_plane

print(f"\nBig plane center: {big_plane_center}")
print(f"Distance small to big plane center: {np.linalg.norm(small_plane_center - big_plane_center):.4f} mm")

# Create 3D visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot points
ax.scatter(*intersection_point, color='red', s=100, label='Intersection (substrate)', marker='o')
ax.scatter(*small_plane_center, color='blue', s=100, label='Small plane center', marker='^')
ax.scatter(*big_plane_center, color='green', s=100, label='Big plane center', marker='v')

# Plot vector from intersection to small plane
ax.quiver(intersection_point[0], intersection_point[1], intersection_point[2],
          vector_to_small_plane[0], vector_to_small_plane[1], vector_to_small_plane[2],
          color='blue', arrow_length_ratio=0.1, linewidth=2, label='Vector to small plane')

# Plot center line between planes
center_line_points = np.array([big_plane_center, small_plane_center])
ax.plot(center_line_points[:, 0], center_line_points[:, 1], center_line_points[:, 2],
        'k--', linewidth=2, label='Center line', alpha=0.6)

# Draw substrate plane (XY plane at Z=0)
xx, yy = np.meshgrid(np.linspace(-40, 60, 10), np.linspace(-30, 30, 10))
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

# Rotation matrix to make plane perpendicular to center line
# Center line is at 40° from horizontal, pointing backward (-X) and up (+Z)
# To rotate the plane normal from +Z to this direction, we need 90° + 40° = 130° rotation
# Or equivalently, -(90° - 40°) = -50° rotation around Y-axis
rotation_angle = -(np.pi/2 - tilt_angle_rad)  # = -50°
rot_matrix = np.array([
    [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
    [0, 1, 0],
    [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
])

# Draw small plane (simplified as rectangle)
small_half_width = small_plane_width / 2
small_half_length = small_plane_length / 2
small_plane_corners_local = np.array([
    [-small_half_width, -small_half_length, 0],
    [small_half_width, -small_half_length, 0],
    [small_half_width, small_half_length, 0],
    [-small_half_width, small_half_length, 0],
    [-small_half_width, -small_half_length, 0]
])
small_plane_corners_rotated = small_plane_corners_local @ rot_matrix.T + small_plane_center
ax.plot(small_plane_corners_rotated[:, 0], small_plane_corners_rotated[:, 1], small_plane_corners_rotated[:, 2],
        'b-', linewidth=2.5, alpha=0.8, label='Small plane (46.13×61.51)')

# Draw big plane (simplified as rectangle)
big_half_width = big_plane_width / 2
big_half_length = big_plane_length / 2
big_plane_corners_local = np.array([
    [-big_half_width, -big_half_length, 0],
    [big_half_width, -big_half_length, 0],
    [big_half_width, big_half_length, 0],
    [-big_half_width, big_half_length, 0],
    [-big_half_width, -big_half_length, 0]
])
big_plane_corners_rotated = big_plane_corners_local @ rot_matrix.T + big_plane_center
ax.plot(big_plane_corners_rotated[:, 0], big_plane_corners_rotated[:, 1], big_plane_corners_rotated[:, 2],
        'g-', linewidth=2.5, alpha=0.8, label='Big plane (89.65×119.53)')

# Add dimension annotations
# Distance from intersection to small plane center
mid_point_to_small = intersection_point + vector_to_small_plane / 2
ax.text(mid_point_to_small[0], mid_point_to_small[1], mid_point_to_small[2] + 3,
        f'{np.linalg.norm(vector_to_small_plane):.2f} mm',
        fontsize=10, color='blue', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.8))

# Distance between plane centers (100mm total)
mid_point_centers = (small_plane_center + big_plane_center) / 2
ax.text(mid_point_centers[0] + 10, mid_point_centers[1], mid_point_centers[2],
        f'100.00 mm\n(total distance)',
        fontsize=10, color='black', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', edgecolor='black', alpha=0.8))

# Tilt angle annotation
angle_annotation_point = np.array([20, 0, -15])
ax.text(angle_annotation_point[0], angle_annotation_point[1], angle_annotation_point[2],
        f'40° tilt',
        fontsize=11, color='red', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='red', alpha=0.9))

# Vector components annotation
vector_text = f'Vector components:\nX: {vector_to_small_plane[0]:.2f} mm\nY: {vector_to_small_plane[1]:.2f} mm\nZ: {vector_to_small_plane[2]:.2f} mm'
ax.text(-35, 15, 25, vector_text,
        fontsize=9, color='darkblue', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='darkblue', alpha=0.9))

# Small plane dimensions annotation
ax.text(small_plane_center[0] - 15, small_plane_center[1], small_plane_center[2] + 8,
        '46.13×61.51 mm',
        fontsize=9, color='blue', fontweight='bold')

# Big plane dimensions annotation
ax.text(big_plane_center[0] + 15, big_plane_center[1], big_plane_center[2] - 8,
        '89.65×119.53 mm',
        fontsize=9, color='green', fontweight='bold')

# Set labels and limits
ax.set_xlabel('X (mm) - Horizontal along tilt')
ax.set_ylabel('Y (mm) - Perpendicular to tilt')
ax.set_zlabel('Z (mm) - Vertical')
ax.set_title(f'Plane System Tilted at {tilt_angle_deg}° to Substrate', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')

# Set equal aspect ratio
max_range = 60
ax.set_xlim([-40, 60])
ax.set_ylim([-30, 30])
ax.set_zlim([-50, 30])

plt.tight_layout()
plt.savefig('_testing_data/plane_intersection_3d.png', dpi=150, bbox_inches='tight')
print("\n" + "=" * 70)
print("3D visualization saved to: plane_intersection_3d.png")
print("=" * 70)

print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)
print(f"Vector from intersection point to small plane center:")
print(f"  X: {dx:.2f} mm")
print(f"  Y: {dy:.2f} mm")
print(f"  Z: {dz:.2f} mm")
print(f"\nMagnitude: {np.linalg.norm(vector_to_small_plane):.2f} mm")
print("=" * 70)