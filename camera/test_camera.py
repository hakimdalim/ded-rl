"""
Test script for the orthographic camera system
Visualizes camera behavior with simple 3D plots
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from _base_camera import BaseCamera, FollowingCameraMixin


# ============================================================================
# STEP 1: Create a simple OrthographicCamera class
# ============================================================================

class OrthographicCamera(BaseCamera):
    """
    Simple orthographic camera for testing.
    Projects points straight onto the image plane (parallel rays).
    """
    
    def _project_cam_to_plane(self, pos_cam):
        """
        Orthographic projection: just use X,Y coordinates directly.
        Ignore Z for projection (but keep it for depth sorting).
        """
        N = pos_cam.shape[0]
        u = pos_cam[:, 0]  # horizontal position (meters)
        v = pos_cam[:, 1]  # vertical position (meters)
        zc = -pos_cam[:, 2]  # depth (negative Z = forward, larger = closer)
        valid = np.ones(N, dtype=bool)  # all points valid in orthographic
        return u, v, zc, valid


class FollowingOrthographicCamera(FollowingCameraMixin, OrthographicCamera):
    """
    Orthographic camera that can follow a moving heat source.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize following behavior with defaults
        self.rel_offset_local = np.array([0.0, -0.12, 0.04])
        self.floor_angle_deg = 30.0
        ang = np.deg2rad(self.floor_angle_deg)
        self._cos_ang = float(np.cos(ang))
        self._sin_ang = float(np.sin(ang))
        self.up_hint = np.array([0.0, 0.0, 1.0])


# ============================================================================
# STEP 2: Create a simple hot object (heat source)
# ============================================================================

def create_simple_heat_source(center=(0.05, 0.05, 0.02), size=0.02):
    """
    Creates a small cube of hot voxels.
    
    Args:
        center: (x, y, z) position in meters
        size: side length of the cube in meters
    
    Returns:
        volume_activated: 3D boolean array
        temperature_field: 3D temperature array
        voxel_size: size of each voxel
    """
    # Create a 100x100x100 grid (10cm x 10cm x 10cm world)
    voxel_size = 0.001  # 1mm per voxel
    grid_size = 100
    
    volume = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    temps = np.full((grid_size, grid_size, grid_size), 300.0, dtype=float)
    
    # Calculate voxel indices for the hot cube
    cx, cy, cz = center
    half = size / 2
    
    ix_min = int((cx - half) / voxel_size)
    ix_max = int((cx + half) / voxel_size)
    iy_min = int((cy - half) / voxel_size)
    iy_max = int((cy + half) / voxel_size)
    iz_min = int((cz - half) / voxel_size)
    iz_max = int((cz + half) / voxel_size)
    
    # Clip to valid range
    ix_min, ix_max = max(0, ix_min), min(grid_size, ix_max)
    iy_min, iy_max = max(0, iy_min), min(grid_size, iy_max)
    iz_min, iz_max = max(0, iz_min), min(grid_size, iz_max)
    
    # Activate voxels and set temperature
    volume[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] = True
    temps[ix_min:ix_max, iy_min:iy_max, iz_min:iz_max] = 400.0  # Hot!
    
    return volume, temps, voxel_size


# ============================================================================
# TEST 1: Basic camera positioning and orientation
# ============================================================================

def test_1_camera_basics():
    """
    Test basic camera setup and visualization.
    Shows where camera is and what it's looking at.
    """
    print("=" * 60)
    print("TEST 1: Camera Position and Orientation")
    print("=" * 60)
    
    # Create camera
    cam = OrthographicCamera(
        pos=(0.05, -0.02, 0.04),  # 5cm forward, 2cm left, 4cm up
        target=(0.05, 0.05, 0.02),  # looking at this point
        plane_size=(0.04, 0.04),  # 4cm x 4cm view
        pixel_size_xy=(0.0005, 0.0005),  # 0.5mm pixels
        voxel_size_xyz=(0.001, 0.001, 0.001)  # 1mm voxels
    )
    
    print(f"Camera position: {cam.pos}")
    print(f"Camera target: {cam.target}")
    print(f"View plane size: {cam.plane_size[0]*100:.1f}cm x {cam.plane_size[1]*100:.1f}cm")
    print(f"Image resolution: {cam.resolution_wh[0]} x {cam.resolution_wh[1]} pixels")
    
    # Get camera basis vectors
    right, up, forward = cam.get_basis()
    print(f"\nCamera orientation:")
    print(f"  Right:   {right}")
    print(f"  Up:      {up}")
    print(f"  Forward: {forward}")
    
    # Visualize in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw camera position
    ax.scatter(*cam.pos, c='blue', s=200, marker='^', label='Camera')
    
    # Draw target
    ax.scatter(*cam.target, c='red', s=200, marker='o', label='Target')
    
    # Draw camera basis vectors
    scale = 0.02  # 2cm long arrows
    ax.quiver(*cam.pos, *right*scale, color='red', arrow_length_ratio=0.3, linewidth=2)
    ax.quiver(*cam.pos, *up*scale, color='green', arrow_length_ratio=0.3, linewidth=2)
    ax.quiver(*cam.pos, *forward*scale, color='blue', arrow_length_ratio=0.3, linewidth=2)
    
    # Draw image plane (simplified as a rectangle)
    plane_w, plane_h = cam.plane_size
    plane_center = cam.pos + forward * 0.001  # 1mm in front
    corners = [
        plane_center - right*plane_w/2 - up*plane_h/2,
        plane_center + right*plane_w/2 - up*plane_h/2,
        plane_center + right*plane_w/2 + up*plane_h/2,
        plane_center - right*plane_w/2 + up*plane_h/2,
        plane_center - right*plane_w/2 - up*plane_h/2,
    ]
    corners = np.array(corners)
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], 'b-', linewidth=2, label='Image plane')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()
    ax.set_title('Camera Setup in 3D Space')
    
    # Equal aspect ratio
    max_range = 0.08
    ax.set_xlim([0, max_range])
    ax.set_ylim([-0.04, 0.08])
    ax.set_zlim([0, max_range])
    
    plt.tight_layout()
    plt.savefig('test1_camera_setup.png', dpi=150)
    print("\n✓ Saved: test1_camera_setup.png")
    plt.show()


# ============================================================================
# TEST 2: Render a simple thermal image
# ============================================================================

def test_2_simple_render():
    """
    Test rendering a thermal image of a hot cube.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Render Thermal Image")
    print("=" * 60)
    
    # Create hot object
    heat_center = (0.05, 0.05, 0.02)
    volume, temps, vox_size = create_simple_heat_source(center=heat_center, size=0.015)
    
    print(f"Heat source center: {heat_center}")
    print(f"Active voxels: {np.sum(volume)}")
    print(f"Temperature range: {temps[volume].min():.1f}K to {temps[volume].max():.1f}K")
    
    # Create camera looking at the heat source
    cam = OrthographicCamera(
        pos=(0.05, -0.03, 0.03),  # positioned to see the cube
        target=heat_center,
        plane_size=(0.04, 0.04),
        pixel_size_xy=(0.0002, 0.0002),  # 0.2mm pixels for detail
        voxel_size_xyz=(vox_size, vox_size, vox_size)
    )
    
    print(f"\nCamera position: {cam.pos}")
    print(f"Rendering {cam.resolution_wh[0]}x{cam.resolution_wh[1]} image...")
    
    # Render the image
    img, extent = cam.render_first_hit(volume, temps, ambient=300.0)
    
    print(f"Image temperature range: {img.min():.1f}K to {img.max():.1f}K")
    print(f"Hot pixels: {np.sum(img > 300.0)} / {img.size}")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show thermal image
    im = ax1.imshow(img.T, origin='lower', extent=extent, cmap='hot', 
                    vmin=300, vmax=400)
    ax1.set_xlabel('Horizontal (meters)')
    ax1.set_ylabel('Vertical (meters)')
    ax1.set_title('Thermal Image (Camera View)')
    plt.colorbar(im, ax=ax1, label='Temperature (K)')
    ax1.grid(True, alpha=0.3)
    
    # Show 3D scene
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot heat source voxels
    idx = np.argwhere(volume)
    vox_pos = idx * vox_size
    ax2.scatter(vox_pos[:, 0], vox_pos[:, 1], vox_pos[:, 2], 
                c='red', s=1, alpha=0.3, label='Hot voxels')
    
    # Plot camera
    ax2.scatter(*cam.pos, c='blue', s=200, marker='^', label='Camera')
    ax2.scatter(*cam.target, c='orange', s=100, marker='x', label='Target')
    
    # Draw sight line
    ax2.plot([cam.pos[0], cam.target[0]], 
             [cam.pos[1], cam.target[1]], 
             [cam.pos[2], cam.target[2]], 'b--', alpha=0.5)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.legend()
    ax2.set_title('3D Scene')
    
    plt.tight_layout()
    plt.savefig('test2_thermal_render.png', dpi=150)
    print("✓ Saved: test2_thermal_render.png")
    plt.show()


# ============================================================================
# TEST 3: Following camera behavior
# ============================================================================

def test_3_following_camera():
    """
    Test camera following a moving heat source.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Following Camera")
    print("=" * 60)
    
    # Create following camera
    cam = FollowingOrthographicCamera(
        plane_size=(0.04, 0.04),
        pixel_size_xy=(0.0004, 0.0004),
        voxel_size_xyz=(0.001, 0.001, 0.001)
    )
    
    # Initialize following behavior
    cam.init_following(
        source_pos=(0.03, 0.03, 0.02),
        rel_offset_local=(0.0, -0.04, 0.02),  # 4cm behind, 2cm above
        floor_angle_deg=25.0
    )
    
    print(f"Following offset: {cam.rel_offset_local}")
    print(f"Floor angle: {cam.floor_angle_deg}°")
    
    # Simulate heat source moving along a path
    n_steps = 8
    path = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        x = 0.03 + 0.03 * t  # move from 3cm to 6cm in X
        y = 0.03 + 0.02 * np.sin(2 * np.pi * t)  # oscillate in Y
        z = 0.02  # constant height
        path.append([x, y, z])
    
    path = np.array(path)
    
    # Track camera positions
    cam_positions = []
    for pos in path:
        cam.follow_heat_source(pos, orient=True)
        cam_positions.append(cam.pos.copy())
    
    cam_positions = np.array(cam_positions)
    
    # Visualize
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw heat source path
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 
            'r-o', linewidth=2, markersize=8, label='Heat source path')
    
    # Draw camera path
    ax.plot(cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
            'b-^', linewidth=2, markersize=8, label='Camera path')
    
    # Draw connections showing camera following
    for i in range(0, n_steps, 2):  # every other point to avoid clutter
        ax.plot([path[i, 0], cam_positions[i, 0]],
                [path[i, 1], cam_positions[i, 1]],
                [path[i, 2], cam_positions[i, 2]],
                'g--', alpha=0.5, linewidth=1)
    
    # Mark start and end
    ax.scatter(*path[0], c='red', s=300, marker='o', 
               edgecolors='black', linewidths=2, label='Start')
    ax.scatter(*path[-1], c='darkred', s=300, marker='s',
               edgecolors='black', linewidths=2, label='End')
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.legend()
    ax.set_title('Following Camera Tracking Moving Heat Source')
    
    # Equal aspect
    ax.set_box_aspect([1,1,0.5])
    
    plt.tight_layout()
    plt.savefig('test3_following_camera.png', dpi=150)
    print("✓ Saved: test3_following_camera.png")
    plt.show()
    
    print(f"\nCamera maintained ~{np.linalg.norm(cam.rel_offset_local):.3f}m offset from source")


# ============================================================================
# TEST 4: Compare multiple camera angles
# ============================================================================

def test_4_multiple_angles():
    """
    Render the same hot object from multiple camera angles.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Multiple Camera Angles")
    print("=" * 60)
    
    # Create hot object
    heat_center = (0.05, 0.05, 0.02)
    volume, temps, vox_size = create_simple_heat_source(center=heat_center, size=0.02)
    
    # Define camera positions (different angles around the object)
    angles = [
        ("Front", (0.05, -0.03, 0.02)),
        ("Side", (0.08, 0.05, 0.02)),
        ("Top-Front", (0.05, 0.00, 0.06)),
        ("Angled", (0.07, 0.00, 0.04))
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (label, cam_pos) in enumerate(angles):
        cam = OrthographicCamera(
            pos=cam_pos,
            target=heat_center,
            plane_size=(0.05, 0.05),
            pixel_size_xy=(0.0003, 0.0003),
            voxel_size_xyz=(vox_size, vox_size, vox_size)
        )
        
        img, extent = cam.render_first_hit(volume, temps, ambient=300.0)
        
        ax = axes[idx]
        im = ax.imshow(img.T, origin='lower', extent=extent, cmap='hot',
                       vmin=300, vmax=400)
        ax.set_xlabel('Horizontal (m)')
        ax.set_ylabel('Vertical (m)')
        ax.set_title(f'{label} View\nCam: ({cam_pos[0]:.2f}, {cam_pos[1]:.2f}, {cam_pos[2]:.2f})')
        plt.colorbar(im, ax=ax, label='Temp (K)')
        ax.grid(True, alpha=0.3)
        
        print(f"{label}: {np.sum(img > 300)} hot pixels")
    
    plt.tight_layout()
    plt.savefig('test4_multiple_angles.png', dpi=150)
    print("✓ Saved: test4_multiple_angles.png")
    plt.show()


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ORTHOGRAPHIC CAMERA TEST SUITE")
    print("="*60)
    
    try:
        test_1_camera_basics()
        test_2_simple_render()
        test_3_following_camera()
        test_4_multiple_angles()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        print("  - test1_camera_setup.png      (3D camera position)")
        print("  - test2_thermal_render.png    (thermal image + scene)")
        print("  - test3_following_camera.png  (camera tracking motion)")
        print("  - test4_multiple_angles.png   (4 different views)")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()