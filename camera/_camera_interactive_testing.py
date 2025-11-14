# _camera_interactive_testing.py
"""
Interactive Testing Application for Camera Classes

This Dash application provides an interactive web interface for testing and visualizing
the behavior of camera classes defined in this package:
- OrthographicCamera (parallel projection)
- PerspectiveCamera (converging rays with FOV)
- FollowingOrthographicCamera (tracks a moving point)
- FollowingPerspectiveCamera (tracks a moving point with perspective)
- CameraCallback (production callback for simulations)

**IMPORTANT - THIS IS A TESTING TOOL, NOT PRODUCTION CODE:**

When bugs or issues are discovered through this interactive interface, fixes must be
applied to the SOURCE camera classes in the following files:
    - camera/_base_camera.py (base functionality)
    - camera/orthographic_camera.py (orthographic projection)
    - camera/perspective_camera.py (perspective projection)
    - callbacks/camera_callback.py (simulation callback)

This testing file should ONLY be modified if the testing interface itself needs changes.

Usage:
    python camera/_camera_interactive_testing.py

Then open http://localhost:8050 in your browser to interact with the camera parameters
and visualize the rendered thermal images in real-time.

Features:
- Real-time camera parameter adjustment
- Side-by-side projection comparisons
- Following camera behavior testing
- Callback configuration testing
- Digital zoom/crop window testing
- Direct callback testing with real CameraCallback instances
"""
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from types import SimpleNamespace
from pathlib import Path

from callbacks.camera_callback import CameraCallback
from camera.orthographic_camera import OrthographicCamera, FollowingOrthographicCamera
from camera.perspective_camera import PerspectiveCamera, FollowingPerspectiveCamera

from callbacks.hdf5_activation_saver import load_activation_volume, load_activation_metadata, list_steps_in_file
from callbacks.hdf5_thermal_saver import load_thermal_field, load_thermal_metadata

# FOV limits for perspective cameras (0, 180) exclusive
FOV_MIN = 1.0   # Minimum FOV in degrees (very narrow telephoto)
FOV_MAX = 179.0 # Maximum FOV in degrees (ultra wide angle)

# ============================================================================
# Load HDF5 Data
# ============================================================================
ACTIVATION_FILE = Path("_testing_data/activation_volumes.h5")
THERMAL_FILE = Path("_testing_data/thermal_fields.h5")

print(f"Loading HDF5 files from:")
print(f"  Activation: {ACTIVATION_FILE}")
print(f"  Thermal:    {THERMAL_FILE}")

# Load available steps
AVAILABLE_STEPS = list_steps_in_file(str(ACTIVATION_FILE))
print(f"  Found {len(AVAILABLE_STEPS)} timesteps: {AVAILABLE_STEPS}")

# Load first step to get metadata
first_metadata = load_activation_metadata(str(ACTIVATION_FILE), step=AVAILABLE_STEPS[0])
VOXEL_SIZE = np.array([
    first_metadata['voxel_size_x'],
    first_metadata['voxel_size_y'],
    first_metadata['voxel_size_z']
])

# Pre-load all data into RAM for fast access
print("Loading all timesteps into RAM...")
HDF5_DATA = {}
for step in AVAILABLE_STEPS:
    activation = load_activation_volume(str(ACTIVATION_FILE), step=step)
    temperature = load_thermal_field(str(THERMAL_FILE), step=step)
    metadata_act = load_activation_metadata(str(ACTIVATION_FILE), step=step)
    metadata_therm = load_thermal_metadata(str(THERMAL_FILE), step=step)

    HDF5_DATA[step] = {
        'activation': activation,
        'temperature': temperature,
        'metadata': {**metadata_act, **metadata_therm},  # Merge both metadata dicts
    }

print(f"Loaded {len(HDF5_DATA)} timesteps into RAM")

# Get volume dimensions from first step
first_activation = HDF5_DATA[AVAILABLE_STEPS[0]]['activation']
VOLUME_SHAPE = first_activation.shape
VOLUME_SIZE = np.array(VOLUME_SHAPE) * VOXEL_SIZE

print(f"  Volume shape: {VOLUME_SHAPE}")
print(f"  Voxel size: {VOXEL_SIZE * 1e6} μm")
print(f"  Volume size: {VOLUME_SIZE * 1e3} mm")

# Helper to get data for a specific step
def get_step_data(step):
    """Get activation, temperature, and metadata for a specific step."""
    return HDF5_DATA[step]['activation'], HDF5_DATA[step]['temperature'], HDF5_DATA[step]['metadata']

# Helper to get target position from metadata (for following cameras)
def get_target_position(metadata):
    """Extract target position from metadata."""
    return np.array([
        metadata.get('position_x', VOLUME_SIZE[0] / 2.0),
        metadata.get('position_y', VOLUME_SIZE[1] / 2.0),
        metadata.get('position_z', VOLUME_SIZE[2] / 2.0),
    ])


# Scene helpers
def scene_params():
    """Calculate scene parameters from loaded HDF5 data."""
    vx, vy, vz = VOXEL_SIZE
    nx, ny, nz = VOLUME_SHAPE
    vol_size = VOLUME_SIZE
    center = vol_size / 2.0
    xy_max = float(max(vol_size[0], vol_size[1]))
    return vx, vy, vz, (nx, ny, nz), vol_size, center, xy_max


def plane_and_res(plane_width, plane_height, voxel_size, res_scale=1.0):
    # Use provided plane size (width and height can be different)
    # Base defaults to square using max(vol_size), but user can adjust independently
    PLANE_SIZE = (plane_width, plane_height)
    px, py, _ = voxel_size
    base_res_w = max(1, int(np.ceil(plane_width / px)) - 10)
    base_res_h = max(1, int(np.ceil(plane_height / py)) - 10)
    W = max(1, int(round(base_res_w * res_scale)))
    H = max(1, int(round(base_res_h * res_scale)))
    return PLANE_SIZE, (W, H)


def render_image(cam, activated, temperature_field, temp_max=None):
    img, extent = cam.render_first_hit(activated, temperature_field, ambient=300.0)
    (xmin, xmax, ymin, ymax) = extent

    # Build heatmap kwargs with optional zmax
    heatmap_kwargs = {
        'z': img.T,
        'x': np.linspace(xmin, xmax, img.shape[0]),
        'y': np.linspace(ymin, ymax, img.shape[1]),
        'colorscale': "Inferno",
        'zsmooth': False
    }
    if temp_max is not None:
        heatmap_kwargs['zmax'] = temp_max
        heatmap_kwargs['zmin'] = 300.0  # Ambient temperature

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=650)
    return fig


def render_image_from_array(img, extent, title: str, temp_max=None):
    xmin, xmax, ymin, ymax = extent

    # Build heatmap kwargs with optional zmax
    heatmap_kwargs = {
        'z': img.T,
        'x': np.linspace(xmin, xmax, img.shape[0]),
        'y': np.linspace(ymin, ymax, img.shape[1]),
        'colorscale': "Inferno",
        'zsmooth': False
    }
    if temp_max is not None:
        heatmap_kwargs['zmax'] = temp_max
        heatmap_kwargs['zmin'] = 300.0  # Ambient temperature

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=650,
        title=dict(text=title, x=0.5, xanchor="center")
    )
    return fig


# Camera creation functions
def make_camera_ortho(plane_size, res, radius, center, voxel_size, az, el, roll):
    cam = OrthographicCamera(plane_size=plane_size, voxel_size_xyz=voxel_size)
    cam.set_resolution(*res)
    cam.set_from_spherical(az_deg=az, el_deg=el, roll_deg=roll, radius=radius, target=center)
    return cam


def make_camera_persp(plane_size, res, radius, center, voxel_size, az, el, roll, fov_y_deg):
    cam = PerspectiveCamera(fov_y_deg=fov_y_deg, plane_size=plane_size, voxel_size_xyz=voxel_size)
    cam.set_resolution(*res)
    cam.set_from_spherical(az_deg=az, el_deg=el, roll_deg=roll, radius=radius, target=center)
    return cam


def make_camera_follow_ortho(plane_size, res, center, voxel_size, follow_back, follow_up, floor_angle):
    cam = FollowingOrthographicCamera(
        source_pos=center,
        offset=(0.0, -follow_back, follow_up),
        plane_size=plane_size,
        voxel_size_xyz=voxel_size
    )
    cam.set_resolution(*res)
    return cam


def make_camera_follow_persp(plane_size, res, center, voxel_size, follow_back, follow_up, floor_angle, fov_y_deg):
    cam = FollowingPerspectiveCamera(
        source_pos=center,
        offset=(0.0, -follow_back, follow_up),
        fov_y_deg=fov_y_deg,
        plane_size=plane_size,
        voxel_size_xyz=voxel_size
    )
    cam.set_resolution(*res)
    return cam


def make_camera_callback_config(plane_size, res, center, voxel_size, rel_x, rel_y, rel_z, floor_angle, fov_y_deg):
    cam = FollowingPerspectiveCamera(
        source_pos=center,
        offset=(rel_x, rel_y, rel_z),
        fov_y_deg=fov_y_deg,
        plane_size=plane_size,
        voxel_size_xyz=voxel_size
    )
    cam.set_resolution(*res)
    return cam


# Helper functions for UI construction
def make_slider(id_str, label, description, min_val, max_val, step, default):
    """Create a labeled slider with description."""
    return html.Div([
        html.Label(label, style={"fontWeight": "500", "marginBottom": "4px"}),
        html.P(description, style={"fontSize": "0.85em", "color": "#666", "marginBottom": "8px"}),
        dcc.Slider(id=id_str, min=min_val, max=max_val, step=step, value=default,
                   marks=None, tooltip={"placement": "bottom", "always_visible": False}),
    ], style={"marginBottom": "20px"})

# Initialize
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
vx, vy, vz, (nx, ny, nz), VOL_SIZE, CENTER, XY_MAX = scene_params()

# Calculate default plane size based on max volume dimension
MAX_VOL_DIM = max(VOL_SIZE)  # For 10x80x10mm, this is 80mm
DEFAULT_PLANE_SIZE = MAX_VOL_DIM * 1.1  # 10% larger than max dimension

# Common controls
# Player controls for timestep
player_controls = html.Div([
    html.Label(f"Timestep (1-{len(AVAILABLE_STEPS)})", style={"fontWeight": "500", "marginBottom": "4px"}),
    html.P("Select simulation timestep to visualize", style={"fontSize": "0.85em", "color": "#666", "marginBottom": "8px"}),

    # Button row
    dbc.Row([
        dbc.Col([
            dbc.Button("⏮", id="step-first", color="secondary", size="sm", style={"width": "100%"}),
        ], width=2),
        dbc.Col([
            dbc.Button("◀", id="step-prev", color="secondary", size="sm", style={"width": "100%"}),
        ], width=2),
        dbc.Col([
            dbc.Button("▶", id="play-pause", color="primary", size="sm", style={"width": "100%"}),
        ], width=2),
        dbc.Col([
            dbc.Button("▶", id="step-next", color="secondary", size="sm", style={"width": "100%"}),
        ], width=2),
        dbc.Col([
            dbc.Button("⏭", id="step-last", color="secondary", size="sm", style={"width": "100%"}),
        ], width=2),
        dbc.Col([
            dcc.Input(
                id="play-speed",
                type="number",
                value=500,
                min=50,
                max=5000,
                step=50,
                placeholder="ms",
                style={"width": "100%", "padding": "4px", "fontSize": "0.85em"}
            ),
        ], width=2),
    ], className="mb-2"),

    # Slider
    dcc.Slider(
        id="timestep",
        min=1,
        max=len(AVAILABLE_STEPS),
        step=1,
        value=1,
        marks=None,
        tooltip={"placement": "bottom", "always_visible": False}
    ),

    # Hidden components for state management
    dcc.Store(id="is-playing", data=False),
    dcc.Interval(id="play-interval", interval=500, disabled=True),
], style={"marginBottom": "20px"})

res_scale_slider = make_slider("res-scale", "Resolution Scale (×pixels)",
    "Multiply base pixel resolution (higher = more detail, slower render)", 0.25, 4.0, 0.25, 1.0)

# Temperature max input (None = auto-scale)
temp_max_input = html.Div([
    html.Label("Temperature Max (K)", style={"fontWeight": "500", "marginBottom": "4px"}),
    html.P("Maximum temperature for colormap (leave empty for auto-scale)",
           style={"fontSize": "0.85em", "color": "#666", "marginBottom": "8px"}),
    dcc.Input(
        id="temp-max",
        type="number",
        placeholder="Auto",
        value=None,
        style={"width": "100%", "padding": "8px"}
    ),
], style={"marginBottom": "20px"})

# Tab contents
tab_ortho = dbc.Card(dbc.CardBody([
    html.H5("Static Orthographic Camera", style={"marginBottom": "20px"}),
    make_slider("ortho-plane-width", "Plane Width (m)", "Sensor width", 0.001, 0.3, 0.0001, DEFAULT_PLANE_SIZE),
    make_slider("ortho-plane-height", "Plane Height (m)", "Sensor height", 0.001, 0.3, 0.0001, DEFAULT_PLANE_SIZE),
    make_slider("ortho-az", "Azimuth (°)", "Horizontal rotation around target (0°=+X axis, 90°=+Y axis)", 0, 360, 1, 45),
    make_slider("ortho-el", "Elevation (°)", "Vertical angle from horizon (0°=horizontal, +90°=zenith, -90°=nadir)", -80, 80, 1, 20),
    make_slider("ortho-roll", "Roll (°)", "Camera rotation around viewing axis (tilts the view)", 0, 180, 1, 0),
    make_slider("ortho-radius", "Radius (m)", "Distance from camera to target point (doesn't affect ortho projection)", 0.5 * XY_MAX, 4.0 * XY_MAX, 0.001, 2.0 * XY_MAX),
]))

tab_persp = dbc.Card(dbc.CardBody([
    html.H5("Static Perspective Camera", style={"marginBottom": "20px"}),
    make_slider("persp-plane-width", "Plane Width (m)", "Sensor width", 0.001, 0.3, 0.0001, DEFAULT_PLANE_SIZE),
    make_slider("persp-plane-height", "Plane Height (m)", "Sensor height", 0.001, 0.3, 0.0001, DEFAULT_PLANE_SIZE),
    make_slider("persp-az", "Azimuth (°)", "Horizontal rotation around target (0°=+X axis, 90°=+Y axis)", 0, 360, 1, 45),
    make_slider("persp-el", "Elevation (°)", "Vertical angle from horizon (0°=horizontal, +90°=zenith, -90°=nadir)", -80, 80, 1, 20),
    make_slider("persp-roll", "Roll (°)", "Camera rotation around viewing axis (tilts the view)", 0, 180, 1, 0),
    make_slider("persp-radius", "Radius (m)", "Distance from camera to target (affects perspective distortion)", 0.5 * XY_MAX, 4.0 * XY_MAX, 0.001, 2.0 * XY_MAX),
    make_slider("persp-fov", "FOV Y (°)", "Vertical field of view (smaller=telephoto, larger=wide angle)", FOV_MIN, FOV_MAX, 1, 40),
]))

tab_follow_ortho = dbc.Card(dbc.CardBody([
    html.H5("Following Orthographic Camera", style={"marginBottom": "20px"}),
    make_slider("follow-ortho-plane-width", "Plane Width (m)", "Sensor width", 0.001, 0.3, 0.0001, DEFAULT_PLANE_SIZE),
    make_slider("follow-ortho-plane-height", "Plane Height (m)", "Sensor height", 0.001, 0.3, 0.0001, DEFAULT_PLANE_SIZE),
    make_slider("follow-ortho-back", "Follow Back (m)", "Distance behind heat source (negative Y in local frame)", 0.2 * XY_MAX, 3.0 * XY_MAX, 0.0005, 0.003),
    make_slider("follow-ortho-up", "Follow Up (m)", "Height above heat source before floor angle rotation (Z in local frame)", 0.0, 1.5 * XY_MAX, 0.0005, 0.001),
    make_slider("follow-ortho-floor", "Tilt Angle (°)", "Rotation about local X-axis (0°=no tilt, +angles rotate offset downward)", 0, 60, 1, 0),
]))

tab_follow_persp = dbc.Card(dbc.CardBody([
    html.H5("Following Perspective Camera", style={"marginBottom": "20px"}),
    make_slider("follow-persp-plane-width", "Plane Width (m)", "Sensor width", 0.001, 0.3, 0.0001, DEFAULT_PLANE_SIZE),
    make_slider("follow-persp-plane-height", "Plane Height (m)", "Sensor height", 0.001, 0.3, 0.0001, DEFAULT_PLANE_SIZE),
    make_slider("follow-persp-back", "Follow Back (m)", "Distance behind heat source (negative Y in local frame)", 0.2 * XY_MAX, 3.0 * XY_MAX, 0.0005, 0.003),
    make_slider("follow-persp-up", "Follow Up (m)", "Height above heat source before floor angle rotation (Z in local frame)", 0.0, 1.5 * XY_MAX, 0.0005, 0.001),
    make_slider("follow-persp-floor", "Tilt Angle (°)", "Rotation about local X-axis (0°=no tilt, +angles rotate offset downward)", 0, 60, 1, 0),
    make_slider("follow-persp-fov", "FOV Y (°)", "Vertical field of view (smaller=telephoto, larger=wide angle)", FOV_MIN, FOV_MAX, 1, 40),
]))

tab_cropping_persp = dbc.Card(dbc.CardBody([
    html.H5("Cropping Persp", style={"marginBottom": "8px"}),
    html.P("Test perspective camera crop with manual configuration", style={"fontSize": "0.9em", "color": "#666", "marginBottom": "20px"}),
    make_slider("cb-rel-x", "offset X (m)", "Lateral offset (perpendicular to scan, +right when looking forward)", -0.2, 0.2, 0.001, -0.12),
    make_slider("cb-rel-y", "offset Y (m)", "Longitudinal offset (along scan direction, -behind/+front of nozzle)", -0.3, 0.1, 0.001, 0.0),
    make_slider("cb-rel-z", "offset Z (m)", "Vertical offset (before floor angle rotation is applied)", 0.0, 0.2, 0.001, 0.04),
    make_slider("cb-azimuth", "Azimuth Rotation (°)", "Rotate offset around tracked point (0°=behind, 90°=right, 180°=front, 270°=left)", 0, 360, 1, 90),
    make_slider("cb-floor-angle", "tilt_angle_deg (°)", "Rotation about local X-axis (0°=no tilt, +angles rotate offset downward)", 0, 90, 1, 0),
    make_slider("cb-fov", "fov_y_deg (°)", "Vertical field of view (smaller=telephoto, larger=wide angle)", FOV_MIN, FOV_MAX, 1, 45),
    html.Hr(style={"marginTop": "30px", "marginBottom": "20px"}),
    make_slider("cb-crop-width", "Crop Window Width (m)", "Width of digital crop window at target (smaller = more zoomed in)", 0.0001, 0.01, 0.00001, 0.002),
    make_slider("cb-crop-height", "Crop Window Height (m)", "Height of digital crop window at target (smaller = more zoomed in)", 0.0001, 0.01, 0.00001, 0.002),
    make_slider("cb-res", "Resolution (pixels)", "Sensor resolution (square pixels)", 128, 2048, 64, 512),
]))

# Cropping comparison tab - test both ortho and persp with same crop parameters
tab_cropping_compare = dbc.Card(dbc.CardBody([
    html.H5("Cropping Compare (Ortho vs Perspective)", style={"marginBottom": "8px"}),
    html.P("Compare orthographic and perspective cropping with same physical window size.",
           style={"fontSize": "0.9em", "color": "#666", "marginBottom": "20px"}),
    make_slider("cbc-rel-x", "offset X (m)", "Lateral offset (perpendicular to scan, +right when looking forward)", -0.2, 0.2, 0.001, -0.12),
    make_slider("cbc-rel-y", "offset Y (m)", "Longitudinal offset (along scan direction, -behind/+front of nozzle)", -0.3, 0.1, 0.001, 0.0),
    make_slider("cbc-rel-z", "offset Z (m)", "Vertical offset (before tilt angle rotation is applied)", 0.0, 0.2, 0.001, 0.04),
    make_slider("cbc-azimuth", "Azimuth Rotation (°)", "Rotate offset around tracked point (0°=behind, 90°=right, 180°=front, 270°=left)", 0, 360, 1, 90),
    make_slider("cbc-floor-angle", "tilt_angle_deg (°)", "Rotation about local X-axis (0°=no tilt, +angles rotate offset downward)", 0, 90, 1, 0),
    make_slider("cbc-fov", "fov_y_deg (°) - Perspective Only", "Vertical field of view for perspective camera (smaller=telephoto, larger=wide angle)", FOV_MIN, FOV_MAX, 1, 45),
    html.Hr(style={"marginTop": "30px", "marginBottom": "20px"}),
    make_slider("cbc-crop-width", "Crop Window Width (m)", "Width of digital crop window at target (both cameras will crop to this size)", 0.0001, 0.01, 0.00001, 0.006),
    make_slider("cbc-crop-height", "Crop Window Height (m)", "Height of digital crop window at target (both cameras will crop to this size)", 0.0001, 0.01, 0.00001, 0.006),
    make_slider("cbc-res", "Resolution (pixels)", "Sensor resolution (square pixels)", 128, 2048, 64, 512),
]))

# Side-by-side comparison tab for following cameras
tab_follow_compare = dbc.Card(dbc.CardBody([
    html.H5("Following Camera Comparison (Ortho vs Perspective)", style={"marginBottom": "8px"}),
    html.P("Compare orthographic and perspective cameras with the same parameters side-by-side",
           style={"fontSize": "0.9em", "color": "#666", "marginBottom": "20px"}),
    make_slider("cmp-plane-width", "Plane Width (m)", "Sensor width", 0.001, 0.3, 0.0001, DEFAULT_PLANE_SIZE),
    make_slider("cmp-plane-height", "Plane Height (m)", "Sensor height", 0.001, 0.3, 0.0001, DEFAULT_PLANE_SIZE),
    make_slider("cmp-follow-back", "Follow Back (m)", "Distance behind heat source (negative Y in local frame)", 0.2 * XY_MAX, 3.0 * XY_MAX, 0.0005, 0.003),
    make_slider("cmp-follow-up", "Follow Up (m)", "Height above heat source before tilt angle rotation (Z in local frame)", 0.0, 1.5 * XY_MAX, 0.0005, 0.001),
    make_slider("cmp-tilt-angle", "Tilt Angle (°)", "Rotation about local X-axis (0°=no tilt, +angles rotate offset downward)", 0, 60, 1, 0),
    make_slider("cmp-fov", "FOV Y (°) - Perspective Only", "Vertical field of view for perspective camera (smaller=telephoto, larger=wide angle)", FOV_MIN, FOV_MAX, 1, 40),
]))

# Real CameraCallback instance tab - uses actual callback as in production
tab_callback_real = dbc.Card(dbc.CardBody([
    html.H5("Callback Compare (Real CameraCallback Instances)", style={"marginBottom": "8px"}),
    html.P("Test with actual CameraCallback instances - production configuration",
           style={"fontSize": "0.9em", "color": "#666", "marginBottom": "20px"}),
    make_slider("cbr-rel-x", "offset X (m)", "Lateral offset (perpendicular to scan, +right when looking forward)", -0.2, 0.2, 0.001, -0.12),
    make_slider("cbr-rel-y", "offset Y (m)", "Longitudinal offset (along scan direction, -behind/+front of nozzle)", -0.3, 0.1, 0.001, 0.0),
    make_slider("cbr-rel-z", "offset Z (m)", "Vertical offset (before floor angle rotation is applied)", 0.0, 0.2, 0.001, 0.04),
    make_slider("cbr-fov", "fov_y_deg (°) - Perspective Only", "Vertical field of view for perspective camera", FOV_MIN, FOV_MAX, 1, 45),
    html.Hr(style={"marginTop": "30px", "marginBottom": "20px"}),
    make_slider("cbr-crop-width", "Crop Window Width (m)", "Width of digital crop window at target (both callbacks)", 0.0001, 0.01, 0.00001, 0.006),
    make_slider("cbr-crop-height", "Crop Window Height (m)", "Height of digital crop window at target (both callbacks)", 0.0001, 0.01, 0.00001, 0.006),
    make_slider("cbr-res", "Resolution (pixels)", "Sensor resolution (square pixels)", 128, 2048, 64, 512),
]))

# New tab: Following camera through HDF5 timesteps
tab_following_timestep = dbc.Card(dbc.CardBody([
    html.H5("Following Camera Through Timesteps", style={"marginBottom": "8px"}),
    html.P("Test following cameras tracking through simulation timesteps from HDF5 data",
           style={"fontSize": "0.9em", "color": "#666", "marginBottom": "20px"}),
    make_slider("fts-plane-width", "Plane Width (m)", "Sensor width", 0.001, 0.3, 0.0001, DEFAULT_PLANE_SIZE),
    make_slider("fts-plane-height", "Plane Height (m)", "Sensor height", 0.001, 0.3, 0.0001, DEFAULT_PLANE_SIZE),
    make_slider("fts-follow-back", "Follow Back (m)", "Distance behind target (negative Y in local frame)", 0.2 * XY_MAX, 3.0 * XY_MAX, 0.0005, 0.003),
    make_slider("fts-follow-up", "Follow Up (m)", "Height above target before tilt angle rotation (Z in local frame)", 0.0, 1.5 * XY_MAX, 0.0005, 0.001),
    make_slider("fts-tilt-angle", "Tilt Angle (°)", "Rotation about local X-axis (0°=no tilt, +angles rotate offset downward)", 0, 60, 1, 0),
    make_slider("fts-fov", "FOV Y (°) - Perspective Only", "Vertical field of view for perspective camera", FOV_MIN, FOV_MAX, 1, 40),
]))

app.layout = dbc.Container([
    html.H3("Interactive Camera Viewer (HDF5 Data)", className="mt-3 mb-3"),
    dbc.Row([
        dbc.Col([
            html.Div(player_controls, className="mb-4"),
            html.Div(res_scale_slider, className="mb-4"),
            html.Div(temp_max_input, className="mb-4"),
            html.Hr(),
            dbc.Tabs([
                dbc.Tab(tab_ortho, label="Orthographic", tab_id="tab-ortho"),
                dbc.Tab(tab_persp, label="Perspective", tab_id="tab-persp"),
                dbc.Tab(tab_follow_ortho, label="Following Ortho", tab_id="tab-follow-ortho"),
                dbc.Tab(tab_follow_persp, label="Following Persp", tab_id="tab-follow-persp"),
                dbc.Tab(tab_cropping_persp, label="Cropping Persp", tab_id="tab-cropping-persp"),
                dbc.Tab(tab_follow_compare, label="Following Compare", tab_id="tab-follow-compare"),
                dbc.Tab(tab_cropping_compare, label="Cropping Compare", tab_id="tab-cropping-compare"),
                dbc.Tab(tab_callback_real, label="Callback Compare", tab_id="tab-callback-real"),
                dbc.Tab(tab_following_timestep, label="Following Timestep", tab_id="tab-following-timestep"),
            ], id="camera-tabs", active_tab="tab-ortho"),
        ], width=8),
        dbc.Col([
            html.Div(id="image-container"),
            html.Div(id="meta", style={"fontFamily": "monospace", "marginTop": "6px", "fontSize": "0.85em"})
        ], width=4),
    ]),
], fluid=True)


# Player control callbacks
@app.callback(
    Output("timestep", "value"),
    Input("step-first", "n_clicks"),
    Input("step-prev", "n_clicks"),
    Input("step-next", "n_clicks"),
    Input("step-last", "n_clicks"),
    Input("play-interval", "n_intervals"),
    State("timestep", "value"),
    State("is-playing", "data"),
    prevent_initial_call=True
)
def update_timestep(btn_first, btn_prev, btn_next, btn_last, n_intervals, current_step, is_playing):
    """Handle timestep navigation from buttons and auto-play."""
    from dash import callback_context

    if not callback_context.triggered:
        return current_step

    trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    max_step = len(AVAILABLE_STEPS)

    if trigger_id == "step-first":
        return 1
    elif trigger_id == "step-prev":
        return max(1, current_step - 1)
    elif trigger_id == "step-next":
        return min(max_step, current_step + 1)
    elif trigger_id == "step-last":
        return max_step
    elif trigger_id == "play-interval" and is_playing:
        # Auto-advance, loop back to 1 if at end
        if current_step >= max_step:
            return 1
        return current_step + 1

    return current_step


@app.callback(
    Output("is-playing", "data"),
    Output("play-pause", "children"),
    Output("play-pause", "color"),
    Input("play-pause", "n_clicks"),
    State("is-playing", "data"),
    prevent_initial_call=True
)
def toggle_play_pause(n_clicks, is_playing):
    """Toggle play/pause state."""
    new_state = not is_playing
    if new_state:
        return True, "⏸", "success"  # Playing - show pause button
    else:
        return False, "▶", "primary"  # Paused - show play button


@app.callback(
    Output("play-interval", "disabled"),
    Output("play-interval", "interval"),
    Input("is-playing", "data"),
    Input("play-speed", "value"),
)
def update_interval(is_playing, speed):
    """Enable/disable interval based on play state and update speed."""
    interval_ms = speed if speed is not None and speed > 0 else 500
    return not is_playing, interval_ms


@app.callback(
    Output("image-container", "children"),
    Output("meta", "children"),
    Input("camera-tabs", "active_tab"),
    Input("timestep", "value"),
    Input("res-scale", "value"),
    Input("temp-max", "value"),
    Input("ortho-plane-width", "value"),
    Input("ortho-plane-height", "value"),
    Input("persp-plane-width", "value"),
    Input("persp-plane-height", "value"),
    Input("follow-ortho-plane-width", "value"),
    Input("follow-ortho-plane-height", "value"),
    Input("follow-persp-plane-width", "value"),
    Input("follow-persp-plane-height", "value"),
    Input("cmp-plane-width", "value"),
    Input("cmp-plane-height", "value"),
    Input("fts-plane-width", "value"),
    Input("fts-plane-height", "value"),
    Input("ortho-az", "value"),
    Input("ortho-el", "value"),
    Input("ortho-roll", "value"),
    Input("ortho-radius", "value"),
    Input("persp-az", "value"),
    Input("persp-el", "value"),
    Input("persp-roll", "value"),
    Input("persp-radius", "value"),
    Input("persp-fov", "value"),
    Input("follow-ortho-back", "value"),
    Input("follow-ortho-up", "value"),
    Input("follow-ortho-floor", "value"),
    Input("follow-persp-back", "value"),
    Input("follow-persp-up", "value"),
    Input("follow-persp-floor", "value"),
    Input("follow-persp-fov", "value"),
    Input("cb-rel-x", "value"),
    Input("cb-rel-y", "value"),
    Input("cb-rel-z", "value"),
    Input("cb-azimuth", "value"),
    Input("cb-floor-angle", "value"),
    Input("cb-fov", "value"),
    Input("cb-crop-width", "value"),
    Input("cb-crop-height", "value"),
    Input("cb-res", "value"),
    Input("cmp-follow-back", "value"),
    Input("cmp-follow-up", "value"),
    Input("cmp-tilt-angle", "value"),
    Input("cmp-fov", "value"),
    Input("cbc-rel-x", "value"),
    Input("cbc-rel-y", "value"),
    Input("cbc-rel-z", "value"),
    Input("cbc-azimuth", "value"),
    Input("cbc-floor-angle", "value"),
    Input("cbc-fov", "value"),
    Input("cbc-crop-width", "value"),
    Input("cbc-crop-height", "value"),
    Input("cbc-res", "value"),
    Input("cbr-rel-x", "value"),
    Input("cbr-rel-y", "value"),
    Input("cbr-rel-z", "value"),
    Input("cbr-fov", "value"),
    Input("cbr-crop-width", "value"),
    Input("cbr-crop-height", "value"),
    Input("cbr-res", "value"),
    Input("fts-follow-back", "value"),
    Input("fts-follow-up", "value"),
    Input("fts-tilt-angle", "value"),
    Input("fts-fov", "value"),
)
def update(active_tab, timestep, res_scale, temp_max,
           ortho_plane_width, ortho_plane_height,
           persp_plane_width, persp_plane_height,
           follow_ortho_plane_width, follow_ortho_plane_height,
           follow_persp_plane_width, follow_persp_plane_height,
           cmp_plane_width, cmp_plane_height,
           fts_plane_width, fts_plane_height,
           ortho_az, ortho_el, ortho_roll, ortho_radius,
           persp_az, persp_el, persp_roll, persp_radius, persp_fov,
           follow_ortho_back, follow_ortho_up, follow_ortho_floor,
           follow_persp_back, follow_persp_up, follow_persp_floor, follow_persp_fov,
           cb_rel_x, cb_rel_y, cb_rel_z, cb_azimuth, cb_floor_angle, cb_fov,
           cb_crop_width, cb_crop_height, cb_res,
           cmp_follow_back, cmp_follow_up, cmp_tilt_angle, cmp_fov,
           cbc_rel_x, cbc_rel_y, cbc_rel_z, cbc_azimuth, cbc_floor_angle, cbc_fov,
           cbc_crop_width, cbc_crop_height, cbc_res,
           cbr_rel_x, cbr_rel_y, cbr_rel_z, cbr_fov,
           cbr_crop_width, cbr_crop_height, cbr_res,
           fts_follow_back, fts_follow_up, fts_tilt_angle, fts_fov):

    # Get current timestep data from HDF5
    step_idx = int(timestep) - 1  # Convert 1-based slider to 0-based index
    current_step = AVAILABLE_STEPS[step_idx]
    activation, temperature_field, metadata = get_step_data(current_step)
    target_position = get_target_position(metadata)
    if active_tab == "tab-follow-compare":
        # Render both orthographic and perspective cameras side-by-side
        PLANE_SIZE, RES = plane_and_res(cmp_plane_width, cmp_plane_height, VOXEL_SIZE, res_scale)

        # Create orthographic camera
        cam_ortho = make_camera_follow_ortho(
            PLANE_SIZE, RES, target_position, VOXEL_SIZE,
            cmp_follow_back, cmp_follow_up, cmp_tilt_angle
        )
        fig_ortho = render_image(cam_ortho, activation, temperature_field, temp_max)
        fig_ortho.update_layout(title=dict(text="Orthographic", x=0.5, xanchor='center'))

        # Create perspective camera
        cam_persp = make_camera_follow_persp(
            PLANE_SIZE, RES, target_position, VOXEL_SIZE,
            cmp_follow_back, cmp_follow_up, cmp_tilt_angle, cmp_fov
        )
        fig_persp = render_image(cam_persp, activation, temperature_field, temp_max)
        fig_persp.update_layout(title=dict(text=f"Perspective (FOV_y={cam_persp.fov_y_deg}°)", x=0.5, xanchor='center'))

        # Debug: Calculate what each camera actually sees
        distance = np.sqrt(cmp_follow_back ** 2 + cmp_follow_up ** 2)
        persp_fov_at_target = distance * 2 * np.tan(np.deg2rad(cam_persp.fov_y_deg / 2))
        persp_focal_length = cam_persp._focal_length

        # Create top-bottom layout
        image_container = html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_ortho, style={"height": "500px"}), width=12),
            ], style={"marginBottom": "10px"}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_persp, style={"height": "500px"}), width=12),
            ])
        ])

        meta = (
            f"Step {current_step} | Compare | back={cmp_follow_back * 1000:.2f}mm up={cmp_follow_up * 1000:.2f}mm "
            f"dist={distance * 1000:.2f}mm | "
            f"Ortho: plane={PLANE_SIZE[0] * 1000:.2f}mm (FOV constant) | "
            f"Persp: plane={cam_persp.plane_size[0] * 1000:.2f}mm "
            f"fov_y={cam_persp.fov_y_deg:.1f}° focal={persp_focal_length * 1000:.2f}mm "
            f"→ FOV@target={persp_fov_at_target * 1000:.2f}mm | RES={RES}"
        )
        return image_container, meta

    elif active_tab == "tab-cropping-persp":
        # Manual perspective camera crop configuration
        # Side view (looking from -X) sees Y-Z plane: (scan_direction, height)
        # Y=80mm is long (horizontal in plot), Z=10mm is short (vertical in plot)
        PLANE_SIZE = (0.06, 0.015)  # 60mm x 15mm sensor for good Y coverage

        # Note: target_window_mm expects mm; sliders are in meters → convert
        target_window_mm = (
            float(cb_crop_width) * 1000.0,
            float(cb_crop_height) * 1000.0
        )

        cb = CameraCallback(
            camera_type="perspective",
            offset=(cb_rel_x, cb_rel_y, cb_rel_z),
            fov_y_deg=float(cb_fov),
            plane_size=PLANE_SIZE,
            resolution_wh=(int(cb_res), int(cb_res)),  # Square resolution
            pixel_size_xy=None,
            ambient_temp=300.0,
            cmap="hot",
            save_images=False,
            target_window_mm=target_window_mm,
        )

        sim = SimpleNamespace(
            step_context={"position": {"x": target_position[0], "y": target_position[1], "z": target_position[2]}},
            config={"voxel_size": VOXEL_SIZE},
            volume_tracker=SimpleNamespace(activated=activation),
            temperature_tracker=SimpleNamespace(temperature=temperature_field),
            progress_tracker=SimpleNamespace(step_count=1),
        )

        cb._execute({"simulation": sim})
        latest = cb.get_latest_image()
        if latest is None:
            image_container = html.Div("No image produced by callback.")
            meta = "Callback: no output"
            return image_container, meta

        img, extent = latest
        fig = render_image_from_array(img, extent, title="Callback (real camera)", temp_max=temp_max)
        image_container = dcc.Graph(figure=fig, config={"displayModeBar": False})
        meta = (
            f"Step {current_step} | Callback | RES={img.shape[0]}×{img.shape[1]} | "
            f"FOV_y={cb_fov:.1f}° | tilt={cb_floor_angle:.1f}° | "
            f"rel=({cb_rel_x:.3f},{cb_rel_y:.3f},{cb_rel_z:.3f}) | "
            f"crop_window(m)=({cb_crop_width:.4f},{cb_crop_height:.4f})"
        )
        return image_container, meta

    elif active_tab == "tab-cropping-compare":
        # Cropping comparison - both ortho and perspective with exact crop parameters
        # Direct camera render_crop() comparison

        # Side view (looking from -X) sees Y-Z plane: (scan_direction, height)
        # Y=80mm is long (horizontal in plot), Z=10mm is short (vertical in plot)
        PLANE_SIZE = (0.06, 0.015)  # 60mm x 15mm sensor for good Y coverage
        RES = (int(cbc_res), int(cbc_res))  # Square resolution
        crop_window = (cbc_crop_width, cbc_crop_height)

        # Create orthographic camera - use same render_crop API as perspective
        cam_ortho = FollowingOrthographicCamera(
            source_pos=target_position,
            offset=(cbc_rel_x, cbc_rel_y, cbc_rel_z),  # Direct offset, no rotation!
            plane_size=PLANE_SIZE,
            voxel_size_xyz=VOXEL_SIZE
        )
        cam_ortho.set_resolution(*RES)

        # Use render_crop with fixed ROI calculation
        img_ortho, extent_ortho = cam_ortho.render_crop(
            activation, temperature_field,
            window_center=target_position,
            window_size=crop_window,
            ambient=300.0
        )
        fig_ortho = render_image_from_array(img_ortho, extent_ortho, "Orthographic", temp_max)

        # Create perspective camera
        cam_persp = FollowingPerspectiveCamera(
            source_pos=target_position,
            offset=(cbc_rel_x, cbc_rel_y, cbc_rel_z),  # Direct offset, no rotation!
            fov_y_deg=cbc_fov,
            plane_size=PLANE_SIZE,
            voxel_size_xyz=VOXEL_SIZE
        )
        cam_persp.set_resolution(*RES)

        # Use render_crop with fixed ROI calculation
        img_persp, extent_persp = cam_persp.render_crop(
            activation, temperature_field,
            window_center=target_position,
            window_size=crop_window,
            ambient=300.0
        )
        fig_persp = render_image_from_array(img_persp, extent_persp, "Perspective", temp_max)

        # Create top-bottom layout
        image_container = html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_ortho, style={"height": "500px"}), width=12),
            ], style={"marginBottom": "10px"}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_persp, style={"height": "500px"}), width=12),
            ])
        ])

        # Calculate actual FOV at target for perspective
        distance_to_target = np.linalg.norm([cbc_rel_x, cbc_rel_y, cbc_rel_z])
        actual_fov_height = distance_to_target * 2 * np.tan(np.deg2rad(cbc_fov / 2))
        actual_fov_width = actual_fov_height  # square aspect

        meta = (
            f"Step {current_step} | Projection Compare (Unified render_crop API) | RES={RES} | "
            f"Crop window: ({cbc_crop_width * 1000:.2f}×{cbc_crop_height * 1000:.2f}mm) (BOTH cameras) | "
            f"Persp optical: FOV_y={cbc_fov:.1f}° → full FOV@target=({actual_fov_width * 1000:.2f}×{actual_fov_height * 1000:.2f}mm) | "
            f"dist={distance_to_target * 1000:.1f}mm"
        )
        return image_container, meta

    elif active_tab == "tab-callback-real":
        # Real CameraCallback instances - production configuration
        # Side view (looking from -X) sees Y-Z plane: (scan_direction, height)
        # Y=80mm is long (horizontal in plot), Z=10mm is short (vertical in plot)
        PLANE_SIZE = (0.06, 0.015)  # 60mm x 15mm sensor for good Y coverage
        RES = (int(cbr_res), int(cbr_res))
        crop_window_m = (cbr_crop_width, cbr_crop_height)
        crop_window_mm = (cbr_crop_width * 1000.0, cbr_crop_height * 1000.0)

        # Create orthographic callback
        cb_ortho = CameraCallback(
            camera_type="orthographic",
            offset=(cbr_rel_x, cbr_rel_y, cbr_rel_z),
            plane_size=PLANE_SIZE,
            resolution_wh=RES,
            ambient_temp=300.0,
            cmap="hot",
            save_images=False,
            target_window_mm=crop_window_mm,
        )

        # Create perspective callback
        cb_persp = CameraCallback(
            camera_type="perspective",
            offset=(cbr_rel_x, cbr_rel_y, cbr_rel_z),
            fov_y_deg=float(cbr_fov),
            plane_size=PLANE_SIZE,
            resolution_wh=RES,
            ambient_temp=300.0,
            cmap="hot",
            save_images=False,
            target_window_mm=crop_window_mm,
        )

        # Create mock simulation context
        sim = SimpleNamespace(
            step_context={"position": {"x": target_position[0], "y": target_position[1], "z": target_position[2]}},
            config={"voxel_size": VOXEL_SIZE},
            volume_tracker=SimpleNamespace(activated=activation),
            temperature_tracker=SimpleNamespace(temperature=temperature_field),
            progress_tracker=SimpleNamespace(step_count=1),
        )

        # Execute both callbacks
        cb_ortho._execute({"simulation": sim})
        cb_persp._execute({"simulation": sim})

        # Get images
        img_ortho, extent_ortho = cb_ortho.get_latest_image()
        img_persp, extent_persp = cb_persp.get_latest_image()

        fig_ortho = render_image_from_array(img_ortho, extent_ortho, "Ortho Callback", temp_max)
        fig_persp = render_image_from_array(img_persp, extent_persp, "Persp Callback", temp_max)

        # Create top-bottom layout
        image_container = html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_ortho, style={"height": "500px"}), width=12),
            ], style={"marginBottom": "10px"}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_persp, style={"height": "500px"}), width=12),
            ])
        ])

        meta = (
            f"Step {current_step} | Real CameraCallback | RES={RES} | "
            f"Crop: ({cbr_crop_width * 1000:.2f}×{cbr_crop_height * 1000:.2f}mm) | "
            f"Persp FOV={cbr_fov:.1f}°"
        )
        return image_container, meta

    elif active_tab == "tab-following-timestep":
        # Following camera through timesteps - test if following actually tracks target
        PLANE_SIZE, RES = plane_and_res(fts_plane_width, fts_plane_height, VOXEL_SIZE, res_scale)

        # Create orthographic following camera
        cam_ortho = FollowingOrthographicCamera(
            source_pos=target_position,
            offset=(0.0, -fts_follow_back, fts_follow_up),
            plane_size=PLANE_SIZE,
            voxel_size_xyz=VOXEL_SIZE
        )
        cam_ortho.set_resolution(*RES)
        img_ortho, extent_ortho = cam_ortho.render_first_hit(
            activation, temperature_field, ambient=300.0
        )
        fig_ortho = render_image_from_array(img_ortho, extent_ortho, "Following Ortho", temp_max)

        # Create perspective following camera
        cam_persp = FollowingPerspectiveCamera(
            source_pos=target_position,
            offset=(0.0, -fts_follow_back, fts_follow_up),
            fov_y_deg=fts_fov,
            plane_size=PLANE_SIZE,
            voxel_size_xyz=VOXEL_SIZE
        )
        cam_persp.set_resolution(*RES)
        img_persp, extent_persp = cam_persp.render_first_hit(
            activation, temperature_field, ambient=300.0
        )
        fig_persp = render_image_from_array(img_persp, extent_persp, "Following Persp", temp_max)

        # Create top-bottom layout
        image_container = html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_ortho, style={"height": "500px"}), width=12),
            ], style={"marginBottom": "10px"}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_persp, style={"height": "500px"}), width=12),
            ])
        ])

        # Calculate distance from camera to target
        distance_ortho = np.linalg.norm(cam_ortho.pos - target_position)
        distance_persp = np.linalg.norm(cam_persp.pos - target_position)

        meta = (
            f"Following Timestep {current_step} | Target: ({target_position[0]*1000:.2f}, "
            f"{target_position[1]*1000:.2f}, {target_position[2]*1000:.2f})mm | "
            f"back={fts_follow_back*1000:.2f}mm up={fts_follow_up*1000:.2f}mm "
            f"tilt={fts_tilt_angle}° | Dist: ortho={distance_ortho*1000:.1f}mm "
            f"persp={distance_persp*1000:.1f}mm FOV={fts_fov}°"
        )
        return image_container, meta

    else:
        # Determine which plane size to use based on active tab
        if active_tab == "tab-ortho":
            PLANE_SIZE, RES = plane_and_res(ortho_plane_width, ortho_plane_height, VOXEL_SIZE, res_scale)
        elif active_tab == "tab-persp":
            PLANE_SIZE, RES = plane_and_res(persp_plane_width, persp_plane_height, VOXEL_SIZE, res_scale)
        elif active_tab == "tab-follow-ortho":
            PLANE_SIZE, RES = plane_and_res(follow_ortho_plane_width, follow_ortho_plane_height, VOXEL_SIZE, res_scale)
        elif active_tab == "tab-follow-persp":
            PLANE_SIZE, RES = plane_and_res(follow_persp_plane_width, follow_persp_plane_height, VOXEL_SIZE, res_scale)
        else:
            # Default fallback
            PLANE_SIZE, RES = plane_and_res(ortho_plane_width, ortho_plane_height, VOXEL_SIZE, res_scale)

        if active_tab == "tab-ortho":
            cam = make_camera_ortho(
                PLANE_SIZE, RES, ortho_radius, CENTER, VOXEL_SIZE,
                ortho_az, ortho_el, ortho_roll
            )
            meta = f"Step {current_step} | Ortho | az={ortho_az}° el={ortho_el}° roll={ortho_roll}° r={ortho_radius:.4f}m | RES={RES}"
        elif active_tab == "tab-persp":
            cam = make_camera_persp(
                PLANE_SIZE, RES, persp_radius, CENTER, VOXEL_SIZE,
                persp_az, persp_el, persp_roll, persp_fov
            )
            meta = (
                f"Step {current_step} | Persp | az={persp_az}° el={persp_el}° roll={persp_roll}° "
                f"r={persp_radius:.4f}m FOV={persp_fov}° | RES={RES}"
            )
        elif active_tab == "tab-follow-ortho":
            cam = make_camera_follow_ortho(
                PLANE_SIZE, RES, CENTER, VOXEL_SIZE,
                follow_ortho_back, follow_ortho_up, follow_ortho_floor
            )
            meta = (
                f"Step {current_step} | Follow Ortho | back={follow_ortho_back:.4f}m "
                f"up={follow_ortho_up:.4f}m floor={follow_ortho_floor}° | RES={RES}"
            )
        elif active_tab == "tab-follow-persp":
            cam = make_camera_follow_persp(
                PLANE_SIZE, RES, CENTER, VOXEL_SIZE,
                follow_persp_back, follow_persp_up, follow_persp_floor, follow_persp_fov
            )
            meta = (
                f"Step {current_step} | Follow Persp | back={follow_persp_back:.4f}m "
                f"up={follow_persp_up:.4f}m floor={follow_persp_floor}° "
                f"FOV={follow_persp_fov}° | RES={RES}"
            )
        else:
            cam = make_camera_ortho(
                PLANE_SIZE, RES, ortho_radius, CENTER, VOXEL_SIZE,
                ortho_az, ortho_el, ortho_roll
            )
            meta = f"Step {current_step} | Unknown tab"

        fig = render_image(cam, activation, temperature_field, temp_max)
        image_container = dcc.Graph(id="image", figure=fig)
        return image_container, meta


if __name__ == "__main__":
    app.run_server(debug=True)