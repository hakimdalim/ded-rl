# dash_camera_app.py
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from camera.orthographic_camera import OrthographicCamera, FollowingOrthographicCamera
from camera.perspective_camera import PerspectiveCamera, FollowingPerspectiveCamera


from voxel.activated_volume import ActivatedVolume
from geometry.clad_profile_function import GenerateParabolicCladProfile

# Create a volume
volume = ActivatedVolume.from_dimensions(
    dimensions=(0.008, 0.008, 0.003),  # 8mm x 8mm x 3mm
    voxel_size=0.00005,  # 50 microns
    substrate_height=0.0002  # 0.2mm substrate
)

# Generate initial track profile (no existing cross-section)
start_profile = GenerateParabolicCladProfile.generate_profile_function(
    width=0.001,  # 1mm width
    height=0.0015,  # 1.5mm height
    track_center=0.004,  # centered at 4mm in x
    cross_section=None  # First track on substrate
)

# Generate end profile with different parameters
end_profile = GenerateParabolicCladProfile.generate_profile_function(
    width=0.0015,  # 1.5mm width
    height=0.001,  # 1mm height
    track_center=0.004,  # centered at 4mm in x
    cross_section=None  # Could provide existing surface here for multi-track
)

# Add track section
volume.add_track_section(
    start_profile=start_profile,
    end_profile=end_profile,
    length_between=0.005,  # 5mm long track
    y_position=0.0015  # starting at 1.5mm in y
)

# Initialize field with warm substrate
temperature_field = np.full(volume.shape, 300.0)
temperature_field[volume.activated] = 500.0

# Warm up substrate voxels (z < substrate_nz) to 400K
if volume.substrate_nz > 0:
    temperature_field[:, :, :volume.substrate_nz] = 400.0

# Track end position in voxel coordinates
track_end_x = int(0.004 / volume.voxel_size[0])   # ~4 mm
track_end_y = int((0.0015 + 0.005) / volume.voxel_size[1])  # ~6.5 mm
track_end_z = int(0.001 / volume.voxel_size[2])   # ~1 mm

# Gaussian hotspot
sigma = 30
peak_temp = 1500
for i in range(volume.shape[0]):
    for j in range(volume.shape[1]):
        for k in range(volume.shape[2]):
            if volume.activated[i, j, k]:
                dist_sq = ((i - track_end_x)**2 +
                           (j - track_end_y)**2 +
                           (k - track_end_z)**2)
                temperature_field[i, j, k] += (peak_temp - 400.0) * np.exp(-dist_sq / (2 * sigma**2))


# --------- scene helpers (auto from volume) ---------
def scene_params(volume):
    vx, vy, vz = volume.voxel_size
    nx, ny, nz = volume.activated.shape
    vol_size = np.array([nx*vx, ny*vy, nz*vz], dtype=float)
    center = vol_size / 2.0
    xy_max = float(max(vol_size[0], vol_size[1]))
    return vx, vy, vz, (nx, ny, nz), vol_size, center, xy_max

def plane_and_res(vol_size, voxel_size, plane_scale=1.1, res_scale=1.0):
    # Plane covers XY with margin; RES uses your "-10" artifact guard then scaled
    PLANE_SIZE = (plane_scale * vol_size[0], plane_scale * vol_size[1])
    px, py, _ = voxel_size
    base_res = (
        max(1, int(np.ceil(PLANE_SIZE[0] / px)) - 10),
        max(1, int(np.ceil(PLANE_SIZE[1] / py)) - 10),
    )
    W = max(1, int(round(base_res[0] * res_scale)))
    H = max(1, int(round(base_res[1] * res_scale)))
    return PLANE_SIZE, (W, H)

def make_camera(camera_key, plane_size, res, radius, center, voxel_size, fov_y_deg=40.0,
                follow_back=0.003, follow_up=0.001, floor_angle=25.0, orient=True):
    common = dict(plane_size=plane_size, voxel_size_xyz=voxel_size)
    if camera_key == "ortho":
        cam = OrthographicCamera(**common)
    elif camera_key == "persp":
        cam = PerspectiveCamera(fov_y_deg=fov_y_deg, **common)
    elif camera_key == "follow_ortho":
        cam = FollowingOrthographicCamera(
            source_pos=(0.0, 0.0, 0.0),
            rel_offset_local=(0.0, -follow_back, follow_up),
            floor_angle_deg=floor_angle,
            **common
        )
    elif camera_key == "follow_persp":
        cam = FollowingPerspectiveCamera(
            source_pos=(0.0, 0.0, 0.0),
            rel_offset_local=(0.0, -follow_back, follow_up),
            floor_angle_deg=floor_angle,
            fov_y_deg=fov_y_deg,
            **common
        )
    else:
        raise ValueError("Unknown camera key")
    cam.set_resolution(*res)
    # For following cams, set initial orientation once
    if "follow" in camera_key:
        cam.follow_heat_source((0.0, 0.0, 0.0), orient=orient)
    # Set pose around target
    cam.set_from_spherical(az_deg=45, el_deg=20, radius=radius, target=center)
    return cam

def render_image(cam, activated, temperature_field, az, el, roll, radius, center):
    cam.set_from_spherical(az_deg=az, el_deg=el, roll_deg=roll, radius=radius, target=center)
    img, extent = cam.render_first_hit(activated, temperature_field)
    (xmin, xmax, ymin, ymax) = extent
    fig = go.Figure(data=go.Heatmap(
        z=img.T, x=np.linspace(xmin, xmax, img.shape[0]),
        y=np.linspace(ymin, ymax, img.shape[1]), colorscale="Inferno", zsmooth=False
    ))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=650)
    return fig

# --------- DASH APP (expects `volume` and `temperature_field` in globals) ---------
app = Dash(__name__)

# Pull initial scene scale from the globally-defined volume
vx, vy, vz, (nx, ny, nz), VOL_SIZE, CENTER, XY_MAX = scene_params(volume)

app.layout = html.Div([
    html.H3("Interactive Camera Viewer (Ortho & Perspective, Static & Following)"),
    html.Div([
        html.Div([
            html.Label("Camera Type"),
            dcc.Dropdown(
                id="camera-type",
                options=[
                    {"label": "Orthographic", "value": "ortho"},
                    {"label": "Perspective", "value": "persp"},
                    {"label": "Following (Orthographic)", "value": "follow_ortho"},
                    {"label": "Following (Perspective)", "value": "follow_persp"},
                ],
                value="ortho", clearable=False
            ),
            html.Br(),
            html.Label("Azimuth (°)"),
            dcc.Slider(id="az", min=0, max=360, step=1, value=45, tooltip={"always_visible": False}),
            html.Label("Elevation (°)"),
            dcc.Slider(id="el", min=-80, max=80, step=1, value=20),
            html.Label("Roll (°)"),
            dcc.Slider(id="roll", min=0, max=180, step=1, value=0),
            html.Br(),
            html.Label("Radius (m)"),
            dcc.Slider(id="radius", min=0.5*XY_MAX, max=4.0*XY_MAX, step=0.001, value=2.0*XY_MAX),
            html.Label("Plane Scale (×XY)"),
            dcc.Slider(id="plane-scale", min=1.0, max=1.6, step=0.02, value=1.1),
            html.Label("Resolution Scale (×pixels)"),
            dcc.Slider(id="res-scale", min=0.25, max=4.0, step=0.25, value=1.0),
        ], style={"width": "32%", "display": "inline-block", "verticalAlign": "top"}),

        html.Div([
            dcc.Graph(id="image"),
            html.Div(id="meta", style={"fontFamily": "monospace", "marginTop": "6px"})
        ], style={"width": "66%", "display": "inline-block", "paddingLeft": "12px"}),
    ]),

    html.Hr(),
    html.Div([
        html.H4("Perspective / Following Options"),
        html.Label("FOV Y (°) — (Perspective only)"),
        dcc.Slider(id="fov", min=20, max=90, step=1, value=40),
        html.Br(),
        html.Label("Follow Back (m) — (Following only)"),
        dcc.Slider(id="follow-back", min=0.2*XY_MAX, max=3.0*XY_MAX, step=0.0005, value=1.5*XY_MAX),
        html.Label("Follow Up (m) — (Following only)"),
        dcc.Slider(id="follow-up", min=0.0, max=1.5*XY_MAX, step=0.0005, value=0.5*XY_MAX),
        html.Label("Floor Angle (°) — (Following only)"),
        dcc.Slider(id="floor-angle", min=0, max=60, step=1, value=25),
        dcc.Checklist(
            id="orient",
            options=[{"label": " Re-orient toward source", "value": "on"}],
            value=["on"]
        ),
    ]),
])

@app.callback(
    Output("image", "figure"),
    Output("meta", "children"),
    Input("camera-type", "value"),
    Input("az", "value"),
    Input("el", "value"),
    Input("roll", "value"),
    Input("radius", "value"),
    Input("plane-scale", "value"),
    Input("res-scale", "value"),
    Input("fov", "value"),
    Input("follow-back", "value"),
    Input("follow-up", "value"),
    Input("floor-angle", "value"),
    Input("orient", "value"),
)
def update(camera_type, az, el, roll, radius, plane_scale, res_scale,
           fov_y_deg, follow_back, follow_up, floor_angle, orient_val):
    PLANE_SIZE, RES = plane_and_res(VOL_SIZE, volume.voxel_size, plane_scale, res_scale)
    cam = make_camera(
        camera_key=camera_type,
        plane_size=PLANE_SIZE,
        res=RES,
        radius=radius,
        center=CENTER,
        voxel_size=volume.voxel_size,
        fov_y_deg=fov_y_deg,
        follow_back=follow_back,
        follow_up=follow_up,
        floor_angle=floor_angle,
        orient=("on" in orient_val) if isinstance(orient_val, list) else False
    )
    fig = render_image(cam, volume.activated, temperature_field, az, el, roll, radius, CENTER)
    meta = f"Type={camera_type} | RES={RES} | Plane={tuple(round(x,6) for x in PLANE_SIZE)} m | FOVy={fov_y_deg:.1f}°"
    return fig, meta

if __name__ == "__main__":
    # In a notebook, use: app.run_server(mode='inline') with jupyter_dash;
    # in a script, just:
    app.run_server(debug=True)
