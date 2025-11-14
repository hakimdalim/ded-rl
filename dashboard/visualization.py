import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
import plotly.express as px
from .styles import COLORS


def create_3d_figure():
    """Create empty 3D figure for build visualization"""
    fig = go.Figure()

    # Add dummy scatter point (will be replaced later)
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(
            size=1,
            color='white',
            opacity=0.0
        ),
        name='Initialization'
    ))

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)'),
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
        title="Build Visualization",
        showlegend=False,
    )

    return fig


def update_3d_figure(build_data):
    """Update 3D visualization with current build data"""
    if not build_data:
        return create_3d_figure()

    fig = go.Figure()

    # Extract data
    x, y, z = build_data.get('x', 0), build_data.get('y', 0), build_data.get('z', 0)
    layer = build_data.get('layer', 0)
    track = build_data.get('track', 0)
    clad_width = build_data.get('clad_width', 0.0007)
    clad_height = build_data.get('clad_height', 0.0002)

    # Convert to mm for display
    x_mm, y_mm, z_mm = x * 1000, y * 1000, z * 1000
    clad_width_mm = clad_width * 1000
    clad_height_mm = clad_height * 1000

    # Create substrate (always present)
    substrate_height = 5  # 5mm substrate height
    substrate_width = 20  # 20mm substrate width
    substrate_length = 20  # 20mm substrate length

    # Add substrate as a surface
    x_substrate = np.linspace(0, substrate_width, 10)
    y_substrate = np.linspace(0, substrate_length, 10)
    z_substrate = np.zeros((10, 10)) + substrate_height

    fig.add_trace(go.Surface(
        x=x_substrate,
        y=y_substrate,
        z=z_substrate,
        colorscale=[[0, 'rgb(80, 80, 80)'], [1, 'rgb(120, 120, 120)']],
        showscale=False,
        opacity=0.8,
        name='Substrate'
    ))

    # Add current position as a marker
    fig.add_trace(go.Scatter3d(
        x=[x_mm], y=[y_mm], z=[z_mm],
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            symbol='diamond'
        ),
        name='Current Position'
    ))

    # Add a cylinder to represent laser beam
    r = 0.2  # Beam radius in mm
    theta = np.linspace(0, 2 * np.pi, 36)
    z_beam = np.linspace(z_mm, z_mm + 10, 10)

    # Create arrays for the beam cylinder
    beam_x = []
    beam_y = []
    beam_z = []
    beam_i = []

    for i, height in enumerate(z_beam):
        intensity = 1 - (i / len(z_beam))  # Decreasing intensity with height
        for t in theta:
            beam_x.append(x_mm + r * np.cos(t) * intensity)
            beam_y.append(y_mm + r * np.sin(t) * intensity)
            beam_z.append(height)
            beam_i.append(intensity)

    fig.add_trace(go.Scatter3d(
        x=beam_x, y=beam_y, z=beam_z,
        mode='markers',
        marker=dict(
            size=3,
            color=beam_i,
            colorscale='Reds',
            opacity=0.5
        ),
        name='Laser Beam'
    ))

    # Add melt pool approximation as a colored sphere
    melt_pool_radius = clad_width_mm / 2
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)

    melt_x = x_mm + melt_pool_radius * np.outer(np.cos(u), np.sin(v))
    melt_y = y_mm + melt_pool_radius * np.outer(np.sin(u), np.sin(v))
    melt_z = z_mm + melt_pool_radius * np.outer(np.ones(np.size(u)), np.cos(v)) * 0.5  # Flattened sphere

    fig.add_trace(go.Surface(
        x=melt_x, y=melt_y, z=melt_z,
        colorscale=[[0, 'orange'], [1, 'red']],
        showscale=False,
        opacity=0.6,
        name='Melt Pool'
    ))

    # Add text annotation for current layer and track
    fig.add_trace(go.Scatter3d(
        x=[x_mm], y=[y_mm], z=[z_mm + 2],
        mode='text',
        text=[f'Layer: {layer}, Track: {track}'],
        textposition='top center',
        textfont=dict(
            color='white',
            size=12
        )
    ))

    # Set layout with appropriate scaling
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (mm)', range=[0, 20]),
            yaxis=dict(title='Y (mm)', range=[0, 20]),
            zaxis=dict(title='Z (mm)', range=[0, 15]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.75),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
        title="Build Visualization",
        showlegend=False,
    )

    return fig


def create_laser_path_figure(path_data):
    """Create 3D figure showing the laser path"""
    if not path_data or 'x' not in path_data:
        return create_3d_figure()

    fig = go.Figure()

    # Extract path data
    x = np.array(path_data['x']) * 1000  # Convert to mm
    y = np.array(path_data['y']) * 1000
    z = np.array(path_data['z']) * 1000
    layers = np.array(path_data['layer'])
    tracks = np.array(path_data['track'])

    # Get unique layers and tracks
    unique_layers = np.unique(layers)
    n_layers = len(unique_layers)

    # Create a colorscale for layers
    layer_colors = px.colors.sample_colorscale('Viridis', n_layers)

    # Plot each layer with a different color
    for i, layer in enumerate(unique_layers):
        layer_mask = layers == layer
        layer_x = x[layer_mask]
        layer_y = y[layer_mask]
        layer_z = z[layer_mask]
        layer_tracks = tracks[layer_mask]

        # Sort by track for proper line drawing
        track_sort_idx = np.argsort(layer_tracks)

        # For each track in the layer
        unique_tracks = np.unique(layer_tracks)

        for track in unique_tracks:
            track_mask = layer_tracks == track
            track_x = layer_x[track_mask]
            track_y = layer_y[track_mask]
            track_z = layer_z[track_mask]

            # Add the track as a line
            fig.add_trace(go.Scatter3d(
                x=track_x, y=track_y, z=track_z,
                mode='lines',
                line=dict(
                    color=layer_colors[i],
                    width=3
                ),
                name=f'Layer {layer}, Track {track}'
            ))

    # Add substrate as a surface
    substrate_height = 5  # 5mm substrate height
    substrate_width = 20  # 20mm substrate width
    substrate_length = 20  # 20mm substrate length

    x_substrate = np.linspace(0, substrate_width, 10)
    y_substrate = np.linspace(0, substrate_length, 10)
    z_substrate = np.zeros((10, 10)) + substrate_height

    fig.add_trace(go.Surface(
        x=x_substrate,
        y=y_substrate,
        z=z_substrate,
        colorscale=[[0, 'rgb(80, 80, 80)'], [1, 'rgb(120, 120, 120)']],
        showscale=False,
        opacity=0.8,
        name='Substrate'
    ))

    # Set layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X (mm)'),
            yaxis=dict(title='Y (mm)'),
            zaxis=dict(title='Z (mm)'),
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark",
        title="Laser Path Visualization",
        showlegend=True,
        legend=dict(
            itemsizing='constant',
            title='Layers and Tracks',
            font=dict(
                family='Arial',
                size=10
            )
        )
    )

    return fig


def create_clad_profile_figure(clad_data):
    """Create figure showing clad cross-section profile"""
    if not clad_data:
        return go.Figure()

    # Extract clad dimensions
    width = clad_data.get('clad.width', 0.0007) * 1000  # Convert to mm
    height = clad_data.get('clad.height', 0.0002) * 1000
    angle = clad_data.get('clad.wetting_angle', 120)  # In degrees

    # Create profile curve
    x = np.linspace(-width / 2, width / 2, 100)

    # Simple parabolic shape for the clad
    y = height * (1 - (2 * x / width) ** 2)
    y[y < 0] = 0  # Clip negative values

    # Create the figure
    fig = go.Figure()

    # Add clad profile
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        fill='tozeroy',
        line=dict(color=COLORS['clad'], width=3),
        name='Clad Profile'
    ))

    # Add horizontal line at base
    fig.add_trace(go.Scatter(
        x=[x.min(), x.max()], y=[0, 0],
        mode='lines',
        line=dict(color='gray', width=1, dash='dash'),
        name='Substrate Surface'
    ))

    # Add annotations for dimensions
    fig.add_annotation(
        x=0, y=height / 2,
        text=f'Height: {height:.2f} mm',
        showarrow=False,
        font=dict(color='white')
    )

    fig.add_annotation(
        x=0, y=-0.05,
        text=f'Width: {width:.2f} mm',
        showarrow=False,
        font=dict(color='white')
    )

    # Add angle markers
    fig.add_trace(go.Scatter(
        x=[width / 2, width / 2 - 0.2 * width],
        y=[0, height * 0.3],
        mode='lines',
        line=dict(color='yellow', width=1),
        name='Wetting Angle'
    ))

    fig.add_annotation(
        x=width / 2 - 0.1 * width,
        y=height * 0.15,
        text=f'{angle:.1f}°',
        showarrow=False,
        font=dict(color='yellow')
    )

    # Set layout
    fig.update_layout(
        title='Clad Cross-Section Profile',
        xaxis=dict(
            title='Width (mm)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray'
        ),
        yaxis=dict(
            title='Height (mm)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='gray'
        ),
        template='plotly_dark',
        showlegend=False
    )

    return fig