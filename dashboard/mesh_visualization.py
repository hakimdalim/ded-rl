import numpy as np
import plotly.graph_objects as go
from .styles import COLORS


def create_mesh_figure():
    """
    Create an empty figure for surface visualization

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Create a small placeholder surface
    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 5, 10)
    z = np.zeros((10, 10))  # Flat surface

    # Add placeholder surface
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='Viridis',
        opacity=0.3,
        showscale=False,
        name='Placeholder Surface'
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
        showlegend=False,
    )

    return fig


def update_mesh_figure(surface_data, colorscale='Viridis'):
    """
    Update surface visualization with current surface data

    Args:
        surface_data: Dictionary containing x, y, z arrays and build_info
        colorscale: Plotly colorscale name for surface coloring

    Returns:
        Updated Plotly figure
    """
    if not surface_data:
        print("update_mesh_figure: No surface data provided")
        return create_mesh_figure()

    try:
        # Check for required keys
        if not all(key in surface_data for key in ['x', 'y', 'z']):
            print(f"update_mesh_figure: Missing required keys in surface_data: {surface_data.keys()}")
            return create_mesh_figure()

        # Extract data
        x = surface_data['x']
        y = surface_data['y']
        z = surface_data['z']

        # Convert to mm for display
        x_mm = x * 1000
        y_mm = y * 1000
        z_mm = z * 1000

        # Get build info if provided, otherwise calculate from data
        if 'build_info' in surface_data:
            info = surface_data['build_info']
            x_min = info['x_min'] * 1000
            x_max = info['x_max'] * 1000
            y_min = info['y_min'] * 1000
            y_max = info['y_max'] * 1000
            z_min = info['z_min'] * 1000
            z_max = info['z_max'] * 1000
        else:
            # Calculate from data
            x_min, x_max = np.min(x_mm), np.max(x_mm)
            y_min, y_max = np.min(y_mm), np.max(y_mm)
            z_min, z_max = np.min(z_mm), np.max(z_mm)

        # Calculate mesh dimensions for display
        x_size = x_max - x_min
        y_size = y_max - y_min
        z_size = z_max - z_min

        # Add margins to axes (proportional to the build size)
        margin_factor = 0.1  # 10% margin
        x_margin = max(x_size * margin_factor, 1.0)  # at least 1mm
        y_margin = max(y_size * margin_factor, 1.0)
        z_margin = max(z_size * margin_factor, 1.0)

        # Set axis ranges with margins
        x_range = [x_min - x_margin, x_max + x_margin]
        y_range = [y_min - y_margin, y_max + y_margin]
        z_range = [z_min - z_margin, z_max + z_margin]

        # Create figure
        fig = go.Figure()

        # Add the surface plot
        fig.add_trace(go.Surface(
            x=x_mm,
            y=y_mm,
            z=z_mm,
            colorscale=colorscale,
            opacity=1.00,
            showscale=True,
            colorbar=dict(
                title='Höhe (mm)',
                titleside='right',
                titlefont=dict(size=12),
                tickfont=dict(size=10)
            ),
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=0.2,
                roughness=0.5,
                fresnel=0.2
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    project=dict(z=True)
                )
            ),
            name='Bauteil Oberfläche'
        ))

        # Set layout with equal unit scaling but different axis ranges
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X (mm)', range=x_range),
                yaxis=dict(title='Y (mm)', range=y_range),
                zaxis=dict(title='Z (mm)', range=z_range),
                aspectmode='manual',  # Use manual scaling
                aspectratio=dict(x=1, y=1, z=1),  # Equal units but different ranges
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_dark",
            showlegend=False,
        )

        print("update_mesh_figure: Successfully created surface figure")
        return fig

    except Exception as e:
        print(f"update_mesh_figure: Error generating figure: {str(e)}")
        import traceback
        traceback.print_exc()
        return create_mesh_figure()  # Return empty figure on error