import numpy as np
import plotly.graph_objects as go
from collections import deque
from .styles import COLORS


def create_clad_metrics_figure():
    """Create an empty clad metrics figure"""
    fig = go.Figure()

    # Set layout
    fig.update_layout(
        title="Geometrie",
        xaxis_title="Simulationsschritt",
        yaxis_title="Kennwert (mm)",
        template="plotly_dark",
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def update_clad_metrics_figure(clad_height_history, melt_pool_depth_history):
    """Update clad metrics figure with current data"""
    if not clad_height_history or not melt_pool_depth_history:
        return create_clad_metrics_figure()

    # Create figure
    fig = go.Figure()

    # Get data
    steps = list(range(len(clad_height_history)))

    # Add clad height trace
    fig.add_trace(go.Scatter(
        x=steps,
        y=clad_height_history,
        mode='lines+markers',
        name='Schichthöhe',
        line=dict(
            color=COLORS.get('clad_height', '#00BFFF'),
            width=2
        ),
        marker=dict(
            size=5,
            color=COLORS.get('clad_height', '#00BFFF')
        )
    ))

    # Add melt pool depth trace
    fig.add_trace(go.Scatter(
        x=steps,
        y=melt_pool_depth_history,
        mode='lines+markers',
        name='Schmelzbad',
        line=dict(
            color=COLORS.get('melt_pool_depth', '#FF6347'),
            width=2
        ),
        marker=dict(
            size=5,
            color=COLORS.get('melt_pool_depth', '#FF6347')
        )
    ))

    # Set layout
    fig.update_layout(
        title="Geometrie",
        xaxis_title="Simulationsschritt",
        yaxis_title="Kennwert (mm)",
        template="plotly_dark",
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_wetting_angle_figure():
    """Create an empty wetting angle figure"""
    fig = go.Figure()

    # Set layout
    fig.update_layout(
        title="Kontaktwinkel",
        xaxis_title="Simulationsschritt",
        yaxis_title="Winkel (°)",
        template="plotly_dark",
        margin=dict(l=50, r=30, t=50, b=50)
    )

    return fig


def update_wetting_angle_figure(wetting_angle_history):
    """Update wetting angle figure with current data"""
    if not wetting_angle_history:
        return create_wetting_angle_figure()

    # Create figure
    fig = go.Figure()

    # Get data
    steps = list(range(len(wetting_angle_history)))

    # Convert radians to degrees if necessary (check one value)
    if wetting_angle_history and abs(max(wetting_angle_history, key=abs)) < 6.3:  # If max value less than 2π
        wetting_angles_deg = [angle * 180 / np.pi for angle in wetting_angle_history]
    else:
        wetting_angles_deg = wetting_angle_history

    # Add wetting angle trace
    fig.add_trace(go.Scatter(
        x=steps,
        y=wetting_angles_deg,
        mode='lines+markers',
        name='Kontaktwinkel',
        line=dict(
            color=COLORS.get('wetting_angle', '#7FFF00'),
            width=2
        ),
        marker=dict(
            size=5,
            color=COLORS.get('wetting_angle', '#7FFF00')
        )
    ))

    # Add reference lines for optimal wetting angle range (typically 30-60 degrees)
    fig.add_shape(
        type="line",
        x0=0, y0=00,
        x1=len(steps) - 1, y1=00,
        line=dict(
            color="rgba(255, 200, 0, 0.7)",
            width=1,
            dash="dot",
        )
    )

    fig.add_shape(
        type="line",
        x0=0, y0=90,
        x1=len(steps) - 1, y1=90,
        line=dict(
            color="rgba(255, 200, 0, 0.7)",
            width=1,
            dash="dot",
        )
    )

    fig.add_annotation(
        x=0, y=30,
        text="Min. optimal (30°)",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(
            color="rgba(255, 200, 0, 0.7)",
            size=10
        )
    )

    fig.add_annotation(
        x=0, y=60,
        text="Max. optimal (60°)",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(
            color="rgba(255, 200, 0, 0.7)",
            size=10
        )
    )

    # Set layout
    fig.update_layout(
        title="Kontaktwinkel",
        xaxis_title="Simulationsschritt",
        yaxis_title="Winkel (°)",
        template="plotly_dark",
        margin=dict(l=50, r=30, t=50, b=50)
    )

    return fig