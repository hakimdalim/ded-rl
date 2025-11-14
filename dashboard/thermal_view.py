import numpy as np
import plotly.graph_objects as go
import plotly.colors
from .styles import COLORS


def create_thermal_figure(view_name):
    """Create an empty thermal view figure"""
    fig = go.Figure()

    # Add dummy heatmap (will be replaced later)
    fig.add_trace(go.Heatmap(
        z=np.zeros((10, 10)),
        colorscale='Hot',
        showscale=True,
        colorbar=dict(
            title='Temperature (K)',
            titleside='right',
            titlefont=dict(size=12),
            tickfont=dict(size=10)
        ),
        name='Temperature'
    ))

    # Set layout
    fig.update_layout(
        title=f"{view_name}",
        xaxis=dict(title='Position (mm)'),
        yaxis=dict(title='Position (mm)'),
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        font=dict(size=10),
        coloraxis=dict(
            colorscale='Hot',
            colorbar=dict(title='Temperature (K)')
        )
    )

    return fig


def update_thermal_figure(thermal_data, view_name, temp_range=(300, 2500)):
    """Update thermal view figure with current data"""

    if thermal_data is None:
        return create_thermal_figure(view_name)

    # Create figure with heatmap
    fig = go.Figure()

    # Clip temperature data to range
    capped_data = np.clip(thermal_data, temp_range[0], temp_range[1])

    # Calculate dimensions based on data shape
    shape_y, shape_x = capped_data.shape
    voxel_size = 0.0002  # 200 µm

    # Create coordinates for heatmap (in mm)
    x = np.linspace(0, shape_x * voxel_size * 1000, 10)
    y = np.linspace(0, shape_y * voxel_size * 1000, 10)

    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=capped_data,
        x=x,
        y=y,
        colorscale='Hot',
        showscale=True,
        colorbar=dict(
            title='Temperature (K)',
            titleside='right',
            titlefont=dict(size=12),
            tickfont=dict(size=10),
            tickvals=[temp_range[0], 1000, 1500, 2000, temp_range[1]],
            ticktext=[f"{temp_range[0]}", "1000", "1500", "2000", f"{temp_range[1]}"],
        ),
        zmin=temp_range[0],
        zmax=temp_range[1],
        name='Temperature'
    ))

    # Add contour for melting temperature (assumed 1700K for typical metal)
    fig.add_trace(go.Contour(
        z=capped_data,
        x=x,
        y=y,
        contours=dict(
            start=1700,
            end=1700,
            size=0,
            coloring='lines',
            showlabels=True,
            labelfont=dict(
                color='white',
                size=10
            )
        ),
        colorscale=[[0, 'white'], [1, 'white']],
        showscale=False,
        name='Melt Pool Boundary'
    ))

    # Set layout
    fig.update_layout(
        title=f"{view_name}",
        xaxis=dict(title='Position (mm)'),
        yaxis=dict(title='Position (mm)'),
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        font=dict(size=10)
    )

    return fig


def create_temperature_history_figure(temperature_data, step_data=None):
    """Create figure showing temperature history at various points"""
    if not temperature_data or len(temperature_data) == 0:
        return go.Figure()

    # Create figure
    fig = go.Figure()

    # Extract data
    steps = list(range(len(temperature_data)))
    max_temps = [t.get('max_temp', 0) for t in temperature_data]

    # Add maximum temperature trace
    fig.add_trace(go.Scatter(
        x=steps,
        y=max_temps,
        mode='lines+markers',
        name='Max Temperature',
        line=dict(
            color=COLORS['temp_max'],
            width=2
        ),
        marker=dict(
            size=5,
            color=COLORS['temp_max']
        )
    ))

    # If available, add layer transitions
    if step_data and 'build.layer' in step_data:
        layers = step_data['build.layer'].values
        layer_transitions = []

        for i in range(1, len(layers)):
            if layers[i] > layers[i - 1]:
                layer_transitions.append(i)

        # Add vertical lines for layer transitions
        for idx in layer_transitions:
            fig.add_shape(
                type="line",
                x0=idx, y0=min(max_temps),
                x1=idx, y1=max(max_temps),
                line=dict(
                    color="rgba(255, 255, 255, 0.5)",
                    width=1,
                    dash="dash",
                )
            )
            fig.add_annotation(
                x=idx, y=max(max_temps),
                text=f"Layer {int(layers[idx])}",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(
                    color="rgba(255, 255, 255, 0.7)",
                    size=10
                )
            )

    # Add horizontal line at melting temperature
    melting_temp = 1700  # Example for steel
    fig.add_shape(
        type="line",
        x0=0, y0=melting_temp,
        x1=len(steps) - 1, y1=melting_temp,
        line=dict(
            color="rgba(255, 200, 0, 0.7)",
            width=1,
            dash="dot",
        )
    )

    fig.add_annotation(
        x=0, y=melting_temp,
        text=f"Melting Point ({melting_temp}K)",
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
        title='Temperature History',
        xaxis=dict(title='Simulation Step'),
        yaxis=dict(title='Temperature (K)'),
        template='plotly_dark',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def create_thermal_distribution_figure(temperature_volume):
    """Create figure showing temperature distribution histogram"""
    if temperature_volume is None:
        return go.Figure()

    # Flatten the temperature volume
    flat_temps = temperature_volume.flatten()

    # Remove values that are at room temperature (300K)
    active_temps = flat_temps[flat_temps > 310]

    if len(active_temps) == 0:
        return go.Figure()

    # Create figure
    fig = go.Figure()

    # Add histogram
    fig.add_trace(go.Histogram(
        x=active_temps,
        histnorm='probability',
        opacity=0.8,
        marker=dict(
            color=COLORS['temp_hist'],
            line=dict(
                color='white',
                width=0.5
            )
        ),
        name='Temperature Distribution'
    ))

    # Add vertical line at melting temperature
    melting_temp = 1700  # Example for steel
    fig.add_shape(
        type="line",
        x0=melting_temp, y0=0,
        x1=melting_temp, y1=1,
        yref="paper",
        line=dict(
            color="rgba(255, 200, 0, 0.9)",
            width=2,
            dash="dash",
        )
    )

    fig.add_annotation(
        x=melting_temp, y=0.9,
        yref="paper",
        text=f"Melting Point ({melting_temp}K)",
        showarrow=False,
        font=dict(
            color="rgba(255, 200, 0, 0.9)",
            size=10
        )
    )

    # Set layout
    fig.update_layout(
        title='Temperature Distribution',
        xaxis=dict(title='Temperature (K)'),
        yaxis=dict(title='Probability'),
        template='plotly_dark',
        bargap=0.1
    )

    return fig