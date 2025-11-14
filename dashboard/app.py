import sys
import os
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import threading
import time
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import pandas as pd
import traceback

# Add parent directory to path so we can import simulation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from dashboard.simulation_controller import SimulationController
from dashboard.data_handler import DataHandler
from dashboard.visualization import create_3d_figure, update_3d_figure
from dashboard.thermal_view import create_thermal_figure, update_thermal_figure, create_temperature_history_figure
from dashboard.mesh_visualization import create_mesh_figure, update_mesh_figure
from dashboard.styles import COLORS, STYLES

# Import the PPO Controller at the top of app.py
from dashboard.ppo_control import PPOController

# Initialize PPO controller with None (will be set later)
ppo_controller = PPOController(model_path=r'C:\Users\schuermm\PycharmProjects\faim-jms-sim\_tensorboard_logs\ppo_parallel_study_v4\job15037670\15037670.model.zip')
# C:\Users\schuermm\PycharmProjects\faim-jms-sim\_tensorboard_logs\ppo_parallel_study_v4\job15037670\15037670.model.zip

# Initialize the Dash app with Bootstrap styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

# Initialize simulation controller and data handler
simulation_controller = SimulationController()
data_handler = DataHandler()

# Define app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("DED Simulations-Dashboard", className="text-center my-4"),
            # Enhanced status display with color-coded alerts and detailed information
            html.Div([
                dbc.Alert(
                    id="simulation-status",
                    color="info",
                    className="mb-3",
                    is_open=True
                ),
                # Collapsible error details
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardHeader("Fehlerdetails"),
                        dbc.CardBody(
                            html.Pre(id="error-details", className="error-details-pre")
                        )
                    ]),
                    id="error-collapse",
                    is_open=False,
                ),
                # Control buttons (visible when simulation stops with error or completes)
                dbc.Collapse(
                    dbc.ButtonGroup([
                        dbc.Button("Simulation zurücksetzen", id="status-reset-button", color="warning",
                                   className="me-2"),
                        dbc.Button("Simulation neu starten", id="status-restart-button", color="success",
                                   className="me-2"),
                        dbc.Button("Details anzeigen/ausblenden", id="toggle-error-details", color="info"),
                    ], className="mt-2 mb-3"),
                    id="status-control-buttons",
                    is_open=False,
                ),
            ], className="text-center")
        ], width=12)
    ]),

    dbc.Row([
        # Left column - Parameter controls and metrics
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prozessparameter"),
                dbc.CardBody([
                    html.Div([
                        html.Label("Laserleistung (W)"),
                        dcc.Slider(
                            id="laser-power-slider",
                            min=600, max=1600, step=10, value=800,
                            marks={i: str(i) for i in range(600, 1601, 200)},
                            className="mb-4"
                        ),
                    ]),
                    html.Div([
                        html.Label("Vorschubgeschwindigkeit (mm/s)"),
                        dcc.Slider(
                            id="scan-speed-slider",
                            min=2, max=20, step=0.1, value=3.0,
                            marks={i: str(i) for i in range(2, 21, 2)},
                            className="mb-4"
                        ),
                    ]),
                    html.Div([
                        html.Label("Pulverförderrate (g/min)"),
                        dcc.Slider(
                            id="powder-feed-slider",
                            min=2.0, max=4.0, step=0.1, value=2.0,
                            marks={i: str(i) for i in range(2, 5, 1)},
                            className="mb-4"
                        ),
                    ]),
                    html.Button("Parameter anwenden", id="apply-params-button",
                                className="btn btn-primary mt-2"),
                ])
            ], className="mb-4"),

            dbc.Card([
                dbc.CardHeader("Simulationssteuerung"),
                dbc.CardBody([
                    dbc.Button("Simulation starten", id="start-stop-button", color="success",
                               className="me-2"),
                    dbc.Button("Simulation zurücksetzen", id="reset-button", color="danger",
                               className="me-2"),
                    dbc.Button("Schritt vorwärts", id="step-button", color="info"),
                ])
            ], className="mb-4"),

            dbc.Card([
                dbc.CardHeader("KI-Steuerung"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(
                                id="model-path-input",
                                type="text",
                                placeholder="Pfad zum PPO-Modell",
                                value="",
                                className="mb-2"
                            ),
                        ], width=8),
                        dbc.Col([
                            dbc.Button("Modell laden", id="load-model-button", color="primary", className="mb-2")
                        ], width=4)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Switch(
                                id="ppo-active-switch",
                                label="KI-Steuerung aktivieren",
                                value=False,
                                className="mt-2"
                            ),
                        ], width=12)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="model-status", className="mt-2 small")
                        ], width=12)
                    ])
                ])
            ], className="mb-4"),

            dbc.Card([
                dbc.CardHeader("Kennzahlen"),
                dbc.CardBody([
                    html.Div(id="metrics-display", className="metrics-container")
                ])
            ]),


            # Hidden div for storing simulation state
            html.Div(id="simulation-state", style={"display": "none"}),

            # Hidden div for storing error information
            html.Div(id="simulation-error", style={"display": "none"}),

            # Interval component for periodic updates - always running
            dcc.Interval(
                id="interval-component",
                interval=1000,  # in milliseconds
                n_intervals=0,
                disabled=False  # Always active to update status
            ),

            # Separate interval for simulation steps
            dcc.Interval(
                id="simulation-interval",
                interval=1000,  # in milliseconds
                n_intervals=0,
                disabled=True  # Active only when simulation is running
            ),
        ], width=3),

        # Right column - Visualizations - Mesh on left, thermal views on right
        dbc.Col([
            dbc.Row([
                # Mesh visualization
                dbc.Col([
                    html.H4("Bauteil", className="text-center mb-2"),
                    dcc.Graph(
                        id="mesh-visualization",
                        figure=create_mesh_figure(),
                        style={"height": "calc(100vh - 190px)"}  # Responsive but fixed height
                    )
                ], width=6),

                # Thermal visualizations - stacked in a column with proportional heights
                dbc.Col([
                    html.H4("Temperaturfeld", className="text-center mb-2"),
                    dcc.Graph(
                        id="thermal-top-view",
                        figure=create_thermal_figure("Draufsicht"),
                        style={"height": "calc((100vh - 190px) / 3)"}  # 1/3 of available height
                    ),
                    dcc.Graph(
                        id="thermal-front-view",
                        figure=create_thermal_figure("Frontansicht"),
                        style={"height": "calc((100vh - 190px) / 3)"}  # 1/3 of available height
                    ),
                    dcc.Graph(
                        id="thermal-side-view",
                        figure=create_thermal_figure("Seitenansicht"),
                        style={"height": "calc((100vh - 190px) / 3)"}  # 1/3 of available height
                    ),
                ], width=3),

                # Time history visualizations - stacked in a column
                dbc.Col([
                    html.H4("Zeitliche Entwicklung", className="text-center mb-2"),
                    dcc.Graph(
                        id="temperature-history",
                        figure=go.Figure(layout={"title": "Temperaturverlauf"}),
                        style={"height": "calc((100vh - 190px) / 3)"}  # 1/3 of available height
                    ),
                    dcc.Graph(
                        id="clad-metrics",
                        figure=go.Figure(layout={"title": "Geometrie"}),
                        style={"height": "calc((100vh - 190px) / 3)"}  # 1/3 of available height
                    ),
                    dcc.Graph(
                        id="wetting-angle",
                        figure=go.Figure(layout={"title": "Kontaktwinkel"}),
                        style={"height": "calc((100vh - 190px) / 3)"}  # 1/3 of available height
                    ),
                ], width=3),
            ]),
        ], width=9),
    ]),
], fluid=True)


# Combined callback for both simulation status updates and toggling error details
@app.callback(
    [Output("simulation-status", "children"),
     Output("simulation-status", "color"),
     Output("error-details", "children"),
     Output("error-collapse", "is_open"),
     Output("status-control-buttons", "is_open")],
    [Input("interval-component", "n_intervals"),
     Input("toggle-error-details", "n_clicks")],
    [State("error-collapse", "is_open")]
)
def update_simulation_status(n_intervals, error_toggle, error_collapse_state):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Check current simulation state
    running = simulation_controller.running
    paused = simulation_controller.paused
    completed = simulation_controller.simulation_completed
    error = simulation_controller.simulation_error
    error_traceback = simulation_controller.error_traceback

    # Default values
    error_details = ""
    show_error_collapse = error_collapse_state
    show_control_buttons = False

    # If toggle error details button was clicked, toggle the collapse state
    if triggered_id == "toggle-error-details" and error_toggle:
        show_error_collapse = not error_collapse_state

    if error:
        # Simulation has an error
        status_message = [
            html.Strong("Bauteil Defekt "),
            html.Span("Die Simulation wurde aufgrund eines Fehlers beendet.")
        ]
        status_color = "danger"
        error_details = f"Error: {str(error)}\n\n{error_traceback}" if error_traceback else str(error)
        show_control_buttons = True

    elif completed:
        # Simulation completed successfully
        status_message = [
            html.Strong("Bauteilfertigung abgeschlossen "),
            html.Span("Die Simulation wurde erfolgreich beendet.")
        ]
        status_color = "success"
        show_control_buttons = True

    elif running and not paused:
        # Simulation is running
        metrics = simulation_controller.get_current_metrics()
        layer = metrics.get('layer', 0)
        track = metrics.get('track', 0)
        status_message = f"Simulation läuft - Schicht: {layer}, Spur: {track}"
        status_color = "primary"

    elif paused and not running:
        # Simulation is paused
        status_message = "Simulation pausiert"
        status_color = "warning"

    else:
        # Simulation is ready or in another state
        status_message = "Simulation bereit"
        status_color = "info"

    return status_message, status_color, error_details, show_error_collapse, show_control_buttons


# Callbacks for simulation control
@app.callback(
    [Output("simulation-interval", "disabled"),
     Output("start-stop-button", "children"),
     Output("start-stop-button", "color")],
    [Input("start-stop-button", "n_clicks"),
     Input("step-button", "n_clicks"),
     Input("reset-button", "n_clicks"),
     Input("status-reset-button", "n_clicks"),
     Input("status-restart-button", "n_clicks")],
    [State("simulation-interval", "disabled")]
)
# Callbacks for simulation control - with allow_duplicate=True for outputs
@app.callback(
    [Output("simulation-interval", "disabled", allow_duplicate=True),
     Output("start-stop-button", "children", allow_duplicate=True),
     Output("start-stop-button", "color", allow_duplicate=True)],
    [Input("start-stop-button", "n_clicks"),
     Input("step-button", "n_clicks"),
     Input("reset-button", "n_clicks"),
     Input("status-reset-button", "n_clicks"),
     Input("status-restart-button", "n_clicks")],
    [State("simulation-interval", "disabled")],
    prevent_initial_call=True  # Prevent firing on initial load
)
def control_simulation(start_clicks, step_clicks, reset_clicks,
                       status_reset_clicks, status_restart_clicks, interval_disabled):
    ctx = dash.callback_context

    if not ctx.triggered:
        return True, "Simulation starten", "success"

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Handle status control buttons
    if button_id == "status-reset-button":
        simulation_controller.reset_simulation()
        return True, "Simulation starten", "success"

    elif button_id == "status-restart-button":
        simulation_controller.reset_simulation()
        simulation_controller.start_simulation()
        return False, "Simulation pausieren", "warning"

    # Handle main simulation control buttons
    elif button_id == "reset-button":
        simulation_controller.reset_simulation()
        return True, "Simulation starten", "success"

    elif button_id == "step-button":
        try:
            simulation_controller.step_simulation()
            return interval_disabled, "Simulation starten" if interval_disabled else "Simulation pausieren", \
                "success" if interval_disabled else "warning"
        except Exception:
            # Error is handled in the simulation_controller and status update
            return True, "Simulation starten", "success"

    elif button_id == "start-stop-button":
        if interval_disabled:
            # Starting simulation - check if we're resuming from pause or starting new
            try:
                if simulation_controller.paused and simulation_controller.current_step > 0:
                    # Resume simulation
                    simulation_controller.paused = False
                    simulation_controller.running = True
                    return False, "Simulation pausieren", "warning"
                else:
                    # Start fresh simulation
                    simulation_controller.start_simulation()
                    return False, "Simulation pausieren", "warning"
            except Exception:
                # Error is handled in the simulation_controller and status update
                return True, "Simulation starten", "success"
        else:
            # Pausing simulation
            simulation_controller.pause_simulation()
            return True, "Simulation starten", "success"

    # Default return
    return interval_disabled, "Simulation starten" if interval_disabled else "Simulation pausieren", \
        "success" if interval_disabled else "warning"


# Callback to process simulation steps
@app.callback(
    Output("simulation-state", "children"),
    [Input("simulation-interval", "n_intervals")],
)
def process_simulation_step(n_intervals):
    # This function is called only when simulation is running
    # The actual step processing happens in the SimulationController thread
    # We just return the current step info for the UI
    if simulation_controller.running and not simulation_controller.paused:
        current_step = simulation_controller.current_step
        return f"Step: {current_step}"
    return "Simulation inactive"


# Callback for parameter updates
@app.callback(
    Output("simulation-error", "children"),
    [Input("apply-params-button", "n_clicks")],
    [State("laser-power-slider", "value"),
     State("scan-speed-slider", "value"),
     State("powder-feed-slider", "value")]
)
def update_parameters(n_clicks, laser_power, scan_speed, powder_feed):
    if n_clicks:
        try:
            params = {
                "laser_power": laser_power,
                "scan_speed": scan_speed / 1000,  # Convert mm/s to m/s
                "powder_feed_rate": powder_feed / (60 * 1000)  # Convert g/min to kg/s
            }
            # Update parameters using set_params to ensure all derived parameters are updated
            from configuration.process_parameters import set_params
            updated_params = set_params(**params)
            simulation_controller.update_parameters(updated_params)
            return f"Parameter aktualisiert: {params}"
        except Exception as e:
            return f"Fehler beim Aktualisieren der Parameter: {str(e)}"
    return "Noch keine Parameter angewendet"


# Callback for metrics updates
@app.callback(
    Output("metrics-display", "children"),
    [Input("interval-component", "n_intervals")]
)
def update_metrics(n_intervals):
    metrics = simulation_controller.get_current_metrics()

    if not metrics:
        return html.Div("Keine Kennzahlen verfügbar")

    # German translations for common metric names
    translations = {
        "temperature": "Temperatur",
        "melt_pool_width": "Schmelzbadbreite",
        "melt_pool_depth": "Schmelzbadtiefe",
        "melt_pool_length": "Schmelzbadlänge",
        "deposition_rate": "Abscheiderate",
        "layer_height": "Schichthöhe",
        "energy_density": "Energiedichte",
        "cooling_rate": "Abkühlrate",
        "build_time": "Bauzeit",
        "material_efficiency": "Materialeffizienz",
        "power_efficiency": "Leistungseffizienz"
    }

    metrics_display = []
    for key, value in metrics.items():
        if isinstance(value, float):
            value_formatted = f"{value:.3f}"
        else:
            value_formatted = str(value)

        # Translate key if available, otherwise use title-case with underscores replaced
        display_key = translations.get(key.lower(), key.replace('_', ' ').title())
        metrics_display.append(html.Div([
            html.Strong(f"{display_key}: "),
            html.Span(value_formatted)
        ]))

    return html.Div(metrics_display)


# Callback for mesh visualization updates
@app.callback(
    Output("mesh-visualization", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_mesh_visualization(n_intervals):
    # Get mesh data directly from the simulation controller
    mesh_data = simulation_controller.get_current_mesh()
    if not mesh_data:
        return create_mesh_figure()

    # Update the mesh visualization
    return update_mesh_figure(mesh_data)


# Callbacks for thermal view updates
@app.callback(
    [Output("thermal-top-view", "figure"),
     Output("thermal-front-view", "figure"),
     Output("thermal-side-view", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_thermal_views(n_intervals):
    thermal_data = simulation_controller.get_current_thermal_data()
    if not thermal_data:
        return [create_thermal_figure("Draufsicht"),
                create_thermal_figure("Frontansicht"),
                create_thermal_figure("Seitenansicht")]

    return [
        update_thermal_figure(thermal_data["xy"], "Draufsicht (XY)"),
        update_thermal_figure(thermal_data["xz"], "Frontansicht (XZ)"),
        update_thermal_figure(thermal_data["yz"], "Seitenansicht (YZ)")
    ]


# Callback for temperature history updates
@app.callback(
    Output("temperature-history", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_temperature_history(n_intervals):
    # Get temperature history from simulation controller
    temperature_history = simulation_controller.get_temperature_history() if hasattr(simulation_controller,
                                                                                     'get_temperature_history') else []

    fig = go.Figure()
    if temperature_history and len(temperature_history) > 0:
        steps = list(range(len(temperature_history)))
        fig.add_trace(go.Scatter(
            x=steps,
            y=temperature_history,
            mode='lines+markers',
            name='Max. Temperatur'
        ))

    fig.update_layout(
        title="Max. Temperatur",
        xaxis_title="Simulationsschritt",
        yaxis_title="Temperatur (K)",
        template="plotly_dark",
        margin=dict(l=50, r=30, t=50, b=50)
    )

    return fig


# Callback for clad metrics updates
@app.callback(
    Output("clad-metrics", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_clad_metrics(n_intervals):
    # Get clad metrics histories from simulation controller
    clad_height_history = simulation_controller.get_clad_height_history() if hasattr(simulation_controller,
                                                                                     'get_clad_height_history') else []
    melt_pool_depth_history = simulation_controller.get_melt_pool_depth_history() if hasattr(simulation_controller,
                                                                                             'get_melt_pool_depth_history') else []

    # Import the visualization function from the new module
    from dashboard.clad_visualization import update_clad_metrics_figure

    # Update the figure
    return update_clad_metrics_figure(clad_height_history, melt_pool_depth_history)


# Callback for wetting angle updates
@app.callback(
    Output("wetting-angle", "figure"),
    [Input("interval-component", "n_intervals")]
)
def update_wetting_angle(n_intervals):
    # Get wetting angle history from simulation controller
    wetting_angle_history = simulation_controller.get_wetting_angle_history() if hasattr(simulation_controller,
                                                                                         'get_wetting_angle_history') else []

    # Import the visualization function from the new module
    from dashboard.clad_visualization import update_wetting_angle_figure

    # Update the figure
    return update_wetting_angle_figure(wetting_angle_history)
    return fig



# Add callbacks for PPO control

# Callback to load model
@app.callback(
    [Output("model-status", "children"),
     Output("ppo-active-switch", "disabled")],
    [Input("load-model-button", "n_clicks")],
    [State("model-path-input", "value")]
)
def load_ppo_model(n_clicks, model_path):
    if n_clicks is None:
        return "Kein Modell geladen", True

    if not model_path:
        return "Bitte geben Sie einen Pfad zum Modell an", True

    success = ppo_controller.load_model(model_path)
    if success:
        return [
            html.Span(f"Modell erfolgreich geladen: {model_path}", className="text-success"),
            False  # Enable the switch
        ]
    else:
        return [
            html.Span(f"Fehler beim Laden des Modells: {model_path}", className="text-danger"),
            True  # Keep the switch disabled
        ]


# Callback for activating/deactivating PPO control
@app.callback(
    Output("model-status", "children", allow_duplicate=True),
    [Input("ppo-active-switch", "value")],
    [State("model-path-input", "value")],
    prevent_initial_call=True
)
def toggle_ppo_control(active, model_path):
    success = ppo_controller.activate(active)

    if active and success:
        return html.Span(f"KI-Steuerung aktiviert: {model_path}", className="text-success")
    elif active and not success:
        return html.Span("Fehler beim Aktivieren der KI-Steuerung", className="text-danger")
    else:
        return html.Span("KI-Steuerung deaktiviert", className="text-warning")


# Modify the update_parameters callback to use PPO controller when active
@app.callback(
    Output("simulation-error", "children", allow_duplicate=True),
    [Input("apply-params-button", "n_clicks"),
     Input("interval-component", "n_intervals")],
    [State("laser-power-slider", "value"),
     State("scan-speed-slider", "value"),
     State("powder-feed-slider", "value")],
    prevent_initial_call=True
)
def update_parameters_with_ppo(n_clicks, n_intervals, laser_power, scan_speed, powder_feed):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # Check if PPO controller is active
    if ppo_controller.is_active() and triggered_id == "interval-component":
        try:
            # Get current metrics and parameters from simulation
            current_params = simulation_controller.params.copy()

            # Get PPO recommendations
            updated_params = ppo_controller.predict_parameters(simulation_controller.current_step_data, current_params)

            # Update simulation parameters using set_params
            from configuration.process_parameters import set_params
            final_params = set_params(**updated_params)
            simulation_controller.update_parameters(final_params)
            return f"Parameter durch KI aktualisiert: {updated_params}"
        except Exception as e:
            return f"Fehler bei KI-gesteuerter Parameteraktualisierung: {str(e)}"

    # Manual parameter update via button click
    elif triggered_id == "apply-params-button" and n_clicks:
        try:
            params = {
                "laser_power": laser_power,
                "scan_speed": scan_speed / 1000,  # Convert mm/s to m/s
                "powder_feed_rate": powder_feed / (60 * 1000)  # Convert g/min to kg/s
            }
            from configuration.process_parameters import set_params
            final_params = set_params(**params)
            simulation_controller.update_parameters(final_params)
            return f"Parameter manuell aktualisiert: {params}"
        except Exception as e:
            return f"Fehler beim Aktualisieren der Parameter: {str(e)}"

    # Initial state or non-triggering input
    return dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True)