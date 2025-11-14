import dash_bootstrap_components as dbc
from dash import html


def create_tutorial_modal():
    """Create modal for dashboard tutorial"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("DED Simulation Dashboard Tutorial")),
            dbc.ModalBody([
                html.H5("Welcome to the DED Simulation Dashboard!"),
                html.P(
                    "This dashboard allows you to control and visualize a Directed Energy Deposition simulation in real-time."),

                html.H6("Dashboard Layout"),
                html.Ul([
                    html.Li([html.Strong("Process Parameters:"),
                             " Adjust laser power, scan speed, and powder feed rate using the sliders on the left."]),
                    html.Li(
                        [html.Strong("Simulation Controls:"), " Start, pause, reset or step through the simulation."]),
                    html.Li([html.Strong("Metrics:"),
                             " View current simulation metrics like melt pool dimensions and temperatures."]),
                    html.Li([html.Strong("Visualizations:"),
                             " Explore different visualizations through the tabs on the right."]),
                ]),

                html.H6("Available Visualizations"),
                html.Ul([
                    html.Li([html.Strong("Build Process:"),
                             " 3D visualization of the build process and the actual build mesh."]),
                    html.Li([html.Strong("Thermal Analysis:"),
                             " Temperature distributions from multiple views and temperature history."]),
                    html.Li([html.Strong("Parameter Analysis:"),
                             " Effects of parameter changes on build characteristics."]),
                    html.Li(
                        [html.Strong("Build Animation:"), " Animated visualization of the build process over time."]),
                ]),

                html.H6("How to Use"),
                html.Ol([
                    html.Li("Set your desired process parameters using the sliders."),
                    html.Li("Click 'Apply Parameters' to update the simulation settings."),
                    html.Li("Click 'Start Simulation' to begin the simulation process."),
                    html.Li("Monitor the build progress and thermal distributions in real-time."),
                    html.Li("Pause or reset the simulation as needed."),
                    html.Li("Adjust parameters during the simulation to see their effects."),
                ]),

                html.H6("Interactive Mesh Visualization"),
                html.P([
                    "The mesh visualization shows the actual build geometry. You can interact with it by:",
                    html.Ul([
                        html.Li("Rotating: Click and drag to rotate the view."),
                        html.Li("Zooming: Scroll to zoom in and out."),
                        html.Li("Panning: Right-click and drag to pan the view."),
                        html.Li("Resetting: Double-click to reset the view."),
                    ])
                ]),

                html.H6("Build Animation"),
                html.P([
                    "To create an animation of the build process:",
                    html.Ol([
                        html.Li("Run the simulation for several steps."),
                        html.Li("Navigate to the 'Build Animation' tab."),
                        html.Li("Click 'Generate Animation' to create an animation from the saved mesh history."),
                        html.Li("Use the animation controls to play, pause, or scrub through the animation."),
                    ])
                ]),
            ]),
            dbc.ModalFooter(
                dbc.Button("Get Started", id="close-tutorial", className="ms-auto", n_clicks=0)
            ),
        ],
        id="tutorial-modal",
        size="lg",
        is_open=True,
    )