"""
Styles for the DED Simulation Dashboard
"""

# Color palette
COLORS = {
    # Primary UI colors
    'primary': '#007bff',
    'secondary': '#6c757d',
    'success': '#28a745',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#17a2b8',

    # Domain-specific colors
    'laser': '#ff5722',  # Orange-red for laser
    'powder': '#90caf9',  # Light blue for powder
    'scan': '#4caf50',  # Green for scan speed
    'melt_pool': '#ff9800',  # Orange for melt pool
    'clad': '#9c27b0',  # Purple for clad
    'substrate': '#607d8b',  # Blue-grey for substrate

    # Temperature colors
    'temp_max': '#ff3d00',  # Bright red for max temperature
    'temp_hist': '#ff7043',  # Lighter red for temperature histogram

    # Layer colors
    'layer_odd': '#3f51b5',  # Indigo for odd layers
    'layer_even': '#2196f3',  # Blue for even layers

    # Background colors
    'bg_dark': '#212121',  # Dark background
    'bg_light': '#424242',  # Lighter background
    'text': '#ffffff',  # White text
    'text_secondary': '#b0bec5'  # Lighter text
}

# CSS Styles
STYLES = {
    'card': {
        'backgroundColor': COLORS['bg_light'],
        'borderRadius': '5px',
        'boxShadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
        'marginBottom': '20px',
        'padding': '15px',
    },

    'container': {
        'backgroundColor': COLORS['bg_dark'],
        'color': COLORS['text'],
        'minHeight': '100vh',
        'padding': '20px',
    },

    'header': {
        'color': COLORS['text'],
        'marginBottom': '30px',
        'textAlign': 'center',
    },

    'slider': {
        'marginBottom': '20px',
    },

    'slider_label': {
        'color': COLORS['text'],
        'marginBottom': '5px',
    },

    'button': {
        'marginRight': '10px',
    },

    'metrics_container': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(2, 1fr)',
        'gridGap': '10px',
        'padding': '10px',
    },

    'metric_item': {
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'flex-start',
        'padding': '10px',
        'backgroundColor': '#303030',
        'borderRadius': '5px',
    },

    'metric_label': {
        'color': COLORS['text_secondary'],
        'fontSize': '0.9rem',
        'marginBottom': '5px',
    },

    'metric_value': {
        'color': COLORS['text'],
        'fontSize': '1.1rem',
        'fontWeight': 'bold',
    },

    'graph_container': {
        'height': '100%',
        'minHeight': '400px',
        'backgroundColor': COLORS['bg_light'],
        'borderRadius': '5px',
        'padding': '15px',
    },

    'tab': {
        'backgroundColor': COLORS['bg_light'],
        'color': COLORS['text'],
        'padding': '10px 15px',
        'border': 'none',
        'borderBottom': f'3px solid {COLORS["bg_light"]}',
        'cursor': 'pointer',
        'transition': 'background-color 0.3s',
    },

    'tab_active': {
        'backgroundColor': COLORS['bg_light'],
        'color': COLORS['primary'],
        'borderBottom': f'3px solid {COLORS["primary"]}',
    },

    'tab_content': {
        'backgroundColor': COLORS['bg_light'],
        'padding': '15px',
        'borderRadius': '0 0 5px 5px',
    },

    'legend': {
        'backgroundColor': 'rgba(0, 0, 0, 0.7)',
        'color': COLORS['text'],
        'borderRadius': '5px',
        'padding': '10px',
        'position': 'absolute',
        'top': '10px',
        'right': '10px',
        'fontSize': '0.9rem',
    },
}

# Plotly figure templates
PLOT_LAYOUT = {
    'font': {
        'family': 'Arial, sans-serif',
        'color': COLORS['text'],
    },
    'paper_bgcolor': COLORS['bg_light'],
    'plot_bgcolor': COLORS['bg_light'],
    'margin': {'t': 40, 'b': 40, 'l': 40, 'r': 40},
}

# 3D Layout specifics
LAYOUT_3D = {
    'scene': {
        'xaxis': {
            'backgroundcolor': COLORS['bg_dark'],
            'gridcolor': 'rgba(255, 255, 255, 0.1)',
            'showbackground': True,
            'zerolinecolor': 'white',
        },
        'yaxis': {
            'backgroundcolor': COLORS['bg_dark'],
            'gridcolor': 'rgba(255, 255, 255, 0.1)',
            'showbackground': True,
            'zerolinecolor': 'white',
        },
        'zaxis': {
            'backgroundcolor': COLORS['bg_dark'],
            'gridcolor': 'rgba(255, 255, 255, 0.1)',
            'showbackground': True,
            'zerolinecolor': 'white',
        },
        'camera': {
            'eye': {'x': 1.5, 'y': 1.5, 'z': 1.2},
        },
        'aspectratio': {'x': 1, 'y': 1, 'z': 0.8},
    },
}

# Colorscales
COLORSCALES = {
    'temperature': [
        [0.0, '#000004'],
        [0.1, '#1b0c41'],
        [0.2, '#4a0c6b'],
        [0.3, '#781c6d'],
        [0.4, '#a52c60'],
        [0.5, '#cf4446'],
        [0.6, '#ed6925'],
        [0.7, '#fb9b06'],
        [0.8, '#f7d13d'],
        [1.0, '#fcffa4'],
    ],
    'layer': [
        [0.0, COLORS['layer_odd']],
        [1.0, COLORS['layer_even']],
    ],
}