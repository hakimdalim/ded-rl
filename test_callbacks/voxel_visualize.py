import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict, Any, Callable


class VoxelVisualizer:
    """Efficient voxel visualization using Plotly surface rendering."""

    def __init__(self, shape: tuple, voxel_size: np.ndarray):
        """
        Args:
            shape: Grid dimensions (nx, ny, nz)
            voxel_size: Physical size per voxel in meters
        """
        self.shape = shape
        self.voxel_size = voxel_size
        self.scale = voxel_size * 1000  # Convert to mm for display

    def _extract_surface_faces(self, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract external faces from labeled volume.

        Args:
            labels: 3D array with voxel labels (0=empty, 1,2,... = different materials)

        Returns:
            Dict with vertices, faces, and labels arrays
        """
        # Pad to handle boundaries
        padded = np.pad(labels, 1, mode='constant', constant_values=0)

        vertices = []
        faces = []
        vertex_labels = []
        vertex_count = 0

        # Process each axis direction
        for axis in range(3):
            diff = np.diff(padded, axis=axis)
            face_indices = np.argwhere(diff != 0)

            for idx in face_indices:
                i, j, k = idx
                label = max(padded[i, j, k], padded[i + (axis == 0), j + (axis == 1), k + (axis == 2)])

                if label > 0:  # Only process non-empty faces
                    # Calculate face corners based on axis
                    if axis == 0:  # YZ plane
                        corners = [
                            [i * self.scale[0], (j - 1) * self.scale[1], (k - 1) * self.scale[2]],
                            [i * self.scale[0], j * self.scale[1], (k - 1) * self.scale[2]],
                            [i * self.scale[0], j * self.scale[1], k * self.scale[2]],
                            [i * self.scale[0], (j - 1) * self.scale[1], k * self.scale[2]]
                        ]
                    elif axis == 1:  # XZ plane
                        corners = [
                            [(i - 1) * self.scale[0], j * self.scale[1], (k - 1) * self.scale[2]],
                            [i * self.scale[0], j * self.scale[1], (k - 1) * self.scale[2]],
                            [i * self.scale[0], j * self.scale[1], k * self.scale[2]],
                            [(i - 1) * self.scale[0], j * self.scale[1], k * self.scale[2]]
                        ]
                    else:  # XY plane
                        corners = [
                            [(i - 1) * self.scale[0], (j - 1) * self.scale[1], k * self.scale[2]],
                            [i * self.scale[0], (j - 1) * self.scale[1], k * self.scale[2]],
                            [i * self.scale[0], j * self.scale[1], k * self.scale[2]],
                            [(i - 1) * self.scale[0], j * self.scale[1], k * self.scale[2]]
                        ]

                    vertices.extend(corners)
                    vertex_labels.extend([label] * 4)

                    # Two triangles per quad
                    base = vertex_count
                    faces.extend([
                        [base, base + 1, base + 2],
                        [base, base + 2, base + 3]
                    ])
                    vertex_count += 4

        return {
            'vertices': np.array(vertices) if vertices else np.empty((0, 3)),
            'faces': np.array(faces) if faces else np.empty((0, 3), dtype=int),
            'labels': np.array(vertex_labels) if vertex_labels else np.empty(0)
        }

    def _apply_color_map(self, labels: np.ndarray, color_map: Callable) -> list:
        """Apply color mapping to vertex labels.

        Args:
            labels: Array of vertex labels
            color_map: Function mapping label -> color string

        Returns:
            List of colors for each vertex
        """
        return [color_map(label) for label in labels]

    def create_figure(self,
                      activated: np.ndarray,
                      substrate_nz: int = 0,
                      color_override: Optional[np.ndarray] = None,
                      colorscale: str = 'Viridis',
                      title: str = "Voxel Visualization") -> go.Figure:
        """Create Plotly figure from voxel data.

        Args:
            activated: Boolean array of activated voxels
            substrate_nz: Number of substrate layers in z
            color_override: Optional array same shape as activated for custom coloring
            colorscale: Plotly colorscale name if using color_override
            title: Figure title

        Returns:
            Plotly figure object
        """
        # Create labels
        if color_override is not None:
            # Use override values directly as labels
            labels = np.zeros(self.shape)
            labels[activated] = color_override[activated]
        else:
            # Default: substrate=1, other material=2
            labels = np.zeros(self.shape, dtype=np.uint8)
            labels[activated] = 2
            if substrate_nz > 0:
                labels[:, :, :substrate_nz] = np.where(
                    activated[:, :, :substrate_nz], 1, 0
                )

        # Extract surface
        surface_data = self._extract_surface_faces(labels)

        if len(surface_data['vertices']) == 0:
            # Empty volume
            fig = go.Figure()
        else:
            # Create mesh
            vertices = surface_data['vertices']
            faces = surface_data['faces']

            if color_override is not None:
                # Use continuous colorscale
                mesh = go.Mesh3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    intensity=surface_data['labels'],
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(title="Value"),
                    flatshading=False,
                    opacity=1.0,
                    hovertemplate='X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>Z: %{z:.2f} mm',
                    contour=dict(
                        show=True,
                        color='#333333',  # Dark gray edges
                        width=1
                    )
                )
            else:
                # Discrete colors for substrate/material
                color_map = lambda l: '#3498db' if l == 1 else '#e74c3c'
                colors = self._apply_color_map(surface_data['labels'], color_map)

                mesh = go.Mesh3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    vertexcolor=colors,
                    flatshading=False,
                    opacity=1.0,
                    hovertemplate='X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>Z: %{z:.2f} mm',
                    showscale=False,
                    contour=dict(
                        show=True,
                        color='#333333',  # Dark gray edges
                        width=1
                    )
                )

            fig = go.Figure(data=[mesh])

        # Add axes
        self._add_axes(fig)

        # Configure layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X (mm)'),
                yaxis=dict(title='Y (mm)'),
                zaxis=dict(title='Z (mm)'),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            title=dict(
                text=title,
                font=dict(size=16)
            ),
            # paper_bgcolor='white',
            # plot_bgcolor='white',
            height=700,
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        # Add stats
        self._add_stats(fig, activated)

        return fig

    def _add_axes(self, fig: go.Figure):
        """Add coordinate axes to figure."""
        axis_length = max(self.shape * self.voxel_size) * 1000 * 0.3

        axes_data = [
            ([0, axis_length], [0, 0], [0, 0], 'red', 'X'),
            ([0, 0], [0, axis_length], [0, 0], 'green', 'Y'),
            ([0, 0], [0, 0], [0, axis_length], 'blue', 'Z')
        ]

        for x, y, z, color, label in axes_data:
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+text',
                line=dict(color=color, width=6),
                text=['', label],
                textfont=dict(size=12, color=color),
                textposition='top center',
                showlegend=False,
                hoverinfo='skip'
            ))

    def _add_stats(self, fig: go.Figure, activated: np.ndarray):
        """Add statistics annotation."""
        total = np.prod(self.shape)
        active = np.sum(activated)

        fig.add_annotation(
            text=f"Grid: {self.shape[0]}×{self.shape[1]}×{self.shape[2]}<br>"
                 f"Active: {active:,}/{total:,} ({100 * active / total:.1f}%)",
            xref="paper", yref="paper", x=0, y=1,
            xanchor="left", yanchor="top",
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.5)"
        )