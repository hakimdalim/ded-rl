import numpy as np
from numpy.typing import ArrayLike, NDArray

from geometry.activation_boundary import ActivationBoundary


class TopSurfaceBoundary(ActivationBoundary):
    """Boundary defined by a ruled surface between two profiles."""

    def __init__(self, surface_function):
        """
        Initialize surface boundary.

        Args:
            surface_function: Function that returns z-coordinates for given x-y-coordinates
        """
        self.surface_function = surface_function

    def contains_points(self, points: ArrayLike) -> NDArray[np.bool_]:
        """Check if points lie below the surface.

        Args:
            points: Array of shape (..., 3) containing world-space coordinates

        Returns:
            Boolean array of shape (...,) where True indicates points below surface
        """
        points = np.asarray(points)
        orig_shape = points.shape
        points = np.atleast_2d(points)

        surface_z = self.surface_function(points)

        # Points are inside if below surface
        result = points[..., 2] <= surface_z
        return result.reshape(orig_shape[:-1])  # Reshape back to match input
