from typing import Union, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ActivationBoundary:
    """Base class for spatial boundary that can check point containment and create activation masks."""

    def contains_points(self, points: ArrayLike) -> NDArray[np.bool_]:
        """Check if points lie inside this boundary instance.

        Args:
            points: Array of shape (..., 3) containing point coordinates

        Returns:
            Boolean array of shape (...,) where True indicates points inside boundary

        Raises:
            NotImplementedError: If method not implemented in subclass
        """
        raise NotImplementedError("contains_points not implemented")

    @staticmethod
    def create_coordinate_grid(
            shape: Tuple[int, int, int],
            voxel_size: Union[float, Tuple[float, float, float]]
    ) -> NDArray[np.float64]:
        """Create a grid of coordinate points for the given shape and voxel size.

        Args:
            shape: Grid dimensions (nx, ny, nz)
            voxel_size: Voxel dimensions, either uniform (float) or per-axis (tuple)

        Returns:
            Array of shape (nx, ny, nz, 3) containing coordinate points
        """
        if isinstance(voxel_size, (int, float)):
            voxel_size = (float(voxel_size),) * 3
        voxel_size = np.array(voxel_size, dtype=float)

        x = np.arange(shape[0]) * voxel_size[0]
        y = np.arange(shape[1]) * voxel_size[1]
        z = np.arange(shape[2]) * voxel_size[2]
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        return np.stack([X, Y, Z], axis=-1)

    def create_mask(
            self,
            shape: Tuple[int, int, int],
            voxel_size: Union[float, Tuple[float, float, float]]
    ) -> NDArray[np.bool_]:
        """Create boolean mask indicating which voxels lie inside this boundary instance.

        Args:
            shape: Grid dimensions (nx, ny, nz)
            voxel_size: Voxel dimensions, either uniform (float) or per-axis (tuple)

        Returns:
            Boolean array of shape (nx, ny, nz) marking voxels inside boundary
        """
        points = self.create_coordinate_grid(shape, voxel_size)
        return self.contains_points(points)

    def activate_inplace(self, array: NDArray[np.bool_], voxel_size: Union[float, Tuple[float, float, float]]) -> None:
        """Set array elements inside this boundary instance to True.

        Args:
            array: Boolean array to modify, shape (nx, ny, nz)
            voxel_size: Voxel dimensions, either uniform (float) or per-axis (tuple)
        """
        mask = self.create_mask(array.shape, voxel_size)
        array |= mask

    def deactivate_inplace(self, array: NDArray[np.bool_],
                           voxel_size: Union[float, Tuple[float, float, float]]) -> None:
        """Set array elements inside this boundary instance to False.

        Args:
            array: Boolean array to modify, shape (nx, ny, nz)
            voxel_size: Voxel dimensions, either uniform (float) or per-axis (tuple)
        """
        mask = self.create_mask(array.shape, voxel_size)
        array &= ~mask