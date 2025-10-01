"""
Gaussian Filter Utilities

This module provides core utilities for implementing masked Gaussian filters in multiple
dimensions. It contains optimized implementations of fundamental operations needed for
Gaussian filtering of masked arrays, with proper handling of boundary and reflection.

Key Features:
    - Gaussian kernel computation
    - Efficient 1D stripe filtering with proper reflection
    - Numba-accelerated implementations
    - Exact matching with scipy.ndimage.gaussian_filter1d behavior

The utilities in this module are designed to be dimension-agnostic and can be used
for both 2D and 3D (or potentially higher-dimensional) implementations.

Example:
    >>> import numpy as np
    >>> from gaussian_filter_utils import compute_gaussian_kernel
    >>> kernel = compute_gaussian_kernel(sigma=2.0)
    >>> print(f"Kernel size: {len(kernel)}")
"""

import numpy as np
from numba import jit
from typing import Tuple, Optional
import numpy.typing as npt

@jit(nopython=True)
def compute_gaussian_kernel(sigma, truncate=4.0) -> npt.NDArray:
    """
    Compute a 1D Gaussian kernel for filtering.

    This function creates a normalized 1D Gaussian kernel based on the specified
    standard deviation and truncation distance. The kernel is symmetric and
    normalized to sum to 1.0.

    Args:
        sigma (float): Standard deviation of the Gaussian kernel. Controls the
            amount of smoothing - larger values create wider kernels.
        truncate (float, optional): Distance, in standard deviations, at which to
            truncate the kernel. Default is 4.0, which includes 99.99% of the
            Gaussian distribution.

    Returns:
        numpy.ndarray: Normalized 1D Gaussian kernel.

    Note:
        The kernel size is determined by sigma * truncate, rounded up to the
        nearest integer. This means the kernel will always have an odd length_between,
        ensuring symmetry around the center point.
    """
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    return kernel / kernel.sum()


@jit(nopython=True)
def apply_gaussian_1d_to_stripe(data: npt.NDArray,
                                start: int,
                                end: int,
                                sigma: float,
                                truncate: float = 4.0) -> npt.NDArray:
    """
    Apply 1D Gaussian filter to a masked stripe with proper reflection at boundary.

    This function is an optimized version of the original implementation, achieving better
    performance through careful memory management and loop optimization while maintaining
    identical numerical results.

    Args:
        data (numpy.ndarray): 1D input array containing the stripe to be filtered.
        start (int): Start index of the valid region (inclusive).
        end (int): End index of the valid region (inclusive).
        sigma (float): Standard deviation for the Gaussian kernel.
        truncate (float, optional): Number of standard deviations at which to
            truncate the kernel. Default is 4.0.

    Returns:
        numpy.ndarray: Filtered array with same shape as input. Only the region
            between start and end (inclusive) is modified.

    Key Optimizations:
        1. Pre-allocated arrays to avoid memory reallocations
        2. Simplified reflection padding calculation
        3. Numba-optimized convolution loop
        4. Minimized array copying operations

    Example:
        >>> data = np.array([0., 1., 2., 3., 4., 0., 0.])
        >>> mask = np.array([0, 1, 1, 1, 1, 0, 0], dtype=bool)
        >>> start, end = 1, 4  # Valid region indices
        >>> filtered = apply_gaussian_1d_to_stripe(data, start, end, sigma=1.0)
    """
    if start >= end:
        return data

    # Get kernel and dimensions
    kernel = compute_gaussian_kernel(sigma, truncate)
    radius = len(kernel) // 2
    n_valid = end - start + 1

    # Pre-allocate arrays
    padded = np.empty(n_valid + 2 * radius, dtype=data.dtype)
    padded[radius:radius + n_valid] = data[start:end + 1]

    # Optimized reflection padding
    left_pad = padded[radius:2 * radius][::-1]
    right_pad = padded[n_valid:n_valid + radius][::-1]
    padded[:radius] = left_pad
    padded[radius + n_valid:] = right_pad

    # Optimized convolution
    result = np.empty(n_valid, dtype=data.dtype)
    kern_len = len(kernel)

    # Manual convolution loop optimized for Numba
    for i in range(n_valid):
        acc = 0.0
        for j in range(kern_len):
            acc += padded[i + j] * kernel[j]
        result[i] = acc

    # Modify output in-place
    out = data.copy()  # One copy still needed to preserve input
    out[start:end + 1] = result
    return out


def validate_inputs(data: np.ndarray,
                    mask: np.ndarray,
                    sigma: float,
                    axis: Optional[int] = None,
                    ndim: Optional[int] = None) -> None:
    """
    Validate inputs for masked Gaussian filtering.

    Args:
        data: Input array to be filtered
        mask: Binary mask where True indicates valid data points
        sigma: Standard deviation for Gaussian kernel
        axis: Optional axis along which to filter. If provided, ndim must also be provided
        ndim: Optional number of dimensions expected. If provided, axis must also be provided

    Raises:
        TypeError: If inputs have incorrect types
        ValueError: If inputs are invalid (shape mismatch, invalid sigma, etc.)
    """
    # Basic type checking
    if not isinstance(data, np.ndarray) or not isinstance(mask, np.ndarray):
        raise TypeError("Data and mask must be numpy arrays")

    if data.ndim != mask.ndim:
        raise ValueError(f"Data and mask must have same number of dimensions, got {data.ndim} and {mask.ndim}")

    if data.shape != mask.shape:
        raise ValueError(f"Data and mask must have same shape, got {data.shape} and {mask.shape}")

    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError("Mask must be a boolean array")

    if not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError(f"Sigma must be a positive number, got {sigma}")

    # Validate axis if provided
    if axis is not None:
        if ndim is None:
            raise ValueError("ndim must be provided when validating axis")

        if not isinstance(axis, int):
            raise TypeError(f"Axis must be an integer, got {type(axis)}")

        if not 0 <= axis < ndim:
            raise ValueError(f"Axis must be between 0 and {ndim - 1} for {ndim}D array, got {axis}")

    # Check for NaN/Inf values
    if not np.all(np.isfinite(data)):
        raise ValueError("Data contains NaN or Inf values")

    # Verify mask is not empty (optional, but might be useful)
    if not np.any(mask):
        raise ValueError("Mask is empty (no valid data points)")