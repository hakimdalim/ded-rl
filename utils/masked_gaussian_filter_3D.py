"""
Masked Gaussian Filter Implementation (3D)

This module provides an efficient implementation of a masked Gaussian filter that operates
on masked 3D numpy arrays. It supports filtering along specified axes with proper reflection
at mask boundary.

Key Features:
    - Separable 3D Gaussian filtering along specified axes
    - Proper reflection padding at mask boundary
    - Efficient boundary detection using binary search
    - Parallel processing for improved performance
    - Exact matching with scipy.ndimage.gaussian_filter1d behavior

Example:
    >>> import numpy as np
    >>> data = np.random.randn(50, 50, 50)
    >>> mask = np.zeros_like(data, dtype=bool)
    >>> mask[10:40, :, :] = True  # Set valid region
    >>> filtered = apply_masked_gaussian_3d(data, mask, sigma=2.0, axis=0)
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from numba import jit, prange

from utils.gaussian_filter_test_utils import run_all_tests
from utils.gaussian_filter_utils import validate_inputs, apply_gaussian_1d_to_stripe


@jit(nopython=True, parallel=True)
def find_boundaries_3d_binary_parallel(mask, axis=0):
    """
    Find the start and end indices of valid regions in a 3D binary mask using parallel binary search.

    Args:
        mask (numpy.ndarray): 3D binary array where True indicates valid data points
        axis (int): Axis along which to find boundary (0, 1, or 2)

    Returns:
        tuple: (starts, ends) arrays containing the start and end indices for each 1D stripe
    """
    # Get dimensions and rearrange if needed
    shape = mask.shape
    n_stripes = shape[(axis + 1) % 3] * shape[(axis + 2) % 3]
    stripe_length = shape[axis]

    # Pre-allocate output arrays
    starts = np.empty(n_stripes, dtype=np.int64)
    ends = np.empty(n_stripes, dtype=np.int64)

    # Create index mapping for different axis orientations
    for i in prange(n_stripes):
        # Convert flat index to 2D indices for other axes
        idx2 = i % shape[(axis + 2) % 3]
        idx1 = i // shape[(axis + 2) % 3]

        # Extract 1D stripe based on axis
        if axis == 0:
            stripe = mask[:, idx1, idx2]
        elif axis == 1:
            stripe = mask[idx1, :, idx2]
        else:  # axis == 2
            stripe = mask[idx1, idx2, :]

        # Find start using binary search
        left, right = 0, stripe_length - 1
        while left < right:
            mid = (left + right) // 2
            if stripe[mid]:
                right = mid
            else:
                left = mid + 1
        starts[i] = left if stripe[left] else stripe_length

        # Find end using binary search
        left, right = 0, stripe_length - 1
        while left < right:
            mid = (left + right + 1) // 2
            if stripe[mid]:
                left = mid
            else:
                right = mid - 1
        ends[i] = right if stripe[right] else -1

    return starts, ends


@jit(nopython=True, parallel=True)
def apply_gaussian_filter_parallel_3d(data, mask, starts, ends, sigma, axis=0):
    """
    Apply Gaussian filter to all stripes in parallel for 3D data.

    Args:
        data (numpy.ndarray): 3D input array
        mask (numpy.ndarray): Binary mask where True indicates valid data
        starts (numpy.ndarray): Array of start indices for each stripe
        ends (numpy.ndarray): Array of end indices for each stripe
        sigma (float): Standard deviation for Gaussian kernel
        axis (int): Axis along which to apply the filter (0, 1, or 2)

    Returns:
        numpy.ndarray: Filtered array with same shape as input
    """
    result = data.copy()
    shape = data.shape
    n_stripes = len(starts)

    for i in prange(n_stripes):
        # Convert flat index to 2D indices for other axes
        idx2 = i % shape[(axis + 2) % 3]
        idx1 = i // shape[(axis + 2) % 3]

        # Extract and filter stripe based on axis
        if axis == 0:
            stripe = result[:, idx1, idx2].copy()
            filtered_stripe = apply_gaussian_1d_to_stripe(stripe, starts[i], ends[i], sigma)
            result[:, idx1, idx2] = filtered_stripe
        elif axis == 1:
            stripe = result[idx1, :, idx2].copy()
            filtered_stripe = apply_gaussian_1d_to_stripe(stripe, starts[i], ends[i], sigma)
            result[idx1, :, idx2] = filtered_stripe
        else:  # axis == 2
            stripe = result[idx1, idx2, :].copy()
            filtered_stripe = apply_gaussian_1d_to_stripe(stripe, starts[i], ends[i], sigma)
            result[idx1, idx2, :] = filtered_stripe

    return result


def apply_masked_gaussian_3d(data, mask, sigma, axis=0):
    """
    Apply 3D masked Gaussian filter along specified axis.

    This function applies a Gaussian filter to masked 3D data, handling boundary through
    reflection. The implementation is optimized for performance using parallel processing
    and matches scipy.ndimage.gaussian_filter1d behavior exactly.

    Args:
        data (numpy.ndarray): 3D input array
        mask (numpy.ndarray): Binary mask where True indicates valid data points
        sigma (float): Standard deviation for Gaussian kernel
        axis (int): Axis along which to apply the filter (0, 1, or 2)

    Returns:
        numpy.ndarray: Filtered array with same shape as input

    Raises:
        ValueError: If inputs are invalid or shapes don't match
    """
    validate_inputs(data, mask, sigma, axis=axis, ndim=3)

    # Find boundary for all stripes
    starts, ends = find_boundaries_3d_binary_parallel(mask, axis)

    # Apply Gaussian filter with parallel processing
    filtered = apply_gaussian_filter_parallel_3d(data, mask, starts, ends, sigma, axis)

    return filtered


if __name__ == "__main__":
    def apply_masked_gaussian_3d_scipy(data, mask, sigma, axis=0):
        """
        Alternative implementation using scipy's gaussian_filter1d for validation.
        """
        if data.shape != mask.shape:
            raise ValueError("Data and mask must have same shape")

        # Find boundary
        starts, ends = find_boundaries_3d_binary_parallel(mask, axis)

        # Create working copy
        filtered = data.copy()
        shape = data.shape
        n_stripes = len(starts)

        # Apply filter to each stripe
        for i in range(n_stripes):
            idx2 = i % shape[(axis + 2) % 3]
            idx1 = i // shape[(axis + 2) % 3]

            if starts[i] <= ends[i]:
                if axis == 0:
                    stripe = filtered[:, idx1, idx2]
                    valid_data = stripe[starts[i]:ends[i] + 1]
                    filtered_valid = gaussian_filter1d(valid_data, sigma=sigma, mode='reflect')
                    filtered[starts[i]:ends[i] + 1, idx1, idx2] = filtered_valid
                elif axis == 1:
                    stripe = filtered[idx1, :, idx2]
                    valid_data = stripe[starts[i]:ends[i] + 1]
                    filtered_valid = gaussian_filter1d(valid_data, sigma=sigma, mode='reflect')
                    filtered[idx1, starts[i]:ends[i] + 1, idx2] = filtered_valid
                else:  # axis == 2
                    stripe = filtered[idx1, idx2, :]
                    valid_data = stripe[starts[i]:ends[i] + 1]
                    filtered_valid = gaussian_filter1d(valid_data, sigma=sigma, mode='reflect')
                    filtered[idx1, idx2, starts[i]:ends[i] + 1] = filtered_valid

        return filtered


    # Run comprehensive tests
    results = run_all_tests(
        apply_masked_gaussian_3d,
        apply_masked_gaussian_3d_scipy,
        sizes=[(20, 20, 20), (50, 50, 50), (100, 100, 100), (250, 250, 250), (500, 500, 500)],
        ndim=3
    )