"""
Gaussian Filter Test Utilities

This module provides shared testing utilities for masked Gaussian filter implementations
across different dimensions, including challenging edge cases and asymmetric scenarios
designed to catch subtle implementation issues. The utilities support testing for
arbitrary N-dimensional arrays.
"""


import numpy as np
import timeit
from typing import Tuple, Callable, List, Union, Optional, Dict
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    size: Tuple[int, ...]
    our_time: float
    scipy_time: float
    speed_ratio: float
    max_diff: float
    mean_diff: float


def create_synthetic_data(size: Tuple[int, ...], pattern: str = 'sinusoidal') -> np.ndarray:
    """
    Create synthetic test data with different patterns.

    Args:
        size: Tuple of dimensions (e.g., (100,100) for 2D or (50,50,50) for 3D)
        pattern: Type of pattern to generate ('sinusoidal', 'gaussian', 'random')

    Returns:
        numpy.ndarray: Generated test data
    """
    if pattern == 'random':
        return np.random.randn(*size)

    # Create meshgrid for the given dimensions
    axes = [np.linspace(-5, 5, s) for s in size]
    grid = np.meshgrid(*axes, indexing='ij')

    if pattern == 'sinusoidal':
        # Create sinusoidal pattern appropriate for the dimensionality
        result = np.ones(size)
        for dim_grid in grid:
            result *= np.sin(dim_grid)
        return result + np.random.randn(*size) * 0.1

    elif pattern == 'gaussian':
        # Create Gaussian blob pattern
        result = np.zeros(size)
        for dim_grid in grid:
            result += dim_grid ** 2
        return np.exp(-result / 8)

    raise ValueError(f"Unknown pattern: {pattern}")


def create_test_mask(size: Tuple[int, ...], pattern: str = 'central_block') -> np.ndarray:
    """
    Create test masks with different patterns.

    Args:
        size: Tuple of dimensions
        pattern: Type of mask ('central_block', 'checkerboard', 'single_point', 'full')

    Returns:
        numpy.ndarray: Boolean mask array
    """
    mask = np.zeros(size, dtype=bool)

    if pattern == 'full':
        return np.ones(size, dtype=bool)

    if pattern == 'single_point':
        center = tuple(s // 2 for s in size)
        mask[center] = True
        return mask

    if pattern == 'central_block':
        slices = []
        for s in size:
            start = s // 4
            end = 3 * s // 4
            slices.append(slice(start, end))
        mask[tuple(slices)] = True
        return mask

    if pattern == 'checkerboard':
        # Create a checkerboard pattern appropriate for the dimensionality
        coords = np.indices(size)
        sum_coords = np.sum(coords, axis=0)
        mask = (sum_coords % 2) == 0
        return mask

    raise ValueError(f"Unknown mask pattern: {pattern}")


def run_benchmark(
        implementation_func: Callable,
        scipy_func: Callable,
        test_sizes: List[Tuple[int, ...]],
        sigma: float = 2.0
) -> List[BenchmarkResult]:
    """
    Run performance benchmark comparing implementation against scipy.

    Args:
        implementation_func: Function implementing the masked Gaussian filter
        scipy_func: Scipy reference implementation
        test_sizes: List of dimension tuples to test
        sigma: Sigma value for Gaussian filter

    Returns:
        List of BenchmarkResult objects containing performance metrics
    """
    results = []

    for size in test_sizes:
        # Create test data
        np.random.seed(42)  # For reproducibility
        data = create_synthetic_data(size, 'random')
        mask = create_test_mask(size, 'central_block')

        # Warm up JIT
        _ = implementation_func(data, mask, sigma)
        _ = scipy_func(data, mask, sigma)

        # Time our implementation
        times = timeit.repeat(
            lambda: implementation_func(data, mask, sigma),
            number=1, repeat=3
        )
        our_time = min(times) * 1000  # Convert to ms

        # Time scipy implementation
        times = timeit.repeat(
            lambda: scipy_func(data, mask, sigma),
            number=1, repeat=3
        )
        scipy_time = min(times) * 1000  # Convert to ms

        # Compute differences
        our_result = implementation_func(data, mask, sigma)
        scipy_result = scipy_func(data, mask, sigma)
        max_diff = np.max(np.abs(our_result - scipy_result))
        mean_diff = np.mean(np.abs(our_result - scipy_result))

        results.append(BenchmarkResult(
            size=size,
            our_time=our_time,
            scipy_time=scipy_time,
            speed_ratio=scipy_time / our_time,
            max_diff=max_diff,
            mean_diff=mean_diff
        ))

    return results


def print_benchmark_results(results: List[BenchmarkResult], dims: int):
    """
    Print formatted benchmark results.

    Args:
        results: List of BenchmarkResult objects
        dims: Number of dimensions (2 or 3)
    """
    print("\nRunning performance benchmarks...")
    print("-" * 100)
    print(f"{'Size':>15} {'Our Time (ms)':>15} {'Scipy Time (ms)':>15} "
          f"{'Speed Ratio':>15} {'Max Diff':>15} {'Mean Diff':>15}")
    print("-" * 100)

    for result in results:
        size_str = 'x'.join(str(s) for s in result.size)
        print(f"{size_str:>15} "
              f"{result.our_time:>15.3f} "
              f"{result.scipy_time:>15.3f} "
              f"{result.speed_ratio:>15.2f}x "
              f"{result.max_diff:>15.2e} "
              f"{result.mean_diff:>15.2e}")


def run_validation_test_cases(
        implementation_func: Callable,
        scipy_func: Callable,
        size: Tuple[int, ...],
        sigma: float = 2.0
) -> None:
    """
    Run a comprehensive set of validation tests.

    Args:
        implementation_func: Function implementing the masked Gaussian filter
        scipy_func: Scipy reference implementation
        size: Base size for test arrays
        sigma: Sigma value for Gaussian filter
    """

    def validate_case(name: str, data: np.ndarray, mask: np.ndarray) -> float:
        our_result = implementation_func(data, mask, sigma)
        scipy_result = scipy_func(data, mask, sigma)
        max_diff = np.max(np.abs(our_result - scipy_result))
        print(f"{name}: Max difference = {max_diff:.2e}")
        return max_diff

    print("\nRunning validation tests...")

    # Test different mask patterns
    print("\nTest 1: Different mask patterns")
    for pattern in ['central_block', 'checkerboard', 'single_point', 'full']:
        data = create_synthetic_data(size, 'random')
        mask = create_test_mask(size, pattern)
        validate_case(f"Mask pattern: {pattern}", data, mask)

    # Test different sigma values
    print("\nTest 2: Different sigma values")
    data = create_synthetic_data(size, 'random')
    mask = create_test_mask(size, 'central_block')
    for test_sigma in [0.5, 1.0, 2.0, 4.0]:
        our_result = implementation_func(data, mask, test_sigma)
        scipy_result = scipy_func(data, mask, test_sigma)
        max_diff = np.max(np.abs(our_result - scipy_result))
        print(f"Sigma {test_sigma}: Max difference = {max_diff:.2e}")

    # Test numerical stability
    print("\nTest 3: Numerical stability")
    mask = create_test_mask(size, 'central_block')

    # Small values
    data_small = create_synthetic_data(size, 'random') * 1e-6
    validate_case("Small values", data_small, mask)

    # Large values
    data_large = create_synthetic_data(size, 'random') * 1e6
    validate_case("Large values", data_large, mask)

    # Mixed values
    scaling = np.logspace(-6, 6, size[0])
    shape_tuple = (size[0],) + (1,) * (len(size) - 1)
    data_mixed = create_synthetic_data(size, 'random') * scaling.reshape(shape_tuple)
    validate_case("Mixed values", data_mixed, mask)




def create_asymmetric_data(size: Tuple[int, ...], complexity: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create challenging asymmetric test cases with known properties.

    Args:
        size: Tuple of dimensions
        complexity: Difficulty level ('easy', 'medium', 'hard')

    Returns:
        Tuple[np.ndarray, np.ndarray]: (data, mask) pair
    """
    data = np.zeros(size)
    mask = np.zeros(size, dtype=bool)

    if complexity == 'easy':
        # Asymmetric valid regions with sharp transitions
        slices = []
        for i, s in enumerate(size):
            start = s // (i + 2)  # Different fraction for each dimension
            end = s - s // (i + 3)  # Asymmetric end point
            slices.append(slice(start, end))

        mask[tuple(slices)] = True
        data[tuple(slices)] = 1.0

    elif complexity == 'medium':
        # Interleaved valid/invalid regions with varying widths
        for i in range(len(size)):
            selector = [slice(None)] * len(size)
            # Create strips of varying width
            for j in range(0, size[i], 3 + i):
                width = 1 + (j % 3)  # Varying width strips
                if j + width <= size[i]:
                    selector[i] = slice(j, j + width)
                    mask[tuple(selector)] = True
                    data[tuple(selector)] = j / size[i]  # Gradient value

    else:  # 'hard'
        # Complex pattern with multiple scales and isolated regions
        coords = np.indices(size)

        # Create complex validity mask using different frequencies per dimension
        mask_pattern = np.ones(size, dtype=bool)
        for i, coord in enumerate(coords):
            freq = 2 + i  # Different frequency per dimension
            phase = i * np.pi / 4  # Different phase per dimension
            mask_pattern &= (np.sin(2 * np.pi * freq * coord / size[i] + phase) > 0.7)

        # Add isolated valid points
        np.random.seed(42)
        random_points = np.random.rand(*size) > 0.95
        mask = mask_pattern | random_points

        # Create data with multiple frequency components
        for i, coord in enumerate(coords):
            data += np.sin(2 * np.pi * (i + 1) * coord / size[i])

        # Add some sharp transitions
        transitions = np.random.rand(*size) > 0.98
        data[transitions] = 10.0

    return data, mask


def create_challenging_cases(size: Tuple[int, ...]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Generate a set of challenging test cases designed to catch implementation issues.

    Args:
        size: Tuple of dimensions

    Returns:
        List of (description, data, mask) tuples
    """
    cases = []

    # Case 1: Highly asymmetric valid regions
    data = np.zeros(size)
    mask = np.zeros(size, dtype=bool)
    for i in range(len(size)):
        slices = [slice(None)] * len(size)
        slices[i] = slice(i, -1, 2 + i)  # Different strides per dimension
        mask[tuple(slices)] = True
        data[tuple(slices)] = 1.0
    cases.append(("Asymmetric strided regions", data, mask))

    # Case 2: Single-pixel wide valid regions
    data = np.zeros(size)
    mask = np.zeros(size, dtype=bool)
    for i in range(len(size)):
        slices = [slice(None)] * len(size)
        slices[i] = i % size[i]  # Single pixel wide line
        mask[tuple(slices)] = True
        data[tuple(slices)] = 1.0
    cases.append(("Single-pixel wide regions", data, mask))

    # Case 3: Alternating valid/invalid with varying frequencies
    data = np.zeros(size)
    mask = np.zeros(size, dtype=bool)
    for i in range(len(size)):
        selector = [slice(None)] * len(size)
        freq = 2 + i  # Different frequency per dimension
        selector[i] = slice(None)
        coord_line = np.arange(size[i])
        pattern = np.sin(2 * np.pi * freq * coord_line / size[i])
        valid_line = pattern > 0

        # Broadcast to full array
        broadcast_shape = [1] * len(size)
        broadcast_shape[i] = size[i]
        valid_pattern = valid_line.reshape(broadcast_shape)
        mask |= np.broadcast_to(valid_pattern, size)
        data += np.broadcast_to(pattern.reshape(broadcast_shape), size)
    cases.append(("Multi-frequency alternating regions", data, mask))

    # Case 4: Sparse isolated points
    np.random.seed(42)
    mask = np.random.rand(*size) > 0.95
    data = np.random.randn(*size) * mask
    cases.append(("Sparse isolated points", data, mask))

    # Case 5: Nested valid regions
    data = np.zeros(size)
    mask = np.zeros(size, dtype=bool)
    for scale in range(1, min(size) // 4):
        slices = []
        for s in size:
            start = scale
            end = s - scale
            slices.append(slice(start, end))
        mask[tuple(slices)] = ~mask[tuple(slices)]  # Toggle nested regions
        data[tuple(slices)] = scale
    cases.append(("Nested alternating regions", data, mask))

    return cases


def run_edge_case_validation(
        implementation_func: Callable,
        scipy_func: Callable,
        size: Tuple[int, ...],
        sigma: float = 2.0
) -> None:
    """
    Run validation tests specifically targeting edge cases and potential issues.

    Args:
        implementation_func: Function implementing the masked Gaussian filter
        scipy_func: Scipy reference implementation
        size: Base size for test arrays
        sigma: Sigma value for Gaussian filter
    """
    print("\nRunning edge case validation...")

    def validate_case(name: str, data: np.ndarray, mask: np.ndarray) -> float:
        our_result = implementation_func(data, mask, sigma)
        scipy_result = scipy_func(data, mask, sigma)
        max_diff = np.max(np.abs(our_result - scipy_result))
        print(f"{name}: Max difference = {max_diff:.2e}")
        return max_diff

    # Test asymmetric cases
    print("\nTest 1: Asymmetric cases")
    for complexity in ['easy', 'medium', 'hard']:
        data, mask = create_asymmetric_data(size, complexity)
        validate_case(f"Asymmetric case ({complexity})", data, mask)

    # Test challenging cases
    print("\nTest 2: Challenging cases")
    for desc, data, mask in create_challenging_cases(size):
        validate_case(desc, data, mask)

    # Test extreme value cases
    print("\nTest 3: Extreme value cases")

    # Case with very large gradient
    data = np.zeros(size)
    mask = create_test_mask(size, 'central_block')
    data[mask] = np.logspace(-6, 6, mask.sum())
    validate_case("Large gradient", data, mask)

    # Case with alternating extreme values
    data = np.ones(size)
    mask = create_test_mask(size, 'central_block')
    data[mask] = np.array([1e-6, 1e6] * (mask.sum() // 2 + 1))[:mask.sum()]
    validate_case("Alternating extreme values", data, mask)

    # Test boundary cases
    print("\nTest 4: Boundary cases")

    # Valid region at array edges
    for dim in range(len(size)):
        mask = np.zeros(size, dtype=bool)
        slices = [slice(None)] * len(size)
        slices[dim] = slice(0, 2)  # Edge region
        mask[tuple(slices)] = True
        data = np.random.randn(*size) * mask
        validate_case(f"Edge region (dim {dim})", data, mask)

    # Single valid point at corners
    for corner in range(2 ** len(size)):
        mask = np.zeros(size, dtype=bool)
        corner_idx = tuple(bool(corner & (1 << i)) * (s - 1) for i, s in enumerate(size))
        mask[corner_idx] = True
        data = np.random.randn(*size) * mask
        validate_case(f"Corner point {corner}", data, mask)


def run_basic_validation(
        implementation_func: Callable,
        scipy_func: Callable,
        size: Tuple[int, ...],
        sigma: float = 2.0
) -> Dict[str, float]:
    """
    Run basic validation tests for the implementation.

    Args:
        implementation_func: Function implementing the masked Gaussian filter
        scipy_func: Scipy reference implementation
        size: Base size for test arrays
        sigma: Sigma value for Gaussian filter

    Returns:
        Dict[str, float]: Dictionary of test names and their maximum differences
    """
    results = {}

    # Test all axes
    data = create_synthetic_data(size, 'random')
    mask = create_test_mask(size, 'central_block')

    for axis in range(len(size)):
        our_result = implementation_func(data, mask, sigma, axis=axis)
        scipy_result = scipy_func(data, mask, sigma, axis=axis)
        max_diff = np.max(np.abs(our_result - scipy_result))
        results[f'axis_{axis}'] = max_diff
        print(f"Basic validation - axis {axis}: Max difference = {max_diff:.2e}")

    # Test composition of all axes
    filtered = data.copy()
    for axis in range(len(size)):
        filtered = implementation_func(filtered, mask, sigma, axis=axis)

    filtered_scipy = data.copy()
    for axis in range(len(size)):
        filtered_scipy = scipy_func(filtered_scipy, mask, sigma, axis=axis)

    max_diff = np.max(np.abs(filtered - filtered_scipy))
    results['all_axes'] = max_diff
    print(f"Basic validation - all axes: Max difference = {max_diff:.2e}")

    return results


def run_error_cases(implementation_func: Callable, ndim: int = 2) -> None:
    """
    Test error handling for invalid inputs, adapting to the specified number of dimensions.

    Args:
        implementation_func: Function implementing the masked Gaussian filter
        ndim: Number of dimensions for test arrays (default 2)
    """
    print("\nTesting error cases...")

    # Create base test arrays with the specified dimensionality
    size = tuple(10 for _ in range(ndim))  # e.g., (10,) for 1D, (10, 10) for 2D, etc.
    data = np.random.randn(*size)
    mask = np.ones(size, dtype=bool)

    # Create mismatched shape by modifying the first dimension
    mismatched_shape = list(size)
    mismatched_shape[0] -= 1
    mismatched_mask = np.ones(tuple(mismatched_shape), dtype=bool)

    test_cases = [
        ("Mismatched shapes",
         lambda: implementation_func(data, mismatched_mask, 1.0)),
        ("Invalid sigma",
         lambda: implementation_func(data, mask, -1.0)),
        ("Invalid axis",
         lambda: implementation_func(data, mask, 1.0, axis=ndim)),
        ("Non-boolean mask",
         lambda: implementation_func(data, mask.astype(float), 1.0)),
        ("NaN in data",
         lambda: implementation_func(np.where(data == data.flat[0], np.nan, data), mask, 1.0)),
        ("Inf in data",
         lambda: implementation_func(np.where(data == data.flat[0], np.inf, data), mask, 1.0))
    ]

    # Add dimension-specific tests for higher dimensions
    if ndim > 1:
        # Test with wrong number of dimensions
        wrong_dim_size = tuple(10 for _ in range(ndim - 1))  # One dimension less
        wrong_dim_data = np.random.randn(*wrong_dim_size)
        wrong_dim_mask = np.ones(wrong_dim_size, dtype=bool)

        test_cases.extend([
            ("Wrong number of dimensions (data)",
             lambda: implementation_func(wrong_dim_data, mask, 1.0)),
            ("Wrong number of dimensions (mask)",
             lambda: implementation_func(data, wrong_dim_mask, 1.0))
        ])

    for test_name, test_func in test_cases:
        try:
            test_func()
            print(f"{test_name}: Failed to raise expected error")
        except Exception as e:
            print(f"{test_name}: Raised {type(e).__name__} - {str(e)}")


def run_numerical_stability_tests(
        implementation_func: Callable,
        scipy_func: Callable,
        size: Tuple[int, ...],
        sigma: float = 2.0
) -> Dict[str, float]:
    """Run tests specifically focused on numerical stability."""
    print("\nRunning numerical stability tests...")
    results = {}

    def test_case(name: str, data: np.ndarray, mask: np.ndarray) -> float:
        max_diffs = []
        for axis in range(len(size)):
            our_result = implementation_func(data, mask, sigma, axis=axis)
            scipy_result = scipy_func(data, mask, sigma, axis=axis)
            max_diffs.append(np.max(np.abs(our_result - scipy_result)))
        max_diff = max(max_diffs)
        results[name] = max_diff
        print(f"{name}: Max difference = {max_diff:.2e}")
        return max_diff

    mask = create_test_mask(size, 'central_block')

    # Very small values - reduced range
    data_tiny = np.random.randn(*size) * 1e-8
    test_case("Tiny values", data_tiny, mask)

    # Very large values - reduced range
    data_huge = np.random.randn(*size) * 1e8
    test_case("Huge values", data_huge, mask)

    # Mixed scales - more moderate range
    data_mixed = np.random.randn(*size)
    for i in range(len(size)):
        slicing = [slice(None)] * len(size)
        slicing[i] = slice(None)
        scale = np.logspace(-8, 8, size[i])  # More moderate range
        data_mixed[tuple(slicing)] *= scale.reshape([-1 if j == i else 1 for j in range(len(size))])
    test_case("Mixed scales", data_mixed, mask)

    # Alternating signs
    data_alternating = np.ones(size)
    mask_alternating = create_test_mask(size, 'checkerboard')
    data_alternating[mask_alternating] = -1
    test_case("Alternating signs", data_alternating, mask_alternating)

    return results


def run_benchmarks(
        implementation_func: Callable,
        scipy_func: Callable,
        sizes: List[Tuple[int, ...]]
) -> List[Dict[str, BenchmarkResult]]:
    """
    Run comprehensive benchmarks across different sizes and axes.

    Args:
        implementation_func: Function implementing the masked Gaussian filter
        scipy_func: Scipy reference implementation
        sizes: List of sizes to test

    Returns:
        List[Dict[str, BenchmarkResult]]: Benchmark results for each size and axis
    """
    all_results = []

    # Print header
    print("\nRunning performance benchmarks...")
    print("-" * 100)
    print(f"{'Size/Axis':>20} {'Our Time (ms)':>15} {'Scipy Time (ms)':>15} "
          f"{'Speed Ratio':>15} {'Max Diff':>15}")
    print("-" * 100)

    for size in sizes:
        results_dict = {}

        # Test each axis separately
        for axis in range(len(size)):
            results = run_benchmark(
                lambda d, m, s: implementation_func(d, m, s, axis=axis),
                lambda d, m, s: scipy_func(d, m, s, axis=axis),
                [size],
                sigma=2.0
            )
            result = results[0]
            results_dict[f'axis_{axis}'] = result

            size_str = 'x'.join(str(s) for s in size)
            print(f"{size_str + f' axis {axis}':>20} "
                  f"{result.our_time:>15.3f} "
                  f"{result.scipy_time:>15.3f} "
                  f"{result.speed_ratio:>15.2f}x "
                  f"{result.max_diff:>15.2e}")

        # Test all axes together
        def apply_all_axes(func, data, mask, sigma):
            result = data.copy()
            for ax in range(len(size)):
                result = func(result, mask, sigma, axis=ax)
            return result

        results = run_benchmark(
            lambda d, m, s: apply_all_axes(implementation_func, d, m, s),
            lambda d, m, s: apply_all_axes(scipy_func, d, m, s),
            [size],
            sigma=2.0
        )
        result = results[0]
        results_dict['all_axes'] = result

        size_str = 'x'.join(str(s) for s in size)
        print(f"{size_str + ' all':>20} "
              f"{result.our_time:>15.3f} "
              f"{result.scipy_time:>15.3f} "
              f"{result.speed_ratio:>15.2f}x "
              f"{result.max_diff:>15.2e}")

        print("-" * 100)  # Separator between sizes
        all_results.append(results_dict)

    return all_results


def run_no_mask_comparison(
        implementation_func: Callable,
        scipy_func: Callable,
        size: Tuple[int, ...],
        sigma: float = 2.0) -> Dict[str, float]:
    """
    Compare results of implementation against direct scipy.ndimage.gaussian_filter
    for the case with no masking (full array filtering).

    Args:
        implementation_func: Function implementing the masked Gaussian filter
        scipy_func: Reference scipy implementation using 1D filters
        size: Dimensions of test array
        sigma: Sigma value for Gaussian filter

    Returns:
        Dict containing maximum and mean differences between implementations
    """
    from scipy.ndimage import gaussian_filter

    print("\nRunning no-mask comparison test...")

    # Create test data
    np.random.seed(42)
    data = create_synthetic_data(size, 'random')
    mask = np.ones(size, dtype=bool)  # Full mask

    # Get results from our implementation (applying 1D filters sequentially)
    result_ours = data.copy()
    for axis in range(len(size)):
        result_ours = implementation_func(result_ours, mask, sigma, axis=axis)

    # Get results from scipy 1D implementation (for verification)
    result_scipy_1d = data.copy()
    for axis in range(len(size)):
        result_scipy_1d = scipy_func(result_scipy_1d, mask, sigma, axis=axis)

    # Get results from direct scipy nd implementation
    result_scipy_nd = gaussian_filter(data, sigma=sigma)

    # Compare results
    diff_ours_vs_scipy_1d = np.abs(result_ours - result_scipy_1d)
    diff_ours_vs_scipy_nd = np.abs(result_ours - result_scipy_nd)
    diff_scipy_1d_vs_nd = np.abs(result_scipy_1d - result_scipy_nd)

    results = {
        'max_diff_ours_vs_scipy_1d': np.max(diff_ours_vs_scipy_1d),
        'mean_diff_ours_vs_scipy_1d': np.mean(diff_ours_vs_scipy_1d),
        'max_diff_ours_vs_scipy_nd': np.max(diff_ours_vs_scipy_nd),
        'mean_diff_ours_vs_scipy_nd': np.mean(diff_ours_vs_scipy_nd),
        'max_diff_scipy_1d_vs_nd': np.max(diff_scipy_1d_vs_nd),
        'mean_diff_scipy_1d_vs_nd': np.mean(diff_scipy_1d_vs_nd)
    }

    # Print results
    print("\nResults for array size:", size)
    print("-" * 60)
    print("Comparison between our implementation and scipy 1D:")
    print(f"Max difference:  {results['max_diff_ours_vs_scipy_1d']:.2e}")
    print(f"Mean difference: {results['mean_diff_ours_vs_scipy_1d']:.2e}")
    print("\nComparison between our implementation and scipy ND:")
    print(f"Max difference:  {results['max_diff_ours_vs_scipy_nd']:.2e}")
    print(f"Mean difference: {results['mean_diff_ours_vs_scipy_nd']:.2e}")
    print("\nComparison between scipy 1D and ND implementations:")
    print(f"Max difference:  {results['max_diff_scipy_1d_vs_nd']:.2e}")
    print(f"Mean difference: {results['mean_diff_scipy_1d_vs_nd']:.2e}")

    return results


def run_all_tests(
        implementation_func: Callable,
        scipy_func: Callable,
        sizes: Optional[List[Tuple[int, ...]]] = None,
        ndim: int = 2
) -> Dict[str, Union[Dict[str, float], List[Dict[str, BenchmarkResult]]]]:
    """
    Run complete test suite for the implementation.

    Args:
        implementation_func: Your implementation
        scipy_func: Reference scipy implementation
        sizes: Optional list of sizes to test. If None, generates appropriate defaults
        ndim: Number of dimensions (default 2)

    Returns:
        Dict containing results from all test categories
    """
    if sizes is None:
        if ndim == 1:
            sizes = [(100,), (1000,), (10000,)]
        elif ndim == 2:
            sizes = [(100, 100), (500, 500), (1000, 1000)]
        else:  # 3D or higher
            sizes = [(20,) * ndim, (50,) * ndim, (100,) * ndim]

    print(f"\nRunning complete test suite for {ndim}D implementation")
    print("=" * 80)

    results = {}

    # 1. Basic validation tests
    results['basic_validation'] = run_basic_validation(
        implementation_func, scipy_func, sizes[0])

    # 2. Error case testing
    run_error_cases(implementation_func, ndim=ndim)

    # 3. Numerical stability tests
    results['numerical_stability'] = run_numerical_stability_tests(
        implementation_func, scipy_func, sizes[0])

    # 4. No mask comparison test (NEW)
    results['no_mask_comparison'] = run_no_mask_comparison(
        implementation_func, scipy_func, sizes[0])

    # 5. Performance benchmarks
    results['benchmarks'] = run_benchmarks(
        implementation_func, scipy_func, sizes)

    # 6. Edge cases and challenging tests
    run_edge_case_validation(
        implementation_func, scipy_func, sizes[0])

    return results