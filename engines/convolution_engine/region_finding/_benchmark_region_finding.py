"""
Region Finding Benchmarking Suite - UPDATED FOR REFACTORED API

Comprehensive benchmarks for all region finder implementations.

Tests:
    1. Correctness verification across all backends
    2. Single signal performance benchmarking
    3. Batched performance benchmarking (NEW)
    4. Scalability analysis
    5. Edge case handling
"""

import numpy as np
import time
from typing import Dict, List
import sys


from convolution_engine.region_finding.region_finding_implementations import (
    find_regions_numpy,
    find_regions_numba,
    find_regions_torch,
    find_regions_python,
    find_regions_scipy,
    list_available_finders,
    find_regions,
)


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")


def create_test_mask(size: int,
                     region_pattern: str = 'alternating',
                     density: float = 0.5,
                     seed: int = 42) -> np.ndarray:
    """Create test mask with various patterns."""
    np.random.seed(seed)

    if region_pattern == 'alternating':
        mask = np.tile([True, True, True, False, False], size // 5 + 1)[:size]
    elif region_pattern == 'random':
        mask = np.random.rand(size) < density
    elif region_pattern == 'single':
        mask = np.zeros(size, dtype=bool)
        start = size // 4
        end = 3 * size // 4
        mask[start:end] = True
    elif region_pattern == 'many_small':
        mask = np.zeros(size, dtype=bool)
        pos = 0
        while pos < size:
            region_size = np.random.randint(3, 6)
            gap_size = np.random.randint(2, 4)
            mask[pos:pos+region_size] = True
            pos += region_size + gap_size
    elif region_pattern == 'few_large':
        mask = np.zeros(size, dtype=bool)
        pos = 0
        while pos < size:
            region_size = np.random.randint(100, 201)
            gap_size = np.random.randint(50, 101)
            mask[pos:min(pos+region_size, size)] = True
            pos += region_size + gap_size
    else:
        raise ValueError(f"Unknown pattern: {region_pattern}")

    return mask


def arrays_equal(arr1, arr2):
    """Compare two region arrays for equality."""
    if isinstance(arr1, list) and isinstance(arr2, list):
        if len(arr1) != len(arr2):
            return False
        return all(arrays_equal(a1, a2) for a1, a2 in zip(arr1, arr2))

    # Handle PyTorch tensors
    if hasattr(arr1, 'cpu'):
        arr1 = arr1.cpu().numpy()
    if hasattr(arr2, 'cpu'):
        arr2 = arr2.cpu().numpy()

    return np.array_equal(arr1, arr2)


def benchmark_correctness():
    """Verify all backends produce identical results."""
    print_section("CORRECTNESS VERIFICATION")

    test_cases = [
        (100, 'alternating', 2, "Small alternating"),
        (1000, 'random', 5, "Medium random"),
        (10000, 'single', 10, "Large single region"),
        (5000, 'many_small', 3, "Many small regions"),
        (10000, 'few_large', 50, "Few large regions"),
    ]

    available = list_available_finders()
    backends = [name for name, avail in available.items() if avail]

    print(f"Testing backends: {', '.join(backends)}\n")
    print(f"{'Test Case':<30} {'Status':<15} {'Details':<35}")
    print("-" * 80)

    all_passed = True

    for size, pattern, min_length, description in test_cases:
        mask = create_test_mask(size, pattern)

        # Get reference result (numpy)
        reference = find_regions_numpy(mask, min_length)

        # Test each backend
        test_name = f"{description} (n={size})"
        passed = True
        details = []

        for backend in backends:
            try:
                result = find_regions(mask, min_length, backend=backend)

                if not arrays_equal(result, reference):
                    passed = False
                    details.append(f"{backend}:MISMATCH")
                    all_passed = False

            except Exception as e:
                passed = False
                details.append(f"{backend}:ERROR")
                all_passed = False

        status = "✓ PASS" if passed else "✗ FAIL"
        detail_str = ', '.join(details) if details else f"{len(reference)} regions"
        print(f"{test_name:<30} {status:<15} {detail_str:<35}")

    print(f"\n{'='*80}")
    if all_passed:
        print("✓ ALL CORRECTNESS TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED - Check results above")
    print(f"{'='*80}")

    return all_passed


def benchmark_multi_region_explicit():
    """Explicitly test multi-region detection."""
    print_section("EXPLICIT MULTI-REGION TESTS")

    test_cases = [
        {
            'name': '3 regions with gaps',
            'mask': [True, True, True, False, False, True, True, True, True, False, True, True],
            'min_length': 2,
            'expected': [[0, 3], [5, 9], [10, 12]]
        },
        {
            'name': '4 regions, mixed lengths',
            'mask': [True]*5 + [False]*2 + [True]*2 + [False] + [True]*6 + [False]*3 + [True]*3 + [False] + [True] + [False]*2 + [True]*4,
            'min_length': 3,
            'expected': [[0, 5], [10, 16], [19, 22], [26, 30]]
        },
        {
            'name': 'No regions (all too short)',
            'mask': [True, False, True, False, True, False],
            'min_length': 2,
            'expected': []
        },
    ]

    available = list_available_finders()
    backends = [name for name, avail in available.items() if avail]

    print(f"{'Test':<30} ", end='')
    for backend in backends:
        print(f"{backend:<12}", end='')
    print()
    print("-" * (30 + 12 * len(backends)))

    for test in test_cases:
        mask = np.array(test['mask'])
        expected = np.array(test['expected'], dtype=np.int64) if test['expected'] else np.empty((0, 2), dtype=np.int64)

        print(f"{test['name']:<30} ", end='', flush=True)

        for backend in backends:
            try:
                result = find_regions(mask, test['min_length'], backend=backend)

                if arrays_equal(result, expected):
                    print(f"✓ {len(result):<10}", end='', flush=True)
                else:
                    print(f"✗ MISMATCH  ", end='', flush=True)

            except Exception as e:
                print(f"✗ ERROR     ", end='', flush=True)

        print()


def benchmark_single(finder_func, mask, min_length, n_warmup=3, n_runs=20):
    """Benchmark a single region finder."""
    # Warmup
    for _ in range(n_warmup):
        try:
            _ = finder_func(mask, min_length)
        except:
            pass

    # Benchmark
    times = []
    result = None
    for _ in range(n_runs):
        try:
            start = time.perf_counter()
            result = finder_func(mask, min_length)
            end = time.perf_counter()
            times.append(end - start)
        except Exception as e:
            return {'error': str(e)[:50]}

    if not times:
        return {'error': 'All runs failed'}

    times = np.array(times)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times),
        'n_regions': len(result),
    }


def benchmark_performance_single():
    """Benchmark single signal performance."""
    print_section("SINGLE SIGNAL PERFORMANCE")

    available = list_available_finders()
    backends = [name for name, avail in available.items() if avail]

    sizes = [100, 1000, 10000, 100000, 1000000]
    patterns = ['alternating', 'random', 'few_large']
    min_length = 5

    print("Testing with min_length=5, 20 runs per benchmark\n")

    for pattern in patterns:
        print(f"\nPattern: {pattern.upper()}")
        print(f"{'Size':<12} ", end='')
        for backend in backends:
            print(f"{backend:<15}", end='')
        print(f"{'Regions':<10}")
        print("-" * (12 + 15 * len(backends) + 10))

        for size in sizes:
            mask = create_test_mask(size, pattern)

            print(f"{size:<12} ", end='', flush=True)

            n_regions = None
            for backend in backends:
                try:
                    func = {
                        'numpy': find_regions_numpy,
                        'numba': find_regions_numba,
                        'torch': find_regions_torch,
                        'python': find_regions_python,
                        'scipy': find_regions_scipy,
                    }[backend]

                    stats = benchmark_single(func, mask, min_length, n_warmup=3, n_runs=20)

                    if 'error' in stats:
                        print(f"{'ERROR':<15}", end='', flush=True)
                    else:
                        mean_time_ms = stats['mean'] * 1000
                        print(f"{mean_time_ms:>13.3f}ms ", end='', flush=True)
                        if n_regions is None:
                            n_regions = stats['n_regions']

                except Exception as e:
                    print(f"{'ERROR':<15}", end='', flush=True)

            print(f"{n_regions:<10}" if n_regions is not None else "")


def benchmark_batched_performance():
    """Benchmark batched input performance (NEW)."""
    print_section("BATCHED PERFORMANCE (NEW)")

    available = list_available_finders()
    backends = [name for name, avail in available.items() if avail]

    # Test configurations: (batch_size, signal_length)
    configs = [
        (10, 1000, "Small batch"),
        (100, 1000, "Medium batch"),
        (1000, 100, "Large batch, short signals"),
        (100, 10000, "Medium batch, long signals"),
    ]

    pattern = 'random'
    min_length = 5

    print(f"Pattern: {pattern}, min_length={min_length}\n")
    print(f"{'Config':<35} ", end='')
    for backend in backends:
        print(f"{backend:<15}", end='')
    print()
    print("-" * (35 + 15 * len(backends)))

    for batch_size, signal_length, description in configs:
        # Create batched mask
        masks = np.random.rand(batch_size, signal_length) < 0.6

        config_str = f"{description} ({batch_size}×{signal_length})"
        print(f"{config_str:<35} ", end='', flush=True)

        for backend in backends:
            try:
                func = {
                    'numpy': find_regions_numpy,
                    'numba': find_regions_numba,
                    'torch': find_regions_torch,
                    'python': find_regions_python,
                    'scipy': find_regions_scipy,
                }[backend]

                # Warmup
                _ = func(masks, min_length)

                # Benchmark
                times = []
                for _ in range(10):
                    start = time.perf_counter()
                    result = func(masks, min_length)
                    end = time.perf_counter()
                    times.append(end - start)

                mean_time_ms = np.mean(times) * 1000
                print(f"{mean_time_ms:>13.3f}ms ", end='', flush=True)

            except Exception as e:
                print(f"{'ERROR':<15}", end='', flush=True)

        print()


def benchmark_batch_vs_sequential():
    """Compare batched processing vs sequential loop."""
    print_section("BATCHED vs SEQUENTIAL COMPARISON")

    batch_size = 100
    signal_length = 10000
    min_length = 5

    masks = np.random.rand(batch_size, signal_length) < 0.6

    print(f"Configuration: {batch_size} signals × {signal_length} elements")
    print(f"Pattern: random, min_length={min_length}\n")

    available = list_available_finders()
    backends = [name for name, avail in available.items() if avail]

    print(f"{'Backend':<15} {'Batched (ms)':<15} {'Sequential (ms)':<15} {'Speedup':<10}")
    print("-" * 55)

    for backend in backends:
        try:
            func = {
                'numpy': find_regions_numpy,
                'numba': find_regions_numba,
                'torch': find_regions_torch,
                'python': find_regions_python,
                'scipy': find_regions_scipy,
            }[backend]

            # Batched approach (NEW API)
            times_batched = []
            for _ in range(10):
                start = time.perf_counter()
                result = func(masks, min_length)
                end = time.perf_counter()
                times_batched.append(end - start)
            batched_time = np.mean(times_batched) * 1000

            # Sequential approach (manual loop)
            times_seq = []
            for _ in range(5):  # Fewer runs since this is slower
                start = time.perf_counter()
                results = []
                for i in range(batch_size):
                    r = func(masks[i], min_length)
                    results.append(r)
                end = time.perf_counter()
                times_seq.append(end - start)
            sequential_time = np.mean(times_seq) * 1000

            speedup = sequential_time / batched_time

            print(f"{backend:<15} {batched_time:>13.3f}  {sequential_time:>15.3f}  {speedup:>8.2f}x")

        except Exception as e:
            print(f"{backend:<15} {'ERROR':<15} {str(e)[:30]}")


def benchmark_scalability():
    """Test how performance scales with number of regions."""
    print_section("SCALABILITY ANALYSIS")

    available = list_available_finders()
    backends = [name for name, avail in available.items() if avail]

    total_size = 100000
    region_configs = [
        (10, "10 large regions (~10k each)"),
        (100, "100 medium regions (~1k each)"),
        (1000, "1000 small regions (~100 each)"),
        (10000, "10000 tiny regions (~10 each)"),
    ]

    print("Fixed total size: 100,000 elements\n")
    print(f"{'Configuration':<35} ", end='')
    for backend in backends:
        print(f"{backend:<15}", end='')
    print()
    print("-" * (35 + 15 * len(backends)))

    for n_regions_target, description in region_configs:
        mask = np.zeros(total_size, dtype=bool)

        total_gaps = n_regions_target - 1
        if total_gaps > 0:
            region_size = max(3, total_size // (2 * n_regions_target))
            gap_size = max(2, (total_size - n_regions_target * region_size) // total_gaps)
        else:
            region_size = total_size
            gap_size = 0

        pos = 0
        actual_regions = 0
        while pos < total_size and actual_regions < n_regions_target:
            end = min(pos + region_size, total_size)
            mask[pos:end] = True
            pos = end + gap_size
            actual_regions += 1

        print(f"{description:<35} ", end='', flush=True)

        min_length = 3
        for backend in backends:
            try:
                func = {
                    'numpy': find_regions_numpy,
                    'numba': find_regions_numba,
                    'torch': find_regions_torch,
                    'python': find_regions_python,
                    'scipy': find_regions_scipy,
                }[backend]

                stats = benchmark_single(func, mask, min_length, n_warmup=3, n_runs=10)

                if 'error' in stats:
                    print(f"{'ERROR':<15}", end='', flush=True)
                else:
                    mean_time_ms = stats['mean'] * 1000
                    print(f"{mean_time_ms:>13.3f}ms ", end='', flush=True)

            except:
                print(f"{'ERROR':<15}", end='', flush=True)

        print()


def benchmark_edge_cases():
    """Test edge cases."""
    print_section("EDGE CASES")

    available = list_available_finders()
    backends = [name for name, avail in available.items() if avail]

    edge_cases = [
        (np.array([]), 1, "Empty mask (should error)"),
        (np.array([True]), 1, "Single True"),
        (np.array([False]), 1, "Single False"),
        (np.array([True] * 100), 1, "All True"),
        (np.array([False] * 100), 1, "All False"),
        (np.array([True, False] * 50), 2, "Alternating (all too short)"),
        (np.array([True] * 50 + [False] * 50), 10, "Two halves"),
        (np.array([False] * 50 + [True] * 50), 10, "Second half only"),
    ]

    print(f"{'Test Case':<40} ", end='')
    for backend in backends:
        print(f"{backend:<12}", end='')
    print()
    print("-" * (40 + 12 * len(backends)))

    for mask, min_length, description in edge_cases:
        print(f"{description:<40} ", end='', flush=True)

        reference = None
        for backend in backends:
            try:
                result = find_regions(mask, min_length, backend=backend)

                if reference is None:
                    reference = result

                if arrays_equal(result, reference):
                    n_regions = len(result) if hasattr(result, '__len__') else 0
                    print(f"✓ {n_regions:<10}", end='', flush=True)
                else:
                    print(f"✗ MISMATCH  ", end='', flush=True)

            except Exception as e:
                if "empty" in description.lower() or "Empty" in str(e):
                    print(f"✓ ERROR     ", end='', flush=True)  # Expected error
                else:
                    print(f"✗ ERROR     ", end='', flush=True)

        print()


def main():
    """Run complete benchmark suite."""
    print("\n" + "="*80)
    print("REGION FINDING - COMPREHENSIVE BENCHMARK SUITE (UPDATED)".center(80))
    print("="*80)

    # Show available backends
    print("\nAvailable backends:")
    available = list_available_finders()
    for backend, is_available in available.items():
        status = "✓" if is_available else "✗"
        print(f"  {status} {backend}")

    # Run benchmarks
    correctness_passed = benchmark_correctness()

    if not correctness_passed:
        print("\n⚠ WARNING: Some correctness tests failed!")
        print("Performance benchmarks may not be meaningful.")
        response = input("\nContinue with performance benchmarks? (y/n): ")
        if response.lower() != 'y':
            return

    benchmark_multi_region_explicit()
    benchmark_edge_cases()
    benchmark_performance_single()
    benchmark_batched_performance()
    benchmark_batch_vs_sequential()
    benchmark_scalability()

    print("\n" + "="*80)
    print("BENCHMARK SUITE COMPLETE".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()