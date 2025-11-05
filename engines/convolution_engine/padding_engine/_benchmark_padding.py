"""
Comprehensive Padding Benchmarking

Benchmarks all padding implementations across:
- All modes: reflect, symmetric, edge, constant, wrap
- All backends: numpy_native, pytorch_cpu, pytorch_gpu, numba_pure, numba_numpy, custom_vectorized
- Various data sizes: 100, 1k, 10k, 100k, 1M
- Various pad widths: 5, 10, 20, 50
- Both 1D and 2D (batched) data

Outputs performance matrix for empirical analysis.
"""

import numpy as np
import time
import sys
from typing import Dict, List, Tuple

from engines.convolution_engine.padding_engine.padding_implementations import (
    PADDING_IMPLEMENTATIONS,
    list_available_implementations
)


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")


def check_backend_available(backend):
    """Check if a backend is available on this system."""
    if backend in ['pytorch_cpu', 'pytorch_gpu']:
        try:
            import torch
            if backend == 'pytorch_gpu' and not torch.cuda.is_available():
                return False
            return True
        except ImportError:
            return False
    elif backend in ['numba_pure', 'numba_numpy']:
        try:
            import numba
            return True
        except ImportError:
            return False
    else:
        # numpy_native, custom_vectorized always available
        return True


def verify_correctness(mode, backends, data_size=100, pad_width=10):
    """
    Verify that all backends produce same results for a given mode.
    
    Args:
        mode: Padding mode
        backends: List of backends to test
        data_size: Size of test data
        pad_width: Padding width
    
    Returns:
        Dict of backend: (max_error, passes) for each backend
    """
    # Create test data
    np.random.seed(42)
    data = np.random.randn(data_size).astype(np.float32)
    
    # Get reference result (numpy_native)
    reference_backend = 'numpy_native'
    reference_func = PADDING_IMPLEMENTATIONS[(mode, reference_backend)]
    reference_result = reference_func(data, pad_width, pad_width, constant_value=0)
    
    results = {}
    for backend in backends:
        if not check_backend_available(backend):
            results[backend] = ('N/A', False)
            continue
        
        try:
            func = PADDING_IMPLEMENTATIONS[(mode, backend)]
            result = func(data, pad_width, pad_width, constant_value=0)
            
            # Convert to numpy if tensor
            if hasattr(result, 'numpy'):
                result = result.numpy()
            
            # Compare
            max_error = np.max(np.abs(result - reference_result))
            passes = max_error < 1e-5
            
            results[backend] = (max_error, passes)
        except Exception as e:
            results[backend] = (f'Error: {str(e)}', False)
    
    return results


def benchmark_single_implementation(func, data, pad_left, pad_right, 
                                   n_warmup=3, n_runs=20):
    """
    Benchmark a single padding implementation.
    
    Args:
        func: Padding function
        data: Input data
        pad_left: Left padding width
        pad_right: Right padding width
        n_warmup: Number of warmup runs
        n_runs: Number of benchmark runs
    
    Returns:
        Dict with timing statistics
    """
    # Warmup
    for _ in range(n_warmup):
        try:
            _ = func(data, pad_left, pad_right, constant_value=0)
        except:
            pass
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        try:
            start = time.perf_counter()
            _ = func(data, pad_left, pad_right, constant_value=0)
            end = time.perf_counter()
            times.append(end - start)
        except Exception as e:
            return {'error': str(e)}
    
    if not times:
        return {'error': 'All runs failed'}
    
    times = np.array(times)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times),
    }


def benchmark_mode_comprehensive(mode, backends, data_sizes, pad_widths, 
                                n_warmup=3, n_runs=20):
    """
    Comprehensive benchmark for a single mode across all parameters.
    
    Args:
        mode: Padding mode
        backends: List of backends to test
        data_sizes: List of data sizes to test
        pad_widths: List of pad widths to test
        n_warmup: Warmup runs
        n_runs: Benchmark runs
    
    Returns:
        Dict of results
    """
    print_section(f"Benchmarking Mode: {mode.upper()}")
    
    results = {}
    
    for data_size in data_sizes:
        print(f"\nData size: {data_size}")
        print(f"{'Backend':<20} ", end='')
        for pad_width in pad_widths:
            print(f"{'pad=' + str(pad_width):<15}", end='')
        print()
        print("-" * (20 + 15 * len(pad_widths)))
        
        # Create test data
        data = np.random.randn(data_size).astype(np.float32)
        
        for backend in backends:
            if not check_backend_available(backend):
                print(f"{backend:<20} {'N/A (not available)'}")
                continue
            
            print(f"{backend:<20} ", end='', flush=True)
            
            for pad_width in pad_widths:
                try:
                    func = PADDING_IMPLEMENTATIONS[(mode, backend)]
                    stats = benchmark_single_implementation(
                        func, data, pad_width, pad_width, n_warmup, n_runs
                    )
                    
                    if 'error' in stats:
                        print(f"{'Error':<15}", end='', flush=True)
                        result_key = (mode, backend, data_size, pad_width)
                        results[result_key] = None
                    else:
                        mean_time_ms = stats['mean'] * 1000
                        print(f"{mean_time_ms:>13.3f}ms ", end='', flush=True)
                        result_key = (mode, backend, data_size, pad_width)
                        results[result_key] = stats['mean']
                
                except Exception as e:
                    print(f"{'Error':<15}", end='', flush=True)
                    result_key = (mode, backend, data_size, pad_width)
                    results[result_key] = None
            
            print()  # New line after backend
    
    return results


def benchmark_batched(mode, backends, batch_sizes, array_size, pad_width,
                     n_warmup=3, n_runs=20):
    """
    Benchmark batched (2D) padding operations.
    
    Args:
        mode: Padding mode
        backends: List of backends
        batch_sizes: List of batch sizes
        array_size: Size of each array in batch
        pad_width: Padding width
        n_warmup: Warmup runs
        n_runs: Benchmark runs
    
    Returns:
        Dict of results
    """
    print_section(f"Batched Benchmarking - Mode: {mode.upper()}")
    print(f"Array size: {array_size}, Pad width: {pad_width}\n")
    
    print(f"{'Backend':<20} ", end='')
    for batch_size in batch_sizes:
        print(f"{'batch=' + str(batch_size):<15}", end='')
    print()
    print("-" * (20 + 15 * len(batch_sizes)))
    
    results = {}
    
    for backend in backends:
        if not check_backend_available(backend):
            print(f"{backend:<20} {'N/A (not available)'}")
            continue
        
        print(f"{backend:<20} ", end='', flush=True)
        
        for batch_size in batch_sizes:
            # Create batched data
            data = np.random.randn(batch_size, array_size).astype(np.float32)
            
            try:
                func = PADDING_IMPLEMENTATIONS[(mode, backend)]

                def batch_wrapper(data_batch, pad_l, pad_r, constant_value=0):
                    results_batch = []
                    for row in data_batch:
                        result_row = func(row, pad_l, pad_r, constant_value)
                        results_batch.append(result_row)

                    # Handle both numpy arrays and torch tensors
                    if hasattr(results_batch[0], '__torch_function__'):
                        import torch
                        return torch.stack(results_batch)
                    else:
                        return np.array(results_batch)
                
                stats = benchmark_single_implementation(
                    batch_wrapper, data, pad_width, pad_width, n_warmup, n_runs
                )
                
                if 'error' in stats:
                    print(f"{'Error':<15}", end='', flush=True)
                    result_key = (mode, backend, batch_size)
                    results[result_key] = None
                else:
                    mean_time_ms = stats['mean'] * 1000
                    print(f"{mean_time_ms:>13.3f}ms ", end='', flush=True)
                    result_key = (mode, backend, batch_size)
                    results[result_key] = stats['mean']
            
            except Exception as e:
                print(f"{'Error':<15}", end='', flush=True)
                result_key = (mode, backend, batch_size)
                results[result_key] = None
        
        print()
    
    return results


def run_correctness_tests():
    """Run correctness verification for all modes and backends."""
    print_section("CORRECTNESS VERIFICATION")
    
    modes = ['reflect', 'symmetric', 'edge', 'constant', 'wrap']
    backends = ['numpy_native', 'pytorch_cpu', 'numba_pure', 'numba_numpy', 
                'custom_vectorized', 'pytorch_gpu']
    
    print(f"Testing all backends against numpy_native reference...\n")
    
    all_pass = True
    
    for mode in modes:
        print(f"\nMode: {mode}")
        print(f"{'Backend':<20} {'Max Error':<15} {'Status':<10}")
        print("-" * 45)
        
        results = verify_correctness(mode, backends)
        
        for backend, (error, passes) in results.items():
            if backend == 'numpy_native':
                continue
            
            if error == 'N/A':
                status = 'N/A'
                error_str = 'N/A'
            elif isinstance(error, str):
                status = 'ERROR'
                error_str = error[:10]
                all_pass = False
            else:
                status = '✓ PASS' if passes else '✗ FAIL'
                error_str = f"{error:.2e}"
                if not passes:
                    all_pass = False
            
            print(f"{backend:<20} {error_str:<15} {status:<10}")
    
    print(f"\n{'='*45}")
    if all_pass:
        print("✓ All correctness tests passed!")
    else:
        print("✗ Some tests failed - check results above")
    print(f"{'='*45}")
    
    return all_pass


def run_comprehensive_benchmarks():
    """Run comprehensive benchmarking suite."""
    print_section("COMPREHENSIVE PADDING BENCHMARKS")
    
    modes = ['reflect', 'symmetric', 'edge', 'constant', 'wrap']
    backends = ['numpy_native', 'pytorch_cpu', 'pytorch_gpu', 'numba_pure', 
                'numba_numpy', 'custom_vectorized']
    
    # 1D benchmarks
    data_sizes = [100, 1000, 10000, 100000, 1000000]
    pad_widths = [5, 10, 20, 50]
    
    all_results = {}
    
    # Benchmark each mode
    for mode in modes:
        mode_results = benchmark_mode_comprehensive(
            mode, backends, data_sizes, pad_widths, n_warmup=3, n_runs=20
        )
        all_results.update(mode_results)
    
    # Batched benchmarks for reflect mode
    print_section("BATCHED (2D) BENCHMARKING")
    batch_sizes = [10, 50, 100, 500]
    batched_results = benchmark_batched(
        'reflect', backends, batch_sizes, array_size=10000, pad_width=10,
        n_warmup=3, n_runs=20
    )
    all_results.update(batched_results)
    
    return all_results


def analyze_results(results):
    """Analyze and summarize benchmark results."""
    print_section("PERFORMANCE ANALYSIS")
    
    # Find fastest backend for each scenario
    print("\nFastest Backend by Scenario:")
    print(f"{'Scenario':<50} {'Backend':<20} {'Time (ms)':<12}")
    print("-" * 82)
    
    # Separate 1D and batched results
    scenarios_1d = {}
    scenarios_batched = {}

    for key, time_val in results.items():
        if time_val is None:
            continue

        # Check if it's a 3-element key (batched) or 4-element key (1D)
        if len(key) == 3:
            # Batched: (mode, backend, batch_size)
            mode, backend, batch_size = key
            if not isinstance(batch_size, int):
                continue
            scenario_key = (mode, batch_size)
            if scenario_key not in scenarios_batched:
                scenarios_batched[scenario_key] = {}
            scenarios_batched[scenario_key][backend] = time_val
        elif len(key) == 4:
            # 1D: (mode, backend, data_size, pad_width)
            mode, backend, data_size, pad_width = key
            if not isinstance(data_size, int):
                continue
            scenario_key = (mode, data_size, pad_width)
            if scenario_key not in scenarios_1d:
                scenarios_1d[scenario_key] = {}
            scenarios_1d[scenario_key][backend] = time_val

    # Print 1D results
    print("\n1D Scenarios:")
    for (mode, data_size, pad_width), backend_times in sorted(scenarios_1d.items()):
        if not backend_times:
            continue

        best_backend = min(backend_times, key=backend_times.get)
        best_time = backend_times[best_backend] * 1000

        scenario_name = f"{mode}, size={data_size}, pad={pad_width}"
        print(f"{scenario_name:<50} {best_backend:<20} {best_time:>10.3f}")

    # Print batched results
    if scenarios_batched:
        print("\nBatched (2D) Scenarios:")
        for (mode, batch_size), backend_times in sorted(scenarios_batched.items()):
            if not backend_times:
                continue

            best_backend = min(backend_times, key=backend_times.get)
            best_time = backend_times[best_backend] * 1000

            scenario_name = f"{mode}, batch={batch_size}"
            print(f"{scenario_name:<50} {best_backend:<20} {best_time:>10.3f}")

    # Speedup analysis (only for 1D results)
    print_section("SPEEDUP ANALYSIS (vs numpy_native) - 1D Data")

    for mode in ['reflect', 'symmetric', 'edge', 'constant', 'wrap']:
        print(f"\nMode: {mode}")
        print(f"{'Backend':<20} ", end='')

        test_sizes = [1000, 10000, 100000]
        for size in test_sizes:
            print(f"{'size=' + str(size):<15}", end='')
        print()
        print("-" * (20 + 15 * len(test_sizes)))

        backends = ['pytorch_cpu', 'pytorch_gpu', 'numba_pure', 'numba_numpy', 'custom_vectorized']
        pad_width = 10

        for backend in backends:
            print(f"{backend:<20} ", end='')

            for size in test_sizes:
                numpy_key = (mode, 'numpy_native', size, pad_width)
                backend_key = (mode, backend, size, pad_width)

                if numpy_key in results and backend_key in results:
                    numpy_time = results[numpy_key]
                    backend_time = results[backend_key]

                    if numpy_time and backend_time:
                        speedup = numpy_time / backend_time
                        print(f"{speedup:>13.2f}x ", end='')
                    else:
                        print(f"{'N/A':<15}", end='')
                else:
                    print(f"{'N/A':<15}", end='')

            print()

    # Batched speedup analysis (if we have batched results)
    if scenarios_batched:
        print_section("SPEEDUP ANALYSIS (vs numpy_native) - Batched (2D) Data")

        # Get the mode used in batched tests (typically 'reflect')
        batched_modes = set(mode for (mode, _) in scenarios_batched.keys())

        for mode in batched_modes:
            print(f"\nMode: {mode}")
            print(f"{'Backend':<20} ", end='')

            # Get batch sizes
            batch_sizes = sorted(set(batch_size for (m, batch_size) in scenarios_batched.keys() if m == mode))
            for batch_size in batch_sizes:
                print(f"{'batch=' + str(batch_size):<15}", end='')
            print()
            print("-" * (20 + 15 * len(batch_sizes)))

            backends = ['pytorch_cpu', 'pytorch_gpu', 'numba_pure', 'numba_numpy', 'custom_vectorized']

            for backend in backends:
                print(f"{backend:<20} ", end='')

                for batch_size in batch_sizes:
                    numpy_key = (mode, 'numpy_native', batch_size)
                    backend_key = (mode, backend, batch_size)

                    if numpy_key in results and backend_key in results:
                        numpy_time = results[numpy_key]
                        backend_time = results[backend_key]

                        if numpy_time and backend_time:
                            speedup = numpy_time / backend_time
                            print(f"{speedup:>13.2f}x ", end='')
                        else:
                            print(f"{'N/A':<15}", end='')
                    else:
                        print(f"{'N/A':<15}", end='')

                print()


def main():
    """Run complete benchmark suite."""
    print("\n" + "="*80)
    print("PADDING IMPLEMENTATIONS - COMPREHENSIVE BENCHMARK SUITE".center(80))
    print("="*80)

    # Check available implementations
    implementations = list_available_implementations()
    print(f"\nTotal implementations available: {len(implementations)}")
    print(f"Modes: 5 (reflect, symmetric, edge, constant, wrap)")
    print(f"Backends: 6 (numpy_native, pytorch_cpu, pytorch_gpu, numba_pure, numba_numpy, custom_vectorized)")

    # Check which backends are available
    print("\nBackend Availability:")
    backends = ['numpy_native', 'pytorch_cpu', 'pytorch_gpu', 'numba_pure',
                'numba_numpy', 'custom_vectorized']
    for backend in backends:
        available = check_backend_available(backend)
        status = "✓" if available else "✗"
        print(f"  {status} {backend}")

    # Run correctness tests
    print("\nRunning correctness tests...")
    correctness_pass = run_correctness_tests()

    if not correctness_pass:
        print("\n⚠ WARNING: Some correctness tests failed!")

    # Run benchmarks
    print("\nRunning benchmarks (this may take some time)...")
    results = run_comprehensive_benchmarks()

    # Analyze results
    analyze_results(results)

    print("\n" + "="*80)
    print("BENCHMARK SUITE COMPLETE".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()