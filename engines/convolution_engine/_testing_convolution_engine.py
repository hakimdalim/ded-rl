"""
Convolution Engine Demo & Test Script

Demonstrates usage and validates all backends.
"""
import time

import numpy as np
import sys

from engines.convolution_engine.engine import ConvolutionEngine


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def demo_basic_usage():
    """Demonstrate basic usage of ConvolutionEngine"""
    print_section("Basic Usage Demo")
    
    # Create simple test data
    data_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data_2d = np.random.randn(3, 10)
    kernel = np.array([0.25, 0.5, 0.25])
    
    print("Input data (1D):", data_1d)
    print("Kernel:", kernel)
    
    # Try each backend
    for backend in ['numpy', 'scipy', 'numba']:
        try:
            engine = ConvolutionEngine(backend=backend)
            result = engine.convolve(data_1d, kernel)
            print(f"\n{backend}: {result}")
        except Exception as e:
            print(f"\n{backend}: Error - {e}")


def demo_backend_availability():
    """Show which backends are available"""
    print_section("Backend Availability")
    
    available = ConvolutionEngine.list_available_backends()
    
    print(f"{'Backend':<20} {'Available':<10}")
    print("-" * 30)
    for name, is_available in available.items():
        status = "✓ Yes" if is_available else "✗ No"
        print(f"{name:<20} {status:<10}")


def demo_backend_info():
    """Show detailed information about each backend"""
    print_section("Backend Details")
    
    for backend in ['numpy', 'scipy', 'numba', 'pytorch_cpu', 'pytorch_gpu']:
        try:
            info = ConvolutionEngine.get_backend_info(backend)
            print(f"\n{backend}:")
            print(f"  Available: {info['available']}")
            print(f"  Supported dimensions: {info['supported_dims']}")
        except Exception as e:
            print(f"\n{backend}: Error - {e}")


def test_dimensionality_support():
    """Test different input dimensions for each backend"""
    print_section("Dimensionality Support Test")
    
    kernel = np.array([0.25, 0.5, 0.25])
    
    test_cases = {
        '1D': np.random.randn(100),
        '2D': np.random.randn(10, 100),
        '3D': np.random.randn(5, 10, 100),
    }
    
    backends = ['numpy', 'scipy', 'numba', 'pytorch_cpu']
    
    print(f"{'Backend':<15} {'1D':<10} {'2D':<10} {'3D':<10}")
    print("-" * 45)
    
    for backend in backends:
        try:
            engine = ConvolutionEngine(backend=backend)
            results = []
            
            for dim_name, data in test_cases.items():
                try:
                    result = engine.convolve(data, kernel)
                    results.append("✓")
                except ValueError:
                    results.append("✗")
                except Exception as e:
                    results.append(f"E")
            
            print(f"{backend:<15} {results[0]:<10} {results[1]:<10} {results[2]:<10}")
        except Exception as e:
            print(f"{backend:<15} Error: {e}")


def test_numerical_consistency():
    """Test that all backends give consistent results"""
    print_section("Numerical Consistency Test")
    
    # Generate test data
    np.random.seed(42)
    data_1d = np.random.randn(100)
    kernel = np.array([0.25, 0.5, 0.25])
    
    print("Testing 1D convolution consistency...")
    print(f"Data shape: {data_1d.shape}, Kernel shape: {kernel.shape}")
    
    results = {}
    
    # Run on all 1D-supporting backends
    for backend in ['numpy', 'scipy', 'numba', 'pytorch_cpu']:
        try:
            engine = ConvolutionEngine(backend=backend)
            result = engine.convolve(data_1d, kernel)
            results[backend] = result
            print(f"\n{backend}: Output shape = {result.shape}")
        except Exception as e:
            print(f"\n{backend}: Error - {e}")
    
    # Compare results
    if len(results) >= 2:
        print("\nComparing results (max absolute difference):")
        backends_list = list(results.keys())
        reference = results[backends_list[0]]
        
        for i in range(1, len(backends_list)):
            backend = backends_list[i]
            diff = np.max(np.abs(results[backend] - reference))
            print(f"  {backends_list[0]} vs {backend}: {diff:.2e}")


def test_batch_processing():
    """Test batch processing capabilities"""
    print_section("Batch Processing Test")
    
    batch_sizes = [1, 10, 100]
    size = 1000
    kernel = np.array([0.25, 0.5, 0.25])
    
    print(f"Testing with array size = {size}, kernel size = {len(kernel)}")
    print(f"\n{'Backend':<15} {'Batch=1':<15} {'Batch=10':<15} {'Batch=100':<15}")
    print("-" * 60)
    
    for backend in ['numba', 'pytorch_cpu']:
        try:
            engine = ConvolutionEngine(backend=backend)
            results = []
            
            for batch in batch_sizes:
                data = np.random.randn(batch, size)
                try:
                    result = engine.convolve(data, kernel)
                    results.append(f"{result.shape}")
                except Exception as e:
                    results.append("Error")
            
            print(f"{backend:<15} {results[0]:<15} {results[1]:<15} {results[2]:<15}")
        except Exception as e:
            print(f"{backend:<15} Initialization error")


def test_error_handling():
    """Test that proper errors are raised for invalid inputs"""
    print_section("Error Handling Test")
    
    print("Testing error conditions...\n")
    
    # Test 1: Invalid backend
    print("1. Invalid backend name:")
    try:
        engine = ConvolutionEngine(backend='invalid')
        print("   ✗ No error raised!")
    except ValueError as e:
        print(f"   ✓ Caught: {e}")
    
    # Test 2: 2D kernel
    print("\n2. Non-1D kernel:")
    try:
        engine = ConvolutionEngine(backend='numpy')
        data = np.random.randn(10)
        kernel = np.random.randn(3, 3)
        result = engine.convolve(data, kernel)
        print("   ✗ No error raised!")
    except ValueError as e:
        print(f"   ✓ Caught: {e}")
    
    # Test 3: Unsupported dimensionality
    print("\n3. Unsupported dimensionality (3D data with numpy backend):")
    try:
        engine = ConvolutionEngine(backend='numpy')
        data = np.random.randn(5, 10, 100)
        kernel = np.random.randn(3)
        result = engine.convolve(data, kernel)
        print("   ✗ No error raised!")
    except ValueError as e:
        print(f"   ✓ Caught: {e}")
    
    # Test 4: GPU backend without CUDA
    print("\n4. GPU backend without CUDA (if CUDA unavailable):")
    try:
        engine = ConvolutionEngine(backend='pytorch_gpu')
        print("   ✓ GPU is available")
    except ValueError as e:
        print(f"   ✓ Caught: {e}")


def test_pytorch_tensor_support():
    """Test that PyTorch backend works with torch tensors"""
    print_section("PyTorch Tensor Support Test")

    try:
        import torch
    except ImportError:
        print("PyTorch not available, skipping tensor tests")
        return

    # Test data
    np.random.seed(42)
    data_np = np.random.randn(10, 100).astype(np.float32)
    kernel_np = np.array([0.25, 0.5, 0.25], dtype=np.float32)

    # Convert to tensors
    data_tensor = torch.from_numpy(data_np)
    kernel_tensor = torch.from_numpy(kernel_np)

    print("Testing PyTorch backend with different input types...")
    print(f"Data shape: {data_np.shape}, Kernel shape: {kernel_np.shape}")

    for backend in ['pytorch_cpu', 'pytorch_gpu']:
        try:
            engine = ConvolutionEngine(backend=backend)
            print(f"\n{backend}:")

            # Test 1: numpy array input
            result_np = engine.convolve(data_np, kernel_np)
            print(f"  numpy → numpy: {type(result_np).__name__}, shape {result_np.shape}")

            # Test 2: torch tensor input
            result_tensor = engine.convolve(data_tensor, kernel_tensor)
            print(f"  tensor → tensor: {type(result_tensor).__name__}, shape {result_tensor.shape}")

            # Test 3: mixed input (numpy data, tensor kernel)
            result_mixed1 = engine.convolve(data_np, kernel_tensor)
            print(f"  numpy data + tensor kernel → {type(result_mixed1).__name__}")

            # Test 4: mixed input (tensor data, numpy kernel)
            result_mixed2 = engine.convolve(data_tensor, kernel_np)
            print(f"  tensor data + numpy kernel → {type(result_mixed2).__name__}")

            # Verify numerical consistency
            result_tensor_np = result_tensor.cpu().numpy() if isinstance(result_tensor, torch.Tensor) else result_tensor
            max_diff = np.max(np.abs(result_np - result_tensor_np))
            print(f"  Max difference (numpy vs tensor): {max_diff:.2e}")

            if max_diff < 1e-5:
                print(f"  ✓ Numerical consistency verified")
            else:
                print(f"  ✗ Numerical inconsistency detected!")

        except ValueError as e:
            print(f"\n{backend}: {e}")
        except Exception as e:
            print(f"\n{backend}: Unexpected error - {e}")


def test_pytorch_tensor_gpu():
    """Test PyTorch tensors on GPU if available"""
    print_section("PyTorch GPU Tensor Test")

    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available, skipping GPU tensor tests")
            return
    except ImportError:
        print("PyTorch not available, skipping GPU tensor tests")
        return

    print("Testing GPU tensor operations...")

    # Create data on GPU
    data_gpu = torch.randn(50, 1000, device='cuda')
    kernel_gpu = torch.tensor([0.25, 0.5, 0.25], device='cuda')

    print(f"Data device: {data_gpu.device}, shape: {data_gpu.shape}")
    print(f"Kernel device: {kernel_gpu.device}, shape: {kernel_gpu.shape}")

    try:
        engine = ConvolutionEngine(backend='pytorch_gpu')

        # Convolve on GPU
        result_gpu = engine.convolve(data_gpu, kernel_gpu)

        print(f"\nResult device: {result_gpu.device}, shape: {result_gpu.shape}")
        print(f"Result type: {type(result_gpu).__name__}")
        print("✓ GPU tensor convolution successful")

        # Compare with CPU version
        data_cpu = data_gpu.cpu()
        kernel_cpu = kernel_gpu.cpu()

        engine_cpu = ConvolutionEngine(backend='pytorch_cpu')
        result_cpu = engine_cpu.convolve(data_cpu, kernel_cpu)

        max_diff = torch.max(torch.abs(result_gpu.cpu() - result_cpu)).item()
        print(f"\nMax difference (GPU vs CPU): {max_diff:.2e}")

        if max_diff < 1e-5:
            print("✓ GPU and CPU results match")
        else:
            print("✗ GPU and CPU results don't match!")

    except Exception as e:
        print(f"Error: {e}")


def test_pytorch_tensor_types():
    """Test different PyTorch tensor dtypes"""
    print_section("PyTorch Tensor Dtype Test")

    try:
        import torch
    except ImportError:
        print("PyTorch not available, skipping dtype tests")
        return

    try:
        engine = ConvolutionEngine(backend='pytorch_cpu')

        dtypes = [torch.float32, torch.float64]
        kernel = torch.tensor([0.25, 0.5, 0.25])

        print(f"{'Dtype':<15} {'Input Shape':<15} {'Output Shape':<15} {'Status':<10}")
        print("-" * 55)

        for dtype in dtypes:
            data = torch.randn(10, 100, dtype=dtype)
            kernel_typed = kernel.to(dtype)

            try:
                result = engine.convolve(data, kernel_typed)
                print(f"{str(dtype):<15} {str(data.shape):<15} {str(result.shape):<15} {'✓':<10}")
            except Exception as e:
                print(f"{str(dtype):<15} {str(data.shape):<15} {'N/A':<15} {'✗ Error':<10}")

    except ValueError as e:
        print(f"Backend not available: {e}")


def test_kernel_size_effects():
    """Test how different kernel sizes affect output shape"""
    print_section("Kernel Size Effect Test")

    input_size = 100
    data = np.random.randn(input_size)

    sigmas = [0.5, 1.0, 2.0, 5.0]

    print(f"Input size: {input_size}")
    print(f"\n{'Sigma':<10} {'Kernel Size':<15} {'Output Size':<15} {'Formula Check':<15}")
    print("-" * 55)

    engine = ConvolutionEngine(backend='numba')

    for sigma in sigmas:
        # Create Gaussian kernel
        truncate = 4.0
        radius = int(truncate * sigma + 0.5)
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-(x * x) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()

        kernel_size = len(kernel)
        expected_output = input_size - kernel_size + 1

        result = engine.convolve(data, kernel)
        actual_output = len(result)

        check = "✓" if actual_output == expected_output else "✗"
        print(f"{sigma:<10.1f} {kernel_size:<15} {actual_output:<15} {check} ({expected_output})")


def test_performance_comprehensive():
    """Comprehensive performance benchmarking across all backends"""
    print_section("Comprehensive Performance Benchmarks")

    # Test configurations
    batch_sizes = [1, 10, 50, 100, 500]
    array_sizes = [100, 500, 1000, 5000, 10000]
    sigmas = [1.0, 2.0, 5.0]
    n_warmup = 3
    n_runs = 20

    # Backends to test
    backends_to_test = []
    available = ConvolutionEngine.list_available_backends()
    for backend in ['numpy', 'scipy', 'numba', 'pytorch_cpu', 'pytorch_gpu']:
        if available[backend]:
            backends_to_test.append(backend)

    print(f"Testing backends: {backends_to_test}")
    print(f"Configurations: {len(batch_sizes)} batch sizes * {len(array_sizes)} array sizes * {len(sigmas)} sigmas")
    print(f"Total tests: {len(backends_to_test) * len(batch_sizes) * len(array_sizes) * len(sigmas)}")
    print(f"Warmup runs: {n_warmup}, Benchmark runs: {n_runs}\n")

    results = {}

    for sigma in sigmas:
        # Create Gaussian kernel
        truncate = 4.0
        radius = int(truncate * sigma + 0.5)
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-(x * x) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        kernel = kernel.astype(np.float32)
        kernel_size = len(kernel)

        print(f"\n{'=' * 70}")
        print(f"Sigma={sigma:.1f}, Kernel size={kernel_size}")
        print(f"{'=' * 70}")

        for array_size in array_sizes:
            print(f"\nArray size: {array_size}")
            print(f"{'Backend':<15} ", end='')
            for batch in batch_sizes:
                print(f"{'B=' + str(batch):<12}", end='')
            print()
            print("-" * (15 + 12 * len(batch_sizes)))

            for backend in backends_to_test:
                try:
                    engine = ConvolutionEngine(backend=backend)
                    supported_dims = engine.supported_dims

                    row_results = []
                    print(f"{backend:<15} ", end='', flush=True)

                    for batch_size in batch_sizes:
                        # Determine if this backend supports batching
                        supports_batching = supported_dims is None or 2 in supported_dims

                        # Create appropriate data shape
                        if supports_batching:
                            # Backend supports batching natively
                            if batch_size == 1:
                                data = np.random.randn(1, array_size).astype(np.float32)
                            else:
                                data = np.random.randn(batch_size, array_size).astype(np.float32)

                            # Warmup
                            for _ in range(n_warmup):
                                try:
                                    _ = engine.convolve(data, kernel)
                                except:
                                    break

                            # Benchmark
                            times = []
                            try:
                                for _ in range(n_runs):
                                    start = time.perf_counter()
                                    _ = engine.convolve(data, kernel)
                                    end = time.perf_counter()
                                    times.append(end - start)

                                mean_time = np.mean(times) * 1000  # ms
                                print(f"{mean_time:>10.2f}ms ", end='', flush=True)

                                # Store result
                                key = (sigma, array_size, batch_size, backend)
                                results[key] = mean_time
                                row_results.append(mean_time)
                            except Exception as e:
                                print(f"{'Error':<12}", end='', flush=True)
                                row_results.append(None)
                        else:
                            # Backend doesn't support batching - iterate manually
                            data = np.random.randn(array_size).astype(np.float32)

                            # Warmup
                            for _ in range(n_warmup):
                                try:
                                    _ = engine.convolve(data, kernel)
                                except:
                                    break

                            # Benchmark by calling multiple times
                            times = []
                            try:
                                for _ in range(n_runs):
                                    start = time.perf_counter()
                                    for _ in range(batch_size):
                                        _ = engine.convolve(data, kernel)
                                    end = time.perf_counter()
                                    times.append(end - start)

                                mean_time = np.mean(times) * 1000  # ms
                                print(f"{mean_time:>10.2f}ms ", end='', flush=True)

                                # Store result
                                key = (sigma, array_size, batch_size, backend)
                                results[key] = mean_time
                                row_results.append(mean_time)
                            except Exception as e:
                                print(f"{'Error':<12}", end='', flush=True)
                                row_results.append(None)

                    print()  # New line after each backend

                except Exception as e:
                    print(f"{backend:<15} Initialization failed: {e}")

    # Summary analysis
    print(f"\n{'=' * 70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 70}\n")

    # Find best backend for different scenarios
    print("Best backend by scenario:")
    print(f"{'Scenario':<40} {'Best Backend':<15} {'Time (ms)':<12}")
    print("-" * 67)

    scenarios = [
        ("Single vector (size=1000)", 1.0, 1000, 1),
        ("Small batch (size=1000, batch=10)", 1.0, 1000, 10),
        ("Medium batch (size=5000, batch=50)", 1.0, 5000, 50),
        ("Large batch (size=10000, batch=100)", 1.0, 10000, 100),
        ("Very large batch (size=10000, batch=500)", 1.0, 10000, 500),
    ]

    for scenario_name, sigma, array_size, batch_size in scenarios:
        best_backend = None
        best_time = float('inf')

        for backend in backends_to_test:
            key = (sigma, array_size, batch_size, backend)
            if key in results and results[key] is not None:
                if results[key] < best_time:
                    best_time = results[key]
                    best_backend = backend

        if best_backend:
            print(f"{scenario_name:<40} {best_backend:<15} {best_time:>10.2f}")

    # Speedup analysis
    print(f"\n{'=' * 70}")
    print("SPEEDUP ANALYSIS (relative to numpy baseline)")
    print(f"{'=' * 70}\n")

    for sigma in sigmas:
        truncate = 4.0
        radius = int(truncate * sigma + 0.5)
        kernel_size = 2 * radius + 1

        print(f"\nSigma={sigma:.1f} (kernel size={kernel_size}):")
        print(f"{'Array Size':<12} {'Batch':<8} ", end='')
        for backend in backends_to_test:
            if backend != 'numpy':
                print(f"{backend:<12}", end='')
        print()
        print("-" * (20 + 12 * (len(backends_to_test) - 1)))

        for array_size in [1000, 5000, 10000]:
            for batch_size in [10, 100, 500]:
                numpy_key = (sigma, array_size, batch_size, 'numpy')

                # Skip if numpy doesn't have this result
                if numpy_key not in results or results[numpy_key] is None:
                    continue

                numpy_time = results[numpy_key]
                print(f"{array_size:<12} {batch_size:<8} ", end='')

                for backend in backends_to_test:
                    if backend == 'numpy':
                        continue

                    key = (sigma, array_size, batch_size, backend)
                    if key in results and results[key] is not None:
                        speedup = numpy_time / results[key]
                        print(f"{speedup:>10.2f}x ", end='')
                    else:
                        print(f"{'N/A':<12}", end='')

                print()


def test_performance_scaling():
    """Test how performance scales with batch size and array size"""
    print_section("Performance Scaling Analysis")

    sigma = 2.0
    truncate = 4.0
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel = kernel.astype(np.float32)

    n_runs = 10

    # Test 1: Scaling with batch size (fixed array size)
    print("\n1. Batch Size Scaling (array_size=5000, sigma=2.0)")
    print(f"{'Batch Size':<15} ", end='')

    backends = ['numba', 'pytorch_cpu']
    available = ConvolutionEngine.list_available_backends()
    if available.get('pytorch_gpu', False):
        backends.append('pytorch_gpu')

    for backend in backends:
        print(f"{backend:<15}", end='')
    print()
    print("-" * (15 + 15 * len(backends)))

    batch_sizes = [1, 5, 10, 20, 50, 100, 200, 500]
    array_size = 5000

    for batch_size in batch_sizes:
        print(f"{batch_size:<15} ", end='', flush=True)

        for backend in backends:
            try:
                engine = ConvolutionEngine(backend=backend)
                data = np.random.randn(batch_size, array_size).astype(np.float32)

                # Warmup
                _ = engine.convolve(data, kernel)

                # Benchmark
                times = []
                for _ in range(n_runs):
                    start = time.perf_counter()
                    _ = engine.convolve(data, kernel)
                    times.append(time.perf_counter() - start)

                mean_time = np.mean(times) * 1000
                print(f"{mean_time:>13.2f}ms ", end='', flush=True)
            except:
                print(f"{'Error':<15}", end='', flush=True)

        print()

    # Test 2: Scaling with array size (fixed batch size)
    print("\n2. Array Size Scaling (batch_size=100, sigma=2.0)")
    print(f"{'Array Size':<15} ", end='')
    for backend in backends:
        print(f"{backend:<15}", end='')
    print()
    print("-" * (15 + 15 * len(backends)))

    array_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]
    batch_size = 100

    for array_size in array_sizes:
        print(f"{array_size:<15} ", end='', flush=True)

        for backend in backends:
            try:
                engine = ConvolutionEngine(backend=backend)
                data = np.random.randn(batch_size, array_size).astype(np.float32)

                # Warmup
                _ = engine.convolve(data, kernel)

                # Benchmark
                times = []
                for _ in range(n_runs):
                    start = time.perf_counter()
                    _ = engine.convolve(data, kernel)
                    times.append(time.perf_counter() - start)

                mean_time = np.mean(times) * 1000
                print(f"{mean_time:>13.2f}ms ", end='', flush=True)
            except:
                print(f"{'Error':<15}", end='', flush=True)

        print()


# Add to main():
def main():
    """Run all demos and tests"""
    print("\n" + "=" * 70)
    print("CONVOLUTION ENGINE - DEMO & TEST SUITE".center(70))
    print("=" * 70)

    demo_backend_availability()
    demo_backend_info()
    demo_basic_usage()
    test_dimensionality_support()
    test_numerical_consistency()
    test_batch_processing()
    test_error_handling()
    test_pytorch_tensor_support()
    test_pytorch_tensor_gpu()
    test_pytorch_tensor_types()
    test_kernel_size_effects()
    test_performance_comprehensive()
    test_performance_scaling()

    print("\n" + "=" * 70)
    print("Demo & Test Suite Complete!".center(70))
    print("=" * 70 + "\n")


"""
alright now it gets interesting for the masked padding! What do you think, should we design an engine for that as well? we probably wont need so many files for it since we dont need to design multiple backends i guess. but we should make sure that it operates on "array likes" so the API is the same between tensors and numpy arrays (they use the same APIs often and we should specifically use operations that are in both APIs). We should also think about two different kinds of paddings. there is the "normal" kind where we would add stuff to the array at the start and end. then there is the masked padding which is a lot more difficult (if done correctly, the old code did it not correctly), we will talk about it later.
"""



if __name__ == "__main__":
    main()
