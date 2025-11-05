"""
Region Finding - Demo & Quick Tests

Simple demonstrations of region finding functionality.
Updated for refactored API with array outputs.
"""

import numpy as np
import sys

from convolution_engine.region_finding.region_finding_implementations import (
    find_regions,
    list_available_finders,
    validate_regions,
    verify_region_correctness,
)


def demo_basic_usage():
    """Demonstrate basic usage."""
    print("="*70)
    print("BASIC USAGE DEMO".center(70))
    print("="*70 + "\n")

    # Simple example
    mask = np.array([True, True, True, False, False, True, True, True, True, False, True])
    print(f"Mask: {mask.astype(int)}")
    print(f"      {' '.join(str(i) for i in range(len(mask)))}")
    print()

    # Find regions with different min_length
    for min_length in [1, 2, 3, 4]:
        regions = find_regions(mask, min_length=min_length)
        if len(regions) > 0:
            print(f"min_length={min_length}: {regions.tolist()}")
        else:
            print(f"min_length={min_length}: [] (no regions found)")

    print("\nExplanation:")
    print("  min_length=1: All True sequences (3, 4, 1 elements)")
    print("  min_length=2: Only sequences >=2 (3, 4 elements)")
    print("  min_length=3: Only sequences >=3 (3, 4 elements)")
    print("  min_length=4: Only sequences >=4 (4 elements)")
    print(f"\nNew format: Returns array([[start, end], ...]), shape (n_regions, 2)")


def demo_all_backends():
    """Test all available backends."""
    print("\n" + "="*70)
    print("ALL BACKENDS DEMO".center(70))
    print("="*70 + "\n")

    mask = np.array([True, True, False, True, True, True, False, True, False])
    min_length = 2

    print(f"Mask: {mask.astype(int)}")
    print(f"min_length: {min_length}\n")

    available = list_available_finders()

    print(f"{'Backend':<15} {'Available':<12} {'Result':<40}")
    print("-" * 67)

    for backend, is_available in available.items():
        if is_available:
            try:
                regions = find_regions(mask, min_length, backend=backend)
                result_str = regions.tolist() if len(regions) > 0 else "[]"
                print(f"{backend:<15} {'✓ Yes':<12} {str(result_str):<40}")
            except Exception as e:
                print(f"{backend:<15} {'✓ Yes':<12} Error: {str(e)[:30]}")
        else:
            print(f"{backend:<15} {'✗ No':<12} {'N/A':<40}")


def demo_batched_input():
    """Demonstrate batched input handling."""
    print("\n" + "="*70)
    print("BATCHED INPUT DEMO".center(70))
    print("="*70 + "\n")

    # 2D batch
    masks_2d = np.array([
        [True, True, False, True, True, True, False, True],
        [True, False, False, True, True, True, True, True],
        [False, False, True, True, True, False, True, True],
    ])

    print("2D Batch (3 signals):")
    print(f"Shape: {masks_2d.shape}")
    results = find_regions(masks_2d, min_length=2)

    for i, r in enumerate(results):
        print(f"  Signal {i}: {r.shape[0]} regions -> {r.tolist()}")

    # 3D batch
    print("\n3D Batch (2x3 grid of signals):")
    masks_3d = np.random.rand(2, 3, 20) > 0.5
    print(f"Shape: {masks_3d.shape}")
    results_3d = find_regions(masks_3d, min_length=3)

    for i in range(len(results_3d)):
        for j in range(len(results_3d[i])):
            n_regions = results_3d[i][j].shape[0]
            print(f"  [{i},{j}]: {n_regions} regions")


def demo_realistic_scenario():
    """Demonstrate realistic use case."""
    print("\n" + "="*70)
    print("REALISTIC SCENARIO: Sensor Data with Gaps".center(70))
    print("="*70 + "\n")

    # Simulate sensor data with dropouts
    np.random.seed(42)
    signal_length = 50

    # Create mask with random dropouts
    mask = np.ones(signal_length, dtype=bool)
    dropout_positions = np.random.choice(signal_length, size=10, replace=False)
    mask[dropout_positions] = False

    # Also create a larger gap
    mask[20:25] = False

    print("Sensor status (1=valid, 0=dropout):")
    print(''.join('█' if m else '·' for m in mask))
    print(''.join(str(i % 10) for i in range(len(mask))))
    print()

    # Find valid regions for convolution
    kernel_size = 5
    min_length = kernel_size  # Need at least kernel_size points

    regions = find_regions(mask, min_length=min_length)

    print(f"Kernel size: {kernel_size}")
    print(f"Valid regions (length >= {min_length}):\n")

    for i in range(len(regions)):
        start, end = regions[i, 0], regions[i, 1]
        length = end - start
        print(f"  Region {i+1}: indices [{start:2d}:{end:2d}], length={length:2d}")

    print(f"\nTotal valid points: {np.sum(mask)}")
    print(f"Usable regions: {len(regions)}")
    total_in_regions = sum(regions[i, 1] - regions[i, 0] for i in range(len(regions)))
    print(f"Points lost to gaps: {np.sum(mask) - total_in_regions}")


def demo_visualization():
    """Visual demonstration of region finding."""
    print("\n" + "="*70)
    print("VISUAL DEMONSTRATION".center(70))
    print("="*70 + "\n")

    # Create interesting pattern
    mask = np.array([
        True, True, True, True, True,          # Region 1 (length 5)
        False, False,                           # Gap
        True, True,                             # Too short (length 2)
        False,                                  # Gap
        True, True, True, True, True, True,    # Region 2 (length 6)
        False, False, False,                    # Gap
        True, True, True,                       # Region 3 (length 3)
        False,                                  # Gap
        True,                                   # Too short (length 1)
        False, False,                           # Gap
        True, True, True, True,                 # Region 4 (length 4)
    ])

    min_length = 3
    regions = find_regions(mask, min_length=min_length)

    print(f"Mask visualization (min_length={min_length}):\n")

    # Create visual representation
    visual = []
    for i, m in enumerate(mask):
        # Check if this index is in any region
        in_region = False
        region_num = None
        for r_idx in range(len(regions)):
            start, end = regions[r_idx, 0], regions[r_idx, 1]
            if start <= i < end:
                in_region = True
                region_num = r_idx + 1
                break

        if in_region:
            visual.append(str(region_num))
        elif m:
            visual.append('x')  # Valid but too short
        else:
            visual.append('·')  # Masked

    print(''.join(visual))
    print(''.join(str(i % 10) for i in range(len(mask))))
    print()

    print("Legend:")
    print("  1,2,3,4 = Region number (valid and long enough)")
    print("  x       = Valid but too short (length < min_length)")
    print("  ·       = Masked (False)")
    print()

    print(f"Found {len(regions)} regions:\n")
    for i in range(len(regions)):
        start, end = regions[i, 0], regions[i, 1]
        print(f"  Region {i+1}: [{start:2d}:{end:2d}], length={end-start}")


def demo_performance_comparison():
    """Quick performance comparison."""
    print("\n" + "="*70)
    print("QUICK PERFORMANCE COMPARISON".center(70))
    print("="*70 + "\n")

    import time

    # Create larger test case
    size = 100000
    mask = np.random.rand(size) < 0.6
    min_length = 5

    print(f"Test: {size} elements, ~60% valid, min_length={min_length}\n")

    available = list_available_finders()
    n_runs = 10

    print(f"{'Backend':<15} {'Time (ms)':<15} {'Regions':<10}")
    print("-" * 40)

    for backend, is_available in available.items():
        if not is_available:
            continue

        try:
            # Warmup
            _ = find_regions(mask, min_length, backend=backend)

            # Benchmark
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                regions = find_regions(mask, min_length, backend=backend)
                end = time.perf_counter()
                times.append(end - start)

            mean_time = np.mean(times) * 1000
            print(f"{backend:<15} {mean_time:>13.3f}  {len(regions):<10}")

        except Exception as e:
            print(f"{backend:<15} {'ERROR':<15} {str(e)[:20]}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("REGION FINDING - DEMO SUITE (UPDATED)".center(70))
    print("="*70)

    demo_basic_usage()
    demo_all_backends()
    demo_batched_input()
    demo_visualization()
    demo_realistic_scenario()
    demo_performance_comparison()

    print("\n" + "="*70)
    print("DEMO COMPLETE".center(70))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()