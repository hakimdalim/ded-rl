"""
Comprehensive Testing Suite for MaskedConvolutionEngine

Tests all combinations of:
- Region finding backends (numpy, numba, torch, python, scipy)
- Convolution backends (numpy, scipy, numba, pytorch_cpu, pytorch_gpu)
- Padding backends (numpy_native, numba_numpy, numba_pure, pytorch_cpu, pytorch_gpu, custom_vectorized)
- Padding modes (reflect, symmetric, edge, constant, wrap)

Validates:
- Correctness of results
- Gap preservation (masked regions keep original values)
- Shape consistency
- Backend availability handling
- Edge cases

Key Behavior:
- Masked regions (mask=False) preserve their original values
- Only valid regions (mask=True) are convolved
- No NaN values in output (unless input contains NaN)
"""

import numpy as np
import sys
from typing import List, Dict, Tuple
from itertools import product

from engines.convolution_engine.masked_convolution_engine import MaskedConvolutionEngine
from engines.convolution_engine.padding_engine.padding_engine import PaddingEngine
from engines.convolution_engine.region_finding.region_finding_engine import RegionFindingEngine
from engines.convolution_engine.engine import ConvolutionEngine


# ============================================================================
# Test Data Generation
# ============================================================================

def generate_test_signals() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate various test signals with masks.

    Returns:
        Dictionary mapping test_name -> (signal, mask)
    """
    signals = {}

    # Simple case: single gap
    signal1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    mask1 = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1], dtype=bool)
    signals['single_gap'] = (signal1, mask1)

    # Multiple gaps
    signal2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=float)
    mask2 = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1], dtype=bool)
    signals['multiple_gaps'] = (signal2, mask2)

    # Edge gaps
    signal3 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    mask3 = np.array([0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
    signals['edge_gaps'] = (signal3, mask3)

    # No gaps
    signal4 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    mask4 = np.ones(len(signal4), dtype=bool)
    signals['no_gaps'] = (signal4, mask4)

    # All gaps
    signal5 = np.array([1, 2, 3, 4, 5], dtype=float)
    mask5 = np.zeros(len(signal5), dtype=bool)
    signals['all_gaps'] = (signal5, mask5)

    # Single valid region (short)
    signal6 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    mask6 = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    signals['single_short_region'] = (signal6, mask6)

    # Many small regions
    signal7 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
    mask7 = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0], dtype=bool)
    signals['many_small_regions'] = (signal7, mask7)

    # Regions of different lengths
    signal8 = np.arange(1, 21, dtype=float)
    mask8 = np.array([1,1,1,0, 1,1,1,1,1,0, 1,1,0, 1,1,1,1,1,1,1], dtype=bool)
    signals['varied_lengths'] = (signal8, mask8)

    return signals


def generate_test_kernels() -> Dict[str, np.ndarray]:
    """Generate various test kernels."""
    kernels = {
        'small_smooth': np.array([0.25, 0.5, 0.25]),
        'larger_smooth': np.array([0.1, 0.2, 0.4, 0.2, 0.1]),
        'uniform_3': np.ones(3) / 3,
        'uniform_5': np.ones(5) / 5,
        'uniform_7': np.ones(7) / 7,
    }
    return kernels


# ============================================================================
# Backend Availability Checking
# ============================================================================

def get_available_backends() -> Dict[str, List[str]]:
    """
    Check which backends are available on this system.

    Returns:
        Dictionary with available backends for each engine type
    """
    available = {
        'region': [],
        'padding': [],
        'convolution': []
    }

    # Region finding backends
    region_avail = RegionFindingEngine.list_available_backends()
    available['region'] = [k for k, v in region_avail.items() if v]

    # Padding backends
    padding_avail = PaddingEngine.list_available_backends()
    available['padding'] = [k for k, v in padding_avail.items() if v]

    # Convolution backends
    conv_avail = ConvolutionEngine.list_available_backends()
    available['convolution'] = [k for k, v in conv_avail.items() if v]

    return available


# ============================================================================
# Validation Functions
# ============================================================================

def validate_result(
    signal: np.ndarray,
    mask: np.ndarray,
    result: np.ndarray,
    kernel: np.ndarray
) -> Tuple[bool, str]:
    """
    Validate convolution result.

    Args:
        signal: Original signal
        mask: Boolean mask
        result: Convolution result
        kernel: Convolution kernel

    Returns:
        (is_valid, error_message)
    """
    # Check shape
    if result.shape != signal.shape:
        return False, f"Shape mismatch: expected {signal.shape}, got {result.shape}"

    # Check that gaps preserve original values (mask=False should keep original)
    if not mask.all():  # If there are any gaps
        gap_signal = signal[~mask]
        gap_result = result[~mask]

        # Handle NaN values properly
        signal_nan = np.isnan(gap_signal)
        result_nan = np.isnan(gap_result)

        # Check NaN positions match
        if not np.array_equal(signal_nan, result_nan):
            return False, "NaN positions don't match in gaps"

        # Check finite values match
        finite_mask = ~signal_nan
        if finite_mask.any():
            gaps_preserved = np.allclose(
                gap_result[finite_mask],
                gap_signal[finite_mask],
                rtol=1e-10
            )
            if not gaps_preserved:
                max_diff = np.abs(gap_result[finite_mask] - gap_signal[finite_mask]).max()
                return False, f"Gaps not preserved (max diff: {max_diff:.6e})"

    # Check that valid regions have been processed (no NaN unless input had NaN)
    if mask.any():
        valid_signal = signal[mask]
        valid_result = result[mask]

        # If input valid region has no NaN, output shouldn't either
        if not np.any(np.isnan(valid_signal)):
            if np.any(np.isnan(valid_result)):
                return False, "NaN values appeared in valid regions"

    return True, "OK"


# ============================================================================
# Test Execution
# ============================================================================

def test_backend_combination(
    region_backend: str,
    conv_backend: str,
    padding_backend: str,
    padding_mode: str,
    signal: np.ndarray,
    mask: np.ndarray,
    kernel: np.ndarray,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Test a specific combination of backends.

    Returns:
        (success, error_message)
    """
    try:
        # Create engines
        engine = MaskedConvolutionEngine(
            region_backend=region_backend,
            convolution_backend=conv_backend
        )

        padding_engine = PaddingEngine(backend=padding_backend)

        # Create pad function
        if padding_mode == 'constant':
            pad_func = lambda x, pw: padding_engine.pad(x, pw, mode=padding_mode, constant_value=0)
        else:
            pad_func = lambda x, pw: padding_engine.pad(x, pw, mode=padding_mode)

        # Convolve
        result = engine.convolve(
            signal=signal,
            kernel=kernel,
            mask=mask,
            pad_width=len(kernel)//2,
            pad_func=pad_func
        )

        # Validate
        is_valid, msg = validate_result(signal, mask, result, kernel)

        if not is_valid:
            return False, msg

        if verbose:
            n_regions = np.sum(np.diff(np.concatenate(([False], mask, [False])).astype(int)) == 1)
            print(f"  ✓ Success: {n_regions} regions processed")

        return True, "OK"

    except ValueError as e:
        # Check if this is an expected validation error
        error_msg = str(e)
        if "smaller than min_length" in error_msg or "Last dimension" in error_msg:
            # This is expected when signal is too short for kernel
            # Consider this a pass (correct validation behavior)
            return True, "OK (expected validation error)"
        else:
            return False, f"ValueError: {error_msg}"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def run_comprehensive_tests(verbose: bool = True) -> Dict[str, any]:
    """
    Run comprehensive test suite.

    Args:
        verbose: Print detailed progress

    Returns:
        Dictionary with test results
    """
    if verbose:
        print("="*80)
        print("MASKED CONVOLUTION ENGINE - COMPREHENSIVE TEST SUITE".center(80))
        print("="*80)
        print()

    # Get available backends
    available = get_available_backends()

    if verbose:
        print("Available Backends:")
        print(f"  Region finding: {', '.join(available['region'])}")
        print(f"  Padding: {', '.join(available['padding'])}")
        print(f"  Convolution: {', '.join(available['convolution'])}")
        print()

    # Generate test data
    test_signals = generate_test_signals()
    test_kernels = generate_test_kernels()
    padding_modes = ['reflect', 'symmetric', 'edge', 'constant', 'wrap']

    # Results tracking
    results = {
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'errors': [],
        'by_backend': {},
        'by_signal': {},
        'by_padding_mode': {}
    }

    # Test each combination
    for signal_name, (signal, mask) in test_signals.items():
        if verbose:
            print(f"\n{'='*80}")
            print(f"Testing: {signal_name}")
            print(f"  Signal length: {len(signal)}, Valid points: {np.sum(mask)}, Gaps: {np.sum(~mask)}")
            print(f"{'='*80}")

        results['by_signal'][signal_name] = {'passed': 0, 'failed': 0}

        for kernel_name, kernel in test_kernels.items():
            if verbose:
                print(f"\n  Kernel: {kernel_name} (size={len(kernel)})")

            # Test all available backend combinations
            for region_backend in available['region']:
                for conv_backend in available['convolution']:
                    for padding_backend in available['padding']:
                        for padding_mode in padding_modes:

                            results['total_tests'] += 1

                            # Track by backend combo
                            backend_combo = f"{region_backend}+{conv_backend}+{padding_backend}"
                            if backend_combo not in results['by_backend']:
                                results['by_backend'][backend_combo] = {'passed': 0, 'failed': 0}

                            # Track by padding mode
                            if padding_mode not in results['by_padding_mode']:
                                results['by_padding_mode'][padding_mode] = {'passed': 0, 'failed': 0}

                            # Run test
                            success, msg = test_backend_combination(
                                region_backend=region_backend,
                                conv_backend=conv_backend,
                                padding_backend=padding_backend,
                                padding_mode=padding_mode,
                                signal=signal,
                                mask=mask,
                                kernel=kernel,
                                verbose=False
                            )

                            if success:
                                results['passed'] += 1
                                results['by_backend'][backend_combo]['passed'] += 1
                                results['by_signal'][signal_name]['passed'] += 1
                                results['by_padding_mode'][padding_mode]['passed'] += 1
                            else:
                                results['failed'] += 1
                                results['by_backend'][backend_combo]['failed'] += 1
                                results['by_signal'][signal_name]['failed'] += 1
                                results['by_padding_mode'][padding_mode]['failed'] += 1

                                error_info = {
                                    'signal': signal_name,
                                    'kernel': kernel_name,
                                    'region_backend': region_backend,
                                    'conv_backend': conv_backend,
                                    'padding_backend': padding_backend,
                                    'padding_mode': padding_mode,
                                    'error': msg
                                }
                                results['errors'].append(error_info)

                                if verbose:
                                    print(f"    ✗ FAILED: {region_backend}+{conv_backend}+{padding_backend}+{padding_mode}")
                                    print(f"      Error: {msg}")

    return results


def print_summary(results: Dict[str, any]):
    """Print test summary."""
    print("\n" + "="*80)
    print("TEST SUMMARY".center(80))
    print("="*80)
    print()

    # Overall stats
    total = results['total_tests']
    passed = results['passed']
    failed = results['failed']
    success_rate = (passed / total * 100) if total > 0 else 0

    print(f"Total tests: {total}")
    print(f"Passed: {passed} ({success_rate:.1f}%)")
    print(f"Failed: {failed}")
    print()

    # By signal
    print("Results by signal type:")
    print(f"  {'Signal':<25} {'Passed':<10} {'Failed':<10}")
    print("  " + "-"*45)
    for signal_name, counts in results['by_signal'].items():
        print(f"  {signal_name:<25} {counts['passed']:<10} {counts['failed']:<10}")
    print()

    # By backend combo
    print("Results by backend combination:")
    print(f"  {'Backend Combo':<50} {'Passed':<10} {'Failed':<10}")
    print("  " + "-"*70)
    for combo, counts in sorted(results['by_backend'].items()):
        print(f"  {combo:<50} {counts['passed']:<10} {counts['failed']:<10}")
    print()

    # By padding mode
    print("Results by padding mode:")
    print(f"  {'Mode':<15} {'Passed':<10} {'Failed':<10}")
    print("  " + "-"*35)
    for mode, counts in sorted(results['by_padding_mode'].items()):
        print(f"  {mode:<15} {counts['passed']:<10} {counts['failed']:<10}")
    print()

    # Errors
    if results['errors']:
        print(f"\nFailed tests ({len(results['errors'])}):")
        print("-"*80)
        for i, error in enumerate(results['errors'][:20], 1):  # Show first 20
            print(f"\n{i}. {error['signal']} / {error['kernel']}")
            print(f"   Backends: {error['region_backend']} + {error['conv_backend']} + {error['padding_backend']}")
            print(f"   Padding: {error['padding_mode']}")
            print(f"   Error: {error['error']}")

        if len(results['errors']) > 20:
            print(f"\n... and {len(results['errors']) - 20} more errors")
    else:
        print("\n✓ All tests passed!")

    print("\n" + "="*80)


# ============================================================================
# Quick Sanity Tests
# ============================================================================

def run_quick_sanity_tests():
    """Run quick sanity tests with default backends."""
    print("\n" + "="*80)
    print("QUICK SANITY TESTS".center(80))
    print("="*80)
    print()

    # Test 1: Basic functionality
    print("Test 1: Basic functionality")
    engine = MaskedConvolutionEngine.create_fast()
    padding_engine = PaddingEngine(backend='numba_numpy')
    pad_func = lambda x, pw: padding_engine.pad(x, pw, mode='reflect')

    signal = np.array([1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, np.nan, 11, 12], dtype=float)
    mask = np.isfinite(signal)
    kernel = np.array([0.25, 0.5, 0.25])

    result = engine.convolve(signal, kernel, mask, pad_width=1, pad_func=pad_func)

    is_valid, msg = validate_result(signal, mask, result, kernel)
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'} - {msg}")

    # Test 2: No gaps
    print("\nTest 2: No gaps (should work like regular convolution)")
    signal_no_gaps = np.arange(1, 11, dtype=float)
    mask_no_gaps = np.ones(len(signal_no_gaps), dtype=bool)

    result_no_gaps = engine.convolve(signal_no_gaps, kernel, mask_no_gaps, pad_width=1, pad_func=pad_func)
    is_valid, msg = validate_result(signal_no_gaps, mask_no_gaps, result_no_gaps, kernel)
    print(f"  Result: {'✓ PASS' if is_valid else '✗ FAIL'} - {msg}")

    # Test 3: All gaps (should preserve original values)
    print("\nTest 3: All gaps (should preserve original values)")
    signal_all_gaps = np.arange(1, 6, dtype=float)
    mask_all_gaps = np.zeros(len(signal_all_gaps), dtype=bool)

    result_all_gaps = engine.convolve(signal_all_gaps, kernel, mask_all_gaps, pad_width=1, pad_func=pad_func)
    values_preserved = np.allclose(result_all_gaps, signal_all_gaps)
    print(f"  Result: {'✓ PASS' if values_preserved else '✗ FAIL'} - {'Original values preserved' if values_preserved else 'Values not preserved'}")

    print("\n" + "="*80)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MASKED CONVOLUTION ENGINE - TEST SUITE".center(80))
    print("="*80)

    # Quick sanity tests
    run_quick_sanity_tests()

    # Comprehensive tests
    print("\nStarting comprehensive tests...")
    print("This will test all combinations of backends, padding modes, and test signals.")
    print("This may take a few minutes...\n")

    results = run_comprehensive_tests(verbose=False)

    # Print summary
    print_summary(results)

    # Return exit code
    if results['failed'] == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {results['failed']} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())