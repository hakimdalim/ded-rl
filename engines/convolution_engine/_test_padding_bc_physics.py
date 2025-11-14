"""

WARNING THIS CODE IS PROBABLY COMPLETE TRASH!!!!!!!








Comprehensive Padding Boundary Condition Tests

Tests the PHYSICS of different padding modes - independent of application.
Verifies that padding strategies actually enforce the boundary conditions
they claim to enforce.

Test Categories:
1. Zero-Flux BC Tests (reflect, symmetric, edge)
2. Zero-Value BC Tests (constant=0, negation)
3. Convolution with BC Tests
4. Physical Reasonableness Tests

Goal: Ensure padding modes give physically sensible results.
"""

import numpy as np
import sys

from engines.convolution_engine.padding_engine.padding_engine import PaddingEngine
from engines.convolution_engine.engine import ConvolutionEngine


# ============================================================================
# Test Data Helpers
# ============================================================================

def create_linear_field(n: int, slope: float = 1.0, intercept: float = 0.0) -> np.ndarray:
    """Create linear temperature field."""
    x = np.arange(n, dtype=float)
    return slope * x + intercept


# ============================================================================
# Test 1: Zero-Flux BC Tests
# ============================================================================

def test_reflect_zero_flux():
    """Test 1.1: Reflect mode creates symmetric temperature distribution."""
    print("\n" + "="*80)
    print("TEST 1.1: Reflect Mode → Symmetric Distribution (Zero-Flux)")
    print("="*80)

    engine = PaddingEngine(backend='numpy_native')
    field = create_linear_field(20, slope=2.0, intercept=100.0)
    padded = engine.pad(field, pad_width=5, mode='reflect')

    print(f"Original: Linear field, slope=2.0")
    print(f"Check: Mirror symmetry around boundary")

    # Check symmetry: padded[boundary-i] should equal padded[boundary+i]
    checks = []
    for i in range(1, 4):
        left = padded[5-i]
        right = padded[5+i]
        diff = abs(left - right)
        checks.append(diff)
        print(f"  Mirror pair {i}: {left:.2f} vs {right:.2f}, diff={diff:.6f}")

    passed = all(d < 1e-10 for d in checks)
    print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_symmetric_zero_flux():
    """Test 1.2: Symmetric mode duplicates boundary value (zero gradient)."""
    print("\n" + "="*80)
    print("TEST 1.2: Symmetric Mode → Zero Gradient at Boundary")
    print("="*80)

    engine = PaddingEngine(backend='numpy_native')
    field = create_linear_field(20, slope=2.0, intercept=100.0)
    padded = engine.pad(field, pad_width=5, mode='symmetric')

    # Check that boundary value is duplicated
    left_dup = padded[4] == padded[5]
    right_dup = padded[-5] == padded[-6]

    print(f"Left boundary:  padded[4]={padded[4]:.2f}, padded[5]={padded[5]:.2f}")
    print(f"Right boundary: padded[-5]={padded[-5]:.2f}, padded[-6]={padded[-6]:.2f}")
    print(f"Gradient at boundary: {abs(padded[5] - padded[4]):.6f}")

    passed = left_dup and right_dup
    print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_edge_zero_flux():
    """Test 1.3: Edge mode extends with constant (zero gradient everywhere)."""
    print("\n" + "="*80)
    print("TEST 1.3: Edge Mode → Constant Padding")
    print("="*80)

    engine = PaddingEngine(backend='numpy_native')
    field = create_linear_field(20, slope=2.0, intercept=100.0)
    padded = engine.pad(field, pad_width=5, mode='edge')

    # Check all padded values are constant
    left_constant = np.all(padded[:5] == padded[0])
    right_constant = np.all(padded[-5:] == padded[-1])

    print(f"Left padding: all = {padded[0]:.2f}")
    print(f"Right padding: all = {padded[-1]:.2f}")

    passed = left_constant and right_constant
    print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================================
# Test 2: Zero-Value BC Tests
# ============================================================================

def test_constant_zero():
    """Test 2.1: Constant mode pads with exact zeros."""
    print("\n" + "="*80)
    print("TEST 2.1: Constant Mode (value=0)")
    print("="*80)

    engine = PaddingEngine(backend='numpy_native')
    field = np.full(20, 100.0, dtype=float)
    padded = engine.pad(field, pad_width=5, mode='constant', constant_value=0.0)

    left_zero = np.allclose(padded[:5], 0.0)
    right_zero = np.allclose(padded[-5:], 0.0)

    print(f"Left padding: {padded[:5]}")
    print(f"Right padding: {padded[-5:]}")

    passed = left_zero and right_zero
    print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_negated_symmetric():
    """Test 2.2: Negated symmetric enforces θ=0 at boundaries."""
    print("\n" + "="*80)
    print("TEST 2.2: Negated Symmetric → θ=0 BC")
    print("="*80)

    engine = PaddingEngine(backend='numpy_native')

    # Field with ZERO at boundaries (hot spot in middle)
    field = np.array([0, 5, 10, 15, 20, 15, 10, 5, 0], dtype=float)
    padded = engine.pad(field, pad_width=3, mode='symmetric')

    print(f"θ field (T-T_ambient): {field}")
    print(f"After symmetric pad: {padded}")

    # Negate padding
    padded[:3] *= -1
    padded[-3:] *= -1

    print(f"After negation: {padded}")

    # Boundaries should be zero
    left = padded[3]
    right = padded[-4]

    print(f"Boundaries: left={left:.6f}, right={right:.6f}")

    passed = abs(left) < 1e-10 and abs(right) < 1e-10
    print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================================
# Test 3: Convolution Tests
# ============================================================================

def test_convolution_energy_conservation():
    """Test 3.1: Zero-flux BC conserves energy."""
    print("\n" + "="*80)
    print("TEST 3.1: Convolution + Zero-Flux → Energy Conservation")
    print("="*80)

    pad_engine = PaddingEngine(backend='numba_numpy')
    conv_engine = ConvolutionEngine(backend='numba')

    # Hot spot
    field = np.zeros(50, dtype=float)
    field[24:26] = 100.0

    # Gaussian kernel
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    kernel /= kernel.sum()

    E_initial = field.sum()

    # Convolve with symmetric padding (zero-flux)
    padded = pad_engine.pad(field, pad_width=2, mode='symmetric')
    convolved = conv_engine.convolve(padded, kernel)
    result = convolved[2:-2]

    E_final = result.sum()
    error = abs(E_final - E_initial) / E_initial

    print(f"Initial energy: {E_initial:.6f}")
    print(f"Final energy:   {E_final:.6f}")
    print(f"Relative error: {error*100:.8f}%")

    passed = error < 1e-6
    print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_convolution_energy_loss():
    """Test 3.2: Zero-value BC loses energy to sink."""
    print("\n" + "="*80)
    print("TEST 3.2: Convolution + Zero-Value → Energy Loss")
    print("="*80)

    pad_engine = PaddingEngine(backend='numba_numpy')
    conv_engine = ConvolutionEngine(backend='numba')

    # Hot spot near edge
    field = np.zeros(20, dtype=float)
    field[2:4] = 100.0

    # Kernel
    kernel = np.array([0.25, 0.5, 0.25])

    E_initial = field.sum()

    # Multiple steps to let energy reach boundary
    result = field.copy()
    for _ in range(3):
        padded = pad_engine.pad(result, pad_width=1, mode='constant', constant_value=0.0)
        convolved = conv_engine.convolve(padded, kernel)
        result = convolved[1:-1]

    E_final = result.sum()
    loss = (E_initial - E_final) / E_initial

    print(f"Initial energy: {E_initial:.6f}")
    print(f"Final energy:   {E_final:.6f}")
    print(f"Energy lost:    {loss*100:.1f}%")

    passed = loss > 0.01  # At least 1% lost
    print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_uniform_stays_uniform():
    """Test 3.3: Uniform field remains uniform."""
    print("\n" + "="*80)
    print("TEST 3.3: Uniform Field Stays Uniform")
    print("="*80)

    pad_engine = PaddingEngine(backend='numpy_native')
    conv_engine = ConvolutionEngine(backend='numpy')

    field = np.full(50, 100.0, dtype=float)
    kernel = np.array([0.25, 0.5, 0.25])

    modes = ['reflect', 'symmetric', 'edge']
    all_uniform = True

    for mode in modes:
        padded = pad_engine.pad(field, pad_width=1, mode=mode)
        convolved = conv_engine.convolve(padded, kernel)
        result = convolved[1:-1]

        std = result.std()
        max_dev = np.abs(result - 100.0).max()

        print(f"  {mode:<12}: std={std:.6e}, max_dev={max_dev:.6e}")

        if std > 1e-10 or max_dev > 1e-10:
            all_uniform = False

    print(f"\nResult: {'✓ PASS' if all_uniform else '✗ FAIL'}")
    return all_uniform


# ============================================================================
# Test 4: Physical Reasonableness
# ============================================================================

def test_gradient_preservation():
    """Test 4.1: Interior gradient preserved with zero-flux."""
    print("\n" + "="*80)
    print("TEST 4.1: Zero-Flux Preserves Interior Gradient")
    print("="*80)

    engine = PaddingEngine(backend='numpy_native')
    slope = 2.5
    field = create_linear_field(30, slope=slope, intercept=50.0)
    padded = engine.pad(field, pad_width=10, mode='symmetric')

    # Measure gradient in interior
    interior = padded[15:-15]
    grad = np.gradient(interior)
    mean_grad = grad.mean()
    std_grad = grad.std()

    print(f"Original slope: {slope:.2f}")
    print(f"Interior gradient: mean={mean_grad:.6f}, std={std_grad:.6f}")

    passed = abs(mean_grad - slope) < 1e-10 and std_grad < 1e-10
    print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_heat_diffusion():
    """Test 4.2: Heat diffusion behaves physically."""
    print("\n" + "="*80)
    print("TEST 4.2: Heat Diffusion Physical Behavior")
    print("="*80)

    pad_engine = PaddingEngine(backend='numba_numpy')
    conv_engine = ConvolutionEngine(backend='numba')

    # Hot spot at indices 5-6 in 12-element array
    field = np.array([250, 250, 250, 250, 250, 600, 600, 250, 250, 250, 250, 250], dtype=float)
    kernel = np.array([0.25, 0.5, 0.25])

    print(f"Original field: {field}")
    print(f"Hot spot at indices [5, 6]")

    # Convolve
    padded = pad_engine.pad(field, pad_width=1, mode='symmetric')
    print(f"\nPadded field (length {len(padded)}): {padded}")

    convolved = conv_engine.convolve(padded, kernel)
    print(f"After convolution (length {len(convolved)}): {convolved}")

    # Check if convolution returned right size
    if len(convolved) == len(field):
        result = convolved  # Already the right size!
        print(f"Convolution returned correct size, using directly")
    else:
        result = convolved[1:-1]
        print(f"After extraction [1:-1] (length {len(result)}): {result}")

    print(f"\nComparison at each index:")
    for i in [4, 5, 6, 7]:
        if i < len(result):
            print(f"  Index {i}: {field[i]:.1f} → {result[i]:.1f} (change: {result[i]-field[i]:+.1f})")

    # Physical checks
    hot_cooled = result[5] < field[5] and result[6] < field[6]
    left_warmed = result[4] > field[4]
    right_warmed = result[7] > field[7]
    peak_reduced = result.max() < field.max()

    passed = hot_cooled and left_warmed and right_warmed and peak_reduced

    print(f"\nChecks:")
    print(f"  Hot cooled: {hot_cooled}")
    print(f"  Left neighbor warmed: {left_warmed}")
    print(f"  Right neighbor warmed: {right_warmed}")
    print(f"  Peak reduced: {peak_reduced}")

    print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


def test_backend_consistency():
    """Test 4.3: All backends give consistent results."""
    print("\n" + "="*80)
    print("TEST 4.3: Backend Consistency")
    print("="*80)

    available = PaddingEngine.list_available_backends()
    available_list = [k for k, v in available.items() if v]

    print(f"Testing {len(available_list)} backends")

    field = create_linear_field(30, slope=2.0, intercept=100.0)

    # Reference
    ref_engine = PaddingEngine(backend='numpy_native')
    ref_results = {}
    for mode in ['reflect', 'symmetric', 'edge']:
        ref_results[mode] = ref_engine.pad(field, pad_width=5, mode=mode)

    # Test others
    failures = 0
    for backend in available_list:
        if backend == 'numpy_native':
            continue

        try:
            engine = PaddingEngine(backend=backend)
            for mode in ['reflect', 'symmetric', 'edge']:
                result = engine.pad(field, pad_width=5, mode=mode)
                if not np.allclose(result, ref_results[mode]):
                    failures += 1
        except Exception:
            failures += 1

    passed = failures == 0
    print(f"Failures: {failures}")
    print(f"\nResult: {'✓ PASS' if passed else '✗ FAIL'}")
    return passed


# ============================================================================
# Main Runner
# ============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("PADDING BOUNDARY CONDITIONS - PHYSICS TESTS")
    print("="*80)
    print("\nGoal: Verify padding modes enforce their claimed boundary conditions")
    print("="*80)

    tests = [
        ("Reflect → Zero-Flux", test_reflect_zero_flux),
        ("Symmetric → Zero-Flux", test_symmetric_zero_flux),
        ("Edge → Zero-Flux", test_edge_zero_flux),
        ("Constant=0 → Zero-Value", test_constant_zero),
        ("Negated Symmetric → Zero-Value", test_negated_symmetric),
        ("Conv + Zero-Flux → Conservation", test_convolution_energy_conservation),
        ("Conv + Zero-Value → Loss", test_convolution_energy_loss),
        ("Uniform Stays Uniform", test_uniform_stays_uniform),
        ("Zero-Flux Preserves Gradient", test_gradient_preservation),
        ("Heat Diffusion Behavior", test_heat_diffusion),
        ("Backend Consistency", test_backend_consistency),
    ]

    results = {}
    for name, test_func in tests:
        try:
            passed = test_func()
            results[name] = passed
        except Exception as e:
            print(f"\n✗ EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    total = len(results)
    passed = sum(results.values())

    print(f"\nTotal: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print()

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print("\n" + "="*80)

    if passed == total:
        print("✓ ALL TESTS PASSED!")
        print("\nConclusion: Padding modes correctly enforce boundary conditions.")
        print("="*80 + "\n")
        return 0
    else:
        print(f"✗ {total - passed} TESTS FAILED")
        print("="*80 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())