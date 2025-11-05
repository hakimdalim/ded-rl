"""
Example: Custom Padding with Post-Transformation

Demonstrates how to use custom padding functions with transformations
for specialized boundary conditions (e.g., heat sink, Neumann BCs).

This shows the power of the pad_func interface - you can do arbitrary
transformations after padding.
"""

import numpy as np
import matplotlib.pyplot as plt

from engines.convolution_engine.masked_convolution_engine import MaskedConvolutionEngine
from engines.convolution_engine.padding_engine.padding_engine import PaddingEngine


# ============================================================================
# Example 1: Negated Padding (Heat Sink / Zero-Temperature BC)
# ============================================================================

def create_negated_pad_func(padding_engine: PaddingEngine, mode: str = 'reflect'):
    """
    Create padding function that negates the padded regions.
    
    Simulates zero-temperature boundary condition (heat sink):
    - Pad with reflection
    - Negate padded values to enforce T=0 at boundary
    
    Args:
        padding_engine: PaddingEngine instance
        mode: Padding mode to use before negation
    
    Returns:
        Padding function compatible with MaskedConvolutionEngine
    """
    def negated_pad(data, pad_width):
        """Pad and negate the padded regions."""
        # Parse pad_width
        if isinstance(pad_width, int):
            pad_left = pad_right = pad_width
        else:
            pad_left, pad_right = pad_width
        
        # Pad with reflection (or other mode)
        padded = padding_engine.pad(data, pad_width, mode=mode)
        
        # For 1D: negate left and right padded regions
        if padded.ndim == 1:
            if pad_left > 0:
                padded[:pad_left] *= -1
            if pad_right > 0:
                padded[-pad_right:] *= -1
        
        # For 2D batch: negate for each row
        elif padded.ndim == 2:
            if pad_left > 0:
                padded[:, :pad_left] *= -1
            if pad_right > 0:
                padded[:, -pad_right:] *= -1
        
        return padded
    
    return negated_pad


# ============================================================================
# Example 2: Gradient Padding (Neumann BC)
# ============================================================================

def create_gradient_pad_func(padding_engine: PaddingEngine):
    """
    Create padding function that extends with linear extrapolation.
    
    Simulates Neumann boundary condition (zero gradient at boundary):
    - Use edge padding (constant extrapolation)
    
    Args:
        padding_engine: PaddingEngine instance
    
    Returns:
        Padding function compatible with MaskedConvolutionEngine
    """
    def gradient_pad(data, pad_width):
        """Pad with constant gradient (edge mode)."""
        # Edge mode already does this, but you could implement
        # more sophisticated gradient extrapolation here
        return padding_engine.pad(data, pad_width, mode='edge')
    
    return gradient_pad


# ============================================================================
# Example 3: Clamped Padding (Fixed Value BC)
# ============================================================================

def create_clamped_pad_func(padding_engine: PaddingEngine, clamp_value: float = 0.0):
    """
    Create padding function that clamps boundaries to a fixed value.
    
    Simulates Dirichlet boundary condition with specific value.
    
    Args:
        padding_engine: PaddingEngine instance
        clamp_value: Value to clamp boundaries to
    
    Returns:
        Padding function compatible with MaskedConvolutionEngine
    """
    def clamped_pad(data, pad_width):
        """Pad with constant value."""
        return padding_engine.pad(data, pad_width, mode='constant', constant_value=clamp_value)
    
    return clamped_pad


# ============================================================================
# Example 4: Scaled Padding (Damped BC)
# ============================================================================

def create_scaled_pad_func(padding_engine: PaddingEngine, scale: float = 0.5, mode: str = 'reflect'):
    """
    Create padding function that scales the padded regions.
    
    Useful for damping boundary effects or simulating partial reflection.
    
    Args:
        padding_engine: PaddingEngine instance
        scale: Scale factor for padded regions (0=zero, 1=no change)
        mode: Padding mode to use before scaling
    
    Returns:
        Padding function compatible with MaskedConvolutionEngine
    """
    def scaled_pad(data, pad_width):
        """Pad and scale the padded regions."""
        if isinstance(pad_width, int):
            pad_left = pad_right = pad_width
        else:
            pad_left, pad_right = pad_width
        
        padded = padding_engine.pad(data, pad_width, mode=mode)
        
        # Scale padded regions
        if padded.ndim == 1:
            if pad_left > 0:
                padded[:pad_left] *= scale
            if pad_right > 0:
                padded[-pad_right:] *= scale
        elif padded.ndim == 2:
            if pad_left > 0:
                padded[:, :pad_left] *= scale
            if pad_right > 0:
                padded[:, -pad_right:] *= scale
        
        return padded
    
    return scaled_pad


# ============================================================================
# Demonstration
# ============================================================================

def demo_negated_padding():
    """Demonstrate negated padding for heat sink BC."""
    print("="*80)
    print("DEMO: Negated Padding (Heat Sink / Zero-Temperature BC)")
    print("="*80)
    print()
    
    # Create engines
    engine = MaskedConvolutionEngine.create_fast()
    padding_engine = PaddingEngine(backend='numba_numpy')
    
    # Signal with gaps (simulating temperature field with gaps)
    signal = np.array([100, 150, 200, np.nan, np.nan, 180, 220, 250, 200, np.nan, 160, 140, 120], dtype=float)
    mask = np.isfinite(signal)
    kernel = np.array([0.25, 0.5, 0.25])  # Smoothing (heat diffusion)
    
    print(f"Signal: {signal}")
    print(f"Mask:   {mask.astype(int)}")
    print(f"Kernel: {kernel}")
    print()
    
    # Compare different padding strategies
    strategies = {
        'Standard Reflect': lambda x, pw: padding_engine.pad(x, pw, mode='reflect'),
        'Negated Reflect (Heat Sink)': create_negated_pad_func(padding_engine, mode='reflect'),
        'Clamped to Zero': create_clamped_pad_func(padding_engine, clamp_value=0.0),
        'Scaled (50% damping)': create_scaled_pad_func(padding_engine, scale=0.5, mode='reflect'),
    }
    
    results = {}
    for name, pad_func in strategies.items():
        result = engine.convolve(
            signal=signal,
            kernel=kernel,
            mask=mask,
            pad_width=len(kernel)//2,
            pad_func=pad_func
        )
        results[name] = result
        print(f"{name}:")
        print(f"  Result: {result}")
        print()
    
    return signal, mask, results


def visualize_padding_comparison():
    """Visualize different padding strategies."""
    print("\n" + "="*80)
    print("VISUALIZATION: Padding Strategy Comparison")
    print("="*80)
    print()
    
    # Create test data
    engine = MaskedConvolutionEngine.create_fast()
    padding_engine = PaddingEngine(backend='numba_numpy')
    
    # Longer signal for better visualization
    np.random.seed(42)
    signal_full = np.sin(np.linspace(0, 4*np.pi, 50)) * 100 + 150
    # Add some gaps
    signal = signal_full.copy()
    signal[10:15] = np.nan
    signal[30:35] = np.nan
    mask = np.isfinite(signal)
    
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])  # 5-point smoother
    
    # Different padding strategies
    strategies = {
        'Reflect': lambda x, pw: padding_engine.pad(x, pw, mode='reflect'),
        'Negated (Heat Sink)': create_negated_pad_func(padding_engine, mode='reflect'),
        'Edge': lambda x, pw: padding_engine.pad(x, pw, mode='edge'),
        'Constant=0': create_clamped_pad_func(padding_engine, clamp_value=0.0),
    }
    
    # Apply each strategy
    fig, axes = plt.subplots(len(strategies) + 1, 1, figsize=(12, 10), sharex=True)
    
    # Plot original
    x = np.arange(len(signal))
    valid = mask
    axes[0].plot(x[valid], signal[valid], 'o-', label='Valid data', markersize=4)
    axes[0].plot(x[~valid], [0]*np.sum(~valid), 'rx', label='Gaps', markersize=8)
    axes[0].set_ylabel('Original')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Masked Convolution: Impact of Padding Strategy on Boundary Effects')
    
    # Plot each strategy
    for i, (name, pad_func) in enumerate(strategies.items(), 1):
        result = engine.convolve(
            signal=signal,
            kernel=kernel,
            mask=mask,
            pad_width=len(kernel)//2,
            pad_func=pad_func
        )
        
        valid_out = np.isfinite(result)
        axes[i].plot(x[valid_out], result[valid_out], 's-', label=f'{name}', markersize=3, alpha=0.8)
        axes[i].plot(x[~valid_out], [0]*np.sum(~valid_out), 'rx', label='Gaps', markersize=8)
        axes[i].set_ylabel(name)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Highlight boundary effects
        axes[i].axvspan(8, 16, alpha=0.1, color='red', label='Gap region')
        axes[i].axvspan(28, 36, alpha=0.1, color='red')
    
    axes[-1].set_xlabel('Index')
    plt.tight_layout()
    plt.show()


def demo_practical_example():
    """Practical example: thermal diffusion with heat sink boundaries."""
    print("\n" + "="*80)
    print("PRACTICAL EXAMPLE: Thermal Diffusion with Heat Sink")
    print("="*80)
    print()

    # Simulate temperature distribution with gaps (sensor failures)
    # Temperature in Kelvin
    temperature = np.array([
        300, 320, 340, 360, 380, 400, 420, 440,  # Heating region
        np.nan, np.nan,  # Sensor failure
        380, 360, 340, 320, 300, 280, 260,  # Cooling region
        np.nan,  # Sensor failure
        240, 220, 200
    ], dtype=float)

    mask = np.isfinite(temperature)

    # Heat diffusion kernel (Gaussian smoothing)
    sigma = 1.0
    kernel_size = 5
    x = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel /= kernel.sum()

    print(f"Temperature field: {len(temperature)} points, {np.sum(~mask)} gaps")
    print(f"Diffusion kernel: {kernel_size} points, sigma={sigma}")
    print()

    # Setup
    engine = MaskedConvolutionEngine.create_fast()
    padding_engine = PaddingEngine(backend='numba_numpy')

    # Compare: adiabatic vs heat sink boundaries
    pad_func_adiabatic = lambda x, pw: padding_engine.pad(x, pw, mode='edge')  # Zero gradient
    pad_func_heatsink = create_negated_pad_func(padding_engine, mode='reflect')  # T=0 at boundary

    result_adiabatic = engine.convolve(
        temperature, kernel, mask,
        pad_width=kernel_size//2,
        pad_func=pad_func_adiabatic
    )

    result_heatsink = engine.convolve(
        temperature, kernel, mask,
        pad_width=kernel_size//2,
        pad_func=pad_func_heatsink
    )

    print("Results at region boundaries:")
    print(f"  Adiabatic (edge padding):  {result_adiabatic[6:10]}")
    print(f"  Heat sink (negated):       {result_heatsink[6:10]}")
    print()
    print("Notice: Heat sink BC pulls temperatures down near boundaries")
    print("        Adiabatic BC preserves temperatures better")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("CUSTOM PADDING TRANSFORMATIONS - EXAMPLES")
    print("="*80)
    print()

    # Demo 1: Basic comparison
    demo_negated_padding()

    # Demo 2: Practical thermal example
    demo_practical_example()

    # Demo 3: Visualization
    visualize_padding_comparison()

    print("\n" + "="*80)
    print("Key Takeaway:")
    print("  The pad_func interface allows arbitrary transformations!")
    print("  - Negate for heat sink BCs")
    print("  - Scale for damping")
    print("  - Clamp for fixed value BCs")
    print("  - Custom logic for specialized physics")
    print("="*80)


if __name__ == "__main__":
    main()