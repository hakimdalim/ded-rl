"""
PaddingEngine Demo

Demonstrates all features of the configurable PaddingEngine.
"""

import numpy as np
import sys

from engines.convolution_engine.padding_engine.padding_engine import PaddingEngine, create_padding_engine


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def demo_basic_usage():
    """Demonstrate basic usage of PaddingEngine."""
    print_section("Basic Usage Demo")
    
    # Create engine
    engine = PaddingEngine(backend='numba_numpy')
    print(f"Created: {engine}\n")
    
    # Simple data
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    print(f"Original data: {data}")
    print(f"Data shape: {data.shape}\n")
    
    # Pad with different modes
    modes = ['reflect', 'symmetric', 'edge', 'constant', 'wrap']
    pad_width = 2
    
    print(f"Padding with pad_width={pad_width}:\n")
    for mode in modes:
        if mode == 'constant':
            padded = engine.pad(data, pad_width, mode=mode, constant_value=-1)
            print(f"  {mode:<10} (value=-1): {padded}")
        else:
            padded = engine.pad(data, pad_width, mode=mode)
            print(f"  {mode:<10}: {padded}")


def demo_all_backends():
    """Test all available backends."""
    print_section("All Backends Demo")
    
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    pad_width = 2
    mode = 'reflect'
    
    # Get available backends
    availability = PaddingEngine.list_available_backends()
    
    print(f"Testing mode='{mode}', pad_width={pad_width}")
    print(f"Original data: {data}\n")
    print(f"{'Backend':<20} {'Available':<12} {'Result':<30}")
    print("-" * 62)
    
    for backend in PaddingEngine.AVAILABLE_BACKENDS:
        available = availability[backend]
        
        if available:
            try:
                engine = PaddingEngine(backend=backend)
                result = engine.pad(data, pad_width, mode=mode)
                print(f"{backend:<20} {'✓ Yes':<12} {str(result):<30}")
            except Exception as e:
                print(f"{backend:<20} {'✓ Yes':<12} Error: {str(e)[:25]}")
        else:
            print(f"{backend:<20} {'✗ No':<12} {'N/A':<30}")


def demo_asymmetric_padding():
    """Demonstrate asymmetric padding."""
    print_section("Asymmetric Padding Demo")
    
    engine = PaddingEngine(backend='numpy_native')
    data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    
    print(f"Original data: {data}\n")
    
    # Different left/right padding
    configs = [
        (1, 3, 'reflect'),
        (3, 1, 'edge'),
        (0, 5, 'constant'),
        (2, 2, 'symmetric'),
    ]
    
    print(f"{'Left':<6} {'Right':<6} {'Mode':<12} {'Result':<35}")
    print("-" * 59)
    
    for left, right, mode in configs:
        if mode == 'constant':
            result = engine.pad(data, (left, right), mode=mode, constant_value=0)
        else:
            result = engine.pad(data, (left, right), mode=mode)
        print(f"{left:<6} {right:<6} {mode:<12} {str(result):<35}")


def demo_batched_padding():
    """Demonstrate padding on batched 2D data."""
    print_section("Batched (2D) Padding Demo")
    
    engine = PaddingEngine(backend='numba_numpy')
    
    # Create small batch
    data = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ], dtype=np.float32)
    
    print(f"Original data shape: {data.shape}")
    print("Original data:")
    print(data)
    print()
    
    # Pad
    padded = engine.pad(data, pad_width=1, mode='reflect')
    
    print(f"Padded data shape: {padded.shape}")
    print("Padded data (reflect, pad_width=1):")
    print(padded)



def demo_backend_availability():
    """Show which backends are available."""
    print_section("Backend Availability")
    
    availability = PaddingEngine.list_available_backends()
    
    print(f"{'Backend':<25} {'Status':<10}")
    print("-" * 35)
    
    for backend, available in availability.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"{backend:<25} {status:<10}")


def demo_modes_visualization():
    """Visualize all padding modes."""
    print_section("All Modes Visualization")
    
    engine = PaddingEngine(backend='numpy_native')
    data = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    pad_width = 3
    
    print(f"Original: {data}")
    print(f"Pad width: {pad_width}\n")
    
    modes = PaddingEngine.list_available_modes()
    
    for mode in modes:
        if mode == 'constant':
            result = engine.pad(data, pad_width, mode=mode, constant_value=0)
            desc = "(pads with 0)"
        else:
            result = engine.pad(data, pad_width, mode=mode)
            desc = ""
        
        print(f"{mode:<12} {desc:<20} {result}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("PADDING ENGINE - DEMO SUITE".center(70))
    print("="*70)
    
    demo_backend_availability()
    demo_basic_usage()
    demo_all_backends()
    demo_modes_visualization()
    demo_asymmetric_padding()
    demo_batched_padding()

    print("\n" + "="*70)
    print("Demo Complete!".center(70))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
