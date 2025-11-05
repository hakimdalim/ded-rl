"""
Convolution Engine

Main engine class that provides a unified interface for convolution operations
across multiple backends.
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, Type, Optional

from engines.convolution_engine.base import ConvolutionBackend
from engines.convolution_engine.backends.numpy_backend import NumpyBackend
from engines.convolution_engine.backends.scipy_backend import ScipyBackend
from engines.convolution_engine.backends.numba_backend import NumbaBackend
from engines.convolution_engine.backends.pytorch_backend import PyTorchBackend


# Registry of all available backend classes
BACKEND_REGISTRY: Dict[str, Type[ConvolutionBackend]] = {
    'numpy': NumpyBackend,
    'scipy': ScipyBackend,
    'numba': NumbaBackend,
    'pytorch_cpu': PyTorchBackend,
    'pytorch_gpu': PyTorchBackend,
}


class ConvolutionEngine:
    """
    Flexible convolution engine with multiple backend support.

    This engine provides a unified interface for 1D convolution operations across multiple
    computational backends. All backends implement "valid" mode convolution (no padding),
    where output_size = input_size - kernel_size + 1.

    Core Behavior:
        - Convolution Type: 1D directional convolution (not 2D/3D spatial convolution)
        - Padding Mode: Valid (no padding) - output is smaller than input
        - Output Size: input_size - kernel_size + 1 along convolution axis
        - Kernel: Must be 1D array, typically Gaussian or other smoothing filter
        - Data Types: NumPy arrays for most backends; PyTorch tensors also supported by pytorch backends

    Backend Capabilities:
        ┌──────────────┬──────────┬──────────┬─────┬──────────────────────────────┐
        │ Backend      │ Dims     │ Batching │ GPU │ Notes                        │
        ├──────────────┼──────────┼──────────┼─────┼──────────────────────────────┤
        │ numpy        │ 1D       │ No       │ No  │ Pure NumPy, always available │
        │ scipy        │ 1D       │ No       │ No  │ scipy.ndimage.correlate1d    │
        │ numba        │ 1D, 2D   │ Yes      │ No  │ JIT-compiled, parallel (CPU) │
        │ pytorch_cpu  │ Any      │ Yes      │ No  │ PyTorch on CPU, flexible     │
        │ pytorch_gpu  │ Any      │ Yes      │ Yes │ PyTorch on CUDA, fastest     │
        └──────────────┴──────────┴──────────┴─────┴──────────────────────────────┘

    Dimensionality Notes:
        - 1D: Single vector (size,) - convolves along the only axis
        - 2D: Batch of vectors (batch, size) - convolves along last axis for each vector
        - 3D+: Higher-dimensional arrays (dim1, dim2, ..., size) - always convolves along LAST axis
        - PyTorch backends support arbitrary dimensions (limited only by memory)

        Important: This is NOT spatial 2D/3D convolution! It's 1D convolution applied to
        multi-dimensional data, always operating along the last axis.

    Performance Guide:
        - CPU batches: Use 'numba' (~3x faster than numpy/scipy, verified in benchmarks)
        - GPU batches: Use 'pytorch_gpu' (~10-20x faster than CPU for large batches, verified)
        - Single vectors (1D): NumPy or SciPy recommended for simplicity (not benchmarked,
          but overhead from JIT compilation or GPU transfer likely outweighs benefits)
        - We suspect small kernels (<10 elements) have minimal performance differences across
          CPU backends, while large kernels (>20 elements) likely benefit more from numba/GPU
          (not systematically benchmarked)

    Input/Output Behavior:
        - Input: NumPy array or PyTorch tensor (backend-dependent)
        - Output: Same type as input (numpy→numpy, tensor→tensor for pytorch backends)
        - Shape: Output has same dimensionality, last axis reduced by (kernel_size - 1)

        Examples:
            Input (100,) + kernel (9,) → Output (92,)
            Input (50, 1000) + kernel (17,) → Output (50, 984)
            Input (10, 20, 500) + kernel (21,) → Output (10, 20, 480)

    Typical Workflow:
        1. User handles boundary padding (e.g., reflection at mask edges)
        2. Engine performs pure convolution (valid mode, no padding)
        3. User extracts/places result back into original data structure

        This separation allows custom boundary handling while keeping convolution fast.

    Example Usage:
        >>> # Basic usage
        >>> engine = ConvolutionEngine(backend='numba')
        >>> data = np.random.randn(100, 1000)  # 100 vectors of length 1000
        >>> kernel = np.array([0.25, 0.5, 0.25])  # 3-element kernel
        >>> result = engine.convolve(data, kernel)
        >>> result.shape  # (100, 998) - reduced by 2 elements per vector

        >>> # With PyTorch tensors
        >>> import torch
        >>> engine = ConvolutionEngine(backend='pytorch_cpu')
        >>> data_tensor = torch.randn(100, 1000)
        >>> kernel_tensor = torch.tensor([0.25, 0.5, 0.25])
        >>> result_tensor = engine.convolve(data_tensor, kernel_tensor)
        >>> type(result_tensor)  # torch.Tensor

        >>> # Check backend availability
        >>> ConvolutionEngine.list_available_backends()
        {'numpy': True, 'scipy': True, 'numba': True,
         'pytorch_cpu': True, 'pytorch_gpu': False}

    Design Philosophy:
        - No silent failures: Explicit errors for unsupported operations
        - No automatic fallbacks: User must handle backend unavailability
        - Consistent interface: All backends expose identical API
        - Type preservation: Output type matches input type (for PyTorch)
        - Performance transparency: User chooses backend based on needs
    """
    
    def __init__(self, backend: str = 'numba', **backend_kwargs):
        """
        Initialize convolution engine with specified backend.
        
        Args:
            backend: Backend name ('numpy', 'scipy', 'numba', 'pytorch_cpu', 'pytorch_gpu')
            **backend_kwargs: Additional arguments passed to backend constructor
                For PyTorch backends: device is set automatically based on backend name
        
        Raises:
            ValueError: If backend name is invalid or backend is not available
        """
        if backend not in BACKEND_REGISTRY:
            available = list(BACKEND_REGISTRY.keys())
            raise ValueError(
                f"Unknown backend '{backend}'. Available backends: {available}"
            )
        
        backend_class = BACKEND_REGISTRY[backend]
        
        # Check if backend is available
        if not backend_class.is_available():
            raise ValueError(
                f"Backend '{backend}' is not available. "
                f"Required dependencies may not be installed."
            )
        
        # Special handling for PyTorch backends
        if backend.startswith('pytorch'):
            device = 'cuda' if backend == 'pytorch_gpu' else 'cpu'
            if device == 'cuda' and not PyTorchBackend.is_cuda_available():
                raise ValueError(
                    "pytorch_gpu backend requested but CUDA is not available. "
                    "Use 'pytorch_cpu' or ensure CUDA is properly installed."
                )
            self._backend = backend_class(device=device, **backend_kwargs)
        else:
            self._backend = backend_class(**backend_kwargs)
        
        self._backend_name = backend
    
    @property
    def backend_name(self) -> str:
        """Get the name of the current backend"""
        return self._backend_name
    
    @property
    def supported_dims(self) -> tuple:
        """Get tuple of supported input dimensions for current backend"""
        return self._backend.supported_dims
    
    def convolve(self, data: npt.NDArray, kernel: npt.NDArray, **kwargs) -> npt.NDArray:
        """
        Apply 1D convolution to data using the configured backend.
        
        Args:
            data: Input array. Supported shapes depend on backend:
                - numpy/scipy: (size,)
                - numba: (size,) or (batch, size)
                - pytorch: (size,), (batch, size), or higher dimensions
            kernel: 1D convolution kernel of shape (kernel_size,)
        
        Returns:
            Convolved result. Shape depends on input and backend:
                - Output length along convolution axis: size - kernel_size + 1
                - Other dimensions preserved
        
        Raises:
            TypeError: If inputs are not numpy arrays
            ValueError: If kernel is not 1D
            ValueError: If data dimensionality is not supported by backend
        
        Example:
            >>> engine = ConvolutionEngine('numba')
            >>> data = np.random.randn(10, 1000)
            >>> kernel = np.array([0.25, 0.5, 0.25])
            >>> result = engine.convolve(data, kernel)
            >>> result.shape
            (10, 998)
        """
        return self._backend.convolve(data, kernel, **kwargs)
    
    @staticmethod
    def list_available_backends() -> Dict[str, bool]:
        """
        List all backends and their availability on current system.
        
        Returns:
            Dictionary mapping backend names to availability status
        
        Example:
            >>> ConvolutionEngine.list_available_backends()
            {
                'numpy': True,
                'scipy': True,
                'numba': True,
                'pytorch_cpu': True,
                'pytorch_gpu': False
            }
        """
        availability = {}
        
        for name, backend_class in BACKEND_REGISTRY.items():
            if name == 'pytorch_gpu':
                # Special check for GPU availability
                availability[name] = PyTorchBackend.is_cuda_available()
            else:
                availability[name] = backend_class.is_available()
        
        return availability
    
    @staticmethod
    def get_backend_info(backend: str) -> Dict[str, any]:
        """
        Get detailed information about a specific backend.
        
        Args:
            backend: Backend name
        
        Returns:
            Dictionary with backend information including:
                - name: Backend name
                - available: Whether backend is available
                - supported_dims: Tuple of supported input dimensions
                - class: Backend class
        
        Raises:
            ValueError: If backend name is invalid
        """
        if backend not in BACKEND_REGISTRY:
            available = list(BACKEND_REGISTRY.keys())
            raise ValueError(
                f"Unknown backend '{backend}'. Available backends: {available}"
            )
        
        backend_class = BACKEND_REGISTRY[backend]
        
        # Create temporary instance to get supported_dims
        # (for PyTorch, use CPU to avoid CUDA requirement)
        if backend.startswith('pytorch'):
            device = 'cpu'  # Always use CPU for info query
            try:
                temp_instance = backend_class(device=device)
                supported_dims = temp_instance.supported_dims
            except:
                supported_dims = None
        else:
            try:
                temp_instance = backend_class()
                supported_dims = temp_instance.supported_dims
            except:
                supported_dims = None
        
        # Check availability
        if backend == 'pytorch_gpu':
            available = PyTorchBackend.is_cuda_available()
        else:
            available = backend_class.is_available()
        
        return {
            'name': backend,
            'available': available,
            'supported_dims': supported_dims,
            'class': backend_class,
        }
    
    def __repr__(self) -> str:
        return f"ConvolutionEngine(backend='{self._backend_name}')"
