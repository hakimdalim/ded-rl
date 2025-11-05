"""
Configurable Padding Engine

Provides unified interface to all padding implementations with backend selection.
All 30 combinations of modes and backends are available.

Example:
    >>> engine = PaddingEngine(backend='numba_numpy')
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> padded = engine.pad(data, pad_width=2, mode='reflect')
    >>> padded
    array([3., 2., 1., 2., 3., 4., 5., 4., 3.])
"""

import numpy as np
from typing import Union, Tuple, Optional
import numpy.typing as npt

from engines.convolution_engine.padding_engine.padding_implementations import (
    PADDING_IMPLEMENTATIONS,
    get_padding_function,
    list_available_implementations
)


class PaddingEngine:
    """
    Configurable padding engine with multiple backend support.
    
    Supports 5 modes:
        - 'reflect': Reflect at boundaries (excludes edge)
        - 'symmetric': Mirror at boundaries (includes edge)
        - 'edge': Repeat edge values
        - 'constant': Pad with constant value
        - 'wrap': Circular/periodic wrapping
    
    Supports 6 backends:
        - 'numpy_native': Uses np.pad() (always available)
        - 'pytorch_cpu': Uses F.pad() on CPU
        - 'pytorch_gpu': Uses F.pad() on GPU (requires CUDA)
        - 'numba_pure': Pure Numba with manual loops
        - 'numba_numpy': Numba with NumPy functions
        - 'custom_vectorized': Custom fancy indexing
    
    Performance characteristics (empirical):
        Small data (<10k):  numba_pure/numba_numpy fastest
        Large data (>100k): numpy_native fastest
        GPU data:           pytorch_gpu fastest (if available)
    """
    
    AVAILABLE_MODES = ['reflect', 'symmetric', 'edge', 'constant', 'wrap']
    AVAILABLE_BACKENDS = ['numpy_native', 'pytorch_cpu', 'pytorch_gpu', 
                          'numba_pure', 'numba_numpy', 'custom_vectorized']
    
    def __init__(self, backend: str = 'numpy_native'):
        """
        Initialize padding engine.
        
        Args:
            backend: Backend to use for padding operations
                    'numpy_native' (default), 'pytorch_cpu', 'pytorch_gpu',
                    'numba_pure', 'numba_numpy', 'custom_vectorized'
        
        Raises:
            ValueError: If backend is not recognized
            ImportError: If backend requires unavailable dependencies
        """
        if backend not in self.AVAILABLE_BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Available backends: {self.AVAILABLE_BACKENDS}"
            )
        
        self.backend = backend
        self._validate_backend()
    
    def _validate_backend(self):
        """Validate that the selected backend is available."""
        backend = self.backend
        
        if backend in ['pytorch_cpu', 'pytorch_gpu']:
            try:
                import torch
                if backend == 'pytorch_gpu' and not torch.cuda.is_available():
                    raise RuntimeError(
                        f"{backend} backend requested but CUDA is not available. "
                        "Use 'pytorch_cpu' or ensure CUDA is properly installed."
                    )
            except ImportError:
                raise ImportError(
                    f"{backend} backend requires PyTorch. "
                    "Install with: pip install torch"
                )
        
        elif backend in ['numba_pure', 'numba_numpy']:
            try:
                import numba
            except ImportError:
                raise ImportError(
                    f"{backend} backend requires Numba. "
                    "Install with: pip install numba"
                )

    def pad(
            self,
            data: Union[npt.NDArray, 'torch.Tensor'],
            pad_width: Union[int, Tuple[int, int]],
            mode: str = 'reflect',
            constant_value: float = 0
    ) -> Union[npt.NDArray, 'torch.Tensor']:
        """
        Pad array along last axis.

        Args:
            data: Input array/tensor
                  - PyTorch backends: Any dimensionality (1D, 2D, 3D, 4D, ...)
                  - Other backends: 1D or 2D only
            pad_width: Padding width
                      - int: pad both sides equally
                      - (left, right): pad asymmetrically
            mode: Padding mode
                  'reflect', 'symmetric', 'edge', 'constant', 'wrap'
            constant_value: Value for constant mode (default 0)

        Returns:
            Padded array (same type as input)

        Raises:
            ValueError: If mode not supported
            TypeError: If data type not supported
            ValueError: If data dimensionality not supported by backend

        Examples:
            >>> # 1D example
            >>> engine = PaddingEngine(backend='numba_numpy')
            >>> data = np.array([1, 2, 3, 4, 5])
            >>> engine.pad(data, 2, mode='reflect')
            array([3., 2., 1., 2., 3., 4., 5., 4., 3.])

            >>> # 3D example (PyTorch only)
            >>> engine_pt = PaddingEngine(backend='pytorch_cpu')
            >>> data_3d = torch.randn(10, 5, 200)
            >>> padded_3d = engine_pt.pad(data_3d, 7, mode='reflect')
            >>> padded_3d.shape
            torch.Size([10, 5, 214])
        """
        # Validate mode
        if mode not in self.AVAILABLE_MODES:
            raise ValueError(
                f"Unknown mode '{mode}'. "
                f"Available modes: {self.AVAILABLE_MODES}"
            )

        # Parse pad_width
        if isinstance(pad_width, int):
            pad_left = pad_right = pad_width
        elif isinstance(pad_width, tuple) and len(pad_width) == 2:
            pad_left, pad_right = pad_width
        else:
            raise ValueError(
                f"pad_width must be int or (left, right) tuple, got {pad_width}"
            )

        # Validate data type
        is_tensor = hasattr(data, '__torch_function__')
        is_numpy = isinstance(data, np.ndarray)

        if not (is_numpy or is_tensor):
            raise TypeError(
                f"Data must be numpy array or torch tensor, got {type(data)}"
            )

        # Get padding function
        try:
            pad_func = get_padding_function(mode, self.backend)
        except KeyError:
            raise ValueError(
                f"Combination mode='{mode}' and backend='{self.backend}' not available"
            )

        # === KEY CHANGE: Handle arbitrary dimensions for PyTorch backends ===

        # PyTorch backends support arbitrary dimensions natively
        if self.backend in ['pytorch_cpu', 'pytorch_gpu']:
            # Direct call - no dimension restrictions
            return pad_func(data, pad_left, pad_right, constant_value)

        # Other backends (numpy, numba, custom) only support 1D
        # Apply padding row-by-row for multi-dimensional data
        if data.ndim == 1:
            # Single vector
            return pad_func(data, pad_left, pad_right, constant_value)

        elif data.ndim == 2:
            # Apply padding to each row
            results = []
            for row in data:
                padded_row = pad_func(row, pad_left, pad_right, constant_value)
                results.append(padded_row)

            # Stack results
            if is_tensor:
                import torch
                return torch.stack(results)
            else:
                return np.array(results)

        else:
            raise ValueError(
                f"{self.backend} backend only supports 1D or 2D data. "
                f"Got {data.ndim}D with shape {data.shape}. "
                f"Use 'pytorch_cpu' or 'pytorch_gpu' backend for arbitrary dimensions."
        )

    @staticmethod
    def list_available_backends() -> dict:
        """
        List which backends are available on this system.
        
        Returns:
            Dict mapping backend name to availability (bool)
        
        Example:
            >>> PaddingEngine.list_available_backends()
            {'numpy_native': True,
             'pytorch_cpu': True,
             'pytorch_gpu': False,
             'numba_pure': True,
             'numba_numpy': True,
             'custom_vectorized': True}
        """
        availability = {}
        
        # NumPy and custom always available
        availability['numpy_native'] = True
        availability['custom_vectorized'] = True
        
        # Check PyTorch
        try:
            import torch
            availability['pytorch_cpu'] = True
            availability['pytorch_gpu'] = torch.cuda.is_available()
        except ImportError:
            availability['pytorch_cpu'] = False
            availability['pytorch_gpu'] = False
        
        # Check Numba
        try:
            import numba
            availability['numba_pure'] = True
            availability['numba_numpy'] = True
        except ImportError:
            availability['numba_pure'] = False
            availability['numba_numpy'] = False
        
        return availability
    
    @staticmethod
    def list_available_modes() -> list:
        """
        List all available padding modes.
        
        Returns:
            List of mode names
        """
        return PaddingEngine.AVAILABLE_MODES.copy()
    
    def __repr__(self):
        return f"PaddingEngine(backend='{self.backend}')"
    
    def __str__(self):
        return (f"PaddingEngine\n"
                f"  Backend: {self.backend}\n"
                f"  Available modes: {', '.join(self.AVAILABLE_MODES)}")
