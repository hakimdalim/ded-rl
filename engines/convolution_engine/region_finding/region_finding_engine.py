"""
Region Finding Engine

Provides unified interface to region finding implementations with backend selection.
Wrapper around region_finding_implementations.py for consistency with other engines.

Example:
    >>> engine = RegionFindingEngine(backend='numba')
    >>> mask = np.array([True, True, False, True, True, True])
    >>> regions = engine.find_regions(mask, min_length=2)
    >>> regions
    array([[0, 2],
           [3, 6]])
"""

import numpy as np
from typing import Union, List, Optional
import numpy.typing as npt

from engines.convolution_engine.region_finding.region_finding_implementations import (
    find_regions,
    find_regions_numpy,
    find_regions_numba,
    find_regions_torch,
    find_regions_python,
    find_regions_scipy,
    list_available_finders,
)


class RegionFindingEngine:
    """
    Configurable region finding engine with multiple backend support.
    
    Finds contiguous True regions in boolean masks. Useful for identifying
    valid (non-masked) segments in time series data.
    
    Supports 5 backends:
        - 'numpy': Vectorized NumPy operations (default, always available)
        - 'numba': JIT-compiled, fastest for most cases
        - 'torch': PyTorch tensor operations
        - 'python': Pure Python (slow, for reference)
        - 'scipy': SciPy ndimage.label (slow, not recommended)
    
    Output Format:
        Single signal (1D):    np.array([[start, end], ...]), shape (n_regions, 2)
        Batched (N-D):         Nested list of arrays matching batch dimensions
    
    Performance characteristics (100k elements):
        - numpy: 0.2-0.3ms (recommended, always works)
        - numba: 0.06-0.4ms (fastest single signals, 3× faster batches)
        - torch: 1-4ms (slower, use only if you need PyTorch tensors)
        - python: 8-9ms (slow, reference only)
        - scipy: 7-58ms (very slow, not recommended)
    
    Example:
        >>> engine = RegionFindingEngine(backend='numba')
        >>> 
        >>> # Single signal
        >>> mask = np.array([True, True, False, True, True, True])
        >>> regions = engine.find_regions(mask, min_length=2)
        >>> regions
        array([[0, 2],
               [3, 6]])
        >>> 
        >>> # Batched
        >>> masks = np.random.rand(100, 10000) < 0.5
        >>> results = engine.find_regions(masks, min_length=5)
        >>> len(results)  # 100 arrays, one per signal
        100
    """
    
    AVAILABLE_BACKENDS = ['numpy', 'numba', 'torch', 'python', 'scipy']
    DEFAULT_BACKEND = 'numpy'
    
    def __init__(self, backend: str = 'numpy'):
        """
        Initialize region finding engine.
        
        Args:
            backend: Backend to use for region finding operations
                    'numpy' (default), 'numba', 'torch', 'python', 'scipy'
        
        Raises:
            ValueError: If backend is not recognized
            ImportError: If backend requires unavailable dependencies
        
        Example:
            >>> engine = RegionFindingEngine(backend='numba')
        """
        if backend not in self.AVAILABLE_BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Available backends: {self.AVAILABLE_BACKENDS}"
            )
        
        self.backend = backend
        self._validate_backend()
        self._backend_func = self._get_backend_function()
    
    def _validate_backend(self):
        """Validate that the selected backend is available."""
        available = list_available_finders()
        
        if not available[self.backend]:
            raise ImportError(
                f"Backend '{self.backend}' is not available. "
                f"Please install required dependencies.\n"
                f"Available backends: {[k for k, v in available.items() if v]}"
            )
    
    def _get_backend_function(self):
        """Get the backend-specific function."""
        backend_map = {
            'numpy': find_regions_numpy,
            'numba': find_regions_numba,
            'torch': find_regions_torch,
            'python': find_regions_python,
            'scipy': find_regions_scipy,
        }
        return backend_map[self.backend]
    
    def find_regions(
        self, 
        mask: Union[np.ndarray, List], 
        min_length: int,
        **kwargs
    ) -> Union[np.ndarray, List]:
        """
        Find contiguous True regions in boolean mask.
        
        Args:
            mask: Boolean array, shape (..., n)
                  True = valid, False = masked/invalid
                  Any number of dimensions supported (regions found along last axis)
            min_length: Minimum region length to include (must be >= 1)
            **kwargs: Backend-specific options:
                     - return_torch: Convert output to PyTorch tensors
                     - return_numpy: Convert output to NumPy arrays
        
        Returns:
            Single signal (1D input):
                np.array([[start, end], ...]), shape (n_regions, 2), dtype=int64
                Empty: shape (0, 2)
            
            Batched (N-D input, N > 1):
                Nested list of arrays matching batch dimensions
                Each element is array of shape (n_regions, 2)
        
        Raises:
            ValueError: If min_length < 1 or last dimension < min_length
        
        Example:
            >>> # Single signal
            >>> mask = np.array([True, True, False, True, True, True])
            >>> regions = engine.find_regions(mask, min_length=2)
            >>> regions
            array([[0, 2],
                   [3, 6]])
            >>> 
            >>> # Process each region
            >>> for i in range(len(regions)):
            ...     start, end = regions[i, 0], regions[i, 1]
            ...     segment = signal[start:end]
            ...     # ... process segment
            >>> 
            >>> # Batched (2D input)
            >>> masks = np.array([[True, True, False, True],
            ...                    [False, True, True, True]])
            >>> results = engine.find_regions(masks, min_length=2)
            >>> results
            [array([[0, 2], [3, 4]]), array([[1, 4]])]
            >>> 
            >>> # Batched (3D input)
            >>> masks_3d = np.random.rand(10, 5, 100) < 0.5
            >>> results_3d = engine.find_regions(masks_3d, min_length=3)
            >>> # Returns nested list: 10 lists, each with 5 arrays
        """
        return self._backend_func(mask, min_length, **kwargs)
    
    def __call__(
        self,
        mask: Union[np.ndarray, List],
        min_length: int,
        **kwargs
    ) -> Union[np.ndarray, List]:
        """
        Callable interface (shorthand for find_regions).
        
        Args:
            mask: Boolean array
            min_length: Minimum region length
            **kwargs: Backend-specific options
        
        Returns:
            Region array(s)
        
        Example:
            >>> engine = RegionFindingEngine()
            >>> regions = engine(mask, min_length=5)  # Same as engine.find_regions()
        """
        return self.find_regions(mask, min_length, **kwargs)
    
    def get_info(self) -> dict:
        """
        Get information about the current backend.
        
        Returns:
            Dictionary with backend information
        
        Example:
            >>> engine = RegionFindingEngine(backend='numba')
            >>> info = engine.get_info()
            >>> info['backend']
            'numba'
        """
        available = list_available_finders()
        
        return {
            'backend': self.backend,
            'available': available[self.backend],
            'all_available_backends': [k for k, v in available.items() if v],
        }
    
    @staticmethod
    def list_available_backends() -> dict:
        """
        List all available backends on this system.
        
        Returns:
            Dictionary mapping backend name to availability (bool)
        
        Example:
            >>> RegionFindingEngine.list_available_backends()
            {'numpy': True, 'numba': True, 'torch': False, 'python': True, 'scipy': True}
        """
        return list_available_finders()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"RegionFindingEngine(backend='{self.backend}')"


# Convenience function for quick use without engine instantiation
def find_regions_quick(
    mask: Union[np.ndarray, List],
    min_length: int,
    backend: str = 'numpy',
    **kwargs
) -> Union[np.ndarray, List]:
    """
    Quick region finding without creating an engine instance.
    
    Convenience function that creates an engine, finds regions, and returns.
    For repeated use, create an engine instance instead.
    
    Args:
        mask: Boolean array
        min_length: Minimum region length
        backend: Backend to use (default: 'numpy')
        **kwargs: Backend-specific options
    
    Returns:
        Region array(s)
    
    Example:
        >>> from region_finding_engine import find_regions_quick
        >>> regions = find_regions_quick(mask, min_length=5, backend='numba')
    """
    engine = RegionFindingEngine(backend=backend)
    return engine.find_regions(mask, min_length, **kwargs)
