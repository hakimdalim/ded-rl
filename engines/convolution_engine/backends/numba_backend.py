"""
Numba Convolution Backend

JIT-compiled implementation using Numba.
Supports 1D and 2D (batched) arrays with parallel processing.
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple

from engines.convolution_engine.base import ConvolutionBackend

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args, **kwargs):
        return range(*args, **kwargs)


@jit(nopython=True)
def _numba_convolve_1d(data: npt.NDArray, kernel: npt.NDArray) -> npt.NDArray:
    """Numba-compiled 1D convolution"""
    n = len(data) - len(kernel) + 1
    result = np.empty(n, dtype=data.dtype)
    kern_len = len(kernel)
    
    for i in range(n):
        acc = 0.0
        for j in range(kern_len):
            acc += data[i + j] * kernel[j]
        result[i] = acc
    
    return result


@jit(nopython=True, parallel=True)
def _numba_convolve_batch(data_batch: npt.NDArray, kernel: npt.NDArray) -> npt.NDArray:
    """Numba-compiled batch convolution with parallel processing"""
    batch_size, size = data_batch.shape
    n = size - len(kernel) + 1
    result = np.empty((batch_size, n), dtype=data_batch.dtype)
    kern_len = len(kernel)
    
    for b in prange(batch_size):
        for i in range(n):
            acc = 0.0
            for j in range(kern_len):
                acc += data_batch[b, i + j] * kernel[j]
            result[b, i] = acc
    
    return result


class NumbaBackend(ConvolutionBackend):
    """
    Numba convolution backend.
    
    JIT-compiled implementation with parallel processing for batches.
    Supports 1D and 2D arrays.
    """
    
    @property
    def name(self) -> str:
        return "numba"
    
    @property
    def supported_dims(self) -> Tuple[int, ...]:
        return (1, 2)  # 1D single vectors and 2D batches
    
    def convolve(self, data: npt.NDArray, kernel: npt.NDArray) -> npt.NDArray:
        """
        Apply 1D convolution using Numba.
        
        Args:
            data: Input array of shape (size,) or (batch, size)
            kernel: 1D kernel array
            
        Returns:
            For 1D input (size,): returns (size - kernel_size + 1,)
            For 2D input (batch, size): returns (batch, size - kernel_size + 1)
            
        Raises:
            ValueError: If data is not 1D or 2D
        """
        self.validate_inputs(data, kernel)
        
        if data.ndim == 1:
            return _numba_convolve_1d(data, kernel)
        elif data.ndim == 2:
            return _numba_convolve_batch(data, kernel)
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if numba is installed"""
        return NUMBA_AVAILABLE
