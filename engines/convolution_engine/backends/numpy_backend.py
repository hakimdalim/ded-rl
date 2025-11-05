"""
NumPy Convolution Backend

Pure NumPy implementation using np.convolve.
Only supports single 1D vectors (no batching).
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple

from engines.convolution_engine.base import ConvolutionBackend


class NumpyBackend(ConvolutionBackend):
    """
    NumPy convolution backend.
    
    Uses np.convolve with mode='valid' (no padding).
    Only supports 1D input arrays.
    """
    
    @property
    def name(self) -> str:
        return "numpy"
    
    @property
    def supported_dims(self) -> Tuple[int, ...]:
        return (1,)  # Only 1D arrays
    
    def convolve(self, data: npt.NDArray, kernel: npt.NDArray, **kwargs) -> npt.NDArray:
        """
        Apply 1D convolution using NumPy.
        
        Args:
            data: 1D input array of shape (size,)
            kernel: 1D kernel array
            
        Returns:
            Convolved array of shape (size - kernel_size + 1,)
            
        Raises:
            ValueError: If data is not 1D
        """
        self.validate_inputs(data, kernel)

        if not 'mode' in kwargs:
            kwargs['mode'] = 'valid'
        
        return np.convolve(data, kernel, **kwargs)
    
    @classmethod
    def is_available(cls) -> bool:
        """NumPy is always available (required dependency)"""
        return True
