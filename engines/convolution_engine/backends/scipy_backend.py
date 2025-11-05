"""
SciPy Convolution Backend

Implementation using scipy.ndimage.correlate1d.
Only supports single 1D vectors (no batching).
"""

import numpy.typing as npt
from typing import Tuple

from engines.convolution_engine.base import ConvolutionBackend


class ScipyBackend(ConvolutionBackend):
    """
    SciPy convolution backend.
    
    Uses scipy.ndimage.correlate1d with mode='constant'.
    Only supports 1D input arrays.
    Returns same size as input (internally pads with zeros).
    """
    
    @property
    def name(self) -> str:
        return "scipy"
    
    @property
    def supported_dims(self) -> Tuple[int, ...]:
        return (1,)  # Only 1D arrays
    
    def convolve(self, data: npt.NDArray, kernel: npt.NDArray, **kwargs) -> npt.NDArray:
        """
        Apply 1D convolution using SciPy.
        
        Args:
            data: 1D input array of shape (size,)
            kernel: 1D kernel array
            
        Returns:
            Convolved array of shape (size - kernel_size + 1,)
            Note: We extract the valid region from scipy's output
            
        Raises:
            ValueError: If data is not 1D
        """
        from scipy.ndimage import correlate1d
        
        self.validate_inputs(data, kernel)

        if not 'mode' in kwargs:
            kwargs['mode'] = 'constant'

        # scipy returns same size as input, extract valid region
        full_result = correlate1d(data, kernel, **kwargs)
        radius = len(kernel) // 2
        n = len(data) - len(kernel) + 1
        
        return full_result[radius:radius + n]
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if scipy is installed"""
        try:
            import scipy.ndimage
            return True
        except ImportError:
            return False
