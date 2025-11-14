"""
Base Backend Abstract Class

Defines the interface that all convolution backends must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Tuple


class ConvolutionBackend(ABC):
    """
    Abstract base class for convolution backends.
    
    All backend implementations must inherit from this class and implement
    the required methods.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier"""
        pass
    
    @property
    @abstractmethod
    def supported_dims(self) -> Tuple[int, ...]:
        """
        Tuple of supported input array dimensions.
        
        Examples:
            (1,) - only 1D arrays
            (1, 2) - 1D and 2D arrays (batched)
            (1, 2, 3, 4, 5) - 1D through 5D arrays
        """
        pass
    
    @abstractmethod
    def convolve(self, data: npt.NDArray, kernel: npt.NDArray) -> npt.NDArray:
        """
        Apply 1D convolution to data using kernel.
        
        Args:
            data: Input array of any supported dimensionality
            kernel: 1D convolution kernel
            
        Returns:
            Convolved result. Output shape depends on backend padding behavior.
            
        Raises:
            ValueError: If data.ndim not in self.supported_dims
            ValueError: If kernel is not 1D
            TypeError: If inputs are not numpy arrays
        """
        pass
    
    def validate_inputs(self, data: npt.NDArray, kernel: npt.NDArray) -> None:
        """
        Validate input data and kernel.
        
        Args:
            data: Input array
            kernel: Convolution kernel
            
        Raises:
            TypeError: If inputs are not numpy arrays
            ValueError: If kernel is not 1D
            ValueError: If data dimensionality is not supported
        """
        # Type checking
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Data must be numpy array, got {type(data)}")
        
        if not isinstance(kernel, np.ndarray):
            raise TypeError(f"Kernel must be numpy array, got {type(kernel)}")
        
        # Kernel must be 1D
        if kernel.ndim != 1:
            raise ValueError(f"Kernel must be 1D array, got shape {kernel.shape}")
        
        # Check dimensionality support
        if data.ndim not in self.supported_dims:
            raise ValueError(
                f"{self.name} backend does not support {data.ndim}D input. "
                f"Supported dimensions: {self.supported_dims}"
            )
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if this backend is available on the current system.
        
        Returns:
            True if backend dependencies are installed and functional
        """
        try:
            # Subclasses can override this to check for specific dependencies
            return True
        except ImportError:
            return False
