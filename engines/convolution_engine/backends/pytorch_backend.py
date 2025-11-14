"""
PyTorch Convolution Backend

Implementation using torch.nn.functional.conv1d.
Supports arbitrary dimensionality (PyTorch handles internally).
Can run on CPU or GPU.
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple

from engines.convolution_engine.base import ConvolutionBackend


class PyTorchBackend(ConvolutionBackend):
    """
    PyTorch convolution backend.

    Uses torch.nn.functional.conv1d with native batch support.
    Supports 1D, 2D, 3D, and higher dimensional arrays.
    Can run on CPU or GPU.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize PyTorch backend.
        
        Args:
            device: 'cpu' or 'cuda' (GPU)
            
        Raises:
            ValueError: If device='cuda' but CUDA is not available
        """
        import torch

        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")
        
        self.device = device
        self._torch = torch
    
    @property
    def name(self) -> str:
        return f"pytorch_{self.device}"

    @property
    def supported_dims(self) -> Tuple[int, ...]:
        # PyTorch supports arbitrary dimensions (no practical limit)
        # Return None or a sentinel to indicate "unlimited"
        return None  # Special case: None means all dimensions supported

    def validate_inputs(self, data, kernel) -> None:
        """
        Validate input data and kernel.

        PyTorch backend accepts both numpy arrays and torch tensors.
        """
        import torch

        # Type checking - accept both numpy and torch
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            raise TypeError(f"Data must be numpy array or torch tensor, got {type(data)}")

        if not isinstance(kernel, (np.ndarray, torch.Tensor)):
            raise TypeError(f"Kernel must be numpy array or torch tensor, got {type(kernel)}")

        # Check kernel dimensionality
        if kernel.ndim != 1:
            raise ValueError(f"Kernel must be 1D array, got shape {kernel.shape}")

        # No dimensionality check for data - PyTorch supports all dimensions
    def convolve(self, data, kernel, **kwargs):
        """Apply 1D convolution using PyTorch."""
        import torch
        import torch.nn.functional as F  # ADD THIS LINE

        self.validate_inputs(data, kernel)

        # Convert to numpy if needed for shape handling
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
            input_was_tensor = True
        else:
            data_np = data
            input_was_tensor = False

        original_shape = data_np.shape
        original_ndim = data_np.ndim

        # Prepare data for conv1d: needs (batch, channels, length)
        if original_ndim == 1:
            data_reshaped = data_np.reshape(1, 1, -1)
        elif original_ndim == 2:
            data_reshaped = data_np.reshape(data_np.shape[0], 1, -1)
        else:
            batch_size = np.prod(original_shape[:-1])
            data_reshaped = data_np.reshape(batch_size, 1, -1)

        # Convert to torch tensors
        data_t = torch.from_numpy(data_reshaped).float().to(self.device)

        if isinstance(kernel, torch.Tensor):
            kernel_t = kernel.reshape(1, 1, -1).float().to(self.device)
        else:
            kernel_t = torch.from_numpy(kernel).reshape(1, 1, -1).float().to(self.device)

        if not 'padding' in kwargs:
            kwargs['padding'] = 0

        # Perform convolution
        result_t = F.conv1d(data_t, kernel_t, **kwargs)

        # Reshape back
        if original_ndim == 1:
            result_t = result_t.squeeze()
        elif original_ndim == 2:
            result_t = result_t.squeeze(1)
        else:
            new_length = result_t.shape[-1]
            new_shape = original_shape[:-1] + (new_length,)
            result_t = result_t.reshape(new_shape)

        # Return tensor if input was tensor, numpy if input was numpy
        if input_was_tensor:
            return result_t
        else:
            return result_t.cpu().numpy()
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if PyTorch is installed"""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    @classmethod
    def is_cuda_available(cls) -> bool:
        """Check if CUDA/GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
