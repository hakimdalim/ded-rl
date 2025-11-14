"""
Comprehensive Padding Implementations

All possible padding implementations for empirical benchmarking.
Each implementation follows the pattern: pad_MODE_BACKEND

Modes: reflect, symmetric, edge, constant, wrap
Backends: numpy_native, pytorch_cpu, pytorch_gpu, numba_pure, numba_numpy, custom_vectorized

Function signature:
    pad_MODE_BACKEND(data, pad_left, pad_right, constant_value=0) -> padded_array

Design Philosophy:
    - NO silent fallbacks: Raise explicit errors if backend unavailable
    - Support arbitrary dimensions for PyTorch (like convolution engine)
    - Preserve input type (numpy→numpy, tensor→tensor)
    - Padding always operates on LAST axis (consistent with convolution)
    - Clean, readable implementations without excessive squeeze/unsqueeze
"""

import numpy as np

# ============================================================================
# DEPENDENCY CHECKS - Fail early with clear errors
# ============================================================================

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def _check_torch_available():
    """Raise error if PyTorch not available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch backend requested but PyTorch is not installed. "
            "Install with: pip install torch"
        )


def _check_torch_cuda_available():
    """Raise error if CUDA not available."""
    _check_torch_available()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA backend requested but CUDA is not available. "
            "Use pytorch_cpu backend or ensure CUDA is properly installed."
        )


def _check_numba_available():
    """Raise error if Numba not available."""
    if not NUMBA_AVAILABLE:
        raise ImportError(
            "Numba backend requested but Numba is not installed. "
            "Install with: pip install numba"
        )


# ============================================================================
# PYTORCH UTILITIES - Support arbitrary dimensions
# ============================================================================

def _pad_pytorch_last_axis(data, pad_left, pad_right, mode, value=0, device='cpu'):
    """
    Pad along last axis for arbitrary-dimensional PyTorch tensors.

    Workaround for PyTorch limitation: F.pad with 2-element padding spec
    only supports up to 3D input. We reshape to 3D, pad, then reshape back.

    Args:
        data: numpy array or torch tensor (any dimensionality)
        pad_left: left padding amount
        pad_right: right padding amount
        mode: PyTorch padding mode ('constant', 'reflect', 'replicate', 'circular')
        value: constant value for mode='constant'
        device: 'cpu' or 'cuda'

    Returns:
        Padded array/tensor (same type as input)
    """
    # Track input type and shape
    input_is_numpy = isinstance(data, np.ndarray)
    original_shape = data.shape
    original_ndim = data.ndim

    # Convert to tensor if needed
    if input_is_numpy:
        data_t = torch.from_numpy(data).float().to(device)
    else:
        data_t = data.to(device) if hasattr(data, 'is_cuda') and not data.is_cuda else data

    # Reshape to 3D for padding (batch, 1, length)
    # This allows us to use F.pad with pad=(left, right) on any dimensionality
    if original_ndim == 1:
        # (n,) → (1, 1, n)
        data_t = data_t.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 2:
        # (batch, n) → (batch, 1, n)
        data_t = data_t.unsqueeze(1)
    else:
        # (d1, d2, ..., dn) → (d1*d2*...*d(n-1), 1, dn)
        batch_size = int(np.prod(original_shape[:-1]))
        data_t = data_t.reshape(batch_size, 1, -1)

    # Pad last dimension
    padding = (pad_left, pad_right)
    if mode == 'constant':
        result = F.pad(data_t, padding, mode=mode, value=value)
    else:
        result = F.pad(data_t, padding, mode=mode)

    # Reshape back to original dimensionality
    new_last_dim = original_shape[-1] + pad_left + pad_right
    if original_ndim == 1:
        result = result.squeeze(0).squeeze(0)
    elif original_ndim == 2:
        result = result.squeeze(1)
    else:
        new_shape = original_shape[:-1] + (new_last_dim,)
        result = result.reshape(new_shape)

    # Convert back to numpy if input was numpy
    return result.cpu().numpy() if input_is_numpy else result


def _reflect_manual_pytorch(data, pad_left, pad_right, device='cpu'):
    """
    Manual reflect implementation for PyTorch (excludes edge).

    PyTorch's 'reflect' mode is actually NumPy's 'symmetric' (includes edge).
    We need to manually implement NumPy's 'reflect' (excludes edge) using indexing.

    Supports arbitrary dimensions - always operates on last axis.
    """
    _check_torch_available()
    if device == 'cuda':
        _check_torch_cuda_available()

    input_is_numpy = isinstance(data, np.ndarray)

    if input_is_numpy:
        data_t = torch.from_numpy(data).float().to(device)
    else:
        data_t = data.to(device) if not data.is_cuda else data

    # Get size of last axis
    last_axis_size = data_t.shape[-1]

    # Create reflection indices for last axis
    # Left: [pad_left, pad_left-1, ..., 1]
    # Right: [n-2, n-3, ..., n-1-pad_right]
    left_indices = torch.arange(pad_left, 0, -1, device=device)
    center_indices = torch.arange(last_axis_size, device=device)
    right_indices = torch.arange(last_axis_size - 2, last_axis_size - 2 - pad_right, -1, device=device)

    all_indices = torch.cat([left_indices, center_indices, right_indices])

    # Index along last axis - works for any dimensionality
    result = torch.index_select(data_t, dim=-1, index=all_indices)

    return result.cpu().numpy() if input_is_numpy else result


def _symmetric_manual_pytorch(data, pad_left, pad_right, device='cpu'):
    """
    Manual symmetric implementation for PyTorch (includes edge).

    PyTorch's 'reflect' mode has size limitations and can be unreliable.
    We implement NumPy's 'symmetric' (includes edge) manually using indexing.

    Supports arbitrary dimensions - always operates on last axis.
    """
    _check_torch_available()
    if device == 'cuda':
        _check_torch_cuda_available()

    input_is_numpy = isinstance(data, np.ndarray)

    if input_is_numpy:
        data_t = torch.from_numpy(data).float().to(device)
    else:
        data_t = data.to(device) if not data.is_cuda else data

    # Get size of last axis
    last_axis_size = data_t.shape[-1]

    # Create symmetric indices for last axis (includes edge)
    # Left: [pad_left-1, pad_left-2, ..., 0]
    # Right: [n-1, n-2, ..., n-pad_right]
    left_indices = torch.arange(pad_left - 1, -1, -1, device=device)
    center_indices = torch.arange(last_axis_size, device=device)
    right_indices = torch.arange(last_axis_size - 1, last_axis_size - 1 - pad_right, -1, device=device)

    all_indices = torch.cat([left_indices, center_indices, right_indices])

    # Index along last axis - works for any dimensionality
    result = torch.index_select(data_t, dim=-1, index=all_indices)

    return result.cpu().numpy() if input_is_numpy else result


# ============================================================================
# NUMPY NATIVE BACKEND
# ============================================================================

def pad_reflect_numpy_native(data, pad_left, pad_right, constant_value=0):
    """Reflect mode - numpy native implementation."""
    return np.pad(data, (pad_left, pad_right), mode='reflect')


def pad_symmetric_numpy_native(data, pad_left, pad_right, constant_value=0):
    """Symmetric mode - numpy native implementation."""
    return np.pad(data, (pad_left, pad_right), mode='symmetric')


def pad_edge_numpy_native(data, pad_left, pad_right, constant_value=0):
    """Edge mode - numpy native implementation."""
    return np.pad(data, (pad_left, pad_right), mode='edge')


def pad_constant_numpy_native(data, pad_left, pad_right, constant_value=0):
    """Constant mode - numpy native implementation."""
    return np.pad(data, (pad_left, pad_right), mode='constant', constant_values=constant_value)


def pad_wrap_numpy_native(data, pad_left, pad_right, constant_value=0):
    """Wrap mode - numpy native implementation."""
    return np.pad(data, (pad_left, pad_right), mode='wrap')


# ============================================================================
# PYTORCH CPU BACKEND - Supports arbitrary dimensions
# ============================================================================

def pad_reflect_pytorch_cpu(data, pad_left, pad_right, constant_value=0):
    """
    Reflect mode - PyTorch CPU implementation.

    Note: PyTorch's 'reflect' is NumPy's 'symmetric' (includes edge).
    We manually implement NumPy's 'reflect' (excludes edge).

    Supports arbitrary dimensions - operates on last axis.
    """
    return _reflect_manual_pytorch(data, pad_left, pad_right, device='cpu')


def pad_symmetric_pytorch_cpu(data, pad_left, pad_right, constant_value=0):
    """
    Symmetric mode - PyTorch CPU implementation.

    Implements NumPy's 'symmetric' mode (includes edge) manually using indexing.
    This avoids issues with PyTorch's 'reflect' mode padding size limitations.
    Supports arbitrary dimensions.
    """
    return _symmetric_manual_pytorch(data, pad_left, pad_right, device='cpu')


def pad_edge_pytorch_cpu(data, pad_left, pad_right, constant_value=0):
    """
    Edge mode - PyTorch CPU implementation.

    PyTorch's 'replicate' mode = NumPy's 'edge' mode.
    Supports arbitrary dimensions.
    """
    return _pad_pytorch_last_axis(data, pad_left, pad_right, mode='replicate', device='cpu')


def pad_constant_pytorch_cpu(data, pad_left, pad_right, constant_value=0):
    """
    Constant mode - PyTorch CPU implementation.
    Supports arbitrary dimensions.
    """
    return _pad_pytorch_last_axis(data, pad_left, pad_right, mode='constant',
                                   value=constant_value, device='cpu')


def pad_wrap_pytorch_cpu(data, pad_left, pad_right, constant_value=0):
    """
    Wrap mode - PyTorch CPU implementation.

    PyTorch's 'circular' mode = NumPy's 'wrap' mode.
    Supports arbitrary dimensions.
    """
    return _pad_pytorch_last_axis(data, pad_left, pad_right, mode='circular', device='cpu')


# ============================================================================
# PYTORCH GPU BACKEND - Supports arbitrary dimensions
# ============================================================================

def pad_reflect_pytorch_gpu(data, pad_left, pad_right, constant_value=0):
    """
    Reflect mode - PyTorch GPU implementation.

    Note: PyTorch's 'reflect' is NumPy's 'symmetric' (includes edge).
    We manually implement NumPy's 'reflect' (excludes edge).

    Supports arbitrary dimensions - operates on last axis.
    """
    return _reflect_manual_pytorch(data, pad_left, pad_right, device='cuda')


def pad_symmetric_pytorch_gpu(data, pad_left, pad_right, constant_value=0):
    """
    Symmetric mode - PyTorch GPU implementation.

    Implements NumPy's 'symmetric' mode (includes edge) manually using indexing.
    This avoids issues with PyTorch's 'reflect' mode padding size limitations.
    Supports arbitrary dimensions.
    """
    return _symmetric_manual_pytorch(data, pad_left, pad_right, device='cuda')


def pad_edge_pytorch_gpu(data, pad_left, pad_right, constant_value=0):
    """
    Edge mode - PyTorch GPU implementation.

    PyTorch's 'replicate' mode = NumPy's 'edge' mode.
    Supports arbitrary dimensions.
    """
    return _pad_pytorch_last_axis(data, pad_left, pad_right, mode='replicate', device='cuda')


def pad_constant_pytorch_gpu(data, pad_left, pad_right, constant_value=0):
    """
    Constant mode - PyTorch GPU implementation.
    Supports arbitrary dimensions.
    """
    return _pad_pytorch_last_axis(data, pad_left, pad_right, mode='constant',
                                   value=constant_value, device='cuda')


def pad_wrap_pytorch_gpu(data, pad_left, pad_right, constant_value=0):
    """
    Wrap mode - PyTorch GPU implementation.

    PyTorch's 'circular' mode = NumPy's 'wrap' mode.
    Supports arbitrary dimensions.
    """
    return _pad_pytorch_last_axis(data, pad_left, pad_right, mode='circular', device='cuda')


# ============================================================================
# NUMBA PURE BACKEND (manual loops only, no numpy functions)
# ============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def pad_reflect_numba_pure(data, pad_left, pad_right, constant_value=0):
        """Reflect mode - pure numba implementation."""
        n = len(data)
        result = np.empty(n + pad_left + pad_right, dtype=data.dtype)

        # Copy center
        for i in range(n):
            result[pad_left + i] = data[i]

        # Reflect left (excludes edge)
        for i in range(pad_left):
            result[i] = data[pad_left - i]

        # Reflect right (excludes edge)
        for i in range(pad_right):
            result[pad_left + n + i] = data[n - 2 - i]

        return result


    @jit(nopython=True)
    def pad_symmetric_numba_pure(data, pad_left, pad_right, constant_value=0):
        """Symmetric mode - pure numba implementation."""
        n = len(data)
        result = np.empty(n + pad_left + pad_right, dtype=data.dtype)

        for i in range(n):
            result[pad_left + i] = data[i]

        # Mirror including edge
        for i in range(pad_left):
            result[i] = data[pad_left - 1 - i]

        for i in range(pad_right):
            result[pad_left + n + i] = data[n - 1 - i]

        return result


    @jit(nopython=True)
    def pad_edge_numba_pure(data, pad_left, pad_right, constant_value=0):
        """Edge mode - pure numba implementation."""
        n = len(data)
        result = np.empty(n + pad_left + pad_right, dtype=data.dtype)

        for i in range(n):
            result[pad_left + i] = data[i]

        left_val = data[0]
        right_val = data[n - 1]

        for i in range(pad_left):
            result[i] = left_val

        for i in range(pad_right):
            result[pad_left + n + i] = right_val

        return result


    @jit(nopython=True)
    def pad_constant_numba_pure(data, pad_left, pad_right, constant_value=0):
        """Constant mode - pure numba implementation."""
        n = len(data)
        result = np.empty(n + pad_left + pad_right, dtype=data.dtype)

        for i in range(n):
            result[pad_left + i] = data[i]

        for i in range(pad_left):
            result[i] = constant_value

        for i in range(pad_right):
            result[pad_left + n + i] = constant_value

        return result


    @jit(nopython=True)
    def pad_wrap_numba_pure(data, pad_left, pad_right, constant_value=0):
        """Wrap mode - pure numba implementation."""
        n = len(data)
        result = np.empty(n + pad_left + pad_right, dtype=data.dtype)

        for i in range(n):
            result[pad_left + i] = data[i]

        for i in range(pad_left):
            idx = (n - pad_left + i) % n
            result[i] = data[idx]

        for i in range(pad_right):
            idx = i % n
            result[pad_left + n + i] = data[idx]

        return result

else:
    # Numba not available - create stub functions that raise errors
    def _numba_not_available(*args, **kwargs):
        _check_numba_available()

    pad_reflect_numba_pure = _numba_not_available
    pad_symmetric_numba_pure = _numba_not_available
    pad_edge_numba_pure = _numba_not_available
    pad_constant_numba_pure = _numba_not_available
    pad_wrap_numba_pure = _numba_not_available


# ============================================================================
# NUMBA + NUMPY FUNCTIONS BACKEND (numpy allocation, manual loops)
# ============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def pad_reflect_numba_numpy(data, pad_left, pad_right, constant_value=0):
        """Reflect mode - numba with numpy functions."""
        n = len(data)
        result = np.empty(n + pad_left + pad_right, dtype=data.dtype)

        # Use slice assignment
        result[pad_left:pad_left + n] = data

        # Reflect using computed indices (excluding edge)
        for i in range(pad_left):
            result[i] = data[pad_left - i]

        for i in range(pad_right):
            result[pad_left + n + i] = data[n - 2 - i]

        return result


    @jit(nopython=True)
    def pad_symmetric_numba_numpy(data, pad_left, pad_right, constant_value=0):
        """Symmetric mode - numba with numpy functions."""
        n = len(data)
        result = np.empty(n + pad_left + pad_right, dtype=data.dtype)
        result[pad_left:pad_left + n] = data

        # Mirror including edge
        for i in range(pad_left):
            result[i] = data[pad_left - 1 - i]

        for i in range(pad_right):
            result[pad_left + n + i] = data[n - 1 - i]

        return result


    @jit(nopython=True)
    def pad_edge_numba_numpy(data, pad_left, pad_right, constant_value=0):
        """Edge mode - numba with numpy functions."""
        n = len(data)
        result = np.empty(n + pad_left + pad_right, dtype=data.dtype)
        result[pad_left:pad_left + n] = data

        left_val = data[0]
        right_val = data[n - 1]

        for i in range(pad_left):
            result[i] = left_val

        for i in range(pad_right):
            result[pad_left + n + i] = right_val

        return result


    @jit(nopython=True)
    def pad_constant_numba_numpy(data, pad_left, pad_right, constant_value=0):
        """Constant mode - numba with numpy functions."""
        n = len(data)
        result = np.empty(n + pad_left + pad_right, dtype=data.dtype)
        result[pad_left:pad_left + n] = data

        for i in range(pad_left):
            result[i] = constant_value

        for i in range(pad_right):
            result[pad_left + n + i] = constant_value

        return result


    @jit(nopython=True)
    def pad_wrap_numba_numpy(data, pad_left, pad_right, constant_value=0):
        """Wrap mode - numba with numpy functions."""
        n = len(data)
        result = np.empty(n + pad_left + pad_right, dtype=data.dtype)
        result[pad_left:pad_left + n] = data

        for i in range(pad_left):
            idx = (n - pad_left + i) % n
            result[i] = data[idx]

        for i in range(pad_right):
            idx = i % n
            result[pad_left + n + i] = data[idx]

        return result

else:
    # Numba not available - create stub functions that raise errors
    pad_reflect_numba_numpy = _numba_not_available
    pad_symmetric_numba_numpy = _numba_not_available
    pad_edge_numba_numpy = _numba_not_available
    pad_constant_numba_numpy = _numba_not_available
    pad_wrap_numba_numpy = _numba_not_available


# ============================================================================
# CUSTOM VECTORIZED BACKEND (fancy indexing, no loops)
# ============================================================================

def pad_reflect_custom_vectorized(data, pad_left, pad_right, constant_value=0):
    """Reflect mode - custom vectorized implementation."""
    n = len(data)

    # Create index array for reflect (excludes edge)
    left_indices = np.arange(pad_left, 0, -1)
    center_indices = np.arange(n)
    right_indices = np.arange(n - 2, n - 2 - pad_right, -1)

    all_indices = np.concatenate([left_indices, center_indices, right_indices])

    return data[all_indices]


def pad_symmetric_custom_vectorized(data, pad_left, pad_right, constant_value=0):
    """Symmetric mode - custom vectorized implementation."""
    n = len(data)

    # For symmetric: includes edge
    left_indices = np.arange(pad_left - 1, -1, -1)
    center_indices = np.arange(n)
    right_indices = np.arange(n - 1, n - 1 - pad_right, -1)

    all_indices = np.concatenate([left_indices, center_indices, right_indices])

    return data[all_indices]


def pad_edge_custom_vectorized(data, pad_left, pad_right, constant_value=0):
    """Edge mode - custom vectorized implementation."""
    n = len(data)

    left_indices = np.zeros(pad_left, dtype=int)
    center_indices = np.arange(n)
    right_indices = np.full(pad_right, n - 1, dtype=int)

    all_indices = np.concatenate([left_indices, center_indices, right_indices])

    return data[all_indices]


def pad_constant_custom_vectorized(data, pad_left, pad_right, constant_value=0):
    """Constant mode - custom vectorized implementation."""
    left = np.full(pad_left, constant_value, dtype=data.dtype)
    right = np.full(pad_right, constant_value, dtype=data.dtype)

    return np.concatenate([left, data, right])


def pad_wrap_custom_vectorized(data, pad_left, pad_right, constant_value=0):
    """Wrap mode - custom vectorized implementation."""
    n = len(data)

    left_indices = np.arange(n - pad_left, n)
    center_indices = np.arange(n)
    right_indices = np.arange(0, pad_right)

    all_indices = np.concatenate([left_indices, center_indices, right_indices])

    return data[all_indices]


# ============================================================================
# REGISTRY - All implementations organized
# ============================================================================

PADDING_IMPLEMENTATIONS = {
    # NumPy Native
    ('reflect', 'numpy_native'): pad_reflect_numpy_native,
    ('symmetric', 'numpy_native'): pad_symmetric_numpy_native,
    ('edge', 'numpy_native'): pad_edge_numpy_native,
    ('constant', 'numpy_native'): pad_constant_numpy_native,
    ('wrap', 'numpy_native'): pad_wrap_numpy_native,

    # PyTorch CPU
    ('reflect', 'pytorch_cpu'): pad_reflect_pytorch_cpu,
    ('symmetric', 'pytorch_cpu'): pad_symmetric_pytorch_cpu,
    ('edge', 'pytorch_cpu'): pad_edge_pytorch_cpu,
    ('constant', 'pytorch_cpu'): pad_constant_pytorch_cpu,
    ('wrap', 'pytorch_cpu'): pad_wrap_pytorch_cpu,

    # PyTorch GPU
    ('reflect', 'pytorch_gpu'): pad_reflect_pytorch_gpu,
    ('symmetric', 'pytorch_gpu'): pad_symmetric_pytorch_gpu,
    ('edge', 'pytorch_gpu'): pad_edge_pytorch_gpu,
    ('constant', 'pytorch_gpu'): pad_constant_pytorch_gpu,
    ('wrap', 'pytorch_gpu'): pad_wrap_pytorch_gpu,

    # Numba Pure
    ('reflect', 'numba_pure'): pad_reflect_numba_pure,
    ('symmetric', 'numba_pure'): pad_symmetric_numba_pure,
    ('edge', 'numba_pure'): pad_edge_numba_pure,
    ('constant', 'numba_pure'): pad_constant_numba_pure,
    ('wrap', 'numba_pure'): pad_wrap_numba_pure,

    # Numba + NumPy
    ('reflect', 'numba_numpy'): pad_reflect_numba_numpy,
    ('symmetric', 'numba_numpy'): pad_symmetric_numba_numpy,
    ('edge', 'numba_numpy'): pad_edge_numba_numpy,
    ('constant', 'numba_numpy'): pad_constant_numba_numpy,
    ('wrap', 'numba_numpy'): pad_wrap_numba_numpy,

    # Custom Vectorized
    ('reflect', 'custom_vectorized'): pad_reflect_custom_vectorized,
    ('symmetric', 'custom_vectorized'): pad_symmetric_custom_vectorized,
    ('edge', 'custom_vectorized'): pad_edge_custom_vectorized,
    ('constant', 'custom_vectorized'): pad_constant_custom_vectorized,
    ('wrap', 'custom_vectorized'): pad_wrap_custom_vectorized,
}


def get_padding_function(mode, backend):
    """
    Get padding function by mode and backend.

    Args:
        mode: 'reflect', 'symmetric', 'edge', 'constant', 'wrap'
        backend: 'numpy_native', 'pytorch_cpu', 'pytorch_gpu',
                 'numba_pure', 'numba_numpy', 'custom_vectorized'

    Returns:
        Padding function

    Raises:
        KeyError: If combination not available
    """
    key = (mode, backend)
    if key not in PADDING_IMPLEMENTATIONS:
        raise KeyError(f"Padding implementation for mode='{mode}' and backend='{backend}' not found")
    return PADDING_IMPLEMENTATIONS[key]


def list_available_implementations():
    """List all available (mode, backend) combinations."""
    return list(PADDING_IMPLEMENTATIONS.keys())