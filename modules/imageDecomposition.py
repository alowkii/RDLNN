import numpy as np
import pywt
import torch
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter

def polar_dyadic_wavelet_transform(ycbcr_img):
    """
    Implement GPU-accelerated Polar Dyadic Wavelet Transform (PDyWT)
    
    Args:
        ycbcr_img: YCbCr preprocessed image as torch.Tensor [C, H, W]
        
    Returns:
        Dictionary with decomposed wavelet coefficients for all channels
    """
    try:
        # Check if GPU is available
        if not torch.cuda.is_available() or not cp.cuda.is_available():
            raise RuntimeError("CUDA GPU not available for CuPy or PyTorch")
        
        # Convert input to numpy if it's a torch tensor
        if isinstance(ycbcr_img, torch.Tensor):
            # Move tensor to CPU and convert to numpy
            if ycbcr_img.is_cuda:
                ycbcr_img = ycbcr_img.cpu()
            ycbcr_np = ycbcr_img.numpy()
        else:
            ycbcr_np = ycbcr_img
            
        # Handle different tensor formats
        if ycbcr_np.ndim == 3 and ycbcr_np.shape[0] == 3:  # [C, H, W] format
            # Rearrange to [H, W, C] for processing
            ycbcr_np = np.transpose(ycbcr_np, (1, 2, 0))
        
        # Transfer image to GPU using CuPy
        ycbcr_gpu = cp.asarray(ycbcr_np, dtype=cp.float32)
        
        # Split YCbCr channels 
        if ycbcr_gpu.ndim == 3 and ycbcr_gpu.shape[2] == 3:
            y_channel = ycbcr_gpu[:, :, 0]
            cb_channel = ycbcr_gpu[:, :, 1]
            cr_channel = ycbcr_gpu[:, :, 2]
        else:
            raise ValueError(f"Expected 3-channel image, got shape {ycbcr_gpu.shape}")
        
        # Make dimensions even for wavelet transform
        h, w = y_channel.shape
        h_pad, w_pad = (h % 2), (w % 2)
        
        # Apply padding if needed
        if h_pad != 0 or w_pad != 0:
            y_channel = cp.pad(y_channel, ((0, h_pad), (0, w_pad)), mode='reflect')
            cb_channel = cp.pad(cb_channel, ((0, h_pad), (0, w_pad)), mode='reflect') 
            cr_channel = cp.pad(cr_channel, ((0, h_pad), (0, w_pad)), mode='reflect')
        
        # Get updated dimensions
        h, w = y_channel.shape
        
        # Apply GPU-based wavelet transform
        coeffs_y = _gpu_swt2(y_channel)
        coeffs_cb = _gpu_swt2(cb_channel)
        coeffs_cr = _gpu_swt2(cr_channel)
        
        # Apply polar transformation
        polar_coeffs_y = _transform_to_polar(coeffs_y)
        polar_coeffs_cb = _transform_to_polar(coeffs_cb)
        polar_coeffs_cr = _transform_to_polar(coeffs_cr)
        
        # Return dictionary with coefficients
        return {
            'y': polar_coeffs_y,
            'cb': polar_coeffs_cb,
            'cr': polar_coeffs_cr
        }
    
    except Exception as e:
        print(f"Error in polar dyadic wavelet transform: {e}")
        raise e

def _gpu_swt2(data_gpu):
    """
    GPU implementation of stationary wavelet transform
    
    Args:
        data_gpu: 2D CuPy array
        
    Returns:
        Tuple of (LL, LH, HL, HH) wavelet coefficients
    """
    # Haar wavelet filters
    low_filter = cp.asarray([0.5, 0.5], dtype=cp.float32)
    high_filter = cp.asarray([0.5, -0.5], dtype=cp.float32)
    
    # Create 2D separable filters
    h, w = data_gpu.shape
    filter_size = 2
    
    # Function to apply separable convolution
    def apply_separable_filter(data, filter_x, filter_y):
        # Apply horizontal filter
        temp = cp.zeros_like(data)
        for i in range(filter_size):
            if i < len(filter_x):
                temp += cp.roll(data, i - filter_size//2, axis=1) * filter_x[i]
        
        # Apply vertical filter to temp result
        result = cp.zeros_like(temp)
        for i in range(filter_size):
            if i < len(filter_y):
                result += cp.roll(temp, i - filter_size//2, axis=0) * filter_y[i]
        
        return result
    
    # Apply filters to get wavelet coefficients
    ll = apply_separable_filter(data_gpu, low_filter, low_filter)
    lh = apply_separable_filter(data_gpu, low_filter, high_filter)
    hl = apply_separable_filter(data_gpu, high_filter, low_filter)
    hh = apply_separable_filter(data_gpu, high_filter, high_filter)
    
    return ll, lh, hl, hh

def _transform_to_polar(coeffs):
    """
    Transform wavelet coefficients to polar coordinates
    
    Args:
        coeffs: Tuple of (LL, LH, HL, HH) wavelet coefficients
        
    Returns:
        Tuple of transformed coefficients
    """
    ll, lh, hl, hh = coeffs
    h, w = ll.shape
    
    # Create coordinate grid
    y_coords, x_coords = cp.mgrid[0:h, 0:w]
    
    # Find center point
    center_y, center_x = h // 2, w // 2
    
    # Calculate polar coordinates (radius and angle)
    r = cp.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    theta = cp.arctan2(y_coords - center_y, x_coords - center_x)
    
    # Normalize radius
    max_radius = cp.sqrt((h/2)**2 + (w/2)**2)
    r_norm = r / max_radius
    
    # Apply polar weighting to coefficients (radial emphasis)
    polar_ll = ll * r_norm
    polar_lh = lh * r_norm
    polar_hl = hl * r_norm
    polar_hh = hh * r_norm
    
    return polar_ll, polar_lh, polar_hl, polar_hh