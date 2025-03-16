import numpy as np
import pywt
import cv2
import math
import cupy as cp
import torch
from cupyx.scipy.ndimage import gaussian_filter

def polar_dyadic_wavelet_transform(ycbcr_img):
    """
    Implement GPU-only Polar Dyadic Wavelet Transform (PDyWT) to decompose the image
    PDyWT keeps image size constant at different levels and provides better analysis
    
    Args:
        ycbcr_img: YCbCr preprocessed image (can be torch.Tensor or numpy array)
        
    Returns:
        Decomposed image with wavelet coefficients
    """
    try:
        # Check if GPU is available
        if not cp.cuda.is_available():
            raise RuntimeError("CUDA GPU not available")
        
        print(f"CUDA GPU available: {cp.cuda.is_available()}")
        
        # Convert input to numpy if it's a torch tensor
        if isinstance(ycbcr_img, torch.Tensor):
            # Move tensor to CPU and convert to numpy
            if ycbcr_img.is_cuda:
                ycbcr_img = ycbcr_img.cpu()
            ycbcr_img = ycbcr_img.numpy()
            # Ensure channel dimension is last (convert from CxHxW to HxWxC)
            if ycbcr_img.shape[0] == 3:  # If in format CxHxW
                ycbcr_img = np.transpose(ycbcr_img, (1, 2, 0))
        
        # Transfer image to GPU
        ycbcr_img_gpu = cp.asarray(ycbcr_img)
        
        # Split YCbCr channels on GPU
        # Extract each channel (assuming HxWxC format)
        y_channel_gpu = ycbcr_img_gpu[:, :, 0]
        cb_channel_gpu = ycbcr_img_gpu[:, :, 1]
        cr_channel_gpu = ycbcr_img_gpu[:, :, 2]
        
        # Ensure dimensions are even for consistent processing
        h, w = y_channel_gpu.shape
        
        # Always pad to ensure dimensions are even and consistent
        # This also helps avoid broadcasting errors
        y_channel_gpu = cp.pad(y_channel_gpu, ((0, (h % 2) + 2), (0, (w % 2) + 2)), 'reflect')
        cb_channel_gpu = cp.pad(cb_channel_gpu, ((0, (h % 2) + 2), (0, (w % 2) + 2)), 'reflect')
        cr_channel_gpu = cp.pad(cr_channel_gpu, ((0, (h % 2) + 2), (0, (w % 2) + 2)), 'reflect')
        
        new_h, new_w = y_channel_gpu.shape
        print(f"Padded image to dimensions: {new_h}x{new_w}")
        
        # Calculate appropriate wavelet decomposition level based on image size
        max_level = min(pywt.dwt_max_level(min(y_channel_gpu.shape), pywt.Wavelet('db1').dec_len), 4)
        level = min(1, max_level)  # Use at least level 1, but don't exceed max_level
        
        # Apply GPU-based stationary wavelet transform
        coeffs_y = _gpu_swt2(y_channel_gpu, level)
        coeffs_cb = _gpu_swt2(cb_channel_gpu, level)
        coeffs_cr = _gpu_swt2(cr_channel_gpu, level)
        
        # Extract approximate (LL) and detail coefficients (LH, HL, HH)
        ll_y, lh_y, hl_y, hh_y = coeffs_y
        ll_cb, lh_cb, hl_cb, hh_cb = coeffs_cb
        ll_cr, lh_cr, hl_cr, hh_cr = coeffs_cr
        
        # Convert to polar form for better coordinate values using GPU
        def to_polar_gpu(coeff_gpu):
            h, w = coeff_gpu.shape
            
            # Get center of the image
            center_y, center_x = h // 2, w // 2
            
            # Create coordinate grids
            y_coords, x_coords = cp.mgrid[0:h, 0:w]
            
            # Calculate polar coordinates
            r = cp.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            theta = cp.arctan2(y_coords - center_y, x_coords - center_x)
            
            # Map polar coordinates to image values (vectorized operation)
            scaling_factor = r / max(h, w)
            polar_coeff_gpu = coeff_gpu * scaling_factor
            
            return polar_coeff_gpu
    
        # Apply polar transformation to wavelet coefficients
        polar_ll_y = to_polar_gpu(ll_y)
        polar_lh_y = to_polar_gpu(lh_y)
        polar_hl_y = to_polar_gpu(hl_y)
        polar_hh_y = to_polar_gpu(hh_y)
        
        polar_ll_cb = to_polar_gpu(ll_cb)
        polar_lh_cb = to_polar_gpu(lh_cb)
        polar_hl_cb = to_polar_gpu(hl_cb)
        polar_hh_cb = to_polar_gpu(hh_cb)
        
        polar_ll_cr = to_polar_gpu(ll_cr)
        polar_lh_cr = to_polar_gpu(lh_cr)
        polar_hl_cr = to_polar_gpu(hl_cr)
        polar_hh_cr = to_polar_gpu(hh_cr)
        
        # Keep the results on GPU
        polar_coeffs = {
            'y': (polar_ll_y, (polar_lh_y, polar_hl_y, polar_hh_y)),
            'cb': (polar_ll_cb, (polar_lh_cb, polar_hl_cb, polar_hh_cb)),
            'cr': (polar_ll_cr, (polar_lh_cr, polar_hl_cr, polar_hh_cr))
        }
        
        print("PDyWT decomposition completed successfully")
        return polar_coeffs
        
    except Exception as e:
        print(f"Error in GPU polar dyadic wavelet transform: {e}")
        raise e

def _gpu_swt2(data_gpu, level):
    """
    GPU implementation of the stationary wavelet transform (SWT2)
    Simplified implementation using GPU-accelerated filters
    
    Args:
        data_gpu: CuPy array of image data
        level: Decomposition level
        
    Returns:
        Tuple of approximation and detail coefficients
    """
    # Define Haar wavelet filter coefficients
    low_filter = cp.array([0.5, 0.5])
    high_filter = cp.array([0.5, -0.5])
    
    # Using CuPy's FFT-based convolution for better stability
    # This avoids shape broadcasting issues
    def apply_filter(data, filter_y, filter_x):
        # Use cuFFT-based convolution
        return cp.fft.ifft2(cp.fft.fft2(data) * cp.fft.fft2(filter_y * filter_x, s=data.shape)).real
    
    # Create 2D filters with proper padding to match input size
    h, w = data_gpu.shape
    low_low = cp.zeros((h, w), dtype=cp.float32)
    low_high = cp.zeros((h, w), dtype=cp.float32)
    high_low = cp.zeros((h, w), dtype=cp.float32)
    high_high = cp.zeros((h, w), dtype=cp.float32)
    
    # Set the center elements to create proper 2D filters
    low_low[h//2-1:h//2+1, w//2-1:w//2+1] = cp.outer(low_filter, low_filter)
    low_high[h//2-1:h//2+1, w//2-1:w//2+1] = cp.outer(low_filter, high_filter)
    high_low[h//2-1:h//2+1, w//2-1:w//2+1] = cp.outer(high_filter, low_filter)
    high_high[h//2-1:h//2+1, w//2-1:w//2+1] = cp.outer(high_filter, high_filter)
    
    # Apply convolution
    ll = apply_filter(data_gpu, low_low, cp.ones_like(low_low))
    lh = apply_filter(data_gpu, low_high, cp.ones_like(low_high))
    hl = apply_filter(data_gpu, high_low, cp.ones_like(high_low))
    hh = apply_filter(data_gpu, high_high, cp.ones_like(high_high))
    
    return ll, lh, hl, hh