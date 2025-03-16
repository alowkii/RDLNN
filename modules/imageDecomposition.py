import numpy as np
import pywt
import torch
import torch.nn.functional as F

def polar_dyadic_wavelet_transform(ycbcr_img):
    """
    Implement GPU-accelerated Polar Dyadic Wavelet Transform (PDyWT) using PyTorch
    
    Args:
        ycbcr_img: YCbCr preprocessed image as torch.Tensor [C, H, W]
        
    Returns:
        Dictionary with decomposed wavelet coefficients for all channels
    """
    try:
        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert input to torch tensor if it's not already
        if not isinstance(ycbcr_img, torch.Tensor):
            ycbcr_img = torch.tensor(ycbcr_img, dtype=torch.float32)
        
        # Ensure tensor is on the right device
        ycbcr_img = ycbcr_img.to(device)
            
        # Handle different tensor formats
        if ycbcr_img.ndim == 3 and ycbcr_img.shape[0] == 3:  # [C, H, W] format
            # Rearrange to [H, W, C] for processing
            ycbcr_img = ycbcr_img.permute(1, 2, 0)
        
        # Split YCbCr channels 
        if ycbcr_img.ndim == 3 and ycbcr_img.shape[2] == 3:
            y_channel = ycbcr_img[:, :, 0]
            cb_channel = ycbcr_img[:, :, 1]
            cr_channel = ycbcr_img[:, :, 2]
        else:
            raise ValueError(f"Expected 3-channel image, got shape {ycbcr_img.shape}")
        
        # Make dimensions even for wavelet transform
        h, w = y_channel.shape
        h_pad = 0 if h % 2 == 0 else 1
        w_pad = 0 if w % 2 == 0 else 1
        
        # Apply padding if needed - ensure we get even dimensions
        if h_pad != 0 or w_pad != 0:
            y_channel = F.pad(y_channel.unsqueeze(0).unsqueeze(0), (0, w_pad, 0, h_pad), mode='reflect').squeeze(0).squeeze(0)
            cb_channel = F.pad(cb_channel.unsqueeze(0).unsqueeze(0), (0, w_pad, 0, h_pad), mode='reflect').squeeze(0).squeeze(0)
            cr_channel = F.pad(cr_channel.unsqueeze(0).unsqueeze(0), (0, w_pad, 0, h_pad), mode='reflect').squeeze(0).squeeze(0)
        
        # Apply PyTorch-based wavelet transform
        coeffs_y = _torch_swt2(y_channel, device)
        coeffs_cb = _torch_swt2(cb_channel, device)
        coeffs_cr = _torch_swt2(cr_channel, device)
        
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

def _torch_swt2(data, device):
    """
    PyTorch implementation of stationary wavelet transform
    
    Args:
        data: 2D PyTorch tensor
        device: PyTorch device
        
    Returns:
        Tuple of (LL, LH, HL, HH) wavelet coefficients
    """
    # Haar wavelet filters
    low_filter = torch.tensor([0.5, 0.5], dtype=torch.float32, device=device)
    high_filter = torch.tensor([0.5, -0.5], dtype=torch.float32, device=device)
    
    # Create 2D separable filters
    low_filter_x = low_filter.view(1, 1, 1, -1)
    low_filter_y = low_filter.view(1, 1, -1, 1)
    high_filter_x = high_filter.view(1, 1, 1, -1)
    high_filter_y = high_filter.view(1, 1, -1, 1)
    
    # Apply 2D separable convolution
    data_4d = data.unsqueeze(0).unsqueeze(0)
    
    # Calculate appropriate padding manually instead of using 'same'
    pad_x = low_filter_x.shape[-1] // 2
    pad_y = low_filter_y.shape[2] // 2
    
    # Apply manual padding for all operations
    data_padded = F.pad(data_4d, (pad_x, pad_x, pad_y, pad_y), mode='reflect')
    
    # Low-Low
    ll_h = F.conv2d(data_padded, low_filter_x, padding=0)
    ll = F.conv2d(ll_h, low_filter_y, padding=0).squeeze(0).squeeze(0)
    
    # Low-High
    lh_h = F.conv2d(data_padded, low_filter_x, padding=0)
    lh = F.conv2d(lh_h, high_filter_y, padding=0).squeeze(0).squeeze(0)
    
    # High-Low
    hl_h = F.conv2d(data_padded, high_filter_x, padding=0)
    hl = F.conv2d(hl_h, low_filter_y, padding=0).squeeze(0).squeeze(0)
    
    # High-High
    hh_h = F.conv2d(data_padded, high_filter_x, padding=0)
    hh = F.conv2d(hh_h, high_filter_y, padding=0).squeeze(0).squeeze(0)
    
    return ll, lh, hl, hh

def _transform_to_polar(coeffs):
    """
    Transform wavelet coefficients to polar coordinates using PyTorch
    
    Args:
        coeffs: Tuple of (LL, LH, HL, HH) wavelet coefficients
        
    Returns:
        Tuple of transformed coefficients
    """
    ll, lh, hl, hh = coeffs
    
    # Ensure all coefficients have the same shape
    h, w = ll.shape
    device = ll.device
    
    # Check and possibly resize other coefficients if needed
    lh = F.interpolate(lh.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    hl = F.interpolate(hl.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    hh = F.interpolate(hh.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    
    # Create coordinate grid
    y_coords = torch.arange(0, h, device=device).view(-1, 1).repeat(1, w)
    x_coords = torch.arange(0, w, device=device).view(1, -1).repeat(h, 1)
    
    # Find center point
    center_y, center_x = h // 2, w // 2
    
    # Calculate polar coordinates (radius and angle)
    r = torch.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    theta = torch.atan2(y_coords - center_y, x_coords - center_x)
    
    # Normalize radius
    max_radius = torch.sqrt(torch.tensor((h/2)**2 + (w/2)**2, device=device))
    r_norm = r / max_radius
    
    # Apply polar weighting to coefficients (radial emphasis)
    polar_ll = ll * r_norm
    polar_lh = lh * r_norm
    polar_hl = hl * r_norm
    polar_hh = hh * r_norm
    
    return polar_ll, polar_lh, polar_hl, polar_hh