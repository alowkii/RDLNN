import numpy as np
import pywt
import cv2
import math

def polar_dyadic_wavelet_transform(ycbcr_img):
    """
    Implement Polar Dyadic Wavelet Transform (PDyWT) to decompose the image
    PDyWT keeps image size constant at different levels and provides better analysis
    
    Args:
        ycbcr_img: YCbCr preprocessed image
        
    Returns:
        Decomposed image with wavelet coefficients
    """
    try:
        # Split channels
        y_channel, cb_channel, cr_channel = cv2.split(ycbcr_img)
        
        # Check if dimensions are even and pad if necessary
        h, w = y_channel.shape
        pad_h = 0 if h % 2 == 0 else 1
        pad_w = 0 if w % 2 == 0 else 1
        
        if pad_h or pad_w:
            y_channel = np.pad(y_channel, ((0, pad_h), (0, pad_w)), 'reflect')
            cb_channel = np.pad(cb_channel, ((0, pad_h), (0, pad_w)), 'reflect')
            cr_channel = np.pad(cr_channel, ((0, pad_h), (0, pad_w)), 'reflect')
            print(f"Padded image to even dimensions: {y_channel.shape}")
        
        # Calculate appropriate wavelet decomposition level based on image size
        max_level = min(pywt.dwt_max_level(min(y_channel.shape), pywt.Wavelet('db1').dec_len), 4)
        level = min(1, max_level)  # Use at least level 1, but don't exceed max_level
        
        # Apply DyWT using stationary wavelet transform with proper error handling
        try:
            coeffs_y = pywt.swt2(y_channel, 'db1', level=level)[0]
            coeffs_cb = pywt.swt2(cb_channel, 'db1', level=level)[0]
            coeffs_cr = pywt.swt2(cr_channel, 'db1', level=level)[0]
        except ValueError as e:
            if "Length of data must be even" in str(e):
                print("Error: Image dimensions must be even for wavelet transform.")
                print(f"Current dimensions after padding: {y_channel.shape}")
                # Apply additional padding if needed
                y_channel = np.pad(y_channel, ((0, 1), (0, 1)), 'reflect')
                cb_channel = np.pad(cb_channel, ((0, 1), (0, 1)), 'reflect')
                cr_channel = np.pad(cr_channel, ((0, 1), (0, 1)), 'reflect')
                print(f"Applied extra padding. New dimensions: {y_channel.shape}")
                # Try again with additional padding
                coeffs_y = pywt.swt2(y_channel, 'db1', level=level)[0]
                coeffs_cb = pywt.swt2(cb_channel, 'db1', level=level)[0]
                coeffs_cr = pywt.swt2(cr_channel, 'db1', level=level)[0]
            else:
                raise
        
        # Extract approximate (LL) and detail coefficients (LH, HL, HH)
        ll_y, (lh_y, hl_y, hh_y) = coeffs_y
        ll_cb, (lh_cb, hl_cb, hh_cb) = coeffs_cb
        ll_cr, (lh_cr, hl_cr, hh_cr) = coeffs_cr
        
        # Convert to polar form for better coordinate values
        def to_polar(coeff):
            h, w = coeff.shape
            polar_coeff = np.zeros_like(coeff)
            
            # Ensure dimensions are even for consistent polar transformation
            if h % 2 != 0 or w % 2 != 0:
                coeff = np.pad(coeff, ((0, h % 2), (0, w % 2)), 'reflect')
                h, w = coeff.shape
            
            # Get center of the image
            center_y, center_x = h // 2, w // 2
            
            for y in range(h):
                for x in range(w):
                    # Calculate polar coordinates
                    r = math.sqrt((y - center_y)**2 + (x - center_x)**2)
                    theta = math.atan2(y - center_y, x - center_x)
                    
                    # Map polar coordinates to image values
                    # This is a simplified version of the polar mapping
                    polar_coeff[y, x] = coeff[y, x] * (r / max(h, w))
            
            return polar_coeff
    
        # Apply polar transformation to wavelet coefficients
        polar_coeffs = {
            'y': (to_polar(ll_y), (to_polar(lh_y), to_polar(hl_y), to_polar(hh_y))),
            'cb': (to_polar(ll_cb), (to_polar(lh_cb), to_polar(hl_cb), to_polar(hh_cb))),
            'cr': (to_polar(ll_cr), (to_polar(lh_cr), to_polar(hl_cr), to_polar(hh_cr)))
        }
    
        print("PDyWT decomposition completed successfully")
        return polar_coeffs
        
    except Exception as e:
        print(f"Error in polar dyadic wavelet transform: {e}")
        import traceback
        traceback.print_exc()
        return None