import torch
import torch.nn.functional as F
import numpy as np

def extract_features(polar_coeffs, ycbcr_img):
    """
    Extract important features from decomposed image using PyTorch with CUDA:
    - SURF-like features with GPU acceleration
    - Edge features
    - Correlation between channels
    - Color histogram
    - Homogeneity
    - Variance features
    
    Args:
        polar_coeffs: Decomposed image using PDyWT (dictionary with wavelet coefficients)
        ycbcr_img: Original YCbCr image (dictionary with 'y', 'cb', 'cr' keys)
        
    Returns:
        Feature vector containing all extracted features (PyTorch tensor on GPU)
    """
    # Force CUDA - no fallback
    device = torch.device("cuda")
    features = []
    
    # Debug info
    print(f"Type of ycbcr_img: {type(ycbcr_img)}")
    if isinstance(ycbcr_img, dict):
        print(f"Keys in ycbcr_img: {list(ycbcr_img.keys())}")
    
    # Extract Y, Cb, Cr channels from dictionary with lowercase keys
    if isinstance(ycbcr_img, dict):
        if 'y' in ycbcr_img and 'cb' in ycbcr_img and 'cr' in ycbcr_img:
            # Get channels from dictionary
            y_data = ycbcr_img['y']
            cb_data = ycbcr_img['cb']
            cr_data = ycbcr_img['cr']
            
            # Convert to tensor and move to GPU
            if not isinstance(y_data, torch.Tensor):
                y_channel = torch.tensor(y_data, dtype=torch.float32, device=device)
                cb_channel = torch.tensor(cb_data, dtype=torch.float32, device=device)
                cr_channel = torch.tensor(cr_data, dtype=torch.float32, device=device)
            else:
                y_channel = y_data.to(device)
                cb_channel = cb_data.to(device)
                cr_channel = cr_data.to(device)
        else:
            raise ValueError(f"Dict doesn't contain expected keys. Available keys: {list(ycbcr_img.keys())}")
    else:
        raise TypeError(f"Expected dict for ycbcr_img, got {type(ycbcr_img)}")
    
    print(f"Using Y channel with shape: {y_channel.shape}")
    
    # 1. SURF-like features using PyTorch equivalent operations
    y_channel_2d = y_channel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    # Apply Sobel filters for gradient computation - create directly on GPU
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
    
    sobel_x = sobel_x.view(1, 1, 3, 3)  # Reshape for convolution
    sobel_y = sobel_y.view(1, 1, 3, 3)  # Reshape for convolution
    
    # Compute gradients using convolution
    grad_x = F.conv2d(y_channel_2d, sobel_x, padding=1)
    grad_y = F.conv2d(y_channel_2d, sobel_y, padding=1)
    
    # Compute gradient magnitude
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    
    # Extract histogram of gradients from several regions
    h, w = y_channel.shape
    regions = [(h//4*i, w//4*j, h//4*(i+1), w//4*(j+1)) for i in range(4) for j in range(4)]
    
    surf_like_features = []
    for top, left, bottom, right in regions[:5]:  # Use first 5 regions for features
        if top >= h or left >= w:
            # Skip invalid regions
            hist = torch.zeros(4, device=device)
        else:
            # Adjust bottom and right if needed
            bottom = min(bottom, h)
            right = min(right, w)
            region_magnitude = grad_magnitude[0, 0, top:bottom, left:right]
            # Calculate histogram with 4 bins
            hist = torch.histc(region_magnitude, bins=4, min=0, max=torch.max(grad_magnitude) + 1e-8)
            hist = hist / (torch.sum(hist) + 1e-8)  # Normalize with epsilon to avoid division by zero
        surf_like_features.append(hist)
    
    # Flatten and add to features list - keep on GPU throughout processing
    surf_features = torch.cat(surf_like_features).flatten()
    features.append(surf_features[:20])  # Take first 20 features
    
    # 2. Edge features using PyTorch gradient operations
    edge_density = torch.mean(grad_magnitude).reshape(1)
    features.append(edge_density)
    
    # 3. Correlation between channels using PyTorch
    def compute_correlation(x, y):
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        x_centered = x - x_mean
        y_centered = y - y_mean
        
        numerator = torch.sum(x_centered * y_centered)
        denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2) + 1e-8)
        
        return numerator / denominator
    
    y_flat = y_channel.flatten()
    cb_flat = cb_channel.flatten()
    cr_flat = cr_channel.flatten()
    
    corr_y_cb = compute_correlation(y_flat, cb_flat).reshape(1)
    corr_y_cr = compute_correlation(y_flat, cr_flat).reshape(1)
    corr_cb_cr = compute_correlation(cb_flat, cr_flat).reshape(1)
    
    features.extend([corr_y_cb, corr_y_cr, corr_cb_cr])
    
    # 4. Color histogram features using PyTorch
    color_hists = []
    for channel in [y_channel, cb_channel, cr_channel]:
        channel_max = torch.max(channel)
        if channel_max > 0:
            hist = torch.histc(channel, bins=8, min=0, max=channel_max + 1e-8)
            hist = hist / (torch.sum(hist) + 1e-8)  # Normalize
        else:
            hist = torch.ones(8, device=device) / 8  # Uniform distribution if channel is all zeros
        color_hists.append(hist)
    
    features.extend(color_hists)
    
    # 5. GLCM-based homogeneity features
    def calculate_homogeneity(channel, distance=1):
        # Reduce channel to 8 gray levels
        max_val = torch.max(channel)
        if max_val > 0:
            channel_reduced = (channel / (max_val / 7.999)).long().clamp(0, 7)
        else:
            # If channel is all zeros, use zeros
            return torch.tensor(1.0, device=device).reshape(1)  # Perfect homogeneity for uniform channel
            
        h, w = channel.shape
        
        # Initialize GLCM on GPU
        glcm = torch.zeros((8, 8), device=device)
        
        # Ensure there are enough columns to compute pairs
        if w <= distance:
            return torch.tensor(1.0, device=device).reshape(1)  # Return perfect homogeneity for small images
            
        # Create pairs of pixel values at specified distance
        indices = torch.arange(w - distance, device=device)
        for i in range(h):
            i_vals = channel_reduced[i, indices]
            j_vals = channel_reduced[i, indices + distance]
            
            # Update GLCM - use GPU-optimized indexing
            # Create coordinate tensors for updating GLCM
            i_coords = i_vals.reshape(-1, 1)
            j_coords = j_vals.reshape(-1, 1)
            coords = torch.cat([i_coords, j_coords], dim=1)
            
            # Count occurrences of each coordinate pair
            values = torch.ones(coords.shape[0], device=device)
            glcm.index_put_(tuple(coords.t()), values, accumulate=True)
        
        # Normalize GLCM
        glcm_sum = torch.sum(glcm)
        if glcm_sum > 0:
            glcm = glcm / glcm_sum
        else:
            return torch.tensor(1.0, device=device).reshape(1)  # Perfect homogeneity for empty GLCM
        
        # Calculate homogeneity
        try:
            i_indices, j_indices = torch.meshgrid(torch.arange(8, device=device), torch.arange(8, device=device), indexing='ij')
        except TypeError:
            # For older PyTorch versions that don't support indexing parameter
            i_indices, j_indices = torch.meshgrid(torch.arange(8, device=device), torch.arange(8, device=device))
            
        weights = 1.0 / (1.0 + torch.abs(i_indices - j_indices).float())
        homogeneity = torch.sum(glcm * weights).reshape(1)
        
        return homogeneity
    
    homogeneity_features = [
        calculate_homogeneity(y_channel),
        calculate_homogeneity(cb_channel),
        calculate_homogeneity(cr_channel)
    ]
    features.extend(homogeneity_features)
    
    # 6. Variance features from wavelet coefficients
    print(f"Type of polar_coeffs: {type(polar_coeffs)}")
    if isinstance(polar_coeffs, dict):
        print(f"Keys in polar_coeffs: {list(polar_coeffs.keys())}")
        
        # Safer processing of wavelet coefficients
        for channel_name, coeffs in polar_coeffs.items():
            print(f"Processing coefficients for channel: {channel_name}")
            print(f"Type of coeffs: {type(coeffs)}")
            
            try:
                # Check what structure coeffs has
                if isinstance(coeffs, tuple):
                    print(f"Coeffs is a tuple of length {len(coeffs)}")
                    if len(coeffs) == 2:
                        ll, others = coeffs
                        print(f"Type of ll: {type(ll)}, Type of others: {type(others)}")
                        
                        # Try different approaches to get the detail coefficients
                        if isinstance(others, tuple):
                            print(f"Others is a tuple of length {len(others)}")
                            if len(others) == 3:
                                lh, hl, hh = others
                            else:
                                print(f"Unexpected length of 'others' tuple: {len(others)}")
                                # Try to access by index anyway
                                lh, hl, hh = others[0], others[1], others[2]
                        elif hasattr(others, '__getitem__'):
                            # If it's list-like but not a tuple
                            print(f"Using __getitem__ to access detail coefficients")
                            lh, hl, hh = others[0], others[1], others[2]
                        else:
                            print(f"Cannot extract detail coefficients from: {type(others)}")
                            # Use zeros as placeholders
                            if isinstance(ll, torch.Tensor):
                                shape = ll.shape
                                lh = torch.zeros_like(ll)
                                hl = torch.zeros_like(ll)
                                hh = torch.zeros_like(ll)
                            else:
                                # If ll is not a tensor, just use scalar zeros
                                lh, hl, hh = 0, 0, 0
                    else:
                        print(f"Unexpected tuple length: {len(coeffs)}")
                        # Try to extract directly if possible
                        if len(coeffs) >= 4:
                            ll, lh, hl, hh = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
                        else:
                            raise ValueError(f"Cannot extract coefficients from tuple of length {len(coeffs)}")
                else:
                    print(f"Coeffs is not a tuple but a {type(coeffs)}")
                    # Try to see if it's a list or similar
                    if hasattr(coeffs, '__getitem__') and hasattr(coeffs, '__len__'):
                        print(f"Trying to treat coeffs as a sequence of length {len(coeffs)}")
                        ll = coeffs[0]
                        # Check what the structure of ll is
                        if hasattr(ll, 'shape'):
                            print(f"Shape of ll: {ll.shape}")
                        # Try to get the other coefficients
                        if len(coeffs) >= 4:
                            lh, hl, hh = coeffs[1], coeffs[2], coeffs[3]
                        else:
                            print(f"Not enough items in coeffs: {len(coeffs)}")
                            raise ValueError("Cannot extract all required coefficients")
                    else:
                        print(f"Cannot determine the structure of coeffs")
                        raise ValueError("Unsupported coefficient structure")
                    
                # Convert to PyTorch tensors directly on GPU
                if not isinstance(ll, torch.Tensor):
                    print(f"Converting coefficients to tensors")
                    ll_t = torch.tensor(ll, dtype=torch.float32, device=device)
                    lh_t = torch.tensor(lh, dtype=torch.float32, device=device)
                    hl_t = torch.tensor(hl, dtype=torch.float32, device=device)
                    hh_t = torch.tensor(hh, dtype=torch.float32, device=device)
                else:
                    ll_t = ll.to(device)
                    lh_t = lh.to(device)
                    hl_t = hl.to(device)
                    hh_t = hh.to(device)
                
                # Add variance features - keep as tensors
                var_features = [
                    torch.var(ll_t).reshape(1),
                    torch.var(lh_t).reshape(1),
                    torch.var(hl_t).reshape(1),
                    torch.var(hh_t).reshape(1)
                ]
                features.extend(var_features)
                
                print(f"Successfully processed wavelet coefficients for {channel_name}")
            except Exception as e:
                print(f"Error processing coefficients for {channel_name}: {e}")
                # Add zeros as fallback - as GPU tensors
                features.extend([torch.zeros(1, device=device) for _ in range(4)])
    else:
        print(f"polar_coeffs is not a dictionary but a {type(polar_coeffs)}")
        # Skip wavelet coefficient processing if structure is unknown
    
    # Concatenate all feature tensors into one
    feature_vector = torch.cat([f.flatten() for f in features])
    print(f"Extracted feature vector with {feature_vector.size(0)} features")
    return feature_vector