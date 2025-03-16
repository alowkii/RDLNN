import torch
import torch.nn.functional as F
import numpy as np
import gc

def extract_features(polar_coeffs, ycbcr_img):
    """
    Extract important features from decomposed image using optimized PyTorch with CUDA:
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
        Feature vector containing all extracted features
    """
    # Force CUDA - no fallback
    device = torch.device("cuda")
    features = []
    
    # Get channels from input
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
                # Move to GPU if not already there
                y_channel = y_data.to(device, non_blocking=True)
                cb_channel = cb_data.to(device, non_blocking=True)
                cr_channel = cr_data.to(device, non_blocking=True)
        else:
            raise ValueError(f"Dict doesn't contain expected keys. Available keys: {list(ycbcr_img.keys())}")
    else:
        raise TypeError(f"Expected dict for ycbcr_img, got {type(ycbcr_img)}")
    
    # Create a CUDA stream for concurrent operations
    stream = torch.cuda.Stream()
    
    # Process within the stream
    with torch.cuda.stream(stream):
        # 1. SURF-like features using optimized PyTorch operations
        y_channel_2d = y_channel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        
        # Pre-compute Sobel filters on GPU
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        # Compute gradients using convolution with non-blocking memory transfer
        grad_x = F.conv2d(y_channel_2d, sobel_x, padding=1)
        grad_y = F.conv2d(y_channel_2d, sobel_y, padding=1)
        
        # Compute gradient magnitude (use in-place operations where possible)
        grad_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
        
        # Extract histogram of gradients from specific regions
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
                
                # Optimize histogram computation
                max_val = torch.max(region_magnitude).item() + 1e-8
                hist = torch.histc(region_magnitude, bins=4, min=0, max=max_val)
                
                # Use in-place division for normalization
                hist_sum = hist.sum().item() + 1e-8
                hist.div_(hist_sum)
            
            surf_like_features.append(hist)
        
        # Concatenate features efficiently
        surf_features = torch.cat(surf_like_features).flatten()
        features.append(surf_features[:20])  # Take first 20 features
        
        # 2. Edge features - just use the pre-computed gradient
        edge_density = torch.mean(grad_magnitude).reshape(1)
        features.append(edge_density)
        
        # 3. Correlation between channels - optimized computation
        def compute_correlation(x, y):
            # Compute means
            x_mean = x.mean()
            y_mean = y.mean()
            
            # Center the data
            x_centered = x - x_mean
            y_centered = y - y_mean
            
            # Compute correlation
            numerator = torch.sum(x_centered * y_centered)
            denominator = torch.sqrt(torch.sum(x_centered**2) * torch.sum(y_centered**2) + 1e-8)
            
            return numerator / denominator
        
        # Flatten tensors for correlation
        y_flat = y_channel.flatten()
        cb_flat = cb_channel.flatten()
        cr_flat = cr_channel.flatten()
        
        # Compute correlations
        corr_y_cb = compute_correlation(y_flat, cb_flat).reshape(1)
        corr_y_cr = compute_correlation(y_flat, cr_flat).reshape(1)
        corr_cb_cr = compute_correlation(cb_flat, cr_flat).reshape(1)
        
        features.extend([corr_y_cb, corr_y_cr, corr_cb_cr])
        
        # 4. Color histogram features - optimized for GPU
        color_hists = []
        for channel in [y_channel, cb_channel, cr_channel]:
            # Get max value for normalization
            channel_max = channel.max().item()
            
            if channel_max > 0:
                # Compute histogram
                hist = torch.histc(channel, bins=8, min=0, max=channel_max + 1e-8)
                # Normalize
                hist.div_(hist.sum() + 1e-8)
            else:
                # Uniform distribution if channel is all zeros
                hist = torch.ones(8, device=device) / 8
                
            color_hists.append(hist)
        
        features.extend(color_hists)
        
        # 5. GLCM-based homogeneity features - optimized GPU implementation
        def calculate_homogeneity(channel, distance=1):
            # Optimize for memory usage and speed
            # Reduce channel to 8 gray levels
            max_val = channel.max().item()
            if max_val > 0:
                channel_reduced = (channel / (max_val / 7.999)).long().clamp(0, 7)
            else:
                # Return perfect homogeneity for uniform channel
                return torch.tensor(1.0, device=device).reshape(1)
                
            h, w = channel.shape
            
            # Return early if image too small
            if w <= distance:
                return torch.tensor(1.0, device=device).reshape(1)
            
            # Initialize GLCM
            glcm = torch.zeros((8, 8), device=device)
            
            # Optimize computation with tensor operations
            # Get all pixel pairs at specified distance
            left_pixels = channel_reduced[:, :-distance].reshape(-1)
            right_pixels = channel_reduced[:, distance:].reshape(-1)
            
            # Create index tensors for efficient histogram computation
            batch_size = left_pixels.size(0)
            indices = torch.stack([left_pixels, right_pixels], dim=1)
            
            # Create coordinate tensors
            for i in range(batch_size):
                i_coord = indices[i, 0]
                j_coord = indices[i, 1]
                glcm[i_coord, j_coord] += 1
            
            # Normalize GLCM
            glcm_sum = glcm.sum()
            if glcm_sum > 0:
                glcm.div_(glcm_sum)
            else:
                return torch.tensor(1.0, device=device).reshape(1)
            
            # Calculate homogeneity
            i_indices, j_indices = torch.meshgrid(torch.arange(8, device=device), 
                                                torch.arange(8, device=device), 
                                                indexing='ij')
            
            # Compute weights matrix
            weights = 1.0 / (1.0 + torch.abs(i_indices - j_indices).float())
            
            # Compute homogeneity score
            homogeneity = torch.sum(glcm * weights).reshape(1)
            
            return homogeneity
        
        # Compute homogeneity for each channel
        homogeneity_features = [
            calculate_homogeneity(y_channel),
            calculate_homogeneity(cb_channel),
            calculate_homogeneity(cr_channel)
        ]
        features.extend(homogeneity_features)
        
        # 6. Variance features from wavelet coefficients - optimized error handling
        if isinstance(polar_coeffs, dict):
            for channel_name, coeffs in polar_coeffs.items():
                try:
                    # Handle different coefficient structures
                    if isinstance(coeffs, tuple):
                        if len(coeffs) == 2:
                            ll, others = coeffs
                            
                            if isinstance(others, tuple) and len(others) == 3:
                                lh, hl, hh = others
                            else:
                                # Try to extract by index
                                try:
                                    lh, hl, hh = others[0], others[1], others[2]
                                except:
                                    if isinstance(ll, torch.Tensor):
                                        lh = hl = hh = torch.zeros_like(ll)
                                    else:
                                        lh = hl = hh = 0
                        elif len(coeffs) >= 4:
                            ll, lh, hl, hh = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
                        else:
                            raise ValueError(f"Cannot extract coefficients from tuple of length {len(coeffs)}")
                    else:
                        # Try to extract from list-like structure
                        if hasattr(coeffs, '__getitem__') and hasattr(coeffs, '__len__'):
                            ll = coeffs[0]
                            if len(coeffs) >= 4:
                                lh, hl, hh = coeffs[1], coeffs[2], coeffs[3]
                            else:
                                raise ValueError("Cannot extract all required coefficients")
                        else:
                            raise ValueError("Unsupported coefficient structure")
                    
                    # Convert to PyTorch tensors on GPU
                    if not isinstance(ll, torch.Tensor):
                        ll_t = torch.tensor(ll, dtype=torch.float32, device=device)
                        lh_t = torch.tensor(lh, dtype=torch.float32, device=device)
                        hl_t = torch.tensor(hl, dtype=torch.float32, device=device)
                        hh_t = torch.tensor(hh, dtype=torch.float32, device=device)
                    else:
                        # Move to GPU with non-blocking transfer
                        ll_t = ll.to(device, non_blocking=True)
                        lh_t = lh.to(device, non_blocking=True)
                        hl_t = hl.to(device, non_blocking=True)
                        hh_t = hh.to(device, non_blocking=True)
                    
                    # Compute variance efficiently
                    var_features = [
                        torch.var(ll_t).reshape(1),
                        torch.var(lh_t).reshape(1),
                        torch.var(hl_t).reshape(1),
                        torch.var(hh_t).reshape(1)
                    ]
                    features.extend(var_features)
                    
                except Exception as e:
                    # Add zeros as fallback
                    features.extend([torch.zeros(1, device=device) for _ in range(4)])
        else:
            # Skip wavelet coefficient processing if structure is unknown
            features.extend([torch.zeros(1, device=device) for _ in range(4)])
    
    # Ensure all CUDA operations are completed
    stream.synchronize()
    
    # Process features for return
    processed_features = []
    for f in features:
        if isinstance(f, torch.Tensor):
            # Move to CPU and convert to numpy
            processed_features.append(f.cpu().numpy())
        elif isinstance(f, list):
            # Process list of tensors
            for item in f:
                if isinstance(item, torch.Tensor):
                    processed_features.append(item.cpu().numpy())
                else:
                    processed_features.append(np.array([item]))
        else:
            processed_features.append(np.array([f]))
    
    # Concatenate all features
    feature_vector = np.concatenate([f.flatten() for f in processed_features])
    
    # Clean up unused tensors
    del features, processed_features
    gc.collect()
    torch.cuda.empty_cache()
    
    # Return as a row vector
    return feature_vector.reshape(1, -1)