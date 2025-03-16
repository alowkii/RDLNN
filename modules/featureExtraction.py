import torch
import torch.nn.functional as F
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor

class BatchFeatureExtractor:
    """
    Optimized batch feature extractor for image forgery detection
    Processes multiple images at once using efficient GPU operations
    """
    
    def __init__(self, device=None, batch_size=16, num_workers=4, use_fp16=True):
        """
        Initialize the batch feature extractor
        
        Args:
            device: PyTorch device (default: CUDA if available)
            batch_size: Number of images to process in a batch
            num_workers: Number of parallel CPU workers for data loading
            use_fp16: Whether to use half precision (FP16) operations
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_fp16 = use_fp16
        
        # Pre-compute filters and tensors used in feature extraction
        self._initialize_filters()
        
        # Create CUDA streams for overlapping operations
        self.streams = [torch.cuda.Stream(device=self.device) for _ in range(3)]
        
    def _initialize_filters(self):
        """Pre-compute filters and tensors used in feature extraction"""
        # Sobel filters
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float16 if self.use_fp16 else torch.float32,
                          device=self.device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float16 if self.use_fp16 else torch.float32,
                          device=self.device).view(1, 1, 3, 3)
                          
        # RGB to YCbCr transformation matrix
        self.rgb_to_ycbcr = torch.tensor([
            [0.299, 0.587, 0.114],      # Y
            [-0.1687, -0.3313, 0.5],    # Cb
            [0.5, -0.4187, -0.0813]     # Cr
        ], dtype=torch.float16 if self.use_fp16 else torch.float32, device=self.device)
        
    def extract_batch_features(self, image_batch, wavelet_coeffs_batch):
        """
        Extract features from a batch of images
        
        Args:
            image_batch: Batch of images as tensor [B, C, H, W]
            wavelet_coeffs_batch: Batch of wavelet coefficients
            
        Returns:
            Tensor of feature vectors for the batch [B, feature_dim]
        """
        batch_size = image_batch.shape[0]
        
        # List to store feature vectors for each image
        batch_features = []
        
        # Process in streams to overlap computation
        with torch.cuda.stream(self.streams[0]):
            # 1. Extract edge features (gradient-based) for the whole batch
            edge_features = self._extract_edge_features_batch(image_batch[:, 0:1, :, :])  # Y channel
            batch_features.append(edge_features)
        
        with torch.cuda.stream(self.streams[1]):
            # 2. Extract color correlation features
            color_features = self._extract_color_features_batch(image_batch)
            batch_features.append(color_features)
            
        with torch.cuda.stream(self.streams[2]):
            # 3. Extract wavelet-based features
            if wavelet_coeffs_batch:
                wavelet_features = self._extract_wavelet_features_batch(wavelet_coeffs_batch)
                batch_features.append(wavelet_features)
        
        # Synchronize streams
        for stream in self.streams:
            stream.synchronize()
            
        # Concatenate all features along feature dimension
        feature_vectors = torch.cat(batch_features, dim=1)
        
        return feature_vectors
        
    def _extract_edge_features_batch(self, y_batch):
        """
        Extract edge features for a batch of Y channel images
        
        Args:
            y_batch: Batch of Y channel images [B, 1, H, W]
            
        Returns:
            Edge features tensor [B, edge_feature_dim]
        """
        batch_size = y_batch.shape[0]
        
        # Add automatic mixed precision
        dtype = torch.float16 if self.use_fp16 else torch.float32
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            # Apply Sobel filters to get gradients
            grad_x = F.conv2d(y_batch, self.sobel_x, padding=1)
            grad_y = F.conv2d(y_batch, self.sobel_y, padding=1)
            
            # Compute gradient magnitude
            grad_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
            
            # Global edge density feature
            edge_density = torch.mean(grad_magnitude, dim=[1, 2, 3]).unsqueeze(1)
            
            # Regional edge features (divide into 3x3 grid)
            b, c, h, w = grad_magnitude.shape
            h_step, w_step = h // 3, w // 3
            
            regional_features = []
            for i in range(3):
                for j in range(3):
                    h_start, h_end = i * h_step, (i + 1) * h_step
                    w_start, w_end = j * w_step, (j + 1) * w_step
                    
                    # Handle edge cases
                    if i == 2: h_end = h
                    if j == 2: w_end = w
                    
                    # Extract region and compute features
                    region = grad_magnitude[:, :, h_start:h_end, w_start:w_end]
                    
                    # Compute mean and variance of the region
                    region_mean = torch.mean(region, dim=[1, 2, 3]).unsqueeze(1)
                    region_var = torch.var(region, dim=[1, 2, 3]).unsqueeze(1)
                    
                    regional_features.extend([region_mean, region_var])
            
            # Concat all edge features
            all_edge_features = torch.cat([edge_density] + regional_features, dim=1)
            
        return all_edge_features
    
    def _extract_color_features_batch(self, ycbcr_batch):
        """
        Extract color features from a batch of YCbCr images
        
        Args:
            ycbcr_batch: Batch of YCbCr images [B, 3, H, W]
            
        Returns:
            Color features tensor [B, color_feature_dim]
        """
        batch_size = ycbcr_batch.shape[0]
        
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            # 1. Channel correlations
            # Reshape and compute correlations
            b, c, h, w = ycbcr_batch.shape
            flattened = ycbcr_batch.view(b, c, -1)  # [B, 3, H*W]
            
            # Mean of each channel
            means = torch.mean(flattened, dim=2, keepdim=True)  # [B, 3, 1]
            
            # Center the data
            centered = flattened - means
            
            # Compute correlation matrix for each image in batch
            corr_features = []
            for i in range(batch_size):
                # Compute correlation between Y-Cb, Y-Cr and Cb-Cr
                norm_y = torch.norm(centered[i, 0])
                norm_cb = torch.norm(centered[i, 1])
                norm_cr = torch.norm(centered[i, 2])
                
                # Avoid division by zero
                eps = 1e-8
                
                corr_y_cb = torch.sum(centered[i, 0] * centered[i, 1]) / (norm_y * norm_cb + eps)
                corr_y_cr = torch.sum(centered[i, 0] * centered[i, 2]) / (norm_y * norm_cr + eps)
                corr_cb_cr = torch.sum(centered[i, 1] * centered[i, 2]) / (norm_cb * norm_cr + eps)
                
                corr_features.append(torch.stack([corr_y_cb, corr_y_cr, corr_cb_cr]))
            
            corr_tensor = torch.stack(corr_features)
            
            # 2. Color histogram features (simplified)
            # Use 8 bins for each channel
            histograms = []
            for channel_idx in range(3):
                channel_data = ycbcr_batch[:, channel_idx]
                
                # Compute min and max for each image in batch
                channel_min = torch.min(channel_data.view(b, -1), dim=1)[0]
                channel_max = torch.max(channel_data.view(b, -1), dim=1)[0] + 1e-8
                
                hist_features = []
                for i in range(batch_size):
                    # Normalize to 0-1 range
                    normalized = (channel_data[i] - channel_min[i]) / (channel_max[i] - channel_min[i])
                    
                    # Compute histogram with 8 bins
                    hist = torch.histc(normalized.flatten(), bins=8, min=0, max=1)
                    
                    # Normalize histogram
                    hist = hist / torch.sum(hist)
                    hist_features.append(hist)
                
                histograms.append(torch.stack(hist_features))
            
            # Concatenate histograms [B, 3*8]
            hist_tensor = torch.cat([h.reshape(batch_size, -1) for h in histograms], dim=1)
            
            # Combine all color features
            all_color_features = torch.cat([corr_tensor, hist_tensor], dim=1)
            
        return all_color_features
    
    def _extract_wavelet_features_batch(self, wavelet_coeffs_batch):
        """
        Extract wavelet-based features from batch of wavelet coefficients
        
        Args:
            wavelet_coeffs_batch: List of wavelet coefficient dictionaries, one per image
            
        Returns:
            Wavelet features tensor [B, wavelet_feature_dim]
        """
        batch_size = len(wavelet_coeffs_batch)
        
        with torch.cuda.amp.autocast(enabled=self.use_fp16):
            # Preallocate tensor for wavelet features
            # We'll extract variance and energy from each subband
            feature_dim = 12  # 3 channels x 4 subbands (LL, LH, HL, HH)
            wavelet_features = torch.zeros((batch_size, feature_dim), 
                                          dtype=torch.float16 if self.use_fp16 else torch.float32,
                                          device=self.device)
            
            for i, coeffs_dict in enumerate(wavelet_coeffs_batch):
                feature_idx = 0
                
                # Process each color channel
                for channel in ['y', 'cb', 'cr']:
                    if channel in coeffs_dict:
                        coeff_tuple = coeffs_dict[channel]
                        
                        # Extract all 4 subbands (LL, LH, HL, HH)
                        if isinstance(coeff_tuple, tuple) and len(coeff_tuple) >= 4:
                            subbands = coeff_tuple[:4]
                        elif isinstance(coeff_tuple, tuple) and len(coeff_tuple) == 2:
                            # Handle case where coeffs are (LL, (LH, HL, HH))
                            ll, others = coeff_tuple
                            if isinstance(others, tuple) and len(others) == 3:
                                subbands = (ll,) + others
                            else:
                                # Skip if format is unknown
                                continue
                        else:
                            # Skip if format is unknown
                            continue
                            
                        # Calculate variance for each subband
                        for subband in subbands:
                            if isinstance(subband, torch.Tensor):
                                # Calculate variance
                                variance = torch.var(subband)
                                wavelet_features[i, feature_idx] = variance
                            feature_idx += 1
                
        return wavelet_features