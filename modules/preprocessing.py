"""
Image preprocessing functions for the forgery detection system
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Dict, Any, Union
from modules.utils import logger

def preprocess_image(image_path: str) -> Optional[torch.Tensor]:
    """
    Load image and convert to YCbCr color space with device agnostic processing
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PyTorch tensor in YCbCr color space [C, H, W] format, or None if processing failed
    """
    try:
        # Check if CUDA is available but don't require it
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load image and convert to tensor
        img = Image.open(image_path).convert("RGB")
        
        # Use transforms to efficiently convert to tensor
        img_tensor = transforms.ToTensor()(img).to(device)
        
        # Define RGB to YCbCr transformation matrix on appropriate device
        transform_matrix = torch.tensor([
            [0.299, 0.587, 0.114],      # Y
            [-0.1687, -0.3313, 0.5],    # Cb
            [0.5, -0.4187, -0.0813]     # Cr
        ], dtype=torch.float32, device=device)

        # Apply transformation efficiently using batch matrix multiplication
        # Reshape [C, H, W] to [H*W, C]
        c, h, w = img_tensor.shape
        img_reshaped = img_tensor.permute(1, 2, 0).reshape(-1, 3)
        
        # Matrix multiplication for color conversion
        ycbcr_reshaped = torch.matmul(img_reshaped, transform_matrix.T)
        
        # Add offsets to Cb and Cr channels
        ycbcr_reshaped[:, 1:] += 128.0
        
        # Reshape back to [C, H, W]
        ycbcr_tensor = ycbcr_reshaped.reshape(h, w, 3).permute(2, 0, 1)
        
        return ycbcr_tensor

    except Exception as e:
        logger.error(f"Error in preprocessing image {image_path}: {e}")
        return None

def normalize_dimensions(image_batch: list) -> Optional[torch.Tensor]:
    """
    Normalize batch images to have same dimensions
    
    Args:
        image_batch: List of image tensors
    
    Returns:
        Tensor batch with consistent dimensions or None if processing failed
    """
    try:
        # Filter out any None results
        image_batch = [img for img in image_batch if img is not None]
        
        if not image_batch:
            return None
        
        # Get consistent dimensions for batching
        first_h, first_w = image_batch[0].shape[1], image_batch[0].shape[2]
        
        # Find common dimensions - use mode of dimensions to handle varying sizes
        heights = [img.shape[1] for img in image_batch]
        widths = [img.shape[2] for img in image_batch]
        
        # Simple approach: use dimensions of first image
        # Advanced approach would be to resize to mode or to a fixed size
        image_batch = [img for img in image_batch if img.shape[1] == first_h and img.shape[2] == first_w]
        
        # Stack into a single batch tensor
        if not image_batch:
            return None
                
        stacked_tensor = torch.stack(image_batch)  # [B, C, H, W]
        
        return stacked_tensor
        
    except Exception as e:
        logger.error(f"Error normalizing image dimensions: {e}")
        return None

def resize_image(image: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    """
    Resize an image tensor to the target size
    
    Args:
        image: Image tensor [C, H, W]
        target_size: Tuple of (height, width)
    
    Returns:
        Resized image tensor
    """
    # Add batch dimension for interpolate
    with_batch = image.unsqueeze(0)
    
    # Resize
    resized = torch.nn.functional.interpolate(
        with_batch, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )
    
    # Remove batch dimension
    return resized.squeeze(0)

def compute_image_stats(ycbcr_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Compute basic statistics of YCbCr image
    
    Args:
        ycbcr_tensor: YCbCr image tensor [C, H, W]
    
    Returns:
        Dictionary of image statistics
    """
    y_channel = ycbcr_tensor[0]
    cb_channel = ycbcr_tensor[1]
    cr_channel = ycbcr_tensor[2]
    
    stats = {
        'y_mean': y_channel.mean().item(),
        'y_std': y_channel.std().item(),
        'cb_mean': cb_channel.mean().item(),
        'cb_std': cb_channel.std().item(),
        'cr_mean': cr_channel.mean().item(),
        'cr_std': cr_channel.std().item(),
    }
    
    return stats