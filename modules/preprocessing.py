import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """
    Load image and convert to YCbCr color space using GPU acceleration
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PyTorch tensor in YCbCr color space [C, H, W] format on GPU
    """
    try:
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This implementation requires GPU acceleration.")
            
        device = torch.device("cuda")

        # Load image and convert to tensor
        img = Image.open(image_path).convert("RGB")
        
        # Use transforms to efficiently convert to tensor
        img_tensor = transforms.ToTensor()(img).to(device)
        
        # Resize if image is too large (optional, helps with memory issues)
        # max_size = 1024
        # if max(img_tensor.shape[1], img_tensor.shape[2]) > max_size:
        #     scale = max_size / max(img_tensor.shape[1], img_tensor.shape[2])
        #     new_h, new_w = int(img_tensor.shape[1] * scale), int(img_tensor.shape[2] * scale)
        #     img_tensor = transforms.functional.resize(img_tensor, (new_h, new_w))

        # Define RGB to YCbCr transformation matrix directly on GPU
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
        
        # Reshape back to [H, W, C] then to [C, H, W]
        ycbcr_tensor = ycbcr_reshaped.reshape(h, w, 3).permute(2, 0, 1)
        
        return ycbcr_tensor

    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None