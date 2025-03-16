import torch
import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(image_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load image and convert to tensor
        img = Image.open(image_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img).to(device)

        # Define RGB to YCbCr transformation matrix as a PyTorch tensor
        transform_matrix = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.169, -0.331, 0.5],
            [0.5, -0.419, -0.081]
        ], dtype=torch.float32, device=device)

        # Apply transformation in PyTorch without converting to NumPy
        ycbcr_tensor = torch.tensordot(img_tensor.permute(1, 2, 0), transform_matrix.T, dims=1)

        # Add offsets for Cb and Cr channels
        ycbcr_tensor[..., 1:] += 128.0

        # Rearrange back to [C, H, W] format
        ycbcr_tensor = ycbcr_tensor.permute(2, 0, 1)

        print(f"Image converted to YCbCr with shape: {ycbcr_tensor.shape}")
        return ycbcr_tensor

    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        return None
