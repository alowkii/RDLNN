import cv2
import numpy as np

def preprocess_image(image_path):
    """
    Convert input RGB image to YCbCr color format for better feature extraction
    
    Args:
        image_path: Path to the input image
        
    Returns:
        YCbCr image
    """
    # Read the input image
    img = cv2.imread(image_path)
    
    # Check if image is loaded properly
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Convert from BGR (OpenCV default) to YCbCr color space
    # Y: luminance component
    # Cb: blue-difference chroma component
    # Cr: red-difference chroma component
    ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    return ycbcr_img