import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from modules.preprocessing import preprocess_image
from modules.imageDecomposition import polar_dyadic_wavelet_transform
from modules.featureExtraction import extract_features
from modules.rdlnn import RegressionDLNN
from modules.batch_processing import batch_process_directory
import gc

def setup_cuda_environment():
    """Configure CUDA environment for optimal performance"""
    # Set CUDA device
    torch.cuda.set_device(0)
    
    # Enable TF32 for faster matrix multiplication on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set cudnn benchmark mode for optimal performance
    torch.backends.cudnn.benchmark = True
    
    # Print CUDA information
    device = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch CUDA: {torch.cuda.is_available()}")

def process_single_image(image_path, model, stream=None):
    """
    Process a single image and determine if it's authentic or forged
    
    Args:
        image_path: Path to the image file
        model: Trained RegressionDLNN model
        stream: CUDA stream for concurrent processing
        
    Returns:
        (prediction, confidence): Tuple with prediction (0: authentic, 1: forged) and confidence level
    """
    try:
        print(f"Processing image: {image_path}")
        
        with torch.cuda.stream(stream) if stream else torch.cuda.stream(torch.cuda.default_stream()):
            # Preprocess the image to YCbCr
            ycbcr_img = preprocess_image(image_path)
            if ycbcr_img is None:
                print("Failed to preprocess image")
                return None, 0
            
            # Apply polar dyadic wavelet transform
            pdywt_coeffs = polar_dyadic_wavelet_transform(ycbcr_img)
            
            # Convert tensor to dictionary format for feature extraction
            ycbcr_img_dict = {
                'y': ycbcr_img[0],
                'cb': ycbcr_img[1],
                'cr': ycbcr_img[2]
            }
            
            # Extract features
            feature_vector = extract_features(pdywt_coeffs, ycbcr_img_dict)
            
            # Make prediction
            prediction, confidence = model.predict(np.array(feature_vector))
            
            result = "FORGED" if prediction == 1 else "AUTHENTIC"
            print(f"Result: {result} (Confidence: {confidence:.2f})")
            
            # Synchronize stream to ensure all GPU operations are complete
            if stream:
                stream.synchronize()
            
        return prediction, confidence
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, 0
    finally:
        # Free CUDA memory
        torch.cuda.empty_cache()

def train_model(train_dir, model_path=None, epochs=50, batch_size=32):
    """
    Train the RDLNN model using images from the specified directory
    
    Args:
        train_dir: Directory containing 'authentic' and 'forged' subdirectories with training images
        model_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Trained model
    """
    # Configure CUDA for optimal training performance
    torch.backends.cudnn.benchmark = True
    
    # Check if required directories exist
    authentic_dir = os.path.join(train_dir, 'authentic')
    forged_dir = os.path.join(train_dir, 'forged')
    
    if not os.path.exists(authentic_dir) or not os.path.exists(forged_dir):
        print(f"Training directory must contain 'authentic' and 'forged' subdirectories")
        return None
    
    # Create CUDA streams for parallel processing
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    print(f"Extracting features from authentic images...")
    authentic_features = []
    authentic_labels = []
    
    # Get list of authentic images
    authentic_images = [img for img in os.listdir(authentic_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process authentic images with progress bar and stream
    for img_name in tqdm(authentic_images, desc="Processing authentic images"):
        try:
            with torch.cuda.stream(stream1):
                img_path = os.path.join(authentic_dir, img_name)
                ycbcr_img = preprocess_image(img_path)
                pdywt_coeffs = polar_dyadic_wavelet_transform(ycbcr_img)
                
                ycbcr_img_dict = {
                    'y': ycbcr_img[0],
                    'cb': ycbcr_img[1],
                    'cr': ycbcr_img[2]
                }
                
                feature_vector = extract_features(pdywt_coeffs, ycbcr_img_dict)
                authentic_features.append(feature_vector)
                authentic_labels.append(0)  # 0 = authentic
        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            
        # Periodically clear cache to prevent OOM errors
        if len(authentic_features) % 50 == 0:
            torch.cuda.empty_cache()
    
    print(f"Extracting features from forged images...")
    forged_features = []
    forged_labels = []
    
    # Get list of forged images
    forged_images = [img for img in os.listdir(forged_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process forged images with progress bar and stream
    for img_name in tqdm(forged_images, desc="Processing forged images"):
        try:
            with torch.cuda.stream(stream2):
                img_path = os.path.join(forged_dir, img_name)
                ycbcr_img = preprocess_image(img_path)
                pdywt_coeffs = polar_dyadic_wavelet_transform(ycbcr_img)
                
                ycbcr_img_dict = {
                    'y': ycbcr_img[0],
                    'cb': ycbcr_img[1],
                    'cr': ycbcr_img[2]
                }
                
                feature_vector = extract_features(pdywt_coeffs, ycbcr_img_dict)
                forged_features.append(feature_vector)
                forged_labels.append(1)  # 1 = forged
        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            return None, 0
        finally:
            torch.cuda.empty_cache()
            
        # Periodically clear cache to prevent OOM errors
        if len(forged_features) % 50 == 0:
            torch.cuda.empty_cache()
    
    # Combine features and labels
    X = np.vstack(authentic_features + forged_features)
    y = np.array(authentic_labels + forged_labels)
    
    # Print training data info
    print(f"Training data: {len(authentic_features)} authentic images, {len(forged_features)} forged images")
    print(f"Feature vector shape: {X.shape}")
    
    # Free memory before training
    del authentic_features, forged_features
    gc.collect()
    torch.cuda.empty_cache()
    
    # Initialize model with input dimension matching feature vector
    input_dim = X.shape[1]
    model = RegressionDLNN(input_dim)
    
    # Train the model
    print(f"Training model with {epochs} epochs and batch size {batch_size}...")
    history = model.train_model(X, y, epochs=epochs, batch_size=batch_size)
    
    # Save the model if path is provided
    if model_path:
        model.save_model(model_path)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Image Forgery Detection using PDyWT and RDLNN')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', type=str, required=True, help='Path to training data directory')
    train_parser.add_argument('--model', type=str, default='models/rdlnn_model.pth', help='Path to save trained model')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    train_parser.add_argument('--early-stopping', type=int, default=5, help='Early stopping patience (epochs)')
    
    # Detect command for single image
    detect_parser = subparsers.add_parser('detect', help='Detect forgery in a single image')
    detect_parser.add_argument('--image', type=str, required=True, help='Path to image for forgery detection')
    detect_parser.add_argument('--model', type=str, default='models/rdlnn_model.pth', help='Path to trained model')
    
    # Batch process command
    batch_parser = subparsers.add_parser('batch', help='Process all images in a directory')
    batch_parser.add_argument('--dir', type=str, required=True, help='Directory containing images to process')
    batch_parser.add_argument('--model', type=str, default='models/rdlnn_model.pth', help='Path to trained model')
    batch_parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Set up CUDA environment
    setup_cuda_environment()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. This implementation requires GPU acceleration.")
        return
    
    # Handle commands
    if args.command == 'train':
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        train_model(args.data, args.model, args.epochs, args.batch_size)
        
    elif args.command == 'detect':
        # Check if model file exists
        if not os.path.exists(args.model):
            print(f"Model file not found: {args.model}")
            print("Please train the model first or provide a valid model path.")
            return
            
        # Check if image file exists
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            return
            
        # Load model
        input_dim = 100  # Placeholder, will be overwritten when loading model
        model = RegressionDLNN(input_dim)
        if model.load_model(args.model):
            # Create a dedicated CUDA stream for this operation
            stream = torch.cuda.Stream()
            process_single_image(args.image, model, stream)
        
    elif args.command == 'batch':
        # Check if model file exists
        if not os.path.exists(args.model):
            print(f"Model file not found: {args.model}")
            print("Please train the model first or provide a valid model path.")
            return
            
        # Check if directory exists
        if not os.path.exists(args.dir):
            print(f"Directory not found: {args.dir}")
            return
            
        # Load model
        input_dim = 100  # Placeholder, will be overwritten when loading model
        model = RegressionDLNN(input_dim)
        if model.load_model(args.model):
            batch_process_directory(args.dir, model, num_workers=args.workers)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()