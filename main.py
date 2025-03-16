import os
import argparse
import torch
import numpy as np
import time
from pathlib import Path

from modules.BatchFeatureExtractor import OptimizedBatchProcessor
from modules.models import RegressionDLNN
from modules.preprocessing import preprocess_image

def main():
    parser = argparse.ArgumentParser(description='Image Forgery Detection System')
    parser.add_argument('--mode', choices=['train', 'test', 'precompute', 'single'], required=True, 
                        help='Operating mode: train, test, precompute features, or single image test')
    parser.add_argument('--input_dir', type=str,
                        help='Directory containing input images')
    parser.add_argument('--image_path', type=str,
                        help='Path to single image for testing')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--model_path', type=str, default='models/forgery_detection_model.pth',
                        help='Path to save/load model')
    parser.add_argument('--features_path', type=str, default='features/precomputed_features.npz',
                        help='Path to save/load precomputed features')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    parser.add_argument('--fp16', action='store_true',
                        help='Use half precision (FP16) operations')
    parser.add_argument('--authentic_dir', type=str,
                        help='Directory containing authentic images (for training)')
    parser.add_argument('--forged_dir', type=str,
                        help='Directory containing forged images (for training)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.features_path), exist_ok=True)
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == 'precompute':
        precompute_features(args)
    elif args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'single':
        test_single_image(args)

def precompute_features(args):
    """Precompute features for training or testing"""
    print(f"Precomputing features with batch size {args.batch_size}...")
    
    # Initialize the batch processor
    processor = OptimizedBatchProcessor(
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_fp16=args.fp16
    )
    
    # Process authentic images if provided
    authentic_features = None
    if args.authentic_dir:
        print(f"\nProcessing authentic images from {args.authentic_dir}")
        authentic_features = processor.precompute_features(
            directory=args.authentic_dir,
            label=0,  # 0 = authentic
            save_path=f"{os.path.splitext(args.features_path)[0]}_authentic.npz"
        )
    
    # Process forged images if provided
    forged_features = None
    if args.forged_dir:
        print(f"\nProcessing forged images from {args.forged_dir}")
        forged_features = processor.precompute_features(
            directory=args.forged_dir,
            label=1,  # 1 = forged
            save_path=f"{os.path.splitext(args.features_path)[0]}_forged.npz"
        )
    
    # Process general input directory if provided
    if args.input_dir and args.input_dir != args.authentic_dir and args.input_dir != args.forged_dir:
        print(f"\nProcessing images from {args.input_dir}")
        processor.precompute_features(
            directory=args.input_dir,
            save_path=args.features_path
        )
    
    # Combine authentic and forged features if both were computed
    if authentic_features and forged_features:
        print("\nCombining authentic and forged features...")
        combined_features = {
            'paths': authentic_features['paths'] + forged_features['paths'],
            'features': np.vstack([authentic_features['features'], forged_features['features']]),
            'labels': authentic_features['labels'] + forged_features['labels']
        }
        
        np.savez(
            args.features_path,
            paths=combined_features['paths'],
            features=combined_features['features'],
            labels=combined_features['labels']
        )
        print(f"Saved combined features to {args.features_path}")

def train_model(args):
    """Train the model using precomputed features"""
    if not (args.authentic_dir and args.forged_dir) and not os.path.exists(args.features_path):
        print("Error: For training, either provide authentic_dir and forged_dir arguments, "
              "or precompute features first and provide features_path.")
        return
    
    # Load or compute features
    if not os.path.exists(args.features_path):
        print("Precomputing features for training...")
        precompute_features(args)
    
    print(f"Loading precomputed features from {args.features_path}...")
    data = np.load(args.features_path)
    features = data['features']
    labels = data['labels']
    
    # Convert to torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(features, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)
    
    # Create and train the model
    input_dim = features.shape[1]
    model = RegressionDLNN(input_dim)
    model.fit(
        X, y, 
        epochs=args.epochs, 
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_fp16=args.fp16
    )
    
    # Save the model
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

def test_model(args):
    """Test the model on images in the input directory"""
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    # Load the model
    model = RegressionDLNN.load(args.model_path)
    
    # Initialize the batch processor with the loaded model
    processor = OptimizedBatchProcessor(
        model=model,
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_fp16=args.fp16
    )
    
    print(f"Processing images from {args.input_dir}...")
    start_time = time.time()
    
    # Process all images in the directory
    results_file = os.path.join(args.output_dir, 'detection_results.txt')
    results = processor.process_directory(
        directory=args.input_dir,
        results_file=results_file
    )
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to {results_file}")

def test_single_image(args):
    """Test the model on a single image"""
    if not args.image_path:
        print("Error: Please provide --image_path argument for single image testing")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    # Load the model
    model = RegressionDLNN.load(args.model_path)
    
    # Initialize the batch processor with the loaded model
    processor = OptimizedBatchProcessor(
        model=model,
        batch_size=1,  # Set to 1 for single image
        num_workers=1,
        use_fp16=args.fp16
    )
    
    print(f"Processing single image: {args.image_path}")
    start_time = time.time()
    
    # Create a temporary directory with the image
    temp_dir = os.path.join(args.output_dir, 'temp_single_image')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy the image to the temporary directory
    import shutil
    image_name = os.path.basename(args.image_path)
    temp_image_path = os.path.join(temp_dir, image_name)
    shutil.copy2(args.image_path, temp_image_path)
    
    try:
        # Process the image
        features = processor._process_image_batch([args.image_path])
        
        if features is not None and len(features) > 0:
            # Make prediction
            predictions, confidences = processor.batch_predict(features)
            
            if predictions is not None and len(predictions) > 0:
                # Display results
                prediction = predictions[0]
                confidence = confidences[0]
                result = "FORGED" if prediction == 1 else "AUTHENTIC"
                
                print(f"\nResult: {result}")
                print(f"Confidence: {confidence:.4f}")
                print(f"Prediction value: {prediction}")
                
                # Save the result
                result_file = os.path.join(args.output_dir, f"{os.path.splitext(image_name)[0]}_result.txt")
                with open(result_file, 'w') as f:
                    f.write(f"Image: {args.image_path}\n")
                    f.write(f"Result: {result}\n")
                    f.write(f"Confidence: {confidence:.4f}\n")
                    f.write(f"Prediction value: {prediction}\n")
                
                print(f"Result saved to {result_file}")
            else:
                print("Failed to make prediction")
        else:
            print("Failed to extract features")
    
    except Exception as e:
        print(f"Error processing image: {e}")
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()