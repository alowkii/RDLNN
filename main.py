import os
import argparse
import torch
import numpy as np
import time
from pathlib import Path
import gc
import matplotlib.pyplot as plt
import signal
import sys

# Import the improved modules
from modules.rdlnn import RegressionDLNN
from modules.data_handling import precompute_features, load_and_verify_features
from modules.preprocessing import preprocess_image

def signal_handler(sig, frame):
    """
    Handle signals for graceful shutdown
    Ensures proper CUDA cleanup before exit
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    print('Cleaning up resources before exit...')
    if torch.cuda.is_available():
        # Synchronize all CUDA streams first
        torch.cuda.synchronize()
        # Then empty cache
        torch.cuda.empty_cache()
    # Collect Python garbage
    gc.collect()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

def main():
    parser = argparse.ArgumentParser(description='Image Forgery Detection System')
    parser.add_argument('--mode', choices=['train', 'test', 'precompute', 'single', 'analyze'], required=True, 
                        help='Operating mode: train, test, precompute features, single image test, or analyze features')
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
        precompute_mode(args)
    elif args.mode == 'train':
        train_mode(args)
    elif args.mode == 'test':
        test_mode(args)
    elif args.mode == 'single':
        test_single_image(args)
    elif args.mode == 'analyze':
        analyze_features(args)

def precompute_mode(args):
    """Precompute features for training or testing"""
    print(f"Precomputing features with batch size {args.batch_size}...")
    
    # Process authentic images if provided
    if args.authentic_dir:
        print(f"\nProcessing authentic images from {args.authentic_dir}")
        authentic_features = precompute_features(
            directory=args.authentic_dir,
            label=0,  # 0 = authentic
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_fp16=args.fp16,
            save_path=f"{os.path.splitext(args.features_path)[0]}_authentic.npz"
        )
    
    # Process forged images if provided
    if args.forged_dir:
        print(f"\nProcessing forged images from {args.forged_dir}")
        forged_features = precompute_features(
            directory=args.forged_dir,
            label=1,  # 1 = forged
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_fp16=args.fp16,
            save_path=f"{os.path.splitext(args.features_path)[0]}_forged.npz"
        )
    
    # Process general input directory if provided
    if args.input_dir and args.input_dir != args.authentic_dir and args.input_dir != args.forged_dir:
        print(f"\nProcessing images from {args.input_dir}")
        precompute_features(
            directory=args.input_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_fp16=args.fp16,
            save_path=args.features_path
        )
    
    # Combine authentic and forged features if both were computed
    if args.authentic_dir and args.forged_dir:
        print("\nCombining authentic and forged features...")
        # Load the individual feature files
        authentic_data = np.load(f"{os.path.splitext(args.features_path)[0]}_authentic.npz")
        forged_data = np.load(f"{os.path.splitext(args.features_path)[0]}_forged.npz")
        
        # Extract data
        authentic_features = authentic_data['features']
        authentic_paths = authentic_data['paths']
        authentic_labels = authentic_data['labels']
        
        forged_features = forged_data['features']
        forged_paths = forged_data['paths']
        forged_labels = forged_data['labels']
        
        # Combine the data
        combined_features = np.vstack([authentic_features, forged_features])
        combined_paths = list(authentic_paths) + list(forged_paths)
        combined_labels = list(authentic_labels) + list(forged_labels)
        
        # Save combined data
        np.savez(
            args.features_path,
            features=combined_features,
            paths=combined_paths,
            labels=combined_labels
        )
        
        print(f"Saved combined features to {args.features_path}")
        print(f"Combined dataset: {len(combined_features)} samples "
              f"({len(authentic_features)} authentic, {len(forged_features)} forged)")

def train_mode(args):
    """Train the model using precomputed features"""
    if not (args.authentic_dir and args.forged_dir) and not os.path.exists(args.features_path):
        print("Error: For training, either provide authentic_dir and forged_dir arguments, "
              "or precompute features first and provide features_path.")
        return
    
    # Load or compute features
    if not os.path.exists(args.features_path):
        print("Precomputing features for training...")
        precompute_mode(args)
    
    # Load and verify features
    features, labels, paths = load_and_verify_features(args.features_path)
    
    if len(features) == 0:
        print("Error: No valid features found. Please check your data.")
        return
    
    if len(labels) == 0:
        print("Error: No labels found. Please make sure your feature file includes labels.")
        return
    
    # Check class balance
    class_counts = np.bincount(labels.astype(int))
    if len(class_counts) < 2:
        print(f"Error: Only found {len(class_counts)} classes. Need at least 2 classes for training.")
        return
    
    # Create and train the model
    input_dim = features.shape[1]
    print(f"Creating model with input dimension: {input_dim}")
    model = RegressionDLNN(input_dim)
    
    # Train the model
    history = model.fit(
        features, labels, 
        epochs=args.epochs, 
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        validation_split=0.2,
        early_stopping=5,
        use_fp16=args.fp16
    )
    
    # Save the model
    model.save(args.model_path)
    
    # Plot training history
    plot_history(history, args.output_dir)

def test_mode(args):
    """Test the model on images in the input directory"""
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    if not args.input_dir:
        print("Error: Please provide --input_dir argument for testing")
        return
    
    # Load the model
    model = RegressionDLNN.load(args.model_path)
    
    # Precompute features if not already done
    features_file = args.features_path
    if not os.path.exists(features_file):
        print("Precomputing features for test images...")
        precompute_features(
            directory=args.input_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_fp16=args.fp16,
            save_path=features_file
        )
    
    # Load features
    test_features, test_labels, test_paths = load_and_verify_features(features_file)
    
    if len(test_features) == 0:
        print("Error: No valid features found for testing.")
        return
    
    # Make predictions
    start_time = time.time()
    predictions, confidences = model.predict(test_features)
    elapsed_time = time.time() - start_time
    
    # Process results
    results = {
        'authentic': [],
        'forged': [],
        'errors': []
    }
    
    # Create detailed results output
    results_file = os.path.join(args.output_dir, 'detection_results.txt')
    with open(results_file, 'w') as f:
        f.write("Image Forgery Detection Results\n")
        f.write("==============================\n\n")
        
        for i, (path, pred, conf) in enumerate(zip(test_paths, predictions, confidences)):
            img_name = os.path.basename(path)
            result_type = 'forged' if pred == 1 else 'authentic'
            results[result_type].append(img_name)
            
            f.write(f"Image: {img_name}\n")
            f.write(f"Result: {result_type.upper()}\n")
            f.write(f"Confidence: {conf:.4f}\n")
            
            if i < len(test_labels):
                true_label = 'forged' if test_labels[i] == 1 else 'authentic'
                f.write(f"True label: {true_label.upper()}\n")
                if pred == test_labels[i]:
                    f.write("Prediction: CORRECT\n")
                else:
                    f.write("Prediction: INCORRECT\n")
            
            f.write("\n")
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Processed {len(predictions)} images in {elapsed_time:.2f} seconds")
    print(f"Authentic images: {len(results['authentic'])}")
    print(f"Forged images: {len(results['forged'])}")
    
    # Compute accuracy if we have labels
    if len(test_labels) > 0:
        accuracy = np.mean(predictions == test_labels)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Compute confusion matrix
        tp = np.sum((predictions == 1) & (test_labels == 1))
        tn = np.sum((predictions == 0) & (test_labels == 0))
        fp = np.sum((predictions == 1) & (test_labels == 0))
        fn = np.sum((predictions == 0) & (test_labels == 1))
        
        print("\nConfusion Matrix:")
        print(f"True Positive: {tp}")
        print(f"True Negative: {tn}")
        print(f"False Positive: {fp}")
        print(f"False Negative: {fn}")
        
        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
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
    
    # Process the image
    print(f"Processing single image: {args.image_path}")
    start_time = time.time()
    
    try:
        # Preprocess the image
        ycbcr_tensor = preprocess_image(args.image_path)
        
        if ycbcr_tensor is None:
            print("Error: Failed to preprocess image")
            return
        
        # Add batch dimension
        ycbcr_tensor = ycbcr_tensor.unsqueeze(0)
        
        # Apply wavelet transform
        from modules.imageDecomposition import polar_dyadic_wavelet_transform
        pdywt_coeffs = polar_dyadic_wavelet_transform(ycbcr_tensor[0])
        
        # Extract features
        from modules.featureExtraction import BatchFeatureExtractor
        feature_extractor = BatchFeatureExtractor(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            batch_size=1,
            num_workers=1,
            use_fp16=args.fp16
        )
        
        feature_vector = feature_extractor.extract_batch_features(
            ycbcr_tensor, [pdywt_coeffs]
        )
        
        if feature_vector is None:
            print("Error: Failed to extract features")
            return
        
        # Make prediction
        predictions, confidences = model.predict(feature_vector.cpu().numpy())
        
        # Display results
        prediction = predictions[0]
        confidence = confidences[0]
        result = "FORGED" if prediction == 1 else "AUTHENTIC"
        
        print(f"\nResult: {result}")
        print(f"Confidence: {confidence:.4f}")
        
        # Save the result
        result_file = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.image_path))[0]}_result.txt")
        with open(result_file, 'w') as f:
            f.write(f"Image: {args.image_path}\n")
            f.write(f"Result: {result}\n")
            f.write(f"Confidence: {confidence:.4f}\n")
        
        print(f"Result saved to {result_file}")
        
    except Exception as e:
        print(f"Error processing image: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds")

def analyze_features(args):
    """Analyze precomputed features to gain insights"""
    if not os.path.exists(args.features_path):
        print(f"Error: Features file not found at {args.features_path}")
        return
    
    # Load features
    features, labels, paths = load_and_verify_features(args.features_path)
    
    if len(features) == 0:
        print("Error: No valid features found for analysis.")
        return
    
    if len(labels) == 0:
        print("Error: No labels found. Feature analysis requires labeled data.")
        return
    
    print("\nFeature Analysis:")
    print("================")
    
    # Split features by class
    authentic_features = features[labels == 0]
    forged_features = features[labels == 1]
    
    print(f"Authentic samples: {len(authentic_features)}")
    print(f"Forged samples: {len(forged_features)}")
    
    # Compute feature statistics by class
    authentic_mean = np.mean(authentic_features, axis=0)
    forged_mean = np.mean(forged_features, axis=0)
    
    # Find most discriminative features
    feature_diff = np.abs(authentic_mean - forged_mean)
    top_features = np.argsort(-feature_diff)[:10]  # Top 10 features
    
    print("\nTop 10 most discriminative features:")
    for i, feature_idx in enumerate(top_features):
        print(f"{i+1}. Feature {feature_idx}: "
              f"Auth={authentic_mean[feature_idx]:.4f}, "
              f"Forged={forged_mean[feature_idx]:.4f}, "
              f"Diff={feature_diff[feature_idx]:.4f}")
    
    # Plot feature distributions
    plt.figure(figsize=(15, 10))
    
    # Plot mean feature values
    plt.subplot(2, 1, 1)
    plt.plot(authentic_mean, 'b-', alpha=0.7, label='Authentic')
    plt.plot(forged_mean, 'r-', alpha=0.7, label='Forged')
    plt.title('Mean Feature Values by Class')
    plt.xlabel('Feature Index')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot feature difference
    plt.subplot(2, 1, 2)
    plt.bar(range(len(feature_diff)), feature_diff)
    plt.title('Absolute Difference in Feature Means')
    plt.xlabel('Feature Index')
    plt.ylabel('Absolute Difference')
    plt.grid(True, alpha=0.3)
    
    # Highlight top discriminative features
    for feature_idx in top_features:
        plt.axvline(x=feature_idx, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    analysis_file = os.path.join(args.output_dir, 'feature_analysis.png')
    plt.savefig(analysis_file)
    print(f"\nFeature analysis plot saved to {analysis_file}")
    
    # PCA visualization if we have enough samples
    if len(features) > 2:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply PCA
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features_scaled)
        
        # Plot PCA
        plt.figure(figsize=(10, 8))
        plt.scatter(features_pca[labels == 0, 0], features_pca[labels == 0, 1], 
                    c='blue', marker='o', alpha=0.7, label='Authentic')
        plt.scatter(features_pca[labels == 1, 0], features_pca[labels == 1, 1], 
                    c='red', marker='x', alpha=0.7, label='Forged')
        plt.title('PCA Visualization of Features')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the PCA plot
        pca_file = os.path.join(args.output_dir, 'pca_visualization.png')
        plt.savefig(pca_file)
        print(f"PCA visualization saved to {pca_file}")

def plot_history(history, output_dir):
    """Plot training history"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    history_file = os.path.join(output_dir, 'training_history.png')
    plt.savefig(history_file)
    print(f"Training history plot saved to {history_file}")

if __name__ == "__main__":
    main()