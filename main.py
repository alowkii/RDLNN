#!/usr/bin/env python3
"""
Image Forgery Detection System - Main entry point
Provides a CLI interface for different modes of operation
"""

import os
import argparse
import torch
import time
import signal
import sys
import gc
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Import modules
from modules.rdlnn import RegressionDLNN
from modules.data_handling import precompute_features, load_and_verify_features
from modules.preprocessing import preprocess_image
from modules.image_decomposition import perform_wavelet_transform
from modules.feature_extraction import extract_features_from_wavelet
from modules.batch_processor import OptimizedBatchProcessor
from modules.utils import setup_signal_handlers, plot_training_history, setup_logging, logger

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.features_path), exist_ok=True)
    
    # Set up logging
    setup_logging(args.output_dir)
    logger.info(f"Starting image forgery detection in {args.mode} mode")
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} - CUDA available: {torch.cuda.is_available()}")
    
    # Execute the requested mode
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
    
    logger.info("Operation completed successfully")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Image Forgery Detection System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--mode', 
                        choices=['train', 'test', 'precompute', 'single', 'analyze'], 
                        required=True, 
                        help='Operating mode: train, test, precompute features, test single image, or analyze features')
    
    parser.add_argument('--input_dir', 
                        type=str,
                        help='Directory containing input images')
    
    parser.add_argument('--image_path', 
                        type=str,
                        help='Path to single image for testing')
    
    parser.add_argument('--output_dir', 
                        type=str, 
                        default='results',
                        help='Directory to save results')
    
    parser.add_argument('--model_path', 
                        type=str, 
                        default='models/forgery_detection_model.pth',
                        help='Path to save/load model')
    
    parser.add_argument('--features_path', 
                        type=str, 
                        default='features/precomputed_features.npz',
                        help='Path to save/load precomputed features')
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=16,
                        help='Batch size for processing')
    
    parser.add_argument('--workers', 
                        type=int, 
                        default=4,
                        help='Number of worker threads for data loading')
    
    parser.add_argument('--fp16', 
                        action='store_true',
                        help='Use half precision (FP16) operations')
    
    parser.add_argument('--authentic_dir', 
                        type=str,
                        help='Directory containing authentic images (for training)')
    
    parser.add_argument('--forged_dir', 
                        type=str,
                        help='Directory containing forged images (for training)')
    
    parser.add_argument('--epochs', 
                        type=int, 
                        default=20,
                        help='Number of training epochs')
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=0.001,
                        help='Learning rate for training')
    
    parser.add_argument('--debug', 
                        action='store_true', 
                        help='Enable debug mode with additional logging')
    
    return parser.parse_args()

def precompute_mode(args: argparse.Namespace) -> None:
    """Precompute features for training or testing
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Precomputing features with batch size {args.batch_size}...")
    
    # Process authentic images if provided
    if args.authentic_dir:
        logger.info(f"Processing authentic images from {args.authentic_dir}")
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
        logger.info(f"Processing forged images from {args.forged_dir}")
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
        logger.info(f"Processing images from {args.input_dir}")
        precompute_features(
            directory=args.input_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            use_fp16=args.fp16,
            save_path=args.features_path
        )
    
    # Combine authentic and forged features if both were computed
    if args.authentic_dir and args.forged_dir:
        logger.info("Combining authentic and forged features...")
        
        # Load the individual feature files
        authentic_data = np.load(f"{os.path.splitext(args.features_path)[0]}_authentic.npz", allow_pickle=True)
        forged_data = np.load(f"{os.path.splitext(args.features_path)[0]}_forged.npz", allow_pickle=True)
        
        # Extract data
        authentic_features = authentic_data['features']
        authentic_paths = authentic_data['paths']
        authentic_labels = authentic_data['labels']
        
        forged_features = forged_data['features']
        forged_paths = forged_data['paths']
        forged_labels = forged_data['labels']
        
        # Combine the data
        import numpy as np
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
        
        logger.info(f"Saved combined features to {args.features_path}")
        logger.info(f"Combined dataset: {len(combined_features)} samples "
              f"({len(authentic_features)} authentic, {len(forged_features)} forged)")

def train_mode(args: argparse.Namespace) -> None:
    """Train the model using precomputed features
    
    Args:
        args: Command line arguments
    """
    import numpy as np
    
    if not (args.authentic_dir and args.forged_dir) and not os.path.exists(args.features_path):
        logger.error("For training, either provide authentic_dir and forged_dir arguments, "
              "or precompute features first and provide features_path.")
        return
    
    # Load or compute features
    if not os.path.exists(args.features_path):
        logger.info("Precomputing features for training...")
        precompute_mode(args)
    
    # Load and verify features
    features, labels, paths = load_and_verify_features(args.features_path)
    
    if len(features) == 0:
        logger.error("No valid features found. Please check your data.")
        return
    
    if len(labels) == 0:
        logger.error("No labels found. Please make sure your feature file includes labels.")
        return
    
    # Check class balance
    class_counts = np.bincount(labels.astype(int))
    if len(class_counts) < 2:
        logger.error(f"Only found {len(class_counts)} classes. Need at least 2 classes for training.")
        return
    
    # Create and train the model
    input_dim = features.shape[1]
    logger.info(f"Creating model with input dimension: {input_dim}")
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
    plot_training_history(history, args.output_dir)

def test_mode(args: argparse.Namespace) -> None:
    """Test the model on images in the input directory
    
    Args:
        args: Command line arguments
    """
    import numpy as np
    
    if not os.path.exists(args.model_path):
        logger.error(f"Error: Model not found at {args.model_path}")
        return
    
    if not args.input_dir:
        logger.error("Error: Please provide --input_dir argument for testing")
        return
    
    # Load the model
    model = RegressionDLNN.load(args.model_path)
    
    # Precompute features if not already done
    features_file = args.features_path
    if not os.path.exists(features_file):
        logger.info("Precomputing features for test images...")
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
        logger.error("Error: No valid features found for testing.")
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
    logger.info("\nProcessing complete!")
    logger.info(f"Processed {len(predictions)} images in {elapsed_time:.2f} seconds")
    logger.info(f"Authentic images: {len(results['authentic'])}")
    logger.info(f"Forged images: {len(results['forged'])}")
    
    # Compute accuracy if we have labels
    if len(test_labels) > 0:
        accuracy = np.mean(predictions == test_labels)
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        # Compute confusion matrix
        tp = np.sum((predictions == 1) & (test_labels == 1))
        tn = np.sum((predictions == 0) & (test_labels == 0))
        fp = np.sum((predictions == 1) & (test_labels == 0))
        fn = np.sum((predictions == 0) & (test_labels == 1))
        
        logger.info("\nConfusion Matrix:")
        logger.info(f"True Positive: {tp}")
        logger.info(f"True Negative: {tn}")
        logger.info(f"False Positive: {fp}")
        logger.info(f"False Negative: {fn}")
        
        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
    
    logger.info(f"Results saved to {results_file}")

def test_single_image(args: argparse.Namespace) -> None:
    """Test the model on a single image
    
    Args:
        args: Command line arguments
    """
    if not args.image_path:
        logger.error("Error: Please provide --image_path argument for single image testing")
        return
    
    if not os.path.exists(args.model_path):
        logger.error(f"Error: Model not found at {args.model_path}")
        return
    
    if not os.path.exists(args.image_path):
        logger.error(f"Error: Image not found at {args.image_path}")
        return
    
    # Load the model
    model = RegressionDLNN.load(args.model_path)
    
    # Process the image
    logger.info(f"Processing single image: {args.image_path}")
    start_time = time.time()
    
    try:
        # Preprocess the image
        ycbcr_tensor = preprocess_image(args.image_path)
        
        if ycbcr_tensor is None:
            logger.error("Error: Failed to preprocess image")
            return
        
        # Add batch dimension
        ycbcr_tensor = ycbcr_tensor.unsqueeze(0)
        
        # Apply wavelet transform
        pdywt_coeffs = perform_wavelet_transform(ycbcr_tensor[0])
        
        # Extract features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feature_vector = extract_features_from_wavelet(
            ycbcr_tensor, 
            [pdywt_coeffs],
            device=device,
            batch_size=1,
            use_fp16=args.fp16
        )
        
        if feature_vector is None:
            logger.error("Error: Failed to extract features")
            return
        
        # Make prediction
        predictions, confidences = model.predict(feature_vector.cpu().numpy())
        
        # Display results
        prediction = predictions[0]
        confidence = confidences[0]
        result = "FORGED" if prediction == 1 else "AUTHENTIC"
        
        logger.info(f"\nResult: {result}")
        logger.info(f"Confidence: {confidence:.4f}")
        
        # Save the result
        result_file = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.image_path))[0]}_result.txt")
        with open(result_file, 'w') as f:
            f.write(f"Image: {args.image_path}\n")
            f.write(f"Result: {result}\n")
            f.write(f"Confidence: {confidence:.4f}\n")
        
        logger.info(f"Result saved to {result_file}")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=args.debug)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")

def analyze_features(args: argparse.Namespace) -> None:
    """Analyze precomputed features to gain insights
    
    Args:
        args: Command line arguments
    """
    import numpy as np
    
    if not os.path.exists(args.features_path):
        logger.error(f"Error: Features file not found at {args.features_path}")
        return
    
    # Load features
    features, labels, paths = load_and_verify_features(args.features_path)
    
    if len(features) == 0:
        logger.error("Error: No valid features found for analysis.")
        return
    
    if len(labels) == 0:
        logger.error("Error: No labels found. Feature analysis requires labeled data.")
        return
    
    logger.info("\nFeature Analysis:")
    logger.info("================")
    
    # Split features by class
    authentic_features = features[labels == 0]
    forged_features = features[labels == 1]
    
    logger.info(f"Authentic samples: {len(authentic_features)}")
    logger.info(f"Forged samples: {len(forged_features)}")
    
    # Compute feature statistics by class
    authentic_mean = np.mean(authentic_features, axis=0)
    forged_mean = np.mean(forged_features, axis=0)
    
    # Find most discriminative features
    feature_diff = np.abs(authentic_mean - forged_mean)
    top_features = np.argsort(-feature_diff)[:10]  # Top 10 features
    
    logger.info("\nTop 10 most discriminative features:")
    for i, feature_idx in enumerate(top_features):
        logger.info(f"{i+1}. Feature {feature_idx}: "
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
    logger.info(f"\nFeature analysis plot saved to {analysis_file}")
    
    # PCA visualization if we have enough samples
    if len(features) > 2:
        try:
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
            logger.info(f"PCA visualization saved to {pca_file}")
        except ImportError:
            logger.warning("scikit-learn not installed. Skipping PCA visualization.")
        except Exception as e:
            logger.error(f"Error during PCA visualization: {e}", exc_info=args.debug)

if __name__ == "__main__":
    main()