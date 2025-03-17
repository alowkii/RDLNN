#!/usr/bin/env python3
"""
Simple precision-tuned training for image forgery detection
"""

import numpy as np
import torch
import os
from sklearn.utils import resample

# Load modules from your existing codebase
from modules.rdlnn import RegressionDLNN
from modules.data_handling import load_and_verify_features
from modules.utils import setup_logging, logger

def precision_tuned_training(features_path, model_path, output_dir,
                            minority_ratio=0.6, pos_weight=1.8,
                            epochs=25, learning_rate=0.001, batch_size=32):
    """
    Precision-focused training with balanced sampling
    
    Args:
        features_path: Path to the features file
        model_path: Path to save the model
        output_dir: Directory to save results
        minority_ratio: Ratio of minority class samples after resampling
        pos_weight: Weight for positive class in loss function
        epochs: Number of epochs to train
        learning_rate: Learning rate
        batch_size: Batch size
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir)
    logger.info("Starting precision-tuned training")
    logger.info(f"Parameters: minority_ratio={minority_ratio}, pos_weight={pos_weight}")
    
    # Load features
    features, labels, paths = load_and_verify_features(features_path)
    
    if len(features) == 0 or len(labels) == 0:
        logger.error("No valid features or labels found")
        return
    
    # Separate minority and majority classes
    X_majority = features[labels == 0]
    X_minority = features[labels == 1]
    y_majority = labels[labels == 0]
    y_minority = labels[labels == 1]
    
    logger.info(f"Majority class samples (authentic): {len(X_majority)}")
    logger.info(f"Minority class samples (forged): {len(X_minority)}")
    
    # Oversample minority class to the specified ratio
    minority_target_count = int(len(X_majority) * minority_ratio)
    
    X_minority_oversampled, y_minority_oversampled = resample(
        X_minority, 
        y_minority,
        replace=True,
        n_samples=minority_target_count,
        random_state=42
    )
    
    # Combine classes
    X_balanced = np.vstack([X_majority, X_minority_oversampled])
    y_balanced = np.concatenate([y_majority, y_minority_oversampled])
    
    # Shuffle
    indices = np.arange(len(X_balanced))
    np.random.shuffle(indices)
    X_balanced = X_balanced[indices]
    y_balanced = y_balanced[indices]
    
    # Check class distribution
    class_counts = np.bincount(y_balanced.astype(int))
    logger.info(f"Re-balanced class distribution: {class_counts}")
    
    # Create model
    input_dim = features.shape[1]
    logger.info(f"Creating model with input dimension: {input_dim}")
    model = RegressionDLNN(input_dim)
    
    # Create pos_weight tensor on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    
    # Set custom loss function with specified pos_weight
    model.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Train model
    history = model.fit(
        X_balanced, y_balanced, 
        epochs=epochs, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_split=0.2,
        early_stopping=5,
        use_fp16=True
    )
    
    # Save the model
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Load the saved model to verify
    test_model = RegressionDLNN.load(model_path)
    logger.info("Successfully loaded model for verification")
    
    # Get predictions - use standard 0.5 threshold
    predictions, confidences = test_model.predict(features)
    
    # Print results for different threshold levels
    logger.info("\nThreshold analysis:")
    logger.info(f"{'Threshold':^10} | {'Precision':^10} | {'Recall':^10} | {'F1 Score':^10} | {'Accuracy':^10}")
    logger.info("-" * 60)
    
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        # Apply threshold
        thresholded_preds = (confidences >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((thresholded_preds == 1) & (labels == 1))
        tn = np.sum((thresholded_preds == 0) & (labels == 0))
        fp = np.sum((thresholded_preds == 1) & (labels == 0))
        fn = np.sum((thresholded_preds == 0) & (labels == 1))
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(labels)
        
        logger.info(f"{threshold:^10.2f} | {precision:^10.4f} | {recall:^10.4f} | {f1:^10.4f} | {accuracy:^10.4f}")
    
    logger.info("\nTraining complete! For production use, you can select the threshold that best balances precision and recall.")
    logger.info("To increase precision further, use a higher threshold like 0.65-0.75.")
    logger.info("To maximize F1 score, choose the threshold with the highest F1 value.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple precision-tuned training for forgery detection")
    parser.add_argument('--features_path', type=str, default='features/casia2_features_fixed.npz',
                      help='Path to features file')
    parser.add_argument('--model_path', type=str, default='models/casia2_precision_model.pth',
                      help='Path to save model')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--minority_ratio', type=float, default=0.6,
                      help='Ratio of minority to majority class samples')
    parser.add_argument('--pos_weight', type=float, default=1.8,
                      help='Weight for positive class in loss function')
    parser.add_argument('--epochs', type=int, default=25,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    
    args = parser.parse_args()
    
    precision_tuned_training(
        args.features_path,
        args.model_path,
        args.output_dir,
        args.minority_ratio,
        args.pos_weight,
        args.epochs,
        args.learning_rate,
        args.batch_size
    )