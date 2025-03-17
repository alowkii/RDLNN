#!/usr/bin/env python3
"""
Precision-tuned balanced training for image forgery detection
Focuses on improving precision while maintaining acceptable recall
"""

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.utils import resample
from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Load modules from your existing codebase
from modules.rdlnn import RegressionDLNN
from modules.data_handling import load_and_verify_features
from modules.utils import setup_logging, logger

def precision_tuned_training(features_path, model_path, output_dir, epochs=25, 
                             learning_rate=0.001, batch_size=32,
                             minority_ratio=0.6, pos_weight_value=1.8,
                             threshold=0.65):
    """
    Training approach specifically tuned to maximize precision
    
    Args:
        features_path: Path to the features file
        model_path: Path to save the model
        output_dir: Directory to save results
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        minority_ratio: Ratio of minority class to majority class after resampling
        pos_weight_value: Positive class weight value in loss function
        threshold: Classification threshold (higher increases precision)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir)
    logger.info("Starting precision-tuned balanced training")
    logger.info(f"Parameters: minority_ratio={minority_ratio}, pos_weight={pos_weight_value}, threshold={threshold}")
    
    # Load features
    features, labels, paths = load_and_verify_features(features_path)
    
    if len(features) == 0 or len(labels) == 0:
        logger.error("No valid features or labels found")
        return
    
    # Split into train/test first to ensure validation set remains unchanged
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Separate minority and majority classes in training set
    X_majority = X_train[y_train == 0]
    X_minority = X_train[y_train == 1]
    y_majority = y_train[y_train == 0]
    y_minority = y_train[y_train == 1]
    
    logger.info(f"Majority class samples (authentic): {len(X_majority)}")
    logger.info(f"Minority class samples (forged): {len(X_minority)}")
    
    # Oversample to the specified ratio of majority class size
    minority_target_count = int(len(X_majority) * minority_ratio)
    
    X_minority_oversampled, y_minority_oversampled = resample(
        X_minority, 
        y_minority,
        replace=True,
        n_samples=minority_target_count,
        random_state=42
    )
    
    # Combine oversampled minority class with majority class
    X_train_balanced = np.vstack([X_majority, X_minority_oversampled])
    y_train_balanced = np.concatenate([y_majority, y_minority_oversampled])
    
    # Shuffle the balanced dataset
    indices = np.arange(len(X_train_balanced))
    np.random.shuffle(indices)
    X_train_balanced = X_train_balanced[indices]
    y_train_balanced = y_train_balanced[indices]
    
    # Check new class distribution
    new_class_counts = np.bincount(y_train_balanced.astype(int))
    logger.info(f"Re-balanced class distribution: {new_class_counts}")
    
    # Create model
    input_dim = features.shape[1]
    logger.info(f"Creating model with input dimension: {input_dim}")
    model = RegressionDLNN(input_dim)
    
    # Set positive class weight for BCE loss 
    # Lower value than before to reduce false positives
    pos_weight = torch.tensor([pos_weight_value], device=model.device)
    model.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Use a slightly higher learning rate than default
    # This helps escape local minima and converge better
    history = model.fit(
        X_train_balanced, y_train_balanced, 
        epochs=epochs, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_split=0.2,  # Use standard validation split
        early_stopping=5,
        use_fp16=True,
        force_class_balance=False  # We're handling balance manually
    )
    
    # Save the model
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # After training, now prepare separate evaluation on validation set
    logger.info(f"Setting classification threshold to {threshold}")
    
    # First, save the model with its standard threshold
    model.save(model_path)
    logger.info(f"Model saved with standard threshold to {model_path}")
    
    # Now evaluate on the validation set
    predictions, confidences = model.predict(X_val)
    
    # Evaluate with different thresholds
    evaluate_with_thresholds(model, X_val, y_val, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    
    # Apply the recommended threshold for final results
    modified_predictions = (confidences >= threshold).astype(int)
    
    # Compute metrics with chosen threshold
    tp = np.sum((modified_predictions == 1) & (y_val == 1))
    fp = np.sum((modified_predictions == 1) & (y_val == 0))
    tn = np.sum((modified_predictions == 0) & (y_val == 0))
    fn = np.sum((modified_predictions == 0) & (y_val == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    logger.info(f"Final results with threshold={threshold}:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Confusion Matrix:")
    logger.info(f"                | Predicted Positive | Predicted Negative |")
    logger.info(f"Actual Positive |       {tp:^18} | {fn:^18} |")
    logger.info(f"Actual Negative |       {fp:^18} | {tn:^18} |")
    
    # Save threshold information with the model
    model.threshold = threshold
    model.save(model_path)
    
    return history

def evaluate_with_thresholds(model, X_val, y_val, thresholds):
    """
    Evaluate model performance with different classification thresholds
    
    Args:
        model: Trained RegressionDLNN model
        X_val: Validation features
        y_val: Validation labels
        thresholds: List of threshold values to try
    """
    _, confidences = model.predict(X_val)
    
    logger.info("\nThreshold analysis:")
    logger.info(f"{'Threshold':^10} | {'Precision':^10} | {'Recall':^10} | {'F1 Score':^10} | {'Accuracy':^10}")
    logger.info("-" * 60)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = (confidences >= threshold).astype(int)
        
        # Compute metrics
        tp = np.sum((predictions == 1) & (y_val == 1))
        fp = np.sum((predictions == 1) & (y_val == 0))
        tn = np.sum((predictions == 0) & (y_val == 0))
        fn = np.sum((predictions == 0) & (y_val == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        logger.info(f"{threshold:^10.2f} | {precision:^10.4f} | {recall:^10.4f} | {f1:^10.4f} | {accuracy:^10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info(f"\nBest threshold based on F1: {best_threshold:.2f} (F1={best_f1:.4f})")

def grid_search_parameters(features_path, model_path, output_dir, epochs=15):
    """
    Perform a grid search over balance parameters to find optimal configuration
    
    Args:
        features_path: Path to the features file
        model_path: Path to save the model
        output_dir: Directory to save results
        epochs: Number of training epochs for each configuration
    """
    # Parameters to search
    minority_ratios = [0.5, 0.6, 0.7]
    pos_weights = [1.5, 2.0, 2.5]
    thresholds = [0.6, 0.65, 0.7]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir)
    logger.info("Starting grid search for balance parameters")
    
    # Create a smaller model path to make it faster
    grid_search_path = os.path.join(os.path.dirname(model_path), "grid_search_model.pth")
    
    # Load features
    features, labels, paths = load_and_verify_features(features_path)
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Track best parameters
    best_f1 = 0
    best_precision = 0
    best_params = {}
    
    # Results table
    logger.info("\nGrid search results:")
    logger.info(f"{'Minority Ratio':^15} | {'Pos Weight':^10} | {'Threshold':^10} | {'Precision':^10} | {'Recall':^10} | {'F1 Score':^10}")
    logger.info("-" * 75)
    
    # Iterate through all parameter combinations
    for minority_ratio in minority_ratios:
        for pos_weight in pos_weights:
            # Train the model with current parameters
            model = train_model_with_params(
                X_train, y_train, 
                grid_search_path, 
                minority_ratio, 
                pos_weight, 
                epochs
            )
            
            # Evaluate on different thresholds
            _, confidences = model.predict(X_val)
            
            for threshold in thresholds:
                predictions = (confidences >= threshold).astype(int)
                
                # Compute metrics
                tp = np.sum((predictions == 1) & (y_val == 1))
                fp = np.sum((predictions == 1) & (y_val == 0))
                tn = np.sum((predictions == 0) & (y_val == 0))
                fn = np.sum((predictions == 0) & (y_val == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Log results
                logger.info(f"{minority_ratio:^15.2f} | {pos_weight:^10.2f} | {threshold:^10.2f} | {precision:^10.4f} | {recall:^10.4f} | {f1:^10.4f}")
                
                # Check if this is the best configuration
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_params = {
                        'minority_ratio': minority_ratio,
                        'pos_weight': pos_weight,
                        'threshold': threshold,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
    
    logger.info("\nBest parameters found:")
    logger.info(f"Minority Ratio: {best_params['minority_ratio']}")
    logger.info(f"Positive Weight: {best_params['pos_weight']}")
    logger.info(f"Threshold: {best_params['threshold']}")
    logger.info(f"Resulting in Precision: {best_params['precision']:.4f}, Recall: {best_params['recall']:.4f}, F1: {best_params['f1']:.4f}")
    
    # Return the best parameters
    return best_params

def train_model_with_params(X_train, y_train, model_path, minority_ratio, pos_weight, epochs):
    """
    Train a model with specific balance parameters
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_path: Path to save the model
        minority_ratio: Ratio of minority class to majority class after resampling
        pos_weight: Positive class weight value
        epochs: Number of training epochs
    
    Returns:
        Trained model
    """
    # Separate minority and majority classes
    X_majority = X_train[y_train == 0]
    X_minority = X_train[y_train == 1]
    y_majority = y_train[y_train == 0]
    y_minority = y_train[y_train == 1]
    
    # Oversample minority class
    minority_target_count = int(len(X_majority) * minority_ratio)
    X_minority_oversampled, y_minority_oversampled = resample(
        X_minority, y_minority, 
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
    
    # Create and train model
    input_dim = X_train.shape[1]
    model = RegressionDLNN(input_dim)
    
    # Set positive class weight
    model.loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=model.device)
    )
    
    # Train with smaller validation set to speed up grid search
    model.fit(
        X_balanced, y_balanced,
        epochs=epochs,
        learning_rate=0.001,
        batch_size=32,
        validation_split=0.1,
        early_stopping=3,
        use_fp16=True
    )
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precision-tuned balanced training for forgery detection")
    parser.add_argument('--features_path', type=str, default='features/casia2_features_fixed.npz',
                      help='Path to features file')
    parser.add_argument('--model_path', type=str, default='models/casia2_precision_model.pth',
                      help='Path to save model')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=25,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--grid_search', action='store_true',
                      help='Perform grid search to find optimal parameters')
    parser.add_argument('--minority_ratio', type=float, default=0.6,
                      help='Ratio of minority class to majority class after resampling')
    parser.add_argument('--pos_weight', type=float, default=1.8,
                      help='Positive class weight value in loss function')
    parser.add_argument('--threshold', type=float, default=0.65,
                      help='Classification threshold (higher increases precision)')
    
    args = parser.parse_args()
    
    if args.grid_search:
        best_params = grid_search_parameters(
            args.features_path,
            args.model_path,
            args.output_dir,
            epochs=15  # Use fewer epochs for grid search
        )
        
        # Train final model with best parameters
        precision_tuned_training(
            args.features_path,
            args.model_path,
            args.output_dir,
            args.epochs,
            args.learning_rate,
            args.batch_size,
            best_params['minority_ratio'],
            best_params['pos_weight'],
            best_params['threshold']
        )
    else:
        # Use provided parameters
        precision_tuned_training(
            args.features_path,
            args.model_path,
            args.output_dir,
            args.epochs,
            args.learning_rate,
            args.batch_size,
            args.minority_ratio,
            args.pos_weight,
            args.threshold
        )