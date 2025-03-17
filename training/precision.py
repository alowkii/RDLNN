#!/usr/bin/env python3
"""
Precision-tuned training for image forgery detection
Optimized to maximize precision while maintaining good F1 score
"""

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from modules.rdlnn import RegressionDLNN
from modules.data_handling import load_and_verify_features
from modules.utils import setup_logging, logger, plot_diagnostic_curves

def precision_tuned_training(features_path, model_path, output_dir,
                           minority_ratio=0.65, pos_weight=2.0, 
                           epochs=25, learning_rate=0.001, batch_size=32,
                           threshold=0.80):
    """
    Precision-focused training with enhanced balancing and fixed threshold
    
    Args:
        features_path: Path to the features file
        model_path: Path to save the model
        output_dir: Directory to save results
        minority_ratio: Ratio of minority class samples after resampling
        pos_weight: Weight for positive class in loss function
        epochs: Number of epochs to train
        learning_rate: Learning rate
        batch_size: Batch size
        threshold: Fixed classification threshold (0.80 for optimal precision/F1)
        
    Returns:
        Trained RegressionDLNN model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Set up logging with reduced verbosity
    setup_logging(output_dir)
    
    logger.info(f"Starting precision-tuned training with threshold={threshold}")
    logger.info(f"Parameters: minority_ratio={minority_ratio}, pos_weight={pos_weight}")
    
    # Load features
    features, labels, paths = load_and_verify_features(features_path)
    
    if len(features) == 0 or len(labels) == 0:
        logger.error("No valid features or labels found")
        return None
    
    # Split data into train/validation/test sets (60/20/20 split)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.4, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    
    logger.info(f"Training: {len(X_train)} samples | Validation: {len(X_val)} | Test: {len(X_test)}")
    
    # Separate minority and majority classes in training set
    X_majority = X_train[y_train == 0]
    X_minority = X_train[y_train == 1]
    y_majority = y_train[y_train == 0]
    y_minority = y_train[y_train == 1]
    
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
    
    # Create model
    input_dim = features.shape[1]
    logger.info(f"Training model with input dimension: {input_dim}")
    model = RegressionDLNN(input_dim)
    
    # Create pos_weight tensor on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    
    # Set custom loss function with specified pos_weight
    model.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    # Train model
    logger.info("Starting training...")
    
    history = model.fit(
        X_balanced, y_balanced, 
        epochs=epochs, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_split=0.2,
        early_stopping=5,
        use_fp16=True
    )
    
    # Evaluate on test set
    predictions, confidences = model.predict(X_test)
    
    # Apply fixed threshold
    thresholded_preds = (confidences >= threshold).astype(int)
    
    # Calculate metrics with fixed threshold
    tp = np.sum((thresholded_preds == 1) & (y_test == 1))
    tn = np.sum((thresholded_preds == 0) & (y_test == 0))
    fp = np.sum((thresholded_preds == 1) & (y_test == 0))
    fn = np.sum((thresholded_preds == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_test)
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    
    # Print results in a concise format
    logger.info(f"\n========== Performance with threshold={threshold:.2f} ==========")
    logger.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f} | Balanced Accuracy: {balanced_accuracy:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    
    # Print confusion matrix in a compact format
    logger.info("\nConfusion Matrix:")
    logger.info(f"TP: {tp} | FN: {fn}")
    logger.info(f"FP: {fp} | TN: {tn}")
    
    # Set threshold in the model
    model.threshold = threshold
    
    # Save the model
    model.save(model_path)
    logger.info(f"Model saved with threshold {threshold} to {model_path}")
    
    # Create diagnostic plots if enabled
    try:
        plot_diagnostic_curves(y_test, confidences, history, output_dir)
        logger.info(f"Diagnostic plots saved to {output_dir}")
    except Exception as e:
        logger.warning(f"Could not create diagnostic plots: {e}")
    
    # Evaluate with different thresholds for reference
    evaluate_with_thresholds(model, X_val, y_val, 
                            [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9], 
                            output_dir)
    
    return model

def evaluate_with_thresholds(model, X_val, y_val, thresholds, output_dir):
    """
    Evaluate model performance with different classification thresholds
    
    Args:
        model: Trained RegressionDLNN model
        X_val: Validation features
        y_val: Validation labels
        thresholds: List of threshold values to try
        output_dir: Directory to save threshold analysis results
    """
    # Get raw predictions
    _, confidences = model.predict(X_val)
    
    results = []
    logger.info("\nThreshold Analysis:")
    logger.info(f"{'Threshold':^10} | {'Precision':^10} | {'Recall':^10} | {'F1 Score':^10} | {'Accuracy':^10}")
    logger.info("-" * 60)
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        # Apply threshold
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
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
        
        logger.info(f"{threshold:^10.2f} | {precision:^10.4f} | {recall:^10.4f} | {f1:^10.4f} | {accuracy:^10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info(f"\nBest threshold based on F1: {best_threshold:.2f} (F1={best_f1:.4f})")
    
    # Create a threshold analysis plot
    try:
        thresholds_arr = np.array([result['threshold'] for result in results])
        precision_arr = np.array([result['precision'] for result in results])
        recall_arr = np.array([result['recall'] for result in results])
        f1_arr = np.array([result['f1'] for result in results])
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_arr, precision_arr, 'b-', label='Precision')
        plt.plot(thresholds_arr, recall_arr, 'g-', label='Recall')
        plt.plot(thresholds_arr, f1_arr, 'r-', label='F1 Score')
        plt.axvline(x=best_threshold, color='k', linestyle='--', alpha=0.3, label=f'Best F1 ({best_threshold:.2f})')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision, Recall and F1 Score vs. Threshold')
        plt.legend()
        
        # Save the plot
        threshold_plot_path = os.path.join(output_dir, 'threshold_analysis.png')
        plt.savefig(threshold_plot_path)
        logger.info(f"Threshold analysis plot saved to {threshold_plot_path}")
    except Exception as e:
        logger.warning(f"Could not create threshold analysis plot: {e}")
    
    return best_threshold, best_f1