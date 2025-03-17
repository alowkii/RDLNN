#!/usr/bin/env python3
"""
Optimized precision-tuned training for image forgery detection
With fixed threshold at 0.80 for optimal precision/F1 score balance
"""

import numpy as np
import torch
import os
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Load modules from your existing codebase
from modules.rdlnn import RegressionDLNN
from modules.data_handling import load_and_verify_features
from modules.utils import setup_logging, logger, plot_diagnostic_curves


def precision_tuned_training(features_path, model_path, output_dir,
                            minority_ratio=0.65, pos_weight=2.2,
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
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Set up logging with reduced verbosity
    setup_logging(output_dir)
    
    # Set custom logging level for console output
    logger.info(f"Starting precision-tuned training with threshold={threshold}")
    
    # Load features
    features, labels, paths = load_and_verify_features(features_path)
    
    if len(features) == 0 or len(labels) == 0:
        logger.error("No valid features or labels found")
        return
    
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
    
    # Override RegressionDLNN's verbose output during training
    class QuietCallback:
        def __init__(self, original_model):
            self.model = original_model
            self.best_epoch = 0
            self.best_val_loss = float('inf')
            
        def on_epoch_end(self, epoch, metrics):
            if metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = metrics['val_loss']
                self.best_epoch = epoch
                logger.info(f"Epoch {epoch+1}: New best validation loss: {self.best_val_loss:.6f}")
    
    # Train model with enhanced parameters and reduced output
    logger.info("Starting training...")
    callback = QuietCallback(model)
    
    # Modify existing fit method to use our callback
    original_fit = model.fit
    
    def quiet_fit(*args, **kwargs):
        history = original_fit(*args, **kwargs)
        return history
    
    model.fit = quiet_fit
    
    history = model.fit(
        X_balanced, y_balanced, 
        epochs=epochs, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_split=0.2,
        early_stopping=5,
        use_fp16=True
    )
    
    logger.info(f"Training completed. Best model found at epoch {callback.best_epoch+1}")
    
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
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized precision-tuned training for forgery detection")
    parser.add_argument('--features_path', type=str, default='features/casia2_features_fixed.npz',
                      help='Path to features file')
    parser.add_argument('--model_path', type=str, default='models/casia2_precision_model.pth',
                      help='Path to save model')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--minority_ratio', type=float, default=0.65,
                      help='Ratio of minority to majority class samples')
    parser.add_argument('--pos_weight', type=float, default=2.2,
                      help='Weight for positive class in loss function')
    parser.add_argument('--epochs', type=int, default=25,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.80,
                      help='Classification threshold')
    
    args = parser.parse_args()
    
    precision_tuned_training(
        args.features_path,
        args.model_path,
        args.output_dir,
        args.minority_ratio,
        args.pos_weight,
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.threshold
    )