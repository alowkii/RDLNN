#!/usr/bin/env python3
"""
Balanced training approaches for image forgery detection
Implements class weight adjustment and oversampling techniques
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

def train_with_balanced_sampling(features_path, model_path, output_dir, epochs=20, 
                                learning_rate=0.001, batch_size=32):
    """
    Train using balanced batch sampling to handle class imbalance
    
    Args:
        features_path: Path to the features file
        model_path: Path to save the model
        output_dir: Directory to save results
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir)
    logger.info("Starting balanced training with WeightedRandomSampler")
    
    # Load features
    features, labels, paths = load_and_verify_features(features_path)
    
    if len(features) == 0 or len(labels) == 0:
        logger.error("No valid features or labels found")
        return
    
    # Calculate class distribution
    class_counts = np.bincount(labels.astype(int))
    logger.info(f"Original class distribution: {class_counts}")
    
    # Calculate class weights (inverse of frequency)
    num_samples = len(labels)
    class_weights = num_samples / (len(class_counts) * class_counts)
    
    # Display calculated weights
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    logger.info(f"Using class weights: {class_weights_dict}")
    
    # Create model
    input_dim = features.shape[1]
    logger.info(f"Creating model with input dimension: {input_dim}")
    model = RegressionDLNN(input_dim)
    
    # Set explicitly high positive class weight for BCE loss to further combat imbalance
    pos_weight = torch.tensor([5.0], device=model.device)  # Even stronger weight for forgeries
    model.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Train the model with our balanced approach
    history = model.fit(
        features, labels, 
        epochs=epochs, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_split=0.2,
        early_stopping=5,
        use_fp16=True,
        force_class_balance=True  # Use our strengthened class balancing
    )
    
    # Save the model
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return history

def train_with_oversampling(features_path, model_path, output_dir, epochs=20, 
                           learning_rate=0.001, batch_size=32):
    """
    Train using oversampling of minority class to handle class imbalance
    
    Args:
        features_path: Path to the features file
        model_path: Path to save the model
        output_dir: Directory to save results
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir)
    logger.info("Starting balanced training with oversampling")
    
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
    
    logger.info(f"Majority class samples: {len(X_majority)}")
    logger.info(f"Minority class samples: {len(X_minority)}")
    
    # Oversample minority class to match majority class
    X_minority_oversampled, y_minority_oversampled = resample(
        X_minority, 
        y_minority,
        replace=True,  # Sample with replacement
        n_samples=len(X_majority),  # Match majority class count
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
    logger.info(f"Balanced class distribution: {new_class_counts}")
    
    # Create model
    input_dim = features.shape[1]
    logger.info(f"Creating model with input dimension: {input_dim}")
    model = RegressionDLNN(input_dim)
    
    # Train with balanced dataset
    history = model.fit(
        X_train_balanced, y_train_balanced, 
        epochs=epochs, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_split=0.2,  # Use default validation split
        early_stopping=5,
        use_fp16=True
    )
    
    # Save the model
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return history

def combined_approach(features_path, model_path, output_dir, epochs=20, 
                     learning_rate=0.001, batch_size=32):
    """
    Combined approach using both oversampling and class weights
    
    Args:
        features_path: Path to the features file
        model_path: Path to save the model
        output_dir: Directory to save results
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Batch size for training
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir)
    logger.info("Starting balanced training with combined approach")
    
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
    
    logger.info(f"Majority class samples: {len(X_majority)}")
    logger.info(f"Minority class samples: {len(X_minority)}")
    
    # Oversample to 75% of majority class size (avoid exact balance which could overfit)
    minority_target_count = int(len(X_majority) * 0.75)
    
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
    logger.info(f"Semi-balanced class distribution: {new_class_counts}")
    
    # Create model
    input_dim = features.shape[1]
    logger.info(f"Creating model with input dimension: {input_dim}")
    model = RegressionDLNN(input_dim)
    
    # Add class weights on top of oversampling
    class_weights = {
        0: 1.0,  # Authentic class
        1: 2.5   # Forged class - still give extra weight, but less extreme
    }
    logger.info(f"Using class weights: {class_weights}")
    
    # Set explicitly higher positive class weight for BCE loss
    pos_weight = torch.tensor([2.5], device=model.device)  # More moderate than in pure class weight approach
    model.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Train with balanced dataset and class weights
    history = model.fit(
        X_train_balanced, y_train_balanced, 
        epochs=epochs, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        validation_split=0.2,  # Use built-in validation split
        early_stopping=5,
        use_fp16=True,
        force_class_balance=True  # Use class balancing
    )
    
    # Save the model
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Balanced training approaches for image forgery detection")
    parser.add_argument('--method', type=str, choices=['weighted', 'oversampling', 'combined'], 
                      default='combined', help='Balancing method to use')
    parser.add_argument('--features_path', type=str, default='features/casia2_features_fixed.npz',
                      help='Path to features file')
    parser.add_argument('--model_path', type=str, default='models/casia2_balanced_model.pth',
                      help='Path to save model')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                      help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    
    args = parser.parse_args()
    
    # Call the appropriate function based on the method
    if args.method == 'weighted':
        train_with_balanced_sampling(
            args.features_path, 
            args.model_path, 
            args.output_dir,
            args.epochs,
            args.learning_rate,
            args.batch_size
        )
    elif args.method == 'oversampling':
        train_with_oversampling(
            args.features_path, 
            args.model_path, 
            args.output_dir,
            args.epochs,
            args.learning_rate,
            args.batch_size
        )
    elif args.method == 'combined':
        combined_approach(
            args.features_path, 
            args.model_path, 
            args.output_dir,
            args.epochs,
            args.learning_rate,
            args.batch_size
        )