"""
Data handling functions for the forgery detection system
"""

import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, List, Dict, Any, Optional
from tqdm import tqdm

from modules.preprocessing import preprocess_image
from modules.image_decomposition import perform_wavelet_transform
from modules.feature_extraction import extract_features_from_wavelet
from modules.utils import logger, clean_cuda_memory

def precompute_features(directory: str, label=None, batch_size=16, num_workers=4, use_fp16=True, save_path=None) -> Dict[str, Any]:
    """
    Precompute and optionally save feature vectors for all images in a directory
    
    Args:
        directory: Directory containing images
        label: Label to assign to all images (0=authentic, 1=forged, None=unknown)
        batch_size: Batch size for processing
        num_workers: Number of worker threads
        use_fp16: Whether to use half precision
        save_path: Path to save feature vectors (optional)
        
    Returns:
        Dictionary with image paths, features, and labels
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Precomputing features using {device}")
    
    image_files = [f for f in os.listdir(directory) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    num_images = len(image_files)
    logger.info(f"Found {num_images} images in {directory}")
    
    # Allocate results
    results = {
        'paths': [],
        'features': [],
        'labels': []
    }
    
    # Process images in batches with a progress bar
    with tqdm(total=num_images, desc="Extracting features", unit="img") as pbar:
        for batch_idx in range(0, num_images, batch_size):
            batch_end = min(batch_idx + batch_size, num_images)
            batch_files = image_files[batch_idx:batch_end]
            batch_paths = [os.path.join(directory, f) for f in batch_files]
            
            # Extract features for batch
            batch_features = _process_image_batch(
                batch_paths, 
                device=device, 
                num_workers=num_workers, 
                batch_size=batch_size,
                use_fp16=use_fp16
            )
            
            if batch_features is not None:
                results['paths'].extend(batch_paths)
                results['features'].append(batch_features.cpu().numpy())
                
                # Add labels if provided
                if label is not None:
                    results['labels'].extend([label] * len(batch_paths))
            
            # Update progress bar with the number of successfully processed images
            pbar.update(len(batch_paths))
            
            # Clean up GPU memory after each batch
            clean_cuda_memory()
    
    # Combine features into a single array
    if results['features']:
        results['features'] = np.vstack(results['features'])
        
        # Save features if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(
                save_path,
                paths=results['paths'],
                features=results['features'],
                labels=results.get('labels', [])
            )
            logger.info(f"Saved {len(results['paths'])} feature vectors to {save_path}")
    
    return results

def _process_image_batch(image_paths: List[str], device: torch.device, num_workers: int, 
                        batch_size: int, use_fp16: bool) -> Optional[torch.Tensor]:
    """
    Process a batch of images and extract features
    
    Args:
        image_paths: List of paths to image files
        device: Computation device
        num_workers: Number of worker threads
        batch_size: Batch size
        use_fp16: Whether to use half precision
        
    Returns:
        Tensor of feature vectors for the batch or None if processing failed
    """
    try:
        # Use ThreadPoolExecutor for parallel preprocessing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            ycbcr_batch = list(executor.map(preprocess_image, image_paths))
        
        # Filter out any None results
        ycbcr_batch = [img for img in ycbcr_batch if img is not None]
        
        if not ycbcr_batch:
            return None
        
        # Get consistent dimensions for batching
        first_h, first_w = ycbcr_batch[0].shape[1], ycbcr_batch[0].shape[2]
        ycbcr_batch = [img for img in ycbcr_batch if img.shape[1] == first_h and img.shape[2] == first_w]
        
        # Stack into a single batch tensor
        if not ycbcr_batch:
            return None
                
        ycbcr_tensor = torch.stack(ycbcr_batch)  # [B, C, H, W]
        
        # Apply wavelet transform to each image in batch
        pdywt_batch = []
        for i in range(len(ycbcr_tensor)):
            pdywt_coeffs = perform_wavelet_transform(ycbcr_tensor[i])
            if pdywt_coeffs is not None:
                pdywt_batch.append(pdywt_coeffs)
        
        # Extract features
        feature_vectors = extract_features_from_wavelet(
            ycbcr_tensor, 
            pdywt_batch,
            device=device,
            batch_size=batch_size,
            use_fp16=use_fp16
        )
        
        return feature_vectors
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return None

def load_and_verify_features(features_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and verify precomputed features, ensuring features and labels match
    
    Args:
        features_path: Path to the features file
        
    Returns:
        Tuple of (features, labels, paths)
    """
    logger.info(f"Loading precomputed features from {features_path}...")
    
    try:
        data = np.load(features_path, allow_pickle=True)
        features = data['features']
        labels = data.get('labels', np.array([]))
        paths = data.get('paths', np.array([]))
        
        logger.info(f"Original dataset - Features: {features.shape}, Labels: {len(labels)}, Paths: {len(paths)}")
        
        # Check the number of paths, features, and labels
        min_length = min(len(features), len(labels)) if len(labels) > 0 else len(features)
        
        # Trim to make everything consistent
        features = features[:min_length]
        
        if len(labels) > 0:
            labels = np.array(labels[:min_length])
        
        if len(paths) > 0:
            paths = paths[:min_length]
        
        # Filter out NaN values
        valid_indices = []
        for i in range(len(features)):
            # Check if feature is valid (no NaN values)
            if not np.isnan(features[i]).any():
                if len(labels) == 0 or not np.isnan(labels[i] if np.isscalar(labels[i]) else labels[i].any()):
                    valid_indices.append(i)
        
        # Apply filtering
        filtered_features = features[valid_indices]
        filtered_labels = labels[valid_indices] if len(labels) > 0 else []
        filtered_paths = [paths[i] for i in valid_indices] if len(paths) > 0 else []
        
        logger.info(f"Filtered dataset - Features: {filtered_features.shape}")
        
        # Check class distribution if we have labels
        if len(filtered_labels) > 0:
            classes, counts = np.unique(filtered_labels, return_counts=True)
            logger.info("Class distribution:")
            for cls, count in zip(classes, counts):
                logger.info(f"  Class {cls}: {count} samples ({count/len(filtered_labels)*100:.2f}%)")
        
        # Check feature statistics
        logger.info(f"Feature statistics:")
        logger.info(f"  Mean: {np.mean(filtered_features):.4f}")
        logger.info(f"  Std: {np.std(filtered_features):.4f}")
        logger.info(f"  Min: {np.min(filtered_features):.4f}")
        logger.info(f"  Max: {np.max(filtered_features):.4f}")
        
        # Check for NaN values
        nan_count = np.isnan(filtered_features).sum()
        if nan_count > 0:
            logger.warning(f"NaN values detected: {nan_count}")
        else:
            logger.info("No NaN values detected")
        
        return filtered_features, filtered_labels, filtered_paths
        
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return np.array([]), np.array([]), []

def create_training_validation_split(features: np.ndarray, labels: np.ndarray, 
                                     validation_split: float = 0.2, random_seed: int = 42) -> Tuple:
    """
    Create a training-validation split of the data
    
    Args:
        features: Feature vectors
        labels: Labels
        validation_split: Fraction to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_features, val_features, train_labels, val_labels)
    """
    from sklearn.model_selection import train_test_split
    
    try:
        # Use sklearn's train_test_split for stratified splitting
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, 
            test_size=validation_split, 
            random_state=random_seed, 
            stratify=labels
        )
        
        # Log split info
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        
        if len(y_train) > 0:
            train_classes, train_counts = np.unique(y_train, return_counts=True)
            logger.info("Training class distribution:")
            for cls, count in zip(train_classes, train_counts):
                logger.info(f"  Class {cls}: {count} samples ({count/len(y_train)*100:.2f}%)")
                
        if len(y_val) > 0:
            val_classes, val_counts = np.unique(y_val, return_counts=True)
            logger.info("Validation class distribution:")
            for cls, count in zip(val_classes, val_counts):
                logger.info(f"  Class {cls}: {count} samples ({count/len(y_val)*100:.2f}%)")
        
        return X_train, X_val, y_train, y_val
        
    except Exception as e:
        logger.error(f"Error creating train-validation split: {e}")
        # Create an emergency split if there's an error
        n = len(features)
        val_size = int(n * validation_split)
        train_size = n - val_size
        return features[:train_size], features[train_size:], labels[:train_size], labels[train_size:]

def get_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Calculate class weights inversely proportional to class frequencies
    
    Args:
        labels: Class labels
        
    Returns:
        Dictionary mapping class indices to weights
    """
    unique_classes = np.unique(labels)
    class_counts = np.bincount(labels.flatten().astype(int))
    total_samples = len(labels)
    
    # Calculate weights inversely proportional to class frequencies
    class_weights = {}
    for i, count in enumerate(class_counts):
        if count > 0:  # Avoid division by zero
            class_weights[i] = total_samples / (len(unique_classes) * count)
    
    return class_weights

def get_balanced_sampler(labels: np.ndarray) -> torch.utils.data.sampler.WeightedRandomSampler:
    """
    Create a weighted sampler for balanced batches
    
    Args:
        labels: Class labels
        
    Returns:
        WeightedRandomSampler instance
    """
    from torch.utils.data import WeightedRandomSampler
    
    # Get class counts
    class_count = np.bincount(labels.flatten().astype(int))
    
    # Compute weights for each sample
    class_weights = 1. / class_count
    weights = torch.tensor([class_weights[int(t)] for t in labels.flatten()], dtype=torch.float32)
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(weights, len(weights))
    
    return sampler

def create_dataloaders(X_train, X_val, y_train, y_val, batch_size=32, num_workers=4, use_balanced_sampling=True):
    """
    Create PyTorch DataLoaders for training and validation
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        batch_size: Batch size
        num_workers: Number of worker threads
        use_balanced_sampling: Whether to use balanced sampling for training
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Calculate optimal number of workers
    num_workers = min(num_workers, os.cpu_count() or 4)
    
    # Create data loaders
    if use_balanced_sampling and np.bincount(y_train.flatten().astype(int))[0] != np.bincount(y_train.flatten().astype(int))[1]:
        logger.info("Using weighted sampler for balanced batches")
        sampler = get_balanced_sampler(y_train)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers
        )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return train_loader, val_loader