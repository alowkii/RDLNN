import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from modules.preprocessing import preprocess_image
from modules.imageDecomposition import polar_dyadic_wavelet_transform
from modules.featureExtraction import BatchFeatureExtractor
from tqdm import tqdm

def precompute_features(directory, label=None, batch_size=16, num_workers=4, use_fp16=True, save_path=None):
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
    
    # Initialize the batch feature extractor
    feature_extractor = BatchFeatureExtractor(
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        use_fp16=use_fp16
    )
    
    image_files = [f for f in os.listdir(directory) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    num_images = len(image_files)
    print(f"Found {num_images} images in {directory}")
    
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
            batch_features = _process_image_batch(batch_paths, feature_extractor, device, num_workers)
            
            if batch_features is not None:
                results['paths'].extend(batch_paths)
                results['features'].append(batch_features.cpu().numpy())
                
                # Add labels if provided
                if label is not None:
                    results['labels'].extend([label] * len(batch_paths))
            
            # Update progress bar with the number of successfully processed images
            pbar.update(len(batch_paths))
            
            # Clean up GPU memory after each batch
            torch.cuda.empty_cache()
    
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
            print(f"Saved {len(results['paths'])} feature vectors to {save_path}")
    
    return results

def _process_image_batch(image_paths, feature_extractor, device, num_workers):
    """
    Process a batch of images and extract features
    
    Args:
        image_paths: List of paths to image files
        feature_extractor: BatchFeatureExtractor instance
        device: Torch device
        num_workers: Number of worker threads
        
    Returns:
        Tensor of feature vectors for the batch
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
            pdywt_coeffs = polar_dyadic_wavelet_transform(ycbcr_tensor[i])
            pdywt_batch.append(pdywt_coeffs)
        
        # Extract features using BatchFeatureExtractor
        feature_vectors = feature_extractor.extract_batch_features(
            ycbcr_tensor, pdywt_batch
        )
        
        return feature_vectors
        
    except Exception as e:
        print(f"Error processing batch: {e}")
        return None

def load_and_verify_features(features_path):
    """
    Load and verify precomputed features, ensuring features and labels match
    
    Args:
        features_path: Path to the features file
        
    Returns:
        Tuple of (features, labels, paths)
    """
    print(f"Loading precomputed features from {features_path}...")
    data = np.load(features_path)
    features = data['features']
    labels = data.get('labels', [])
    paths = data.get('paths', [])
    
    print(f"Original dataset - Features: {features.shape}, Labels: {len(labels)}, Paths: {len(paths)}")
    
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
    
    print(f"Filtered dataset - Features: {filtered_features.shape}")
    
    # Check class distribution if we have labels
    if len(filtered_labels) > 0:
        classes, counts = np.unique(filtered_labels, return_counts=True)
        print("Class distribution:")
        for cls, count in zip(classes, counts):
            print(f"  Class {cls}: {count} samples ({count/len(filtered_labels)*100:.2f}%)")
    
    # Check feature statistics
    print(f"Feature statistics:")
    print(f"  Mean: {np.mean(filtered_features):.4f}")
    print(f"  Std: {np.std(filtered_features):.4f}")
    print(f"  Min: {np.min(filtered_features):.4f}")
    print(f"  Max: {np.max(filtered_features):.4f}")
    print(f"  NaN values: {np.isnan(filtered_features).sum()}")
    
    return filtered_features, filtered_labels, filtered_paths