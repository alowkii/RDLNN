import os
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from modules.preprocessing import preprocess_image
from modules.imageDecomposition import polar_dyadic_wavelet_transform
from modules.featureExtraction import extract_features
import gc

def batch_process_directory(directory, model, num_workers=4):
    """
    Process all images in a directory and classify them as authentic or forged
    using GPU acceleration and parallel processing
    
    Args:
        directory: Directory containing images to process
        model: Trained RegressionDLNN model
        num_workers: Number of parallel workers for processing
        
    Returns:
        Dictionary with results categorized as authentic, forged, and errors
    """
    results = {
        'authentic': [],
        'forged': [],
        'errors': []
    }
    
    # Get list of valid image files
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"Found {len(image_files)} images in {directory}")
    
    # Create CUDA streams for parallel GPU operations
    streams = [torch.cuda.Stream() for _ in range(min(num_workers, 8))]  # Cap at 8 streams
    
    # Process images in batches to avoid CUDA memory issues
    batch_size = 16  # Adjust based on GPU memory
    
    for batch_idx in range(0, len(image_files), batch_size):
        batch_files = image_files[batch_idx:batch_idx + batch_size]
        
        print(f"\nProcessing batch {batch_idx//batch_size + 1}/{(len(image_files)-1)//batch_size + 1} ({len(batch_files)} images)")
        
        # Process images in the batch with parallel workers
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for i, img_name in enumerate(batch_files):
                img_path = os.path.join(directory, img_name)
                stream_idx = i % len(streams)
                futures.append(executor.submit(
                    process_single_image, 
                    img_path, 
                    img_name, 
                    model, 
                    streams[stream_idx]
                ))
            
            # Collect results with progress bar
            for future in tqdm(futures, desc="Processing images", unit="img"):
                img_name, prediction, confidence, error = future.result()
                
                if error:
                    results['errors'].append(img_name)
                    print(f"Error processing {img_name}: {error}")
                else:
                    result_type = 'forged' if prediction == 1 else 'authentic'
                    results[result_type].append(img_name)
                    print(f"Image {img_name}: {result_type.upper()} (Confidence: {confidence:.2f})")
        
        # Clean up CUDA memory after each batch
        torch.cuda.empty_cache()
        gc.collect()
    
    # Print summary
    print("\nProcessing complete!")
    print(f"Authentic images: {len(results['authentic'])}")
    print(f"Forged images: {len(results['forged'])}")
    print(f"Errors: {len(results['errors'])}")
    
    return results

def process_single_image(img_path, img_name, model, stream=None):
    """
    Process a single image with GPU acceleration
    
    Args:
        img_path: Path to the image file
        img_name: Name of the image file
        model: Trained RegressionDLNN model
        stream: CUDA stream for concurrent processing
        
    Returns:
        Tuple of (img_name, prediction, confidence, error)
    """
    try:
        with torch.cuda.stream(stream) if stream else torch.cuda.default_stream():
            # Preprocess the image to YCbCr
            ycbcr_img = preprocess_image(img_path)
            if ycbcr_img is None:
                return img_name, None, 0, "Failed to preprocess image"
            
            # Apply polar dyadic wavelet transform
            pdywt_coeffs = polar_dyadic_wavelet_transform(ycbcr_img)
            
            # Convert tensor to dictionary format for feature extraction
            ycbcr_img_dict = {
                'y': ycbcr_img[0],
                'cb': ycbcr_img[1],
                'cr': ycbcr_img[2]
            }
            
            # Extract features
            feature_vector = extract_features(pdywt_coeffs, ycbcr_img_dict)
            
            # Make prediction
            prediction, confidence = model.predict(np.array(feature_vector))
            
            # Synchronize stream to ensure all GPU operations are complete
            if stream:
                stream.synchronize()
                
            return img_name, prediction, confidence, None
            
    except Exception as e:
        return img_name, None, 0, str(e)
    finally:
        # Explicitly free GPU memory
        torch.cuda.empty_cache()