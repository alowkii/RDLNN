import os
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import gc
from modules.preprocessing import preprocess_image
from modules.imageDecomposition import polar_dyadic_wavelet_transform
from modules.featureExtraction import BatchFeatureExtractor

class OptimizedBatchProcessor:
    """
    Optimized batch processor for image forgery detection
    Processes images in batches keeping data on GPU throughout the pipeline
    """
    
    def __init__(self, model=None, batch_size=16, num_workers=4, use_fp16=True):
        """
        Initialize the batch processor
        
        Args:
            model: Trained RegressionDLNN model (optional)
            batch_size: Number of images to process in a batch
            num_workers: Number of parallel workers for data loading
            use_fp16: Whether to use half precision (FP16) operations
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.num_workers = min(num_workers, os.cpu_count() or 4)
        self.use_fp16 = use_fp16
        self.model = model
        
        # Create multiple CUDA streams for overlapping operations
        self.streams = {
            'preprocess': torch.cuda.Stream(device=self.device),
            'transform': torch.cuda.Stream(device=self.device),
            'features': torch.cuda.Stream(device=self.device),
            'predict': torch.cuda.Stream(device=self.device)
        }
        
        # Initialize the batch feature extractor
        self.feature_extractor = BatchFeatureExtractor(
            device=self.device,
            batch_size=batch_size,
            num_workers=self.num_workers,
            use_fp16=use_fp16
        )
        
        # Set up mixed precision if enabled
        self.amp_dtype = torch.float16 if use_fp16 else torch.float32
        self.scaler = torch.cuda.amp.GradScaler() if use_fp16 else None
    
    def precompute_features(self, directory, label=None, save_path=None):
        """
        Precompute and optionally save feature vectors for all images in a directory
        
        Args:
            directory: Directory containing images
            label: Label to assign to all images (0=authentic, 1=forged, None=unknown)
            save_path: Path to save feature vectors (optional)
            
        Returns:
            Dictionary with image paths, features, and labels
        """
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
            for batch_idx in range(0, num_images, self.batch_size):
                batch_end = min(batch_idx + self.batch_size, num_images)
                batch_files = image_files[batch_idx:batch_end]
                batch_paths = [os.path.join(directory, f) for f in batch_files]
                
                # Extract features for batch
                batch_features = self._process_image_batch(batch_paths)
                
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
    
    def _process_image_batch(self, image_paths):
        """
        Process a batch of images and extract features
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Tensor of feature vectors for the batch
        """
        try:
            batch_size = len(image_paths)
            
            # STAGE 1: Preprocess images in parallel (CPU) and move to GPU
            with torch.cuda.stream(self.streams['preprocess']):
                # Use ThreadPoolExecutor for parallel preprocessing
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
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
                
            # STAGE 2: Apply wavelet transform
            with torch.cuda.stream(self.streams['transform']):
                # Record the current stream to manage synchronization
                self.streams['preprocess'].synchronize()
                
                # Apply wavelet transform to each image in batch
                pdywt_batch = []
                for i in range(len(ycbcr_tensor)):
                    pdywt_coeffs = polar_dyadic_wavelet_transform(ycbcr_tensor[i])
                    pdywt_batch.append(pdywt_coeffs)
            
            # STAGE 3: Extract features
            with torch.cuda.stream(self.streams['features']):
                # Ensure wavelet transform is complete
                self.streams['transform'].synchronize()
                
                # Prepare image batch for feature extraction
                ycbcr_dict_batch = []
                for i in range(len(ycbcr_tensor)):
                    ycbcr_dict = {
                        'y': ycbcr_tensor[i, 0],
                        'cb': ycbcr_tensor[i, 1],
                        'cr': ycbcr_tensor[i, 2]
                    }
                    ycbcr_dict_batch.append(ycbcr_dict)
                
                # Use the BatchFeatureExtractor for efficient feature extraction
                with torch.amp.autocast(device_type='cuda',enabled=self.use_fp16):
                    feature_vectors = self.feature_extractor.extract_batch_features(
                        ycbcr_tensor, pdywt_batch
                    )
            
            # Ensure all operations are complete
            self.streams['features'].synchronize()
            
            return feature_vectors
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            return None
        
    def batch_predict(self, features):
        """
        Make predictions on a batch of feature vectors
        
        Args:
            features: Tensor or numpy array of feature vectors
            
        Returns:
            Tuple of predictions and confidences
        """
        if self.model is None:
            raise ValueError("Model not initialized. Please provide a model during initialization.")
        
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Convert to tensor if numpy array
            if isinstance(features, np.ndarray):
                features = torch.tensor(features, dtype=torch.float32, device=self.device)
            
            # Ensure features are on the correct device
            features = features.to(self.device)
            
            # Apply scaling if model has a scaler
            if hasattr(self.model, 'scaler') and self.model.scaler is not None:
                features_np = features.cpu().numpy()
                features_scaled = self.model.scaler.transform(features_np)
                features = torch.tensor(features_scaled, dtype=torch.float32, device=self.device)
            
            # Make predictions with mixed precision
            with torch.no_grad(), torch.cuda.stream(self.streams['predict']), \
                 torch.amp.autocast(device_type='cuda',enabled=self.use_fp16):
                confidences = self.model.model(features).squeeze(-1)
            
            # Convert to binary predictions
            predictions = (confidences >= 0.5).float()
            
            # Return as numpy arrays
            return predictions.cpu().numpy(), confidences.cpu().numpy()
            
        except Exception as e:
            print(f"Error during batch prediction: {e}")
            return None, None
            
    def process_directory(self, directory, results_file=None):
        """
        Process all images in a directory and classify them as authentic or forged
        
        Args:
            directory: Directory containing images to process
            results_file: Path to save results (optional)
            
        Returns:
            Dictionary with results
        """
        # First, precompute features for all images
        print("Precomputing features for all images...")
        precomputed = self.precompute_features(directory)
        
        if not precomputed['features'].size:
            print("No valid features were extracted.")
            return {'authentic': [], 'forged': [], 'errors': []}
        
        # Make predictions on all features at once
        print("Making predictions on all images...")
        predictions, confidences = self.batch_predict(precomputed['features'])
        
        if predictions is None:
            print("Error during prediction")
            return {'authentic': [], 'forged': [], 'errors': precomputed['paths']}
        
        # Organize results
        results = {
            'authentic': [],
            'forged': [],
            'errors': []
        }
        
        # Display results with a progress bar
        with tqdm(total=len(precomputed['paths']), desc="Processing results", unit="img") as pbar:
            for i, (path, pred, conf) in enumerate(zip(precomputed['paths'], predictions, confidences)):
                img_name = os.path.basename(path)
                result_type = 'forged' if pred == 1 else 'authentic'
                results[result_type].append(img_name)
                pbar.set_postfix(file=img_name, result=result_type, conf=f"{conf:.2f}")
                pbar.update(1)
        
        # Print summary
        print("\nProcessing complete!")
        print(f"Authentic images: {len(results['authentic'])}")
        print(f"Forged images: {len(results['forged'])}")
        print(f"Errors: {len(results['errors'])}")
        
        # Save results if file path provided
        if results_file:
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'w') as f:
                f.write("Authentic Images:\n")
                for img in results['authentic']:
                    f.write(f"{img}\n")
                
                f.write("\nForged Images:\n")
                for img in results['forged']:
                    f.write(f"{img}\n")
                    
                f.write("\nError Images:\n")
                for img in results['errors']:
                    f.write(f"{img}\n")
            
            print(f"Results saved to {results_file}")
        
        return results