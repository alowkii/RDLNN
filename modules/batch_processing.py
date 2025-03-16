import os
import numpy as np
from modules.preprocessing import preprocess_image
from modules.imageDecomposition import polar_dyadic_wavelet_transform
from modules.featureExtraction import extract_features

def batch_process_directory(directory, model):
    """
    Process all images in a directory and classify them as authentic or forged
    
    Args:
        directory: Directory containing images to process
        model: Trained RegressionDLNN model
    """
    results = {
        'authentic': [],
        'forged': [],
        'errors': []
    }
    
    print(f"Processing all images in {directory}...")
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        
        try:
            # Process image
            ycbcr_img = preprocess_image(img_path)
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
            prediction = model.predict(np.array(feature_vector))
            
            # Store result
            if prediction == 0:
                results['authentic'].append(img_name)
            else:
                results['forged'].append(img_name)
                
            print(f"Image {img_name}: {'FORGED' if prediction == 1 else 'AUTHENTIC'}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results['errors'].append(img_name)
            
    # Print summary
    print("\nProcessing complete!")
    print(f"Authentic images: {len(results['authentic'])}")
    print(f"Forged images: {len(results['forged'])}")
    print(f"Errors: {len(results['errors'])}")
    
    return results