from modules.preprocessing import preprocess_image
from modules.imageDecomposition import polar_dyadic_wavelet_transform
from modules.featureExtraction import extract_features

def main():
    try:
        ycbcr_img_tensor = preprocess_image("./Data/test_image.jpg")
        polar_coeffs = polar_dyadic_wavelet_transform(ycbcr_img_tensor)
        
        # Convert tensor to dictionary format expected by extract_features
        ycbcr_img_dict = {
            'y': ycbcr_img_tensor[0],  # Y channel
            'cb': ycbcr_img_tensor[1], # Cb channel
            'cr': ycbcr_img_tensor[2]  # Cr channel
        }
        
        features = extract_features(polar_coeffs, ycbcr_img_dict)
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == '__main__':
    main()