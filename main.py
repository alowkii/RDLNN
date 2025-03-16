def main():
    try:
        ycbcr_img = preprocess_image("./Data/test_image.jpg")
        if ycbcr_img is not None:
            polar_coeffs = polar_dyadic_wavelet_transform(ycbcr_img)
            if polar_coeffs is not None:
                print("Successfully generated polar dyadic wavelet coefficients")
                # Additional processing can be done here
            else:
                print("Failed to generate polar dyadic wavelet coefficients")
        else:
            print("Image preprocessing failed")
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()