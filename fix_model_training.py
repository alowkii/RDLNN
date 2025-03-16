# run_improved_training.py
import os
import sys
import argparse
import torch
import numpy as np
from modules.rdlnn import RegressionDLNN
from modules.data_handling import load_and_verify_features

def fix_model_training():
    """
    Run an improved training session specifically designed to address class imbalance
    and prevent the model from predicting only one class.
    """
    # Create output directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Step 1: Load the features
    print("Loading precomputed features...")
    features_path = "features/my_features.npz"
    features, labels, paths = load_and_verify_features(features_path)
    
    if len(features) == 0:
        print("Error: No valid features found. Please check your data.")
        return
    
    # Step 2: Analyze feature importances to check data quality
    analyze_features(features, labels)
    
    # Step 3: Create and train the model with special initialization
    input_dim = features.shape[1]
    print(f"Creating model with input dimension: {input_dim}")
    model = RegressionDLNN(input_dim)
    
    # Modify the final layer to have a bias toward the minority class (forged)
    with torch.no_grad():
        model.model[-1].bias.fill_(0.5)  # Initialize with a bias toward the positive class
    
    # Step 4: Train the model with aggressive settings
    history = model.fit(
        features, labels,
        epochs=5,                    # More epochs for better learning
        learning_rate=0.0001,          # Much higher learning rate
        batch_size=64,                # Larger batch size
        validation_split=0.2,
        early_stopping=10,
        use_fp16=True,                # Use FP16 for speed
        force_class_balance=True      # Explicitly force class balance
    )
    
    # Step 5: Save the model
    model.save("models/balanced_model.pth")
    print("Model saved to models/balanced_model.pth")
    
    # Step 6: Evaluate the model
    evaluate_model(model, features, labels)

def analyze_features(features, labels):
    """Analyze features to find the most discriminative ones"""
    try:
        # Split by class
        authentic_features = features[labels == 0]
        forged_features = features[labels == 1]
        
        # Compute mean difference
        authentic_mean = np.mean(authentic_features, axis=0)
        forged_mean = np.mean(forged_features, axis=0)
        
        # Calculate absolute difference
        feature_diff = np.abs(authentic_mean - forged_mean)
        
        # Get indices of top 10 most discriminative features
        top_indices = np.argsort(-feature_diff)[:10]
        
        print("\nTop 10 most discriminative features:")
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. Feature {idx}: Difference = {feature_diff[idx]:.4f} "
                  f"(Auth={authentic_mean[idx]:.4f}, Forged={forged_mean[idx]:.4f})")
        
        # Check if features have good separation
        if np.max(feature_diff) < 0.1:
            print("\nWARNING: Features show very little difference between classes!")
            print("This may indicate poor feature quality for classification.")
        
        # Plot feature importance 
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(feature_diff)), feature_diff)
        plt.title('Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Absolute Difference Between Classes')
        plt.savefig('results/feature_importance.png')
        plt.close()
    except Exception as e:
        print(f"Error analyzing features: {e}")

def evaluate_model(model, features, labels):
    """Evaluate the trained model and check prediction balance"""
    print("\nEvaluating model on all data...")
    
    # Make predictions
    predictions, confidences = model.predict(features)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == labels)
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nConfusion Matrix:")
    print(f"                | Predicted Positive | Predicted Negative |")
    print(f"Actual Positive | {tp:^18} | {fn:^18} |")
    print(f"Actual Negative | {fp:^18} | {tn:^18} |")
    print(f"\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Check class distribution in predictions
    pred_positive = np.sum(predictions == 1)
    pred_negative = np.sum(predictions == 0)
    print(f"\nPrediction distribution:")
    print(f"  Class 0 (Authentic): {pred_negative} ({pred_negative/len(predictions)*100:.2f}%)")
    print(f"  Class 1 (Forged): {pred_positive} ({pred_positive/len(predictions)*100:.2f}%)")
    
    # Check confidence distribution
    low_conf = np.sum((confidences > 0.4) & (confidences < 0.6))
    print(f"\nPrediction confidence:")
    print(f"  Low confidence predictions (0.4-0.6): {low_conf} ({low_conf/len(predictions)*100:.2f}%)")
    
    if pred_positive == 0:
        print("\nWARNING: Model is still predicting only class 0!")
        print("Try using an even higher learning rate or different model architecture.")

if __name__ == "__main__":
    fix_model_training()