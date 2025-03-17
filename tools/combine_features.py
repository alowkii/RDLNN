#!/usr/bin/env python3
"""
Utility to combine authentic and forged features into a single file
"""

import numpy as np
import os
import argparse
from pathlib import Path

def combine_feature_files(authentic_path, forged_path, output_path):
    """
    Combine authentic and forged feature files into a single file with explicit labels.
    
    Args:
        authentic_path: Path to the authentic features file
        forged_path: Path to the forged features file
        output_path: Path to save the combined features
    
    Returns:
        Tuple of (features, labels, paths) from the combined data
    """
    # Load authentic data
    print(f"Loading authentic features from {authentic_path}")
    authentic_data = np.load(authentic_path, allow_pickle=True)
    authentic_features = authentic_data['features']
    authentic_paths = authentic_data['paths']

    # Load forged data
    print(f"Loading forged features from {forged_path}")
    forged_data = np.load(forged_path, allow_pickle=True)
    forged_features = forged_data['features']
    forged_paths = forged_data['paths']

    # Create explicit labels
    authentic_labels = np.zeros(len(authentic_features), dtype=np.int32)
    forged_labels = np.ones(len(forged_features), dtype=np.int32)

    # Combine all data
    combined_features = np.vstack([authentic_features, forged_features])
    combined_paths = list(authentic_paths) + list(forged_paths)
    combined_labels = np.concatenate([authentic_labels, forged_labels])

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save combined data
    np.savez(output_path,
             features=combined_features,
             paths=combined_paths,
             labels=combined_labels)

    # Verify the contents
    print("Combined features saved successfully:")
    print(f"- Path: {output_path}")
    print(f"- Features shape: {combined_features.shape}")
    print(f"- Labels shape: {combined_labels.shape}")
    print(f"- Unique labels: {np.unique(combined_labels)}")
    print(f"- Label counts: {np.bincount(combined_labels)}")
    
    return combined_features, combined_labels, combined_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine authentic and forged feature files")
    parser.add_argument('--authentic', type=str, required=True, help='Path to authentic features file')
    parser.add_argument('--forged', type=str, required=True, help='Path to forged features file')
    parser.add_argument('--output', type=str, required=True, help='Path to save combined features')
    
    args = parser.parse_args()
    combine_feature_files(args.authentic, args.forged, args.output)