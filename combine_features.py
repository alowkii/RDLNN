import numpy as np
import os

# Load authentic data
authentic_data = np.load('features/casia2_features_authentic.npz', allow_pickle=True)
authentic_features = authentic_data['features']
authentic_paths = authentic_data['paths']

# Load forged data
forged_data = np.load('features/casia2_features_forged.npz', allow_pickle=True)
forged_features = forged_data['features']
forged_paths = forged_data['paths']

# Create explicit labels
authentic_labels = np.zeros(len(authentic_features), dtype=np.int32)
forged_labels = np.ones(len(forged_features), dtype=np.int32)

# Combine all data
combined_features = np.vstack([authentic_features, forged_features])
combined_paths = list(authentic_paths) + list(forged_paths)
combined_labels = np.concatenate([authentic_labels, forged_labels])

# Save combined data
np.savez('features/casia2_features_fixed.npz',
         features=combined_features,
         paths=combined_paths,
         labels=combined_labels)

# Verify the contents
data = np.load('features/casia2_features_fixed.npz')
print(f"Combined features shape: {data['features'].shape}")
print(f"Labels shape: {data['labels'].shape}")
print(f"Unique labels: {np.unique(data['labels'])}")
print(f"Label counts: {np.bincount(data['labels'])}")