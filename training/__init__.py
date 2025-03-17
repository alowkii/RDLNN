"""
Training methods for image forgery detection
"""

# Import training functions so they can be accessed from training package
from .balanced import train_with_balanced_sampling, train_with_oversampling, combined_approach
from .precision import precision_tuned_training

# Define what gets imported with "from training import *"
__all__ = [
    'train_with_balanced_sampling',
    'train_with_oversampling',
    'combined_approach', 
    'precision_tuned_training'
]