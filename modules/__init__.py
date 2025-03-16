"""
Image Forgery Detection System - Module initialization
Provides exports for the main project modules
"""

# Import main classes and functions to make them available directly from the package
from .rdlnn import RegressionDLNN
from .data_handling import precompute_features, load_and_verify_features
from .preprocessing import preprocess_image
from .image_decomposition import perform_wavelet_transform
from .feature_extraction import extract_features_from_wavelet, BatchFeatureExtractor
from .batch_processor import OptimizedBatchProcessor
from .utils import logger, plot_training_history, setup_logging, setup_signal_handlers

# Define what gets imported with "from modules import *"
__all__ = [
    'RegressionDLNN',
    'precompute_features',
    'load_and_verify_features',
    'preprocess_image',
    'perform_wavelet_transform',
    'extract_features_from_wavelet',
    'BatchFeatureExtractor',
    'OptimizedBatchProcessor',
    'logger',
    'plot_training_history',
    'setup_logging',
    'setup_signal_handlers'
]

# Version information
__version__ = '1.0.0'