"""
Utility functions for the image forgery detection system
"""

import os
import sys
import gc
import signal
import logging
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger('forgery_detection')

def setup_logging(output_dir: str, level: int = logging.INFO) -> None:
    """
    Configure logging for the application
    
    Args:
        output_dir: Directory to store log files
        level: Logging level
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure file handler
    log_path = os.path.join(output_dir, 'forgery_detection.log')
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure logger
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent log messages from being propagated to the root logger
    logger.propagate = False

def signal_handler(sig: int, frame: Any) -> None:
    """
    Handle signals for graceful shutdown
    Ensures proper CUDA cleanup before exit
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    logger.info('Received termination signal. Cleaning up resources before exit...')
    
    # Flush any open file handles
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Clean up CUDA resources if available
    if torch.cuda.is_available():
        try:
            # Synchronize all CUDA streams first
            torch.cuda.synchronize()
            # Then empty cache
            torch.cuda.empty_cache()
            logger.info('CUDA resources cleaned up.')
        except Exception as e:
            logger.error(f"Error during CUDA cleanup: {e}")
    
    # Collect Python garbage
    gc.collect()
    
    logger.info('Cleanup complete. Exiting...')
    sys.exit(0)

def setup_signal_handlers() -> None:
    """Register signal handlers for graceful shutdown"""
    import torch
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    logger.debug("Signal handlers registered for graceful shutdown")

def plot_training_history(history: Dict[str, Any], output_dir: str) -> None:
    """
    Plot training history metrics
    
    Args:
        history: Dictionary containing training metrics
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    history_file = os.path.join(output_dir, 'training_history.png')
    plt.savefig(history_file)
    logger.info(f"Training history plot saved to {history_file}")

def plot_diagnostic_curves(y_true: Any, y_pred_prob: Any, history: Dict[str, Any], output_dir: str) -> None:
    """
    Generate ROC and Precision-Recall curves for model diagnostics
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        history: Dictionary containing training metrics
        output_dir: Directory to save the plot
    """
    try:
        import sklearn.metrics as metrics
        import numpy as np
        
        # Only proceed if we have predictions from both classes
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            logger.warning("Cannot generate diagnostic curves: not enough classes in validation set")
            return
        
        # Calculate ROC curve
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob)
        roc_auc = metrics.auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred_prob)
        pr_auc = metrics.average_precision_score(y_true, y_pred_prob)
        
        # Create diagnostic plots
        plt.figure(figsize=(15, 5))
        
        # ROC curve
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        
        # Precision-Recall curve
        plt.subplot(1, 3, 2)
        plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        
        # Learning rate curve
        plt.subplot(1, 3, 3)
        plt.semilogy(history['learning_rates'], label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate (log scale)')
        plt.title('Learning Rate Decay')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_diagnostics.png'))
        logger.info(f"Model diagnostic curves saved to {os.path.join(output_dir, 'model_diagnostics.png')}")
            
    except ImportError:
        logger.warning("Scikit-learn is required for diagnostic curves")
    except Exception as e:
        logger.error(f"Error generating diagnostic curves: {e}")

def clean_cuda_memory() -> None:
    """Free CUDA memory and collect garbage"""
    import torch
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()