"""
Regression Deep Learning Neural Network (RDLNN) for image forgery detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os
import gc
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from tqdm import tqdm

from modules.utils import logger, plot_diagnostic_curves, clean_cuda_memory
from modules.data_handling import create_training_validation_split, create_dataloaders, get_class_weights

class RegressionDLNN:
    """
    Regression Deep Learning Neural Network (RDLNN) for image forgery detection
    Implemented with PyTorch and CUDA support with improved handling of imbalanced data
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize the RDLNN model
        
        Args:
            input_dim: Number of input features
        """
        super(RegressionDLNN, self).__init__()
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the model architecture with improved layers
        self.model = nn.Sequential(
            # Input layer with larger size
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            # Hidden layers
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            # Output layer - sigmoid is handled in loss function
            nn.Linear(64, 1)
        ).to(self.device)
        
        # Initialize weights using Xavier initialization
        self._init_weights()
        
        # Define loss function - will be updated with class weights
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # For feature normalization
        self.scaler = StandardScaler()
        
        # Initialize gradient scaler for mixed precision training
        self.scaler_amp = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None
        
        # Print device information
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for computation")
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            y: Union[np.ndarray, torch.Tensor], 
            epochs: int = 50, 
            learning_rate: float = 0.001, 
            batch_size: int = 32, 
            validation_split: float = 0.2, 
            early_stopping: int = 10, 
            use_fp16: bool = False, 
            force_class_balance: bool = False) -> Dict[str, List[float]]:
        """
        Train the RDLNN model with optimized training loop and balanced sampling
        
        Args:
            X: Feature vectors
            y: Labels (0: authentic, 1: forged)
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            early_stopping: Number of epochs with no improvement after which training will stop
            use_fp16: Whether to use half precision (FP16) operations
            force_class_balance: Whether to enforce class balance during training
            
        Returns:
            Training history (dictionary with loss and accuracy metrics)
        """
        # Initialize optimizer with the given learning rate
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,  # L2 regularization
            betas=(0.9, 0.999)  # Default Adam parameters
        )
        
        logger.info(f"Training with learning rate: {learning_rate:.6f}")
        
        # Learning rate scheduler with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # Restart every 5 epochs
            T_mult=2,  # Multiply period by 2 at each restart
            eta_min=learning_rate / 10,
        )
        
        # Ensure we're working with numpy arrays for the preprocessing steps
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        else:
            X_np = X
            
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = y
        
        # Properly reshape y if needed
        if len(y_np.shape) == 1:
            y_np = y_np.reshape(-1, 1)
            
        # Handle class imbalance by using class weights
        unique_classes = np.unique(y_np)
        class_counts = np.bincount(y_np.flatten().astype(int))
        total_samples = len(y_np)
        
        # Calculate class weights inversely proportional to class frequencies
        class_weights = {}
        for i, count in enumerate(class_counts):
            class_weights[i] = total_samples / (len(unique_classes) * count)
        
        # Print class weights
        logger.info(f"Class weights: {class_weights}")
        
        # Use MUCH higher weight for minority class (forged)
        # This is crucial to fix the "all one class" prediction problem
        if force_class_balance:
            pos_weight_value = 5.0  # Force very high weight for positive class
            logger.info(f"Forcing positive class weight to {pos_weight_value}")
        else:
            pos_weight_value = class_weights[1]/class_weights[0]
            
        # Update loss function to use weights
        weight_tensor = torch.tensor([pos_weight_value], device=self.device)
        logger.info(f"Using positive weight: {pos_weight_value:.4f}")
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=weight_tensor
        )
            
        # Split data into training and validation sets using sklearn to ensure balance
        X_train, X_val, y_train, y_val = create_training_validation_split(
            X_np, y_np, validation_split=validation_split
        )
        
        # Verify balance after split
        logger.info(f"Training class distribution: {np.bincount(y_train.flatten().astype(int))}")
        logger.info(f"Validation class distribution: {np.bincount(y_val.flatten().astype(int))}")
        
        # Normalize features - fit only on training data
        self.scaler = StandardScaler().fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            X_train_scaled, X_val_scaled, y_train, y_val,
            batch_size=batch_size,
            num_workers=min(4, os.cpu_count() or 4),
            use_balanced_sampling=True if not force_class_balance else False
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Early stopping setup
        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Training metrics
            train_loss = 0.0
            train_correct = 0
            train_samples = 0
            
            # Set model to training mode
            self.model.train()
            
            # Training loop with tqdm progress bar
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for inputs, targets in train_pbar:
                # Move inputs and targets to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Add noise to target values to stabilize training
                if force_class_balance:
                    # Add noise to break symmetry - very important!
                    target_noise = torch.randn_like(targets) * 0.05
                    targets = targets + target_noise
                    targets = torch.clamp(targets, 0.0, 1.0)
                
                # Forward and backward pass
                if use_fp16 and self.device.type == 'cuda':
                    # Using mixed precision
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, targets)
                    
                    # Scale loss and do backward pass
                    self.scaler_amp.scale(loss).backward()
                    
                    # Gradient clipping to stabilize training
                    self.scaler_amp.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler_amp.step(self.optimizer)
                    self.scaler_amp.update()
                else:
                    # Standard precision
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * inputs.size(0)
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                train_correct += (predicted == targets.round()).sum().item()
                train_samples += inputs.size(0)
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': train_correct / train_samples
                })
            
            # Calculate epoch metrics
            avg_train_loss = train_loss / train_samples
            train_accuracy = train_correct / train_samples
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples = 0
            
            val_preds = []
            val_targets_list = []
            val_outputs_list = []
            
            # Validation loop with tqdm progress bar
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            with torch.no_grad():
                for inputs, targets in val_pbar:
                    # Move inputs and targets to device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    if use_fp16 and self.device.type == 'cuda':
                        with torch.amp.autocast("cuda"):
                            outputs = self.model(inputs)
                            loss = self.loss_fn(outputs, targets)
                    else:
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    
                    # Explicitly calculate sigmoid to get probabilities
                    probabilities = torch.sigmoid(outputs)
                    predicted = (probabilities >= 0.5).float()
                    
                    val_correct += (predicted == targets).sum().item()
                    val_samples += inputs.size(0)
                    
                    # Store predictions and targets for confusion matrix
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets_list.extend(targets.cpu().numpy())
                    val_outputs_list.extend(probabilities.cpu().numpy())
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        'loss': loss.item(),
                        'acc': val_correct / val_samples
                    })
            
            # Calculate validation metrics
            avg_val_loss = val_loss / val_samples
            val_accuracy = val_correct / val_samples
            
            # Print confusion matrix for the epoch
            val_preds = np.array(val_preds).flatten()
            val_targets_list = np.array(val_targets_list).flatten()
            
            # Compute confusion matrix
            tp = np.sum((val_preds == 1) & (val_targets_list == 1))
            tn = np.sum((val_preds == 0) & (val_targets_list == 0))
            fp = np.sum((val_preds == 1) & (val_targets_list == 0))
            fn = np.sum((val_preds == 0) & (val_targets_list == 1))
            
            # Calculate precision, recall, and F1 score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Update learning rate after each epoch
            self.scheduler.step()
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                # Save best model state
                best_model_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                logger.info(f"[OK] New best validation loss: {best_val_loss:.6f}")
            else:
                early_stopping_counter += 1
                logger.info(f"! Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping}")
                
                if early_stopping_counter >= early_stopping:
                    logger.info("\n" + "="*80)
                    logger.info(f"EARLY STOPPING TRIGGERED (after {epoch+1} epochs)")
                    logger.info("="*80)
                    
                    # Check if model is just predicting one class
                    unique_preds, counts = np.unique(val_preds, return_counts=True)
                    pred_dist = {int(k): int(v) for k, v in zip(unique_preds, counts)}
                    
                    if len(pred_dist) == 1:
                        only_class = list(pred_dist.keys())[0]
                        logger.warning(f"WARNING: Model is predicting ONLY class {only_class} for ALL samples!")
                        logger.warning("This suggests the model hasn't learned to differentiate between classes.")
                        logger.info("Suggestions:")
                        logger.info("1. Increase learning rate (try 0.001 or 0.005)")
                        logger.info("2. Train for more epochs")
                        logger.info("3. Check feature quality/usefulness")
                        logger.info("4. Consider using a different model architecture")
                    else:
                        logger.info(f"Final prediction distribution: {pred_dist}")
                    
                    # Restore best model state
                    self.model.load_state_dict(best_model_state)
                    break
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_accuracy)
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Print progress with detailed metrics in a cleaner format
            logger.info("\n" + "="*80)
            logger.info(f"EPOCH {epoch+1}/{epochs} (Time: {epoch_time:.2f}s, LR: {current_lr:.6f})")
            logger.info("-"*80)
            logger.info(f"Training   | Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.2%}")
            logger.info(f"Validation | Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2%}")
            logger.info("-"*80)
            logger.info(f"Confusion Matrix:")
            logger.info(f"                | Predicted Positive | Predicted Negative |")
            logger.info(f"Actual Positive | {tp:^18} | {fn:^18} |")
            logger.info(f"Actual Negative | {fp:^18} | {tn:^18} |")
            logger.info("-"*80)
            logger.info(f"Metrics | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            
            # Show prediction distribution
            unique_preds, counts = np.unique(val_preds, return_counts=True)
            pred_dist = {int(k): int(v) for k, v in zip(unique_preds, counts)}
            
            # Format nicely
            if len(pred_dist) == 1:
                only_class = list(pred_dist.keys())[0]
                logger.warning(f"WARNING: Model is predicting ONLY class {only_class} for ALL validation samples!")
            else:
                logger.info(f"Prediction distribution: {pred_dist}")
            
            # Calculate and print probability distribution
            val_outputs_array = np.array(val_outputs_list).flatten()
            if len(val_outputs_array) > 0:
                # Calculate percentiles for better understanding
                p10 = np.percentile(val_outputs_array, 10)
                p25 = np.percentile(val_outputs_array, 25)
                p50 = np.percentile(val_outputs_array, 50)
                p75 = np.percentile(val_outputs_array, 75)
                p90 = np.percentile(val_outputs_array, 90)
                
                logger.info("-"*80)
                logger.info("Probability Distribution:")
                logger.info(f"Range: [{val_outputs_array.min():.4f} - {val_outputs_array.max():.4f}] | Mean: {val_outputs_array.mean():.4f} | Std: {val_outputs_array.std():.4f}")
                logger.info(f"Percentiles: 10%: {p10:.4f} | 25%: {p25:.4f} | 50%: {p50:.4f} | 75%: {p75:.4f} | 90%: {p90:.4f}")
                
                # Analyze confidence levels
                high_conf = np.sum((val_outputs_array > 0.9) | (val_outputs_array < 0.1))
                medium_conf = np.sum((val_outputs_array > 0.7) & (val_outputs_array < 0.9) | 
                                    (val_outputs_array > 0.1) & (val_outputs_array < 0.3))
                uncertain = np.sum((val_outputs_array >= 0.3) & (val_outputs_array <= 0.7))
                
                logger.info(f"Confidence levels:")
                logger.info(f"  High (>0.9 or <0.1): {high_conf} samples ({high_conf/len(val_outputs_array)*100:.1f}%)")
                logger.info(f"  Medium: {medium_conf} samples ({medium_conf/len(val_outputs_array)*100:.1f}%)")
                logger.info(f"  Uncertain (0.3-0.7): {uncertain} samples ({uncertain/len(val_outputs_array)*100:.1f}%)")
                logger.info("="*80)
            
            # Periodically clear CUDA cache
            if self.device.type == 'cuda' and epoch % 5 == 0:
                clean_cuda_memory()
        
        # Final cleanup
        if best_model_state:
            clean_cuda_memory()
        
        # Generate model diagnostics
        try:
            plot_diagnostic_curves(val_targets_list, val_outputs_array, history, 'results')
        except Exception as e:
            logger.error(f"Error generating diagnostic curves: {e}")
        
        return history
    
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict if an image is forged (1) or authentic (0)
        
        Args:
            X: Feature vector or batch of feature vectors
            
        Returns:
            Tuple of (predictions, confidences)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Handle single feature vector or batch
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Normalize features
        if hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
            logger.warning("Warning: StandardScaler not fitted yet. Using raw features.")
        
        # Convert to tensor and move to device
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        
        # Make prediction
        with torch.no_grad():
            # Get raw logits from model
            logits = self.model(X_tensor)
            # Apply sigmoid to get probabilities
            confidences = torch.sigmoid(logits).cpu().numpy()
        
        # Convert to binary output with confidence
        predictions = (confidences >= 0.5).astype(int)
        
        # Return predictions and confidences
        return predictions.flatten(), confidences.flatten()
    
    def save(self, filepath: str) -> None:
        """Save the model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state, scaler, and metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'scaler': self.scaler,
            'input_dim': self.model[0].in_features,
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'RegressionDLNN':
        """Load a trained model from disk
        
        Args:
            filepath: Path to saved model file
            
        Returns:
            Loaded RegressionDLNN model
        """
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Get input dimension from checkpoint
        input_dim = checkpoint.get('input_dim')
        if not input_dim:
            raise ValueError("Input dimension not found in checkpoint")
            
        # Create new model instance
        model = cls(input_dim)
        
        # Load model state
        model.model.load_state_dict(checkpoint['model_state_dict'])

        # Load scaler
        model.scaler = checkpoint['scaler']

        # Move model to the correct device after loading
        model.model = model.model.to(model.device)
        
        logger.info(f"Model loaded successfully from {filepath}")
        return model