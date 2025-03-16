import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os
import gc
from torch.amp import autocast, GradScaler
import time

class RegressionDLNN:
    """
    Regression Deep Learning Neural Network (RDLNN) for image forgery detection
    Implemented with PyTorch and CUDA support
    """
    
    def __init__(self, input_dim):
        """
        Initialize the RDLNN model
        
        Args:
            input_dim: Number of input features
        """
        super(RegressionDLNN, self).__init__()
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the model architecture
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Reduced dropout
            
            # Hidden layer
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),  # Reduced dropout
            
            # Output layer
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Initialize weights using Xavier initialization
        self._init_weights()
        
        # Define loss function
        self.loss_fn = nn.BCELoss()
        
        # For feature normalization
        self.scaler = StandardScaler()
        
        # Initialize gradient scaler for mixed precision training
        self.scaler_amp = GradScaler()
        
        # Print device information
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for computation")
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def fit(self, X, y, epochs=50, learning_rate=0.001, batch_size=32, validation_split=0.2, early_stopping=10, use_fp16=False):
        """
        Train the RDLNN model with optimized training loop
        
        Args:
            X: Feature vectors
            y: Labels (0: authentic, 1: forged)
            epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            early_stopping: Number of epochs with no improvement after which training will stop
            use_fp16: Whether to use half precision (FP16) operations
            
        Returns:
            Training history (dictionary with loss and accuracy metrics)
        """
        # Initialize optimizer with the given learning rate
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4  # L2 regularization
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
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
            
        # Split data into training and validation sets using sklearn to ensure balance
        X_train, X_val, y_train, y_val = train_test_split(
            X_np, y_np, test_size=validation_split, random_state=42, stratify=y_np
        )
        
        # Normalize features - fit only on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Convert to torch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Calculate optimal number of workers
        num_workers = min(4, os.cpu_count() or 4)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        # Print dataset information
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Class distribution in training: {np.bincount(y_train.flatten().astype(int))}")
        print(f"Class distribution in validation: {np.bincount(y_val.flatten().astype(int))}")
        
        # Set model to training mode
        self.model.train()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Early stopping setup
        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training metrics
            train_loss = 0.0
            train_correct = 0
            train_samples = 0
            
            # Training loop with mixed precision if requested
            for inputs, targets in train_loader:
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward and backward pass
                if use_fp16 and self.device.type == 'cuda':
                    # Using mixed precision
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, targets)
                    
                    # Scale loss and do backward pass
                    self.scaler_amp.scale(loss).backward()
                    self.scaler_amp.step(self.optimizer)
                    self.scaler_amp.update()
                else:
                    # Standard precision
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * inputs.size(0)
                predicted = (outputs >= 0.5).float()
                train_correct += (predicted == targets).sum().item()
                train_samples += inputs.size(0)
            
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
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    if use_fp16 and self.device.type == 'cuda':
                        with autocast(device_type='cuda', dtype=torch.float16):
                            outputs = self.model(inputs)
                            loss = self.loss_fn(outputs, targets)
                    else:
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    predicted = (outputs >= 0.5).float()
                    val_correct += (predicted == targets).sum().item()
                    val_samples += inputs.size(0)
                    
                    # Store predictions and targets for confusion matrix
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets_list.extend(targets.cpu().numpy())
            
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
            
            # Update learning rate scheduler
            self.scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                # Save best model state
                best_model_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    # Restore best model state
                    self.model.load_state_dict(best_model_state)
                    break
            
            # Back to training mode
            self.model.train()
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_accuracy)
            
            # Calculate epoch time
            epoch_time = time.time() - start_time
            
            # Print progress with detailed metrics
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"time: {epoch_time:.2f}s - "
                  f"loss: {avg_train_loss:.4f} - acc: {train_accuracy:.4f} - "
                  f"val_loss: {avg_val_loss:.4f} - val_acc: {val_accuracy:.4f}")
            print(f"Confusion Matrix - TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
            
            # Periodically clear CUDA cache
            if self.device.type == 'cuda' and epoch % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Final cleanup
        if best_model_state:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        return history
    
    def predict(self, X):
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
            print("Warning: StandardScaler not fitted yet. Using raw features.")
        
        # Convert to tensor and move to device
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        
        # Make prediction
        with torch.no_grad():
            confidences = self.model(X_tensor).cpu().numpy()
        
        # Convert to binary output with confidence
        predictions = (confidences >= 0.5).astype(int)
        
        # Return predictions and confidences
        return predictions.flatten(), confidences.flatten()
    
    def save(self, filepath):
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
        
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Load a trained model from disk
        
        Args:
            filepath: Path to saved model file
            
        Returns:
            Loaded RegressionDLNN model
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
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
        
        print(f"Model loaded successfully from {filepath}")
        return model