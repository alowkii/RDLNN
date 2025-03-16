import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import gc
from torch.amp import autocast, GradScaler
import time

class RegressionDLNN(nn.Module):
    """
    Regression Deep Learning Neural Network (RDLNN) for image forgery detection
    Implemented with PyTorch and CUDA support with additional optimizations
    """
    
    def __init__(self, input_dim):
        """
        Initialize the RDLNN model with optimized architecture
        
        Args:
            input_dim: Number of input features
        """
        super(RegressionDLNN, self).__init__()
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the model architecture with stronger regularization
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, 8),  # Reduced from 64 to 8
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.9),  # Increased from 0.4 to 0.8
            
            # Output layer - removed middle layer
            nn.Linear(8, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Initialize weights using Kaiming initialization
        self._init_weights()
        
        # Define loss function
        self.loss_fn = nn.BCELoss()
        
        # Will initialize optimizer in fit method with stronger weight decay
        self.optimizer = None
        self.scheduler = None
        
        # For feature normalization
        self.scaler = StandardScaler()
        
        # Initialize gradient scaler for mixed precision training
        self.scaler_amp = GradScaler()
        
        # Print device information
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("Using CPU for computation")
        
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)
    
    def fit(self, X, y, epochs=50, learning_rate=0.001, batch_size=32, validation_split=0.3, early_stopping=3, use_fp16=False):
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
        # Initialize optimizer and scheduler with the given learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-2  # Increased from 5e-3
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        # If CUDA is available and using it
        if torch.cuda.is_available() and self.device.type == 'cuda':
            # Set appropriate CUDA device
            if self.device.type == 'cuda' and hasattr(self.device, 'index') and self.device.index is not None:
                torch.cuda.set_device(self.device.index)
        
        # Normalize features if X is a numpy array
        if isinstance(X, np.ndarray):
            X_scaled = self.scaler.fit_transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            # Assume they're already torch tensors
            X_tensor = X
            y_tensor = y
            
        # Ensure y is 2D
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.reshape(-1, 1)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into training and validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        
        # Use generator for reproducibility
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
        # Calculate optimal number of workers based on system
        num_workers = min(4, os.cpu_count() or 4)
        
        # Create data loaders with device optimizations
        # Create data loaders with device optimizations
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=False,  # Change this from True or from device.type=='cuda'
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size * 2,
            pin_memory=False,  # Change this from True or from device.type=='cuda'
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        # Set model to training mode
        self.model.train()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # CUDA events for timing and synchronization
        if self.device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
            stream1 = torch.cuda.Stream()
            stream2 = torch.cuda.Stream()
        
        # Early stopping setup
        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        
        # Preallocate tensors for metrics to avoid recreating them
        train_loss_tensor = torch.zeros(1, device=self.device)
        val_loss_tensor = torch.zeros(1, device=self.device)
        
        for epoch in range(epochs):
            if self.device.type == 'cuda':
                start_event.record()
            else:
                start_time = time.time()
            
            # Training metrics
            train_loss = 0.0
            train_correct = 0
            train_samples = 0
            
            # Training loop with mixed precision if requested
            for inputs, targets in train_loader:
                # Zero the gradients
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Forward and backward pass
                if use_fp16 and self.device.type == 'cuda':
                    # Using mixed precision
                    with autocast(device_type='cuda', dtype=torch.float16):
                        # Add noise during training for regularization
                        if self.training:
                            noise = torch.randn_like(inputs) * 0.05  # 5% noise
                            noisy_inputs = inputs + noise
                            outputs = self.model(noisy_inputs)
                        else:
                            outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, targets)
                    
                    # Scale loss and do backward pass
                    self.scaler_amp.scale(loss).backward()
                    
                    # Update weights with gradient scaling
                    self.scaler_amp.step(self.optimizer)
                    self.scaler_amp.update()
                else:
                    # Standard precision
                    # Add noise during training for regularization
                    if self.training:
                        noise = torch.randn_like(inputs) * 0.05  # 5% noise
                        noisy_inputs = inputs + noise
                        outputs = self.model(noisy_inputs)
                    else:
                        outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                
                # Track metrics
                if self.device.type == 'cuda':
                    with torch.cuda.stream(stream1):
                        train_loss_tensor.copy_(loss)
                        train_loss += train_loss_tensor.item() * inputs.size(0)
                        predicted = (outputs >= 0.5).float()
                        train_correct += (predicted == targets).sum().item()
                        train_samples += inputs.size(0)
                else:
                    train_loss += loss.item() * inputs.size(0)
                    predicted = (outputs >= 0.5).float()
                    train_correct += (predicted == targets).sum().item()
                    train_samples += inputs.size(0)

                # Calculate the regular BCE loss
                outputs = self.model(inputs)  # or noisy_inputs if you're using noise
                bce_loss = self.loss_fn(outputs, targets)

                # Add L1 regularization
                l1_lambda = 0.01  # Adjust this value as needed
                l1_reg = 0
                for param in self.model.parameters():
                    l1_reg += torch.norm(param, 1)

                # Combined loss
                loss = bce_loss + l1_lambda * l1_reg
            
            # Calculate epoch metrics
            avg_train_loss = train_loss / train_samples
            train_accuracy = train_correct / train_samples
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples = 0
            
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.cuda.stream(stream2):
                        for inputs, targets in val_loader:
                            if use_fp16:
                                with autocast(device_type='cuda', dtype=torch.float16):
                                    outputs = self.model(inputs)
                                    loss = self.loss_fn(outputs, targets)
                            else:
                                outputs = self.model(inputs)
                                loss = self.loss_fn(outputs, targets)
                            
                            val_loss_tensor.copy_(loss)
                            val_loss += val_loss_tensor.item() * inputs.size(0)
                            predicted = (outputs >= 0.5).float()
                            val_correct += (predicted == targets).sum().item()
                            val_samples += inputs.size(0)
                else:
                    for inputs, targets in val_loader:
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, targets)
                        
                        val_loss += loss.item() * inputs.size(0)
                        predicted = (outputs >= 0.5).float()
                        val_correct += (predicted == targets).sum().item()
                        val_samples += inputs.size(0)
            
            # Calculate validation metrics
            avg_val_loss = val_loss / val_samples
            val_accuracy = val_correct / val_samples
            
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
            
            # End timing and calculate epoch time
            if self.device.type == 'cuda':
                # End timing and synchronize CUDA streams
                stream1.synchronize()
                stream2.synchronize()
                end_event.record()
                torch.cuda.synchronize()
                epoch_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
            else:
                epoch_time = time.time() - start_time
            
            # Print progress with timing info
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"time: {epoch_time:.2f}s - "
                  f"loss: {avg_train_loss:.4f} - acc: {train_accuracy:.4f} - "
                  f"val_loss: {avg_val_loss:.4f} - val_acc: {val_accuracy:.4f}")
            
            # Periodically clear CUDA cache
            if self.device.type == 'cuda' and epoch % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Final cleanup
        del best_model_state
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        return history
    
    def predict(self, X):
        """
        Predict if an image is forged (1) or authentic (0) using optimized inference
        
        Args:
            X: Feature vector or batch of feature vectors
            
        Returns:
            Tuple of (predictions, confidences)
        """
        # Check if scaler is fitted
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            print("Warning: StandardScaler not fitted yet. Please train the model first.")
            return np.zeros(X.shape[0] if X.ndim > 1 else 1), np.full(X.shape[0] if X.ndim > 1 else 1, 0.5)
        
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Handle single feature vector or batch
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Normalize features
            X_scaled = self.scaler.transform(X)
            
            # Convert to tensor and move to device
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            
            # Make prediction
            with torch.no_grad():
                confidences = self.model(X_tensor).cpu().numpy()
            
            # Convert to binary output with confidence
            predictions = (confidences >= 0.5).astype(int)
            
            # Return predictions and confidences
            return predictions.flatten(), confidences.flatten()
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return default predictions in case of error
            return np.zeros(X.shape[0]), np.full(X.shape[0], 0.5)
    
    def save(self, filepath):
        """Save the model to disk with optimized format"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Synchronize CUDA operations before moving data
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Move model to CPU for saving to avoid GPU memory issues
            cpu_model = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            
            # Make sure optimizer state is also moved to CPU
            optimizer_state = None
            if self.optimizer:
                optimizer_state = {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in self.optimizer.state_dict().items()
                }
            
            # Save model state, optimizer state, and scaler
            torch.save({
                'model_state_dict': cpu_model,
                'optimizer_state_dict': optimizer_state,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'scaler': self.scaler,
                'input_dim': self.model[0].in_features,  # Store input dimension for loading
                'architecture': 'rdlnn_v1'  # Version tag
            }, filepath)
            
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
        finally:
            # Ensure proper cleanup
            if 'cpu_model' in locals():
                del cpu_model
            if 'optimizer_state' in locals():
                del optimizer_state
            # Synchronize before leaving the function
            if self.device.type == 'cuda':
                torch.cuda.synchronize()

    def load(cls, filepath):
        """Load a trained model from disk
        
        Args:
            filepath: Path to saved model file
            
        Returns:
            Loaded RegressionDLNN model
        """
        try:
            # Load checkpoint with CPU mapping first to avoid OOM
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Check model architecture version
            architecture = checkpoint.get('architecture', 'rdlnn_v1')
            print(f"Loading model architecture: {architecture}")
            
            # Get input dimension from checkpoint
            input_dim = checkpoint.get('input_dim')
            if not input_dim:
                raise ValueError("Input dimension not found in checkpoint")
                
            # Create new model instance
            model = cls(input_dim)
            
            # Load model state
            model.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state if present
            if checkpoint['optimizer_state_dict'] is not None:
                model.optimizer = optim.AdamW(
                    model.model.parameters(),
                    lr=0.001,
                    weight_decay=1e-5
                )
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore scheduler state if it exists
            if checkpoint.get('scheduler_state_dict') is not None:
                model.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    model.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5
                )
                model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load scaler
            model.scaler = checkpoint['scaler']

            # Move model to the correct device after loading
            model.model = model.model.to(model.device)
            
            print(f"Model loaded successfully from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None