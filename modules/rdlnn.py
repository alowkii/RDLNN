import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import gc
from torch.cuda.amp import autocast, GradScaler

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
        
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This implementation requires a CUDA-capable GPU.")
        
        # Set device to CUDA with specific device selection
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        
        # Define the model architecture with optimized layer sizes and batch normalization
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layers with batch normalization
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Output layer with sigmoid activation for binary classification
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Initialize weights using Kaiming initialization
        self._init_weights()
        
        # Define loss function and optimizer with weight decay
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # For feature normalization
        self.scaler = StandardScaler()
        
        # Initialize gradient scaler for mixed precision training
        self.scaler_amp = GradScaler()
        
        # Print CUDA information
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        
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
    
    def train_model(self, X, y, validation_split=0.2, epochs=50, batch_size=32, early_stopping=10):
        """
        Train the RDLNN model with optimized training loop
        
        Args:
            X: Feature vectors
            y: Labels (0: authentic, 1: forged)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping: Number of epochs with no improvement after which training will stop
            
        Returns:
            Training history (dictionary with loss and accuracy metrics)
        """
        # Set appropriate CUDA device
        torch.cuda.set_device(0)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors directly on CUDA
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)
        
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
        
        # Create data loaders with CUDA optimizations
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=2  # Prefetch 2 batches per worker
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size * 2,  # Larger batch size for validation
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=2
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
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Training loop with CUDA benchmarking and mixed precision
        torch.backends.cudnn.benchmark = True
        
        # Early stopping setup
        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        
        # Create CUDA streams for parallelize data transfers and computations
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        # Preallocate tensors for metrics to avoid recreating them
        train_loss_tensor = torch.zeros(1, device=self.device)
        val_loss_tensor = torch.zeros(1, device=self.device)
        
        for epoch in range(epochs):
            start_event.record()
            
            # Training metrics
            train_loss = 0.0
            train_correct = 0
            train_samples = 0
            
            # Training loop with mixed precision
            for inputs, targets in train_loader:
                # Zero the gradients
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Using mixed precision for forward and backward pass
                with autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Calculate loss
                    loss = self.loss_fn(outputs, targets)
                
                # Scale loss and do backward pass
                self.scaler_amp.scale(loss).backward()
                
                # Update weights with gradient scaling
                self.scaler_amp.step(self.optimizer)
                self.scaler_amp.update()
                
                # Track metrics - use different stream to avoid blocking computation
                with torch.cuda.stream(stream1):
                    train_loss_tensor.copy_(loss)
                    train_loss += train_loss_tensor.item() * inputs.size(0)
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
            
            with torch.no_grad(), torch.cuda.stream(stream2):
                for inputs, targets in val_loader:
                    # Use mixed precision for inference
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, targets)
                    
                    val_loss_tensor.copy_(loss)
                    val_loss += val_loss_tensor.item() * inputs.size(0)
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
            
            # End timing and synchronize CUDA streams
            stream1.synchronize()
            stream2.synchronize()
            end_event.record()
            torch.cuda.synchronize()
            epoch_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
            
            # Print progress with timing info
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"time: {epoch_time:.2f}s - "
                  f"loss: {avg_train_loss:.4f} - acc: {train_accuracy:.4f} - "
                  f"val_loss: {avg_val_loss:.4f} - val_acc: {val_accuracy:.4f}")
            
            # Periodically clear CUDA cache
            if epoch % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Final cleanup
        del best_model_state
        torch.cuda.empty_cache()
        gc.collect()
        
        return history
    
    def predict(self, X):
        """
        Predict if an image is forged (1) or authentic (0) using optimized inference
        
        Args:
            X: Feature vector
            
        Returns:
            Prediction (0: authentic, 1: forged) and confidence level (0-1)
        """
        # Check if scaler is fitted
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            print("Warning: StandardScaler not fitted yet. Please train the model first.")
            return 0, 0.5  # Default prediction
        
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Normalize features
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            
            # Convert to tensor and move to device
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            
            # Make prediction with mixed precision
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                confidence = self.model(X_tensor).item()
            
            # Convert to binary output with confidence
            prediction = 1 if confidence >= 0.5 else 0
            
            # Return confidence level appropriate for the prediction
            return prediction, confidence if prediction == 1 else (1 - confidence)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return 0, 0.5  # Default prediction in case of error
    
    def save_model(self, filepath):
        """Save the model to disk with optimized format"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Move model to CPU for saving to avoid GPU memory issues
        cpu_model = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        # Save model state, optimizer state, and scaler
        torch.save({
            'model_state_dict': cpu_model,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler': self.scaler,
            'input_dim': self.model[0].in_features,  # Store input dimension for loading
            'architecture': 'rdlnn_v2'  # Version tag
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk with optimized loading"""
        try:
            # Load checkpoint with CPU mapping first to avoid OOM
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Check model architecture version
            architecture = checkpoint.get('architecture', 'rdlnn_v1')
            print(f"Loading model architecture: {architecture}")
            
            # Check if the model architecture matches
            input_dim = checkpoint.get('input_dim')
            if input_dim and input_dim != self.model[0].in_features:
                print(f"Rebuilding model with input dimension {input_dim}")
                # Rebuild model with correct dimensions and batch normalization
                self.model = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                ).to(self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state - use AdamW to match the save operation
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=0.001,
                weight_decay=1e-5
            )
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Restore scheduler state if it exists
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True
                )
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load scaler
            self.scaler = checkpoint['scaler']

            # Move model to the correct device after loading
            self.model = self.model.to(self.device)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            print(f"Model loaded successfully from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False