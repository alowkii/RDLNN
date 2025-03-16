import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

class RegressionDLNN(nn.Module):
    """
    Regression Deep Learning Neural Network (RDLNN) for image forgery detection
    Implemented with PyTorch and CUDA support only
    """
    
    def __init__(self, input_dim):
        """
        Initialize the RDLNN model
        
        Args:
            input_dim: Number of input features
        """
        super(RegressionDLNN, self).__init__()
        
        # Ensure CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This implementation requires a CUDA-capable GPU.")
        
        # Set device to CUDA
        self.device = torch.device("cuda")
        
        # Define the model architecture
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layers
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            
            # Output layer with sigmoid activation for binary classification
            # (0: authentic image, 1: forged image)
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Define loss function and optimizer
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
        # For feature normalization
        self.scaler = StandardScaler()
        
        # Print CUDA information
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x)
    
    def train_model(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """
        Train the RDLNN model
        
        Args:
            X: Feature vectors
            y: Labels (0: authentic, 1: forged)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history (dictionary with loss and accuracy metrics)
        """
        # Set appropriate CUDA device
        torch.cuda.set_device(0)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors directly on CUDA
        X_tensor = torch.cuda.FloatTensor(X_scaled)
        y_tensor = torch.cuda.FloatTensor(y).reshape(-1, 1)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into training and validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders with CUDA optimizations
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=4,  # Use multiple workers for data loading
            persistent_workers=True  # Keep workers alive between iterations
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True
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
        
        # CUDA event for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Training loop with CUDA benchmarking
        torch.backends.cudnn.benchmark = True
        
        for epoch in range(epochs):
            start_event.record()
            
            # Training metrics
            train_loss = 0.0
            train_correct = 0
            train_samples = 0
            
            # Training loop
            for inputs, targets in train_loader:
                # No need to move tensors to device as they're already on CUDA
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.loss_fn(outputs, targets)
                
                # Backward pass and optimize
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
            
            with torch.no_grad():
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
            
            # Back to training mode
            self.model.train()
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_accuracy)
            
            # End timing
            end_event.record()
            torch.cuda.synchronize()
            epoch_time = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
            
            # Print progress with timing info
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"time: {epoch_time:.2f}s - "
                  f"loss: {avg_train_loss:.4f} - acc: {train_accuracy:.4f} - "
                  f"val_loss: {avg_val_loss:.4f} - val_acc: {val_accuracy:.4f}")
            
            # Periodically clear CUDA cache to prevent memory leaks
            if epoch % 10 == 0:
                torch.cuda.empty_cache()
        
        return history
    
    def predict(self, X):
        """
        Predict if an image is forged (1) or authentic (0)
        
        Args:
            X: Feature vector
            
        Returns:
            Prediction (0: authentic, 1: forged) and confidence level (0-1)
        """
        # Check if scaler is fitted before using it
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            print("Warning: StandardScaler not fitted yet. Please train the model first.")
            print("Returning default prediction (authentic) with 50% confidence.")
            return 0, 0.5  # Default to authentic with 50% confidence
        
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # Normalize features
            X_scaled = self.scaler.transform(X.reshape(1, -1))
            
            # Convert to tensor and move to device
            X_tensor = torch.cuda.FloatTensor(X_scaled)
            
            # Make prediction
            with torch.no_grad():
                confidence = self.model(X_tensor).item()
            
            # Convert to binary output with confidence
            prediction = 1 if confidence >= 0.5 else 0
            
            return prediction, confidence if prediction == 1 else (1 - confidence)
        except Exception as e:
            print(f"Error during prediction: {e}")
            print("Returning default prediction (authentic) with 50% confidence.")
            return 0, 0.5  # Default prediction in case of error
    
    def save_model(self, filepath):
        """Save the model to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state, optimizer state, and scaler
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'input_dim': self.model[0].in_features  # Store input dimension for loading
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from disk"""
        try:
            # Load checkpoint directly to CUDA
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Check if the model architecture matches
            input_dim = checkpoint.get('input_dim')
            if input_dim and input_dim != self.model[0].in_features:
                print(f"Warning: Loaded model has different input dimension ({input_dim}) "
                      f"than current model ({self.model[0].in_features})")
                # Rebuild model with correct dimensions
                self.model = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                ).to(self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer = optim.Adam(self.model.parameters())
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scaler
            self.scaler = checkpoint['scaler']
            
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False