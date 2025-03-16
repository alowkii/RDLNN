import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import os
import gc
from torch.amp import autocast, GradScaler
import time
import matplotlib.pyplot as plt

class RegressionDLNN:
    """
    Regression Deep Learning Neural Network (RDLNN) for image forgery detection
    Implemented with PyTorch and CUDA support with improved handling of imbalanced data
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
        
        # Define the model architecture with improved layers
        self.model = nn.Sequential(
            # Input layer with larger size
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),  # Use LeakyReLU for better gradient flow
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
            
            # Output layer - remove sigmoid activation
            nn.Linear(64, 1)
        ).to(self.device)
        
        # Initialize weights using Xavier initialization
        self._init_weights()
        
        # Define loss function - will be updated with class weights
        self.loss_fn = nn.BCEWithLogitsLoss()
        
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
    
    def fit(self, X, y, epochs=50, learning_rate=0.001, batch_size=32, validation_split=0.2, 
            early_stopping=10, use_fp16=False, force_class_balance=False):
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
        
        print(f"Training with learning rate: {learning_rate:.6f}")
        
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
        print(f"Class weights: {class_weights}")
        
        # Use MUCH higher weight for minority class (forged)
        # This is crucial to fix the "all one class" prediction problem
        if force_class_balance:
            pos_weight_value = 5.0  # Force very high weight for positive class
            print(f"Forcing positive class weight to {pos_weight_value}")
        else:
            pos_weight_value = class_weights[1]/class_weights[0]
            
        # Update loss function to use weights
        weight_tensor = torch.tensor([pos_weight_value], device=self.device)
        print(f"Using positive weight: {pos_weight_value:.4f}")
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=weight_tensor
        )
            
        # Split data into training and validation sets using sklearn to ensure balance
        X_train, X_val, y_train, y_val = train_test_split(
            X_np, y_np, test_size=validation_split, random_state=42, stratify=y_np
        )
        
        # Verify balance after split
        print(f"Training class distribution: {np.bincount(y_train.flatten().astype(int))}")
        print(f"Validation class distribution: {np.bincount(y_val.flatten().astype(int))}")
        
        # Normalize features - fit only on training data
        self.scaler = StandardScaler().fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
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
        
        # Handle class imbalance using weighted sampling
        if np.bincount(y_train.flatten().astype(int))[0] != np.bincount(y_train.flatten().astype(int))[1]:
            print("Using weighted sampler for balanced batches")
            weights = 1.0 / np.bincount(y_train.flatten().astype(int))
            sample_weights = torch.tensor([weights[int(t)] for t in y_train.flatten()], dtype=torch.float32)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers
            )
        else:
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
            'val_acc': [],
            'learning_rates': []
        }
        
        # Early stopping setup
        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Training metrics
            train_loss = 0.0
            train_correct = 0
            train_samples = 0
            
            # Training loop with mixed precision if requested
            for inputs, targets in train_loader:
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
                    with autocast(device_type='cuda', dtype=torch.float16):
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
                    
                    # Explicitly calculate sigmoid to get probabilities
                    probabilities = torch.sigmoid(outputs)
                    predicted = (probabilities >= 0.5).float()
                    
                    val_correct += (predicted == targets).sum().item()
                    val_samples += inputs.size(0)
                    
                    # Store predictions and targets for confusion matrix
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets_list.extend(targets.cpu().numpy())
                    val_outputs_list.extend(probabilities.cpu().numpy())
            
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
                print(f"✓ New best validation loss: {best_val_loss:.6f}")
            else:
                early_stopping_counter += 1
                print(f"! Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping}")
                
                if early_stopping_counter >= early_stopping:
                    print("\n" + "="*80)
                    print(f"EARLY STOPPING TRIGGERED (after {epoch+1} epochs)")
                    print("="*80)
                    
                    # Check if model is just predicting one class
                    unique_preds, counts = np.unique(val_preds, return_counts=True)
                    pred_dist = {int(k): int(v) for k, v in zip(unique_preds, counts)}
                    
                    if len(pred_dist) == 1:
                        only_class = list(pred_dist.keys())[0]
                        print(f"WARNING: Model is predicting ONLY class {only_class} for ALL samples!")
                        print("This suggests the model hasn't learned to differentiate between classes.")
                        print("Suggestions:")
                        print("1. Increase learning rate (try 0.001 or 0.005)")
                        print("2. Train for more epochs")
                        print("3. Check feature quality/usefulness")
                        print("4. Consider using a different model architecture")
                    else:
                        print(f"Final prediction distribution: {pred_dist}")
                    
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
            
            # Print progress with detailed metrics in a cleaner format
            print("\n" + "="*80)
            print(f"EPOCH {epoch+1}/{epochs} (Time: {epoch_time:.2f}s, LR: {current_lr:.6f})")
            print("-"*80)
            print(f"Training   | Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.2%}")
            print(f"Validation | Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2%}")
            print("-"*80)
            print(f"Confusion Matrix:")
            print(f"                | Predicted Positive | Predicted Negative |")
            print(f"Actual Positive | {tp:^18} | {fn:^18} |")
            print(f"Actual Negative | {fp:^18} | {tn:^18} |")
            print("-"*80)
            print(f"Metrics | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            
            # Show prediction distribution
            unique_preds, counts = np.unique(val_preds, return_counts=True)
            pred_dist = {int(k): int(v) for k, v in zip(unique_preds, counts)}
            
            # Format nicely
            if len(pred_dist) == 1:
                only_class = list(pred_dist.keys())[0]
                print(f"WARNING: Model is predicting ONLY class {only_class} for ALL validation samples!")
            else:
                print(f"Prediction distribution: {pred_dist}")
            
            # Calculate and print probability distribution
            val_outputs_array = np.array(val_outputs_list).flatten()
            if len(val_outputs_array) > 0:
                # Calculate percentiles for better understanding
                p10 = np.percentile(val_outputs_array, 10)
                p25 = np.percentile(val_outputs_array, 25)
                p50 = np.percentile(val_outputs_array, 50)
                p75 = np.percentile(val_outputs_array, 75)
                p90 = np.percentile(val_outputs_array, 90)
                
                print("-"*80)
                print("Probability Distribution:")
                print(f"Range: [{val_outputs_array.min():.4f} - {val_outputs_array.max():.4f}] | Mean: {val_outputs_array.mean():.4f} | Std: {val_outputs_array.std():.4f}")
                print(f"Percentiles: 10%: {p10:.4f} | 25%: {p25:.4f} | 50%: {p50:.4f} | 75%: {p75:.4f} | 90%: {p90:.4f}")
                
                # Analyze confidence levels
                high_conf = np.sum((val_outputs_array > 0.9) | (val_outputs_array < 0.1))
                medium_conf = np.sum((val_outputs_array > 0.7) & (val_outputs_array < 0.9) | 
                                    (val_outputs_array > 0.1) & (val_outputs_array < 0.3))
                uncertain = np.sum((val_outputs_array >= 0.3) & (val_outputs_array <= 0.7))
                
                print(f"Confidence levels:")
                print(f"  High (>0.9 or <0.1): {high_conf} samples ({high_conf/len(val_outputs_array)*100:.1f}%)")
                print(f"  Medium: {medium_conf} samples ({medium_conf/len(val_outputs_array)*100:.1f}%)")
                print(f"  Uncertain (0.3-0.7): {uncertain} samples ({uncertain/len(val_outputs_array)*100:.1f}%)")
                print("="*80)
            
            # Periodically clear CUDA cache
            if self.device.type == 'cuda' and epoch % 5 == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
        
        # Final cleanup
        if best_model_state:
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
        
        # Generate model diagnostics
        self.plot_diagnostic_curves(val_targets_list, val_outputs_list, history)
        
        return history
    
    def plot_diagnostic_curves(self, y_true, y_pred_prob, history):
        """Generate ROC and Precision-Recall curves for model diagnostics"""
        try:
            import sklearn.metrics as metrics
            
            # Only proceed if we have predictions from both classes
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                print("Cannot generate diagnostic curves: not enough classes in validation set")
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
            plt.savefig('results/model_diagnostics.png')
            print(f"Model diagnostic curves saved to results/model_diagnostics.png")
            
        except ImportError:
            print("Scikit-learn is required for diagnostic curves")
        except Exception as e:
            print(f"Error generating diagnostic curves: {e}")
    
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
            # Get raw logits from model
            logits = self.model(X_tensor)
            # Apply sigmoid to get probabilities
            confidences = torch.sigmoid(logits).cpu().numpy()
        
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