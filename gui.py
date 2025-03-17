#!/usr/bin/env python3
"""
GUI application for Image Forgery Detection System
"""

import sys
import os
import threading
import queue
import time
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Import from our modules
from modules.rdlnn import RegressionDLNN
from modules.preprocessing import preprocess_image
from modules.image_decomposition import perform_wavelet_transform
from modules.feature_extraction import extract_features_from_wavelet
from modules.data_handling import precompute_features
from training.precision import precision_tuned_training
from training.balanced import train_with_balanced_sampling, train_with_oversampling, combined_approach
from tools.combine_features import combine_feature_files

# Configure logging to capture output for the GUI
import logging

# Create a logger
logger = logging.getLogger('forgery_detection')
logger.setLevel(logging.INFO)

# Create a custom handler that will store logs in a queue for the GUI
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

queue_handler = QueueHandler(log_queue)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
queue_handler.setFormatter(formatter)
logger.addHandler(queue_handler)

class ForgeryDetectionGUI:
    def __init__(self, root):
        self.root = root
        root.title("Image Forgery Detection System")
        root.geometry("1024x768")
        root.minsize(800, 600)
        
        # Set theme
        style = ttk.Style()
        if "default" in style.theme_names():
            style.theme_use("default")
        
        self.create_variables()
        self.create_widgets()
        self.arrange_layout()
        self.configure_bindings()
        
        # Start thread to read logs
        self.log_thread = threading.Thread(target=self.process_log_queue, daemon=True)
        self.log_thread.start()
        
        # Set default paths
        self.default_paths()
        
    def create_variables(self):
        """Initialize all variables used in the application"""
        # Paths
        self.input_dir_var = tk.StringVar()
        self.image_path_var = tk.StringVar()
        self.authentic_dir_var = tk.StringVar()
        self.forged_dir_var = tk.StringVar()
        self.features_path_var = tk.StringVar()
        self.model_path_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        
        # Configuration options
        self.mode_var = tk.StringVar(value="single")
        self.training_method_var = tk.StringVar(value="precision")
        self.epochs_var = tk.IntVar(value=25)
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        self.batch_size_var = tk.IntVar(value=32)
        self.threshold_var = tk.DoubleVar(value=0.75)
        self.use_fp16_var = tk.BooleanVar(value=True)
        self.workers_var = tk.IntVar(value=4)
        
        # Progress tracking
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Ready")
        
        # For image preview
        self.current_image = None
        self.current_image_result = None
        
    def create_widgets(self):
        """Create all widgets for the GUI"""
        # Create notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Main tab
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="Main")
        
        # Training tab
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Training")
        
        # Batch Processing tab
        self.batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_frame, text="Batch Processing")
        
        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")
        
        # Log tab
        self.log_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.log_frame, text="Log")
        
        # --- MAIN TAB ---
        # Image preview frame
        self.preview_frame = ttk.LabelFrame(self.main_frame, text="Image Preview")
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="white", width=400, height=300)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Result frame
        self.result_frame = ttk.LabelFrame(self.main_frame, text="Detection Result")
        self.result_label = ttk.Label(self.result_frame, text="No result yet", font=("Arial", 14))
        self.result_confidence = ttk.Label(self.result_frame, text="")
        self.result_label.pack(pady=10)
        self.result_confidence.pack(pady=5)
        
        # Single image detection panel
        self.single_frame = ttk.LabelFrame(self.main_frame, text="Single Image Detection")
        
        ttk.Label(self.single_frame, text="Image:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.single_frame, textvariable=self.image_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.single_frame, text="Browse...", command=self.browse_image).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(self.single_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.single_frame, textvariable=self.model_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.single_frame, text="Browse...", command=self.browse_model).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(self.single_frame, text="Threshold:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        threshold_entry = ttk.Entry(self.single_frame, textvariable=self.threshold_var, width=10)
        threshold_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(self.single_frame, text="Detect", command=self.detect_single_image).grid(row=3, column=0, columnspan=3, pady=10)
        
        # --- TRAINING TAB ---
        self.train_frame = ttk.LabelFrame(self.training_frame, text="Training Configuration")
        
        # Feature files
        ttk.Label(self.train_frame, text="Authentic Images:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.train_frame, textvariable=self.authentic_dir_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.train_frame, text="Browse...", command=self.browse_authentic_dir).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(self.train_frame, text="Forged Images:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.train_frame, textvariable=self.forged_dir_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.train_frame, text="Browse...", command=self.browse_forged_dir).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(self.train_frame, text="Features File:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.train_frame, textvariable=self.features_path_var, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.train_frame, text="Browse...", command=self.browse_features_path).grid(row=2, column=2, padx=5, pady=5)
        
        ttk.Label(self.train_frame, text="Output Model:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.train_frame, textvariable=self.model_path_var, width=50).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(self.train_frame, text="Browse...", command=self.browse_model).grid(row=3, column=2, padx=5, pady=5)
        
        ttk.Label(self.train_frame, text="Output Directory:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.train_frame, textvariable=self.output_dir_var, width=50).grid(row=4, column=1, padx=5, pady=5)
        ttk.Button(self.train_frame, text="Browse...", command=self.browse_output_dir).grid(row=4, column=2, padx=5, pady=5)
        
        # Training parameters
        ttk.Label(self.train_frame, text="Training Method:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        method_combo = ttk.Combobox(self.train_frame, textvariable=self.training_method_var, width=20)
        method_combo['values'] = ('precision', 'balanced', 'oversampling', 'combined')
        method_combo.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(self.train_frame, text="Epochs:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.train_frame, textvariable=self.epochs_var, width=10).grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(self.train_frame, text="Learning Rate:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.train_frame, textvariable=self.learning_rate_var, width=10).grid(row=7, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(self.train_frame, text="Batch Size:").grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.train_frame, textvariable=self.batch_size_var, width=10).grid(row=8, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(self.train_frame, text="Threshold:").grid(row=9, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.train_frame, textvariable=self.threshold_var, width=10).grid(row=9, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(self.train_frame, text="Use FP16:").grid(row=10, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Checkbutton(self.train_frame, variable=self.use_fp16_var).grid(row=10, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Action buttons
        ttk.Button(self.train_frame, text="Precompute Features", command=self.precompute_features).grid(row=11, column=0, padx=5, pady=10)
        ttk.Button(self.train_frame, text="Train Model", command=self.train_model).grid(row=11, column=1, padx=5, pady=10)
        
        # --- BATCH PROCESSING TAB ---
        self.batch_process_frame = ttk.LabelFrame(self.batch_frame, text="Batch Processing")
        
        ttk.Label(self.batch_process_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.batch_process_frame, textvariable=self.input_dir_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.batch_process_frame, text="Browse...", command=self.browse_input_dir).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(self.batch_process_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.batch_process_frame, textvariable=self.model_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.batch_process_frame, text="Browse...", command=self.browse_model).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(self.batch_process_frame, text="Features Path:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.batch_process_frame, textvariable=self.features_path_var, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.batch_process_frame, text="Browse...", command=self.browse_features_path).grid(row=2, column=2, padx=5, pady=5)
        
        ttk.Label(self.batch_process_frame, text="Output Directory:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.batch_process_frame, textvariable=self.output_dir_var, width=50).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(self.batch_process_frame, text="Browse...", command=self.browse_output_dir).grid(row=3, column=2, padx=5, pady=5)
        
        ttk.Label(self.batch_process_frame, text="Threshold:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.batch_process_frame, textvariable=self.threshold_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(self.batch_process_frame, text="Process Batch", command=self.process_batch).grid(row=5, column=0, columnspan=3, pady=10)
        
        # --- SETTINGS TAB ---
        self.settings_panel = ttk.LabelFrame(self.settings_frame, text="Advanced Settings")
        
        ttk.Label(self.settings_panel, text="Worker Threads:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.settings_panel, textvariable=self.workers_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(self.settings_panel, text="Use FP16 Precision:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Checkbutton(self.settings_panel, variable=self.use_fp16_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # About panel
        self.about_panel = ttk.LabelFrame(self.settings_frame, text="About")
        ttk.Label(self.about_panel, text="Image Forgery Detection System", font=("Arial", 12, "bold")).pack(pady=5)
        ttk.Label(self.about_panel, text="Version 1.0.0").pack()
        ttk.Label(self.about_panel, text="© 2025").pack()
        ttk.Label(self.about_panel, text="Uses deep learning with wavelet features to detect image forgeries").pack(pady=5)
        
        # --- LOG TAB ---
        self.log_text = scrolledtext.ScrolledText(self.log_frame, width=80, height=30)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Progress bar and status at the bottom
        self.status_frame = ttk.Frame(self.root)
        self.progress_bar = ttk.Progressbar(self.status_frame, variable=self.progress_var, maximum=100)
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        
    def arrange_layout(self):
        """Arrange the layout for all widgets"""
        # Pack the notebook to fill the window
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main tab layout
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.result_frame.pack(fill=tk.X, padx=10, pady=10)
        self.single_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Training tab layout
        self.train_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Batch processing tab layout
        self.batch_process_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Settings tab layout
        self.settings_panel.pack(fill=tk.X, padx=10, pady=10)
        self.about_panel.pack(fill=tk.X, padx=10, pady=10)
        
        # Status bar at the bottom
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        self.status_label.pack(side=tk.LEFT, padx=5)
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
    
    def configure_bindings(self):
        """Set up event bindings"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def default_paths(self):
        """Set default paths based on current directory"""
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        if not data_dir.exists():
            data_dir.mkdir()
            
        # Create subdirectories
        for subdir in ["features", "models", "results"]:
            path = data_dir / subdir
            if not path.exists():
                path.mkdir()
        
        # Set default paths
        self.features_path_var.set(str(data_dir / "features" / "features.npz"))
        self.model_path_var.set(str(data_dir / "models" / "model.pth"))
        self.output_dir_var.set(str(data_dir / "results"))
    
    def browse_image(self):
        """Browse for an image file"""
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if filename:
            self.image_path_var.set(filename)
            self.load_preview_image(filename)
    
    def browse_input_dir(self):
        """Browse for input directory"""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_dir_var.set(directory)
    
    def browse_authentic_dir(self):
        """Browse for authentic images directory"""
        directory = filedialog.askdirectory(title="Select Authentic Images Directory")
        if directory:
            self.authentic_dir_var.set(directory)
    
    def browse_forged_dir(self):
        """Browse for forged images directory"""
        directory = filedialog.askdirectory(title="Select Forged Images Directory")
        if directory:
            self.forged_dir_var.set(directory)
    
    def browse_features_path(self):
        """Browse for features path"""
        filename = filedialog.asksaveasfilename(
            title="Select Features File",
            defaultextension=".npz",
            filetypes=[("NumPy Compressed", "*.npz")]
        )
        if filename:
            self.features_path_var.set(filename)
    
    def browse_model(self):
        """Browse for model path"""
        filename = filedialog.asksaveasfilename(
            title="Select Model File",
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def load_preview_image(self, image_path):
        """Load and display image in preview area"""
        try:
            # Load the image
            image = Image.open(image_path)
            
            # Resize while maintaining aspect ratio
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            if canvas_width < 10:  # Not fully initialized yet
                canvas_width = 400
                canvas_height = 300
            
            img_width, img_height = image.size
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.current_image = ImageTk.PhotoImage(image)
            
            # Display on canvas
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                canvas_width // 2, canvas_height // 2, 
                image=self.current_image, 
                anchor=tk.CENTER
            )
            
            # Clear previous results
            self.result_label.config(text="No result yet")
            self.result_confidence.config(text="")
            self.current_image_result = None
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def detect_single_image(self):
        """Detect forgery in a single image"""
        image_path = self.image_path_var.get()
        model_path = self.model_path_var.get()
        threshold = self.threshold_var.get()
        
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please select a valid image file.")
            return
        
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file.")
            return
        
        self.status_var.set("Detecting forgery...")
        self.progress_var.set(10)
        
        # Run detection in a separate thread
        threading.Thread(target=self._run_detection, args=(image_path, model_path, threshold), daemon=True).start()
    
    def _run_detection(self, image_path, model_path, threshold):
        """Run detection in a separate thread"""
        try:
            # Load the model
            model = RegressionDLNN.load(model_path)
            
            # Get model's threshold or use provided one
            model_threshold = getattr(model, 'threshold', None)
            if model_threshold is not None and threshold == 0.5:  # Default threshold
                threshold = model_threshold
                self.threshold_var.set(threshold)
            
            self.root.after(0, lambda: self.status_var.set("Preprocessing image..."))
            self.root.after(0, lambda: self.progress_var.set(30))
            
            # Preprocess the image
            ycbcr_tensor = preprocess_image(image_path)
            if ycbcr_tensor is None:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to preprocess image"))
                return
            
            # Add batch dimension
            ycbcr_tensor = ycbcr_tensor.unsqueeze(0)
            
            self.root.after(0, lambda: self.status_var.set("Applying wavelet transform..."))
            self.root.after(0, lambda: self.progress_var.set(50))
            
            # Apply wavelet transform
            pdywt_coeffs = perform_wavelet_transform(ycbcr_tensor[0])
            
            self.root.after(0, lambda: self.status_var.set("Extracting features..."))
            self.root.after(0, lambda: self.progress_var.set(70))
            
            # Extract features
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            feature_vector = extract_features_from_wavelet(
                ycbcr_tensor, 
                [pdywt_coeffs],
                device=device,
                batch_size=1,
                use_fp16=self.use_fp16_var.get()
            )
            
            if feature_vector is None:
                self.root.after(0, lambda: messagebox.showerror("Error", "Failed to extract features"))
                return
            
            self.root.after(0, lambda: self.status_var.set("Making prediction..."))
            self.root.after(0, lambda: self.progress_var.set(90))
            
            # Make prediction
            _, confidences = model.predict(feature_vector.cpu().numpy())
            
            # Apply threshold
            prediction = 1 if confidences[0] >= threshold else 0
            confidence = confidences[0]
            
            # Update UI with result
            result_text = "FORGED" if prediction == 1 else "AUTHENTIC"
            confidence_text = f"Confidence: {confidence:.4f} (Threshold: {threshold:.2f})"
            
            # Change color based on result (red for forged, green for authentic)
            result_color = "red" if prediction == 1 else "green"
            
            self.root.after(0, lambda: self._update_result(result_text, confidence_text, result_color))
            
            # Save result to file if output directory is set
            output_dir = self.output_dir_var.get()
            if output_dir and os.path.isdir(output_dir):
                result_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_result.txt")
                with open(result_file, 'w') as f:
                    f.write(f"Image: {image_path}\n")
                    f.write(f"Result: {result_text}\n")
                    f.write(f"Confidence: {confidence:.4f}\n")
                    f.write(f"Threshold: {threshold:.2f}\n")
                    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                self.log_message(f"Result saved to {result_file}")
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {str(e)}"))
            self.log_message(f"Error during detection: {str(e)}")
        finally:
            self.root.after(0, lambda: self.status_var.set("Ready"))
            self.root.after(0, lambda: self.progress_var.set(0))
    
    def _update_result(self, result_text, confidence_text, result_color):
        """Update result display in the UI"""
        self.result_label.config(text=result_text, foreground=result_color)
        self.result_confidence.config(text=confidence_text)
    
    def precompute_features(self):
        """Precompute features for training"""
        authentic_dir = self.authentic_dir_var.get()
        forged_dir = self.forged_dir_var.get()
        features_path = self.features_path_var.get()
        
        if not authentic_dir or not os.path.isdir(authentic_dir):
            messagebox.showerror("Error", "Please select a valid authentic images directory.")
            return
        
        if not forged_dir or not os.path.isdir(forged_dir):
            messagebox.showerror("Error", "Please select a valid forged images directory.")
            return
        
        if not features_path:
            messagebox.showerror("Error", "Please provide a valid features file path.")
            return
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        
        self.status_var.set("Precomputing features...")
        self.progress_var.set(10)
        
        # Run feature extraction in a separate thread
        threading.Thread(target=self._run_precompute_features, 
                         args=(authentic_dir, forged_dir, features_path), 
                         daemon=True).start()
    
    def _run_precompute_features(self, authentic_dir, forged_dir, features_path):
        """Run feature precomputation in a separate thread"""
        try:
            # Create temp filenames for authentic and forged features
            authentic_features = os.path.splitext(features_path)[0] + "_authentic.npz"
            forged_features = os.path.splitext(features_path)[0] + "_forged.npz"
            
            # Precompute authentic features
            self.root.after(0, lambda: self.status_var.set("Processing authentic images..."))
            self.root.after(0, lambda: self.progress_var.set(20))
            
            precompute_features(
                directory=authentic_dir,
                label=0,  # 0 = authentic
                batch_size=self.batch_size_var.get(),
                num_workers=self.workers_var.get(),
                use_fp16=self.use_fp16_var.get(),
                save_path=authentic_features
            )
            
            # Precompute forged features
            self.root.after(0, lambda: self.status_var.set("Processing forged images..."))
            self.root.after(0, lambda: self.progress_var.set(60))
            
            precompute_features(
                directory=forged_dir,
                label=1,  # 1 = forged
                batch_size=self.batch_size_var.get(),
                num_workers=self.workers_var.get(),
                use_fp16=self.use_fp16_var.get(),
                save_path=forged_features
            )
            
            # Combine features
            self.root.after(0, lambda: self.status_var.set("Combining features..."))
            self.root.after(0, lambda: self.progress_var.set(90))
            
            combine_feature_files(authentic_features, forged_features, features_path)
            
            self.root.after(0, lambda: messagebox.showinfo("Success", "Features precomputed successfully"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Feature extraction failed: {str(e)}"))
            self.log_message(f"Error during feature extraction: {str(e)}")
        finally:
            self.root.after(0, lambda: self.status_var.set("Ready"))
            self.root.after(0, lambda: self.progress_var.set(0))
    
    def train_model(self):
        """Train a model with the selected configuration"""
        features_path = self.features_path_var.get()
        model_path = self.model_path_var.get()
        output_dir = self.output_dir_var.get()
        training_method = self.training_method_var.get()
        
        if not features_path or not os.path.exists(features_path):
            messagebox.showerror("Error", "Please select a valid features file.")
            return
        
        if not model_path:
            messagebox.showerror("Error", "Please provide a valid model output path.")
            return
        
        if not output_dir:
            messagebox.showerror("Error", "Please select a valid output directory.")
            return
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get training parameters
        epochs = self.epochs_var.get()
        learning_rate = self.learning_rate_var.get()
        batch_size = self.batch_size_var.get()
        threshold = self.threshold_var.get()
        use_fp16 = self.use_fp16_var.get()
        
        self.status_var.set(f"Training model using {training_method} method...")
        self.progress_var.set(10)
        
        # Run training in a separate thread
        threading.Thread(target=self._run_training, 
                         args=(features_path, model_path, output_dir, training_method,
                              epochs, learning_rate, batch_size, threshold, use_fp16), 
                         daemon=True).start()
    
    def _run_training(self, features_path, model_path, output_dir, training_method,
                     epochs, learning_rate, batch_size, threshold, use_fp16):
        """Run model training in a separate thread"""
        try:
            self.root.after(0, lambda: self.progress_var.set(20))
            
            if training_method == 'precision':
                precision_tuned_training(
                    features_path,
                    model_path,
                    output_dir,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    threshold=threshold
                )
            elif training_method == 'balanced':
                train_with_balanced_sampling(
                    features_path,
                    model_path,
                    output_dir,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size
                )
            elif training_method == 'oversampling':
                train_with_oversampling(
                    features_path,
                    model_path,
                    output_dir,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size
                )
            elif training_method == 'combined':
                combined_approach(
                    features_path,
                    model_path,
                    output_dir,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    batch_size=batch_size
                )
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Unknown training method: {training_method}"))
                return
            
            self.root.after(0, lambda: messagebox.showinfo("Success", "Model trained successfully"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))
            self.log_message(f"Error during training: {str(e)}")
        finally:
            self.root.after(0, lambda: self.status_var.set("Ready"))
            self.root.after(0, lambda: self.progress_var.set(0))
    
    def process_batch(self):
        """Process a batch of images"""
        input_dir = self.input_dir_var.get()
        model_path = self.model_path_var.get()
        features_path = self.features_path_var.get()
        output_dir = self.output_dir_var.get()
        threshold = self.threshold_var.get()
        
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Please select a valid input directory.")
            return
        
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file.")
            return
        
        if not features_path:
            messagebox.showerror("Error", "Please provide a valid features file path.")
            return
        
        if not output_dir:
            messagebox.showerror("Error", "Please select a valid output directory.")
            return
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        
        self.status_var.set("Processing batch...")
        self.progress_var.set(10)
        
        # Run batch processing in a separate thread
        threading.Thread(target=self._run_batch_processing, 
                         args=(input_dir, model_path, features_path, output_dir, threshold), 
                         daemon=True).start()
    
    def _run_batch_processing(self, input_dir, model_path, features_path, output_dir, threshold):
        """Run batch processing in a separate thread"""
        try:
            # Step 1: Precompute features
            self.root.after(0, lambda: self.status_var.set("Precomputing features..."))
            self.root.after(0, lambda: self.progress_var.set(20))
            
            if not os.path.exists(features_path):
                precompute_features(
                    directory=input_dir,
                    batch_size=self.batch_size_var.get(),
                    num_workers=self.workers_var.get(),
                    use_fp16=self.use_fp16_var.get(),
                    save_path=features_path
                )
            
            # Step 2: Load model
            self.root.after(0, lambda: self.status_var.set("Loading model..."))
            self.root.after(0, lambda: self.progress_var.set(50))
            
            model = RegressionDLNN.load(model_path)
            
            # Get model's threshold or use provided one
            model_threshold = getattr(model, 'threshold', None)
            if model_threshold is not None and threshold == 0.5:  # Default threshold
                threshold = model_threshold
                
            # Step 3: Make predictions
            self.root.after(0, lambda: self.status_var.set("Running detection..."))
            self.root.after(0, lambda: self.progress_var.set(70))
            
            from modules.data_handling import load_and_verify_features
            import numpy as np
            
            # Load features
            features, labels, paths = load_and_verify_features(features_path)
            
            if len(features) == 0:
                self.root.after(0, lambda: messagebox.showerror("Error", "No valid features found for testing."))
                return
            
            # Make predictions
            _, confidences = model.predict(features)
            
            # Apply threshold
            predictions = (confidences >= threshold).astype(int)
            
            # Process results
            authentic = []
            forged = []
            errors = []
            
            # Create detailed results output
            results_file = os.path.join(output_dir, 'detection_results.txt')
            
            with open(results_file, 'w') as f:
                f.write("Image Forgery Detection Results\n")
                f.write("==============================\n\n")
                
                for i, (path, pred, conf) in enumerate(zip(paths, predictions, confidences)):
                    img_name = os.path.basename(path)
                    result_type = 'forged' if pred == 1 else 'authentic'
                    
                    if result_type == 'forged':
                        forged.append(img_name)
                    else:
                        authentic.append(img_name)
                    
                    f.write(f"Image: {img_name}\n")
                    f.write(f"Result: {result_type.upper()}\n")
                    f.write(f"Confidence: {conf:.4f}\n")
                    
                    if i < len(labels) and len(labels) > 0:
                        true_label = 'forged' if labels[i] == 1 else 'authentic'
                        f.write(f"True label: {true_label.upper()}\n")
                        if pred == labels[i]:
                            f.write("Prediction: CORRECT\n")
                        else:
                            f.write("Prediction: INCORRECT\n")
                    
                    f.write("\n")
            
            # Calculate stats if we have labels
            if len(labels) > 0:
                accuracy = np.mean(predictions == labels)
                
                # Compute confusion matrix
                tp = np.sum((predictions == 1) & (labels == 1))
                tn = np.sum((predictions == 0) & (labels == 0))
                fp = np.sum((predictions == 1) & (labels == 0))
                fn = np.sum((predictions == 0) & (labels == 1))
                
                # Calculate precision, recall, and F1 score
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                result_summary = (
                    f"Processed {len(predictions)} images\n"
                    f"Authentic: {len(authentic)}, Forged: {len(forged)}\n"
                    f"Accuracy: {accuracy:.4f}\n"
                    f"Precision: {precision:.4f}\n"
                    f"Recall: {recall:.4f}\n"
                    f"F1 Score: {f1:.4f}\n"
                )
            else:
                result_summary = (
                    f"Processed {len(predictions)} images\n"
                    f"Authentic: {len(authentic)}, Forged: {len(forged)}\n"
                )
            
            self.root.after(0, lambda: messagebox.showinfo("Batch Processing Complete", result_summary))
            self.log_message(f"Results saved to {results_file}")
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Batch processing failed: {str(e)}"))
            self.log_message(f"Error during batch processing: {str(e)}")
        finally:
            self.root.after(0, lambda: self.status_var.set("Ready"))
            self.root.after(0, lambda: self.progress_var.set(0))
    
    def process_log_queue(self):
        """Process the log queue and update the log text widget"""
        try:
            while True:
                record = log_queue.get_nowait()
                self.log_message(self.formatter.format(record))
                log_queue.task_done()
        except queue.Empty:
            self.root.after(100, self.process_log_queue)
    
    def log_message(self, message):
        """Add a message to the log text widget"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # Auto-scroll to the end
    
    def on_close(self):
        """Handle window close event"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()

if __name__ == "__main__":
    # Set up root window
    root = tk.Tk()
    app = ForgeryDetectionGUI(root)
    
    # Start application
    root.mainloop()