#!/usr/bin/env python3
"""
Launcher for the Image Forgery Detection GUI
"""

import os
import sys
import subprocess
import tkinter as tk
import tkinter.messagebox as messagebox

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import torch
        import numpy
        import PIL
        return True
    except ImportError as e:
        return False

def install_dependencies():
    """Install missing dependencies"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        return False

def check_folders():
    """Check if required folders exist and create them if not"""
    required_dirs = [
        "data",
        "data/features",
        "data/models",
        "data/results"
    ]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except OSError:
                return False
    
    return True

def run_gui():
    """Run the GUI application"""
    try:
        import gui
        root = tk.Tk()
        app = gui.ForgeryDetectionGUI(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Error starting the GUI: {str(e)}")
        raise

if __name__ == "__main__":
    # Check and create required folders
    if not check_folders():
        print("Error: Could not create required folders. Please check permissions.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("Some dependencies are missing. Attempting to install them...")
        if not install_dependencies():
            print("Error: Failed to install dependencies.")
            sys.exit(1)
        else:
            print("Dependencies installed successfully.")
    
    # Run the GUI
    run_gui()