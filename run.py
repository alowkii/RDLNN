#!/usr/bin/env python3
"""
Image Forgery Detection System - Main Entry Point
"""

import os
import sys
import argparse

# Ensure all modules can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create module files if they don't exist
def create_module_files():
    """Create the necessary module files"""
    os.makedirs('modules', exist_ok=True)
    
    # Copy the modules from original code
    modules_to_preserve = [
        'preprocessing.py',
        'imageDecomposition.py',
        'featureExtraction.py'
    ]
    
    # Import improved modules
    from improved_main import main

if __name__ == "__main__":
    # Ensure module files exist
    create_module_files()
    
    # Import the main function from improved main
    from improved_main import main
    
    # Run the main function
    main()