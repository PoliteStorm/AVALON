#!/usr/bin/env python3
"""
Setup script for Fungal Electrical Activity Monitoring System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements/requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ["output", "logs", "temp"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import numpy as np
        import pandas as pd
        import scipy
        import matplotlib.pyplot as plt
        import numba
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def main():
    print("Setting up Fungal Electrical Activity Monitoring System...")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        print("Setup failed. Please check the error messages above.")
        return
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_imports():
        print("Setup failed. Please check the error messages above.")
        return
    
    print("=" * 60)
    print("✓ Setup completed successfully!")
    print("\nYou can now run the system using:")
    print("  python scripts/ultra_optimized_fungal_monitoring_simple.py data/")
    print("  python scripts/fungal_electrical_monitoring_with_wave_transform.py data/")

if __name__ == "__main__":
    main() 