#!/usr/bin/env python3
"""
🧪 Test Real Data Loading - Phase 3
====================================

This script tests the ability to load and process real CSV data files
from the fungal electrical monitoring system.

Author: Environmental Sensing Research Team
Date: August 12, 2025
"""

import sys
import pandas as pd
from pathlib import Path
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_real_data_loading():
    """Test loading real CSV data files."""
    print("🧪 Testing Real Data Loading for Phase 3")
    print("=" * 50)
    
    # Test data sources
    test_files = [
        "../../DATA/raw/15061491/Activity_time_part1.csv",
        "../../DATA/raw/15061491/Ch1-2.csv",
        "../../DATA/raw/15061491/GL1.csv"
    ]
    
    for file_path in test_files:
        print(f"\n📊 Testing: {file_path}")
        
        try:
            # Check if file exists
            if not Path(file_path).exists():
                print(f"❌ File not found: {file_path}")
                continue
            
            # Load CSV data
            data = pd.read_csv(file_path)
            print(f"✅ Loaded successfully: {len(data)} rows, {len(data.columns)} columns")
            
            # Show column names
            print(f"📋 Columns: {list(data.columns[:10])}{'...' if len(data.columns) > 10 else ''}")
            
            # Show data types
            print(f"🔧 Data types: {dict(data.dtypes)}")
            
            # Show first few rows
            print(f"📈 First 3 rows:")
            print(data.head(3).to_string())
            
            # Check for missing values
            missing = data.isnull().sum()
            if missing.sum() > 0:
                print(f"⚠️  Missing values: {missing[missing > 0].to_dict()}")
            else:
                print("✅ No missing values found")
                
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    print("\n" + "=" * 50)
    print("🧪 Real Data Loading Test Complete")

def test_data_processing():
    """Test basic data processing capabilities."""
    print("\n🔧 Testing Data Processing Capabilities")
    print("=" * 50)
    
    try:
        # Load a sample file
        file_path = "../../DATA/raw/15061491/Activity_time_part1.csv"
        
        if Path(file_path).exists():
            data = pd.read_csv(file_path)
            
            # Test data cleaning
            print(f"📊 Original data: {len(data)} rows")
            
            # Remove rows with all NaN values
            data_clean = data.dropna(how='all')
            print(f"🧹 After cleaning: {len(data_clean)} rows")
            
            # Check for numeric columns
            numeric_cols = data_clean.select_dtypes(include=['number']).columns
            print(f"🔢 Numeric columns: {list(numeric_cols)}")
            
            # Basic statistics
            if len(numeric_cols) > 0:
                print(f"📈 Basic statistics:")
                print(data_clean[numeric_cols].describe())
            
        else:
            print(f"❌ Test file not found: {file_path}")
            
    except Exception as e:
        print(f"❌ Error in data processing test: {e}")

if __name__ == "__main__":
    test_real_data_loading()
    test_data_processing() 