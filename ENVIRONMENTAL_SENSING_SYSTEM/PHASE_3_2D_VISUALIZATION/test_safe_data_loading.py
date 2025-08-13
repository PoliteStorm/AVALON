#!/usr/bin/env python3
"""
🧪 Safe Data Loading Test - Phase 3
====================================

This script safely tests data loading without memory issues by:
1. Reading only file headers first
2. Sampling small portions of data
3. Using chunked reading for large files

Author: Environmental Sensing Research Team
Date: August 12, 2025
"""

import sys
import pandas as pd
from pathlib import Path
import json

def test_file_access():
    """Test basic file access without loading full content."""
    print("🧪 Testing Safe File Access for Phase 3")
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
            
            # Get file size
            file_size = Path(file_path).stat().st_size
            print(f"📏 File size: {file_size / (1024*1024):.2f} MB")
            
            # Read only the header (first few lines)
            with open(file_path, 'r') as f:
                header_lines = []
                for i, line in enumerate(f):
                    if i < 5:  # Read first 5 lines
                        header_lines.append(line.strip())
                    else:
                        break
            
            print(f"📋 First 5 lines:")
            for i, line in enumerate(header_lines):
                print(f"  {i+1}: {line[:100]}{'...' if len(line) > 100 else ''}")
            
            # Try to read just column names using pandas
            try:
                # Read only first row to get column names
                columns = pd.read_csv(file_path, nrows=0).columns.tolist()
                print(f"✅ Column names: {columns[:10]}{'...' if len(columns) > 10 else ''}")
                print(f"📊 Total columns: {len(columns)}")
            except Exception as e:
                print(f"⚠️  Could not read columns with pandas: {e}")
            
        except Exception as e:
            print(f"❌ Error accessing {file_path}: {e}")
    
    print("\n" + "=" * 50)
    print("🧪 Safe File Access Test Complete")

def test_chunked_loading():
    """Test loading data in chunks to avoid memory issues."""
    print("\n🔧 Testing Chunked Data Loading")
    print("=" * 50)
    
    try:
        # Test with a smaller file first
        file_path = "../../DATA/raw/15061491/Ch1-2.csv"
        
        if Path(file_path).exists():
            print(f"📊 Testing chunked loading: {file_path}")
            
            # Read in chunks of 1000 rows
            chunk_size = 1000
            total_rows = 0
            columns = None
            
            for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                if chunk_num == 0:
                    columns = chunk.columns.tolist()
                    print(f"📋 Columns: {columns[:10]}{'...' if len(columns) > 10 else ''}")
                
                total_rows += len(chunk)
                print(f"📈 Chunk {chunk_num + 1}: {len(chunk)} rows (Total: {total_rows})")
                
                # Only process first few chunks to avoid memory issues
                if chunk_num >= 2:
                    print(f"⏹️  Stopping after 3 chunks to avoid memory issues")
                    break
            
            print(f"✅ Successfully loaded {total_rows} rows in chunks")
            
        else:
            print(f"❌ Test file not found: {file_path}")
            
    except Exception as e:
        print(f"❌ Error in chunked loading test: {e}")

def test_small_sample():
    """Test loading a small sample of data."""
    print("\n🎯 Testing Small Sample Loading")
    print("=" * 50)
    
    try:
        # Try to load just 100 rows from a file
        file_path = "../../DATA/raw/15061491/Ch1-2.csv"
        
        if Path(file_path).exists():
            print(f"📊 Loading small sample: {file_path}")
            
            # Load only first 100 rows
            sample_data = pd.read_csv(file_path, nrows=100)
            print(f"✅ Loaded sample: {len(sample_data)} rows, {len(sample_data.columns)} columns")
            
            # Show basic info
            print(f"📋 Columns: {list(sample_data.columns)}")
            print(f"🔧 Data types: {dict(sample_data.dtypes)}")
            
            # Show first few rows
            print(f"📈 First 3 rows:")
            print(sample_data.head(3).to_string())
            
            # Check for missing values
            missing = sample_data.isnull().sum()
            if missing.sum() > 0:
                print(f"⚠️  Missing values: {missing[missing > 0].to_dict()}")
            else:
                print("✅ No missing values in sample")
                
        else:
            print(f"❌ Test file not found: {file_path}")
            
    except Exception as e:
        print(f"❌ Error in small sample test: {e}")

if __name__ == "__main__":
    test_file_access()
    test_chunked_loading()
    test_small_sample() 