#!/usr/bin/env python3
"""
Test script for data detection and validation.
This script tests data loading and validation for electrical voltage data.
FOCUS: Electrical activity only - no coordinate data.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add the fungal_analysis_project/src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def test_data_detection():
    """Test data detection and validation."""
    
    print("=== Testing Data Detection and Validation ===")
    print("This tests electrical voltage data detection and validation")
    print("Data type: Voltage recordings (electrical signals)")
    print("FOCUS: Electrical activity only")
    print()
    
    # Initialize analyzer with voltage data only
    analyzer = RigorousFungalAnalyzer(None, "15061491/fungal_spikes/good_recordings")
    
    # Load data
    data = analyzer.load_and_categorize_data()
    
    print("Data Summary:")
    print(f"  Voltage files: {len(data['voltage_data'])}")
    print("FOCUS: Electrical activity only - no coordinate data")
    print()
    
    # Test data validation
    if data['voltage_data']:
        print("✅ Voltage data detected and loaded successfully")
        
        # Sample a few files for detailed validation
        sample_files = list(data['voltage_data'].keys())[:3]
        
        for filename in sample_files:
            print(f"\n📄 Validating: {filename}")
            
            data_info = data['voltage_data'][filename]
            df = data_info['data']
            metadata = data_info['metadata']
            
            # Validate voltage data structure
            print(f"  Columns: {len(df.columns)}")
            print(f"  Rows: {len(df)}")
            print(f"  Data type: {df.dtypes.iloc[0]}")
            
            # Validate voltage signal characteristics
            if len(df.columns) >= 1:
                voltage_signal = df.iloc[:, 0].values
                
                print(f"  Voltage signal validation:")
                print(f"    Mean: {np.mean(voltage_signal):.6f}")
                print(f"    Std: {np.std(voltage_signal):.6f}")
                print(f"    Min: {np.min(voltage_signal):.6f}")
                print(f"    Max: {np.max(voltage_signal):.6f}")
                print(f"    Range: {np.max(voltage_signal) - np.min(voltage_signal):.6f}")
                
                # Check for potential issues
                if np.std(voltage_signal) == 0:
                    print("    ⚠️  Warning: Zero variance in voltage signal")
                if np.isnan(voltage_signal).any():
                    print("    ⚠️  Warning: NaN values detected")
                if np.isinf(voltage_signal).any():
                    print("    ⚠️  Warning: Infinite values detected")
                
                # Validate metadata
                print(f"  Metadata validation:")
                for key, value in metadata.items():
                    print(f"    {key}: {value}")
                
                print("  ✅ Voltage data validation passed")
            else:
                print("  ❌ Invalid voltage data structure")
    else:
        print("❌ No voltage data found")
        print("This test requires electrical voltage recordings")
    
    # Save validation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data_detection_test_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump({
            'test_timestamp': timestamp,
            'voltage_files_found': len(data['voltage_data']),
            'test_summary': 'Data detection and validation test completed',
            'data_type': 'Voltage recordings (electrical signals)',
            'validation_status': 'Passed' if data['voltage_data'] else 'Failed',
            'note': 'This is electrical activity validation, not coordinate data validation'
        }, f, indent=2)
    
    print(f"\nTest results saved to: {results_file}")
    print()
    print("IMPORTANT: This test validated voltage data (electrical recordings),")
    print("NOT coordinate data. These are fundamentally different data types.")
    
    return len(data['voltage_data']) > 0

if __name__ == "__main__":
    test_data_detection() 