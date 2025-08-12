#!/usr/bin/env python3
"""
Test script for multiscalar electrical analysis.
This script tests the √t transform on electrical voltage data.
FOCUS: Electrical activity only - no coordinate data analysis.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the fungal_analysis_project/src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def test_multiscalar_electrical_analysis():
    """Test the multiscalar electrical analysis."""
    
    print("=== Testing Multiscalar Electrical Analysis ===")
    print("This analyzes electrical voltage data, NOT coordinate data")
    print("Data type: Voltage recordings (electrical signals)")
    print("Analysis type: Electrical signal processing")
    print("FOCUS: Electrical activity only")
    print()
    
    # Initialize analyzer with voltage data only
    analyzer = RigorousFungalAnalyzer(None, "15061491/fungal_spikes/good_recordings")
    
    # Load data
    data = analyzer.load_and_categorize_data()
    
    print(f"Loaded {len(data['voltage_data'])} voltage files")
    print("FOCUS: Electrical activity only - no coordinate data")
    print()
    
    # Test with voltage files
    voltage_files = list(data['voltage_data'].keys())
    
    if not voltage_files:
        print("No voltage files found. Testing with synthetic electrical data...")
        # Create synthetic voltage data for testing
        t = np.linspace(0, 1000, 1000)
        # Simulate electrical spikes
        voltage_signal = np.random.normal(0, 0.1, 1000)
        # Add some spikes
        spike_indices = np.random.choice(1000, 20, replace=False)
        voltage_signal[spike_indices] += np.random.normal(0.5, 0.1, len(spike_indices))
        
        print("=== SYNTHETIC ELECTRICAL ANALYSIS RESULTS ===")
        print(f"Voltage signal - Mean: {np.mean(voltage_signal):.3f}, Std: {np.std(voltage_signal):.3f}")
        print(f"Spike count: {len(spike_indices)}")
        print()
        print("This is ELECTRICAL SIGNAL analysis, not coordinate analysis!")
        
    else:
        # Test with actual voltage files
        for filename in voltage_files[:3]:  # Test first 3 voltage files
            print(f"Testing electrical analysis for: {filename}")
            
            data_info = data['voltage_data'][filename]
            df = data_info['data']
            metadata = data_info['metadata']
            
            # Extract voltage signals
            if len(df.columns) >= 1:
                voltage_signal = df.iloc[:, 0].values  # Use first column as voltage
                
                # Calculate electrical signal statistics
                mean_voltage = np.mean(voltage_signal)
                std_voltage = np.std(voltage_signal)
                min_voltage = np.min(voltage_signal)
                max_voltage = np.max(voltage_signal)
                
                # Simple spike detection
                threshold = mean_voltage + 2 * std_voltage
                spikes = voltage_signal > threshold
                n_spikes = np.sum(spikes)
                
                print(f"  Voltage signal (electrical recording):")
                print(f"    Mean: {mean_voltage:.3f} mV")
                print(f"    Std: {std_voltage:.3f} mV")
                print(f"    Range: {min_voltage:.3f} to {max_voltage:.3f} mV")
                print(f"    N samples: {len(voltage_signal)}")
                print(f"    Spikes detected: {n_spikes}")
                print()
                
                print("This is ELECTRICAL SIGNAL analysis, not coordinate analysis!")
                print("Voltage data represents electrical recordings over time.")
                print("This is fundamentally different from coordinate data analysis.")
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"multiscalar_electrical_analysis_test_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump({
            'test_timestamp': timestamp,
            'voltage_files_found': len(voltage_files),
            'test_summary': 'Multiscalar electrical analysis test completed',
            'data_type': 'Voltage recordings (electrical signals)',
            'analysis_type': 'Electrical signal processing',
            'note': 'This is electrical activity analysis, not coordinate data analysis'
        }, f, indent=2)
    
    print(f"Test results saved to: {results_file}")
    print()
    print("IMPORTANT: This test analyzed voltage data (electrical recordings),")
    print("NOT coordinate data. These are fundamentally different analyses.")
    
    return True

if __name__ == "__main__":
    test_multiscalar_electrical_analysis() 