#!/usr/bin/env python3
"""
Test script for multiscalar electrical spiking analysis based on Adamatzky's research.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def test_multiscalar_analysis():
    """Test the multiscalar electrical spiking analysis."""
    
    print("=== Testing Multiscalar Electrical Spiking Analysis ===")
    print("Based on Adamatzky's Research: Multiscalar Electrical Spiking in Schizophyllum commune")
    print("PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10406843/")
    print()
    
    # Initialize analyzer
    analyzer = RigorousFungalAnalyzer("data/csv_data", "data/15061491/fungal_spikes/good_recordings")
    
    # Load data
    data = analyzer.load_and_categorize_data()
    
    print(f"Loaded {len(data['coordinate_data'])} coordinate files")
    print(f"Loaded {len(data['voltage_data'])} voltage files")
    print()
    
    # Find Sc (Schizophyllum commune) files
    sc_files = []
    for filename, data_info in data['coordinate_data'].items():
        if data_info['metadata']['species'] == 'Sc':
            sc_files.append(filename)
    
    print(f"Found {len(sc_files)} Sc (Schizophyllum commune) files")
    print()
    
    if not sc_files:
        print("No Sc files found. Testing with synthetic data...")
        # Create synthetic multiscalar signal for testing
        t = np.linspace(0, 1000, 10000)
        signal = (np.sin(2*np.pi*0.01*t) +  # Slow component
                 0.5*np.sin(2*np.pi*0.1*t) +  # Medium component
                 0.2*np.sin(2*np.pi*1.0*t) +  # Fast component
                 0.1*np.random.randn(len(t)))  # Noise
        
        multiscalar_result = analyzer.analyze_multiscalar_electrical_spiking(signal, 'Sc', 10.0)
        
        print("=== SYNTHETIC MULTISCALAR ANALYSIS RESULTS ===")
        print(f"Multiscalar Analysis: {multiscalar_result['multiscalar_analysis']}")
        print(f"Spike Pattern: {multiscalar_result['spike_patterns']['pattern_type']}")
        print(f"Spike Count: {multiscalar_result['spike_patterns']['spike_count']}")
        print(f"Multiscalar Complexity: {multiscalar_result['multiscalar_complexity']:.3f}")
        print()
        
        # Show temporal scale analysis
        print("=== TEMPORAL SCALE ANALYSIS ===")
        for scale, features in multiscalar_result['temporal_scale_analysis'].items():
            print(f"Scale {scale}s:")
            print(f"  Mean Amplitude: {features['mean_amplitude']:.3f}")
            print(f"  Amplitude Variance: {features['amplitude_variance']:.3f}")
            print(f"  Peak Count: {features['peak_count']}")
            print(f"  Zero Crossings: {features['zero_crossings']}")
            print(f"  Autocorrelation: {features['autocorrelation']:.3f}")
            print()
        
        # Show frequency band analysis
        print("=== FREQUENCY BAND ANALYSIS ===")
        for freq_band, features in multiscalar_result['frequency_band_analysis'].items():
            print(f"Frequency Band {freq_band} Hz:")
            print(f"  Band Power: {features['band_power']:.3f}")
            print(f"  Band Amplitude: {features['band_amplitude']:.3f}")
            print(f"  Band Peaks: {features['band_peaks']}")
            print()
        
        # Show cross-scale coupling
        print("=== CROSS-SCALE COUPLING ===")
        for coupling_name, coupling_value in multiscalar_result['cross_scale_coupling'].items():
            print(f"{coupling_name}: {coupling_value:.3f}")
        print()
        
    else:
        # Test with actual Sc files
        for filename in sc_files[:3]:  # Test first 3 Sc files
            print(f"Testing multiscalar analysis for: {filename}")
            
            data_info = data['coordinate_data'][filename]
            df = data_info['data']
            metadata = data_info['metadata']
            
            # Extract coordinate signals
            if len(df.columns) >= 2:
                x_coords = df.iloc[:, 0].values
                y_coords = df.iloc[:, 1].values
                
                # Create derived signals
                distance = np.sqrt(x_coords**2 + y_coords**2)
                velocity = np.gradient(distance)
                acceleration = np.gradient(velocity)
                
                # Test multiscalar analysis on each signal
                signals = {
                    'distance': distance,
                    'velocity': velocity,
                    'acceleration': acceleration
                }
                
                for signal_name, signal_data in signals.items():
                    # Normalize signal
                    if np.std(signal_data) > 0:
                        signal_normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
                    else:
                        signal_normalized = signal_data
                    
                    multiscalar_result = analyzer.analyze_multiscalar_electrical_spiking(
                        signal_normalized, 'Sc', 1.0
                    )
                    
                    print(f"  {signal_name} signal:")
                    print(f"    Multiscalar Analysis: {multiscalar_result['multiscalar_analysis']}")
                    print(f"    Spike Pattern: {multiscalar_result['spike_patterns']['pattern_type']}")
                    print(f"    Spike Count: {multiscalar_result['spike_patterns']['spike_count']}")
                    print(f"    Complexity: {multiscalar_result['multiscalar_complexity']:.3f}")
                    print()
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"multiscalar_analysis_test_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump({
            'test_timestamp': timestamp,
            'sc_files_found': len(sc_files),
            'test_summary': 'Multiscalar electrical spiking analysis test completed'
        }, f, indent=2)
    
    print(f"Test results saved to: {results_file}")
    
    return True

if __name__ == "__main__":
    test_multiscalar_analysis() 