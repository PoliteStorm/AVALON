#!/usr/bin/env python3
"""
Quick test to check what the transform is detecting in the actual data.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def test_data_detection():
    """Test what the transform is detecting in the actual data."""
    
    print("=== Testing Data Detection ===")
    
    # Initialize analyzer
    analyzer = RigorousFungalAnalyzer("data/csv_data", "data/15061491/fungal_spikes/good_recordings")
    
    # Load a sample of data
    data = analyzer.load_and_categorize_data()
    
    print(f"\nData loaded:")
    print(f"  Coordinate files: {len(data['coordinate_data'])}")
    print(f"  Voltage files: {len(data['voltage_data'])}")
    
    # Test with a few sample files
    sample_files = list(data['coordinate_data'].keys())[:3]
    
    for filename in sample_files:
        print(f"\n--- Testing {filename} ---")
        
        data_info = data['coordinate_data'][filename]
        df = data_info['data']
        metadata = data_info['metadata']
        
        print(f"  Species: {metadata['species']}")
        print(f"  Duration: {metadata['duration_hours']:.1f} hours")
        print(f"  Data points: {len(df)}")
        
        # Extract coordinate signals
        if len(df.columns) >= 2:
            x_coords = df.iloc[:, 0].values
            y_coords = df.iloc[:, 1].values
            
            # Create derived signals
            distance = np.sqrt(x_coords**2 + y_coords**2)
            velocity = np.gradient(distance)
            
            print(f"  Distance range: {np.min(distance):.1f} to {np.max(distance):.1f}")
            print(f"  Velocity range: {np.min(velocity):.3f} to {np.max(velocity):.3f}")
            
            # Test transform on distance signal
            if np.std(distance) > 0:
                signal_normalized = (distance - np.mean(distance)) / np.std(distance)
                
                # Apply transform
                transform_results = analyzer.improved_sqrt_transform(
                    signal_normalized,
                    metadata['species'],
                    metadata['treatment'],
                    metadata['duration_hours']
                )
                
                features = transform_results['transform_results']['features']
                print(f"  Features detected: {len(features)}")
                
                if features:
                    magnitudes = [f['magnitude'] for f in features]
                    frequencies = [f['frequency'] for f in features]
                    time_scales = [f['time_scale'] for f in features]
                    
                    print(f"    Magnitude range: {np.min(magnitudes):.3f} to {np.max(magnitudes):.3f}")
                    print(f"    Frequency range: {np.min(frequencies):.3f} to {np.max(frequencies):.3f} Hz")
                    print(f"    Time scale range: {np.min(time_scales):.1f} to {np.max(time_scales):.1f} s")
                    
                    # Check validation
                    validation = transform_results['validation_results']
                    print(f"    Valid features: {sum(validation['frequency_validation'])}/{len(features)}")
                    
                    # Check false positives
                    fp_analysis = transform_results['false_positive_analysis']
                    print(f"    False positive assessment: {fp_analysis['overall_assessment']}")
                    
                    # Check biological plausibility
                    bio_assessment = transform_results['biological_assessment']
                    print(f"    Biological plausibility: {bio_assessment['overall_plausibility']}")
                else:
                    print("    No features detected")
            else:
                print("    Signal has no variance - skipping transform")
    
    # Test voltage data
    print(f"\n--- Testing Voltage Data ---")
    voltage_files = list(data['voltage_data'].keys())[:2]
    
    for filename in voltage_files:
        print(f"\nTesting voltage file: {filename}")
        
        data_info = data['voltage_data'][filename]
        df = data_info['data']
        
        print(f"  Data points: {len(df)}")
        print(f"  Sampling rate: {data_info['sampling_rate']:.1f} Hz")
        
        if len(df.columns) >= 2:
            voltage_signal = df.iloc[:, 1].values
            print(f"  Voltage range: {np.min(voltage_signal):.3f} to {np.max(voltage_signal):.3f} mV")
            
            # Preprocess and test
            voltage_processed = analyzer.preprocess_voltage_signal(voltage_signal, data_info['sampling_rate'])
            
            transform_results = analyzer.improved_sqrt_transform(
                voltage_processed,
                'Unknown',
                'Standard',
                len(voltage_processed) / data_info['sampling_rate'] / 3600
            )
            
            features = transform_results['transform_results']['features']
            print(f"  Features detected: {len(features)}")
            
            if features:
                magnitudes = [f['magnitude'] for f in features]
                frequencies = [f['frequency'] for f in features]
                time_scales = [f['time_scale'] for f in features]
                
                print(f"    Magnitude range: {np.min(magnitudes):.3f} to {np.max(magnitudes):.3f}")
                print(f"    Frequency range: {np.min(frequencies):.3f} to {np.max(frequencies):.3f} Hz")
                print(f"    Time scale range: {np.min(time_scales):.1f} to {np.max(time_scales):.1f} s")

if __name__ == "__main__":
    test_data_detection() 