#!/usr/bin/env python3
"""
Test script for electrical spiking analysis using actual voltage data.
This script tests both Adamatzky's method and the current code's method.
FOCUS: Electrical activity only - no coordinate data analysis.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

def load_voltage_data(file_path):
    """Load voltage data from CSV file."""
    try:
        data = pd.read_csv(file_path, header=None)
        if len(data.columns) == 1:
            voltage_signal = data.iloc[:, 0].values
        else:
            voltage_signal = data.iloc[:, 1].values
        return voltage_signal
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def test_adamatzkys_method(voltage_signal, sampling_rate=1.0):
    """
    Test Adamatzky's actual spike detection method.
    """
    print("=== TESTING ADAMATZKY'S SPIKE DETECTION METHOD ===")
    
    # Adamatzky's parameters
    w = 20  # Window size
    delta = 0.01  # Threshold (mV)
    
    spikes = []
    n_samples = len(voltage_signal)
    
    # Adamatzky's algorithm
    for i in range(w, n_samples - w):
        window_start = i - w
        window_end = i + w
        avg = np.mean(voltage_signal[window_start:window_end])
        
        if abs(voltage_signal[i]) - abs(avg) > delta:
            spike_info = {
                'time_index': i,
                'time_seconds': i / sampling_rate,
                'amplitude_mv': voltage_signal[i],
                'threshold': avg + delta,
                'baseline': avg
            }
            spikes.append(spike_info)
    
    # Calculate statistics
    if len(spikes) > 1:
        isi = []
        for i in range(1, len(spikes)):
            interval = spikes[i]['time_seconds'] - spikes[i-1]['time_seconds']
            isi.append(interval)
        
        mean_isi = np.mean(isi)
        mean_amplitude = np.mean([s['amplitude_mv'] for s in spikes])
    else:
        mean_isi = 0
        mean_amplitude = np.mean([s['amplitude_mv'] for s in spikes]) if spikes else 0
    
    print(f"Spikes detected: {len(spikes)}")
    print(f"Mean amplitude: {mean_amplitude:.6f} mV")
    print(f"Mean ISI: {mean_isi:.2f} seconds")
    print(f"Spike rate: {len(spikes) / (n_samples / sampling_rate):.3f} Hz")
    
    return {
        'spikes': spikes,
        'n_spikes': len(spikes),
        'mean_amplitude': mean_amplitude,
        'mean_isi': mean_isi,
        'spike_rate': len(spikes) / (n_samples / sampling_rate)
    }

def test_current_method(voltage_signal, sampling_rate=1.0):
    """
    Test the current code's √t transform method.
    """
    print("\n=== TESTING CURRENT CODE'S √t TRANSFORM METHOD ===")
    
    # Simplified version of current method
    k_values = np.logspace(-2, 1, 30)
    tau_values = np.logspace(-1, 3, 30)
    
    # Create time vector
    t = np.arange(len(voltage_signal)) / sampling_rate
    
    # Apply √t transform
    features = []
    for k in k_values:
        for tau in tau_values:
            sqrt_t = np.sqrt(t)
            window = np.exp(-(sqrt_t / tau)**2)
            phase = np.exp(-1j * k * sqrt_t)
            integrand = voltage_signal * window * phase
            magnitude = np.abs(np.trapz(integrand, t))
            
            if magnitude > np.mean(voltage_signal) * 0.1:  # Simple threshold
                features.append({
                    'k': k,
                    'tau': tau,
                    'magnitude': magnitude,
                    'frequency': k / (2 * np.pi),
                    'time_scale': tau
                })
    
    if features:
        mean_magnitude = np.mean([f['magnitude'] for f in features])
        mean_frequency = np.mean([f['frequency'] for f in features])
        mean_time_scale = np.mean([f['time_scale'] for f in features])
    else:
        mean_magnitude = mean_frequency = mean_time_scale = 0
    
    print(f"Features detected: {len(features)}")
    print(f"Mean magnitude: {mean_magnitude:.6f}")
    print(f"Mean frequency: {mean_frequency:.6f} Hz")
    print(f"Mean time scale: {mean_time_scale:.2f} seconds")
    
    return {
        'features': features,
        'n_features': len(features),
        'mean_magnitude': mean_magnitude,
        'mean_frequency': mean_frequency,
        'mean_time_scale': mean_time_scale
    }

def test_electrical_analysis():
    """Test electrical spiking analysis on actual voltage data."""
    
    print("=== ELECTRICAL SPIKING ANALYSIS TEST ===")
    print("Testing on actual voltage recordings from fungal data")
    print("FOCUS: Electrical activity only - no coordinate data")
    print()
    
    # Test on available voltage files
    voltage_dir = "15061491/fungal_spikes/good_recordings"
    test_file = "15061491/fungal_spikes/good_recordings/New_Oyster_with spray_as_mV_seconds_SigView.csv"
    
    if not os.path.exists(test_file):
        print(f"Error: Voltage data file not found: {test_file}")
        return False
    
    # Load voltage data
    voltage_signal = load_voltage_data(test_file)
    if voltage_signal is None:
        print("Failed to load voltage data")
        return False
    
    print(f"Loaded {len(voltage_signal)} voltage samples")
    print(f"Voltage range: {np.min(voltage_signal):.6f} to {np.max(voltage_signal):.6f} mV")
    print(f"Mean voltage: {np.mean(voltage_signal):.6f} mV")
    print(f"Std voltage: {np.std(voltage_signal):.6f} mV")
    print()
    
    # Test Adamatzky's method
    adamatzky_results = test_adamatzkys_method(voltage_signal)
    
    # Test current method
    current_results = test_current_method(voltage_signal)
    
    # Comparison
    print("\n=== COMPARISON ===")
    print(f"Adamatzky's method: {adamatzky_results['n_spikes']} spikes")
    print(f"Current method: {current_results['n_features']} features")
    print()
    
    if adamatzky_results['n_spikes'] > 0:
        print("Adamatzky's method found actual electrical spikes in voltage data")
    else:
        print("Adamatzky's method found no spikes (may need parameter adjustment)")
    
    if current_results['n_features'] > 0:
        print("Current method found mathematical features in transformed signal")
    else:
        print("Current method found no features (may need parameter adjustment)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"electrical_analysis_test_{timestamp}.json"
    
    results = {
        'test_timestamp': timestamp,
        'voltage_file': test_file,
        'n_samples': len(voltage_signal),
        'voltage_stats': {
            'min': float(np.min(voltage_signal)),
            'max': float(np.max(voltage_signal)),
            'mean': float(np.mean(voltage_signal)),
            'std': float(np.std(voltage_signal))
        },
        'adamatzky_results': adamatzky_results,
        'current_results': current_results,
        'analysis_type': 'Electrical activity only',
        'note': 'No coordinate data analysis - electrical spikes only'
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to: {results_file}")
    
    return True

if __name__ == "__main__":
    test_electrical_analysis() 