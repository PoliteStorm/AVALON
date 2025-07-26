#!/usr/bin/env python3
"""
Algorithm Comparison: Adamatzky's Spike Detection vs √t Transform
Compare the actual results from both methods on real voltage data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
from sklearn.cluster import DBSCAN
import os
from datetime import datetime
import json

class AdamatzkySpikeDetector:
    """
    Implementation of Adamatzky's actual spike detection algorithm.
    Based on: "Multiscalar electrical spiking in Schizophyllum commune" (2023)
    """
    
    def __init__(self):
        # Adamatzky's published parameters
        self.w = 20  # Window size for moving average
        self.delta = 0.01  # Threshold parameter (mV)
        self.d = 30  # Distance parameter
        
    def detect_spikes_adamatzkys_method(self, voltage_signal, sampling_rate=1.0):
        """
        Adamatzky's spike detection algorithm.
        
        Args:
            voltage_signal: Voltage signal in mV
            sampling_rate: Sampling rate in Hz
            
        Returns:
            dict: Spike detection results
        """
        spikes = []
        n_samples = len(voltage_signal)
        
        # Adamatzky's algorithm
        for i in range(self.w, n_samples - self.w):
            # Calculate moving average
            window_start = i - self.w
            window_end = i + self.w
            avg = np.mean(voltage_signal[window_start:window_end])
            
            # Check if sample exceeds threshold
            if abs(voltage_signal[i]) - abs(avg) > self.delta:
                # Calculate spike duration and amplitude
                spike_info = {
                    'time_index': i,
                    'time_seconds': i / sampling_rate,
                    'amplitude_mv': voltage_signal[i],
                    'threshold': avg + self.delta,
                    'baseline': avg
                }
                spikes.append(spike_info)
        
        # Calculate inter-spike intervals
        if len(spikes) > 1:
            isi = []
            for i in range(1, len(spikes)):
                interval = spikes[i]['time_seconds'] - spikes[i-1]['time_seconds']
                isi.append(interval)
            
            # Classify spikes based on Adamatzky's three families
            spike_classifications = self.classify_spikes_adamatzkys_families(isi)
        else:
            spike_classifications = {'very_fast': 0, 'slow': 0, 'very_slow': 0}
        
        return {
            'spikes': spikes,
            'n_spikes': len(spikes),
            'inter_spike_intervals': isi if len(spikes) > 1 else [],
            'classifications': spike_classifications,
            'mean_amplitude': np.mean([s['amplitude_mv'] for s in spikes]) if spikes else 0,
            'mean_isi': np.mean(isi) if len(spikes) > 1 else 0
        }
    
    def classify_spikes_adamatzkys_families(self, isi):
        """
        Classify spikes according to Adamatzky's three families.
        
        Adamatzky's published ranges:
        - Very fast spikes: ~24s (24s ± 0.07s)
        - Slow spikes: ~8 min (457s ± 120s)
        - Very slow spikes: ~43 min (2573s ± 168s)
        """
        if not isi:
            return {'very_fast': 0, 'slow': 0, 'very_slow': 0}
        
        classifications = {'very_fast': 0, 'slow': 0, 'very_slow': 0}
        
        for interval in isi:
            if interval < 100:  # < 100s = very fast
                classifications['very_fast'] += 1
            elif interval < 1000:  # 100s - 1000s = slow
                classifications['slow'] += 1
            else:  # > 1000s = very slow
                classifications['very_slow'] += 1
        
        return classifications

class CurrentMethodAnalyzer:
    """
    Implementation of the current code's √t transform method.
    """
    
    def __init__(self):
        self.biological_constraints = {
            'frequency_range': (0.001, 10.0),
            'growth_time_scales': (0.1, 100000),
            'spike_amplitude_range': (0.02, 0.15)
        }
    
    def apply_sqrt_transform(self, signal, params, sampling_rate=1.0):
        """
        Apply √t transform (current code's method).
        """
        k_values = params['k_range']
        tau_values = params['tau_range']
        
        # Create time vector
        t = np.arange(len(signal)) / sampling_rate
        
        # Apply transform
        W = np.zeros((len(k_values), len(tau_values)), dtype=complex)
        
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                # Gaussian window with √t scaling
                sqrt_t = np.sqrt(t)
                window = np.exp(-(sqrt_t / tau)**2)
                phase = np.exp(-1j * k * sqrt_t)
                integrand = signal * window * phase
                W[i, j] = np.trapezoid(integrand, t)
        
        magnitude = np.abs(W)
        
        # Feature detection
        features = self.detect_features(magnitude, k_values, tau_values, params)
        
        return {
            'magnitude': magnitude,
            'phase': np.angle(W),
            'k_values': k_values,
            'tau_values': tau_values,
            'features': features
        }
    
    def detect_features(self, magnitude, k_values, tau_values, params):
        """
        Detect features using current code's method.
        """
        features = []
        
        amplitude_threshold = params.get('amplitude_threshold', 0.05)
        frequency_threshold = params.get('frequency_threshold', 0.01)
        time_scale_threshold = params.get('time_scale_threshold', 0.1)
        
        # Calculate adaptive threshold
        mag_flat = magnitude.flatten()
        mean_mag = np.mean(mag_flat)
        threshold = mean_mag + amplitude_threshold * (np.max(mag_flat) - mean_mag)
        
        # Find peaks above threshold
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                if magnitude[i, j] > threshold:
                    # Check if it's a local maximum
                    is_local_max = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < len(k_values) and 
                                0 <= nj < len(tau_values) and
                                magnitude[ni, nj] > magnitude[i, j]):
                                is_local_max = False
                                break
                        if not is_local_max:
                            break
                    
                    if is_local_max:
                        frequency = k / (2 * np.pi)
                        time_scale = tau
                        
                        if (frequency >= frequency_threshold and 
                            time_scale >= time_scale_threshold):
                            
                            features.append({
                                'k': k,
                                'tau': tau,
                                'magnitude': magnitude[i, j],
                                'frequency': frequency,
                                'time_scale': time_scale,
                                'amplitude_ratio': magnitude[i, j] / np.max(mag_flat)
                            })
        
        return features
    
    def analyze_with_current_method(self, voltage_signal, sampling_rate=1.0):
        """
        Analyze voltage signal using current code's method.
        """
        # Get parameters (similar to current code)
        params = {
            'k_range': np.logspace(-2, 1, 30),
            'tau_range': np.logspace(-1, 3, 30),
            'amplitude_threshold': 0.05,
            'frequency_threshold': 0.01,
            'time_scale_threshold': 0.1
        }
        
        # Apply transform
        transform_results = self.apply_sqrt_transform(voltage_signal, params, sampling_rate)
        
        return {
            'features': transform_results['features'],
            'n_features': len(transform_results['features']),
            'magnitudes': [f['magnitude'] for f in transform_results['features']],
            'frequencies': [f['frequency'] for f in transform_results['features']],
            'time_scales': [f['time_scale'] for f in transform_results['features']]
        }

def load_voltage_data(file_path):
    """
    Load voltage data from CSV file.
    """
    try:
        # Try to read as single column
        data = pd.read_csv(file_path, header=None)
        if len(data.columns) == 1:
            voltage_signal = data.iloc[:, 0].values
        else:
            # Assume second column is voltage
            voltage_signal = data.iloc[:, 1].values
        
        return voltage_signal
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compare_algorithms(voltage_file):
    """
    Compare Adamatzky's method vs current code's method.
    """
    print(f"Comparing algorithms on: {voltage_file}")
    print("="*60)
    
    # Load voltage data
    voltage_signal = load_voltage_data(voltage_file)
    if voltage_signal is None:
        print("Failed to load voltage data")
        return
    
    print(f"Loaded {len(voltage_signal)} voltage samples")
    print(f"Voltage range: {np.min(voltage_signal):.6f} to {np.max(voltage_signal):.6f} mV")
    print(f"Mean voltage: {np.mean(voltage_signal):.6f} mV")
    print(f"Std voltage: {np.std(voltage_signal):.6f} mV")
    print()
    
    # Method 1: Adamatzky's spike detection
    print("METHOD 1: ADAMATZKY'S SPIKE DETECTION")
    print("-" * 40)
    adamatzky_detector = AdamatzkySpikeDetector()
    adamatzky_results = adamatzky_detector.detect_spikes_adamatzkys_method(voltage_signal)
    
    print(f"Spikes detected: {adamatzky_results['n_spikes']}")
    if adamatzky_results['n_spikes'] > 0:
        print(f"Mean amplitude: {adamatzky_results['mean_amplitude']:.6f} mV")
        print(f"Mean ISI: {adamatzky_results['mean_isi']:.2f} seconds")
        print("Spike classifications:")
        for family, count in adamatzky_results['classifications'].items():
            print(f"  {family}: {count} spikes")
    
    print()
    
    # Method 2: Current code's √t transform
    print("METHOD 2: CURRENT CODE'S √t TRANSFORM")
    print("-" * 40)
    current_analyzer = CurrentMethodAnalyzer()
    current_results = current_analyzer.analyze_with_current_method(voltage_signal)
    
    print(f"Features detected: {current_results['n_features']}")
    if current_results['n_features'] > 0:
        print(f"Mean magnitude: {np.mean(current_results['magnitudes']):.6f}")
        print(f"Mean frequency: {np.mean(current_results['frequencies']):.6f} Hz")
        print(f"Mean time scale: {np.mean(current_results['time_scales']):.2f} seconds")
    
    print()
    
    # Comparison
    print("COMPARISON")
    print("-" * 40)
    print(f"Adamatzky spikes: {adamatzky_results['n_spikes']}")
    print(f"Current features: {current_results['n_features']}")
    
    if adamatzky_results['n_spikes'] > 0 and current_results['n_features'] > 0:
        print("\nAdamatzky's method found specific electrical spikes")
        print("Current method found mathematical features")
        print("These are fundamentally different types of analysis!")
    
    return {
        'adamatzky_results': adamatzky_results,
        'current_results': current_results,
        'voltage_signal': voltage_signal
    }

def main():
    """
    Main function to run the comparison.
    """
    print("ALGORITHM COMPARISON: ADAMATZKY vs CURRENT CODE")
    print("="*60)
    print("This script compares:")
    print("1. Adamatzky's actual spike detection algorithm")
    print("2. Current code's √t transform method")
    print("on real voltage data from fungal recordings.")
    print()
    
    # Test on available voltage files
    voltage_dir = "15061491/fungal_spikes/good_recordings"
    
    # Use a smaller file for testing
    test_file = "15061491/fungal_spikes/good_recordings/New_Oyster_with spray_as_mV_seconds_SigView.csv"
    
    if os.path.exists(test_file):
        results = compare_algorithms(test_file)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"algorithm_comparison_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            'adamatzky_results': {
                'n_spikes': results['adamatzky_results']['n_spikes'],
                'mean_amplitude': float(results['adamatzky_results']['mean_amplitude']),
                'mean_isi': float(results['adamatzky_results']['mean_isi']),
                'classifications': results['adamatzky_results']['classifications']
            },
            'current_results': {
                'n_features': results['current_results']['n_features'],
                'mean_magnitude': float(np.mean(results['current_results']['magnitudes'])) if results['current_results']['magnitudes'] else 0,
                'mean_frequency': float(np.mean(results['current_results']['frequencies'])) if results['current_results']['frequencies'] else 0,
                'mean_time_scale': float(np.mean(results['current_results']['time_scales'])) if results['current_results']['time_scales'] else 0
            },
            'voltage_stats': {
                'n_samples': len(results['voltage_signal']),
                'min_voltage': float(np.min(results['voltage_signal'])),
                'max_voltage': float(np.max(results['voltage_signal'])),
                'mean_voltage': float(np.mean(results['voltage_signal'])),
                'std_voltage': float(np.std(results['voltage_signal']))
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
    else:
        print(f"Test file not found: {test_file}")
        print("Please check the file path.")

if __name__ == "__main__":
    main() 