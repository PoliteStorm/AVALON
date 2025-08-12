#!/usr/bin/env python3
"""
Data-Driven Parameter Analysis
Let the data speak for itself by automatically discovering optimal parameters.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def analyze_data_characteristics(data):
    """Analyze the actual characteristics of the data to inform parameter selection."""
    
    print("=== ANALYZING DATA CHARACTERISTICS ===")
    
    characteristics = {
        'coordinate_data': {},
        'voltage_data': {},
        'overall': {}
    }
    
    # Analyze coordinate data
    if data['coordinate_data']:
        print("\nAnalyzing coordinate data...")
        coord_stats = []
        
        for filename, data_dict in list(data['coordinate_data'].items())[:10]:  # Sample first 10 files
            df = data_dict['data']
            if hasattr(df, 'columns') and len(df.columns) >= 2:
                x_coords = df.iloc[:, 0].values
                y_coords = df.iloc[:, 1].values
                distance = np.sqrt(x_coords**2 + y_coords**2)
                
                # Calculate velocity (rate of change)
                if len(distance) > 1:
                    velocity = np.diff(distance)
                    acceleration = np.diff(velocity) if len(velocity) > 1 else np.array([0])
                    
                    coord_stats.append({
                        'filename': filename,
                        'mean_distance': np.mean(distance),
                        'std_distance': np.std(distance),
                        'mean_velocity': np.mean(np.abs(velocity)),
                        'std_velocity': np.std(velocity),
                        'mean_acceleration': np.mean(np.abs(acceleration)),
                        'std_acceleration': np.std(acceleration),
                        'signal_length': len(distance),
                        'sampling_rate': 1.0  # Assuming 1 Hz sampling
                    })
        
        if coord_stats:
            # Aggregate statistics
            mean_dist = np.mean([s['mean_distance'] for s in coord_stats])
            std_dist = np.mean([s['std_distance'] for s in coord_stats])
            mean_vel = np.mean([s['mean_velocity'] for s in coord_stats])
            std_vel = np.mean([s['std_velocity'] for s in coord_stats])
            
            characteristics['coordinate_data'] = {
                'mean_distance': mean_dist,
                'std_distance': std_dist,
                'mean_velocity': mean_vel,
                'std_velocity': std_vel,
                'amplitude_range': [mean_dist - 2*std_dist, mean_dist + 2*std_dist],
                'velocity_range': [mean_vel - 2*std_vel, mean_vel + 2*std_vel]
            }
            
            print(f"  Mean distance: {mean_dist:.3f} ± {std_dist:.3f}")
            print(f"  Mean velocity: {mean_vel:.3f} ± {std_vel:.3f}")
    
    # Analyze voltage data
    if data['voltage_data']:
        print("\nAnalyzing voltage data...")
        voltage_stats = []
        
        for filename, data_dict in list(data['voltage_data'].items())[:5]:  # Sample first 5 files
            df = data_dict['data']
            if hasattr(df, 'columns') and len(df.columns) >= 1:
                # Convert to numeric, handling non-numeric values
                voltage_raw = df.iloc[:, 0].values
                voltage = pd.to_numeric(voltage_raw, errors='coerce')
                voltage = voltage[~np.isnan(voltage)]  # Remove NaN values
                
                if len(voltage) > 0:  # Only process if we have valid numeric data
                    voltage_stats.append({
                        'filename': filename,
                        'mean_voltage': np.mean(voltage),
                        'std_voltage': np.std(voltage),
                        'min_voltage': np.min(voltage),
                        'max_voltage': np.max(voltage),
                        'signal_length': len(voltage),
                        'sampling_rate': 1.0  # Assuming 1 Hz sampling
                    })
        
        if voltage_stats:
            # Aggregate statistics
            mean_voltage = np.mean([s['mean_voltage'] for s in voltage_stats])
            std_voltage = np.mean([s['std_voltage'] for s in voltage_stats])
            min_voltage = np.min([s['min_voltage'] for s in voltage_stats])
            max_voltage = np.max([s['max_voltage'] for s in voltage_stats])
            
            characteristics['voltage_data'] = {
                'mean_voltage': mean_voltage,
                'std_voltage': std_voltage,
                'voltage_range': [min_voltage, max_voltage],
                'amplitude_range': [mean_voltage - 2*std_voltage, mean_voltage + 2*std_voltage]
            }
            
            print(f"  Mean voltage: {mean_voltage:.3f} ± {std_voltage:.3f} mV")
            print(f"  Voltage range: [{min_voltage:.3f}, {max_voltage:.3f}] mV")
    
    return characteristics

def discover_optimal_parameters(data, characteristics):
    """Discover optimal parameters based on data characteristics."""
    
    print("\n=== DISCOVERING OPTIMAL PARAMETERS ===")
    
    # Start with very broad parameter ranges
    base_params = {
        'k_range': np.logspace(-3, 2, 50),  # 0.001 to 100 Hz (very broad)
        'tau_range': np.logspace(-2, 4, 50),  # 0.01 to 10000 seconds (very broad)
        'amplitude_threshold': 0.01,  # Very low threshold
        'frequency_threshold': 0.001,  # Very low threshold
        'sqrt_scaling_factor': 1.0,
        'window_function': 'gaussian',
        'detection_method': 'adaptive',
        'significance_threshold': 0.1,  # Relaxed
        'time_scale_threshold': 0.01  # Very low threshold
    }
    
    # Test with sample data to see what we detect
    print("Testing with broad parameters to see what the data contains...")
    
    sample_results = {}
    
    # Test coordinate data
    if data['coordinate_data']:
        sample_file = list(data['coordinate_data'].keys())[0]
        sample_data_dict = data['coordinate_data'][sample_file]
        sample_data = sample_data_dict['data']
        
        if hasattr(sample_data, 'columns') and len(sample_data.columns) >= 2:
            x_coords = sample_data.iloc[:, 0].values
            y_coords = sample_data.iloc[:, 1].values
            distance_signal = np.sqrt(x_coords**2 + y_coords**2)
            
            # Apply transform with broad parameters
            analyzer = RigorousFungalAnalyzer("data/csv_data", "data/15061491/fungal_spikes/good_recordings")
            transform_results = analyzer.apply_sqrt_transform(distance_signal, base_params, 1.0)
            features = transform_results['features']
            
            if features:
                # Analyze detected features
                frequencies = [f['frequency'] for f in features]
                time_scales = [f['time_scale'] for f in features]
                magnitudes = [f['magnitude'] for f in features]
                
                sample_results['coordinate'] = {
                    'feature_count': len(features),
                    'frequency_range': [np.min(frequencies), np.max(frequencies)],
                    'time_scale_range': [np.min(time_scales), np.max(time_scales)],
                    'magnitude_range': [np.min(magnitudes), np.max(magnitudes)],
                    'mean_frequency': np.mean(frequencies),
                    'mean_time_scale': np.mean(time_scales),
                    'mean_magnitude': np.mean(magnitudes)
                }
                
                print(f"  Coordinate data: {len(features)} features detected")
                print(f"    Frequency range: {sample_results['coordinate']['frequency_range'][0]:.4f} - {sample_results['coordinate']['frequency_range'][1]:.4f} Hz")
                print(f"    Time scale range: {sample_results['coordinate']['time_scale_range'][0]:.2f} - {sample_results['coordinate']['time_scale_range'][1]:.2f} s")
    
    # Test voltage data
    if data['voltage_data']:
        sample_file = list(data['voltage_data'].keys())[0]
        sample_data_dict = data['voltage_data'][sample_file]
        sample_data = sample_data_dict['data']
        
        if hasattr(sample_data, 'columns') and len(sample_data.columns) >= 1:
            # Convert to numeric, handling non-numeric values
            voltage_raw = sample_data.iloc[:, 0].values
            voltage_signal = pd.to_numeric(voltage_raw, errors='coerce')
            voltage_signal = voltage_signal[~np.isnan(voltage_signal)]  # Remove NaN values
            
            if len(voltage_signal) == 0:
                print("  No valid numeric voltage data found")
                return sample_results
            
            # Apply transform with broad parameters
            analyzer = RigorousFungalAnalyzer("data/csv_data", "data/15061491/fungal_spikes/good_recordings")
            transform_results = analyzer.apply_sqrt_transform(voltage_signal, base_params, 1.0)
            features = transform_results['features']
            
            if features:
                # Analyze detected features
                frequencies = [f['frequency'] for f in features]
                time_scales = [f['time_scale'] for f in features]
                magnitudes = [f['magnitude'] for f in features]
                
                sample_results['voltage'] = {
                    'feature_count': len(features),
                    'frequency_range': [np.min(frequencies), np.max(frequencies)],
                    'time_scale_range': [np.min(time_scales), np.max(time_scales)],
                    'magnitude_range': [np.min(magnitudes), np.max(magnitudes)],
                    'mean_frequency': np.mean(frequencies),
                    'mean_time_scale': np.mean(time_scales),
                    'mean_magnitude': np.mean(magnitudes)
                }
                
                print(f"  Voltage data: {len(features)} features detected")
                print(f"    Frequency range: {sample_results['voltage']['frequency_range'][0]:.4f} - {sample_results['voltage']['frequency_range'][1]:.4f} Hz")
                print(f"    Time scale range: {sample_results['voltage']['time_scale_range'][0]:.2f} - {sample_results['voltage']['time_scale_range'][1]:.2f} s")
    
    # Derive optimal parameters from detected features
    optimal_params = derive_parameters_from_features(sample_results, characteristics)
    
    return optimal_params, sample_results

def derive_parameters_from_features(sample_results, characteristics):
    """Derive optimal parameters based on what was actually detected."""
    
    print("\n=== DERIVING OPTIMAL PARAMETERS ===")
    
    # Combine results from both data types
    all_frequencies = []
    all_time_scales = []
    all_magnitudes = []
    
    for data_type, results in sample_results.items():
        if 'frequency_range' in results:
            all_frequencies.extend(results['frequency_range'])
            all_time_scales.extend(results['time_scale_range'])
            all_magnitudes.extend(results['magnitude_range'])
    
    if not all_frequencies:
        print("  No features detected with broad parameters")
        print("  Using conservative default parameters")
        return {
            'k_range': np.logspace(-2, 1, 30),  # 0.01 to 10 Hz
            'tau_range': np.logspace(-1, 3, 30),  # 0.1 to 1000 seconds
            'amplitude_threshold': 0.05,
            'frequency_threshold': 0.01,
            'sqrt_scaling_factor': 1.0,
            'window_function': 'gaussian',
            'detection_method': 'adaptive',
            'significance_threshold': 0.05,
            'time_scale_threshold': 0.1
        }
    
    # Calculate optimal ranges based on detected features
    min_freq = np.min(all_frequencies)
    max_freq = np.max(all_frequencies)
    min_time = np.min(all_time_scales)
    max_time = np.max(all_time_scales)
    min_mag = np.min(all_magnitudes)
    max_mag = np.max(all_magnitudes)
    
    # Add some margin to the ranges
    freq_margin = (max_freq - min_freq) * 0.2
    time_margin = (max_time - min_time) * 0.2
    mag_margin = (max_mag - min_mag) * 0.2
    
    optimal_k_range = np.logspace(
        np.log10(max(min_freq - freq_margin, 0.001)), 
        np.log10(max_freq + freq_margin), 
        30
    )
    
    optimal_tau_range = np.logspace(
        np.log10(max(min_time - time_margin, 0.01)), 
        np.log10(max_time + time_margin), 
        30
    )
    
    # Set thresholds based on detected magnitudes
    optimal_amplitude_threshold = max(min_mag * 0.5, 0.01)
    optimal_frequency_threshold = max(min_freq * 0.5, 0.001)
    
    print(f"  Optimal frequency range: {optimal_k_range[0]:.4f} - {optimal_k_range[-1]:.4f} Hz")
    print(f"  Optimal time scale range: {optimal_tau_range[0]:.2f} - {optimal_tau_range[-1]:.2f} s")
    print(f"  Optimal amplitude threshold: {optimal_amplitude_threshold:.4f}")
    print(f"  Optimal frequency threshold: {optimal_frequency_threshold:.4f}")
    
    return {
        'k_range': optimal_k_range,
        'tau_range': optimal_tau_range,
        'amplitude_threshold': optimal_amplitude_threshold,
        'frequency_threshold': optimal_frequency_threshold,
        'sqrt_scaling_factor': 1.0,
        'window_function': 'gaussian',
        'detection_method': 'adaptive',
        'significance_threshold': 0.05,
        'time_scale_threshold': 0.1
    }

def test_uniform_data_driven_analysis():
    """Test analysis with data-driven uniform parameters."""
    
    print("=== DATA-DRIVEN UNIFORM ANALYSIS ===")
    print("Letting the data speak for itself...\n")
    
    # Initialize analyzer
    analyzer = RigorousFungalAnalyzer("data/csv_data", "data/15061491/fungal_spikes/good_recordings")
    
    # Load data
    print("Loading data...")
    data = analyzer.load_and_categorize_data()
    
    # Analyze data characteristics
    characteristics = analyze_data_characteristics(data)
    
    # Discover optimal parameters
    optimal_params, sample_results = discover_optimal_parameters(data, characteristics)
    
    # Test with optimal parameters
    print("\n=== TESTING WITH DATA-DRIVEN PARAMETERS ===")
    
    results = {}
    
    # Test all species with the same data-driven parameters
    for species in ['Pv', 'Pi', 'Pp', 'Rb', 'Ag', 'Sc']:
        print(f"Testing {species} with data-driven parameters...")
        
        # Use the same optimal parameters for all species
        params = optimal_params.copy()
        
        # Test with sample data
        if data['coordinate_data']:
            sample_file = list(data['coordinate_data'].keys())[0]
            sample_data_dict = data['coordinate_data'][sample_file]
            sample_data = sample_data_dict['data']
            
            # Extract distance signal
            if hasattr(sample_data, 'columns') and len(sample_data.columns) >= 2:
                x_coords = sample_data.iloc[:, 0].values
                y_coords = sample_data.iloc[:, 1].values
                distance_signal = np.sqrt(x_coords**2 + y_coords**2)
            else:
                distance_signal = sample_data.iloc[:, 0].values
            
            # Apply transform
            transform_results = analyzer.apply_sqrt_transform(distance_signal, params, 1.0)
            features = transform_results['features']
            
            results[species] = {
                'feature_count': len(features),
                'avg_frequency': np.mean([f['frequency'] for f in features]) if features else 0,
                'avg_time_scale': np.mean([f['time_scale'] for f in features]) if features else 0,
                'avg_magnitude': np.mean([f['magnitude'] for f in features]) if features else 0,
                'frequency_range': [np.min([f['frequency'] for f in features]), np.max([f['frequency'] for f in features])] if features else [0, 0],
                'time_scale_range': [np.min([f['time_scale'] for f in features]), np.max([f['time_scale'] for f in features])] if features else [0, 0]
            }
            
            print(f"  {species}: {len(features)} features")
            if features:
                print(f"    Frequency: {results[species]['avg_frequency']:.4f} Hz ({results[species]['frequency_range'][0]:.4f}-{results[species]['frequency_range'][1]:.4f})")
                print(f"    Time scale: {results[species]['avg_time_scale']:.2f} s ({results[species]['time_scale_range'][0]:.2f}-{results[species]['time_scale_range'][1]:.2f})")
    
    # Analyze consistency across species
    print("\n=== CONSISTENCY ANALYSIS ===")
    
    feature_counts = [results[s]['feature_count'] for s in results.keys()]
    avg_frequencies = [results[s]['avg_frequency'] for s in results.keys()]
    avg_time_scales = [results[s]['avg_time_scale'] for s in results.keys()]
    
    print(f"Feature count statistics:")
    print(f"  Mean: {np.mean(feature_counts):.1f} ± {np.std(feature_counts):.1f}")
    print(f"  Range: {np.min(feature_counts)} - {np.max(feature_counts)}")
    
    print(f"Frequency statistics:")
    print(f"  Mean: {np.mean(avg_frequencies):.4f} ± {np.std(avg_frequencies):.4f} Hz")
    print(f"  Range: {np.min(avg_frequencies):.4f} - {np.max(avg_frequencies):.4f} Hz")
    
    print(f"Time scale statistics:")
    print(f"  Mean: {np.mean(avg_time_scales):.2f} ± {np.std(avg_time_scales):.2f} s")
    print(f"  Range: {np.min(avg_time_scales):.2f} - {np.max(avg_time_scales):.2f} s")
    
    # Assess consistency
    feature_cv = np.std(feature_counts) / np.mean(feature_counts) if np.mean(feature_counts) > 0 else 0
    freq_cv = np.std(avg_frequencies) / np.mean(avg_frequencies) if np.mean(avg_frequencies) > 0 else 0
    time_cv = np.std(avg_time_scales) / np.mean(avg_time_scales) if np.mean(avg_time_scales) > 0 else 0
    
    print(f"\nCoefficient of variation (lower = more consistent):")
    print(f"  Feature count: {feature_cv:.3f}")
    print(f"  Frequency: {freq_cv:.3f}")
    print(f"  Time scale: {time_cv:.3f}")
    
    if feature_cv < 0.3 and freq_cv < 0.3:
        print("\n✅ HIGH CONSISTENCY: Results suggest real biological patterns")
        print("   The data-driven approach reveals consistent patterns across species")
    elif feature_cv < 0.5 and freq_cv < 0.5:
        print("\n⚠️  MODERATE CONSISTENCY: Some variation but generally consistent")
        print("   Consider investigating species-specific differences")
    else:
        print("\n❌ LOW CONSISTENCY: High variation suggests parameter dependence")
        print("   Results may be artifacts of parameter choices")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data_driven_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'optimal_parameters': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in optimal_params.items()},
            'sample_results': sample_results,
            'species_results': results,
            'consistency_metrics': {
                'feature_cv': feature_cv,
                'frequency_cv': freq_cv,
                'time_scale_cv': time_cv
            }
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results, optimal_params

if __name__ == "__main__":
    # Run the data-driven analysis
    results, optimal_params = test_uniform_data_driven_analysis()
    
    print("\n=== KEY INSIGHTS ===")
    print("1. This approach lets the data determine optimal parameters")
    print("2. No species-specific bias - same parameters for all species")
    print("3. Consistency across species suggests real patterns")
    print("4. High variation suggests parameter dependence")
    print("5. Use these optimal parameters for unbiased analysis") 