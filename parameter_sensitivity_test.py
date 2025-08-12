#!/usr/bin/env python3
"""
Parameter Sensitivity Test for √t Transform
This script tests how much results depend on k and τ parameter choices.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def test_parameter_sensitivity():
    """Test how much results depend on k and τ parameter choices."""
    
    print("=== PARAMETER SENSITIVITY TEST ===")
    print("Testing how much results depend on k and τ choices\n")
    
    # Initialize analyzer
    analyzer = RigorousFungalAnalyzer("../csv_data", "../15061491/fungal_spikes/good_recordings")
    
    # Load sample data
    data = analyzer.load_and_categorize_data()
    
    # Test with different parameter sets
    test_cases = [
        {
            'name': 'Current Parameters (Species-Specific)',
            'description': 'Your current species-specific parameters',
            'k_range': np.logspace(-1.0, 0.8, 30),  # Pi species
            'tau_range': np.logspace(0.2, 2.2, 30),
            'amplitude_threshold': 0.045,
            'frequency_threshold': 0.02,
            'sqrt_scaling_factor': 1.0
        },
        {
            'name': 'Broad Parameters (Generic)',
            'description': 'Generic broad parameter ranges',
            'k_range': np.logspace(-2, 1, 30),
            'tau_range': np.logspace(-1, 3, 30),
            'amplitude_threshold': 0.05,
            'frequency_threshold': 0.01,
            'sqrt_scaling_factor': 1.0
        },
        {
            'name': 'Narrow Parameters (Conservative)',
            'description': 'Conservative narrow parameter ranges',
            'k_range': np.logspace(-0.5, 0.5, 30),
            'tau_range': np.logspace(0.5, 2.5, 30),
            'amplitude_threshold': 0.06,
            'frequency_threshold': 0.03,
            'sqrt_scaling_factor': 1.0
        },
        {
            'name': 'High Frequency Focus',
            'description': 'Focus on high frequency patterns',
            'k_range': np.logspace(0.0, 1.5, 30),
            'tau_range': np.logspace(-0.5, 1.5, 30),
            'amplitude_threshold': 0.04,
            'frequency_threshold': 0.05,
            'sqrt_scaling_factor': 1.2
        },
        {
            'name': 'Low Frequency Focus',
            'description': 'Focus on low frequency patterns',
            'k_range': np.logspace(-2.5, -0.5, 30),
            'tau_range': np.logspace(1.5, 3.5, 30),
            'amplitude_threshold': 0.03,
            'frequency_threshold': 0.005,
            'sqrt_scaling_factor': 0.8
        }
    ]
    
    results = {}
    
    # Test each parameter set
    for i, test_case in enumerate(test_cases):
        print(f"Test {i+1}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        
        # Create params dictionary
        params = {
            'k_range': test_case['k_range'],
            'tau_range': test_case['tau_range'],
            'amplitude_threshold': test_case['amplitude_threshold'],
            'frequency_threshold': test_case['frequency_threshold'],
            'sqrt_scaling_factor': test_case['sqrt_scaling_factor'],
            'window_function': 'gaussian',
            'detection_method': 'adaptive',
            'significance_threshold': 0.05,
            'time_scale_threshold': 0.1
        }
        
        # Test with sample voltage data
        if data['voltage_data']:
            sample_file = list(data['voltage_data'].keys())[0]
            sample_data = data['voltage_data'][sample_file]['data']
            
            # Extract voltage signal
            if len(sample_data.columns) > 1:
                voltage_signal = sample_data.iloc[:, 1].values  # Second column as voltage
            else:
                voltage_signal = sample_data.iloc[:, 0].values
            
            # Apply transform with these parameters
            transform_results = analyzer.apply_sqrt_transform(voltage_signal, params, 1.0)
            
            # Count features detected
            features = transform_results['features']
            feature_count = len(features)
            
            # Calculate average feature characteristics
            if features:
                avg_amplitude = np.mean([f['amplitude'] for f in features])
                avg_frequency = np.mean([f['frequency'] for f in features])
                avg_time_scale = np.mean([f['time_scale'] for f in features])
            else:
                avg_amplitude = avg_frequency = avg_time_scale = 0
            
            results[test_case['name']] = {
                'feature_count': feature_count,
                'avg_amplitude': avg_amplitude,
                'avg_frequency': avg_frequency,
                'avg_time_scale': avg_time_scale,
                'parameters': test_case
            }
            
            print(f"  Features detected: {feature_count}")
            print(f"  Avg amplitude: {avg_amplitude:.4f}")
            print(f"  Avg frequency: {avg_frequency:.4f} Hz")
            print(f"  Avg time scale: {avg_time_scale:.2f} s")
            print()
    
    # Analyze parameter sensitivity
    print("=== PARAMETER SENSITIVITY ANALYSIS ===")
    
    feature_counts = [results[name]['feature_count'] for name in results.keys()]
    avg_amplitudes = [results[name]['avg_amplitude'] for name in results.keys()]
    avg_frequencies = [results[name]['avg_frequency'] for name in results.keys()]
    
    print(f"Feature count range: {min(feature_counts)} to {max(feature_counts)}")
    print(f"Feature count variation: {max(feature_counts) - min(feature_counts)}")
    print(f"Feature count coefficient of variation: {np.std(feature_counts) / np.mean(feature_counts):.2f}")
    
    print(f"\nAmplitude range: {min(avg_amplitudes):.4f} to {max(avg_amplitudes):.4f}")
    print(f"Frequency range: {min(avg_frequencies):.4f} to {max(avg_frequencies):.4f} Hz")
    
    # Calculate sensitivity metrics
    sensitivity_score = (max(feature_counts) - min(feature_counts)) / np.mean(feature_counts)
    
    print(f"\n=== SENSITIVITY ASSESSMENT ===")
    if sensitivity_score > 0.5:
        print("⚠️  HIGH SENSITIVITY: Results heavily depend on parameter choices")
        print("   This suggests the method may be overfitting to parameter selection")
    elif sensitivity_score > 0.2:
        print("⚠️  MODERATE SENSITIVITY: Results show some parameter dependence")
        print("   Consider parameter validation and cross-validation")
    else:
        print("✅  LOW SENSITIVITY: Results are relatively robust to parameter choices")
        print("   This suggests the method is finding real patterns")
    
    # Create visualization
    create_sensitivity_plot(results)
    
    return results

def create_sensitivity_plot(results):
    """Create visualization of parameter sensitivity."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=16)
    
    # Extract data
    names = list(results.keys())
    feature_counts = [results[name]['feature_count'] for name in names]
    avg_amplitudes = [results[name]['avg_amplitude'] for name in names]
    avg_frequencies = [results[name]['avg_frequency'] for name in names]
    avg_time_scales = [results[name]['avg_time_scale'] for name in names]
    
    # Feature count comparison
    axes[0, 0].bar(range(len(names)), feature_counts, color='skyblue')
    axes[0, 0].set_title('Feature Count by Parameter Set')
    axes[0, 0].set_ylabel('Number of Features Detected')
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels([name.split('(')[0].strip() for name in names], rotation=45, ha='right')
    
    # Amplitude comparison
    axes[0, 1].bar(range(len(names)), avg_amplitudes, color='lightgreen')
    axes[0, 1].set_title('Average Feature Amplitude')
    axes[0, 1].set_ylabel('Amplitude (mV)')
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels([name.split('(')[0].strip() for name in names], rotation=45, ha='right')
    
    # Frequency comparison
    axes[1, 0].bar(range(len(names)), avg_frequencies, color='lightcoral')
    axes[1, 0].set_title('Average Feature Frequency')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels([name.split('(')[0].strip() for name in names], rotation=45, ha='right')
    
    # Time scale comparison
    axes[1, 1].bar(range(len(names)), avg_time_scales, color='gold')
    axes[1, 1].set_title('Average Feature Time Scale')
    axes[1, 1].set_ylabel('Time Scale (seconds)')
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels([name.split('(')[0].strip() for name in names], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSensitivity plot saved as 'parameter_sensitivity_analysis.png'")

def test_parameter_validation():
    """Test parameter validation methods."""
    
    print("\n=== PARAMETER VALIDATION TEST ===")
    
    # Test with synthetic data
    print("Testing with synthetic data to validate parameter robustness...")
    
    # Generate synthetic fungal-like signal
    t = np.linspace(0, 3600, 3600)  # 1 hour at 1 Hz
    synthetic_signal = (
        0.1 * np.sin(2 * np.pi * 0.001 * t) +  # Very slow oscillation
        0.05 * np.sin(2 * np.pi * 0.01 * t) +   # Slow oscillation
        0.02 * np.sin(2 * np.pi * 0.1 * t) +    # Medium oscillation
        0.01 * np.random.randn(len(t))           # Noise
    )
    
    # Test different parameter sets on synthetic data
    test_params = [
        {'name': 'Broad', 'k_range': np.logspace(-2, 1, 30), 'tau_range': np.logspace(-1, 3, 30)},
        {'name': 'Narrow', 'k_range': np.logspace(-0.5, 0.5, 30), 'tau_range': np.logspace(0.5, 2.5, 30)},
        {'name': 'High Freq', 'k_range': np.logspace(0, 1.5, 30), 'tau_range': np.logspace(-0.5, 1.5, 30)},
        {'name': 'Low Freq', 'k_range': np.logspace(-2.5, -0.5, 30), 'tau_range': np.logspace(1.5, 3.5, 30)}
    ]
    
    analyzer = RigorousFungalAnalyzer("../csv_data", "../15061491/fungal_spikes/good_recordings")
    
    for param_set in test_params:
        params = {
            'k_range': param_set['k_range'],
            'tau_range': param_set['tau_range'],
            'amplitude_threshold': 0.05,
            'frequency_threshold': 0.01,
            'sqrt_scaling_factor': 1.0,
            'window_function': 'gaussian',
            'detection_method': 'adaptive',
            'significance_threshold': 0.05,
            'time_scale_threshold': 0.1
        }
        
        transform_results = analyzer.apply_sqrt_transform(synthetic_signal, params, 1.0)
        features = transform_results['features']
        
        print(f"{param_set['name']}: {len(features)} features detected")
    
    print("\nIf all parameter sets detect similar numbers of features in synthetic data,")
    print("this suggests the method is robust. If not, it suggests parameter sensitivity.")

if __name__ == "__main__":
    # Run parameter sensitivity test
    results = test_parameter_sensitivity()
    
    # Run parameter validation test
    test_parameter_validation()
    
    print("\n=== RECOMMENDATIONS ===")
    print("1. If sensitivity is HIGH: Consider parameter validation and cross-validation")
    print("2. If sensitivity is MODERATE: Document parameter choices and rationale")
    print("3. If sensitivity is LOW: Results are more likely to be robust")
    print("4. Always test with synthetic data to validate parameter robustness") 