#!/usr/bin/env python3
"""
Simple Parameter Sensitivity Test
Tests how much results depend on k and τ parameter choices.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def test_parameter_sensitivity():
    """Test how much results depend on k and τ parameter choices."""
    
    print("=== PARAMETER SENSITIVITY TEST ===")
    print("Testing how much results depend on k and τ choices\n")
    
    # Initialize analyzer
    analyzer = RigorousFungalAnalyzer("data/csv_data", "data/15061491/fungal_spikes/good_recordings")
    
    # Generate synthetic test signal (more reliable than real data for this test)
    t = np.linspace(0, 3600, 3600)  # 1 hour at 1 Hz
    synthetic_signal = (
        0.1 * np.sin(2 * np.pi * 0.001 * t) +  # Very slow oscillation
        0.05 * np.sin(2 * np.pi * 0.01 * t) +   # Slow oscillation
        0.02 * np.sin(2 * np.pi * 0.1 * t) +    # Medium oscillation
        0.01 * np.random.randn(len(t))           # Noise
    )
    
    # Test different parameter sets
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
        
        # Apply transform with these parameters
        transform_results = analyzer.apply_sqrt_transform(synthetic_signal, params, 1.0)
        
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
    if np.mean(feature_counts) > 0:
        print(f"Feature count coefficient of variation: {np.std(feature_counts) / np.mean(feature_counts):.2f}")
    
    print(f"\nAmplitude range: {min(avg_amplitudes):.4f} to {max(avg_amplitudes):.4f}")
    print(f"Frequency range: {min(avg_frequencies):.4f} to {max(avg_frequencies):.4f} Hz")
    
    # Calculate sensitivity metrics
    if np.mean(feature_counts) > 0:
        sensitivity_score = (max(feature_counts) - min(feature_counts)) / np.mean(feature_counts)
        
        print(f"\n=== SENSITIVITY ASSESSMENT ===")
        if sensitivity_score > 0.5:
            print("⚠️  HIGH SENSITIVITY: Results heavily depend on parameter choices")
            print("   This suggests the method may be overfitting to parameter selection")
            print("   RECOMMENDATION: Use cross-validation and parameter validation")
        elif sensitivity_score > 0.2:
            print("⚠️  MODERATE SENSITIVITY: Results show some parameter dependence")
            print("   RECOMMENDATION: Document parameter choices and rationale")
        else:
            print("✅  LOW SENSITIVITY: Results are relatively robust to parameter choices")
            print("   This suggests the method is finding real patterns")
    else:
        print("⚠️  NO FEATURES DETECTED: All parameter sets failed to detect features")
        print("   This suggests the method may be too restrictive")
    
    return results

def analyze_parameter_ranges():
    """Analyze the parameter ranges used in your analysis."""
    
    print("\n=== PARAMETER RANGE ANALYSIS ===")
    
    # Your current parameter ranges
    species_params = {
        'Pv': {'k_range': np.logspace(-0.5, 1.2, 30), 'tau_range': np.logspace(-0.3, 1.7, 30)},
        'Pi': {'k_range': np.logspace(-1.0, 0.8, 30), 'tau_range': np.logspace(0.2, 2.2, 30)},
        'Pp': {'k_range': np.logspace(-0.2, 1.5, 30), 'tau_range': np.logspace(-0.5, 1.2, 30)},
        'Rb': {'k_range': np.logspace(-2.5, -0.5, 30), 'tau_range': np.logspace(1.5, 3.5, 30)},
        'Ag': {'k_range': np.logspace(-1.2, 0.6, 30), 'tau_range': np.logspace(0.5, 2.5, 30)},
        'Sc': {'k_range': np.logspace(-1.3, 0.7, 30), 'tau_range': np.logspace(0.3, 2.7, 30)}
    }
    
    print("Your species-specific parameter ranges:")
    for species, params in species_params.items():
        k_min, k_max = params['k_range'].min(), params['k_range'].max()
        tau_min, tau_max = params['tau_range'].min(), params['tau_range'].max()
        print(f"{species}: k = {k_min:.3f}-{k_max:.3f} Hz, τ = {tau_min:.1f}-{tau_max:.1f} s")
    
    # Check for overlap
    print("\nParameter overlap analysis:")
    all_k_ranges = [params['k_range'] for params in species_params.values()]
    all_tau_ranges = [params['tau_range'] for params in species_params.values()]
    
    k_overlap_min = max([k_range.min() for k_range in all_k_ranges])
    k_overlap_max = min([k_range.max() for k_range in all_k_ranges])
    tau_overlap_min = max([tau_range.min() for tau_range in all_tau_ranges])
    tau_overlap_max = min([tau_range.max() for tau_range in all_tau_ranges])
    
    print(f"k overlap: {k_overlap_min:.3f}-{k_overlap_max:.3f} Hz")
    print(f"τ overlap: {tau_overlap_min:.1f}-{tau_overlap_max:.1f} s")
    
    if k_overlap_max > k_overlap_min and tau_overlap_max > tau_overlap_min:
        print("✅ Good parameter overlap - species can be compared")
    else:
        print("⚠️  Limited parameter overlap - species comparison may be biased")

if __name__ == "__main__":
    # Run parameter sensitivity test
    results = test_parameter_sensitivity()
    
    # Analyze parameter ranges
    analyze_parameter_ranges()
    
    print("\n=== KEY INSIGHTS ===")
    print("1. Parameter sensitivity indicates how much your results depend on k/τ choices")
    print("2. High sensitivity suggests overfitting to parameter selection")
    print("3. Low sensitivity suggests robust pattern detection")
    print("4. Species-specific parameters may enforce expected results")
    print("5. Consider testing with uniform parameters across all species") 