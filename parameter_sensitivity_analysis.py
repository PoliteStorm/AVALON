#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis for √t Transform
Tests how much results depend on k and τ parameter choices.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def analyze_parameter_sensitivity():
    """Analyze how much results depend on k and τ parameter choices."""
    
    print("=== PARAMETER SENSITIVITY ANALYSIS ===")
    print("Testing how much results depend on k and τ choices\n")
    
    # Initialize analyzer
    analyzer = RigorousFungalAnalyzer("data/csv_data", "data/15061491/fungal_spikes/good_recordings")
    
    # Generate synthetic test signal with known patterns
    t = np.linspace(0, 3600, 3600)  # 1 hour at 1 Hz
    synthetic_signal = (
        0.1 * np.sin(2 * np.pi * 0.001 * t) +  # Very slow oscillation (0.001 Hz)
        0.05 * np.sin(2 * np.pi * 0.01 * t) +   # Slow oscillation (0.01 Hz)
        0.02 * np.sin(2 * np.pi * 0.1 * t) +    # Medium oscillation (0.1 Hz)
        0.01 * np.random.randn(len(t))           # Noise
    )
    
    # Test different parameter sets
    test_cases = [
        {
            'name': 'Broad Parameters',
            'description': 'Very broad parameter ranges',
            'k_range': np.logspace(-3, 2, 30),  # 0.001 to 100 Hz
            'tau_range': np.logspace(-2, 4, 30),  # 0.01 to 10000 seconds
            'amplitude_threshold': 0.03,
            'frequency_threshold': 0.001,
            'sqrt_scaling_factor': 1.0
        },
        {
            'name': 'Narrow Parameters',
            'description': 'Very narrow parameter ranges',
            'k_range': np.logspace(-0.5, 0.5, 30),  # 0.3 to 3 Hz
            'tau_range': np.logspace(0.5, 2.5, 30),  # 3 to 300 seconds
            'amplitude_threshold': 0.06,
            'frequency_threshold': 0.03,
            'sqrt_scaling_factor': 1.0
        },
        {
            'name': 'High Frequency Focus',
            'description': 'Focus on high frequency patterns',
            'k_range': np.logspace(0.0, 2.0, 30),  # 1 to 100 Hz
            'tau_range': np.logspace(-1, 2, 30),  # 0.1 to 100 seconds
            'amplitude_threshold': 0.04,
            'frequency_threshold': 0.05,
            'sqrt_scaling_factor': 1.2
        },
        {
            'name': 'Low Frequency Focus',
            'description': 'Focus on low frequency patterns',
            'k_range': np.logspace(-3, -0.5, 30),  # 0.001 to 0.3 Hz
            'tau_range': np.logspace(2, 5, 30),  # 100 to 100000 seconds
            'amplitude_threshold': 0.02,
            'frequency_threshold': 0.001,
            'sqrt_scaling_factor': 0.8
        },
        {
            'name': 'Your Current Parameters (Pi)',
            'description': 'Your species-specific parameters for Pi',
            'k_range': np.logspace(-1.0, 0.8, 30),  # 0.1 to 6 Hz
            'tau_range': np.logspace(0.2, 2.2, 30),  # 1.6 to 160 seconds
            'amplitude_threshold': 0.045,
            'frequency_threshold': 0.02,
            'sqrt_scaling_factor': 1.0
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
            avg_magnitude = np.mean([f['magnitude'] for f in features])
            avg_frequency = np.mean([f['frequency'] for f in features])
            avg_time_scale = np.mean([f['time_scale'] for f in features])
            avg_significance = np.mean([f['significance'] for f in features])
        else:
            avg_magnitude = avg_frequency = avg_time_scale = avg_significance = 0
        
        results[test_case['name']] = {
            'feature_count': feature_count,
            'avg_magnitude': avg_magnitude,
            'avg_frequency': avg_frequency,
            'avg_time_scale': avg_time_scale,
            'avg_significance': avg_significance,
            'parameters': test_case
        }
        
        print(f"  Features detected: {feature_count}")
        print(f"  Avg magnitude: {avg_magnitude:.4f}")
        print(f"  Avg frequency: {avg_frequency:.4f} Hz")
        print(f"  Avg time scale: {avg_time_scale:.2f} s")
        print(f"  Avg significance: {avg_significance:.2f}")
        print()
    
    # Analyze parameter sensitivity
    print("=== PARAMETER SENSITIVITY ANALYSIS ===")
    
    feature_counts = [results[name]['feature_count'] for name in results.keys()]
    avg_magnitudes = [results[name]['avg_magnitude'] for name in results.keys()]
    avg_frequencies = [results[name]['avg_frequency'] for name in results.keys()]
    
    print(f"Feature count range: {min(feature_counts)} to {max(feature_counts)}")
    print(f"Feature count variation: {max(feature_counts) - min(feature_counts)}")
    if np.mean(feature_counts) > 0:
        cv = np.std(feature_counts) / np.mean(feature_counts)
        print(f"Feature count coefficient of variation: {cv:.2f}")
        
        # Sensitivity assessment
        if cv > 0.5:
            print("⚠️  HIGH SENSITIVITY: Results heavily depend on parameter choices")
            print("   This suggests potential overfitting to parameter selection")
        elif cv > 0.2:
            print("⚠️  MODERATE SENSITIVITY: Results show some parameter dependence")
            print("   Consider parameter validation and cross-validation")
        else:
            print("✅  LOW SENSITIVITY: Results are relatively robust to parameter choices")
            print("   This suggests the method is finding real patterns")
    else:
        print("⚠️  NO FEATURES DETECTED: All parameter sets failed to detect features")
        print("   This suggests the method may be too restrictive")
    
    print(f"\nMagnitude range: {min(avg_magnitudes):.4f} to {max(avg_magnitudes):.4f}")
    print(f"Frequency range: {min(avg_frequencies):.4f} to {max(avg_frequencies):.4f} Hz")
    
    return results

def analyze_species_specific_parameters():
    """Analyze how species-specific parameters might enforce results."""
    
    print("\n=== SPECIES-SPECIFIC PARAMETER ANALYSIS ===")
    
    # Your current species-specific parameters
    species_params = {
        'Pv': {
            'k_range': np.logspace(-0.5, 1.2, 30),  # 0.3 to 16 Hz
            'tau_range': np.logspace(-0.3, 1.7, 30),  # 0.5 to 50 seconds
            'description': 'High frequency bursts'
        },
        'Pi': {
            'k_range': np.logspace(-1.0, 0.8, 30),  # 0.1 to 6 Hz
            'tau_range': np.logspace(0.2, 2.2, 30),  # 1.6 to 160 seconds
            'description': 'Medium frequency regular'
        },
        'Pp': {
            'k_range': np.logspace(-0.2, 1.5, 30),  # 0.6 to 32 Hz
            'tau_range': np.logspace(-0.5, 1.2, 30),  # 0.3 to 16 seconds
            'description': 'Very high frequency irregular'
        },
        'Rb': {
            'k_range': np.logspace(-2.5, -0.5, 30),  # 0.003 to 0.3 Hz
            'tau_range': np.logspace(1.5, 3.5, 30),  # 32 to 3200 seconds
            'description': 'Low frequency slow'
        },
        'Ag': {
            'k_range': np.logspace(-1.2, 0.6, 30),  # 0.06 to 4 Hz
            'tau_range': np.logspace(0.5, 2.5, 30),  # 3 to 300 seconds
            'description': 'Medium frequency steady'
        },
        'Sc': {
            'k_range': np.logspace(-1.3, 0.7, 30),  # 0.05 to 5 Hz
            'tau_range': np.logspace(0.3, 2.7, 30),  # 2 to 500 seconds
            'description': 'Medium frequency variable'
        }
    }
    
    print("Your species-specific parameter ranges:")
    for species, params in species_params.items():
        k_min, k_max = params['k_range'].min(), params['k_range'].max()
        tau_min, tau_max = params['tau_range'].min(), params['tau_range'].max()
        print(f"{species}: k = {k_min:.3f}-{k_max:.3f} Hz, τ = {tau_min:.1f}-{tau_max:.1f} s")
        print(f"  Description: {params['description']}")
    
    # Check for parameter overlap
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
    
    # Check if parameters are designed to find expected patterns
    print("\nParameter bias analysis:")
    for species, params in species_params.items():
        k_center = np.mean(params['k_range'])
        tau_center = np.mean(params['tau_range'])
        print(f"{species}: k_center = {k_center:.3f} Hz, τ_center = {tau_center:.1f} s")
        
        # Check if parameters are optimized for expected species characteristics
        if species == 'Pv' and k_center > 1.0:
            print("  ⚠️  Pv parameters biased toward high frequency")
        elif species == 'Rb' and k_center < 0.1:
            print("  ⚠️  Rb parameters biased toward low frequency")
        elif species == 'Sc' and 0.1 < k_center < 1.0:
            print("  ✅ Sc parameters allow medium frequency detection")

def test_uniform_parameters():
    """Test with uniform parameters across all species."""
    
    print("\n=== UNIFORM PARAMETER TEST ===")
    print("Testing with uniform parameters across all species...")
    
    # Create uniform parameter set
    uniform_params = {
        'k_range': np.logspace(-2, 1, 30),  # 0.01 to 10 Hz
        'tau_range': np.logspace(-1, 3, 30),  # 0.1 to 1000 seconds
        'amplitude_threshold': 0.05,
        'frequency_threshold': 0.01,
        'sqrt_scaling_factor': 1.0
    }
    
    # Generate synthetic signal
    t = np.linspace(0, 3600, 3600)
    synthetic_signal = (
        0.1 * np.sin(2 * np.pi * 0.001 * t) +
        0.05 * np.sin(2 * np.pi * 0.01 * t) +
        0.02 * np.sin(2 * np.pi * 0.1 * t) +
        0.01 * np.random.randn(len(t))
    )
    
    analyzer = RigorousFungalAnalyzer("data/csv_data", "data/15061491/fungal_spikes/good_recordings")
    
    # Test with uniform parameters
    params = {
        'k_range': uniform_params['k_range'],
        'tau_range': uniform_params['tau_range'],
        'amplitude_threshold': uniform_params['amplitude_threshold'],
        'frequency_threshold': uniform_params['frequency_threshold'],
        'sqrt_scaling_factor': uniform_params['sqrt_scaling_factor'],
        'window_function': 'gaussian',
        'detection_method': 'adaptive',
        'significance_threshold': 0.05,
        'time_scale_threshold': 0.1
    }
    
    transform_results = analyzer.apply_sqrt_transform(synthetic_signal, params, 1.0)
    features = transform_results['features']
    
    print(f"Uniform parameters detected {len(features)} features")
    if features:
        avg_freq = np.mean([f['frequency'] for f in features])
        avg_time = np.mean([f['time_scale'] for f in features])
        print(f"Average frequency: {avg_freq:.4f} Hz")
        print(f"Average time scale: {avg_time:.2f} s")
    
    print("\nRECOMMENDATION: Compare this with species-specific results")
    print("If uniform parameters detect similar patterns, this suggests robustness")
    print("If species-specific parameters detect very different patterns, this suggests bias")

if __name__ == "__main__":
    # Run parameter sensitivity analysis
    results = analyze_parameter_sensitivity()
    
    # Analyze species-specific parameters
    analyze_species_specific_parameters()
    
    # Test uniform parameters
    test_uniform_parameters()
    
    print("\n=== KEY FINDINGS ===")
    print("1. Parameter sensitivity indicates how much results depend on k/τ choices")
    print("2. High sensitivity suggests overfitting to parameter selection")
    print("3. Species-specific parameters may enforce expected results")
    print("4. Consider testing with uniform parameters across all species")
    print("5. Document parameter choices and rationale for transparency") 