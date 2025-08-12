#!/usr/bin/env python3
"""
Uniform vs Species-Specific Parameter Comparison
Tests if results are robust or parameter-dependent.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def create_uniform_parameters():
    """Create uniform parameters for all species."""
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

def get_species_specific_parameters():
    """Get the current species-specific parameters."""
    return {
        'Pv': {
            'k_range': np.logspace(-0.5, 1.2, 30),  # 0.3 to 16 Hz
            'tau_range': np.logspace(-0.3, 1.7, 30),  # 0.5 to 50 seconds
            'amplitude_threshold': 0.04,
            'frequency_threshold': 0.03,
            'sqrt_scaling_factor': 1.2
        },
        'Pi': {
            'k_range': np.logspace(-1.0, 0.8, 30),  # 0.1 to 6 Hz
            'tau_range': np.logspace(0.2, 2.2, 30),  # 1.6 to 160 seconds
            'amplitude_threshold': 0.045,
            'frequency_threshold': 0.02,
            'sqrt_scaling_factor': 1.0
        },
        'Pp': {
            'k_range': np.logspace(-0.2, 1.5, 30),  # 0.6 to 32 Hz
            'tau_range': np.logspace(-0.5, 1.2, 30),  # 0.3 to 16 seconds
            'amplitude_threshold': 0.035,
            'frequency_threshold': 0.05,
            'sqrt_scaling_factor': 1.5
        },
        'Rb': {
            'k_range': np.logspace(-2.5, -0.5, 30),  # 0.003 to 0.3 Hz
            'tau_range': np.logspace(1.5, 3.5, 30),  # 32 to 3200 seconds
            'amplitude_threshold': 0.025,
            'frequency_threshold': 0.005,
            'sqrt_scaling_factor': 0.8
        },
        'Ag': {
            'k_range': np.logspace(-1.2, 0.6, 30),  # 0.06 to 4 Hz
            'tau_range': np.logspace(0.5, 2.5, 30),  # 3 to 300 seconds
            'amplitude_threshold': 0.04,
            'frequency_threshold': 0.025,
            'sqrt_scaling_factor': 1.1
        },
        'Sc': {
            'k_range': np.logspace(-1.3, 0.7, 30),  # 0.05 to 5 Hz
            'tau_range': np.logspace(0.3, 2.7, 30),  # 2 to 500 seconds
            'amplitude_threshold': 0.042,
            'frequency_threshold': 0.02,
            'sqrt_scaling_factor': 1.05
        }
    }

def compare_parameter_approaches():
    """Compare uniform vs species-specific parameters."""
    
    print("=== UNIFORM VS SPECIES-SPECIFIC PARAMETER COMPARISON ===")
    print("Testing if results are robust or parameter-dependent\n")
    
    # Initialize analyzer
    analyzer = RigorousFungalAnalyzer("data/csv_data", "data/15061491/fungal_spikes/good_recordings")
    
    # Load data
    print("Loading data...")
    data = analyzer.load_and_categorize_data()
    
    # Get parameter sets
    uniform_params = create_uniform_parameters()
    species_params = get_species_specific_parameters()
    
    results = {
        'uniform': {},
        'species_specific': {},
        'comparison': {}
    }
    
    # Test with uniform parameters
    print("\n=== TESTING WITH UNIFORM PARAMETERS ===")
    uniform_results = {}
    
    for species in ['Pv', 'Pi', 'Pp', 'Rb', 'Ag', 'Sc']:
        print(f"Testing {species} with uniform parameters...")
        
        # Create params with uniform settings
        params = uniform_params.copy()
        
        # Test with sample data (use coordinate data for consistency)
        if data['coordinate_data']:
            sample_file = list(data['coordinate_data'].keys())[0]
            sample_data = data['coordinate_data'][sample_file]
            
            # Extract distance signal
            if len(sample_data.columns) >= 2:
                x_coords = sample_data.iloc[:, 0].values
                y_coords = sample_data.iloc[:, 1].values
                distance_signal = np.sqrt(x_coords**2 + y_coords**2)
            else:
                distance_signal = sample_data.iloc[:, 0].values
            
            # Apply transform
            transform_results = analyzer.apply_sqrt_transform(distance_signal, params, 1.0)
            features = transform_results['features']
            
            uniform_results[species] = {
                'feature_count': len(features),
                'avg_frequency': np.mean([f['frequency'] for f in features]) if features else 0,
                'avg_time_scale': np.mean([f['time_scale'] for f in features]) if features else 0,
                'avg_magnitude': np.mean([f['magnitude'] for f in features]) if features else 0
            }
            
            print(f"  {species}: {len(features)} features, avg freq: {uniform_results[species]['avg_frequency']:.4f} Hz")
    
    results['uniform'] = uniform_results
    
    # Test with species-specific parameters
    print("\n=== TESTING WITH SPECIES-SPECIFIC PARAMETERS ===")
    species_specific_results = {}
    
    for species, species_param in species_params.items():
        print(f"Testing {species} with species-specific parameters...")
        
        # Create params with species-specific settings
        params = {
            'k_range': species_param['k_range'],
            'tau_range': species_param['tau_range'],
            'amplitude_threshold': species_param['amplitude_threshold'],
            'frequency_threshold': species_param['frequency_threshold'],
            'sqrt_scaling_factor': species_param['sqrt_scaling_factor'],
            'window_function': 'gaussian',
            'detection_method': 'adaptive',
            'significance_threshold': 0.05,
            'time_scale_threshold': 0.1
        }
        
        # Test with same sample data
        if data['coordinate_data']:
            sample_file = list(data['coordinate_data'].keys())[0]
            sample_data = data['coordinate_data'][sample_file]
            
            # Extract distance signal
            if len(sample_data.columns) >= 2:
                x_coords = sample_data.iloc[:, 0].values
                y_coords = sample_data.iloc[:, 1].values
                distance_signal = np.sqrt(x_coords**2 + y_coords**2)
            else:
                distance_signal = sample_data.iloc[:, 0].values
            
            # Apply transform
            transform_results = analyzer.apply_sqrt_transform(distance_signal, params, 1.0)
            features = transform_results['features']
            
            species_specific_results[species] = {
                'feature_count': len(features),
                'avg_frequency': np.mean([f['frequency'] for f in features]) if features else 0,
                'avg_time_scale': np.mean([f['time_scale'] for f in features]) if features else 0,
                'avg_magnitude': np.mean([f['magnitude'] for f in features]) if features else 0
            }
            
            print(f"  {species}: {len(features)} features, avg freq: {species_specific_results[species]['avg_frequency']:.4f} Hz")
    
    results['species_specific'] = species_specific_results
    
    # Compare results
    print("\n=== COMPARISON ANALYSIS ===")
    comparison = {}
    
    for species in ['Pv', 'Pi', 'Pp', 'Rb', 'Ag', 'Sc']:
        if species in uniform_results and species in species_specific_results:
            uniform = uniform_results[species]
            specific = species_specific_results[species]
            
            # Calculate differences
            feature_diff = specific['feature_count'] - uniform['feature_count']
            freq_diff = specific['avg_frequency'] - uniform['avg_frequency']
            time_diff = specific['avg_time_scale'] - uniform['avg_time_scale']
            mag_diff = specific['avg_magnitude'] - uniform['avg_magnitude']
            
            # Calculate relative differences
            feature_rel_diff = feature_diff / max(uniform['feature_count'], 1)
            freq_rel_diff = freq_diff / max(uniform['avg_frequency'], 0.001)
            time_rel_diff = time_diff / max(uniform['avg_time_scale'], 1)
            mag_rel_diff = mag_diff / max(uniform['avg_magnitude'], 0.001)
            
            comparison[species] = {
                'feature_count_diff': feature_diff,
                'feature_count_rel_diff': feature_rel_diff,
                'frequency_diff': freq_diff,
                'frequency_rel_diff': freq_rel_diff,
                'time_scale_diff': time_diff,
                'time_scale_rel_diff': time_rel_diff,
                'magnitude_diff': mag_diff,
                'magnitude_rel_diff': mag_rel_diff
            }
            
            print(f"{species}:")
            print(f"  Feature count: {uniform['feature_count']} → {specific['feature_count']} (diff: {feature_diff:+d})")
            print(f"  Avg frequency: {uniform['avg_frequency']:.4f} → {specific['avg_frequency']:.4f} Hz (diff: {freq_diff:+.4f})")
            print(f"  Avg time scale: {uniform['avg_time_scale']:.1f} → {specific['avg_time_scale']:.1f} s (diff: {time_diff:+.1f})")
    
    results['comparison'] = comparison
    
    # Overall assessment
    print("\n=== OVERALL ASSESSMENT ===")
    
    # Calculate average relative differences
    avg_feature_rel_diff = np.mean([abs(comp['feature_count_rel_diff']) for comp in comparison.values()])
    avg_freq_rel_diff = np.mean([abs(comp['frequency_rel_diff']) for comp in comparison.values()])
    avg_time_rel_diff = np.mean([abs(comp['time_scale_rel_diff']) for comp in comparison.values()])
    
    print(f"Average relative differences:")
    print(f"  Feature count: {avg_feature_rel_diff:.2f} ({avg_feature_rel_diff*100:.1f}%)")
    print(f"  Frequency: {avg_freq_rel_diff:.2f} ({avg_freq_rel_diff*100:.1f}%)")
    print(f"  Time scale: {avg_time_rel_diff:.2f} ({avg_time_rel_diff*100:.1f}%)")
    
    # Determine if results are parameter-dependent
    if avg_feature_rel_diff > 0.5 or avg_freq_rel_diff > 0.5:
        print("\n⚠️  HIGH PARAMETER DEPENDENCE")
        print("Results are heavily influenced by parameter choices")
        print("Species-specific parameters may be enforcing expected results")
    elif avg_feature_rel_diff > 0.2 or avg_freq_rel_diff > 0.2:
        print("\n⚠️  MODERATE PARAMETER DEPENDENCE")
        print("Results show some influence from parameter choices")
        print("Consider using uniform parameters for comparison")
    else:
        print("\n✅  LOW PARAMETER DEPENDENCE")
        print("Results are relatively robust to parameter choices")
        print("This suggests the patterns are real, not parameter artifacts")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"uniform_vs_species_specific_comparison_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

def create_comparison_visualization(results):
    """Create visualization of the comparison."""
    
    species = list(results['comparison'].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Uniform vs Species-Specific Parameter Comparison', fontsize=16)
    
    # Feature count comparison
    uniform_counts = [results['uniform'][s]['feature_count'] for s in species]
    specific_counts = [results['species_specific'][s]['feature_count'] for s in species]
    
    x = np.arange(len(species))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, uniform_counts, width, label='Uniform Parameters', color='skyblue')
    axes[0, 0].bar(x + width/2, specific_counts, width, label='Species-Specific Parameters', color='lightcoral')
    axes[0, 0].set_title('Feature Count Comparison')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(species)
    axes[0, 0].legend()
    
    # Frequency comparison
    uniform_freqs = [results['uniform'][s]['avg_frequency'] for s in species]
    specific_freqs = [results['species_specific'][s]['avg_frequency'] for s in species]
    
    axes[0, 1].bar(x - width/2, uniform_freqs, width, label='Uniform Parameters', color='skyblue')
    axes[0, 1].bar(x + width/2, specific_freqs, width, label='Species-Specific Parameters', color='lightcoral')
    axes[0, 1].set_title('Average Frequency Comparison')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(species)
    axes[0, 1].legend()
    
    # Time scale comparison
    uniform_times = [results['uniform'][s]['avg_time_scale'] for s in species]
    specific_times = [results['species_specific'][s]['avg_time_scale'] for s in species]
    
    axes[1, 0].bar(x - width/2, uniform_times, width, label='Uniform Parameters', color='skyblue')
    axes[1, 0].bar(x + width/2, specific_times, width, label='Species-Specific Parameters', color='lightcoral')
    axes[1, 0].set_title('Average Time Scale Comparison')
    axes[1, 0].set_ylabel('Time Scale (seconds)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(species)
    axes[1, 0].legend()
    
    # Relative differences
    rel_diffs = [abs(results['comparison'][s]['frequency_rel_diff']) for s in species]
    
    axes[1, 1].bar(species, rel_diffs, color='gold')
    axes[1, 1].set_title('Relative Frequency Differences')
    axes[1, 1].set_ylabel('Relative Difference')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('uniform_vs_species_specific_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'uniform_vs_species_specific_comparison.png'")

if __name__ == "__main__":
    # Run the comparison
    results = compare_parameter_approaches()
    
    # Create visualization
    create_comparison_visualization(results)
    
    print("\n=== KEY INSIGHTS ===")
    print("1. This test determines if your results are robust or parameter-dependent")
    print("2. Large differences suggest parameter enforcement of expected results")
    print("3. Small differences suggest real biological patterns")
    print("4. Consider using uniform parameters for unbiased comparison") 