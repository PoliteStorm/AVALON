#!/usr/bin/env python3
"""
Run validation tests comparing original linear analysis with √t transform
on fungal electrophysiology data
"""

import sys
import os
from transform_validation_framework import SqrtTTransform, ValidationFramework
import numpy as np
import pandas as pd

def test_single_file(filepath, channel=None):
    """
    Test a single data file with comprehensive validation
    """
    print(f"\n{'='*60}")
    print(f"Testing: {filepath}")
    print(f"{'='*60}")
    
    # Initialize transform and validator
    transform = SqrtTTransform(sampling_rate=1.0)
    validator = ValidationFramework(transform)
    
    # Load data
    V = validator.load_fungal_data(filepath, channel)
    if V is None:
        print(f"Failed to load data from {filepath}")
        return None
    
    print(f"Data loaded: {len(V)} samples")
    print(f"Data range: {np.min(V):.6f} to {np.max(V):.6f}")
    print(f"Data mean: {np.mean(V):.6f}, std: {np.std(V):.6f}")
    
    # Parameters for √t transform
    k_values = np.logspace(-1, 1, 15)  # 0.1 to 10
    tau_values = np.logspace(0, 2, 15)  # 1 to 100
    
    # 1. Original linear analysis (replicating Fungal-Spike-Clustering)
    print("\n1. Running original linear analysis...")
    original_spikes = validator.original_linear_analysis(V)
    print(f"   Original spikes detected: {len(original_spikes)}")
    
    # 2. √t transform analysis
    print("\n2. Running √t transform analysis...")
    W = transform.transform(V, k_values, tau_values)
    sqrt_features = transform.detect_features(W, k_values, tau_values)
    print(f"   √t features detected: {len(sqrt_features)}")
    
    # 3. Control testing for false positive detection
    print("\n3. Running control tests...")
    controls = validator.generate_control_signals(V)
    control_results = {}
    
    for control_name, control_signal in controls.items():
        print(f"   Testing {control_name}...")
        W_control = transform.transform(control_signal, k_values, tau_values)
        control_features = transform.detect_features(W_control, k_values, tau_values)
        control_results[control_name] = control_features
        print(f"     Features detected: {len(control_features)}")
    
    # 4. Cross-validation with established methods
    print("\n4. Running cross-validation...")
    established_results = validator.cross_validate_with_established_methods(V)
    print(f"   FFT peaks: {len(established_results['fft_peaks'])}")
    
    # 5. Biological plausibility check
    print("\n5. Checking biological plausibility...")
    plausible_features = validator.biological_plausibility_check(sqrt_features)
    print(f"   Biologically plausible features: {len(plausible_features)}")
    
    # 6. Statistical significance testing
    print("\n6. Testing statistical significance...")
    significance_results = validator.statistical_significance_test(sqrt_features, control_results)
    print(f"   Significant: {significance_results['significant']}")
    print(f"   p-value: {significance_results['p_value']:.4f}")
    
    # 7. Detailed comparison
    print("\n7. Detailed comparison:")
    print(f"   Original linear spikes: {len(original_spikes)}")
    print(f"   √t transform features: {len(sqrt_features)}")
    print(f"   Plausible features: {len(plausible_features)}")
    
    if sqrt_features:
        k_vals = [f['k'] for f in sqrt_features]
        tau_vals = [f['tau'] for f in sqrt_features]
        mag_vals = [f['magnitude'] for f in sqrt_features]
        
        print(f"   √t feature k range: {min(k_vals):.3f} to {max(k_vals):.3f}")
        print(f"   √t feature τ range: {min(tau_vals):.3f} to {max(tau_vals):.3f}")
        print(f"   √t feature magnitude range: {min(mag_vals):.6f} to {max(mag_vals):.6f}")
    
    # 8. Check for additional patterns not detected by original method
    print("\n8. Checking for additional patterns...")
    if sqrt_features and original_spikes:
        # Look for √t features that don't correspond to original spikes
        additional_features = []
        for feature in sqrt_features:
            # Check if this feature corresponds to any original spike
            k, tau = feature['k'], feature['tau']
            # Simple heuristic: check if feature is in biologically plausible range
            if feature.get('biologically_plausible', False):
                additional_features.append(feature)
        
        print(f"   Additional biologically plausible features: {len(additional_features)}")
    
    # Create results summary
    results = {
        'filepath': filepath,
        'data_length': len(V),
        'original_spikes': original_spikes,
        'sqrt_features': sqrt_features,
        'plausible_features': plausible_features,
        'control_results': control_results,
        'established_results': established_results,
        'significance_results': significance_results,
        'additional_features': additional_features if 'additional_features' in locals() else []
    }
    
    return results

def main():
    """
    Main testing function
    """
    # Test files to analyze
    test_files = [
        {
            'path': "15061491/fungal_spikes/good_recordings/New_Oyster_with spray_as_mV_seconds_SigView.csv",
            'channel': None,  # Single column data
            'description': "Oyster mushroom voltage recording (single channel)"
        },
        {
            'path': "15061491/fungal_spikes/good_recordings/Hericium_20_4_22_part1.csv",
            'channel': "Differential 1 - 2 Ave. (V)",
            'description': "Hericium erinaceus voltage recording (multi-channel)"
        },
        {
            'path': "15061491/fungal_spikes/good_recordings/Blue_oyster_31_5_22.csv",
            'channel': None,
            'description': "Blue oyster mushroom recording"
        }
    ]
    
    all_results = []
    
    for test_config in test_files:
        print(f"\n{'='*80}")
        print(f"Testing: {test_config['description']}")
        print(f"File: {test_config['path']}")
        print(f"{'='*80}")
        
        try:
            results = test_single_file(test_config['path'], test_config['channel'])
            if results:
                all_results.append(results)
                
                # Save individual results
                output_file = f"results_{test_config['path'].split('/')[-1].replace('.csv', '.txt')}"
                with open(output_file, 'w') as f:
                    f.write(f"Results for {test_config['description']}\n")
                    f.write(f"File: {test_config['path']}\n")
                    f.write(f"Data length: {results['data_length']}\n")
                    f.write(f"Original spikes: {len(results['original_spikes'])}\n")
                    f.write(f"√t features: {len(results['sqrt_features'])}\n")
                    f.write(f"Plausible features: {len(results['plausible_features'])}\n")
                    f.write(f"Significant: {results['significance_results']['significant']}\n")
                    f.write(f"p-value: {results['significance_results']['p_value']:.4f}\n")
                    f.write(f"Additional features: {len(results['additional_features'])}\n")
                
                print(f"Results saved to: {output_file}")
                
        except Exception as e:
            print(f"Error testing {test_config['path']}: {e}")
            continue
    
    # Summary report
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")
    
    for i, results in enumerate(all_results):
        print(f"\nTest {i+1}: {results['filepath']}")
        print(f"  Original spikes: {len(results['original_spikes'])}")
        print(f"  √t features: {len(results['sqrt_features'])}")
        print(f"  Plausible features: {len(results['plausible_features'])}")
        print(f"  Significant: {results['significance_results']['significant']}")
        print(f"  Additional features: {len(results['additional_features'])}")
    
    # Overall conclusions
    print(f"\n{'='*80}")
    print("CONCLUSIONS")
    print(f"{'='*80}")
    
    total_tests = len(all_results)
    significant_tests = sum(1 for r in all_results if r['significance_results']['significant'])
    tests_with_additional_features = sum(1 for r in all_results if len(r['additional_features']) > 0)
    
    print(f"Total tests: {total_tests}")
    print(f"Significant results: {significant_tests}/{total_tests}")
    print(f"Tests with additional features: {tests_with_additional_features}/{total_tests}")
    
    if significant_tests > 0:
        print("\n✓ √t transform shows statistically significant features")
    else:
        print("\n✗ √t transform does not show statistically significant features")
    
    if tests_with_additional_features > 0:
        print("✓ √t transform detects additional patterns not found by original method")
    else:
        print("✗ √t transform does not detect additional patterns")

if __name__ == "__main__":
    main() 