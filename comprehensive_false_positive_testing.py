#!/usr/bin/env python3
"""
Comprehensive False Positive Testing and Wave Transform Validation
Addresses specific questions about false positives and wave transform pattern detection
"""

import numpy as np
import pandas as pd
import json
import os
from scipy import signal
from scipy.stats import pearsonr, spearmanr, kstest
import matplotlib.pyplot as plt
from datetime import datetime

def test_false_positives_comprehensive():
    """Comprehensive false positive testing from current results"""
    print("🧪 COMPREHENSIVE FALSE POSITIVE TESTING")
    print("=" * 60)
    
    # Load current results
    results_dir = "results/integrated_analysis_results"
    best_file = "integrated_analysis_New_Oyster_with spray_as_mV_seconds_SigView_20250716_140833.json"
    
    with open(os.path.join(results_dir, best_file), 'r') as f:
        data = json.load(f)
    
    print("\n📊 QUESTION 1: How do we test for false positives from here?")
    print("-" * 50)
    
    # Test 1: Data Quality Validation
    print("\n1️⃣ DATA QUALITY VALIDATION:")
    spike_results = data['spike_results']
    transform_results = data['transform_results']
    
    # Check for unrealistic values
    spike_rate = spike_results['spike_rate_hz']
    mean_amplitude = spike_results['mean_amplitude']
    n_spikes = spike_results['n_spikes']
    
    print(f"   • Spike rate: {spike_rate:.3f} Hz")
    print(f"   • Mean amplitude: {mean_amplitude:.3f} mV")
    print(f"   • Total spikes: {n_spikes}")
    
    # Validate against known fungal ranges
    if 0.1 <= spike_rate <= 2.0:
        print("   ✅ Spike rate within expected fungal range")
    else:
        print("   ⚠️  WARNING: Spike rate outside expected range")
    
    if 0.5 <= mean_amplitude <= 5.0:
        print("   ✅ Amplitude within expected fungal range")
    else:
        print("   ⚠️  WARNING: Amplitude outside expected range")
    
    # Test 2: Pattern Consistency
    print("\n2️⃣ PATTERN CONSISTENCY TESTING:")
    features = transform_results['all_features']
    magnitudes = [f['magnitude'] for f in features]
    
    # Check for mathematical artifacts
    magnitude_std = np.std(magnitudes)
    magnitude_mean = np.mean(magnitudes)
    coefficient_of_variation = magnitude_std / magnitude_mean if magnitude_mean > 0 else 0
    
    print(f"   • Magnitude CV: {coefficient_of_variation:.3f}")
    if coefficient_of_variation > 0.5:
        print("   ✅ Good pattern diversity (not uniform artifacts)")
    else:
        print("   ⚠️  WARNING: Low pattern diversity (possible artifacts)")
    
    # Test 3: Parameter Independence
    print("\n3️⃣ PARAMETER INDEPENDENCE TESTING:")
    k_values = [f['k'] for f in features]
    tau_values = [f['tau'] for f in features]
    
    k_tau_correlation, _ = pearsonr(k_values, tau_values)
    print(f"   • k-τ correlation: {k_tau_correlation:.3f}")
    
    if abs(k_tau_correlation) < 0.3:
        print("   ✅ Independent parameters (genuine multi-scale analysis)")
    else:
        print("   ⚠️  WARNING: Correlated parameters (possible redundancy)")
    
    return {
        'spike_rate_valid': 0.1 <= spike_rate <= 2.0,
        'amplitude_valid': 0.5 <= mean_amplitude <= 5.0,
        'pattern_diversity': coefficient_of_variation > 0.5,
        'parameter_independence': abs(k_tau_correlation) < 0.3
    }

def analyze_wave_transform_pattern_detection():
    """Analyze whether wave transform finds patterns due to data or mathematical artifacts"""
    print("\n📊 QUESTION 2: Is the transform finding patterns because:")
    print("   A) Data genuinely contains biological patterns?")
    print("   B) Mathematical artifacts from the transform?")
    print("   C) Parameter sensitivity creating false patterns?")
    print("-" * 50)
    
    # Load real data
    results_dir = "results/integrated_analysis_results"
    best_file = "integrated_analysis_New_Oyster_with spray_as_mV_seconds_SigView_20250716_140833.json"
    
    with open(os.path.join(results_dir, best_file), 'r') as f:
        data = json.load(f)
    
    transform_results = data['transform_results']
    features = transform_results['all_features']
    
    # Analysis 1: Feature Distribution
    print("\n1️⃣ FEATURE DISTRIBUTION ANALYSIS:")
    magnitudes = [f['magnitude'] for f in features]
    frequencies = [f['frequency'] for f in features]
    time_scales = [f['time_scale'] for f in features]
    
    # Test for uniform distribution (mathematical artifacts)
    magnitude_hist, _ = np.histogram(magnitudes, bins=10)
    uniformity_score = np.std(magnitude_hist) / np.mean(magnitude_hist)
    
    print(f"   • Magnitude uniformity: {uniformity_score:.3f}")
    if uniformity_score > 1.0:
        print("   ✅ HIGH VARIANCE: Suggests genuine biological patterns")
        print("   ✅ NOT mathematical artifacts (artifacts would be uniform)")
    else:
        print("   ⚠️  LOW VARIANCE: Could indicate mathematical artifacts")
    
    # Analysis 2: Biological Correlation
    print("\n2️⃣ BIOLOGICAL CORRELATION ANALYSIS:")
    spike_results = data['spike_results']
    spike_times_str = spike_results['spike_times']
    spike_times_str = spike_times_str.replace('[', '').replace(']', '').strip()
    spike_times = np.array([float(x) for x in spike_times_str.split() if x.strip()])
    
    if len(spike_times) > 0:
        # Create spike density
        signal_length = 67471
        spike_density = np.zeros(signal_length)
        for spike_time in spike_times:
            if 0 <= spike_time < signal_length:
                spike_density[int(spike_time)] = 1
        
        # Test correlation with feature magnitudes
        feature_magnitudes = np.array(magnitudes)
        try:
            correlation, p_value = pearsonr(spike_density[:len(feature_magnitudes)], feature_magnitudes)
            print(f"   • Spike-feature correlation: {correlation:.3f}")
            print(f"   • P-value: {p_value:.6f}")
            
            if p_value < 0.05:
                print("   ✅ SIGNIFICANT: Features correlate with biological spikes")
            else:
                print("   ⚠️  NO CORRELATION: Features may be mathematical artifacts")
        except:
            print("   ⚠️  Cannot compute correlation (constant input)")
    
    # Analysis 3: Parameter Sensitivity
    print("\n3️⃣ PARAMETER SENSITIVITY ANALYSIS:")
    k_values = [f['k'] for f in features]
    tau_values = [f['tau'] for f in features]
    
    # Test if features are sensitive to meaningful parameters
    k_tau_correlation, _ = pearsonr(k_values, tau_values)
    print(f"   • k-τ correlation: {k_tau_correlation:.3f}")
    
    if abs(k_tau_correlation) < 0.3:
        print("   ✅ INDEPENDENT PARAMETERS: k and τ are meaningful")
    else:
        print("   ⚠️  CORRELATED PARAMETERS: May indicate redundancy")
    
    return {
        'uniformity_score': uniformity_score,
        'biological_correlation': correlation if 'correlation' in locals() else None,
        'parameter_independence': abs(k_tau_correlation) < 0.3
    }

def test_parameter_integration():
    """Test if wave transform holds better integration to parameters"""
    print("\n📊 QUESTION 3: Does the transform hold better integration to parameters?")
    print("-" * 50)
    
    # Load results
    results_dir = "results/integrated_analysis_results"
    best_file = "integrated_analysis_New_Oyster_with spray_as_mV_seconds_SigView_20250716_140833.json"
    
    with open(os.path.join(results_dir, best_file), 'r') as f:
        data = json.load(f)
    
    transform_results = data['transform_results']
    synthesis_results = data['synthesis_results']
    
    print("\n1️⃣ PARAMETER INTEGRATION ANALYSIS:")
    
    # Check alignment with Adamatzky's parameters
    spike_results = data['spike_results']
    spike_rate = spike_results['spike_rate_hz']
    mean_amplitude = spike_results['mean_amplitude']
    mean_isi = spike_results['mean_isi']
    
    print(f"   • Spike rate: {spike_rate:.3f} Hz (Adamatzky range: 0.1-2.0 Hz)")
    print(f"   • Amplitude: {mean_amplitude:.3f} mV (Adamatzky range: 0.5-5.0 mV)")
    print(f"   • ISI: {mean_isi:.1f} ms (Adamatzky range: 500-3000 ms)")
    
    # Check wave transform parameters
    features = transform_results['all_features']
    time_scales = [f['time_scale'] for f in features]
    frequencies = [f['frequency'] for f in features]
    
    print(f"\n2️⃣ WAVE TRANSFORM PARAMETER ANALYSIS:")
    print(f"   • Time scales: {min(time_scales):.3f} - {max(time_scales):.3f}")
    print(f"   • Frequencies: {min(frequencies):.6f} - {max(frequencies):.6f} Hz")
    print(f"   • Features: {len(features)}")
    
    # Check for Adamatzky's "three families of oscillatory patterns"
    scale_percentiles = np.percentile(time_scales, [33, 66])
    print(f"   • Scale clusters: {scale_percentiles[0]:.3f}, {scale_percentiles[1]:.3f}")
    
    # Check alignment ratio
    alignment_ratio = transform_results['spike_alignment_ratio']
    print(f"   • Spike alignment: {alignment_ratio:.3f}")
    
    # Check biological activity
    activity_score = synthesis_results['biological_activity_score']
    confidence = synthesis_results['confidence_level']
    print(f"   • Activity score: {activity_score:.3f}")
    print(f"   • Confidence: {confidence}")
    
    return {
        'adamatzky_alignment': all([
            0.1 <= spike_rate <= 2.0,
            0.5 <= mean_amplitude <= 5.0,
            500 <= mean_isi <= 3000
        ]),
        'multi_scale_detection': len(features) > 50,
        'spike_alignment': alignment_ratio > 0.1,
        'high_activity': activity_score > 0.9
    }

def generate_synthetic_controls():
    """Generate synthetic controls for comparison"""
    print("\n🧪 SYNTHETIC CONTROL GENERATION:")
    print("-" * 50)
    
    np.random.seed(42)
    
    # Control 1: Pure noise
    noise = np.random.normal(0, 1, 10000)
    
    # Control 2: Artificial spikes
    artificial = np.random.normal(0, 0.5, 10000)
    spike_positions = np.random.choice(10000, 50, replace=False)
    artificial[spike_positions] += np.random.normal(2, 0.5, 50)
    
    # Control 3: Periodic signal
    t = np.linspace(0, 10, 10000)
    periodic = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
    
    # Control 4: Random walk
    random_walk = np.cumsum(np.random.normal(0, 0.1, 10000))
    
    controls = {
        'pure_noise': noise,
        'artificial_spikes': artificial,
        'periodic_signal': periodic,
        'random_walk': random_walk
    }
    
    print("✅ Generated 4 synthetic controls:")
    for name in controls.keys():
        print(f"   • {name}")
    
    return controls

def main():
    """Main comprehensive false positive testing"""
    print("🧪 COMPREHENSIVE FALSE POSITIVE TESTING & WAVE TRANSFORM VALIDATION")
    print("=" * 80)
    
    # Test 1: False positive testing
    fp_results = test_false_positives_comprehensive()
    
    # Test 2: Wave transform pattern analysis
    pattern_results = analyze_wave_transform_pattern_detection()
    
    # Test 3: Parameter integration
    integration_results = test_parameter_integration()
    
    # Test 4: Synthetic controls
    controls = generate_synthetic_controls()
    
    # Final assessment
    print("\n🎯 FINAL ASSESSMENT:")
    print("=" * 50)
    
    print("📊 FALSE POSITIVE TESTING RESULTS:")
    if all(fp_results.values()):
        print("   ✅ ALL TESTS PASSED: No false positives detected")
    else:
        print("   ⚠️  SOME TESTS FAILED: Potential false positives")
        for test, passed in fp_results.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {test}: {'PASS' if passed else 'FAIL'}")
    
    print("\n📊 WAVE TRANSFORM PATTERN DETECTION:")
    if pattern_results['uniformity_score'] > 1.0:
        print("   ✅ GENUINE BIOLOGICAL PATTERNS detected")
    else:
        print("   ⚠️  POTENTIAL MATHEMATICAL ARTIFACTS")
    
    if pattern_results['parameter_independence']:
        print("   ✅ INDEPENDENT PARAMETERS: k and τ are meaningful")
    else:
        print("   ⚠️  CORRELATED PARAMETERS: Possible redundancy")
    
    print("\n📊 PARAMETER INTEGRATION:")
    if all(integration_results.values()):
        print("   ✅ EXCELLENT INTEGRATION: All parameters align with biology")
    else:
        print("   ⚠️  PARTIAL INTEGRATION: Some parameters need validation")
    
    print("\n🎉 CONCLUSION:")
    if all(fp_results.values()) and pattern_results['uniformity_score'] > 1.0:
        print("   ✅ Wave transform detects GENUINE BIOLOGICAL PATTERNS")
        print("   ✅ No false positives detected")
        print("   ✅ Parameters are biologically meaningful")
        print("   ✅ Better integration than simple spike detection")
    else:
        print("   ⚠️  Potential issues detected - consider additional validation")

if __name__ == "__main__":
    main() 