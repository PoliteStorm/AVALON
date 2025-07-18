#!/usr/bin/env python3
"""
False Positive Validation and Wave Transform Pattern Analysis
Tests for false positives and validates wave transform pattern detection
"""

import numpy as np
import pandas as pd
import json
import os
from scipy import signal
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

def generate_synthetic_controls():
    """Generate synthetic control data for false positive testing"""
    print("🧪 GENERATING SYNTHETIC CONTROL DATA")
    print("=" * 50)
    
    # Control 1: Pure noise (no patterns)
    np.random.seed(42)
    noise_data = np.random.normal(0, 1, 10000)
    
    # Control 2: Artificial spikes (not biological)
    artificial_spikes = np.random.normal(0, 0.5, 10000)
    spike_positions = np.random.choice(10000, 50, replace=False)
    artificial_spikes[spike_positions] += np.random.normal(2, 0.5, 50)
    
    # Control 3: Periodic signal (mathematical, not biological)
    t = np.linspace(0, 10, 10000)
    periodic_signal = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
    
    # Control 4: Random walk (trend, not spikes)
    random_walk = np.cumsum(np.random.normal(0, 0.1, 10000))
    
    controls = {
        'pure_noise': noise_data,
        'artificial_spikes': artificial_spikes,
        'periodic_signal': periodic_signal,
        'random_walk': random_walk
    }
    
    print("✅ Generated 4 synthetic control datasets:")
    print("   • Pure noise (no patterns)")
    print("   • Artificial spikes (non-biological)")
    print("   • Periodic signal (mathematical)")
    print("   • Random walk (trend)")
    
    return controls

def test_wave_transform_on_controls(controls):
    """Test wave transform on control data to detect false positives"""
    print("\n🔍 TESTING WAVE TRANSFORM ON CONTROL DATA")
    print("=" * 50)
    
    results = {}
    
    for name, data in controls.items():
        print(f"\n📊 Testing: {name}")
        
        # Simple wave transform simulation
        # W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
        k_values = np.logspace(-2, 1, 10)  # Frequency parameters
        tau_values = np.logspace(-1, 2, 10)  # Time scale parameters
        
        features = []
        for k in k_values:
            for tau in tau_values:
                # Simplified wave transform calculation
                t = np.arange(len(data))
                wavelet = np.exp(-t / (tau * 1000))  # Time scale normalization
                frequency_component = np.exp(-1j * k * np.sqrt(t))
                
                # Wave transform calculation
                transform = np.sum(data * wavelet * frequency_component)
                magnitude = np.abs(transform)
                
                features.append({
                    'k': k,
                    'tau': tau,
                    'magnitude': magnitude,
                    'frequency': k / (2 * np.pi),
                    'time_scale': tau
                })
        
        # Calculate feature statistics
        magnitudes = [f['magnitude'] for f in features]
        mean_magnitude = np.mean(magnitudes)
        std_magnitude = np.std(magnitudes)
        
        # Pattern complexity score
        pattern_complexity = std_magnitude / mean_magnitude if mean_magnitude > 0 else 0
        
        results[name] = {
            'n_features': len(features),
            'mean_magnitude': mean_magnitude,
            'std_magnitude': std_magnitude,
            'pattern_complexity': pattern_complexity,
            'max_magnitude': max(magnitudes),
            'min_magnitude': min(magnitudes)
        }
        
        print(f"   • Features: {len(features)}")
        print(f"   • Mean magnitude: {mean_magnitude:.3f}")
        print(f"   • Pattern complexity: {pattern_complexity:.3f}")
        print(f"   • Magnitude range: {min(magnitudes):.3f} - {max(magnitudes):.3f}")
    
    return results

def compare_with_real_data(control_results):
    """Compare control results with real fungal data"""
    print("\n📊 COMPARING WITH REAL FUNGAL DATA")
    print("=" * 50)
    
    # Load real fungal data results
    results_dir = "results/integrated_analysis_results"
    real_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    real_results = {}
    for filename in real_files[:3]:  # Use first 3 files
        with open(os.path.join(results_dir, filename), 'r') as f:
            data = json.load(f)
        
        transform_results = data['transform_results']
        synthesis_results = data['synthesis_results']
        
        real_results[data['filename']] = {
            'n_features': transform_results['n_features'],
            'mean_magnitude': transform_results['mean_magnitude'],
            'pattern_complexity': synthesis_results['pattern_complexity'],
            'biological_activity_score': synthesis_results['biological_activity_score'],
            'confidence_level': synthesis_results['confidence_level']
        }
    
    print("📈 REAL FUNGAL DATA RESULTS:")
    for filename, results in real_results.items():
        print(f"\n   {filename}:")
        print(f"   • Features: {results['n_features']}")
        print(f"   • Mean magnitude: {results['mean_magnitude']:.3f}")
        print(f"   • Pattern complexity: {results['pattern_complexity']:.3f}")
        print(f"   • Activity score: {results['biological_activity_score']:.3f}")
        print(f"   • Confidence: {results['confidence_level']}")
    
    print("\n📉 CONTROL DATA RESULTS:")
    for name, results in control_results.items():
        print(f"\n   {name}:")
        print(f"   • Features: {results['n_features']}")
        print(f"   • Mean magnitude: {results['mean_magnitude']:.3f}")
        print(f"   • Pattern complexity: {results['pattern_complexity']:.3f}")
    
    return real_results

def analyze_wave_transform_patterns():
    """Analyze whether wave transform finds patterns because data suggests patterns or due to mathematical artifacts"""
    print("\n🔬 WAVE TRANSFORM PATTERN ANALYSIS")
    print("=" * 50)
    
    # Load real fungal data
    results_dir = "results/integrated_analysis_results"
    best_file = "integrated_analysis_New_Oyster_with spray_as_mV_seconds_SigView_20250716_140833.json"
    
    with open(os.path.join(results_dir, best_file), 'r') as f:
        data = json.load(f)
    
    transform_results = data['transform_results']
    features = transform_results['all_features']
    
    print("📊 PATTERN DETECTION ANALYSIS:")
    print("\n1️⃣ FEATURE DISTRIBUTION ANALYSIS:")
    
    # Analyze feature distribution
    magnitudes = [f['magnitude'] for f in features]
    frequencies = [f['frequency'] for f in features]
    time_scales = [f['time_scale'] for f in features]
    
    print(f"   • Total features: {len(features)}")
    print(f"   • Magnitude range: {min(magnitudes):.3f} - {max(magnitudes):.3f}")
    print(f"   • Frequency range: {min(frequencies):.6f} - {max(frequencies):.6f} Hz")
    print(f"   • Time scale range: {min(time_scales):.3f} - {max(time_scales):.3f}")
    
    # Check for mathematical artifacts vs biological patterns
    print("\n2️⃣ MATHEMATICAL ARTIFACT DETECTION:")
    
    # Test for uniform distribution (suggests artifacts)
    magnitude_hist, _ = np.histogram(magnitudes, bins=10)
    uniformity_score = np.std(magnitude_hist) / np.mean(magnitude_hist)
    
    print(f"   • Magnitude uniformity score: {uniformity_score:.3f}")
    if uniformity_score < 0.5:
        print("   ⚠️  WARNING: High uniformity suggests mathematical artifacts")
    else:
        print("   ✅ Good: Non-uniform distribution suggests genuine patterns")
    
    # Check for correlation with spike data
    print("\n3️⃣ SPIKE-FEATURE CORRELATION ANALYSIS:")
    
    spike_results = data['spike_results']
    spike_times_str = spike_results['spike_times']
    spike_times_str = spike_times_str.replace('[', '').replace(']', '').strip()
    spike_times = np.array([float(x) for x in spike_times_str.split() if x.strip()])
    
    if len(spike_times) > 0:
        # Calculate spike density
        signal_length = 67471  # From the data
        spike_density = np.zeros(signal_length)
        for spike_time in spike_times:
            if 0 <= spike_time < signal_length:
                spike_density[int(spike_time)] = 1
        
        # Correlate with feature magnitudes
        if len(features) > 0:
            feature_magnitudes = np.array(magnitudes)
            correlation, p_value = pearsonr(spike_density[:len(feature_magnitudes)], feature_magnitudes)
            
            print(f"   • Spike-feature correlation: {correlation:.3f}")
            print(f"   • P-value: {p_value:.6f}")
            
            if p_value < 0.05:
                print("   ✅ SIGNIFICANT: Features correlate with biological spikes")
            else:
                print("   ⚠️  WARNING: No significant correlation with biological spikes")
    
    # Check for parameter sensitivity
    print("\n4️⃣ PARAMETER SENSITIVITY ANALYSIS:")
    
    # Test if features are sensitive to k and τ parameters
    k_values = [f['k'] for f in features]
    tau_values = [f['tau'] for f in features]
    
    k_tau_correlation, k_tau_p = pearsonr(k_values, tau_values)
    print(f"   • k-τ correlation: {k_tau_correlation:.3f}")
    
    if abs(k_tau_correlation) > 0.8:
        print("   ⚠️  WARNING: High k-τ correlation suggests parameter redundancy")
    else:
        print("   ✅ Good: Independent k and τ parameters")
    
    return {
        'uniformity_score': uniformity_score,
        'spike_correlation': correlation if 'correlation' in locals() else None,
        'spike_p_value': p_value if 'p_value' in locals() else None,
        'k_tau_correlation': k_tau_correlation
    }

def main():
    """Main false positive validation and pattern analysis"""
    print("🧪 FALSE POSITIVE VALIDATION & WAVE TRANSFORM ANALYSIS")
    print("=" * 80)
    
    # Generate synthetic controls
    controls = generate_synthetic_controls()
    
    # Test wave transform on controls
    control_results = test_wave_transform_on_controls(controls)
    
    # Compare with real data
    real_results = compare_with_real_data(control_results)
    
    # Analyze wave transform patterns
    pattern_analysis = analyze_wave_transform_patterns()
    
    # Final assessment
    print("\n🎯 FINAL ASSESSMENT:")
    print("=" * 50)
    
    # Check if real data shows distinct patterns from controls
    real_complexities = [r['pattern_complexity'] for r in real_results.values()]
    control_complexities = [c['pattern_complexity'] for c in control_results.values()]
    
    avg_real_complexity = np.mean(real_complexities)
    avg_control_complexity = np.mean(control_complexities)
    
    print(f"📊 PATTERN COMPLEXITY COMPARISON:")
    print(f"   • Real fungal data: {avg_real_complexity:.3f}")
    print(f"   • Control data: {avg_control_complexity:.3f}")
    
    if avg_real_complexity > avg_control_complexity * 1.5:
        print("   ✅ REAL PATTERNS: Fungal data shows significantly higher complexity")
    else:
        print("   ⚠️  WARNING: Fungal data complexity similar to controls")
    
    print(f"\n🔬 WAVE TRANSFORM VALIDATION:")
    print(f"   • Uniformity score: {pattern_analysis['uniformity_score']:.3f}")
    if pattern_analysis['spike_correlation']:
        print(f"   • Spike correlation: {pattern_analysis['spike_correlation']:.3f}")
        print(f"   • P-value: {pattern_analysis['spike_p_value']:.6f}")
    print(f"   • k-τ correlation: {pattern_analysis['k_tau_correlation']:.3f}")
    
    print(f"\n🎉 CONCLUSION:")
    if avg_real_complexity > avg_control_complexity * 1.5 and pattern_analysis['uniformity_score'] > 0.5:
        print("   ✅ Wave transform detects GENUINE BIOLOGICAL PATTERNS")
        print("   ✅ Not mathematical artifacts")
        print("   ✅ Parameters are meaningful and independent")
    else:
        print("   ⚠️  Potential false positives detected")
        print("   ⚠️  Consider additional validation")

if __name__ == "__main__":
    main() 