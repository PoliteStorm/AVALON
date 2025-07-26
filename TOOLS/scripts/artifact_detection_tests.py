#!/usr/bin/env python3
"""
Artifact Detection Tests for √t Transform
Since there's no ground truth √t-scaled data, we test for artifacts using:
1. Transform self-consistency
2. Mathematical properties
3. Physical plausibility
4. Cross-validation with known patterns
"""

import numpy as np
import matplotlib.pyplot as plt
from transform_validation_framework import SqrtTTransform, ValidationFramework

class ArtifactDetector:
    """
    Comprehensive artifact detection for √t transform
    """
    
    def __init__(self):
        self.transform = SqrtTTransform(sampling_rate=1.0)
        
    def test_transform_self_consistency(self, V):
        """
        Test 1: Does the transform create its own artifacts?
        Apply transform → reconstruct → apply transform again
        If second transform shows structure, it's likely an artifact
        """
        print("Test 1: Transform Self-Consistency")
        
        # First application
        k_values = np.logspace(-1, 1, 10)
        tau_values = np.logspace(0, 2, 10)
        W1 = self.transform.transform(V, k_values, tau_values)
        
        # Attempt reconstruction (simplified)
        reconstructed = self.simple_reconstruction(W1, k_values, tau_values, len(V))
        
        # Second application
        W2 = self.transform.transform(reconstructed, k_values, tau_values)
        
        # Compare magnitudes
        mag1 = np.abs(W1)
        mag2 = np.abs(W2)
        
        correlation = np.corrcoef(mag1.flatten(), mag2.flatten())[0,1]
        
        print(f"   Self-consistency correlation: {correlation:.4f}")
        print(f"   High correlation (>0.8) suggests transform artifacts")
        
        return correlation > 0.8  # If true, likely artifact
    
    def simple_reconstruction(self, W, k_values, tau_values, signal_length):
        """Simplified reconstruction for testing"""
        t = np.arange(signal_length)
        reconstructed = np.zeros(signal_length)
        
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                if abs(W[i,j]) > 0.1 * np.max(np.abs(W)):
                    window = self.transform.gaussian_window(t, tau)
                    phase = np.exp(-1j * k * np.sqrt(t))
                    contribution = np.real(W[i,j] * window * phase)
                    reconstructed += contribution
        
        return reconstructed
    
    def test_mathematical_properties(self, V):
        """
        Test 2: Do the results violate expected mathematical properties?
        """
        print("Test 2: Mathematical Properties")
        
        k_values = np.logspace(-1, 1, 10)
        tau_values = np.logspace(0, 2, 10)
        W = self.transform.transform(V, k_values, tau_values)
        
        # Property 1: Linearity
        V1 = V[:len(V)//2]
        V2 = V[len(V)//2:]
        W1 = self.transform.transform(V1, k_values, tau_values)
        W2 = self.transform.transform(V2, k_values, tau_values)
        W_combined = self.transform.transform(V, k_values, tau_values)
        
        linearity_error = np.mean(np.abs(W_combined - (W1 + W2)))
        print(f"   Linearity error: {linearity_error:.6f}")
        
        # Property 2: Scale invariance (should be approximately preserved)
        V_scaled = V * 2.0
        W_scaled = self.transform.transform(V_scaled, k_values, tau_values)
        scale_invariance = np.corrcoef(np.abs(W).flatten(), np.abs(W_scaled).flatten())[0,1]
        print(f"   Scale invariance correlation: {scale_invariance:.4f}")
        
        return linearity_error < 0.1 and scale_invariance > 0.7
    
    def test_physical_plausibility(self, V):
        """
        Test 3: Do the detected features make physical sense?
        """
        print("Test 3: Physical Plausibility")
        
        k_values = np.logspace(-1, 1, 10)
        tau_values = np.logspace(0, 2, 10)
        W = self.transform.transform(V, k_values, tau_values)
        
        features = self.transform.detect_features(W, k_values, tau_values)
        
        # Check for physically impossible patterns
        impossible_features = []
        for feature in features:
            k, tau = feature['k'], feature['tau']
            
            # Check for impossible frequency relationships
            freq_estimate = k / (2 * np.pi * np.sqrt(tau))
            
            # Impossible patterns:
            # 1. Negative frequencies
            # 2. Frequencies > 1000 Hz (beyond biological range)
            # 3. Perfect harmonics (suggest digital artifacts)
            # 4. Scale parameters outside reasonable range
            
            if freq_estimate < 0 or freq_estimate > 1000:
                impossible_features.append(feature)
            elif tau < 0.1 or tau > 10000:  # Unreasonable time scales
                impossible_features.append(feature)
        
        print(f"   Impossible features detected: {len(impossible_features)}")
        print(f"   Total features: {len(features)}")
        
        return len(impossible_features) == 0
    
    def test_cross_validation_with_known_patterns(self, V):
        """
        Test 4: Test against known synthetic patterns
        """
        print("Test 4: Cross-Validation with Known Patterns")
        
        # Create synthetic patterns with known properties
        synthetic_tests = {}
        
        # Test 1: Pure sine wave
        t = np.arange(len(V))
        synthetic_tests['sine_wave'] = np.sin(2 * np.pi * 0.1 * t)
        
        # Test 2: Exponential decay (should show √t scaling)
        synthetic_tests['exponential'] = np.exp(-t/1000)
        
        # Test 3: Random walk (should NOT show √t scaling)
        synthetic_tests['random_walk'] = np.cumsum(np.random.normal(0, 1, len(V)))
        
        # Test 4: Spike train
        spike_train = np.zeros(len(V))
        spike_positions = np.random.choice(len(V), len(V)//100, replace=False)
        for pos in spike_positions:
            spike_train[pos] = 1.0
        synthetic_tests['spike_train'] = spike_train
        
        results = {}
        k_values = np.logspace(-1, 1, 10)
        tau_values = np.logspace(0, 2, 10)
        
        for test_name, test_signal in synthetic_tests.items():
            W_test = self.transform.transform(test_signal, k_values, tau_values)
            features = self.transform.detect_features(W_test, k_values, tau_values)
            results[test_name] = len(features)
            print(f"   {test_name}: {len(features)} features")
        
        # Expected behavior:
        # - Exponential should show √t features
        # - Random walk should show few features
        # - Sine wave should show frequency-specific features
        # - Spike train should show temporal features
        
        return results
    
    def test_parameter_sensitivity(self, V):
        """
        Test 5: Are results sensitive to parameter changes?
        Artifacts often show parameter-dependent behavior
        """
        print("Test 5: Parameter Sensitivity")
        
        # Test with different parameter ranges
        k_ranges = [
            np.logspace(-1, 1, 10),
            np.logspace(-0.5, 0.5, 10),
            np.logspace(-2, 2, 10)
        ]
        
        tau_ranges = [
            np.logspace(0, 2, 10),
            np.logspace(0.5, 1.5, 10),
            np.logspace(-1, 3, 10)
        ]
        
        feature_counts = []
        
        for i, k_vals in enumerate(k_ranges):
            for j, tau_vals in enumerate(tau_ranges):
                W = self.transform.transform(V, k_vals, tau_vals)
                features = self.transform.detect_features(W, k_vals, tau_vals)
                feature_counts.append(len(features))
                print(f"   k_range_{i}, tau_range_{j}: {len(features)} features")
        
        # Calculate stability
        feature_std = np.std(feature_counts)
        feature_mean = np.mean(feature_counts)
        stability = feature_mean / (feature_std + 1e-10)
        
        print(f"   Parameter stability: {stability:.2f}")
        print(f"   High stability suggests robust features, low stability suggests artifacts")
        
        return stability > 2.0  # Arbitrary threshold
    
    def comprehensive_artifact_test(self, V):
        """
        Run all artifact detection tests
        """
        print(f"\n{'='*60}")
        print("COMPREHENSIVE ARTIFACT DETECTION")
        print(f"{'='*60}")
        
        results = {}
        
        # Test 1: Self-consistency
        results['self_consistent'] = self.test_transform_self_consistency(V)
        
        # Test 2: Mathematical properties
        results['mathematical_valid'] = self.test_mathematical_properties(V)
        
        # Test 3: Physical plausibility
        results['physically_plausible'] = self.test_physical_plausibility(V)
        
        # Test 4: Cross-validation
        results['cross_validation'] = self.test_cross_validation_with_known_patterns(V)
        
        # Test 5: Parameter sensitivity
        results['parameter_stable'] = self.test_parameter_sensitivity(V)
        
        # Summary
        print(f"\n{'='*60}")
        print("ARTIFACT DETECTION SUMMARY")
        print(f"{'='*60}")
        
        # Count boolean results only
        boolean_results = {k: v for k, v in results.items() if isinstance(v, bool)}
        passed_tests = sum(boolean_results.values())
        total_tests = len(boolean_results)
        
        print(f"Tests passed: {passed_tests}/{total_tests}")
        
        if passed_tests >= 4:
            print("✓ Transform appears robust with minimal artifacts")
        elif passed_tests >= 2:
            print("⚠ Transform shows some artifacts but may still be useful")
        else:
            print("✗ Transform shows significant artifacts")
        
        return results

def main():
    """
    Run artifact detection on fungal data
    """
    detector = ArtifactDetector()
    
    # Test with fungal data
    validator = ValidationFramework(detector.transform)
    
    # Test files
    test_files = [
        "15061491/fungal_spikes/good_recordings/New_Oyster_with spray_as_mV_seconds_SigView.csv",
        "15061491/fungal_spikes/good_recordings/Hericium_20_4_22_part1.csv"
    ]
    
    for filepath in test_files:
        print(f"\n{'='*80}")
        print(f"Testing artifacts in: {filepath}")
        print(f"{'='*80}")
        
        V = validator.load_fungal_data(filepath)
        if V is not None:
            # Use first 10000 samples for faster testing
            V_test = V[:10000]
            results = detector.comprehensive_artifact_test(V_test)
        else:
            print(f"Could not load {filepath}")

if __name__ == "__main__":
    main() 