#!/usr/bin/env python3
"""
Synthetic √t Data Testing
Create signals with known √t features and test if the transform can recover them
"""

import numpy as np
import matplotlib.pyplot as plt
from transform_validation_framework import SqrtTTransform, ValidationFramework

class SyntheticSqrtTTest:
    """
    Test the √t transform with synthetic data containing known √t features
    """
    
    def __init__(self):
        self.transform = SqrtTTransform(sampling_rate=1.0)
        
    def create_synthetic_sqrt_t_signal(self, length=10000):
        """
        Create synthetic signals with known √t features
        """
        t = np.arange(length)
        sqrt_t = np.sqrt(t)
        
        # Signal 1: Pure √t oscillation
        # This should be perfectly detected by the transform
        signal1 = np.sin(2 * np.pi * 0.1 * sqrt_t)
        
        # Signal 2: √t-modulated amplitude
        # Amplitude varies with √t
        signal2 = np.sin(2 * np.pi * 0.05 * t) * np.exp(-sqrt_t / 100)
        
        # Signal 3: √t-spaced spikes
        # Spikes occur at √t intervals
        signal3 = np.zeros(length)
        sqrt_spike_times = np.arange(0, np.sqrt(length), 10)  # Every 10 √t units
        spike_times = (sqrt_spike_times ** 2).astype(int)
        spike_times = spike_times[spike_times < length]
        signal3[spike_times] = 1.0
        
        # Signal 4: √t frequency modulation
        # Frequency varies as √t
        signal4 = np.sin(2 * np.pi * 0.01 * sqrt_t * t)
        
        # Signal 5: √t decay with oscillation
        # Exponential decay in √t time
        signal5 = np.exp(-sqrt_t / 50) * np.sin(2 * np.pi * 0.1 * t)
        
        return {
            'pure_sqrt_oscillation': signal1,
            'sqrt_modulated_amplitude': signal2,
            'sqrt_spaced_spikes': signal3,
            'sqrt_frequency_modulation': signal4,
            'sqrt_decay_oscillation': signal5
        }
    
    def create_control_signals(self, length=10000):
        """
        Create control signals that should NOT show √t features
        """
        t = np.arange(length)
        
        # Control 1: Pure linear oscillation
        control1 = np.sin(2 * np.pi * 0.1 * t)
        
        # Control 2: Linear decay
        control2 = np.exp(-t / 1000)
        
        # Control 3: Random noise
        control3 = np.random.normal(0, 1, length)
        
        # Control 4: Linear frequency modulation
        control4 = np.sin(2 * np.pi * 0.01 * t * t)
        
        return {
            'linear_oscillation': control1,
            'linear_decay': control2,
            'random_noise': control3,
            'linear_frequency_modulation': control4
        }
    
    def test_transform_recovery(self, signal, signal_name):
        """
        Test if the transform can recover known √t features
        """
        print(f"\nTesting: {signal_name}")
        print("=" * 50)
        
        # Parameters for transform
        k_values = np.logspace(-1, 1, 15)
        tau_values = np.logspace(0, 2, 15)
        
        # Apply transform
        W = self.transform.transform(signal, k_values, tau_values)
        features = self.transform.detect_features(W, k_values, tau_values)
        
        print(f"Features detected: {len(features)}")
        
        if features:
            # Analyze feature characteristics
            k_vals = [f['k'] for f in features]
            tau_vals = [f['tau'] for f in features]
            mag_vals = [f['magnitude'] for f in features]
            
            print(f"k range: {min(k_vals):.3f} to {max(k_vals):.3f}")
            print(f"τ range: {min(tau_vals):.3f} to {max(tau_vals):.3f}")
            print(f"Magnitude range: {min(mag_vals):.6f} to {max(mag_vals):.6f}")
            
            # Check for √t-specific patterns
            sqrt_t_indicators = self.check_sqrt_t_indicators(features)
            print(f"√t indicators: {sqrt_t_indicators}")
            
            return {
                'features': len(features),
                'k_range': (min(k_vals), max(k_vals)),
                'tau_range': (min(tau_vals), max(tau_vals)),
                'magnitude_range': (min(mag_vals), max(mag_vals)),
                'sqrt_t_indicators': sqrt_t_indicators
            }
        else:
            print("No features detected")
            return {'features': 0}
    
    def check_sqrt_t_indicators(self, features):
        """
        Check if detected features are consistent with √t structure
        """
        indicators = []
        
        for feature in features:
            k, tau = feature['k'], feature['tau']
            
            # Indicator 1: Frequency estimate in √t domain
            freq_estimate = k / (2 * np.pi * np.sqrt(tau))
            
            # Indicator 2: Scale parameter in reasonable range
            if 1 <= tau <= 1000:
                indicators.append(f"Reasonable τ: {tau:.2f}")
            
            # Indicator 3: Frequency in biological range
            if 0.01 <= freq_estimate <= 10:
                indicators.append(f"Biological freq: {freq_estimate:.3f} Hz")
            
            # Indicator 4: Strong magnitude (real features should be strong)
            if feature['magnitude'] > 0.1 * np.max([f['magnitude'] for f in features]):
                indicators.append(f"Strong magnitude: {feature['magnitude']:.3f}")
        
        return indicators
    
    def comprehensive_synthetic_test(self):
        """
        Run comprehensive test with synthetic √t data
        """
        print("SYNTHETIC √t TRANSFORM TESTING")
        print("=" * 60)
        
        # Create synthetic signals
        sqrt_signals = self.create_synthetic_sqrt_t_signal()
        control_signals = self.create_control_signals()
        
        results = {}
        
        # Test √t signals (should show features)
        print("\nTESTING √t SIGNALS (should show features):")
        print("-" * 40)
        
        for name, signal in sqrt_signals.items():
            result = self.test_transform_recovery(signal, name)
            results[f"sqrt_{name}"] = result
        
        # Test control signals (should show few/no features)
        print("\nTESTING CONTROL SIGNALS (should show few features):")
        print("-" * 40)
        
        for name, signal in control_signals.items():
            result = self.test_transform_recovery(signal, name)
            results[f"control_{name}"] = result
        
        # Summary analysis
        self.analyze_results(results)
        
        return results
    
    def analyze_results(self, results):
        """
        Analyze test results
        """
        print("\n" + "=" * 60)
        print("RESULTS ANALYSIS")
        print("=" * 60)
        
        sqrt_results = {k: v for k, v in results.items() if k.startswith('sqrt_')}
        control_results = {k: v for k, v in results.items() if k.startswith('control_')}
        
        # Count features
        sqrt_features = sum(r['features'] for r in sqrt_results.values())
        control_features = sum(r['features'] for r in control_results.values())
        
        print(f"√t signals total features: {sqrt_features}")
        print(f"Control signals total features: {control_features}")
        
        # Success criteria
        success_criteria = []
        
        # Criterion 1: √t signals should have more features than controls
        if sqrt_features > control_features:
            success_criteria.append("✓ √t signals show more features than controls")
        else:
            success_criteria.append("✗ √t signals don't show more features than controls")
        
        # Criterion 2: At least some √t signals should show features
        sqrt_signals_with_features = sum(1 for r in sqrt_results.values() if r['features'] > 0)
        if sqrt_signals_with_features > 0:
            success_criteria.append(f"✓ {sqrt_signals_with_features} √t signals show features")
        else:
            success_criteria.append("✗ No √t signals show features")
        
        # Criterion 3: Controls should show few features
        control_signals_with_features = sum(1 for r in control_results.values() if r['features'] > 0)
        if control_signals_with_features < len(control_results):
            success_criteria.append(f"✓ Only {control_signals_with_features} control signals show features")
        else:
            success_criteria.append("✗ Too many control signals show features")
        
        print("\nSuccess Criteria:")
        for criterion in success_criteria:
            print(f"  {criterion}")
        
        # Overall assessment
        passed_criteria = sum(1 for c in success_criteria if c.startswith("✓"))
        total_criteria = len(success_criteria)
        
        print(f"\nOverall: {passed_criteria}/{total_criteria} criteria passed")
        
        if passed_criteria >= 2:
            print("✓ Transform shows promise for detecting √t features")
        else:
            print("✗ Transform does not reliably detect √t features")

def main():
    """
    Run synthetic √t testing
    """
    tester = SyntheticSqrtTTest()
    results = tester.comprehensive_synthetic_test()
    
    return results

if __name__ == "__main__":
    main() 