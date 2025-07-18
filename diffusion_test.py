#!/usr/bin/env python3
"""
Diffusion Data Testing for √t Transform
Test the transform on diffusion processes where √t scaling is well-established
"""

import numpy as np
import matplotlib.pyplot as plt
from transform_validation_framework import SqrtTTransform, ValidationFramework

class DiffusionTest:
    """
    Test √t transform on diffusion processes
    """
    
    def __init__(self):
        self.transform = SqrtTTransform(sampling_rate=1.0)
        
    def create_diffusion_signals(self, length=10000):
        """
        Create diffusion signals with known √t scaling
        """
        t = np.arange(length)
        sqrt_t = np.sqrt(t)
        
        # Signal 1: Pure diffusion (mean squared displacement ∝ t)
        # This should show √t scaling in the variance
        diffusion1 = np.zeros(length)
        for i in range(1, length):
            diffusion1[i] = diffusion1[i-1] + np.random.normal(0, np.sqrt(0.1))
        
        # Signal 2: Diffusion with drift
        # MSD ∝ t, but with linear drift component
        diffusion2 = np.zeros(length)
        for i in range(1, length):
            diffusion2[i] = diffusion2[i-1] + np.random.normal(0, np.sqrt(0.1)) + 0.01
        
        # Signal 3: Anomalous diffusion (MSD ∝ t^α, where α < 1)
        # This should show different scaling than √t
        diffusion3 = np.zeros(length)
        for i in range(1, length):
            diffusion3[i] = diffusion3[i-1] + np.random.normal(0, np.sqrt(0.1 * (i/1000)**0.5))
        
        # Signal 4: Superdiffusion (MSD ∝ t^α, where α > 1)
        # This should show different scaling than √t
        diffusion4 = np.zeros(length)
        for i in range(1, length):
            diffusion4[i] = diffusion4[i-1] + np.random.normal(0, np.sqrt(0.1 * (i/1000)**1.5))
        
        # Signal 5: Diffusion with √t-modulated noise
        # Explicit √t scaling in the noise term
        diffusion5 = np.zeros(length)
        for i in range(1, length):
            noise_amplitude = 0.1 * np.sqrt(i)  # √t scaling
            diffusion5[i] = diffusion5[i-1] + np.random.normal(0, noise_amplitude)
        
        return {
            'pure_diffusion': diffusion1,
            'diffusion_with_drift': diffusion2,
            'anomalous_diffusion': diffusion3,
            'superdiffusion': diffusion4,
            'sqrt_modulated_diffusion': diffusion5
        }
    
    def create_non_diffusion_controls(self, length=10000):
        """
        Create control signals that should NOT show √t scaling
        """
        t = np.arange(length)
        
        # Control 1: Random walk with constant step size
        control1 = np.cumsum(np.random.normal(0, 1, length))
        
        # Control 2: Linear trend with noise
        control2 = 0.01 * t + np.random.normal(0, 1, length)
        
        # Control 3: Oscillatory signal
        control3 = np.sin(2 * np.pi * 0.01 * t) + 0.1 * np.random.normal(0, 1, length)
        
        # Control 4: Exponential growth
        control4 = np.exp(0.001 * t) + 0.1 * np.random.normal(0, 1, length)
        
        # Control 5: Stationary process
        control5 = np.random.normal(0, 1, length)
        
        return {
            'random_walk': control1,
            'linear_trend': control2,
            'oscillatory': control3,
            'exponential_growth': control4,
            'stationary': control5
        }
    
    def analyze_diffusion_properties(self, signal):
        """
        Analyze the diffusion properties of a signal
        """
        # Calculate mean squared displacement
        msd = []
        max_lag = min(1000, len(signal)//10)
        
        for lag in range(1, max_lag):
            diff = signal[lag:] - signal[:-lag]
            msd.append(np.mean(diff**2))
        
        lags = np.arange(1, max_lag)
        
        # Fit power law: MSD ∝ t^α
        if len(msd) > 10:
            log_lags = np.log(lags)
            log_msd = np.log(msd)
            
            # Linear fit
            coeffs = np.polyfit(log_lags, log_msd, 1)
            alpha = coeffs[0]  # Power law exponent
            r_squared = np.corrcoef(log_lags, log_msd)[0,1]**2
            
            return {
                'alpha': alpha,
                'r_squared': r_squared,
                'is_diffusive': abs(alpha - 1.0) < 0.2,  # α ≈ 1 for normal diffusion
                'is_sqrt_scaled': abs(alpha - 0.5) < 0.2,  # α ≈ 0.5 for √t scaling
                'msd': msd,
                'lags': lags
            }
        else:
            return None
    
    def test_diffusion_signals(self):
        """
        Test the transform on diffusion signals
        """
        print("DIFFUSION TESTING FOR √t TRANSFORM")
        print("=" * 60)
        
        # Create signals
        diffusion_signals = self.create_diffusion_signals()
        control_signals = self.create_non_diffusion_controls()
        
        # Parameters
        k_values = np.logspace(-1, 1, 15)
        tau_values = np.logspace(0, 2, 15)
        
        results = {}
        
        # Test diffusion signals
        print("\nDIFFUSION SIGNALS (should show √t features):")
        print("-" * 50)
        
        for name, signal in diffusion_signals.items():
            print(f"\nTesting: {name}")
            
            # Analyze diffusion properties
            diff_props = self.analyze_diffusion_properties(signal)
            if diff_props:
                print(f"  Diffusion exponent (α): {diff_props['alpha']:.3f}")
                print(f"  R² fit: {diff_props['r_squared']:.3f}")
                print(f"  Is diffusive: {diff_props['is_diffusive']}")
                print(f"  Shows √t scaling: {diff_props['is_sqrt_scaled']}")
            
            # Apply transform
            W = self.transform.transform(signal, k_values, tau_values)
            features = self.transform.detect_features(W, k_values, tau_values)
            
            print(f"  √t features detected: {len(features)}")
            
            if features:
                k_vals = [f['k'] for f in features]
                tau_vals = [f['tau'] for f in features]
                print(f"  k range: {min(k_vals):.3f} to {max(k_vals):.3f}")
                print(f"  τ range: {min(tau_vals):.3f} to {max(tau_vals):.3f}")
            
            results[f'diffusion_{name}'] = {
                'signal': signal,
                'features': features,
                'diffusion_props': diff_props
            }
        
        # Test control signals
        print("\nCONTROL SIGNALS (should show few √t features):")
        print("-" * 50)
        
        for name, signal in control_signals.items():
            print(f"\nTesting: {name}")
            
            # Analyze diffusion properties
            diff_props = self.analyze_diffusion_properties(signal)
            if diff_props:
                print(f"  Diffusion exponent (α): {diff_props['alpha']:.3f}")
                print(f"  R² fit: {diff_props['r_squared']:.3f}")
                print(f"  Is diffusive: {diff_props['is_diffusive']}")
                print(f"  Shows √t scaling: {diff_props['is_sqrt_scaled']}")
            
            # Apply transform
            W = self.transform.transform(signal, k_values, tau_values)
            features = self.transform.detect_features(W, k_values, tau_values)
            
            print(f"  √t features detected: {len(features)}")
            
            results[f'control_{name}'] = {
                'signal': signal,
                'features': features,
                'diffusion_props': diff_props
            }
        
        return results
    
    def analyze_results(self, results):
        """
        Analyze the diffusion test results
        """
        print("\n" + "=" * 60)
        print("DIFFUSION TEST ANALYSIS")
        print("=" * 60)
        
        # Separate diffusion and control results
        diffusion_results = {k: v for k, v in results.items() if k.startswith('diffusion_')}
        control_results = {k: v for k, v in results.items() if k.startswith('control_')}
        
        # Count features
        diffusion_features = [len(r['features']) for r in diffusion_results.values()]
        control_features = [len(r['features']) for r in control_results.values()]
        
        print(f"Diffusion signals: {len(diffusion_features)} total features")
        print(f"Control signals: {len(control_features)} total features")
        
        # Analyze diffusion properties
        diffusion_alphas = []
        sqrt_scaled_signals = []
        
        for name, result in diffusion_results.items():
            if result['diffusion_props']:
                alpha = result['diffusion_props']['alpha']
                diffusion_alphas.append(alpha)
                if result['diffusion_props']['is_sqrt_scaled']:
                    sqrt_scaled_signals.append(name)
        
        print(f"\nDiffusion exponents (α): {[f'{a:.3f}' for a in diffusion_alphas]}")
        print(f"Signals with √t scaling: {sqrt_scaled_signals}")
        
        # Success criteria
        success_criteria = []
        
        # Criterion 1: Diffusion signals should have more features than controls
        if np.mean(diffusion_features) > np.mean(control_features):
            success_criteria.append("✓ Diffusion signals show more features than controls")
        else:
            success_criteria.append("✗ Diffusion signals don't show more features than controls")
        
        # Criterion 2: √t-scaled signals should show features
        sqrt_scaled_features = []
        for name in sqrt_scaled_signals:
            if name in diffusion_results:
                sqrt_scaled_features.append(len(diffusion_results[name]['features']))
        
        if sqrt_scaled_features and np.mean(sqrt_scaled_features) > 0:
            success_criteria.append(f"✓ √t-scaled signals show features (avg: {np.mean(sqrt_scaled_features):.1f})")
        else:
            success_criteria.append("✗ √t-scaled signals don't show features")
        
        # Criterion 3: Controls should show few features
        if np.mean(control_features) < 5:  # Arbitrary threshold
            success_criteria.append(f"✓ Controls show few features (avg: {np.mean(control_features):.1f})")
        else:
            success_criteria.append(f"✗ Controls show too many features (avg: {np.mean(control_features):.1f})")
        
        print("\nSuccess Criteria:")
        for criterion in success_criteria:
            print(f"  {criterion}")
        
        # Overall assessment
        passed_criteria = sum(1 for c in success_criteria if c.startswith("✓"))
        total_criteria = len(success_criteria)
        
        print(f"\nOverall: {passed_criteria}/{total_criteria} criteria passed")
        
        if passed_criteria >= 2:
            print("✓ Transform shows promise for diffusion analysis")
        else:
            print("✗ Transform does not reliably detect diffusion patterns")

def main():
    """
    Run diffusion testing
    """
    tester = DiffusionTest()
    results = tester.test_diffusion_signals()
    tester.analyze_results(results)
    
    return results

if __name__ == "__main__":
    main() 