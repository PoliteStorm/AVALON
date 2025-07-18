#!/usr/bin/env python3
"""
Refined √t Transform Testing
- Better detection thresholds to reduce false positives
- More sophisticated √t patterns
- Quantitative comparison methods
"""

import numpy as np
import matplotlib.pyplot as plt
from transform_validation_framework import SqrtTTransform, ValidationFramework

class RefinedSqrtTTest:
    """
    Refined testing with better thresholds and sophisticated patterns
    """
    
    def __init__(self):
        self.transform = SqrtTTransform(sampling_rate=1.0)
        
    def create_sophisticated_sqrt_t_signals(self, length=10000):
        """
        Create more sophisticated √t signals with known properties
        """
        t = np.arange(length)
        sqrt_t = np.sqrt(t)
        
        # Signal 1: Multi-scale √t oscillation
        # Combines multiple √t frequencies
        signal1 = (np.sin(2 * np.pi * 0.05 * sqrt_t) + 
                   0.5 * np.sin(2 * np.pi * 0.1 * sqrt_t) +
                   0.25 * np.sin(2 * np.pi * 0.2 * sqrt_t))
        
        # Signal 2: √t-chirp (frequency varies with √t)
        # Frequency increases as √t
        signal2 = np.sin(2 * np.pi * 0.01 * sqrt_t * t)
        
        # Signal 3: √t-modulated pulse train
        # Pulses with √t-modulated spacing
        signal3 = np.zeros(length)
        pulse_times = []
        for i in range(1, int(np.sqrt(length))):
            pulse_time = int(i**2)  # √t spacing
            if pulse_time < length:
                pulse_times.append(pulse_time)
                # Create pulse with √t-modulated width
                pulse_width = int(10 * np.sqrt(i))
                start = max(0, pulse_time - pulse_width//2)
                end = min(length, pulse_time + pulse_width//2)
                signal3[start:end] += np.exp(-((np.arange(end-start) - pulse_width//2)**2) / (2 * (pulse_width//4)**2))
        
        # Signal 4: √t-fractal pattern
        # Self-similar pattern at different √t scales
        signal4 = np.zeros(length)
        for scale in [1, 2, 4, 8, 16]:
            if scale < length:
                pattern = np.sin(2 * np.pi * 0.1 * sqrt_t[:length//scale] * scale)
                # Upsample to full length
                for i in range(0, length, scale):
                    if i + scale < length:
                        signal4[i:i+scale] += pattern[:scale] / scale
        
        # Signal 5: √t-diffusion process
        # Simulates diffusion with √t scaling
        signal5 = np.zeros(length)
        for i in range(1, length):
            signal5[i] = signal5[i-1] + np.random.normal(0, 1/np.sqrt(i))
        
        return {
            'multi_scale_sqrt_oscillation': signal1,
            'sqrt_chirp': signal2,
            'sqrt_modulated_pulses': signal3,
            'sqrt_fractal': signal4,
            'sqrt_diffusion': signal5
        }
    
    def create_improved_controls(self, length=10000):
        """
        Create improved control signals that should show minimal √t features
        """
        t = np.arange(length)
        
        # Control 1: Pure linear oscillation (should show NO √t features)
        control1 = np.sin(2 * np.pi * 0.1 * t)
        
        # Control 2: Linear chirp (frequency varies linearly, not √t)
        control2 = np.sin(2 * np.pi * 0.001 * t * t)
        
        # Control 3: White noise (should show NO structure)
        control3 = np.random.normal(0, 1, length)
        
        # Control 4: Linear pulse train (constant spacing)
        control4 = np.zeros(length)
        for i in range(0, length, 100):
            if i + 50 < length:
                control4[i:i+50] = 1.0
        
        # Control 5: Exponential decay (linear time, not √t)
        control5 = np.exp(-t / 1000)
        
        return {
            'linear_oscillation': control1,
            'linear_chirp': control2,
            'white_noise': control3,
            'linear_pulses': control4,
            'linear_decay': control5
        }
    
    def adaptive_threshold_detection(self, W, k_values, tau_values, method='percentile'):
        """
        Improved feature detection with adaptive thresholds
        """
        magnitude = np.abs(W)
        
        if method == 'percentile':
            # Use 95th percentile as threshold
            threshold = np.percentile(magnitude, 95)
        elif method == 'adaptive':
            # Adaptive threshold based on local maxima
            from scipy.signal import find_peaks
            magnitude_flat = magnitude.flatten()
            peaks, _ = find_peaks(magnitude_flat, height=np.percentile(magnitude_flat, 90))
            if len(peaks) > 0:
                threshold = np.percentile(magnitude_flat[peaks], 75)
            else:
                threshold = np.percentile(magnitude_flat, 95)
        elif method == 'conservative':
            # Very conservative threshold
            threshold = np.percentile(magnitude, 98)
        else:
            threshold = 0.1 * np.max(magnitude)
        
        features = []
        
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                if magnitude[i, j] > threshold:
                    # Additional filtering: check if it's a local maximum
                    is_local_max = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < len(k_values) and 
                                0 <= nj < len(tau_values) and
                                magnitude[ni, nj] > magnitude[i, j]):
                                is_local_max = False
                                break
                        if not is_local_max:
                            break
                    
                    if is_local_max:
                        features.append({
                            'k': k,
                            'tau': tau,
                            'magnitude': magnitude[i, j],
                            'phase': np.angle(W[i, j]),
                            'relative_magnitude': magnitude[i, j] / np.max(magnitude)
                        })
        
        return features
    
    def quantitative_comparison(self, sqrt_results, control_results):
        """
        Quantitative comparison between √t and control signals
        """
        # Extract feature counts
        sqrt_counts = [len(r['features']) for r in sqrt_results.values()]
        control_counts = [len(r['features']) for r in control_results.values()]
        
        # Statistical tests
        from scipy import stats
        
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(sqrt_counts, control_counts, alternative='greater')
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(sqrt_counts) - np.mean(control_counts)
        pooled_std = np.sqrt(((len(sqrt_counts) - 1) * np.var(sqrt_counts) + 
                             (len(control_counts) - 1) * np.var(control_counts)) / 
                            (len(sqrt_counts) + len(control_counts) - 2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Magnitude analysis
        sqrt_magnitudes = []
        control_magnitudes = []
        
        for result in sqrt_results.values():
            if result['features']:
                sqrt_magnitudes.extend([f['relative_magnitude'] for f in result['features']])
        
        for result in control_results.values():
            if result['features']:
                control_magnitudes.extend([f['relative_magnitude'] for f in result['features']])
        
        # Magnitude comparison
        if sqrt_magnitudes and control_magnitudes:
            mag_statistic, mag_p_value = stats.mannwhitneyu(sqrt_magnitudes, control_magnitudes, alternative='greater')
        else:
            mag_statistic, mag_p_value = 0, 1
        
        return {
            'feature_count_test': {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'effect_size': cohens_d,
            'magnitude_test': {
                'statistic': mag_statistic,
                'p_value': mag_p_value,
                'significant': mag_p_value < 0.05
            },
            'summary_stats': {
                'sqrt_mean_features': np.mean(sqrt_counts),
                'control_mean_features': np.mean(control_counts),
                'sqrt_std_features': np.std(sqrt_counts),
                'control_std_features': np.std(control_counts)
            }
        }
    
    def comprehensive_refined_test(self):
        """
        Run comprehensive refined testing
        """
        print("REFINED √t TRANSFORM TESTING")
        print("=" * 60)
        
        # Create signals
        sqrt_signals = self.create_sophisticated_sqrt_t_signals()
        control_signals = self.create_improved_controls()
        
        # Test different threshold methods
        threshold_methods = ['percentile', 'adaptive', 'conservative']
        
        all_results = {}
        
        for method in threshold_methods:
            print(f"\nTesting with {method} threshold:")
            print("-" * 40)
            
            sqrt_results = {}
            control_results = {}
            
            # Parameters
            k_values = np.logspace(-1, 1, 15)
            tau_values = np.logspace(0, 2, 15)
            
            # Test √t signals
            for name, signal in sqrt_signals.items():
                W = self.transform.transform(signal, k_values, tau_values)
                features = self.adaptive_threshold_detection(W, k_values, tau_values, method)
                sqrt_results[name] = {'features': features, 'count': len(features)}
                print(f"  {name}: {len(features)} features")
            
            # Test control signals
            for name, signal in control_signals.items():
                W = self.transform.transform(signal, k_values, tau_values)
                features = self.adaptive_threshold_detection(W, k_values, tau_values, method)
                control_results[name] = {'features': features, 'count': len(features)}
                print(f"  {name}: {len(features)} features")
            
            # Quantitative comparison
            comparison = self.quantitative_comparison(sqrt_results, control_results)
            
            all_results[method] = {
                'sqrt_results': sqrt_results,
                'control_results': control_results,
                'comparison': comparison
            }
            
            print(f"\nQuantitative Results ({method}):")
            print(f"  Feature count p-value: {comparison['feature_count_test']['p_value']:.4f}")
            print(f"  Magnitude p-value: {comparison['magnitude_test']['p_value']:.4f}")
            print(f"  Effect size (Cohen's d): {comparison['effect_size']:.3f}")
        
        return all_results

def main():
    """
    Run refined testing
    """
    tester = RefinedSqrtTTest()
    results = tester.comprehensive_refined_test()
    
    # Summary
    print("\n" + "=" * 60)
    print("REFINED TESTING SUMMARY")
    print("=" * 60)
    
    for method, result in results.items():
        comparison = result['comparison']
        print(f"\n{method.upper()} THRESHOLD:")
        print(f"  Feature count significant: {comparison['feature_count_test']['significant']}")
        print(f"  Magnitude significant: {comparison['magnitude_test']['significant']}")
        print(f"  Effect size: {comparison['effect_size']:.3f}")
        print(f"  √t mean features: {comparison['summary_stats']['sqrt_mean_features']:.1f}")
        print(f"  Control mean features: {comparison['summary_stats']['control_mean_features']:.1f}")
    
    return results

if __name__ == "__main__":
    main() 