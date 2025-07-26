#!/usr/bin/env python3
"""
Proper Validation Framework for √t Transform
Tests the transform against known patterns and controls.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal
import pandas as pd
from improved_sqrt_transform import ImprovedSqrtTTransform
import os
from datetime import datetime

class ProperValidationFramework:
    def __init__(self):
        self.transform = ImprovedSqrtTTransform()
        
    def generate_ground_truth_signals(self, t, n_samples=1000):
        """
        Generate signals with known √t patterns for validation.
        
        Returns:
            dict: Dictionary of test signals with known properties
        """
        signals = {}
        
        # 1. Pure √t oscillation (ground truth)
        sqrt_t_osc = np.sin(2 * np.pi * np.sqrt(t))
        signals['pure_sqrt_oscillation'] = {
            'signal': sqrt_t_osc,
            'has_sqrt_pattern': True,
            'description': 'Pure √t oscillation: sin(2π√t)'
        }
        
        # 2. √t-modulated amplitude
        sqrt_mod_amp = np.sin(2 * np.pi * t) * np.exp(-np.sqrt(t) / 10)
        signals['sqrt_modulated_amplitude'] = {
            'signal': sqrt_mod_amp,
            'has_sqrt_pattern': True,
            'description': '√t-modulated amplitude: sin(2πt) * exp(-√t/10)'
        }
        
        # 3. √t-spaced spikes
        sqrt_spikes = np.zeros_like(t)
        spike_times = np.arange(0, np.sqrt(t[-1]), 1)**2  # Spikes at t = n²
        for spike_time in spike_times:
            if spike_time < t[-1]:
                idx = np.argmin(np.abs(t - spike_time))
                sqrt_spikes[idx] = 1.0
        signals['sqrt_spaced_spikes'] = {
            'signal': sqrt_spikes,
            'has_sqrt_pattern': True,
            'description': 'Spikes spaced by √t intervals'
        }
        
        # 4. √t frequency modulation
        sqrt_freq_mod = np.sin(2 * np.pi * np.sqrt(t) * t)
        signals['sqrt_frequency_modulation'] = {
            'signal': sqrt_freq_mod,
            'has_sqrt_pattern': True,
            'description': '√t frequency modulation: sin(2π√t * t)'
        }
        
        # 5. √t decay oscillation
        sqrt_decay = np.sin(2 * np.pi * np.sqrt(t)) * np.exp(-t / 100)
        signals['sqrt_decay_oscillation'] = {
            'signal': sqrt_decay,
            'has_sqrt_pattern': True,
            'description': '√t oscillation with decay: sin(2π√t) * exp(-t/100)'
        }
        
        # Control signals (should NOT have √t patterns)
        # 6. Linear oscillation
        linear_osc = np.sin(2 * np.pi * t)
        signals['linear_oscillation'] = {
            'signal': linear_osc,
            'has_sqrt_pattern': False,
            'description': 'Linear oscillation: sin(2πt)'
        }
        
        # 7. Exponential decay
        exp_decay = np.exp(-t / 50)
        signals['exponential_decay'] = {
            'signal': exp_decay,
            'has_sqrt_pattern': False,
            'description': 'Exponential decay: exp(-t/50)'
        }
        
        # 8. Random noise
        noise = np.random.normal(0, 1, len(t))
        signals['random_noise'] = {
            'signal': noise,
            'has_sqrt_pattern': False,
            'description': 'Random Gaussian noise'
        }
        
        # 9. Linear frequency modulation
        linear_freq_mod = np.sin(2 * np.pi * t * t)
        signals['linear_frequency_modulation'] = {
            'signal': linear_freq_mod,
            'has_sqrt_pattern': False,
            'description': 'Linear frequency modulation: sin(2πt²)'
        }
        
        # 10. Chaotic signal (Lorenz system)
        def lorenz_system(t, sigma=10, rho=28, beta=8/3):
            dt = t[1] - t[0]
            x = np.zeros_like(t)
            y = np.zeros_like(t)
            z = np.zeros_like(t)
            
            x[0], y[0], z[0] = 1, 1, 1
            
            for i in range(1, len(t)):
                dx = sigma * (y[i-1] - x[i-1])
                dy = x[i-1] * (rho - z[i-1]) - y[i-1]
                dz = x[i-1] * y[i-1] - beta * z[i-1]
                
                x[i] = x[i-1] + dx * dt
                y[i] = y[i-1] + dy * dt
                z[i] = z[i-1] + dz * dt
            
            return x
        
        chaotic_signal = lorenz_system(t)
        signals['chaotic_signal'] = {
            'signal': chaotic_signal,
            'has_sqrt_pattern': False,
            'description': 'Chaotic signal (Lorenz system)'
        }
        
        return signals
    
    def compute_transform_features(self, signal, name="signal"):
        """
        Compute comprehensive features from the √t transform.
        
        Returns:
            dict: Dictionary of transform features
        """
        try:
            # Use comprehensive test from improved transform
            results = self.transform.comprehensive_test(signal, name)
            
            # Extract key features
            features = {
                'signal_name': name,
                'signal_length': len(signal),
                'mean_magnitude': np.mean(results['magnitudes']) if results['magnitudes'] else 0,
                'max_magnitude': np.max(results['magnitudes']) if results['magnitudes'] else 0,
                'std_magnitude': np.std(results['magnitudes']) if results['magnitudes'] else 0,
                'feature_count': len(results['features']) if results['features'] else 0,
                'window_scores': results.get('window_scores', {}),
                'parameter_scores': results.get('parameter_scores', {}),
                'detection_scores': results.get('detection_scores', {}),
                'overall_score': results.get('overall_score', 0)
            }
            
            return features
            
        except Exception as e:
            print(f"Error computing features for {name}: {e}")
            return None
    
    def statistical_validation(self, ground_truth_results, n_permutations=1000):
        """
        Perform statistical validation of the transform.
        
        Args:
            ground_truth_results: Results from ground truth signals
            n_permutations: Number of permutations for null distribution
            
        Returns:
            dict: Statistical validation results
        """
        print("Performing statistical validation...")
        
        # Separate ground truth and control signals
        sqrt_signals = {k: v for k, v in ground_truth_results.items() 
                       if v and v.get('has_sqrt_pattern', False)}
        control_signals = {k: v for k, v in ground_truth_results.items() 
                          if v and not v.get('has_sqrt_pattern', False)}
        
        # Extract scores
        sqrt_scores = [v['overall_score'] for v in sqrt_signals.values() if v]
        control_scores = [v['overall_score'] for v in control_signals.values() if v]
        
        # Statistical tests
        results = {}
        
        if sqrt_scores and control_scores:
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(sqrt_scores, control_scores, 
                                                alternative='greater')
            results['mann_whitney'] = {
                'u_statistic': u_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(sqrt_scores) - 1) * np.var(sqrt_scores) + 
                                 (len(control_scores) - 1) * np.var(control_scores)) / 
                                (len(sqrt_scores) + len(control_scores) - 2))
            cohens_d = (np.mean(sqrt_scores) - np.mean(control_scores)) / pooled_std
            results['effect_size'] = {
                'cohens_d': cohens_d,
                'interpretation': self.interpret_cohens_d(cohens_d)
            }
            
            # Permutation test
            all_scores = sqrt_scores + control_scores
            n_sqrt = len(sqrt_scores)
            permutation_p_values = []
            
            for _ in range(n_permutations):
                np.random.shuffle(all_scores)
                perm_sqrt = all_scores[:n_sqrt]
                perm_control = all_scores[n_sqrt:]
                perm_diff = np.mean(perm_sqrt) - np.mean(perm_control)
                real_diff = np.mean(sqrt_scores) - np.mean(control_scores)
                permutation_p_values.append(perm_diff >= real_diff)
            
            perm_p_value = np.mean(permutation_p_values)
            results['permutation_test'] = {
                'p_value': perm_p_value,
                'significant': perm_p_value < 0.05
            }
        
        # Summary statistics
        results['summary'] = {
            'n_sqrt_signals': len(sqrt_scores),
            'n_control_signals': len(control_scores),
            'mean_sqrt_score': np.mean(sqrt_scores) if sqrt_scores else 0,
            'mean_control_score': np.mean(control_scores) if control_scores else 0,
            'std_sqrt_score': np.std(sqrt_scores) if sqrt_scores else 0,
            'std_control_score': np.std(control_scores) if control_scores else 0
        }
        
        return results
    
    def interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size"""
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"
    
    def mathematical_property_validation(self):
        """
        Validate mathematical properties of the transform.
        
        Returns:
            dict: Mathematical validation results
        """
        print("Validating mathematical properties...")
        
        results = {}
        
        # Test 1: Linearity
        t = np.linspace(0, 100, 1000)
        f1 = np.sin(2 * np.pi * np.sqrt(t))
        f2 = np.cos(2 * np.pi * np.sqrt(t))
        f_sum = f1 + f2
        
        # Apply transform to individual signals and sum
        W1 = self.transform.transform_with_window(f1, [0.1, 0.5, 1.0], [10, 50, 100])
        W2 = self.transform.transform_with_window(f2, [0.1, 0.5, 1.0], [10, 50, 100])
        W_sum = self.transform.transform_with_window(f_sum, [0.1, 0.5, 1.0], [10, 50, 100])
        
        # Check linearity
        W_expected = W1 + W2
        linearity_error = np.mean(np.abs(W_sum - W_expected))
        results['linearity'] = {
            'error': linearity_error,
            'is_linear': linearity_error < 1e-10
        }
        
        # Test 2: Scale invariance (for √t scaling)
        # Create signal with √t scaling
        sqrt_scaled = np.sin(2 * np.pi * np.sqrt(t))
        
        # Scale time axis
        t_scaled = t * 4  # Scale by factor of 4
        sqrt_scaled_shifted = np.sin(2 * np.pi * np.sqrt(t_scaled))
        
        # Apply transform with scaled parameters
        W_original = self.transform.transform_with_window(sqrt_scaled, [0.1, 0.5, 1.0], [10, 50, 100])
        W_scaled = self.transform.transform_with_window(sqrt_scaled_shifted, [0.1/2, 0.5/2, 1.0/2], [5, 25, 50])
        
        # Check scale invariance
        scale_invariance_error = np.mean(np.abs(np.abs(W_original) - np.abs(W_scaled)))
        results['scale_invariance'] = {
            'error': scale_invariance_error,
            'is_scale_invariant': scale_invariance_error < 0.1
        }
        
        # Test 3: Energy conservation
        # Check if transform preserves signal energy
        signal_energy = np.sum(f1**2)
        transform_energy = np.sum(np.abs(W1)**2)
        energy_ratio = transform_energy / signal_energy
        results['energy_conservation'] = {
            'signal_energy': signal_energy,
            'transform_energy': transform_energy,
            'energy_ratio': energy_ratio,
            'conserves_energy': 0.1 < energy_ratio < 10.0
        }
        
        return results
    
    def create_validation_report(self, ground_truth_results, statistical_results, 
                                mathematical_results, output_dir="validation_results"):
        """
        Create comprehensive validation report.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Ground truth vs control scores
        sqrt_scores = [v['overall_score'] for v in ground_truth_results.values() 
                      if v and v.get('has_sqrt_pattern', False)]
        control_scores = [v['overall_score'] for v in ground_truth_results.values() 
                         if v and not v.get('has_sqrt_pattern', False)]
        
        ax1 = axes[0, 0]
        ax1.boxplot([sqrt_scores, control_scores], labels=['√t Signals', 'Control Signals'])
        ax1.set_title('Transform Scores: Ground Truth vs Controls')
        ax1.set_ylabel('Overall Score')
        ax1.grid(True, alpha=0.3)
        
        # 2. Individual signal scores
        ax2 = axes[0, 1]
        signal_names = list(ground_truth_results.keys())
        scores = [ground_truth_results[name]['overall_score'] 
                 for name in signal_names if ground_truth_results[name]]
        colors = ['green' if ground_truth_results[name].get('has_sqrt_pattern', False) 
                 else 'red' for name in signal_names if ground_truth_results[name]]
        
        bars = ax2.bar(range(len(scores)), scores, color=colors, alpha=0.7)
        ax2.set_title('Individual Signal Scores')
        ax2.set_ylabel('Overall Score')
        ax2.set_xticks(range(len(signal_names)))
        ax2.set_xticklabels(signal_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Statistical test results
        ax3 = axes[1, 0]
        if statistical_results.get('mann_whitney'):
            tests = ['Mann-Whitney U', 'Permutation Test']
            p_values = [statistical_results['mann_whitney']['p_value'],
                       statistical_results['permutation_test']['p_value']]
            colors = ['green' if p < 0.05 else 'red' for p in p_values]
            
            bars = ax3.bar(tests, p_values, color=colors, alpha=0.7)
            ax3.axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
            ax3.set_title('Statistical Test Results')
            ax3.set_ylabel('p-value')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Mathematical property validation
        ax4 = axes[1, 1]
        if mathematical_results:
            properties = ['Linearity', 'Scale Invariance', 'Energy Conservation']
            errors = [mathematical_results['linearity']['error'],
                     mathematical_results['scale_invariance']['error'],
                     mathematical_results['energy_conservation']['energy_ratio']]
            colors = ['green' if mathematical_results['linearity']['is_linear'] else 'red',
                     'green' if mathematical_results['scale_invariance']['is_scale_invariant'] else 'red',
                     'green' if mathematical_results['energy_conservation']['conserves_energy'] else 'red']
            
            bars = ax4.bar(properties, errors, color=colors, alpha=0.7)
            ax4.set_title('Mathematical Property Validation')
            ax4.set_ylabel('Error/Ratio')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{output_dir}/validation_report_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved validation report to: {plot_filename}")
        
        # Save detailed results
        import json
        results_filename = f"{output_dir}/validation_results_{timestamp}.json"
        all_results = {
            'ground_truth_results': ground_truth_results,
            'statistical_results': statistical_results,
            'mathematical_results': mathematical_results
        }
        with open(results_filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Saved detailed results to: {results_filename}")
        
        plt.show()
        
        # Print summary
        self.print_validation_summary(ground_truth_results, statistical_results, mathematical_results)
    
    def print_validation_summary(self, ground_truth_results, statistical_results, mathematical_results):
        """Print validation summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        # Ground truth performance
        sqrt_signals = {k: v for k, v in ground_truth_results.items() 
                       if v and v.get('has_sqrt_pattern', False)}
        control_signals = {k: v for k, v in ground_truth_results.items() 
                          if v and not v.get('has_sqrt_pattern', False)}
        
        sqrt_scores = [v['overall_score'] for v in sqrt_signals.values() if v]
        control_scores = [v['overall_score'] for v in control_signals.values() if v]
        
        print(f"\nGround Truth Performance:")
        print(f"  √t signals detected: {len(sqrt_scores)}")
        print(f"  Control signals detected: {len(control_scores)}")
        if sqrt_scores:
            print(f"  Mean √t score: {np.mean(sqrt_scores):.4f}")
        if control_scores:
            print(f"  Mean control score: {np.mean(control_scores):.4f}")
        
        # Statistical validation
        if statistical_results.get('mann_whitney'):
            print(f"\nStatistical Validation:")
            print(f"  Mann-Whitney U test: p = {statistical_results['mann_whitney']['p_value']:.4f}")
            print(f"  Significant difference: {statistical_results['mann_whitney']['significant']}")
            if 'effect_size' in statistical_results:
                print(f"  Effect size (Cohen's d): {statistical_results['effect_size']['cohens_d']:.4f}")
                print(f"  Effect interpretation: {statistical_results['effect_size']['interpretation']}")
        
        # Mathematical validation
        if mathematical_results:
            print(f"\nMathematical Properties:")
            print(f"  Linearity: {mathematical_results['linearity']['is_linear']}")
            print(f"  Scale invariance: {mathematical_results['scale_invariance']['is_scale_invariant']}")
            print(f"  Energy conservation: {mathematical_results['energy_conservation']['conserves_energy']}")
        
        # Overall assessment
        print(f"\nOverall Assessment:")
        if (statistical_results.get('mann_whitney', {}).get('significant', False) and
            mathematical_results.get('linearity', {}).get('is_linear', False)):
            print("  ✅ Transform VALIDATED - Passes statistical and mathematical tests")
        elif statistical_results.get('mann_whitney', {}).get('significant', False):
            print("  ⚠️  Transform PARTIALLY VALIDATED - Passes statistical tests but fails mathematical properties")
        else:
            print("  ❌ Transform NOT VALIDATED - Fails statistical tests")
    
    def run_complete_validation(self):
        """
        Run the complete validation pipeline.
        """
        print("=== Complete √t Transform Validation ===")
        
        # Generate ground truth signals
        t = np.linspace(0, 100, 1000)
        ground_truth_signals = self.generate_ground_truth_signals(t)
        
        # Compute transform features for all signals
        ground_truth_results = {}
        for name, signal_info in ground_truth_signals.items():
            print(f"Analyzing {name}...")
            features = self.compute_transform_features(signal_info['signal'], name)
            if features:
                features['has_sqrt_pattern'] = signal_info['has_sqrt_pattern']
                features['description'] = signal_info['description']
                ground_truth_results[name] = features
        
        # Statistical validation
        statistical_results = self.statistical_validation(ground_truth_results)
        
        # Mathematical property validation
        mathematical_results = self.mathematical_property_validation()
        
        # Create comprehensive report
        self.create_validation_report(ground_truth_results, statistical_results, mathematical_results)
        
        return ground_truth_results, statistical_results, mathematical_results

if __name__ == "__main__":
    validator = ProperValidationFramework()
    validator.run_complete_validation() 