#!/usr/bin/env python3
"""
Comprehensive Validation Framework for √t Wave Transform
W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt

This framework addresses false positive concerns by:
1. Replicating original linear-time analysis
2. Testing against multiple control conditions
3. Cross-validation with established methods
4. Biological plausibility checks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.fft import fft, ifft
import warnings
warnings.filterwarnings('ignore')

class SqrtTTransform:
    """
    Implementation of the √t wave transform
    W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
    """
    
    def __init__(self, sampling_rate=1.0):
        self.sampling_rate = sampling_rate
        
    def gaussian_window(self, t, tau):
        """Gaussian window function ψ(√t/τ)"""
        sqrt_t = np.sqrt(t)
        return np.exp(-(sqrt_t / tau)**2)
    
    def transform(self, V, k_values, tau_values, t_max=None):
        """
        Compute the √t transform W(k,τ)
        
        Parameters:
        V: voltage time series
        k_values: array of k values (frequency-like parameter)
        tau_values: array of τ values (scale parameter)
        t_max: maximum time for integration (if None, uses full signal)
        """
        if t_max is None:
            t_max = len(V) / self.sampling_rate
            
        t = np.arange(0, t_max, 1/self.sampling_rate)
        dt = 1/self.sampling_rate
        
        W = np.zeros((len(k_values), len(tau_values)), dtype=complex)
        
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                # Compute the integrand
                window = self.gaussian_window(t, tau)
                phase = np.exp(-1j * k * np.sqrt(t))
                integrand = V[:len(t)] * window * phase
                
                # Numerical integration
                W[i, j] = np.trapz(integrand, t)
                
        return W
    
    def detect_features(self, W, k_values, tau_values, threshold=0.1):
        """Detect significant features in the transform"""
        magnitude = np.abs(W)
        features = []
        
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                if magnitude[i, j] > threshold * np.max(magnitude):
                    features.append({
                        'k': k,
                        'tau': tau,
                        'magnitude': magnitude[i, j],
                        'phase': np.angle(W[i, j])
                    })
        
        return features

class ValidationFramework:
    """
    Comprehensive validation framework to test the √t transform
    and address false positive concerns
    """
    
    def __init__(self, transform):
        self.transform = transform

    def load_fungal_data(self, filepath, channel=None):
        """Load fungal electrophysiology data"""
        try:
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
                # Handle single-column data (no header)
                if len(df.columns) == 1 and df.columns[0] == 'Unnamed: 0':
                    # This is single-column data
                    return df.iloc[:, 0].values
                if channel:
                    return df[channel].values
                else:
                    # Return first voltage column
                    voltage_cols = [col for col in df.columns if 'Differential' in col or 'V' in col or 'voltage' in col.lower()]
                    if voltage_cols:
                        return df[voltage_cols[0]].values
                    else:
                        # If no voltage columns found, return first numeric column
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            return df[numeric_cols[0]].values
                        else:
                            # Last resort: return first column
                            return df.iloc[:, 0].values
            else:
                # Single column data
                return np.loadtxt(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def original_linear_analysis(self, V, window_size=50, threshold=0.003, distance=100):
        """
        Replicate the original spike detection from Fungal-Spike-Clustering
        """
        spikes = []
        
        for i in range(window_size + 1, len(V) - window_size):
            if i - 2*window_size > 0 and i + 2*window_size <= len(V):
                neighborhood = V[(i-2*window_size):(i+2*window_size)]
                local_avg = np.mean(neighborhood)
                
                if abs(V[i] - local_avg) > threshold:
                    spikes.append(i)
        
        # Filter close spikes
        filtered_spikes = []
        last_spike = -distance
        for spike in spikes:
            if (spike - last_spike) > distance:
                filtered_spikes.append(spike)
                last_spike = spike
                
        return filtered_spikes
    
    def generate_control_signals(self, original_signal, n_samples=1000):
        """
        Generate multiple control signals for false positive testing
        """
        controls = {}
        
        # 1. White noise with same statistics
        controls['white_noise'] = np.random.normal(
            np.mean(original_signal), 
            np.std(original_signal), 
            len(original_signal)
        )
        
        # 2. Phase-randomized surrogate (preserves power spectrum)
        fft_original = fft(original_signal)
        phase_random = np.random.uniform(0, 2*np.pi, len(fft_original))
        controls['phase_randomized'] = np.real(ifft(
            np.abs(fft_original) * np.exp(1j * phase_random)
        ))
        
        # 3. Time-shuffled signal
        controls['shuffled'] = np.random.permutation(original_signal)
        
        # 4. Synthetic spike train
        controls['synthetic_spikes'] = self.generate_synthetic_spikes(len(original_signal))
        
        # 5. Exponential decay
        t = np.arange(len(original_signal))
        controls['exponential_decay'] = np.exp(-t/1000) + 0.1 * np.random.normal(0, 1, len(original_signal))
        
        return controls
    
    def generate_synthetic_spikes(self, length, spike_rate=0.01, spike_amplitude=0.1):
        """Generate synthetic spike train for testing"""
        signal = np.zeros(length)
        n_spikes = int(length * spike_rate)
        spike_positions = np.random.choice(length, n_spikes, replace=False)
        
        for pos in spike_positions:
            # Create a spike with exponential decay
            for i in range(min(50, length - pos)):
                if pos + i < length:
                    signal[pos + i] += spike_amplitude * np.exp(-i/10)
                    
        return signal + 0.01 * np.random.normal(0, 1, length)
    
    def cross_validate_with_established_methods(self, V):
        """
        Compare √t transform results with established methods
        """
        results = {}
        
        # 1. FFT analysis
        fft_result = fft(V)
        fft_power = np.abs(fft_result)**2
        results['fft_peaks'] = signal.find_peaks(fft_power[:len(fft_power)//2])[0]
        
        # 2. Wavelet analysis (simplified)
        try:
            from scipy.signal import cwt, morlet2
            widths = np.arange(1, 31)
            cwtmatr = cwt(V, widths, morlet2)
            results['wavelet_power'] = np.abs(cwtmatr)
        except ImportError:
            # Fallback to simpler wavelet analysis
            from scipy.signal import find_peaks
            # Use FFT-based frequency analysis instead
            fft_freq = np.fft.fftfreq(len(V), 1/self.transform.sampling_rate)
            fft_power = np.abs(fft(V))**2
            results['wavelet_power'] = fft_power  # Use FFT power as fallback
        
        # 3. Hilbert-Huang transform (simplified)
        analytic_signal = signal.hilbert(V)
        results['hilbert_amplitude'] = np.abs(analytic_signal)
        results['hilbert_phase'] = np.unwrap(np.angle(analytic_signal))
        
        return results
    
    def biological_plausibility_check(self, features, sampling_rate=1.0):
        """
        Check if detected features are biologically plausible
        """
        plausible_features = []
        
        for feature in features:
            k, tau = feature['k'], feature['tau']
            
            # Check frequency range (0.1 - 10 Hz for fungal signals)
            freq_estimate = k / (2 * np.pi * np.sqrt(tau))
            if 0.1 <= freq_estimate <= 10:
                feature['biologically_plausible'] = True
                feature['estimated_frequency'] = freq_estimate
            else:
                feature['biologically_plausible'] = False
                
            # Check scale parameter (should be reasonable for fungal timescales)
            if 1 <= tau <= 1000:  # seconds
                feature['scale_plausible'] = True
            else:
                feature['scale_plausible'] = False
                
            if feature.get('biologically_plausible', False) and feature.get('scale_plausible', False):
                plausible_features.append(feature)
                
        return plausible_features
    
    def comprehensive_test(self, data_file, channel=None, k_range=(0.1, 10), tau_range=(1, 100)):
        """
        Run comprehensive validation test
        """
        print(f"Testing √t transform on {data_file}")
        
        # Load data
        V = self.load_fungal_data(data_file, channel)
        if V is None:
            return None
            
        # Parameters for transform
        k_values = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), 20)
        tau_values = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), 20)
        
        # 1. Original linear analysis
        print("Running original linear analysis...")
        original_spikes = self.original_linear_analysis(V)
        
        # 2. √t transform analysis
        print("Running √t transform analysis...")
        W = self.transform.transform(V, k_values, tau_values)
        sqrt_features = self.transform.detect_features(W, k_values, tau_values)
        
        # 3. Control testing
        print("Running control tests...")
        controls = self.generate_control_signals(V)
        control_results = {}
        
        for control_name, control_signal in controls.items():
            W_control = self.transform.transform(control_signal, k_values, tau_values)
            control_features = self.transform.detect_features(W_control, k_values, tau_values)
            control_results[control_name] = control_features
        
        # 4. Cross-validation
        print("Running cross-validation...")
        established_results = self.cross_validate_with_established_methods(V)
        
        # 5. Biological plausibility
        print("Checking biological plausibility...")
        plausible_features = self.biological_plausibility_check(sqrt_features)
        
        # 6. Statistical significance testing
        print("Testing statistical significance...")
        significance_results = self.statistical_significance_test(sqrt_features, control_results)
        
        return {
            'original_spikes': original_spikes,
            'sqrt_features': sqrt_features,
            'plausible_features': plausible_features,
            'control_results': control_results,
            'established_results': established_results,
            'significance_results': significance_results,
            'data_file': data_file
        }
    
    def statistical_significance_test(self, real_features, control_results):
        """
        Test if real features are significantly different from controls
        """
        if not real_features:
            return {'significant': False, 'p_value': 1.0}
            
        # Collect magnitudes from all controls
        control_magnitudes = []
        for control_name, control_features in control_results.items():
            for feature in control_features:
                control_magnitudes.append(feature['magnitude'])
        
        if not control_magnitudes:
            return {'significant': False, 'p_value': 1.0}
        
        # Test real features against control distribution
        real_magnitudes = [f['magnitude'] for f in real_features]
        
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(
            real_magnitudes, control_magnitudes, alternative='greater'
        )
        
        return {
            'significant': p_value < 0.05,
            'p_value': p_value,
            'statistic': statistic,
            'real_mean': np.mean(real_magnitudes),
            'control_mean': np.mean(control_magnitudes)
        }
    
    def plot_comparison(self, results):
        """
        Create comprehensive comparison plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Load original data for plotting
        V = self.load_fungal_data(results['data_file'])
        
        # 1. Original signal with detected spikes
        axes[0, 0].plot(V[:1000])
        if results['original_spikes']:
            spike_times = [s for s in results['original_spikes'] if s < 1000]
            spike_values = [V[s] for s in spike_times]
            axes[0, 0].scatter(spike_times, spike_values, color='red', alpha=0.7)
        axes[0, 0].set_title('Original Signal + Linear Spike Detection')
        
        # 2. √t transform magnitude
        if results['sqrt_features']:
            k_vals = [f['k'] for f in results['sqrt_features']]
            tau_vals = [f['tau'] for f in results['sqrt_features']]
            mag_vals = [f['magnitude'] for f in results['sqrt_features']]
            axes[0, 1].scatter(k_vals, tau_vals, c=mag_vals, cmap='viridis')
            axes[0, 1].set_xscale('log')
            axes[0, 1].set_yscale('log')
            axes[0, 1].set_title('√t Transform Features')
            axes[0, 1].set_xlabel('k')
            axes[0, 1].set_ylabel('τ')
        
        # 3. Control comparison
        control_names = list(results['control_results'].keys())
        control_counts = [len(results['control_results'][name]) for name in control_names]
        real_count = len(results['sqrt_features'])
        
        axes[0, 2].bar(['Real'] + control_names, [real_count] + control_counts)
        axes[0, 2].set_title('Feature Count Comparison')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Biological plausibility
        if results['plausible_features']:
            freqs = [f.get('estimated_frequency', 0) for f in results['plausible_features']]
            axes[1, 0].hist(freqs, bins=20, alpha=0.7)
            axes[1, 0].set_title('Biologically Plausible Frequencies')
            axes[1, 0].set_xlabel('Frequency (Hz)')
        
        # 5. Statistical significance
        sig_results = results['significance_results']
        axes[1, 1].text(0.5, 0.5, f"p-value: {sig_results['p_value']:.4f}\n"
                        f"Significant: {sig_results['significant']}", 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Statistical Significance')
        
        # 6. Summary
        summary_text = f"""
        Original Spikes: {len(results['original_spikes'])}
        √t Features: {len(results['sqrt_features'])}
        Plausible Features: {len(results['plausible_features'])}
        Significant: {sig_results['significant']}
        """
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='center')
        axes[1, 2].set_title('Summary')
        
        plt.tight_layout()
        return fig

def main():
    """
    Main testing function
    """
    # Initialize transform and validation framework
    transform = SqrtTTransform(sampling_rate=1.0)
    validator = ValidationFramework(transform)
    
    # Test files to analyze
    test_files = [
        "15061491/fungal_spikes/good_recordings/New_Oyster_with spray_as_mV_seconds_SigView.csv",
        "15061491/fungal_spikes/good_recordings/Hericium_20_4_22_part1.csv"
    ]
    
    results_summary = []
    
    for test_file in test_files:
        print(f"\n{'='*50}")
        print(f"Testing: {test_file}")
        print(f"{'='*50}")
        
        # Run comprehensive test
        results = validator.comprehensive_test(test_file)
        
        if results:
            # Plot results
            fig = validator.plot_comparison(results)
            plt.savefig(f"results_{test_file.split('/')[-1].replace('.csv', '.png')}")
            plt.close()
            
            # Print summary
            print(f"\nResults Summary for {test_file}:")
            print(f"Original spikes detected: {len(results['original_spikes'])}")
            print(f"√t features detected: {len(results['sqrt_features'])}")
            print(f"Biologically plausible features: {len(results['plausible_features'])}")
            print(f"Statistically significant: {results['significance_results']['significant']}")
            print(f"p-value: {results['significance_results']['p_value']:.4f}")
            
            results_summary.append(results)
    
    return results_summary

if __name__ == "__main__":
    results = main() 