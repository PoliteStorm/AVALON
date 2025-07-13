"""
√t Wavelet Transform Implementation

Implements the transform: W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt

This transform is designed to detect patterns that scale with √t time,
which is relevant for many biological systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import quad
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class SqrtWaveletTransform:
    """
    Implementation of the √t wavelet transform:
    W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
    
    This transform is specifically designed to detect patterns
    that scale with √t time rather than linear time.
    """
    
    def __init__(self, sampling_rate: float = 10.0, 
                 k_range: Tuple[float, float] = (0.1, 10.0),
                 tau_range: Tuple[float, float] = (0.1, 10.0),
                 n_k: int = 50, n_tau: int = 50):
        """
        Initialize the √t wavelet transform.
        
        Args:
            sampling_rate: Sampling rate in Hz
            k_range: Range of k values (frequency-like parameter)
            tau_range: Range of τ values (time scale parameter)
            n_k: Number of k values to compute
            n_tau: Number of τ values to compute
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        
        # Define k and τ ranges
        self.k_values = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), n_k)
        self.tau_values = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), n_tau)
        
        # Wavelet function (Morlet-like wavelet adapted for √t scaling)
        self.wavelet_family = 'sqrt_morlet'
        
    def sqrt_morlet_wavelet(self, t: np.ndarray, tau: float) -> np.ndarray:
        """
        √t-scaled Morlet wavelet: ψ(√t/τ)
        
        This wavelet is designed to be sensitive to √t-scaled patterns
        rather than linear time patterns.
        """
        # Normalize time by τ
        sqrt_t_normalized = np.sqrt(t) / np.sqrt(tau)
        
        # Morlet wavelet adapted for √t scaling
        # Center frequency and bandwidth parameters
        omega_0 = 2 * np.pi  # Central frequency
        sigma = 1.0  # Bandwidth parameter
        
        # Complex Morlet wavelet with √t scaling
        wavelet = np.exp(1j * omega_0 * sqrt_t_normalized) * np.exp(-0.5 * (sqrt_t_normalized / sigma)**2)
        
        # Normalize
        norm_factor = np.sqrt(np.sum(np.abs(wavelet)**2))
        if norm_factor > 0:
            wavelet = wavelet / norm_factor
            
        return wavelet
    
    def compute_transform_kernel(self, t: np.ndarray, k: float, tau: float) -> np.ndarray:
        """
        Compute the transform kernel: ψ(√t/τ) · e^(-ik√t)
        
        This is the core of the √t wavelet transform.
        """
        # Get the wavelet
        wavelet = self.sqrt_morlet_wavelet(t, tau)
        
        # Compute the √t-scaled exponential
        sqrt_t = np.sqrt(t)
        exponential = np.exp(-1j * k * sqrt_t)
        
        # Combine wavelet and exponential
        kernel = wavelet * exponential
        
        return kernel
    
    def transform_single_point(self, signal: np.ndarray, k: float, tau: float) -> complex:
        """
        Compute W(k,τ) for a single (k,τ) point.
        
        Args:
            signal: Input signal V(t)
            k: Frequency-like parameter
            tau: Time scale parameter
            
        Returns:
            W(k,τ): Complex transform value
        """
        # Time array
        t = np.arange(len(signal)) * self.dt
        
        # Compute kernel
        kernel = self.compute_transform_kernel(t, k, tau)
        
        # Apply transform (discrete approximation of integral)
        # W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
        transform_value = np.sum(signal * kernel) * self.dt
        
        return transform_value
    
    def analyze_signal(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the √t wavelet transform to a signal.
        
        Args:
            signal: Input signal V(t)
            
        Returns:
            coeffs: Complex wavelet coefficients W(k,τ)
            magnitude: Magnitude of coefficients |W(k,τ)|
            phase: Phase of coefficients arg(W(k,τ))
        """
        print(f"Computing √t wavelet transform for signal of length {len(signal)}")
        print(f"k range: [{self.k_values[0]:.3f}, {self.k_values[-1]:.3f}]")
        print(f"τ range: [{self.tau_values[0]:.3f}, {self.tau_values[-1]:.3f}]")
        
        # Initialize coefficient matrix
        coeffs = np.zeros((len(self.tau_values), len(self.k_values)), dtype=complex)
        
        # Compute transform for each (k,τ) pair
        total_points = len(self.tau_values) * len(self.k_values)
        current_point = 0
        
        for i, tau in enumerate(self.tau_values):
            for j, k in enumerate(self.k_values):
                # Compute W(k,τ)
                coeffs[i, j] = self.transform_single_point(signal, k, tau)
                
                # Progress indicator
                current_point += 1
                if current_point % 100 == 0:
                    progress = (current_point / total_points) * 100
                    print(f"Progress: {progress:.1f}%")
        
        # Compute magnitude and phase
        magnitude = np.abs(coeffs)
        phase = np.angle(coeffs)
        
        print("√t wavelet transform computation complete!")
        
        return coeffs, magnitude, phase
    
    def analyze_signal_fast(self, signal: np.ndarray, max_length: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast version of the transform for validation and testing.
        Uses a subset of the signal for speed.
        """
        # Use subset for faster computation
        if len(signal) > max_length:
            # Take evenly spaced samples
            indices = np.linspace(0, len(signal)-1, max_length, dtype=int)
            signal_subset = signal[indices]
        else:
            signal_subset = signal
            
        return self.analyze_signal(signal_subset)
    
    def plot_transform(self, magnitude: np.ndarray, phase: np.ndarray, 
                      title: str = "√t Wavelet Transform") -> None:
        """
        Plot the transform results.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Magnitude plot
        im1 = axes[0].imshow(magnitude, cmap='viridis', aspect='auto',
                             extent=[self.k_values[0], self.k_values[-1], 
                                    self.tau_values[0], self.tau_values[-1]])
        axes[0].set_xlabel('k')
        axes[0].set_ylabel('τ')
        axes[0].set_title(f'{title} - Magnitude')
        plt.colorbar(im1, ax=axes[0])
        
        # Phase plot
        im2 = axes[1].imshow(phase, cmap='twilight', aspect='auto',
                             extent=[self.k_values[0], self.k_values[-1], 
                                    self.tau_values[0], self.tau_values[-1]])
        axes[1].set_xlabel('k')
        axes[1].set_ylabel('τ')
        axes[1].set_title(f'{title} - Phase')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
    
    def compute_sqrt_signature(self, magnitude: np.ndarray) -> float:
        """
        Compute the √t signature strength.
        
        This measures how well the transform follows the expected
        √t scaling relationship.
        """
        try:
            # Expected √t scaling: magnitude should scale as τ^(-0.5) × k^(-1.5)
            tau_indices = np.arange(magnitude.shape[0])
            k_indices = np.arange(magnitude.shape[1])
            
            # Expected scaling pattern
            expected_scaling = np.outer(tau_indices**(-0.5), k_indices**(-1.5))
            expected_scaling = expected_scaling / np.max(expected_scaling)
            
            # Flatten arrays for correlation
            mag_flat = magnitude.flatten()
            expected_flat = expected_scaling.flatten()
            
            # Remove any NaN or inf values
            valid_mask = ~(np.isnan(mag_flat) | np.isinf(mag_flat) | 
                          np.isnan(expected_flat) | np.isinf(expected_flat))
            
            if np.sum(valid_mask) < 10:  # Need at least 10 valid points
                return 0.0
                
            mag_valid = mag_flat[valid_mask]
            expected_valid = expected_flat[valid_mask]
            
            # Compute correlation
            if len(mag_valid) > 1 and np.std(mag_valid) > 0:
                correlation = np.corrcoef(mag_valid, expected_valid)[0, 1]
                return float(correlation) if not np.isnan(correlation) else 0.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error computing √t signature: {e}")
            return 0.0
    
    def compute_alternative_signature(self, magnitude: np.ndarray) -> Dict[str, float]:
        """
        Compute alternative signature metrics for √t patterns.
        """
        signatures = {}
        try:
            # 1. τ-scaling signature (how magnitude scales with τ)
            tau_means = np.mean(magnitude, axis=1)
            tau_indices = np.arange(1, len(tau_means)+1)
            # Expected τ^(-0.5) scaling
            expected_tau = tau_indices**(-0.5)
            expected_tau = expected_tau / np.max(expected_tau)
            # 2. k-scaling signature (how magnitude scales with k)
            k_means = np.mean(magnitude, axis=0)
            k_indices = np.arange(1, len(k_means)+1)
            # Expected k^(-1.5) scaling
            expected_k = k_indices**(-1.5)
            expected_k = expected_k / np.max(expected_k)
            # Correlations
            if len(tau_means) > 1 and np.std(tau_means) > 0:
                tau_corr = np.corrcoef(tau_means, expected_tau)[0, 1]
                signatures['tau_scaling'] = float(tau_corr) if not np.isnan(tau_corr) else 0.0
            else:
                signatures['tau_scaling'] = 0.0
            if len(k_means) > 1 and np.std(k_means) > 0:
                k_corr = np.corrcoef(k_means, expected_k)[0, 1]
                signatures['k_scaling'] = float(k_corr) if not np.isnan(k_corr) else 0.0
            else:
                signatures['k_scaling'] = 0.0
            # 3. Overall √t signature (geometric mean of τ and k scaling)
            signatures['sqrt_signature'] = np.sqrt(signatures['tau_scaling'] * signatures['k_scaling'])
            # 4. Pattern concentration (how focused the patterns are)
            peak_mag = np.max(magnitude)
            mean_mag = np.mean(magnitude)
            signatures['pattern_concentration'] = peak_mag / mean_mag if mean_mag > 0 else 0.0
        except Exception as e:
            print(f"Error computing alternative signatures: {e}")
            signatures = {
                'tau_scaling': 0.0,
                'k_scaling': 0.0,
                'sqrt_signature': 0.0,
                'pattern_concentration': 0.0
            }
        return signatures
    
    def get_transform_info(self) -> Dict:
        """
        Get information about the transform parameters.
        """
        return {
            'transform_type': '√t Wavelet Transform',
            'equation': 'W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt',
            'sampling_rate': self.sampling_rate,
            'k_range': [self.k_values[0], self.k_values[-1]],
            'tau_range': [self.tau_values[0], self.tau_values[-1]],
            'n_k': len(self.k_values),
            'n_tau': len(self.tau_values),
            'wavelet_family': self.wavelet_family
        }


# Test function
def test_sqrt_wavelet_transform():
    """
    Test the √t wavelet transform with various signals.
    """
    print("🧪 TESTING √t WAVELET TRANSFORM")
    print("=" * 50)
    
    # Create transform
    swt = SqrtWaveletTransform(sampling_rate=10.0)
    
    # Test signals
    t = np.linspace(0, 100, 1000)
    
    # 1. Linear time signal (should show weak √t patterns)
    linear_signal = np.sin(2 * np.pi * 0.1 * t)
    
    # 2. √t time signal (should show strong √t patterns)
    sqrt_signal = np.sin(2 * np.pi * 0.1 * np.sqrt(t))
    
    # 3. Pure noise (should show minimal patterns)
    noise_signal = np.random.normal(0, 1, len(t))
    
    # Test each signal
    signals = {
        'Linear Signal': linear_signal,
        '√t Signal': sqrt_signal,
        'Noise': noise_signal
    }
    
    results = {}
    
    for name, signal in signals.items():
        print(f"\nTesting {name}...")
        
        # Apply transform
        coeffs, magnitude, phase = swt.analyze_signal_fast(signal, max_length=500)
        
        # Compute signature
        sqrt_signature = swt.compute_sqrt_signature(magnitude)
        
        # Store results
        results[name] = {
            'peak_magnitude': np.max(magnitude),
            'mean_magnitude': np.mean(magnitude),
            'sqrt_signature': sqrt_signature
        }
        
        print(f"  Peak magnitude: {results[name]['peak_magnitude']:.3f}")
        print(f"  Mean magnitude: {results[name]['mean_magnitude']:.3f}")
        print(f"  √t signature: {sqrt_signature:.3f}")
        
        # Plot results
        swt.plot_transform(magnitude, phase, title=f"√t Wavelet Transform - {name}")
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Peak magnitude: {result['peak_magnitude']:.3f}")
        print(f"  √t signature: {result['sqrt_signature']:.3f}")
        
        if '√t' in name:
            expected = "HIGH"
        elif 'Linear' in name:
            expected = "MODERATE"
        else:
            expected = "LOW"
            
        print(f"  Expected: {expected}")
        print()
    
    return results


if __name__ == "__main__":
    # Run test
    test_results = test_sqrt_wavelet_transform() 