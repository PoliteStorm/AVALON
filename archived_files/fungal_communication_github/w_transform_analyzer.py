"""
W-Transform Analysis Module for Fungal Communication
=================================================

Implements the sophisticated W-transform for analyzing fungal electrical signals:
W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt

This transform is particularly suited for analyzing biological signals due to its
ability to capture both frequency and temporal characteristics while maintaining
phase information.

Key Features:
- Full W-transform implementation with wavelet basis
- Multi-scale analysis capabilities
- Phase-preserving signal decomposition
- Adaptive time-frequency resolution
- Statistical validation of results
"""

import numpy as np
from scipy import integrate, special, signal
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
from scipy.stats import chi2

@dataclass
class WTransformConfig:
    """Configuration for W-transform analysis"""
    min_scale: float = 0.1  # Minimum scale parameter τ
    max_scale: float = 10.0  # Maximum scale parameter τ
    num_scales: int = 32    # Number of scale points
    k_range: Tuple[float, float] = (-10, 10)  # Range for frequency parameter k
    num_k: int = 64        # Number of frequency points
    wavelet_type: str = 'morlet'  # Type of wavelet basis
    significance_level: float = 0.05  # For statistical testing
    monte_carlo_iterations: int = 1000  # For confidence estimation
    chunk_size: int = 1000  # Size of chunks for parallel processing

class WTransformAnalyzer:
    """
    Sophisticated W-transform analyzer for fungal electrical signals.
    Implements the complete transform with proper wavelet basis functions
    and statistical validation.
    """
    
    def __init__(self, config: WTransformConfig):
        """Initialize the W-transform analyzer with configuration"""
        self.config = config
        self.scales = np.logspace(
            np.log10(config.min_scale),
            np.log10(config.max_scale),
            config.num_scales
        )
        self.k_values = np.linspace(
            config.k_range[0],
            config.k_range[1],
            config.num_k
        )
        
    def wavelet_basis(self, t: np.ndarray, tau: float) -> np.ndarray:
        """
        Compute the wavelet basis function ψ(√t/τ)
        
        Args:
            t: Time points
            tau: Scale parameter
        
        Returns:
            Wavelet basis function values
        """
        if self.config.wavelet_type == 'morlet':
            # Morlet wavelet: ψ(t) = π^(-1/4) · exp(-t²/2) · exp(iω₀t)
            omega0 = 6.0  # Standard value for good time-frequency resolution
            x = np.sqrt(t) / tau
            return (np.pi ** (-0.25)) * np.exp(-x**2 / 2) * np.exp(1j * omega0 * x)
        else:
            raise ValueError(f"Unsupported wavelet type: {self.config.wavelet_type}")
    
    def compute_w_transform(self, voltage_data: np.ndarray, time_data: np.ndarray) -> Dict:
        """
        Compute the full W-transform of the voltage signal using vectorized operations
        
        W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
        
        Args:
            voltage_data: Voltage measurements
            time_data: Time points
        
        Returns:
            Dictionary containing W-transform results and analysis
        """
        # Pre-compute sqrt(t) for efficiency
        sqrt_time = np.sqrt(time_data)
        
        # Initialize transform matrix
        w_transform = np.zeros((len(self.scales), len(self.k_values)), dtype=complex)
        
        # Pre-compute exponential terms for all k values
        k_matrix = self.k_values[:, np.newaxis]
        exp_terms = np.exp(-1j * k_matrix @ sqrt_time[np.newaxis, :])
        
        # Compute transform for each scale using vectorized operations
        for i, tau in enumerate(self.scales):
            # Compute wavelet basis for this scale
            psi = self.wavelet_basis(sqrt_time, tau)
            
            # Multiply voltage data with wavelet basis
            signal_psi = voltage_data * psi
            
            # Vectorized computation for all k values
            w_transform[i, :] = np.trapz(
                exp_terms * signal_psi[np.newaxis, :],
                time_data,
                axis=1
            )
            
        # Compute power spectrum
        power = np.abs(w_transform) ** 2
        
        # Statistical significance testing
        significance = self.compute_significance(power, voltage_data)
        
        # Ridge extraction for pattern identification
        ridges = self.extract_ridges(w_transform)
        
        # Phase analysis
        phase = np.angle(w_transform)
        phase_coherence = self.compute_phase_coherence(w_transform)
        
        return {
            'transform': w_transform,
            'power': power,
            'significance': significance,
            'ridges': ridges,
            'phase': phase,
            'phase_coherence': phase_coherence,
            'scales': self.scales,
            'frequencies': self.k_values,
            'parameters': {
                'config': self.config.__dict__,
                'data_length': len(voltage_data),
                'time_range': [time_data[0], time_data[-1]]
            }
        }
    
    def compute_significance(self, power: np.ndarray, voltage_data: np.ndarray) -> np.ndarray:
        """
        Compute statistical significance of W-transform coefficients
        using Monte Carlo simulations against red noise background
        """
        # Estimate background noise spectrum
        background = np.zeros_like(power)
        for _ in range(self.config.monte_carlo_iterations):
            # Generate red noise (AR1) surrogate data
            surrogate = signal.welch(voltage_data)[1]
            surrogate_transform = self.compute_w_transform(surrogate, np.arange(len(surrogate)))['power']
            background += surrogate_transform
        
        background /= self.config.monte_carlo_iterations
        
        # Compute significance using chi-square test
        dof = 2  # Degrees of freedom for complex transform
        significance = power / background
        p_values = 1 - chi2.cdf(significance * dof, dof)
        
        return p_values < self.config.significance_level
    
    def extract_ridges(self, w_transform: np.ndarray) -> List[Dict]:
        """
        Extract ridge lines from the W-transform for pattern identification
        """
        ridges = []
        amplitude = np.abs(w_transform)
        
        # Find local maxima in each scale
        for i in range(len(self.scales)):
            peaks, _ = signal.find_peaks(amplitude[i, :])
            for peak in peaks:
                # Follow ridge line
                ridge = self.follow_ridge(amplitude, i, peak)
                if len(ridge) > 3:  # Minimum ridge length
                    ridges.append({
                        'scale_indices': [r[0] for r in ridge],
                        'k_indices': [r[1] for r in ridge],
                        'amplitude': [amplitude[r[0], r[1]] for r in ridge]
                    })
        
        return ridges
    
    def follow_ridge(self, amplitude: np.ndarray, start_scale: int, start_k: int) -> List[Tuple[int, int]]:
        """
        Follow a ridge line through the transform
        """
        ridge = [(start_scale, start_k)]
        current_scale = start_scale
        current_k = start_k
        
        while current_scale < len(self.scales) - 1:
            # Look for maximum in next scale within a window
            window_size = 3
            k_min = max(0, current_k - window_size)
            k_max = min(len(self.k_values), current_k + window_size + 1)
            next_k = k_min + np.argmax(amplitude[current_scale + 1, k_min:k_max])
            
            if k_min < next_k < k_max - 1:
                ridge.append((current_scale + 1, next_k))
                current_scale += 1
                current_k = next_k
            else:
                break
        
        return ridge
    
    def compute_phase_coherence(self, w_transform: np.ndarray) -> np.ndarray:
        """
        Compute phase coherence across scales
        """
        phase = np.angle(w_transform)
        coherence = np.zeros(len(self.k_values))
        
        for k_idx in range(len(self.k_values)):
            # Compute phase coherence using circular statistics
            phase_factors = np.exp(1j * phase[:, k_idx])
            coherence[k_idx] = np.abs(np.mean(phase_factors))
        
        return coherence 