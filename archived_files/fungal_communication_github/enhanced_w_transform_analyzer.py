"""
Enhanced W-Transform Analyzer for Fungal Communication
===================================================

This module implements an advanced W-transform analysis system specifically designed
for fungal communication patterns. It combines rigorous statistical validation with
biological interpretation based on latest research findings.

Key Features:
- Multi-scale W-transform analysis
- Advanced pattern detection
- Statistical validation framework
- Biological interpretation engine
- Research-backed parameter optimization
"""

import numpy as np
from scipy import stats
from scipy.stats import norm, chi2
from scipy import signal
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import warnings

@dataclass
class WTransformConfig:
    """Configuration for enhanced W-transform analysis"""
    # Core parameters
    min_scale: float = 0.1
    max_scale: float = 10.0
    num_scales: int = 32
    k_range: Tuple[float, float] = (-10, 10)
    num_k: int = 64
    
    # Statistical validation
    significance_level: float = 0.05
    monte_carlo_iterations: int = 1000
    confidence_threshold: float = 0.95
    
    # Pattern detection
    min_pattern_confidence: float = 0.7
    min_ridge_length: int = 3
    peak_prominence: float = 0.3
    
    # Biological parameters
    voltage_range: Tuple[float, float] = (0.03, 2.1)  # mV (Adamatzky 2021)
    typical_frequencies: List[float] = [0.1, 0.5, 1.0, 2.0]  # Hz
    species_specific_timing: Dict[str, float] = {
        'Cordyceps_militaris': 116.0,  # minutes
        'Pleurotus_djamor': 98.0,
        'Schizophyllum_commune': 144.0
    }

class EnhancedWTransformAnalyzer:
    """
    Advanced W-transform analyzer for fungal communication patterns.
    Implements the sophisticated W-transform:
    W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
    """
    
    def __init__(self, config: Optional[WTransformConfig] = None):
        """Initialize the analyzer with optional custom configuration"""
        self.config = config or WTransformConfig()
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize analysis parameters"""
        # Scale parameters
        self.scales = np.logspace(
            np.log10(self.config.min_scale),
            np.log10(self.config.max_scale),
            self.config.num_scales
        )
        
        # Frequency parameters
        self.k_values = np.linspace(
            self.config.k_range[0],
            self.config.k_range[1],
            self.config.num_k
        )
        
        # Pre-compute statistical thresholds
        self.chi2_threshold = stats.chi2.ppf(
            1 - self.config.significance_level,
            df=2  # Complex transform has 2 degrees of freedom
        )
    
    def compute_w_transform(self, voltage_data: np.ndarray, 
                          time_data: np.ndarray,
                          species: str = None) -> Dict[str, Any]:
        """
        Compute the enhanced W-transform with full analysis suite
        
        Args:
            voltage_data: Voltage measurements
            time_data: Time points
            species: Optional species name for species-specific analysis
            
        Returns:
            Comprehensive analysis results
        """
        # Input validation
        if len(voltage_data) != len(time_data):
            raise ValueError("Voltage and time data must have same length")
        
        # Normalize time to start at 0
        t_normalized = time_data - time_data[0]
        sqrt_time = np.sqrt(np.abs(t_normalized) + 1e-10)
        
        # Initialize transform matrix
        w_transform = np.zeros((len(self.scales), len(self.k_values)), 
                             dtype=complex)
        
        # Pre-compute exponential terms
        k_matrix = self.k_values[:, np.newaxis]
        exp_terms = np.exp(-1j * k_matrix @ sqrt_time[np.newaxis, :])
        
        # Compute transform for each scale
        for i, tau in enumerate(self.scales):
            # Wavelet basis
            psi = self._compute_wavelet(sqrt_time, tau)
            
            # Signal * wavelet
            signal_psi = voltage_data * psi
            
            # Vectorized computation
            w_transform[i, :] = np.trapz(
                exp_terms * signal_psi[np.newaxis, :],
                time_data,
                axis=1
            )
        
        # Compute power spectrum
        power = np.abs(w_transform) ** 2
        
        # Full analysis suite
        results = {
            'transform': w_transform,
            'power': power,
            'significance': self._compute_significance(power, voltage_data),
            'patterns': self._detect_patterns(w_transform, power),
            'ridges': self._extract_ridges(w_transform),
            'biological_interpretation': self._interpret_biological(
                w_transform, power, species
            ),
            'parameters': {
                'scales': self.scales,
                'frequencies': self.k_values,
                'config': self.config.__dict__
            }
        }
        
        # Add validation metrics
        results.update(self._compute_validation_metrics(results))
        
        return results
    
    def _compute_wavelet(self, sqrt_time: np.ndarray, tau: float) -> np.ndarray:
        """Compute the wavelet basis function"""
        psi_arg = sqrt_time / tau
        return np.exp(-psi_arg**2 / 2)  # Gaussian wavelet
    
    def _compute_significance(self, power: np.ndarray, 
                            voltage_data: np.ndarray) -> Dict[str, Any]:
        """Compute statistical significance using Monte Carlo"""
        try:
            background = np.zeros_like(power)
            
            for _ in range(self.config.monte_carlo_iterations):
                # Generate surrogate data
                surrogate = np.random.permutation(voltage_data)
                surrogate_transform = self.compute_w_transform(
                    surrogate,
                    np.arange(len(surrogate))
                )['power']
                background += surrogate_transform
            
            background /= self.config.monte_carlo_iterations
            
            # Compute significance using updated stats functions
            significance = power / (background + 1e-10)
            p_values = 1 - chi2.cdf(significance * 2, df=2)
            
            # Normalize significance scores
            z_scores = (significance - np.mean(significance)) / (np.std(significance) + 1e-10)
            norm_p_values = 1 - norm.cdf(np.abs(z_scores))
            
            return {
                'significant_coefficients': p_values < self.config.significance_level,
                'p_values': p_values,
                'z_scores': z_scores,
                'normalized_p_values': norm_p_values,
                'background_spectrum': background
            }
            
        except Exception as e:
            warnings.warn(f"Error in significance computation: {str(e)}")
            return {
                'significant_coefficients': np.zeros_like(power, dtype=bool),
                'p_values': np.ones_like(power),
                'z_scores': np.zeros_like(power),
                'normalized_p_values': np.ones_like(power),
                'background_spectrum': np.zeros_like(power),
                'error': str(e)
            }
    
    def _detect_patterns(self, w_transform: np.ndarray, 
                        power: np.ndarray) -> List[Dict[str, Any]]:
        """Detect significant patterns in the transform"""
        patterns = []
        
        # Multi-modal timescales
        tau_profile = np.mean(power, axis=0)
        peaks, properties = signal.find_peaks(
            tau_profile,
            prominence=np.max(tau_profile) * self.config.peak_prominence
        )
        
        if len(peaks) > 1:
            patterns.append({
                'type': 'multi_modal_timescales',
                'confidence': min(1.0, len(peaks) / 5.0),
                'description': f'Multiple distinct timescales: {len(peaks)} peaks',
                'peaks': peaks.tolist(),
                'timescales': self.scales[peaks].tolist()
            })
        
        # Frequency-timescale coupling
        freq_profile = np.mean(power, axis=1)
        coupling = np.corrcoef(freq_profile, np.mean(power, axis=0))[0, 1]
        
        if abs(coupling) > self.config.min_pattern_confidence:
            patterns.append({
                'type': 'frequency_timescale_coupling',
                'confidence': abs(coupling),
                'description': 'Strong coupling between frequency and timescale',
                'coupling_strength': float(coupling)
            })
        
        # Scale invariance
        scale_ratios = []
        for i in range(len(self.scales)-1):
            ratio = np.mean(power[i+1]) / (np.mean(power[i]) + 1e-10)
            scale_ratios.append(ratio)
        
        scale_invariance = 1.0 - np.std(scale_ratios)
        if scale_invariance > self.config.min_pattern_confidence:
            patterns.append({
                'type': 'scale_invariance',
                'confidence': scale_invariance,
                'description': 'Scale-invariant patterns detected',
                'invariance_measure': float(scale_invariance)
            })
        
        return patterns
    
    def _extract_ridges(self, w_transform: np.ndarray) -> List[Dict[str, Any]]:
        """Extract ridge lines from transform"""
        ridges = []
        amplitude = np.abs(w_transform)
        
        for i in range(len(self.scales)):
            peaks, _ = signal.find_peaks(amplitude[i, :])
            for peak in peaks:
                ridge = self._follow_ridge(amplitude, i, peak)
                if len(ridge) >= self.config.min_ridge_length:
                    ridges.append({
                        'scale_indices': [r[0] for r in ridge],
                        'k_indices': [r[1] for r in ridge],
                        'amplitude': [amplitude[r[0], r[1]] for r in ridge],
                        'length': len(ridge)
                    })
        
        return ridges
    
    def _follow_ridge(self, amplitude: np.ndarray, 
                     start_scale: int, 
                     start_k: int) -> List[Tuple[int, int]]:
        """Follow a ridge line through the transform"""
        ridge = [(start_scale, start_k)]
        current_scale = start_scale
        current_k = start_k
        
        while current_scale < len(self.scales) - 1:
            # Search window in next scale
            window_size = 3
            k_min = max(0, current_k - window_size)
            k_max = min(len(self.k_values), current_k + window_size + 1)
            
            # Find maximum in window
            next_k = k_min + np.argmax(
                amplitude[current_scale + 1, k_min:k_max]
            )
            
            if k_min < next_k < k_max - 1:
                ridge.append((current_scale + 1, next_k))
                current_scale += 1
                current_k = next_k
            else:
                break
        
        return ridge
    
    def _interpret_biological(self, w_transform: np.ndarray,
                            power: np.ndarray,
                            species: str) -> Dict[str, Any]:
        """Biological interpretation of transform results"""
        # Find dominant components
        max_idx = np.unravel_index(np.argmax(power), power.shape)
        dominant_k = self.k_values[max_idx[1]]
        dominant_tau = self.scales[max_idx[0]]
        
        # Basic interpretation
        interpretation = {
            'dominant_frequency': dominant_k,
            'dominant_timescale': dominant_tau,
            'process_type': self._classify_biological_process(dominant_k)
        }
        
        # Species-specific analysis
        if species and species in self.config.species_specific_timing:
            expected_timing = self.config.species_specific_timing[species]
            detected_timing = 1.0 / (dominant_k + 1e-10)  # Convert frequency to period
            timing_match = abs(detected_timing - expected_timing) / expected_timing
            
            interpretation['species_analysis'] = {
                'expected_timing': expected_timing,
                'detected_timing': detected_timing,
                'timing_match_quality': 1.0 - min(1.0, timing_match)
            }
        
        # Energy distribution
        total_energy = np.sum(power)
        energy_distribution = power / total_energy
        entropy = -np.sum(energy_distribution * np.log2(energy_distribution + 1e-10))
        
        interpretation['complexity_metrics'] = {
            'entropy': entropy,
            'energy_concentration': 1.0 / np.sum(energy_distribution**2)
        }
        
        return interpretation
    
    def _classify_biological_process(self, frequency: float) -> str:
        """Classify biological process based on frequency"""
        if frequency < 0.1:
            return "Slow metabolic process"
        elif frequency < 0.5:
            return "Information processing"
        elif frequency < 1.0:
            return "Active signaling"
        else:
            return "Rapid response"
    
    def _compute_validation_metrics(self, 
                                  results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute validation metrics for results"""
        # Pattern confidence
        pattern_confidences = [p['confidence'] for p in results['patterns']]
        mean_confidence = np.mean(pattern_confidences) if pattern_confidences else 0.0
        
        # Ridge quality
        ridge_lengths = [r['length'] for r in results['ridges']]
        mean_ridge_length = np.mean(ridge_lengths) if ridge_lengths else 0.0
        
        # Statistical strength
        significant_ratio = np.mean(results['significance']['significant_coefficients'])
        
        return {
            'validation_metrics': {
                'pattern_confidence': mean_confidence,
                'ridge_quality': mean_ridge_length / self.config.num_scales,
                'statistical_strength': significant_ratio,
                'overall_quality': np.mean([
                    mean_confidence,
                    mean_ridge_length / self.config.num_scales,
                    significant_ratio
                ])
            }
        } 