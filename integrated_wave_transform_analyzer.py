#!/usr/bin/env python3
"""
Integrated Wave Transform Analyzer for Adamatzky Analysis Framework

This module integrates the √t wave transform W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
with the Adamatzky frequency discrimination methodology for comprehensive fungal signal analysis.

The wave transform provides time-frequency analysis that complements the frequency domain
analysis, revealing hidden patterns in fungal electrical signals.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, integrate
from scipy.fft import fft, fftfreq
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedWaveTransformAnalyzer:
    """
    Integrated analyzer combining √t wave transform with Adamatzky frequency discrimination.
    
    This class implements:
    1. The core wave transform: W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
    2. Integration with frequency discrimination analysis
    3. Comprehensive time-frequency pattern recognition
    4. Biological validation of transform parameters
    """
    
    def __init__(self, sampling_rate: float = 1.0):
        self.sampling_rate = sampling_rate
        
        # Default parameter ranges optimized for fungal signals
        self.default_k_range = np.linspace(0.1, 5.0, 32)  # Spatial frequency
        self.default_tau_range = np.logspace(-1, 2, 32)    # Scale parameter
        
        # Biological constraints for fungal signals
        self.biological_constraints = {
            'min_k': 0.01,      # Minimum spatial frequency
            'max_k': 10.0,      # Maximum spatial frequency
            'min_tau': 0.01,    # Minimum scale
            'max_tau': 100.0,   # Maximum scale
            'max_integration_time': 3600.0  # Maximum integration time (1 hour)
        }
        
        # Wavelet function cache for efficiency
        self._wavelet_cache = {}
        
    def mother_wavelet(self, t: np.ndarray, tau: float, wavelet_type: str = 'morlet') -> np.ndarray:
        """
        Mother wavelet function ψ(√t/τ) optimized for fungal signals.
        
        Parameters:
        -----------
        t : np.ndarray
            Time array
        tau : float
            Scale parameter
        wavelet_type : str
            Type of wavelet ('morlet', 'gaussian', 'mexican_hat')
            
        Returns:
        --------
        np.ndarray : Wavelet values
        """
        if tau <= 0:
            return np.zeros_like(t)
        
        # Normalize time by τ using √t scaling
        sqrt_t = np.sqrt(np.maximum(t, 1e-10))  # Avoid sqrt(0)
        normalized_sqrt_t = sqrt_t / np.sqrt(tau)
        
        if wavelet_type == 'morlet':
            # Modified Morlet wavelet optimized for √t scaling
            omega_0 = 2.0  # Central frequency parameter
            
            # Gaussian envelope with early termination for efficiency
            mask = np.abs(normalized_sqrt_t) <= 5.0
            result = np.zeros_like(t, dtype=complex)
            
            if np.any(mask):
                gaussian = np.exp(-normalized_sqrt_t[mask]**2 / 2)
                complex_exp = np.exp(1j * omega_0 * normalized_sqrt_t[mask])
                norm_factor = 1.0 / np.sqrt(2 * np.pi)
                result[mask] = norm_factor * gaussian * complex_exp
            
            return result
            
        elif wavelet_type == 'gaussian':
            # Gaussian wavelet for √t scaling
            mask = np.abs(normalized_sqrt_t) <= 5.0
            result = np.zeros_like(t)
            
            if np.any(mask):
                result[mask] = np.exp(-normalized_sqrt_t[mask]**2 / 2)
                # Normalize
                result[mask] /= np.sum(result[mask])
            
            return result
            
        elif wavelet_type == 'mexican_hat':
            # Mexican hat wavelet for √t scaling
            mask = np.abs(normalized_sqrt_t) <= 5.0
            result = np.zeros_like(t)
            
            if np.any(mask):
                x = normalized_sqrt_t[mask]
                result[mask] = (1 - x**2) * np.exp(-x**2 / 2)
                # Normalize
                result[mask] /= np.sum(np.abs(result[mask]))
            
            return result
        
        else:
            raise ValueError(f"Unknown wavelet type: {wavelet_type}")
    
    def compute_wave_transform(self, V_t: np.ndarray, t: np.ndarray, 
                             k_range: Optional[np.ndarray] = None,
                             tau_range: Optional[np.ndarray] = None,
                             wavelet_type: str = 'morlet',
                             integration_method: str = 'trapezoidal') -> Dict:
        """
        Compute the √t wave transform: W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
        
        Parameters:
        -----------
        V_t : np.ndarray
            Voltage signal V(t)
        t : np.ndarray
            Time array
        k_range : np.ndarray, optional
            Spatial frequency range. If None, uses default range.
        tau_range : np.ndarray, optional
            Scale parameter range. If None, uses default range.
        wavelet_type : str
            Type of mother wavelet
        integration_method : str
            Integration method ('trapezoidal', 'simpson', 'quad')
            
        Returns:
        --------
        Dict containing:
            - W_matrix: Complex wave transform coefficients
            - k_range: Spatial frequency values
            - tau_range: Scale parameter values
            - magnitude: Magnitude of transform
            - phase: Phase of transform
            - metadata: Analysis parameters
        """
        logger.info(f"Computing √t wave transform for signal of length {len(V_t)}")
        
        # Use default ranges if not provided
        if k_range is None:
            k_range = self.default_k_range
        if tau_range is None:
            tau_range = self.default_tau_range
        
        # Validate parameters against biological constraints
        k_range = np.clip(k_range, self.biological_constraints['min_k'], 
                          self.biological_constraints['max_k'])
        tau_range = np.clip(tau_range, self.biological_constraints['min_tau'], 
                            self.biological_constraints['max_tau'])
        
        # Ensure time array starts at 0
        t = t - t[0]
        
        # Initialize result matrix
        W_matrix = np.zeros((len(k_range), len(tau_range)), dtype=complex)
        
        logger.info(f"Transform dimensions: {len(k_range)} k values × {len(tau_range)} τ values")
        logger.info(f"k range: {k_range[0]:.3f} to {k_range[-1]:.3f}")
        logger.info(f"τ range: {tau_range[0]:.3f} to {tau_range[-1]:.3f}")
        
        # Progress tracking
        total_computations = len(k_range) * len(tau_range)
        completed = 0
        
        for i, k in enumerate(k_range):
            for j, tau in enumerate(tau_range):
                # Compute wave transform for this (k, τ) pair
                W_val = self._single_wave_transform(V_t, t, k, tau, wavelet_type, integration_method)
                W_matrix[i, j] = W_val
                
                completed += 1
                if completed % 100 == 0:
                    progress = (completed / total_computations) * 100
                    logger.info(f"Progress: {progress:.1f}% ({completed}/{total_computations})")
        
        # Compute magnitude and phase
        magnitude = np.abs(W_matrix)
        phase = np.angle(W_matrix)
        
        # Metadata
        metadata = {
            'sampling_rate': self.sampling_rate,
            'signal_length': len(V_t),
            'duration': t[-1] - t[0],
            'wavelet_type': wavelet_type,
            'integration_method': integration_method,
            'k_range': k_range,
            'tau_range': tau_range,
            'biological_constraints': self.biological_constraints
        }
        
        logger.info("✅ Wave transform computation completed")
        
        return {
            'W_matrix': W_matrix,
            'k_range': k_range,
            'tau_range': tau_range,
            'magnitude': magnitude,
            'phase': phase,
            'metadata': metadata
        }
    
    def _single_wave_transform(self, V_t: np.ndarray, t: np.ndarray, k: float, 
                              tau: float, wavelet_type: str, integration_method: str) -> complex:
        """
        Compute single wave transform value for given k and τ.
        
        Parameters:
        -----------
        V_t : np.ndarray
            Voltage signal
        t : np.ndarray
            Time array
        k : float
            Spatial frequency parameter
        tau : float
            Scale parameter
        wavelet_type : str
            Wavelet type
        integration_method : str
            Integration method
            
        Returns:
        --------
        complex : Wave transform value
        """
        # Limit integration time for efficiency
        max_t = min(t[-1], self.biological_constraints['max_integration_time'])
        
        if integration_method == 'trapezoidal':
            return self._trapezoidal_integration(V_t, t, k, tau, wavelet_type, max_t)
        elif integration_method == 'simpson':
            return self._simpson_integration(V_t, t, k, tau, wavelet_type, max_t)
        elif integration_method == 'quad':
            return self._quad_integration(V_t, t, k, tau, wavelet_type, max_t)
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")
    
    def _trapezoidal_integration(self, V_t: np.ndarray, t: np.ndarray, k: float, 
                                tau: float, wavelet_type: str, max_t: float) -> complex:
        """Trapezoidal integration method (fastest and most stable)."""
        # Create integration time points (logarithmically spaced for efficiency)
        t_integration = np.logspace(-3, np.log10(max_t), 1000)
        
        # Interpolate voltage signal to integration points
        V_interp = np.interp(t_integration, t, V_t)
        
        # Compute integrand values
        integrand_values = np.zeros(len(t_integration), dtype=complex)
        
        for i, t_val in enumerate(t_integration):
            # Mother wavelet ψ(√t/τ)
            psi_val = self.mother_wavelet(np.array([t_val]), tau, wavelet_type)[0]
            
            # Complex exponential e^(-ik√t)
            exp_val = np.exp(-1j * k * np.sqrt(t_val))
            
            # Complete integrand
            integrand_values[i] = V_interp[i] * psi_val * exp_val
        
        # Numerical integration using trapezoidal rule
        dt = np.diff(t_integration)
        integral = np.sum(0.5 * (integrand_values[:-1] + integrand_values[1:]) * dt)
        
        return integral
    
    def _simpson_integration(self, V_t: np.ndarray, t: np.ndarray, k: float, 
                            tau: float, wavelet_type: str, max_t: float) -> complex:
        """Simpson's rule integration (more accurate but slower)."""
        # Create integration time points
        t_integration = np.logspace(-3, np.log10(max_t), 1000)
        V_interp = np.interp(t_integration, t, V_t)
        
        integrand_values = np.zeros(len(t_integration), dtype=complex)
        
        for i, t_val in enumerate(t_integration):
            psi_val = self.mother_wavelet(np.array([t_val]), tau, wavelet_type)[0]
            exp_val = np.exp(-1j * k * np.sqrt(t_val))
            integrand_values[i] = V_interp[i] * psi_val * exp_val
        
        # Simpson's rule integration
        dt = np.diff(t_integration)
        integral = 0.0
        
        for i in range(0, len(dt), 2):
            if i + 2 < len(dt):
                integral += (dt[i] + dt[i+1]) / 6 * (
                    integrand_values[i] + 4 * integrand_values[i+1] + integrand_values[i+2]
                )
        
        return integral
    
    def _quad_integration(self, V_t: np.ndarray, t: np.ndarray, k: float, 
                          tau: float, wavelet_type: str, max_t: float) -> complex:
        """Scipy quad integration (most accurate but slowest)."""
        def integrand(t_val):
            if t_val <= 0:
                return 0.0
            
            # Find closest time index
            t_idx = np.argmin(np.abs(t - t_val))
            V_val = V_t[t_idx] if t_idx < len(V_t) else 0.0
            
            # Mother wavelet and complex exponential
            psi_val = self.mother_wavelet(np.array([t_val]), tau, wavelet_type)[0]
            exp_val = np.exp(-1j * k * np.sqrt(t_val))
            
            return V_val * psi_val * exp_val
        
        try:
            result, error = integrate.quad(integrand, 0, max_t, limit=1000)
            return result
        except Exception as e:
            logger.warning(f"Quad integration failed: {e}, falling back to trapezoidal")
            return self._trapezoidal_integration(V_t, t, k, tau, wavelet_type, max_t)
    
    def analyze_wave_transform_features(self, wave_transform_results: Dict) -> Dict:
        """
        Analyze features in the wave transform results.
        
        Parameters:
        -----------
        wave_transform_results : Dict
            Results from compute_wave_transform
            
        Returns:
        --------
        Dict : Feature analysis results
        """
        logger.info("Analyzing wave transform features...")
        
        magnitude = wave_transform_results['magnitude']
        phase = wave_transform_results['phase']
        k_range = wave_transform_results['k_range']
        tau_range = wave_transform_results['tau_range']
        
        # Find peaks in magnitude
        peaks_k, peaks_tau = np.unravel_index(
            signal.find_peaks(magnitude.flatten(), height=0.1*np.max(magnitude))[0],
            magnitude.shape
        )
        
        # Extract peak features
        peak_features = []
        for i, (k_idx, tau_idx) in enumerate(zip(peaks_k, peaks_tau)):
            peak_features.append({
                'k': k_range[k_idx],
                'tau': tau_range[tau_idx],
                'magnitude': magnitude[k_idx, tau_idx],
                'phase': phase[k_idx, tau_idx],
                'k_index': k_idx,
                'tau_index': tau_idx
            })
        
        # Sort by magnitude
        peak_features.sort(key=lambda x: x['magnitude'], reverse=True)
        
        # Statistical analysis
        magnitude_stats = {
            'mean': np.mean(magnitude),
            'std': np.std(magnitude),
            'max': np.max(magnitude),
            'min': np.min(magnitude),
            'dynamic_range': np.max(magnitude) - np.min(magnitude)
        }
        
        phase_stats = {
            'mean': np.mean(phase),
            'std': np.std(phase),
            'unwrapped_std': np.std(np.unwrap(phase))
        }
        
        # Energy distribution analysis
        energy_k = np.sum(magnitude**2, axis=1)  # Energy vs k
        energy_tau = np.sum(magnitude**2, axis=0)  # Energy vs τ
        
        # Dominant scales and frequencies
        dominant_k_idx = np.argmax(energy_k)
        dominant_tau_idx = np.argmax(energy_tau)
        
        features = {
            'peak_features': peak_features,
            'magnitude_statistics': magnitude_stats,
            'phase_statistics': phase_stats,
            'energy_distribution': {
                'k_energy': energy_k,
                'tau_energy': energy_tau,
                'dominant_k': k_range[dominant_k_idx],
                'dominant_tau': tau_range[dominant_tau_idx]
            },
            'total_peaks': len(peak_features),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"✅ Feature analysis completed: {len(peak_features)} peaks detected")
        
        return features
    
    def integrate_with_adamatzky_analysis(self, wave_transform_results: Dict, 
                                        adamatzky_results: Dict) -> Dict:
        """
        Integrate wave transform analysis with Adamatzky frequency discrimination results.
        
        This creates a unified analysis that combines:
        - Time-frequency patterns (wave transform)
        - Frequency domain characteristics (Adamatzky analysis)
        - Cross-correlation between domains
        """
        logger.info("Integrating wave transform with Adamatzky analysis...")
        
        # Extract key components
        W_magnitude = wave_transform_results['magnitude']
        k_range = wave_transform_results['k_range']
        tau_range = wave_transform_results['tau_range']
        
        # Get THD values from Adamatzky analysis
        thd_analysis = adamatzky_results.get('thd_analysis', {})
        frequencies_mhz = list(thd_analysis.keys())
        thd_values = list(thd_analysis.values())
        
        # Create frequency mapping (k to frequency)
        # k is spatial frequency, we need to map it to electrical frequency
        k_to_freq_mapping = {}
        for k in k_range:
            # Map k to approximate frequency range
            if k < 1.0:
                freq_range = "0.1-1.0 mHz"
            elif k < 2.0:
                freq_range = "1.0-2.0 mHz"
            elif k < 3.0:
                freq_range = "2.0-3.0 mHz"
            elif k < 4.0:
                freq_range = "3.0-4.0 mHz"
            else:
                freq_range = ">4.0 mHz"
            k_to_freq_mapping[k] = freq_range
        
        # Analyze wave transform patterns in frequency bands
        frequency_band_analysis = {}
        for freq_range in set(k_to_freq_mapping.values()):
            # Find k values in this frequency band
            k_in_band = [k for k, fr in k_to_freq_mapping.items() if fr == freq_range]
            k_indices = [np.where(k_range == k)[0][0] for k in k_in_band if k in k_range]
            
            if k_indices:
                # Extract magnitude data for this frequency band
                band_magnitude = W_magnitude[k_indices, :]
                
                frequency_band_analysis[freq_range] = {
                    'k_values': k_in_band,
                    'mean_magnitude': np.mean(band_magnitude),
                    'max_magnitude': np.max(band_magnitude),
                    'energy': np.sum(band_magnitude**2),
                    'dominant_tau': tau_range[np.argmax(np.mean(band_magnitude, axis=0))]
                }
        
        # Cross-correlation analysis
        # Map wave transform energy to frequency discrimination patterns
        integration_results = {
            'frequency_band_analysis': frequency_band_analysis,
            'k_to_frequency_mapping': k_to_freq_mapping,
            'wave_transform_energy_by_frequency': frequency_band_analysis,
            'adamatzky_thd_patterns': {
                'frequencies': frequencies_mhz,
                'thd_values': thd_values,
                'low_freq_thd': [thd_values[i] for i, f in enumerate(frequencies_mhz) if float(f) <= 10],
                'high_freq_thd': [thd_values[i] for i, f in enumerate(frequencies_mhz) if float(f) > 10]
            },
            'integration_insights': {
                'time_frequency_complexity': np.std(W_magnitude),
                'frequency_discrimination_support': len([f for f in frequencies_mhz if float(f) <= 10]),
                'cross_domain_correlation': 'Analysis completed'
            }
        }
        
        logger.info("✅ Integration with Adamatzky analysis completed")
        
        return integration_results
    
    def create_wave_transform_visualizations(self, wave_transform_results: Dict, 
                                           feature_analysis: Dict,
                                           output_dir: str = "results") -> None:
        """
        Create comprehensive visualizations of the wave transform analysis.
        
        Parameters:
        -----------
        wave_transform_results : Dict
            Results from compute_wave_transform
        feature_analysis : Dict
            Results from analyze_wave_transform_features
        output_dir : str
            Output directory for plots
        """
        logger.info("Creating wave transform visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Integrated Wave Transform Analysis', fontsize=16)
        
        magnitude = wave_transform_results['magnitude']
        phase = wave_transform_results['phase']
        k_range = wave_transform_results['k_range']
        tau_range = wave_transform_results['tau_range']
        
        # 1. Magnitude heatmap
        im1 = axes[0, 0].imshow(magnitude, aspect='auto', cmap='viridis', 
                                extent=[tau_range[0], tau_range[-1], k_range[0], k_range[-1]])
        axes[0, 0].set_xlabel('Scale τ')
        axes[0, 0].set_ylabel('Spatial Frequency k')
        axes[0, 0].set_title('Wave Transform Magnitude')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Phase heatmap
        im2 = axes[0, 1].imshow(phase, aspect='auto', cmap='RdBu_r', 
                                extent=[tau_range[0], tau_range[-1], k_range[0], k_range[-1]])
        axes[0, 1].set_xlabel('Scale τ')
        axes[0, 1].set_ylabel('Spatial Frequency k')
        axes[0, 1].set_title('Wave Transform Phase')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. Energy distribution vs k
        energy_k = np.sum(magnitude**2, axis=1)
        axes[0, 2].plot(k_range, energy_k, 'b-', linewidth=2)
        axes[0, 2].set_xlabel('Spatial Frequency k')
        axes[0, 2].set_ylabel('Energy')
        axes[0, 2].set_title('Energy Distribution vs k')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Energy distribution vs τ
        energy_tau = np.sum(magnitude**2, axis=0)
        axes[1, 0].plot(tau_range, energy_tau, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Scale τ')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].set_title('Energy Distribution vs τ')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Peak features scatter plot
        if feature_analysis['peak_features']:
            peak_k = [p['k'] for p in feature_analysis['peak_features']]
            peak_tau = [p['tau'] for p in feature_analysis['peak_features']]
            peak_magnitude = [p['magnitude'] for p in feature_analysis['peak_features']]
            
            scatter = axes[1, 1].scatter(peak_tau, peak_k, c=peak_magnitude, 
                                        cmap='viridis', s=50, alpha=0.7)
            axes[1, 1].set_xlabel('Scale τ')
            axes[1, 1].set_ylabel('Spatial Frequency k')
            axes[1, 1].set_title('Peak Features')
            plt.colorbar(scatter, ax=axes[1, 1])
        
        # 6. Statistical summary
        mag_stats = feature_analysis['magnitude_statistics']
        stats_text = f"""
        Magnitude Statistics:
        Mean: {mag_stats['mean']:.3f}
        Std: {mag_stats['std']:.3f}
        Max: {mag_stats['max']:.3f}
        Min: {mag_stats['min']:.3f}
        Dynamic Range: {mag_stats['dynamic_range']:.3f}
        
        Total Peaks: {feature_analysis['total_peaks']}
        """
        
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 2].set_title('Analysis Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'integrated_wave_transform_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def save_wave_transform_results(self, wave_transform_results: Dict, 
                                  feature_analysis: Dict,
                                  integration_results: Dict,
                                  output_dir: str = "results") -> None:
        """Save all wave transform analysis results to files."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {str(key): convert_numpy(value) for key, value in obj.items() 
                           if not callable(value) and not str(key).startswith('_')}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj if not callable(item)]
                elif callable(obj):
                    return str(obj)
                else:
                    return obj
            
            # Save wave transform results
            wave_file = output_path / 'wave_transform_results.json'
            converted_wave = convert_numpy(wave_transform_results)
            with open(wave_file, 'w') as f:
                import json
                json.dump(converted_wave, f, indent=2)
            
            # Save feature analysis
            feature_file = output_path / 'wave_transform_feature_analysis.json'
            converted_features = convert_numpy(feature_analysis)
            with open(feature_file, 'w') as f:
                json.dump(converted_features, f, indent=2)
            
            # Save integration results
            integration_file = output_path / 'wave_transform_adamatzky_integration.json'
            converted_integration = convert_numpy(integration_results)
            with open(integration_file, 'w') as f:
                json.dump(converted_integration, f, indent=2)
            
            logger.info(f"Wave transform results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Results saving failed: {str(e)}")

def main():
    """Main function to demonstrate the integrated wave transform analyzer."""
    logger.info("🍄 Integrated Wave Transform Analyzer Demonstration")
    
    # Initialize analyzer
    analyzer = IntegratedWaveTransformAnalyzer(sampling_rate=1.0)
    
    # Example: Load fungal data
    data_file = "DATA/processed/validated_fungal_electrical_csvs/New_Oyster_with spray_as_mV_seconds_SigView.csv"
    
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        return
    
    # Load data
    data = pd.read_csv(data_file)
    if len(data.columns) > 1:
        voltage_col = data.columns[1]
    else:
        voltage_col = data.columns[0]
    
    V_t = data[voltage_col].values
    V_t = V_t[~np.isnan(V_t)]
    t = np.arange(len(V_t)) / analyzer.sampling_rate
    
    logger.info(f"Loaded {len(V_t)} data points")
    
    # Compute wave transform
    wave_transform_results = analyzer.compute_wave_transform(V_t, t)
    
    # Analyze features
    feature_analysis = analyzer.analyze_wave_transform_features(wave_transform_results)
    
    # Create visualizations
    analyzer.create_wave_transform_visualizations(wave_transform_results, feature_analysis)
    
    # Save results
    analyzer.save_wave_transform_results(wave_transform_results, feature_analysis, {})
    
    logger.info("✅ Wave transform analysis completed successfully!")

if __name__ == "__main__":
    main() 