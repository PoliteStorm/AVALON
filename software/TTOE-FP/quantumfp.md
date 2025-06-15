import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import integrate
from scipy.optimize import minimize_scalar
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

@dataclass
class QHOParameters:
    """Parameters for the Quantum Harmonic Oscillator."""
    omega: float = 2.0      # Angular frequency
    amplitude: float = 1.0   # Amplitude
    phase: float = 0.0       # Phase
    
@dataclass
class TransformParameters:
    """Parameters for the wavelet transform computation."""
    k_min: float = -10.0     # Minimum k value
    k_max: float = 10.0      # Maximum k value
    tau_min: float = 0.1     # Minimum tau value
    tau_max: float = 5.0     # Maximum tau value
    k_points: int = 100      # Number of k points
    tau_points: int = 100    # Number of tau points
    t_max: float = 20.0      # Maximum integration time
    t_points: int = 1000     # Number of integration points

class QuantumFingerprintSimulator:
    """
    Complete simulation of the quantum fingerprint experiment using
    wavelet-integral transforms on quantum harmonic oscillator dynamics.
    """
    
    def __init__(self, qho_params: QHOParameters, transform_params: TransformParameters):
        self.qho = qho_params
        self.transform = transform_params
        self.k_values = np.linspace(transform_params.k_min, transform_params.k_max, transform_params.k_points)
        self.tau_values = np.linspace(transform_params.tau_min, transform_params.tau_max, transform_params.tau_points)
        self.t_values = np.linspace(0.01, transform_params.t_max, transform_params.t_points)
        
        # Results storage
        self.W_transform = None
        self.fingerprint_metrics = None
        
    def qho_signal(self, t: np.ndarray) -> np.ndarray:
        """
        Quantum Harmonic Oscillator expectation value: <x(t)> = A*cos(ω*t + φ)
        """
        return self.qho.amplitude * np.cos(self.qho.omega * t + self.qho.phase)
    
    def gaussian_wavelet(self, x: np.ndarray) -> np.ndarray:
        """
        Gaussian wavelet function: ψ(x) = exp(-x²/2) / √(2π)
        """
        return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    
    def morlet_wavelet(self, x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """
        Morlet wavelet function: ψ(x) = exp(-x²/2σ²) * cos(5x) / √(σ√π)
        """
        return np.exp(-x**2 / (2 * sigma**2)) * np.cos(5 * x) / np.sqrt(sigma * np.sqrt(np.pi))
    
    def compute_wavelet_transform_numerical(self, wavelet_type: str = 'gaussian') -> np.ndarray:
        """
        Numerically compute the wavelet transform W(k,τ) using integration.
        
        W(k,τ) = ∫ V(t) * ψ(√(t/τ)) * exp(-i*k*√t) dt
        """
        print("Computing wavelet transform numerically...")
        
        # Choose wavelet function
        if wavelet_type == 'gaussian':
            wavelet_func = self.gaussian_wavelet
        elif wavelet_type == 'morlet':
            wavelet_func = self.morlet_wavelet
        else:
            raise ValueError("Wavelet type must be 'gaussian' or 'morlet'")
        
        # Initialize result array
        W_real = np.zeros((len(self.tau_values), len(self.k_values)))
        W_imag = np.zeros((len(self.tau_values), len(self.k_values)))
        
        # Compute for each (k, tau) pair
        for i, tau in enumerate(self.tau_values):
            for j, k in enumerate(self.k_values):
                
                # Define integrand for this (k, tau) pair
                def integrand_real(t):
                    if t <= 0:
                        return 0
                    signal = self.qho_signal(t)
                    wavelet = wavelet_func(np.sqrt(t / tau))
                    momentum_real = np.cos(k * np.sqrt(t))
                    return signal * wavelet * momentum_real
                
                def integrand_imag(t):
                    if t <= 0:
                        return 0
                    signal = self.qho_signal(t)
                    wavelet = wavelet_func(np.sqrt(t / tau))
                    momentum_imag = -np.sin(k * np.sqrt(t))
                    return signal * wavelet * momentum_imag
                
                # Numerical integration
                try:
                    real_part, _ = integrate.quad(integrand_real, 0.01, self.transform.t_max, 
                                                limit=100, epsabs=1e-8)
                    imag_part, _ = integrate.quad(integrand_imag, 0.01, self.transform.t_max, 
                                                limit=100, epsabs=1e-8)
                    
                    W_real[i, j] = real_part
                    W_imag[i, j] = imag_part
                    
                except:
                    W_real[i, j] = 0
                    W_imag[i, j] = 0
            
            # Progress indicator
            if i % 10 == 0:
                print(f"Progress: {i+1}/{len(self.tau_values)} tau values completed")
        
        # Store complex transform
        self.W_transform = W_real + 1j * W_imag
        return self.W_transform
    
    def compute_wavelet_transform_analytical(self) -> np.ndarray:
        """
        Analytical approximation using the localization property of the wavelet.
        For computational efficiency and comparison with numerical results.
        """
        print("Computing wavelet transform analytically (approximation)...")
        
        W_result = np.zeros((len(self.tau_values), len(self.k_values)), dtype=complex)
        
        for i, tau in enumerate(self.tau_values):
            for j, k in enumerate(self.k_values):
                # Use localization: main contribution at t ≈ τ
                t_peak = tau
                
                # Signal value at peak
                signal = self.qho_signal(t_peak)
                
                # Wavelet value at peak (x = √(t/τ) = 1)
                wavelet = self.gaussian_wavelet(1.0)
                
                # Momentum term
                momentum = np.exp(-1j * k * np.sqrt(t_peak))
                
                # Normalization factor (approximate integral of scaled wavelet)
                normalization = np.sqrt(tau) * np.sqrt(2 * np.pi)
                
                W_result[i, j] = signal * wavelet * momentum * normalization
        
        self.W_transform = W_result
        return W_result
    
    def compute_fingerprint_metrics(self) -> Dict[str, float]:
        """
        Extract comprehensive fingerprint metrics from the wavelet transform.
        """
        if self.W_transform is None:
            raise ValueError("Must compute wavelet transform first")
        
        # Magnitude of the transform
        W_magnitude = np.abs(self.W_transform)
        W_squared = W_magnitude ** 2
        
        # Find peak
        peak_idx = np.unravel_index(np.argmax(W_magnitude), W_magnitude.shape)
        peak_magnitude = W_magnitude[peak_idx]
        dominant_tau = self.tau_values[peak_idx[0]]
        dominant_k = self.k_values[peak_idx[1]]
        
        # Total energy
        total_energy = np.sum(W_squared)
        
        # Create meshgrids for centroid calculations
        K, TAU = np.meshgrid(self.k_values, self.tau_values)
        
        # Centroids (weighted averages)
        k_centroid = np.sum(K * W_squared) / total_energy
        tau_centroid = np.sum(TAU * W_squared) / total_energy
        
        # Spreads (weighted standard deviations)
        k_spread = np.sqrt(np.sum((K - k_centroid)**2 * W_squared) / total_energy)
        tau_spread = np.sqrt(np.sum((TAU - tau_centroid)**2 * W_squared) / total_energy)
        
        # Additional metrics
        # Skewness and kurtosis for shape analysis
        k_skewness = np.sum(((K - k_centroid) / k_spread)**3 * W_squared) / total_energy
        tau_skewness = np.sum(((TAU - tau_centroid) / tau_spread)**3 * W_squared) / total_energy
        
        # Information entropy
        prob_density = W_squared / total_energy
        prob_density = prob_density[prob_density > 1e-12]  # Avoid log(0)
        information_entropy = -np.sum(prob_density * np.log(prob_density))
        
        # Peak-to-noise ratio
        noise_level = np.std(W_magnitude[W_magnitude < 0.1 * peak_magnitude])
        peak_to_noise = peak_magnitude / (noise_level + 1e-12)
        
        self.fingerprint_metrics = {
            'peak_magnitude': peak_magnitude,
            'dominant_k': dominant_k,
            'dominant_tau': dominant_tau,
            'k_centroid': k_centroid,
            'tau_centroid': tau_centroid,
            'k_spread': k_spread,
            'tau_spread': tau_spread,
            'k_skewness': k_skewness,
            'tau_skewness': tau_skewness,
            'total_energy': total_energy,
            'information_entropy': information_entropy,
            'peak_to_noise_ratio': peak_to_noise,
            'coherence_factor': peak_magnitude / np.sqrt(total_energy),
            'localization_parameter': 1.0 / (k_spread * tau_spread)
        }
        
        return self.fingerprint_metrics
    
    def analyze_quantum_features(self) -> Dict[str, float]:
        """
        Extract physics-based interpretations from the fingerprint.
        """
        if self.fingerprint_metrics is None:
            raise ValueError("Must compute fingerprint metrics first")
        
        metrics = self.fingerprint_metrics
        
        # Theoretical predictions for QHO
        theoretical_period = 2 * np.pi / self.qho.omega
        theoretical_tau = theoretical_period / (2 * np.pi)  # Characteristic timescale
        
        # Compare with measured values
        tau_accuracy = 1.0 - abs(metrics['dominant_tau'] - theoretical_tau) / theoretical_tau
        
        # Quantum coherence indicators
        coherence_strength = metrics['coherence_factor']
        localization_quality = metrics['localization_parameter']
        
        # Classical vs quantum behavior
        classical_score = coherence_strength * localization_quality
        
        analysis = {
            'theoretical_tau': theoretical_tau,
            'tau_prediction_accuracy': tau_accuracy,
            'coherence_strength': coherence_strength,
            'localization_quality': localization_quality,
            'classical_behavior_score': classical_score,
            'quantum_signature_strength': metrics['information_entropy'],
            'phase_sensitivity': abs(metrics['k_skewness']),
            'temporal_stability': 1.0 / (1.0 + metrics['tau_spread'])
        }
        
        return analysis
    
    def create_comprehensive_plots(self, figsize: Tuple[int, int] = (20, 16)):
        """
        Create comprehensive visualization of the quantum fingerprint analysis.
        """
        if self.W_transform is None:
            raise ValueError("Must compute wavelet transform first")
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Original QHO Signal
        ax1 = fig.add_subplot(gs[0, :2])
        t_plot = np.linspace(0, 4*np.pi/self.qho.omega, 1000)
        signal_plot = self.qho_signal(t_plot)
        ax1.plot(t_plot, signal_plot, 'b-', linewidth=2, label=r'$\langle x(t) \rangle = A\cos(\omega t + \phi)$')
        ax1.axhline(y=self.qho.amplitude, color='r', linestyle='--', alpha=0.5, label=r'$\pm A$')
        ax1.axhline(y=-self.qho.amplitude, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Time t')
        ax1.set_ylabel(r'$\langle x(t) \rangle$')
        ax1.set_title(f'Quantum Harmonic Oscillator Signal\n'
                      f'ω={self.qho.omega:.2f}, A={self.qho.amplitude:.2f}, φ={self.qho.phase:.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Wavelet Transform Magnitude (Heatmap)
        ax2 = fig.add_subplot(gs[0, 2:])
        W_magnitude = np.abs(self.W_transform)
        im = ax2.imshow(W_magnitude, aspect='auto', origin='lower', 
                       extent=[self.k_values[0], self.k_values[-1], 
                              self.tau_values[0], self.tau_values[-1]],
                       cmap='plasma')
        ax2.set_xlabel('Momentum-like parameter k')
        ax2.set_ylabel('Timescale τ')
        ax2.set_title('Wavelet Transform |W(k,τ)|')
        plt.colorbar(im, ax=ax2, label='Magnitude')
        
        # Mark peak
        peak_idx = np.unravel_index(np.argmax(W_magnitude), W_magnitude.shape)
        peak_tau = self.tau_values[peak_idx[0]]
        peak_k = self.k_values[peak_idx[1]]
        ax2.plot(peak_k, peak_tau, 'r*', markersize=15, label='Peak')
        ax2.legend()
        
        # 3. Cross-sections through peak
        ax3 = fig.add_subplot(gs[1, :2])
        k_cross = W_magnitude[peak_idx[0], :]
        ax3.plot(self.k_values, k_cross, 'g-', linewidth=2)
        ax3.axvline(x=peak_k, color='r', linestyle='--', alpha=0.7)
        ax3.set_xlabel('k')
        ax3.set_ylabel('|W(k,τ)|')
        ax3.set_title(f'k-section at τ = {peak_tau:.3f}')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 2:])
        tau_cross = W_magnitude[:, peak_idx[1]]
        ax4.plot(self.tau_values, tau_cross, 'm-', linewidth=2)
        ax4.axvline(x=peak_tau, color='r', linestyle='--', alpha=0.7)
        ax4.set_xlabel('τ')
        ax4.set_ylabel('|W(k,τ)|')
        ax4.set_title(f'τ-section at k = {peak_k:.3f}')
        ax4.grid(True, alpha=0.3)
        
        # 4. Fingerprint Metrics Bar Chart
        ax5 = fig.add_subplot(gs[2, :2])
        if self.fingerprint_metrics:
            metrics_names = ['Peak Mag', 'Dom k', 'Dom τ', 'k Centroid', 'τ Centroid']
            metrics_values = [
                self.fingerprint_metrics['peak_magnitude'],
                self.fingerprint_metrics['dominant_k'],
                self.fingerprint_metrics['dominant_tau'],
                self.fingerprint_metrics['k_centroid'],
                self.fingerprint_metrics['tau_centroid']
            ]
            bars = ax5.bar(metrics_names, metrics_values, color=['red', 'blue', 'green', 'orange', 'purple'])
            ax5.set_title('Key Fingerprint Metrics')
            ax5.set_ylabel('Value')
            plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Phase Information
        ax6 = fig.add_subplot(gs[2, 2:])
        W_phase = np.angle(self.W_transform)
        im_phase = ax6.imshow(W_phase, aspect='auto', origin='lower',
                             extent=[self.k_values[0], self.k_values[-1],
                                    self.tau_values[0], self.tau_values[-1]],
                             cmap='hsv')
        ax6.set_xlabel('k')
        ax6.set_ylabel('τ')
        ax6.set_title('Phase of W(k,τ)')
        plt.colorbar(im_phase, ax=ax6, label='Phase (radians)')
        
        # 6. Theoretical Comparison
        ax7 = fig.add_subplot(gs[3, :])
        if hasattr(self, 'fingerprint_metrics') and self.fingerprint_metrics:
            analysis = self.analyze_quantum_features()
            
            comparison_data = {
                'Metric': ['Dominant τ', 'Theoretical τ', 'Coherence', 'Localization', 'Classical Score'],
                'Value': [
                    self.fingerprint_metrics['dominant_tau'],
                    analysis['theoretical_tau'],
                    analysis['coherence_strength'],
                    analysis['localization_quality'] / 10,  # Scale for visibility
                    analysis['classical_behavior_score']
                ]
            }
            
            bars = ax7.bar(comparison_data['Metric'], comparison_data['Value'], 
                          color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
            ax7.set_title('Theoretical vs Measured Quantum Features')
            ax7.set_ylabel('Value')
            plt.setp(ax7.get_xticklabels(), rotation=45, ha='right')
            
            # Add accuracy annotation
            accuracy = analysis['tau_prediction_accuracy'] * 100
            ax7.text(0.02, 0.98, f'τ Prediction Accuracy: {accuracy:.1f}%', 
                    transform=ax7.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('Quantum Fingerprint Analysis: Complete Results', fontsize=16, y=0.98)
        plt.tight_layout()
        return fig
    
    def export_results(self, filename: str = 'quantum_fingerprint_results'):
        """
        Export all results to files for further analysis.
        """
        # Save fingerprint metrics
        if self.fingerprint_metrics:
            metrics_df = pd.DataFrame([self.fingerprint_metrics])
            metrics_df.to_csv(f'{filename}_metrics.csv', index=False)
            
        # Save transform data
        if self.W_transform is not None:
            np.save(f'{filename}_transform_real.npy', self.W_transform.real)
            np.save(f'{filename}_transform_imag.npy', self.W_transform.imag)
            
        # Save parameters
        params_dict = {
            'qho_omega': self.qho.omega,
            'qho_amplitude': self.qho.amplitude,
            'qho_phase': self.qho.phase,
            'k_min': self.transform.k_min,
            'k_max': self.transform.k_max,
            'tau_min': self.transform.tau_min,
            'tau_max': self.transform.tau_max,
            'k_points': self.transform.k_points,
            'tau_points': self.transform.tau_points
        }
        params_df = pd.DataFrame([params_dict])
        params_df.to_csv(f'{filename}_parameters.csv', index=False)
        
        print(f"Results exported to {filename}_*.csv and {filename}_*.npy files")

def run_comprehensive_experiment():
    """
    Run a comprehensive quantum fingerprint experiment with multiple scenarios.
    """
    print("=" * 80)
    print("QUANTUM FINGERPRINT EXPERIMENT - COMPREHENSIVE SIMULATION")
    print("=" * 80)
    
    # Define multiple test scenarios
    scenarios = [
        ("Standard QHO", QHOParameters(omega=2.0, amplitude=1.0, phase=0.0)),
        ("High Frequency", QHOParameters(omega=5.0, amplitude=1.0, phase=0.0)),
        ("Low Frequency", QHOParameters(omega=0.5, amplitude=1.0, phase=0.0)),
        ("Phase Shifted", QHOParameters(omega=2.0, amplitude=1.0, phase=np.pi/4)),
        ("High Amplitude", QHOParameters(omega=2.0, amplitude=2.0, phase=0.0)),
    ]
    
    # Transform parameters
    transform_params = TransformParameters(
        k_min=-8.0, k_max=8.0, tau_min=0.1, tau_max=4.0,
        k_points=80, tau_points=80, t_max=15.0
    )
    
    all_results = []
    
    for scenario_name, qho_params in scenarios:
        print(f"\n--- Analyzing {scenario_name} ---")
        
        # Create simulator
        simulator = QuantumFingerprintSimulator(qho_params, transform_params)
        
        # Compute transform (using analytical approximation for speed)
        simulator.compute_wavelet_transform_analytical()
        
        # Compute fingerprint
        fingerprint = simulator.compute_fingerprint_metrics()
        
        # Analyze quantum features
        analysis = simulator.analyze_quantum_features()
        
        # Store results
        result = {
            'scenario': scenario_name,
            'qho_params': qho_params,
            'fingerprint': fingerprint,
            'analysis': analysis
        }
        all_results.append(result)
        
        # Print key results
        print(f"Peak Magnitude: {fingerprint['peak_magnitude']:.4f}")
        print(f"Dominant τ: {fingerprint['dominant_tau']:.3f} (Theory: {analysis['theoretical_tau']:.3f})")
        print(f"Dominant k: {fingerprint['dominant_k']:.3f}")
        print(f"Coherence: {analysis['coherence_strength']:.4f}")
        print(f"τ Accuracy: {analysis['tau_prediction_accuracy']*100:.1f}%")
        
        # Create plots for first scenario
        if scenario_name == "Standard QHO":
            print("Creating comprehensive visualization...")
            fig = simulator.create_comprehensive_plots()
            plt.show()
            
            # Export results
            simulator.export_results(f'quantum_fingerprint_{scenario_name.lower().replace(" ", "_")}')
    
    # Comparative analysis
    print(f"\n{'='*50}")
    print("COMPARATIVE ANALYSIS ACROSS SCENARIOS")
    print(f"{'='*50}")
    
    comparison_df = pd.DataFrame([
        {
            'Scenario': result['scenario'],
            'ω': result['qho_params'].omega,
            'A': result['qho_params'].amplitude,
            'φ': result['qho_params'].phase,
            'Peak_Mag': result['fingerprint']['peak_magnitude'],
            'Dom_τ': result['fingerprint']['dominant_tau'],
            'Dom_k': result['fingerprint']['dominant_k'],
            'Coherence': result['analysis']['coherence_strength'],
            'τ_Accuracy': result['analysis']['tau_prediction_accuracy'],
            'Classical_Score': result['analysis']['classical_behavior_score']
        }
        for result in all_results
    ])
    
    print(comparison_df.round(4))
    
    # Save comparison
    comparison_df.to_csv('quantum_fingerprint_comparison.csv', index=False)
    
    return all_results, comparison_df

def demonstrate_numerical_vs_analytical():
    """
    Compare numerical integration vs analytical approximation.
    """
    print("\n" + "="*60)
    print("NUMERICAL vs ANALYTICAL COMPARISON")
    print("="*60)
    
    # Use simpler parameters for numerical computation
    qho_params = QHOParameters(omega=2.0, amplitude=1.0, phase=0.0)
    transform_params = TransformParameters(
        k_min=-5.0, k_max=5.0, tau_min=0.5, tau_max=3.0,
        k_points=30, tau_points=30, t_max=10.0
    )
    
    simulator = QuantumFingerprintSimulator(qho_params, transform_params)
    
    # Analytical method
    print("Computing analytical approximation...")
    W_analytical = simulator.compute_wavelet_transform_analytical()
    fingerprint_analytical = simulator.compute_fingerprint_metrics()
    
    # Numerical method (warning: this is slow!)
    print("Computing numerical integration (this may take several minutes)...")
    W_numerical = simulator.compute_wavelet_transform_numerical()
    fingerprint_numerical = simulator.compute_fingerprint_metrics()
    
    # Compare results
    print("\nComparison of Key Metrics:")
    print(f"{'Metric':<20} {'Analytical':<12} {'Numerical':<12} {'Diff %':<10}")
    print("-" * 60)
    
    for key in ['peak_magnitude', 'dominant_tau', 'dominant_k']:
        anal_val = fingerprint_analytical[key]
        num_val = fingerprint_numerical[key]
        diff_pct = abs(anal_val - num_val) / abs(num_val) * 100 if num_val != 0 else 0
        print(f"{key:<20} {anal_val:<12.4f} {num_val:<12.4f} {diff_pct:<10.2f}")
    
    return W_analytical, W_numerical

if __name__ == "__main__":
    # Run the comprehensive experiment
    results, comparison = run_comprehensive_experiment()
    
    # Uncomment the following line to run numerical vs analytical comparison
    # (Warning: this is computationally intensive)
    # numerical_comparison = demonstrate_numerical_vs_analytical()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("Check the generated CSV files and plots for detailed results.")
    print("="*80)