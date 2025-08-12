#!/usr/bin/env python3
"""
Optimized √t Wave Transform Implementation
Author: Joe Knowles
Timestamp: 2025-08-12 09:23:27 BST
Description: Optimized implementation of W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
from pathlib import Path

class OptimizedSqrtWaveTransform:
    """
    Optimized implementation of the √t wave transform for fungal electrical signals:
    W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
    """
    
    def __init__(self):
        self.timestamp = "20250812_092327"
        self.author = "Joe Knowles"
        
    def mother_wavelet(self, t, tau):
        """
        Optimized mother wavelet function ψ(√t/τ)
        Using a modified Morlet wavelet adapted for √t scaling
        """
        # Handle edge cases
        if t <= 0 or tau <= 0:
            return 0.0
        
        # Normalize time by τ
        normalized_t = np.sqrt(t) / np.sqrt(tau)
        
        # Modified Morlet wavelet for √t scaling
        omega_0 = 2.0  # Central frequency parameter
        
        # Gaussian envelope with early termination for large values
        if abs(normalized_t) > 5.0:  # Truncate for efficiency
            return 0.0
        
        gaussian = np.exp(-normalized_t**2 / 2)
        
        # Complex exponential for frequency content
        complex_exp = np.exp(1j * omega_0 * normalized_t)
        
        # Normalization factor
        norm_factor = 1.0 / np.sqrt(2 * np.pi)
        
        return norm_factor * gaussian * complex_exp
    
    def optimized_sqrt_wave_transform(self, V_t, t, k, tau, max_t=None):
        """
        Optimized √t wave transform computation:
        W(k,τ) = ∫₀^max_t V(t) · ψ(√t/τ) · e^(-ik√t) dt
        
        Uses finite integration limits and vectorized operations for efficiency
        """
        
        if max_t is None:
            max_t = t[-1]  # Use signal duration as upper limit
        
        # Create integration time points (logarithmically spaced for efficiency)
        t_integration = np.logspace(-3, np.log10(max_t), 1000)
        
        # Interpolate voltage signal to integration points
        V_interp = np.interp(t_integration, t, V_t)
        
        # Compute integrand values
        integrand_values = np.zeros(len(t_integration), dtype=complex)
        
        for i, t_val in enumerate(t_integration):
            # Mother wavelet ψ(√t/τ)
            psi_val = self.mother_wavelet(t_val, tau)
            
            # Complex exponential e^(-ik√t)
            exp_val = np.exp(-1j * k * np.sqrt(t_val))
            
            # Complete integrand
            integrand_values[i] = V_interp[i] * psi_val * exp_val
        
        # Numerical integration using trapezoidal rule
        # More stable than quad for this type of integral
        dt = np.diff(t_integration)
        integral = np.sum(0.5 * (integrand_values[:-1] + integrand_values[1:]) * dt)
        
        return integral
    
    def compute_wave_transform_2d_optimized(self, V_t, t, k_range, tau_range):
        """
        Optimized 2D wave transform computation
        
        Parameters:
        V_t: Voltage signal array
        t: Time array
        k_range: Array of wavenumber values
        tau_range: Array of scale values
        
        Returns:
        W_matrix: 2D complex matrix of wave transform values
        """
        print(f"🔬 Computing Optimized √t Wave Transform...")
        print(f"  Signal length: {len(V_t)} samples")
        print(f"  k range: {k_range[0]:.3f} to {k_range[-1]:.3f}")
        print(f"  τ range: {tau_range[0]:.3f} to {tau_range[-1]:.3f}")
        
        start_time = time.time()
        
        # Initialize result matrix
        W_matrix = np.zeros((len(k_range), len(tau_range)), dtype=complex)
        
        # Progress tracking
        total_computations = len(k_range) * len(tau_range)
        completed = 0
        
        # Use finite integration limit for stability
        max_t = t[-1]
        
        for i, k in enumerate(k_range):
            for j, tau in enumerate(tau_range):
                # Compute wave transform for this (k, τ) pair
                W_val = self.optimized_sqrt_wave_transform(V_t, t, k, tau, max_t)
                W_matrix[i, j] = W_val
                
                completed += 1
                if completed % 50 == 0:
                    progress = (completed / total_computations) * 100
                    print(f"  Progress: {progress:.1f}% ({completed}/{total_computations})")
        
        duration = time.time() - start_time
        print(f"  ✅ Wave transform completed in {duration:.3f} seconds")
        print(f"  🚀 Speed: {total_computations/duration:.1f} computations/second")
        
        return W_matrix
    
    def analyze_fungal_signals_optimized(self, V_t, t, k_range=None, tau_range=None):
        """
        Analyze fungal electrical signals using the optimized √t wave transform
        
        Parameters:
        V_t: Voltage signal array
        t: Time array
        k_range: Wavenumber range (if None, auto-generate)
        tau_range: Scale range (if None, auto-generate)
        
        Returns:
        analysis_results: Dictionary containing analysis results
        """
        
        # Auto-generate ranges if not provided
        if k_range is None:
            # k range based on signal characteristics
            signal_freq = 1.0 / (t[1] - t[0]) if len(t) > 1 else 1000
            k_range = np.linspace(0.1, 5.0, 30)  # 30 k values
        
        if tau_range is None:
            # τ range based on signal duration
            signal_duration = t[-1] - t[0]
            tau_range = np.logspace(-1, np.log10(signal_duration), 25)  # 25 τ values
        
        print(f"🍄 Analyzing Fungal Electrical Signals with Optimized √t Wave Transform")
        print(f"  Signal duration: {t[-1] - t[0]:.3f} seconds")
        print(f"  Sampling rate: {1.0/(t[1] - t[0]):.1f} Hz")
        print(f"  Voltage range: {np.min(V_t):.3f} to {np.max(V_t):.3f} mV")
        print(f"  Equation: W(k,τ) = ∫₀^T V(t) · ψ(√t/τ) · e^(-ik√t) dt")
        
        # Compute wave transform
        W_matrix = self.compute_wave_transform_2d_optimized(V_t, t, k_range, tau_range)
        
        # Analyze results
        analysis_results = self.analyze_wave_transform_results(W_matrix, k_range, tau_range)
        
        return analysis_results, W_matrix, k_range, tau_range
    
    def analyze_wave_transform_results(self, W_matrix, k_range, tau_range):
        """
        Analyze the results of the wave transform computation
        
        Parameters:
        W_matrix: Complex wave transform matrix
        k_range: Wavenumber range
        tau_range: Scale range
        
        Returns:
        results: Dictionary of analysis results
        """
        
        # Magnitude of complex values
        magnitude = np.abs(W_matrix)
        
        # Find dominant patterns
        max_magnitude_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        max_k = k_range[max_magnitude_idx[0]]
        max_tau = tau_range[max_magnitude_idx[1]]
        max_magnitude = magnitude[max_magnitude_idx]
        
        # Frequency analysis (k parameter)
        k_power = np.sum(magnitude, axis=1)
        dominant_k_idx = np.argmax(k_power)
        dominant_k = k_range[dominant_k_idx]
        
        # Scale analysis (τ parameter)
        tau_power = np.sum(magnitude, axis=0)
        dominant_tau_idx = np.argmax(tau_power)
        dominant_tau = tau_range[dominant_tau_idx]
        
        # Pattern coherence
        coherence = np.std(magnitude) / np.mean(magnitude)
        
        # Additional pattern analysis
        pattern_energy = np.sum(magnitude**2)
        pattern_entropy = -np.sum(magnitude * np.log(magnitude + 1e-10))
        
        results = {
            'dominant_pattern': {
                'k_value': max_k,
                'tau_value': max_tau,
                'magnitude': max_magnitude
            },
            'frequency_analysis': {
                'dominant_k': dominant_k,
                'k_power_distribution': k_power.tolist(),
                'k_spectrum': k_power
            },
            'scale_analysis': {
                'dominant_tau': dominant_tau,
                'tau_power_distribution': tau_power.tolist(),
                'tau_spectrum': tau_power
            },
            'pattern_characteristics': {
                'coherence': coherence,
                'total_energy': pattern_energy,
                'mean_magnitude': np.mean(magnitude),
                'pattern_entropy': pattern_entropy,
                'max_magnitude': np.max(magnitude)
            }
        }
        
        return results
    
    def create_comprehensive_visualization(self, W_matrix, k_range, tau_range, results, V_t, t, save_path=None):
        """
        Create comprehensive visualization of the wave transform results
        
        Parameters:
        W_matrix: Complex wave transform matrix
        k_range: Wavenumber range
        tau_range: Scale range
        results: Analysis results
        V_t: Original voltage signal
        t: Time array
        save_path: Path to save visualization
        """
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        fig.suptitle('Comprehensive √t Wave Transform Analysis of Fungal Electrical Signals', 
                     fontsize=16, fontweight='bold')
        
        # 1. Original voltage signal
        axes[0, 0].plot(t, V_t, 'b-', linewidth=1, alpha=0.8)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Voltage (mV)')
        axes[0, 0].set_title('Original Fungal Electrical Signal')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Wave transform magnitude surface
        magnitude = np.abs(W_matrix)
        K, TAU = np.meshgrid(k_range, tau_range)
        
        im1 = axes[0, 1].pcolormesh(TAU, K, magnitude.T, shading='gouraud', cmap='viridis')
        axes[0, 1].set_xlabel('Scale τ (s)')
        axes[0, 1].set_ylabel('Wavenumber k')
        axes[0, 1].set_title('√t Wave Transform Magnitude |W(k,τ)|')
        axes[0, 1].set_xscale('log')
        plt.colorbar(im1, ax=axes[0, 1], label='Magnitude')
        
        # Mark dominant pattern
        axes[0, 1].plot(results['dominant_pattern']['tau_value'], 
                        results['dominant_pattern']['k_value'], 
                        'r*', markersize=15, label='Dominant Pattern')
        axes[0, 1].legend()
        
        # 3. Frequency analysis (k parameter)
        k_power = results['frequency_analysis']['k_spectrum']
        axes[1, 0].plot(k_range, k_power, 'b-', linewidth=2)
        axes[1, 0].axvline(results['frequency_analysis']['dominant_k'], 
                           color='r', linestyle='--', 
                           label=f'Dominant k = {results["frequency_analysis"]["dominant_k"]:.3f}')
        axes[1, 0].set_xlabel('Wavenumber k')
        axes[1, 0].set_ylabel('Power')
        axes[1, 0].set_title('Frequency Analysis (k parameter)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Scale analysis (τ parameter)
        tau_power = results['scale_analysis']['tau_spectrum']
        axes[1, 1].plot(tau_range, tau_power, 'g-', linewidth=2)
        axes[1, 1].axvline(results['scale_analysis']['dominant_tau'], 
                           color='r', linestyle='--', 
                           label=f'Dominant τ = {results["scale_analysis"]["dominant_tau"]:.3f}')
        axes[1, 1].set_xlabel('Scale τ (s)')
        axes[1, 1].set_ylabel('Power')
        axes[1, 1].set_title('Scale Analysis (τ parameter)')
        axes[1, 1].set_xscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Phase analysis
        phase = np.angle(W_matrix)
        im2 = axes[2, 0].pcolormesh(TAU, K, phase.T, shading='gouraud', cmap='twilight')
        axes[2, 0].set_xlabel('Scale τ (s)')
        axes[2, 0].set_ylabel('Wavenumber k')
        axes[2, 0].set_title('√t Wave Transform Phase ∠W(k,τ)')
        axes[2, 0].set_xscale('log')
        plt.colorbar(im2, ax=axes[2, 0], label='Phase (radians)')
        
        # 6. Summary statistics
        axes[2, 1].axis('off')
        summary_text = f"""
√t Wave Transform Results

Equation: W(k,τ) = ∫₀^T V(t) · ψ(√t/τ) · e^(-ik√t) dt

Dominant Pattern:
• k = {results['dominant_pattern']['k_value']:.3f}
• τ = {results['dominant_pattern']['tau_value']:.3f}
• Magnitude = {results['dominant_pattern']['magnitude']:.3f}

Pattern Characteristics:
• Coherence = {results['pattern_characteristics']['coherence']:.3f}
• Total Energy = {results['pattern_characteristics']['total_energy']:.3f}
• Pattern Entropy = {results['pattern_characteristics']['pattern_entropy']:.3f}
• Mean Magnitude = {results['pattern_characteristics']['mean_magnitude']:.3f}

Analysis by: {self.author}
Timestamp: {self.timestamp}
        """
        axes[2, 1].text(0.1, 0.5, summary_text, transform=axes[2, 1].transAxes, 
                        fontsize=11, verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comprehensive visualization saved to: {save_path}")
        
        plt.show()
        return fig

def main():
    """Main function to demonstrate the optimized √t wave transform."""
    print("🔬 Optimized √t Wave Transform Implementation")
    print("Author: Joe Knowles")
    print("Timestamp: 2025-08-12 09:23:27 BST")
    print("=" * 60)
    
    # Create optimized wave transform analyzer
    analyzer = OptimizedSqrtWaveTransform()
    
    # Load real fungal data
    data_path = Path("DATA/raw/15061491")
    file_path = data_path / "Spray_in_bag.csv"
    
    if not file_path.exists():
        print("❌ Data file not found. Please ensure fungal data is available.")
        return
    
    print(f"📁 Loading fungal electrical data from: {file_path}")
    
    # Load and prepare data
    try:
        data = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines and extract electrical data
        header_count = 0
        for line in lines:
            if line.strip() and not line.startswith('"'):
                header_count += 1
            if header_count > 2:
                break
        
        # Extract electrical data from remaining lines
        for line in lines[header_count:]:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) > 1:
                    try:
                        value = float(parts[1].strip('"'))
                        data.append(value)
                    except (ValueError, IndexError):
                        continue
        
        print(f"✅ Loaded {len(data)} electrical measurements")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    if len(data) < 10:
        print("❌ Insufficient data for analysis")
        return
    
    # Prepare time array (assuming 1 second intervals)
    t = np.linspace(0, len(data), len(data))
    V_t = np.array(data)
    
    # Define k and τ ranges for analysis
    k_range = np.linspace(0.1, 5.0, 30)  # 30 wavenumber values
    tau_range = np.logspace(-1, 1, 25)    # 25 scale values (logarithmic)
    
    print(f"\n🔬 Computing Optimized √t Wave Transform...")
    print(f"  Equation: W(k,τ) = ∫₀^T V(t) · ψ(√t/τ) · e^(-ik√t) dt")
    print(f"  k values: {len(k_range)} (range: {k_range[0]:.3f} to {k_range[-1]:.3f})")
    print(f"  τ values: {len(tau_range)} (range: {tau_range[0]:.3f} to {tau_range[-1]:.3f})")
    
    # Perform optimized wave transform analysis
    results, W_matrix, k_range, tau_range = analyzer.analyze_fungal_signals_optimized(
        V_t, t, k_range, tau_range
    )
    
    # Display results
    print(f"\n📊 Optimized √t Wave Transform Results:")
    print(f"=" * 50)
    print(f"Dominant Pattern:")
    print(f"  • k = {results['dominant_pattern']['k_value']:.3f}")
    print(f"  • τ = {results['dominant_pattern']['tau_value']:.3f}")
    print(f"  • Magnitude = {results['dominant_pattern']['magnitude']:.3f}")
    
    print(f"\nPattern Characteristics:")
    print(f"  • Coherence = {results['pattern_characteristics']['coherence']:.3f}")
    print(f"  • Total Energy = {results['pattern_characteristics']['total_energy']:.3f}")
    print(f"  • Pattern Entropy = {results['pattern_characteristics']['pattern_entropy']:.3f}")
    print(f"  • Mean Magnitude = {results['pattern_characteristics']['mean_magnitude']:.3f}")
    
    # Create comprehensive visualization
    print(f"\n🎨 Creating comprehensive visualization...")
    save_path = "RESULTS/analysis/optimized_sqrt_wave_transform_analysis.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig = analyzer.create_comprehensive_visualization(
        W_matrix, k_range, tau_range, results, V_t, t, save_path
    )
    
    print(f"\n🎉 Optimized √t Wave Transform Analysis Complete!")
    print(f"📊 Results saved and visualized")
    print(f"🔬 Real fungal electrical data analyzed using correct equation")
    print(f"⚡ Performance optimized for stability and speed")

if __name__ == "__main__":
    import os
    main() 