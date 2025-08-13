#!/usr/bin/env python3
"""
Correct √t Wave Transform Implementation
Author: Joe Knowles
Timestamp: 2025-08-12 09:23:27 BST
Description: Implements the correct wave transform W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import signal
import time
from pathlib import Path

class SqrtWaveTransform:
    """
    Implements the correct √t wave transform for fungal electrical signals:
    W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
    """
    
    def __init__(self):
        self.timestamp = "20250812_092327"
        self.author = "Joe Knowles"
        
    def mother_wavelet(self, t, tau):
        """
        Mother wavelet function ψ(√t/τ)
        Using a modified Morlet wavelet adapted for √t scaling
        """
        # Normalize time by τ
        normalized_t = np.sqrt(t) / np.sqrt(tau)
        
        # Modified Morlet wavelet for √t scaling
        # ψ(x) = (1/√(2π)) * e^(-x²/2) * e^(iω₀x)
        omega_0 = 2.0  # Central frequency parameter
        
        # Gaussian envelope
        gaussian = np.exp(-normalized_t**2 / 2)
        
        # Complex exponential for frequency content
        complex_exp = np.exp(1j * omega_0 * normalized_t)
        
        # Normalization factor
        norm_factor = 1.0 / np.sqrt(2 * np.pi)
        
        return norm_factor * gaussian * complex_exp
    
    def sqrt_wave_transform(self, V_t, t, k, tau):
        """
        Compute the √t wave transform:
        W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
        
        Parameters:
        V_t: Voltage signal V(t)
        t: Time array
        k: Wavenumber parameter
        tau: Scale parameter
        
        Returns:
        W_k_tau: Complex wave transform value
        """
        
        def integrand(t_val):
            """Integrand function for the wave transform"""
            if t_val <= 0:
                return 0.0
            
            # Find closest time index
            t_idx = np.argmin(np.abs(t - t_val))
            
            # Get voltage value at this time
            V_val = V_t[t_idx] if t_idx < len(V_t) else 0.0
            
            # Mother wavelet ψ(√t/τ)
            psi_val = self.mother_wavelet(t_val, tau)
            
            # Complex exponential e^(-ik√t)
            exp_val = np.exp(-1j * k * np.sqrt(t_val))
            
            # Complete integrand
            return V_val * psi_val * exp_val
        
        # Perform numerical integration
        try:
            # Use scipy's quad for numerical integration
            result, error = integrate.quad(integrand, 0, np.inf, limit=1000)
            return result, error
        except Exception as e:
            print(f"Integration error: {e}")
            return 0.0, 0.0
    
    def compute_wave_transform_2d(self, V_t, t, k_range, tau_range):
        """
        Compute 2D wave transform across k and τ ranges
        
        Parameters:
        V_t: Voltage signal V(t)
        t: Time array
        k_range: Array of wavenumber values
        tau_range: Array of scale values
        
        Returns:
        W_matrix: 2D complex matrix of wave transform values
        """
        print(f"🔬 Computing √t Wave Transform...")
        print(f"  Signal length: {len(V_t)} samples")
        print(f"  k range: {k_range[0]:.3f} to {k_range[-1]:.3f}")
        print(f"  τ range: {tau_range[0]:.3f} to {tau_range[-1]:.3f}")
        
        start_time = time.time()
        
        # Initialize result matrix
        W_matrix = np.zeros((len(k_range), len(tau_range)), dtype=complex)
        
        # Progress tracking
        total_computations = len(k_range) * len(tau_range)
        completed = 0
        
        for i, k in enumerate(k_range):
            for j, tau in enumerate(tau_range):
                # Compute wave transform for this (k, τ) pair
                W_val, error = self.sqrt_wave_transform(V_t, t, k, tau)
                W_matrix[i, j] = W_val
                
                completed += 1
                if completed % 100 == 0:
                    progress = (completed / total_computations) * 100
                    print(f"  Progress: {progress:.1f}% ({completed}/{total_computations})")
        
        duration = time.time() - start_time
        print(f"  ✅ Wave transform completed in {duration:.3f} seconds")
        print(f"  🚀 Speed: {total_computations/duration:.1f} computations/second")
        
        return W_matrix
    
    def analyze_fungal_signals(self, V_t, t, k_range=None, tau_range=None):
        """
        Analyze fungal electrical signals using the √t wave transform
        
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
            k_range = np.linspace(0.1, 10.0, 50)  # 50 k values
        
        if tau_range is None:
            # τ range based on signal duration
            signal_duration = t[-1] - t[0]
            tau_range = np.logspace(-2, np.log10(signal_duration), 30)  # 30 τ values
        
        print(f"🍄 Analyzing Fungal Electrical Signals with √t Wave Transform")
        print(f"  Signal duration: {t[-1] - t[0]:.3f} seconds")
        print(f"  Sampling rate: {1.0/(t[1] - t[0]):.1f} Hz")
        print(f"  Voltage range: {np.min(V_t):.3f} to {np.max(V_t):.3f} mV")
        
        # Compute wave transform
        W_matrix = self.compute_wave_transform_2d(V_t, t, k_range, tau_range)
        
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
        
        results = {
            'dominant_pattern': {
                'k_value': max_k,
                'tau_value': max_tau,
                'magnitude': max_magnitude
            },
            'frequency_analysis': {
                'dominant_k': dominant_k,
                'k_power_distribution': k_power.tolist()
            },
            'scale_analysis': {
                'dominant_tau': dominant_tau,
                'tau_power_distribution': tau_power.tolist()
            },
            'pattern_characteristics': {
                'coherence': coherence,
                'total_energy': np.sum(magnitude**2),
                'mean_magnitude': np.mean(magnitude)
            }
        }
        
        return results
    
    def visualize_wave_transform(self, W_matrix, k_range, tau_range, results, save_path=None):
        """
        Visualize the wave transform results
        
        Parameters:
        W_matrix: Complex wave transform matrix
        k_range: Wavenumber range
        tau_range: Scale range
        results: Analysis results
        save_path: Path to save visualization
        """
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('√t Wave Transform Analysis of Fungal Electrical Signals', fontsize=16, fontweight='bold')
        
        # 1. Wave transform magnitude surface
        magnitude = np.abs(W_matrix)
        K, TAU = np.meshgrid(k_range, tau_range)
        
        im1 = axes[0, 0].pcolormesh(TAU, K, magnitude.T, shading='gouraud', cmap='viridis')
        axes[0, 0].set_xlabel('Scale τ (s)')
        axes[0, 0].set_ylabel('Wavenumber k')
        axes[0, 0].set_title('Wave Transform Magnitude |W(k,τ)|')
        axes[0, 0].set_xscale('log')
        plt.colorbar(im1, ax=axes[0, 0], label='Magnitude')
        
        # Mark dominant pattern
        axes[0, 0].plot(results['dominant_pattern']['tau_value'], 
                        results['dominant_pattern']['k_value'], 
                        'r*', markersize=15, label='Dominant Pattern')
        axes[0, 0].legend()
        
        # 2. Frequency analysis (k parameter)
        k_power = np.sum(magnitude, axis=1)
        axes[0, 1].plot(k_range, k_power, 'b-', linewidth=2)
        axes[0, 1].axvline(results['frequency_analysis']['dominant_k'], 
                           color='r', linestyle='--', 
                           label=f'Dominant k = {results["frequency_analysis"]["dominant_k"]:.3f}')
        axes[0, 1].set_xlabel('Wavenumber k')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].set_title('Frequency Analysis (k parameter)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Scale analysis (τ parameter)
        tau_power = np.sum(magnitude, axis=0)
        axes[1, 0].plot(tau_range, tau_power, 'g-', linewidth=2)
        axes[1, 0].axvline(results['scale_analysis']['dominant_tau'], 
                           color='r', linestyle='--', 
                           label=f'Dominant τ = {results["scale_analysis"]["dominant_tau"]:.3f}')
        axes[1, 0].set_xlabel('Scale τ (s)')
        axes[1, 0].set_ylabel('Power')
        axes[1, 0].set_title('Scale Analysis (τ parameter)')
        axes[1, 0].set_xscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
√t Wave Transform Results

Dominant Pattern:
• k = {results['dominant_pattern']['k_value']:.3f}
• τ = {results['dominant_pattern']['tau_value']:.3f}
• Magnitude = {results['dominant_pattern']['magnitude']:.3f}

Pattern Characteristics:
• Coherence = {results['pattern_characteristics']['coherence']:.3f}
• Total Energy = {results['pattern_characteristics']['total_energy']:.3f}
• Mean Magnitude = {results['pattern_characteristics']['mean_magnitude']:.3f}

Analysis by: {self.author}
Timestamp: {self.timestamp}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=12, verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        
        plt.show()
        return fig

def main():
    """Main function to demonstrate the correct √t wave transform."""
    print("🔬 Correct √t Wave Transform Implementation")
    print("Author: Joe Knowles")
    print("Timestamp: 2025-08-12 09:23:27 BST")
    print("=" * 60)
    
    # Create wave transform analyzer
    analyzer = SqrtWaveTransform()
    
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
    k_range = np.linspace(0.1, 5.0, 25)  # 25 wavenumber values
    tau_range = np.logspace(-1, 1, 20)    # 20 scale values (logarithmic)
    
    print(f"\n🔬 Computing √t Wave Transform...")
    print(f"  Equation: W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt")
    print(f"  k values: {len(k_range)} (range: {k_range[0]:.3f} to {k_range[-1]:.3f})")
    print(f"  τ values: {len(tau_range)} (range: {tau_range[0]:.3f} to {tau_range[-1]:.3f})")
    
    # Perform wave transform analysis
    results, W_matrix, k_range, tau_range = analyzer.analyze_fungal_signals(
        V_t, t, k_range, tau_range
    )
    
    # Display results
    print(f"\n📊 √t Wave Transform Results:")
    print(f"=" * 40)
    print(f"Dominant Pattern:")
    print(f"  • k = {results['dominant_pattern']['k_value']:.3f}")
    print(f"  • τ = {results['dominant_pattern']['tau_value']:.3f}")
    print(f"  • Magnitude = {results['dominant_pattern']['magnitude']:.3f}")
    
    print(f"\nPattern Characteristics:")
    print(f"  • Coherence = {results['pattern_characteristics']['coherence']:.3f}")
    print(f"  • Total Energy = {results['pattern_characteristics']['total_energy']:.3f}")
    print(f"  • Mean Magnitude = {results['pattern_characteristics']['mean_magnitude']:.3f}")
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    save_path = "RESULTS/analysis/sqrt_wave_transform_analysis.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig = analyzer.visualize_wave_transform(W_matrix, k_range, tau_range, results, save_path)
    
    print(f"\n🎉 √t Wave Transform Analysis Complete!")
    print(f"📊 Results saved and visualized")
    print(f"🔬 Real fungal electrical data analyzed using correct equation")

if __name__ == "__main__":
    import os
    main() 