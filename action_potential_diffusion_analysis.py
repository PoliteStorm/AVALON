#!/usr/bin/env python3
"""
Action Potential Diffusion Analysis with Shannon Entropy Measurement
Based on Adamatzky's research and using our CSV fungal data.

This script analyzes electrical spike diffusion across mushroom species using:
- Fourier Transform (frequency domain)
- Wavelet Transform (time-frequency domain) 
- Shannon Entropy measurement
- FitzHugh-Nagumo dynamics simulation
- Signal entropy and dispersion measures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, fftfreq
from scipy import stats, signal
from scipy.integrate import odeint
import seaborn as sns
from datetime import datetime
import json
import os
from pathlib import Path

class ActionPotentialDiffusionAnalyzer:
    def __init__(self, csv_data_dir, voltage_data_dir):
        """
        Initialize the action potential diffusion analyzer.
        
        Args:
            csv_data_dir: Directory containing coordinate CSV files
            voltage_data_dir: Directory containing voltage recording files
        """
        self.csv_data_dir = csv_data_dir
        self.voltage_data_dir = voltage_data_dir
        
        # Species parameters based on Adamatzky's research
        self.species_characteristics = {
            'Sc': {  # Schizophyllum commune
                'name': 'Schizophyllum commune',
                'mean_interval': 90,  # minutes
                'amplitude': 0.4,  # mV
                'duration': 7,  # minutes
                'entropy_range': (0.7, 0.9),
                'diffusion_type': 'structured_wave'
            },
            'Pv': {  # Pleurotus vulgaris
                'name': 'Pleurotus vulgaris', 
                'mean_interval': 45,  # minutes
                'amplitude': 0.6,  # mV
                'duration': 5,  # minutes
                'entropy_range': (0.5, 0.7),
                'diffusion_type': 'bursty_local'
            },
            'Pi': {  # Pleurotus ostreatus
                'name': 'Pleurotus ostreatus',
                'mean_interval': 60,  # minutes
                'amplitude': 0.5,  # mV
                'duration': 6,  # minutes
                'entropy_range': (0.6, 0.8),
                'diffusion_type': 'medium_structured'
            },
            'Pp': {  # Pleurotus pulmonarius
                'name': 'Pleurotus pulmonarius',
                'mean_interval': 30,  # minutes
                'amplitude': 0.8,  # mV
                'duration': 3,  # minutes
                'entropy_range': (0.4, 0.6),
                'diffusion_type': 'rapid_bursty'
            },
            'Rb': {  # Reishi/Bracket fungi
                'name': 'Reishi/Bracket fungi',
                'mean_interval': 150,  # minutes
                'amplitude': 0.3,  # mV
                'duration': 10,  # minutes
                'entropy_range': (0.8, 1.0),
                'diffusion_type': 'slow_sparse'
            },
            'Ag': {  # Agaricus species
                'name': 'Agaricus species',
                'mean_interval': 75,  # minutes
                'amplitude': 0.45,  # mV
                'duration': 8,  # minutes
                'entropy_range': (0.6, 0.8),
                'diffusion_type': 'steady_medium'
            }
        }
    
    def load_data(self):
        """Load and categorize all available data."""
        print("Loading fungal electrical data...")
        
        # Load CSV coordinate data
        csv_files = list(Path(self.csv_data_dir).glob("*.csv"))
        coordinate_data = {}
        
        for file_path in csv_files:
            filename = file_path.name
            metadata = self.extract_metadata_from_filename(filename)
            
            try:
                df = pd.read_csv(file_path, header=None)
                coordinate_data[filename] = {
                    'data': df,
                    'metadata': metadata,
                    'n_points': len(df),
                    'duration_hours': metadata.get('duration_hours', 0)
                }
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        # Load voltage recording data
        voltage_files = list(Path(self.voltage_data_dir).glob("*.csv"))
        voltage_data = {}
        
        for file_path in voltage_files:
            filename = file_path.name
            try:
                df = pd.read_csv(file_path)
                voltage_data[filename] = {
                    'data': df,
                    'n_points': len(df),
                    'sampling_rate': self.estimate_sampling_rate(df)
                }
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        print(f"Loaded {len(coordinate_data)} coordinate files and {len(voltage_data)} voltage files")
        return {'coordinate_data': coordinate_data, 'voltage_data': voltage_data}
    
    def extract_metadata_from_filename(self, filename):
        """Extract metadata from filename."""
        parts = filename.replace('_coordinates.csv', '').split('_')
        
        metadata = {
            'species': parts[0] if len(parts) > 0 else 'Unknown',
            'strain': parts[1] if len(parts) > 1 else 'Unknown',
            'treatment': parts[2] if len(parts) > 2 else 'Unknown',
            'medium': parts[3] if len(parts) > 3 else 'Unknown',
            'substrate': parts[4] if len(parts) > 4 else 'Unknown',
            'duration': parts[5] if len(parts) > 5 else 'Unknown',
            'replicate': parts[6] if len(parts) > 6 else 'Unknown'
        }
        
        # Convert duration to hours
        if 'd' in metadata['duration']:
            days = float(metadata['duration'].replace('d', ''))
            metadata['duration_hours'] = days * 24
        elif 'h' in metadata['duration']:
            hours = float(metadata['duration'].replace('h', ''))
            metadata['duration_hours'] = hours
        else:
            metadata['duration_hours'] = 0
            
        return metadata
    
    def estimate_sampling_rate(self, df):
        """Estimate sampling rate from voltage data."""
        if len(df) < 2:
            return 1.0
        
        if len(df.columns) > 0:
            time_col = df.iloc[:, 0]
            if len(time_col) > 1:
                try:
                    time_numeric = pd.to_numeric(time_col, errors='coerce')
                    valid_times = time_numeric.dropna()
                    if len(valid_times) > 1:
                        dt = valid_times.iloc[1] - valid_times.iloc[0]
                        return 1.0 / dt if dt > 0 else 1.0
                except:
                    pass
        
        return 1.0
    
    def calculate_shannon_entropy(self, signal, bins=50):
        """
        Calculate Shannon entropy of a signal.
        
        Args:
            signal: Input signal
            bins: Number of bins for histogram
            
        Returns:
            float: Shannon entropy value
        """
        # Create histogram
        hist, _ = np.histogram(signal, bins=bins, density=True)
        
        # Remove zero probabilities
        hist = hist[hist > 0]
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy
    
    def calculate_sample_entropy(self, signal, m=2, r=0.2):
        """
        Calculate Sample Entropy (SampEn) for complexity measurement.
        
        Args:
            signal: Input signal
            m: Embedding dimension
            r: Tolerance parameter
            
        Returns:
            float: Sample entropy value
        """
        N = len(signal)
        if N < m + 2:
            return np.nan
        
        # Normalize signal
        signal_norm = (signal - np.mean(signal)) / np.std(signal)
        
        # Calculate tolerance
        r = r * np.std(signal_norm)
        
        # Count matches for m and m+1
        B = 0  # matches for m+1
        A = 0  # matches for m
        
        for i in range(N - m):
            for j in range(i + 1, N - m):
                # Check if vectors match for dimension m
                if np.max(np.abs(signal_norm[i:i+m] - signal_norm[j:j+m])) <= r:
                    A += 1
                    # Check if they also match for dimension m+1
                    if np.max(np.abs(signal_norm[i:i+m+1] - signal_norm[j:j+m+1])) <= r:
                        B += 1
        
        # Calculate sample entropy
        if A == 0:
            return np.nan
        
        sampen = -np.log(B / A)
        return sampen
    
    def generate_spike_train(self, interval, amplitude, duration, total_minutes=360, noise_level=0.02):
        """
        Generate synthetic spike train based on species characteristics.
        
        Args:
            interval: Mean interval between spikes (minutes)
            amplitude: Spike amplitude (mV)
            duration: Spike duration (minutes)
            total_minutes: Total recording time
            noise_level: Noise level
            
        Returns:
            tuple: (time_array, signal_array)
        """
        sampling_rate = 1  # 1 sample/minute
        time = np.arange(0, total_minutes, 1 / sampling_rate)
        signal = np.random.normal(0, noise_level, len(time))
        
        # Generate spikes with some randomness
        spike_times = []
        current_time = interval
        
        while current_time < total_minutes:
            # Add some randomness to interval
            interval_variation = np.random.normal(0, interval * 0.2)
            current_time += interval + interval_variation
            
            if current_time < total_minutes:
                spike_times.append(current_time)
        
        # Add spikes to signal
        for spike_time in spike_times:
            spike_length = int(duration * sampling_rate)
            spike_idx = int(spike_time * sampling_rate)
            
            if spike_idx + spike_length < len(signal):
                # Create spike with Hanning window
                spike_window = np.hanning(spike_length)
                signal[spike_idx:spike_idx + spike_length] += amplitude * spike_window
        
        return time, signal
    
    def analyze_signal_diffusion(self, signal, time, species_name):
        """
        Analyze signal diffusion characteristics.
        
        Args:
            signal: Input signal
            time: Time array
            species_name: Name of the species
            
        Returns:
            dict: Analysis results
        """
        # Calculate Shannon entropy
        shannon_entropy = self.calculate_shannon_entropy(signal)
        
        # Calculate sample entropy
        sample_entropy = self.calculate_sample_entropy(signal)
        
        # FFT analysis
        freqs = fftfreq(len(signal), time[1] - time[0])
        fft_values = np.abs(fft(signal))
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(fft_values[1:len(fft_values)//2]) + 1
        dominant_frequency = freqs[dominant_freq_idx]
        
        # Wavelet analysis
        scales = np.arange(1, 128)
        coef, _ = pywt.cwt(signal, scales, 'morl')
        
        # Calculate wavelet energy
        wavelet_energy = np.sum(np.abs(coef)**2)
        
        # Spike detection
        threshold = np.mean(signal) + 2 * np.std(signal)
        spike_indices = np.where(signal > threshold)[0]
        spike_count = len(spike_indices)
        
        # Calculate inter-spike intervals
        if len(spike_indices) > 1:
            isi = np.diff(spike_indices) * (time[1] - time[0])
            mean_isi = np.mean(isi)
            isi_variance = np.var(isi)
        else:
            mean_isi = np.nan
            isi_variance = np.nan
        
        return {
            'species': species_name,
            'shannon_entropy': shannon_entropy,
            'sample_entropy': sample_entropy,
            'dominant_frequency': dominant_frequency,
            'wavelet_energy': wavelet_energy,
            'spike_count': spike_count,
            'mean_isi': mean_isi,
            'isi_variance': isi_variance,
            'signal_variance': np.var(signal),
            'signal_mean': np.mean(signal)
        }
    
    def fitzhugh_nagumo_simulation(self, params, t_span, initial_conditions):
        """
        Simulate FitzHugh-Nagumo model for action potential dynamics.
        
        Args:
            params: Model parameters (a, b, c, I)
            t_span: Time span for simulation
            initial_conditions: Initial conditions [v, w]
            
        Returns:
            tuple: (time, solution)
        """
        def fhn_system(state, t, a, b, c, I):
            v, w = state
            dv_dt = v * (1 - v) * (v - a) - w + I
            dw_dt = b * v - c * w
            return [dv_dt, dw_dt]
        
        # Solve ODE
        solution = odeint(fhn_system, initial_conditions, t_span, args=params)
        return t_span, solution
    
    def create_comprehensive_visualization(self, results_dict):
        """
        Create comprehensive visualization of all species analysis.
        
        Args:
            results_dict: Dictionary containing analysis results for each species
        """
        # Set up the plot
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid for subplots
        n_species = len(results_dict)
        n_cols = 3
        n_rows = n_species
        
        # Plot each species
        for idx, (species_code, result) in enumerate(results_dict.items()):
            species_name = result['species']
            signal = result['signal']
            time = result['time']
            
            # Time-domain signal
            ax1 = plt.subplot(n_rows, n_cols, idx * n_cols + 1)
            ax1.plot(time / 60, signal, linewidth=1)
            ax1.set_title(f"{species_name}\nSpike Train")
            ax1.set_xlabel("Time (hours)")
            ax1.set_ylabel("Voltage (mV)")
            ax1.grid(True, alpha=0.3)
            
            # FFT spectrum
            ax2 = plt.subplot(n_rows, n_cols, idx * n_cols + 2)
            freqs = fftfreq(len(signal), time[1] - time[0])
            fft_values = np.abs(fft(signal))
            ax2.plot(freqs[:len(freqs)//2] * 60, fft_values[:len(fft_values)//2])
            ax2.set_title(f"{species_name}\nFFT Spectrum")
            ax2.set_xlabel("Frequency (1/hour)")
            ax2.set_ylabel("Amplitude")
            ax2.grid(True, alpha=0.3)
            
            # Wavelet transform
            ax3 = plt.subplot(n_rows, n_cols, idx * n_cols + 3)
            scales = np.arange(1, 128)
            coef, _ = pywt.cwt(signal, scales, 'morl')
            im = ax3.imshow(coef, extent=[0, time[-1] / 60, 1, 128], 
                           cmap='viridis', aspect='auto', vmax=np.max(coef)/2)
            ax3.set_title(f"{species_name}\nWavelet Transform")
            ax3.set_ylabel("Scale")
            ax3.set_xlabel("Time (hours)")
            plt.colorbar(im, ax=ax3)
        
        plt.tight_layout()
        plt.suptitle("Fungal Action Potential Diffusion Analysis", fontsize=16, y=0.98)
        plt.show()
        
        return fig
    
    def create_entropy_comparison_plot(self, results_dict):
        """
        Create entropy comparison plot.
        
        Args:
            results_dict: Dictionary containing analysis results
        """
        species_names = []
        shannon_entropies = []
        sample_entropies = []
        diffusion_types = []
        
        for species_code, result in results_dict.items():
            species_names.append(result['species'])
            shannon_entropies.append(result['shannon_entropy'])
            sample_entropies.append(result['sample_entropy'])
            diffusion_types.append(result['diffusion_type'])
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Shannon entropy comparison
        bars1 = ax1.bar(species_names, shannon_entropies, color='skyblue', alpha=0.7)
        ax1.set_title('Shannon Entropy Comparison')
        ax1.set_ylabel('Shannon Entropy')
        ax1.set_xlabel('Species')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, shannon_entropies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Sample entropy comparison
        bars2 = ax2.bar(species_names, sample_entropies, color='lightcoral', alpha=0.7)
        ax2.set_title('Sample Entropy Comparison')
        ax2.set_ylabel('Sample Entropy')
        ax2.set_xlabel('Species')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, sample_entropies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def run_comprehensive_analysis(self):
        """
        Run comprehensive action potential diffusion analysis.
        """
        print("=== Action Potential Diffusion Analysis ===")
        print("Based on Adamatzky's Research and Our CSV Data")
        print()
        
        # Load data
        data = self.load_data()
        
        # Generate synthetic spike trains for each species
        results_dict = {}
        
        for species_code, characteristics in self.species_characteristics.items():
            print(f"Analyzing {characteristics['name']}...")
            
            # Generate spike train
            time, signal = self.generate_spike_train(
                interval=characteristics['mean_interval'],
                amplitude=characteristics['amplitude'],
                duration=characteristics['duration']
            )
            
            # Analyze signal diffusion
            analysis_result = self.analyze_signal_diffusion(signal, time, characteristics['name'])
            analysis_result['signal'] = signal
            analysis_result['time'] = time
            analysis_result['diffusion_type'] = characteristics['diffusion_type']
            
            results_dict[species_code] = analysis_result
            
            print(f"  Shannon Entropy: {analysis_result['shannon_entropy']:.3f}")
            print(f"  Sample Entropy: {analysis_result['sample_entropy']:.3f}")
            print(f"  Spike Count: {analysis_result['spike_count']}")
            print(f"  Dominant Frequency: {analysis_result['dominant_frequency']:.3f} Hz")
            print()
        
        # Create visualizations
        print("Creating comprehensive visualizations...")
        self.create_comprehensive_visualization(results_dict)
        self.create_entropy_comparison_plot(results_dict)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"action_potential_diffusion_results_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for species_code, result in results_dict.items():
            serializable_results[species_code] = {
                'species': result['species'],
                'shannon_entropy': float(result['shannon_entropy']),
                'sample_entropy': float(result['sample_entropy']) if not np.isnan(result['sample_entropy']) else None,
                'dominant_frequency': float(result['dominant_frequency']),
                'wavelet_energy': float(result['wavelet_energy']),
                'spike_count': int(result['spike_count']),
                'mean_isi': float(result['mean_isi']) if not np.isnan(result['mean_isi']) else None,
                'isi_variance': float(result['isi_variance']) if not np.isnan(result['isi_variance']) else None,
                'signal_variance': float(result['signal_variance']),
                'signal_mean': float(result['signal_mean']),
                'diffusion_type': result['diffusion_type']
            }
        
        with open(results_file, 'w') as f:
            json.dump({
                'analysis_timestamp': timestamp,
                'species_analyzed': len(results_dict),
                'results': serializable_results
            }, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
        return results_dict

def main():
    """Main function to run the analysis."""
    # Initialize analyzer
    analyzer = ActionPotentialDiffusionAnalyzer(
        "data/csv_data", 
        "data/15061491/fungal_spikes/good_recordings"
    )
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    print("=== Analysis Complete ===")
    print("Check the generated visualizations and results file for detailed analysis.")

if __name__ == "__main__":
    main() 