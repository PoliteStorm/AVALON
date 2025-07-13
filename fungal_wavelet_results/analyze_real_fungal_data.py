#!/usr/bin/env python3
"""
Analyze real fungal electrical activity data using the √t wavelet transform.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqrt_wavelet_transform import SqrtWaveletTransform
import os
from datetime import datetime

class RealFungalAnalyzer:
    def __init__(self, csv_file_path, sampling_rate=10.0):
        """
        Initialize the analyzer for real fungal data.
        
        Args:
            csv_file_path: Path to the CSV file containing fungal electrical activity
            sampling_rate: Sampling rate in Hz (default: 10.0 Hz)
        """
        self.csv_file_path = csv_file_path
        self.sampling_rate = sampling_rate
        self.transform = SqrtWaveletTransform()
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the fungal electrical activity data.
        
        Returns:
            dict: Dictionary containing processed signals
        """
        print(f"Loading data from: {self.csv_file_path}")
        
        # Load CSV data
        try:
            df = pd.read_csv(self.csv_file_path, header=None)
            print(f"Data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Extract x and y coordinates
            x_coords = df.iloc[:, 0].values
            y_coords = df.iloc[:, 1].values
            
            # Create time vector
            time = np.arange(len(x_coords)) / self.sampling_rate
            
            # Preprocess signals
            signals = {}
            
            # Raw signals
            signals['x_coordinate'] = x_coords
            signals['y_coordinate'] = y_coords
            
            # Derived signals
            signals['distance'] = np.sqrt(x_coords**2 + y_coords**2)
            signals['velocity'] = np.gradient(signals['distance'])
            signals['acceleration'] = np.gradient(signals['velocity'])
            
            # Normalize signals (except distance which might have biological meaning)
            for key in ['x_coordinate', 'y_coordinate', 'velocity', 'acceleration']:
                sig = signals[key]
                if np.std(sig) > 0:
                    signals[key] = (sig - np.mean(sig)) / np.std(sig)
            
            # Print signal statistics
            print("\nSignal Statistics:")
            for key, sig in signals.items():
                print(f"  {key}:")
                print(f"    min: {np.nanmin(sig):.4g}, max: {np.nanmax(sig):.4g}")
                print(f"    mean: {np.nanmean(sig):.4g}, std: {np.nanstd(sig):.4g}")
                print(f"    nan count: {np.isnan(sig).sum()}, inf count: {np.isinf(sig).sum()}")
            
            return signals, time
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def analyze_signal(self, signal, signal_name, time):
        """
        Analyze a single signal using the √t wavelet transform.
        
        Args:
            signal: The signal to analyze
            signal_name: Name of the signal
            time: Time vector
            
        Returns:
            dict: Analysis results
        """
        print(f"\nAnalyzing {signal_name}...")
        
        try:
            # Apply √t wavelet transform
            coeffs, magnitude, phase = self.transform.analyze_signal(signal)
            
            # Compute signature metrics
            signatures = self.transform.compute_alternative_signature(magnitude)
            
            # Basic statistics
            peak_magnitude = np.max(magnitude) if magnitude.size > 0 else 0
            mean_magnitude = np.mean(magnitude) if magnitude.size > 0 else 0
            
            results = {
                'signal_name': signal_name,
                'signal_length': len(signal),
                'peak_magnitude': peak_magnitude,
                'mean_magnitude': mean_magnitude,
                'sqrt_t_signature': signatures.get('sqrt_signature', 0.0),
                'tau_scaling': signatures.get('tau_scaling', 0.0),
                'k_scaling': signatures.get('k_scaling', 0.0),
                'magnitude_shape': magnitude.shape,
                'tau_range': (float(self.transform.tau_values[0]), float(self.transform.tau_values[-1])) if len(self.transform.tau_values) > 0 else (0, 0),
                'k_range': (float(self.transform.k_values[0]), float(self.transform.k_values[-1])) if len(self.transform.k_values) > 0 else (0, 0)
            }
            
            print(f"  √t signature: {results['sqrt_t_signature']:.4f}")
            print(f"  τ scaling: {results['tau_scaling']:.4f}")
            print(f"  k scaling: {results['k_scaling']:.4f}")
            print(f"  Peak magnitude: {results['peak_magnitude']:.4f}")
            
            return results, magnitude, None, None
            
        except Exception as e:
            print(f"  Error analyzing {signal_name}: {e}")
            return None, None, None, None
    
    def create_visualizations(self, signals, time, results_dict, output_dir="real_fungal_analysis"):
        """
        Create comprehensive visualizations of the analysis.
        
        Args:
            signals: Dictionary of signals
            time: Time vector
            results_dict: Dictionary of analysis results
            output_dir: Output directory for plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Raw signals
        ax1 = plt.subplot(3, 3, 1)
        for key in ['x_coordinate', 'y_coordinate']:
            plt.plot(time, signals[key], label=key, alpha=0.7)
        plt.title('Raw Coordinates')
        plt.xlabel('Time (s)')
        plt.ylabel('Coordinate Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Derived signals
        ax2 = plt.subplot(3, 3, 2)
        for key in ['distance', 'velocity', 'acceleration']:
            plt.plot(time, signals[key], label=key, alpha=0.7)
        plt.title('Derived Signals')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Signature comparison
        ax3 = plt.subplot(3, 3, 3)
        signal_names = [r['signal_name'] for r in results_dict.values() if r is not None]
        signatures = [r['sqrt_t_signature'] for r in results_dict.values() if r is not None]
        
        bars = plt.bar(range(len(signatures)), signatures, alpha=0.7)
        plt.title('√t Signature Comparison')
        plt.xlabel('Signal Type')
        plt.ylabel('√t Signature')
        plt.xticks(range(len(signal_names)), signal_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Color bars based on signature strength
        for i, (bar, sig) in enumerate(zip(bars, signatures)):
            if sig > 0.3:
                bar.set_color('green')  # Strong √t pattern
            elif sig > 0.1:
                bar.set_color('orange')  # Moderate √t pattern
            else:
                bar.set_color('red')  # Weak or no √t pattern
        
        # 4-9. Individual signal analysis plots
        signal_keys = list(signals.keys())
        n_plots = min(len(signal_keys), 6)  # Maximum 6 plots
        
        # Create a 2x3 grid for individual plots
        for i in range(n_plots):
            key = signal_keys[i]
            if key in results_dict and results_dict[key] is not None:
                # Use proper subplot indexing: 2 rows, 3 columns
                ax = plt.subplot(2, 3, i + 1)
                
                # Plot signal
                plt.plot(time, signals[key], 'b-', alpha=0.7, label=f'{key}')
                plt.title(f'{key}\n√t Signature: {results_dict[key]["sqrt_t_signature"]:.4f}')
                plt.xlabel('Time (s)')
                plt.ylabel('Signal Value')
                plt.grid(True, alpha=0.3)
                
                # Add signature strength indicator
                sig = results_dict[key]["sqrt_t_signature"]
                if np.isnan(sig):
                    plt.text(0.02, 0.98, 'No √t Pattern', transform=ax.transAxes,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="gray", alpha=0.7),
                            verticalalignment='top')
                elif sig > 0.3:
                    plt.text(0.02, 0.98, 'Strong √t Pattern', transform=ax.transAxes, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7),
                            verticalalignment='top')
                elif sig > 0.1:
                    plt.text(0.02, 0.98, 'Moderate √t Pattern', transform=ax.transAxes,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                            verticalalignment='top')
                else:
                    plt.text(0.02, 0.98, 'Weak √t Pattern', transform=ax.transAxes,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                            verticalalignment='top')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{output_dir}/real_fungal_analysis_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {plot_filename}")
        
        # Save results summary
        summary_filename = f"{output_dir}/analysis_summary_{timestamp}.json"
        import json
        with open(summary_filename, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"Saved analysis summary to: {summary_filename}")
        
        plt.show()
    
    def run_analysis(self):
        """
        Run the complete analysis pipeline.
        """
        print("=== Real Fungal Data Analysis with √t Wavelet Transform ===\n")
        
        # Load and preprocess data
        signals, time = self.load_and_preprocess_data()
        if signals is None:
            return
        
        # Analyze each signal
        results_dict = {}
        magnitude_dict = {}
        tau_dict = {}
        k_dict = {}
        
        for signal_name, signal in signals.items():
            results, magnitude, tau, k = self.analyze_signal(signal, signal_name, time)
            if results is not None:
                results_dict[signal_name] = results
                magnitude_dict[signal_name] = magnitude
                tau_dict[signal_name] = tau
                k_dict[signal_name] = k
        
        # --- Negative Controls ---
        # 1. Shuffled distance (multiple times)
        n_shuffles = 100
        shuffled_signatures = []
        if 'distance' in signals:
            for _ in range(n_shuffles):
                shuffled_distance = np.copy(signals['distance'])
                np.random.shuffle(shuffled_distance)
                results, magnitude, tau, k = self.analyze_signal(shuffled_distance, 'shuffled_distance', time)
                if results is not None and not np.isnan(results['sqrt_t_signature']):
                    shuffled_signatures.append(results['sqrt_t_signature'])
            # Analyze one example for visualization
            shuffled_distance = np.copy(signals['distance'])
            np.random.shuffle(shuffled_distance)
            results, magnitude, tau, k = self.analyze_signal(shuffled_distance, 'shuffled_distance_example', time)
            if results is not None:
                results_dict['shuffled_distance_example'] = results
        # 2. Synthetic noise
        synthetic_noise = np.random.normal(0, 1, len(time))
        results, magnitude, tau, k = self.analyze_signal(synthetic_noise, 'synthetic_noise', time)
        if results is not None:
            results_dict['synthetic_noise'] = results
        
        # Print shuffle stats
        if shuffled_signatures:
            import scipy.stats as stats
            real_sig = results_dict['distance']['sqrt_t_signature'] if 'distance' in results_dict else None
            mean_sig = np.mean(shuffled_signatures)
            std_sig = np.std(shuffled_signatures)
            percentile = stats.percentileofscore(shuffled_signatures, real_sig) if real_sig is not None else None
            print(f"\nShuffled control (n={n_shuffles}):")
            print(f"  Mean signature: {mean_sig:.4f}")
            print(f"  Std signature: {std_sig:.4f}")
            print(f"  Real distance signature: {real_sig:.4f}")
            print(f"  Percentile rank of real: {percentile:.1f}%")
            # Plot histogram
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7,4))
            plt.hist(shuffled_signatures, bins=20, alpha=0.7, label='Shuffled')
            if real_sig is not None:
                plt.axvline(real_sig, color='red', linestyle='--', label='Real distance')
            plt.xlabel('√t Signature')
            plt.ylabel('Count')
            plt.title('Distribution of √t Signatures (Shuffled Controls)')
            plt.legend()
            plt.tight_layout()
            plt.savefig('real_fungal_analysis/shuffled_signature_histogram.png', dpi=200)
            plt.show()
        
        # Create visualizations
        self.create_visualizations(signals, time, results_dict)
        
        # Print summary
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Dataset: {os.path.basename(self.csv_file_path)}")
        print(f"Signals analyzed: {len(results_dict)}")
        print(f"Data points: {len(time)}")
        print(f"Duration: {time[-1]:.2f} seconds")
        
        # Find strongest √t pattern
        if results_dict:
            strongest_signal = max(results_dict.items(), 
                                 key=lambda x: x[1]['sqrt_t_signature'])
            print(f"\nStrongest √t pattern: {strongest_signal[0]}")
            print(f"Signature value: {strongest_signal[1]['sqrt_t_signature']:.4f}")
            
            # Overall assessment
            avg_signature = np.mean([r['sqrt_t_signature'] for r in results_dict.values()])
            print(f"Average signature: {avg_signature:.4f}")
            
            if avg_signature > 0.2:
                print("Assessment: Strong √t patterns detected in fungal activity")
            elif avg_signature > 0.1:
                print("Assessment: Moderate √t patterns detected in fungal activity")
            else:
                print("Assessment: Weak or no √t patterns detected in fungal activity")

if __name__ == "__main__":
    # Analyze the small fungal dataset
    csv_file = "csv_data/Rb_M_I_Fc-M_N_16d_3_coordinates.csv"
    
    analyzer = RealFungalAnalyzer(csv_file, sampling_rate=10.0)
    analyzer.run_analysis() 