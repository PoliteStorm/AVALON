#!/usr/bin/env python3
"""
Test Best Files Script
Analyze and visualize the top-performing CSV files from the batch analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultra_optimized_fungal_monitoring_simple import UltraOptimizedFungalMonitor
import json
from datetime import datetime

# Best files to test (from the batch analysis results)
BEST_FILES = [
    "fungal_analysis_project/data/raw/15061491/Norm_vs_deep_tip_crop.csv",
    "fungal_analysis_project/data/15061491/Norm_vs_deep_tip_crop.csv", 
    "fungal_analysis_project/data/raw/15061491/Ch1-2_moisture_added.csv",
    "fungal_analysis_project/data/15061491/Ch1-2_moisture_added.csv"
]

def load_voltage_data(csv_path):
    """Load voltage data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        # Try to find voltage column
        voltage_columns = ['voltage', 'signal', 'amplitude', 'data', 'V', 'mv']
        for col in voltage_columns:
            if col in df.columns:
                return df[col].values, col
        # Fallback to first column
        return df.iloc[:, 0].values, df.columns[0]
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

def analyze_and_visualize(file_path, monitor, file_index):
    """Analyze a file and create visualizations"""
    print(f"\n=== Testing File {file_index + 1}: {file_path} ===")
    
    # Load data
    voltage_data, column_name = load_voltage_data(file_path)
    if voltage_data is None:
        print(f"Could not load data from {file_path}")
        return None
    
    print(f"Data loaded: {len(voltage_data)} samples from column '{column_name}'")
    print(f"Data range: {voltage_data.min():.4f} to {voltage_data.max():.4f}")
    
    # Analyze with ultra-optimized method
    results = monitor.analyze_recording_ultra_optimized(voltage_data)
    
    # Print results
    stats = results['stats']
    wave_features = results['wave_features']
    
    print(f"\nAnalysis Results:")
    print(f"- Quality Score: {stats['quality_score']:.3f}")
    print(f"- SNR: {stats['snr']:.2f}")
    print(f"- Spikes Detected: {stats['n_spikes']}")
    print(f"- Spike Rate: {stats['spike_rate']:.3f} Hz")
    print(f"- Mean Amplitude: {stats['mean_amplitude']:.4f} mV")
    print(f"- Mean ISI: {stats['mean_isi']:.3f} s")
    print(f"- Wave Patterns: {wave_features['wave_patterns']}")
    print(f"- Wave Confidence: {wave_features['confidence']:.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Analysis of {file_path.split("/")[-1]}', fontsize=16)
    
    # Plot 1: Raw signal
    axes[0, 0].plot(voltage_data[:min(10000, len(voltage_data))], linewidth=0.5)
    axes[0, 0].set_title('Raw Electrical Signal (First 10k samples)')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Voltage')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Spike locations
    if results['spikes']:
        spike_times = [spike['time_seconds'] for spike in results['spikes']]
        spike_amplitudes = [spike['amplitude_mv'] for spike in results['spikes']]
        
        # Show spikes on signal
        sample_indices = [int(spike['time_seconds'] * 1000) for spike in results['spikes']]  # Assuming 1kHz
        sample_indices = [i for i in sample_indices if i < len(voltage_data)]
        
        axes[0, 1].plot(voltage_data[:min(10000, len(voltage_data))], linewidth=0.5, alpha=0.7)
        if sample_indices:
            spike_samples = [i for i in sample_indices if i < 10000]
            if spike_samples:
                axes[0, 1].scatter(spike_samples, voltage_data[spike_samples], 
                                  color='red', s=20, zorder=5, label=f'{len(spike_samples)} spikes')
        axes[0, 1].set_title('Signal with Detected Spikes')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Voltage')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No spikes detected', ha='center', va='center', 
                        transform=axes[0, 1].transAxes, fontsize=14)
        axes[0, 1].set_title('No Spikes Detected')
    
    # Plot 3: Spike amplitude distribution
    if results['spikes']:
        amplitudes = [spike['amplitude_mv'] for spike in results['spikes']]
        axes[1, 0].hist(amplitudes, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Spike Amplitude Distribution')
        axes[1, 0].set_xlabel('Amplitude (mV)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No spikes for histogram', ha='center', va='center',
                        transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].set_title('No Spikes for Histogram')
    
    # Plot 4: Wave patterns
    if results['wave_patterns']:
        patterns = results['wave_patterns']
        scales = [p['scale'] for p in patterns]
        strengths = [p['strength'] for p in patterns]
        
        axes[1, 1].scatter(scales, strengths, alpha=0.7, s=50)
        axes[1, 1].set_title('Wave Pattern Strengths vs Scales')
        axes[1, 1].set_xlabel('Scale')
        axes[1, 1].set_ylabel('Strength')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No wave patterns detected', ha='center', va='center',
                        transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('No Wave Patterns')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"test_file_{file_index + 1}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {plot_filename}")
    
    plt.show()
    
    return results

def main():
    """Test the best files from batch analysis"""
    print("=== Testing Best Files from Batch Analysis ===")
    
    # Initialize monitor
    monitor = UltraOptimizedFungalMonitor()
    monitor.get_species_parameters('pleurotus')
    
    all_results = []
    
    for i, file_path in enumerate(BEST_FILES):
        try:
            results = analyze_and_visualize(file_path, monitor, i)
            if results:
                all_results.append({
                    'file': file_path,
                    'results': results
                })
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Successfully analyzed {len(all_results)} files")
    
    if all_results:
        # Find best overall file
        best_file = max(all_results, key=lambda x: x['results']['stats']['quality_score'])
        print(f"\nBest overall file: {best_file['file']}")
        print(f"Quality Score: {best_file['results']['stats']['quality_score']:.3f}")
        print(f"Spikes: {best_file['results']['stats']['n_spikes']}")
        print(f"Wave Patterns: {best_file['results']['wave_features']['wave_patterns']}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"test_best_files_results_{timestamp}.json"
    
    # Convert results for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    json_results = []
    for result in all_results:
        json_result = {
            'file': result['file'],
            'stats': result['results']['stats'],
            'wave_features': result['results']['wave_features']
        }
        # Convert numpy objects
        json_result = json.loads(json.dumps(json_result, default=convert_numpy))
        json_results.append(json_result)
    
    with open(results_filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to {results_filename}")

if __name__ == "__main__":
    main() 