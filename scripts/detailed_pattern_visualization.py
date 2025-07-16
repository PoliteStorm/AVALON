#!/usr/bin/env python3
"""
Detailed Pattern Visualization for Fungal Electrical Activity
Shows comprehensive visualizations of spike detection, wave transforms, and their alignment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
from scipy import signal
from scipy.stats import linregress

warnings.filterwarnings('ignore')

def load_analysis_results(results_dir):
    """Load all analysis results from the results directory"""
    results = []
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and 'integrated_analysis' in filename:
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append(data)
    return results

def create_comprehensive_visualizations(results, output_dir="detailed_visualizations"):
    """Create comprehensive visualizations for all analysis results"""
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        filename = result['filename']
        print(f"\n📊 Creating visualizations for: {filename}")
        
        # Load the original voltage data
        voltage_file = f"validated_fungal_electrical_csvs/{filename}"
        if os.path.exists(voltage_file):
            try:
                voltage_data = pd.read_csv(voltage_file, header=None, low_memory=False)
                # Convert to numeric, handling any non-numeric values
                voltage_signal = pd.to_numeric(voltage_data.iloc[:, 0], errors='coerce').fillna(0).values
                print(f"✅ Loaded voltage data: {len(voltage_signal)} samples")
            except Exception as e:
                print(f"Warning: Could not load voltage file for {filename}: {e}")
                continue
        else:
            print(f"Warning: Could not find voltage file for {filename}")
            continue
        
        # Create comprehensive visualization
        create_single_file_visualization(result, voltage_signal, output_dir)
    
    # Create summary comparison
    create_summary_comparison(results, output_dir)

def create_single_file_visualization(result, voltage_signal, output_dir):
    """Create detailed visualization for a single file"""
    filename = result['filename']
    spike_results = result['spike_results']
    transform_results = result['transform_results']
    synthesis_results = result['synthesis_results']
    
    # Parse spike data - handle numpy array format properly
    spike_times_str = spike_results['spike_times']
    spike_amplitudes_str = spike_results['spike_amplitudes']
    
    # Convert string representation to numpy arrays
    try:
        # Remove brackets and split by spaces/newlines
        spike_times_str = spike_times_str.replace('[', '').replace(']', '').strip()
        spike_amplitudes_str = spike_amplitudes_str.replace('[', '').replace(']', '').strip()
        
        # Split and convert to float
        spike_times = np.array([float(x) for x in spike_times_str.split() if x.strip()])
        spike_amplitudes = np.array([float(x) for x in spike_amplitudes_str.split() if x.strip()])
    except Exception as e:
        print(f"Warning: Could not parse spike data for {filename}: {e}")
        spike_times = np.array([])
        spike_amplitudes = np.array([])
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Comprehensive Analysis: {filename}', fontsize=16, fontweight='bold')
    
    # 1. Raw voltage signal with spikes
    ax1 = plt.subplot(4, 2, 1)
    time_axis = np.arange(len(voltage_signal)) / 1000  # Convert to seconds
    plt.plot(time_axis, voltage_signal, 'b-', alpha=0.7, linewidth=0.5, label='Voltage Signal')
    if len(spike_times) > 0:
        plt.scatter(spike_times/1000, spike_amplitudes, color='red', s=30, alpha=0.8, label='Detected Spikes')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (mV)')
    plt.title('Raw Voltage Signal with Detected Spikes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Spike amplitude distribution
    ax2 = plt.subplot(4, 2, 2)
    if len(spike_amplitudes) > 0:
        plt.hist(spike_amplitudes, bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(np.mean(spike_amplitudes), color='blue', linestyle='--', label=f'Mean: {np.mean(spike_amplitudes):.3f}')
    plt.xlabel('Spike Amplitude (mV)')
    plt.ylabel('Frequency')
    plt.title('Spike Amplitude Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Inter-spike intervals
    ax3 = plt.subplot(4, 2, 3)
    if len(spike_times) > 1:
        isi = np.diff(spike_times) / 1000  # Convert to seconds
        plt.hist(isi, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(np.mean(isi), color='blue', linestyle='--', label=f'Mean ISI: {np.mean(isi):.3f}s')
        plt.xlabel('Inter-Spike Interval (seconds)')
        plt.ylabel('Frequency')
        plt.title('Inter-Spike Interval Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 4. Spike rate over time
    ax4 = plt.subplot(4, 2, 4)
    if len(spike_times) > 1:
        # Calculate spike rate in sliding windows
        window_size = 5000  # 5 seconds
        time_windows = np.arange(0, len(voltage_signal), window_size)
        spike_rates = []
        
        for window_start in time_windows:
            window_end = min(window_start + window_size, len(voltage_signal))
            spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
            rate = spikes_in_window / (window_size / 1000)  # spikes per second
            spike_rates.append(rate)
        
        plt.plot(time_windows/1000, spike_rates, 'g-', linewidth=2)
        plt.axhline(spike_results['spike_rate_hz'], color='red', linestyle='--', 
                   label=f'Overall Rate: {spike_results["spike_rate_hz"]:.3f} Hz')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Spike Rate (Hz)')
        plt.title('Spike Rate Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 5. Wave transform features
    ax5 = plt.subplot(4, 2, 5)
    if 'all_features' in transform_results and transform_results['all_features']:
        features = transform_results['all_features']
        k_values = [f['k'] for f in features]
        tau_values = [f['tau'] for f in features]
        magnitudes = [f['magnitude'] for f in features]
        
        scatter = plt.scatter(k_values, tau_values, c=magnitudes, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(scatter, label='Magnitude')
        plt.xlabel('k (frequency parameter)')
        plt.ylabel('τ (time scale parameter)')
        plt.title('Wave Transform Features (k vs τ)')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    # 6. Feature magnitude distribution
    ax6 = plt.subplot(4, 2, 6)
    if 'all_features' in transform_results and transform_results['all_features']:
        features = transform_results['all_features']
        magnitudes = [f['magnitude'] for f in features]
        plt.hist(magnitudes, bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(np.mean(magnitudes), color='blue', linestyle='--', 
                   label=f'Mean: {np.mean(magnitudes):.3f}')
        plt.xlabel('Feature Magnitude')
        plt.ylabel('Frequency')
        plt.title('Wave Transform Feature Magnitudes')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 7. Power spectral density
    ax7 = plt.subplot(4, 2, 7)
    frequencies, psd = signal.welch(voltage_signal, fs=1000, nperseg=min(2048, len(voltage_signal)//4))
    plt.semilogy(frequencies, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Power Spectral Density')
    plt.grid(True, alpha=0.3)
    
    # 8. Summary statistics
    ax8 = plt.subplot(4, 2, 8)
    ax8.axis('off')
    
    # Create summary text
    summary_text = f"""
    SUMMARY STATISTICS
    
    SPIKE DETECTION (Adamatzky):
    • Total Spikes: {spike_results['n_spikes']}
    • Mean Amplitude: {spike_results['mean_amplitude']:.3f} mV
    • Spike Rate: {spike_results['spike_rate_hz']:.3f} Hz
    • Mean ISI: {spike_results['mean_isi']:.1f} ms
    
    WAVE TRANSFORM:
    • Total Features: {transform_results['n_features']}
    • Spike-Aligned Features: {transform_results['n_spike_aligned']}
    • Alignment Ratio: {transform_results['spike_alignment_ratio']:.3f}
    
    SYNTHESIS:
    • Biological Activity Score: {synthesis_results['biological_activity_score']:.3f}
    • Method Agreement: {synthesis_results['method_agreement']:.3f}
    • Pattern Complexity: {synthesis_results['pattern_complexity']:.3f}
    • Recommended Analysis: {synthesis_results['recommended_analysis']}
    • Confidence Level: {synthesis_results['confidence_level']}
    """
    
    plt.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the visualization
    output_file = os.path.join(output_dir, f"detailed_analysis_{filename.replace('.csv', '')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")

def create_summary_comparison(results, output_dir):
    """Create a summary comparison across all files"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Summary Comparison Across All Files', fontsize=16, fontweight='bold')
    
    # Extract data for comparison
    filenames = [r['filename'] for r in results]
    spike_counts = [r['spike_results']['n_spikes'] for r in results]
    spike_rates = [r['spike_results']['spike_rate_hz'] for r in results]
    mean_amplitudes = [r['spike_results']['mean_amplitude'] for r in results]
    feature_counts = [r['transform_results']['n_features'] for r in results]
    alignment_ratios = [r['transform_results']['spike_alignment_ratio'] for r in results]
    activity_scores = [r['synthesis_results']['biological_activity_score'] for r in results]
    
    # 1. Spike counts comparison
    axes[0, 0].bar(range(len(filenames)), spike_counts, color='red', alpha=0.7)
    axes[0, 0].set_title('Spike Counts')
    axes[0, 0].set_ylabel('Number of Spikes')
    axes[0, 0].set_xticks(range(len(filenames)))
    axes[0, 0].set_xticklabels([f.split('_')[0] for f in filenames], rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Spike rates comparison
    axes[0, 1].bar(range(len(filenames)), spike_rates, color='blue', alpha=0.7)
    axes[0, 1].set_title('Spike Rates')
    axes[0, 1].set_ylabel('Spike Rate (Hz)')
    axes[0, 1].set_xticks(range(len(filenames)))
    axes[0, 1].set_xticklabels([f.split('_')[0] for f in filenames], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Alignment ratios comparison
    axes[1, 0].bar(range(len(filenames)), alignment_ratios, color='green', alpha=0.7)
    axes[1, 0].set_title('Method Alignment Ratios')
    axes[1, 0].set_ylabel('Alignment Ratio')
    axes[1, 0].set_xticks(range(len(filenames)))
    axes[1, 0].set_xticklabels([f.split('_')[0] for f in filenames], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Activity scores comparison
    axes[1, 1].bar(range(len(filenames)), activity_scores, color='purple', alpha=0.7)
    axes[1, 1].set_title('Biological Activity Scores')
    axes[1, 1].set_ylabel('Activity Score')
    axes[1, 1].set_xticks(range(len(filenames)))
    axes[1, 1].set_xticklabels([f.split('_')[0] for f in filenames], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the comparison
    output_file = os.path.join(output_dir, "summary_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_file}")

def main():
    """Main function to create all visualizations"""
    print("🎨 CREATING DETAILED PATTERN VISUALIZATIONS")
    print("=" * 60)
    
    # Load analysis results
    results_dir = "results/integrated_analysis_results"
    results = load_analysis_results(results_dir)
    
    if not results:
        print("❌ No analysis results found!")
        return
    
    print(f"📊 Found {len(results)} analysis results")
    
    # Create visualizations
    create_comprehensive_visualizations(results)
    
    print("\n🎉 ALL VISUALIZATIONS COMPLETE!")
    print("📁 Check the 'detailed_visualizations' directory for all plots")

if __name__ == "__main__":
    main() 