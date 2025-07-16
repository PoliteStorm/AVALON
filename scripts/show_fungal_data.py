#!/usr/bin/env python3
"""
Show Fungal Data
Display the raw electrical data from the best files with detailed analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultra_optimized_fungal_monitoring_simple import UltraOptimizedFungalMonitor
import json

# Best files from our analysis
BEST_FILES = [
    "fungal_analysis_project/data/raw/15061491/Norm_vs_deep_tip_crop.csv",
    "fungal_analysis_project/data/raw/15061491/Ch1-2_moisture_added.csv"
]

def load_and_display_data(file_path):
    """Load and display detailed data from a CSV file"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {file_path}")
    print(f"{'='*60}")
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        print(f"\n📊 FILE INFO:")
        print(f"- Shape: {df.shape}")
        print(f"- Columns: {list(df.columns)}")
        print(f"- Data types: {df.dtypes.to_dict()}")
        
        # Get voltage data
        voltage_columns = ['voltage', 'signal', 'amplitude', 'data', 'V', 'mv']
        voltage_col = None
        for col in voltage_columns:
            if col in df.columns:
                voltage_col = col
                break
        
        if voltage_col is None:
            voltage_col = df.columns[0]
        
        voltage_data = df[voltage_col].values
        
        print(f"\n⚡ VOLTAGE DATA:")
        print(f"- Column used: '{voltage_col}'")
        print(f"- Number of samples: {len(voltage_data):,}")
        print(f"- Min voltage: {voltage_data.min():.6f} V")
        print(f"- Max voltage: {voltage_data.max():.6f} V")
        print(f"- Mean voltage: {voltage_data.mean():.6f} V")
        print(f"- Std deviation: {voltage_data.std():.6f} V")
        print(f"- Recording duration: {len(voltage_data)/1000:.1f} seconds")
        
        # Show first 20 values
        print(f"\n📈 FIRST 20 SAMPLES:")
        for i, val in enumerate(voltage_data[:20]):
            print(f"  Sample {i+1:2d}: {val:8.6f} V")
        
        # Analyze with our method
        monitor = UltraOptimizedFungalMonitor()
        monitor.get_species_parameters('pleurotus')
        
        results = monitor.analyze_recording_ultra_optimized(voltage_data)
        
        print(f"\n🔍 ANALYSIS RESULTS:")
        stats = results['stats']
        wave_features = results['wave_features']
        
        print(f"- Quality Score: {stats['quality_score']:.3f}")
        print(f"- Signal-to-Noise Ratio: {stats['snr']:.2f}")
        print(f"- Spikes Detected: {stats['n_spikes']}")
        print(f"- Spike Rate: {stats['spike_rate']:.3f} Hz")
        print(f"- Mean Spike Amplitude: {stats['mean_amplitude']:.6f} V")
        print(f"- Mean Inter-Spike Interval: {stats['mean_isi']:.3f} s")
        print(f"- Wave Patterns Detected: {wave_features['wave_patterns']}")
        print(f"- Wave Confidence: {wave_features['confidence']:.3f}")
        
        # Show spike details
        if results['spikes']:
            print(f"\n⚡ SPIKE DETAILS:")
            for i, spike in enumerate(results['spikes'][:10]):  # Show first 10 spikes
                print(f"  Spike {i+1:2d}: Time={spike['time_seconds']:6.3f}s, "
                      f"Amplitude={spike['amplitude_mv']:8.6f}V, "
                      f"Index={spike['index']}")
            if len(results['spikes']) > 10:
                print(f"  ... and {len(results['spikes']) - 10} more spikes")
        
        # Show wave pattern details
        if results['wave_patterns']:
            print(f"\n🌊 WAVE PATTERN DETAILS:")
            for i, pattern in enumerate(results['wave_patterns'][:5]):  # Show first 5 patterns
                print(f"  Pattern {i+1}: Scale={pattern['scale']:6.3f}, "
                      f"Shift={pattern['shift']:6.3f}, "
                      f"Strength={pattern['strength']:6.3f}")
            if len(results['wave_patterns']) > 5:
                print(f"  ... and {len(results['wave_patterns']) - 5} more patterns")
        
        # Create detailed visualization
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle(f'Detailed Analysis: {file_path.split("/")[-1]}', fontsize=16)
        
        # Plot 1: Full signal overview
        time_axis = np.arange(len(voltage_data)) / 1000  # Convert to seconds
        axes[0].plot(time_axis, voltage_data, linewidth=0.5, alpha=0.7)
        axes[0].set_title('Full Electrical Signal')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Voltage (V)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Zoomed section with spikes
        if results['spikes']:
            # Show a 10-second window around the middle
            mid_point = len(voltage_data) // 2
            start_idx = max(0, mid_point - 5000)
            end_idx = min(len(voltage_data), mid_point + 5000)
            
            zoom_time = time_axis[start_idx:end_idx]
            zoom_data = voltage_data[start_idx:end_idx]
            
            axes[1].plot(zoom_time, zoom_data, linewidth=0.8, alpha=0.8)
            
            # Mark spikes in this window
            spike_times = [spike['time_seconds'] for spike in results['spikes']]
            spike_amplitudes = [spike['amplitude_mv'] for spike in results['spikes']]
            
            # Filter spikes in the zoom window
            zoom_spikes = []
            for spike in results['spikes']:
                if start_idx/1000 <= spike['time_seconds'] <= end_idx/1000:
                    zoom_spikes.append(spike)
            
            if zoom_spikes:
                spike_x = [spike['time_seconds'] for spike in zoom_spikes]
                spike_y = [voltage_data[spike['index']] for spike in zoom_spikes]
                axes[1].scatter(spike_x, spike_y, color='red', s=30, zorder=5, 
                               label=f'{len(zoom_spikes)} spikes in window')
                axes[1].legend()
            
            axes[1].set_title('Zoomed Section (10 seconds) with Detected Spikes')
            axes[1].set_xlabel('Time (seconds)')
            axes[1].set_ylabel('Voltage (V)')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No spikes detected', ha='center', va='center',
                        transform=axes[1].transAxes, fontsize=14)
            axes[1].set_title('No Spikes Detected')
        
        # Plot 3: Spike amplitude distribution
        if results['spikes']:
            amplitudes = [spike['amplitude_mv'] for spike in results['spikes']]
            axes[2].hist(amplitudes, bins=15, alpha=0.7, edgecolor='black')
            axes[2].set_title('Spike Amplitude Distribution')
            axes[2].set_xlabel('Amplitude (V)')
            axes[2].set_ylabel('Count')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No spikes for histogram', ha='center', va='center',
                        transform=axes[2].transAxes, fontsize=14)
            axes[2].set_title('No Spikes for Histogram')
        
        plt.tight_layout()
        plt.show()
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing {file_path}: {e}")
        return None

def main():
    """Display detailed data from the best files"""
    print("🍄 FUNGAL ELECTRICAL ACTIVITY DATA ANALYSIS")
    print("=" * 60)
    
    all_results = []
    
    for file_path in BEST_FILES:
        results = load_and_display_data(file_path)
        if results:
            all_results.append({
                'file': file_path,
                'results': results
            })
    
    # Summary comparison
    if all_results:
        print(f"\n{'='*60}")
        print("📊 COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        for i, result in enumerate(all_results):
            stats = result['results']['stats']
            wave_features = result['results']['wave_features']
            
            print(f"\n📁 File {i+1}: {result['file'].split('/')[-1]}")
            print(f"   Quality: {stats['quality_score']:.3f} | "
                  f"Spikes: {stats['n_spikes']} | "
                  f"Wave Patterns: {wave_features['wave_patterns']} | "
                  f"SNR: {stats['snr']:.2f}")

if __name__ == "__main__":
    main() 