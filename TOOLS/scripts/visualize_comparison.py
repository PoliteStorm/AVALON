#!/usr/bin/env python3
"""
Visualize the comparison between Adamatzky's method and current code's method.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

def load_comparison_results(results_file):
    """Load the comparison results."""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_comparison_visualization(results, output_file="algorithm_comparison_visualization.png"):
    """
    Create a comprehensive visualization of the algorithm comparison.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Algorithm Comparison: Adamatzky vs Current Code', fontsize=16, fontweight='bold')
    
    # 1. Spike/Feature Count Comparison
    ax1 = axes[0, 0]
    methods = ['Adamatzky\nSpike Detection', 'Current Code\n√t Transform']
    counts = [results['adamatzky_results']['n_spikes'], results['current_results']['n_features']]
    colors = ['#2E86AB', '#A23B72']
    
    bars = ax1.bar(methods, counts, color=colors, alpha=0.7)
    ax1.set_title('Detection Count Comparison')
    ax1.set_ylabel('Number of Detections')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Adamatzky Spike Classifications
    ax2 = axes[0, 1]
    classifications = results['adamatzky_results']['classifications']
    spike_types = list(classifications.keys())
    spike_counts = list(classifications.values())
    
    colors_spike = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax2.bar(spike_types, spike_counts, color=colors_spike, alpha=0.7)
    ax2.set_title('Adamatzky Spike Classifications')
    ax2.set_ylabel('Number of Spikes')
    
    # Add value labels
    for bar, count in zip(bars, spike_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(spike_counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Amplitude Comparison
    ax3 = axes[0, 2]
    adamatzky_amp = results['adamatzky_results']['mean_amplitude']
    current_mag = results['current_results']['mean_magnitude']
    
    # Normalize current magnitude for comparison (log scale)
    comparison_data = [adamatzky_amp, current_mag/1000]  # Scale down for visualization
    labels = ['Adamatzky\nMean Amplitude (mV)', 'Current Code\nMean Magnitude/1000']
    colors_amp = ['#2E86AB', '#A23B72']
    
    bars = ax3.bar(labels, comparison_data, color=colors_amp, alpha=0.7)
    ax3.set_title('Amplitude/Magnitude Comparison')
    ax3.set_ylabel('Amplitude (mV) / Magnitude (scaled)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, comparison_data):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(comparison_data)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Time Scale Comparison
    ax4 = axes[1, 0]
    adamatzky_isi = results['adamatzky_results']['mean_isi']
    current_timescale = results['current_results']['mean_time_scale']
    
    time_data = [adamatzky_isi, current_timescale]
    time_labels = ['Adamatzky\nMean ISI (s)', 'Current Code\nMean Time Scale (s)']
    colors_time = ['#2E86AB', '#A23B72']
    
    bars = ax4.bar(time_labels, time_data, color=colors_time, alpha=0.7)
    ax4.set_title('Time Scale Comparison')
    ax4.set_ylabel('Time (seconds)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, time_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(time_data)*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Frequency Analysis
    ax5 = axes[1, 1]
    current_freq = results['current_results']['mean_frequency']
    
    # Adamatzky doesn't directly measure frequency, but we can calculate from ISI
    if adamatzky_isi > 0:
        adamatzky_freq = 1.0 / adamatzky_isi
    else:
        adamatzky_freq = 0
    
    freq_data = [adamatzky_freq, current_freq]
    freq_labels = ['Adamatzky\nDerived Freq (Hz)', 'Current Code\nMean Freq (Hz)']
    colors_freq = ['#2E86AB', '#A23B72']
    
    bars = ax5.bar(freq_labels, freq_data, color=colors_freq, alpha=0.7)
    ax5.set_title('Frequency Comparison')
    ax5.set_ylabel('Frequency (Hz)')
    ax5.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, freq_data):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + max(freq_data)*0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Summary Statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""
ALGORITHM COMPARISON SUMMARY

ADAMATZKY'S METHOD:
• Detected {results['adamatzky_results']['n_spikes']:,} electrical spikes
• Mean amplitude: {results['adamatzky_results']['mean_amplitude']:.3f} mV
• Mean ISI: {results['adamatzky_results']['mean_isi']:.2f} seconds
• Spike types: {results['adamatzky_results']['classifications']}

CURRENT CODE'S METHOD:
• Detected {results['current_results']['n_features']} mathematical features
• Mean magnitude: {results['current_results']['mean_magnitude']:.1f}
• Mean frequency: {results['current_results']['mean_frequency']:.4f} Hz
• Mean time scale: {results['current_results']['mean_time_scale']:.1f} seconds

KEY DIFFERENCES:
• Different detection algorithms
• Different units and scales
• Different biological interpretations
• Fundamentally different analysis types
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()

def main():
    """Main function to create the visualization."""
    # Find the most recent results file
    import glob
    results_files = glob.glob("algorithm_comparison_results_*.json")
    
    if not results_files:
        print("No comparison results found. Please run algorithm_comparison.py first.")
        return
    
    # Use the most recent file
    latest_file = max(results_files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")
    
    # Load and visualize results
    results = load_comparison_results(latest_file)
    create_comparison_visualization(results)

if __name__ == "__main__":
    main() 