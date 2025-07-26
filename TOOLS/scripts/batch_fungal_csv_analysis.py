#!/usr/bin/env python3
"""
Batch Fungal CSV Analysis
Scans all CSVs in csv_data/ and fungal_analysis_project/data/, analyzes each using the ultra-optimized Adamatzky + wave transform script, and outputs a ranked summary of the best files.
"""

import os
import sys
import numpy as np
import pandas as pd
import traceback
from ultra_optimized_fungal_monitoring_simple import UltraOptimizedFungalMonitor

# Directories to scan
CSV_DIRS = [
    'csv_data',
    os.path.join('fungal_analysis_project', 'data')
]

# Acceptable column names for voltage data
VOLTAGE_COLUMNS = ['voltage', 'signal', 'amplitude', 'data', 'V', 'mv']

# Output summary file
SUMMARY_CSV = 'batch_fungal_csv_analysis_summary.csv'
SUMMARY_JSON = 'batch_fungal_csv_analysis_summary.json'


def find_csv_files(directories):
    """Recursively find all CSV files in the given directories."""
    csv_files = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
    return csv_files


def load_voltage_data(csv_path):
    """Load voltage data from a CSV file, trying common column names or the first column."""
    try:
        df = pd.read_csv(csv_path)
        for col in VOLTAGE_COLUMNS:
            if col in df.columns:
                return df[col].values
        # Fallback: use the first column
        return df.iloc[:, 0].values
    except Exception as e:
        print(f"[WARN] Could not load {csv_path}: {e}")
        return None


def analyze_file(csv_path, monitor):
    """Analyze a single CSV file and return its metrics or None if failed."""
    voltage_data = load_voltage_data(csv_path)
    if voltage_data is None or len(voltage_data) < 100:
        return None
    try:
        results = monitor.analyze_recording_ultra_optimized(voltage_data)
        stats = results['stats']
        wave_features = results['wave_features']
        return {
            'file': csv_path,
            'quality_score': stats['quality_score'],
            'snr': stats['snr'],
            'n_spikes': stats['n_spikes'],
            'spike_rate': stats['spike_rate'],
            'mean_amplitude': stats['mean_amplitude'],
            'mean_isi': stats['mean_isi'],
            'wave_patterns': wave_features['wave_patterns'],
            'wave_confidence': wave_features['confidence']
        }
    except Exception as e:
        print(f"[ERROR] Analysis failed for {csv_path}: {e}")
        traceback.print_exc()
        return None


def main():
    print("=== Batch Fungal CSV Analysis ===")
    csv_files = find_csv_files(CSV_DIRS)
    print(f"Found {len(csv_files)} CSV files to analyze.")

    monitor = UltraOptimizedFungalMonitor()
    monitor.get_species_parameters('pleurotus')

    results = []
    for i, csv_path in enumerate(csv_files):
        print(f"[{i+1}/{len(csv_files)}] Analyzing: {csv_path}")
        metrics = analyze_file(csv_path, monitor)
        if metrics:
            results.append(metrics)

    if not results:
        print("No valid results found.")
        return

    # Sort by quality score (descending)
    results.sort(key=lambda x: x['quality_score'], reverse=True)

    # Save summary CSV
    df = pd.DataFrame(results)
    df.to_csv(SUMMARY_CSV, index=False)
    df.to_json(SUMMARY_JSON, orient='records', indent=2)
    print(f"\nTop 10 files by quality score:")
    print(df.head(10).to_string(index=False))
    print(f"\nFull summary saved to {SUMMARY_CSV} and {SUMMARY_JSON}")

if __name__ == "__main__":
    main() 