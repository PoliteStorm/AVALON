#!/usr/bin/env python3
"""
Rigorous fungal electrical analysis with visual progress and clear results.
Compares Adamatzky's spike detection (with correct threshold) and the wave (sqrt t) transform.
Processes all voltage files in the directory, with progress bars and robust data handling.
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm

def load_voltage_data(file_path):
    """Load voltage data from CSV file, robust to headers and time columns."""
    try:
        # Try reading with header and skip time column
        data = pd.read_csv(file_path, header=0)
        # Find the first column with numeric data (skip time column)
        for col in data.columns:
            voltage_signal = pd.to_numeric(data[col], errors='coerce')
            if voltage_signal.notnull().sum() > 0.9 * len(voltage_signal):
                voltage_signal = voltage_signal.dropna().values
                return voltage_signal
        # Fallback: try reading with no header, as before
        data = pd.read_csv(file_path, header=None, skiprows=1)
        voltage_signal = pd.to_numeric(data.iloc[:, 1], errors='coerce').dropna().values
        return voltage_signal
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def adamatzky_spike_detection(voltage_signal, w=20, delta=0.01):
    """Adamatzky's moving average threshold spike detection."""
    spikes = []
    n_samples = len(voltage_signal)
    for i in range(w, n_samples - w):
        avg = np.mean(voltage_signal[i-w:i+w])
        if abs(voltage_signal[i]) - abs(avg) > delta:
            spikes.append(i)
    return spikes

def wave_transform_features(voltage_signal):
    """Current wave (sqrt t) transform method, optimized for speed."""
    # Reduce parameter space for faster computation
    k_values = np.logspace(-2, 1, 15)  # Reduced from 30 to 15
    tau_values = np.logspace(-1, 3, 15)  # Reduced from 30 to 15
    t = np.arange(len(voltage_signal))
    features = []
    
    # Precompute sqrt_t once
    sqrt_t = np.sqrt(t)
    
    for k in k_values:
        for tau in tau_values:
            window = np.exp(-(sqrt_t / tau)**2)
            phase = np.exp(-1j * k * sqrt_t)
            integrand = voltage_signal * window * phase
            # Use trapezoid instead of deprecated trapz
            magnitude = np.abs(np.trapezoid(integrand, t))
            if magnitude > np.mean(voltage_signal) * 0.1:
                features.append({
                    'k': k,
                    'tau': tau,
                    'magnitude': magnitude,
                    'frequency': k / (2 * np.pi),
                    'time_scale': tau
                })
    return features

def main():
    print("=== Rigorous Fungal Electrical Analysis ===")
    print("Comparing Adamatzky's spike detection (with correct threshold) and the wave (sqrt t) transform.")
    print("All voltage files in the directory will be processed.")
    print()

    voltage_dir = "../15061491/fungal_spikes/good_recordings"
    results_dir = "results/rigorous_analysis_results"
    os.makedirs(results_dir, exist_ok=True)

    files = [f for f in os.listdir(voltage_dir) if f.endswith('.csv')]
    if not files:
        print(f"No voltage CSV files found in {voltage_dir}")
        return

    all_results = []
    print(f"Found {len(files)} voltage files. Starting analysis...\n")

    for file_idx, filename in enumerate(tqdm(files, desc="Files", unit="file")):
        voltage_file = os.path.join(voltage_dir, filename)
        voltage_signal = load_voltage_data(voltage_file)
        if voltage_signal is None or len(voltage_signal) < 100:
            print(f"[SKIP] {filename}: Could not load valid voltage data.")
            continue
        print(f"\n=== File {file_idx+1}/{len(files)}: {filename} ===")
        print(f"Loaded {len(voltage_signal)} voltage samples.")
        print(f"Voltage range: {np.min(voltage_signal):.6f} to {np.max(voltage_signal):.6f} mV")
        print(f"Mean voltage: {np.mean(voltage_signal):.6f} mV")
        print(f"Std voltage: {np.std(voltage_signal):.6f} mV")

        # Adamatzky's method with correct threshold
        noise_std = np.std(voltage_signal)
        delta = 2 * noise_std
        print(f"--- Adamatzky's Spike Detection ---")
        print(f"Spike detection threshold (delta): {delta:.6f} mV (2x noise std)")
        spikes = []
        for i in tqdm(range(20, len(voltage_signal)-20), desc="Adamatzky", leave=False):
            avg = np.mean(voltage_signal[i-20:i+20])
            if abs(voltage_signal[i]) - abs(avg) > delta:
                spikes.append(i)
        spike_times = np.array(spikes)
        if len(spike_times) > 1:
            isi = np.diff(spike_times)
            mean_isi = np.mean(isi)
            spike_rate = len(spike_times) / len(voltage_signal)
            # Convert to more meaningful units
            spike_rate_per_1000 = spike_rate * 1000
            spike_rate_per_second = spike_rate * 1000  # Assuming 1kHz sampling rate
        else:
            mean_isi = 0
            spike_rate = 0
            spike_rate_per_1000 = 0
            spike_rate_per_second = 0
        mean_amplitude = np.mean(voltage_signal[spike_times]) if len(spike_times) > 0 else 0
        print(f"Spikes detected: {len(spike_times)}")
        print(f"Mean amplitude: {mean_amplitude:.6f} mV")
        print(f"Mean ISI: {mean_isi:.2f} samples")
        print(f"Spike rate: {spike_rate:.6f} per sample ({spike_rate_per_1000:.3f} per 1000 samples, ~{spike_rate_per_second:.3f} Hz)")

        # Wave (sqrt t) transform method with progress
        print("\n--- Wave (sqrt t) Transform Method ---")
        features = []
        k_values = np.logspace(-2, 1, 15)  # Reduced from 30 to 15
        tau_values = np.logspace(-1, 3, 15)  # Reduced from 30 to 15
        t = np.arange(len(voltage_signal))
        sqrt_t = np.sqrt(t)  # Precompute once
        
        for k in tqdm(k_values, desc="Wave k", leave=False):
            for tau in tau_values:
                window = np.exp(-(sqrt_t / tau)**2)
                phase = np.exp(-1j * k * sqrt_t)
                integrand = voltage_signal * window * phase
                # Use trapezoid instead of deprecated trapz
                magnitude = np.abs(np.trapezoid(integrand, t))
                if magnitude > np.mean(voltage_signal) * 0.1:
                    features.append({
                        'k': k,
                        'tau': tau,
                        'magnitude': magnitude,
                        'frequency': k / (2 * np.pi),
                        'time_scale': tau
                    })
        if features:
            mean_magnitude = np.mean([f['magnitude'] for f in features])
            mean_frequency = np.mean([f['frequency'] for f in features])
            mean_time_scale = np.mean([f['time_scale'] for f in features])
        else:
            mean_magnitude = mean_frequency = mean_time_scale = 0
        print(f"Features detected: {len(features)}")
        print(f"Mean magnitude: {mean_magnitude:.6f}")
        print(f"Mean frequency: {mean_frequency:.6f} Hz")
        print(f"Mean time scale: {mean_time_scale:.2f} seconds")

        # Save results for this file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"rigorous_electrical_analysis_{os.path.splitext(filename)[0]}_{timestamp}.json")
        results = {
            'test_timestamp': timestamp,
            'voltage_file': voltage_file,
            'n_samples': len(voltage_signal),
            'voltage_stats': {
                'min': float(np.min(voltage_signal)),
                'max': float(np.max(voltage_signal)),
                'mean': float(np.mean(voltage_signal)),
                'std': float(np.std(voltage_signal))
            },
            'adamatzky_results': {
                'n_spikes': int(len(spike_times)),
                'mean_amplitude': float(mean_amplitude),
                'mean_isi': float(mean_isi),
                'spike_rate': float(spike_rate),
                'spike_rate_per_1000': float(spike_rate_per_1000),
                'spike_rate_hz': float(spike_rate_per_second),
                'threshold_delta': float(delta)
            },
            'wave_transform_results': {
                'n_features': int(len(features)),
                'mean_magnitude': float(mean_magnitude),
                'mean_frequency': float(mean_frequency),
                'mean_time_scale': float(mean_time_scale)
            }
        }
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        all_results.append(results)
        print(f"\nResults saved to: {results_file}\n")

    print("\n=== All files processed. Summary ===")
    for idx, res in enumerate(all_results):
        print(f"File {idx+1}: {os.path.basename(res['voltage_file'])}")
        print(f"  Adamatzky: {res['adamatzky_results']['n_spikes']} spikes, threshold {res['adamatzky_results']['threshold_delta']:.6g}")
        print(f"  Wave (sqrt t): {res['wave_transform_results']['n_features']} features")
    print("\nAll results saved to:", results_dir)
    print("=== Analysis Complete ===")

if __name__ == "__main__":
    main() 