#!/usr/bin/env python3
"""
Integrated Adamatzky-Transform Analysis
Combines Adamatzky's spike detection methodologies with the wave (sqrt t) transform
for comprehensive fungal electrical analysis.
"""

import sys
import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
from scipy import signal, stats
from sklearn.cluster import DBSCAN

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

def adamatzky_spike_detection_enhanced(voltage_signal, w=20, delta=None, adaptive_threshold=True):
    """
    Enhanced Adamatzky spike detection with multiple methodologies.
    
    Args:
        voltage_signal: Input voltage signal
        w: Window size for moving average
        delta: Threshold (if None, calculated adaptively)
        adaptive_threshold: Whether to use adaptive thresholding
    
    Returns:
        dict: Comprehensive spike detection results
    """
    n_samples = len(voltage_signal)
    
    # Method 1: Original Adamatzky (moving average threshold)
    if delta is None:
        if adaptive_threshold:
            # Adaptive threshold based on noise characteristics
            noise_std = np.std(voltage_signal)
            delta = 2 * noise_std  # 2x noise standard deviation
        else:
            delta = 0.01  # Fixed threshold
    
    spikes_original = []
    for i in range(w, n_samples - w):
        avg = np.mean(voltage_signal[i-w:i+w])
        if abs(voltage_signal[i]) - abs(avg) > delta:
            spikes_original.append(i)
    
    # Method 2: Peak detection with prominence
    peaks, properties = signal.find_peaks(np.abs(voltage_signal), 
                                        prominence=delta/2,
                                        distance=w)
    
    # Method 3: Statistical thresholding
    z_scores = np.abs((voltage_signal - np.mean(voltage_signal)) / np.std(voltage_signal))
    spikes_statistical = np.where(z_scores > 2.5)[0]  # 2.5 sigma threshold
    
    # Method 4: Wavelet-based spike detection
    # Simple approximation using bandpass filter
    nyquist = 1000 / 2  # Assuming 1kHz sampling
    low = 0.1 / nyquist
    high = 10.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, voltage_signal)
    
    # Find peaks in filtered signal
    peaks_filtered, _ = signal.find_peaks(np.abs(filtered_signal), 
                                         height=delta/2,
                                         distance=w)
    
    # Combine and validate spikes
    all_spike_candidates = set(spikes_original + list(peaks) + list(spikes_statistical) + list(peaks_filtered))
    
    # Remove duplicates and validate
    validated_spikes = []
    for spike_idx in sorted(all_spike_candidates):
        if w <= spike_idx < n_samples - w:
            # Check if it's a local maximum
            local_max = True
            for offset in range(-w//2, w//2 + 1):
                if (0 <= spike_idx + offset < n_samples and 
                    abs(voltage_signal[spike_idx + offset]) > abs(voltage_signal[spike_idx])):
                    local_max = False
                    break
            if local_max:
                validated_spikes.append(spike_idx)
    
    # Calculate spike statistics
    spike_times = np.array(validated_spikes)
    spike_amplitudes = voltage_signal[spike_times] if len(spike_times) > 0 else np.array([])
    
    if len(spike_times) > 1:
        isi = np.diff(spike_times)
        mean_isi = np.mean(isi)
        spike_rate = len(spike_times) / len(voltage_signal)
        spike_rate_per_1000 = spike_rate * 1000
        spike_rate_hz = spike_rate * 1000  # Assuming 1kHz sampling
    else:
        mean_isi = 0
        spike_rate = 0
        spike_rate_per_1000 = 0
        spike_rate_hz = 0
    
    return {
        'spike_times': spike_times,
        'spike_amplitudes': spike_amplitudes,
        'n_spikes': len(spike_times),
        'mean_amplitude': float(np.mean(spike_amplitudes)) if len(spike_amplitudes) > 0 else 0.0,
        'mean_isi': float(mean_isi),
        'spike_rate': float(spike_rate),
        'spike_rate_per_1000': float(spike_rate_per_1000),
        'spike_rate_hz': float(spike_rate_hz),
        'threshold_delta': float(delta),
        'methods': {
            'original_adamatzky': len(spikes_original),
            'peak_detection': len(peaks),
            'statistical': len(spikes_statistical),
            'wavelet_filtered': len(peaks_filtered),
            'validated': len(validated_spikes)
        }
    }

def wave_transform_with_spike_guidance(voltage_signal, spike_times, k_values=None, tau_values=None):
    """
    Wave transform with guidance from Adamatzky spike detection.
    
    Args:
        voltage_signal: Input voltage signal
        spike_times: Spike times from Adamatzky detection
        k_values: Frequency parameters (if None, auto-generated)
        tau_values: Time scale parameters (if None, auto-generated)
    
    Returns:
        dict: Wave transform results with spike-guided analysis
    """
    if k_values is None:
        k_values = np.logspace(-2, 1, 10)  # Reduced from 15 to 10
    if tau_values is None:
        tau_values = np.logspace(-1, 3, 10)  # Reduced from 15 to 10
    
    t = np.arange(len(voltage_signal))
    sqrt_t = np.sqrt(t)
    
    # Standard wave transform
    features = []
    spike_guided_features = []
    
    # Add progress tracking
    total_combinations = len(k_values) * len(tau_values)
    combination_count = 0
    
    # Pre-calculate spike density for efficiency (only if we have spikes)
    spike_density_cache = {}
    if len(spike_times) > 0:
        print(f"  Pre-calculating spike density for {len(spike_times)} spikes...")
        # Sample tau values for density calculation to avoid excessive computation
        sample_taus = np.logspace(-1, 3, 5)  # Only 5 sample taus for density calculation
        for tau in sample_taus:
            if tau > 0:
                spike_density_cache[tau] = calculate_spike_density_around_feature(
                    spike_times, tau, len(voltage_signal)
                )
    
    for k in k_values:
        for tau in tau_values:
            combination_count += 1
            if combination_count % 25 == 0:  # More frequent progress updates
                print(f"  Progress: {combination_count}/{total_combinations} combinations processed")
            
            # Safety check for tau
            if tau <= 0:
                continue
                
            window = np.exp(-(sqrt_t / tau)**2)
            phase = np.exp(-1j * k * sqrt_t)
            integrand = voltage_signal * window * phase
            magnitude = np.abs(np.trapezoid(integrand, t))
            
            if magnitude > np.mean(voltage_signal) * 0.1:
                feature = {
                    'k': k,
                    'tau': tau,
                    'magnitude': magnitude,
                    'frequency': k / (2 * np.pi),
                    'time_scale': tau
                }
                features.append(feature)
                
                # Spike-guided analysis: Use cached density or estimate
                if len(spike_times) > 0:
                    try:
                        # Use cached density if available, otherwise estimate
                        if tau in spike_density_cache:
                            spike_density = spike_density_cache[tau]
                        else:
                            # Estimate based on nearest cached value
                            cached_taus = list(spike_density_cache.keys())
                            if cached_taus:
                                nearest_tau = min(cached_taus, key=lambda x: abs(x - tau))
                                spike_density = spike_density_cache[nearest_tau]
                            else:
                                spike_density = 0.0
                        
                        # Enhanced feature with spike correlation
                        feature['spike_correlation'] = spike_density
                        feature['spike_aligned'] = spike_density > 0.1  # Threshold for alignment
                        
                        if feature['spike_aligned']:
                            spike_guided_features.append(feature)
                    except Exception as e:
                        # If spike density calculation fails, continue without spike guidance
                        feature['spike_correlation'] = 0.0
                        feature['spike_aligned'] = False
                        print(f"Warning: Spike density calculation failed for tau={tau}: {e}")
    
    return {
        'all_features': features,
        'spike_guided_features': spike_guided_features,
        'n_features': len(features),
        'n_spike_aligned': len(spike_guided_features),
        'mean_magnitude': np.mean([f['magnitude'] for f in features]) if features else 0,
        'mean_frequency': np.mean([f['frequency'] for f in features]) if features else 0,
        'mean_time_scale': np.mean([f['time_scale'] for f in features]) if features else 0,
        'spike_alignment_ratio': len(spike_guided_features) / len(features) if features else 0
    }

def calculate_spike_density_around_feature(spike_times, tau, signal_length):
    """Calculate spike density around a feature's time scale."""
    if len(spike_times) == 0:
        return 0.0
    
    # Ensure tau is at least 1 to avoid division by zero
    tau = max(tau, 1.0)
    
    # For very large signals, sample to improve performance
    if signal_length > 100000:  # If signal is very long
        # Sample every 1000th point for efficiency
        sample_indices = np.arange(0, signal_length, 1000)
        signal_length_sampled = len(sample_indices)
        step_size = max(1, int(tau / 1000))  # Adjust step size for sampling
    else:
        signal_length_sampled = signal_length
        step_size = max(1, int(tau))
    
    # Count spikes within time windows around the feature's scale
    density = 0
    n_windows = 0
    
    # Limit the number of windows to prevent excessive computation
    max_windows = 1000
    window_count = 0
    
    for i in range(0, signal_length_sampled, step_size):
        if window_count >= max_windows:  # Safety limit
            break
            
        window_start = max(0, i - int(tau/2))
        window_end = min(signal_length, i + int(tau/2))
        
        # Count spikes in this window using efficient numpy operations
        spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
        density += spikes_in_window
        n_windows += 1
        window_count += 1
    
    return density / max(n_windows, 1)

def integrated_analysis(voltage_signal, filename):
    """
    Integrated analysis combining Adamatzky and wave transform methods.
    """
    print(f"\n=== Integrated Analysis: {filename} ===")
    print(f"Signal length: {len(voltage_signal)} samples")
    print(f"Voltage range: {np.min(voltage_signal):.6f} to {np.max(voltage_signal):.6f} mV")
    
    # Step 1: Enhanced Adamatzky spike detection
    print("\n--- Step 1: Enhanced Adamatzky Spike Detection ---")
    spike_results = adamatzky_spike_detection_enhanced(voltage_signal)
    
    print(f"Spikes detected: {spike_results['n_spikes']}")
    print(f"Mean amplitude: {spike_results['mean_amplitude']:.6f} mV")
    print(f"Spike rate: {spike_results['spike_rate_hz']:.6f} Hz")
    print(f"Threshold: {spike_results['threshold_delta']:.6f} mV")
    
    # Step 2: Wave transform with spike guidance
    print("\n--- Step 2: Wave Transform with Spike Guidance ---")
    
    # Add timeout warning for large datasets
    if len(voltage_signal) > 500000:  # Very large signal
        print(f"  Warning: Large signal detected ({len(voltage_signal)} samples)")
        print(f"  Analysis may take several minutes. Press Ctrl+C to stop if needed.")
    
    try:
        transform_results = wave_transform_with_spike_guidance(
            voltage_signal, spike_results['spike_times']
        )
        
        print(f"Total features: {transform_results['n_features']}")
        print(f"Spike-aligned features: {transform_results['n_spike_aligned']}")
        print(f"Alignment ratio: {transform_results['spike_alignment_ratio']:.3f}")
        
    except KeyboardInterrupt:
        print("\n  Analysis interrupted by user. Using partial results...")
        # Return partial results if interrupted
        transform_results = {
            'all_features': [],
            'spike_guided_features': [],
            'n_features': 0,
            'n_spike_aligned': 0,
            'mean_magnitude': 0,
            'mean_frequency': 0,
            'mean_time_scale': 0,
            'spike_alignment_ratio': 0
        }
    
    # Step 3: Cross-validation and synthesis
    print("\n--- Step 3: Cross-Validation and Synthesis ---")
    synthesis_results = synthesize_methods(spike_results, transform_results)
    
    return {
        'filename': filename,
        'spike_results': spike_results,
        'transform_results': transform_results,
        'synthesis_results': synthesis_results,
        'voltage_stats': {
            'min': float(np.min(voltage_signal)),
            'max': float(np.max(voltage_signal)),
            'mean': float(np.mean(voltage_signal)),
            'std': float(np.std(voltage_signal))
        }
    }

def synthesize_methods(spike_results, transform_results):
    """
    Synthesize results from both methods for comprehensive analysis.
    """
    synthesis = {
        'biological_activity_score': 0.0,
        'pattern_complexity': 0.0,
        'method_agreement': 0.0,
        'recommended_analysis': 'unknown',
        'confidence_level': 'low'
    }
    
    # Calculate biological activity score
    spike_score = min(spike_results['spike_rate_hz'] * 10, 1.0)  # Normalize to 0-1
    transform_score = min(transform_results['n_features'] / 100, 1.0)  # Normalize to 0-1
    
    synthesis['biological_activity_score'] = (spike_score + transform_score) / 2
    
    # Calculate pattern complexity
    if transform_results['n_features'] > 0:
        complexity = transform_results['n_features'] * transform_results['spike_alignment_ratio']
        synthesis['pattern_complexity'] = min(complexity / 50, 1.0)
    
    # Calculate method agreement
    if spike_results['n_spikes'] > 0 and transform_results['n_features'] > 0:
        agreement = transform_results['spike_alignment_ratio']
        synthesis['method_agreement'] = agreement
    
    # Determine recommended analysis approach
    if synthesis['biological_activity_score'] > 0.7:
        synthesis['recommended_analysis'] = 'high_activity'
        synthesis['confidence_level'] = 'high'
    elif synthesis['biological_activity_score'] > 0.3:
        synthesis['recommended_analysis'] = 'moderate_activity'
        synthesis['confidence_level'] = 'medium'
    else:
        synthesis['recommended_analysis'] = 'low_activity'
        synthesis['confidence_level'] = 'low'
    
    return synthesis

def main():
    import sys
    
    print("=== Integrated Adamatzky-Transform Analysis ===")
    print("Combining Adamatzky's spike detection with wave transform analysis.")
    print()
    
    # Accept directory path as command line argument, or use default
    if len(sys.argv) > 1:
        voltage_dir = sys.argv[1]
        print(f"Using directory: {voltage_dir}")
    else:
        voltage_dir = "../15061491/fungal_spikes/good_recordings"
        print(f"Using default directory: {voltage_dir}")
    
    # Check if directory exists
    if not os.path.exists(voltage_dir):
        print(f"Error: Directory '{voltage_dir}' does not exist.")
        print("Usage: python integrated_adamatzky_transform_analysis.py [directory_path]")
        return
    
    results_dir = "results/integrated_analysis_results"
    os.makedirs(results_dir, exist_ok=True)
    
    files = [f for f in os.listdir(voltage_dir) if f.endswith('.csv')]
    if not files:
        print(f"No voltage CSV files found in {voltage_dir}")
        return
    
    all_results = []
    print(f"Found {len(files)} voltage files. Starting integrated analysis...\n")
    
    for file_idx, filename in enumerate(tqdm(files, desc="Files", unit="file")):
        voltage_file = os.path.join(voltage_dir, filename)
        voltage_signal = load_voltage_data(voltage_file)
        
        if voltage_signal is None or len(voltage_signal) < 100:
            print(f"[SKIP] {filename}: Could not load valid voltage data.")
            continue
        
        # Run integrated analysis
        results = integrated_analysis(voltage_signal, filename)
        all_results.append(results)
        
        # Save individual results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"integrated_analysis_{os.path.splitext(filename)[0]}_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}\n")
    
    # Generate summary report
    print("\n=== Integrated Analysis Summary ===")
    for idx, res in enumerate(all_results):
        print(f"File {idx+1}: {os.path.basename(res['filename'])}")
        print(f"  Spikes: {res['spike_results']['n_spikes']} ({res['spike_results']['spike_rate_hz']:.3f} Hz)")
        print(f"  Features: {res['transform_results']['n_features']} ({res['transform_results']['n_spike_aligned']} aligned)")
        print(f"  Activity Score: {res['synthesis_results']['biological_activity_score']:.3f}")
        print(f"  Recommended: {res['synthesis_results']['recommended_analysis']}")
        print()
    
    print("All results saved to:", results_dir)
    print("=== Integrated Analysis Complete ===")

if __name__ == "__main__":
    main() 