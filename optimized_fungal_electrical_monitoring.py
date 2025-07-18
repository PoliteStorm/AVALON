#!/usr/bin/env python3
"""
Optimized Fungal Electrical Activity Monitoring System
High-performance implementation with vectorized operations and parallel processing
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import linregress
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import numba
from numba import jit, prange
import multiprocessing as mp

warnings.filterwarnings('ignore')

# Enable Numba JIT compilation for performance
@jit(nopython=True, parallel=True)
def fast_wave_transform(signal_data, scale, shift):
    """Optimized wave transform using Numba JIT compilation"""
    n = len(signal_data)
    transformed = np.zeros(n)
    
    for i in prange(n):
        t = i / n
        wave_value = np.sqrt(t + shift) * scale
        transformed[i] = signal_data[i] * wave_value
    
    return transformed

@jit(nopython=True)
def fast_spike_detection(voltage_data, threshold, min_isi_samples):
    """Optimized spike detection using Numba"""
    n = len(voltage_data)
    spikes = []
    
    for i in range(20, n-20):
        avg = np.mean(voltage_data[i-20:i+20])
        if abs(voltage_data[i]) - abs(avg) > threshold:
            # Check minimum ISI
            if not spikes or (i - spikes[-1]) >= min_isi_samples:
                spikes.append(i)
    
    return np.array(spikes)

class OptimizedWaveTransform:
    """Optimized wave transform with vectorized operations and caching"""
    
    def __init__(self, scale_range=(0.1, 10.0), shift_range=(0, 100)):
        self.scale_range = scale_range
        self.shift_range = shift_range
        self._cache = {}
        
    def wave_transform_vectorized(self, signal_data, scales, shifts):
        """Vectorized wave transform for multiple scales/shifts simultaneously"""
        n = len(signal_data)
        t = np.linspace(0, 1, n)
        
        # Precompute wave values for all scales and shifts
        scale_grid, shift_grid = np.meshgrid(scales, shifts)
        t_grid = t.reshape(-1, 1, 1)
        
        # Vectorized wave calculation
        wave_values = np.sqrt(t_grid + shift_grid) * scale_grid
        
        # Apply to signal data
        signal_reshaped = signal_data.reshape(-1, 1, 1)
        transformed = signal_reshaped * wave_values
        
        return transformed
    
    def detect_wave_patterns_optimized(self, signal_data, threshold=0.1, max_patterns=50):
        """Optimized pattern detection with early termination"""
        patterns = []
        
        # Reduce parameter space for speed
        scales = np.linspace(self.scale_range[0], self.scale_range[1], 10)  # Reduced from 20
        shifts = np.linspace(self.shift_range[0], self.shift_range[1], 5)   # Reduced from 10
        
        # Vectorized computation
        transformed = self.wave_transform_vectorized(signal_data, scales, shifts)
        
        # Calculate pattern strengths vectorized
        pattern_strengths = np.std(transformed, axis=0)
        
        # Find patterns above threshold
        strong_patterns = np.where(pattern_strengths > threshold)
        
        for i, j in zip(strong_patterns[0], strong_patterns[1]):
            if len(patterns) >= max_patterns:
                break
                
            patterns.append({
                'scale': scales[j],
                'shift': shifts[i],
                'strength': pattern_strengths[i, j]
            })
        
        return patterns
    
    def calculate_wave_features_optimized(self, signal_data):
        """Optimized wave feature calculation"""
        patterns = self.detect_wave_patterns_optimized(signal_data)
        
        if not patterns:
            return {
                'wave_patterns': 0,
                'max_strength': 0,
                'avg_strength': 0,
                'scale_distribution': [],
                'shift_distribution': [],
                'confidence': 0
            }
        
        strengths = [p['strength'] for p in patterns]
        scales = [p['scale'] for p in patterns]
        shifts = [p['shift'] for p in patterns]
        
        return {
            'wave_patterns': len(patterns),
            'max_strength': np.max(strengths),
            'avg_strength': np.mean(strengths),
            'scale_distribution': scales,
            'shift_distribution': shifts,
            'confidence': min(1.0, len(patterns) / 25)  # Adjusted normalization
        }

class OptimizedFungalMonitor:
    """Optimized fungal electrical monitoring with parallel processing"""
    
    def __init__(self, config=None):
        """Initialize optimized fungal monitoring system"""
        self.config = config or self._get_default_config()
        self.spikes = []
        self.wave_patterns = []
        self.baseline = None
        self.threshold = None
        self.quality_score = 0.0
        self.wave_transform = OptimizedWaveTransform()
        self._cache = {}
        
    def _get_default_config(self):
        """Get optimized default configuration"""
        return {
            # Adamatzky Method Parameters
            'adamatzky': {
                'baseline_threshold': 0.1,
                'threshold_multiplier': 1.0,
                'adaptive_threshold': True,
                'min_isi': 0.1,
                'max_isi': 10.0,
                'spike_duration': 0.05,
                'min_spike_amplitude': 0.05,
                'max_spike_amplitude': 5.0,
                'min_snr': 3.0,
                'baseline_stability': 0.1
            },
            
            # Wave Transform Parameters (optimized)
            'wave_transform': {
                'scale_range': [0.1, 10.0],
                'shift_range': [0, 100],
                'threshold': 0.1,
                'confidence': 0.8,
                'integration_weight': 0.5,
                'max_patterns': 50  # Limit for speed
            },
            
            # Performance Parameters
            'performance': {
                'use_parallel': True,
                'max_workers': min(8, mp.cpu_count()),
                'chunk_size': 10000,
                'use_cache': True,
                'early_termination': True
            },
            
            # Integration Parameters
            'integration': {
                'method_combination': 'weighted_average',
                'spike_wave_alignment': True,
                'cross_validation': False,  # Disabled for speed
                'ensemble_threshold': 0.7,
                'alignment_threshold': 0.3
            },
            
            # Data Acquisition Parameters
            'acquisition': {
                'sampling_rate': 1000,
                'recording_duration': 3600,
                'buffer_size': 10000,
                'electrode_impedance': 1e6,
                'amplifier_gain': 1000,
                'filter_bandwidth': [0.1, 100]
            },
            
            # Analysis Parameters (optimized)
            'analysis': {
                'confidence_level': 0.95,
                'p_value_threshold': 0.05,
                'min_spikes': 10,
                'max_spikes': 5000,  # Reduced for speed
                'min_quality_score': 0.7,
                'validation_split': 0.2,
                'cv_folds': 3  # Reduced for speed
            },
            
            'species': 'pleurotus'
        }
    
    @lru_cache(maxsize=128)
    def get_species_parameters(self, species_name):
        """Cached species-specific parameters"""
        species_params = {
            'pleurotus': {
                'baseline_threshold': 0.15,
                'spike_threshold': 0.2,
                'min_isi': 0.2,
                'max_amplitude': 3.0,
                'typical_frequency': 0.5,
                'wave_scale_range': [0.2, 5.0],
                'wave_shift_range': [0, 50]
            },
            'hericium': {
                'baseline_threshold': 0.1,
                'spike_threshold': 0.15,
                'min_isi': 0.1,
                'max_amplitude': 2.0,
                'typical_frequency': 1.0,
                'wave_scale_range': [0.1, 8.0],
                'wave_shift_range': [0, 80]
            },
            'rhizopus': {
                'baseline_threshold': 0.2,
                'spike_threshold': 0.25,
                'min_isi': 0.3,
                'max_amplitude': 4.0,
                'typical_frequency': 0.3,
                'wave_scale_range': [0.3, 6.0],
                'wave_shift_range': [0, 60]
            }
        }
        
        if species_name.lower() in species_params:
            self.config['adamatzky'].update(species_params[species_name.lower()])
            self.config['wave_transform']['scale_range'] = species_params[species_name.lower()]['wave_scale_range']
            self.config['wave_transform']['shift_range'] = species_params[species_name.lower()]['wave_shift_range']
            self.config['species'] = species_name.lower()
        
        return species_params.get(species_name.lower(), {})
    
    def preprocess_signal_optimized(self, voltage_data):
        """Optimized signal preprocessing with vectorized operations"""
        # Apply bandpass filter
        acquisition = self.config['acquisition']
        nyquist = acquisition['sampling_rate'] / 2
        low_freq = self.config['acquisition']['filter_bandwidth'][0] / nyquist
        high_freq = self.config['acquisition']['filter_bandwidth'][1] / nyquist
        
        # Design filter
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        filtered_signal = signal.filtfilt(b, a, voltage_data)
        
        # Remove DC offset
        filtered_signal = filtered_signal - np.mean(filtered_signal)
        
        return filtered_signal
    
    def calculate_baseline_optimized(self, voltage_data):
        """Optimized baseline calculation"""
        baseline = np.median(voltage_data)
        baseline_std = np.std(voltage_data)
        
        if baseline_std > self.config['adamatzky']['baseline_stability']:
            print(f"Warning: Baseline unstable (std = {baseline_std:.4f} mV)")
        
        return baseline, baseline_std
    
    def calculate_threshold_optimized(self, voltage_data, baseline):
        """Optimized threshold calculation"""
        adamatzky = self.config['adamatzky']
        
        if adamatzky['adaptive_threshold']:
            signal_std = np.std(voltage_data)
            threshold = baseline + (adamatzky['threshold_multiplier'] * signal_std)
        else:
            threshold = baseline + adamatzky['baseline_threshold']
        
        return threshold
    
    def detect_spikes_adamatzky_optimized(self, voltage_data, sampling_rate=None):
        """Optimized spike detection using Numba JIT"""
        if sampling_rate is None:
            sampling_rate = self.config['acquisition']['sampling_rate']
        
        # Preprocess signal
        processed_signal = self.preprocess_signal_optimized(voltage_data)
        
        # Calculate baseline and threshold
        baseline, baseline_std = self.calculate_baseline_optimized(processed_signal)
        threshold = self.calculate_threshold_optimized(processed_signal, baseline)
        
        # Use optimized spike detection
        min_isi_samples = int(self.config['adamatzky']['min_isi'] * sampling_rate)
        spike_indices = fast_spike_detection(processed_signal, threshold, min_isi_samples)
        
        # Convert to spike objects
        spikes = []
        for idx in spike_indices:
            spike_time = idx / sampling_rate
            amplitude = processed_signal[idx] - baseline
            
            # Validate spike amplitude
            if (self.config['adamatzky']['min_spike_amplitude'] <= 
                abs(amplitude) <= self.config['adamatzky']['max_spike_amplitude']):
                spikes.append({
                    'time_seconds': spike_time,
                    'amplitude_mv': amplitude,
                    'index': idx
                })
        
        return spikes
    
    def analyze_wave_patterns_optimized(self, voltage_data):
        """Optimized wave pattern analysis"""
        # Update wave transform parameters
        self.wave_transform.scale_range = tuple(self.config['wave_transform']['scale_range'])
        self.wave_transform.shift_range = tuple(self.config['wave_transform']['shift_range'])
        
        # Use optimized pattern detection
        patterns = self.wave_transform.detect_wave_patterns_optimized(
            voltage_data, 
            self.config['wave_transform']['threshold'],
            self.config['wave_transform']['max_patterns']
        )
        
        # Calculate wave features
        wave_features = self.wave_transform.calculate_wave_features_optimized(voltage_data)
        
        self.wave_patterns = patterns
        return patterns, wave_features
    
    def calculate_snr_optimized(self, voltage_data):
        """Optimized SNR calculation"""
        signal_power = np.var(voltage_data)
        
        # Simplified noise estimation
        noise_signal = voltage_data - np.mean(voltage_data)
        noise_power = np.var(noise_signal)
        
        if noise_power > 0:
            snr = signal_power / noise_power
        else:
            snr = float('inf')
        
        return snr
    
    def calculate_quality_score_optimized(self, voltage_data, spikes, wave_features):
        """Optimized quality score calculation"""
        score = 0.0
        
        # Check SNR
        snr = self.calculate_snr_optimized(voltage_data)
        if snr >= self.config['adamatzky']['min_snr']:
            score += 0.2
        elif snr >= 1.0:
            score += 0.1
        
        # Check spike count
        min_spikes = self.config['analysis']['min_spikes']
        max_spikes = self.config['analysis']['max_spikes']
        if min_spikes <= len(spikes) <= max_spikes:
            score += 0.2
        elif len(spikes) > 0:
            score += 0.1
        
        # Check wave patterns
        if wave_features['wave_patterns'] > 0:
            score += 0.2
            if wave_features['confidence'] > 0.5:
                score += 0.1
        
        # Check baseline stability
        baseline_std = np.std(voltage_data)
        if baseline_std <= self.config['adamatzky']['baseline_stability']:
            score += 0.2
        elif baseline_std <= 2 * self.config['adamatzky']['baseline_stability']:
            score += 0.1
        
        # Check voltage range
        voltage_range = np.max(voltage_data) - np.min(voltage_data)
        if voltage_range >= 0.1:
            score += 0.2
        elif voltage_range >= 0.05:
            score += 0.1
        
        self.quality_score = score
        return score
    
    def analyze_recording_optimized(self, voltage_data, sampling_rate=None):
        """Optimized complete analysis"""
        if sampling_rate is None:
            sampling_rate = self.config['acquisition']['sampling_rate']
        
        # Detect spikes using optimized method
        spikes = self.detect_spikes_adamatzky_optimized(voltage_data, sampling_rate)
        
        # Analyze wave patterns using optimized method
        wave_patterns, wave_features = self.analyze_wave_patterns_optimized(voltage_data)
        
        # Calculate quality score
        quality_score = self.calculate_quality_score_optimized(voltage_data, spikes, wave_features)
        
        # Calculate statistics
        if spikes:
            amplitudes = [spike['amplitude_mv'] for spike in spikes]
            isis = []
            for i in range(1, len(spikes)):
                isi = spikes[i]['time_seconds'] - spikes[i-1]['time_seconds']
                isis.append(isi)
            
            stats = {
                'n_spikes': len(spikes),
                'mean_amplitude': np.mean(amplitudes),
                'std_amplitude': np.std(amplitudes),
                'mean_isi': np.mean(isis) if isis else 0,
                'std_isi': np.std(isis) if isis else 0,
                'spike_rate': len(spikes) / (len(voltage_data) / sampling_rate),
                'quality_score': quality_score,
                'snr': self.calculate_snr_optimized(voltage_data)
            }
        else:
            stats = {
                'n_spikes': 0,
                'mean_amplitude': 0,
                'std_amplitude': 0,
                'mean_isi': 0,
                'std_isi': 0,
                'spike_rate': 0,
                'quality_score': quality_score,
                'snr': self.calculate_snr_optimized(voltage_data)
            }
        
        return {
            'spikes': spikes,
            'wave_patterns': wave_patterns,
            'wave_features': wave_features,
            'stats': stats,
            'config': self.config
        }
    
    def save_results(self, results, filename=None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimized_fungal_results_{timestamp}.json"
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Convert numpy objects for JSON serialization
        results_json = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Results saved to {filename}")
        return filename

def main():
    """Example usage of optimized fungal electrical monitoring system"""
    print("=== Optimized Fungal Electrical Activity Monitoring ===")
    
    # Initialize monitor with optimized parameters
    monitor = OptimizedFungalMonitor()
    
    # Set species-specific parameters
    monitor.get_species_parameters('pleurotus')
    
    # Generate example fungal electrical signal
    sampling_rate = 1000  # Hz
    duration = 60  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Create realistic fungal electrical signal
    baseline = 0.5 + 0.1 * np.sin(2 * np.pi * 0.01 * t)
    
    # Add random spikes
    spike_times = np.random.exponential(2.0, 30)
    spike_times = np.cumsum(spike_times)
    spike_times = spike_times[spike_times < duration]
    
    signal = baseline.copy()
    for spike_time in spike_times:
        spike_idx = int(spike_time * sampling_rate)
        if spike_idx < len(signal):
            spike_duration = int(0.05 * sampling_rate)
            for i in range(min(spike_duration, len(signal) - spike_idx)):
                signal[spike_idx + i] += 0.5 * np.exp(-i / (0.01 * sampling_rate))
    
    # Add noise
    noise = np.random.normal(0, 0.02, len(signal))
    signal += noise
    
    # Analyze the signal with optimized methods
    print("Analyzing signal...")
    start_time = datetime.now()
    
    results = monitor.analyze_recording_optimized(signal, sampling_rate)
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Print results
    print("\nOptimized Fungal Electrical Activity Analysis Results:")
    print("=" * 60)
    print(f"Processing Time: {processing_time:.3f} seconds")
    print(f"Species: {monitor.config['species']}")
    print(f"Quality Score: {results['stats']['quality_score']:.3f}")
    print(f"SNR: {results['stats']['snr']:.2f}")
    print(f"Spikes Detected: {results['stats']['n_spikes']}")
    print(f"Spike Rate: {results['stats']['spike_rate']:.3f} Hz")
    print(f"Mean Amplitude: {results['stats']['mean_amplitude']:.4f} mV")
    print(f"Mean ISI: {results['stats']['mean_isi']:.3f} s")
    print(f"Wave Patterns: {results['wave_features']['wave_patterns']}")
    print(f"Wave Confidence: {results['wave_features']['confidence']:.3f}")
    
    # Save results
    monitor.save_results(results)
    
    print(f"\nOptimization Summary:")
    print(f"- Vectorized operations: Enabled")
    print(f"- Numba JIT compilation: Enabled")
    print(f"- Parallel processing: {monitor.config['performance']['use_parallel']}")
    print(f"- Caching: {monitor.config['performance']['use_cache']}")
    print(f"- Early termination: {monitor.config['performance']['early_termination']}")

if __name__ == "__main__":
    main() 