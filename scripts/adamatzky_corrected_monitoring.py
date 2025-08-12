#!/usr/bin/env python3
"""
Adamatzky-Corrected Fungal Electrical Monitoring System
Implements proper temporal parameters with time compression for simulation
"""

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import json
import os

warnings.filterwarnings('ignore')

class AdamatzkyCorrectedMonitor:
    """Fungal electrical monitoring with Adamatzky-corrected parameters"""
    
    def __init__(self, time_compression_factor=86400):
        """
        Initialize with Adamatzky-corrected parameters
        
        Args:
            time_compression_factor: Factor to compress time (86400 = 1 day in 1 second)
        """
        self.time_compression_factor = time_compression_factor
        self.config = self._get_adamatzky_corrected_config()
        
    def _get_adamatzky_corrected_config(self):
        """Get configuration based on Adamatzky's actual research"""
        return {
            # Adamatzky's Corrected Parameters
            'adamatzky': {
                'baseline_threshold': 0.1,        # mV
                'threshold_multiplier': 1.0,
                'adaptive_threshold': True,
                'min_isi': 30,                    # seconds (very fast = half-minute)
                'max_isi': 3600,                  # seconds (very slow = hours)
                'spike_duration': 30,             # seconds (very fast spike duration)
                'min_spike_amplitude': 0.05,      # mV
                'max_spike_amplitude': 5.0,       # mV
                'min_snr': 2.0,                   # Lowered from 3.0 for fungal signals
                'baseline_stability': 0.1         # mV
            },
            
            # Adamatzky's Sampling Parameters
            'acquisition': {
                'sampling_rate': 1,               # Hz (Adamatzky's rate)
                'recording_duration': 259200,     # 3 days (compressed)
                'buffer_size': 86400,             # 1 day of 1 Hz data
                'electrode_impedance': 1e6,       # Ω
                'amplifier_gain': 1000,
                'filter_bandwidth': [0.001, 0.1] # Hz (much lower for fungal signals)
            },
            
            # Adamatzky's Three Categories
            'categories': {
                'very_fast': {
                    'min_duration': 30,           # 30 seconds
                    'max_duration': 60,           # 60 seconds
                    'min_isi': 30,                # 30 seconds
                    'max_isi': 300,               # 5 minutes
                    'description': 'Half-minute scale activity'
                },
                'slow': {
                    'min_duration': 600,          # 10 minutes
                    'max_duration': 3600,         # 1 hour
                    'min_isi': 600,               # 10 minutes
                    'max_isi': 3600,              # 1 hour
                    'description': '10-minute scale activity'
                },
                'very_slow': {
                    'min_duration': 3600,         # 1 hour
                    'max_duration': float('inf'), # Unlimited
                    'min_isi': 3600,              # 1 hour
                    'max_isi': float('inf'),      # Unlimited
                    'description': 'Hour-scale activity'
                }
            },
            
            # Analysis Parameters
            'analysis': {
                'confidence_level': 0.95,
                'min_spikes': 3,                  # Lower for longer recordings
                'max_spikes': 1000,               # Higher for longer recordings
                'min_quality_score': 0.7,
                'validation_split': 0.2,
                'cv_folds': 5
            },
            
            # Time Compression Settings
            'compression': {
                'factor': self.time_compression_factor,
                'simulate_days': True,
                'compression_ratio': '1_second = 1_day'
            }
        }
    
    def compress_time_series(self, voltage_data, original_sampling_rate=1000):
        """
        Compress time series to simulate longer recording periods
        
        Args:
            voltage_data: Original voltage data
            original_sampling_rate: Original sampling rate (Hz)
            
        Returns:
            compressed_data: Time-compressed data
            compressed_sampling_rate: New sampling rate
        """
        print(f"🕐 TIME COMPRESSION: {self.time_compression_factor}x")
        print(f"   Original: {len(voltage_data)} samples at {original_sampling_rate} Hz")
        
        # Calculate compression parameters
        compression_ratio = original_sampling_rate / self.config['acquisition']['sampling_rate']
        target_samples = len(voltage_data) // compression_ratio
        
        # Resample data to 1 Hz (Adamatzky's rate)
        if original_sampling_rate > 1:
            # Downsample to 1 Hz
            downsample_factor = int(original_sampling_rate)
            compressed_data = voltage_data[::downsample_factor]
            
            # Apply additional compression if needed
            if len(compressed_data) > target_samples:
                compression_factor = int(len(compressed_data) // target_samples)
                if compression_factor > 1:
                    compressed_data = compressed_data[::compression_factor]
        else:
            compressed_data = voltage_data
        
        compressed_sampling_rate = 1  # Hz (Adamatzky's rate)
        
        print(f"   Compressed: {len(compressed_data)} samples at {compressed_sampling_rate} Hz")
        print(f"   Simulated duration: {len(compressed_data) / 86400:.1f} days")
        
        return compressed_data, compressed_sampling_rate
    
    def detect_spikes_adamatzky_corrected(self, voltage_data, sampling_rate=1):
        """
        Detect spikes using Adamatzky's corrected parameters
        """
        print("🔬 DETECTING SPIKES WITH ADAMATZKY-CORRECTED PARAMETERS")
        
        # Preprocess signal for fungal timescales
        processed_signal = self.preprocess_signal_fungal(voltage_data)
        
        # Calculate baseline and threshold
        baseline, baseline_std = self.calculate_baseline_fungal(processed_signal)
        threshold = self.calculate_threshold_fungal(processed_signal, baseline)
        
        # Find spike candidates with corrected parameters
        spike_candidates = []
        min_isi_samples = int(self.config['adamatzky']['min_isi'] * sampling_rate)
        max_isi_samples = int(self.config['adamatzky']['max_isi'] * sampling_rate)
        
        # Find all points above threshold
        above_threshold = processed_signal > threshold
        
        # Find spike peaks with fungal timescales
        i = 0
        while i < len(processed_signal):
            if above_threshold[i]:
                # Find peak of this spike (30+ second duration)
                peak_idx = i
                peak_amplitude = processed_signal[i]
                
                # Look for maximum in this spike (30+ second window)
                spike_duration_samples = int(self.config['adamatzky']['spike_duration'] * sampling_rate)
                j = i
                while j < len(processed_signal) and j < i + spike_duration_samples:
                    if processed_signal[j] > peak_amplitude:
                        peak_idx = j
                        peak_amplitude = processed_signal[j]
                    j += 1
                
                # Check amplitude constraints
                if (peak_amplitude >= self.config['adamatzky']['min_spike_amplitude'] and 
                    peak_amplitude <= self.config['adamatzky']['max_spike_amplitude']):
                    
                    # Check ISI constraint (30+ seconds between spikes)
                    if not spike_candidates or (peak_idx - spike_candidates[-1]['time_index']) >= min_isi_samples:
                        spike_candidates.append({
                            'time_index': peak_idx,
                            'time_seconds': peak_idx / sampling_rate,
                            'amplitude_mv': peak_amplitude,
                            'threshold': threshold,
                            'baseline': baseline
                        })
                
                i = j
            else:
                i += 1
        
        return spike_candidates
    
    def preprocess_signal_fungal(self, voltage_data):
        """Preprocess signal for fungal electrical activity timescales"""
        # Apply very low-frequency bandpass filter for fungal signals
        nyquist = self.config['acquisition']['sampling_rate'] / 2
        low_freq = self.config['acquisition']['filter_bandwidth'][0] / nyquist
        high_freq = self.config['acquisition']['filter_bandwidth'][1] / nyquist
        
        # Design filter for fungal timescales
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        filtered_signal = signal.filtfilt(b, a, voltage_data)
        
        # Remove DC offset
        filtered_signal = filtered_signal - np.mean(filtered_signal)
        
        return filtered_signal
    
    def calculate_baseline_fungal(self, voltage_data):
        """Calculate baseline for fungal electrical activity"""
        # Use median as baseline (more robust for long recordings)
        baseline = np.median(voltage_data)
        
        # Check baseline stability over longer periods
        baseline_std = np.std(voltage_data)
        if baseline_std > self.config['adamatzky']['baseline_stability']:
            print(f"⚠️  Warning: Baseline unstable (std = {baseline_std:.4f} mV)")
        
        return baseline, baseline_std
    
    def calculate_threshold_fungal(self, voltage_data, baseline):
        """Calculate threshold for fungal spike detection"""
        adamatzky = self.config['adamatzky']
        
        if adamatzky['adaptive_threshold']:
            # Adaptive threshold based on signal variability
            signal_std = np.std(voltage_data)
            threshold = baseline + (adamatzky['threshold_multiplier'] * signal_std)
        else:
            # Fixed threshold
            threshold = baseline + adamatzky['baseline_threshold']
        
        return threshold
    
    def classify_spikes_adamatzky(self, spikes):
        """Classify spikes according to Adamatzky's three categories"""
        if not spikes:
            return {}
        
        # Calculate ISIs
        isis = []
        for i in range(1, len(spikes)):
            isi = spikes[i]['time_seconds'] - spikes[i-1]['time_seconds']
            isis.append(isi)
        
        if not isis:
            return {}
        
        # Classify based on Adamatzky's categories
        categories = self.config['categories']
        classified = {
            'very_fast': [],
            'slow': [],
            'very_slow': []
        }
        
        for i, spike in enumerate(spikes):
            if i == 0:
                # First spike - classify based on amplitude/duration
                if spike['amplitude_mv'] > 0.3:
                    classified['very_fast'].append(spike)
                else:
                    classified['slow'].append(spike)
            else:
                isi = isis[i-1]
                
                # Classify based on ISI
                if isi <= categories['very_fast']['max_isi']:
                    classified['very_fast'].append(spike)
                elif isi <= categories['slow']['max_isi']:
                    classified['slow'].append(spike)
                else:
                    classified['very_slow'].append(spike)
        
        return classified
    
    def calculate_snr_fungal(self, voltage_data):
        """Calculate SNR for fungal electrical activity"""
        # Calculate signal power (variance of signal)
        signal_power = np.var(voltage_data)
        
        # Estimate noise power (variance of high-frequency components)
        # For fungal signals, use very low-frequency noise
        nyquist = self.config['acquisition']['sampling_rate'] / 2
        high_freq = 0.01 / nyquist  # 0.01 Hz high-pass for fungal signals
        b, a = signal.butter(4, high_freq, btype='high')
        noise_signal = signal.filtfilt(b, a, voltage_data)
        noise_power = np.var(noise_signal)
        
        if noise_power > 0:
            snr = signal_power / noise_power
        else:
            snr = float('inf')
        
        return snr
    
    def analyze_recording_adamatzky_corrected(self, voltage_data, original_sampling_rate=1000):
        """Analyze recording with Adamatzky-corrected parameters"""
        print("🧬 ADAMATZKY-CORRECTED FUNGAL ELECTRICAL ACTIVITY ANALYSIS")
        print("=" * 70)
        
        # Compress time series
        compressed_data, compressed_sampling_rate = self.compress_time_series(
            voltage_data, original_sampling_rate
        )
        
        # Detect spikes with corrected parameters
        spikes = self.detect_spikes_adamatzky_corrected(compressed_data, compressed_sampling_rate)
        
        # Classify spikes according to Adamatzky's categories
        classified_spikes = self.classify_spikes_adamatzky(spikes)
        
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
                'spike_rate': len(spikes) / (len(compressed_data) / compressed_sampling_rate),
                'snr': self.calculate_snr_fungal(compressed_data),
                'classified_spikes': classified_spikes
            }
        else:
            stats = {
                'n_spikes': 0,
                'mean_amplitude': 0,
                'std_amplitude': 0,
                'mean_isi': 0,
                'std_isi': 0,
                'spike_rate': 0,
                'snr': self.calculate_snr_fungal(compressed_data),
                'classified_spikes': {}
            }
        
        # Display results
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   Total Spikes: {stats['n_spikes']}")
        print(f"   Spike Rate: {stats['spike_rate']:.6f} Hz")
        print(f"   Mean Amplitude: {stats['mean_amplitude']:.4f} mV")
        print(f"   Mean ISI: {stats['mean_isi']:.1f} seconds")
        print(f"   SNR: {stats['snr']:.2f}")
        
        print(f"\n🎯 ADAMATZKY CLASSIFICATION:")
        for category, spikes_list in stats['classified_spikes'].items():
            print(f"   {category.replace('_', ' ').title()}: {len(spikes_list)} spikes")
        
        return {
            'spikes': spikes,
            'stats': stats,
            'config': self.config,
            'compression_info': {
                'original_samples': len(voltage_data),
                'compressed_samples': len(compressed_data),
                'compression_factor': self.time_compression_factor,
                'simulated_days': len(compressed_data) / 86400
            }
        }

def main():
    """Main function to test Adamatzky-corrected monitoring"""
    print("🔬 ADAMATZKY-CORRECTED FUNGAL ELECTRICAL MONITORING")
    print("=" * 60)
    
    # Initialize with time compression (1 second = 1 day)
    monitor = AdamatzkyCorrectedMonitor(time_compression_factor=86400)
    
    # Test with existing data
    test_file = "data/Norm_vs_deep_tip_crop.csv"
    if os.path.exists(test_file):
        print(f"\n📁 Testing with: {test_file}")
        
        # Load data
        data = pd.read_csv(test_file)
        voltage_data = data.iloc[:, 1].values  # Assuming second column is voltage
        
        # Analyze with Adamatzky-corrected parameters
        results = monitor.analyze_recording_adamatzky_corrected(voltage_data)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"adamatzky_corrected_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Display compression summary
        compression = results['compression_info']
        print(f"\n⏱️  TIME COMPRESSION SUMMARY:")
        print(f"   Original: {compression['original_samples']} samples")
        print(f"   Compressed: {compression['compressed_samples']} samples")
        print(f"   Compression Factor: {compression['compression_factor']}x")
        print(f"   Simulated Duration: {compression['simulated_days']:.1f} days")
        
    else:
        print(f"❌ Test file not found: {test_file}")

if __name__ == "__main__":
    main() 