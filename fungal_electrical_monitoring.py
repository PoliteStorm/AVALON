#!/usr/bin/env python3
"""
Fungal Electrical Activity Monitoring System
Implements Adamatzky's method with comprehensive parameter validation
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import linregress
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class FungalElectricalMonitor:
    def __init__(self, config=None):
        """Initialize fungal electrical monitoring system"""
        self.config = config or self._get_default_config()
        self.spikes = []
        self.baseline = None
        self.threshold = None
        self.quality_score = 0.0
        
    def _get_default_config(self):
        """Get default configuration for fungal electrical monitoring"""
        return {
            # Adamatzky Method Parameters
            'adamatzky': {
                'baseline_threshold': 0.1,        # mV
                'threshold_multiplier': 1.0,
                'adaptive_threshold': True,
                'min_isi': 0.1,                   # seconds
                'max_isi': 10.0,                  # seconds
                'spike_duration': 0.05,           # seconds
                'min_spike_amplitude': 0.05,      # mV
                'max_spike_amplitude': 5.0,       # mV
                'min_snr': 3.0,
                'baseline_stability': 0.1         # mV
            },
            
            # Data Acquisition Parameters
            'acquisition': {
                'sampling_rate': 1000,            # Hz
                'recording_duration': 3600,       # seconds
                'buffer_size': 10000,             # samples
                'electrode_impedance': 1e6,       # Ω
                'amplifier_gain': 1000,
                'filter_bandwidth': [0.1, 100]    # Hz
            },
            
            # Analysis Parameters
            'analysis': {
                'confidence_level': 0.95,
                'p_value_threshold': 0.05,
                'min_spikes': 10,
                'max_spikes': 10000,
                'min_quality_score': 0.7,
                'validation_split': 0.2,
                'cv_folds': 5
            },
            
            # Species-specific parameters
            'species': 'pleurotus'  # Default species
        }
    
    def get_species_parameters(self, species_name):
        """Get species-specific parameters"""
        species_params = {
            'pleurotus': {
                'baseline_threshold': 0.15,
                'spike_threshold': 0.2,
                'min_isi': 0.2,
                'max_amplitude': 3.0,
                'typical_frequency': 0.5
            },
            'hericium': {
                'baseline_threshold': 0.1,
                'spike_threshold': 0.15,
                'min_isi': 0.1,
                'max_amplitude': 2.0,
                'typical_frequency': 1.0
            },
            'rhizopus': {
                'baseline_threshold': 0.2,
                'spike_threshold': 0.25,
                'min_isi': 0.3,
                'max_amplitude': 4.0,
                'typical_frequency': 0.3
            }
        }
        
        if species_name.lower() in species_params:
            # Update config with species-specific parameters
            self.config['adamatzky'].update(species_params[species_name.lower()])
            self.config['species'] = species_name.lower()
            print(f"Applied {species_name} parameters")
        
        return species_params.get(species_name.lower(), {})
    
    def validate_parameters(self):
        """Validate all monitoring parameters"""
        errors = []
        
        # Check Adamatzky parameters
        adamatzky = self.config['adamatzky']
        if adamatzky['baseline_threshold'] < 0.05:
            errors.append("Baseline threshold too low (< 0.05 mV)")
        if adamatzky['min_isi'] < 0.05:
            errors.append("Minimum ISI too low (< 0.05 s)")
        if adamatzky['min_spike_amplitude'] < 0.01:
            errors.append("Minimum spike amplitude too low (< 0.01 mV)")
        if adamatzky['max_spike_amplitude'] > 10.0:
            errors.append("Maximum spike amplitude too high (> 10 mV)")
        
        # Check acquisition parameters
        acquisition = self.config['acquisition']
        if acquisition['sampling_rate'] < 100:
            errors.append("Sampling rate too low (< 100 Hz)")
        if acquisition['amplifier_gain'] < 100:
            errors.append("Amplifier gain too low (< 100)")
        if acquisition['electrode_impedance'] < 1e5:
            errors.append("Electrode impedance too low (< 100 kΩ)")
        
        # Check analysis parameters
        analysis = self.config['analysis']
        if analysis['min_spikes'] < 1:
            errors.append("Minimum spikes too low (< 1)")
        if analysis['max_spikes'] > 100000:
            errors.append("Maximum spikes too high (> 100,000)")
        if analysis['min_quality_score'] < 0.0 or analysis['min_quality_score'] > 1.0:
            errors.append("Quality score out of range (0-1)")
        
        return errors
    
    def preprocess_signal(self, voltage_data):
        """Preprocess voltage signal for analysis"""
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
    
    def calculate_baseline(self, voltage_data):
        """Calculate baseline voltage using robust method"""
        # Use median as baseline (more robust than mean)
        baseline = np.median(voltage_data)
        
        # Check baseline stability
        baseline_std = np.std(voltage_data)
        if baseline_std > self.config['adamatzky']['baseline_stability']:
            print(f"Warning: Baseline unstable (std = {baseline_std:.4f} mV)")
        
        return baseline, baseline_std
    
    def calculate_threshold(self, voltage_data, baseline):
        """Calculate spike detection threshold"""
        adamatzky = self.config['adamatzky']
        
        if adamatzky['adaptive_threshold']:
            # Adaptive threshold based on signal variability
            signal_std = np.std(voltage_data)
            threshold = baseline + (adamatzky['threshold_multiplier'] * signal_std)
        else:
            # Fixed threshold
            threshold = baseline + adamatzky['baseline_threshold']
        
        return threshold
    
    def detect_spikes_adamatzky(self, voltage_data, sampling_rate=None):
        """Detect spikes using Adamatzky's method"""
        if sampling_rate is None:
            sampling_rate = self.config['acquisition']['sampling_rate']
        
        # Preprocess signal
        processed_signal = self.preprocess_signal(voltage_data)
        
        # Calculate baseline and threshold
        self.baseline, baseline_std = self.calculate_baseline(processed_signal)
        self.threshold = self.calculate_threshold(processed_signal, self.baseline)
        
        # Find spike candidates
        spike_candidates = []
        min_isi_samples = int(self.config['adamatzky']['min_isi'] * sampling_rate)
        
        # Find all points above threshold
        above_threshold = processed_signal > self.threshold
        
        # Find spike peaks
        i = 0
        while i < len(processed_signal):
            if above_threshold[i]:
                # Find peak of this spike
                peak_idx = i
                peak_amplitude = processed_signal[i]
                
                # Look for maximum in this spike
                j = i
                while j < len(processed_signal) and above_threshold[j]:
                    if processed_signal[j] > peak_amplitude:
                        peak_idx = j
                        peak_amplitude = processed_signal[j]
                    j += 1
                
                # Check amplitude constraints
                if (peak_amplitude >= self.config['adamatzky']['min_spike_amplitude'] and 
                    peak_amplitude <= self.config['adamatzky']['max_spike_amplitude']):
                    
                    # Check ISI constraint
                    if not spike_candidates or (peak_idx - spike_candidates[-1]['time_index']) >= min_isi_samples:
                        spike_candidates.append({
                            'time_index': peak_idx,
                            'time_seconds': peak_idx / sampling_rate,
                            'amplitude_mv': peak_amplitude,
                            'threshold': self.threshold,
                            'baseline': self.baseline
                        })
                
                i = j
            else:
                i += 1
        
        self.spikes = spike_candidates
        return spike_candidates
    
    def calculate_snr(self, voltage_data):
        """Calculate signal-to-noise ratio"""
        # Calculate signal power (variance of signal)
        signal_power = np.var(voltage_data)
        
        # Estimate noise power (variance of high-frequency components)
        # Apply high-pass filter to isolate noise
        nyquist = self.config['acquisition']['sampling_rate'] / 2
        high_freq = 50 / nyquist  # 50 Hz high-pass
        b, a = signal.butter(4, high_freq, btype='high')
        noise_signal = signal.filtfilt(b, a, voltage_data)
        noise_power = np.var(noise_signal)
        
        if noise_power > 0:
            snr = signal_power / noise_power
        else:
            snr = float('inf')
        
        return snr
    
    def calculate_quality_score(self, voltage_data, spikes):
        """Calculate overall quality score for the recording"""
        score = 0.0
        
        # Check SNR
        snr = self.calculate_snr(voltage_data)
        if snr >= self.config['adamatzky']['min_snr']:
            score += 0.3
        elif snr >= 1.0:
            score += 0.1
        
        # Check spike count
        min_spikes = self.config['analysis']['min_spikes']
        max_spikes = self.config['analysis']['max_spikes']
        if min_spikes <= len(spikes) <= max_spikes:
            score += 0.3
        elif len(spikes) > 0:
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
    
    def classify_spikes(self, spikes):
        """Classify spikes by frequency"""
        if not spikes:
            return {}
        
        # Calculate ISIs
        isis = []
        for i in range(1, len(spikes)):
            isi = spikes[i]['time_seconds'] - spikes[i-1]['time_seconds']
            isis.append(isi)
        
        if not isis:
            return {}
        
        # Calculate frequencies
        frequencies = [1.0 / isi for isi in isis if isi > 0]
        
        # Classify based on ISI ranges
        classifications = {
            'very_fast': 0,
            'fast': 0,
            'slow': 0,
            'very_slow': 0
        }
        
        for isi in isis:
            if isi < 0.1:
                classifications['very_fast'] += 1
            elif isi < 1.0:
                classifications['fast'] += 1
            elif isi < 10.0:
                classifications['slow'] += 1
            else:
                classifications['very_slow'] += 1
        
        return classifications
    
    def detect_bursts(self, spikes):
        """Detect burst firing patterns"""
        if len(spikes) < 3:
            return []
        
        bursts = []
        current_burst = [spikes[0]]
        max_isi_within_burst = self.config['adamatzky']['min_isi'] * 5  # 5x min ISI
        
        for i in range(1, len(spikes)):
            isi = spikes[i]['time_seconds'] - spikes[i-1]['time_seconds']
            
            if isi <= max_isi_within_burst:
                # Continue current burst
                current_burst.append(spikes[i])
            else:
                # End current burst if it has enough spikes
                if len(current_burst) >= 3:
                    bursts.append({
                        'start_time': current_burst[0]['time_seconds'],
                        'end_time': current_burst[-1]['time_seconds'],
                        'duration': current_burst[-1]['time_seconds'] - current_burst[0]['time_seconds'],
                        'spike_count': len(current_burst),
                        'spikes': current_burst
                    })
                
                # Start new burst
                current_burst = [spikes[i]]
        
        # Handle last burst
        if len(current_burst) >= 3:
            bursts.append({
                'start_time': current_burst[0]['time_seconds'],
                'end_time': current_burst[-1]['time_seconds'],
                'duration': current_burst[-1]['time_seconds'] - current_burst[0]['time_seconds'],
                'spike_count': len(current_burst),
                'spikes': current_burst
            })
        
        return bursts
    
    def analyze_recording(self, voltage_data, sampling_rate=None):
        """Complete analysis of fungal electrical recording"""
        if sampling_rate is None:
            sampling_rate = self.config['acquisition']['sampling_rate']
        
        # Validate parameters
        errors = self.validate_parameters()
        if errors:
            print("Parameter validation errors:")
            for error in errors:
                print(f"  - {error}")
            return None
        
        # Detect spikes
        spikes = self.detect_spikes_adamatzky(voltage_data, sampling_rate)
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(voltage_data, spikes)
        
        # Classify spikes
        classifications = self.classify_spikes(spikes)
        
        # Detect bursts
        bursts = self.detect_bursts(spikes)
        
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
                'classifications': classifications,
                'n_bursts': len(bursts),
                'quality_score': quality_score,
                'snr': self.calculate_snr(voltage_data)
            }
        else:
            stats = {
                'n_spikes': 0,
                'mean_amplitude': 0,
                'std_amplitude': 0,
                'mean_isi': 0,
                'std_isi': 0,
                'spike_rate': 0,
                'classifications': {},
                'n_bursts': 0,
                'quality_score': quality_score,
                'snr': self.calculate_snr(voltage_data)
            }
        
        return {
            'spikes': spikes,
            'bursts': bursts,
            'stats': stats,
            'config': self.config
        }
    
    def save_results(self, results, filename=None):
        """Save analysis results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fungal_electrical_analysis_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Convert results for JSON serialization
        json_results = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {filename}")
        return filename

def main():
    """Example usage of fungal electrical monitoring system"""
    # Initialize monitor with default parameters
    monitor = FungalElectricalMonitor()
    
    # Set species-specific parameters
    monitor.get_species_parameters('pleurotus')
    
    # Generate example fungal electrical signal
    sampling_rate = 1000  # Hz
    duration = 60  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Create realistic fungal electrical signal
    # Baseline with slow oscillations
    baseline = 0.5 + 0.1 * np.sin(2 * np.pi * 0.01 * t)
    
    # Add random spikes
    spike_times = np.random.exponential(2.0, 30)  # Average 2 second intervals
    spike_times = np.cumsum(spike_times)
    spike_times = spike_times[spike_times < duration]
    
    signal = baseline.copy()
    for spike_time in spike_times:
        spike_idx = int(spike_time * sampling_rate)
        if spike_idx < len(signal):
            # Add spike with exponential decay
            spike_duration = int(0.05 * sampling_rate)  # 50ms spike
            for i in range(min(spike_duration, len(signal) - spike_idx)):
                signal[spike_idx + i] += 0.5 * np.exp(-i / (0.01 * sampling_rate))
    
    # Add noise
    noise = np.random.normal(0, 0.02, len(signal))
    signal += noise
    
    # Analyze the signal
    results = monitor.analyze_recording(signal, sampling_rate)
    
    # Print results
    print("Fungal Electrical Activity Analysis Results:")
    print("=" * 50)
    print(f"Species: {monitor.config['species']}")
    print(f"Quality Score: {results['stats']['quality_score']:.3f}")
    print(f"SNR: {results['stats']['snr']:.2f}")
    print(f"Spikes Detected: {results['stats']['n_spikes']}")
    print(f"Spike Rate: {results['stats']['spike_rate']:.3f} Hz")
    print(f"Mean Amplitude: {results['stats']['mean_amplitude']:.4f} mV")
    print(f"Mean ISI: {results['stats']['mean_isi']:.3f} s")
    print(f"Bursts Detected: {results['stats']['n_bursts']}")
    
    if results['stats']['classifications']:
        print("\nSpike Classifications:")
        for classification, count in results['stats']['classifications'].items():
            print(f"  {classification}: {count}")
    
    # Save results
    monitor.save_results(results)
    
    return results

if __name__ == "__main__":
    main() 