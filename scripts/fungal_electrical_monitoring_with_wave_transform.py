#!/usr/bin/env python3
"""
Fungal Electrical Activity Monitoring System
Integrates Adamatzky's spike detection with wave transform analysis
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

warnings.filterwarnings('ignore')

class WaveTransform:
    """Wave transform implementation for multi-scale pattern detection"""
    
    def __init__(self, scale_range=(0.1, 10.0), shift_range=(0, 100)):
        self.scale_range = scale_range
        self.shift_range = shift_range
        
    def wave_transform(self, signal_data, scale, shift):
        """Apply wave transform with given scale and shift parameters"""
        n = len(signal_data)
        transformed = np.zeros(n)
        
        for i in range(n):
            # Apply wave function: sqrt(t + shift) * scale
            t = i / n  # Normalize time to [0, 1]
            wave_value = np.sqrt(t + shift) * scale
            transformed[i] = signal_data[i] * wave_value
            
        return transformed
    
    def detect_wave_patterns(self, signal_data, threshold=0.1):
        """Detect patterns using wave transform across multiple scales"""
        patterns = []
        
        # Sample scale and shift parameters
        scales = np.linspace(self.scale_range[0], self.scale_range[1], 20)
        shifts = np.linspace(self.shift_range[0], self.shift_range[1], 10)
        
        for scale in scales:
            for shift in shifts:
                # Apply wave transform
                transformed = self.wave_transform(signal_data, scale, shift)
                
                # Calculate pattern strength
                pattern_strength = np.std(transformed)
                
                if pattern_strength > threshold:
                    patterns.append({
                        'scale': scale,
                        'shift': shift,
                        'strength': pattern_strength,
                        'transformed_signal': transformed
                    })
        
        return patterns
    
    def calculate_wave_features(self, signal_data):
        """Calculate comprehensive wave transform features"""
        patterns = self.detect_wave_patterns(signal_data)
        
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
            'confidence': min(1.0, len(patterns) / 50)  # Normalize confidence
        }

class IntegratedFungalMonitor:
    """Integrated fungal electrical monitoring with both Adamatzky and wave transform"""
    
    def __init__(self, config=None):
        """Initialize integrated fungal monitoring system"""
        self.config = config or self._get_default_config()
        self.spikes = []
        self.wave_patterns = []
        self.baseline = None
        self.threshold = None
        self.quality_score = 0.0
        self.wave_transform = WaveTransform()
        
    def _get_default_config(self):
        """Get default configuration for integrated monitoring"""
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
            
            # Wave Transform Parameters
            'wave_transform': {
                'scale_range': [0.1, 10.0],
                'shift_range': [0, 100],
                'threshold': 0.1,
                'confidence': 0.8,
                'integration_weight': 0.5
            },
            
            # Integration Parameters
            'integration': {
                'method_combination': 'weighted_average',
                'spike_wave_alignment': True,
                'cross_validation': True,
                'ensemble_threshold': 0.7,
                'alignment_threshold': 0.3
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
            # Update config with species-specific parameters
            self.config['adamatzky'].update(species_params[species_name.lower()])
            self.config['wave_transform']['scale_range'] = species_params[species_name.lower()]['wave_scale_range']
            self.config['wave_transform']['shift_range'] = species_params[species_name.lower()]['wave_shift_range']
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
        
        # Check wave transform parameters
        wave = self.config['wave_transform']
        if wave['scale_range'][0] <= 0:
            errors.append("Wave scale range must be positive")
        if wave['shift_range'][0] < 0:
            errors.append("Wave shift range must be non-negative")
        if wave['threshold'] <= 0:
            errors.append("Wave threshold must be positive")
        
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
    
    def analyze_wave_patterns(self, voltage_data):
        """Analyze signal using wave transform"""
        # Update wave transform parameters
        self.wave_transform.scale_range = tuple(self.config['wave_transform']['scale_range'])
        self.wave_transform.shift_range = tuple(self.config['wave_transform']['shift_range'])
        
        # Detect wave patterns
        patterns = self.wave_transform.detect_wave_patterns(
            voltage_data, 
            self.config['wave_transform']['threshold']
        )
        
        # Calculate wave features
        wave_features = self.wave_transform.calculate_wave_features(voltage_data)
        
        self.wave_patterns = patterns
        return patterns, wave_features
    
    def align_spikes_with_waves(self, spikes, wave_patterns):
        """Align spike detection with wave transform patterns"""
        if not spikes or not wave_patterns:
            return 0.0
        
        alignment_scores = []
        
        for spike in spikes:
            spike_time = spike['time_seconds']
            
            # Find wave patterns near spike time
            for pattern in wave_patterns:
                # Calculate pattern strength at spike time
                pattern_time = pattern.get('time', 0)
                time_diff = abs(spike_time - pattern_time)
                
                if time_diff < self.config['integration']['alignment_threshold']:
                    alignment_score = pattern['strength'] * (1 - time_diff)
                    alignment_scores.append(alignment_score)
        
        if alignment_scores:
            return np.mean(alignment_scores)
        else:
            return 0.0
    
    def calculate_integrated_score(self, spike_results, wave_results):
        """Calculate integrated score combining both methods"""
        integration = self.config['integration']
        weight = self.config['wave_transform']['integration_weight']
        
        # Spike-based score
        spike_score = 0.0
        if spike_results['n_spikes'] > 0:
            spike_score = min(1.0, spike_results['n_spikes'] / 100)  # Normalize
        
        # Wave-based score
        wave_score = wave_results.get('confidence', 0.0)
        
        # Combined score
        if integration['method_combination'] == 'weighted_average':
            combined_score = weight * wave_score + (1 - weight) * spike_score
        else:  # voting
            combined_score = 1.0 if (spike_score > 0.5 and wave_score > 0.5) else 0.0
        
        return combined_score
    
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
    
    def calculate_quality_score(self, voltage_data, spikes, wave_features):
        """Calculate overall quality score for the recording"""
        score = 0.0
        
        # Check SNR
        snr = self.calculate_snr(voltage_data)
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
        """Complete integrated analysis of fungal electrical recording"""
        if sampling_rate is None:
            sampling_rate = self.config['acquisition']['sampling_rate']
        
        # Validate parameters
        errors = self.validate_parameters()
        if errors:
            print("Parameter validation errors:")
            for error in errors:
                print(f"  - {error}")
            return None
        
        # Detect spikes using Adamatzky's method
        spikes = self.detect_spikes_adamatzky(voltage_data, sampling_rate)
        
        # Analyze wave patterns
        wave_patterns, wave_features = self.analyze_wave_patterns(voltage_data)
        
        # Align spikes with wave patterns
        alignment_score = self.align_spikes_with_waves(spikes, wave_patterns)
        
        # Calculate integrated score
        spike_stats = self._calculate_spike_statistics(spikes, voltage_data, sampling_rate)
        integrated_score = self.calculate_integrated_score(spike_stats, wave_features)
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(voltage_data, spikes, wave_features)
        
        # Classify spikes
        classifications = self.classify_spikes(spikes)
        
        # Detect bursts
        bursts = self.detect_bursts(spikes)
        
        # Combine results
        results = {
            'spikes': spikes,
            'wave_patterns': wave_patterns,
            'wave_features': wave_features,
            'bursts': bursts,
            'alignment_score': alignment_score,
            'integrated_score': integrated_score,
            'stats': spike_stats,
            'quality_score': quality_score,
            'config': self.config
        }
        
        return results
    
    def _calculate_spike_statistics(self, spikes, voltage_data, sampling_rate):
        """Calculate spike statistics"""
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
                'classifications': self.classify_spikes(spikes),
                'n_bursts': len(self.detect_bursts(spikes)),
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
                'snr': self.calculate_snr(voltage_data)
            }
        
        return stats
    
    def save_results(self, results, filename=None):
        """Save analysis results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"integrated_fungal_analysis_{timestamp}.json"
        
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
    """Example usage of integrated fungal electrical monitoring system"""
    # Initialize monitor with default parameters
    monitor = IntegratedFungalMonitor()
    
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
    
    # Analyze the signal with both methods
    results = monitor.analyze_recording(signal, sampling_rate)
    
    # Print results
    print("Integrated Fungal Electrical Activity Analysis Results:")
    print("=" * 60)
    print(f"Species: {monitor.config['species']}")
    print(f"Quality Score: {results['quality_score']:.3f}")
    print(f"Integrated Score: {results['integrated_score']:.3f}")
    print(f"Alignment Score: {results['alignment_score']:.3f}")
    print(f"SNR: {results['stats']['snr']:.2f}")
    print(f"Spikes Detected: {results['stats']['n_spikes']}")
    print(f"Wave Patterns: {results['wave_features']['wave_patterns']}")
    print(f"Wave Confidence: {results['wave_features']['confidence']:.3f}")
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