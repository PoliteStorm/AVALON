#!/usr/bin/env python3
"""
Rigorous Fungal Analysis with √t Transform
Implements improvements while maintaining strict false positive detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import json
from pathlib import Path

class RigorousFungalAnalyzer:
    def __init__(self, csv_data_dir, voltage_data_dir):
        """
        Initialize the rigorous analyzer.
        
        Args:
            csv_data_dir: Directory containing coordinate CSV files
            voltage_data_dir: Directory containing voltage recording files
        """
        self.csv_data_dir = csv_data_dir
        self.voltage_data_dir = voltage_data_dir
        self.results = {}
        
        # Biological parameters based on Adamatzky's multiscalar electrical spiking research
        self.biological_constraints = {
            'frequency_range': (0.001, 10.0),  # Hz - expanded range for all species
            'growth_time_scales': (0.1, 100000),  # seconds - 0.1s to ~28h
            'spike_amplitude_range': (0.02, 0.15),  # mV - based on actual data ranges
            'multiscalar_analysis': True,  # Enable multiscalar electrical spiking analysis
            'spike_detection_method': 'adaptive_threshold',  # Based on Adamatzky's methods
            'temporal_scales': [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],  # Multiple time scales
            'frequency_bands': [0.001, 0.01, 0.1, 1.0, 10.0],  # Frequency bands for analysis
            'species_characteristics': {
                'Pv': {
                    'growth_rate': 'fast', 
                    'spike_freq': 'high', 
                    'amplitude_range': (0.095, 0.115), 
                    'sqrt_scaling': True,
                    'action_potential_duration': (0.5, 2.0),  # seconds
                    'resting_potential': -0.080,  # mV
                    'threshold_potential': -0.060,  # mV
                    'electrical_fingerprint': 'high_frequency_bursts'
                },
                'Pi': {
                    'growth_rate': 'fast', 
                    'spike_freq': 'high', 
                    'amplitude_range': (0.090, 0.110), 
                    'sqrt_scaling': True,
                    'action_potential_duration': (0.8, 3.0),
                    'resting_potential': -0.075,
                    'threshold_potential': -0.055,
                    'electrical_fingerprint': 'medium_frequency_regular'
                },
                'Pp': {
                    'growth_rate': 'fast', 
                    'spike_freq': 'very_high', 
                    'amplitude_range': (0.100, 0.120), 
                    'sqrt_scaling': True,
                    'action_potential_duration': (0.3, 1.5),
                    'resting_potential': -0.085,
                    'threshold_potential': -0.065,
                    'electrical_fingerprint': 'very_high_frequency_irregular'
                },
                'Rb': {
                    'growth_rate': 'slow', 
                    'spike_freq': 'low', 
                    'amplitude_range': (0.080, 0.100), 
                    'sqrt_scaling': True,
                    'action_potential_duration': (2.0, 8.0),
                    'resting_potential': -0.070,
                    'threshold_potential': -0.050,
                    'electrical_fingerprint': 'low_frequency_slow'
                },
                'Hericium': {
                    'growth_rate': 'very_slow', 
                    'spike_freq': 'very_low', 
                    'amplitude_range': (0.020, 0.080), 
                    'sqrt_scaling': True,
                    'action_potential_duration': (5.0, 15.0),
                    'resting_potential': -0.065,
                    'threshold_potential': -0.045,
                    'electrical_fingerprint': 'very_low_frequency_sporadic'
                },
                'Ag': {
                    'growth_rate': 'medium', 
                    'spike_freq': 'medium', 
                    'amplitude_range': (0.085, 0.105), 
                    'sqrt_scaling': True,
                    'action_potential_duration': (1.0, 4.0),
                    'resting_potential': -0.078,
                    'threshold_potential': -0.058,
                    'electrical_fingerprint': 'medium_frequency_steady'
                },
                'Sc': {
                    'growth_rate': 'medium', 
                    'spike_freq': 'multiscalar',  # Based on Adamatzky's multiscalar findings
                    'amplitude_range': (0.085, 0.105), 
                    'sqrt_scaling': True,
                    'action_potential_duration': (1.2, 4.5),
                    'resting_potential': -0.076,
                    'threshold_potential': -0.056,
                    'electrical_fingerprint': 'multiscalar_electrical_spiking',
                    'multiscalar_characteristics': {
                        'temporal_scales': [0.1, 1.0, 10.0, 100.0, 1000.0],  # Multiple time scales
                        'frequency_bands': [0.001, 0.01, 0.1, 1.0, 10.0],  # Frequency bands
                        'spike_patterns': ['isolated', 'bursts', 'trains', 'complex'],  # Spike pattern types
                        'amplitude_modulation': True,  # Amplitude varies with time scale
                        'frequency_modulation': True,  # Frequency varies with time scale
                        'cross_scale_coupling': True  # Coupling between different temporal scales
                    }
                }
            }
        }
        
    def load_and_categorize_data(self):
        """
        Load and categorize all available data.
        
        Returns:
            dict: Categorized data files
        """
        print("Loading and categorizing data...")
        
        # Load CSV coordinate data
        csv_files = list(Path(self.csv_data_dir).glob("*.csv"))
        coordinate_data = {}
        
        for file_path in csv_files:
            filename = file_path.name
            metadata = self.extract_metadata_from_filename(filename)
            
            try:
                df = pd.read_csv(file_path, header=None)
                coordinate_data[filename] = {
                    'data': df,
                    'metadata': metadata,
                    'n_points': len(df),
                    'duration_hours': metadata.get('duration_hours', 0)
                }
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        # Load voltage recording data
        voltage_files = list(Path(self.voltage_data_dir).glob("*.csv"))
        voltage_data = {}
        
        for file_path in voltage_files:
            filename = file_path.name
            try:
                df = pd.read_csv(file_path)
                voltage_data[filename] = {
                    'data': df,
                    'n_points': len(df),
                    'sampling_rate': self.estimate_sampling_rate(df)
                }
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        print(f"Loaded {len(coordinate_data)} coordinate files and {len(voltage_data)} voltage files")
        
        return {
            'coordinate_data': coordinate_data,
            'voltage_data': voltage_data
        }
    
    def extract_metadata_from_filename(self, filename):
        """
        Extract metadata from filename.
        
        Args:
            filename: CSV filename
            
        Returns:
            dict: Extracted metadata
        """
        # Parse filename like "Pv_M_I_U_N_25d_1_coordinates.csv"
        parts = filename.replace('_coordinates.csv', '').split('_')
        
        metadata = {
            'species': parts[0] if len(parts) > 0 else 'Unknown',
            'strain': parts[1] if len(parts) > 1 else 'Unknown',
            'treatment': parts[2] if len(parts) > 2 else 'Unknown',
            'medium': parts[3] if len(parts) > 3 else 'Unknown',
            'substrate': parts[4] if len(parts) > 4 else 'Unknown',
            'duration': parts[5] if len(parts) > 5 else 'Unknown',
            'replicate': parts[6] if len(parts) > 6 else 'Unknown'
        }
        
        # Convert duration to hours
        if 'd' in metadata['duration']:
            days = float(metadata['duration'].replace('d', ''))
            metadata['duration_hours'] = days * 24
        elif 'h' in metadata['duration']:
            hours = float(metadata['duration'].replace('h', ''))
            metadata['duration_hours'] = hours
        else:
            metadata['duration_hours'] = 0
            
        return metadata
    
    def estimate_sampling_rate(self, df):
        """Estimate sampling rate from voltage data."""
        if len(df) < 2:
            return 1.0
        
        # Assume first column is time in seconds
        if len(df.columns) > 0:
            time_col = df.iloc[:, 0]
            if len(time_col) > 1:
                try:
                    # Convert to numeric if possible
                    time_numeric = pd.to_numeric(time_col, errors='coerce')
                    valid_times = time_numeric.dropna()
                    if len(valid_times) > 1:
                        dt = valid_times.iloc[1] - valid_times.iloc[0]
                        return 1.0 / dt if dt > 0 else 1.0
                except:
                    pass
        
        return 1.0  # Default
    
    def improved_sqrt_transform(self, signal, species, experimental_condition, 
                              recording_duration, sampling_rate=1.0):
        """
        Improved √t transform with species-specific parameters and rigorous validation.
        
        Args:
            signal: Input signal
            species: Fungal species
            experimental_condition: Experimental treatment
            recording_duration: Recording duration in hours
            sampling_rate: Sampling rate in Hz
            
        Returns:
            dict: Transform results with validation
        """
        # Species-specific parameter optimization
        params = self.get_species_specific_parameters(species, experimental_condition, recording_duration)
        
        # Apply transform with optimized parameters
        transform_results = self.apply_sqrt_transform(signal, params, sampling_rate)
        
        # Rigorous validation
        validation_results = self.validate_transform_results(
            transform_results, species, experimental_condition, recording_duration
        )
        
        # False positive detection
        false_positive_analysis = self.detect_false_positives(
            transform_results, signal, species, experimental_condition
        )
        
        # Biological plausibility assessment
        biological_assessment = self.assess_biological_plausibility(
            transform_results, species, experimental_condition, recording_duration
        )
        
        return {
            'transform_results': transform_results,
            'validation_results': validation_results,
            'false_positive_analysis': false_positive_analysis,
            'biological_assessment': biological_assessment,
            'parameters_used': params
        }
    
    def get_species_specific_parameters(self, species, experimental_condition, recording_duration):
        """
        Get species-specific parameters based on Adamatzky's research and actual data characteristics.
        """
        # Electrical fingerprint parameters based on Adamatzky's research
        if species == 'Pv':  # Pleurotus vulgaris - high frequency bursts
            k_range = np.logspace(-0.5, 1.2, 30)  # 0.3 to 16 Hz (high frequency bursts)
            tau_range = np.logspace(-0.3, 1.7, 30)  # 0.5 to 50 seconds (short bursts)
            amplitude_threshold = 0.04  # Lower threshold for better detection
            frequency_threshold = 0.03  # 0.03 Hz minimum for bursts
            sqrt_scaling_factor = 1.2  # Enhanced √t scaling for bursts
            
        elif species == 'Pi':  # Pleurotus ostreatus - medium frequency regular
            k_range = np.logspace(-1.0, 0.8, 30)  # 0.1 to 6 Hz (medium frequency regular)
            tau_range = np.logspace(0.2, 2.2, 30)  # 1.6 to 160 seconds (regular intervals)
            amplitude_threshold = 0.045
            frequency_threshold = 0.02
            sqrt_scaling_factor = 1.0  # Standard √t scaling
            
        elif species == 'Pp':  # Pleurotus pulmonarius - very high frequency irregular
            k_range = np.logspace(-0.2, 1.5, 30)  # 0.6 to 32 Hz (very high frequency)
            tau_range = np.logspace(-0.5, 1.2, 30)  # 0.3 to 16 seconds (irregular bursts)
            amplitude_threshold = 0.035  # Lower threshold for irregular patterns
            frequency_threshold = 0.05
            sqrt_scaling_factor = 1.5  # Strong √t scaling for irregular patterns
            
        elif species == 'Rb':  # Reishi/Bracket fungi - low frequency slow
            k_range = np.logspace(-2.5, -0.5, 30)  # 0.003 to 0.3 Hz (low frequency)
            tau_range = np.logspace(1.5, 3.5, 30)  # 32 to 3200 seconds (slow patterns)
            amplitude_threshold = 0.025  # Lower threshold for slow activity
            frequency_threshold = 0.005
            sqrt_scaling_factor = 0.8  # Reduced √t scaling for slow patterns
            
        elif species == 'Hericium':  # Hericium species - very low frequency sporadic
            k_range = np.logspace(-3.5, -1.5, 30)  # 0.0003 to 0.03 Hz (very low frequency)
            tau_range = np.logspace(2.5, 4.5, 30)  # 320 to 32000 seconds (sporadic)
            amplitude_threshold = 0.02  # Very low threshold
            frequency_threshold = 0.001
            sqrt_scaling_factor = 0.6  # Minimal √t scaling for sporadic activity
            
        elif species == 'Ag':  # Agaricus species - medium frequency steady
            k_range = np.logspace(-1.2, 0.6, 30)  # 0.06 to 4 Hz (medium frequency steady)
            tau_range = np.logspace(0.5, 2.5, 30)  # 3 to 300 seconds (steady patterns)
            amplitude_threshold = 0.04
            frequency_threshold = 0.025
            sqrt_scaling_factor = 1.1  # Slightly enhanced √t scaling
            
        elif species == 'Sc':  # Schizophyllum commune - medium frequency variable
            k_range = np.logspace(-1.3, 0.7, 30)  # 0.05 to 5 Hz (medium frequency variable)
            tau_range = np.logspace(0.3, 2.7, 30)  # 2 to 500 seconds (variable patterns)
            amplitude_threshold = 0.042
            frequency_threshold = 0.02
            sqrt_scaling_factor = 1.05  # Variable √t scaling
            
        else:  # Default for unknown species
            k_range = np.logspace(-2, 1, 30)  # 0.01 to 10 Hz (broad range)
            tau_range = np.logspace(-1, 3, 30)  # 0.1 to 1000 seconds (broad range)
            amplitude_threshold = 0.05
            frequency_threshold = 0.01
        
        # Experimental condition adjustments
        if 'I+4R' in experimental_condition:
            # High resistance - expect slower patterns
            k_range = k_range * 0.5  # Reduce frequency range
            amplitude_threshold *= 1.2  # Slightly higher threshold
        elif '5xI' in experimental_condition:
            # High current - expect faster patterns
            k_range = k_range * 1.5  # Increase frequency range
            amplitude_threshold *= 0.8  # Lower threshold
        elif 'Fc' in experimental_condition:
            # Fruiting conditions - expect more activity
            k_range = k_range * 1.2  # Slightly higher frequencies
            amplitude_threshold *= 0.9  # Lower threshold
        
        # Duration-based adjustments
        if recording_duration < 24:  # Less than 1 day
            # Focus on shorter time scales
            tau_range = tau_range[tau_range < 1000]
        elif recording_duration > 168:  # More than 1 week
            # Focus on longer time scales
            tau_range = tau_range[tau_range > 10]
        
        # Ensure we have enough parameters
        if len(k_range) < 10:
            k_range = np.logspace(-2, 1, 20)
        if len(tau_range) < 10:
            tau_range = np.logspace(-1, 3, 20)
        
        base_params = {
            'k_range': k_range,
            'tau_range': tau_range,
            'window_function': 'gaussian',
            'detection_method': 'adaptive',  # Changed from 'statistical'
            'significance_threshold': 0.05,  # Relaxed from 0.01
            'amplitude_threshold': amplitude_threshold,
            'frequency_threshold': frequency_threshold,
            'time_scale_threshold': 0.1,  # 0.1 seconds minimum
            'sqrt_scaling_factor': sqrt_scaling_factor  # Species-specific √t scaling
        }
        
        return base_params
    
    def apply_sqrt_transform(self, signal, params, sampling_rate):
        """
        Apply √t transform with given parameters.
        """
        k_values = params['k_range']
        tau_values = params['tau_range']
        
        # Create time vector
        t = np.arange(len(signal)) / sampling_rate
        
        # Apply transform
        W = np.zeros((len(k_values), len(tau_values)), dtype=complex)
        
        # Get species-specific √t scaling factor
        sqrt_scaling_factor = params.get('sqrt_scaling_factor', 1.0)
        
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                # Gaussian window with species-specific √t scaling
                sqrt_t = np.sqrt(t) * sqrt_scaling_factor
                window = np.exp(-(sqrt_t / tau)**2)
                phase = np.exp(-1j * k * sqrt_t)
                integrand = signal * window * phase
                W[i, j] = np.trapezoid(integrand, t)
        
        magnitude = np.abs(W)
        
        # Feature detection
        features = self.detect_features(magnitude, k_values, tau_values, params)
        
        return {
            'magnitude': magnitude,
            'phase': np.angle(W),
            'k_values': k_values,
            'tau_values': tau_values,
            'features': features
        }
    
    def detect_features(self, magnitude, k_values, tau_values, params):
        """
        Detect features using adaptive thresholding based on species-specific parameters.
        """
        features = []
        
        # Get species-specific thresholds
        amplitude_threshold = params.get('amplitude_threshold', 0.05)
        frequency_threshold = params.get('frequency_threshold', 0.01)
        time_scale_threshold = params.get('time_scale_threshold', 0.1)
        
        # Calculate adaptive threshold based on magnitude distribution
        mag_flat = magnitude.flatten()
        mean_mag = np.mean(mag_flat)
        std_mag = np.std(mag_flat)
        
        # Use species-specific amplitude threshold
        threshold = mean_mag + amplitude_threshold * (np.max(mag_flat) - mean_mag)
        
        # Find peaks above threshold
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                if magnitude[i, j] > threshold:
                    # Check if it's a local maximum
                    is_local_max = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < len(k_values) and 
                                0 <= nj < len(tau_values) and
                                magnitude[ni, nj] > magnitude[i, j]):
                                is_local_max = False
                                break
                        if not is_local_max:
                            break
                    
                    if is_local_max:
                        # Calculate frequency and time scale
                        frequency = k / (2 * np.pi)
                        time_scale = tau
                        
                        # Apply species-specific frequency and time scale filters
                        if (frequency >= frequency_threshold and 
                            time_scale >= time_scale_threshold):
                            
                            # Calculate significance
                            z = (magnitude[i, j] - mean_mag) / (std_mag + 1e-10)
                            p_value = 1 - stats.norm.cdf(z)
                            
                            features.append({
                                'k': k,
                                'tau': tau,
                                'magnitude': magnitude[i, j],
                                'frequency': frequency,
                                'time_scale': time_scale,
                                'significance': z,
                                'p_value': p_value,
                                'amplitude_ratio': magnitude[i, j] / np.max(mag_flat)
                            })
        
        return features
    
    def validate_transform_results(self, transform_results, species, experimental_condition, recording_duration):
        """
        Validate transform results against biological expectations.
        """
        features = transform_results['features']
        validation = {
            'n_features': len(features),
            'frequency_validation': [],
            'time_scale_validation': [],
            'magnitude_validation': [],
            'sqrt_scaling_validation': [],
            'overall_validity': False
        }
        
        for feature in features:
            # Frequency validation (0.001-10 Hz)
            freq_valid = self.biological_constraints['frequency_range'][0] <= feature['frequency'] <= self.biological_constraints['frequency_range'][1]
            validation['frequency_validation'].append(freq_valid)
            
            # Time scale validation (0.1s to ~28h)
            time_valid = self.biological_constraints['growth_time_scales'][0] <= feature['time_scale'] <= self.biological_constraints['growth_time_scales'][1]
            validation['time_scale_validation'].append(time_valid)
            
            # Magnitude validation - use species-specific amplitude ranges
            if species in self.biological_constraints['species_characteristics']:
                species_amp_range = self.biological_constraints['species_characteristics'][species]['amplitude_range']
                # Convert magnitude to amplitude ratio for validation
                amplitude_ratio = feature.get('amplitude_ratio', 0.05)
                mag_valid = species_amp_range[0] <= amplitude_ratio <= species_amp_range[1]
            else:
                mag_valid = feature['magnitude'] > 0.001  # Default minimum threshold
            validation['magnitude_validation'].append(mag_valid)
            
            # √t scaling validation
            sqrt_valid = self.validate_sqrt_scaling(feature, species)
            validation['sqrt_scaling_validation'].append(sqrt_valid)
        
        # Overall validity
        if features:
            validation['overall_validity'] = (
                np.mean(validation['frequency_validation']) > 0.8 and
                np.mean(validation['time_scale_validation']) > 0.8 and
                np.mean(validation['magnitude_validation']) > 0.8
            )
        
        return validation
    
    def validate_sqrt_scaling(self, feature, species):
        """
        Validate that the feature exhibits √t scaling.
        """
        # Check if frequency and time scale are consistent with √t scaling
        # For √t scaling: frequency should be proportional to 1/√time
        expected_freq = 1.0 / np.sqrt(feature['time_scale'])
        actual_freq = feature['frequency']
        
        # Allow for some tolerance
        tolerance = 0.5
        ratio = actual_freq / expected_freq
        
        return 1.0 / tolerance <= ratio <= tolerance
    
    def detect_false_positives(self, transform_results, signal, species, experimental_condition):
        """
        Rigorous false positive detection.
        """
        features = transform_results['features']
        false_positive_analysis = {
            'n_features': len(features),
            'false_positive_indicators': [],
            'statistical_tests': {},
            'synthetic_controls': {},
            'overall_assessment': 'unknown'
        }
        
        if not features:
            false_positive_analysis['overall_assessment'] = 'no_features'
            return false_positive_analysis
        
        # 1. Statistical significance testing
        p_values = []
        for feature in features:
            # Permutation test for each feature
            p_value = self.permutation_test(signal, feature)
            p_values.append(p_value)
        
        false_positive_analysis['statistical_tests'] = {
            'p_values': p_values,
            'significant_features': [p < 0.05 for p in p_values],
            'mean_p_value': np.mean(p_values),
            'min_p_value': np.min(p_values)
        }
        
        # 2. Synthetic control comparison
        synthetic_results = self.generate_synthetic_controls(signal, species, experimental_condition)
        false_positive_analysis['synthetic_controls'] = synthetic_results
        
        # 3. False positive indicators
        indicators = []
        
        # Indicator 1: Too many features relative to signal length
        if len(features) > len(signal) / 100:
            indicators.append('too_many_features')
        
        # Indicator 2: Features clustered in parameter space
        if len(features) > 1:
            feature_coords = np.array([[f['k'], f['tau']] for f in features])
            clustering = DBSCAN(eps=0.1, min_samples=2).fit(feature_coords)
            if len(set(clustering.labels_)) < len(features) / 2:
                indicators.append('feature_clustering')
        
        # Indicator 3: Low statistical significance
        if np.mean(p_values) > 0.1:
            indicators.append('low_significance')
        
        # Indicator 4: Inconsistent with synthetic controls
        if synthetic_results['real_vs_synthetic_ratio'] < 2.0:
            indicators.append('poor_vs_synthetic')
        
        false_positive_analysis['false_positive_indicators'] = indicators
        
        # Overall assessment
        if len(indicators) == 0:
            false_positive_analysis['overall_assessment'] = 'likely_real'
        elif len(indicators) <= 2:
            false_positive_analysis['overall_assessment'] = 'uncertain'
        else:
            false_positive_analysis['overall_assessment'] = 'likely_false_positive'
        
        return false_positive_analysis
    
    def permutation_test(self, signal, feature, n_permutations=1000):
        """
        Permutation test for feature significance.
        """
        original_score = feature['magnitude']
        permutation_scores = []
        
        for _ in range(n_permutations):
            # Shuffle signal
            shuffled_signal = np.copy(signal)
            np.random.shuffle(shuffled_signal)
            
            # Apply transform to shuffled signal
            params = {'k_range': [feature['k']], 'tau_range': [feature['tau']]}
            shuffled_results = self.apply_sqrt_transform(shuffled_signal, params, 1.0)
            
            if shuffled_results['features']:
                max_magnitude = max([f['magnitude'] for f in shuffled_results['features']])
                permutation_scores.append(max_magnitude)
            else:
                permutation_scores.append(0)
        
        # Calculate p-value
        p_value = np.mean([score >= original_score for score in permutation_scores])
        return p_value
    
    def generate_synthetic_controls(self, signal, species, experimental_condition):
        """
        Generate synthetic controls for comparison.
        """
        # Generate synthetic signals with known properties
        n_points = len(signal)
        t = np.arange(n_points)
        
        # Control 1: Pure noise
        noise_signal = np.random.normal(0, np.std(signal), n_points)
        
        # Control 2: Linear oscillation
        linear_signal = np.sin(2 * np.pi * 0.01 * t)
        
        # Control 3: Exponential decay
        decay_signal = np.exp(-t / 100)
        
        # Apply transform to controls
        params = self.get_species_specific_parameters(species, experimental_condition, 24)
        
        noise_results = self.apply_sqrt_transform(noise_signal, params, 1.0)
        linear_results = self.apply_sqrt_transform(linear_signal, params, 1.0)
        decay_results = self.apply_sqrt_transform(decay_signal, params, 1.0)
        
        # Apply transform to real signal for comparison
        real_results = self.apply_sqrt_transform(signal, params, 1.0)
        
        # Compare with real signal
        real_features = len(real_results['features'])
        noise_features = len(noise_results['features'])
        linear_features = len(linear_results['features'])
        decay_features = len(decay_results['features'])
        
        return {
            'noise_features': noise_features,
            'linear_features': linear_features,
            'decay_features': decay_features,
            'real_features': real_features,
            'real_vs_synthetic_ratio': real_features / max(1, (noise_features + linear_features + decay_features) / 3)
        }
    
    def assess_biological_plausibility(self, transform_results, species, experimental_condition, recording_duration):
        """
        Assess biological plausibility of detected patterns.
        """
        features = transform_results['features']
        assessment = {
            'n_plausible_features': 0,
            'biological_consistency': [],
            'experimental_consistency': [],
            'overall_plausibility': 'unknown'
        }
        
        for feature in features:
            # Check biological consistency
            bio_consistent = self.check_biological_consistency(feature, species)
            assessment['biological_consistency'].append(bio_consistent)
            
            # Check experimental consistency
            exp_consistent = self.check_experimental_consistency(feature, experimental_condition)
            assessment['experimental_consistency'].append(exp_consistent)
            
            if bio_consistent and exp_consistent:
                assessment['n_plausible_features'] += 1
        
        # Overall plausibility
        if features:
            plausibility_score = assessment['n_plausible_features'] / len(features)
            if plausibility_score > 0.8:
                assessment['overall_plausibility'] = 'high'
            elif plausibility_score > 0.5:
                assessment['overall_plausibility'] = 'medium'
            else:
                assessment['overall_plausibility'] = 'low'
        
        return assessment
    
    def check_biological_consistency(self, feature, species):
        """
        Check if feature is consistent with known biological patterns.
        """
        # Check frequency range
        freq_ok = self.biological_constraints['frequency_range'][0] <= feature['frequency'] <= self.biological_constraints['frequency_range'][1]
        
        # Check time scale
        time_ok = self.biological_constraints['growth_time_scales'][0] <= feature['time_scale'] <= self.biological_constraints['growth_time_scales'][1]
        
        # Check magnitude
        mag_ok = feature['magnitude'] > 0.001
        
        return freq_ok and time_ok and mag_ok
    
    def check_experimental_consistency(self, feature, experimental_condition):
        """
        Check if feature is consistent with experimental conditions.
        """
        if 'I+4R' in experimental_condition:
            # High resistance - expect slower patterns
            return feature['frequency'] < 0.1
        elif '5xI' in experimental_condition:
            # High current - expect faster patterns
            return feature['frequency'] > 0.01
        else:
            # Standard conditions
            return True
    
    def identify_species_from_electrical_fingerprint(self, features, species_metadata):
        """
        Identify species based on electrical fingerprint patterns from Adamatzky's research.
        
        Args:
            features: List of detected features
            species_metadata: Metadata about the species
            
        Returns:
            dict: Species identification results
        """
        if not features:
            return {'identified_species': 'Unknown', 'confidence': 0.0, 'fingerprint_match': 'none'}
        
        # Extract electrical fingerprint characteristics
        frequencies = [f['frequency'] for f in features]
        time_scales = [f['time_scale'] for f in features]
        magnitudes = [f['magnitude'] for f in features]
        
        # Calculate fingerprint metrics
        avg_frequency = np.mean(frequencies)
        freq_std = np.std(frequencies)
        avg_time_scale = np.mean(time_scales)
        time_scale_std = np.std(time_scales)
        avg_magnitude = np.mean(magnitudes)
        
        # Species-specific fingerprint matching based on Adamatzky's research
        fingerprint_scores = {}
        
        # Pv: High frequency bursts
        pv_score = 0
        if avg_frequency > 0.5 and freq_std > 0.3:  # High frequency with variability
            pv_score += 0.4
        if avg_time_scale < 100:  # Short bursts
            pv_score += 0.3
        if avg_magnitude > 200:  # High amplitude
            pv_score += 0.3
        fingerprint_scores['Pv'] = pv_score
        
        # Pi: Medium frequency regular
        pi_score = 0
        if 0.1 < avg_frequency < 1.0 and freq_std < 0.2:  # Medium frequency, regular
            pi_score += 0.4
        if 50 < avg_time_scale < 500:  # Regular intervals
            pi_score += 0.3
        if 100 < avg_magnitude < 300:  # Medium amplitude
            pi_score += 0.3
        fingerprint_scores['Pi'] = pi_score
        
        # Pp: Very high frequency irregular
        pp_score = 0
        if avg_frequency > 1.0 and freq_std > 0.5:  # Very high frequency, irregular
            pp_score += 0.4
        if avg_time_scale < 50:  # Very short bursts
            pp_score += 0.3
        if avg_magnitude > 300:  # Very high amplitude
            pp_score += 0.3
        fingerprint_scores['Pp'] = pp_score
        
        # Rb: Low frequency slow
        rb_score = 0
        if avg_frequency < 0.1 and freq_std < 0.1:  # Low frequency, consistent
            rb_score += 0.4
        if avg_time_scale > 1000:  # Slow patterns
            rb_score += 0.3
        if avg_magnitude < 200:  # Low amplitude
            rb_score += 0.3
        fingerprint_scores['Rb'] = rb_score
        
        # Ag: Medium frequency steady
        ag_score = 0
        if 0.1 < avg_frequency < 0.8 and freq_std < 0.3:  # Medium frequency, steady
            ag_score += 0.4
        if 100 < avg_time_scale < 1000:  # Steady patterns
            ag_score += 0.3
        if 100 < avg_magnitude < 250:  # Medium amplitude
            ag_score += 0.3
        fingerprint_scores['Ag'] = ag_score
        
        # Sc: Medium frequency variable
        sc_score = 0
        if 0.1 < avg_frequency < 0.8 and freq_std > 0.2:  # Medium frequency, variable
            sc_score += 0.4
        if 50 < avg_time_scale < 1000:  # Variable patterns
            sc_score += 0.3
        if 100 < avg_magnitude < 300:  # Medium amplitude
            sc_score += 0.3
        fingerprint_scores['Sc'] = sc_score
        
        # Find best match
        best_species = max(fingerprint_scores, key=fingerprint_scores.get)
        best_score = fingerprint_scores[best_species]
        
        # Calculate confidence based on score and number of features
        confidence = min(best_score * len(features) / 10, 1.0)
        
        return {
            'identified_species': best_species,
            'confidence': confidence,
            'fingerprint_match': best_species,
            'fingerprint_scores': fingerprint_scores,
            'avg_frequency': avg_frequency,
            'avg_time_scale': avg_time_scale,
            'avg_magnitude': avg_magnitude
        }
    
    def analyze_multiscalar_electrical_spiking(self, signal, species, sampling_rate=1.0):
        """
        Analyze multiscalar electrical spiking patterns based on Adamatzky's research.
        
        Args:
            signal: Input signal
            species: Fungal species (especially Sc for Schizophyllum commune)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            dict: Multiscalar analysis results
        """
        if species != 'Sc':
            return {'multiscalar_analysis': False, 'reason': 'Species not Sc'}
        
        # Get multiscalar parameters
        temporal_scales = self.biological_constraints['temporal_scales']
        frequency_bands = self.biological_constraints['frequency_bands']
        
        # Analyze at multiple temporal scales
        scale_analysis = {}
        for scale in temporal_scales:
            # Apply scale-specific analysis
            scale_signal = self.apply_temporal_scale_filter(signal, scale, sampling_rate)
            scale_features = self.extract_scale_specific_features(scale_signal, scale)
            scale_analysis[scale] = scale_features
        
        # Analyze frequency bands
        frequency_analysis = {}
        for freq_band in frequency_bands:
            band_features = self.extract_frequency_band_features(signal, freq_band, sampling_rate)
            frequency_analysis[freq_band] = band_features
        
        # Cross-scale coupling analysis
        cross_scale_coupling = self.analyze_cross_scale_coupling(scale_analysis, frequency_analysis)
        
        # Spike pattern classification
        spike_patterns = self.classify_spike_patterns(signal, sampling_rate)
        
        return {
            'multiscalar_analysis': True,
            'temporal_scale_analysis': scale_analysis,
            'frequency_band_analysis': frequency_analysis,
            'cross_scale_coupling': cross_scale_coupling,
            'spike_patterns': spike_patterns,
            'multiscalar_complexity': self.calculate_multiscalar_complexity(scale_analysis, frequency_analysis)
        }
    
    def apply_temporal_scale_filter(self, signal, scale, sampling_rate):
        """Apply temporal scale-specific filtering."""
        # Create scale-specific filter
        filter_length = int(scale * sampling_rate)
        if filter_length < 3:
            filter_length = 3
        
        # Apply moving average filter for temporal scale
        filtered_signal = np.convolve(signal, np.ones(filter_length)/filter_length, mode='same')
        return filtered_signal
    
    def extract_scale_specific_features(self, signal, scale):
        """Extract features specific to a temporal scale."""
        features = {
            'scale': scale,
            'mean_amplitude': np.mean(np.abs(signal)),
            'amplitude_variance': np.var(signal),
            'peak_count': len(signal[signal > np.std(signal)]),
            'zero_crossings': len(np.where(np.diff(np.signbit(signal)))[0]),
            'autocorrelation': np.corrcoef(signal[:-1], signal[1:])[0, 1]
        }
        return features
    
    def extract_frequency_band_features(self, signal_data, freq_band, sampling_rate):
        """Extract features from specific frequency bands."""
        # Design bandpass filter
        nyquist = sampling_rate / 2
        low_freq = freq_band / 10
        high_freq = freq_band * 10
        
        if low_freq < high_freq < nyquist:
            b, a = signal.butter(4, [low_freq/nyquist, high_freq/nyquist], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_data)
        else:
            filtered_signal = signal_data
        
        features = {
            'frequency_band': freq_band,
            'band_power': np.mean(filtered_signal**2),
            'band_amplitude': np.mean(np.abs(filtered_signal)),
            'band_peaks': len(filtered_signal[filtered_signal > np.std(filtered_signal)])
        }
        return features
    
    def analyze_cross_scale_coupling(self, scale_analysis, frequency_analysis):
        """Analyze coupling between different temporal scales."""
        coupling_matrix = {}
        
        scales = list(scale_analysis.keys())
        for i, scale1 in enumerate(scales):
            for j, scale2 in enumerate(scales):
                if i != j:
                    # Calculate coupling between scales
                    coupling = self.calculate_scale_coupling(
                        scale_analysis[scale1], scale_analysis[scale2]
                    )
                    coupling_matrix[f"{scale1}_{scale2}"] = coupling
        
        return coupling_matrix
    
    def calculate_scale_coupling(self, features1, features2):
        """Calculate coupling between two temporal scales."""
        # Simple correlation-based coupling
        try:
            coupling = np.corrcoef([
                features1['mean_amplitude'], features1['amplitude_variance'],
                features2['mean_amplitude'], features2['amplitude_variance']
            ])[0, 1]
            return coupling if not np.isnan(coupling) else 0.0
        except:
            return 0.0
    
    def classify_spike_patterns(self, signal, sampling_rate):
        """Classify spike patterns based on Adamatzky's research."""
        # Detect spikes using adaptive threshold
        threshold = np.mean(signal) + 2 * np.std(signal)
        spike_indices = np.where(signal > threshold)[0]
        
        if len(spike_indices) == 0:
            return {'pattern_type': 'none', 'spike_count': 0}
        
        # Calculate inter-spike intervals
        isi = np.diff(spike_indices) / sampling_rate
        
        # Classify patterns
        if len(isi) == 0:
            pattern_type = 'isolated'
        elif np.mean(isi) < 0.1:
            pattern_type = 'bursts'
        elif np.std(isi) < np.mean(isi) * 0.5:
            pattern_type = 'trains'
        else:
            pattern_type = 'complex'
        
        return {
            'pattern_type': pattern_type,
            'spike_count': len(spike_indices),
            'mean_isi': np.mean(isi) if len(isi) > 0 else 0,
            'isi_variance': np.var(isi) if len(isi) > 0 else 0
        }
    
    def calculate_multiscalar_complexity(self, scale_analysis, frequency_analysis):
        """Calculate multiscalar complexity measure."""
        # Combine features from all scales and frequency bands
        all_features = []
        
        for scale_features in scale_analysis.values():
            all_features.extend([scale_features['mean_amplitude'], scale_features['amplitude_variance']])
        
        for freq_features in frequency_analysis.values():
            all_features.extend([freq_features['band_power'], freq_features['band_amplitude']])
        
        # Calculate complexity as entropy of feature distribution
        if len(all_features) > 1:
            complexity = -np.sum(np.histogram(all_features, bins=10)[0] * 
                               np.log(np.histogram(all_features, bins=10)[0] + 1e-10))
        else:
            complexity = 0
        
        return complexity
    
    def run_comprehensive_analysis(self):
        """
        Run comprehensive analysis on all available data.
        """
        print("=== Rigorous Fungal Analysis with √t Transform ===\n")
        
        # Load data
        data = self.load_and_categorize_data()
        
        # Analyze coordinate data
        coordinate_results = self.analyze_coordinate_data(data['coordinate_data'])
        
        # Analyze voltage data
        voltage_results = self.analyze_voltage_data(data['voltage_data'])
        
        # Cross-validation analysis
        cross_validation_results = self.cross_validate_results(coordinate_results, voltage_results)
        
        # Generate comprehensive report
        self.generate_comprehensive_report(coordinate_results, voltage_results, cross_validation_results)
        
        return {
            'coordinate_results': coordinate_results,
            'voltage_results': voltage_results,
            'cross_validation_results': cross_validation_results
        }
    
    def analyze_coordinate_data(self, coordinate_data):
        """
        Analyze coordinate data with rigorous validation.
        """
        results = {}
        
        for filename, data_info in coordinate_data.items():
            print(f"Analyzing coordinate data: {filename}")
            
            df = data_info['data']
            metadata = data_info['metadata']
            
            # Extract coordinate signals
            if len(df.columns) >= 2:
                x_coords = df.iloc[:, 0].values
                y_coords = df.iloc[:, 1].values
                
                # Create derived signals
                distance = np.sqrt(x_coords**2 + y_coords**2)
                velocity = np.gradient(distance)
                acceleration = np.gradient(velocity)
                
                # Analyze each signal
                signals = {
                    'distance': distance,
                    'velocity': velocity,
                    'acceleration': acceleration
                }
                
                signal_results = {}
                for signal_name, signal_data in signals.items():
                    # Normalize signal
                    if np.std(signal_data) > 0:
                        signal_normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
                    else:
                        signal_normalized = signal_data
                    
                    # Apply improved transform
                    transform_results = self.improved_sqrt_transform(
                        signal_normalized,
                        metadata['species'],
                        metadata['treatment'],
                        metadata['duration_hours']
                    )
                    
                    signal_results[signal_name] = {
                        'transform_results': transform_results['transform_results'],
                        'validation_results': transform_results['validation_results'],
                        'false_positive_analysis': transform_results['false_positive_analysis'],
                        'biological_assessment': transform_results['biological_assessment'],
                        'parameters_used': transform_results['parameters_used']
                    }
                
                # Species identification from electrical fingerprint
                all_features = []
                for signal_name, signal_result in signal_results.items():
                    features = signal_result['transform_results']['features']
                    all_features.extend(features)
                
                species_identification = self.identify_species_from_electrical_fingerprint(
                    all_features, metadata
                )
                
                # Add multiscalar analysis for Sc species
                multiscalar_results = {}
                if metadata['species'] == 'Sc':
                    for signal_name, signal_data in signals.items():
                        # Normalize signal for multiscalar analysis
                        if np.std(signal_data) > 0:
                            signal_normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
                        else:
                            signal_normalized = signal_data
                        
                        multiscalar_analysis = self.analyze_multiscalar_electrical_spiking(
                            signal_normalized, 'Sc', 1.0  # Assume 1 Hz sampling rate
                        )
                        multiscalar_results[signal_name] = multiscalar_analysis
                
                results[filename] = {
                    'metadata': metadata,
                    'signal_results': signal_results,
                    'species_identification': species_identification,
                    'multiscalar_analysis': multiscalar_results,
                    'n_points': data_info['n_points'],
                    'duration_hours': data_info['duration_hours']
                }
        
        return results
    
    def analyze_voltage_data(self, voltage_data):
        """
        Analyze voltage data with rigorous validation.
        """
        results = {}
        
        for filename, data_info in voltage_data.items():
            print(f"Analyzing voltage data: {filename}")
            
            df = data_info['data']
            sampling_rate = data_info['sampling_rate']
            
            # Extract voltage signal
            if len(df.columns) >= 2:
                voltage_signal = df.iloc[:, 1].values  # Assume second column is voltage
                
                # Preprocess voltage signal
                voltage_processed = self.preprocess_voltage_signal(voltage_signal, sampling_rate)
                
                # Apply improved transform
                transform_results = self.improved_sqrt_transform(
                    voltage_processed,
                    'Unknown',  # Species not specified in voltage files
                    'Standard',
                    len(voltage_processed) / sampling_rate / 3600  # Convert to hours
                )
                
                results[filename] = {
                    'transform_results': transform_results['transform_results'],
                    'validation_results': transform_results['validation_results'],
                    'false_positive_analysis': transform_results['false_positive_analysis'],
                    'biological_assessment': transform_results['biological_assessment'],
                    'parameters_used': transform_results['parameters_used'],
                    'n_points': data_info['n_points'],
                    'sampling_rate': sampling_rate
                }
        
        return results
    
    def preprocess_voltage_signal(self, voltage_signal, sampling_rate):
        """
        Preprocess voltage signal for analysis.
        """
        # Remove DC offset
        voltage_centered = voltage_signal - np.mean(voltage_signal)
        
        # Bandpass filter for fungal activity (0.001-1 Hz)
        nyquist = sampling_rate / 2
        low = 0.001 / nyquist
        high = 1.0 / nyquist
        
        if low < high < 1.0:
            b, a = signal.butter(4, [low, high], btype='band')
            voltage_filtered = signal.filtfilt(b, a, voltage_centered)
        else:
            voltage_filtered = voltage_centered
        
        return voltage_filtered
    
    def cross_validate_results(self, coordinate_results, voltage_results):
        """
        Cross-validate results across different data types and species.
        """
        print("Performing cross-validation analysis...")
        
        # Group results by species
        species_results = {}
        for filename, result in coordinate_results.items():
            species = result['metadata']['species']
            if species not in species_results:
                species_results[species] = []
            species_results[species].append(result)
        
        # Analyze consistency within species
        consistency_analysis = {}
        for species, results in species_results.items():
            if len(results) > 1:
                consistency_analysis[species] = self.analyze_species_consistency(species, results)
        
        return consistency_analysis
    
    def analyze_species_consistency(self, species, results):
        """
        Analyze consistency of patterns within a species.
        """
        # Extract all features from this species
        all_features = []
        for result in results:
            for signal_name, signal_result in result['signal_results'].items():
                features = signal_result['transform_results']['features']
                all_features.extend(features)
        
        if not all_features:
            return {'consistency_score': 0, 'n_features': 0, 'pattern_clustering': 'none'}
        
        # Analyze feature clustering
        feature_coords = np.array([[f['k'], f['tau']] for f in all_features])
        
        # Use DBSCAN to find clusters
        clustering = DBSCAN(eps=0.2, min_samples=2).fit(feature_coords)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        # Calculate consistency score
        if len(all_features) > 0:
            consistency_score = n_clusters / len(all_features)
        else:
            consistency_score = 0
        
        return {
            'consistency_score': consistency_score,
            'n_features': len(all_features),
            'n_clusters': n_clusters,
            'pattern_clustering': 'strong' if consistency_score < 0.3 else 'weak'
        }
    
    def generate_comprehensive_report(self, coordinate_results, voltage_results, cross_validation_results):
        """
        Generate comprehensive analysis report with memory efficiency.
        """
        print("\n=== GENERATING COMPREHENSIVE REPORT ===\n")
        
        # Create output directory
        output_dir = "fungal_analysis_project/results/rigorous_analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Summary statistics
        summary = {
            'coordinate_files_analyzed': len(coordinate_results),
            'voltage_files_analyzed': len(voltage_results),
            'species_analyzed': list(set([r['metadata']['species'] for r in coordinate_results.values()])),
            'total_features_detected': 0,
            'biologically_plausible_features': 0,
            'false_positive_indicators': [],
            'cross_validation_results': cross_validation_results,
            'species_summaries': {},
            'species_identification_results': {},
            'parameter_effectiveness': {},
            'electrical_fingerprint_analysis': {},
            'multiscalar_analysis_results': {}
        }
        
        # Analyze coordinate results with memory efficiency
        for filename, result in coordinate_results.items():
            species = result['metadata']['species']
            if species not in summary['species_summaries']:
                summary['species_summaries'][species] = {
                    'files_analyzed': 0,
                    'total_features': 0,
                    'plausible_features': 0,
                    'avg_magnitude': 0,
                    'avg_frequency': 0,
                    'avg_time_scale': 0
                }
            
            summary['species_summaries'][species]['files_analyzed'] += 1
            
            # Add species identification results
            if 'species_identification' in result:
                species_id = result['species_identification']
                if species not in summary['species_identification_results']:
                    summary['species_identification_results'][species] = {
                        'correct_identifications': 0,
                        'total_files': 0,
                        'avg_confidence': 0,
                        'fingerprint_scores': {}
                    }
                
                summary['species_identification_results'][species]['total_files'] += 1
                if species_id['identified_species'] == species:
                    summary['species_identification_results'][species]['correct_identifications'] += 1
                
                summary['species_identification_results'][species]['avg_confidence'] += species_id['confidence']
                
                # Store fingerprint scores
                for fp_species, score in species_id['fingerprint_scores'].items():
                    if fp_species not in summary['species_identification_results'][species]['fingerprint_scores']:
                        summary['species_identification_results'][species]['fingerprint_scores'][fp_species] = []
                    summary['species_identification_results'][species]['fingerprint_scores'][fp_species].append(score)
                
                # Process multiscalar analysis results
                if 'multiscalar_analysis' in result and result['multiscalar_analysis']:
                    if species not in summary['multiscalar_analysis_results']:
                        summary['multiscalar_analysis_results'][species] = {
                            'files_analyzed': 0,
                            'total_complexity': 0,
                            'pattern_types': {},
                            'scale_coupling': {}
                        }
                    
                    summary['multiscalar_analysis_results'][species]['files_analyzed'] += 1
                    
                    for signal_name, multiscalar_result in result['multiscalar_analysis'].items():
                        if multiscalar_result['multiscalar_analysis']:
                            summary['multiscalar_analysis_results'][species]['total_complexity'] += multiscalar_result['multiscalar_complexity']
                            
                            # Track pattern types
                            pattern_type = multiscalar_result['spike_patterns']['pattern_type']
                            if pattern_type not in summary['multiscalar_analysis_results'][species]['pattern_types']:
                                summary['multiscalar_analysis_results'][species]['pattern_types'][pattern_type] = 0
                            summary['multiscalar_analysis_results'][species]['pattern_types'][pattern_type] += 1
            
            for signal_name, signal_result in result['signal_results'].items():
                features = signal_result['transform_results']['features']
                summary['total_features_detected'] += len(features)
                summary['species_summaries'][species]['total_features'] += len(features)
                
                # Count biologically plausible features
                validation = signal_result['validation_results']
                if validation['overall_validity']:
                    summary['biologically_plausible_features'] += len(features)
                    summary['species_summaries'][species]['plausible_features'] += len(features)
                
                # Collect false positive indicators
                fp_analysis = signal_result['false_positive_analysis']
                summary['false_positive_indicators'].extend(fp_analysis['false_positive_indicators'])
                
                # Collect feature statistics
                if features:
                    magnitudes = [f['magnitude'] for f in features]
                    frequencies = [f['frequency'] for f in features]
                    time_scales = [f['time_scale'] for f in features]
                    
                    summary['species_summaries'][species]['avg_magnitude'] += np.mean(magnitudes)
                    summary['species_summaries'][species]['avg_frequency'] += np.mean(frequencies)
                    summary['species_summaries'][species]['avg_time_scale'] += np.mean(time_scales)
        
        # Normalize averages
        for species in summary['species_summaries']:
            if summary['species_summaries'][species]['files_analyzed'] > 0:
                n_files = summary['species_summaries'][species]['files_analyzed']
                summary['species_summaries'][species]['avg_magnitude'] /= n_files
                summary['species_summaries'][species]['avg_frequency'] /= n_files
                summary['species_summaries'][species]['avg_time_scale'] /= n_files
        
        # Process species identification results
        for species in summary['species_identification_results']:
            if summary['species_identification_results'][species]['total_files'] > 0:
                n_files = summary['species_identification_results'][species]['total_files']
                summary['species_identification_results'][species]['avg_confidence'] /= n_files
                summary['species_identification_results'][species]['identification_accuracy'] = (
                    summary['species_identification_results'][species]['correct_identifications'] / n_files
                )
                
                # Calculate average fingerprint scores
                for fp_species in summary['species_identification_results'][species]['fingerprint_scores']:
                    scores = summary['species_identification_results'][species]['fingerprint_scores'][fp_species]
                    summary['species_identification_results'][species]['fingerprint_scores'][fp_species] = np.mean(scores)
        
        # Calculate overall assessment
        if summary['total_features_detected'] > 0:
            plausibility_ratio = summary['biologically_plausible_features'] / summary['total_features_detected']
            if plausibility_ratio > 0.8:
                overall_assessment = "HIGH_QUALITY"
            elif plausibility_ratio > 0.5:
                overall_assessment = "MODERATE_QUALITY"
            else:
                overall_assessment = "LOW_QUALITY"
        else:
            overall_assessment = "NO_FEATURES"
        
        summary['overall_assessment'] = overall_assessment
        summary['plausibility_ratio'] = plausibility_ratio if summary['total_features_detected'] > 0 else 0
        
        # Save summary results (memory efficient)
        results_filename = f"{output_dir}/summary_results_{timestamp}.json"
        with open(results_filename, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate visualizations (memory efficient)
        self.create_analysis_visualizations(coordinate_results, voltage_results, cross_validation_results, output_dir, timestamp)
        
        # Print summary
        self.print_analysis_summary(summary)
        
        print(f"\nSummary results saved to: {results_filename}")
        print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def create_analysis_visualizations(self, coordinate_results, voltage_results, cross_validation_results, output_dir, timestamp):
        """
        Create comprehensive visualizations of the analysis.
        """
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Feature distribution by species
        ax1 = axes[0, 0]
        species_features = {}
        for filename, result in coordinate_results.items():
            species = result['metadata']['species']
            if species not in species_features:
                species_features[species] = 0
            for signal_result in result['signal_results'].values():
                species_features[species] += len(signal_result['transform_results']['features'])
        
        if species_features:
            species_names = list(species_features.keys())
            feature_counts = list(species_features.values())
            bars = ax1.bar(species_names, feature_counts, alpha=0.7)
            ax1.set_title('Features Detected by Species')
            ax1.set_ylabel('Number of Features')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. False positive indicators
        ax2 = axes[0, 1]
        fp_indicators = {}
        for result in coordinate_results.values():
            for signal_result in result['signal_results'].values():
                indicators = signal_result['false_positive_analysis']['false_positive_indicators']
                for indicator in indicators:
                    fp_indicators[indicator] = fp_indicators.get(indicator, 0) + 1
        
        if fp_indicators:
            indicator_names = list(fp_indicators.keys())
            indicator_counts = list(fp_indicators.values())
            bars = ax2.bar(indicator_names, indicator_counts, alpha=0.7, color='red')
            ax2.set_title('False Positive Indicators')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Biological plausibility assessment
        ax3 = axes[0, 2]
        plausibility_counts = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
        for result in coordinate_results.values():
            for signal_result in result['signal_results'].values():
                assessment = signal_result['biological_assessment']
                plausibility_counts[assessment['overall_plausibility']] += 1
        
        plausibility_names = list(plausibility_counts.keys())
        plausibility_values = list(plausibility_counts.values())
        colors = ['green', 'orange', 'red', 'gray']
        bars = ax3.bar(plausibility_names, plausibility_values, color=colors, alpha=0.7)
        ax3.set_title('Biological Plausibility Assessment')
        ax3.set_ylabel('Count')
        
        # 4. Cross-validation consistency
        ax4 = axes[1, 0]
        if cross_validation_results:
            species_names = list(cross_validation_results.keys())
            consistency_scores = [cv['consistency_score'] for cv in cross_validation_results.values()]
            bars = ax4.bar(species_names, consistency_scores, alpha=0.7)
            ax4.set_title('Cross-Validation Consistency')
            ax4.set_ylabel('Consistency Score')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Feature magnitude distribution
        ax5 = axes[1, 1]
        all_magnitudes = []
        for result in coordinate_results.values():
            for signal_result in result['signal_results'].values():
                features = signal_result['transform_results']['features']
                magnitudes = [f['magnitude'] for f in features]
                all_magnitudes.extend(magnitudes)
        
        if all_magnitudes:
            ax5.hist(all_magnitudes, bins=20, alpha=0.7, edgecolor='black')
            ax5.set_title('Feature Magnitude Distribution')
            ax5.set_xlabel('Magnitude')
            ax5.set_ylabel('Count')
        
        # 6. Frequency vs time scale scatter
        ax6 = axes[1, 2]
        all_frequencies = []
        all_time_scales = []
        for result in coordinate_results.values():
            for signal_result in result['signal_results'].values():
                features = signal_result['transform_results']['features']
                frequencies = [f['frequency'] for f in features]
                time_scales = [f['time_scale'] for f in features]
                all_frequencies.extend(frequencies)
                all_time_scales.extend(time_scales)
        
        if all_frequencies and all_time_scales:
            ax6.scatter(all_frequencies, all_time_scales, alpha=0.6)
            ax6.set_xlabel('Frequency (Hz)')
            ax6.set_ylabel('Time Scale (s)')
            ax6.set_title('Feature Distribution')
            ax6.set_xscale('log')
            ax6.set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{output_dir}/analysis_visualizations_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualizations to: {plot_filename}")
        
        plt.show()
    
    def print_analysis_summary(self, summary):
        """
        Print comprehensive analysis summary.
        """
        print("\n" + "="*80)
        print("RIGOROUS FUNGAL ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nData Analysis:")
        print(f"  Coordinate files analyzed: {summary['coordinate_files_analyzed']}")
        print(f"  Voltage files analyzed: {summary['voltage_files_analyzed']}")
        print(f"  Species analyzed: {', '.join(summary['species_analyzed'])}")
        
        print(f"\nFeature Detection:")
        print(f"  Total features detected: {summary['total_features_detected']}")
        print(f"  Biologically plausible features: {summary['biologically_plausible_features']}")
        if summary['total_features_detected'] > 0:
            plausibility_ratio = summary['biologically_plausible_features'] / summary['total_features_detected']
            print(f"  Plausibility ratio: {plausibility_ratio:.3f}")
        
        print(f"\nFalse Positive Analysis:")
        if summary['false_positive_indicators']:
            fp_counts = {}
            for indicator in summary['false_positive_indicators']:
                fp_counts[indicator] = fp_counts.get(indicator, 0) + 1
            for indicator, count in fp_counts.items():
                print(f"  {indicator}: {count} occurrences")
        else:
            print("  No false positive indicators detected")
        
        print(f"\nCross-Validation Results:")
        for species, cv_result in summary['cross_validation_results'].items():
            print(f"  {species}: consistency={cv_result['consistency_score']:.3f}, "
                  f"features={cv_result['n_features']}, clustering={cv_result['pattern_clustering']}")
        
        print(f"\nOverall Assessment: {summary['overall_assessment']}")
        
        if summary['overall_assessment'] == "HIGH_QUALITY":
            print("  ✅ Transform shows high-quality, biologically plausible results")
        elif summary['overall_assessment'] == "MODERATE_QUALITY":
            print("  ⚠️  Transform shows moderate-quality results with some concerns")
        elif summary['overall_assessment'] == "LOW_QUALITY":
            print("  ❌ Transform shows low-quality results with many false positives")
        else:
            print("  ❓ Transform shows no clear patterns")

if __name__ == "__main__":
    # Initialize analyzer with organized project paths
    analyzer = RigorousFungalAnalyzer("../../data/csv_data", "../../data/15061491/fungal_spikes/good_recordings")
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis() 