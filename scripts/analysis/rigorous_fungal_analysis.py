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
        
        # Biological parameters based on Adamatzky's research and actual data
        self.biological_constraints = {
            'frequency_range': (0.001, 10.0),  # Hz - expanded range for all species
            'growth_time_scales': (0.1, 100000),  # seconds - 0.1s to ~28h
            'spike_amplitude_range': (0.02, 0.15),  # mV - based on actual data ranges
            'species_characteristics': {
                'Pv': {'growth_rate': 'fast', 'spike_freq': 'high', 'amplitude_range': (0.095, 0.115), 'sqrt_scaling': True},
                'Pi': {'growth_rate': 'fast', 'spike_freq': 'high', 'amplitude_range': (0.090, 0.110), 'sqrt_scaling': True},
                'Pp': {'growth_rate': 'fast', 'spike_freq': 'very_high', 'amplitude_range': (0.100, 0.120), 'sqrt_scaling': True},
                'Rb': {'growth_rate': 'slow', 'spike_freq': 'low', 'amplitude_range': (0.080, 0.100), 'sqrt_scaling': True},
                'Hericium': {'growth_rate': 'very_slow', 'spike_freq': 'very_low', 'amplitude_range': (0.020, 0.080), 'sqrt_scaling': True},
                'Ag': {'growth_rate': 'medium', 'spike_freq': 'medium', 'amplitude_range': (0.085, 0.105), 'sqrt_scaling': True},
                'Sc': {'growth_rate': 'medium', 'spike_freq': 'medium', 'amplitude_range': (0.085, 0.105), 'sqrt_scaling': True}
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
        # Base parameter ranges based on Adamatzky's fungal action potential research
        if species == 'Pv':  # Pleurotus vulgaris - fast-growing, high activity
            k_range = np.logspace(-1, 1, 25)  # 0.1 to 10 Hz (fast activity)
            tau_range = np.logspace(0, 2, 25)  # 1 to 100 seconds (short bursts)
            amplitude_threshold = 0.05  # 5% of max amplitude
            frequency_threshold = 0.05  # 0.05 Hz minimum
            
        elif species == 'Pi':  # Pleurotus ostreatus - medium-fast activity
            k_range = np.logspace(-1.5, 0.5, 25)  # 0.03 to 3 Hz (medium-fast)
            tau_range = np.logspace(0.5, 2.5, 25)  # 3 to 300 seconds (medium bursts)
            amplitude_threshold = 0.05
            frequency_threshold = 0.03
            
        elif species == 'Pp':  # Pleurotus pulmonarius - very fast activity
            k_range = np.logspace(-0.5, 1.5, 25)  # 0.3 to 30 Hz (very fast)
            tau_range = np.logspace(-0.5, 1.5, 25)  # 0.3 to 30 seconds (very short bursts)
            amplitude_threshold = 0.05
            frequency_threshold = 0.1
            
        elif species == 'Rb':  # Reishi/Bracket fungi - slow activity
            k_range = np.logspace(-2, 0, 25)  # 0.01 to 1 Hz (slow activity)
            tau_range = np.logspace(1, 3, 25)  # 10 to 1000 seconds (long periods)
            amplitude_threshold = 0.03  # Lower threshold for slower activity
            frequency_threshold = 0.01
            
        elif species == 'Hericium':  # Hericium species - very slow activity
            k_range = np.logspace(-3, -1, 25)  # 0.001 to 0.1 Hz (very slow)
            tau_range = np.logspace(2, 4, 25)  # 100 to 10000 seconds (very long periods)
            amplitude_threshold = 0.02  # Even lower threshold
            frequency_threshold = 0.001
            
        elif species == 'Ag':  # Agaricus species - medium activity
            k_range = np.logspace(-1.5, 0.5, 25)  # 0.03 to 3 Hz
            tau_range = np.logspace(0.5, 2.5, 25)  # 3 to 300 seconds
            amplitude_threshold = 0.05
            frequency_threshold = 0.03
            
        elif species == 'Sc':  # Schizophyllum commune - medium activity
            k_range = np.logspace(-1.5, 0.5, 25)  # 0.03 to 3 Hz
            tau_range = np.logspace(0.5, 2.5, 25)  # 3 to 300 seconds
            amplitude_threshold = 0.05
            frequency_threshold = 0.03
            
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
            'time_scale_threshold': 0.1  # 0.1 seconds minimum
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
        
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                # Gaussian window with √t scaling
                sqrt_t = np.sqrt(t)
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
                
                results[filename] = {
                    'metadata': metadata,
                    'signal_results': signal_results,
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
        output_dir = "rigorous_analysis_results"
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
            'parameter_effectiveness': {}
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
    # Initialize analyzer
    analyzer = RigorousFungalAnalyzer("csv_data", "15061491/fungal_spikes/good_recordings")
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis() 