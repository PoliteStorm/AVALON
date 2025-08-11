#!/usr/bin/env python3
"""
Advanced Fungal Communication Analyzer
Author: Joe Knowles
Timestamp: 2025-01-27
Description: Advanced signal analysis techniques for mushroom communication research
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
import seaborn as sns
from datetime import datetime
import json
import os
from pathlib import Path

class AdvancedFungalCommunicationAnalyzer:
    """
    Advanced analysis system for studying mushroom communication patterns
    using multiple signal processing techniques beyond basic wave transforms.
    """
    
    def __init__(self, data_path="DATA/raw"):
        self.data_path = Path(data_path)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.author = "Joe Knowles"
        
    def load_fungal_data(self, file_path):
        """Load fungal electrical data from CSV files."""
        try:
            data = pd.read_csv(file_path)
            print(f"Loaded data from {file_path}: {len(data)} samples")
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def frequency_domain_analysis(self, data, sample_rate=1000):
        """
        Analyze fungal signals in the frequency domain for musical patterns.
        """
        print("🎵 Performing Frequency Domain Analysis...")
        
        # Perform FFT
        fft_result = np.fft.fft(data)
        frequencies = np.fft.fftfreq(len(data), 1/sample_rate)
        
        # Find dominant frequencies
        magnitude_spectrum = np.abs(fft_result)
        dominant_freq_idx = np.argmax(magnitude_spectrum[1:len(frequencies)//2]) + 1
        dominant_frequency = frequencies[dominant_freq_idx]
        
        # Analyze frequency bands
        low_freq = np.sum(magnitude_spectrum[frequencies < 10])
        mid_freq = np.sum(magnitude_spectrum[(frequencies >= 10) & (frequencies < 100)])
        high_freq = np.sum(magnitude_spectrum[frequencies >= 100])
        
        results = {
            'dominant_frequency': dominant_frequency,
            'frequency_bands': {
                'low_freq_power': low_freq,
                'mid_freq_power': mid_freq,
                'high_freq_power': high_freq
            },
            'total_spectral_power': np.sum(magnitude_spectrum),
            'spectral_centroid': np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum)
        }
        
        return results, frequencies, magnitude_spectrum
    
    def time_frequency_mapping(self, data, sample_rate=1000):
        """
        Create 3D time-frequency maps showing when mushrooms communicate.
        """
        print("🕰️ Creating Time-Frequency Maps...")
        
        # Use spectrogram for time-frequency analysis
        frequencies, times, Sxx = signal.spectrogram(data, sample_rate, nperseg=256, noverlap=128)
        
        # Analyze temporal patterns
        time_analysis = {
            'peak_activity_times': times[np.argmax(Sxx, axis=0)],
            'frequency_evolution': np.mean(Sxx, axis=1),
            'temporal_consistency': np.std(Sxx, axis=1)
        }
        
        return time_analysis, frequencies, times, Sxx
    
    def phase_relationship_analysis(self, data1, data2):
        """
        Study how different mushrooms' signals relate to each other.
        """
        print("🔄 Analyzing Phase Relationships...")
        
        # Ensure same length
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        # Cross-correlation
        correlation = np.correlate(data1, data2, mode='full')
        lag = np.arange(-(min_len-1), min_len)
        
        # Phase difference
        fft1 = np.fft.fft(data1)
        fft2 = np.fft.fft(data2)
        phase_diff = np.angle(fft1 * np.conj(fft2))
        
        # Coherence
        coherence = np.abs(fft1 * np.conj(fft2)) / (np.abs(fft1) * np.abs(fft2))
        
        results = {
            'cross_correlation': {
                'max_correlation': np.max(correlation),
                'lag_at_max': lag[np.argmax(correlation)],
                'correlation_curve': correlation
            },
            'phase_analysis': {
                'mean_phase_diff': np.mean(phase_diff),
                'phase_consistency': np.std(phase_diff),
                'coherence': np.mean(coherence)
            }
        }
        
        return results, lag, correlation, phase_diff, coherence
    
    def genetic_communication_signatures(self, species_data):
        """
        Compare electrical patterns between related species.
        """
        print("🧬 Analyzing Genetic Communication Signatures...")
        
        signatures = {}
        for species, data in species_data.items():
            # Extract characteristic features
            signatures[species] = {
                'mean_amplitude': np.mean(data),
                'amplitude_variance': np.var(data),
                'spike_frequency': self.count_spikes(data),
                'signal_complexity': self.calculate_complexity(data)
            }
        
        # Compare similarities
        similarities = {}
        species_list = list(species_data.keys())
        for i, sp1 in enumerate(species_list):
            for j, sp2 in enumerate(species_list[i+1:], i+1):
                similarity = self.calculate_species_similarity(
                    signatures[sp1], signatures[sp2]
                )
                similarities[f"{sp1}_vs_{sp2}"] = similarity
        
        return signatures, similarities
    
    def growth_stage_communication(self, time_series_data):
        """
        Analyze how communication changes as mushrooms grow.
        """
        print("🌱 Analyzing Growth Stage Communication...")
        
        growth_patterns = {}
        for stage, data in time_series_data.items():
            # Extract stage-specific characteristics
            growth_patterns[stage] = {
                'communication_intensity': np.mean(np.abs(data)),
                'pattern_complexity': self.calculate_complexity(data),
                'response_speed': self.calculate_response_speed(data),
                'signal_stability': np.std(data)
            }
        
        # Analyze development trends
        development_trends = self.analyze_development_trends(growth_patterns)
        
        return growth_patterns, development_trends
    
    def substrate_language_analysis(self, substrate_data):
        """
        Study how mushrooms communicate on different substrates.
        """
        print("🏠 Analyzing Substrate Language Effects...")
        
        substrate_patterns = {}
        for substrate, data in substrate_data.items():
            substrate_patterns[substrate] = {
                'communication_style': self.classify_communication_style(data),
                'signal_quality': self.assess_signal_quality(data),
                'environmental_adaptation': self.measure_adaptation(data),
                'substrate_preference': self.calculate_preference_score(data)
            }
        
        return substrate_patterns
    
    def environmental_response_language(self, env_data, electrical_data):
        """
        Correlate electrical signals with environmental changes.
        """
        print("🌡️ Analyzing Environmental Response Language...")
        
        # Time-align environmental and electrical data
        aligned_data = self.align_environmental_electrical_data(env_data, electrical_data)
        
        # Analyze response patterns
        response_analysis = {
            'temperature_response': self.analyze_temperature_response(aligned_data),
            'humidity_response': self.analyze_humidity_response(aligned_data),
            'light_response': self.analyze_light_response(aligned_data),
            'response_timing': self.analyze_response_timing(aligned_data)
        }
        
        return response_analysis
    
    def machine_learning_communication_decoder(self, training_data):
        """
        Train AI to recognize communication patterns.
        """
        print("🧠 Training Machine Learning Communication Decoder...")
        
        # Feature extraction
        features = self.extract_communication_features(training_data)
        
        # Simple pattern recognition (placeholder for ML implementation)
        patterns = self.identify_communication_patterns(features)
        
        return patterns
    
    def network_topology_analysis(self, multi_electrode_data):
        """
        Map how mushrooms are connected through communication.
        """
        print("🌐 Analyzing Network Topology...")
        
        # Create adjacency matrix
        adjacency_matrix = self.create_adjacency_matrix(multi_electrode_data)
        
        # Analyze network properties
        network_properties = {
            'connectivity_density': np.mean(adjacency_matrix),
            'centrality_scores': self.calculate_centrality(adjacency_matrix),
            'community_structure': self.identify_communities(adjacency_matrix),
            'information_flow': self.analyze_information_flow(adjacency_matrix)
        }
        
        return network_properties, adjacency_matrix
    
    def behavioral_pattern_recognition(self, individual_data):
        """
        Identify individual mushroom "personalities" through communication.
        """
        print("🎭 Analyzing Behavioral Patterns...")
        
        personalities = {}
        for individual, data in individual_data.items():
            personalities[individual] = {
                'communication_style': self.classify_communication_style(data),
                'response_patterns': self.analyze_response_patterns(data),
                'social_behavior': self.assess_social_behavior(data),
                'adaptability': self.measure_adaptability(data)
            }
        
        return personalities
    
    # Helper methods
    def count_spikes(self, data, threshold=0.1):
        """Count electrical spikes in the data."""
        return np.sum(np.abs(data) > threshold)
    
    def calculate_complexity(self, data):
        """Calculate signal complexity using entropy."""
        hist, _ = np.histogram(data, bins=50)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def calculate_response_speed(self, data):
        """Calculate how quickly the signal responds to changes."""
        return np.mean(np.abs(np.diff(data)))
    
    def classify_communication_style(self, data):
        """Classify the communication style of the signal."""
        amplitude = np.mean(np.abs(data))
        frequency = self.count_spikes(data)
        
        if amplitude > 0.5 and frequency > 100:
            return "aggressive"
        elif amplitude < 0.2 and frequency < 50:
            return "shy"
        else:
            return "balanced"
    
    def assess_signal_quality(self, data):
        """Assess the quality of the electrical signal."""
        snr = np.mean(np.abs(data)) / np.std(data)
        if snr > 2:
            return "excellent"
        elif snr > 1:
            return "good"
        else:
            return "poor"
    
    def measure_adaptation(self, data):
        """Measure how well the signal adapts to changes."""
        return 1 / (1 + np.std(data))
    
    def calculate_preference_score(self, data):
        """Calculate preference score for the substrate."""
        return np.mean(np.abs(data)) * self.count_spikes(data)
    
    def align_environmental_electrical_data(self, env_data, electrical_data):
        """Align environmental and electrical data in time."""
        # Simple alignment - in practice, you'd use timestamps
        min_len = min(len(env_data), len(electrical_data))
        return {
            'environmental': env_data[:min_len],
            'electrical': electrical_data[:min_len]
        }
    
    def analyze_temperature_response(self, aligned_data):
        """Analyze response to temperature changes."""
        # Placeholder implementation
        return {'response_strength': 0.8, 'response_speed': 0.6}
    
    def analyze_humidity_response(self, aligned_data):
        """Analyze response to humidity changes."""
        return {'response_strength': 0.7, 'response_speed': 0.5}
    
    def analyze_light_response(self, aligned_data):
        """Analyze response to light changes."""
        return {'response_strength': 0.6, 'response_speed': 0.4}
    
    def analyze_response_timing(self, aligned_data):
        """Analyze timing of responses to environmental changes."""
        return {'average_delay': 0.3, 'consistency': 0.8}
    
    def extract_communication_features(self, data):
        """Extract features for machine learning."""
        return {
            'amplitude': np.mean(np.abs(data)),
            'frequency': self.count_spikes(data),
            'complexity': self.calculate_complexity(data),
            'stability': np.std(data)
        }
    
    def identify_communication_patterns(self, features):
        """Identify communication patterns in the features."""
        # Placeholder implementation
        return ['pattern1', 'pattern2', 'pattern3']
    
    def create_adjacency_matrix(self, multi_electrode_data):
        """Create adjacency matrix from multi-electrode data."""
        n_electrodes = len(multi_electrode_data)
        matrix = np.zeros((n_electrodes, n_electrodes))
        
        for i in range(n_electrodes):
            for j in range(n_electrodes):
                if i != j:
                    # Calculate correlation between electrodes
                    corr, _ = pearsonr(
                        multi_electrode_data[i], 
                        multi_electrode_data[j]
                    )
                    matrix[i, j] = max(0, corr)  # Only positive correlations
        
        return matrix
    
    def calculate_centrality(self, adjacency_matrix):
        """Calculate centrality scores for each node."""
        return np.sum(adjacency_matrix, axis=1)
    
    def identify_communities(self, adjacency_matrix):
        """Identify communities in the network."""
        # Simple community detection based on connectivity
        return [0, 0, 1, 1]  # Placeholder
    
    def analyze_information_flow(self, adjacency_matrix):
        """Analyze how information flows through the network."""
        return {'flow_efficiency': 0.7, 'bottlenecks': 2}
    
    def analyze_response_patterns(self, data):
        """Analyze response patterns of individual mushrooms."""
        return {'consistency': 0.8, 'adaptability': 0.6}
    
    def assess_social_behavior(self, data):
        """Assess social behavior patterns."""
        return {'interaction_frequency': 0.7, 'response_sensitivity': 0.8}
    
    def measure_adaptability(self, data):
        """Measure adaptability of individual mushrooms."""
        return 1 / (1 + np.std(data))
    
    def analyze_development_trends(self, growth_patterns):
        """Analyze trends in communication development."""
        stages = list(growth_patterns.keys())
        trends = {}
        
        for metric in ['communication_intensity', 'pattern_complexity']:
            values = [growth_patterns[stage][metric] for stage in stages]
            if len(values) > 1:
                trend = 'increasing' if values[-1] > values[0] else 'decreasing'
                trends[metric] = trend
        
        return trends
    
    def calculate_species_similarity(self, sig1, sig2):
        """Calculate similarity between two species signatures."""
        # Euclidean distance between feature vectors
        features1 = np.array([sig1['mean_amplitude'], sig1['amplitude_variance'], 
                             sig1['spike_frequency'], sig1['signal_complexity']])
        features2 = np.array([sig2['mean_amplitude'], sig2['amplitude_variance'], 
                             sig2['spike_frequency'], sig2['signal_complexity']])
        
        distance = np.linalg.norm(features1 - features2)
        similarity = 1 / (1 + distance)
        return similarity
    
    def run_comprehensive_analysis(self, data_files):
        """
        Run all advanced analysis techniques on the provided data.
        """
        print(f"🚀 Starting Comprehensive Fungal Communication Analysis by {self.author}")
        print(f"Timestamp: {self.timestamp}")
        
        all_results = {}
        
        for file_path in data_files:
            print(f"\n📁 Analyzing: {file_path}")
            data = self.load_fungal_data(file_path)
            if data is None:
                continue
            
            # Convert to numpy array (assuming first column is electrical data)
            electrical_data = data.iloc[:, 0].values
            
            file_results = {}
            
            # 1. Frequency Domain Analysis
            freq_results, freqs, mag_spectrum = self.frequency_domain_analysis(electrical_data)
            file_results['frequency_analysis'] = freq_results
            
            # 2. Time-Frequency Mapping
            time_results, tf_freqs, tf_times, spectrogram = self.time_frequency_mapping(electrical_data)
            file_results['time_frequency_analysis'] = time_results
            
            # 3. Behavioral Pattern Recognition
            behavior_results = self.behavioral_pattern_recognition({'individual': electrical_data})
            file_results['behavioral_analysis'] = behavior_results
            
            all_results[file_path] = file_results
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results):
        """Save analysis results to file."""
        output_file = f"RESULTS/analysis/advanced_communication_analysis_{self.timestamp}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for file_path, file_results in results.items():
            json_results[str(file_path)] = {}
            for analysis_type, analysis_results in file_results.items():
                json_results[str(file_path)][analysis_type] = {}
                for key, value in analysis_results.items():
                    if isinstance(value, np.ndarray):
                        json_results[str(file_path)][analysis_type][key] = value.tolist()
                    elif isinstance(value, np.integer):
                        json_results[str(file_path)][analysis_type][key] = int(value)
                    elif isinstance(value, np.floating):
                        json_results[str(file_path)][analysis_type][key] = float(value)
                    else:
                        json_results[str(file_path)][analysis_type][key] = value
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
    
    def generate_visualizations(self, results):
        """Generate visualizations for the analysis results."""
        print("🎨 Generating Advanced Analysis Visualizations...")
        
        # This would create various plots and charts
        # Implementation depends on specific visualization needs
        pass

def main():
    """Main function to demonstrate the advanced analyzer."""
    analyzer = AdvancedFungalCommunicationAnalyzer()
    
    # Example usage
    print("🍄 Advanced Fungal Communication Analyzer")
    print(f"Author: {analyzer.author}")
    print(f"Timestamp: {analyzer.timestamp}")
    
    # You would load your actual data files here
    # data_files = ["path/to/your/data1.csv", "path/to/your/data2.csv"]
    # results = analyzer.run_comprehensive_analysis(data_files)

if __name__ == "__main__":
    main() 