#!/usr/bin/env python3
"""
Fungal Audio-Linguistic Correlation Analysis

This script analyzes correlations between:
1. Fungal electrical signals
2. Audio synthesis outputs
3. Linguistic patterns (Adamatzky's methodology)

Based on Adamatzky's research on fungal communication and language complexity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from scipy.fft import fft, fftfreq
import librosa
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FungalAudioLinguisticAnalyzer:
    """
    Analyzes correlations between fungal electrical signals, audio synthesis, and linguistic patterns.
    """
    
    def __init__(self, sampling_rate: float = 1.0):
        self.sampling_rate = sampling_rate
        
    def load_audio_file(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """Load audio file and return waveform and sample rate."""
        try:
            waveform, sr = librosa.load(audio_path, sr=None)
            logger.info(f"Loaded audio: {len(waveform)} samples at {sr} Hz")
            return waveform, sr
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {str(e)}")
            return np.array([]), 0.0
    
    def extract_audio_features(self, waveform: np.ndarray, sr: float) -> Dict:
        """Extract comprehensive audio features."""
        try:
            features = {}
            
            # Spectral features
            features['spectral_centroid'] = librosa.feature.spectral_centroid(y=waveform, sr=sr).flatten()
            features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=waveform, sr=sr).flatten()
            features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=waveform, sr=sr).flatten()
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
            features['mfccs'] = mfccs
            
            # Rhythm features
            tempo, beats = librosa.beat.beat_track(y=waveform, sr=sr)
            features['tempo'] = tempo
            features['beat_frames'] = beats
            
            # Harmonic features
            harmonic, percussive = librosa.effects.hpss(waveform)
            features['harmonic_ratio'] = np.sum(harmonic**2) / (np.sum(harmonic**2) + np.sum(percussive**2))
            
            # Zero crossing rate
            features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(waveform).flatten()
            
            return features
            
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {str(e)}")
            return {}
    
    def analyze_linguistic_patterns(self, spike_times: np.ndarray) -> Dict:
        """
        Implement Adamatzky's linguistic analysis methodology for fungal signals.
        """
        try:
            if len(spike_times) < 2:
                return {'error': 'Insufficient spikes for analysis'}
            
            # Calculate inter-spike intervals
            intervals = np.diff(spike_times)
            
            # Group spikes into words (Adamatzky's method)
            theta = np.mean(intervals)  # Threshold for word boundaries
            words = []
            current_word = [0]
            
            for i in range(1, len(spike_times)):
                if intervals[i-1] <= theta:
                    current_word.append(i)
                else:
                    words.append(current_word)
                    current_word = [i]
            
            if current_word:
                words.append(current_word)
            
            # Analyze word statistics
            word_lengths = [len(word) for word in words]
            
            # Calculate vocabulary size (unique word patterns)
            word_patterns = [tuple(word) for word in words]
            vocabulary_size = len(set(word_patterns))
            
            # Calculate complexity measures
            if len(word_lengths) > 0:
                mean_word_length = np.mean(word_lengths)
                word_length_entropy = -np.sum([(word_lengths.count(l)/len(word_lengths)) * 
                                             np.log2(word_lengths.count(l)/len(word_lengths)) 
                                             for l in set(word_lengths)])
            else:
                mean_word_length = 0
                word_length_entropy = 0
            
            # Analyze syntax (word transitions)
            transition_matrix = {}
            if len(words) > 1:
                for i in range(len(words)-1):
                    current_pattern = tuple(words[i])
                    next_pattern = tuple(words[i+1])
                    
                    if current_pattern not in transition_matrix:
                        transition_matrix[current_pattern] = {}
                    
                    if next_pattern not in transition_matrix[current_pattern]:
                        transition_matrix[current_pattern][next_pattern] = 0
                    
                    transition_matrix[current_pattern][next_pattern] += 1
            
            return {
                'total_spikes': len(spike_times),
                'total_words': len(words),
                'word_lengths': word_lengths,
                'mean_word_length': mean_word_length,
                'vocabulary_size': vocabulary_size,
                'word_length_entropy': word_length_entropy,
                'transition_matrix': transition_matrix,
                'theta_threshold': theta
            }
            
        except Exception as e:
            logger.error(f"Linguistic analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def detect_spikes(self, signal_data: np.ndarray, threshold_factor: float = 2.0) -> np.ndarray:
        """Detect spikes in electrical signal using adaptive thresholding."""
        try:
            # Calculate baseline and threshold
            baseline = np.median(signal_data)
            noise_level = np.std(signal_data)
            threshold = baseline + threshold_factor * noise_level
            
            # Find peaks above threshold
            peaks, _ = signal.find_peaks(signal_data, height=threshold, distance=10)
            
            return peaks
            
        except Exception as e:
            logger.error(f"Spike detection failed: {str(e)}")
            return np.array([])
    
    def correlate_electrical_audio(self, electrical_signal: np.ndarray, 
                                 audio_features: Dict) -> Dict:
        """Analyze correlations between electrical signals and audio features."""
        try:
            correlations = {}
            
            # Ensure signals have same length for correlation
            min_length = min(len(electrical_signal), len(audio_features.get('spectral_centroid', [])))
            if min_length == 0:
                return {'error': 'No overlapping data for correlation'}
            
            # Truncate signals to same length
            elec_truncated = electrical_signal[:min_length]
            
            # Correlate with each audio feature
            for feature_name, feature_data in audio_features.items():
                if isinstance(feature_data, np.ndarray) and len(feature_data) >= min_length:
                    feature_truncated = feature_data[:min_length]
                    
                    # Calculate correlations
                    pearson_corr, pearson_p = pearsonr(elec_truncated, feature_truncated)
                    spearman_corr, spearman_p = spearmanr(elec_truncated, feature_truncated)
                    
                    correlations[feature_name] = {
                        'pearson_correlation': pearson_corr,
                        'pearson_p_value': pearson_p,
                        'spearman_correlation': spearman_corr,
                        'spearman_p_value': spearman_p
                    }
            
            return correlations
            
        except Exception as e:
            logger.error(f"Electrical-audio correlation failed: {str(e)}")
            return {'error': str(e)}
    
    def analyze_frequency_content(self, signal_data: np.ndarray) -> Dict:
        """Analyze frequency content of electrical signals."""
        try:
            # Compute FFT
            fft_result = fft(signal_data)
            freqs = fftfreq(len(signal_data), 1/self.sampling_rate)
            
            # Get positive frequencies
            positive_mask = freqs >= 0
            positive_freqs = freqs[positive_mask]
            positive_fft = np.abs(fft_result[positive_mask])
            
            # Find dominant frequencies
            dominant_indices = signal.find_peaks(positive_fft, height=np.max(positive_fft)*0.1)
            dominant_freqs = positive_freqs[dominant_indices[0]]
            dominant_amplitudes = positive_fft[dominant_indices[0]]
            
            # Calculate power spectral density
            psd = np.abs(positive_fft)**2
            
            return {
                'frequencies': positive_freqs,
                'amplitudes': positive_fft,
                'dominant_frequencies': dominant_freqs,
                'dominant_amplitudes': dominant_amplitudes,
                'power_spectral_density': psd,
                'total_power': np.sum(psd)
            }
            
        except Exception as e:
            logger.error(f"Frequency content analysis failed: {str(e)}")
            return {}
    
    def comprehensive_analysis(self, electrical_file: str, audio_file: str) -> Dict:
        """Perform comprehensive analysis of electrical and audio signals."""
        logger.info("Starting comprehensive fungal audio-linguistic analysis...")
        
        # Load electrical data
        electrical_data = pd.read_csv(electrical_file)
        if len(electrical_data.columns) > 1:
            voltage_col = electrical_data.columns[1]  # Assume second column is voltage
        else:
            voltage_col = electrical_data.columns[0]
        
        electrical_signal = electrical_data[voltage_col].values
        electrical_signal = electrical_signal[~np.isnan(electrical_signal)]
        
        # Load audio data
        audio_waveform, audio_sr = self.load_audio_file(audio_file)
        if len(audio_waveform) == 0:
            logger.error("Failed to load audio file")
            return {}
        
        # Extract audio features
        audio_features = self.extract_audio_features(audio_waveform, audio_sr)
        
        # Detect spikes in electrical signal
        spike_times = self.detect_spikes(electrical_signal)
        
        # Perform linguistic analysis
        linguistic_results = self.analyze_linguistic_patterns(spike_times)
        
        # Analyze frequency content
        frequency_analysis = self.analyze_frequency_content(electrical_signal)
        
        # Correlate electrical and audio signals
        correlations = self.correlate_electrical_audio(electrical_signal, audio_features)
        
        # Compile results
        results = {
            'electrical_signal': {
                'length': len(electrical_signal),
                'mean': float(np.mean(electrical_signal)),
                'std': float(np.std(electrical_signal)),
                'spike_count': len(spike_times)
            },
            'audio_features': {
                'sample_rate': audio_sr,
                'duration': len(audio_waveform) / audio_sr,
                'tempo': audio_features.get('tempo', 0),
                'harmonic_ratio': audio_features.get('harmonic_ratio', 0)
            },
            'linguistic_analysis': linguistic_results,
            'frequency_analysis': frequency_analysis,
            'correlations': correlations,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info("Comprehensive analysis completed")
        return results
    
    def create_correlation_visualizations(self, results: Dict, output_dir: str = "results"):
        """Create visualizations of the correlation analysis."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Create correlation heatmap
            if 'correlations' in results and 'error' not in results['correlations']:
                correlation_data = []
                feature_names = []
                
                for feature_name, corr_data in results['correlations'].items():
                    if 'error' not in corr_data:
                        feature_names.append(feature_name)
                        correlation_data.append([
                            corr_data['pearson_correlation'],
                            corr_data['spearman_correlation']
                        ])
                
                if correlation_data:
                    correlation_matrix = np.array(correlation_data)
                    
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(correlation_matrix, 
                               xticklabels=['Pearson', 'Spearman'],
                               yticklabels=feature_names,
                               annot=True, cmap='RdBu_r', center=0,
                               fmt='.3f')
                    plt.title('Electrical-Audio Feature Correlations')
                    plt.tight_layout()
                    plt.savefig(output_path / 'electrical_audio_correlations.png', dpi=300, bbox_inches='tight')
                    plt.show()
            
            # Create linguistic analysis visualization
            if 'linguistic_analysis' in results and 'error' not in results['linguistic_analysis']:
                ling = results['linguistic_analysis']
                
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Word length distribution
                if 'word_lengths' in ling and ling['word_lengths']:
                    axes[0].hist(ling['word_lengths'], bins='auto', alpha=0.7, edgecolor='black')
                    axes[0].set_xlabel('Word Length')
                    axes[0].set_ylabel('Frequency')
                    axes[0].set_title('Fungal Word Length Distribution')
                    axes[0].grid(True, alpha=0.3)
                
                # Vocabulary analysis
                if 'vocabulary_size' in ling:
                    axes[1].bar(['Total Words', 'Vocabulary Size'], 
                               [ling.get('total_words', 0), ling.get('vocabulary_size', 0)],
                               color=['skyblue', 'lightcoral'])
                    axes[1].set_ylabel('Count')
                    axes[1].set_title('Fungal Language Statistics')
                    axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_path / 'linguistic_analysis.png', dpi=300, bbox_inches='tight')
                plt.show()
            
            logger.info(f"Visualizations saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
    
    def save_analysis_results(self, results: Dict, output_dir: str = "results"):
        """Save analysis results to JSON file."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Convert numpy types to native Python types
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            converted_results = convert_numpy(results)
            
            output_file = output_path / 'fungal_audio_linguistic_analysis.json'
            with open(output_file, 'w') as f:
                json.dump(converted_results, f, indent=2)
            
            logger.info(f"Analysis results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Results saving failed: {str(e)}")

def main():
    """Main analysis function."""
    logger.info("Starting Fungal Audio-Linguistic Correlation Analysis")
    
    # Initialize analyzer
    analyzer = FungalAudioLinguisticAnalyzer(sampling_rate=1.0)
    
    # Example file paths (modify as needed)
    electrical_file = "DATA/processed/validated_fungal_electrical_csvs/New_Oyster_with spray_as_mV_seconds_SigView.csv"
    audio_file = "RESULTS/audio/New_Oyster_with spray_as_mV.csv_basic_sound.wav"
    
    # Check if files exist
    if not Path(electrical_file).exists():
        logger.error(f"Electrical data file not found: {electrical_file}")
        return
    
    if not Path(audio_file).exists():
        logger.error(f"Audio file not found: {audio_file}")
        return
    
    # Perform comprehensive analysis
    results = analyzer.comprehensive_analysis(electrical_file, audio_file)
    
    if results:
        # Create visualizations
        analyzer.create_correlation_visualizations(results)
        
        # Save results
        analyzer.save_analysis_results(results)
        
        # Print summary
        logger.info("Analysis Summary:")
        logger.info(f"Electrical signal: {results['electrical_signal']['length']} samples, {results['electrical_signal']['spike_count']} spikes")
        logger.info(f"Audio: {results['audio_features']['duration']:.2f}s, tempo: {results['audio_features']['tempo']:.1f} BPM")
        
        if 'linguistic_analysis' in results and 'error' not in results['linguistic_analysis']:
            ling = results['linguistic_analysis']
            logger.info(f"Linguistic: {ling.get('total_words', 0)} words, vocabulary size: {ling.get('vocabulary_size', 0)}")
        
        logger.info("Analysis completed successfully!")
    else:
        logger.error("Analysis failed")

if __name__ == "__main__":
    main() 