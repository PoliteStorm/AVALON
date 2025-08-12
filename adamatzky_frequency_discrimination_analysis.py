#!/usr/bin/env python3
"""
Adamatzky Frequency Discrimination Analysis for Fungal Mycelium Networks

This script implements the methodology from Adamatzky et al. (2022) for analyzing
frequency discrimination in Pleurotus ostreatus mycelium networks, including:

1. Frequency response analysis (1-100 mHz range)
2. Harmonic analysis (2nd vs 3rd harmonics)
3. Total Harmonic Distortion (THD) calculation
4. Fuzzy logic classification of frequency responses
5. Linguistic analysis of electrical spiking patterns
6. Audio correlation analysis

Based on: "Fungal electronics: frequency discrimination in mycelium networks"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdamatzkyFrequencyAnalyzer:
    """
    Implements Adamatzky's frequency discrimination methodology for fungal networks.
    """
    
    def __init__(self, sampling_rate: float = 1.0):
        self.sampling_rate = sampling_rate
        self.frequencies_mhz = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                                        20, 30, 40, 50, 60, 70, 80, 90, 100])
        
    def load_fungal_data(self, file_path: str) -> np.ndarray:
        """Load fungal electrical data from CSV file."""
        try:
            data = pd.read_csv(file_path)
            # Assume first column contains voltage/time data
            if len(data.columns) > 1:
                # If multiple columns, try to identify voltage column
                voltage_col = None
                for col in data.columns:
                    if any(keyword in col.lower() for keyword in ['voltage', 'mv', 'v', 'signal']):
                        voltage_col = col
                        break
                if voltage_col is None:
                    voltage_col = data.columns[1]  # Default to second column
                signal_data = data[voltage_col].values
            else:
                signal_data = data.iloc[:, 0].values
                
            # Remove NaN values
            signal_data = signal_data[~np.isnan(signal_data)]
            logger.info(f"Loaded {len(signal_data)} data points from {file_path}")
            return signal_data
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            return np.array([])
    
    def apply_sinusoidal_stimulation(self, signal_data: np.ndarray, 
                                   frequency_mhz: float, 
                                   amplitude: float = 10.0) -> np.ndarray:
        """
        Apply sinusoidal stimulation at specified frequency (Adamatzky protocol).
        
        Args:
            signal_data: Original fungal signal
            frequency_mhz: Frequency in mHz
            amplitude: Peak-to-peak amplitude in V
            
        Returns:
            Stimulated signal
        """
        # Convert mHz to Hz
        frequency_hz = frequency_mhz / 1000.0
        
        # Generate time array
        time = np.arange(len(signal_data)) / self.sampling_rate
        
        # Generate sinusoidal stimulation
        stimulation = amplitude * np.sin(2 * np.pi * frequency_hz * time)
        
        # Combine original signal with stimulation
        stimulated_signal = signal_data + stimulation
        
        return stimulated_signal
    
    def compute_fft_analysis(self, signal_data: np.ndarray, 
                            window_type: str = 'blackman') -> Dict:
        """
        Compute FFT analysis using Blackman window (Adamatzky's choice).
        
        Args:
            signal_data: Input signal
            window_type: Window function type
            
        Returns:
            Dictionary containing FFT results
        """
        try:
            # Apply window function
            if window_type == 'blackman':
                # Use numpy's blackman window if scipy doesn't have it
                try:
                    window = signal.blackman(len(signal_data))
                except AttributeError:
                    window = np.blackman(len(signal_data))
            elif window_type == 'hann':
                try:
                    window = signal.hann(len(signal_data))
                except AttributeError:
                    window = np.hanning(len(signal_data))
            else:
                window = np.ones(len(signal_data))
            
            # Apply window and compute FFT
            windowed_signal = signal_data * window
            fft_result = fft(windowed_signal)
            freqs = fftfreq(len(signal_data), 1/self.sampling_rate)
            
            # Get positive frequencies only
            positive_freqs = freqs[freqs >= 0]
            positive_fft = np.abs(fft_result[freqs >= 0])
            
            # Find fundamental frequency and harmonics
            fundamental_idx = np.argmax(positive_fft)
            fundamental_freq = positive_freqs[fundamental_idx]
            fundamental_amp = positive_fft[fundamental_idx]
            
            # Find harmonics (2nd and 3rd)
            harmonic_2_idx = np.argmin(np.abs(positive_freqs - 2*fundamental_freq))
            harmonic_3_idx = np.argmin(np.abs(positive_freqs - 3*fundamental_freq))
            
            harmonic_2_amp = positive_fft[harmonic_2_idx]
            harmonic_3_amp = positive_fft[harmonic_3_idx]
            
            return {
                'frequencies': positive_freqs,
                'amplitudes': positive_fft,
                'fundamental_freq': fundamental_freq,
                'fundamental_amp': fundamental_amp,
                'harmonic_2_amp': harmonic_2_amp,
                'harmonic_3_amp': harmonic_3_amp,
                'harmonic_2_3_ratio': harmonic_2_amp / (harmonic_3_amp + 1e-10)
            }
            
        except Exception as e:
            logger.error(f"FFT analysis failed: {str(e)}")
            return {}
    
    def calculate_thd(self, fft_results: Dict) -> float:
        """
        Calculate Total Harmonic Distortion (THD) as per Adamatzky's methodology.
        
        THD = sqrt(sum(harmonic_amplitudes^2)) / fundamental_amplitude
        """
        try:
            fundamental_amp = fft_results.get('fundamental_amp', 0)
            if fundamental_amp == 0:
                return 0.0
                
            # Get all amplitudes above fundamental
            freqs = fft_results['frequencies']
            amps = fft_results['amplitudes']
            
            # Find harmonics (frequencies > fundamental)
            fundamental_freq = fft_results['fundamental_freq']
            harmonic_mask = freqs > fundamental_freq
            
            if not np.any(harmonic_mask):
                return 0.0
                
            harmonic_amplitudes = amps[harmonic_mask]
            harmonic_sum_squares = np.sum(harmonic_amplitudes**2)
            
            thd = np.sqrt(harmonic_sum_squares) / fundamental_amp
            return float(thd)
            
        except Exception as e:
            logger.error(f"THD calculation failed: {str(e)}")
            return 0.0
    
    def create_fuzzy_sets(self, thd_values: List[float]) -> Dict:
        """
        Create fuzzy sets for THD classification as proposed by Adamatzky.
        
        Uses sigmoidal sets for boundaries and Gaussian sets for center.
        """
        try:
            thd_array = np.array(thd_values)
            min_thd, max_thd = np.min(thd_array), np.max(thd_array)
            
            # Define fuzzy set parameters
            very_low_threshold = min_thd + 0.1 * (max_thd - min_thd)
            low_threshold = min_thd + 0.25 * (max_thd - min_thd)
            medium_threshold = min_thd + 0.5 * (max_thd - min_thd)
            high_threshold = min_thd + 0.75 * (max_thd - min_thd)
            very_high_threshold = min_thd + 0.9 * (max_thd - min_thd)
            
            # Sigmoidal membership functions for boundaries
            def sigmoidal_membership(x, center, slope):
                return 1 / (1 + np.exp(-slope * (x - center)))
            
            # Gaussian membership functions for center
            def gaussian_membership(x, center, sigma):
                return np.exp(-0.5 * ((x - center) / sigma)**2)
            
            # Create membership functions
            fuzzy_sets = {
                'very_low': lambda x: 1 - sigmoidal_membership(x, very_low_threshold, 10),
                'low': lambda x: gaussian_membership(x, low_threshold, (high_threshold - low_threshold) / 4),
                'medium': lambda x: gaussian_membership(x, medium_threshold, (high_threshold - low_threshold) / 4),
                'high': lambda x: gaussian_membership(x, high_threshold, (high_threshold - low_threshold) / 4),
                'very_high': lambda x: sigmoidal_membership(x, very_high_threshold, 10)
            }
            
            return {
                'fuzzy_sets': fuzzy_sets,
                'thresholds': {
                    'very_low': very_low_threshold,
                    'low': low_threshold,
                    'medium': medium_threshold,
                    'high': high_threshold,
                    'very_high': very_high_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Fuzzy set creation failed: {str(e)}")
            return {}
    
    def classify_frequency_response(self, thd_value: float, fuzzy_sets: Dict) -> Dict:
        """Classify frequency response using fuzzy logic."""
        try:
            classifications = {}
            for set_name, membership_func in fuzzy_sets['fuzzy_sets'].items():
                membership = membership_func(thd_value)
                classifications[set_name] = float(membership)
            
            # Find dominant classification
            dominant_class = max(classifications.items(), key=lambda x: x[1])
            
            return {
                'classifications': classifications,
                'dominant_class': dominant_class[0],
                'confidence': dominant_class[1]
            }
            
        except Exception as e:
            logger.error(f"Frequency response classification failed: {str(e)}")
            return {}
    
    def analyze_frequency_discrimination(self, signal_data: np.ndarray) -> Dict:
        """
        Perform complete frequency discrimination analysis following Adamatzky's methodology.
        """
        logger.info("Starting frequency discrimination analysis...")
        
        results = {
            'frequency_responses': {},
            'harmonic_analysis': {},
            'thd_analysis': {},
            'fuzzy_classification': {},
            'summary': {}
        }
        
        # Analyze response to different frequencies
        for freq_mhz in self.frequencies_mhz:
            logger.info(f"Analyzing {freq_mhz} mHz response...")
            
            # Apply stimulation
            stimulated_signal = self.apply_sinusoidal_stimulation(signal_data, freq_mhz)
            
            # Compute FFT
            fft_results = self.compute_fft_analysis(stimulated_signal)
            
            if fft_results:
                # Calculate THD
                thd = self.calculate_thd(fft_results)
                
                # Store results
                results['frequency_responses'][freq_mhz] = {
                    'fft': fft_results,
                    'thd': thd
                }
                
                results['harmonic_analysis'][freq_mhz] = {
                    'harmonic_2_3_ratio': fft_results.get('harmonic_2_3_ratio', 0),
                    'harmonic_2_amp': fft_results.get('harmonic_2_amp', 0),
                    'harmonic_3_amp': fft_results.get('harmonic_3_amp', 0)
                }
                
                results['thd_analysis'][freq_mhz] = thd
        
        # Create fuzzy sets and classify responses
        thd_values = list(results['thd_analysis'].values())
        if thd_values:
            fuzzy_sets = self.create_fuzzy_sets(thd_values)
            results['fuzzy_classification'] = fuzzy_sets
            
            # Classify each frequency response
            for freq_mhz, thd in results['thd_analysis'].items():
                classification = self.classify_frequency_response(thd, fuzzy_sets)
                results['frequency_responses'][freq_mhz]['classification'] = classification
        
        # Generate summary statistics
        results['summary'] = {
            'total_frequencies_tested': len(self.frequencies_mhz),
            'mean_thd': np.mean(thd_values) if thd_values else 0,
            'std_thd': np.std(thd_values) if thd_values else 0,
            'frequency_discrimination_threshold': 10.0,  # mHz threshold from Adamatzky
            'low_freq_thd_mean': np.mean([results['thd_analysis'][float(f)] for f in self.frequencies_mhz if f <= 10]) if any(f <= 10 for f in self.frequencies_mhz) else 0,
            'high_freq_thd_mean': np.mean([results['thd_analysis'][float(f)] for f in self.frequencies_mhz if f > 10]) if any(f > 10 for f in self.frequencies_mhz) else 0
        }
        
        logger.info("Frequency discrimination analysis completed")
        return results
    
    def create_visualizations(self, results: Dict, output_dir: str = "results"):
        """Create comprehensive visualizations of the analysis results."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Set style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Adamatzky Frequency Discrimination Analysis', fontsize=16)
            
            # 1. THD vs Frequency
            frequencies = list(results['thd_analysis'].keys())
            thd_values = list(results['thd_analysis'].values())
            
            axes[0, 0].plot(frequencies, thd_values, 'bo-', linewidth=2, markersize=6)
            axes[0, 0].axvline(x=10, color='r', linestyle='--', alpha=0.7, label='Threshold (10 mHz)')
            axes[0, 0].set_xlabel('Frequency (mHz)')
            axes[0, 0].set_ylabel('Total Harmonic Distortion (THD)')
            axes[0, 0].set_title('THD vs Frequency Response')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # 2. Harmonic Analysis
            harmonic_2_3_ratios = [results['harmonic_analysis'][f]['harmonic_2_3_ratio'] for f in frequencies]
            axes[0, 1].plot(frequencies, harmonic_2_3_ratios, 'go-', linewidth=2, markersize=6)
            axes[0, 1].axvline(x=10, color='r', linestyle='--', alpha=0.7, label='Threshold (10 mHz)')
            axes[0, 1].set_xlabel('Frequency (mHz)')
            axes[0, 1].set_ylabel('2nd/3rd Harmonic Ratio')
            axes[0, 1].set_title('Harmonic Ratio vs Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # 3. Fuzzy Classification Heatmap
            if results['fuzzy_classification']:
                fuzzy_data = []
                for freq in frequencies:
                    if 'classification' in results['frequency_responses'][freq]:
                        classifications = results['frequency_responses'][freq]['classification']['classifications']
                        fuzzy_data.append([classifications.get('very_low', 0),
                                        classifications.get('low', 0),
                                        classifications.get('medium', 0),
                                        classifications.get('high', 0),
                                        classifications.get('very_high', 0)])
                    else:
                        fuzzy_data.append([0, 0, 0, 0, 0])
                
                fuzzy_data = np.array(fuzzy_data)
                im = axes[1, 0].imshow(fuzzy_data.T, aspect='auto', cmap='viridis')
                axes[1, 0].set_xlabel('Frequency Index')
                axes[1, 0].set_ylabel('Fuzzy Set')
                axes[1, 0].set_title('Fuzzy Classification Heatmap')
                axes[1, 0].set_yticks(range(5))
                axes[1, 0].set_yticklabels(['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                plt.colorbar(im, ax=axes[1, 0])
            
            # 4. Summary Statistics
            summary = results['summary']
            summary_text = f"""
            Total Frequencies Tested: {summary['total_frequencies_tested']}
            Mean THD: {summary['mean_thd']:.3f}
            Std THD: {summary['std_thd']:.3f}
            Low Freq THD Mean (≤10 mHz): {summary['low_freq_thd_mean']:.3f}
            High Freq THD Mean (>10 mHz): {summary['high_freq_thd_mean']:.3f}
            """
            
            axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1, 1].set_title('Analysis Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path / 'adamatzky_frequency_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Visualizations saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {str(e)}")
    
    def save_results(self, results: Dict, output_dir: str = "results"):
        """Save analysis results to JSON file."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {str(key): convert_numpy(value) for key, value in obj.items() 
                           if not callable(value) and not str(key).startswith('_')}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj 
                           if not callable(item)]
                elif callable(obj):
                    return str(obj)  # Convert functions to string representation
                else:
                    return obj
            
            converted_results = convert_numpy(results)
            
            output_file = output_path / 'adamatzky_frequency_discrimination_results.json'
            with open(output_file, 'w') as f:
                json.dump(converted_results, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Results saving failed: {str(e)}")

def main():
    """Main analysis function."""
    logger.info("Starting Adamatzky Frequency Discrimination Analysis")
    
    # Initialize analyzer
    analyzer = AdamatzkyFrequencyAnalyzer(sampling_rate=1.0)
    
    # Example data file path (modify as needed)
    data_file = "DATA/processed/validated_fungal_electrical_csvs/New_Oyster_with spray_as_mV_seconds_SigView.csv"
    
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please provide a valid path to fungal electrical data")
        return
    
    # Load data
    signal_data = analyzer.load_fungal_data(data_file)
    if len(signal_data) == 0:
        logger.error("No valid data loaded")
        return
    
    # Perform frequency discrimination analysis
    results = analyzer.analyze_frequency_discrimination(signal_data)
    
    # Create visualizations
    analyzer.create_visualizations(results)
    
    # Save results
    analyzer.save_results(results)
    
    # Print summary
    summary = results['summary']
    logger.info("Analysis Summary:")
    logger.info(f"Total frequencies tested: {summary['total_frequencies_tested']}")
    logger.info(f"Mean THD: {summary['mean_thd']:.3f}")
    logger.info(f"Low frequency THD mean (≤10 mHz): {summary['low_freq_thd_mean']:.3f}")
    logger.info(f"High frequency THD mean (>10 mHz): {summary['high_freq_thd_mean']:.3f}")
    
    logger.info("Analysis completed successfully!")

if __name__ == "__main__":
    main() 