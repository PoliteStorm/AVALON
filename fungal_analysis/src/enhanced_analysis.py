"""
Enhanced Analysis Framework for Fungal Signals

This module provides an integrated approach combining multiple analysis methods:
- Wavelet transform (existing)
- Non-linear pattern detection
- Information theory metrics
- Network state analysis
- Linguistic analysis (Adamatzky's methodology)
"""

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import logging
import pywt
from linguistic_analyzer import LinguisticAnalyzer
import psutil
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EnhancedAnalysis:
    def __init__(self, sampling_rate: float = 1.0, max_cpu_percent: float = 80.0):
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        self.max_cpu_percent = max_cpu_percent
        self.chunk_size = 5000  # Reduced chunk size for Chromebook
        
    def _check_resources(self):
        """Monitor CPU usage and pause if necessary."""
        cpu_percent = psutil.cpu_percent()
        if cpu_percent > self.max_cpu_percent:
            time.sleep(1)  # Give system time to recover
            
    def denoise_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """Simple denoising using wavelet thresholding."""
        try:
            # For very long signals, process in chunks
            if len(signal_data) > self.chunk_size:
                chunks = np.array_split(signal_data, len(signal_data) // self.chunk_size + 1)
                denoised_chunks = []
                
                for chunk in tqdm(chunks, desc="Denoising signal"):
                    self._check_resources()
                    denoised_chunk = self._denoise_chunk(chunk)
                    denoised_chunks.append(denoised_chunk)
                    
                return np.concatenate(denoised_chunks)
            else:
                return self._denoise_chunk(signal_data)
                
        except Exception as e:
            logger.warning(f"Wavelet denoising failed: {str(e)}. Using original signal.")
            return signal_data
            
    def _denoise_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Denoise a single chunk of data."""
        # Simple wavelet denoising
        wavelet = 'db4'
        level = min(3, int(np.log2(len(chunk))))  # Reduced levels
        
        coeffs = pywt.wavedec(chunk, wavelet, level=level)
        for i in range(1, len(coeffs)):
            thresh = np.std(coeffs[i]) * np.sqrt(2*np.log(len(coeffs[i])))
            coeffs[i] = pywt.threshold(coeffs[i], thresh, mode='soft')
            
        denoised = pywt.waverec(coeffs, wavelet)
        return denoised[:len(chunk)]

    def detect_spikes(self, signal_data: np.ndarray) -> np.ndarray:
        """Detect spikes in the signal using wavelet-based detection."""
        try:
            # Process in smaller chunks
            if len(signal_data) > self.chunk_size:
                chunks = np.array_split(signal_data, len(signal_data) // self.chunk_size + 1)
                all_peaks = []
                offset = 0
                
                for chunk in tqdm(chunks, desc="Detecting spikes"):
                    self._check_resources()
                    chunk_peaks = self._detect_spikes_chunk(chunk)
                    all_peaks.extend([p + offset for p in chunk_peaks])
                    offset += len(chunk)
                    
                return np.array(sorted(all_peaks))
            else:
                return np.array(self._detect_spikes_chunk(signal_data))
                
        except Exception as e:
            logger.warning(f"Spike detection failed: {str(e)}")
            return np.array([])
            
    def _detect_spikes_chunk(self, chunk: np.ndarray) -> List[int]:
        """Detect spikes in a single chunk."""
        # Use fewer wavelet scales for efficiency
        scales = np.arange(1, 16)  # Reduced from 32
        coeffs = pywt.cwt(chunk, scales, 'morl')[0]
        
        # Compute power and normalize
        power = np.abs(coeffs) ** 2
        power_z = (power - np.mean(power)) / (np.std(power) + 1e-10)
        
        # Find peaks
        peaks = []
        threshold = 2.0
        
        for i in range(power_z.shape[0]):
            scale_peaks = signal.find_peaks(power_z[i], height=threshold,
                                          distance=20)[0]
            peaks.extend(scale_peaks)
            
        return sorted(list(set(peaks)))

    def compute_multi_scale_features(self, signal_data: np.ndarray) -> Dict:
        """Analyze signal across multiple temporal scales."""
        try:
            # Process in smaller chunks
            if len(signal_data) > self.chunk_size:
                chunks = np.array_split(signal_data, len(signal_data) // self.chunk_size + 1)
                results = {}
                
                for i, chunk in enumerate(tqdm(chunks, desc="Multi-scale analysis")):
                    self._check_resources()
                    chunk_results = self._compute_multi_scale_chunk(chunk)
                    
                    # Merge results
                    if i == 0:
                        results = chunk_results
                    else:
                        for scale_name in chunk_results:
                            if scale_name in results:
                                results[scale_name]['power'] = np.concatenate(
                                    [results[scale_name]['power'], 
                                     chunk_results[scale_name]['power']], axis=1)
                                results[scale_name]['significant_events'] = np.concatenate(
                                    [results[scale_name]['significant_events'],
                                     chunk_results[scale_name]['significant_events']], axis=1)
                            
                return results
            else:
                return self._compute_multi_scale_chunk(signal_data)
                
        except Exception as e:
            logger.warning(f"Multi-scale analysis failed: {str(e)}")
            return {}
            
    def _compute_multi_scale_chunk(self, signal_data: np.ndarray) -> Dict:
        """Compute multi-scale features for a single chunk."""
        if len(signal_data) < 100:
            return {}
            
        # Simplified scale definitions for efficiency
        scales = {
            'short': np.arange(1, min(30, len(signal_data)//4)),
            'medium': np.arange(30, min(100, len(signal_data)//4), 5),
            'long': np.arange(100, min(300, len(signal_data)//4), 20)
        }
        
        results = {}
        for scale_name, scale_values in scales.items():
            if len(scale_values) == 0:
                continue
                
            try:
                coeffs = pywt.cwt(signal_data, scale_values, 'morl')[0]
                power = np.abs(coeffs) ** 2
                
                # Simplified normalization
                power_z = (power - np.mean(power)) / (np.std(power) + 1e-10)
                significant = power_z > 2.0
                
                results[scale_name] = {
                    'power': power,
                    'significant_events': significant,
                    'mean_power': np.mean(power, axis=1)
                }
            except Exception as e:
                logger.warning(f"Scale {scale_name} analysis failed: {str(e)}")
                continue
                
        return results

    def detect_nonlinear_patterns(self, signal_data: np.ndarray, 
                                embed_dim: int = 3, tau: int = 1) -> Dict:
        """Detect non-linear patterns using phase space reconstruction."""
        try:
            # Ensure sufficient data points
            if len(signal_data) < embed_dim * tau:
                raise ValueError("Signal too short for embedding")
                
            # Phase space reconstruction
            N = len(signal_data) - (embed_dim - 1) * tau
            embedded = np.zeros((N, embed_dim))
            for i in range(embed_dim):
                embedded[:, i] = signal_data[i*tau:i*tau + N]
                
            # Compute distance matrix
            dist_matrix = np.zeros((N, N))
            for i in range(N):
                dist_matrix[i] = np.linalg.norm(embedded - embedded[i], axis=1)
                
            # Compute recurrence plot
            threshold = np.mean(dist_matrix) + np.std(dist_matrix)
            recurrence = dist_matrix < threshold
            
            # Analyze patterns
            diag_lines = []
            for i in range(-N+1, N):
                line = np.diag(recurrence, k=i)
                if len(line) > 0:
                    diag_lines.extend(self._count_lines(line))
                    
            return {
                'recurrence_rate': float(np.mean(recurrence)),
                'determinism': float(len(diag_lines) / N if diag_lines else 0),
                'mean_line_length': float(np.mean(diag_lines) if diag_lines else 0)
            }
        except Exception as e:
            logger.warning(f"Nonlinear pattern detection failed: {str(e)}")
            return {
                'recurrence_rate': 0.0,
                'determinism': 0.0,
                'mean_line_length': 0.0
            }
        
    def _count_lines(self, diagonal: np.ndarray) -> List[int]:
        """Helper function to count diagonal line lengths in recurrence plot."""
        lines = []
        current_line = 0
        for val in diagonal:
            if val:
                current_line += 1
            elif current_line > 0:
                if current_line >= 2:  # Only count lines of length >= 2
                    lines.append(current_line)
                current_line = 0
        if current_line >= 2:
            lines.append(current_line)
        return lines
        
    def compute_information_metrics(self, signal_data: np.ndarray, 
                                  window_size: Optional[int] = None) -> Dict:
        """Compute information theory based metrics."""
        try:
            # Adjust window size based on signal length
            if window_size is None:
                window_size = min(1000, len(signal_data) // 10)
            window_size = max(10, min(window_size, len(signal_data) // 2))
            
            # Prepare windows
            n_windows = max(1, len(signal_data) // (window_size // 2) - 1)
            entropy = np.zeros(n_windows)
            
            for i in range(n_windows):
                start = i * (window_size // 2)
                window = signal_data[start:start + window_size]
                
                # Compute histogram for entropy calculation
                hist, _ = np.histogram(window, bins='auto', density=True)
                hist = hist[hist > 0]  # Remove zero probabilities
                if len(hist) > 0:
                    entropy[i] = -np.sum(hist * np.log2(hist))
                
            # Compute complexity measures
            mean_entropy = float(np.mean(entropy))
            complexity = float(np.std(entropy) / (mean_entropy + 1e-10))
            
            return {
                'entropy_series': entropy.tolist(),
                'complexity_index': complexity,
                'mean_entropy': mean_entropy
            }
        except Exception as e:
            logger.warning(f"Information metrics calculation failed: {str(e)}")
            return {
                'entropy_series': [],
                'complexity_index': 0.0,
                'mean_entropy': 0.0
            }
        
    def analyze_network_states(self, signal_data: np.ndarray,
                             n_states: int = 5,
                             window_size: Optional[int] = None) -> Dict:
        """Identify distinct network states using clustering."""
        try:
            # Adjust window size based on signal length
            if window_size is None:
                window_size = min(1000, len(signal_data) // 10)
            window_size = max(10, min(window_size, len(signal_data) // 2))
            
            # Extract features in sliding windows
            n_windows = max(1, len(signal_data) // (window_size // 2) - 1)
            features = np.zeros((n_windows, 4))
            
            for i in range(n_windows):
                start = i * (window_size // 2)
                window = signal_data[start:start + window_size]
                
                features[i] = [
                    np.mean(window),
                    np.std(window),
                    np.max(window) - np.min(window),
                    len(np.where(np.diff(window) > 0)[0]) / window_size
                ]
                
            # Adjust number of states based on data size
            n_states = min(n_states, n_windows // 2)
            if n_states < 2:
                raise ValueError("Not enough data for state analysis")
                
            # Normalize features
            features = self.scaler.fit_transform(features)
            
            # Cluster into states
            kmeans = KMeans(n_clusters=n_states, random_state=42)
            states = kmeans.fit_predict(features)
            
            # Analyze state transitions
            transitions = np.diff(states)
            transition_matrix = np.zeros((n_states, n_states))
            for i in range(len(transitions)):
                transition_matrix[states[i], states[i+1]] += 1
                
            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = np.divide(transition_matrix, 
                                       row_sums[:, np.newaxis],
                                       where=row_sums[:, np.newaxis] != 0)
            
            return {
                'states': states.tolist(),
                'state_centers': kmeans.cluster_centers_.tolist(),
                'transition_matrix': transition_matrix.tolist(),
                'state_frequencies': (np.bincount(states) / len(states)).tolist()
            }
        except Exception as e:
            logger.warning(f"Network state analysis failed: {str(e)}")
            return {
                'states': [],
                'state_centers': [],
                'transition_matrix': [],
                'state_frequencies': []
            }
        
    def comprehensive_analysis(self, signal_data: np.ndarray) -> Dict:
        """Perform comprehensive analysis using all methods."""
        logger.info("Starting comprehensive analysis...")
        start_time = time.time()
        
        # Preprocess and denoise
        logger.info("Denoising signal...")
        denoised_data = self.denoise_signal(signal_data)
        
        # Detect spikes
        logger.info("Detecting spikes...")
        spike_times = self.detect_spikes(denoised_data)
        
        # Initialize linguistic analyzer
        logger.info("Performing linguistic analysis...")
        linguistic_analyzer = LinguisticAnalyzer(self.sampling_rate)
        
        # Run analyses
        results = {
            'multi_scale': self.compute_multi_scale_features(denoised_data),
            'linguistic': linguistic_analyzer.analyze_linguistic_features(spike_times),
            'spikes': {
                'times': spike_times.tolist(),
                'count': len(spike_times),
                'mean_interval': np.mean(np.diff(spike_times)) if len(spike_times) > 1 else 0
            }
        }
        
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
        
        return results 