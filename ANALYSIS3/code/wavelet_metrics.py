import numpy as np
from scipy.stats import entropy
from typing import Tuple, List
from sqrt_wavelet import SqrtWaveletTransform

class WaveletMetrics:
    """Calculate various metrics from wavelet transforms of fungal signals"""
    
    def __init__(self):
        self.wavelet = SqrtWaveletTransform()
        
    def calculate_signal_complexity(self, signal: np.ndarray) -> float:
        """
        Calculate signal complexity using wavelet entropy
        Higher values indicate more complex signals
        """
        # Get wavelet coefficients
        coeffs = self.wavelet.transform(signal)
        
        # Calculate normalized magnitude spectrum
        magnitude = np.abs(coeffs)
        prob_dist = magnitude / np.sum(magnitude)
        
        # Calculate Shannon entropy
        return entropy(prob_dist.flatten())
    
    def calculate_response_latency(self, signal: np.ndarray, 
                                 event_time: int) -> float:
        """
        Calculate response latency after an environmental change
        Returns time (in samples) until significant response
        """
        # Get wavelet coefficients before and after event
        pre_event = signal[:event_time]
        post_event = signal[event_time:]
        
        pre_coeffs = self.wavelet.transform(pre_event)
        post_coeffs = self.wavelet.transform(post_event)
        
        # Calculate magnitude change
        pre_mag = np.abs(pre_coeffs)
        post_mag = np.abs(post_coeffs)
        
        # Find first significant deviation
        baseline = np.mean(pre_mag) + 2 * np.std(pre_mag)
        response_idx = np.where(post_mag > baseline)[0]
        
        return response_idx[0] if len(response_idx) > 0 else np.inf
    
    def calculate_pattern_consistency(self, signal: np.ndarray, 
                                   window_size: int = 1000) -> float:
        """
        Calculate consistency of patterns across time windows
        Returns value between 0 (inconsistent) and 1 (highly consistent)
        """
        # Split signal into windows
        n_windows = len(signal) // window_size
        windows = np.array_split(signal[:n_windows*window_size], n_windows)
        
        # Calculate wavelet coefficients for each window
        window_coeffs = [self.wavelet.transform(w) for w in windows]
        
        # Calculate pairwise correlations between windows
        correlations = []
        for i in range(len(window_coeffs)):
            for j in range(i+1, len(window_coeffs)):
                corr = np.corrcoef(np.abs(window_coeffs[i]).flatten(),
                                 np.abs(window_coeffs[j]).flatten())[0,1]
                correlations.append(corr)
        
        # Return mean correlation
        return np.mean(correlations)
    
    def extract_characteristic_frequencies(self, signal: np.ndarray, 
                                        n_freqs: int = 5) -> List[float]:
        """
        Extract most prominent frequency components
        Returns list of frequencies in Hz
        """
        coeffs = self.wavelet.transform(signal)
        magnitude = np.abs(coeffs)
        
        # Average across time
        freq_spectrum = np.mean(magnitude, axis=1)
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(freq_spectrum)
        
        # Sort by magnitude and return top frequencies
        peak_mags = freq_spectrum[peaks]
        top_indices = np.argsort(peak_mags)[-n_freqs:]
        
        return peaks[top_indices]
    
    def detect_anomalies(self, signal: np.ndarray, 
                        threshold: float = 3.0) -> List[int]:
        """
        Detect unusual patterns in the signal
        Returns indices of anomalous events
        """
        coeffs = self.wavelet.transform(signal)
        magnitude = np.abs(coeffs)
        
        # Calculate rolling statistics
        window = 100
        rolling_mean = np.convolve(magnitude.mean(axis=0), 
                                 np.ones(window)/window, mode='valid')
        rolling_std = np.std(magnitude, axis=0)
        
        # Find points exceeding threshold standard deviations
        anomalies = np.where(np.abs(magnitude.mean(axis=0) - rolling_mean) > 
                           threshold * rolling_std)[0]
        
        return list(anomalies) 