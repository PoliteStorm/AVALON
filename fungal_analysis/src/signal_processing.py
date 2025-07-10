import numpy as np
import pywt
from scipy import signal
from typing import Tuple, List, Optional
from sqrt_wavelet_adapter import sqrt_cwt  # NEW IMPORT

class FungalSignalProcessor:
    def __init__(self, sampling_rate: float):
        """
        Initialize the signal processor
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.sampling_rate = float(sampling_rate)  # Ensure float type
        
    def preprocess_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the raw signal data
        
        Args:
            data: Raw electrical recording data
            
        Returns:
            Preprocessed signal
        """
        # Convert data to float numpy array
        data = np.array(data, dtype=np.float64)
        
        # Remove DC offset
        data = data - np.mean(data)
        
        # Apply bandpass filter (0.1-4 Hz for typical fungal signals)
        nyquist = self.sampling_rate / 2.0
        low_freq = 0.1  # 0.1 Hz lower cutoff
        high_freq = min(4.0, nyquist * 0.95)  # 4 Hz upper cutoff or 95% of Nyquist
        
        # Normalize frequencies for scipy
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design and apply filter
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data
    
    def compute_cwt(self, data: np.ndarray, 
                   wavelet: str = 'mexh',
                   scales: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Continuous Wavelet Transform
        
        Args:
            data: Input signal
            wavelet: Wavelet type ('mexh' for Mexican Hat recommended for spikes)
            scales: Custom scales for wavelet transform
            
        Returns:
            Tuple of (coefficients, frequencies)
        """
        # Ensure data is float
        data = np.array(data, dtype=np.float64)
        
        if scales is None:
            # Adjust scales based on sampling rate
            # Cover periods from 1 second to 60 seconds
            min_scale = self.sampling_rate / 60.0  # For highest frequency (1 Hz)
            max_scale = self.sampling_rate * 60.0   # For lowest frequency (1/60 Hz)
            num_scales = 100
            scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
        
        # Use √t wavelet transform instead of classic CWT
        coefficients, frequencies = sqrt_cwt(data, sampling_rate=self.sampling_rate,
                                            taus=scales, ks=None, normalize=False)
        
        return coefficients, frequencies
    
    def detect_spikes_wavelet(self, data: np.ndarray, 
                            threshold: float = 3.0) -> List[int]:
        """
        Detect spikes using wavelet transform
        
        Args:
            data: Input signal
            threshold: Number of standard deviations for spike detection
            
        Returns:
            List of spike indices
        """
        # Ensure data is float
        data = np.array(data, dtype=np.float64)
        
        # Compute CWT
        coefficients, _ = self.compute_cwt(data)
        
        # Sum across scales to get ridge lines
        ridge_line = np.sum(np.abs(coefficients), axis=0)
        
        # Detect peaks above threshold
        threshold_value = np.mean(ridge_line) + threshold * np.std(ridge_line)
        min_distance = int(0.1 * self.sampling_rate)  # Min 100ms between spikes
        if min_distance < 1:
            min_distance = 1
            
        peaks, _ = signal.find_peaks(ridge_line, height=threshold_value,
                                   distance=min_distance)
        
        return peaks.tolist()
    
    def extract_spike_features(self, data: np.ndarray, 
                             spike_indices: List[int],
                             window_ms: float = 500) -> np.ndarray:
        """
        Extract wavelet-based features for each spike
        
        Args:
            data: Input signal
            spike_indices: List of spike locations
            window_ms: Time window around spike in milliseconds
            
        Returns:
            Feature matrix for spikes
        """
        # Ensure data is float
        data = np.array(data, dtype=np.float64)
        
        window_samples = max(int(window_ms * self.sampling_rate / 1000), 2)
        features = []
        
        for spike_idx in spike_indices:
            start = max(0, spike_idx - window_samples//2)
            end = min(len(data), spike_idx + window_samples//2)
            
            # Extract spike window
            spike_window = data[start:end]
            if len(spike_window) < window_samples:
                continue
                
            # Compute wavelet transform for the window
            coeffs, _ = self.compute_cwt(spike_window)
            
            # Extract features (e.g., max coefficient values at different scales)
            feature_vector = np.max(np.abs(coeffs), axis=1)
            features.append(feature_vector)
            
        return np.array(features) if features else np.array([]) 