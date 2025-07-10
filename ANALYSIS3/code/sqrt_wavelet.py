import numpy as np
from scipy import signal
from typing import Optional, Tuple

class SqrtWaveletTransform:
    """Implementation of the square root time wavelet transform."""
    
    def __init__(self, num_scales: int = 50):
        self.num_scales = num_scales
        
    def mother_wavelet(self, t: np.ndarray) -> np.ndarray:
        """
        Mother wavelet function (Morlet wavelet)
        """
        return np.exp(1j * 5.0 * t) * np.exp(-t**2 / 2.0)
        
    def transform(self, data: np.ndarray, sampling_rate: float = 1.0) -> np.ndarray:
        """
        Perform the square root time wavelet transform
        
        Args:
            data: Input signal
            sampling_rate: Sampling rate of the signal
            
        Returns:
            2D array of wavelet coefficients (scales x time)
        """
        N = len(data)
        scales = np.logspace(0, 3, self.num_scales)
        coeffs = np.zeros((len(scales), N), dtype=complex)
        
        # Create sqrt time grid
        t = np.arange(N) / sampling_rate
        sqrt_t = np.sqrt(t)
        
        for idx, scale in enumerate(scales):
            # Create wavelet at current scale
            wavelet = self.mother_wavelet(sqrt_t / scale)
            
            # Normalize
            wavelet = wavelet / np.sqrt(scale)
            
            # Convolve with signal
            coeffs[idx, :] = signal.convolve(data, wavelet, mode='same')
            
        return coeffs
        
    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Perform the inverse square root time wavelet transform
        
        Args:
            coeffs: Wavelet coefficients
            
        Returns:
            Reconstructed signal
        """
        # This is a simplified reconstruction
        return np.real(np.sum(coeffs, axis=0)) 