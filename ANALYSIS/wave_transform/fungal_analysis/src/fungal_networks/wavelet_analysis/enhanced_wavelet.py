"""
Enhanced Wavelet Analysis Framework for Fungal Network Data

This module provides tools for analyzing fungal network data using wavelet transforms,
with support for both time series (.mat) and spatial coordinate (.csv) data.
"""

import numpy as np
import pywt
from scipy import signal
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from fungal_networks.wavelet_analysis.sqrt_adapter import sqrt_cwt  # NEW IMPORT

logger = logging.getLogger(__name__)

class EnhancedWaveletAnalysis:
    """Enhanced wavelet analysis framework for fungal network data."""
    
    def __init__(self, wavelet_type: str = 'morl', sampling_rate: float = 1.0):
        """
        Initialize the wavelet analysis framework.
        
        Args:
            wavelet_type: Type of wavelet to use (default: 'morl' for Morlet wavelet)
            sampling_rate: Sampling rate of the data in Hz (default: 1.0)
        """
        self.wavelet_type = wavelet_type
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        
    def preprocess_signal(self, signal_data: np.ndarray, detrend: bool = True,
                         denoise: bool = True) -> np.ndarray:
        """
        Preprocess the input signal.
        
        Args:
            signal_data: Input signal as numpy array
            detrend: Whether to remove linear trend
            denoise: Whether to apply denoising
            
        Returns:
            Preprocessed signal
        """
        # Ensure 1D array
        signal_data = np.asarray(signal_data).flatten()
        
        # Remove NaN values
        signal_data = np.nan_to_num(signal_data, nan=np.nanmean(signal_data))
        
        # Detrend if requested
        if detrend:
            signal_data = signal.detrend(signal_data)
            
        # Standardize
        signal_data = self.scaler.fit_transform(signal_data.reshape(-1, 1)).flatten()
        
        # Denoise if requested
        if denoise:
            # Wavelet denoising
            wavelet = 'db4'  # Daubechies 4 wavelet
            level = int(np.log2(len(signal_data)))
            coeffs = pywt.wavedec(signal_data, wavelet, level=level)
            
            # Threshold coefficients
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            
            # Reconstruct signal
            signal_data = pywt.waverec(coeffs, wavelet)
            
        return signal_data
        
    def compute_wavelet_transform(self, signal_data: np.ndarray,
                                scales: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute continuous wavelet transform.
        
        Args:
            signal_data: Input signal
            scales: Wavelet scales (if None, automatically determined)
            
        Returns:
            Tuple of (wavelet coefficients, frequencies)
        """
        # Determine scales if not provided
        if scales is None:
            scales = np.arange(1, min(len(signal_data) // 2, 128))
            
        # Compute CWT
        coefficients, frequencies = sqrt_cwt(signal_data,
                                             sampling_rate=self.sampling_rate,
                                             taus=self.scales,
                                             ks=None,
                                             normalize=False)
        
        return coefficients, frequencies
        
    def detect_patterns(self, coefficients: np.ndarray, frequencies: np.ndarray,
                       threshold: float = 2.0) -> Dict:
        """
        Detect significant patterns in wavelet coefficients.
        
        Args:
            coefficients: Wavelet coefficients
            frequencies: Corresponding frequencies
            threshold: Z-score threshold for significance
            
        Returns:
            Dictionary containing detected patterns
        """
        # Compute power spectrum
        power = np.abs(coefficients) ** 2
        
        # Normalize power
        power_z = zscore(power, axis=1)
        
        # Find significant coefficients
        significant = power_z > threshold
        
        # Find peak frequencies
        peak_freqs = frequencies[np.where(np.any(significant, axis=1))[0]]
        
        # Find peak times
        peak_times = np.where(np.any(significant, axis=0))[0]
        
        # Compute average power distribution
        power_dist = np.mean(power, axis=1)
        
        return {
            'peak_frequencies': peak_freqs,
            'peak_times': peak_times,
            'power_distribution': power_dist,
            'significance_mask': significant
        }
        
    def analyze_signal(self, signal_data: np.ndarray, detrend: bool = True,
                      denoise: bool = True) -> Dict:
        """
        Perform complete wavelet analysis on a signal.
        
        Args:
            signal_data: Input signal
            detrend: Whether to remove linear trend
            denoise: Whether to apply denoising
            
        Returns:
            Dictionary containing analysis results
        """
        # Preprocess signal
        processed_signal = self.preprocess_signal(signal_data, detrend, denoise)
        
        # Compute wavelet transform
        coefficients, frequencies = self.compute_wavelet_transform(processed_signal)
        
        # Detect patterns
        patterns = self.detect_patterns(coefficients, frequencies)
        
        # Add raw coefficients and frequencies
        patterns['coefficients'] = coefficients
        patterns['frequencies'] = frequencies
        
        return patterns
        
    def plot_analysis(self, results: Dict, title: str = '') -> plt.Figure:
        """
        Create visualization of analysis results.
        
        Args:
            results: Analysis results from analyze_signal
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplots
        gs = plt.GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])  # Wavelet coefficients
        ax2 = fig.add_subplot(gs[1, 0])  # Power spectrum
        ax3 = fig.add_subplot(gs[1, 1])  # Peak frequencies
        ax4 = fig.add_subplot(gs[2, :])  # Temporal patterns
        
        # Plot wavelet coefficients
        im = ax1.imshow(
            np.abs(results['coefficients']),
            aspect='auto',
            cmap='viridis'
        )
        ax1.set_title('Wavelet Coefficients')
        ax1.set_ylabel('Scale')
        plt.colorbar(im, ax=ax1)
        
        # Plot power spectrum
        power_dist = results['power_distribution']
        ax2.plot(results['frequencies'], power_dist)
        ax2.set_title('Power Spectrum')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Power')
        
        # Plot peak frequencies
        if len(results['peak_frequencies']) > 0:
            ax3.hist(results['peak_frequencies'], bins=20)
        ax3.set_title('Peak Frequency Distribution')
        ax3.set_xlabel('Frequency')
        ax3.set_ylabel('Count')
        
        # Plot temporal patterns
        temporal = np.sum(results['significance_mask'], axis=0)
        ax4.plot(temporal)
        ax4.set_title('Temporal Pattern Strength')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Number of Significant Coefficients')
        
        # Set overall title
        fig.suptitle(title, fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig 