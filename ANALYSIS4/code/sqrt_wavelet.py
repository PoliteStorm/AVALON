import numpy as np
from scipy import signal
import pywt
from typing import Optional, Tuple, Generator
from tqdm import tqdm

class SqrtWaveletTransform:
    """Implementation of the square root time wavelet transform with memory-efficient chunking."""
    
    def __init__(self, num_scales: int = 50, chunk_size: int = 10000):
        self.num_scales = num_scales
        self.chunk_size = max(chunk_size, 100)  # Ensure minimum chunk size
        
    def mother_wavelet(self, t: np.ndarray) -> np.ndarray:
        """
        Mother wavelet function (Morlet wavelet)
        """
        return np.exp(1j * 5.0 * t) * np.exp(-t**2 / 2.0)
        
    def get_scales(self, signal_length: int) -> np.ndarray:
        """Get wavelet scales appropriate for signal length"""
        return np.logspace(0, 2, self.num_scales)
        
    def process_chunk(self, chunk: np.ndarray, scales: np.ndarray,
                     start_idx: int, total_length: int,
                     overlap: int = 10) -> Tuple[np.ndarray, slice]:
        """
        Process a single chunk of data with overlap
        
        Args:
            chunk: Input signal chunk
            scales: Wavelet scales
            start_idx: Starting index of chunk in full signal
            total_length: Total length of the signal
            overlap: Number of points to overlap
            
        Returns:
            Tuple of (coefficients, valid slice)
        """
        N = len(chunk)
        coeffs = np.zeros((len(scales), N), dtype=complex)
        
        # Create sqrt time grid for entire signal
        t = np.arange(total_length)
        sqrt_t = np.sqrt(t)
        
        # Extract relevant portion of sqrt time grid
        chunk_t = sqrt_t[start_idx:start_idx + N]
        
        for idx, scale in enumerate(scales):
            # Create wavelet at current scale
            wavelet = self.mother_wavelet(chunk_t / scale)
            wavelet = wavelet / np.sqrt(scale)
            
            # Convolve
            coeffs[idx, :] = signal.convolve(chunk, wavelet, mode='same', method='direct')
        
        # Define valid region (excluding overlap)
        if start_idx == 0:
            valid_start = 0
            valid_end = N - overlap
        elif start_idx + N >= total_length:
            valid_start = overlap
            valid_end = N
        else:
            valid_start = overlap
            valid_end = N - overlap
            
        return coeffs, slice(valid_start, valid_end)
        
    def transform(self, data: np.ndarray, sampling_rate: float = 1.0) -> np.ndarray:
        """
        Perform the square root time wavelet transform with chunking
        
        Args:
            data: Input signal
            sampling_rate: Sampling rate of the signal
            
        Returns:
            2D array of wavelet coefficients (scales x time)
        """
        N = len(data)
        scales = self.get_scales(N)
        overlap = min(10, self.chunk_size // 4)  # Adaptive overlap
        step = max(1, self.chunk_size - 2*overlap)  # Ensure positive step
        
        try:
            # Initialize output array
            coeffs = np.zeros((len(scales), N), dtype=complex)
            
            # Process data in chunks with progress bar
            with tqdm(total=N, desc="Processing chunks") as pbar:
                for i in range(0, N, step):
                    # Get chunk with overlap
                    chunk_start = max(0, i - overlap)
                    chunk_end = min(N, i + self.chunk_size + overlap)
                    chunk = data[chunk_start:chunk_end]
                    
                    # Process chunk
                    chunk_coeffs, valid_slice = self.process_chunk(
                        chunk, scales, chunk_start, N, overlap
                    )
                    
                    # Copy valid portion to output
                    out_start = chunk_start + valid_slice.start
                    out_end = chunk_start + valid_slice.stop
                    coeffs[:, out_start:out_end] = chunk_coeffs[:, valid_slice]
                    
                    pbar.update(min(step, N - i))
                    
        except MemoryError:
            raise MemoryError(
                "Ran out of memory. Try:\n"
                f"1. Reducing num_scales (currently {self.num_scales})\n"
                f"2. Reducing chunk_size (currently {self.chunk_size})\n"
                "3. Downsampling your input data"
            )
            
        return coeffs
        
    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Perform the inverse square root time wavelet transform
        
        Args:
            coeffs: Wavelet coefficients
            
        Returns:
            Reconstructed signal
        """
        # Get scales used in transform
        N = coeffs.shape[1]
        scales = self.get_scales(N)
        
        # Initialize output
        reconstructed = np.zeros(N)
        
        # Create sqrt time grid
        t = np.arange(N)
        sqrt_t = np.sqrt(t)
        
        # Reconstruct signal
        for i, scale in enumerate(scales):
            # Create wavelet at current scale
            wavelet = self.mother_wavelet(sqrt_t / scale)
            wavelet = wavelet / np.sqrt(scale)
            
            # Inverse transform at this scale
            reconstructed += np.real(
                signal.convolve(
                    coeffs[i, :],
                    np.conj(wavelet)[::-1],  # Time reverse for reconstruction
                    mode='same',
                    method='direct'
                )
            )
            
        # Normalize
        reconstructed /= len(scales)
        
        return reconstructed 