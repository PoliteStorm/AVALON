"""Data processing module for fungal communication analysis."""

import numpy as np
import scipy.signal
from typing import Dict, Optional
from .w_transform_analyzer import WTransformAnalyzer, WTransformConfig
import gc
import psutil

def process_data_chunk(chunk_data: np.ndarray, w_resolution: int, q_threshold: float,
                      max_memory_gb: Optional[float] = None) -> Dict:
    """
    Process a chunk of data with sophisticated W-transform and quantum analysis.
    
    Args:
        chunk_data: Input voltage data chunk
        w_resolution: Resolution for W-transform
        q_threshold: Threshold for quantum analysis
        max_memory_gb: Maximum memory usage in GB (None for no limit)
    
    Returns:
        Dictionary containing analysis results
    """
    result = {}
    
    try:
        # Check input data
        if not isinstance(chunk_data, np.ndarray) or chunk_data.size == 0:
            raise ValueError("Invalid input data")
            
        # Monitor memory usage
        if max_memory_gb is not None:
            current_memory = psutil.Process().memory_info().rss / (1024**3)  # GB
            if current_memory > max_memory_gb:
                raise MemoryError(f"Memory usage ({current_memory:.1f}GB) exceeds limit ({max_memory_gb}GB)")
        
        # Initialize W-transform analyzer with research-backed configuration
        w_config = WTransformConfig(
            min_scale=0.1,  # Minimum scale for fine temporal features
            max_scale=10.0,  # Maximum scale for long-term patterns
            num_scales=w_resolution,  # Scale resolution
            k_range=(-10, 10),  # Frequency range
            num_k=64,  # Frequency resolution
            wavelet_type='morlet',  # Optimal for biological signals
            significance_level=0.05,  # Standard statistical significance
            monte_carlo_iterations=1000,  # For robust confidence estimation
            chunk_size=min(1000, len(chunk_data))  # Adaptive chunk size
        )
        w_analyzer = WTransformAnalyzer(w_config)
        
        # Generate time array for the chunk
        time_data = np.arange(len(chunk_data)) / len(chunk_data)  # Normalized time
        
        # Compute sophisticated W-transform with memory management
        try:
            w_transform_results = w_analyzer.compute_w_transform(chunk_data, time_data)
        except MemoryError:
            # If memory error occurs, try with reduced resolution
            w_config.num_scales = max(16, w_config.num_scales // 2)
            w_config.num_k = max(32, w_config.num_k // 2)
            w_analyzer = WTransformAnalyzer(w_config)
            w_transform_results = w_analyzer.compute_w_transform(chunk_data, time_data)
        
        # Extract key features from W-transform
        # Convert large arrays to more memory-efficient types
        result['w_transform'] = {
            'transform': w_transform_results['transform'].astype(np.complex64).tolist(),  # Use single precision
            'power': w_transform_results['power'].astype(np.float32).tolist(),  # Use single precision
            'scales': w_transform_results['scales'].astype(np.float32).tolist(),
            'frequencies': w_transform_results['frequencies'].astype(np.float32).tolist(),
            'significance': w_transform_results['significance'].astype(np.float32).tolist(),
            'phase_coherence': w_transform_results['phase_coherence'].astype(np.float32).tolist(),
            'ridges': w_transform_results['ridges'],  # Already optimized
            'parameters': w_transform_results['parameters']
        }
        
        # Basic quantum state analysis with memory optimization
        # Use lower precision and efficient binning
        chunk_min, chunk_max = np.min(chunk_data), np.max(chunk_data)
        bins = np.linspace(chunk_min, chunk_max, min(100, len(chunk_data)))
        digitized = np.digitize(chunk_data, bins)
        energy_levels = np.unique(digitized)
        coherence = np.exp(-np.std(chunk_data) / q_threshold)
        
        result['quantum_state'] = {
            'energy_levels': int(len(energy_levels)),
            'coherence': float(coherence),
            'threshold': float(q_threshold)
        }
        
        # Extract sophisticated patterns using W-transform ridges
        # Optimize peak detection for memory efficiency
        peaks, _ = scipy.signal.find_peaks(chunk_data, distance=max(1, len(chunk_data)//100))
        
        result['patterns'] = {
            'mean': float(np.mean(chunk_data)),
            'std': float(np.std(chunk_data)),
            'peaks': len(peaks),
            'significant_patterns': [
                {
                    'scale': ridge['scale_indices'],
                    'frequency': ridge['k_indices'],
                    'amplitude': float(ridge['amplitude'])  # Ensure Python native type
                }
                for ridge in w_transform_results['ridges']
            ]
        }
        
        # Force cleanup of large temporary arrays
        del w_transform_results
        gc.collect()
        
        return result
        
    except Exception as e:
        # Return partial results if available, otherwise error info
        if not result:
            result = {
                'error': str(e),
                'partial_results': False
            }
        else:
            result['error'] = str(e)
            result['partial_results'] = True
        
        return result 