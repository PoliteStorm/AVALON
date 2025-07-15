#!/usr/bin/env python3
"""
Improved √t Transform with Multiple Enhancements
1. Different window functions
2. Alternative parameter ranges  
3. Refined detection methods
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from transform_validation_framework import SqrtTTransform, ValidationFramework

class ImprovedSqrtTTransform:
    """
    Improved √t transform with multiple enhancements
    """
    
    def __init__(self, sampling_rate=1.0):
        self.sampling_rate = sampling_rate
        
    def gaussian_window(self, t, tau):
        """Original Gaussian window"""
        sqrt_t = np.sqrt(t)
        return np.exp(-(sqrt_t / tau)**2)
    
    def morlet_window(self, t, tau):
        """Morlet wavelet-inspired window"""
        sqrt_t = np.sqrt(t)
        omega_0 = 2 * np.pi  # Central frequency
        return np.exp(-(sqrt_t / tau)**2) * np.cos(omega_0 * sqrt_t / tau)
    
    def mexican_hat_window(self, t, tau):
        """Mexican hat wavelet-inspired window"""
        sqrt_t = np.sqrt(t)
        x = sqrt_t / tau
        return (1 - x**2) * np.exp(-x**2 / 2)
    
    def exponential_window(self, t, tau):
        """Exponential window for √t"""
        sqrt_t = np.sqrt(t)
        return np.exp(-sqrt_t / tau)
    
    def polynomial_window(self, t, tau):
        """Polynomial window with √t scaling"""
        sqrt_t = np.sqrt(t)
        x = sqrt_t / tau
        return np.maximum(0, 1 - x**2)  # Quadratic window
    
    def transform_with_window(self, V, k_values, tau_values, window_func='gaussian', t_max=None):
        """
        Compute √t transform with different window functions
        """
        if t_max is None:
            t_max = len(V) / self.sampling_rate
            
        t = np.arange(0, t_max, 1/self.sampling_rate)
        
        # Select window function
        if window_func == 'gaussian':
            window_func = self.gaussian_window
        elif window_func == 'morlet':
            window_func = self.morlet_window
        elif window_func == 'mexican_hat':
            window_func = self.mexican_hat_window
        elif window_func == 'exponential':
            window_func = self.exponential_window
        elif window_func == 'polynomial':
            window_func = self.polynomial_window
        else:
            window_func = self.gaussian_window
        
        W = np.zeros((len(k_values), len(tau_values)), dtype=complex)
        
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                # Compute the integrand with selected window
                window = window_func(t, tau)
                phase = np.exp(-1j * k * np.sqrt(t))
                integrand = V[:len(t)] * window * phase
                
                # Numerical integration
                W[i, j] = np.trapz(integrand, t)
                
        return W
    
    def alternative_parameter_ranges(self):
        """
        Generate alternative parameter ranges for testing
        """
        # Range 1: Focus on low frequencies (biological range)
        k_range1 = np.logspace(-2, 0, 20)  # 0.01 to 1
        tau_range1 = np.logspace(1, 3, 20)  # 10 to 1000
        
        # Range 2: Focus on medium frequencies
        k_range2 = np.logspace(-1, 1, 20)  # 0.1 to 10
        tau_range2 = np.logspace(0, 2, 20)  # 1 to 100
        
        # Range 3: Focus on high frequencies
        k_range3 = np.logspace(0, 2, 20)  # 1 to 100
        tau_range3 = np.logspace(-1, 1, 20)  # 0.1 to 10
        
        # Range 4: Dense sampling in biological range
        k_range4 = np.linspace(0.01, 1.0, 30)  # Linear spacing
        tau_range4 = np.linspace(10, 1000, 30)  # Linear spacing
        
        # Range 5: Adaptive based on signal length
        signal_length = 10000
        k_range5 = np.logspace(-2, 0, 25)  # Focus on low frequencies
        tau_range5 = np.logspace(1, np.log10(signal_length/10), 25)
        
        return {
            'biological_low': (k_range1, tau_range1),
            'medium_freq': (k_range2, tau_range2),
            'high_freq': (k_range3, tau_range3),
            'linear_dense': (k_range4, tau_range4),
            'adaptive': (k_range5, tau_range5)
        }
    
    def refined_detection_methods(self, W, k_values, tau_values, method='multi_scale'):
        """
        Refined detection methods for features
        """
        magnitude = np.abs(W)
        
        if method == 'multi_scale':
            # Multi-scale peak detection
            features = self.multi_scale_peak_detection(magnitude, k_values, tau_values)
        elif method == 'statistical':
            # Statistical thresholding
            features = self.statistical_thresholding(magnitude, k_values, tau_values)
        elif method == 'adaptive':
            # Adaptive thresholding
            features = self.adaptive_thresholding(magnitude, k_values, tau_values)
        elif method == 'clustering':
            # Clustering-based detection
            features = self.clustering_detection(magnitude, k_values, tau_values)
        else:
            features = self.basic_detection(magnitude, k_values, tau_values)
        
        return features
    
    def multi_scale_peak_detection(self, magnitude, k_values, tau_values):
        """Multi-scale peak detection"""
        features = []
        
        # Find peaks at different scales
        for scale in [1, 2, 4]:
            # Downsample magnitude
            if scale > 1:
                k_down = k_values[::scale]
                tau_down = tau_values[::scale]
                mag_down = magnitude[::scale, ::scale]
            else:
                k_down, tau_down, mag_down = k_values, tau_values, magnitude
            
            # Find peaks in downsampled data
            peaks = signal.find_peaks(mag_down.flatten(), height=np.percentile(mag_down, 90))[0]
            
            for peak_idx in peaks:
                i, j = np.unravel_index(peak_idx, mag_down.shape)
                if i < len(k_down) and j < len(tau_down):
                    features.append({
                        'k': k_down[i],
                        'tau': tau_down[j],
                        'magnitude': mag_down[i, j],
                        'scale': scale
                    })
        
        return features
    
    def statistical_thresholding(self, magnitude, k_values, tau_values):
        """Statistical thresholding based on magnitude distribution"""
        features = []
        
        # Calculate statistics
        mag_flat = magnitude.flatten()
        mean_mag = np.mean(mag_flat)
        std_mag = np.std(mag_flat)
        
        # Use 2 standard deviations above mean as threshold
        threshold = mean_mag + 2 * std_mag
        
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                if magnitude[i, j] > threshold:
                    # Check if it's a local maximum
                    is_local_max = True
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if (0 <= ni < len(k_values) and 
                                0 <= nj < len(tau_values) and
                                magnitude[ni, nj] > magnitude[i, j]):
                                is_local_max = False
                                break
                        if not is_local_max:
                            break
                    
                    if is_local_max:
                        features.append({
                            'k': k,
                            'tau': tau,
                            'magnitude': magnitude[i, j],
                            'z_score': (magnitude[i, j] - mean_mag) / std_mag
                        })
        
        return features
    
    def adaptive_thresholding(self, magnitude, k_values, tau_values):
        """Adaptive thresholding based on local statistics"""
        features = []
        
        # Calculate local statistics
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(magnitude, size=3)
        local_std = uniform_filter(magnitude**2, size=3) - local_mean**2
        local_std = np.sqrt(np.maximum(local_std, 0))
        
        # Adaptive threshold: local_mean + 2 * local_std
        threshold = local_mean + 2 * local_std
        
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                if magnitude[i, j] > threshold[i, j]:
                    features.append({
                        'k': k,
                        'tau': tau,
                        'magnitude': magnitude[i, j],
                        'local_threshold': threshold[i, j]
                    })
        
        return features
    
    def clustering_detection(self, magnitude, k_values, tau_values):
        """Clustering-based feature detection"""
        features = []
        
        # Flatten magnitude and find high-value regions
        mag_flat = magnitude.flatten()
        
        # Use percentile-based clustering
        high_threshold = np.percentile(mag_flat, 95)
        high_indices = np.where(mag_flat > high_threshold)[0]
        
        if len(high_indices) > 0:
            # Cluster high-value points
            from sklearn.cluster import DBSCAN
            high_points = np.column_stack([
                high_indices // len(tau_values),  # k index
                high_indices % len(tau_values)    # tau index
            ])
            
            if len(high_points) > 1:
                clustering = DBSCAN(eps=2, min_samples=1).fit(high_points)
                
                # Take the strongest point from each cluster
                for cluster_id in set(clustering.labels_):
                    cluster_points = high_points[clustering.labels_ == cluster_id]
                    cluster_magnitudes = [magnitude[i, j] for i, j in cluster_points]
                    best_idx = np.argmax(cluster_magnitudes)
                    i, j = cluster_points[best_idx]
                    
                    features.append({
                        'k': k_values[i],
                        'tau': tau_values[j],
                        'magnitude': magnitude[i, j],
                        'cluster_size': len(cluster_points)
                    })
            else:
                # Single high point
                i, j = high_points[0]
                features.append({
                    'k': k_values[i],
                    'tau': tau_values[j],
                    'magnitude': magnitude[i, j],
                    'cluster_size': 1
                })
        
        return features
    
    def basic_detection(self, magnitude, k_values, tau_values):
        """Basic detection method"""
        features = []
        threshold = np.percentile(magnitude, 95)
        
        for i, k in enumerate(k_values):
            for j, tau in enumerate(tau_values):
                if magnitude[i, j] > threshold:
                    features.append({
                        'k': k,
                        'tau': tau,
                        'magnitude': magnitude[i, j]
                    })
        
        return features
    
    def comprehensive_test(self, signal, name="test_signal"):
        """
        Comprehensive test with all improvements
        """
        print(f"\nComprehensive Testing: {name}")
        print("=" * 60)
        
        # Get alternative parameter ranges
        param_ranges = self.alternative_parameter_ranges()
        
        # Window functions to test
        window_functions = ['gaussian', 'morlet', 'mexican_hat', 'exponential', 'polynomial']
        
        # Detection methods to test
        detection_methods = ['multi_scale', 'statistical', 'adaptive', 'clustering', 'basic']
        
        all_results = {}
        
        for window_func in window_functions:
            print(f"\nWindow Function: {window_func}")
            print("-" * 40)
            
            for param_name, (k_vals, tau_vals) in param_ranges.items():
                print(f"  Parameter Range: {param_name}")
                
                # Apply transform with current window and parameters
                W = self.transform_with_window(signal, k_vals, tau_vals, window_func)
                
                for det_method in detection_methods:
                    features = self.refined_detection_methods(W, k_vals, tau_vals, det_method)
                    
                    result_key = f"{window_func}_{param_name}_{det_method}"
                    all_results[result_key] = {
                        'features': features,
                        'count': len(features),
                        'window': window_func,
                        'params': param_name,
                        'method': det_method
                    }
                    
                    print(f"    {det_method}: {len(features)} features")
        
        return all_results
    
    def analyze_comprehensive_results(self, results):
        """
        Analyze comprehensive test results
        """
        print("\n" + "=" * 60)
        print("COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        # Group results by window function
        window_results = {}
        for key, result in results.items():
            window = result['window']
            if window not in window_results:
                window_results[window] = []
            window_results[window].append(result)
        
        # Find best performing combinations
        best_combinations = []
        for window, window_data in window_results.items():
            for result in window_data:
                if result['count'] > 0:  # Only consider combinations that find features
                    best_combinations.append({
                        'window': result['window'],
                        'params': result['params'],
                        'method': result['method'],
                        'count': result['count']
                    })
        
        # Sort by feature count
        best_combinations.sort(key=lambda x: x['count'], reverse=True)
        
        print("\nBest Performing Combinations:")
        for i, combo in enumerate(best_combinations[:10]):
            print(f"  {i+1}. {combo['window']} + {combo['params']} + {combo['method']}: {combo['count']} features")
        
        return best_combinations

def main():
    """
    Run comprehensive improved testing
    """
    # Create test signals
    t = np.arange(10000)
    sqrt_t = np.sqrt(t)
    
    # Test signal with √t structure
    test_signal = np.sin(2 * np.pi * 0.1 * sqrt_t) + 0.1 * np.random.normal(0, 1, 10000)
    
    # Test diffusion signal
    diffusion_signal = np.zeros(10000)
    for i in range(1, 10000):
        diffusion_signal[i] = diffusion_signal[i-1] + np.random.normal(0, np.sqrt(0.1))
    
    # Initialize improved transform
    improved_transform = ImprovedSqrtTTransform()
    
    # Test √t signal
    results1 = improved_transform.comprehensive_test(test_signal, "√t Signal")
    best1 = improved_transform.analyze_comprehensive_results(results1)
    
    # Test diffusion signal
    results2 = improved_transform.comprehensive_test(diffusion_signal, "Diffusion Signal")
    best2 = improved_transform.analyze_comprehensive_results(results2)
    
    return results1, results2, best1, best2

if __name__ == "__main__":
    main() 