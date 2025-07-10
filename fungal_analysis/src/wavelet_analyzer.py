import numpy as np
from scipy import signal
import pywt
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
from sqrt_wavelet_adapter import sqrt_cwt  # NEW IMPORT

class WaveletAnalyzer:
    def __init__(self, data_dir: str = "/home/kronos/AVALON/fungal_networks"):
        self.data_dir = Path(data_dir)
        
    def load_mat_data(self, filepath: Path) -> np.ndarray:
        """Load time series data from .mat file"""
        try:
            data = loadmat(filepath)
            # Extract coordinates or time series data
            # Assuming the data is stored in a field named 'coordinates'
            if 'coordinates' in data:
                return data['coordinates']
            return None
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None

    def preprocess_signal(self, data: np.ndarray) -> np.ndarray:
        """Preprocess the signal for wavelet analysis"""
        if data is None or len(data) == 0:
            return None
            
        # If data is multi-dimensional, we'll analyze the first dimension
        if len(data.shape) > 1:
            data = data[:, 0]
            
        # Remove DC offset
        data = data - np.mean(data)
        
        # Apply bandpass filter (0.01-0.45 normalized frequency)
        b, a = signal.butter(4, [0.01, 0.45], btype='band')
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data

    def compute_wavelet_transform(self, data: np.ndarray, 
                                wavelet: str = 'cmor1.5-1.0',
                                scales: np.ndarray = None) -> tuple:
        """Compute continuous wavelet transform"""
        if scales is None:
            scales = np.arange(1, 128)
            
        # Compute CWT
        coef, freqs = sqrt_cwt(data, sampling_rate=1.0, taus=scales)
        
        return coef, freqs

    def detect_spikes(self, cwt_coef: np.ndarray, threshold: float = 2.0) -> list:
        """Detect spike-like patterns in wavelet coefficients"""
        # Compute standard deviation for thresholding
        std_dev = np.std(cwt_coef)
        threshold_val = std_dev * threshold
        
        # Find peaks in wavelet coefficients
        peaks = []
        for scale_idx in range(cwt_coef.shape[0]):
            scale_peaks = signal.find_peaks(np.abs(cwt_coef[scale_idx]), 
                                          height=threshold_val,
                                          distance=20)[0]
            peaks.extend([(scale_idx, peak) for peak in scale_peaks])
            
        return peaks

    def analyze_temporal_patterns(self, peaks: list, 
                                min_pattern_length: int = 3) -> dict:
        """Analyze temporal patterns in detected spikes"""
        if not peaks:
            return {}
            
        # Convert peaks to time series
        peak_times = sorted(list(set([p[1] for p in peaks])))
        intervals = np.diff(peak_times)
        
        # Basic statistics
        stats = {
            'total_spikes': len(peak_times),
            'mean_interval': np.mean(intervals) if len(intervals) > 0 else 0,
            'std_interval': np.std(intervals) if len(intervals) > 0 else 0
        }
        
        # Find repeating patterns
        patterns = {}
        for length in range(min_pattern_length, min(len(intervals) + 1, 10)):
            for i in range(len(intervals) - length + 1):
                pattern = tuple(intervals[i:i+length])
                if pattern in patterns:
                    patterns[pattern] += 1
                else:
                    patterns[pattern] = 1
                    
        # Keep only patterns that repeat at least twice
        significant_patterns = {k: v for k, v in patterns.items() if v > 1}
        stats['repeating_patterns'] = significant_patterns
        
        return stats

    def plot_wavelet_analysis(self, data: np.ndarray, cwt_coef: np.ndarray, 
                            freqs: np.ndarray, peaks: list, 
                            title: str = "Wavelet Analysis",
                            save_path: Path = None) -> None:
        """Plot original signal, wavelet transform, and detected spikes"""
        plt.figure(figsize=(15, 10))
        
        # Plot original signal
        plt.subplot(2, 1, 1)
        plt.plot(data)
        plt.title(f"{title} - Original Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        
        # Plot wavelet transform
        plt.subplot(2, 1, 2)
        plt.imshow(np.abs(cwt_coef), aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude')
        
        # Plot detected peaks
        if peaks:
            peak_x = [p[1] for p in peaks]
            peak_y = [p[0] for p in peaks]
            plt.plot(peak_x, peak_y, 'k.', markersize=1)
            
        plt.title("Wavelet Transform and Detected Spikes")
        plt.xlabel("Time")
        plt.ylabel("Scale")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def analyze_file(self, filepath: Path) -> dict:
        """Analyze a single file and return results"""
        # Load and preprocess data
        raw_data = self.load_mat_data(filepath)
        if raw_data is None:
            return None
            
        processed_data = self.preprocess_signal(raw_data)
        if processed_data is None:
            return None
            
        # Compute wavelet transform
        cwt_coef, freqs = self.compute_wavelet_transform(processed_data)
        
        # Detect spikes
        peaks = self.detect_spikes(cwt_coef)
        
        # Analyze patterns
        pattern_analysis = self.analyze_temporal_patterns(peaks)
        
        # Create output directory for plots
        plot_dir = self.data_dir / 'wavelet_analysis'
        plot_dir.mkdir(exist_ok=True)
        
        # Plot results
        plot_path = plot_dir / f"{filepath.stem}_wavelet.png"
        self.plot_wavelet_analysis(processed_data, cwt_coef, freqs, peaks,
                                 title=filepath.name, save_path=plot_path)
        
        return {
            'filename': filepath.name,
            'analysis': pattern_analysis
        }

    def batch_analyze(self, species_filter: str = None, max_files: int = 5) -> dict:
        """Analyze multiple files and compile results"""
        results = {}
        
        # Get list of files
        files = list(self.data_dir.glob('*.mat'))
        if species_filter:
            files = [f for f in files if f.name.startswith(species_filter)]
            
        # Limit number of files for testing
        files = files[:max_files]
            
        for file in files:
            print(f"Analyzing {file.name}...")
            result = self.analyze_file(file)
            if result:
                results[file.name] = result
                
        return results

def main():
    analyzer = WaveletAnalyzer()
    
    # Analyze samples from each species
    species = ['Pv', 'Pp', 'Pi', 'Ag']
    all_results = {}
    
    for sp in species:
        print(f"\nAnalyzing {sp} samples...")
        results = analyzer.batch_analyze(species_filter=sp)
        all_results.update(results)
    
    # Print summary
    print("\nAnalysis Summary:")
    for filename, result in all_results.items():
        if 'analysis' in result:
            analysis = result['analysis']
            print(f"\n{filename}:")
            print(f"Total spikes: {analysis['total_spikes']}")
            print(f"Mean interval: {analysis['mean_interval']:.2f}")
            print(f"Std interval: {analysis['std_interval']:.2f}")
            if analysis['repeating_patterns']:
                print("Found repeating patterns!")

if __name__ == "__main__":
    main() 