import numpy as np
from scipy import signal, spatial, stats
import pywt
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sqrt_wavelet_adapter import sqrt_cwt  # NEW IMPORT

class NetworkDynamicsAnalyzer:
    def __init__(self, data_dir: str = "/home/kronos/AVALON/fungal_networks"):
        self.data_dir = Path(data_dir)
        
    def load_mat_data(self, filepath: Path) -> dict:
        """Load coordinate and time series data from .mat file"""
        try:
            data = loadmat(filepath)
            if 'coordinates' in data:
                return {
                    'coordinates': data['coordinates'],
                    'time_points': data.get('time_points', None),
                    'metadata': self._parse_filename(filepath.name)
                }
            return None
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None
            
    def _parse_filename(self, filename: str) -> dict:
        """Parse metadata from filename"""
        parts = filename.replace('.mat', '').split('_')
        return {
            'species': parts[0],
            'scale': parts[1],
            'treatment': parts[2],
            'condition': parts[3],
            'state': parts[4],
            'time': parts[5],
            'replicate': parts[6]
        }

    def analyze_network_geometry(self, coordinates: np.ndarray) -> dict:
        """Analyze geometric properties of the network"""
        if coordinates is None or len(coordinates) < 2:
            return None
            
        # Calculate basic geometric properties
        centroid = np.mean(coordinates, axis=0)
        distances = spatial.distance.cdist([centroid], coordinates)[0]
        
        # Calculate network metrics
        metrics = {
            'area': spatial.ConvexHull(coordinates).area,
            'perimeter': spatial.ConvexHull(coordinates).area,
            'radius_mean': np.mean(distances),
            'radius_std': np.std(distances),
            'density': len(coordinates) / spatial.ConvexHull(coordinates).area,
            'branching_points': self._estimate_branching_points(coordinates)
        }
        
        # Calculate network growth metrics if multiple time points
        if coordinates.shape[0] > 1:
            growth_metrics = self._analyze_growth_dynamics(coordinates)
            metrics.update(growth_metrics)
            
        return metrics
        
    def _estimate_branching_points(self, coordinates: np.ndarray, 
                                 radius: float = 5.0) -> int:
        """Estimate number of branching points using density-based clustering"""
        # Calculate pairwise distances
        distances = pdist(coordinates)
        distance_matrix = squareform(distances)
        
        # Find points with multiple neighbors within radius
        branching_points = 0
        for point_distances in distance_matrix:
            neighbors = np.sum(point_distances < radius)
            if neighbors >= 3:  # Points with 3+ neighbors are potential branches
                branching_points += 1
                
        return branching_points
        
    def _analyze_growth_dynamics(self, coordinates: np.ndarray) -> dict:
        """Analyze network growth patterns"""
        # Calculate growth rates between consecutive points
        diffs = np.diff(coordinates, axis=0)
        velocities = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Analyze growth patterns
        growth_metrics = {
            'growth_rate_mean': np.mean(velocities),
            'growth_rate_std': np.std(velocities),
            'growth_direction': self._calculate_growth_direction(diffs),
            'growth_uniformity': self._calculate_growth_uniformity(velocities)
        }
        
        return growth_metrics
        
    def _calculate_growth_direction(self, diffs: np.ndarray) -> float:
        """Calculate predominant growth direction consistency"""
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        return stats.circmean(angles)
        
    def _calculate_growth_uniformity(self, velocities: np.ndarray) -> float:
        """Calculate how uniform the growth is"""
        return 1 - (np.std(velocities) / np.mean(velocities))

    def analyze_electrical_activity(self, coordinates: np.ndarray,
                                  time_points: np.ndarray = None) -> dict:
        """Analyze electrical activity patterns"""
        if coordinates is None or len(coordinates) < 2:
            return None
            
        # Extract electrical signal from coordinate changes
        if time_points is None:
            time_points = np.arange(len(coordinates))
            
        # Calculate various electrical activity metrics
        signal = self._extract_electrical_signal(coordinates)
        
        # Compute wavelet transform
        wavelet = 'cmor1.5-1.0'
        scales = np.arange(1, min(128, len(signal)//2))
        coef, freqs = sqrt_cwt(signal, sampling_rate=1.0, taus=scales)
        
        # Detect spikes and patterns
        peaks = self._detect_spikes(coef)
        patterns = self._analyze_temporal_patterns(peaks)
        
        # Calculate power in different frequency bands
        power_bands = self._calculate_power_bands(coef, freqs)
        
        metrics = {
            'spike_count': len(peaks),
            'mean_interval': patterns.get('mean_interval', 0),
            'std_interval': patterns.get('std_interval', 0),
            'power_bands': power_bands,
            'patterns': patterns.get('repeating_patterns', {})
        }
        
        return metrics, coef, freqs
        
    def _extract_electrical_signal(self, coordinates: np.ndarray) -> np.ndarray:
        """Extract electrical activity signal from coordinate changes"""
        # Calculate point-wise changes
        diffs = np.diff(coordinates, axis=0)
        magnitudes = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Normalize and detrend
        signal = magnitudes - np.mean(magnitudes)
        return signal
        
    def _detect_spikes(self, cwt_coef: np.ndarray, 
                      threshold: float = 2.0) -> list:
        """Detect spike-like patterns in wavelet coefficients"""
        std_dev = np.std(cwt_coef)
        threshold_val = std_dev * threshold
        
        peaks = []
        for scale_idx in range(cwt_coef.shape[0]):
            scale_peaks = signal.find_peaks(np.abs(cwt_coef[scale_idx]), 
                                          height=threshold_val,
                                          distance=20)[0]
            peaks.extend([(scale_idx, peak) for peak in scale_peaks])
            
        return peaks
        
    def _analyze_temporal_patterns(self, peaks: list) -> dict:
        """Analyze temporal patterns in detected spikes"""
        if not peaks:
            return {}
            
        peak_times = sorted(list(set([p[1] for p in peaks])))
        intervals = np.diff(peak_times)
        
        patterns = {}
        for length in range(3, min(len(intervals) + 1, 10)):
            for i in range(len(intervals) - length + 1):
                pattern = tuple(intervals[i:i+length])
                patterns[pattern] = patterns.get(pattern, 0) + 1
                
        return {
            'mean_interval': np.mean(intervals) if len(intervals) > 0 else 0,
            'std_interval': np.std(intervals) if len(intervals) > 0 else 0,
            'repeating_patterns': {k: v for k, v in patterns.items() if v > 1}
        }
        
    def _calculate_power_bands(self, cwt_coef: np.ndarray, 
                             freqs: np.ndarray) -> dict:
        """Calculate power in different frequency bands"""
        power = np.abs(cwt_coef)**2
        total_power = np.sum(power)
        
        # Define frequency bands (adjust based on your data)
        bands = {
            'low': (0.01, 0.1),
            'medium': (0.1, 1),
            'high': (1, 10)
        }
        
        power_bands = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            band_power = np.sum(power[mask]) / total_power
            power_bands[band_name] = band_power
            
        return power_bands

    def analyze_correlations(self, geometry_metrics: dict,
                           electrical_metrics: dict) -> dict:
        """Analyze correlations between geometric and electrical patterns"""
        if not geometry_metrics or not electrical_metrics:
            return None
            
        correlations = {}
        
        # Compare growth rate with spike frequency
        if 'growth_rate_mean' in geometry_metrics and 'spike_count' in electrical_metrics:
            correlations['growth_spike_correlation'] = {
                'growth_rate': geometry_metrics['growth_rate_mean'],
                'spike_frequency': electrical_metrics['spike_count']
            }
            
        # Compare network density with electrical activity power
        if 'density' in geometry_metrics and 'power_bands' in electrical_metrics:
            correlations['density_power_correlation'] = {
                'density': geometry_metrics['density'],
                'total_power': sum(electrical_metrics['power_bands'].values())
            }
            
        # Compare branching with spike patterns
        if 'branching_points' in geometry_metrics and 'patterns' in electrical_metrics:
            correlations['branching_pattern_correlation'] = {
                'branching_points': geometry_metrics['branching_points'],
                'pattern_complexity': len(electrical_metrics['patterns'])
            }
            
        return correlations

    def plot_combined_analysis(self, coordinates: np.ndarray,
                             electrical_data: tuple,
                             geometry_metrics: dict,
                             save_path: Path = None) -> None:
        """Plot combined geometric and electrical analysis"""
        coef, freqs = electrical_data
        
        fig = plt.figure(figsize=(15, 12))
        
        # Plot network geometry
        ax1 = fig.add_subplot(221)
        ax1.scatter(coordinates[:, 0], coordinates[:, 1], s=1, alpha=0.5)
        ax1.set_title('Network Geometry')
        ax1.set_aspect('equal')
        
        # Plot electrical signal
        ax2 = fig.add_subplot(222)
        signal = self._extract_electrical_signal(coordinates)
        ax2.plot(signal)
        ax2.set_title('Electrical Activity')
        
        # Plot wavelet transform
        ax3 = fig.add_subplot(223)
        im = ax3.imshow(np.abs(coef), aspect='auto', cmap='jet')
        ax3.set_title('Wavelet Transform')
        plt.colorbar(im, ax=ax3)
        
        # Plot metrics summary
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        summary = [
            "Geometry Metrics:",
            f"Area: {geometry_metrics['area']:.2f}",
            f"Density: {geometry_metrics['density']:.2f}",
            f"Branching Points: {geometry_metrics['branching_points']}",
            "\nElectrical Metrics:",
            f"Spike Count: {len(self._detect_spikes(coef))}",
            "Power Bands:",
        ]
        ax4.text(0.1, 0.9, '\n'.join(summary),
                transform=ax4.transAxes, fontsize=10,
                verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def analyze_file(self, filepath: Path) -> dict:
        """Analyze both geometric and electrical patterns in a single file"""
        # Load data
        data = self.load_mat_data(filepath)
        if data is None:
            return None
            
        # Analyze geometry
        geometry_metrics = self.analyze_network_geometry(data['coordinates'])
        
        # Analyze electrical activity
        electrical_results = self.analyze_electrical_activity(
            data['coordinates'],
            data['time_points']
        )
        if electrical_results:
            electrical_metrics, coef, freqs = electrical_results
        else:
            return None
            
        # Analyze correlations
        correlations = self.analyze_correlations(geometry_metrics, electrical_metrics)
        
        # Create output directory
        output_dir = self.data_dir / 'network_analysis'
        output_dir.mkdir(exist_ok=True)
        
        # Plot results
        plot_path = output_dir / f"{filepath.stem}_analysis.png"
        self.plot_combined_analysis(
            data['coordinates'],
            (coef, freqs),
            geometry_metrics,
            save_path=plot_path
        )
        
        return {
            'filename': filepath.name,
            'metadata': data['metadata'],
            'geometry': geometry_metrics,
            'electrical': electrical_metrics,
            'correlations': correlations
        }

    def batch_analyze(self, species_filter: str = None, max_files: int = 5) -> dict:
        """Analyze multiple files and compile results"""
        results = {}
        
        files = list(self.data_dir.glob('*.mat'))
        if species_filter:
            files = [f for f in files if f.name.startswith(species_filter)]
        files = files[:max_files]
        
        for file in files:
            print(f"Analyzing {file.name}...")
            result = self.analyze_file(file)
            if result:
                results[file.name] = result
                
        return results

    def generate_summary_report(self, results: dict) -> None:
        """Generate a summary report of the analysis"""
        report = ["=== Network Dynamics Analysis Report ===\n"]
        
        # Group by species
        species_data = {}
        for filename, data in results.items():
            species = data['metadata']['species']
            if species not in species_data:
                species_data[species] = []
            species_data[species].append(data)
            
        for species, data_list in species_data.items():
            report.extend([
                f"\n=== {species} Analysis ===",
                f"Number of samples: {len(data_list)}"
            ])
            
            # Aggregate metrics
            geo_metrics = {
                'area': [],
                'density': [],
                'branching_points': []
            }
            elec_metrics = {
                'spike_count': [],
                'mean_interval': []
            }
            
            for data in data_list:
                for key in geo_metrics:
                    if key in data['geometry']:
                        geo_metrics[key].append(data['geometry'][key])
                for key in elec_metrics:
                    if key in data['electrical']:
                        elec_metrics[key].append(data['electrical'][key])
                        
            # Report averages
            report.extend([
                "\nGeometric Patterns:",
                f"Mean Area: {np.mean(geo_metrics['area']):.2f} ± {np.std(geo_metrics['area']):.2f}",
                f"Mean Density: {np.mean(geo_metrics['density']):.2f} ± {np.std(geo_metrics['density']):.2f}",
                f"Mean Branching Points: {np.mean(geo_metrics['branching_points']):.1f} ± {np.std(geo_metrics['branching_points']):.1f}",
                "\nElectrical Patterns:",
                f"Mean Spike Count: {np.mean(elec_metrics['spike_count']):.1f} ± {np.std(elec_metrics['spike_count']):.1f}",
                f"Mean Interval: {np.mean(elec_metrics['mean_interval']):.2f} ± {np.std(elec_metrics['mean_interval']):.2f}"
            ])
            
        # Save report
        report_path = self.data_dir / 'network_analysis' / 'analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

def main():
    analyzer = NetworkDynamicsAnalyzer()
    
    # Analyze samples from each species
    species = ['Pv', 'Pp', 'Pi', 'Ag']
    all_results = {}
    
    for sp in species:
        print(f"\nAnalyzing {sp} samples...")
        results = analyzer.batch_analyze(species_filter=sp)
        all_results.update(results)
    
    # Generate summary report
    analyzer.generate_summary_report(all_results)
    print("\nAnalysis complete. Check the network_analysis directory for results.")

if __name__ == "__main__":
    main() 