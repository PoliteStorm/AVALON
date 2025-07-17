#!/usr/bin/env python3
"""
Comprehensive Wave Transform Analysis with Adamatzky Parameters
Processes Adamatzky-compliant CSV files through wave transform W(k,τ) with validation and visualization

Wave Transform: W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt

Uses centralized configuration to eliminate forced parameters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.optimize import curve_fit
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import warnings
from typing import Dict, List, Tuple, Optional

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from enhanced_adamatzky_processor import EnhancedAdamatzkyProcessor
from comprehensive_wave_transform_validation import ComprehensiveWaveTransformValidator

# Import centralized configuration
sys.path.append(str(Path(__file__).parent.parent / "config"))
from analysis_config import config

warnings.filterwarnings('ignore')

class ComprehensiveWaveTransformAnalyzer:
    """
    Comprehensive analyzer for wave transform W(k,τ) with Adamatzky parameters
    Uses centralized configuration to eliminate forced parameters
    """
    
    def __init__(self):
        self.processor = EnhancedAdamatzkyProcessor()
        self.validator = ComprehensiveWaveTransformValidator()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get configuration
        self.config = config
        self.adamatzky_params = self.config.get_adamatzky_params()
        self.output_dirs = self.config.get_output_dirs()
        
        # Create output directories
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def apply_wave_transform_wkt(self, signal_data: np.ndarray, filename: str) -> Dict:
        """
        Apply wave transform W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
        Args:
            signal_data: Voltage signal V(t)
            filename: Name of the file for labeling
        Returns:
            Dictionary containing wave transform results
        """
        print(f"\n🌊 APPLYING WAVE TRANSFORM W(k,τ)")
        print("=" * 60)
        print(f"W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt")
        
        n_samples = len(signal_data)
        
        # Get wave transform parameters from configuration
        wt_params = self.config.get_wave_transform_params()
        magnitude_threshold = wt_params['magnitude_threshold']
        
        # Define k values from configuration
        k_values = np.linspace(wt_params['k_values']['min'], 
                              wt_params['k_values']['max'], 
                              wt_params['k_values']['steps'])
        
        # Define tau values from configuration
        tau_params = wt_params['tau_values']
        tau_values = np.concatenate([
            np.linspace(tau_params['very_fast_range'][0], tau_params['very_fast_range'][1], 
                       tau_params['steps_per_range'][0]),      # Very fast (half-minute)
            np.linspace(tau_params['slow_range'][0], tau_params['slow_range'][1], 
                       tau_params['steps_per_range'][1]),      # Slow (10-minute)
            np.linspace(tau_params['very_slow_range'][0], tau_params['very_slow_range'][1], 
                       tau_params['steps_per_range'][2])       # Very slow (hour)
        ])
        
        wave_transform_results = []
        
        for k in k_values:
            for tau in tau_values:
                transformed = np.zeros(n_samples, dtype=complex)
                for i in range(n_samples):
                    t = i / self.adamatzky_params['sampling_rate']  # Time in seconds
                    wave_function = np.sqrt(t / tau) if t > 0 else 0
                    frequency_component = np.exp(-1j * k * np.sqrt(t)) if t > 0 else 0
                    wave_value = wave_function * frequency_component
                    transformed[i] = signal_data[i] * wave_value
                
                magnitude = np.abs(np.sum(transformed))
                phase = np.angle(np.sum(transformed))
                
                # Only add features that are actually detected (magnitude > threshold)
                if magnitude > magnitude_threshold:
                    temporal_scale = self._classify_temporal_scale(tau)
                    wave_transform_results.append({
                        'k': k,
                        'tau': tau,
                        'magnitude': magnitude,
                        'phase': phase,
                        'frequency': k / (2 * np.pi),
                        'temporal_scale': temporal_scale
                    })
        
        return {
            'all_features': wave_transform_results,
            'n_features': len(wave_transform_results),
            'max_magnitude': max([f['magnitude'] for f in wave_transform_results]) if wave_transform_results else 0,
            'avg_magnitude': np.mean([f['magnitude'] for f in wave_transform_results]) if wave_transform_results else 0,
            'scale_distribution': [f['tau'] for f in wave_transform_results],
            'frequency_distribution': [f['frequency'] for f in wave_transform_results],
            'temporal_scale_distribution': [f['temporal_scale'] for f in wave_transform_results]
        }
    
    def _classify_temporal_scale(self, tau: float) -> str:
        """Classify tau according to Adamatzky's temporal scales"""
        if tau <= 300:
            return 'very_fast'
        elif tau <= 3600:
            return 'slow'
        else:
            return 'very_slow'
    
    def create_comprehensive_heatmaps(self, wave_results: Dict, original_signal: np.ndarray) -> Dict:
        """
        Create comprehensive heatmap visualizations for wave transform analysis
        
        Args:
            wave_results: Wave transform results
            original_signal: Original signal data
            
        Returns:
            Dictionary of saved plot paths
        """
        if not wave_results['all_features']:
            return {}
        
        plot_paths = {}
        
        # Extract data for heatmaps
        k_values = [f['k'] for f in wave_results['all_features']]
        tau_values = [f['tau'] for f in wave_results['all_features']]
        magnitudes = [f['magnitude'] for f in wave_results['all_features']]
        phases = [f['phase'] for f in wave_results['all_features']]
        temporal_scales = [f['temporal_scale'] for f in wave_results['all_features']]
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'Comprehensive Wave Transform Analysis: {wave_results["filename"]}', fontsize=16)
        
        # 1. k vs tau heatmap (magnitude)
        unique_k = np.unique(k_values)
        unique_tau = np.unique(tau_values)
        heatmap_data = np.zeros((len(unique_tau), len(unique_k)))
        
        for i, tau in enumerate(unique_tau):
            for j, k in enumerate(unique_k):
                mask = (np.array(k_values) == k) & (np.array(tau_values) == tau)
                if np.any(mask):
                    heatmap_data[i, j] = np.mean(np.array(magnitudes)[mask])
        
        im1 = axes[0, 0].imshow(heatmap_data, aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Wave Transform Magnitude: k vs τ')
        axes[0, 0].set_xlabel('Frequency Parameter k')
        axes[0, 0].set_ylabel('Time Scale τ (seconds)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. k vs tau heatmap (phase)
        phase_heatmap = np.zeros((len(unique_tau), len(unique_k)))
        for i, tau in enumerate(unique_tau):
            for j, k in enumerate(unique_k):
                mask = (np.array(k_values) == k) & (np.array(tau_values) == tau)
                if np.any(mask):
                    phase_heatmap[i, j] = np.mean(np.array(phases)[mask])
        
        im2 = axes[0, 1].imshow(phase_heatmap, aspect='auto', cmap='RdBu_r')
        axes[0, 1].set_title('Wave Transform Phase: k vs τ')
        axes[0, 1].set_xlabel('Frequency Parameter k')
        axes[0, 1].set_ylabel('Time Scale τ (seconds)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # 3. Temporal scale distribution
        scale_counts = pd.Series(temporal_scales).value_counts()
        axes[0, 2].pie(scale_counts.values, labels=scale_counts.index, autopct='%1.1f%%')
        axes[0, 2].set_title('Temporal Scale Distribution')
        
        # 4. Magnitude distribution
        axes[1, 0].hist(magnitudes, bins=30, alpha=0.7, color='blue')
        axes[1, 0].set_title('Feature Magnitude Distribution')
        axes[1, 0].set_xlabel('Magnitude')
        axes[1, 0].set_ylabel('Frequency')
        
        # 5. k distribution
        axes[1, 1].hist(k_values, bins=30, alpha=0.7, color='green')
        axes[1, 1].set_title('Frequency Parameter k Distribution')
        axes[1, 1].set_xlabel('k')
        axes[1, 1].set_ylabel('Frequency')
        
        # 6. τ distribution
        axes[1, 2].hist(tau_values, bins=30, alpha=0.7, color='red')
        axes[1, 2].set_title('Time Scale τ Distribution')
        axes[1, 2].set_xlabel('τ (seconds)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_xscale('log')
        
        # 7. Original signal
        axes[2, 0].plot(original_signal[:1000])
        axes[2, 0].set_title('Original Signal (First 1000 samples)')
        axes[2, 0].set_xlabel('Sample')
        axes[2, 0].set_ylabel('Amplitude (mV)')
        
        # 8. Phase distribution
        axes[2, 1].hist(phases, bins=30, alpha=0.7, color='orange')
        axes[2, 1].set_title('Phase Distribution')
        axes[2, 1].set_xlabel('Phase (radians)')
        axes[2, 1].set_ylabel('Frequency')
        
        # 9. Magnitude vs k scatter
        axes[2, 2].scatter(k_values, magnitudes, alpha=0.6, c=tau_values, cmap='viridis')
        axes[2, 2].set_title('Magnitude vs Frequency Parameter k')
        axes[2, 2].set_xlabel('k')
        axes[2, 2].set_ylabel('Magnitude')
        plt.colorbar(axes[2, 2].collections[0], ax=axes[2, 2], label='τ (seconds)')
        
        plt.tight_layout()
        
        # Save comprehensive plot
        plot_filename = f"comprehensive_wave_transform_{wave_results['filename'].replace('.csv', '')}_{self.timestamp}.png"
        plot_path = self.output_dirs['visualizations'] / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['comprehensive'] = str(plot_path)
        
        # Create detailed heatmap
        detailed_heatmap_path = self._create_detailed_heatmap(wave_results)
        if detailed_heatmap_path:
            plot_paths['detailed_heatmap'] = detailed_heatmap_path
        
        return plot_paths
    
    def _create_detailed_heatmap(self, wave_results: Dict) -> Optional[str]:
        """Create detailed heatmap with temporal scale annotations"""
        
        k_values = [f['k'] for f in wave_results['all_features']]
        tau_values = [f['tau'] for f in wave_results['all_features']]
        magnitudes = [f['magnitude'] for f in wave_results['all_features']]
        
        # Create detailed heatmap
        unique_k = np.unique(k_values)
        unique_tau = np.unique(tau_values)
        heatmap_data = np.zeros((len(unique_tau), len(unique_k)))
        
        for i, tau in enumerate(unique_tau):
            for j, k in enumerate(unique_k):
                mask = (np.array(k_values) == k) & (np.array(tau_values) == tau)
                if np.any(mask):
                    heatmap_data[i, j] = np.mean(np.array(magnitudes)[mask])
        
        plt.figure(figsize=(15, 10))
        
        # Create heatmap with temporal scale annotations
        im = plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.title(f'Detailed Wave Transform Heatmap: {wave_results["filename"]}')
        plt.xlabel('Frequency Parameter k')
        plt.ylabel('Time Scale τ (seconds)')
        
        # Add temporal scale annotations
        tau_positions = []
        tau_labels = []
        for i, tau in enumerate(unique_tau):
            if tau <= 300:
                tau_labels.append('Very Fast')
            elif tau <= 3600:
                tau_labels.append('Slow')
            else:
                tau_labels.append('Very Slow')
            tau_positions.append(i)
        
        plt.yticks(tau_positions[::5], [tau_labels[i] for i in tau_positions[::5]])
        
        # Save detailed heatmap
        heatmap_filename = f"detailed_heatmap_{wave_results['filename'].replace('.csv', '')}_{self.timestamp}.png"
        heatmap_path = self.output_dirs['visualizations'] / heatmap_filename
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(heatmap_path)
    
    def compare_methods(self, wave_results: Dict, original_signal: np.ndarray) -> Dict:
        """
        Compare wave transform results with standard methods
        
        Args:
            wave_results: Wave transform results
            original_signal: Original signal data
            
        Returns:
            Dictionary containing comparison results
        """
        print(f"\n🔍 COMPARISON ANALYSIS")
        print("=" * 60)
        
        comparison_results = {
            'wave_transform': wave_results,
            'standard_methods': {},
            'comparison_metrics': {}
        }
        
        # 1. Standard peak detection
        peaks, _ = signal.find_peaks(np.abs(original_signal), height=np.std(original_signal))
        peak_intervals = np.diff(peaks) / self.adamatzky_params['sampling_rate']
        
        comparison_results['standard_methods']['peak_detection'] = {
            'n_peaks': len(peaks),
            'avg_interval': np.mean(peak_intervals) if len(peak_intervals) > 0 else 0,
            'peak_positions': peaks.tolist()
        }
        
        # 2. FFT analysis
        fft_result = np.fft.fft(original_signal)
        fft_freq = np.fft.fftfreq(len(original_signal), 1/self.adamatzky_params['sampling_rate'])
        
        # Find dominant frequencies
        dominant_freq_idx = np.argsort(np.abs(fft_result))[-10:]  # Top 10 frequencies
        dominant_frequencies = fft_freq[dominant_freq_idx]
        
        comparison_results['standard_methods']['fft_analysis'] = {
            'dominant_frequencies': dominant_frequencies.tolist(),
            'fft_magnitude': np.abs(fft_result).tolist()
        }
        
        # 3. Statistical comparison
        wave_magnitudes = [f['magnitude'] for f in wave_results['all_features']]
        
        comparison_results['comparison_metrics'] = {
            'wave_transform_features': len(wave_results['all_features']),
            'standard_peaks': len(peaks),
            'wave_avg_magnitude': np.mean(wave_magnitudes) if wave_magnitudes else 0,
            'signal_variance': np.var(original_signal),
            'signal_range': np.max(original_signal) - np.min(original_signal)
        }
        
        print(f"📊 Comparison Results:")
        print(f"   Wave transform features: {comparison_results['comparison_metrics']['wave_transform_features']}")
        print(f"   Standard peaks detected: {comparison_results['comparison_metrics']['standard_peaks']}")
        print(f"   Signal variance: {comparison_results['comparison_metrics']['signal_variance']:.3f}")
        
        return comparison_results
    
    def process_single_file(self, csv_file: str) -> Dict:
        """
        Process a single CSV file with comprehensive wave transform analysis
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            Dictionary containing complete analysis results
        """
        print(f"\n🔬 Processing: {Path(csv_file).name}")
        print("=" * 60)
        
        # Load and preprocess data
        compressed_data, metadata = self.processor.load_and_preprocess_csv(csv_file)
        if compressed_data is None:
            return {}
        
        print(f"📊 Data Summary:")
        print(f"   Original samples: {metadata['original_samples']}")
        print(f"   Compressed samples: {metadata['compressed_samples']}")
        print(f"   Mean amplitude: {metadata['mean_amplitude']:.3f} mV")
        print(f"   Max amplitude: {metadata['max_amplitude']:.3f} mV")
        
        # Apply wave transform W(k,τ)
        wave_results = self.apply_wave_transform_wkt(compressed_data, Path(csv_file).name)
        
        # Create comprehensive visualizations
        plot_paths = self.create_comprehensive_heatmaps(wave_results, compressed_data)
        
        # Compare with standard methods
        comparison_results = self.compare_methods(wave_results, compressed_data)
        
        # Comprehensive validation
        validation_results = self.validator.comprehensive_validation(
            wave_results, compressed_data, Path(csv_file).name
        )
        
        # Compile complete results
        complete_results = {
            'filename': Path(csv_file).name,
            'timestamp': self.timestamp,
            'metadata': metadata,
            'wave_transform_results': wave_results,
            'comparison_results': comparison_results,
            'validation_results': validation_results,
            'plot_paths': plot_paths
        }
        
        # Save results
        results_filename = f"wave_transform_analysis_{Path(csv_file).stem}_{self.timestamp}.json"
        results_path = self.output_dirs['results'] / results_filename
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_for_json(complete_results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")
        
        return complete_results
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            # If dtype is complex, convert to list of dicts with magnitude and phase
            if np.issubdtype(obj.dtype, np.complexfloating):
                return [{'magnitude': float(np.abs(x)), 'phase': float(np.angle(x))} for x in obj]
            else:
                return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj
    
    def process_all_files(self, data_dir: str = "../data/processed") -> Dict:
        """
        Process all CSV files in the processed directory
        
        Args:
            data_dir: Directory containing CSV files
            
        Returns:
            Dictionary containing batch results
        """
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {data_dir}")
            return {}
        
        print(f"🚀 Starting comprehensive wave transform analysis of {len(csv_files)} files...")
        print("=" * 60)
        
        all_results = {}
        for csv_file in csv_files:
            try:
                results = self.process_single_file(str(csv_file))
                if results:
                    all_results[csv_file.name] = results
            except Exception as e:
                print(f"❌ Error processing {csv_file}: {e}")
        
        # Create comprehensive summary report
        summary = self.create_comprehensive_summary(all_results)
        
        # Save batch summary
        summary_filename = f"comprehensive_wave_transform_summary_{self.timestamp}.json"
        summary_path = self.output_dirs['reports'] / summary_filename
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Comprehensive wave transform analysis complete!")
        print(f"   Processed files: {len(all_results)}")
        print(f"   Summary saved to: {summary_path}")
        print(f"   Visualizations: {self.output_dirs['visualizations']}")
        
        return all_results
    
    def create_comprehensive_summary(self, all_results: Dict) -> Dict:
        """Create comprehensive summary report"""
        summary = {
            'timestamp': self.timestamp,
            'total_files': len(all_results),
            'files_processed': list(all_results.keys()),
            'wave_transform_summary': {},
            'comparison_summary': {},
            'validation_summary': {},
            'overall_statistics': {}
        }
        
        # Aggregate wave transform statistics
        all_wave_features = []
        all_magnitudes = []
        all_temporal_scales = []
        
        for filename, results in all_results.items():
            wave_results = results['wave_transform_results']
            magnitudes = [f['magnitude'] for f in wave_results['all_features']]
            temporal_scales = wave_results['temporal_scale_distribution']
            
            all_wave_features.extend(wave_results['all_features'])
            all_magnitudes.extend(magnitudes)
            all_temporal_scales.extend(temporal_scales)
        
        # Wave transform summary
        if all_magnitudes:
            summary['wave_transform_summary'] = {
                'total_features': len(all_wave_features),
                'mean_magnitude': float(np.mean(all_magnitudes)),
                'max_magnitude': float(np.max(all_magnitudes)),
                'std_magnitude': float(np.std(all_magnitudes))
            }
        
        # Temporal scale summary
        if all_temporal_scales:
            scale_dist = pd.Series(all_temporal_scales).value_counts()
            summary['temporal_scale_summary'] = {
                scale: {'count': int(count), 'percentage': float((count/len(all_temporal_scales))*100)}
                for scale, count in scale_dist.items()
            }
        
        return summary

def main():
    """Main execution function"""
    
    analyzer = ComprehensiveWaveTransformAnalyzer()
    
    # Process all files in the processed directory
    results = analyzer.process_all_files()
    
    print(f"\n✅ Comprehensive wave transform analysis complete!")
    print(f"   Results saved in: {analyzer.output_dirs['results']}")
    print(f"   Visualizations: {analyzer.output_dirs['visualizations']}")
    print(f"   Reports: {analyzer.output_dirs['reports']}")

if __name__ == "__main__":
    main() 