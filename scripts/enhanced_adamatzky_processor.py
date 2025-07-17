#!/usr/bin/env python3
"""
Enhanced Adamatzky Wave Transform Processor
Comprehensive analysis of fungal electrical activity using Adamatzky's validated parameters

Based on Adamatzky 2023: "Growing colonies of the split-gill fungus Schizophyllum commune 
show action potential-like spikes of extracellular electrical potential"

Key Features:
- Three temporal scales: very slow (hour), slow (10 min), very fast (half-minute)
- FitzHugh-Nagumo model integration
- Comprehensive validation against biological parameters
- Advanced visualization and documentation
- Centralized configuration management (no forced parameters)
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from comprehensive_wave_transform_validation import ComprehensiveWaveTransformValidator

# Import centralized configuration
sys.path.append(str(Path(__file__).parent.parent / "config"))
from analysis_config import config

warnings.filterwarnings('ignore')

class EnhancedAdamatzkyProcessor:
    """
    Enhanced processor for fungal electrical activity analysis using Adamatzky's validated parameters
    
    Implements the three families of oscillatory patterns:
    1. Very slow activity (hour scale)
    2. Slow activity (10-minute scale) 
    3. Very fast activity (half-minute scale)
    
    Uses centralized configuration to eliminate forced parameters
    """
    
    def __init__(self, output_dir: str = "../results"):
        self.validator = ComprehensiveWaveTransformValidator()
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get configuration
        self.config = config
        self.adamatzky_params = self.config.get_adamatzky_params()
        self.output_dirs = self.config.get_output_dirs()
        
        # Create output directories
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # FitzHugh-Nagumo model parameters from config
        self.fhn_params = self.config.config['fitzhugh_nagumo_parameters']
    
    def load_and_preprocess_csv(self, csv_file: str) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Load and preprocess CSV data with Adamatzky-corrected preprocessing
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            Tuple of (processed_data, metadata)
        """
        try:
            df = pd.read_csv(csv_file, header=None)
            # Only keep numeric columns
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(axis=1, how='all')
            
            # Find voltage column (highest variance)
            voltage_data = None
            max_variance = 0
            best_column = 0
            
            for col in range(min(4, len(df.columns))):
                col_data = df.iloc[:, col].values
                if not np.issubdtype(col_data.dtype, np.number):
                    continue
                variance = np.var(col_data)
                if variance > max_variance:
                    max_variance = variance
                    voltage_data = col_data
                    best_column = col
            
            if voltage_data is None:
                voltage_data = df.select_dtypes(include=[np.number]).iloc[:, 0].values
                best_column = 0
            
            # Apply dynamic time compression based on data length
            original_samples = len(voltage_data)
            compression_factor = self.config.get_compression_factor(original_samples)
            compressed_data = voltage_data[::compression_factor]
            
            metadata = {
                'original_samples': original_samples,
                'compressed_samples': len(compressed_data),
                'compression_factor': compression_factor,
                'best_column': best_column,
                'variance': max_variance,
                'mean_amplitude': np.mean(np.abs(voltage_data)),
                'max_amplitude': np.max(np.abs(voltage_data))
            }
            
            return compressed_data, metadata
            
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")
            return None, {}
    
    def apply_adamatzky_wave_transform(self, signal_data: np.ndarray) -> Dict:
        """
        Apply Adamatzky-corrected wave transform W(k,τ) with biological parameters
        Args:
            signal_data: Preprocessed signal data
        Returns:
            Dictionary containing wave transform features
        """
        n = len(signal_data)
        
        # Get wave transform parameters from configuration
        wt_params = self.config.get_wave_transform_params()
        magnitude_threshold = wt_params['magnitude_threshold']
        
        # Generate scales based on configuration
        tau_params = wt_params['tau_values']
        scales = np.concatenate([
            np.linspace(tau_params['very_fast_range'][0], tau_params['very_fast_range'][1], 
                       tau_params['steps_per_range'][0]),      # Very fast (half-minute)
            np.linspace(tau_params['slow_range'][0], tau_params['slow_range'][1], 
                       tau_params['steps_per_range'][1]),      # Slow (10-minute)
            np.linspace(tau_params['very_slow_range'][0], tau_params['very_slow_range'][1], 
                       tau_params['steps_per_range'][2])       # Very slow (hour)
        ])
        
        shifts = np.linspace(0, 86400, 20)  # Up to 1 day
        features = []
        
        for scale in scales:
            for shift in shifts:
                transformed = np.zeros(n, dtype=complex)
                # Use dynamic compression factor from metadata
                compression_factor = getattr(self, 'current_compression_factor', 360)
                compressed_t = np.arange(n) * compression_factor
                
                for i in range(n):
                    t = compressed_t[i]
                    if t + shift > 0:
                        wave_function = np.sqrt(t + shift) * scale
                        frequency_component = np.exp(-1j * scale * np.sqrt(t))
                        wave_value = wave_function * frequency_component
                        transformed[i] = signal_data[i] * wave_value
                
                magnitude = np.abs(np.sum(transformed))
                phase = np.angle(np.sum(transformed))
                
                # Only add features that are actually detected (magnitude > threshold)
                if magnitude > magnitude_threshold:
                    temporal_scale = self._classify_temporal_scale(scale)
                    features.append({
                        'scale': float(scale),
                        'shift': float(shift),
                        'magnitude': float(magnitude),
                        'phase': float(phase),
                        'frequency': float(scale / (2 * np.pi)),
                        'temporal_scale': temporal_scale
                    })
        
        return {
            'all_features': features,
            'n_features': len(features),
            'max_magnitude': max([f['magnitude'] for f in features]) if features else 0,
            'avg_magnitude': np.mean([f['magnitude'] for f in features]) if features else 0,
            'scale_distribution': [f['scale'] for f in features],
            'shift_distribution': [f['shift'] for f in features],
            'temporal_scale_distribution': [f['temporal_scale'] for f in features]
        }
    
    def _classify_temporal_scale(self, scale: float) -> str:
        """Classify scale according to Adamatzky's three families"""
        if scale <= 300:
            return 'very_fast'
        elif scale <= 3600:
            return 'slow'
        else:
            return 'very_slow'
    
    def create_comprehensive_visualizations(self, wave_features: Dict, 
                                         original_signal: np.ndarray, 
                                         filename: str) -> Dict:
        """
        Create comprehensive visualizations for Adamatzky analysis
        
        Args:
            wave_features: Wave transform results
            original_signal: Original signal data
            filename: Input filename for labeling
            
        Returns:
            Dictionary of saved plot paths
        """
        if not wave_features['all_features']:
            return {}
        
        plot_paths = {}
        
        # 1. Temporal Scale Distribution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Adamatzky Wave Transform Analysis: {filename}', fontsize=16)
        
        # Temporal scale distribution
        temporal_scales = wave_features['temporal_scale_distribution']
        scale_counts = pd.Series(temporal_scales).value_counts()
        
        axes[0, 0].pie(scale_counts.values, labels=scale_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Temporal Scale Distribution')
        
        # Magnitude distribution by scale
        magnitudes = [f['magnitude'] for f in wave_features['all_features']]
        scales = [f['scale'] for f in wave_features['all_features']]
        
        axes[0, 1].scatter(scales, magnitudes, alpha=0.6)
        axes[0, 1].set_xlabel('Scale (seconds)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].set_title('Scale vs Magnitude')
        axes[0, 1].set_xscale('log')
        
        # Original signal
        axes[0, 2].plot(original_signal[:1000])  # First 1000 samples
        axes[0, 2].set_title('Original Signal (First 1000 samples)')
        axes[0, 2].set_xlabel('Sample')
        axes[0, 2].set_ylabel('Amplitude')
        
        # Magnitude distribution
        axes[1, 0].hist(magnitudes, bins=30, alpha=0.7, color='blue')
        axes[1, 0].set_title('Feature Magnitude Distribution')
        axes[1, 0].set_xlabel('Magnitude')
        axes[1, 0].set_ylabel('Frequency')
        
        # Scale distribution
        axes[1, 1].hist(scales, bins=30, alpha=0.7, color='green')
        axes[1, 1].set_title('Scale Distribution')
        axes[1, 1].set_xlabel('Scale (seconds)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_xscale('log')
        
        # Phase distribution
        phases = [f['phase'] for f in wave_features['all_features']]
        axes[1, 2].hist(phases, bins=30, alpha=0.7, color='red')
        axes[1, 2].set_title('Phase Distribution')
        axes[1, 2].set_xlabel('Phase (radians)')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save comprehensive plot
        plot_filename = f"adamatzky_analysis_{filename.replace('.csv', '')}_{self.timestamp}.png"
        plot_path = self.dirs['visualizations'] / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['comprehensive'] = str(plot_path)
        
        # 2. Create heatmap visualization
        heatmap_path = self._create_heatmap_visualization(wave_features, filename)
        if heatmap_path:
            plot_paths['heatmap'] = heatmap_path
        
        return plot_paths
    
    def _create_heatmap_visualization(self, wave_features: Dict, filename: str) -> Optional[str]:
        """Create heatmap visualization of scale vs shift"""
        if not wave_features['all_features']:
            return None
        
        scales = [f['scale'] for f in wave_features['all_features']]
        shifts = [f['shift'] for f in wave_features['all_features']]
        magnitudes = [f['magnitude'] for f in wave_features['all_features']]
        
        # Create heatmap data
        unique_scales = np.unique(scales)
        unique_shifts = np.unique(shifts)
        heatmap_data = np.zeros((len(unique_shifts), len(unique_scales)))
        
        for i, shift in enumerate(unique_shifts):
            for j, scale in enumerate(unique_scales):
                mask = (np.array(scales) == scale) & (np.array(shifts) == shift)
                if np.any(mask):
                    heatmap_data[i, j] = np.mean(np.array(magnitudes)[mask])
        
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.title(f'Scale vs Shift Heatmap: {filename}')
        plt.xlabel('Scale (seconds)')
        plt.ylabel('Shift (seconds)')
        
        # Add temporal scale annotations
        scale_positions = []
        scale_labels = []
        for i, scale in enumerate(unique_scales):
            if scale <= 300:
                scale_labels.append('Very Fast')
            elif scale <= 3600:
                scale_labels.append('Slow')
            else:
                scale_labels.append('Very Slow')
            scale_positions.append(i)
        
        plt.xticks(scale_positions[::3], [scale_labels[i] for i in scale_positions[::3]], rotation=45)
        
        # Save heatmap
        heatmap_filename = f"heatmap_{filename.replace('.csv', '')}_{self.timestamp}.png"
        heatmap_path = self.dirs['visualizations'] / heatmap_filename
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(heatmap_path)
    
    def process_single_file(self, csv_file: str) -> Dict:
        """
        Process a single CSV file with comprehensive Adamatzky analysis
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            Dictionary containing analysis results
        """
        print(f"\n🔬 Processing: {csv_file}")
        print("=" * 60)
        
        # Load and preprocess
        compressed_data, metadata = self.load_and_preprocess_csv(csv_file)
        if compressed_data is None:
            return {}
        
        # Store compression factor for wave transform
        self.current_compression_factor = metadata['compression_factor']
        
        print(f"📊 Data Summary:")
        print(f"   Original samples: {metadata['original_samples']}")
        print(f"   Compressed samples: {metadata['compressed_samples']}")
        print(f"   Compression factor: {metadata['compression_factor']}x")
        print(f"   Mean amplitude: {metadata['mean_amplitude']:.3f} mV")
        print(f"   Max amplitude: {metadata['max_amplitude']:.3f} mV")
        
        # Apply Adamatzky wave transform
        wave_features = self.apply_adamatzky_wave_transform(compressed_data)
        
        print(f"\n🌊 Wave Transform Results:")
        print(f"   Total features: {wave_features['n_features']}")
        print(f"   Max magnitude: {wave_features['max_magnitude']:.3f}")
        print(f"   Avg magnitude: {wave_features['avg_magnitude']:.3f}")
        
        # Temporal scale distribution
        temporal_dist = pd.Series(wave_features['temporal_scale_distribution']).value_counts()
        print(f"\n⏰ Temporal Scale Distribution:")
        for scale, count in temporal_dist.items():
            percentage = (count / len(wave_features['temporal_scale_distribution'])) * 100
            print(f"   {scale}: {count} ({percentage:.1f}%)")
        
        # Create visualizations
        plot_paths = self.create_comprehensive_visualizations(wave_features, compressed_data, 
                                                            Path(csv_file).name)
        
        # Comprehensive validation
        validation_results = self.validator.comprehensive_validation(
            wave_features, compressed_data, Path(csv_file).name
        )
        
        # Compile results
        results = {
            'filename': Path(csv_file).name,
            'timestamp': self.timestamp,
            'metadata': metadata,
            'wave_features': wave_features,
            'validation_results': validation_results,
            'plot_paths': plot_paths
        }
        
        # Save results
        results_filename = f"analysis_{Path(csv_file).stem}_{self.timestamp}.json"
        results_path = self.dirs['analysis'] / results_filename
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._prepare_for_json(results)
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")
        
        return results
    
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
    
    def batch_process_all_files(self, data_dir: str = "../../data") -> Dict:
        """
        Process all CSV files in the data directory
        
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
        
        print(f"🚀 Starting batch processing of {len(csv_files)} files...")
        print("=" * 60)
        
        all_results = {}
        for csv_file in csv_files:
            try:
                results = self.process_single_file(str(csv_file))
                if results:
                    all_results[csv_file.name] = results
            except Exception as e:
                print(f"❌ Error processing {csv_file}: {e}")
        
        # Create summary report
        summary = self.create_summary_report(all_results)
        
        # Save batch summary
        summary_filename = f"batch_summary_{self.timestamp}.json"
        summary_path = self.dirs['reports'] / summary_filename
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Batch processing complete!")
        print(f"   Processed files: {len(all_results)}")
        print(f"   Summary saved to: {summary_path}")
        
        return all_results
    
    def create_summary_report(self, all_results: Dict) -> Dict:
        """Create comprehensive summary report"""
        summary = {
            'timestamp': self.timestamp,
            'total_files': len(all_results),
            'files_processed': list(all_results.keys()),
            'temporal_scale_summary': {},
            'validation_summary': {},
            'overall_statistics': {}
        }
        
        # Aggregate temporal scale distributions
        all_temporal_scales = []
        all_magnitudes = []
        validation_scores = []
        
        for filename, results in all_results.items():
            temporal_scales = results['wave_features']['temporal_scale_distribution']
            magnitudes = [f['magnitude'] for f in results['wave_features']['all_features']]
            
            all_temporal_scales.extend(temporal_scales)
            all_magnitudes.extend(magnitudes)
            
            # Collect validation scores
            if 'validation_results' in results:
                validation_scores.append(results['validation_results'])
        
        # Temporal scale summary
        if all_temporal_scales:
            scale_dist = pd.Series(all_temporal_scales).value_counts()
            summary['temporal_scale_summary'] = {
                scale: {'count': int(count), 'percentage': float((count/len(all_temporal_scales))*100)}
                for scale, count in scale_dist.items()
            }
        
        # Overall statistics
        if all_magnitudes:
            summary['overall_statistics'] = {
                'total_features': len(all_magnitudes),
                'mean_magnitude': float(np.mean(all_magnitudes)),
                'max_magnitude': float(np.max(all_magnitudes)),
                'std_magnitude': float(np.std(all_magnitudes))
            }
        
        return summary

def main():
    """Main execution function (modified to process only two specific files)"""
    processor = EnhancedAdamatzkyProcessor()
    files_to_process = [
        "Ch1-2_moisture_added.csv",
        "Spray_in_bag_crop.csv"
    ]
    all_results = {}
    for csv_file in files_to_process:
        if Path(csv_file).exists():
            results = processor.process_single_file(csv_file)
            if results:
                all_results[csv_file] = results
        else:
            print(f"❌ File not found: {csv_file}")
    if all_results:
        summary = processor.create_summary_report(all_results)
        summary_filename = f"summary_{processor.timestamp}.json"
        summary_path = processor.dirs['reports'] / summary_filename
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n📊 Summary saved to: {summary_path}")
    print(f"\n✅ Enhanced Adamatzky analysis complete!")
    print(f"   Results saved in: {processor.output_dir}")
    print(f"   Visualizations: {processor.dirs['visualizations']}")
    print(f"   Reports: {processor.dirs['reports']}")

if __name__ == "__main__":
    main() 