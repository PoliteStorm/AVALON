#!/usr/bin/env python3
"""
Batch Wave Transform Processor with Comprehensive Validation
Processes all CSV files with Adamatzky-corrected wave transform and validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import json
import os
from datetime import datetime
from pathlib import Path
import sys
import warnings

# Add parent directory to path to import validation module
sys.path.append(str(Path(__file__).parent))
from comprehensive_wave_transform_validation import ComprehensiveWaveTransformValidator

warnings.filterwarnings('ignore')

class BatchWaveTransformProcessor:
    """Batch processor for wave transform analysis with comprehensive validation"""
    
    def __init__(self):
        self.validator = ComprehensiveWaveTransformValidator()
        self.results_dir = Path("../results/analysis")
        self.plots_dir = Path("../results/visualizations")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Adamatzky 2023 validated parameters
        self.adamatzky_params = {
            'temporal_scales': {
                'very_slow': {'min_isi': 3600, 'max_isi': float('inf'), 'description': 'Hour scale (43 min avg)'},
                'slow': {'min_isi': 600, 'max_isi': 3600, 'description': '10-minute scale (8 min avg)'},
                'very_fast': {'min_isi': 30, 'max_isi': 300, 'description': 'Half-minute scale (24s avg)'}
            },
            'spike_characteristics': {
                'very_slow': {'duration': 2573, 'amplitude': 0.16, 'distance': 2656},
                'slow': {'duration': 457, 'amplitude': 0.4, 'distance': 1819},
                'very_fast': {'duration': 24, 'amplitude': 0.36, 'distance': 148}
            },
            'sampling_rate': 1,  # Hz
            'min_spike_amplitude': 0.05,  # mV
            'max_spike_amplitude': 5.0,   # mV
            'time_compression': 3000     # 3000 seconds = 1 day (longer acquisition)
        }
        
        # Wave transform parameters
        self.wave_params = {
            'scale_range': [30, 3600],  # Adamatzky temporal scales (seconds)
            'shift_range': [0, 86400],   # Up to 1 day
            'time_compression': 360,   # 360 seconds = 1 day (less aggressive)
            'sampling_rate': 1,          # Hz
            'min_snr': 2.0
        }
    
    def load_and_preprocess_csv(self, csv_file):
        """Load CSV and apply Adamatzky-corrected preprocessing"""
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
            
            # Apply time compression
            original_samples = len(voltage_data)
            compressed_samples = original_samples // self.wave_params['time_compression']
            
            if compressed_samples < 10:
                compression_factor = max(1, original_samples // 100)
                compressed_data = voltage_data[::compression_factor]
            else:
                compressed_data = voltage_data[::self.wave_params['time_compression']]
            
            return compressed_data, original_samples, len(compressed_data)
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            return None, 0, 0
    
    def apply_wave_transform(self, signal_data):
        """Apply Adamatzky-corrected wave transform W(k,τ)"""
        n = len(signal_data)
        
        # Adamatzky-corrected scale and shift parameters
        # Focus on the three temporal scales identified by Adamatzky
        scales = np.concatenate([
            np.linspace(30, 300, 10),      # Very fast (half-minute)
            np.linspace(600, 3600, 15),    # Slow (10-minute)
            np.linspace(3600, 86400, 10)   # Very slow (hour)
        ])
        
        shifts = np.linspace(self.wave_params['shift_range'][0], 
                           self.wave_params['shift_range'][1], 20)
        
        features = []
        for scale in scales:
            for shift in shifts:
                # Apply wave transform: W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
                transformed = np.zeros(n, dtype=complex)
                compressed_t = np.arange(n) / self.wave_params['time_compression']
                
                for i in range(n):
                    t = compressed_t[i]
                    if t + shift > 0:
                        wave_function = np.sqrt(t + shift) * scale
                        frequency_component = np.exp(-1j * scale * np.sqrt(t))
                        wave_value = wave_function * frequency_component
                        transformed[i] = signal_data[i] * wave_value
                
                # Calculate feature magnitude and phase
                magnitude = np.abs(np.sum(transformed))
                phase = np.angle(np.sum(transformed))
                
                if magnitude > 0:
                    # Classify by Adamatzky's temporal scales
                    temporal_scale = self._classify_temporal_scale(scale)
                    
                    features.append({
                        'scale': scale,
                        'shift': shift,
                        'magnitude': magnitude,
                        'phase': phase,
                        'frequency': scale / (2 * np.pi),
                        'temporal_scale': temporal_scale,
                        'transformed_signal': transformed
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
    
    def create_heatmap_visualization(self, wave_features, filename):
        """Create comprehensive heatmap visualizations"""
        if not wave_features['all_features']:
            return None
        
        # Prepare data for heatmaps
        scales = [f['scale'] for f in wave_features['all_features']]
        shifts = [f['shift'] for f in wave_features['all_features']]
        magnitudes = [f['magnitude'] for f in wave_features['all_features']]
        
        # Create scale vs shift heatmap
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Wave Transform Analysis: {filename}', fontsize=16)
        
        # 1. Scale vs Shift Heatmap
        unique_scales = np.unique(scales)
        unique_shifts = np.unique(shifts)
        heatmap_data = np.zeros((len(unique_shifts), len(unique_scales)))
        
        for i, shift in enumerate(unique_shifts):
            for j, scale in enumerate(unique_scales):
                # Find features with this scale and shift
                mask = (np.array(scales) == scale) & (np.array(shifts) == shift)
                if np.any(mask):
                    heatmap_data[i, j] = np.mean(np.array(magnitudes)[mask])
        
        im1 = axes[0, 0].imshow(heatmap_data, aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Scale vs Shift Heatmap')
        axes[0, 0].set_xlabel('Scale (seconds)')
        axes[0, 0].set_ylabel('Shift (seconds)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. Magnitude Distribution
        axes[0, 1].hist(magnitudes, bins=20, alpha=0.7, color='blue')
        axes[0, 1].set_title('Feature Magnitude Distribution')
        axes[0, 1].set_xlabel('Magnitude')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Scale Distribution
        axes[1, 0].hist(scales, bins=20, alpha=0.7, color='green')
        axes[1, 0].set_title('Scale Distribution')
        axes[1, 0].set_xlabel('Scale (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Shift Distribution
        axes[1, 1].hist(shifts, bins=20, alpha=0.7, color='red')
        axes[1, 1].set_title('Shift Distribution')
        axes[1, 1].set_xlabel('Shift (seconds)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"wave_transform_heatmap_{filename.replace('.csv', '')}.png"
        plot_path = self.plots_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def process_single_file(self, csv_file):
        """Process a single CSV file with comprehensive validation"""
        print(f"\n🔬 Processing: {csv_file}")
        print("=" * 60)
        
        # Load and preprocess
        compressed_data, original_samples, compressed_samples = self.load_and_preprocess_csv(csv_file)
        if compressed_data is None:
            return None
        
        print(f"📊 Data Summary:")
        print(f"   Original samples: {original_samples}")
        print(f"   Compressed samples: {compressed_samples}")
        print(f"   Compression factor: {self.wave_params['time_compression']}x")
        
        # Apply wave transform
        wave_features = self.apply_wave_transform(compressed_data)
        
        print(f"🌊 Wave Transform Results:")
        print(f"   Features detected: {wave_features['n_features']}")
        print(f"   Max magnitude: {wave_features['max_magnitude']:.4f}")
        print(f"   Avg magnitude: {wave_features['avg_magnitude']:.4f}")
        
        # Comprehensive validation
        validation_results = self.validator.comprehensive_validation(
            wave_features, compressed_data, os.path.basename(csv_file)
        )
        
        # Create visualizations
        plot_path = self.create_heatmap_visualization(wave_features, os.path.basename(csv_file))
        
        # Combine results
        results = {
            'filename': os.path.basename(csv_file),
            'processing_info': {
                'original_samples': original_samples,
                'compressed_samples': compressed_samples,
                'compression_factor': self.wave_params['time_compression'],
                'sampling_rate': self.wave_params['sampling_rate']
            },
            'wave_features': wave_features,
            'validation_results': validation_results,
            'plot_path': str(plot_path) if plot_path else None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        json_filename = f"wave_transform_results_{os.path.basename(csv_file).replace('.csv', '')}.json"
        json_path = self.results_dir / json_filename
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {json_path}")
        if plot_path:
            print(f"📊 Plot saved to: {plot_path}")
        
        return results
    
    def batch_process_all_files(self, data_dir="../../data"):
        """Process all CSV files in the data directory"""
        print("🚀 BATCH WAVE TRANSFORM PROCESSING")
        print("=" * 80)
        
        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {data_dir}")
            return []
        
        print(f"📁 Found {len(csv_files)} CSV files to process")
        
        all_results = []
        for csv_file in csv_files:
            try:
                result = self.process_single_file(str(csv_file))
                if result:
                    all_results.append(result)
            except Exception as e:
                print(f"❌ Error processing {csv_file}: {e}")
                continue
        
        # Create summary report
        self.create_summary_report(all_results)
        
        return all_results
    
    def create_summary_report(self, results):
        """Create a summary report of all processed files"""
        if not results:
            return
        
        summary = {
            'total_files': len(results),
            'processing_timestamp': datetime.now().isoformat(),
            'file_summaries': [],
            'overall_statistics': {
                'avg_validation_score': 0.0,
                'files_with_issues': 0,
                'excellent_files': 0,
                'good_files': 0,
                'caution_files': 0,
                'reject_files': 0
            }
        }
        
        validation_scores = []
        for result in results:
            validation = result['validation_results']
            score = validation['overall_score']
            validation_scores.append(score)
            
            file_summary = {
                'filename': result['filename'],
                'validation_score': score,
                'recommendation': validation['recommendation'],
                'n_features': result['wave_features']['n_features'],
                'issues': validation['all_issues']
            }
            summary['file_summaries'].append(file_summary)
            
            # Count by recommendation
            if 'EXCELLENT' in validation['recommendation']:
                summary['overall_statistics']['excellent_files'] += 1
            elif 'GOOD' in validation['recommendation']:
                summary['overall_statistics']['good_files'] += 1
            elif 'CAUTION' in validation['recommendation']:
                summary['overall_statistics']['caution_files'] += 1
            else:
                summary['overall_statistics']['reject_files'] += 1
            
            if validation['all_issues']:
                summary['overall_statistics']['files_with_issues'] += 1
        
        if validation_scores:
            summary['overall_statistics']['avg_validation_score'] = np.mean(validation_scores)
        
        # Save summary
        summary_path = self.results_dir / "batch_processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print summary
        print(f"\n📊 BATCH PROCESSING SUMMARY")
        print(f"Total files processed: {summary['total_files']}")
        print(f"Average validation score: {summary['overall_statistics']['avg_validation_score']:.3f}")
        print(f"Files with issues: {summary['overall_statistics']['files_with_issues']}")
        print(f"Excellent: {summary['overall_statistics']['excellent_files']}")
        print(f"Good: {summary['overall_statistics']['good_files']}")
        print(f"Caution: {summary['overall_statistics']['caution_files']}")
        print(f"Reject: {summary['overall_statistics']['reject_files']}")
        print(f"Summary saved to: {summary_path}")

def main():
    """Main function"""
    processor = BatchWaveTransformProcessor()
    results = processor.batch_process_all_files()
    print(f"\n✅ Batch processing complete! Processed {len(results)} files.")

if __name__ == "__main__":
    main() 