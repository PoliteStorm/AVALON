#!/usr/bin/env python3
"""
Focused Adamatzky Analysis
Process the 2 compliant files with comprehensive validation and comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FocusedAdamatzkyAnalyzer:
    """Focused analysis of Adamatzky-compliant files with comprehensive validation"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("../results/focused_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Adamatzky 2023 parameters (not forcing patterns)
        self.adamatzky_params = {
            'temporal_scales': {
                'very_slow': {'min_isi': 3600, 'max_isi': float('inf'), 'description': 'Hour scale'},
                'slow': {'min_isi': 600, 'max_isi': 3600, 'description': '10-minute scale'},
                'very_fast': {'min_isi': 30, 'max_isi': 300, 'description': 'Half-minute scale'}
            },
            'sampling_rate': 1,  # Hz
            'voltage_range': {'min': -39, 'max': 39},  # mV
            'spike_amplitude': {'min': 0.05, 'max': 5.0},  # mV
            'time_compression': 3000  # 3000 seconds = 1 day (longer acquisition)
        }
        
        # Validation thresholds (not forcing patterns)
        self.validation_thresholds = {
            'biological_plausibility': 0.6,
            'mathematical_consistency': 0.7,
            'false_positive_rate': 0.15,
            'signal_quality': 0.5
        }
    
    def load_and_preprocess(self, csv_file):
        """Load and preprocess CSV with Adamatzky-corrected methods"""
        try:
            df = pd.read_csv(csv_file, header=None, low_memory=False)
            
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
            
            # Apply time compression (not forcing patterns)
            original_samples = len(voltage_data)
            compressed_samples = original_samples // self.adamatzky_params['time_compression']
            
            if compressed_samples < 10:
                compression_factor = max(1, original_samples // 100)
                compressed_data = voltage_data[::compression_factor]
            else:
                compressed_data = voltage_data[::self.adamatzky_params['time_compression']]
            
            return compressed_data, {
                'filename': Path(csv_file).name,
                'original_samples': original_samples,
                'compressed_samples': len(compressed_data),
                'best_column': best_column,
                'compression_factor': self.adamatzky_params['time_compression']
            }
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            return None, {}
    
    def apply_wave_transform(self, signal_data):
        """Apply wave transform W(k,τ) with Adamatzky parameters"""
        n = len(signal_data)
        
        # Adamatzky-corrected scale and shift parameters
        scales = np.concatenate([
            np.linspace(30, 300, 10),      # Very fast (half-minute)
            np.linspace(600, 3600, 15),    # Slow (10-minute)
            np.linspace(3600, 86400, 10)   # Very slow (hour)
        ])
        
        shifts = np.linspace(0, 86400, 20)  # Up to 1 day
        
        features = []
        for scale in scales:
            for shift in shifts:
                # Apply wave transform: W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
                transformed = np.zeros(n, dtype=complex)
                compressed_t = np.arange(n) / self.adamatzky_params['time_compression']
                
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
                        'scale': float(scale),
                        'shift': float(shift),
                        'magnitude': float(magnitude),
                        'phase': float(phase),
                        'frequency': float(scale / (2 * np.pi)),
                        'temporal_scale': temporal_scale
                    })
        
        return features
    
    def _classify_temporal_scale(self, scale):
        """Classify scale according to Adamatzky's three families"""
        if scale <= 300:
            return 'very_fast'
        elif scale <= 3600:
            return 'slow'
        else:
            return 'very_slow'
    
    def validate_against_adamatzky(self, features, original_signal):
        """Validate results against Adamatzky's methods (not forcing patterns)"""
        
        validation_results = {
            'temporal_alignment': 0.0,
            'biological_plausibility': 0.0,
            'mathematical_consistency': 0.0,
            'false_positive_rate': 0.0,
            'signal_quality': 0.0,
            'issues': []
        }
        
        if not features:
            validation_results['issues'].append("No features detected")
            return validation_results
        
        # Extract magnitudes and temporal scales
        magnitudes = [f['magnitude'] for f in features]
        temporal_scales = [f['temporal_scale'] for f in features]
        
        # 1. Temporal Alignment Check (Adamatzky expects primarily slow/very slow)
        scale_counts = {}
        for scale in temporal_scales:
            scale_counts[scale] = scale_counts.get(scale, 0) + 1
        
        total_features = len(temporal_scales)
        if total_features > 0:
            slow_ratio = (scale_counts.get('slow', 0) + scale_counts.get('very_slow', 0)) / total_features
            validation_results['temporal_alignment'] = slow_ratio
            
            if slow_ratio < 0.4:
                validation_results['issues'].append("Poor alignment with Adamatzky temporal scales")
        
        # 2. Biological Plausibility Check
        # Check for log-normal distribution in magnitudes (biological signals)
        if len(magnitudes) > 10:
            try:
                log_magnitudes = np.log(magnitudes)
                _, p_value = stats.normaltest(log_magnitudes)
                if p_value > 0.05:  # Normal distribution in log space
                    validation_results['biological_plausibility'] = 0.8
                else:
                    validation_results['biological_plausibility'] = 0.3
            except:
                validation_results['biological_plausibility'] = 0.5
        
        # 3. Mathematical Consistency Check
        # Check for reasonable magnitude distribution
        if magnitudes:
            magnitude_cv = np.std(magnitudes) / np.mean(magnitudes)
            if 0.1 < magnitude_cv < 10.0:  # Reasonable coefficient of variation
                validation_results['mathematical_consistency'] = 0.8
            else:
                validation_results['mathematical_consistency'] = 0.3
        
        # 4. False Positive Detection
        # Check for suspicious patterns (too regular, too random)
        if len(magnitudes) > 20:
            # Test for randomness using autocorrelation
            autocorr = np.correlate(magnitudes, magnitudes, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            randomness_score = np.mean(np.abs(autocorr[1:10]))
            validation_results['false_positive_rate'] = 1.0 - randomness_score
            
            if randomness_score < 0.1:
                validation_results['issues'].append("Suspiciously random patterns detected")
            elif randomness_score > 0.9:
                validation_results['issues'].append("Suspiciously regular patterns detected")
        
        # 5. Signal Quality Assessment
        # Check original signal characteristics
        if len(original_signal) > 0:
            signal_variance = np.var(original_signal)
            signal_range = np.max(original_signal) - np.min(original_signal)
            
            if signal_variance > 0.001 and signal_range > 0.01:
                validation_results['signal_quality'] = 0.8
            else:
                validation_results['signal_quality'] = 0.3
        
        return validation_results
    
    def create_comprehensive_visualization(self, features, original_signal, filename):
        """Create comprehensive visualization of results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Adamatzky Wave Transform Analysis: {filename}', fontsize=16)
        
        # Extract data
        magnitudes = [f['magnitude'] for f in features]
        phases = [f['phase'] for f in features]
        scales = [f['scale'] for f in features]
        temporal_scales = [f['temporal_scale'] for f in features]
        
        # 1. Original Signal
        axes[0, 0].plot(original_signal[:1000], 'b-', alpha=0.7)
        axes[0, 0].set_title('Original Signal (First 1000 samples)')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Voltage (mV)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Magnitude Distribution
        axes[0, 1].hist(magnitudes, bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('Wave Transform Magnitude Distribution')
        axes[0, 1].set_xlabel('Magnitude')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Temporal Scale Distribution
        scale_counts = {}
        for scale in temporal_scales:
            scale_counts[scale] = scale_counts.get(scale, 0) + 1
        
        scales_list = list(scale_counts.keys())
        counts_list = list(scale_counts.values())
        
        axes[1, 0].bar(scales_list, counts_list, color=['red', 'orange', 'blue'])
        axes[1, 0].set_title('Temporal Scale Distribution')
        axes[1, 0].set_xlabel('Temporal Scale')
        axes[1, 0].set_ylabel('Feature Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Phase vs Magnitude Scatter
        axes[1, 1].scatter(phases, magnitudes, alpha=0.6, c=scales, cmap='viridis')
        axes[1, 1].set_title('Phase vs Magnitude (colored by scale)')
        axes[1, 1].set_xlabel('Phase (radians)')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"focused_analysis_{Path(filename).stem}_{self.timestamp}.png"
        plot_path = self.results_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def analyze_file(self, csv_file):
        """Comprehensive analysis of a single file"""
        print(f"🔬 Analyzing: {csv_file}")
        
        # Load and preprocess
        compressed_data, metadata = self.load_and_preprocess(csv_file)
        if compressed_data is None:
            return None
        
        print(f"   📊 Original samples: {metadata['original_samples']}")
        print(f"   📊 Compressed samples: {metadata['compressed_samples']}")
        
        # Apply wave transform
        features = self.apply_wave_transform(compressed_data)
        print(f"   📊 Features detected: {len(features)}")
        
        # Validate against Adamatzky
        validation = self.validate_against_adamatzky(features, compressed_data)
        
        # Create visualization
        plot_path = self.create_comprehensive_visualization(features, compressed_data, metadata['filename'])
        
        # Compile results
        results = {
            'filename': metadata['filename'],
            'timestamp': self.timestamp,
            'metadata': metadata,
            'features': features,
            'validation': validation,
            'plot_path': plot_path,
            'summary': {
                'n_features': len(features),
                'max_magnitude': max([f['magnitude'] for f in features]) if features else 0,
                'avg_magnitude': np.mean([f['magnitude'] for f in features]) if features else 0,
                'temporal_distribution': {
                    'very_fast': len([f for f in features if f['temporal_scale'] == 'very_fast']),
                    'slow': len([f for f in features if f['temporal_scale'] == 'slow']),
                    'very_slow': len([f for f in features if f['temporal_scale'] == 'very_slow'])
                }
            }
        }
        
        # Save results
        results_filename = f"focused_analysis_{Path(csv_file).stem}_{self.timestamp}.json"
        results_path = self.results_dir / results_filename
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   ✅ Analysis complete - saved to {results_path}")
        print(f"   📊 Validation score: {np.mean(list(validation.values())[:5]):.3f}")
        
        return results
    
    def run_focused_analysis(self):
        """Run focused analysis on the 2 Adamatzky-compliant files"""
        
        print("🎯 FOCUSED ADAMATZKY ANALYSIS")
        print("=" * 60)
        print("Processing 2 Adamatzky-compliant files with comprehensive validation")
        print()
        
        # The 2 compliant files from fast scan
        compliant_files = [
            "wave_transform_batch_analysis/Spray_in_bag_crop.csv",
            "wave_transform_batch_analysis/Ch1-2_moisture_added.csv"
        ]
        
        all_results = []
        
        for csv_file in compliant_files:
            file_path = Path(csv_file)
            if file_path.exists():
                results = self.analyze_file(str(file_path))
                if results:
                    all_results.append(results)
            else:
                print(f"❌ File not found: {csv_file}")
        
        # Create comparison report
        self.create_comparison_report(all_results)
        
        print(f"\n✅ Focused analysis complete!")
        print(f"   Files processed: {len(all_results)}")
        print(f"   Results saved in: {self.results_dir}")
        
        return all_results
    
    def create_comparison_report(self, results):
        """Create comparison report between the two files"""
        
        if len(results) < 2:
            return
        
        comparison = {
            'timestamp': self.timestamp,
            'files_compared': [r['filename'] for r in results],
            'comparison_metrics': {}
        }
        
        # Compare key metrics
        for i, result in enumerate(results):
            filename = result['filename']
            summary = result['summary']
            validation = result['validation']
            
            comparison['comparison_metrics'][filename] = {
                'n_features': summary['n_features'],
                'max_magnitude': summary['max_magnitude'],
                'avg_magnitude': summary['avg_magnitude'],
                'temporal_distribution': summary['temporal_distribution'],
                'validation_scores': {
                    'temporal_alignment': validation['temporal_alignment'],
                    'biological_plausibility': validation['biological_plausibility'],
                    'mathematical_consistency': validation['mathematical_consistency'],
                    'false_positive_rate': validation['false_positive_rate'],
                    'signal_quality': validation['signal_quality']
                }
            }
        
        # Save comparison
        comparison_filename = f"focused_comparison_{self.timestamp}.json"
        comparison_path = self.results_dir / comparison_filename
        
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"📊 Comparison report saved: {comparison_path}")

def main():
    """Main function for focused Adamatzky analysis"""
    analyzer = FocusedAdamatzkyAnalyzer()
    results = analyzer.run_focused_analysis()
    
    # Print summary
    print(f"\n📊 FOCUSED ANALYSIS SUMMARY")
    print("=" * 60)
    for result in results:
        filename = result['filename']
        summary = result['summary']
        validation = result['validation']
        
        print(f"\n📁 {filename}:")
        print(f"   Features: {summary['n_features']}")
        print(f"   Max magnitude: {summary['max_magnitude']:.3f}")
        print(f"   Temporal distribution:")
        for scale, count in summary['temporal_distribution'].items():
            print(f"     {scale}: {count}")
        
        avg_validation = np.mean(list(validation.values())[:5])
        print(f"   Validation score: {avg_validation:.3f}")

if __name__ == "__main__":
    main() 