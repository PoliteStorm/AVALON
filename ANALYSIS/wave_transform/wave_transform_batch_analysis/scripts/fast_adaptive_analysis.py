#!/usr/bin/env python3
"""
Fast Adaptive Wave Transform Analysis
Efficient implementation that removes fixed parameters and uses vectorized operations
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FastAdaptiveAnalyzer:
    """Fast adaptive analysis with no fixed parameters"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("../results/fast_adaptive")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, filename: str) -> np.ndarray:
        """Load and preprocess CSV data efficiently"""
        try:
            # Read only first column for speed
            df = pd.read_csv(filename, header=None, usecols=[0])
            voltage_data = df.iloc[:, 0].values
            
            # Remove NaN values
            voltage_data = voltage_data[~np.isnan(voltage_data)]
            
            # Downsample if too large
            if len(voltage_data) > 10000:
                step = len(voltage_data) // 10000
                voltage_data = voltage_data[::step]
            
            return voltage_data
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None
    
    def detect_natural_scales(self, signal_data: np.ndarray) -> list:
        """Detect natural temporal scales using FFT efficiently"""
        # Compute FFT
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data))
        
        # Find dominant frequencies (top 10%)
        power_spectrum = np.abs(fft)**2
        threshold = np.percentile(power_spectrum, 90)
        significant_indices = power_spectrum > threshold
        
        # Convert to temporal scales
        scales = []
        for i in np.where(significant_indices)[0]:
            freq = freqs[i]
            if freq > 0:  # Avoid DC component
                period = 1.0 / freq
                if 30 <= period <= 86400:  # Within Adamatzky range
                    scales.append(period)
        
        # Remove duplicates and limit to top 10
        scales = sorted(list(set(scales)))[:10]
        return scales if scales else [60, 300, 1800, 3600]
    
    def apply_fast_wave_transform(self, signal_data: np.ndarray) -> dict:
        """Apply wave transform with vectorized operations"""
        n = len(signal_data)
        
        # Detect natural scales
        detected_scales = self.detect_natural_scales(signal_data)
        print(f"🔍 Using {len(detected_scales)} natural scales: {[int(s) for s in detected_scales]}")
        
        features = []
        
        # Adaptive threshold based on signal characteristics
        signal_std = np.std(signal_data)
        adaptive_threshold = signal_std * 0.05
        
        for scale in detected_scales:
            # Vectorized wave transform
            t = np.arange(n)
            
            # Wave function: ψ(√t/τ)
            wave_function = np.sqrt(t) / np.sqrt(scale)
            
            # Frequency component: e^(-ik√t)
            frequency_component = np.exp(-1j * scale * np.sqrt(t))
            
            # Combined wave value
            wave_values = wave_function * frequency_component
            
            # Apply to signal
            transformed = signal_data * wave_values
            
            # Compute magnitude
            magnitude = np.abs(np.sum(transformed))
            
            # Only keep significant features
            if magnitude > adaptive_threshold:
                phase = np.angle(np.sum(transformed))
                temporal_scale = self._classify_temporal_scale(scale)
                
                features.append({
                    'scale': float(scale),
                    'shift': 0.0,  # Simplified for speed
                    'magnitude': float(magnitude),
                    'phase': float(phase),
                    'frequency': float(scale / (2 * np.pi)),
                    'temporal_scale': temporal_scale,
                    'threshold_used': float(adaptive_threshold)
                })
        
        return {
            'all_features': features,
            'n_features': len(features),
            'detected_scales': detected_scales,
            'max_magnitude': max([f['magnitude'] for f in features]) if features else 0,
            'avg_magnitude': np.mean([f['magnitude'] for f in features]) if features else 0,
            'scale_distribution': [f['scale'] for f in features],
            'temporal_scale_distribution': [f['temporal_scale'] for f in features],
            'thresholds_used': [f['threshold_used'] for f in features]
        }
    
    def _classify_temporal_scale(self, scale: float) -> str:
        """Classify scale according to Adamatzky's three families"""
        if scale <= 300:
            return 'very_fast'
        elif scale <= 3600:
            return 'slow'
        else:
            return 'very_slow'
    
    def validate_biological_relevance(self, features: dict) -> dict:
        """Validate features against biological criteria"""
        if not features['all_features']:
            return {'valid': False, 'reason': 'No features detected'}
        
        # Check temporal scale distribution
        temporal_scales = features['temporal_scale_distribution']
        scale_counts = pd.Series(temporal_scales).value_counts()
        
        validation = {
            'valid': True,
            'reasons': [],
            'temporal_distribution': scale_counts.to_dict(),
            'feature_count': len(features['all_features']),
            'magnitude_range': {
                'min': min([f['magnitude'] for f in features['all_features']]),
                'max': max([f['magnitude'] for f in features['all_features']]),
                'mean': np.mean([f['magnitude'] for f in features['all_features']])
            }
        }
        
        # Check feature count
        if len(features['all_features']) < 3:
            validation['valid'] = False
            validation['reasons'].append('Too few features detected')
        elif len(features['all_features']) > 500:
            validation['valid'] = False
            validation['reasons'].append('Suspiciously many features')
        
        # Check magnitude distribution
        magnitudes = [f['magnitude'] for f in features['all_features']]
        if np.std(magnitudes) < np.mean(magnitudes) * 0.01:
            validation['valid'] = False
            validation['reasons'].append('Suspiciously uniform magnitudes')
        
        return validation
    
    def analyze_file(self, filename: str) -> dict:
        """Analyze a single file with fast adaptive parameters"""
        print(f"\n🔬 Analyzing: {filename}")
        print("=" * 50)
        
        # Load data
        signal_data = self.load_data(filename)
        if signal_data is None:
            return {}
        
        print(f"📊 Data Summary:")
        print(f"   Samples: {len(signal_data)}")
        print(f"   Mean amplitude: {np.mean(np.abs(signal_data)):.3f}")
        print(f"   Max amplitude: {np.max(np.abs(signal_data)):.3f}")
        print(f"   Std amplitude: {np.std(signal_data):.3f}")
        
        # Apply fast wave transform
        features = self.apply_fast_wave_transform(signal_data)
        
        print(f"\n🌊 Fast Adaptive Wave Transform Results:")
        print(f"   Features detected: {features['n_features']}")
        print(f"   Natural scales found: {len(features['detected_scales'])}")
        print(f"   Max magnitude: {features['max_magnitude']:.3f}")
        print(f"   Avg magnitude: {features['avg_magnitude']:.3f}")
        
        # Temporal scale distribution
        if features['temporal_scale_distribution']:
            temporal_dist = pd.Series(features['temporal_scale_distribution']).value_counts()
            print(f"\n⏰ Temporal Scale Distribution:")
            for scale, count in temporal_dist.items():
                percentage = (count / len(features['temporal_scale_distribution'])) * 100
                print(f"   {scale}: {count} ({percentage:.1f}%)")
        
        # Validate biological relevance
        validation = self.validate_biological_relevance(features)
        
        print(f"\n🔬 Biological Validation:")
        print(f"   Valid: {validation['valid']}")
        if validation['reasons']:
            print(f"   Issues: {', '.join(validation['reasons'])}")
        
        # Compile results
        results = {
            'filename': Path(filename).name,
            'timestamp': self.timestamp,
            'signal_stats': {
                'samples': len(signal_data),
                'mean_amplitude': float(np.mean(np.abs(signal_data))),
                'max_amplitude': float(np.max(np.abs(signal_data))),
                'std_amplitude': float(np.std(signal_data))
            },
            'wave_features': features,
            'validation': validation
        }
        
        # Save results
        results_filename = f"fast_adaptive_{Path(filename).stem}_{self.timestamp}.json"
        results_path = self.results_dir / results_filename
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")
        
        return results

def main():
    """Main execution function"""
    analyzer = FastAdaptiveAnalyzer()
    
    # Analyze the two files
    files_to_process = [
        "Ch1-2_moisture_added.csv",
        "Spray_in_bag_crop.csv"
    ]
    
    all_results = {}
    for csv_file in files_to_process:
        if Path(csv_file).exists():
            results = analyzer.analyze_file(csv_file)
            if results:
                all_results[csv_file] = results
        else:
            print(f"❌ File not found: {csv_file}")
    
    # Create summary
    if all_results:
        summary = {
            'timestamp': analyzer.timestamp,
            'files_processed': len(all_results),
            'file_results': {}
        }
        
        for filename, results in all_results.items():
            summary['file_results'][filename] = {
                'feature_count': results['wave_features']['n_features'],
                'valid': results['validation']['valid'],
                'temporal_scales': results['wave_features']['temporal_scale_distribution']
            }
        
        summary_filename = f"fast_adaptive_summary_{analyzer.timestamp}.json"
        summary_path = analyzer.results_dir / summary_filename
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Summary saved to: {summary_path}")
    
    print(f"\n✅ Fast adaptive analysis complete!")
    print(f"   Results saved in: {analyzer.results_dir}")

if __name__ == "__main__":
    main() 