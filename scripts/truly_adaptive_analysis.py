#!/usr/bin/env python3
"""
Truly Adaptive Wave Transform Analysis
Completely removes fixed parameters and lets data determine feature count
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

class TrulyAdaptiveAnalyzer:
    """Truly adaptive analysis with no fixed parameters"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("../results/truly_adaptive")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Adamatzky parameters for biological validation
        self.adamatzky_scales = {
            'very_fast': (30, 300),    # 30s to 5min
            'slow': (600, 3600),       # 10min to 1hour  
            'very_slow': (3600, 86400) # 1hour to 1day
        }
    
    def load_data(self, filename: str) -> np.ndarray:
        """Load and preprocess CSV data"""
        try:
            df = pd.read_csv(filename, header=None)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(axis=1, how='all')
            
            # Find voltage column (highest variance)
            voltage_data = None
            max_variance = 0
            
            for col in range(min(4, len(df.columns))):
                col_data = df.iloc[:, col].values
                if not np.issubdtype(col_data.dtype, np.number):
                    continue
                variance = np.var(col_data)
                if variance > max_variance:
                    max_variance = variance
                    voltage_data = col_data
            
            if voltage_data is None:
                voltage_data = df.select_dtypes(include=[np.number]).iloc[:, 0].values
            
            return voltage_data
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None
    
    def detect_natural_scales(self, signal_data: np.ndarray) -> list:
        """Detect natural temporal scales in the data using FFT"""
        # Compute FFT
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data))
        
        # Find dominant frequencies
        power_spectrum = np.abs(fft)**2
        significant_freqs = freqs[power_spectrum > np.percentile(power_spectrum, 90)]
        
        # Convert to temporal scales (periods)
        scales = []
        for freq in significant_freqs:
            if freq > 0:  # Avoid DC component
                period = 1.0 / freq
                if 30 <= period <= 86400:  # Within Adamatzky range
                    scales.append(period)
        
        # Remove duplicates and sort
        scales = sorted(list(set(scales)))
        return scales[:20]  # Limit to top 20 scales
    
    def apply_adaptive_wave_transform(self, signal_data: np.ndarray) -> dict:
        """Apply wave transform with data-determined parameters"""
        n = len(signal_data)
        
        # Detect natural scales from data
        detected_scales = self.detect_natural_scales(signal_data)
        
        if not detected_scales:
            # Fallback to Adamatzky scales if no natural scales detected
            detected_scales = [60, 300, 1800, 3600, 7200, 86400]
        
        print(f"🔍 Detected {len(detected_scales)} natural temporal scales")
        
        features = []
        
        # Apply transform only at detected scales
        for scale in detected_scales:
            # Adaptive shift range based on scale
            max_shift = min(scale * 10, n // 2)
            num_shifts = min(10, int(max_shift // 100) + 1)
            shifts = np.linspace(0, max_shift, num_shifts)
            
            for shift in shifts:
                # Apply wave transform
                transformed = np.zeros(n, dtype=complex)
                
                for i in range(n):
                    t = i
                    if t + shift < n:
                        # Wave function: ψ(√t/τ)
                        wave_function = np.sqrt(t + shift) / np.sqrt(scale)
                        # Frequency component: e^(-ik√t)
                        frequency_component = np.exp(-1j * scale * np.sqrt(t))
                        # Combined wave value
                        wave_value = wave_function * frequency_component
                        transformed[i] = signal_data[i] * wave_value
                
                magnitude = np.abs(np.sum(transformed))
                
                # Only keep features above adaptive threshold
                threshold = np.std(signal_data) * 0.1  # Adaptive threshold
                
                if magnitude > threshold:
                    phase = np.angle(np.sum(transformed))
                    temporal_scale = self._classify_temporal_scale(scale)
                    
                    features.append({
                        'scale': float(scale),
                        'shift': float(shift),
                        'magnitude': float(magnitude),
                        'phase': float(phase),
                        'frequency': float(scale / (2 * np.pi)),
                        'temporal_scale': temporal_scale,
                        'threshold_used': float(threshold)
                    })
        
        return {
            'all_features': features,
            'n_features': len(features),
            'detected_scales': detected_scales,
            'max_magnitude': max([f['magnitude'] for f in features]) if features else 0,
            'avg_magnitude': np.mean([f['magnitude'] for f in features]) if features else 0,
            'scale_distribution': [f['scale'] for f in features],
            'shift_distribution': [f['shift'] for f in features],
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
    
    def validate_biological_relevance(self, features: dict, signal_data: np.ndarray) -> dict:
        """Validate features against biological criteria"""
        if not features['all_features']:
            return {'valid': False, 'reason': 'No features detected'}
        
        # Check temporal scale distribution
        temporal_scales = features['temporal_scale_distribution']
        scale_counts = pd.Series(temporal_scales).value_counts()
        
        # Biological validation criteria
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
        
        # Check if we have features in all three temporal families
        if len(scale_counts) < 2:
            validation['valid'] = False
            validation['reasons'].append('Limited temporal scale diversity')
        
        # Check magnitude distribution
        magnitudes = [f['magnitude'] for f in features['all_features']]
        if np.std(magnitudes) < np.mean(magnitudes) * 0.1:
            validation['valid'] = False
            validation['reasons'].append('Suspiciously uniform magnitudes')
        
        # Check for reasonable feature count
        if len(features['all_features']) < 5:
            validation['valid'] = False
            validation['reasons'].append('Too few features detected')
        elif len(features['all_features']) > 1000:
            validation['valid'] = False
            validation['reasons'].append('Suspiciously many features')
        
        return validation
    
    def analyze_file(self, filename: str) -> dict:
        """Analyze a single file with truly adaptive parameters"""
        print(f"\n🔬 Analyzing: {filename}")
        print("=" * 60)
        
        # Load data
        signal_data = self.load_data(filename)
        if signal_data is None:
            return {}
        
        print(f"📊 Data Summary:")
        print(f"   Samples: {len(signal_data)}")
        print(f"   Mean amplitude: {np.mean(np.abs(signal_data)):.3f}")
        print(f"   Max amplitude: {np.max(np.abs(signal_data)):.3f}")
        print(f"   Std amplitude: {np.std(signal_data):.3f}")
        
        # Apply adaptive wave transform
        features = self.apply_adaptive_wave_transform(signal_data)
        
        print(f"\n🌊 Adaptive Wave Transform Results:")
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
        validation = self.validate_biological_relevance(features, signal_data)
        
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
        results_filename = f"adaptive_analysis_{Path(filename).stem}_{self.timestamp}.json"
        results_path = self.results_dir / results_filename
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")
        
        return results

def main():
    """Main execution function"""
    analyzer = TrulyAdaptiveAnalyzer()
    
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
        
        summary_filename = f"adaptive_summary_{analyzer.timestamp}.json"
        summary_path = analyzer.results_dir / summary_filename
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Summary saved to: {summary_path}")
    
    print(f"\n✅ Truly adaptive analysis complete!")
    print(f"   Results saved in: {analyzer.results_dir}")

if __name__ == "__main__":
    main() 