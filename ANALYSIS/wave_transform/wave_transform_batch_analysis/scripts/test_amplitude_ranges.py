#!/usr/bin/env python3
"""
Test Different Amplitude Ranges
Explore what patterns emerge at different sensitivity levels

This script tests the transform with different amplitude thresholds
to see what patterns are detected at various sensitivity levels.
"""

import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
import warnings

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

# Import centralized configuration
sys.path.append(str(Path(__file__).parent.parent / "config"))
from analysis_config import config

warnings.filterwarnings('ignore')

class AmplitudeRangeTester:
    """
    Test different amplitude ranges to explore pattern detection
    """
    
    def __init__(self):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Test different amplitude ranges
        self.amplitude_ranges = [
            {"name": "ultra_low", "min": 0.001, "max": 0.01, "description": "Ultra-low (0.001-0.01 mV)"},
            {"name": "very_low", "min": 0.01, "max": 0.1, "description": "Very low (0.01-0.1 mV)"},
            {"name": "low", "min": 0.1, "max": 1.0, "description": "Low (0.1-1.0 mV)"},
            {"name": "medium", "min": 1.0, "max": 10.0, "description": "Medium (1.0-10.0 mV)"},
            {"name": "high", "min": 10.0, "max": 100.0, "description": "High (10.0-100.0 mV)"},
            {"name": "very_high", "min": 100.0, "max": 1000.0, "description": "Very high (100.0-1000.0 mV)"},
            {"name": "ultra_high", "min": 1000.0, "max": 10000.0, "description": "Ultra-high (1000.0-10000.0 mV)"},
            {"name": "adamatzky_range", "min": 0.05, "max": 5.0, "description": "Adamatzky 2023 range (0.05-5.0 mV)"}
        ]
        
        print("🔬 AMPLITUDE RANGE TESTER")
        print("=" * 60)
        print("Testing different amplitude ranges for pattern detection")
    
    def load_data(self, filename: str) -> np.ndarray:
        """Load CSV data with error handling"""
        try:
            data = pd.read_csv(filename, header=None)
            # Use the column with highest variance (best signal)
            variances = data.var()
            best_column = variances.idxmax()
            signal_data = data.iloc[:, best_column].values
            return signal_data
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None
    
    def test_amplitude_range(self, signal_data: np.ndarray, amplitude_range: dict) -> dict:
        """
        Test pattern detection at a specific amplitude range
        
        Args:
            signal_data: Input signal data
            amplitude_range: Dictionary with min, max, name, description
            
        Returns:
            Results dictionary for this amplitude range
        """
        print(f"\n🔍 Testing: {amplitude_range['description']}")
        
        # Get adaptive parameters
        adaptive_percentile = self.config.get_adaptive_percentile(signal_data)
        adaptive_limits = self.config.get_adaptive_scale_limits(signal_data)
        
        # Compute FFT
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data))
        power_spectrum = np.abs(fft)**2
        
        # Use adaptive percentile
        threshold = np.percentile(power_spectrum, adaptive_percentile)
        significant_indices = power_spectrum > threshold
        
        # Get temporal ranges
        temporal_ranges = self.config.get_adaptive_temporal_ranges(signal_data)
        
        # Detect scales
        scales = []
        for i in np.where(significant_indices)[0]:
            freq = freqs[i]
            if freq > 0:
                period = 1.0 / freq
                
                # Check if period falls within any adaptive range
                very_fast_range = (temporal_ranges['very_fast']['min_isi'], 
                                 temporal_ranges['very_fast']['max_isi'])
                slow_range = (temporal_ranges['slow']['min_isi'], 
                            temporal_ranges['slow']['max_isi'])
                very_slow_range = (temporal_ranges['very_slow']['min_isi'], 
                                 temporal_ranges['very_slow']['max_isi'])
                
                if (very_fast_range[0] <= period <= very_fast_range[1] or
                    slow_range[0] <= period <= slow_range[1] or
                    very_slow_range[0] <= period <= very_slow_range[1]):
                    scales.append(period)
        
        max_scales = adaptive_limits['max_scales']
        detected_scales = sorted(list(set(scales)))[:max_scales]
        
        # Apply wave transform with amplitude filtering
        features = []
        n = len(signal_data)
        
        for scale in detected_scales:
            t = np.arange(n)
            wave_function = np.sqrt(t) / np.sqrt(scale)
            frequency_component = np.exp(-1j * scale * np.sqrt(t))
            wave_values = wave_function * frequency_component
            
            transformed = signal_data * wave_values
            magnitude = np.abs(np.sum(transformed))
            
            # Filter by amplitude range
            if amplitude_range['min'] <= magnitude <= amplitude_range['max']:
                phase = np.angle(np.sum(transformed))
                
                # Classify temporal scale
                if scale <= 300:
                    temporal_scale = 'very_fast'
                elif scale <= 3600:
                    temporal_scale = 'slow'
                else:
                    temporal_scale = 'very_slow'
                
                features.append({
                    'scale': float(scale),
                    'magnitude': float(magnitude),
                    'phase': float(phase),
                    'temporal_scale': temporal_scale
                })
        
        # Analyze results
        temporal_distribution = {}
        if features:
            temporal_scales = [f['temporal_scale'] for f in features]
            for scale in set(temporal_scales):
                temporal_distribution[scale] = temporal_scales.count(scale)
        
        return {
            'amplitude_range': amplitude_range,
            'features_detected': len(features),
            'temporal_distribution': temporal_distribution,
            'magnitude_range': {
                'min': min([f['magnitude'] for f in features]) if features else 0,
                'max': max([f['magnitude'] for f in features]) if features else 0,
                'mean': np.mean([f['magnitude'] for f in features]) if features else 0
            },
            'scales_detected': [f['scale'] for f in features],
            'adaptive_parameters': {
                'percentile_used': adaptive_percentile,
                'max_scales_used': max_scales
            }
        }
    
    def test_all_ranges(self, filename: str) -> dict:
        """
        Test all amplitude ranges on a single file
        
        Args:
            filename: Path to CSV file
            
        Returns:
            Results for all amplitude ranges
        """
        print(f"\n🔬 Testing amplitude ranges on: {filename}")
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
        
        # Test each amplitude range
        all_results = {}
        
        for amplitude_range in self.amplitude_ranges:
            results = self.test_amplitude_range(signal_data, amplitude_range)
            all_results[amplitude_range['name']] = results
            
            print(f"   {amplitude_range['description']}: {results['features_detected']} features")
            if results['temporal_distribution']:
                for scale, count in results['temporal_distribution'].items():
                    print(f"     - {scale}: {count}")
        
        return {
            'filename': Path(filename).name,
            'timestamp': self.timestamp,
            'signal_stats': {
                'samples': len(signal_data),
                'mean_amplitude': float(np.mean(np.abs(signal_data))),
                'max_amplitude': float(np.max(np.abs(signal_data))),
                'std_amplitude': float(np.std(signal_data))
            },
            'amplitude_range_results': all_results
        }

def main():
    """Main execution function"""
    tester = AmplitudeRangeTester()
    
    # Test on one file - use a smaller file that we know works
    data_dir = Path("data/raw")
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found")
        return
    
    # Use a smaller file that we know works (Ag_M_I+4R file)
    small_files = [f for f in csv_files if f.stat().st_size < 100000]  # Less than 100KB
    if small_files:
        test_file = small_files[0]  # Use first small file
    else:
        test_file = csv_files[0]  # Fallback to first file
    
    print(f"Testing file: {test_file.name}")
    results = tester.test_all_ranges(str(test_file))
    
    if results:
        # Save results
        results_filename = f"amplitude_range_test_{Path(test_file).stem}_{tester.timestamp}.json"
        results_path = Path("results/analysis/latest") / results_filename
        
        # Ensure directory exists
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert NumPy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_numpy_types(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")
        
        # Print summary
        print(f"\n📊 AMPLITUDE RANGE TEST SUMMARY")
        print("=" * 60)
        for range_name, range_results in results['amplitude_range_results'].items():
            print(f"\n{range_results['amplitude_range']['description']}:")
            print(f"   Features detected: {range_results['features_detected']}")
            if range_results['temporal_distribution']:
                for scale, count in range_results['temporal_distribution'].items():
                    print(f"   - {scale}: {count} features")
            else:
                print("   - No features in this amplitude range")
        
    else:
        print("❌ Failed to analyze file")

if __name__ == "__main__":
    main() 