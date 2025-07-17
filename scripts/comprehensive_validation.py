#!/usr/bin/env python3
"""
Comprehensive Validation for Wave Transform
Tests multiple parameter combinations and assesses biological relevance
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

class ComprehensiveValidator:
    """Comprehensive validation framework"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("../results/comprehensive_validation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, filename: str) -> np.ndarray:
        """Load and preprocess CSV data"""
        try:
            df = pd.read_csv(filename, header=None, usecols=[0])
            voltage_data = df.iloc[:, 0].values
            voltage_data = voltage_data[~np.isnan(voltage_data)]
            
            # Downsample if too large
            if len(voltage_data) > 10000:
                step = len(voltage_data) // 10000
                voltage_data = voltage_data[::step]
            
            return voltage_data
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return None
    
    def test_parameter_combinations(self, signal_data: np.ndarray) -> dict:
        """Test multiple parameter combinations"""
        results = {}
        
        # Test different scale detection methods
        methods = {
            'fft_dominant': self._detect_scales_fft_dominant,
            'fft_peaks': self._detect_scales_fft_peaks,
            'autocorrelation': self._detect_scales_autocorrelation,
            'wavelet': self._detect_scales_wavelet
        }
        
        for method_name, method_func in methods.items():
            try:
                scales = method_func(signal_data)
                results[method_name] = {
                    'scales': scales,
                    'n_scales': len(scales),
                    'scale_range': (min(scales), max(scales)) if scales else (0, 0)
                }
            except Exception as e:
                results[method_name] = {'error': str(e)}
        
        return results
    
    def _detect_scales_fft_dominant(self, signal_data: np.ndarray) -> list:
        """Detect scales using dominant FFT frequencies"""
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data))
        power_spectrum = np.abs(fft)**2
        
        # Find dominant frequencies
        threshold = np.percentile(power_spectrum, 90)
        significant_indices = power_spectrum > threshold
        
        scales = []
        for i in np.where(significant_indices)[0]:
            freq = freqs[i]
            if freq > 0:
                period = 1.0 / freq
                if 30 <= period <= 86400:
                    scales.append(period)
        
        return sorted(list(set(scales)))[:10]
    
    def _detect_scales_fft_peaks(self, signal_data: np.ndarray) -> list:
        """Detect scales using FFT peak detection"""
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(len(signal_data))
        power_spectrum = np.abs(fft)**2
        
        # Find peaks in power spectrum
        peaks, _ = signal.find_peaks(power_spectrum, height=np.percentile(power_spectrum, 80))
        
        scales = []
        for peak_idx in peaks:
            freq = freqs[peak_idx]
            if freq > 0:
                period = 1.0 / freq
                if 30 <= period <= 86400:
                    scales.append(period)
        
        return sorted(list(set(scales)))[:10]
    
    def _detect_scales_autocorrelation(self, signal_data: np.ndarray) -> list:
        """Detect scales using autocorrelation"""
        # Compute autocorrelation
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[len(signal_data)-1:]
        
        # Find peaks in autocorrelation
        peaks, _ = signal.find_peaks(autocorr, distance=10)
        
        scales = []
        for peak_idx in peaks:
            if peak_idx > 0:
                scales.append(float(peak_idx))
        
        return sorted(list(set(scales)))[:10]
    
    def _detect_scales_wavelet(self, signal_data: np.ndarray) -> list:
        """Detect scales using wavelet-like approach"""
        # Simple wavelet-like scale detection
        scales = []
        n = len(signal_data)
        
        for scale_factor in [1, 2, 4, 8, 16, 32]:
            scale = n // scale_factor
            if 30 <= scale <= 86400:
                scales.append(scale)
        
        return scales
    
    def cross_validate_features(self, signal_data: np.ndarray, scales: list) -> dict:
        """Cross-validate feature detection"""
        # Split data into segments
        n_segments = 5
        segment_length = len(signal_data) // n_segments
        
        segment_features = []
        
        for i in range(n_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            segment = signal_data[start_idx:end_idx]
            
            # Apply wave transform to segment
            features = self._apply_wave_transform_segment(segment, scales)
            segment_features.append(features)
        
        # Analyze consistency across segments
        all_magnitudes = []
        all_scales = []
        
        for features in segment_features:
            all_magnitudes.extend([f['magnitude'] for f in features])
            all_scales.extend([f['scale'] for f in features])
        
        consistency = {
            'magnitude_std': np.std(all_magnitudes),
            'magnitude_cv': np.std(all_magnitudes) / np.mean(all_magnitudes) if all_magnitudes else 0,
            'scale_consistency': len(set(all_scales)) / len(all_scales) if all_scales else 0,
            'n_segments': n_segments,
            'avg_features_per_segment': np.mean([len(f) for f in segment_features])
        }
        
        return consistency
    
    def _apply_wave_transform_segment(self, segment: np.ndarray, scales: list) -> list:
        """Apply wave transform to a data segment"""
        features = []
        n = len(segment)
        
        for scale in scales:
            t = np.arange(n)
            wave_function = np.sqrt(t) / np.sqrt(scale)
            frequency_component = np.exp(-1j * scale * np.sqrt(t))
            wave_values = wave_function * frequency_component
            
            transformed = segment * wave_values
            magnitude = np.abs(np.sum(transformed))
            
            if magnitude > np.std(segment) * 0.01:
                features.append({
                    'scale': float(scale),
                    'magnitude': float(magnitude),
                    'phase': float(np.angle(np.sum(transformed)))
                })
        
        return features
    
    def assess_biological_relevance(self, signal_data: np.ndarray, features: list) -> dict:
        """Assess biological relevance of detected features"""
        if not features:
            return {'relevant': False, 'reason': 'No features detected'}
        
        # Biological criteria
        criteria = {
            'temporal_scales': bool(self._check_temporal_scales(features)),
            'magnitude_distribution': bool(self._check_magnitude_distribution(features)),
            'phase_coherence': bool(self._check_phase_coherence(features)),
            'signal_noise_ratio': bool(self._check_signal_noise_ratio(signal_data, features)),
            'stability': bool(self._check_stability(features))
        }
        
        # Overall assessment
        relevant_criteria = sum(criteria.values())
        total_criteria = len(criteria)
        relevance_score = relevant_criteria / total_criteria
        
        return {
            'relevant': relevance_score > 0.6,
            'relevance_score': relevance_score,
            'criteria': criteria,
            'n_features': len(features)
        }
    
    def _check_temporal_scales(self, features: list) -> bool:
        """Check if temporal scales are biologically plausible"""
        scales = [f['scale'] for f in features]
        
        # Check if scales fall within Adamatzky's ranges
        very_fast = sum(1 for s in scales if s <= 300)
        slow = sum(1 for s in scales if 300 < s <= 3600)
        very_slow = sum(1 for s in scales if s > 3600)
        
        # Should have at least some very_fast features
        return very_fast > 0
    
    def _check_magnitude_distribution(self, features: list) -> bool:
        """Check if magnitude distribution is reasonable"""
        magnitudes = [f['magnitude'] for f in features]
        
        # Check for reasonable variation
        cv = np.std(magnitudes) / np.mean(magnitudes) if magnitudes else 0
        return 0.1 < cv < 10.0  # Reasonable coefficient of variation
    
    def _check_phase_coherence(self, features: list) -> bool:
        """Check phase coherence"""
        phases = [f['phase'] for f in features]
        
        # Check if phases are not all identical
        phase_std = np.std(phases)
        return phase_std > 0.1
    
    def _check_signal_noise_ratio(self, signal_data: np.ndarray, features: list) -> bool:
        """Check signal-to-noise ratio"""
        if not features:
            return False
        
        signal_power = np.mean([f['magnitude']**2 for f in features])
        noise_power = np.var(signal_data)
        
        snr = signal_power / noise_power if noise_power > 0 else 0
        return snr > 0.01
    
    def _check_stability(self, features: list) -> bool:
        """Check feature stability"""
        if len(features) < 2:
            return False
        
        magnitudes = [f['magnitude'] for f in features]
        # Check if magnitudes are reasonably stable
        return np.std(magnitudes) < np.mean(magnitudes)
    
    def validate_file(self, filename: str) -> dict:
        """Comprehensive validation of a single file"""
        print(f"\n🔬 Comprehensive Validation: {filename}")
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
        
        # Test parameter combinations
        print(f"\n🧪 Testing Parameter Combinations:")
        param_results = self.test_parameter_combinations(signal_data)
        
        for method, result in param_results.items():
            if 'error' not in result:
                print(f"   {method}: {result['n_scales']} scales, range: {result['scale_range']}")
            else:
                print(f"   {method}: Error - {result['error']}")
        
        # Use best method for further analysis
        best_method = max(param_results.keys(), 
                         key=lambda x: param_results[x].get('n_scales', 0) if 'error' not in param_results[x] else 0)
        
        if 'error' not in param_results[best_method]:
            scales = param_results[best_method]['scales']
            print(f"\n✅ Using {best_method} method with {len(scales)} scales")
            
            # Cross-validate features
            consistency = self.cross_validate_features(signal_data, scales)
            
            print(f"\n🔄 Cross-Validation Results:")
            print(f"   Magnitude CV: {consistency['magnitude_cv']:.3f}")
            print(f"   Scale consistency: {consistency['scale_consistency']:.3f}")
            print(f"   Avg features per segment: {consistency['avg_features_per_segment']:.1f}")
            
            # Apply wave transform with best scales
            features = self._apply_wave_transform_segment(signal_data, scales)
            
            # Assess biological relevance
            biological_assessment = self.assess_biological_relevance(signal_data, features)
            
            print(f"\n🧬 Biological Relevance Assessment:")
            print(f"   Relevant: {biological_assessment['relevant']}")
            print(f"   Relevance score: {biological_assessment['relevance_score']:.3f}")
            print(f"   Features: {biological_assessment['n_features']}")
            
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
                'parameter_tests': param_results,
                'best_method': best_method,
                'cross_validation': consistency,
                'features': features,
                'biological_assessment': biological_assessment
            }
            
            # Save results
            results_filename = f"comprehensive_validation_{Path(filename).stem}_{self.timestamp}.json"
            results_path = self.results_dir / results_filename
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Results saved to: {results_path}")
            
            return results
        
        return {}

def main():
    """Main execution function"""
    validator = ComprehensiveValidator()
    
    # Validate the two files
    files_to_validate = [
        "Ch1-2_moisture_added.csv",
        "Spray_in_bag_crop.csv"
    ]
    
    all_results = {}
    for csv_file in files_to_validate:
        if Path(csv_file).exists():
            results = validator.validate_file(csv_file)
            if results:
                all_results[csv_file] = results
        else:
            print(f"❌ File not found: {csv_file}")
    
    # Create summary
    if all_results:
        summary = {
            'timestamp': validator.timestamp,
            'files_validated': len(all_results),
            'validation_summary': {}
        }
        
        for filename, results in all_results.items():
            if 'biological_assessment' in results:
                summary['validation_summary'][filename] = {
                    'relevant': results['biological_assessment']['relevant'],
                    'relevance_score': results['biological_assessment']['relevance_score'],
                    'n_features': results['biological_assessment']['n_features'],
                    'best_method': results.get('best_method', 'unknown')
                }
        
        summary_filename = f"comprehensive_validation_summary_{validator.timestamp}.json"
        summary_path = validator.results_dir / summary_filename
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Validation summary saved to: {summary_path}")
    
    print(f"\n✅ Comprehensive validation complete!")
    print(f"   Results saved in: {validator.results_dir}")

if __name__ == "__main__":
    main() 