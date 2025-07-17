#!/usr/bin/env python3
"""
Adaptive Wave Transform - Let Data Speak for Itself
Data-driven approach that adapts to each file's unique characteristics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.optimize import curve_fit
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdaptiveWaveTransform:
    """Adaptive wave transform that lets data determine optimal parameters"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("../results/adaptive_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Adaptive parameters (not fixed)
        self.adaptive_params = {
            'min_scale_samples': 5,      # Minimum scales to test
            'max_scale_samples': 50,     # Maximum scales to test
            'min_shift_samples': 5,      # Minimum shifts to test
            'max_shift_samples': 30,     # Maximum shifts to test
            'magnitude_threshold': 0.01, # Minimum magnitude to keep feature
            'correlation_threshold': 0.3  # Minimum correlation for feature selection
        }
    
    def analyze_signal_characteristics(self, signal_data):
        """Analyze signal to determine optimal parameters"""
        print("🔬 ANALYZING SIGNAL CHARACTERISTICS")
        print("=" * 50)
        
        n_samples = len(signal_data)
        
        # 1. Frequency domain analysis
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(n_samples)
        power_spectrum = np.abs(fft)**2
        
        # Find dominant frequencies
        dominant_freqs = freqs[np.argsort(power_spectrum)[-10:]]
        dominant_periods = 1 / np.abs(dominant_freqs[dominant_freqs != 0])
        
        print(f"📊 Signal length: {n_samples} samples")
        print(f"📊 Dominant periods: {dominant_periods[:5]}")
        
        # 2. Temporal scale analysis
        # Auto-correlation to find natural temporal scales
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation (natural temporal scales)
        peaks, _ = signal.find_peaks(autocorr, height=np.max(autocorr)*0.1)
        natural_scales = peaks[:10]  # Top 10 natural scales
        
        print(f"📊 Natural temporal scales: {natural_scales[:5]}")
        
        # 3. Variance analysis
        # Sliding window variance to find scale-dependent patterns
        window_sizes = [10, 50, 100, 500, 1000]
        variances = []
        
        for window_size in window_sizes:
            if window_size < n_samples:
                rolling_var = pd.Series(signal_data).rolling(window_size).var().dropna()
                variances.append(np.mean(rolling_var))
            else:
                variances.append(np.var(signal_data))
        
        print(f"📊 Scale-dependent variances: {variances}")
        
        # 4. Determine adaptive parameters
        optimal_scales = self._determine_optimal_scales(dominant_periods, natural_scales, variances)
        optimal_shifts = self._determine_optimal_shifts(n_samples, natural_scales)
        
        return {
            'optimal_scales': optimal_scales,
            'optimal_shifts': optimal_shifts,
            'dominant_periods': dominant_periods,
            'natural_scales': natural_scales,
            'scale_variances': variances,
            'signal_characteristics': {
                'length': n_samples,
                'variance': np.var(signal_data),
                'skewness': stats.skew(signal_data),
                'kurtosis': stats.kurtosis(signal_data)
            }
        }
    
    def _determine_optimal_scales(self, dominant_periods, natural_scales, variances):
        """Determine optimal scales based on data characteristics"""
        
        # Combine different scale sources
        all_scales = []
        
        # 1. Add dominant frequency periods
        all_scales.extend(dominant_periods)
        
        # 2. Add natural temporal scales
        all_scales.extend(natural_scales)
        
        # 3. Add variance-based scales (where variance changes significantly)
        variance_threshold = np.std(variances) * 0.5
        for i, var in enumerate(variances):
            if i > 0 and abs(var - variances[i-1]) > variance_threshold:
                all_scales.append(10 * (2**i))  # Exponential scale
        
        # 4. Add log-spaced scales for comprehensive coverage
        min_scale = max(1, int(np.min(all_scales) * 0.1))
        max_scale = min(len(variances) * 100, int(np.max(all_scales) * 10))
        log_scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 20)
        all_scales.extend(log_scales)
        
        # Remove duplicates and sort
        unique_scales = np.unique(all_scales)
        unique_scales = unique_scales[(unique_scales >= 1) & (unique_scales <= len(variances) * 1000)]
        
        # Limit to reasonable number of scales
        n_scales = min(len(unique_scales), self.adaptive_params['max_scale_samples'])
        optimal_scales = np.sort(unique_scales)[:n_scales]
        
        print(f"📊 Optimal scales determined: {len(optimal_scales)} scales")
        print(f"📊 Scale range: {optimal_scales[0]:.1f} - {optimal_scales[-1]:.1f}")
        
        return optimal_scales
    
    def _determine_optimal_shifts(self, n_samples, natural_scales):
        """Determine optimal shifts based on signal length and natural scales"""
        
        # Adaptive shift strategy
        if n_samples < 100:
            # Short signal: dense sampling
            shifts = np.linspace(0, n_samples, min(10, n_samples//2))
        elif n_samples < 1000:
            # Medium signal: moderate sampling
            shifts = np.linspace(0, n_samples, 15)
        else:
            # Long signal: sparse sampling
            shifts = np.linspace(0, n_samples, 20)
        
        # Add natural scale-based shifts
        natural_shifts = natural_scales[natural_scales < n_samples//2]
        shifts = np.concatenate([shifts, natural_shifts])
        shifts = np.unique(shifts)
        
        # Limit number of shifts
        n_shifts = min(len(shifts), self.adaptive_params['max_shift_samples'])
        optimal_shifts = np.sort(shifts)[:n_shifts]
        
        print(f"📊 Optimal shifts determined: {len(optimal_shifts)} shifts")
        print(f"📊 Shift range: {optimal_shifts[0]:.1f} - {optimal_shifts[-1]:.1f}")
        
        return optimal_shifts
    
    def apply_adaptive_wave_transform(self, signal_data):
        """Apply adaptive wave transform W(k,τ)"""
        print("\n🌊 APPLYING ADAPTIVE WAVE TRANSFORM")
        print("=" * 50)
        
        # 1. Analyze signal characteristics
        characteristics = self.analyze_signal_characteristics(signal_data)
        
        # 2. Apply adaptive wave transform
        optimal_scales = characteristics['optimal_scales']
        optimal_shifts = characteristics['optimal_shifts']
        
        features = []
        n_samples = len(signal_data)
        
        print(f"🔬 Testing {len(optimal_scales)} scales × {len(optimal_shifts)} shifts = {len(optimal_scales) * len(optimal_shifts)} combinations")
        
        for scale in optimal_scales:
            for shift in optimal_shifts:
                # Apply adaptive wave transform
                transformed = np.zeros(n_samples, dtype=complex)
                
                for i in range(n_samples):
                    t = i
                    if t + shift < n_samples:
                        # Adaptive wave function based on data characteristics
                        wave_function = np.sqrt(t + shift) * scale
                        frequency_component = np.exp(-1j * scale * np.sqrt(t))
                        wave_value = wave_function * frequency_component
                        transformed[i] = signal_data[i] * wave_value
                
                # Calculate feature magnitude and phase
                magnitude = np.abs(np.sum(transformed))
                phase = np.angle(np.sum(transformed))
                
                # Only keep features above threshold
                if magnitude > self.adaptive_params['magnitude_threshold']:
                    # Determine temporal scale based on data characteristics
                    temporal_scale = self._classify_adaptive_temporal_scale(scale, characteristics)
                    
                    features.append({
                        'scale': float(scale),
                        'shift': float(shift),
                        'magnitude': float(magnitude),
                        'phase': float(phase),
                        'frequency': float(scale / (2 * np.pi)),
                        'temporal_scale': temporal_scale,
                        'data_driven': True
                    })
        
        print(f"📊 Features detected: {len(features)} (data-driven)")
        
        return {
            'all_features': features,
            'n_features': len(features),
            'max_magnitude': max([f['magnitude'] for f in features]) if features else 0,
            'avg_magnitude': np.mean([f['magnitude'] for f in features]) if features else 0,
            'scale_distribution': [f['scale'] for f in features],
            'shift_distribution': [f['shift'] for f in features],
            'temporal_scale_distribution': [f['temporal_scale'] for f in features],
            'signal_characteristics': characteristics,
            'adaptive_parameters': {
                'scales_used': len(optimal_scales),
                'shifts_used': len(optimal_shifts),
                'total_combinations': len(optimal_scales) * len(optimal_shifts),
                'features_kept': len(features)
            }
        }
    
    def _classify_adaptive_temporal_scale(self, scale, characteristics):
        """Classify scale based on data characteristics, not fixed thresholds"""
        
        # Use data-driven classification
        dominant_periods = characteristics['dominant_periods']
        natural_scales = characteristics['natural_scales']
        
        # Find closest natural scale
        if len(natural_scales) > 0:
            closest_natural = natural_scales[np.argmin(np.abs(natural_scales - scale))]
            scale_ratio = scale / closest_natural
            
            if scale_ratio < 0.5:
                return 'very_fast'
            elif scale_ratio < 2.0:
                return 'slow'
            else:
                return 'very_slow'
        else:
            # Fallback to data-relative classification
            avg_period = np.mean(dominant_periods) if len(dominant_periods) > 0 else 100
            
            if scale < avg_period * 0.5:
                return 'very_fast'
            elif scale < avg_period * 2.0:
                return 'slow'
            else:
                return 'very_slow'
    
    def analyze_file(self, csv_file):
        """Analyze a single file with adaptive approach"""
        print(f"\n🔬 ADAPTIVE ANALYSIS: {csv_file}")
        print("=" * 60)
        
        # Load data
        try:
            df = pd.read_csv(csv_file, header=None)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.dropna(axis=1, how='all')
            
            # Find voltage column
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
            
            print(f"📊 Original samples: {len(voltage_data)}")
            
            # Apply adaptive wave transform
            results = self.apply_adaptive_wave_transform(voltage_data)
            
            # Save results
            filename = Path(csv_file).stem
            results_file = self.results_dir / f"adaptive_analysis_{filename}_{self.timestamp}.json"
            
            # Prepare for JSON serialization
            json_results = self._prepare_for_json(results)
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"✅ Analysis complete - saved to {results_file}")
            print(f"📊 Features detected: {results['n_features']}")
            print(f"📊 Max magnitude: {results['max_magnitude']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error analyzing {csv_file}: {e}")
            return None
    
    def _prepare_for_json(self, obj):
        """Prepare object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

def main():
    """Run adaptive analysis on compliant files"""
    print("🎯 ADAPTIVE WAVE TRANSFORM ANALYSIS")
    print("=" * 60)
    print("Letting data speak for itself...")
    
    analyzer = AdaptiveWaveTransform()
    
    # Analyze the compliant files
    compliant_files = [
        "wave_transform_batch_analysis/Spray_in_bag_crop.csv",
        "wave_transform_batch_analysis/Ch1-2_moisture_added.csv"
    ]
    
    all_results = []
    
    for csv_file in compliant_files:
        results = analyzer.analyze_file(csv_file)
        if results:
            all_results.append(results)
    
    # Create comparison report
    if all_results:
        comparison_file = analyzer.results_dir / f"adaptive_comparison_{analyzer.timestamp}.json"
        
        comparison_data = {
            'timestamp': analyzer.timestamp,
            'files_analyzed': len(all_results),
            'results': all_results,
            'summary': {
                'total_features': sum(r['n_features'] for r in all_results),
                'avg_features_per_file': np.mean([r['n_features'] for r in all_results]),
                'feature_ranges': [r['n_features'] for r in all_results],
                'max_magnitudes': [r['max_magnitude'] for r in all_results]
            }
        }
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\n📊 ADAPTIVE ANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Files analyzed: {len(all_results)}")
        print(f"Total features: {sum(r['n_features'] for r in all_results)}")
        print(f"Average features per file: {np.mean([r['n_features'] for r in all_results]):.1f}")
        print(f"Feature ranges: {[r['n_features'] for r in all_results]}")
        print(f"Comparison saved: {comparison_file}")

if __name__ == "__main__":
    main() 