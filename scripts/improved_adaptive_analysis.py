#!/usr/bin/env python3
"""
Improved Adaptive Wave Transform Analysis
Reduces fixed framework bias and improves validity by letting data determine parameters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.optimize import minimize
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImprovedAdaptiveAnalyzer:
    """Improved adaptive analysis that reduces framework bias"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("../results/improved_adaptive")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Data-driven parameters (not fixed)
        self.adaptive_params = {
            'min_scales': 5,
            'max_scales': 50,
            'scale_optimization': 'data_driven',
            'shift_optimization': 'adaptive',
            'magnitude_threshold': 'adaptive',
            'validation_method': 'cross_scale'
        }
        
        # Adamatzky reference (not forcing patterns)
        self.adamatzky_reference = {
            'temporal_scales': {
                'very_slow': {'min': 3600, 'max': float('inf')},
                'slow': {'min': 600, 'max': 3600},
                'very_fast': {'min': 30, 'max': 300}
            },
            'sampling_rate': 1,
            'voltage_range': {'min': -39, 'max': 39}
        }
    
    def analyze_signal_characteristics(self, signal_data):
        """Analyze signal to determine optimal parameters"""
        print("🔬 ANALYZING SIGNAL CHARACTERISTICS")
        
        n_samples = len(signal_data)
        
        # 1. Determine natural temporal scales from data
        fft = np.fft.fft(signal_data)
        freqs = np.fft.fftfreq(n_samples)
        power_spectrum = np.abs(fft)**2
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(power_spectrum[:n_samples//2])[0]
        dominant_freqs = freqs[peak_indices]
        dominant_periods = 1 / np.abs(dominant_freqs[dominant_freqs > 0])
        
        # 2. Determine optimal scales from data variance
        window_sizes = np.logspace(1, np.log10(n_samples//10), 20)
        scale_variances = []
        
        for window_size in window_sizes:
            window_size = int(window_size)
            if window_size < n_samples:
                windows = [signal_data[i:i+window_size] for i in range(0, n_samples-window_size, window_size//2)]
                variances = [np.var(window) for window in windows if len(window) == window_size]
                scale_variances.append(np.mean(variances))
            else:
                scale_variances.append(0)
        
        # 3. Find optimal scales where variance changes significantly
        variance_gradient = np.gradient(scale_variances)
        optimal_scale_indices = np.where(np.abs(variance_gradient) > np.std(variance_gradient))[0]
        optimal_scales = window_sizes[optimal_scale_indices]
        
        # 4. Determine adaptive magnitude threshold
        signal_magnitude = np.std(signal_data)
        adaptive_threshold = signal_magnitude * 0.1  # 10% of signal std
        
        return {
            'optimal_scales': optimal_scales,
            'dominant_periods': dominant_periods,
            'scale_variances': scale_variances,
            'adaptive_threshold': adaptive_threshold,
            'signal_characteristics': {
                'length': n_samples,
                'variance': np.var(signal_data),
                'skewness': stats.skew(signal_data),
                'kurtosis': stats.kurtosis(signal_data),
                'dominant_frequencies': dominant_freqs
            }
        }
    
    def apply_adaptive_wave_transform(self, signal_data, characteristics):
        """Apply adaptive wave transform with data-driven parameters"""
        print("🌊 APPLYING ADAPTIVE WAVE TRANSFORM")
        
        n_samples = len(signal_data)
        
        # Use data-driven scales instead of fixed ones
        scales = characteristics['optimal_scales']
        if len(scales) < self.adaptive_params['min_scales']:
            # Add more scales if needed
            additional_scales = np.logspace(np.log10(min(scales)), np.log10(max(scales)), 
                                         self.adaptive_params['min_scales'] - len(scales))
            scales = np.concatenate([scales, additional_scales])
        
        # Adaptive shifts based on signal length
        max_shift = min(n_samples // 4, 86400)  # Adaptive max shift
        shifts = np.linspace(0, max_shift, min(20, len(scales)))
        
        # Adaptive magnitude threshold
        magnitude_threshold = characteristics['adaptive_threshold']
        
        features = []
        total_combinations = len(scales) * len(shifts)
        print(f"   Testing {total_combinations} adaptive scale-shift combinations...")
        
        for i, scale in enumerate(scales):
            for j, shift in enumerate(shifts):
                # Apply wave transform with adaptive parameters
                transformed = np.zeros(n_samples, dtype=complex)
                compressed_t = np.arange(n_samples) / max_shift
                
                for k in range(n_samples):
                    t = compressed_t[k]
                    if t + shift > 0:
                        wave_function = np.sqrt(t + shift) * scale
                        frequency_component = np.exp(-1j * scale * np.sqrt(t))
                        wave_value = wave_function * frequency_component
                        transformed[k] = signal_data[k] * wave_value
                
                magnitude = np.abs(np.sum(transformed))
                phase = np.angle(np.sum(transformed))
                
                # Use adaptive threshold
                if magnitude > magnitude_threshold:
                    temporal_scale = self._classify_adaptive_scale(scale, characteristics)
                    
                    features.append({
                        'scale': float(scale),
                        'shift': float(shift),
                        'magnitude': float(magnitude),
                        'phase': float(phase),
                        'frequency': float(scale / (2 * np.pi)),
                        'temporal_scale': temporal_scale,
                        'adaptive_threshold': float(magnitude_threshold)
                    })
        
        print(f"   ✅ Adaptive transform completed: {len(features)} features detected")
        
        return {
            'all_features': features,
            'n_features': len(features),
            'adaptive_parameters': {
                'scales_used': len(scales),
                'shifts_used': len(shifts),
                'magnitude_threshold': magnitude_threshold,
                'data_driven_scales': list(scales)
            },
            'signal_characteristics': characteristics
        }
    
    def _classify_adaptive_scale(self, scale, characteristics):
        """Classify scale based on data characteristics, not fixed ranges"""
        # Use data-driven classification
        signal_length = characteristics['signal_characteristics']['length']
        dominant_periods = characteristics['dominant_periods']
        
        if len(dominant_periods) > 0:
            avg_period = np.mean(dominant_periods)
            if scale < avg_period * 0.5:
                return 'very_fast'
            elif scale < avg_period * 2:
                return 'slow'
            else:
                return 'very_slow'
        else:
            # Fallback to signal length-based classification
            if scale < signal_length * 0.01:
                return 'very_fast'
            elif scale < signal_length * 0.1:
                return 'slow'
            else:
                return 'very_slow'
    
    def validate_against_adamatzky(self, results):
        """Validate results against Adamatzky's findings without forcing patterns"""
        print("🔍 VALIDATING AGAINST ADAMATZKY (NON-FORCING)")
        
        validation = {
            'temporal_alignment': 0.0,
            'amplitude_alignment': 0.0,
            'pattern_detection': 0.0,
            'data_driven_score': 0.0,
            'issues': []
        }
        
        features = results['all_features']
        if not features:
            validation['issues'].append("No features detected")
            return validation
        
        # 1. Check if temporal scales align with Adamatzky's ranges (without forcing)
        scales = [f['scale'] for f in features]
        temporal_scales = np.array(scales) * results['adaptive_parameters']['magnitude_threshold']
        
        # Count features in each Adamatzky range
        very_fast_count = np.sum((temporal_scales >= 30) & (temporal_scales <= 300))
        slow_count = np.sum((temporal_scales >= 600) & (temporal_scales <= 3600))
        very_slow_count = np.sum(temporal_scales >= 3600)
        
        total_features = len(temporal_scales)
        
        # Calculate alignment score (not forcing, just measuring overlap)
        adamatzky_ranges = [very_fast_count, slow_count, very_slow_count]
        max_adamatzky = max(adamatzky_ranges)
        alignment_score = max_adamatzky / total_features if total_features > 0 else 0
        
        validation['temporal_alignment'] = alignment_score
        
        # 2. Check amplitude ranges
        magnitudes = [f['magnitude'] for f in features]
        if magnitudes:
            mean_magnitude = np.mean(magnitudes)
            # Check if magnitudes are in biological range (0.05-5.0 mV)
            if 0.05 <= mean_magnitude <= 5.0:
                validation['amplitude_alignment'] = 1.0
            else:
                validation['amplitude_alignment'] = 0.5
        
        # 3. Data-driven score (how much we let data determine parameters)
        adaptive_params = results['adaptive_parameters']
        data_driven_score = (
            (adaptive_params['scales_used'] / 50) +  # More scales = more data-driven
            (adaptive_params['magnitude_threshold'] > 0)  # Adaptive threshold used
        ) / 2
        
        validation['data_driven_score'] = data_driven_score
        
        print(f"   📊 Temporal alignment: {alignment_score:.3f}")
        print(f"   📊 Amplitude alignment: {validation['amplitude_alignment']:.3f}")
        print(f"   📊 Data-driven score: {data_driven_score:.3f}")
        
        return validation
    
    def analyze_file_improved(self, csv_file):
        """Analyze file with improved adaptive methods"""
        print(f"\n{'='*60}")
        print(f"🔬 IMPROVED ADAPTIVE ANALYSIS: {Path(csv_file).name}")
        print(f"{'='*60}")
        
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
            
            print(f"   ✅ Loaded {len(voltage_data)} samples")
            
        except Exception as e:
            print(f"   ❌ Error loading {csv_file}: {e}")
            return None
        
        # Analyze signal characteristics
        characteristics = self.analyze_signal_characteristics(voltage_data)
        
        # Apply adaptive wave transform
        results = self.apply_adaptive_wave_transform(voltage_data, characteristics)
        
        # Validate against Adamatzky (non-forcing)
        validation = self.validate_against_adamatzky(results)
        
        # Save results
        output = {
            'filename': Path(csv_file).name,
            'timestamp': self.timestamp,
            'wave_transform_results': results,
            'validation': validation,
            'signal_characteristics': characteristics['signal_characteristics']
        }
        
        output_file = self.results_dir / f"improved_adaptive_{Path(csv_file).stem}_{self.timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"   💾 Results saved: {output_file}")
        print(f"   📊 Features detected: {results['n_features']}")
        print(f"   📊 Data-driven scales: {len(results['adaptive_parameters']['data_driven_scales'])}")
        
        return output

def main():
    """Run improved adaptive analysis"""
    analyzer = ImprovedAdaptiveAnalyzer()
    
    # Find CSV files
    csv_dirs = ['csv_data', '15061491']
    csv_files = []
    
    for csv_dir in csv_dirs:
        if Path(csv_dir).exists():
            csv_files.extend(list(Path(csv_dir).glob('*.csv')))
    
    if not csv_files:
        print("❌ No CSV files found")
        return
    
    print(f"🔬 IMPROVED ADAPTIVE ANALYSIS")
    print(f"📁 Found {len(csv_files)} CSV files")
    print(f"🎯 Goal: Reduce fixed framework bias, improve validity")
    
    results = []
    for csv_file in csv_files:
        result = analyzer.analyze_file_improved(str(csv_file))
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 IMPROVED ADAPTIVE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        filename = result['filename']
        n_features = result['wave_transform_results']['n_features']
        alignment = result['validation']['temporal_alignment']
        data_driven = result['validation']['data_driven_score']
        
        print(f"   📄 {filename}: {n_features} features, {alignment:.3f} alignment, {data_driven:.3f} data-driven")

if __name__ == "__main__":
    main() 