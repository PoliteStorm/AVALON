#!/usr/bin/env python3
"""
Cross-Validation Analysis for Wave Transform
Improves validity by testing multiple parameter combinations and measuring consistency
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.model_selection import KFold
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class CrossValidationAnalyzer:
    """Cross-validation analysis to improve wave transform validity"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("../results/cross_validation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cross-validation parameters
        self.cv_params = {
            'n_folds': 5,
            'n_parameter_combinations': 10,
            'consistency_threshold': 0.7,
            'robustness_threshold': 0.6
        }
        
        # Parameter ranges to test
        self.parameter_ranges = {
            'scale_ranges': [
                np.linspace(30, 300, 5),      # Very fast
                np.linspace(600, 3600, 8),    # Slow
                np.linspace(3600, 86400, 5)   # Very slow
            ],
            'shift_ranges': [
                np.linspace(0, 1000, 5),
                np.linspace(0, 10000, 8),
                np.linspace(0, 86400, 5)
            ],
            'magnitude_thresholds': [0.01, 0.05, 0.1, 0.5, 1.0]
        }
    
    def cross_validate_wave_transform(self, signal_data, filename):
        """Perform cross-validation on wave transform parameters"""
        print(f"🔬 CROSS-VALIDATION ANALYSIS: {filename}")
        print("=" * 60)
        
        n_samples = len(signal_data)
        kf = KFold(n_splits=self.cv_params['n_folds'], shuffle=True, random_state=42)
        
        cv_results = {
            'parameter_combinations': [],
            'fold_results': [],
            'consistency_scores': [],
            'robustness_scores': [],
            'optimal_parameters': None
        }
        
        # Test different parameter combinations
        for i in range(self.cv_params['n_parameter_combinations']):
            # Randomly select parameters
            scale_range_idx = np.random.randint(0, len(self.parameter_ranges['scale_ranges']))
            shift_range_idx = np.random.randint(0, len(self.parameter_ranges['shift_ranges']))
            threshold_idx = np.random.randint(0, len(self.parameter_ranges['magnitude_thresholds']))
            
            scales = self.parameter_ranges['scale_ranges'][scale_range_idx]
            shifts = self.parameter_ranges['shift_ranges'][shift_range_idx]
            magnitude_threshold = self.parameter_ranges['magnitude_thresholds'][threshold_idx]
            
            parameter_combo = {
                'scales': scales,
                'shifts': shifts,
                'magnitude_threshold': magnitude_threshold,
                'combo_id': i
            }
            
            fold_features = []
            
            # Cross-validate with k-fold
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(signal_data)):
                train_data = signal_data[train_idx]
                test_data = signal_data[test_idx]
                
                # Apply wave transform to training data
                train_features = self._apply_wave_transform_fold(
                    train_data, scales, shifts, magnitude_threshold
                )
                
                # Apply to test data
                test_features = self._apply_wave_transform_fold(
                    test_data, scales, shifts, magnitude_threshold
                )
                
                fold_result = {
                    'fold': fold_idx,
                    'train_features': len(train_features),
                    'test_features': len(test_features),
                    'train_magnitudes': [f['magnitude'] for f in train_features],
                    'test_magnitudes': [f['magnitude'] for f in test_features],
                    'train_scales': [f['scale'] for f in train_features],
                    'test_scales': [f['scale'] for f in test_features]
                }
                
                fold_features.append(fold_result)
            
            # Calculate consistency across folds
            consistency_score = self._calculate_consistency(fold_features)
            robustness_score = self._calculate_robustness(fold_features)
            
            cv_results['parameter_combinations'].append(parameter_combo)
            cv_results['fold_results'].append(fold_features)
            cv_results['consistency_scores'].append(consistency_score)
            cv_results['robustness_scores'].append(robustness_score)
            
            print(f"   📊 Combo {i}: Consistency={consistency_score:.3f}, Robustness={robustness_score:.3f}")
        
        # Find optimal parameters
        best_combo_idx = np.argmax(cv_results['consistency_scores'])
        cv_results['optimal_parameters'] = cv_results['parameter_combinations'][best_combo_idx]
        
        print(f"   🎯 Optimal parameters: Combo {best_combo_idx}")
        print(f"   📊 Best consistency: {cv_results['consistency_scores'][best_combo_idx]:.3f}")
        
        return cv_results
    
    def _apply_wave_transform_fold(self, signal_data, scales, shifts, magnitude_threshold):
        """Apply wave transform to a fold of data"""
        n_samples = len(signal_data)
        features = []
        
        for scale in scales:
            for shift in shifts:
                transformed = np.zeros(n_samples, dtype=complex)
                compressed_t = np.arange(n_samples) / 3000  # Time compression
                
                for i in range(n_samples):
                    t = compressed_t[i]
                    if t + shift > 0:
                        wave_function = np.sqrt(t + shift) * scale
                        frequency_component = np.exp(-1j * scale * np.sqrt(t))
                        wave_value = wave_function * frequency_component
                        transformed[i] = signal_data[i] * wave_value
                
                magnitude = np.abs(np.sum(transformed))
                phase = np.angle(np.sum(transformed))
                
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
        
        return features
    
    def _calculate_consistency(self, fold_features):
        """Calculate consistency across folds"""
        if not fold_features:
            return 0.0
        
        # Compare feature counts across folds
        feature_counts = [fold['train_features'] for fold in fold_features]
        mean_count = np.mean(feature_counts)
        std_count = np.std(feature_counts)
        
        if mean_count == 0:
            return 0.0
        
        # Consistency based on coefficient of variation
        cv = std_count / mean_count
        consistency = max(0, 1 - cv)  # Higher consistency = lower CV
        
        return consistency
    
    def _calculate_robustness(self, fold_features):
        """Calculate robustness across folds"""
        if not fold_features:
            return 0.0
        
        # Compare magnitude distributions across folds
        all_magnitudes = []
        for fold in fold_features:
            all_magnitudes.extend(fold['train_magnitudes'])
        
        if not all_magnitudes:
            return 0.0
        
        # Robustness based on magnitude stability
        magnitudes = np.array(all_magnitudes)
        mean_magnitude = np.mean(magnitudes)
        std_magnitude = np.std(magnitudes)
        
        if mean_magnitude == 0:
            return 0.0
        
        # Robustness based on relative standard deviation
        relative_std = std_magnitude / mean_magnitude
        robustness = max(0, 1 - relative_std)
        
        return robustness
    
    def _classify_temporal_scale(self, scale):
        """Classify temporal scale"""
        if scale <= 300:
            return 'very_fast'
        elif scale <= 3600:
            return 'slow'
        else:
            return 'very_slow'
    
    def validate_with_adamatzky(self, cv_results):
        """Validate cross-validation results against Adamatzky's findings"""
        print("🔍 VALIDATING CROSS-VALIDATION RESULTS")
        
        validation = {
            'parameter_stability': 0.0,
            'biological_alignment': 0.0,
            'reproducibility': 0.0,
            'overall_validity': 0.0
        }
        
        # 1. Parameter stability
        consistency_scores = cv_results['consistency_scores']
        validation['parameter_stability'] = np.mean(consistency_scores)
        
        # 2. Biological alignment (using optimal parameters)
        optimal_params = cv_results['optimal_parameters']
        if optimal_params:
            scales = optimal_params['scales']
            temporal_scales = scales * 3000  # Convert to seconds
            
            # Check alignment with Adamatzky's ranges
            very_fast_count = np.sum((temporal_scales >= 30) & (temporal_scales <= 300))
            slow_count = np.sum((temporal_scales >= 600) & (temporal_scales <= 3600))
            very_slow_count = np.sum(temporal_scales >= 3600)
            
            total_scales = len(temporal_scales)
            if total_scales > 0:
                max_alignment = max(very_fast_count, slow_count, very_slow_count)
                validation['biological_alignment'] = max_alignment / total_scales
        
        # 3. Reproducibility (robustness across folds)
        robustness_scores = cv_results['robustness_scores']
        validation['reproducibility'] = np.mean(robustness_scores)
        
        # 4. Overall validity
        validation['overall_validity'] = (
            validation['parameter_stability'] * 0.4 +
            validation['biological_alignment'] * 0.4 +
            validation['reproducibility'] * 0.2
        )
        
        print(f"   📊 Parameter stability: {validation['parameter_stability']:.3f}")
        print(f"   📊 Biological alignment: {validation['biological_alignment']:.3f}")
        print(f"   📊 Reproducibility: {validation['reproducibility']:.3f}")
        print(f"   📊 Overall validity: {validation['overall_validity']:.3f}")
        
        return validation
    
    def analyze_file_with_cv(self, csv_file):
        """Analyze file with cross-validation"""
        print(f"\n{'='*60}")
        print(f"🔬 CROSS-VALIDATION ANALYSIS: {Path(csv_file).name}")
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
        
        # Perform cross-validation
        cv_results = self.cross_validate_wave_transform(voltage_data, Path(csv_file).name)
        
        # Validate against Adamatzky
        validation = self.validate_with_adamatzky(cv_results)
        
        # Save results
        output = {
            'filename': Path(csv_file).name,
            'timestamp': self.timestamp,
            'cv_results': cv_results,
            'validation': validation,
            'signal_info': {
                'length': len(voltage_data),
                'variance': np.var(voltage_data),
                'mean': np.mean(voltage_data)
            }
        }
        
        output_file = self.results_dir / f"cv_analysis_{Path(csv_file).stem}_{self.timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"   💾 Results saved: {output_file}")
        
        return output

def main():
    """Run cross-validation analysis"""
    analyzer = CrossValidationAnalyzer()
    
    # Find CSV files
    csv_dirs = ['csv_data', '15061491']
    csv_files = []
    
    for csv_dir in csv_dirs:
        if Path(csv_dir).exists():
            csv_files.extend(list(Path(csv_dir).glob('*.csv')))
    
    if not csv_files:
        print("❌ No CSV files found")
        return
    
    print(f"🔬 CROSS-VALIDATION ANALYSIS")
    print(f"📁 Found {len(csv_files)} CSV files")
    print(f"🎯 Goal: Improve validity through parameter testing and consistency measurement")
    
    results = []
    for csv_file in csv_files:
        result = analyzer.analyze_file_with_cv(str(csv_file))
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for result in results:
        filename = result['filename']
        validity = result['validation']['overall_validity']
        stability = result['validation']['parameter_stability']
        alignment = result['validation']['biological_alignment']
        
        print(f"   📄 {filename}: Validity={validity:.3f}, Stability={stability:.3f}, Alignment={alignment:.3f}")

if __name__ == "__main__":
    main() 