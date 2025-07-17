#!/usr/bin/env python3
"""
Comprehensive Wave Transform Validation with Adamatzky Parameters
Mathematical validation methods to ensure biological accuracy and prevent false positives
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
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

class ComprehensiveWaveTransformValidator:
    """Comprehensive validation of wave transform with Adamatzky's biological parameters"""
    
    def __init__(self):
        # Adamatzky 2023 validated parameters
        self.adamatzky_params = {
            'temporal_scales': {
                'very_fast': {'min_isi': 30, 'max_isi': 300, 'description': 'Half-minute scale'},
                'slow': {'min_isi': 600, 'max_isi': 3600, 'description': '10-minute scale'},
                'very_slow': {'min_isi': 3600, 'max_isi': float('inf'), 'description': 'Hour scale'}
            },
            'sampling_rate': 1,  # Hz (Adamatzky's rate)
            'min_spike_amplitude': 0.05,  # mV
            'max_spike_amplitude': 5.0,   # mV
            'min_snr': 2.0,
            'baseline_stability': 0.1,    # mV
            'time_compression': 86400     # 1 second = 1 day
        }
        
        # Mathematical validation thresholds
        self.validation_thresholds = {
            'biological_plausibility': 0.7,
            'mathematical_consistency': 0.8,
            'false_positive_rate': 0.1,
            'signal_quality': 0.6
        }
        
    def validate_adamatzky_temporal_scales(self, wave_features):
        """Validate wave transform results against Adamatzky's temporal scales"""
        print("🔬 VALIDATING AGAINST ADAMATZKY TEMPORAL SCALES")
        print("=" * 60)
        
        validation_results = {
            'temporal_alignment': 0.0,
            'scale_distribution': {},
            'biological_plausibility': 0.0,
            'issues': []
        }
        
        # Check if wave scales align with Adamatzky's three families
        scales = wave_features.get('scale_distribution', [])
        if not scales:
            validation_results['issues'].append("No wave scales detected")
            return validation_results
        
        # Convert scales to temporal units (seconds)
        temporal_scales = np.array(scales) * self.adamatzky_params['time_compression']
        
        # Check alignment with Adamatzky's categories
        very_fast_count = np.sum((temporal_scales >= 30) & (temporal_scales <= 300))
        slow_count = np.sum((temporal_scales >= 600) & (temporal_scales <= 3600))
        very_slow_count = np.sum(temporal_scales >= 3600)
        
        total_scales = len(temporal_scales)
        if total_scales > 0:
            very_fast_ratio = very_fast_count / total_scales
            slow_ratio = slow_count / total_scales
            very_slow_ratio = very_slow_count / total_scales
            
            # Adamatzky expects primarily slow and very slow patterns
            expected_slow_ratio = 0.6  # 60% should be slow/very slow
            actual_slow_ratio = slow_ratio + very_slow_ratio
            
            temporal_alignment = min(1.0, actual_slow_ratio / expected_slow_ratio)
            
            validation_results.update({
                'temporal_alignment': temporal_alignment,
                'scale_distribution': {
                    'very_fast': very_fast_ratio,
                    'slow': slow_ratio,
                    'very_slow': very_slow_ratio
                }
            })
            
            print(f"📊 Temporal Scale Distribution:")
            print(f"   Very Fast (30-300s): {very_fast_ratio:.2%}")
            print(f"   Slow (600-3600s): {slow_ratio:.2%}")
            print(f"   Very Slow (>3600s): {very_slow_ratio:.2%}")
            print(f"   Temporal Alignment Score: {temporal_alignment:.3f}")
            
            if temporal_alignment < 0.5:
                validation_results['issues'].append("Poor alignment with Adamatzky temporal scales")
            elif temporal_alignment < 0.7:
                validation_results['issues'].append("Moderate alignment with Adamatzky temporal scales")
            else:
                print("   ✅ Good alignment with Adamatzky temporal scales")
        
        return validation_results
    
    def mathematical_consistency_check(self, wave_features, original_signal):
        """Mathematical consistency validation using rigorous statistical methods"""
        print("\n🧮 MATHEMATICAL CONSISTENCY VALIDATION")
        print("=" * 60)
        
        consistency_results = {
            'energy_conservation': 0.0,
            'orthogonality_check': 0.0,
            'scale_invariance': 0.0,
            'overall_consistency': 0.0,
            'issues': []
        }
        
        # 1. Energy Conservation Check
        original_energy = np.sum(original_signal**2)
        wave_energy = np.sum([f['magnitude']**2 for f in wave_features.get('all_features', [])])
        
        if original_energy > 0:
            energy_ratio = wave_energy / original_energy
            energy_conservation = min(1.0, energy_ratio) if energy_ratio <= 1.0 else 1.0 / energy_ratio
            consistency_results['energy_conservation'] = energy_conservation
            
            print(f"⚡ Energy Conservation: {energy_conservation:.3f}")
            if energy_conservation < 0.8:
                consistency_results['issues'].append("Poor energy conservation")
        
        # 2. Orthogonality Check (wavelet basis functions should be orthogonal)
        features = wave_features.get('all_features', [])
        if len(features) > 1:
            # Calculate correlation between different scale features
            magnitudes = [f['magnitude'] for f in features]
            if len(magnitudes) > 1:
                correlation_matrix = np.corrcoef(magnitudes)
                if correlation_matrix.ndim == 2 and correlation_matrix.shape[0] > 1:
                    # Remove diagonal elements
                    off_diagonal = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
                    avg_correlation = np.mean(np.abs(off_diagonal))
                    orthogonality = 1.0 - avg_correlation
                    consistency_results['orthogonality_check'] = orthogonality
                    
                    print(f"📐 Orthogonality Score: {orthogonality:.3f}")
                    if orthogonality < 0.6:
                        consistency_results['issues'].append("Poor orthogonality in wave basis")
                else:
                    print("   Skipping orthogonality check: not enough features for 2D correlation matrix.")
        else:
            print("   Skipping orthogonality check: not enough features.")
        
        # 3. Scale Invariance Check
        if len(features) > 2:
            scales = [f['scale'] for f in features]
            magnitudes = [f['magnitude'] for f in features]
            
            # Check if magnitude scales properly with scale parameter
            try:
                # Fit power law: magnitude = a * scale^b
                log_scales = np.log(scales)
                log_magnitudes = np.log(magnitudes)
                
                # Linear fit in log space
                slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_magnitudes)
                
                scale_invariance = r_value**2  # R-squared value
                consistency_results['scale_invariance'] = scale_invariance
                
                print(f"📏 Scale Invariance (R²): {scale_invariance:.3f}")
                if scale_invariance < 0.5:
                    consistency_results['issues'].append("Poor scale invariance")
            except:
                consistency_results['issues'].append("Could not compute scale invariance")
        
        # Overall consistency score
        consistency_scores = [v for k, v in consistency_results.items() 
                           if k in ['energy_conservation', 'orthogonality_check', 'scale_invariance']]
        if consistency_scores:
            consistency_results['overall_consistency'] = np.mean(consistency_scores)
            print(f"🎯 Overall Mathematical Consistency: {consistency_results['overall_consistency']:.3f}")
        
        return consistency_results
    
    def false_positive_detection(self, wave_features, original_signal):
        """Detect false positives using multiple validation methods"""
        print("\n🚨 FALSE POSITIVE DETECTION")
        print("=" * 60)
        
        fp_results = {
            'uniformity_test': 0.0,
            'randomness_test': 0.0,
            'biological_plausibility': 0.0,
            'false_positive_score': 0.0,
            'issues': []
        }
        
        # 1. Uniformity Test (false positives often show uniform patterns)
        magnitudes = [f['magnitude'] for f in wave_features.get('all_features', [])]
        if magnitudes:
            # Test for uniform distribution using Kolmogorov-Smirnov test
            try:
                uniform_data = np.random.uniform(min(magnitudes), max(magnitudes), len(magnitudes))
                ks_statistic, p_value = stats.ks_2samp(magnitudes, uniform_data)
                uniformity_score = 1.0 - ks_statistic  # Lower KS = more uniform = worse
                fp_results['uniformity_test'] = uniformity_score
                
                print(f"📊 Uniformity Test: {uniformity_score:.3f}")
                if uniformity_score < 0.3:
                    fp_results['issues'].append("Suspiciously uniform patterns detected")
            except:
                fp_results['issues'].append("Could not perform uniformity test")
        
        # 2. Randomness Test (biological signals should not be random)
        if len(magnitudes) > 10:
            # Test for randomness using autocorrelation
            autocorr = np.correlate(magnitudes, magnitudes, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Calculate randomness score (lower autocorrelation = more random = worse)
            randomness_score = np.mean(np.abs(autocorr[1:10]))  # First 10 lags
            fp_results['randomness_test'] = randomness_score
            
            print(f"🎲 Randomness Test: {randomness_score:.3f}")
            if randomness_score < 0.1:
                fp_results['issues'].append("Suspiciously random patterns detected")
        
        # 3. Biological Plausibility Check
        # Check if patterns align with known fungal electrical characteristics
        biological_score = 0.0
        
        # Check temporal scales
        temporal_alignment = self.validate_adamatzky_temporal_scales(wave_features)
        biological_score += temporal_alignment['temporal_alignment'] * 0.4
        
        # Check amplitude distribution (should be log-normal for biological signals)
        if magnitudes:
            try:
                # Test for log-normal distribution
                log_magnitudes = np.log(magnitudes)
                _, p_value = stats.normaltest(log_magnitudes)
                if p_value > 0.05:  # Normal distribution in log space
                    biological_score += 0.3
                    print("   ✅ Log-normal amplitude distribution (biological)")
                else:
                    print("   ⚠️  Non-log-normal amplitude distribution")
            except:
                pass
        
        # Check for burst patterns (common in fungal activity)
        if len(magnitudes) > 5:
            # Look for clustering in magnitudes (bursts)
            sorted_mags = np.sort(magnitudes)
            gaps = np.diff(sorted_mags)
            if np.std(gaps) < np.mean(gaps) * 0.5:  # Clustered
                biological_score += 0.3
                print("   ✅ Burst-like patterns detected")
        
        fp_results['biological_plausibility'] = biological_score
        print(f"🧬 Biological Plausibility: {biological_score:.3f}")
        
        # Overall false positive score
        fp_scores = [fp_results['uniformity_test'], fp_results['randomness_test'], fp_results['biological_plausibility']]
        fp_results['false_positive_score'] = np.mean(fp_scores)
        
        if fp_results['false_positive_score'] < 0.5:
            fp_results['issues'].append("High risk of false positives")
        
        return fp_results
    
    def signal_quality_assessment(self, wave_features, original_signal):
        """Assess signal quality using SNR and other metrics"""
        print("\n📡 SIGNAL QUALITY ASSESSMENT")
        print("=" * 60)
        
        quality_results = {
            'snr_estimate': 0.0,
            'feature_diversity': 0.0,
            'scale_resolution': 0.0,
            'overall_quality': 0.0,
            'issues': []
        }
        
        # 1. SNR Estimation
        signal_power = np.var(original_signal)
        if len(wave_features.get('all_features', [])) > 0:
            feature_power = np.var([f['magnitude'] for f in wave_features['all_features']])
            if feature_power > 0:
                snr = signal_power / feature_power
                quality_results['snr_estimate'] = min(10.0, snr) / 10.0  # Normalize to [0,1]
                
                print(f"📊 SNR Estimate: {snr:.2f}")
                if snr < 2.0:
                    quality_results['issues'].append("Low signal-to-noise ratio")
        
        # 2. Feature Diversity
        features = wave_features.get('all_features', [])
        if features:
            magnitudes = [f['magnitude'] for f in features]
            diversity = np.std(magnitudes) / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 0
            quality_results['feature_diversity'] = min(1.0, diversity)
            
            print(f"🎨 Feature Diversity: {diversity:.3f}")
            if diversity < 0.1:
                quality_results['issues'].append("Low feature diversity")
        
        # 3. Scale Resolution
        if features:
            scales = [f['scale'] for f in features]
            unique_scales = len(np.unique(scales))
            scale_resolution = min(1.0, unique_scales / 20)  # Normalize to [0,1]
            quality_results['scale_resolution'] = scale_resolution
            
            print(f"📏 Scale Resolution: {unique_scales} unique scales")
            if unique_scales < 5:
                quality_results['issues'].append("Poor scale resolution")
        
        # Overall quality
        quality_scores = [quality_results['snr_estimate'], quality_results['feature_diversity'], quality_results['scale_resolution']]
        quality_results['overall_quality'] = np.mean(quality_scores)
        
        print(f"🎯 Overall Signal Quality: {quality_results['overall_quality']:.3f}")
        
        return quality_results
    
    def comprehensive_validation(self, wave_features, original_signal, filename):
        """Comprehensive validation combining all methods"""
        print(f"\n🔬 COMPREHENSIVE WAVE TRANSFORM VALIDATION")
        print(f"File: {filename}")
        print("=" * 80)
        
        # Run all validation methods
        temporal_validation = self.validate_adamatzky_temporal_scales(wave_features)
        mathematical_validation = self.mathematical_consistency_check(wave_features, original_signal)
        fp_validation = self.false_positive_detection(wave_features, original_signal)
        quality_validation = self.signal_quality_assessment(wave_features, original_signal)
        
        # Combine results
        validation_summary = {
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'temporal_validation': temporal_validation,
            'mathematical_validation': mathematical_validation,
            'false_positive_validation': fp_validation,
            'quality_validation': quality_validation,
            'overall_score': 0.0,
            'recommendation': '',
            'all_issues': []
        }
        
        # Calculate overall score
        scores = [
            temporal_validation['temporal_alignment'],
            mathematical_validation['overall_consistency'],
            fp_validation['false_positive_score'],
            quality_validation['overall_quality']
        ]
        
        validation_summary['overall_score'] = np.mean(scores)
        
        # Collect all issues
        all_issues = []
        for validation in [temporal_validation, mathematical_validation, fp_validation, quality_validation]:
            all_issues.extend(validation.get('issues', []))
        validation_summary['all_issues'] = all_issues
        
        # Generate recommendation
        if validation_summary['overall_score'] >= 0.8:
            validation_summary['recommendation'] = 'EXCELLENT - High confidence in biological accuracy'
        elif validation_summary['overall_score'] >= 0.6:
            validation_summary['recommendation'] = 'GOOD - Moderate confidence, minor issues'
        elif validation_summary['overall_score'] >= 0.4:
            validation_summary['recommendation'] = 'CAUTION - Significant issues detected'
        else:
            validation_summary['recommendation'] = 'REJECT - High risk of false positives'
        
        # Print summary
        print(f"\n📊 VALIDATION SUMMARY")
        print(f"Overall Score: {validation_summary['overall_score']:.3f}")
        print(f"Recommendation: {validation_summary['recommendation']}")
        
        if all_issues:
            print(f"\n⚠️  Issues Detected:")
            for issue in all_issues:
                print(f"   • {issue}")
        
        return validation_summary

def main():
    """Main validation function"""
    validator = ComprehensiveWaveTransformValidator()
    
    # Example usage (will be called by batch processor)
    print("🔬 Comprehensive Wave Transform Validator")
    print("Ready for batch processing with Adamatzky parameters")
    print("=" * 60)

if __name__ == "__main__":
    main() 