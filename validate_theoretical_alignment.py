#!/usr/bin/env python3
"""
Validate Theoretical Alignment: Compare Results with Fungal Electrical Activity Theory
"""

import json
import os
import numpy as np

def load_results(filename):
    """Load results from JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def validate_theoretical_alignment():
    """Validate results against theoretical expectations"""
    print("🧬 THEORETICAL VALIDATION: Fungal Electrical Activity")
    print("=" * 60)
    
    # Load ultra-optimized results
    ultra_results = load_results("ultra_optimized_fungal_results_20250716_145728.json")
    
    # Theoretical expectations for fungal electrical activity
    theoretical_expectations = {
        'spike_rate': {
            'range': (0.1, 2.0),  # Hz - fungal spike rates
            'description': 'Fungal spike rates typically 0.1-2.0 Hz'
        },
        'amplitude': {
            'range': (0.05, 5.0),  # mV - fungal spike amplitudes
            'description': 'Fungal spike amplitudes typically 0.05-5.0 mV'
        },
        'isi': {
            'range': (0.5, 10.0),  # seconds - inter-spike intervals
            'description': 'Fungal ISI typically 0.5-10.0 seconds'
        },
        'snr': {
            'range': (1.0, 10.0),  # signal-to-noise ratio
            'description': 'Good fungal recordings have SNR > 1.0'
        },
        'quality_score': {
            'range': (0.7, 1.0),  # quality score
            'description': 'High-quality recordings have score > 0.7'
        }
    }
    
    print("\n📊 THEORETICAL EXPECTATIONS:")
    print("-" * 40)
    for metric, expectation in theoretical_expectations.items():
        print(f"   {metric.upper()}: {expectation['range'][0]}-{expectation['range'][1]} {expectation['description']}")
    
    if ultra_results:
        stats = ultra_results.get('stats', {})
        
        print(f"\n🔬 ULTRA-OPTIMIZED RESULTS VALIDATION:")
        print("-" * 40)
        
        # Validate each metric
        validations = {}
        
        # Spike Rate Validation
        spike_rate = stats.get('spike_rate', 0)
        spike_rate_valid = theoretical_expectations['spike_rate']['range'][0] <= spike_rate <= theoretical_expectations['spike_rate']['range'][1]
        validations['spike_rate'] = {
            'value': spike_rate,
            'expected': theoretical_expectations['spike_rate']['range'],
            'valid': spike_rate_valid,
            'description': 'Spike Rate (Hz)'
        }
        
        # Amplitude Validation
        mean_amplitude = abs(stats.get('mean_amplitude', 0))
        amplitude_valid = theoretical_expectations['amplitude']['range'][0] <= mean_amplitude <= theoretical_expectations['amplitude']['range'][1]
        validations['amplitude'] = {
            'value': mean_amplitude,
            'expected': theoretical_expectations['amplitude']['range'],
            'valid': amplitude_valid,
            'description': 'Mean Amplitude (mV)'
        }
        
        # ISI Validation
        mean_isi = stats.get('mean_isi', 0)
        if mean_isi > 0:  # Only validate positive ISI
            isi_valid = theoretical_expectations['isi']['range'][0] <= mean_isi <= theoretical_expectations['isi']['range'][1]
        else:
            isi_valid = False
        validations['isi'] = {
            'value': mean_isi,
            'expected': theoretical_expectations['isi']['range'],
            'valid': isi_valid,
            'description': 'Mean ISI (seconds)'
        }
        
        # SNR Validation
        snr = stats.get('snr', 0)
        snr_valid = theoretical_expectations['snr']['range'][0] <= snr <= theoretical_expectations['snr']['range'][1]
        validations['snr'] = {
            'value': snr,
            'expected': theoretical_expectations['snr']['range'],
            'valid': snr_valid,
            'description': 'Signal-to-Noise Ratio'
        }
        
        # Quality Score Validation
        quality_score = stats.get('quality_score', 0)
        quality_valid = theoretical_expectations['quality_score']['range'][0] <= quality_score <= theoretical_expectations['quality_score']['range'][1]
        validations['quality_score'] = {
            'value': quality_score,
            'expected': theoretical_expectations['quality_score']['range'],
            'valid': quality_valid,
            'description': 'Quality Score'
        }
        
        # Display validation results
        for metric, validation in validations.items():
            status = "✅ PASS" if validation['valid'] else "❌ FAIL"
            print(f"   {validation['description']}: {validation['value']:.4f} | Expected: {validation['expected'][0]}-{validation['expected'][1]} | {status}")
        
        # Calculate overall validation score
        passed_metrics = sum(1 for v in validations.values() if v['valid'])
        total_metrics = len(validations)
        validation_score = (passed_metrics / total_metrics) * 100
        
        print(f"\n🎯 OVERALL VALIDATION:")
        print(f"   Passed: {passed_metrics}/{total_metrics} metrics")
        print(f"   Validation Score: {validation_score:.1f}%")
        
        if validation_score >= 80:
            print(f"   ✅ EXCELLENT: Results align well with theoretical expectations")
        elif validation_score >= 60:
            print(f"   ⚠️  GOOD: Results mostly align with theoretical expectations")
        else:
            print(f"   ❌ POOR: Results don't align well with theoretical expectations")
        
        # Analyze spike timing patterns
        if 'spikes' in ultra_results:
            spike_timings = [spike['time_seconds'] for spike in ultra_results['spikes']]
            spike_timings.sort()
            
            print(f"\n📈 SPIKE TIMING PATTERN ANALYSIS:")
            print(f"   Total spikes: {len(spike_timings)}")
            print(f"   Recording duration: {spike_timings[-1] - spike_timings[0]:.1f} seconds")
            print(f"   Average interval: {np.mean(np.diff(spike_timings)):.2f} seconds")
            
            # Check for burst patterns
            intervals = np.diff(spike_timings)
            short_intervals = intervals[intervals < 1.0]  # Bursts have intervals < 1s
            if len(short_intervals) > 0:
                print(f"   Burst patterns: {len(short_intervals)} short intervals detected")
            else:
                print(f"   Burst patterns: None detected (all intervals > 1s)")
            
            # Check for regularity
            interval_std = np.std(intervals)
            interval_cv = interval_std / np.mean(intervals) if np.mean(intervals) > 0 else 0
            print(f"   Interval variability (CV): {interval_cv:.2f}")
            
            if interval_cv < 0.5:
                print(f"   ✅ Regular spiking pattern detected")
            elif interval_cv < 1.0:
                print(f"   ⚠️  Moderately irregular spiking pattern")
            else:
                print(f"   ❌ Highly irregular spiking pattern")
        
        print(f"\n🧬 BIOLOGICAL INTERPRETATION:")
        print(f"   • Spike rate {spike_rate:.3f} Hz: {'✅ Normal fungal range' if spike_rate_valid else '❌ Outside normal range'}")
        print(f"   • Amplitude {mean_amplitude:.4f} mV: {'✅ Normal fungal range' if amplitude_valid else '❌ Outside normal range'}")
        print(f"   • ISI {mean_isi:.2f} s: {'✅ Normal fungal range' if isi_valid else '❌ Outside normal range'}")
        print(f"   • SNR {snr:.2f}: {'✅ Good signal quality' if snr_valid else '❌ Poor signal quality'}")
        print(f"   • Quality {quality_score:.2f}: {'✅ High quality recording' if quality_valid else '❌ Low quality recording'}")
        
        if validation_score >= 80:
            print(f"\n✅ CONCLUSION: Results strongly support genuine fungal electrical activity")
        elif validation_score >= 60:
            print(f"\n⚠️  CONCLUSION: Results suggest fungal electrical activity with some concerns")
        else:
            print(f"\n❌ CONCLUSION: Results may not represent genuine fungal electrical activity")

if __name__ == "__main__":
    validate_theoretical_alignment() 