#!/usr/bin/env python3
"""
Thorough Results Validation Script
Validates all analysis results for completeness and correctness
"""

import json
import os
import numpy as np
from datetime import datetime

def validate_json_structure(data, filename):
    """Validate JSON structure and required fields"""
    print(f"\n🔍 VALIDATING: {filename}")
    print("=" * 60)
    
    # Check top-level structure
    required_top_level = ['filename', 'spike_results', 'transform_results', 'synthesis_results']
    missing_top = [key for key in required_top_level if key not in data]
    
    if missing_top:
        print(f"❌ MISSING TOP-LEVEL KEYS: {missing_top}")
        return False
    else:
        print("✅ All top-level keys present")
    
    # Validate spike results
    spike_results = data['spike_results']
    required_spike_keys = ['spike_times', 'spike_amplitudes', 'n_spikes', 'mean_amplitude', 
                          'spike_rate_hz', 'mean_isi']
    missing_spike = [key for key in required_spike_keys if key not in spike_results]
    
    if missing_spike:
        print(f"❌ MISSING SPIKE KEYS: {missing_spike}")
        return False
    else:
        print("✅ All spike result keys present")
    
    # Validate transform results
    transform_results = data['transform_results']
    required_transform_keys = ['all_features', 'n_features', 'n_spike_aligned', 
                             'spike_alignment_ratio']
    missing_transform = [key for key in required_transform_keys if key not in transform_results]
    
    if missing_transform:
        print(f"❌ MISSING TRANSFORM KEYS: {missing_transform}")
        return False
    else:
        print("✅ All transform result keys present")
    
    # Validate synthesis results
    synthesis_results = data['synthesis_results']
    required_synthesis_keys = ['biological_activity_score', 'method_agreement', 
                             'recommended_analysis', 'confidence_level']
    missing_synthesis = [key for key in required_synthesis_keys if key not in synthesis_results]
    
    if missing_synthesis:
        print(f"❌ MISSING SYNTHESIS KEYS: {missing_synthesis}")
        return False
    else:
        print("✅ All synthesis result keys present")
    
    return True

def validate_data_consistency(data, filename):
    """Validate data consistency and logical relationships"""
    print(f"\n📊 DATA CONSISTENCY CHECK: {filename}")
    print("-" * 40)
    
    # Check spike data consistency
    spike_results = data['spike_results']
    n_spikes = spike_results['n_spikes']
    
    # Parse spike times and amplitudes
    try:
        spike_times_str = spike_results['spike_times']
        spike_amplitudes_str = spike_results['spike_amplitudes']
        
        # Convert string representations to arrays
        spike_times_str = spike_times_str.replace('[', '').replace(']', '').strip()
        spike_amplitudes_str = spike_amplitudes_str.replace('[', '').replace(']', '').strip()
        
        spike_times = np.array([float(x) for x in spike_times_str.split() if x.strip()])
        spike_amplitudes = np.array([float(x) for x in spike_amplitudes_str.split() if x.strip()])
        
        # Check consistency
        if len(spike_times) != n_spikes:
            print(f"❌ SPIKE COUNT MISMATCH: Expected {n_spikes}, got {len(spike_times)}")
            return False
        else:
            print(f"✅ Spike count consistent: {n_spikes}")
        
        if len(spike_amplitudes) != n_spikes:
            print(f"❌ AMPLITUDE COUNT MISMATCH: Expected {n_spikes}, got {len(spike_amplitudes)}")
            return False
        else:
            print(f"✅ Amplitude count consistent: {n_spikes}")
        
        # Check amplitude statistics
        calculated_mean = np.mean(spike_amplitudes)
        reported_mean = spike_results['mean_amplitude']
        if abs(calculated_mean - reported_mean) > 0.001:
            print(f"❌ MEAN AMPLITUDE MISMATCH: Calculated {calculated_mean:.3f}, reported {reported_mean:.3f}")
            return False
        else:
            print(f"✅ Mean amplitude consistent: {reported_mean:.3f}")
        
    except Exception as e:
        print(f"❌ SPIKE DATA PARSING ERROR: {e}")
        return False
    
    # Check transform data consistency
    transform_results = data['transform_results']
    n_features = transform_results['n_features']
    all_features = transform_results['all_features']
    
    if len(all_features) != n_features:
        print(f"❌ FEATURE COUNT MISMATCH: Expected {n_features}, got {len(all_features)}")
        return False
    else:
        print(f"✅ Feature count consistent: {n_features}")
    
    # Check alignment ratio
    n_aligned = transform_results['n_spike_aligned']
    alignment_ratio = transform_results['spike_alignment_ratio']
    calculated_ratio = n_aligned / n_features if n_features > 0 else 0
    
    if abs(calculated_ratio - alignment_ratio) > 0.001:
        print(f"❌ ALIGNMENT RATIO MISMATCH: Calculated {calculated_ratio:.3f}, reported {alignment_ratio:.3f}")
        return False
    else:
        print(f"✅ Alignment ratio consistent: {alignment_ratio:.3f}")
    
    return True

def validate_biological_plausibility(data, filename):
    """Validate that results are biologically plausible"""
    print(f"\n🧬 BIOLOGICAL PLAUSIBILITY CHECK: {filename}")
    print("-" * 40)
    
    spike_results = data['spike_results']
    synthesis_results = data['synthesis_results']
    
    # Check spike rate (should be 0.1-10 Hz for fungi)
    spike_rate = spike_results['spike_rate_hz']
    if spike_rate < 0.01 or spike_rate > 10:
        print(f"⚠️  UNUSUAL SPIKE RATE: {spike_rate:.3f} Hz (expected 0.01-10 Hz)")
    else:
        print(f"✅ Spike rate plausible: {spike_rate:.3f} Hz")
    
    # Check mean amplitude (should be 0.1-10 mV for fungi)
    mean_amplitude = spike_results['mean_amplitude']
    if mean_amplitude < 0.01 or mean_amplitude > 10:
        print(f"⚠️  UNUSUAL AMPLITUDE: {mean_amplitude:.3f} mV (expected 0.01-10 mV)")
    else:
        print(f"✅ Mean amplitude plausible: {mean_amplitude:.3f} mV")
    
    # Check activity score (should be 0-1)
    activity_score = synthesis_results['biological_activity_score']
    if activity_score < 0 or activity_score > 1:
        print(f"❌ INVALID ACTIVITY SCORE: {activity_score:.3f} (should be 0-1)")
        return False
    else:
        print(f"✅ Activity score valid: {activity_score:.3f}")
    
    # Check confidence level
    confidence = synthesis_results['confidence_level']
    valid_confidences = ['low', 'medium', 'high']
    if confidence not in valid_confidences:
        print(f"❌ INVALID CONFIDENCE LEVEL: {confidence} (should be {valid_confidences})")
        return False
    else:
        print(f"✅ Confidence level valid: {confidence}")
    
    return True

def main():
    """Main validation function"""
    print("🔍 THOROUGH RESULTS VALIDATION")
    print("=" * 60)
    
    results_dir = "results/integrated_analysis_results"
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    print(f"📁 Found {len(files)} analysis files")
    
    if not files:
        print("❌ No JSON files found in results directory")
        return
    
    validation_results = {}
    
    for filename in files:
        filepath = os.path.join(results_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Run all validations
            structure_valid = validate_json_structure(data, filename)
            consistency_valid = validate_data_consistency(data, filename)
            biological_valid = validate_biological_plausibility(data, filename)
            
            validation_results[filename] = {
                'structure': structure_valid,
                'consistency': consistency_valid,
                'biological': biological_valid,
                'overall': structure_valid and consistency_valid and biological_valid
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON PARSE ERROR in {filename}: {e}")
            validation_results[filename] = {
                'structure': False,
                'consistency': False,
                'biological': False,
                'overall': False
            }
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR in {filename}: {e}")
            validation_results[filename] = {
                'structure': False,
                'consistency': False,
                'biological': False,
                'overall': False
            }
    
    # Summary
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    all_valid = True
    for filename, results in validation_results.items():
        status = "✅ PASS" if results['overall'] else "❌ FAIL"
        print(f"{status} {filename}")
        
        if not results['overall']:
            all_valid = False
            if not results['structure']:
                print(f"    - Structure validation failed")
            if not results['consistency']:
                print(f"    - Data consistency failed")
            if not results['biological']:
                print(f"    - Biological plausibility failed")
    
    if all_valid:
        print(f"\n🎉 ALL RESULTS VALIDATED SUCCESSFULLY!")
        print(f"✅ {len(files)} files passed all checks")
    else:
        print(f"\n⚠️  SOME RESULTS HAVE ISSUES")
        print(f"❌ {sum(1 for r in validation_results.values() if not r['overall'])} files failed validation")

if __name__ == "__main__":
    main() 