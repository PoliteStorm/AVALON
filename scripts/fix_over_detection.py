#!/usr/bin/env python3
"""
Fix Over-Detection Issues in Fungal Electrical Analysis
Implements biological validation and data type detection
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def detect_data_type(file_path, data):
    """Detect if data is electrical, coordinate, or moisture"""
    
    # Check file name patterns
    file_lower = file_path.lower()
    if 'coordinate' in file_lower:
        return 'coordinate'
    if 'moisture' in file_lower:
        return 'moisture'
    if 'electrical' in file_lower or 'voltage' in file_lower:
        return 'electrical'
    
    # Check data characteristics
    if isinstance(data, pd.DataFrame):
        voltage_cols = [col for col in data.columns if any(keyword in col.lower() 
                        for keyword in ['voltage', 'v', 'mv', 'electrical'])]
        coord_cols = [col for col in data.columns if any(keyword in col.lower() 
                       for keyword in ['x', 'y', 'coord', 'position'])]
        
        if voltage_cols:
            voltage_data = data[voltage_cols[0]]
        elif len(data.columns) == 1:
            voltage_data = data.iloc[:, 0]
        else:
            return 'unknown'
    else:
        voltage_data = data
    
    voltage_range = np.max(voltage_data) - np.min(voltage_data)
    voltage_std = np.std(voltage_data)
    
    # Electrical data characteristics
    if voltage_range > 0.1 and voltage_std > 0.05:
        return 'electrical'
    
    # Coordinate data characteristics (usually smaller ranges)
    if voltage_range < 10:
        return 'coordinate'
    
    return 'unknown'

def validate_spike_rate(spike_count, total_samples, species='unknown'):
    """Validate spike rate against biological constraints"""
    
    # Adamatzky's biological limits
    max_spike_rates = {
        'pleurotus_ostreatus': 0.05,  # 5% max for oyster mushrooms
        'lentinula_edodes': 0.03,     # 3% max for shiitake
        'ganoderma_lucidum': 0.02,    # 2% max for reishi
        'unknown': 0.02               # 2% max for unknown species
    }
    
    spike_rate = spike_count / total_samples
    max_allowed = max_spike_rates.get(species, 0.02)
    
    if spike_rate > max_allowed:
        return False, f"Spike rate {spike_rate:.3f} exceeds biological limit {max_allowed:.3f}"
    
    return True, f"Spike rate {spike_rate:.3f} within biological limits"

def calculate_biological_confidence(results):
    """Calculate confidence score based on biological validity"""
    
    score = 0.0
    max_score = 1.0
    
    # Spike rate validation (30% of score)
    spike_rate = results.get('spike_rate', 0)
    if spike_rate <= 0.05:  # Within biological limits
        score += 0.3
    elif spike_rate <= 0.1:  # Borderline
        score += 0.15
    # 0 points for excessive rates
    
    # Amplitude validation (25% of score)
    mean_amplitude = results.get('mean_amplitude', 0)
    if 0.05 <= mean_amplitude <= 2.0:
        score += 0.25
    elif 0.02 <= mean_amplitude <= 5.0:
        score += 0.125
    
    # Interval validation (25% of score)
    mean_isi = results.get('mean_isi', 0)
    if mean_isi >= 50:  # Realistic intervals
        score += 0.25
    elif mean_isi >= 10:
        score += 0.125
    
    # Signal quality (20% of score)
    snr = results.get('snr', 0)
    if snr > 1.5:
        score += 0.2
    elif snr > 1.0:
        score += 0.1
    
    return min(score, max_score)

def fix_batch_results(input_file, output_file):
    """Fix over-detection issues in batch results"""
    
    print(f"Loading batch results from {input_file}...")
    
    with open(input_file, 'r') as f:
        results = json.load(f)
    
    fixed_results = []
    rejected_results = []
    
    for result in results:
        file_path = result.get('file', '')
        spike_rate = result.get('spike_rate', 0)
        n_spikes = result.get('n_spikes', 0)
        mean_amplitude = result.get('mean_amplitude', 0)
        mean_isi = result.get('mean_isi', 0)
        
        # 1. Detect data type
        data_type = detect_data_type(file_path, None)
        
        # 2. Validate spike rate
        is_valid_rate, rate_message = validate_spike_rate(n_spikes, 1000)  # Assuming 1000 samples
        
        # 3. Calculate biological confidence
        biological_confidence = calculate_biological_confidence(result)
        
        # 4. Apply filters
        should_reject = False
        rejection_reasons = []
        
        # Reject coordinate data from electrical analysis
        if data_type == 'coordinate':
            should_reject = True
            rejection_reasons.append("Coordinate data processed as electrical")
        
        # Reject excessive spike rates
        if spike_rate > 0.1:  # More than 10% spike rate
            should_reject = True
            rejection_reasons.append(f"Excessive spike rate: {spike_rate:.3f}")
        
        # Reject biologically impossible intervals
        if mean_isi < 10:  # Less than 10ms intervals
            should_reject = True
            rejection_reasons.append(f"Impossible spike intervals: {mean_isi:.1f}ms")
        
        # Reject low confidence results
        if biological_confidence < 0.3:
            should_reject = True
            rejection_reasons.append(f"Low biological confidence: {biological_confidence:.2f}")
        
        # Add validation info to result
        result['data_type'] = data_type
        result['biological_confidence'] = biological_confidence
        result['is_valid_rate'] = is_valid_rate
        result['rate_validation_message'] = rate_message
        
        if should_reject:
            result['rejected'] = True
            result['rejection_reasons'] = rejection_reasons
            rejected_results.append(result)
        else:
            result['rejected'] = False
            fixed_results.append(result)
    
    # Save fixed results
    with open(output_file, 'w') as f:
        json.dump(fixed_results, f, indent=2)
    
    # Save rejected results
    rejected_file = output_file.replace('.json', '_rejected.json')
    with open(rejected_file, 'w') as f:
        json.dump(rejected_results, f, indent=2)
    
    print(f"Fixed results saved to {output_file}")
    print(f"Rejected results saved to {rejected_file}")
    print(f"Original results: {len(results)}")
    print(f"Valid results: {len(fixed_results)}")
    print(f"Rejected results: {len(rejected_results)}")
    
    return fixed_results, rejected_results

def main():
    """Main function to fix over-detection issues"""
    
    print("=== Fixing Over-Detection Issues ===")
    
    # Fix batch results
    input_file = 'results/batch_fungal_csv_analysis_summary.json'
    output_file = 'results/fixed_batch_results.json'
    
    if os.path.exists(input_file):
        fixed_results, rejected_results = fix_batch_results(input_file, output_file)
        
        print("\n=== Summary ===")
        print(f"Total files processed: {len(fixed_results) + len(rejected_results)}")
        print(f"Valid results: {len(fixed_results)}")
        print(f"Rejected results: {len(rejected_results)}")
        
        if rejected_results:
            print("\n=== Rejection Reasons ===")
            reasons = {}
            for result in rejected_results:
                for reason in result.get('rejection_reasons', []):
                    reasons[reason] = reasons.get(reason, 0) + 1
            
            for reason, count in reasons.items():
                print(f"  {reason}: {count} files")
    else:
        print(f"Input file {input_file} not found")

if __name__ == "__main__":
    main() 