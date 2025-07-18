#!/usr/bin/env python3
"""
Fast Adamatzky Validation Test
Quick test of √t transform alignment with Adamatzky's published findings.
FOCUS: Electrical activity only - no coordinate data analysis.
"""

import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add fungal analysis project to path
sys.path.insert(0, 'fungal_analysis_project/src')
from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def test_fast_alignment():
    """Quick test of alignment with Adamatzky's findings."""
    
    print("="*60)
    print("⚡ FAST ADAMATZKY ELECTRICAL VALIDATION ⚡")
    print("="*60)
    print("Testing √t transform against Adamatzky's electrical findings")
    print("FOCUS: Electrical activity only")
    print("="*60)
    print()
    
    # Initialize analyzer with voltage data only
    analyzer = RigorousFungalAnalyzer(None, "15061491/fungal_spikes/good_recordings")
    
    # Load data
    print("📁 Loading fungal electrical data...")
    data = analyzer.load_and_categorize_data()
    
    print(f"✅ Loaded {len(data['voltage_data'])} voltage files")
    print("FOCUS: Electrical activity only - no coordinate data")
    print()
    
    # Adamatzky's published characteristics
    adamatzky_characteristics = {
        'Pv': {'frequency_range': (0.1, 10), 'time_scale': (1, 100)},
        'Pi': {'frequency_range': (0.01, 1), 'time_scale': (10, 1000)},
        'Pp': {'frequency_range': (0.5, 20), 'time_scale': (0.1, 10)},
        'Rb': {'frequency_range': (0.001, 0.1), 'time_scale': (100, 10000)}
    }
    
    # Test each species
    results = {}
    
    for species in ['Pv', 'Pi', 'Pp', 'Rb']:
        print(f"🔍 Testing {species}...")
        
        # Get species-specific voltage files
        species_files = [f for f in data['voltage_data'].keys() if f.startswith(species)]
        
        if not species_files:
            print(f"   ❌ No {species} voltage files found")
            continue
        
        # Test with first file for speed
        filename = species_files[0]
        file_data = data['voltage_data'][filename]
        
        # Extract voltage signal
        df = file_data['data']
        if len(df.columns) >= 1:
            voltage_signal = df.iloc[:, 0].values
            
            # Quick electrical analysis
            features = quick_electrical_analysis(voltage_signal)
            
            if features:
                # Check alignment
                alignment = check_alignment(features, adamatzky_characteristics[species])
                results[species] = {
                    'alignment': alignment,
                    'n_features': len(features),
                    'avg_freq': np.mean([f['frequency'] for f in features]),
                    'avg_time': np.mean([f['time_scale'] for f in features])
                }
                print(f"   ✅ {species}: {len(features)} features, alignment: {alignment:.2f}")
            else:
                print(f"   ❌ {species}: No electrical features detected")
    
    # Summary
    print("\n" + "="*60)
    print("📊 FAST VALIDATION RESULTS")
    print("="*60)
    
    if results:
        overall_score = np.mean([r['alignment'] for r in results.values()])
        print(f"Overall alignment score: {overall_score:.2f}")
        
        if overall_score > 0.7:
            print("✅ GOOD ALIGNMENT with Adamatzky's electrical findings")
        elif overall_score > 0.4:
            print("⚠️  MODERATE ALIGNMENT - needs refinement")
        else:
            print("❌ POOR ALIGNMENT - potential false positives")
        
        print("\nDetailed results:")
        for species, result in results.items():
            status = "✅" if result['alignment'] > 0.5 else "❌"
            print(f"  {species}: {result['alignment']:.2f} {status}")
    else:
        print("❌ No electrical features detected in any species")
    
    return results

def quick_electrical_analysis(voltage_signal):
    """Quick electrical signal analysis."""
    features = []
    
    if len(voltage_signal) > 50:
        # Basic electrical characteristics
        mean_voltage = np.mean(voltage_signal)
        std_voltage = np.std(voltage_signal)
        
        # Simple spike detection
        threshold = mean_voltage + 1.5 * std_voltage
        spikes = voltage_signal > threshold
        
        if np.any(spikes):
            spike_indices = np.where(spikes)[0]
            if len(spike_indices) > 1:
                # Calculate basic electrical parameters
                isi = np.diff(spike_indices)
                mean_isi = np.mean(isi)
                frequency = 1.0 / mean_isi if mean_isi > 0 else 0
                
                features.append({
                    'frequency': frequency,
                    'time_scale': mean_isi,
                    'amplitude': np.mean(voltage_signal[spikes])
                })
    
    return features

def check_alignment(features, expected):
    """Check alignment with Adamatzky's characteristics."""
    if not features:
        return 0.0
    
    avg_freq = np.mean([f['frequency'] for f in features])
    avg_time = np.mean([f['time_scale'] for f in features])
    
    # Check if within expected ranges
    freq_ok = expected['frequency_range'][0] <= avg_freq <= expected['frequency_range'][1]
    time_ok = expected['time_scale'][0] <= avg_time <= expected['time_scale'][1]
    
    return (freq_ok + time_ok) / 2

if __name__ == "__main__":
    test_fast_alignment() 