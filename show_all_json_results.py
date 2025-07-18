#!/usr/bin/env python3
"""
Display all JSON analysis results
"""

import json
import os

def main():
    results_dir = "results/integrated_analysis_results"
    files = [f for f in os.listdir(results_dir) if f.endswith('.json') and 'integrated_analysis' in f]
    
    print("📊 ALL JSON ANALYSIS RESULTS:")
    print("=" * 60)
    
    for filename in files:
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"\n📁 {filename}:")
        print(f"  Spikes: {data['spike_results']['n_spikes']}, Rate: {data['spike_results']['spike_rate_hz']:.3f} Hz")
        print(f"  Features: {data['transform_results']['n_features']}, Alignment: {data['transform_results']['spike_alignment_ratio']:.3f}")
        print(f"  Activity Score: {data['synthesis_results']['biological_activity_score']:.3f}, Confidence: {data['synthesis_results']['confidence_level']}")

if __name__ == "__main__":
    main() 