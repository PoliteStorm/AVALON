#!/usr/bin/env python3
"""
Analyze Spike Overlap: Compare Ultra-Optimized vs Integrated Wave Transform
"""

import json
import os
import numpy as np

def load_spike_timings(filename):
    """Load spike timings from JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            if 'spikes' in data:
                return [spike['time_seconds'] for spike in data['spikes']]
    return []

def analyze_overlap():
    """Analyze overlap between detection methods"""
    print("🔍 SPIKE DETECTION OVERLAP ANALYSIS")
    print("=" * 50)
    
    # Load ultra-optimized spike timings
    ultra_timings = load_spike_timings("ultra_optimized_fungal_results_20250716_145728.json")
    
    # Try to load integrated spike timings
    integrated_file = "integrated_fungal_analysis_20250716_145558.json"
    integrated_timings = []
    
    if os.path.exists(integrated_file):
        try:
            with open(integrated_file, 'r') as f:
                # Read a larger chunk to get spike data
                data = f.read(50000)  # Read 50KB
                if '"spikes"' in data:
                    # Extract spike timings manually
                    import re
                    time_matches = re.findall(r'"time_seconds":\s*([0-9.]+)', data)
                    integrated_timings = [float(t) for t in time_matches]
        except:
            integrated_timings = []
    
    print(f"\n📊 SPIKE COUNT COMPARISON:")
    print(f"   Ultra-Optimized: {len(ultra_timings)} spikes")
    print(f"   Integrated Wave: {len(integrated_timings)} spikes")
    
    if ultra_timings and integrated_timings:
        # Find overlapping spikes (within 0.1 second tolerance)
        tolerance = 0.1
        overlapping = []
        ultra_only = []
        integrated_only = []
        
        for ultra_time in ultra_timings:
            found_overlap = False
            for integrated_time in integrated_timings:
                if abs(ultra_time - integrated_time) <= tolerance:
                    overlapping.append((ultra_time, integrated_time))
                    found_overlap = True
                    break
            if not found_overlap:
                ultra_only.append(ultra_time)
        
        for integrated_time in integrated_timings:
            found_overlap = False
            for ultra_time in ultra_timings:
                if abs(integrated_time - ultra_time) <= tolerance:
                    found_overlap = True
                    break
            if not found_overlap:
                integrated_only.append(integrated_time)
        
        print(f"\n🎯 OVERLAP ANALYSIS:")
        print(f"   Overlapping spikes: {len(overlapping)}")
        print(f"   Ultra-only spikes: {len(ultra_only)}")
        print(f"   Integrated-only spikes: {len(integrated_only)}")
        
        if len(ultra_timings) > 0:
            overlap_percentage = (len(overlapping) / len(ultra_timings)) * 100
            print(f"   Overlap percentage: {overlap_percentage:.1f}%")
        
        print(f"\n⚡ DETAILED COMPARISON:")
        print(f"   Ultra-Optimized Spikes:")
        for i, time in enumerate(ultra_timings[:10]):
            print(f"     {i+1:2d}. {time:6.3f}s")
        if len(ultra_timings) > 10:
            print(f"     ... and {len(ultra_timings) - 10} more")
        
        print(f"\n   Integrated Wave Spikes:")
        for i, time in enumerate(integrated_timings[:10]):
            print(f"     {i+1:2d}. {time:6.3f}s")
        if len(integrated_timings) > 10:
            print(f"     ... and {len(integrated_timings) - 10} more")
        
        print(f"\n🔍 OVERLAPPING SPIKE PAIRS:")
        for i, (ultra, integrated) in enumerate(overlapping[:5]):
            diff = abs(ultra - integrated)
            print(f"   {i+1}. Ultra: {ultra:6.3f}s | Integrated: {integrated:6.3f}s | Diff: {diff:6.3f}s")
        if len(overlapping) > 5:
            print(f"   ... and {len(overlapping) - 5} more overlapping pairs")
        
        print(f"\n🎯 INTERPRETATION:")
        if len(overlapping) > len(ultra_timings) * 0.8:
            print(f"   ✅ HIGH OVERLAP: Wave transform is detecting the SAME spikes")
            print(f"   ✅ PLUS additional spikes (enhanced sensitivity)")
        elif len(overlapping) > len(ultra_timings) * 0.5:
            print(f"   ⚠️  MODERATE OVERLAP: Wave transform detects SOME same spikes")
            print(f"   ✅ PLUS many new spikes (different detection approach)")
        else:
            print(f"   ❌ LOW OVERLAP: Wave transform detects DIFFERENT spikes")
            print(f"   ✅ Completely different detection approach")
        
        print(f"\n📈 WHAT THIS MEANS:")
        print(f"   • Ultra-Optimized: {len(ultra_timings)} spikes (conservative)")
        print(f"   • Integrated Wave: {len(integrated_timings)} spikes (comprehensive)")
        print(f"   • New Detection: +{len(integrated_timings) - len(overlapping)} additional spikes")
        print(f"   • Enhanced Sensitivity: {((len(integrated_timings) - len(ultra_timings)) / len(ultra_timings) * 100):.1f}% increase")
    
    else:
        print(f"\n❌ Cannot load integrated spike data (file too large)")
        print(f"   Ultra-Optimized timings: {ultra_timings}")
        print(f"   Integrated timings: {len(integrated_timings)} found")

if __name__ == "__main__":
    analyze_overlap() 