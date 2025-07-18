#!/usr/bin/env python3
"""
Live Results Monitor for Adamatzky-Corrected Testing
Shows constant new data from ongoing tests
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
import glob

def get_latest_results():
    """Get the latest test results"""
    results_dir = Path("continuous_testing_results")
    if not results_dir.exists():
        return []
    
    json_files = list(results_dir.glob("continuous_test_*.json"))
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    results = []
    for file in json_files[:5]:  # Latest 5 results
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                data['filename'] = file.name
                results.append(data)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    return results

def print_live_summary():
    """Print live summary of all results"""
    results = get_latest_results()
    
    if not results:
        print("🔄 Waiting for test results...")
        return
    
    print(f"\n📊 LIVE ADAMATZKY-CORRECTED TESTING SUMMARY")
    print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Summary statistics
    total_tests = len(results)
    files_tested = set(r['file'] for r in results)
    total_spikes = sum(r['metrics'].get('n_spikes', 0) for r in results)
    
    print(f"📈 Total Tests: {total_tests}")
    print(f"📁 Files Tested: {len(files_tested)}")
    print(f"⚡ Total Spikes Detected: {total_spikes}")
    print(f"🔬 Latest Results:")
    
    # Show latest 3 results
    for i, result in enumerate(results[:3]):
        print(f"\n--- Test #{i+1} ---")
        print(f"File: {result['file']}")
        print(f"Spikes: {result['metrics'].get('n_spikes', 0)}")
        print(f"Mean Amplitude: {result['metrics'].get('mean_amplitude', 0):.4f} mV")
        print(f"Mean ISI: {result['metrics'].get('mean_isi', 0):.1f} seconds")
        print(f"Spike Rate: {result['metrics'].get('spike_rate', 0):.4f} Hz")
        print(f"Mean SNR: {result['metrics'].get('mean_snr', 0):.2f}")
        
        # Classification
        classified = result['metrics'].get('classified_counts', {})
        print(f"Classification: VF={classified.get('very_fast', 0)}, "
              f"S={classified.get('slow', 0)}, VS={classified.get('very_slow', 0)}")
        
        # Biological plausibility
        plausibility = result['metrics'].get('biological_plausibility', {})
        score = plausibility.get('score', 0)
        issues = plausibility.get('issues', [])
        print(f"Biological Score: {score}/100")
        if issues:
            print(f"Issues: {', '.join(issues)}")
    
    print("\n" + "=" * 80)

def monitor_continuously(interval_seconds=10):
    """Monitor results continuously"""
    print("🔬 Starting Live Adamatzky-Corrected Results Monitor")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    try:
        while True:
            print_live_summary()
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped")

if __name__ == "__main__":
    monitor_continuously(interval_seconds=10) 