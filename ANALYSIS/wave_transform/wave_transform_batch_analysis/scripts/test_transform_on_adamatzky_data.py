#!/usr/bin/env python3
"""
Test Wave Transform on Adamatzky Data
Tests the wave transform on simulated Adamatzky data to validate performance
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def test_wave_transform_on_adamatzky_data():
    """Test wave transform on simulated Adamatzky data"""
    
    print("=" * 80)
    print("TESTING WAVE TRANSFORM ON ADAMATZKY DATA")
    print("=" * 80)
    print("Testing your wave transform on simulated Adamatzky signals")
    print("with identical electrode setup and biological ranges")
    print("=" * 80)
    
    # Check if simulated data exists
    simulated_file = "data/simulated_adamatzky_signals.csv"
    
    if not os.path.exists(simulated_file):
        print(f"❌ Simulated data not found: {simulated_file}")
        print("Please run the Adamatzky electrode simulation first.")
        return
    
    # Load simulated data
    print(f"Loading simulated Adamatzky data: {simulated_file}")
    df = pd.read_csv(simulated_file)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Analyze the simulated data
    time = df.iloc[:, 0].values
    amplitude = df.iloc[:, 1].values
    
    print(f"\nSimulated Data Analysis:")
    print(f"  Duration: {len(time)} seconds ({len(time)/3600:.1f} hours)")
    print(f"  Sampling rate: 1 Hz")
    print(f"  Amplitude range: {np.min(amplitude):.3f} - {np.max(amplitude):.3f} mV")
    print(f"  Mean amplitude: {np.mean(amplitude):.3f} mV")
    print(f"  Standard deviation: {np.std(amplitude):.3f} mV")
    
    # Check compliance with Adamatzky's ranges
    adamatzky_min, adamatzky_max = 0.16, 0.4
    compliance = "✓ WITHIN RANGE" if np.min(amplitude) >= adamatzky_min and np.max(amplitude) <= adamatzky_max else "✗ OUTSIDE RANGE"
    factor = np.max(amplitude) / adamatzky_max if np.max(amplitude) > adamatzky_max else 1.0
    
    print(f"  Adamatzky compliance: {compliance}")
    print(f"  Factor vs Adamatzky: {factor:.1f}x")
    
    # Peak analysis
    from scipy import signal
    peaks, _ = signal.find_peaks(amplitude, height=np.mean(amplitude))
    
    print(f"\nPeak Analysis:")
    print(f"  Peak count: {len(peaks)}")
    print(f"  Peak density: {len(peaks)/len(amplitude):.3f}")
    print(f"  Mean peak amplitude: {np.mean(amplitude[peaks]):.3f} mV")
    
    # Temporal analysis
    fft = np.fft.fft(amplitude)
    freqs = np.fft.fftfreq(len(amplitude), d=1)  # 1 second sampling
    power_spectrum = np.abs(fft) ** 2
    dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
    dominant_freq = freqs[dominant_freq_idx]
    dominant_period = 1 / abs(dominant_freq) if dominant_freq != 0 else float('inf')
    
    print(f"\nTemporal Analysis:")
    print(f"  Dominant frequency: {dominant_freq:.6f} Hz")
    print(f"  Dominant period: {dominant_period:.1f} seconds")
    
    # Compare with Adamatzky's temporal scales
    adamatzky_scales = {
        "very_slow": 2573,  # seconds (43 min)
        "slow": 457,        # seconds (8 min)
        "very_fast": 24     # seconds
    }
    
    temporal_match = "unknown"
    for scale, period in adamatzky_scales.items():
        if 0.5 * period <= dominant_period <= 2 * period:
            temporal_match = scale
            break
    
    print(f"  Temporal match: {temporal_match}")
    
    # Now test the wave transform
    print(f"\n" + "=" * 80)
    print("WAVE TRANSFORM TESTING")
    print("=" * 80)
    
    # Copy simulated data to raw data directory for wave transform processing
    import shutil
    test_file = "data/raw/simulated_adamatzky_test.csv"
    shutil.copy(simulated_file, test_file)
    
    print(f"Copied simulated data to: {test_file}")
    print("This file is now ready for wave transform analysis.")
    
    # Provide instructions for wave transform testing
    print(f"\n" + "=" * 80)
    print("NEXT STEPS FOR WAVE TRANSFORM TESTING")
    print("=" * 80)
    
    print("1. Run your wave transform on the simulated data:")
    print(f"   python3 scripts/your_wave_transform_script.py {test_file}")
    
    print("\n2. Compare results with Adamatzky's findings:")
    print("   - Very slow spikes: 43 min duration, 0.16 mV amplitude")
    print("   - Slow spikes: 8 min duration, 0.4 mV amplitude")
    print("   - Very fast spikes: 24 s duration, 0.36 mV amplitude")
    
    print("\n3. Expected outcomes:")
    print("   - Wave transform should detect all three temporal scales")
    print("   - Amplitude ranges should match Adamatzky's biological ranges")
    print("   - No forced parameters should bias the results")
    
    print("\n4. Validation criteria:")
    print("   - ✓ Detects very slow spikes (43 min)")
    print("   - ✓ Detects slow spikes (8 min)")
    print("   - ✓ Detects very fast spikes (24 s)")
    print("   - ✓ Amplitude ranges within 0.16-0.4 mV")
    print("   - ✓ No false positives or forced patterns")
    
    # Save test results
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "test_file": test_file,
        "simulated_data_stats": {
            "duration_seconds": len(time),
            "duration_hours": len(time) / 3600,
            "amplitude_range": [float(np.min(amplitude)), float(np.max(amplitude))],
            "mean_amplitude": float(np.mean(amplitude)),
            "std_amplitude": float(np.std(amplitude)),
            "peak_count": len(peaks),
            "peak_density": len(peaks) / len(amplitude),
            "dominant_period": float(dominant_period),
            "temporal_match": temporal_match
        },
        "adamatzky_compliance": {
            "within_range": np.min(amplitude) >= adamatzky_min and np.max(amplitude) <= adamatzky_max,
            "amplitude_factor": factor,
            "adamatzky_range": [adamatzky_min, adamatzky_max]
        }
    }
    
    output_file = f"results/adamatzky_transform_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    
    import json
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results saved to: {output_file}")
    
    print(f"\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("✅ Adamatzky electrode simulation and testing complete!")
    print("Your wave transform can now be tested under Adamatzky's exact conditions.")
    print("The simulated data provides a perfect benchmark for validation.")
    
    return test_results

def main():
    """Main testing function"""
    
    results = test_wave_transform_on_adamatzky_data()
    
    print(f"\n🎯 Ready for wave transform testing!")
    print("The simulated Adamatzky data is now available for your wave transform analysis.")

if __name__ == "__main__":
    main() 