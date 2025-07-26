#!/usr/bin/env python3
"""
Continuous Adamatzky-Corrected Fungal Electrical Testing
Provides constant new data from ongoing tests with biological accuracy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import time
from datetime import datetime
import warnings
from scipy import signal
from pathlib import Path

warnings.filterwarnings('ignore')

class ContinuousAdamatzkyTester:
    """Continuous testing with Adamatzky-corrected parameters"""
    
    def __init__(self):
        self.results_dir = Path("continuous_testing_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Adamatzky-corrected parameters
        self.config = {
            "sampling_rate": 1,  # Hz
            "min_isi": 30,  # seconds
            "max_isi": 3600,  # seconds  
            "spike_duration": 30,  # seconds
            "min_amplitude": 0.05,  # mV
            "max_amplitude": 5.0,  # mV
            "min_snr": 2.0,
            "time_compression": 86400,  # 1 second = 1 day
            "baseline_threshold": 0.1,
            "adaptive_threshold": True
        }
        
        # Spike categories from Adamatzky 2023
        self.categories = {
            "very_fast": {"min_duration": 30, "max_duration": 60, "description": "Half-minute scale"},
            "slow": {"min_duration": 600, "max_duration": 3600, "description": "10-minute scale"}, 
            "very_slow": {"min_duration": 3600, "max_duration": float('inf'), "description": "Hour-scale"}
        }
        
        self.test_count = 0
        self.session_start = datetime.now()
        
    def load_and_compress_data(self, csv_file):
        """Load CSV and apply time compression"""
        try:
            df = pd.read_csv(csv_file, header=None)
            
            # Try different columns to find voltage data
            voltage_data = None
            for col in range(min(4, len(df.columns))):
                col_data = df.iloc[:, col].values
                if np.std(col_data) > 0.01:  # Look for column with variation
                    voltage_data = col_data
                    break
            
            if voltage_data is None:
                voltage_data = df.iloc[:, 1].values  # Fallback to second column
            
            # Apply time compression
            original_samples = len(voltage_data)
            compressed_samples = original_samples // self.config["time_compression"]
            
            if compressed_samples < 10:
                # If too compressed, use a smaller factor
                compression_factor = max(1, original_samples // 100)
                compressed_data = voltage_data[::compression_factor]
            else:
                compressed_data = voltage_data[::self.config["time_compression"]]
            
            return compressed_data, original_samples, len(compressed_data)
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            return None, 0, 0
    
    def detect_spikes_adamatzky(self, voltage_data):
        """Adamatzky-corrected spike detection"""
        spikes = []
        
        # Calculate baseline using robust median
        baseline = np.median(voltage_data)
        baseline_std = np.std(voltage_data)
        
        # Adaptive threshold
        if self.config["adaptive_threshold"]:
            threshold = baseline + (self.config["baseline_threshold"] * baseline_std)
        else:
            threshold = baseline + self.config["baseline_threshold"]
        
        # Find peaks above threshold
        peaks, _ = signal.find_peaks(voltage_data, height=threshold, distance=self.config["min_isi"])
        
        for peak in peaks:
            amplitude = voltage_data[peak] - baseline
            
            # Check amplitude bounds
            if self.config["min_amplitude"] <= amplitude <= self.config["max_amplitude"]:
                # Calculate SNR
                noise_level = baseline_std
                snr = amplitude / noise_level if noise_level > 0 else 0
                
                if snr >= self.config["min_snr"]:
                    spike_info = {
                        "time_index": peak,
                        "time_seconds": peak,
                        "amplitude_mv": amplitude,
                        "threshold": threshold,
                        "baseline": baseline,
                        "snr": snr
                    }
                    spikes.append(spike_info)
        
        return spikes
    
    def classify_spikes(self, spikes):
        """Classify spikes into Adamatzky categories"""
        classified = {"very_fast": [], "slow": [], "very_slow": []}
        
        if len(spikes) < 2:
            return classified
        
        # Calculate ISIs
        times = [s["time_seconds"] for s in spikes]
        isis = np.diff(times)
        
        for i, spike in enumerate(spikes):
            if i == 0:
                isi = isis[0] if len(isis) > 0 else 0
            else:
                isi = isis[i-1]
            
            # Classify based on ISI
            if 30 <= isi <= 300:
                classified["very_fast"].append(spike)
            elif 600 <= isi <= 3600:
                classified["slow"].append(spike)
            elif isi >= 3600:
                classified["very_slow"].append(spike)
        
        return classified
    
    def calculate_biological_metrics(self, spikes, classified_spikes):
        """Calculate biologically relevant metrics"""
        if not spikes:
            return {
                "n_spikes": 0,
                "mean_amplitude": 0.0,
                "std_amplitude": 0.0,
                "mean_isi": 0.0,
                "std_isi": 0.0,
                "spike_rate": 0.0,
                "mean_snr": 0.0,
                "classified_counts": {k: len(v) for k, v in classified_spikes.items()},
                "biological_plausibility": self.assess_biological_plausibility(spikes, classified_spikes)
            }
        
        amplitudes = [s["amplitude_mv"] for s in spikes]
        snrs = [s["snr"] for s in spikes]
        
        # Calculate ISIs
        times = [s["time_seconds"] for s in spikes]
        isis = np.diff(times) if len(times) > 1 else [0]
        
        metrics = {
            "n_spikes": len(spikes),
            "mean_amplitude": np.mean(amplitudes),
            "std_amplitude": np.std(amplitudes),
            "mean_isi": np.mean(isis),
            "std_isi": np.std(isis),
            "spike_rate": len(spikes) / max(times[-1] - times[0], 1),
            "mean_snr": np.mean(snrs),
            "classified_counts": {k: len(v) for k, v in classified_spikes.items()},
            "biological_plausibility": self.assess_biological_plausibility(spikes, classified_spikes)
        }
        
        return metrics
    
    def assess_biological_plausibility(self, spikes, classified_spikes):
        """Assess if results align with Adamatzky's findings"""
        if not spikes:
            return {"score": 0, "issues": ["No spikes detected"]}
        
        issues = []
        score = 100
        
        # Check temporal scale alignment
        if len(spikes) > 0:
            times = [s["time_seconds"] for s in spikes]
            total_duration = times[-1] - times[0]
            
            if total_duration < 3600:  # Less than 1 hour
                issues.append("Recording duration too short for fungal activity")
                score -= 20
            
            if len(spikes) > 10:  # Too many spikes for fungal activity
                issues.append("Too many spikes - likely noise artifacts")
                score -= 30
        
        # Check amplitude distribution
        amplitudes = [s["amplitude_mv"] for s in spikes]
        if amplitudes:
            if np.std(amplitudes) < 0.01:  # Too uniform
                issues.append("Suspiciously uniform amplitudes")
                score -= 15
        
        # Check category distribution
        total_classified = sum(len(v) for v in classified_spikes.values())
        if total_classified > 0:
            if classified_spikes["very_fast"] and not classified_spikes["slow"]:
                issues.append("Only fast spikes detected - may be noise")
                score -= 10
        
        return {"score": max(0, score), "issues": issues}
    
    def run_single_test(self, csv_file):
        """Run a single test on a CSV file"""
        print(f"\n🔬 Testing: {csv_file}")
        
        # Load and compress data
        compressed_data, original_samples, compressed_samples = self.load_and_compress_data(csv_file)
        if compressed_data is None:
            return None
        
        # Detect spikes
        spikes = self.detect_spikes_adamatzky(compressed_data)
        
        # Classify spikes
        classified_spikes = self.classify_spikes(spikes)
        
        # Calculate metrics
        metrics = self.calculate_biological_metrics(spikes, classified_spikes)
        
        # Create result
        result = {
            "test_id": f"test_{self.test_count:04d}",
            "timestamp": datetime.now().isoformat(),
            "file": csv_file,
            "spikes": spikes,
            "classified_spikes": classified_spikes,
            "metrics": metrics,
            "config": self.config,
            "compression_info": {
                "original_samples": original_samples,
                "compressed_samples": compressed_samples,
                "compression_factor": self.config["time_compression"]
            }
        }
        
        self.test_count += 1
        return result
    
    def save_result(self, result):
        """Save test result to JSON"""
        if result is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"continuous_test_{timestamp}_{self.test_count:04d}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"💾 Saved: {filepath}")
        return filepath
    
    def print_live_results(self, result):
        """Print live results for constant data feed"""
        if result is None:
            return
        
        print(f"\n📊 LIVE RESULTS - Test #{self.test_count}")
        print(f"File: {result['file']}")
        print(f"Spikes detected: {result['metrics']['n_spikes']}")
        print(f"Mean amplitude: {result['metrics']['mean_amplitude']:.4f} mV")
        print(f"Mean ISI: {result['metrics']['mean_isi']:.1f} seconds")
        print(f"Spike rate: {result['metrics']['spike_rate']:.4f} Hz")
        print(f"Mean SNR: {result['metrics']['mean_snr']:.2f}")
        
        # Classification summary
        classified = result['metrics']['classified_counts']
        print(f"Classification: Very Fast={classified.get('very_fast', 0)}, "
              f"Slow={classified.get('slow', 0)}, Very Slow={classified.get('very_slow', 0)}")
        
        # Biological plausibility
        plausibility = result['metrics']['biological_plausibility']
        print(f"Biological Score: {plausibility['score']}/100")
        if plausibility['issues']:
            print(f"Issues: {', '.join(plausibility['issues'])}")
        
        print("-" * 60)
    
    def run_continuous_tests(self, csv_files, interval_seconds=30):
        """Run continuous tests on multiple files"""
        print(f"🚀 Starting continuous Adamatzky-corrected testing")
        print(f"Session started: {self.session_start}")
        print(f"Testing interval: {interval_seconds} seconds")
        print(f"Files to test: {len(csv_files)}")
        print("=" * 60)
        
        while True:
            for csv_file in csv_files:
                try:
                    # Run test
                    result = self.run_single_test(csv_file)
                    
                    # Save result
                    saved_file = self.save_result(result)
                    
                    # Print live results
                    self.print_live_results(result)
                    
                    # Wait before next test
                    time.sleep(interval_seconds)
                    
                except KeyboardInterrupt:
                    print("\n⏹️  Testing stopped by user")
                    return
                except Exception as e:
                    print(f"❌ Error in test: {e}")
                    continue

def main():
    """Main function for continuous testing"""
    tester = ContinuousAdamatzkyTester()
    
    # Find CSV files to test
    csv_files = [
        "data/Norm_vs_deep_tip_crop.csv",
        "data/Ch1-2_1second_sampling.csv", 
        "data/New_Oyster_with spray_as_mV_seconds_SigView.csv"
    ]
    
    # Filter to existing files
    existing_files = [f for f in csv_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No CSV files found to test")
        return
    
    print(f"📁 Found {len(existing_files)} files to test")
    
    # Run continuous tests
    tester.run_continuous_tests(existing_files, interval_seconds=15)

if __name__ == "__main__":
    main() 