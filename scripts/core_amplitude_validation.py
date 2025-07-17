#!/usr/bin/env python3
"""
Core Amplitude Validation System
Uses Adamatzky's methods with improved quality testing and no forced parameters
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy import signal

class CoreAmplitudeValidator:
    """Core amplitude validation using Adamatzky's methods"""
    
    def __init__(self):
        # Adamatzky's validated parameters
        self.adamatzky_params = {
            "amplitude_ranges": {
                "very_slow_spikes": {"min": 0.16, "max": 0.16, "unit": "mV"},
                "slow_spikes": {"min": 0.4, "max": 0.4, "unit": "mV"},
                "very_fast_spikes": {"min": 0.36, "max": 0.36, "unit": "mV"}
            },
            "temporal_scales": {
                "very_slow": {"min": 2573, "max": 2573, "unit": "seconds"},
                "slow": {"min": 457, "max": 457, "unit": "seconds"},
                "very_fast": {"min": 24, "max": 24, "unit": "seconds"}
            },
            "electrode_setup": {
                "type": "Iridium-coated stainless steel sub-dermal needle electrodes",
                "voltage_range": 78,  # mV
                "sampling_rate": 1,  # Hz
                "distance": 10  # mm between electrodes
            }
        }
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "validation_tests": {},
            "amplitude_analysis": {},
            "quality_metrics": {},
            "adamatzky_compliance": {}
        }
    
    def analyze_amplitude_quality(self, data_file):
        """Analyze amplitude quality using Adamatzky's methods"""
        
        print(f"Analyzing amplitude quality for: {data_file}")
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Get amplitude column (assuming it's the second column)
        if len(df.columns) >= 2:
            amplitude_col = df.columns[1]
            amplitudes = df[amplitude_col].values
        else:
            print(f"Warning: Could not find amplitude column in {data_file}")
            return None
        
        # Remove NaN values
        amplitudes = amplitudes[~np.isnan(amplitudes)]
        
        if len(amplitudes) == 0:
            print(f"Warning: No valid amplitude data in {data_file}")
            return None
        
        # Basic statistics
        stats = {
            "min": np.min(amplitudes),
            "max": np.max(amplitudes),
            "mean": np.mean(amplitudes),
            "median": np.median(amplitudes),
            "std": np.std(amplitudes),
            "range": np.max(amplitudes) - np.min(amplitudes)
        }
        
        # Adamatzky compliance check
        compliance = self.check_adamatzky_compliance(stats)
        
        # Signal quality metrics
        quality_metrics = self.calculate_signal_quality(amplitudes)
        
        # Cross-validation of amplitude patterns
        cv_results = self.cross_validate_amplitude_patterns(amplitudes)
        
        # Transform analysis without forced parameters
        transform_analysis = self.analyze_transform_output(amplitudes)
        
        return {
            "file": data_file,
            "statistics": stats,
            "adamatzky_compliance": compliance,
            "quality_metrics": quality_metrics,
            "cross_validation": cv_results,
            "transform_analysis": transform_analysis
        }
    
    def check_adamatzky_compliance(self, stats):
        """Check compliance with Adamatzky's biological ranges"""
        
        compliance = {
            "within_biological_range": False,
            "amplitude_factor": 0,
            "compliance_score": 0,
            "recommendations": []
        }
        
        # Check if within Adamatzky's range (0.16-0.4 mV)
        adamatzky_min = 0.16
        adamatzky_max = 0.4
        
        if stats["min"] >= adamatzky_min and stats["max"] <= adamatzky_max:
            compliance["within_biological_range"] = True
            compliance["compliance_score"] = 100
        else:
            # Calculate how far outside the range
            if stats["max"] > adamatzky_max:
                compliance["amplitude_factor"] = stats["max"] / adamatzky_max
                compliance["compliance_score"] = max(0, 100 - (compliance["amplitude_factor"] - 1) * 10)
                compliance["recommendations"].append(f"Amplitude {compliance['amplitude_factor']:.1f}x higher than biological range")
            else:
                compliance["amplitude_factor"] = adamatzky_min / stats["max"]
                compliance["compliance_score"] = max(0, 100 - (compliance["amplitude_factor"] - 1) * 10)
                compliance["recommendations"].append(f"Amplitude {compliance['amplitude_factor']:.1f}x lower than biological range")
        
        return compliance
    
    def calculate_signal_quality(self, amplitudes):
        """Calculate signal quality metrics"""
        
        # Signal-to-noise ratio estimation
        signal_power = np.mean(amplitudes ** 2)
        noise_power = np.var(amplitudes)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Dynamic range
        dynamic_range = 20 * np.log10(np.max(amplitudes) / np.min(amplitudes)) if np.min(amplitudes) > 0 else 0
        
        # Peak detection quality
        peaks, _ = signal.find_peaks(amplitudes, height=np.mean(amplitudes))
        peak_quality = len(peaks) / len(amplitudes) if len(amplitudes) > 0 else 0
        
        # Amplitude stability
        amplitude_stability = 1 / (1 + np.std(amplitudes) / np.mean(amplitudes)) if np.mean(amplitudes) > 0 else 0
        
        return {
            "signal_to_noise_ratio": snr,
            "dynamic_range_db": dynamic_range,
            "peak_density": peak_quality,
            "peak_count": len(peaks),
            "signal_stability": amplitude_stability,
            "amplitude_consistency": 1 - (np.std(amplitudes) / np.mean(amplitudes)) if np.mean(amplitudes) > 0 else 0
        }
    
    def cross_validate_amplitude_patterns(self, amplitudes):
        """Cross-validate amplitude patterns without forced parameters"""
        
        # Simple cross-validation using data splitting
        n_samples = len(amplitudes)
        if n_samples < 100:
            return {"error": "Insufficient data for cross-validation"}
        
        # Split data into 5 folds
        fold_size = n_samples // 5
        cv_scores = []
        
        for i in range(5):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < 4 else n_samples
            
            # Test fold
            test_data = amplitudes[start_idx:end_idx]
            # Training data (all other folds)
            train_data = np.concatenate([amplitudes[:start_idx], amplitudes[end_idx:]])
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            # Simple prediction: use mean of training data
            prediction = np.mean(train_data)
            actual = np.mean(test_data)
            
            # Calculate error
            error = abs(prediction - actual) / actual if actual != 0 else 0
            cv_scores.append(1 - error)  # Convert to score
        
        return {
            "cv_scores": cv_scores,
            "mean_cv_score": np.mean(cv_scores) if cv_scores else 0,
            "std_cv_score": np.std(cv_scores) if cv_scores else 0,
            "cv_quality": "Good" if np.mean(cv_scores) > 0.8 else "Poor" if cv_scores else "Insufficient Data"
        }
    
    def analyze_transform_output(self, amplitudes):
        """Analyze transform output without forced parameters"""
        
        # Detect peaks without forced thresholds
        peaks, properties = signal.find_peaks(amplitudes)
        
        # Analyze peak characteristics
        if len(peaks) > 0:
            peak_amplitudes = amplitudes[peaks]
            peak_intervals = np.diff(peaks)
            
            analysis = {
                "peak_count": len(peaks),
                "peak_density": len(peaks) / len(amplitudes),
                "mean_peak_amplitude": np.mean(peak_amplitudes),
                "std_peak_amplitude": np.std(peak_amplitudes),
                "mean_peak_interval": np.mean(peak_intervals) if len(peak_intervals) > 0 else 0,
                "peak_amplitude_range": {
                    "min": np.min(peak_amplitudes),
                    "max": np.max(peak_amplitudes)
                }
            }
        else:
            analysis = {
                "peak_count": 0,
                "peak_density": 0,
                "mean_peak_amplitude": 0,
                "std_peak_amplitude": 0,
                "mean_peak_interval": 0,
                "peak_amplitude_range": {"min": 0, "max": 0}
            }
        
        # Check for temporal patterns (Adamatzky's three scales)
        temporal_analysis = self.analyze_temporal_patterns(amplitudes)
        analysis["temporal_patterns"] = temporal_analysis
        
        return analysis
    
    def analyze_temporal_patterns(self, amplitudes):
        """Analyze temporal patterns without forced parameters"""
        
        # Use FFT to find dominant frequencies
        fft = np.fft.fft(amplitudes)
        freqs = np.fft.fftfreq(len(amplitudes))
        
        # Find dominant frequencies
        power_spectrum = np.abs(fft) ** 2
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        
        # Convert to temporal scales
        if dominant_freq != 0:
            dominant_period = 1 / abs(dominant_freq)
        else:
            dominant_period = float('inf')
        
        # Compare with Adamatzky's temporal scales
        adamatzky_scales = {
            "very_slow": 2573,  # seconds
            "slow": 457,        # seconds
            "very_fast": 24     # seconds
        }
        
        temporal_match = "unknown"
        for scale, period in adamatzky_scales.items():
            if 0.5 * period <= dominant_period <= 2 * period:
                temporal_match = scale
                break
        
        return {
            "dominant_frequency": dominant_freq,
            "dominant_period": dominant_period,
            "temporal_match": temporal_match,
            "power_spectrum_entropy": -np.sum(power_spectrum * np.log(power_spectrum + 1e-10))
        }
    
    def validate_all_files(self):
        """Validate amplitude quality across all files"""
        
        raw_data_dir = Path("data/raw")
        csv_files = list(raw_data_dir.glob("*.csv"))
        
        print(f"Found {len(csv_files)} CSV files to validate")
        
        all_results = []
        
        for csv_file in csv_files:
            try:
                result = self.analyze_amplitude_quality(csv_file)
                if result:
                    all_results.append(result)
                    print(f"✓ Validated: {csv_file.name}")
                else:
                    print(f"✗ Failed: {csv_file.name}")
            except Exception as e:
                print(f"✗ Error processing {csv_file.name}: {e}")
        
        # Aggregate results
        self.aggregate_validation_results(all_results)
        
        return all_results
    
    def aggregate_validation_results(self, results):
        """Aggregate validation results"""
        
        if not results:
            return
        
        # Overall statistics
        compliance_scores = [r["adamatzky_compliance"]["compliance_score"] for r in results]
        quality_scores = [r["quality_metrics"]["signal_stability"] for r in results]
        
        self.results["validation_summary"] = {
            "total_files": len(results),
            "average_compliance_score": np.mean(compliance_scores),
            "average_quality_score": np.mean(quality_scores),
            "files_within_biological_range": sum(1 for r in results if r["adamatzky_compliance"]["within_biological_range"]),
            "best_compliance_file": max(results, key=lambda x: x["adamatzky_compliance"]["compliance_score"])["file"],
            "worst_compliance_file": min(results, key=lambda x: x["adamatzky_compliance"]["compliance_score"])["file"]
        }
        
        # Save detailed results
        output_file = f"results/core_amplitude_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        self.print_validation_summary()
        
        return output_file
    
    def print_validation_summary(self):
        """Print validation summary"""
        
        summary = self.results["validation_summary"]
        
        print("\n" + "=" * 80)
        print("CORE AMPLITUDE VALIDATION SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal Files Analyzed: {summary['total_files']}")
        print(f"Average Compliance Score: {summary['average_compliance_score']:.1f}%")
        print(f"Average Quality Score: {summary['average_quality_score']:.3f}")
        print(f"Files Within Biological Range: {summary['files_within_biological_range']}/{summary['total_files']}")
        
        print(f"\nBest Compliance File: {summary['best_compliance_file']}")
        print(f"Worst Compliance File: {summary['worst_compliance_file']}")
        
        print("\n" + "=" * 80)

def main():
    """Main validation function"""
    
    validator = CoreAmplitudeValidator()
    
    print("Starting Core Amplitude Validation...")
    print("Using Adamatzky's methods with improved quality testing")
    print("No forced parameters - letting data speak for itself")
    
    results = validator.validate_all_files()
    
    print(f"\nValidation complete! Results saved to results/ directory")
    
    return results

if __name__ == "__main__":
    main() 