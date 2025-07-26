#!/usr/bin/env python3
"""
Improved Amplitude Validation System
Uses Adamatzky's methods with improved cross-validation and no forced parameters
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy import signal
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

class ImprovedAmplitudeValidator:
    """Advanced amplitude validation using Adamatzky's methods"""
    
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
        
        # Cross-validation parameters
        self.cv_params = {
            "folds": 5,
            "random_state": 42,
            "scoring": "neg_mean_squared_error"
        }
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "validation_tests": {},
            "amplitude_analysis": {},
            "cross_validation_results": {},
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
        
        return {
            "file": data_file,
            "statistics": stats,
            "adamatzky_compliance": compliance,
            "quality_metrics": quality_metrics,
            "cross_validation": cv_results
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
        
        return {
            "signal_to_noise_ratio": snr,
            "dynamic_range_db": dynamic_range,
            "peak_density": peak_quality,
            "peak_count": len(peaks),
            "signal_stability": 1 / (1 + np.std(amplitudes) / np.mean(amplitudes)) if np.mean(amplitudes) > 0 else 0
        }
    
    def cross_validate_amplitude_patterns(self, amplitudes):
        """Cross-validate amplitude patterns without forced parameters"""
        
        # Create features for cross-validation
        features = []
        targets = []
        
        # Use sliding window approach
        window_size = min(100, len(amplitudes) // 10)
        
        for i in range(len(amplitudes) - window_size):
            window = amplitudes[i:i+window_size]
            features.append([
                np.mean(window),
                np.std(window),
                np.max(window),
                np.min(window),
                len(signal.find_peaks(window)[0])
            ])
            targets.append(amplitudes[i + window_size])
        
        if len(features) < 10:
            return {"error": "Insufficient data for cross-validation"}
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Simple linear model for cross-validation
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, features, targets, 
            cv=self.cv_params["folds"],
            scoring=self.cv_params["scoring"]
        )
        
        return {
            "cv_scores": cv_scores.tolist(),
            "mean_cv_score": np.mean(cv_scores),
            "std_cv_score": np.std(cv_scores),
            "cv_quality": "Good" if np.mean(cv_scores) > -0.1 else "Poor"
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
        output_file = f"results/improved_amplitude_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Print summary
        self.print_validation_summary()
        
        return output_file
    
    def print_validation_summary(self):
        """Print validation summary"""
        
        summary = self.results["validation_summary"]
        
        print("\n" + "=" * 80)
        print("IMPROVED AMPLITUDE VALIDATION SUMMARY")
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
    
    validator = ImprovedAmplitudeValidator()
    
    print("Starting Improved Amplitude Validation...")
    print("Using Adamatzky's methods with improved cross-validation")
    print("No forced parameters - letting data speak for itself")
    
    results = validator.validate_all_files()
    
    print(f"\nValidation complete! Results saved to results/ directory")
    
    return results

if __name__ == "__main__":
    main() 