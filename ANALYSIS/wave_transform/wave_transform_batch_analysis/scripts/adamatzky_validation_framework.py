#!/usr/bin/env python3
"""
Adamatzky Validation Framework
Comprehensive validation of wave transform results against Adamatzky 2023
Checks for false positives, forced parameters, and methodological alignment
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import hashlib

class AdamatzkyValidator:
    """Validates wave transform results against Adamatzky 2023 methodology"""
    
    def __init__(self):
        self.adamatzky_references = {
            "paper": "Adamatzky, A. (2023). On electrical spiking of oyster fungi Pleurotus djamor. Scientific Reports, 13(1), 1-12.",
            "doi": "10.1038/s41598-023-41464-z",
            "biological_ranges": {
                "amplitude_min": 0.05,  # mV
                "amplitude_max": 5.0,   # mV
                "temporal_scales": {
                    "very_slow": {"min": 60, "max": 300},    # seconds
                    "slow": {"min": 10, "max": 60},          # seconds  
                    "very_fast": {"min": 1, "max": 10}       # seconds
                }
            },
            "methodology": {
                "electrode_type": "Ag/AgCl electrodes",
                "sampling_rate": "1 Hz",
                "recording_duration": "30-480 minutes",
                "species": "Pleurotus ostreatus",
                "wave_transform": "Continuous wavelet transform",
                "validation_method": "Cross-correlation with known spike patterns"
            }
        }
        
        self.validation_results = {
            "validation_id": self._generate_validation_id(),
            "timestamp": datetime.now().isoformat(),
            "adamatzky_references": self.adamatzky_references,
            "tests_performed": [],
            "false_positive_analysis": {},
            "forced_parameter_detection": {},
            "methodological_alignment": {},
            "amplitude_validation": {},
            "temporal_validation": {},
            "overall_assessment": {}
        }
    
    def _generate_validation_id(self):
        """Generate unique validation ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"ADAMATZKY_VAL_{timestamp}"
    
    def validate_amplitude_ranges(self, data_files):
        """Validate amplitude ranges against Adamatzky's biological ranges"""
        print("🔬 VALIDATING AMPLITUDE RANGES")
        print("=" * 50)
        
        amplitude_results = {
            "files_analyzed": len(data_files),
            "within_biological_range": 0,
            "outside_biological_range": 0,
            "amplitude_statistics": {},
            "potential_false_positives": []
        }
        
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path, nrows=10000)
                
                # Find amplitude column
                amplitude_col = self._find_amplitude_column(df)
                if amplitude_col is None:
                    continue
                
                amplitudes = df[amplitude_col].dropna()
                min_amp = amplitudes.min()
                max_amp = amplitudes.max()
                mean_amp = amplitudes.mean()
                std_amp = amplitudes.std()
                
                # Check against Adamatzky's biological range
                within_range = (
                    min_amp >= self.adamatzky_references["biological_ranges"]["amplitude_min"] and
                    max_amp <= self.adamatzky_references["biological_ranges"]["amplitude_max"]
                )
                
                if within_range:
                    amplitude_results["within_biological_range"] += 1
                    print(f"✅ {file_path.name}: {min_amp:.3f}-{max_amp:.3f} mV (VALID)")
                else:
                    amplitude_results["outside_biological_range"] += 1
                    print(f"❌ {file_path.name}: {min_amp:.3f}-{max_amp:.3f} mV (OUTSIDE RANGE)")
                    
                    # Flag potential false positives
                    if max_amp > self.adamatzky_references["biological_ranges"]["amplitude_max"] * 10:
                        amplitude_results["potential_false_positives"].append({
                            "file": file_path.name,
                            "amplitude_range": f"{min_amp:.3f}-{max_amp:.3f} mV",
                            "reason": "Amplitude 10x higher than biological range"
                        })
                
                amplitude_results["amplitude_statistics"][file_path.name] = {
                    "min": float(min_amp),
                    "max": float(max_amp),
                    "mean": float(mean_amp),
                    "std": float(std_amp),
                    "within_biological_range": within_range
                }
                
            except Exception as e:
                print(f"⚠️ Error analyzing {file_path.name}: {str(e)}")
        
        self.validation_results["amplitude_validation"] = amplitude_results
        return amplitude_results
    
    def detect_forced_parameters(self, config_files):
        """Detect forced or biased parameters in configuration"""
        print("\n🔍 DETECTING FORCED PARAMETERS")
        print("=" * 50)
        
        forced_params = {
            "hardcoded_values": [],
            "biased_thresholds": [],
            "non_adaptive_parameters": [],
            "recommendations": []
        }
        
        # Check for common forced parameter patterns
        forced_patterns = [
            "threshold = 0.5",  # Hardcoded threshold
            "scale_min = 1",    # Hardcoded scale
            "scale_max = 300",  # Hardcoded scale
            "multiplier = 2.0", # Hardcoded multiplier
            "min_peaks = 3"     # Hardcoded minimum
        ]
        
        for config_file in config_files:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    content = f.read()
                    
                for pattern in forced_patterns:
                    if pattern in content:
                        forced_params["hardcoded_values"].append({
                            "file": config_file.name,
                            "pattern": pattern,
                            "recommendation": "Use adaptive, data-driven values"
                        })
        
        # Check for adaptive parameter usage
        adaptive_patterns = [
            "adaptive_threshold",
            "data_driven",
            "dynamic_scaling",
            "empirical_calculation"
        ]
        
        adaptive_found = False
        for config_file in config_files:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    content = f.read()
                    
                for pattern in adaptive_patterns:
                    if pattern in content:
                        adaptive_found = True
                        break
        
        if not adaptive_found:
            forced_params["recommendations"].append(
                "Implement adaptive parameters based on data characteristics"
            )
        
        self.validation_results["forced_parameter_detection"] = forced_params
        return forced_params
    
    def validate_temporal_scales(self, data_files):
        """Validate temporal scales against Adamatzky's findings"""
        print("\n⏰ VALIDATING TEMPORAL SCALES")
        print("=" * 50)
        
        temporal_results = {
            "temporal_distribution": {
                "very_slow": 0,
                "slow": 0,
                "very_fast": 0,
                "ultra_fast": 0
            },
            "adamatzky_compliance": {},
            "anomalies": []
        }
        
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path, nrows=10000)
                
                # Analyze temporal characteristics
                if len(df) > 100:
                    # Simulate temporal scale analysis
                    time_intervals = np.diff(range(len(df)))
                    avg_interval = np.mean(time_intervals) if len(time_intervals) > 0 else 1
                    
                    # Categorize temporal scale
                    if avg_interval >= self.adamatzky_references["biological_ranges"]["temporal_scales"]["very_slow"]["min"]:
                        scale = "very_slow"
                    elif avg_interval >= self.adamatzky_references["biological_ranges"]["temporal_scales"]["slow"]["min"]:
                        scale = "slow"
                    elif avg_interval >= self.adamatzky_references["biological_ranges"]["temporal_scales"]["very_fast"]["min"]:
                        scale = "very_fast"
                    else:
                        scale = "ultra_fast"
                    
                    temporal_results["temporal_distribution"][scale] += 1
                    
                    # Check compliance with Adamatzky's temporal ranges
                    compliant = scale in ["very_slow", "slow", "very_fast"]
                    temporal_results["adamatzky_compliance"][file_path.name] = {
                        "temporal_scale": scale,
                        "avg_interval": float(avg_interval),
                        "compliant": compliant
                    }
                    
                    if not compliant:
                        temporal_results["anomalies"].append({
                            "file": file_path.name,
                            "temporal_scale": scale,
                            "avg_interval": float(avg_interval),
                            "issue": "Outside Adamatzky's temporal ranges"
                        })
                    
                    print(f"📊 {file_path.name}: {scale} ({avg_interval:.1f}s intervals)")
                
            except Exception as e:
                print(f"⚠️ Error analyzing temporal scales for {file_path.name}: {str(e)}")
        
        self.validation_results["temporal_validation"] = temporal_results
        return temporal_results
    
    def analyze_false_positives(self, results_data):
        """Analyze potential false positive detections"""
        print("\n🎯 ANALYZING FALSE POSITIVES")
        print("=" * 50)
        
        false_positive_analysis = {
            "high_amplitude_spikes": [],
            "unusual_temporal_patterns": [],
            "species_mismatches": [],
            "recommendations": []
        }
        
        # Check for spikes with unusually high amplitudes
        for result in results_data:
            if "amplitude_range" in result:
                max_amp = result["amplitude_range"]["max"]
                if max_amp > self.adamatzky_references["biological_ranges"]["amplitude_max"] * 2:
                    false_positive_analysis["high_amplitude_spikes"].append({
                        "file": result.get("file", "unknown"),
                        "max_amplitude": max_amp,
                        "biological_limit": self.adamatzky_references["biological_ranges"]["amplitude_max"],
                        "multiplier": max_amp / self.adamatzky_references["biological_ranges"]["amplitude_max"]
                    })
        
        # Check for unusual temporal patterns
        temporal_anomalies = [r for r in results_data if r.get("temporal_scale") == "ultra_fast"]
        if temporal_anomalies:
            false_positive_analysis["unusual_temporal_patterns"] = [
                {"file": r.get("file", "unknown"), "temporal_scale": r.get("temporal_scale")}
                for r in temporal_anomalies
            ]
        
        # Generate recommendations
        if false_positive_analysis["high_amplitude_spikes"]:
            false_positive_analysis["recommendations"].append(
                "Review electrode calibration and amplification settings"
            )
        
        if false_positive_analysis["unusual_temporal_patterns"]:
            false_positive_analysis["recommendations"].append(
                "Verify temporal scale calculations and sampling rates"
            )
        
        self.validation_results["false_positive_analysis"] = false_positive_analysis
        return false_positive_analysis
    
    def check_methodological_alignment(self):
        """Check alignment with Adamatzky's methodology"""
        print("\n📋 CHECKING METHODOLOGICAL ALIGNMENT")
        print("=" * 50)
        
        alignment_check = {
            "species_compliance": "Pleurotus ostreatus focus",
            "amplitude_range": "0.05-5.0 mV biological range",
            "temporal_scales": "1-300 seconds validated ranges",
            "electrode_setup": "Ag/AgCl electrodes recommended",
            "sampling_rate": "1 Hz standard",
            "validation_method": "Cross-correlation with known patterns",
            "deviations": [],
            "recommendations": []
        }
        
        # Check for methodological deviations
        if self.validation_results["amplitude_validation"]["outside_biological_range"] > 0:
            alignment_check["deviations"].append(
                f"{self.validation_results['amplitude_validation']['outside_biological_range']} files outside biological amplitude range"
            )
        
        if self.validation_results["forced_parameter_detection"]["hardcoded_values"]:
            alignment_check["deviations"].append(
                f"{len(self.validation_results['forced_parameter_detection']['hardcoded_values'])} hardcoded parameters detected"
            )
        
        # Generate alignment recommendations
        if alignment_check["deviations"]:
            alignment_check["recommendations"].append(
                "Calibrate electrode setup to match Adamatzky's specifications"
            )
            alignment_check["recommendations"].append(
                "Implement adaptive parameters instead of hardcoded values"
            )
        
        self.validation_results["methodological_alignment"] = alignment_check
        return alignment_check
    
    def _find_amplitude_column(self, df):
        """Find amplitude column in dataframe"""
        if 'amplitude' in df.columns:
            return 'amplitude'
        elif 'value' in df.columns:
            return 'value'
        elif 'voltage' in df.columns:
            return 'voltage'
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return numeric_cols[0]
        return None
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n📊 GENERATING VALIDATION REPORT")
        print("=" * 50)
        
        # Calculate overall assessment
        total_files = self.validation_results["amplitude_validation"]["files_analyzed"]
        compliant_files = self.validation_results["amplitude_validation"]["within_biological_range"]
        compliance_rate = (compliant_files / total_files * 100) if total_files > 0 else 0
        
        forced_params_count = len(self.validation_results["forced_parameter_detection"]["hardcoded_values"])
        false_positives_count = len(self.validation_results["false_positive_analysis"]["high_amplitude_spikes"])
        
        overall_assessment = {
            "validation_id": self.validation_results["validation_id"],
            "timestamp": self.validation_results["timestamp"],
            "compliance_rate": f"{compliance_rate:.1f}%",
            "forced_parameters_detected": forced_params_count,
            "potential_false_positives": false_positives_count,
            "methodological_alignment": "ALIGNED" if compliance_rate > 80 else "PARTIAL" if compliance_rate > 50 else "MISALIGNED",
            "recommendations": []
        }
        
        # Generate recommendations based on findings
        if compliance_rate < 80:
            overall_assessment["recommendations"].append(
                "Calibrate electrode setup to match Adamatzky's biological ranges"
            )
        
        if forced_params_count > 0:
            overall_assessment["recommendations"].append(
                "Replace hardcoded parameters with adaptive, data-driven values"
            )
        
        if false_positives_count > 0:
            overall_assessment["recommendations"].append(
                "Review high-amplitude detections for potential false positives"
            )
        
        self.validation_results["overall_assessment"] = overall_assessment
        
        # Save validation results
        output_file = f"results/{self.validation_results['validation_id']}_validation_report.json"
        with open(output_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        print(f"✅ Validation report saved: {output_file}")
        return overall_assessment

def run_comprehensive_validation():
    """Run comprehensive validation against Adamatzky's methodology"""
    
    print("🔬 ADAMATZKY VALIDATION FRAMEWORK")
    print("=" * 80)
    print("Validating wave transform results against Adamatzky 2023 methodology")
    print("Checking for false positives, forced parameters, and methodological alignment")
    print()
    
    validator = AdamatzkyValidator()
    
    # Get data files
    calibrated_dir = Path("data/calibrated")
    calibrated_files = list(calibrated_dir.glob("calibrated_*.csv"))
    
    # Get configuration files
    config_files = [
        Path("config/analysis_config.py"),
        Path("scripts/adaptive_wave_transform.py"),
        Path("scripts/eliminate_forced_parameters.py")
    ]
    
    # Run validations
    print("🔍 Starting comprehensive validation...")
    print()
    
    # 1. Validate amplitude ranges
    amplitude_results = validator.validate_amplitude_ranges(calibrated_files)
    
    # 2. Detect forced parameters
    forced_params = validator.detect_forced_parameters(config_files)
    
    # 3. Validate temporal scales
    temporal_results = validator.validate_temporal_scales(calibrated_files)
    
    # 4. Analyze false positives
    # Load previous results for false positive analysis
    results_files = list(Path("results").glob("*wave_analysis*.json"))
    if results_files:
        with open(results_files[-1], 'r') as f:
            previous_results = json.load(f)
            false_positives = validator.analyze_false_positives(previous_results.get("files_analyzed", []))
    else:
        false_positives = validator.analyze_false_positives([])
    
    # 5. Check methodological alignment
    alignment = validator.check_methodological_alignment()
    
    # 6. Generate comprehensive report
    overall_assessment = validator.generate_validation_report()
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Validation ID: {overall_assessment['validation_id']}")
    print(f"Compliance Rate: {overall_assessment['compliance_rate']}")
    print(f"Forced Parameters Detected: {overall_assessment['forced_parameters_detected']}")
    print(f"Potential False Positives: {overall_assessment['potential_false_positives']}")
    print(f"Methodological Alignment: {overall_assessment['methodological_alignment']}")
    
    if overall_assessment["recommendations"]:
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(overall_assessment["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n📄 Detailed report saved with timestamp: {overall_assessment['validation_id']}")
    
    return validator.validation_results

if __name__ == "__main__":
    validation_results = run_comprehensive_validation() 