#!/usr/bin/env python3
"""
False Positive Analysis for Electrical Activity Transform
Analyzes JSON files to identify potential false positives when voltage isn't present.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

class FalsePositiveAnalyzer:
    def __init__(self):
        self.json_files = {}
        self.analysis_results = {}
        
    def load_all_json_files(self):
        """Load all JSON files in the directory"""
        json_files = list(Path('.').glob('*.json'))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                self.json_files[json_file.name] = data
                print(f"Loaded {json_file.name}: {len(str(data))} characters")
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
        
        print(f"\nLoaded {len(self.json_files)} JSON files")
    
    def analyze_algorithm_comparison_results(self):
        """Analyze the algorithm comparison results for false positives"""
        if 'algorithm_comparison_results_20250715_233116.json' not in self.json_files:
            return
        
        data = self.json_files['algorithm_comparison_results_20250715_233116.json']
        
        print("\n" + "="*60)
        print("ALGORITHM COMPARISON RESULTS ANALYSIS")
        print("="*60)
        
        # Analyze Adamatzky results
        adamatzky = data.get('adamatzky_results', {})
        print(f"\nAdamatzky Spike Detection:")
        print(f"  Total spikes: {adamatzky.get('n_spikes', 0):,}")
        print(f"  Mean amplitude: {adamatzky.get('mean_amplitude', 0):.4f}")
        print(f"  Mean ISI: {adamatzky.get('mean_isi', 0):.4f}")
        
        # Analyze current (wave transform) results
        current = data.get('current_results', {})
        print(f"\nWave Transform Results:")
        print(f"  Features detected: {current.get('n_features', 0)}")
        print(f"  Mean magnitude: {current.get('mean_magnitude', 0):.4f}")
        print(f"  Mean frequency: {current.get('mean_frequency', 0):.4f}")
        print(f"  Mean time scale: {current.get('mean_time_scale', 0):.4f}")
        
        # Analyze voltage statistics
        voltage_stats = data.get('voltage_stats', {})
        print(f"\nVoltage Statistics:")
        print(f"  Samples: {voltage_stats.get('n_samples', 0):,}")
        print(f"  Min voltage: {voltage_stats.get('min_voltage', 0):.4f}")
        print(f"  Max voltage: {voltage_stats.get('max_voltage', 0):.4f}")
        print(f"  Mean voltage: {voltage_stats.get('mean_voltage', 0):.4f}")
        print(f"  Std voltage: {voltage_stats.get('std_voltage', 0):.4f}")
        
        # Check for potential false positives
        voltage_range = voltage_stats.get('max_voltage', 0) - voltage_stats.get('min_voltage', 0)
        voltage_std = voltage_stats.get('std_voltage', 0)
        
        print(f"\nFalse Positive Analysis:")
        print(f"  Voltage range: {voltage_range:.4f}")
        print(f"  Voltage std: {voltage_std:.4f}")
        
        # Criteria for false positives
        if voltage_range < 0.1:
            print(f"  ⚠️  LOW VOLTAGE RANGE: {voltage_range:.4f} - Possible false positive")
        if voltage_std < 0.05:
            print(f"  ⚠️  LOW VOLTAGE VARIABILITY: {voltage_std:.4f} - Possible false positive")
        if adamatzky.get('n_spikes', 0) == 0:
            print(f"  ⚠️  NO SPIKES DETECTED by Adamatzky method")
        if current.get('n_features', 0) == 0:
            print(f"  ⚠️  NO FEATURES DETECTED by wave transform")
        
        return {
            'adamatzky_spikes': adamatzky.get('n_spikes', 0),
            'wave_features': current.get('n_features', 0),
            'voltage_range': voltage_range,
            'voltage_std': voltage_std,
            'potential_false_positive': voltage_range < 0.1 or voltage_std < 0.05
        }
    
    def analyze_electrical_activity_results(self):
        """Analyze the main electrical activity results for false positives"""
        if 'electrical_activity_results_20250716_003156.json' not in self.json_files:
            return
        
        data = self.json_files['electrical_activity_results_20250716_003156.json']
        
        print("\n" + "="*60)
        print("ELECTRICAL ACTIVITY RESULTS FALSE POSITIVE ANALYSIS")
        print("="*60)
        
        # Analyze by file type
        file_types = {}
        false_positive_candidates = []
        
        for file_name, file_data in data.items():
            file_type = file_data.get('file_type', 'unknown')
            if file_type not in file_types:
                file_types[file_type] = []
            file_types[file_type].append((file_name, file_data))
            
            # Check for potential false positives
            if self._is_potential_false_positive(file_data):
                false_positive_candidates.append((file_name, file_data))
        
        print(f"\nFile Type Distribution:")
        for file_type, files in file_types.items():
            print(f"  {file_type}: {len(files)} files")
        
        print(f"\nPotential False Positive Analysis:")
        print(f"  Total files analyzed: {len(data)}")
        print(f"  Potential false positives: {len(false_positive_candidates)}")
        
        # Analyze false positive candidates
        if false_positive_candidates:
            print(f"\nTop False Positive Candidates:")
            for i, (file_name, file_data) in enumerate(false_positive_candidates[:10]):
                reasons = self._get_false_positive_reasons(file_data)
                print(f"  {i+1}. {file_name}")
                for reason in reasons:
                    print(f"     - {reason}")
        
        return {
            'total_files': len(data),
            'false_positive_candidates': len(false_positive_candidates),
            'file_types': {k: len(v) for k, v in file_types.items()}
        }
    
    def _is_potential_false_positive(self, file_data):
        """Check if a file might be a false positive"""
        file_type = file_data.get('file_type', 'unknown')
        
        if file_type == 'coordinate':
            # Check coordinate data for false positives
            velocity_rms = file_data.get('velocity_rms', 0)
            curvature_rms = file_data.get('curvature_rms', 0)
            distance_range = file_data.get('distance_range', 0)
            
            # Very low movement suggests false positive
            if velocity_rms < 1.0 and curvature_rms < 0.1:
                return True
        
        elif file_type in ['direct_electrical', 'fungal_spike']:
            # Check electrical data for false positives
            voltage_rms = file_data.get('voltage_rms', 0)
            voltage_peak_rate = file_data.get('voltage_peak_rate', 0)
            voltage_zero_crossing_rate = file_data.get('voltage_zero_crossing_rate', 0)
            
            # Very low electrical activity suggests false positive
            if voltage_rms < 0.1 and voltage_peak_rate < 0.001:
                return True
            
            # No zero crossings might indicate flat signal
            if voltage_zero_crossing_rate < 0.001:
                return True
        
        elif file_type == 'moisture':
            # Check moisture data for false positives
            moisture_std = file_data.get('moisture_std', 0)
            moisture_range = file_data.get('moisture_range', 0)
            
            # Very low moisture variability suggests false positive
            if moisture_std < 0.01 and moisture_range < 0.1:
                return True
        
        return False
    
    def _get_false_positive_reasons(self, file_data):
        """Get specific reasons why a file might be a false positive"""
        reasons = []
        file_type = file_data.get('file_type', 'unknown')
        
        if file_type == 'coordinate':
            velocity_rms = file_data.get('velocity_rms', 0)
            curvature_rms = file_data.get('curvature_rms', 0)
            
            if velocity_rms < 1.0:
                reasons.append(f"Very low velocity RMS: {velocity_rms:.4f}")
            if curvature_rms < 0.1:
                reasons.append(f"Very low curvature RMS: {curvature_rms:.4f}")
        
        elif file_type in ['direct_electrical', 'fungal_spike']:
            voltage_rms = file_data.get('voltage_rms', 0)
            voltage_peak_rate = file_data.get('voltage_peak_rate', 0)
            voltage_zero_crossing_rate = file_data.get('voltage_zero_crossing_rate', 0)
            
            if voltage_rms < 0.1:
                reasons.append(f"Very low voltage RMS: {voltage_rms:.4f}")
            if voltage_peak_rate < 0.001:
                reasons.append(f"Very low peak rate: {voltage_peak_rate:.4f}")
            if voltage_zero_crossing_rate < 0.001:
                reasons.append(f"No zero crossings: {voltage_zero_crossing_rate:.4f}")
        
        elif file_type == 'moisture':
            moisture_std = file_data.get('moisture_std', 0)
            moisture_range = file_data.get('moisture_range', 0)
            
            if moisture_std < 0.01:
                reasons.append(f"Very low moisture std: {moisture_std:.4f}")
            if moisture_range < 0.1:
                reasons.append(f"Very low moisture range: {moisture_range:.4f}")
        
        return reasons
    
    def analyze_electrical_analysis_results(self):
        """Analyze the electrical analysis results"""
        if 'electrical_analysis_results_20250716_003343.json' not in self.json_files:
            return
        
        data = self.json_files['electrical_analysis_results_20250716_003343.json']
        
        print("\n" + "="*60)
        print("ELECTRICAL ANALYSIS RESULTS ANALYSIS")
        print("="*60)
        
        print(f"\nFile Type Distribution:")
        file_types = data.get('file_type_distribution', {})
        for file_type, count in file_types.items():
            print(f"  {file_type}: {count} files")
        
        print(f"\nTop Electrical Files (Potential False Positives):")
        top_electrical = data.get('top_electrical_files', [])
        for i, file_info in enumerate(top_electrical[:5]):
            file_name = file_info.get('file', 'Unknown')
            voltage_rms = file_info.get('voltage_rms', 0)
            peak_rate = file_info.get('peak_rate', 0)
            
            print(f"  {i+1}. {file_name}")
            print(f"     Voltage RMS: {voltage_rms:.4f}")
            print(f"     Peak Rate: {peak_rate:.4f}")
            
            # Check for false positives
            if voltage_rms > 1e6:  # Extremely high voltage
                print(f"     ⚠️  SUSPICIOUS: Extremely high voltage RMS")
            if peak_rate == 0.0:
                print(f"     ⚠️  SUSPICIOUS: Zero peak rate despite high voltage")
        
        print(f"\nTop Movement Files (Potential False Positives):")
        top_movement = data.get('top_movement_files', [])
        for i, file_info in enumerate(top_movement[:5]):
            file_name = file_info.get('file', 'Unknown')
            velocity_rms = file_info.get('velocity_rms', 0)
            curvature_rms = file_info.get('curvature_rms', 0)
            
            print(f"  {i+1}. {file_name}")
            print(f"     Velocity RMS: {velocity_rms:.4f}")
            print(f"     Curvature RMS: {curvature_rms:.4f}")
            
            # Check for false positives
            if velocity_rms > 200:  # Very high velocity
                print(f"     ⚠️  SUSPICIOUS: Extremely high velocity")
            if curvature_rms < 0.1 and velocity_rms > 100:
                print(f"     ⚠️  SUSPICIOUS: High velocity but low curvature")
        
        return {
            'file_types': file_types,
            'top_electrical_count': len(top_electrical),
            'top_movement_count': len(top_movement)
        }
    
    def analyze_test_results(self):
        """Analyze the test results files for false positives"""
        test_files = [name for name in self.json_files.keys() if 'test' in name.lower()]
        
        print("\n" + "="*60)
        print("TEST RESULTS FALSE POSITIVE ANALYSIS")
        print("="*60)
        
        for test_file in test_files:
            print(f"\nAnalyzing {test_file}:")
            data = self.json_files[test_file]
            
            # Extract key metrics
            voltage_stats = data.get('voltage_stats', {})
            adamatzky_results = data.get('adamatzky_results', {})
            
            if voltage_stats:
                voltage_range = voltage_stats.get('max', 0) - voltage_stats.get('min', 0)
                voltage_std = voltage_stats.get('std', 0)
                
                print(f"  Voltage range: {voltage_range:.4f}")
                print(f"  Voltage std: {voltage_std:.4f}")
                
                if voltage_range < 0.1:
                    print(f"  ⚠️  LOW VOLTAGE RANGE - Possible false positive")
                if voltage_std < 0.05:
                    print(f"  ⚠️  LOW VOLTAGE VARIABILITY - Possible false positive")
            
            if adamatzky_results:
                n_spikes = len(adamatzky_results.get('spikes', []))
                print(f"  Adamatzky spikes detected: {n_spikes}")
                
                if n_spikes == 0:
                    print(f"  ⚠️  NO SPIKES DETECTED - Possible false positive")
                elif n_spikes > 1000:
                    print(f"  ⚠️  EXCESSIVE SPIKES - Possible false positive")
    
    def generate_false_positive_report(self):
        """Generate a comprehensive false positive report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE FALSE POSITIVE ANALYSIS REPORT")
        print("="*60)
        
        # Run all analyses
        algorithm_analysis = self.analyze_algorithm_comparison_results()
        electrical_analysis = self.analyze_electrical_activity_results()
        analysis_results = self.analyze_electrical_analysis_results()
        self.analyze_test_results()
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"  Total JSON files analyzed: {len(self.json_files)}")
        
        if algorithm_analysis:
            print(f"  Algorithm comparison - Adamatzky spikes: {algorithm_analysis.get('adamatzky_spikes', 0)}")
            print(f"  Algorithm comparison - Wave features: {algorithm_analysis.get('wave_features', 0)}")
            print(f"  Potential false positive in algorithm comparison: {algorithm_analysis.get('potential_false_positive', False)}")
        
        if electrical_analysis:
            print(f"  Electrical activity - Total files: {electrical_analysis.get('total_files', 0)}")
            print(f"  Electrical activity - False positive candidates: {electrical_analysis.get('false_positive_candidates', 0)}")
        
        # Key findings about false positives
        print(f"\nFALSE POSITIVE RISK FACTORS:")
        print(f"  1. Low voltage range (< 0.1) - Indicates flat or constant signals")
        print(f"  2. Low voltage variability (< 0.05 std) - Suggests noise-free but uninteresting data")
        print(f"  3. Zero peak rates - No spike-like activity detected")
        print(f"  4. Zero zero-crossings - Completely flat signal")
        print(f"  5. Extremely high voltage values (> 1e6) - Possible data corruption or scaling issues")
        print(f"  6. High velocity with low curvature - Unrealistic movement patterns")
        
        print(f"\nRECOMMENDATIONS:")
        print(f"  1. Apply voltage range thresholds before analysis")
        print(f"  2. Check for data quality issues (corruption, scaling)")
        print(f"  3. Validate coordinate data for realistic movement patterns")
        print(f"  4. Use multiple detection methods to cross-validate results")
        print(f"  5. Implement confidence scores for electrical activity detection")

def main():
    """Main function to run the false positive analysis"""
    analyzer = FalsePositiveAnalyzer()
    analyzer.load_all_json_files()
    analyzer.generate_false_positive_report()

if __name__ == "__main__":
    main() 