#!/usr/bin/env python3
"""
Improvement Comparison Script
Compares original vs improved ultra-simple scaling analysis results
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

class ImprovementAnalyzer:
    """Analyze improvements in ultra-simple scaling analysis"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Directory paths
        self.original_dir = Path("results/ultra_simple_scaling_analysis")
        self.improved_dir = Path("results/ultra_simple_scaling_analysis_improved")
        
        # Create comparison output directory
        self.comparison_dir = Path("results/improvement_comparison")
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔬 IMPROVEMENT COMPARISON ANALYSIS")
        print("=" * 60)
        print("Comparing original vs improved ultra-simple scaling analysis")
        print("=" * 60)
    
    def load_results(self, directory: Path) -> dict:
        """Load results from a directory"""
        results = {}
        
        if not directory.exists():
            print(f"❌ Directory not found: {directory}")
            return results
        
        # Load JSON results
        json_dir = directory / "json_results"
        if json_dir.exists():
            for json_file in json_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    results[json_file.stem] = data
                except Exception as e:
                    print(f"❌ Error loading {json_file}: {e}")
        
        # Load summary reports
        reports_dir = directory / "reports"
        if reports_dir.exists():
            for report_file in reports_dir.glob("*.json"):
                try:
                    with open(report_file, 'r') as f:
                        data = json.load(f)
                    results[f"summary_{report_file.stem}"] = data
                except Exception as e:
                    print(f"❌ Error loading {report_file}: {e}")
        
        return results
    
    def compare_spike_detection(self, original_results: dict, improved_results: dict) -> dict:
        """Compare spike detection results"""
        comparison = {
            'original_spikes': {},
            'improved_spikes': {},
            'differences': {}
        }
        
        # Extract spike data from both versions
        for filename, data in original_results.items():
            if 'rate_1.0' in data:
                spike_data = data['rate_1.0']['spike_detection']
                comparison['original_spikes'][filename] = {
                    'n_spikes': spike_data.get('n_spikes', 0),
                    'mean_amplitude': spike_data.get('mean_amplitude', 0),
                    'mean_isi': spike_data.get('mean_isi', 0),
                    'isi_cv': spike_data.get('isi_cv', 0),
                    'threshold_used': spike_data.get('threshold_used', 0)
                }
        
        for filename, data in improved_results.items():
            if 'rate_1.0' in data:
                spike_data = data['rate_1.0']['spike_detection']
                comparison['improved_spikes'][filename] = {
                    'n_spikes': spike_data.get('n_spikes', 0),
                    'mean_amplitude': spike_data.get('mean_amplitude', 0),
                    'mean_isi': spike_data.get('mean_isi', 0),
                    'isi_cv': spike_data.get('isi_cv', 0),
                    'threshold_used': spike_data.get('threshold_used', 0)
                }
        
        # Calculate differences
        for filename in comparison['original_spikes'].keys():
            if filename in comparison['improved_spikes']:
                orig = comparison['original_spikes'][filename]
                impr = comparison['improved_spikes'][filename]
                
                comparison['differences'][filename] = {
                    'spike_count_change': impr['n_spikes'] - orig['n_spikes'],
                    'amplitude_change': impr['mean_amplitude'] - orig['mean_amplitude'],
                    'isi_change': impr['mean_isi'] - orig['mean_isi'],
                    'threshold_change': impr['threshold_used'] - orig['threshold_used']
                }
        
        return comparison
    
    def compare_wave_transform_results(self, original_results: dict, improved_results: dict) -> dict:
        """Compare wave transform results"""
        comparison = {
            'original_features': {},
            'improved_features': {},
            'differences': {}
        }
        
        # Extract feature data from both versions
        for filename, data in original_results.items():
            if 'rate_1.0' in data:
                sqrt_data = data['rate_1.0']['square_root_results']
                linear_data = data['rate_1.0']['linear_results']
                
                comparison['original_features'][filename] = {
                    'sqrt_features': len(sqrt_data.get('all_features', [])),
                    'linear_features': len(linear_data.get('all_features', [])),
                    'sqrt_max_magnitude': sqrt_data.get('max_magnitude', 0),
                    'linear_max_magnitude': linear_data.get('max_magnitude', 0),
                    'sqrt_avg_magnitude': sqrt_data.get('avg_magnitude', 0),
                    'linear_avg_magnitude': linear_data.get('avg_magnitude', 0)
                }
        
        for filename, data in improved_results.items():
            if 'rate_1.0' in data:
                sqrt_data = data['rate_1.0']['square_root_results']
                linear_data = data['rate_1.0']['linear_results']
                
                comparison['improved_features'][filename] = {
                    'sqrt_features': len(sqrt_data.get('all_features', [])),
                    'linear_features': len(linear_data.get('all_features', [])),
                    'sqrt_max_magnitude': sqrt_data.get('max_magnitude', 0),
                    'linear_max_magnitude': linear_data.get('max_magnitude', 0),
                    'sqrt_avg_magnitude': sqrt_data.get('avg_magnitude', 0),
                    'linear_avg_magnitude': linear_data.get('avg_magnitude', 0)
                }
        
        # Calculate differences
        for filename in comparison['original_features'].keys():
            if filename in comparison['improved_features']:
                orig = comparison['original_features'][filename]
                impr = comparison['improved_features'][filename]
                
                comparison['differences'][filename] = {
                    'sqrt_features_change': impr['sqrt_features'] - orig['sqrt_features'],
                    'linear_features_change': impr['linear_features'] - orig['linear_features'],
                    'sqrt_magnitude_change': impr['sqrt_max_magnitude'] - orig['sqrt_max_magnitude'],
                    'linear_magnitude_change': impr['linear_max_magnitude'] - orig['linear_max_magnitude']
                }
        
        return comparison
    
    def create_comparison_report(self, spike_comparison: dict, feature_comparison: dict) -> dict:
        """Create comprehensive comparison report"""
        report = {
            'timestamp': self.timestamp,
            'improvements_implemented': [
                'Removed forced amplitude ranges',
                'Implemented adaptive thresholds',
                'Eliminated artificial noise',
                'Data-driven scale detection'
            ],
            'spike_detection_comparison': spike_comparison,
            'feature_detection_comparison': feature_comparison,
            'summary_statistics': {}
        }
        
        # Calculate summary statistics
        if spike_comparison['differences']:
            spike_changes = list(spike_comparison['differences'].values())
            report['summary_statistics']['spike_detection'] = {
                'avg_spike_count_change': np.mean([d['spike_count_change'] for d in spike_changes]),
                'avg_amplitude_change': np.mean([d['amplitude_change'] for d in spike_changes]),
                'files_with_more_spikes': sum(1 for d in spike_changes if d['spike_count_change'] > 0),
                'files_with_fewer_spikes': sum(1 for d in spike_changes if d['spike_count_change'] < 0)
            }
        
        if feature_comparison['differences']:
            feature_changes = list(feature_comparison['differences'].values())
            report['summary_statistics']['feature_detection'] = {
                'avg_sqrt_features_change': np.mean([d['sqrt_features_change'] for d in feature_changes]),
                'avg_linear_features_change': np.mean([d['linear_features_change'] for d in feature_changes]),
                'avg_sqrt_magnitude_change': np.mean([d['sqrt_magnitude_change'] for d in feature_changes]),
                'avg_linear_magnitude_change': np.mean([d['linear_magnitude_change'] for d in feature_changes])
            }
        
        return report
    
    def save_comparison_report(self, report: dict):
        """Save comparison report"""
        report_filename = f"improvement_comparison_report_{self.timestamp}.json"
        report_path = self.comparison_dir / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Comparison report saved: {report_path}")
        return report_path
    
    def run_comparison(self):
        """Run the complete comparison analysis"""
        print("\n📊 Loading original results...")
        original_results = self.load_results(self.original_dir)
        print(f"   Found {len(original_results)} original result files")
        
        print("\n📊 Loading improved results...")
        improved_results = self.load_results(self.improved_dir)
        print(f"   Found {len(improved_results)} improved result files")
        
        if not original_results or not improved_results:
            print("❌ Cannot run comparison - missing results")
            return
        
        print("\n🔍 Comparing spike detection...")
        spike_comparison = self.compare_spike_detection(original_results, improved_results)
        
        print("\n🔍 Comparing wave transform results...")
        feature_comparison = self.compare_wave_transform_results(original_results, improved_results)
        
        print("\n📊 Creating comparison report...")
        report = self.create_comparison_report(spike_comparison, feature_comparison)
        
        # Save report
        report_path = self.save_comparison_report(report)
        
        # Print summary
        print(f"\n🎯 COMPARISON SUMMARY:")
        print("=" * 60)
        
        if 'spike_detection' in report['summary_statistics']:
            spike_stats = report['summary_statistics']['spike_detection']
            print(f"📈 Spike Detection Changes:")
            print(f"   Average spike count change: {spike_stats['avg_spike_count_change']:.1f}")
            print(f"   Average amplitude change: {spike_stats['avg_amplitude_change']:.3f} mV")
            print(f"   Files with more spikes: {spike_stats['files_with_more_spikes']}")
            print(f"   Files with fewer spikes: {spike_stats['files_with_fewer_spikes']}")
        
        if 'feature_detection' in report['summary_statistics']:
            feature_stats = report['summary_statistics']['feature_detection']
            print(f"🌊 Feature Detection Changes:")
            print(f"   Average sqrt features change: {feature_stats['avg_sqrt_features_change']:.1f}")
            print(f"   Average linear features change: {feature_stats['avg_linear_features_change']:.1f}")
            print(f"   Average sqrt magnitude change: {feature_stats['avg_sqrt_magnitude_change']:.1f}")
            print(f"   Average linear magnitude change: {feature_stats['avg_linear_magnitude_change']:.1f}")
        
        print(f"\n✅ Comparison complete! Check: {report_path}")

def main():
    """Main execution function"""
    analyzer = ImprovementAnalyzer()
    analyzer.run_comparison()

if __name__ == "__main__":
    main() 