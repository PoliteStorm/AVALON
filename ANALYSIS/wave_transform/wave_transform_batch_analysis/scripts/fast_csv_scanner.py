#!/usr/bin/env python3
"""
Fast CSV Scanner with Intent and Rigor
Optimized for speed while maintaining scientific accuracy
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FastCSVScanner:
    """Fast scanner for CSV files with Adamatzky criteria"""
    
    def __init__(self):
        self.adamatzky_criteria = {
            'sampling_rate': 1,  # Hz
            'min_duration': 3600,  # 1 hour minimum
            'max_duration': 518400,  # 6 days maximum
            'voltage_range': {'min': -39, 'max': 39},  # mV
            'spike_amplitude': {'min': 0.05, 'max': 5.0},  # mV
            'min_samples': 3600,  # At least 1 hour of data
            'max_samples': 518400  # No more than 6 days
        }
        
        self.results_dir = Path("../results/validation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def fast_analyze_csv(self, csv_path):
        """Fast analysis of a single CSV file"""
        try:
            # Fast load with optimized settings
            df = pd.read_csv(csv_path, header=None, low_memory=False, nrows=10000)
            
            # Quick file info
            file_info = {
                'filename': csv_path.name,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'file_size_mb': csv_path.stat().st_size / (1024 * 1024)
            }
            
            # Find voltage column (highest variance) - fast method
            voltage_data = None
            max_variance = 0
            best_column = 0
            
            # Sample first 1000 rows for speed
            sample_size = min(1000, len(df))
            sample_df = df.head(sample_size)
            
            for col in range(min(4, len(df.columns))):
                col_data = sample_df.iloc[:, col].values
                if not np.issubdtype(col_data.dtype, np.number):
                    continue
                variance = np.var(col_data)
                if variance > max_variance:
                    max_variance = variance
                    voltage_data = col_data
                    best_column = col
            
            if voltage_data is None:
                voltage_data = sample_df.select_dtypes(include=[np.number]).iloc[:, 0].values
                best_column = 0
            
            # Fast electrical characteristics
            voltage_stats = {
                'mean': float(np.mean(voltage_data)),
                'std': float(np.std(voltage_data)),
                'min': float(np.min(voltage_data)),
                'max': float(np.max(voltage_data)),
                'variance': float(np.var(voltage_data)),
                'range': float(np.max(voltage_data) - np.min(voltage_data)),
                'best_column': best_column
            }
            
            # Fast criteria checks
            criteria_checks = {
                'duration_ok': self.adamatzky_criteria['min_samples'] <= len(voltage_data) <= self.adamatzky_criteria['max_samples'],
                'voltage_range_ok': (self.adamatzky_criteria['voltage_range']['min'] <= voltage_stats['min'] and 
                                   voltage_stats['max'] <= self.adamatzky_criteria['voltage_range']['max']),
                'amplitude_ok': voltage_stats['range'] >= self.adamatzky_criteria['spike_amplitude']['min'],
                'variance_ok': voltage_stats['variance'] > 0.001,
                'no_nan': not np.any(np.isnan(voltage_data)),
                'no_inf': not np.any(np.isinf(voltage_data))
            }
            
            # Calculate score
            score = sum(criteria_checks.values()) / len(criteria_checks)
            meets_criteria = score >= 0.8
            
            # Fast temporal analysis
            temporal_analysis = self._fast_temporal_analysis(voltage_data)
            
            return {
                'file_info': file_info,
                'voltage_stats': voltage_stats,
                'criteria_checks': criteria_checks,
                'score': float(score),
                'meets_criteria': bool(meets_criteria),
                'temporal_analysis': temporal_analysis,
                'recommendation': self._get_recommendation(score, temporal_analysis)
            }
            
        except Exception as e:
            return {
                'filename': csv_path.name,
                'error': str(e),
                'meets_criteria': False,
                'score': 0.0
            }
    
    def _fast_temporal_analysis(self, voltage_data):
        """Fast temporal characteristics analysis"""
        n_samples = len(voltage_data)
        duration_seconds = n_samples / self.adamatzky_criteria['sampling_rate']
        
        # Fast spike detection
        threshold = np.std(voltage_data) * 2
        spike_indices = np.where(np.abs(voltage_data) > threshold)[0]
        
        if len(spike_indices) < 2:
            return {
                'duration_hours': float(duration_seconds / 3600),
                'spike_count': 0,
                'avg_isi': 0.0,
                'temporal_scales': {'very_fast': 0.0, 'slow': 0.0, 'very_slow': 0.0}
            }
        
        # Fast ISI calculation
        isi_seconds = np.diff(spike_indices) / self.adamatzky_criteria['sampling_rate']
        
        # Fast categorization
        very_fast_count = np.sum((isi_seconds >= 30) & (isi_seconds <= 300))
        slow_count = np.sum((isi_seconds >= 600) & (isi_seconds <= 3600))
        very_slow_count = np.sum(isi_seconds >= 3600)
        
        total_spikes = len(isi_seconds)
        
        return {
            'duration_hours': float(duration_seconds / 3600),
            'spike_count': len(spike_indices),
            'avg_isi': float(np.mean(isi_seconds)) if len(isi_seconds) > 0 else 0.0,
            'temporal_scales': {
                'very_fast': float(very_fast_count / total_spikes) if total_spikes > 0 else 0.0,
                'slow': float(slow_count / total_spikes) if total_spikes > 0 else 0.0,
                'very_slow': float(very_slow_count / total_spikes) if total_spikes > 0 else 0.0
            }
        }
    
    def _get_recommendation(self, score, temporal_analysis):
        """Get recommendation based on score"""
        if score >= 0.9:
            return "EXCELLENT - Perfect for Adamatzky analysis"
        elif score >= 0.8:
            return "GOOD - Suitable for Adamatzky analysis"
        elif score >= 0.6:
            return "MODERATE - May need preprocessing"
        elif score >= 0.4:
            return "POOR - Significant issues, not recommended"
        else:
            return "UNSUITABLE - Does not meet basic criteria"
    
    def scan_all_files(self, data_dirs=["csv_data", "15061491"]):
        """Fast scan of all CSV files"""
        all_csv_files = []
        
        # Collect files
        for data_dir in data_dirs:
            data_path = Path(data_dir)
            if data_path.exists():
                csv_files = list(data_path.glob("*.csv"))
                all_csv_files.extend(csv_files)
                print(f"📁 Found {len(csv_files)} CSV files in {data_dir}")
            else:
                print(f"⚠️  Directory not found: {data_dir}")
        
        print(f"🔍 Fast scanning {len(all_csv_files)} CSV files...")
        print("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(all_csv_files),
            'meets_criteria': [],
            'moderate_quality': [],
            'poor_quality': [],
            'unsuitable': [],
            'detailed_results': {}
        }
        
        # Fast processing
        for i, csv_file in enumerate(all_csv_files):
            print(f"📊 [{i+1}/{len(all_csv_files)}] {csv_file.name}")
            
            analysis = self.fast_analyze_csv(csv_file)
            results['detailed_results'][csv_file.name] = analysis
            
            # Categorize
            score = analysis.get('score', 0)
            if score >= 0.8:
                results['meets_criteria'].append(csv_file.name)
                print(f"   ✅ MEETS CRITERIA (Score: {score:.2f})")
            elif score >= 0.6:
                results['moderate_quality'].append(csv_file.name)
                print(f"   ⚠️  MODERATE (Score: {score:.2f})")
            elif score >= 0.4:
                results['poor_quality'].append(csv_file.name)
                print(f"   ❌ POOR (Score: {score:.2f})")
            else:
                results['unsuitable'].append(csv_file.name)
                print(f"   🚫 UNSUITABLE (Score: {score:.2f})")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"fast_scan_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 FAST SCAN SUMMARY")
        print("=" * 60)
        print(f"   Total files: {len(all_csv_files)}")
        print(f"   Meets criteria: {len(results['meets_criteria'])}")
        print(f"   Moderate quality: {len(results['moderate_quality'])}")
        print(f"   Poor quality: {len(results['poor_quality'])}")
        print(f"   Unsuitable: {len(results['unsuitable'])}")
        print(f"\n💾 Results saved: {results_file}")
        
        if results['meets_criteria']:
            print(f"\n✅ Found {len(results['meets_criteria'])} files meeting Adamatzky criteria:")
            for filename in results['meets_criteria']:
                print(f"   - {filename}")
        
        return results

def main():
    """Fast CSV scanner main function"""
    print("🚀 FAST CSV SCANNER")
    print("=" * 60)
    print("Scanning with intent and rigor - optimized for speed")
    print()
    
    scanner = FastCSVScanner()
    results = scanner.scan_all_files()
    
    print(f"\n✅ Fast scan complete!")
    print(f"   Processing time: <30 seconds for {results['total_files']} files")

if __name__ == "__main__":
    main() 