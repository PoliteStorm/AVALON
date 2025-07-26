#!/usr/bin/env python3
"""
Comprehensive Anomaly Scanning for Fungal Electrical Activity Datasets
Performs deep analysis to find additional anomalies missed in initial detection
"""

import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import csv

class ComprehensiveAnomalyScanner:
    """Performs comprehensive anomaly scanning"""
    
    def __init__(self):
        self.additional_anomalies = []
        self.temporal_issues = []
        self.format_issues = []
        self.sampling_issues = []
        self.missing_data_issues = []
        
    def load_data(self) -> Dict:
        """Load the extraction results"""
        json_file = Path("environmental_analysis/environmental_parameters_20250718_021033.json")
        if json_file.exists():
            with open(json_file, 'r') as f:
                return json.load(f)
        return {}
    
    def scan_temporal_anomalies(self, data: Dict):
        """Scan for temporal inconsistencies"""
        print("⏰ TEMPORAL ANOMALY SCANNING")
        print("=" * 50)
        
        electrical = data.get('electrical_datasets', [])
        
        for dataset in electrical:
            filename = dataset['filename']
            
            # Check for temporal patterns in filenames
            if 'part' in filename.lower():
                # Check for missing parts in series
                base_name = filename.replace('_part1.csv', '').replace('_part2.csv', '').replace('_part3.csv', '')
                if '_part1.csv' in filename:
                    # Look for missing part2, part3
                    expected_parts = [f"{base_name}_part2.csv", f"{base_name}_part3.csv"]
                    missing_parts = []
                    for expected in expected_parts:
                        if not any(expected in d['filename'] for d in electrical):
                            missing_parts.append(expected)
                    
                    if missing_parts:
                        self.temporal_issues.append({
                            'type': 'MISSING_TEMPORAL_PARTS',
                            'file': filename,
                            'missing': missing_parts,
                            'description': f"Expected temporal series parts missing: {missing_parts}"
                        })
            
            # Check for unrealistic time durations
            if 'd' in filename or 'h' in filename:
                time_match = re.search(r'(\d+)([dh])', filename)
                if time_match:
                    duration = int(time_match.group(1))
                    unit = time_match.group(2)
                    
                    if unit == 'd' and duration > 365:  # More than a year
                        self.temporal_issues.append({
                            'type': 'UNREALISTIC_DURATION',
                            'file': filename,
                            'duration': f"{duration} days",
                            'description': f"Duration of {duration} days seems unrealistic for fungal experiments"
                        })
                    
                    if unit == 'h' and duration > 24:  # More than a day in hours
                        self.temporal_issues.append({
                            'type': 'HOURS_INSTEAD_OF_DAYS',
                            'file': filename,
                            'duration': f"{duration} hours",
                            'description': f"Duration of {duration} hours might be better expressed in days"
                        })
    
    def scan_format_anomalies(self, data: Dict):
        """Scan for data format issues"""
        print("📄 FORMAT ANOMALY SCANNING")
        print("=" * 50)
        
        # Check CSV files for format issues
        csv_files = []
        for directory in ['15061491', 'csv_data']:
            if Path(directory).exists():
                csv_files.extend(list(Path(directory).glob('*.csv')))
        
        for csv_file in csv_files[:10]:  # Sample first 10 files
            try:
                with open(csv_file, 'r') as f:
                    first_lines = [next(f) for _ in range(5)]
                
                # Check for encoding issues
                if any('' in line for line in first_lines):
                    self.format_issues.append({
                        'type': 'ENCODING_ISSUE',
                        'file': str(csv_file),
                        'description': "Contains encoding characters () indicating potential encoding problems"
                    })
                
                # Check for inconsistent delimiters
                delimiter_counts = {}
                for line in first_lines:
                    for delim in [',', ';', '\t']:
                        delimiter_counts[delim] = delimiter_counts.get(delim, 0) + line.count(delim)
                
                if len([d for d in delimiter_counts.values() if d > 0]) > 1:
                    self.format_issues.append({
                        'type': 'MIXED_DELIMITERS',
                        'file': str(csv_file),
                        'delimiters': delimiter_counts,
                        'description': "Mixed delimiters detected - may cause parsing issues"
                    })
                
                # Check for empty lines
                if any(line.strip() == '' for line in first_lines):
                    self.format_issues.append({
                        'type': 'EMPTY_LINES',
                        'file': str(csv_file),
                        'description': "Contains empty lines which may cause parsing issues"
                    })
                    
            except Exception as e:
                self.format_issues.append({
                    'type': 'FILE_READ_ERROR',
                    'file': str(csv_file),
                    'error': str(e),
                    'description': f"Cannot read file: {e}"
                })
    
    def scan_sampling_anomalies(self, data: Dict):
        """Scan for sampling rate and consistency issues"""
        print("📊 SAMPLING ANOMALY SCANNING")
        print("=" * 50)
        
        electrical = data.get('electrical_datasets', [])
        
        for dataset in electrical:
            filename = dataset['filename']
            metadata = dataset['metadata']
            
            # Check for unrealistic sampling rates
            sampling_rate = metadata.get('sampling_rate_hz', 1.0)
            if sampling_rate > 1000:  # More than 1kHz
                self.sampling_issues.append({
                    'type': 'UNREALISTIC_SAMPLING_RATE',
                    'file': filename,
                    'rate': f"{sampling_rate} Hz",
                    'description': f"Sampling rate of {sampling_rate} Hz seems unrealistic for fungal electrical monitoring"
                })
            
            # Check for very low sampling rates
            if sampling_rate < 0.001:  # Less than 1mHz
                self.sampling_issues.append({
                    'type': 'VERY_LOW_SAMPLING_RATE',
                    'file': filename,
                    'rate': f"{sampling_rate} Hz",
                    'description': f"Sampling rate of {sampling_rate} Hz is extremely low"
                })
            
            # Check file size vs line count for sampling consistency
            try:
                size_str = dataset['size_mb']
                size = float(size_str.replace(' MB', '').replace(' KB', ''))
                if 'KB' in size_str:
                    size *= 1024
                
                lines_str = dataset['lines']
                if '>' in lines_str:
                    lines = 100000  # Estimate
                else:
                    lines = int(lines_str)
                
                # Calculate expected file size based on lines
                if lines > 0 and size > 0:
                    bytes_per_line = (size * 1024 * 1024) / lines
                    if bytes_per_line > 1000:  # More than 1KB per line
                        self.sampling_issues.append({
                            'type': 'UNUSUAL_BYTES_PER_LINE',
                            'file': filename,
                            'bytes_per_line': f"{bytes_per_line:.1f}",
                            'description': f"Unusually high bytes per line ({bytes_per_line:.1f}) - may indicate data corruption"
                        })
                        
            except:
                pass
    
    def scan_missing_data_patterns(self, data: Dict):
        """Scan for missing data patterns"""
        print("🔍 MISSING DATA PATTERN SCANNING")
        print("=" * 50)
        
        # Check for files with very small sizes
        electrical = data.get('electrical_datasets', [])
        environmental = data.get('environmental_datasets', [])
        
        for dataset in electrical + environmental:
            filename = dataset['filename']
            size_str = dataset['size_mb']
            
            try:
                size = float(size_str.replace(' MB', '').replace(' KB', ''))
                if 'KB' in size_str:
                    size /= 1024
                
                # Check for suspiciously small files
                if size < 0.001:  # Less than 1KB
                    self.missing_data_issues.append({
                        'type': 'SUSPICIOUSLY_SMALL_FILE',
                        'file': filename,
                        'size': f"{size:.6f} MB",
                        'description': f"File size of {size:.6f} MB is suspiciously small"
                    })
                
                # Check for files with very few lines
                lines_str = dataset['lines']
                if lines_str.isdigit() and int(lines_str) < 10:
                    self.missing_data_issues.append({
                        'type': 'VERY_FEW_LINES',
                        'file': filename,
                        'lines': lines_str,
                        'description': f"Only {lines_str} lines - may indicate missing data"
                    })
                    
            except:
                pass
    
    def scan_metadata_anomalies(self, data: Dict):
        """Scan for metadata inconsistencies"""
        print("🏷️ METADATA ANOMALY SCANNING")
        print("=" * 50)
        
        electrical = data.get('electrical_datasets', [])
        
        # Check for inconsistent species naming
        species_variations = {}
        for dataset in electrical:
            species = dataset['metadata']['species']
            if species not in species_variations:
                species_variations[species] = []
            species_variations[species].append(dataset['filename'])
        
        # Check for files with same species but different naming
        for species, files in species_variations.items():
            if len(files) > 1:
                # Check for naming inconsistencies
                base_names = [f.split('_')[0] for f in files]
                if len(set(base_names)) > 1:
                    self.additional_anomalies.append({
                        'type': 'INCONSISTENT_SPECIES_NAMING',
                        'species': species,
                        'files': files,
                        'description': f"Same species '{species}' with inconsistent naming patterns"
                    })
        
        # Check for missing electrode information
        unknown_electrodes = [d for d in electrical if d['metadata']['electrode_type'] == 'Unknown']
        if len(unknown_electrodes) > len(electrical) * 0.4:  # More than 40%
            self.additional_anomalies.append({
                'type': 'MISSING_ELECTRODE_INFO',
                'count': len(unknown_electrodes),
                'total': len(electrical),
                'description': f"{len(unknown_electrodes)}/{len(electrical)} files missing electrode information"
            })
    
    def scan_coordinate_anomalies(self, data: Dict):
        """Scan for coordinate-specific anomalies"""
        print("🗺️ COORDINATE ANOMALY SCANNING")
        print("=" * 50)
        
        coordinate = data.get('coordinate_datasets', [])
        
        # Check for coordinate file naming patterns
        naming_patterns = {}
        for dataset in coordinate:
            filename = dataset['filename']
            # Extract pattern: Pv_L_I+4xR_Fc_N_36d_1_coordinates.csv
            parts = filename.split('_')
            if len(parts) >= 6:
                pattern = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}"
                if pattern not in naming_patterns:
                    naming_patterns[pattern] = []
                naming_patterns[pattern].append(filename)
        
        # Check for incomplete coordinate series
        for pattern, files in naming_patterns.items():
            if len(files) > 1:
                # Check for missing replicates
                replicate_numbers = []
                for file in files:
                    match = re.search(r'_(\d+)_coordinates', file)
                    if match:
                        replicate_numbers.append(int(match.group(1)))
                
                if replicate_numbers:
                    expected_range = range(min(replicate_numbers), max(replicate_numbers) + 1)
                    missing_replicates = [i for i in expected_range if i not in replicate_numbers]
                    
                    if missing_replicates:
                        self.additional_anomalies.append({
                            'type': 'MISSING_COORDINATE_REPLICATES',
                            'pattern': pattern,
                            'missing': missing_replicates,
                            'description': f"Missing coordinate replicates: {missing_replicates}"
                        })
    
    def scan_environmental_anomalies(self, data: Dict):
        """Scan for environmental-specific anomalies"""
        print("🌡️ ENVIRONMENTAL ANOMALY SCANNING")
        print("=" * 50)
        
        environmental = data.get('environmental_datasets', [])
        
        # Check for moisture sensor consistency
        moisture_sensors = {}
        for dataset in environmental:
            if 'moisture_range_m3_m3' in dataset['metadata']:
                min_moist, max_moist = dataset['metadata']['moisture_range_m3_m3']
                sensor_type = dataset['metadata']['sensor_type']
                
                if sensor_type not in moisture_sensors:
                    moisture_sensors[sensor_type] = []
                moisture_sensors[sensor_type].append((min_moist, max_moist))
        
        # Check for sensor calibration issues
        for sensor_type, ranges in moisture_sensors.items():
            if len(ranges) > 1:
                all_min = min([r[0] for r in ranges])
                all_max = max([r[1] for r in ranges])
                
                if all_max - all_min > 0.5:  # More than 0.5 m³/m³ range
                    self.additional_anomalies.append({
                        'type': 'LARGE_MOISTURE_RANGE',
                        'sensor_type': sensor_type,
                        'range': f"{all_min:.4f} to {all_max:.4f} m³/m³",
                        'description': f"Large moisture range across {sensor_type} sensors may indicate calibration issues"
                    })
    
    def generate_comprehensive_report(self):
        """Generate comprehensive anomaly report"""
        print("\n📋 COMPREHENSIVE ANOMALY REPORT")
        print("=" * 60)
        
        all_issues = (
            self.additional_anomalies + 
            self.temporal_issues + 
            self.format_issues + 
            self.sampling_issues + 
            self.missing_data_issues
        )
        
        # Categorize by type
        categories = {}
        for issue in all_issues:
            issue_type = issue['type']
            if issue_type not in categories:
                categories[issue_type] = []
            categories[issue_type].append(issue)
        
        print(f"🔍 Total Additional Issues Found: {len(all_issues)}")
        print(f"📊 Issue Categories: {len(categories)}")
        
        for category, issues in categories.items():
            print(f"\n📌 {category}: {len(issues)} issues")
            for issue in issues[:3]:  # Show first 3 examples
                if 'file' in issue:
                    print(f"   • {issue['file']}: {issue['description']}")
                elif 'description' in issue:
                    print(f"   • {issue['description']}")
        
        # Save detailed report
        self.save_comprehensive_report(all_issues, categories)
        
        return all_issues
    
    def save_comprehensive_report(self, all_issues: List[Dict], categories: Dict):
        """Save comprehensive anomaly report"""
        output_dir = Path("environmental_analysis")
        output_dir.mkdir(exist_ok=True)
        
        report_file = output_dir / "comprehensive_anomaly_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Comprehensive Anomaly Detection Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Additional Issues**: {len(all_issues)}\n")
            f.write(f"- **Issue Categories**: {len(categories)}\n\n")
            
            f.write("## Issue Categories\n\n")
            for category, issues in categories.items():
                f.write(f"### {category}\n")
                f.write(f"**Count**: {len(issues)}\n\n")
                
                for issue in issues:
                    f.write(f"- **Description**: {issue.get('description', 'No description')}\n")
                    if 'file' in issue:
                        f.write(f"  - **File**: {issue['file']}\n")
                    if 'value' in issue:
                        f.write(f"  - **Value**: {issue['value']}\n")
                    f.write("\n")
        
        print(f"📄 Comprehensive report saved: {report_file}")
    
    def run_comprehensive_scan(self):
        """Run comprehensive anomaly scanning"""
        print("🔍 COMPREHENSIVE ANOMALY SCANNING")
        print("=" * 80)
        
        # Load data
        data = self.load_data()
        if not data:
            print("❌ Could not load extraction data")
            return
        
        # Run all scanning methods
        self.scan_temporal_anomalies(data)
        self.scan_format_anomalies(data)
        self.scan_sampling_anomalies(data)
        self.scan_missing_data_patterns(data)
        self.scan_metadata_anomalies(data)
        self.scan_coordinate_anomalies(data)
        self.scan_environmental_anomalies(data)
        
        # Generate report
        all_issues = self.generate_comprehensive_report()
        
        return all_issues

def main():
    scanner = ComprehensiveAnomalyScanner()
    issues = scanner.run_comprehensive_scan()
    
    print(f"\n" + "=" * 80)
    print("✅ COMPREHENSIVE ANOMALY SCANNING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main() 