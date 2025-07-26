#!/usr/bin/env python3
"""
Anomaly Detection Analysis for Fungal Electrical Activity Datasets
Identifies potential issues that could compromise simulation accuracy
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

class AnomalyDetector:
    """Detects anomalies in environmental parameter extraction results"""
    
    def __init__(self):
        self.adamatzky_ranges = {
            'amplitude_mv': (0.16, 0.5),
            'sampling_rate_hz': (0.001, 0.1),
            'moisture_m3_m3': (-0.2, 0.3),
            'temperature_c': (15, 30),
            'duration_hours': (0.5, 5000)
        }
        
        self.critical_anomalies = []
        self.warnings = []
        self.recommendations = []
    
    def load_data(self) -> Dict:
        """Load the extraction results"""
        json_file = Path("environmental_analysis/environmental_parameters_20250718_021033.json")
        if json_file.exists():
            with open(json_file, 'r') as f:
                return json.load(f)
        return {}
    
    def detect_electrical_anomalies(self, data: Dict) -> List[Dict]:
        """Detect anomalies in electrical activity datasets"""
        anomalies = []
        electrical = data.get('electrical_datasets', [])
        
        print("🔌 ELECTRICAL ACTIVITY ANOMALY DETECTION")
        print("=" * 60)
        
        for dataset in electrical:
            filename = dataset['filename']
            metadata = dataset['metadata']
            validation = dataset['validation']
            
            # Check for extreme amplitude values
            if 'amplitude_range_mv' in metadata:
                min_amp, max_amp = metadata['amplitude_range_mv']
                
                # Anomaly: Extreme values
                if abs(max_amp) > 1000 or abs(min_amp) > 1000:
                    anomalies.append({
                        'type': 'EXTREME_AMPLITUDE',
                        'severity': 'CRITICAL',
                        'file': filename,
                        'value': f"{min_amp:.6f} to {max_amp:.6f} mV",
                        'description': f"Amplitude range {min_amp:.6f} to {max_amp:.6f} mV is biologically impossible"
                    })
                
                # Anomaly: Negative values (should be positive for fungal electrical activity)
                if min_amp < 0:
                    anomalies.append({
                        'type': 'NEGATIVE_AMPLITUDE',
                        'severity': 'WARNING',
                        'file': filename,
                        'value': f"{min_amp:.6f} mV",
                        'description': f"Negative amplitude {min_amp:.6f} mV may indicate measurement error"
                    })
                
                # Anomaly: Zero or near-zero values
                if abs(max_amp) < 0.001 and abs(min_amp) < 0.001:
                    anomalies.append({
                        'type': 'ZERO_AMPLITUDE',
                        'severity': 'WARNING',
                        'file': filename,
                        'value': f"{min_amp:.6f} to {max_amp:.6f} mV",
                        'description': f"Near-zero amplitudes may indicate sensor failure or no activity"
                    })
            
            # Check for missing electrode information
            if metadata.get('electrode_type') == 'Unknown':
                anomalies.append({
                    'type': 'UNKNOWN_ELECTRODE',
                    'severity': 'MEDIUM',
                    'file': filename,
                    'value': 'Unknown',
                    'description': "Electrode type unknown - may affect signal interpretation"
                })
            
            # Check for unrealistic channel counts
            channels = metadata.get('channels', 0)
            if channels == 0 and 'differential' in filename.lower():
                anomalies.append({
                    'type': 'MISSING_CHANNELS',
                    'severity': 'MEDIUM',
                    'file': filename,
                    'value': f"{channels} channels",
                    'description': f"Expected differential electrodes but found {channels} channels"
                })
        
        return anomalies
    
    def detect_environmental_anomalies(self, data: Dict) -> List[Dict]:
        """Detect anomalies in environmental datasets"""
        anomalies = []
        environmental = data.get('environmental_datasets', [])
        
        print("\n🌡️  ENVIRONMENTAL ANOMALY DETECTION")
        print("=" * 60)
        
        for dataset in environmental:
            filename = dataset['filename']
            metadata = dataset['metadata']
            
            # Check for extreme moisture values
            if 'moisture_range_m3_m3' in metadata:
                min_moist, max_moist = metadata['moisture_range_m3_m3']
                
                # Anomaly: Extreme negative moisture (below -0.3)
                if min_moist < -0.3:
                    anomalies.append({
                        'type': 'EXTREME_DRY',
                        'severity': 'WARNING',
                        'file': filename,
                        'value': f"{min_moist:.4f} m³/m³",
                        'description': f"Extremely dry condition {min_moist:.4f} m³/m³ may be sensor error"
                    })
                
                # Anomaly: Extreme positive moisture (above 0.3)
                if max_moist > 0.3:
                    anomalies.append({
                        'type': 'EXTREME_WET',
                        'severity': 'WARNING',
                        'file': filename,
                        'value': f"{max_moist:.4f} m³/m³",
                        'description': f"Extremely wet condition {max_moist:.4f} m³/m³ may be sensor error"
                    })
                
                # Anomaly: No moisture variation
                if abs(max_moist - min_moist) < 0.001:
                    anomalies.append({
                        'type': 'STATIC_MOISTURE',
                        'severity': 'MEDIUM',
                        'file': filename,
                        'value': f"{min_moist:.4f} to {max_moist:.4f} m³/m³",
                        'description': "No moisture variation detected - may indicate sensor malfunction"
                    })
        
        return anomalies
    
    def detect_coordinate_anomalies(self, data: Dict) -> List[Dict]:
        """Detect anomalies in coordinate datasets"""
        anomalies = []
        coordinate = data.get('coordinate_datasets', [])
        
        print("\n🗺️  COORDINATE ANOMALY DETECTION")
        print("=" * 60)
        
        # Analyze coordinate patterns
        coordinate_values = []
        for dataset in coordinate:
            filename = dataset['filename']
            
            # Check for empty metadata (should have species/duration info)
            if not dataset.get('metadata'):
                anomalies.append({
                    'type': 'EMPTY_METADATA',
                    'severity': 'LOW',
                    'file': filename,
                    'value': 'No metadata',
                    'description': "Coordinate file missing parsed metadata"
                })
            
            # Check file size anomalies
            size_str = dataset['size_mb']
            try:
                size = float(size_str.replace(' MB', '').replace(' KB', ''))
                if 'KB' in size_str:
                    size /= 1024
                coordinate_values.append(size)
            except:
                pass
        
        # Check for size anomalies
        if coordinate_values:
            mean_size = np.mean(coordinate_values)
            std_size = np.std(coordinate_values)
            
            for dataset in coordinate:
                size_str = dataset['size_mb']
                try:
                    size = float(size_str.replace(' MB', '').replace(' KB', ''))
                    if 'KB' in size_str:
                        size /= 1024
                    
                    # Anomaly: Unusually large coordinate files
                    if size > mean_size + 2 * std_size:
                        anomalies.append({
                            'type': 'LARGE_COORDINATE_FILE',
                            'severity': 'MEDIUM',
                            'file': dataset['filename'],
                            'value': f"{size:.3f} MB",
                            'description': f"Unusually large coordinate file ({size:.3f} MB vs mean {mean_size:.3f} MB)"
                        })
                except:
                    pass
        
        return anomalies
    
    def detect_data_quality_issues(self, data: Dict) -> List[Dict]:
        """Detect general data quality issues"""
        issues = []
        
        print("\n🔍 DATA QUALITY ISSUES")
        print("=" * 60)
        
        # Check for missing species information
        electrical = data.get('electrical_datasets', [])
        unknown_species = [d for d in electrical if d['metadata']['species'] == 'Unknown']
        
        if len(unknown_species) > len(electrical) * 0.5:
            issues.append({
                'type': 'MISSING_SPECIES_INFO',
                'severity': 'HIGH',
                'files_affected': len(unknown_species),
                'description': f"{len(unknown_species)}/{len(electrical)} electrical files have unknown species"
            })
        
        # Check for file size inconsistencies
        electrical_sizes = []
        for dataset in electrical:
            size_str = dataset['size_mb']
            try:
                size = float(size_str.replace(' MB', '').replace(' KB', ''))
                if 'KB' in size_str:
                    size /= 1024
                electrical_sizes.append(size)
            except:
                pass
        
        if electrical_sizes:
            # Check for unusually large files
            large_files = [s for s in electrical_sizes if s > 50]  # >50MB
            if large_files:
                issues.append({
                    'type': 'LARGE_ELECTRICAL_FILES',
                    'severity': 'MEDIUM',
                    'files_affected': len(large_files),
                    'description': f"{len(large_files)} files exceed 50MB - may cause memory issues"
                })
        
        return issues
    
    def detect_simulation_compromises(self, data: Dict) -> List[Dict]:
        """Detect issues that could compromise simulation accuracy"""
        compromises = []
        
        print("\n⚠️  SIMULATION COMPROMISE DETECTION")
        print("=" * 60)
        
        # Check biological plausibility
        electrical = data.get('electrical_datasets', [])
        biologically_plausible = [d for d in electrical if d['simulation_ready']]
        
        if len(biologically_plausible) == 0:
            compromises.append({
                'type': 'NO_BIOLOGICALLY_PLAUSIBLE_ELECTRICAL',
                'severity': 'CRITICAL',
                'description': "No electrical datasets meet Adamatzky's biological criteria",
                'impact': "Simulation will not reflect real fungal electrical activity",
                'solution': "Amplitude normalization required for all electrical data"
            })
        
        # Check parameter range consistency
        ranges = data.get('parameter_ranges', {})
        if 'amplitude_mv' in ranges:
            min_amp, max_amp = ranges['amplitude_mv']
            if max_amp > 1000:
                compromises.append({
                    'type': 'EXTREME_AMPLITUDE_RANGE',
                    'severity': 'CRITICAL',
                    'description': f"Amplitude range {min_amp:.6f} to {max_amp:.6f} mV is biologically impossible",
                    'impact': "Simulation parameters will be unrealistic",
                    'solution': "Implement amplitude clipping and normalization"
                })
        
        # Check environmental coverage
        environmental = data.get('environmental_datasets', [])
        if len(environmental) < 5:
            compromises.append({
                'type': 'LIMITED_ENVIRONMENTAL_DATA',
                'severity': 'MEDIUM',
                'description': f"Only {len(environmental)} environmental datasets available",
                'impact': "Limited environmental parameter coverage for simulation",
                'solution': "Consider additional environmental monitoring data"
            })
        
        return compromises
    
    def generate_anomaly_report(self, all_anomalies: List[Dict], issues: List[Dict], compromises: List[Dict]):
        """Generate comprehensive anomaly report"""
        print("\n📋 ANOMALY REPORT SUMMARY")
        print("=" * 60)
        
        # Count by severity
        critical = [a for a in all_anomalies if a['severity'] == 'CRITICAL']
        warnings = [a for a in all_anomalies if a['severity'] == 'WARNING']
        medium = [a for a in all_anomalies if a['severity'] == 'MEDIUM']
        low = [a for a in all_anomalies if a['severity'] == 'LOW']
        
        print(f"🔴 Critical Anomalies: {len(critical)}")
        print(f"🟡 Warnings: {len(warnings)}")
        print(f"🟠 Medium Issues: {len(medium)}")
        print(f"🟢 Low Issues: {len(low)}")
        print(f"⚠️  Data Quality Issues: {len(issues)}")
        print(f"🚨 Simulation Compromises: {len(compromises)}")
        
        # Show critical anomalies
        if critical:
            print(f"\n🔴 CRITICAL ANOMALIES:")
            for anomaly in critical[:5]:  # Show top 5
                print(f"   • {anomaly['type']}: {anomaly['file']}")
                print(f"     {anomaly['description']}")
        
        # Show simulation compromises
        if compromises:
            print(f"\n🚨 SIMULATION COMPROMISES:")
            for compromise in compromises:
                print(f"   • {compromise['type']}")
                print(f"     Impact: {compromise['impact']}")
                print(f"     Solution: {compromise['solution']}")
        
        return {
            'critical_count': len(critical),
            'warning_count': len(warnings),
            'medium_count': len(medium),
            'low_count': len(low),
            'quality_issues': len(issues),
            'compromises': len(compromises),
            'critical_anomalies': critical,
            'simulation_compromises': compromises
        }
    
    def run_analysis(self):
        """Run complete anomaly detection analysis"""
        print("🔍 ANOMALY DETECTION ANALYSIS")
        print("=" * 80)
        
        # Load data
        data = self.load_data()
        if not data:
            print("❌ Could not load extraction data")
            return
        
        # Run all anomaly detection
        electrical_anomalies = self.detect_electrical_anomalies(data)
        environmental_anomalies = self.detect_environmental_anomalies(data)
        coordinate_anomalies = self.detect_coordinate_anomalies(data)
        quality_issues = self.detect_data_quality_issues(data)
        simulation_compromises = self.detect_simulation_compromises(data)
        
        # Combine all anomalies
        all_anomalies = electrical_anomalies + environmental_anomalies + coordinate_anomalies
        
        # Generate report
        report = self.generate_anomaly_report(all_anomalies, quality_issues, simulation_compromises)
        
        # Save detailed report
        self.save_detailed_report(all_anomalies, quality_issues, simulation_compromises, report)
        
        return report
    
    def save_detailed_report(self, anomalies: List[Dict], issues: List[Dict], compromises: List[Dict], summary: Dict):
        """Save detailed anomaly report"""
        output_dir = Path("environmental_analysis")
        output_dir.mkdir(exist_ok=True)
        
        report_file = output_dir / "anomaly_detection_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Anomaly Detection Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Critical Anomalies**: {summary['critical_count']}\n")
            f.write(f"- **Warnings**: {summary['warning_count']}\n")
            f.write(f"- **Medium Issues**: {summary['medium_count']}\n")
            f.write(f"- **Low Issues**: {summary['low_count']}\n")
            f.write(f"- **Data Quality Issues**: {summary['quality_issues']}\n")
            f.write(f"- **Simulation Compromises**: {summary['compromises']}\n\n")
            
            if summary['critical_anomalies']:
                f.write("## Critical Anomalies\n\n")
                for anomaly in summary['critical_anomalies']:
                    f.write(f"### {anomaly['type']}\n")
                    f.write(f"- **File**: {anomaly['file']}\n")
                    f.write(f"- **Value**: {anomaly['value']}\n")
                    f.write(f"- **Description**: {anomaly['description']}\n\n")
            
            if summary['simulation_compromises']:
                f.write("## Simulation Compromises\n\n")
                for compromise in summary['simulation_compromises']:
                    f.write(f"### {compromise['type']}\n")
                    f.write(f"- **Impact**: {compromise['impact']}\n")
                    f.write(f"- **Solution**: {compromise['solution']}\n\n")
        
        print(f"📄 Detailed report saved: {report_file}")

def main():
    detector = AnomalyDetector()
    report = detector.run_analysis()
    
    print(f"\n" + "=" * 80)
    print("✅ ANOMALY DETECTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main() 