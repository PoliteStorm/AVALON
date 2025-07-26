#!/usr/bin/env python3
"""
Environmental Parameter Extraction for Fungal Electrical Activity Simulation
Extracts and analyzes environmental parameters from profiled CSV datasets
"""

import pandas as pd
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

class EnvironmentalParameterExtractor:
    """
    Extracts environmental parameters from CSV datasets for simulation input
    """
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("environmental_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Adamatzky's biological ranges for validation
        self.adamatzky_ranges = {
            'amplitude_mv': (0.16, 0.5),  # mV range from Adamatzky (2022)
            'sampling_rate_hz': (0.001, 0.1),  # Hz range for fungal electrical activity
            'moisture_m3_m3': (-0.2, 0.3),  # Typical soil moisture range
            'temperature_c': (15, 30),  # Typical fungal growth temperature
            'duration_hours': (0.5, 5000)  # Hours (0.5h to 208d)
        }
        
        # Species mapping from filenames
        self.species_mapping = {
            'Pv': 'Pleurotus ostreatus (Oyster)',
            'Pi': 'Pleurotus ostreatus variant',
            'Pp': 'Pleurotus pulmonarius (Phoenix oyster)',
            'Rb': 'Rubus species (Raspberry/Blackberry)',
            'Ag': 'Agaricus species',
            'Sc': 'Schizophyllum commune',
            'Hericium': 'Hericium erinaceus (Lion\'s Mane)',
            'Oyster': 'Pleurotus ostreatus',
            'Lions': 'Hericium erinaceus'
        }
        
    def load_profile_data(self) -> pd.DataFrame:
        """Load the CSV profile data"""
        try:
            df = pd.read_csv('csv_profile_summary.csv')
            print(f"✅ Loaded {len(df)} profiled files")
            return df
        except FileNotFoundError:
            print("❌ csv_profile_summary.csv not found. Run profile_all_csvs.py first.")
            return pd.DataFrame()
    
    def classify_files(self, df: pd.DataFrame) -> Dict[str, List]:
        """Classify files into categories"""
        classifications = {
            'electrical_activity': [],
            'environmental_logs': [],
            'coordinate_data': [],
            'unknown': []
        }
        
        for _, row in df.iterrows():
            filename = row['file']
            directory = row['directory']
            columns = str(row['columns'])
            preview = str(row['preview'])
            
            # Classify based on filename patterns and content
            if 'coordinates' in filename:
                classifications['coordinate_data'].append(row)
            elif any(keyword in filename.lower() for keyword in ['moisture', 'logger', 'gl']):
                classifications['environmental_logs'].append(row)
            elif any(keyword in columns.lower() for keyword in ['differential', 'voltage', 'mv', 'v']):
                classifications['electrical_activity'].append(row)
            elif any(keyword in preview.lower() for keyword in ['water content', 'moisture']):
                classifications['environmental_logs'].append(row)
            else:
                classifications['unknown'].append(row)
        
        return classifications
    
    def extract_coordinate_metadata(self, filename: str) -> Dict:
        """Extract metadata from coordinate filenames"""
        # Example: Pv_L_I+4xR_Fc_N_36d_1_coordinates.csv
        pattern = r'([A-Za-z]+)_([A-Za-z0-9\+x]+)_([A-Za-z]+)_([A-Za-z]+)_([\d]+[dh])_([\d]+)_coordinates'
        match = re.match(pattern, filename)
        
        if match:
            return {
                'species': match.group(1),
                'variant': match.group(2),
                'unknown1': match.group(3),
                'unknown2': match.group(4),
                'duration': match.group(5),
                'replicate': match.group(6),
                'species_full': self.species_mapping.get(match.group(1), match.group(1))
            }
        return {}
    
    def extract_electrical_metadata(self, filename: str, columns: str, preview: str) -> Dict:
        """Extract metadata from electrical activity files"""
        metadata = {
            'species': 'Unknown',
            'electrode_type': 'Unknown',
            'sampling_rate_hz': 1.0,  # Default
            'channels': 0,
            'amplitude_range_mv': (0, 0)
        }
        
        # Extract species from filename
        for species_code, full_name in self.species_mapping.items():
            if species_code.lower() in filename.lower():
                metadata['species'] = full_name
                break
        
        # Extract electrode information
        if 'differential' in columns.lower():
            metadata['electrode_type'] = 'Differential electrodes'
            # Count channels from column names
            channel_count = columns.count('Differential')
            metadata['channels'] = channel_count
        
        # Extract amplitude range from preview
        try:
            values = []
            for line in preview.split('|'):
                for val in line.split(','):
                    try:
                        val = float(val.strip().replace('"', ''))
                        values.append(val)
                    except:
                        continue
            
            if values:
                metadata['amplitude_range_mv'] = (min(values), max(values))
        except:
            pass
        
        return metadata
    
    def extract_environmental_metadata(self, filename: str, preview: str) -> Dict:
        """Extract environmental parameters from moisture logs"""
        metadata = {
            'sensor_type': 'Moisture logger',
            'moisture_range_m3_m3': (0, 0),
            'sampling_interval_seconds': 1,
            'location': 'Unknown'
        }
        
        # Extract moisture range from preview
        try:
            moisture_values = []
            for line in preview.split('|'):
                if 'Water Content' in line or any(char.isdigit() for char in line):
                    parts = line.split(',')
                    for part in parts:
                        try:
                            val = float(part.strip().replace('"', ''))
                            if -1 < val < 1:  # Reasonable moisture range
                                moisture_values.append(val)
                        except:
                            continue
            
            if moisture_values:
                metadata['moisture_range_m3_m3'] = (min(moisture_values), max(moisture_values))
        except:
            pass
        
        # Extract location from filename
        if 'blue' in filename.lower():
            metadata['location'] = 'Blue oyster experiment'
        elif 'dry' in filename.lower():
            metadata['location'] = 'Dry condition'
        
        return metadata
    
    def validate_biological_plausibility(self, metadata: Dict, file_type: str) -> Dict:
        """Validate parameters against Adamatzky's biological ranges"""
        validation = {
            'is_biologically_plausible': True,
            'warnings': [],
            'recommendations': []
        }
        
        if file_type == 'electrical_activity':
            # Check amplitude range
            if 'amplitude_range_mv' in metadata:
                min_amp, max_amp = metadata['amplitude_range_mv']
                if min_amp < self.adamatzky_ranges['amplitude_mv'][0] or max_amp > self.adamatzky_ranges['amplitude_mv'][1]:
                    validation['warnings'].append(f"Amplitude range ({min_amp:.3f}, {max_amp:.3f}) mV outside Adamatzky range {self.adamatzky_ranges['amplitude_mv']}")
                    validation['is_biologically_plausible'] = False
        
        elif file_type == 'environmental_logs':
            # Check moisture range
            if 'moisture_range_m3_m3' in metadata:
                min_moist, max_moist = metadata['moisture_range_m3_m3']
                if min_moist < self.adamatzky_ranges['moisture_m3_m3'][0] or max_moist > self.adamatzky_ranges['moisture_m3_m3'][1]:
                    validation['warnings'].append(f"Moisture range ({min_moist:.3f}, {max_moist:.3f}) outside expected range {self.adamatzky_ranges['moisture_m3_m3']}")
        
        return validation
    
    def create_simulation_parameters(self, classifications: Dict) -> Dict:
        """Create simulation-ready parameter sets"""
        simulation_params = {
            'electrical_datasets': [],
            'environmental_datasets': [],
            'coordinate_datasets': [],
            'species_summary': {},
            'parameter_ranges': {},
            'recommendations': []
        }
        
        # Process electrical activity files
        for file_data in classifications['electrical_activity']:
            metadata = self.extract_electrical_metadata(
                file_data['file'], 
                file_data['columns'], 
                file_data['preview']
            )
            validation = self.validate_biological_plausibility(metadata, 'electrical_activity')
            
            simulation_params['electrical_datasets'].append({
                'filename': file_data['file'],
                'size_mb': file_data['size'],
                'lines': file_data['lines'],
                'metadata': metadata,
                'validation': validation,
                'simulation_ready': validation['is_biologically_plausible']
            })
        
        # Process environmental logs
        for file_data in classifications['environmental_logs']:
            metadata = self.extract_environmental_metadata(
                file_data['file'], 
                file_data['preview']
            )
            validation = self.validate_biological_plausibility(metadata, 'environmental_logs')
            
            simulation_params['environmental_datasets'].append({
                'filename': file_data['file'],
                'size_mb': file_data['size'],
                'lines': file_data['lines'],
                'metadata': metadata,
                'validation': validation,
                'simulation_ready': True  # Environmental data is always useful
            })
        
        # Process coordinate data
        for file_data in classifications['coordinate_data']:
            metadata = self.extract_coordinate_metadata(file_data['file'])
            
            simulation_params['coordinate_datasets'].append({
                'filename': file_data['file'],
                'size_mb': file_data['size'],
                'lines': file_data['lines'],
                'metadata': metadata,
                'simulation_ready': True  # Coordinate data is always useful
            })
        
        # Create species summary
        species_counts = {}
        for dataset in simulation_params['electrical_datasets']:
            species = dataset['metadata']['species']
            species_counts[species] = species_counts.get(species, 0) + 1
        
        simulation_params['species_summary'] = species_counts
        
        # Calculate parameter ranges
        all_amplitudes = []
        all_moistures = []
        
        for dataset in simulation_params['electrical_datasets']:
            if 'amplitude_range_mv' in dataset['metadata']:
                all_amplitudes.extend(dataset['metadata']['amplitude_range_mv'])
        
        for dataset in simulation_params['environmental_datasets']:
            if 'moisture_range_m3_m3' in dataset['metadata']:
                all_moistures.extend(dataset['metadata']['moisture_range_m3_m3'])
        
        simulation_params['parameter_ranges'] = {
            'amplitude_mv': (min(all_amplitudes), max(all_amplitudes)) if all_amplitudes else None,
            'moisture_m3_m3': (min(all_moistures), max(all_moistures)) if all_moistures else None
        }
        
        return simulation_params
    
    def generate_reports(self, simulation_params: Dict):
        """Generate comprehensive reports"""
        
        # JSON report
        json_path = self.output_dir / f"environmental_parameters_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(simulation_params, f, indent=2)
        
        # Markdown report
        md_path = self.output_dir / f"environmental_parameters_{self.timestamp}.md"
        with open(md_path, 'w') as f:
            f.write("# Environmental Parameter Analysis Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Electrical Activity Files**: {len(simulation_params['electrical_datasets'])}\n")
            f.write(f"- **Environmental Logs**: {len(simulation_params['environmental_datasets'])}\n")
            f.write(f"- **Coordinate Datasets**: {len(simulation_params['coordinate_datasets'])}\n")
            f.write(f"- **Total Files**: {len(simulation_params['electrical_datasets']) + len(simulation_params['environmental_datasets']) + len(simulation_params['coordinate_datasets'])}\n\n")
            
            f.write("## Species Distribution\n\n")
            for species, count in simulation_params['species_summary'].items():
                f.write(f"- **{species}**: {count} files\n")
            f.write("\n")
            
            f.write("## Parameter Ranges\n\n")
            if simulation_params['parameter_ranges']['amplitude_mv']:
                min_amp, max_amp = simulation_params['parameter_ranges']['amplitude_mv']
                f.write(f"- **Amplitude Range**: {min_amp:.3f} to {max_amp:.3f} mV\n")
            if simulation_params['parameter_ranges']['moisture_m3_m3']:
                min_moist, max_moist = simulation_params['parameter_ranges']['moisture_m3_m3']
                f.write(f"- **Moisture Range**: {min_moist:.3f} to {max_moist:.3f} m³/m³\n")
            f.write("\n")
            
            f.write("## Simulation-Ready Datasets\n\n")
            ready_electrical = [d for d in simulation_params['electrical_datasets'] if d['simulation_ready']]
            f.write(f"- **Biologically Plausible Electrical**: {len(ready_electrical)} files\n")
            f.write(f"- **Environmental Logs**: {len(simulation_params['environmental_datasets'])} files\n")
            f.write(f"- **Coordinate Data**: {len(simulation_params['coordinate_datasets'])} files\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **Priority Electrical Datasets**:\n")
            for dataset in ready_electrical[:5]:  # Top 5
                f.write(f"   - {dataset['filename']} ({dataset['metadata']['species']})\n")
            f.write("\n")
            
            f.write("2. **Environmental Parameters**:\n")
            for dataset in simulation_params['environmental_datasets'][:3]:  # Top 3
                f.write(f"   - {dataset['filename']} ({dataset['metadata']['sensor_type']})\n")
            f.write("\n")
        
        print(f"✅ Reports generated:")
        print(f"   📄 {json_path}")
        print(f"   📄 {md_path}")
        
        return json_path, md_path
    
    def run_analysis(self):
        """Run the complete environmental parameter analysis"""
        print("🔬 ENVIRONMENTAL PARAMETER EXTRACTION")
        print("=" * 50)
        
        # Load profile data
        df = self.load_profile_data()
        if df.empty:
            return
        
        # Classify files
        print("📊 Classifying files...")
        classifications = self.classify_files(df)
        
        print(f"   ✅ Electrical Activity: {len(classifications['electrical_activity'])} files")
        print(f"   ✅ Environmental Logs: {len(classifications['environmental_logs'])} files")
        print(f"   ✅ Coordinate Data: {len(classifications['coordinate_data'])} files")
        print(f"   ⚠️  Unknown: {len(classifications['unknown'])} files")
        
        # Create simulation parameters
        print("\n🔧 Creating simulation parameters...")
        simulation_params = self.create_simulation_parameters(classifications)
        
        # Generate reports
        print("\n📋 Generating reports...")
        json_path, md_path = self.generate_reports(simulation_params)
        
        # Summary
        ready_count = len([d for d in simulation_params['electrical_datasets'] if d['simulation_ready']])
        total_electrical = len(simulation_params['electrical_datasets'])
        
        print("\n" + "=" * 50)
        print("📈 ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"✅ Biologically plausible electrical datasets: {ready_count}/{total_electrical}")
        print(f"✅ Environmental parameter datasets: {len(simulation_params['environmental_datasets'])}")
        print(f"✅ Coordinate/spatial datasets: {len(simulation_params['coordinate_datasets'])}")
        print(f"✅ Total simulation-ready datasets: {ready_count + len(simulation_params['environmental_datasets']) + len(simulation_params['coordinate_datasets'])}")
        
        if simulation_params['parameter_ranges']['amplitude_mv']:
            min_amp, max_amp = simulation_params['parameter_ranges']['amplitude_mv']
            print(f"📊 Amplitude range: {min_amp:.3f} to {max_amp:.3f} mV")
        
        if simulation_params['parameter_ranges']['moisture_m3_m3']:
            min_moist, max_moist = simulation_params['parameter_ranges']['moisture_m3_m3']
            print(f"💧 Moisture range: {min_moist:.3f} to {max_moist:.3f} m³/m³")
        
        print("\n🎯 Ready for simulation integration!")

def main():
    extractor = EnvironmentalParameterExtractor()
    extractor.run_analysis()

if __name__ == "__main__":
    main() 