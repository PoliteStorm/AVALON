#!/usr/bin/env python3
"""
Detailed Summary of Environmental Parameter Extraction Results
Shows comprehensive analysis of the extracted parameters and their implications
"""

import json
import pandas as pd
from pathlib import Path

def load_extraction_results():
    """Load the environmental parameter extraction results"""
    json_file = Path("environmental_analysis/environmental_parameters_20250718_021033.json")
    if json_file.exists():
        with open(json_file, 'r') as f:
            return json.load(f)
    return None

def analyze_electrical_datasets(data):
    """Analyze electrical activity datasets"""
    print("🔌 ELECTRICAL ACTIVITY DATASETS ANALYSIS")
    print("=" * 60)
    
    electrical = data['electrical_datasets']
    print(f"Total electrical files: {len(electrical)}")
    
    # Group by species
    species_groups = {}
    for dataset in electrical:
        species = dataset['metadata']['species']
        if species not in species_groups:
            species_groups[species] = []
        species_groups[species].append(dataset)
    
    print(f"\n📊 Species Distribution:")
    for species, datasets in species_groups.items():
        print(f"   • {species}: {len(datasets)} files")
    
    # Analyze amplitude ranges
    print(f"\n📈 Amplitude Analysis:")
    all_amplitudes = []
    for dataset in electrical:
        if 'amplitude_range_mv' in dataset['metadata']:
            min_amp, max_amp = dataset['metadata']['amplitude_range_mv']
            all_amplitudes.extend([min_amp, max_amp])
    
    if all_amplitudes:
        print(f"   • Global range: {min(all_amplitudes):.6f} to {max(all_amplitudes):.6f} mV")
        print(f"   • Adamatzky expected range: 0.16 to 0.5 mV")
        
        # Find closest to Adamatzky range
        closest_to_adamatzky = []
        for dataset in electrical:
            if 'amplitude_range_mv' in dataset['metadata']:
                min_amp, max_amp = dataset['metadata']['amplitude_range_mv']
                # Check if any part overlaps with Adamatzky range
                if (min_amp <= 0.5 and max_amp >= 0.16):
                    closest_to_adamatzky.append(dataset)
        
        print(f"   • Files with Adamatzky-compatible ranges: {len(closest_to_adamatzky)}")
        
        if closest_to_adamatzky:
            print(f"   • Best candidates:")
            for dataset in closest_to_adamatzky[:3]:
                min_amp, max_amp = dataset['metadata']['amplitude_range_mv']
                print(f"     - {dataset['filename']}: {min_amp:.6f} to {max_amp:.6f} mV")
    
    # Channel analysis
    print(f"\n🔌 Electrode Configuration:")
    channel_counts = {}
    for dataset in electrical:
        channels = dataset['metadata']['channels']
        channel_counts[channels] = channel_counts.get(channels, 0) + 1
    
    for channels, count in sorted(channel_counts.items()):
        print(f"   • {channels} channels: {count} files")
    
    # File size analysis
    print(f"\n💾 File Size Analysis:")
    sizes = []
    for dataset in electrical:
        size_str = dataset['size_mb']
        try:
            size = float(size_str.replace(' MB', '').replace(' KB', ''))
            if 'KB' in size_str:
                size /= 1024
            sizes.append(size)
        except:
            continue
    
    if sizes:
        print(f"   • Average size: {sum(sizes)/len(sizes):.2f} MB")
        print(f"   • Largest: {max(sizes):.2f} MB")
        print(f"   • Smallest: {min(sizes):.2f} MB")

def analyze_environmental_datasets(data):
    """Analyze environmental datasets"""
    print(f"\n🌡️  ENVIRONMENTAL DATASETS ANALYSIS")
    print("=" * 60)
    
    environmental = data['environmental_datasets']
    print(f"Total environmental files: {len(environmental)}")
    
    # Moisture analysis
    print(f"\n💧 Moisture Analysis:")
    moisture_ranges = []
    for dataset in environmental:
        if 'moisture_range_m3_m3' in dataset['metadata']:
            min_moist, max_moist = dataset['metadata']['moisture_range_m3_m3']
            moisture_ranges.append((min_moist, max_moist))
    
    if moisture_ranges:
        all_min = min([r[0] for r in moisture_ranges])
        all_max = max([r[1] for r in moisture_ranges])
        print(f"   • Global moisture range: {all_min:.4f} to {all_max:.4f} m³/m³")
        
        # Categorize by moisture levels
        dry_files = []
        wet_files = []
        for dataset in environmental:
            if 'moisture_range_m3_m3' in dataset['metadata']:
                min_moist, max_moist = dataset['metadata']['moisture_range_m3_m3']
                if max_moist < 0:
                    dry_files.append(dataset)
                else:
                    wet_files.append(dataset)
        
        print(f"   • Dry condition files: {len(dry_files)}")
        print(f"   • Wet condition files: {len(wet_files)}")
        
        if dry_files:
            print(f"   • Dry condition examples:")
            for dataset in dry_files[:3]:
                min_moist, max_moist = dataset['metadata']['moisture_range_m3_m3']
                print(f"     - {dataset['filename']}: {min_moist:.4f} to {max_moist:.4f} m³/m³")
        
        if wet_files:
            print(f"   • Wet condition examples:")
            for dataset in wet_files[:3]:
                min_moist, max_moist = dataset['metadata']['moisture_range_m3_m3']
                print(f"     - {dataset['filename']}: {min_moist:.4f} to {max_moist:.4f} m³/m³")
    
    # Location analysis
    print(f"\n📍 Location Analysis:")
    locations = {}
    for dataset in environmental:
        location = dataset['metadata']['location']
        locations[location] = locations.get(location, 0) + 1
    
    for location, count in locations.items():
        print(f"   • {location}: {count} files")

def analyze_coordinate_datasets(data):
    """Analyze coordinate datasets"""
    print(f"\n🗺️  COORDINATE DATASETS ANALYSIS")
    print("=" * 60)
    
    coordinate = data['coordinate_datasets']
    print(f"Total coordinate files: {len(coordinate)}")
    
    # Extract metadata from filenames
    species_counts = {}
    duration_counts = {}
    
    for dataset in coordinate:
        filename = dataset['filename']
        # Parse filename pattern: Pv_L_I+4xR_Fc_N_36d_1_coordinates.csv
        parts = filename.split('_')
        if len(parts) >= 6:
            species = parts[0]
            duration = parts[4]  # e.g., "36d"
            
            species_counts[species] = species_counts.get(species, 0) + 1
            duration_counts[duration] = duration_counts.get(duration, 0) + 1
    
    print(f"\n🍄 Species Distribution:")
    for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {species}: {count} files")
    
    print(f"\n⏱️  Duration Distribution:")
    for duration, count in sorted(duration_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   • {duration}: {count} files")
    
    # File size analysis
    print(f"\n💾 Coordinate File Sizes:")
    sizes = []
    for dataset in coordinate:
        size_str = dataset['size_mb']
        try:
            size = float(size_str.replace(' MB', '').replace(' KB', ''))
            if 'KB' in size_str:
                size /= 1024
            sizes.append(size)
        except:
            continue
    
    if sizes:
        print(f"   • Average size: {sum(sizes)/len(sizes):.3f} MB")
        print(f"   • Total data: {sum(sizes):.2f} MB")

def analyze_parameter_ranges(data):
    """Analyze overall parameter ranges"""
    print(f"\n📊 OVERALL PARAMETER RANGES")
    print("=" * 60)
    
    ranges = data['parameter_ranges']
    
    print(f"Amplitude Range: {ranges['amplitude_mv'][0]:.6f} to {ranges['amplitude_mv'][1]:.6f} mV")
    print(f"Moisture Range: {ranges['moisture_m3_m3'][0]:.4f} to {ranges['moisture_m3_m3'][1]:.4f} m³/m³")
    
    # Biological plausibility assessment
    print(f"\n🧬 Biological Plausibility Assessment:")
    
    # Check amplitude range
    min_amp, max_amp = ranges['amplitude_mv']
    adamatzky_min, adamatzky_max = 0.16, 0.5
    
    if min_amp >= adamatzky_min and max_amp <= adamatzky_max:
        print(f"   ✅ Amplitude range is within Adamatzky's biological range")
    else:
        print(f"   ⚠️  Amplitude range is outside Adamatzky's range (0.16-0.5 mV)")
        print(f"      Current range: {min_amp:.6f} to {max_amp:.6f} mV")
    
    # Check moisture range
    min_moist, max_moist = ranges['moisture_m3_m3']
    if -0.2 <= min_moist and max_moist <= 0.3:
        print(f"   ✅ Moisture range is within expected biological range")
    else:
        print(f"   ⚠️  Moisture range may be outside expected range")
        print(f"      Current range: {min_moist:.4f} to {max_moist:.4f} m³/m³")

def provide_recommendations(data):
    """Provide simulation recommendations"""
    print(f"\n🎯 SIMULATION RECOMMENDATIONS")
    print("=" * 60)
    
    electrical = data['electrical_datasets']
    environmental = data['environmental_datasets']
    coordinate = data['coordinate_datasets']
    
    # Count simulation-ready datasets
    ready_electrical = len([d for d in electrical if d['simulation_ready']])
    ready_environmental = len(environmental)  # All environmental are ready
    ready_coordinate = len(coordinate)  # All coordinate are ready
    
    print(f"📈 Simulation-Ready Datasets:")
    print(f"   • Electrical Activity: {ready_electrical}/{len(electrical)} files")
    print(f"   • Environmental Parameters: {ready_environmental} files")
    print(f"   • Coordinate/Spatial Data: {ready_coordinate} files")
    print(f"   • Total Ready: {ready_electrical + ready_environmental + ready_coordinate} files")
    
    print(f"\n🔧 Integration Strategy:")
    print(f"   1. Use {ready_environmental} environmental parameter files for moisture/temperature simulation")
    print(f"   2. Use {ready_coordinate} coordinate files for spatial network modeling")
    print(f"   3. Process {len(electrical)} electrical files with amplitude normalization")
    
    print(f"\n⚠️  Key Considerations:")
    print(f"   • Electrical amplitudes need normalization to Adamatzky range (0.16-0.5 mV)")
    print(f"   • Moisture data shows both dry (-0.55) and wet (0.15) conditions")
    print(f"   • Coordinate data provides extensive spatial network information")
    print(f"   • Species diversity: {len(data['species_summary'])} different species represented")

def main():
    """Main analysis function"""
    print("🔬 DETAILED ENVIRONMENTAL PARAMETER EXTRACTION SUMMARY")
    print("=" * 80)
    
    # Load data
    data = load_extraction_results()
    if not data:
        print("❌ Could not load extraction results")
        return
    
    # Run analyses
    analyze_electrical_datasets(data)
    analyze_environmental_datasets(data)
    analyze_coordinate_datasets(data)
    analyze_parameter_ranges(data)
    provide_recommendations(data)
    
    print(f"\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE - Ready for simulation integration!")
    print("=" * 80)

if __name__ == "__main__":
    main() 