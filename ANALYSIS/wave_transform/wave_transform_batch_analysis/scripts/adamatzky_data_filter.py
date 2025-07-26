#!/usr/bin/env python3
"""
Adamatzky Data Filter
Filters fungal electrical activity data to meet Adamatzky 2023 validated parameters
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def filter_adamatzky_compliant_data():
    """Filter data to meet Adamatzky's validated biological ranges"""
    
    # Adamatzky's validated parameters
    ADAMATZKY_PARAMS = {
        "amplitude_range": {
            "min": 0.05,  # mV
            "max": 5.0    # mV
        },
        "temporal_scales": {
            "very_slow": {"min": 60, "max": 300},    # seconds
            "slow": {"min": 10, "max": 60},          # seconds  
            "very_fast": {"min": 1, "max": 10}       # seconds
        },
        "species": ["Pv"],  # Pleurotus only
        "recording_duration": {
            "min_minutes": 30,
            "max_minutes": 480  # 8 hours max
        }
    }
    
    raw_data_dir = Path("data/raw")
    filtered_results = {
        "timestamp": datetime.now().isoformat(),
        "adamatzky_parameters": ADAMATZKY_PARAMS,
        "filtered_files": [],
        "rejected_files": [],
        "compliance_summary": {}
    }
    
    print("=" * 80)
    print("ADAMATZKY COMPLIANCE FILTER")
    print("=" * 80)
    print(f"Filtering for amplitude range: {ADAMATZKY_PARAMS['amplitude_range']['min']}-{ADAMATZKY_PARAMS['amplitude_range']['max']} mV")
    print(f"Species filter: {ADAMATZKY_PARAMS['species']}")
    print()
    
    # Process each CSV file
    csv_files = list(raw_data_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        filename = csv_file.name
        print(f"Processing: {filename}")
        
        # Check species compliance
        species_code = filename.split('_')[0] if '_' in filename else "Unknown"
        
        if species_code not in ADAMATZKY_PARAMS["species"]:
            print(f"  ❌ Rejected: Species {species_code} not in Adamatzky's study")
            filtered_results["rejected_files"].append({
                "file": filename,
                "reason": f"Species {species_code} not in Adamatzky's study",
                "species": species_code
            })
            continue
        
        try:
            # Read data efficiently
            df = pd.read_csv(csv_file, nrows=1000)  # Sample for analysis
            
            # Check amplitude compliance
            if 'amplitude' in df.columns:
                amplitude_col = 'amplitude'
            elif 'value' in df.columns:
                amplitude_col = 'value'
            elif 'voltage' in df.columns:
                amplitude_col = 'voltage'
            else:
                # Assume first numeric column is amplitude
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    amplitude_col = numeric_cols[0]
                else:
                    print(f"  ❌ Rejected: No amplitude data found")
                    filtered_results["rejected_files"].append({
                        "file": filename,
                        "reason": "No amplitude data found",
                        "species": species_code
                    })
                    continue
            
            # Analyze amplitude range
            amplitudes = df[amplitude_col].dropna()
            min_amp = amplitudes.min()
            max_amp = amplitudes.max()
            mean_amp = amplitudes.mean()
            
            # Check if within Adamatzky's biological range
            if (min_amp >= ADAMATZKY_PARAMS["amplitude_range"]["min"] and 
                max_amp <= ADAMATZKY_PARAMS["amplitude_range"]["max"]):
                
                print(f"  ✅ ACCEPTED: Amplitude range {min_amp:.3f}-{max_amp:.3f} mV within biological range")
                
                # Analyze temporal characteristics
                if 'time' in df.columns or 'timestamp' in df.columns:
                    time_col = 'time' if 'time' in df.columns else 'timestamp'
                    time_data = pd.to_datetime(df[time_col], errors='coerce').dropna()
                    
                    if len(time_data) > 1:
                        time_diff = time_data.diff().dropna()
                        avg_interval = time_diff.mean().total_seconds()
                        
                        # Determine temporal scale
                        if avg_interval >= ADAMATZKY_PARAMS["temporal_scales"]["very_slow"]["min"]:
                            temporal_scale = "very_slow"
                        elif avg_interval >= ADAMATZKY_PARAMS["temporal_scales"]["slow"]["min"]:
                            temporal_scale = "slow"
                        elif avg_interval >= ADAMATZKY_PARAMS["temporal_scales"]["very_fast"]["min"]:
                            temporal_scale = "very_fast"
                        else:
                            temporal_scale = "ultra_fast"
                        
                        print(f"    Temporal scale: {temporal_scale} ({avg_interval:.1f}s intervals)")
                        
                        filtered_results["filtered_files"].append({
                            "file": filename,
                            "species": species_code,
                            "amplitude_range": {
                                "min": float(min_amp),
                                "max": float(max_amp),
                                "mean": float(mean_amp)
                            },
                            "temporal_scale": temporal_scale,
                            "avg_interval_seconds": float(avg_interval),
                            "compliance": "adamatzky_validated"
                        })
                    else:
                        print(f"    ⚠️  Warning: Insufficient temporal data")
                        filtered_results["filtered_files"].append({
                            "file": filename,
                            "species": species_code,
                            "amplitude_range": {
                                "min": float(min_amp),
                                "max": float(max_amp),
                                "mean": float(mean_amp)
                            },
                            "temporal_scale": "unknown",
                            "compliance": "amplitude_only"
                        })
                else:
                    print(f"    ⚠️  Warning: No temporal data available")
                    filtered_results["filtered_files"].append({
                        "file": filename,
                        "species": species_code,
                        "amplitude_range": {
                            "min": float(min_amp),
                            "max": float(max_amp),
                            "mean": float(mean_amp)
                        },
                        "temporal_scale": "unknown",
                        "compliance": "amplitude_only"
                    })
            else:
                print(f"  ❌ Rejected: Amplitude range {min_amp:.3f}-{max_amp:.3f} mV outside biological range")
                filtered_results["rejected_files"].append({
                    "file": filename,
                    "reason": f"Amplitude range {min_amp:.3f}-{max_amp:.3f} mV outside Adamatzky's biological range",
                    "species": species_code,
                    "amplitude_range": {
                        "min": float(min_amp),
                        "max": float(max_amp),
                        "mean": float(mean_amp)
                    }
                })
                
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {str(e)}")
            filtered_results["rejected_files"].append({
                "file": filename,
                "reason": f"Processing error: {str(e)}",
                "species": species_code
            })
    
    # Generate compliance summary
    total_files = len(csv_files)
    accepted_files = len(filtered_results["filtered_files"])
    rejected_files = len(filtered_results["rejected_files"])
    
    filtered_results["compliance_summary"] = {
        "total_files_processed": total_files,
        "accepted_files": accepted_files,
        "rejected_files": rejected_files,
        "compliance_rate": f"{(accepted_files/total_files)*100:.1f}%" if total_files > 0 else "0%"
    }
    
    print("\n" + "=" * 80)
    print("FILTERING RESULTS")
    print("=" * 80)
    print(f"Total files processed: {total_files}")
    print(f"Files meeting Adamatzky criteria: {accepted_files}")
    print(f"Files rejected: {rejected_files}")
    print(f"Compliance rate: {filtered_results['compliance_summary']['compliance_rate']}")
    
    if accepted_files > 0:
        print(f"\nACCEPTED FILES (Adamatzky-compliant):")
        for file_info in filtered_results["filtered_files"]:
            amp_range = file_info["amplitude_range"]
            print(f"  • {file_info['file']}")
            print(f"    Amplitude: {amp_range['min']:.3f}-{amp_range['max']:.3f} mV")
            print(f"    Temporal scale: {file_info.get('temporal_scale', 'unknown')}")
    
    return filtered_results

def save_filtered_results(results):
    """Save the filtered results"""
    output_file = f"results/adamatzky_data_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFiltered results saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    results = filter_adamatzky_compliant_data()
    output_file = save_filtered_results(results) 