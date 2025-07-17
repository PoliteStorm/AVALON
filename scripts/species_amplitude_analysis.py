#!/usr/bin/env python3
"""
Species and Amplitude Analysis for Fungal Electrical Activity
Compares amplitude ranges with Adamatzky 2023 biological ranges
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def analyze_species_and_amplitudes():
    """Analyze species identification and amplitude ranges in CSV files"""
    
    raw_data_dir = Path("data/raw")
    results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "species_identification": {},
        "amplitude_ranges": {},
        "comparison_with_adamatzky": {
            "adamatzky_biological_range": "0.05-5.0 mV",
            "adamatzky_species": "Pleurotus ostreatus (oyster mushroom)",
            "adamatzky_recording_duration": "24-48 hours",
            "adamatzky_electrode_type": "Ag/AgCl electrodes"
        },
        "summary": {}
    }
    
    # Species code mapping based on filename patterns
    species_codes = {
        "Rb": "Rhododendron (likely R. ponticum)",
        "Pv": "Pleurotus (likely P. ostreatus - oyster mushroom)",
        "Sc": "Sclerotinia (fungal pathogen)",
        "Ag": "Agaricus (button mushroom)",
        "Pi": "Pinus (pine tree - not fungal)"
    }
    
    # Analyze each CSV file
    for csv_file in raw_data_dir.glob("*.csv"):
        if csv_file.name.startswith("."):
            continue
            
        print(f"Analyzing: {csv_file.name}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file, header=None)
            
            # Determine species from filename
            species_code = None
            for code in species_codes.keys():
                if code in csv_file.name:
                    species_code = code
                    break
            
            if species_code is None:
                species_code = "Unknown"
            
            # Analyze amplitude ranges
            if df.shape[1] >= 2:  # Has at least 2 columns
                # Flatten all numeric values
                all_values = df.values.flatten()
                all_values = all_values[~pd.isna(all_values)]  # Remove NaN
                
                if len(all_values) > 0:
                    min_amp = float(np.min(all_values))
                    max_amp = float(np.max(all_values))
                    mean_amp = float(np.mean(all_values))
                    std_amp = float(np.std(all_values))
                    
                    # Convert to mV if values are very large (likely in microvolts)
                    conversion_factor = 1.0
                    unit = "unknown"
                    
                    if max_amp > 1000:
                        conversion_factor = 1000  # Convert to mV
                        unit = "mV (converted from μV)"
                    elif max_amp > 1:
                        unit = "mV"
                    else:
                        unit = "V"
                    
                    amplitude_info = {
                        "species_code": species_code,
                        "species_name": species_codes.get(species_code, "Unknown"),
                        "min_amplitude": min_amp,
                        "max_amplitude": max_amp,
                        "mean_amplitude": mean_amp,
                        "std_amplitude": std_amp,
                        "conversion_factor": conversion_factor,
                        "unit": unit,
                        "min_amplitude_mv": min_amp * conversion_factor,
                        "max_amplitude_mv": max_amp * conversion_factor,
                        "mean_amplitude_mv": mean_amp * conversion_factor,
                        "data_points": len(all_values),
                        "file_size_kb": csv_file.stat().st_size / 1024
                    }
                    
                    # Compare with Adamatzky's biological range
                    adamatzky_min = 0.05  # mV
                    adamatzky_max = 5.0   # mV
                    
                    amplitude_info["vs_adamatzky"] = {
                        "within_biological_range": (
                            amplitude_info["min_amplitude_mv"] >= adamatzky_min and 
                            amplitude_info["max_amplitude_mv"] <= adamatzky_max
                        ),
                        "overlap_with_biological_range": (
                            amplitude_info["max_amplitude_mv"] >= adamatzky_min and 
                            amplitude_info["min_amplitude_mv"] <= adamatzky_max
                        ),
                        "factor_higher_than_adamatzky": amplitude_info["max_amplitude_mv"] / adamatzky_max,
                        "factor_lower_than_adamatzky": amplitude_info["min_amplitude_mv"] / adamatzky_min
                    }
                    
                    results["amplitude_ranges"][csv_file.name] = amplitude_info
                    
                    # Track species statistics
                    if species_code not in results["species_identification"]:
                        results["species_identification"][species_code] = {
                            "name": species_codes.get(species_code, "Unknown"),
                            "files": [],
                            "amplitude_ranges": []
                        }
                    
                    results["species_identification"][species_code]["files"].append(csv_file.name)
                    results["species_identification"][species_code]["amplitude_ranges"].append({
                        "min_mv": amplitude_info["min_amplitude_mv"],
                        "max_mv": amplitude_info["max_amplitude_mv"],
                        "mean_mv": amplitude_info["mean_amplitude_mv"]
                    })
                    
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            results["amplitude_ranges"][csv_file.name] = {
                "error": str(e),
                "species_code": "Unknown"
            }
    
    # Generate summary statistics
    summary = {
        "total_files_analyzed": len(results["amplitude_ranges"]),
        "species_found": list(results["species_identification"].keys()),
        "amplitude_statistics": {}
    }
    
    # Calculate species-level statistics
    for species_code, data in results["species_identification"].items():
        if data["amplitude_ranges"]:
            amps = data["amplitude_ranges"]
            summary["amplitude_statistics"][species_code] = {
                "name": data["name"],
                "file_count": len(data["files"]),
                "min_amplitude_mv": min([a["min_mv"] for a in amps]),
                "max_amplitude_mv": max([a["max_mv"] for a in amps]),
                "mean_amplitude_mv": np.mean([a["mean_mv"] for a in amps])
            }
    
    results["summary"] = summary
    
    # Save results
    output_file = f"results/species_amplitude_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n=== SPECIES AND AMPLITUDE ANALYSIS ===")
    print(f"Results saved to: {output_file}")
    
    # Print key findings
    print(f"\nSpecies Found:")
    for species_code, data in results["species_identification"].items():
        print(f"  {species_code}: {data['name']} ({len(data['files'])} files)")
    
    print(f"\nAmplitude Ranges (converted to mV):")
    for species_code, stats in summary["amplitude_statistics"].items():
        print(f"  {species_code} ({stats['name']}): {stats['min_amplitude_mv']:.2f} - {stats['max_amplitude_mv']:.2f} mV")
    
    print(f"\nComparison with Adamatzky (0.05-5.0 mV):")
    for species_code, stats in summary["amplitude_statistics"].items():
        factor = stats['max_amplitude_mv'] / 5.0
        print(f"  {species_code}: {factor:.1f}x higher than Adamatzky's maximum")
    
    return results

if __name__ == "__main__":
    analyze_species_and_amplitudes() 