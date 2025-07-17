#!/usr/bin/env python3
"""
Adamatzky Setup Analysis
Compares Adamatzky's exact experimental setup with our data
"""

import json
from datetime import datetime

def analyze_adamatzky_setup():
    """Analyze Adamatzky's experimental setup vs our data"""
    
    # Adamatzky's exact specifications from the paper
    ADAMATZKY_SETUP = {
        "species": "Schizophyllum commune (split-gill fungus)",
        "strain": "H4-8A (Utrecht University, The Netherlands)",
        "growth_medium": "S. commune minimal medium (SCMM) + 1.5% agar",
        "growth_conditions": "3 days at 30°C in dark",
        "electrodes": {
            "type": "Iridium-coated stainless steel sub-dermal needle electrodes",
            "manufacturer": "Spes Medica S.r.l., Italy",
            "configuration": "Pairs of differential electrodes",
            "distance": "~10 mm between electrodes",
            "placement": "Through melted openings in Petri dish lids, touching bottom"
        },
        "data_logger": {
            "type": "ADC-24 (Pico Technology, UK)",
            "resolution": "24-bit A/D converter",
            "features": "Galvanic isolation, software-selectable sample rates",
            "voltage_range": "78 mV",
            "sampling": "1 sample per second (with averaging of up to 600 measurements/s)"
        },
        "recording_duration": "Nearly 6 days",
        "electrode_pairs": 16,
        "amplitude_ranges": {
            "very_slow_spikes": "0.16 mV (average)",
            "slow_spikes": "0.4 mV (average)", 
            "very_fast_spikes": "0.36 mV (average)"
        },
        "temporal_scales": {
            "very_slow": "2573 s (43 min) average duration",
            "slow": "457 s (8 min) average duration",
            "very_fast": "24 s average duration"
        }
    }
    
    # Our setup (inferred from data)
    OUR_SETUP = {
        "species": "Various (Pv=Pleurotus, Ag=Agaricus, etc.)",
        "electrodes": "Unknown type",
        "amplitude_ranges": {
            "pleurotus": "0.00 - 2,446,573.70 mV",
            "agaricus": "0.00 - 219.31 mV",
            "rhododendron": "0.00 - 242.19 mV"
        },
        "sampling": "Unknown rate",
        "voltage_range": "Unknown"
    }
    
    # Analysis
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "adamatzky_setup": ADAMATZKY_SETUP,
        "our_setup": OUR_SETUP,
        "key_differences": {
            "electrode_type": {
                "adamatzky": "Iridium-coated stainless steel sub-dermal needle electrodes",
                "our_data": "Unknown",
                "impact": "Different electrode types have different sensitivity and amplification"
            },
            "voltage_range": {
                "adamatzky": "78 mV acquisition range",
                "our_data": "Unknown",
                "impact": "Our data may have much higher voltage ranges"
            },
            "amplification": {
                "adamatzky": "24-bit A/D converter with galvanic isolation",
                "our_data": "Unknown",
                "impact": "Different amplification settings could explain amplitude differences"
            },
            "species": {
                "adamatzky": "Schizophyllum commune (specific strain)",
                "our_data": "Pleurotus, Agaricus, etc. (different species)",
                "impact": "Different fungal species may have different electrical properties"
            },
            "recording_conditions": {
                "adamatzky": "Controlled agar medium, sealed Petri dishes",
                "our_data": "Unknown conditions",
                "impact": "Environmental factors affect signal strength"
            }
        },
        "amplitude_comparison": {
            "adamatzky_max": "0.4 mV (slow spikes)",
            "our_pleurotus_max": "2,446,573.70 mV",
            "magnitude_difference": "6,116,434.25x higher",
            "likely_causes": [
                "Different electrode types and sensitivity",
                "Different amplification settings",
                "Different voltage acquisition ranges",
                "Different recording conditions",
                "Species-specific electrical properties"
            ]
        }
    }
    
    # Save analysis
    output_file = f"results/adamatzky_setup_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print("=" * 80)
    print("ADAMATZKY EXPERIMENTAL SETUP ANALYSIS")
    print("=" * 80)
    
    print("\nADAMATZKY'S EXACT SETUP:")
    print(f"- Species: {ADAMATZKY_SETUP['species']}")
    print(f"- Electrodes: {ADAMATZKY_SETUP['electrodes']['type']}")
    print(f"- Voltage Range: {ADAMATZKY_SETUP['data_logger']['voltage_range']}")
    print(f"- Amplitude Range: 0.16-0.4 mV")
    
    print("\nOUR SETUP (INFERRED):")
    print(f"- Species: Various (Pleurotus, Agaricus, etc.)")
    print(f"- Electrodes: Unknown type")
    print(f"- Voltage Range: Unknown")
    print(f"- Amplitude Range: 0.00 - 2,446,573.70 mV")
    
    print("\nKEY DIFFERENCES:")
    for key, diff in analysis['key_differences'].items():
        print(f"- {key.replace('_', ' ').title()}:")
        print(f"  Adamatzky: {diff['adamatzky']}")
        print(f"  Our Data: {diff['our_data']}")
        print(f"  Impact: {diff['impact']}")
    
    print(f"\nAMPLITUDE DIFFERENCE: {analysis['amplitude_comparison']['magnitude_difference']}")
    print("\nLIKELY CAUSES:")
    for cause in analysis['amplitude_comparison']['likely_causes']:
        print(f"- {cause}")
    
    print(f"\nDetailed analysis saved to: {output_file}")
    
    return analysis

if __name__ == "__main__":
    analyze_adamatzky_setup() 