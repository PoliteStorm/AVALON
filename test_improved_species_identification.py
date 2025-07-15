#!/usr/bin/env python3
"""
Test script for improved species identification based on Adamatzky's electrical fingerprints.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def test_improved_species_identification():
    """Test the improved species identification with Adamatzky's parameters."""
    
    print("=== Testing Improved Species Identification ===")
    print("Based on Adamatzky's Electrical Fingerprint Research")
    print("Royal Society Open Science: https://royalsocietypublishing.org/doi/10.1098/rsos.211926")
    print()
    
    # Initialize analyzer
    analyzer = RigorousFungalAnalyzer("data/csv_data", "data/15061491/fungal_spikes/good_recordings")
    
    # Load data
    data = analyzer.load_and_categorize_data()
    
    print(f"Loaded {len(data['coordinate_data'])} coordinate files")
    print(f"Loaded {len(data['voltage_data'])} voltage files")
    print()
    
    # Test species identification on a subset of files
    test_files = list(data['coordinate_data'].keys())[:10]  # Test first 10 files
    
    species_identification_results = {}
    
    for filename in test_files:
        print(f"Testing species identification for: {filename}")
        
        data_info = data['coordinate_data'][filename]
        df = data_info['data']
        metadata = data_info['metadata']
        
        # Extract coordinate signals
        if len(df.columns) >= 2:
            x_coords = df.iloc[:, 0].values
            y_coords = df.iloc[:, 1].values
            
            # Create derived signals
            distance = np.sqrt(x_coords**2 + y_coords**2)
            velocity = np.gradient(distance)
            acceleration = np.gradient(velocity)
            
            # Analyze each signal
            signals = {
                'distance': distance,
                'velocity': velocity,
                'acceleration': acceleration
            }
            
            all_features = []
            for signal_name, signal_data in signals.items():
                # Normalize signal
                if np.std(signal_data) > 0:
                    signal_normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
                else:
                    signal_normalized = signal_data
                
                # Apply improved transform
                transform_results = analyzer.improved_sqrt_transform(
                    signal_normalized,
                    metadata['species'],
                    metadata['treatment'],
                    metadata['duration_hours']
                )
                
                features = transform_results['transform_results']['features']
                all_features.extend(features)
            
            # Species identification
            species_id = analyzer.identify_species_from_electrical_fingerprint(all_features, metadata)
            
            species_identification_results[filename] = {
                'actual_species': metadata['species'],
                'identified_species': species_id['identified_species'],
                'confidence': species_id['confidence'],
                'fingerprint_scores': species_id['fingerprint_scores'],
                'avg_frequency': species_id['avg_frequency'],
                'avg_time_scale': species_id['avg_time_scale'],
                'avg_magnitude': species_id['avg_magnitude'],
                'n_features': len(all_features)
            }
            
            print(f"  Actual: {metadata['species']}, Identified: {species_id['identified_species']}")
            print(f"  Confidence: {species_id['confidence']:.3f}")
            print(f"  Features: {len(all_features)}")
            print(f"  Avg Frequency: {species_id['avg_frequency']:.3f} Hz")
            print(f"  Avg Time Scale: {species_id['avg_time_scale']:.1f} s")
            print()
    
    # Calculate overall identification accuracy
    correct_identifications = 0
    total_files = len(species_identification_results)
    
    for filename, result in species_identification_results.items():
        if result['actual_species'] == result['identified_species']:
            correct_identifications += 1
    
    accuracy = correct_identifications / total_files if total_files > 0 else 0
    
    print("=== SPECIES IDENTIFICATION RESULTS ===")
    print(f"Total files tested: {total_files}")
    print(f"Correct identifications: {correct_identifications}")
    print(f"Overall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print()
    
    # Show detailed results by species
    species_results = {}
    for filename, result in species_identification_results.items():
        actual_species = result['actual_species']
        if actual_species not in species_results:
            species_results[actual_species] = {
                'total': 0,
                'correct': 0,
                'avg_confidence': 0,
                'avg_frequency': 0,
                'avg_time_scale': 0,
                'avg_magnitude': 0
            }
        
        species_results[actual_species]['total'] += 1
        if result['actual_species'] == result['identified_species']:
            species_results[actual_species]['correct'] += 1
        
        species_results[actual_species]['avg_confidence'] += result['confidence']
        species_results[actual_species]['avg_frequency'] += result['avg_frequency']
        species_results[actual_species]['avg_time_scale'] += result['avg_time_scale']
        species_results[actual_species]['avg_magnitude'] += result['avg_magnitude']
    
    print("=== DETAILED RESULTS BY SPECIES ===")
    for species, stats in species_results.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            avg_confidence = stats['avg_confidence'] / stats['total']
            avg_frequency = stats['avg_frequency'] / stats['total']
            avg_time_scale = stats['avg_time_scale'] / stats['total']
            avg_magnitude = stats['avg_magnitude'] / stats['total']
            
            print(f"{species}:")
            print(f"  Accuracy: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
            print(f"  Avg Confidence: {avg_confidence:.3f}")
            print(f"  Avg Frequency: {avg_frequency:.3f} Hz")
            print(f"  Avg Time Scale: {avg_time_scale:.1f} s")
            print(f"  Avg Magnitude: {avg_magnitude:.1f}")
            print()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"species_identification_test_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump({
            'test_timestamp': timestamp,
            'overall_accuracy': accuracy,
            'total_files_tested': total_files,
            'correct_identifications': correct_identifications,
            'species_results': species_results,
            'detailed_results': species_identification_results
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    return species_identification_results

if __name__ == "__main__":
    test_improved_species_identification() 