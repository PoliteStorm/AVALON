#!/usr/bin/env python3
"""
Main script for running fungal language analysis
"""

import numpy as np
from pathlib import Path
from enhanced_fungal_language_decoder import EnhancedFungalLanguageDecoder
from datetime import datetime

def load_sample_data():
    """Load sample data for testing based on Adamatzky's research timescales"""
    # Generate 24 hours of data at 1 sample per second
    time = np.linspace(0, 86400, 86400)  # 24 hours in seconds
    
    # Create a complex signal with multiple patterns matching observed fungal behavior
    base_signal = (
        # Base rhythm (~14 min period, matching P. djamor low-freq)
        0.03 * np.sin(2 * np.pi * (1/840) * time) +  
        # Medium frequency component (~2.6 min period)
        0.02 * np.sin(2 * np.pi * (1/156) * time) +
        # Background noise (very small amplitude)
        0.005 * np.random.randn(len(time))
    )
    
    # Add sporadic larger spikes (0.1-2.1 mV range)
    spike_times = np.random.choice(len(time), size=int(len(time)/3600), replace=False)  # ~1 spike per hour
    for t in spike_times:
        if np.random.random() < 0.3:  # 30% chance of strong spike
            amplitude = np.random.uniform(1.0, 2.1)  # Strong spike
        else:
            amplitude = np.random.uniform(0.1, 0.5)  # Normal spike
        
        # Create spike with realistic duration (5-8 min, based on G. resinaceum research)
        duration = np.random.randint(300, 480)  # 5-8 minutes in seconds
        spike = amplitude * np.exp(-np.linspace(0, 5, duration)**2)
        
        # Add spike to signal if there's room
        if t + len(spike) < len(time):
            base_signal[t:t+len(spike)] += spike
    
    return {
        'time_array': time,
        'signal_array': base_signal,
        'spatial_data': np.random.rand(100, 100),  # Synthetic spatial data
        'metadata': {
            'species': 'Schizophyllum commune',
            'environment': 'controlled_lab',
            'temperature': 22.5,
            'humidity': 85.0,
            'recording_duration': '24h',
            'sample_rate': '1Hz'
        }
    }

def main():
    # Initialize decoder
    decoder = EnhancedFungalLanguageDecoder()
    
    # Load data
    data = load_sample_data()
    
    # Analyze temporal patterns
    temporal_results = decoder.analyze_temporal_patterns(
        data['time_array'], 
        data['signal_array']
    )
    print("\nTemporal Analysis Results:")
    print(f"Found {len(temporal_results['patterns'])} patterns")
    print(f"Confidence: {temporal_results['confidence']:.2f}")
    
    # Analyze amplitude patterns
    amplitude_results = decoder.analyze_amplitude_patterns(data['signal_array'])
    print("\nAmplitude Analysis Results:")
    print(f"Found {len(amplitude_results['words'])} amplitude words")
    print(f"Confidence: {amplitude_results['confidence']:.2f}")
    
    # Analyze frequency patterns
    frequency_results = decoder.analyze_frequency_patterns(
        data['time_array'],
        data['signal_array']
    )
    print("\nFrequency Analysis Results:")
    print(f"Found {len(frequency_results['words'])} frequency words")
    print(f"Confidence: {frequency_results['confidence']:.2f}")
    
    # Build vocabulary from all analyses
    multi_modal_data = {
        'temporal': temporal_results,
        'amplitude': amplitude_results,
        'frequency': frequency_results,
        'metadata': data['metadata']
    }
    
    vocabulary = decoder.build_vocabulary(multi_modal_data)
    print(f"\nBuilt vocabulary with {len(vocabulary)} words")
    
    # Generate and save report
    report = decoder.generate_language_report(vocabulary, [])
    
    # Save report
    output_dir = Path('language_analysis_outputs')
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / f"analysis_report_{data['metadata']['species']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nAnalysis report saved to: {report_path}")
    
    # Generate visualizations
    try:
        viz_path = output_dir / f"language_patterns_{data['metadata']['species']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        decoder.visualize_language_patterns(vocabulary, [], save_path=str(viz_path))
        print(f"Visualizations saved to: {viz_path}")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")

if __name__ == '__main__':
    main() 