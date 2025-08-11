#!/usr/bin/env python3
"""
Test Script for Advanced Fungal Communication Analysis
Author: Joe Knowles
Timestamp: 2025-01-27
Description: Demonstrates advanced signal analysis techniques for mushroom communication
"""

import numpy as np
import matplotlib.pyplot as plt
from advanced_fungal_communication_analyzer import AdvancedFungalCommunicationAnalyzer

def generate_test_data():
    """Generate synthetic fungal electrical data for testing."""
    print("🧪 Generating Test Fungal Data...")
    
    # Simulate different fungal species with unique signatures
    sample_rate = 1000
    duration = 10  # seconds
    t = np.linspace(0, duration, sample_rate * duration)
    
    # Species 1: Pleurotus ostreatus (oyster mushroom) - rhythmic spiker
    pleurotus_data = (
        0.5 * np.sin(2 * np.pi * 5 * t) +  # 5 Hz base rhythm
        0.3 * np.sin(2 * np.pi * 15 * t) +  # 15 Hz modulation
        0.1 * np.random.randn(len(t)) +      # noise
        0.2 * np.exp(-((t - 5) ** 2) / 2)   # environmental response
    )
    
    # Species 2: Ganoderma lucidum (reishi) - steady communicator
    ganoderma_data = (
        0.3 * np.sin(2 * np.pi * 3 * t) +   # 3 Hz steady signal
        0.4 * np.sin(2 * np.pi * 12 * t) +  # 12 Hz carrier
        0.05 * np.random.randn(len(t)) +     # low noise
        0.1 * np.exp(-((t - 7) ** 2) / 1.5) # different response timing
    )
    
    # Species 3: Hericium erinaceus (lion's mane) - adaptive learner
    hericium_data = (
        0.2 * np.sin(2 * np.pi * 8 * t) +   # 8 Hz base
        0.3 * np.sin(2 * np.pi * 20 * t) +  # 20 Hz high frequency
        0.15 * np.random.randn(len(t)) +     # moderate noise
        0.3 * np.exp(-((t - 3) ** 2) / 3) + # early response
        0.1 * np.exp(-((t - 8) ** 2) / 2)   # secondary response
    )
    
    return {
        'pleurotus': pleurotus_data,
        'ganoderma': ganoderma_data,
        'hericium': hericium_data,
        'time': t,
        'sample_rate': sample_rate
    }

def test_frequency_analysis(analyzer, test_data):
    """Test frequency domain analysis."""
    print("\n🎵 Testing Frequency Domain Analysis...")
    
    for species, data in test_data.items():
        if species == 'time' or species == 'sample_rate':
            continue
            
        print(f"  Analyzing {species}...")
        results, frequencies, magnitude = analyzer.frequency_domain_analysis(
            data, test_data['sample_rate']
        )
        
        print(f"    Dominant frequency: {results['dominant_frequency']:.2f} Hz")
        print(f"    Total spectral power: {results['total_spectral_power']:.2f}")
        print(f"    Spectral centroid: {results['spectral_centroid']:.2f} Hz")
        
        # Plot frequency spectrum
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(test_data['time'], data)
        plt.title(f'{species.capitalize()} - Time Domain Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        plt.subplot(2, 1, 2)
        plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
        plt.title(f'{species.capitalize()} - Frequency Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.tight_layout()
        plt.savefig(f'RESULTS/analysis/{species}_frequency_analysis.png')
        plt.close()

def test_time_frequency_mapping(analyzer, test_data):
    """Test time-frequency analysis."""
    print("\n🕰️ Testing Time-Frequency Mapping...")
    
    for species, data in test_data.items():
        if species == 'time' or species == 'sample_rate':
            continue
            
        print(f"  Mapping {species} time-frequency patterns...")
        time_results, frequencies, times, spectrogram = analyzer.time_frequency_mapping(
            data, test_data['sample_rate']
        )
        
        print(f"    Peak activity times: {len(time_results['peak_activity_times'])} detected")
        print(f"    Frequency evolution: {len(time_results['frequency_evolution'])} frequency bands")
        
        # Plot spectrogram
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram))
        plt.colorbar(label='Power (dB)')
        plt.title(f'{species.capitalize()} - Time-Frequency Spectrogram')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(f'RESULTS/analysis/{species}_spectrogram.png')
        plt.close()

def test_phase_relationships(analyzer, test_data):
    """Test phase relationship analysis."""
    print("\n🔄 Testing Phase Relationship Analysis...")
    
    species_list = ['pleurotus', 'ganoderma', 'hericium']
    
    for i, sp1 in enumerate(species_list):
        for j, sp2 in enumerate(species_list[i+1:], i+1):
            print(f"  Analyzing {sp1} vs {sp2}...")
            
            results, lag, correlation, phase_diff, coherence = analyzer.phase_relationship_analysis(
                test_data[sp1], test_data[sp2]
            )
            
            print(f"    Max correlation: {results['cross_correlation']['max_correlation']:.3f}")
            print(f"    Lag at max: {results['cross_correlation']['lag_at_max']} samples")
            print(f"    Phase consistency: {results['phase_analysis']['phase_consistency']:.3f}")
            print(f"    Coherence: {results['phase_analysis']['coherence']:.3f}")
            
            # Plot cross-correlation
            plt.figure(figsize=(10, 6))
            plt.plot(lag, correlation)
            plt.title(f'Cross-Correlation: {sp1.capitalize()} vs {sp2.capitalize()}')
            plt.xlabel('Lag (samples)')
            plt.ylabel('Correlation')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'RESULTS/analysis/{sp1}_vs_{sp2}_correlation.png')
            plt.close()

def test_behavioral_patterns(analyzer, test_data):
    """Test behavioral pattern recognition."""
    print("\n🎭 Testing Behavioral Pattern Recognition...")
    
    for species, data in test_data.items():
        if species == 'time' or species == 'sample_rate':
            continue
            
        print(f"  Analyzing {species} personality...")
        behavior_results = analyzer.behavioral_pattern_recognition({'individual': data})
        
        personality = behavior_results['individual']
        print(f"    Communication style: {personality['communication_style']}")
        print(f"    Response patterns: {personality['response_patterns']['consistency']:.3f}")
        print(f"    Social behavior: {personality['social_behavior']['interaction_frequency']:.3f}")
        print(f"    Adaptability: {personality['adaptability']:.3f}")

def test_genetic_signatures(analyzer, test_data):
    """Test genetic communication signature analysis."""
    print("\n🧬 Testing Genetic Communication Signatures...")
    
    # Prepare species data
    species_data = {}
    for species, data in test_data.items():
        if species != 'time' and species != 'sample_rate':
            species_data[species] = data
    
    signatures, similarities = analyzer.genetic_communication_signatures(species_data)
    
    print("\n  Species Signatures:")
    for species, sig in signatures.items():
        print(f"    {species.capitalize()}:")
        print(f"      Mean amplitude: {sig['mean_amplitude']:.3f}")
        print(f"      Spike frequency: {sig['spike_frequency']}")
        print(f"      Signal complexity: {sig['signal_complexity']:.3f}")
    
    print("\n  Species Similarities:")
    for comparison, similarity in similarities.items():
        print(f"    {comparison}: {similarity:.3f}")

def main():
    """Main test function."""
    print("🍄 Advanced Fungal Communication Analysis Test Suite")
    print("Author: Joe Knowles")
    print("Timestamp: 2025-01-27")
    print("=" * 60)
    
    # Create analyzer
    analyzer = AdvancedFungalCommunicationAnalyzer()
    
    # Generate test data
    test_data = generate_test_data()
    
    # Create results directory
    import os
    os.makedirs('RESULTS/analysis', exist_ok=True)
    
    # Run tests
    test_frequency_analysis(analyzer, test_data)
    test_time_frequency_mapping(analyzer, test_data)
    test_phase_relationships(analyzer, test_data)
    test_behavioral_patterns(analyzer, test_data)
    test_genetic_signatures(analyzer, test_data)
    
    print("\n✅ All tests completed successfully!")
    print("📁 Results saved to RESULTS/analysis/")
    print("🎨 Visualizations generated for each analysis type")

if __name__ == "__main__":
    main() 