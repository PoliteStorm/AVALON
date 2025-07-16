#!/usr/bin/env python3
"""
Identify Strongest CSV for Further Research
"""

import json
import os
import numpy as np

def load_results(filename):
    """Load results from JSON file"""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def analyze_csv_strength():
    """Analyze all CSV files to identify the strongest one"""
    print("🔬 IDENTIFYING STRONGEST CSV FOR FURTHER RESEARCH")
    print("=" * 60)
    
    # Define the CSV files we analyzed
    csv_files = [
        "data/Norm_vs_deep_tip_crop.csv",
        "data/New_Oyster_with spray_as_mV_seconds_SigView.csv", 
        "data/Ch1-2_1second_sampling.csv"
    ]
    
    # Analysis results from our tests
    analysis_results = {
        "Norm_vs_deep_tip_crop.csv": {
            "ultra_optimized": {
                "spikes": 16,
                "spike_rate": 0.267,
                "amplitude": 0.1881,
                "quality_score": 0.90,
                "wave_patterns": 12,
                "snr": 1.0
            },
            "integrated_wave": {
                "spikes": 25,
                "spike_rate": 0.417,
                "amplitude": 0.4134,
                "quality_score": 1.100,
                "wave_patterns": 185,
                "snr": 6.71
            }
        },
        "New_Oyster_with spray_as_mV_seconds_SigView.csv": {
            "ultra_optimized": {
                "spikes": 14,
                "spike_rate": 0.233,
                "amplitude": 0.1832,
                "quality_score": 1.00,
                "wave_patterns": 12,
                "snr": 1.0
            },
            "integrated_wave": {
                "spikes": 24,
                "spike_rate": 0.400,
                "quality_score": 1.100,
                "wave_patterns": 185,
                "snr": 6.60
            }
        },
        "Ch1-2_1second_sampling.csv": {
            "ultra_optimized": {
                "spikes": 13,
                "spike_rate": 0.217,
                "amplitude": 0.1463,
                "quality_score": 0.90,
                "wave_patterns": 12,
                "snr": 1.0
            },
            "integrated_wave": {
                "spikes": 25,
                "spike_rate": 0.417,
                "quality_score": 1.100,
                "wave_patterns": 185,
                "snr": 6.71
            }
        }
    }
    
    # Scoring criteria weights
    weights = {
        "spike_count": 0.25,
        "spike_rate": 0.20,
        "amplitude": 0.15,
        "quality_score": 0.20,
        "wave_patterns": 0.10,
        "snr": 0.10
    }
    
    print("\n📊 ANALYSIS RESULTS:")
    print("-" * 40)
    
    scores = {}
    
    for csv_file, results in analysis_results.items():
        print(f"\n📁 {csv_file}:")
        
        # Calculate scores for both methods
        ultra_score = 0
        integrated_score = 0
        
        # Ultra-optimized scoring
        ultra = results["ultra_optimized"]
        ultra_score += (ultra["spikes"] / 25) * weights["spike_count"] * 100
        ultra_score += (ultra["spike_rate"] / 2.0) * weights["spike_rate"] * 100
        ultra_score += (ultra["amplitude"] / 5.0) * weights["amplitude"] * 100
        ultra_score += ultra["quality_score"] * weights["quality_score"] * 100
        ultra_score += (ultra["wave_patterns"] / 200) * weights["wave_patterns"] * 100
        ultra_score += (ultra["snr"] / 10.0) * weights["snr"] * 100
        
        # Integrated wave scoring
        integrated = results["integrated_wave"]
        integrated_score += (integrated["spikes"] / 25) * weights["spike_count"] * 100
        integrated_score += (integrated["spike_rate"] / 2.0) * weights["spike_rate"] * 100
        integrated_score += (integrated.get("amplitude", 0.4) / 5.0) * weights["amplitude"] * 100
        integrated_score += integrated["quality_score"] * weights["quality_score"] * 100
        integrated_score += (integrated["wave_patterns"] / 200) * weights["wave_patterns"] * 100
        integrated_score += (integrated["snr"] / 10.0) * weights["snr"] * 100
        
        # Combined score (weighted average)
        combined_score = (ultra_score * 0.4) + (integrated_score * 0.6)
        
        scores[csv_file] = {
            "ultra_score": ultra_score,
            "integrated_score": integrated_score,
            "combined_score": combined_score,
            "results": results
        }
        
        print(f"   Ultra-Optimized: {ultra_score:.1f}/100")
        print(f"   Integrated Wave: {integrated_score:.1f}/100")
        print(f"   Combined Score: {combined_score:.1f}/100")
        
        # Display key metrics
        print(f"   Spikes: {ultra['spikes']} → {integrated['spikes']} (+{integrated['spikes'] - ultra['spikes']})")
        print(f"   Spike Rate: {ultra['spike_rate']:.3f} → {integrated['spike_rate']:.3f} Hz")
        print(f"   Quality: {ultra['quality_score']:.2f} → {integrated['quality_score']:.2f}")
        print(f"   Wave Patterns: {ultra['wave_patterns']} → {integrated['wave_patterns']}")
        print(f"   SNR: {ultra['snr']:.2f} → {integrated['snr']:.2f}")
    
    # Find the strongest CSV
    strongest_csv = max(scores.keys(), key=lambda x: scores[x]["combined_score"])
    strongest_score = scores[strongest_csv]["combined_score"]
    
    print(f"\n🏆 STRONGEST CSV IDENTIFIED:")
    print("=" * 40)
    print(f"   📁 File: {strongest_csv}")
    print(f"   🎯 Combined Score: {strongest_score:.1f}/100")
    print(f"   🏅 Rank: #1 of {len(scores)} files")
    
    # Show detailed analysis of the strongest file
    strongest_results = scores[strongest_csv]["results"]
    
    print(f"\n📈 DETAILED ANALYSIS OF STRONGEST FILE:")
    print("-" * 40)
    print(f"   Ultra-Optimized Method:")
    print(f"     • Spikes: {strongest_results['ultra_optimized']['spikes']}")
    print(f"     • Spike Rate: {strongest_results['ultra_optimized']['spike_rate']:.3f} Hz")
    print(f"     • Amplitude: {strongest_results['ultra_optimized']['amplitude']:.4f} mV")
    print(f"     • Quality Score: {strongest_results['ultra_optimized']['quality_score']:.2f}")
    print(f"     • Wave Patterns: {strongest_results['ultra_optimized']['wave_patterns']}")
    print(f"     • SNR: {strongest_results['ultra_optimized']['snr']:.2f}")
    
    print(f"\n   Integrated Wave Transform Method:")
    print(f"     • Spikes: {strongest_results['integrated_wave']['spikes']}")
    print(f"     • Spike Rate: {strongest_results['integrated_wave']['spike_rate']:.3f} Hz")
    print(f"     • Quality Score: {strongest_results['integrated_wave']['quality_score']:.2f}")
    print(f"     • Wave Patterns: {strongest_results['integrated_wave']['wave_patterns']}")
    print(f"     • SNR: {strongest_results['integrated_wave']['snr']:.2f}")
    
    # Calculate improvement metrics
    spike_improvement = ((strongest_results['integrated_wave']['spikes'] - strongest_results['ultra_optimized']['spikes']) / strongest_results['ultra_optimized']['spikes']) * 100
    rate_improvement = ((strongest_results['integrated_wave']['spike_rate'] - strongest_results['ultra_optimized']['spike_rate']) / strongest_results['ultra_optimized']['spike_rate']) * 100
    
    print(f"\n📊 IMPROVEMENT METRICS:")
    print(f"   • Spike Detection: +{spike_improvement:.1f}%")
    print(f"   • Spike Rate: +{rate_improvement:.1f}%")
    print(f"   • Wave Patterns: +{strongest_results['integrated_wave']['wave_patterns'] - strongest_results['ultra_optimized']['wave_patterns']} patterns")
    print(f"   • SNR Improvement: +{strongest_results['integrated_wave']['snr'] - strongest_results['ultra_optimized']['snr']:.2f}")
    
    # Research recommendations
    print(f"\n🎯 RESEARCH RECOMMENDATIONS:")
    print("-" * 40)
    print(f"   ✅ FOCUS ON: {strongest_csv}")
    print(f"   ✅ Use Integrated Wave Transform method")
    print(f"   ✅ Analyze {strongest_results['integrated_wave']['spikes']} spikes")
    print(f"   ✅ Study {strongest_results['integrated_wave']['wave_patterns']} wave patterns")
    print(f"   ✅ Monitor {strongest_results['integrated_wave']['spike_rate']:.3f} Hz activity")
    print(f"   ✅ Validate with {strongest_results['integrated_wave']['snr']:.2f} SNR")
    
    return strongest_csv, scores

if __name__ == "__main__":
    strongest_csv, scores = analyze_csv_strength() 