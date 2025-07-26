#!/usr/bin/env python3
"""
Show Adamatzky's Methods and Standards
Simple explanation of the analysis methods and what they mean
"""

from config.analysis_config import config

def show_adamatzky_methods():
    """Display Adamatzky's 2023 methods and standards"""
    
    print("=" * 60)
    print("ADAMATZKY 2023 METHODS & STANDARDS")
    print("=" * 60)
    print()
    
    # Get Adamatzky parameters
    adamatzky = config.get_adamatzky_params()
    
    print("🌱 BIOLOGICAL BACKGROUND")
    print("-" * 40)
    print("Adamatzky studied the split-gill fungus Schizophyllum commune")
    print("He discovered three families of electrical oscillatory patterns")
    print("These patterns represent different types of fungal electrical activity")
    print()
    
    print("⏰ TEMPORAL SCALES (3 Families of Patterns)")
    print("-" * 40)
    
    temporal_scales = adamatzky['temporal_scales']
    for scale_name, params in temporal_scales.items():
        print(f"\n📊 {scale_name.upper().replace('_', ' ')}:")
        print(f"   Time Range: {params['min_isi']} - {params['max_isi']} seconds")
        print(f"   Description: {params['description']}")
        print(f"   Duration: {params['duration']} ± {params['duration']*0.1:.0f} seconds")
        print(f"   Amplitude: {params['amplitude']} mV")
        print(f"   Distance: {params['distance']} seconds")
        
        # Convert to human-readable time
        if params['min_isi'] < 60:
            time_desc = f"{params['min_isi']} seconds"
        elif params['min_isi'] < 3600:
            time_desc = f"{params['min_isi']/60:.1f} minutes"
        else:
            time_desc = f"{params['min_isi']/3600:.1f} hours"
        print(f"   Human Time: {time_desc}")
    
    print("\n🔬 SIGNAL PARAMETERS")
    print("-" * 40)
    print(f"Sampling Rate: {adamatzky['sampling_rate']} Hz (1 measurement per second)")
    print(f"Voltage Range: {adamatzky['voltage_range']['min']} to {adamatzky['voltage_range']['max']} mV")
    print(f"Spike Amplitude Range: {adamatzky['min_spike_amplitude']} to {adamatzky['max_spike_amplitude']} mV")
    
    print("\n🌊 WAVE TRANSFORM METHOD")
    print("-" * 40)
    print("Formula: W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt")
    print()
    print("What this means:")
    print("• V(t) = Your voltage signal (fungal electrical activity)")
    print("• ψ(√t/τ) = Wave function that scales with time")
    print("• e^(-ik√t) = Complex exponential capturing frequency patterns")
    print("• k = Frequency parameter (how fast patterns repeat)")
    print("• τ = Time scale parameter (how long patterns last)")
    
    print("\n🎯 WHAT THE ANALYSIS LOOKS FOR")
    print("-" * 40)
    print("1. VERY FAST PATTERNS (30-300 seconds):")
    print("   • Quick electrical spikes")
    print("   • Rapid fungal responses")
    print("   • Short-term electrical activity")
    print()
    print("2. SLOW PATTERNS (600-3600 seconds):")
    print("   • Medium-term electrical rhythms")
    print("   • 10-minute scale oscillations")
    print("   • Sustained electrical patterns")
    print()
    print("3. VERY SLOW PATTERNS (3600+ seconds):")
    print("   • Long-term electrical cycles")
    print("   • Hour-scale patterns")
    print("   • Extended fungal electrical behavior")
    
    print("\n✅ VALIDATION STANDARDS")
    print("-" * 40)
    validation = config.get_validation_thresholds()
    print("Biological Plausibility: >", validation['biological_plausibility'])
    print("Mathematical Consistency: >", validation['mathematical_consistency'])
    print("False Positive Rate: <", validation['false_positive_rate'])
    print("Signal Quality: >", validation['signal_quality'])
    print("Energy Conservation: >", validation['energy_conservation'])
    
    print("\n📈 WHAT THE RESULTS MEAN")
    print("-" * 40)
    print("• Feature Count: How many electrical patterns were detected")
    print("• Magnitude: How strong each pattern is")
    print("• Temporal Scale: Which time scale the pattern belongs to")
    print("• Validation Score: How reliable the detection is")
    print("• Phase: Timing information about the patterns")
    
    print("\n🔍 QUALITY INDICATORS")
    print("-" * 40)
    print("Excellent (0.8-1.0): High confidence, reliable results")
    print("Good (0.6-0.8): Moderate confidence, minor issues")
    print("Caution (0.4-0.6): Low confidence, significant issues")
    print("Reject (<0.4): Poor quality, unreliable results")

if __name__ == "__main__":
    show_adamatzky_methods() 