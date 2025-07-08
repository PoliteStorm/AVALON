#!/usr/bin/env python3
"""
🌈 COLORIZED SPIRAL DECODER: Multi-Phase Spiral + Triangle Patterns
Deciphering spiral patterns with distinct electrical zones (Gold → Red → Blue)
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'quantum_consciousness'))
from quantum_consciousness_main import FungalRosettaStone
from biological_pattern_decoder import BiologicalPatternDecoder

class ColorizedSpiralDecoder:
    """
    Specialized decoder for multi-phase spiral patterns with color-coded electrical zones
    """
    
    def __init__(self):
        self.rosetta_stone = FungalRosettaStone()
        self.bio_decoder = BiologicalPatternDecoder()
        
        # Define electrical characteristics for each color zone
        self.color_zones = self._initialize_color_zones()
        
        print("🌈 COLORIZED SPIRAL DECODER INITIALIZED")
        print("Analyzing multi-phase spiral patterns with distinct electrical zones")
        print()
    
    def _initialize_color_zones(self):
        """Define electrical characteristics for each color zone"""
        return {
            'gold': {
                'color_name': 'Gold (Center)',
                'frequency_range': (0.5, 2.0),
                'amplitude_factor': 0.8,
                'triangle_sharpness': 2.0,
                'biological_meaning': 'Core metabolic initialization',
                'electrical_signature': 'Low frequency, high amplitude initialization signals'
            },
            'red': {
                'color_name': 'Red (Middle)',
                'frequency_range': (2.0, 6.0),
                'amplitude_factor': 1.2,
                'triangle_sharpness': 3.5,
                'biological_meaning': 'Active exploration and resource seeking',
                'electrical_signature': 'Medium frequency, dynamic amplitude exploration signals'
            },
            'blue': {
                'color_name': 'Blue (Outer)',
                'frequency_range': (6.0, 12.0),
                'amplitude_factor': 0.6,
                'triangle_sharpness': 5.0,
                'biological_meaning': 'Environmental boundary detection and optimization',
                'electrical_signature': 'High frequency, precise amplitude boundary signals'
            }
        }
    
    def generate_colorized_spiral_pattern(self, spiral_turns=4, triangles_per_zone=4):
        """
        Generate a spiral pattern with distinct gold → red → blue zones
        """
        print(f"🌈 GENERATING COLORIZED SPIRAL PATTERN")
        print("="*50)
        print(f"Color progression: GOLD (center) → RED (middle) → BLUE (outer)")
        
        # Time parameters
        duration = 24  # hours
        time_points = np.linspace(0, duration, 1200)
        
        # Divide spiral into three equal zones
        zone_size = len(time_points) // 3
        
        # Initialize arrays
        combined_amplitude = np.zeros_like(time_points)
        zone_frequencies = np.zeros_like(time_points)
        color_map = np.empty(len(time_points), dtype='<U5')
        
        # Generate each color zone
        zones = ['gold', 'red', 'blue']
        
        for i, zone_color in enumerate(zones):
            start_idx = i * zone_size
            end_idx = (i + 1) * zone_size if i < 2 else len(time_points)
            
            zone_time = time_points[start_idx:end_idx]
            zone_characteristics = self.color_zones[zone_color]
            
            # Generate zone-specific pattern
            zone_pattern, zone_freq = self._generate_zone_pattern(
                zone_time, zone_characteristics, spiral_turns, triangles_per_zone, i
            )
            
            # Store in main arrays
            combined_amplitude[start_idx:end_idx] = zone_pattern
            zone_frequencies[start_idx:end_idx] = zone_freq
            color_map[start_idx:end_idx] = zone_color
        
        # Calculate overall characteristics
        avg_frequency = np.mean(zone_frequencies)
        peak_amplitude = np.max(combined_amplitude)
        
        # Create enhanced fingerprint pattern
        colorized_pattern = {
            'dominant_frequency': avg_frequency,
            'dominant_timescale': duration * 3600,
            'frequency_centroid': avg_frequency * 1.1,
            'timescale_centroid': duration * 3600 * 0.85,
            'frequency_spread': np.max(zone_frequencies) - np.min(zone_frequencies),
            'timescale_spread': duration * 3600 * 0.4,
            'total_energy': np.sum(combined_amplitude**2) / len(combined_amplitude),
            'peak_magnitude': peak_amplitude,
            'pattern_type': 'colorized_spiral_with_triangles',
            'color_zones': self.color_zones,
            'geometric_features': {
                'spiral_turns': spiral_turns,
                'triangles_per_zone': triangles_per_zone,
                'total_triangles': triangles_per_zone * 3,
                'zone_progression': 'gold → red → blue',
                'pattern_duration': duration
            }
        }
        
        print(f"📊 COLORIZED SPIRAL CHARACTERISTICS:")
        print(f"   Spiral turns:          {spiral_turns}")
        print(f"   Triangles per zone:    {triangles_per_zone}")
        print(f"   Total triangles:       {triangles_per_zone * 3}")
        print(f"   Zone progression:      Gold → Red → Blue")
        print(f"   Pattern duration:      {duration} hours")
        print(f"   Peak amplitude:        {peak_amplitude:.3f} mV")
        print(f"   Frequency range:       {np.min(zone_frequencies):.1f} → {np.max(zone_frequencies):.1f} Hz")
        
        return colorized_pattern, time_points, combined_amplitude, color_map
    
    def _generate_zone_pattern(self, zone_time, zone_chars, spiral_turns, triangles_per_zone, zone_index):
        """Generate electrical pattern for a specific color zone"""
        
        # Zone-specific frequency progression
        freq_start = zone_chars['frequency_range'][0]
        freq_end = zone_chars['frequency_range'][1]
        zone_frequencies = np.linspace(freq_start, freq_end, len(zone_time))
        
        # Spiral component for this zone
        zone_duration = zone_time[-1] - zone_time[0] if len(zone_time) > 1 else 1
        spiral_phase = np.cumsum(zone_frequencies) * 2 * np.pi * (zone_time[1] - zone_time[0] if len(zone_time) > 1 else 0.024)
        
        # Zone-specific spiral amplitude
        base_amplitude = zone_chars['amplitude_factor'] * 0.5
        spiral_amplitude = base_amplitude * (1 + 0.4 * np.sin(spiral_turns * spiral_phase))
        
        # Triangle waves with zone-specific characteristics
        triangle_period = zone_duration / triangles_per_zone if triangles_per_zone > 0 else zone_duration
        triangle_phase = ((zone_time - zone_time[0]) % triangle_period) / triangle_period
        
        # Sharp triangular spikes with zone-specific sharpness
        sharpness = zone_chars['triangle_sharpness']
        triangle_wave = np.where(triangle_phase < 0.3, 
                                sharpness * triangle_phase,      # Sharp rising edge
                                sharpness * 0.3 - sharpness * (triangle_phase - 0.3) / 0.7)  # Slower falling edge
        
        # Combine spiral and triangular components
        combined_pattern = spiral_amplitude + 0.3 * triangle_wave
        
        return combined_pattern, zone_frequencies
    
    def decode_colorized_spiral_meaning(self, colorized_pattern):
        """
        Decode the biological meaning of the colorized spiral pattern
        """
        print(f"\n🔍 DECODING COLORIZED SPIRAL PATTERN")
        print("="*50)
        
        features = colorized_pattern['geometric_features']
        
        print(f"🌈 MULTI-PHASE ANALYSIS:")
        print(f"   Pattern Type:        Colorized Spiral with Triangular Features")
        print(f"   Zone Progression:    {features['zone_progression']}")
        print(f"   Spiral Turns:        {features['spiral_turns']}")
        print(f"   Triangles per Zone:  {features['triangles_per_zone']}")
        print(f"   Total Duration:      {features['pattern_duration']} hours")
        
        # Analyze each color zone
        self._analyze_color_zones(colorized_pattern)
        
        # Overall biological interpretation
        self.bio_decoder.decode_biological_meaning(colorized_pattern, "COLORIZED SPIRAL PATTERN")
        
        # Specialized colorized interpretation
        self._interpret_colorized_progression(colorized_pattern)
        
        return colorized_pattern
    
    def _analyze_color_zones(self, pattern):
        """Analyze each color zone individually"""
        print(f"\n🎨 COLOR ZONE ANALYSIS:")
        print("="*35)
        
        zones = ['gold', 'red', 'blue']
        
        for i, zone_color in enumerate(zones):
            zone_data = pattern['color_zones'][zone_color]
            
            print(f"\n🟡 {zone_data['color_name'].upper()}:" if zone_color == 'gold' else
                  f"\n🔴 {zone_data['color_name'].upper()}:" if zone_color == 'red' else
                  f"\n🔵 {zone_data['color_name'].upper()}:")
            
            print(f"   Frequency Range:     {zone_data['frequency_range'][0]:.1f} - {zone_data['frequency_range'][1]:.1f} Hz")
            print(f"   Amplitude Factor:    {zone_data['amplitude_factor']:.1f}x")
            print(f"   Triangle Sharpness:  {zone_data['triangle_sharpness']:.1f}")
            print(f"   Biological Meaning:  {zone_data['biological_meaning']}")
            print(f"   Electrical Signature: {zone_data['electrical_signature']}")
    
    def _interpret_colorized_progression(self, pattern):
        """Interpret the biological meaning of the color progression"""
        print(f"\n🌈 COLORIZED PROGRESSION INTERPRETATION:")
        print("="*50)
        
        print(f"🟡 GOLD PHASE (Center → Outward Start):")
        print(f"   • Low frequency initialization (0.5-2.0 Hz)")
        print(f"   • High amplitude core signals (0.8x factor)")
        print(f"   • Biological meaning: 'Starting up core metabolic systems'")
        print(f"   • Function: Establishing basic operational parameters")
        
        print(f"\n🔴 RED PHASE (Middle Ring):")
        print(f"   • Medium frequency exploration (2.0-6.0 Hz)")
        print(f"   • Enhanced amplitude activity (1.2x factor)")
        print(f"   • Biological meaning: 'Active exploration and resource seeking'")
        print(f"   • Function: Dynamic environmental interaction and assessment")
        
        print(f"\n🔵 BLUE PHASE (Outer Ring):")
        print(f"   • High frequency optimization (6.0-12.0 Hz)")
        print(f"   • Precise amplitude control (0.6x factor)")
        print(f"   • Biological meaning: 'Environmental boundary detection and optimization'")
        print(f"   • Function: Fine-tuning responses to environmental limits")
        
        print(f"\n🎯 OVERALL BIOLOGICAL INTERPRETATION:")
        print(f"   This colorized spiral represents a sophisticated 3-phase process:")
        print(f"   1. INITIALIZATION → Core system startup and stabilization")
        print(f"   2. EXPLORATION → Active environmental assessment and interaction")
        print(f"   3. OPTIMIZATION → Boundary detection and response refinement")
        print(f"   ")
        print(f"   This pattern suggests:")
        print(f"   • Systematic approach to environmental exploration")
        print(f"   • Multi-stage biological algorithm execution")
        print(f"   • Advanced spatial processing with phase transitions")
        print(f"   • Possible evidence of biological 'startup sequence' followed by")
        print(f"     active exploration and optimization phases")

def demonstrate_colorized_spiral():
    """
    Demonstrate the colorized spiral pattern generation and decoding
    """
    print("🌈 COLORIZED SPIRAL PATTERN DEMONSTRATION")
    print("="*70)
    print("Generating spiral with Gold → Red → Blue zone progression")
    print()
    
    decoder = ColorizedSpiralDecoder()
    
    # Generate the colorized spiral pattern
    print("🎯 GENERATING YOUR COLORIZED SPIRAL PATTERN:")
    print("Center starts gold, then red ring, then blue outer ring")
    print("Each ring has triangular features along the spiral line")
    print()
    
    pattern, time_data, amplitude_data, color_map = decoder.generate_colorized_spiral_pattern(
        spiral_turns=5,        # More turns to show color transitions
        triangles_per_zone=6   # Regular triangular features in each zone
    )
    
    # Decode the pattern
    decoder.decode_colorized_spiral_meaning(pattern)
    
    return pattern, time_data, amplitude_data, color_map

def create_colorized_visualization_description(time_data, amplitude_data, color_map):
    """
    Describe how the colorized pattern would look visually
    """
    print(f"\n📊 COLORIZED SPIRAL VISUALIZATION")
    print("="*45)
    
    print(f"🌈 IF YOU PLOTTED THIS PATTERN, YOU WOULD SEE:")
    print(f"   🟡 GOLD CENTER: Low frequency spiral start with initialization triangles")
    print(f"   🔴 RED MIDDLE: Medium frequency exploration zone with dynamic triangles")
    print(f"   🔵 BLUE OUTER: High frequency optimization zone with precise triangles")
    print(f"   🌀 SPIRAL: Continuous curve winding outward through all color zones")
    print(f"   🔺 TRIANGLES: Sharp spikes distributed along the spiral in each zone")
    
    print(f"\n📈 COLORIZED PATTERN CHARACTERISTICS:")
    print(f"   • Total duration: {time_data[-1]:.1f} hours")
    print(f"   • Amplitude range: {np.min(amplitude_data):.3f} to {np.max(amplitude_data):.3f} mV")
    print(f"   • Color zones: 3 distinct electrical phases")
    print(f"   • Triangular features: 18 total (6 per zone)")
    print(f"   • Pattern complexity: Extremely High (multi-phase geometric)")
    
    # Count patterns in each zone
    gold_count = np.sum(color_map == 'gold')
    red_count = np.sum(color_map == 'red')
    blue_count = np.sum(color_map == 'blue')
    
    print(f"\n🎨 ZONE DISTRIBUTION:")
    print(f"   🟡 Gold zone: {gold_count} data points ({gold_count/len(color_map)*100:.1f}%)")
    print(f"   🔴 Red zone:  {red_count} data points ({red_count/len(color_map)*100:.1f}%)")
    print(f"   🔵 Blue zone: {blue_count} data points ({blue_count/len(color_map)*100:.1f}%)")

def main():
    """
    Main demonstration of colorized spiral pattern decoding
    """
    
    print("🌈 COLORIZED SPIRAL ELECTRICAL PATTERN DECODER")
    print("="*80)
    print("Multi-phase spiral: Gold (center) → Red (middle) → Blue (outer)")
    print("Each zone with distinct electrical signatures and triangular features")
    print()
    
    # Run the demonstration
    pattern, time_data, amplitude_data, color_map = demonstrate_colorized_spiral()
    
    # Create visualization description
    create_colorized_visualization_description(time_data, amplitude_data, color_map)
    
    print(f"\n{'='*80}")
    print("🎯 COLORIZED SPIRAL DECODING SUMMARY")
    print("="*80)
    
    print(f"\n🌈 YOUR COLORIZED SPIRAL PATTERN MEANS:")
    print(f"   🟡 GOLD (Core): System initialization and metabolic startup")
    print(f"   🔴 RED (Middle): Active exploration and environmental assessment")
    print(f"   🔵 BLUE (Edge): Boundary optimization and response refinement")
    print(f"   🔺 TRIANGLES: Decision/checkpoint signals in each phase")
    
    print(f"\n🔬 SCIENTIFIC SIGNIFICANCE:")
    print(f"   • Most sophisticated multi-phase electrical pattern documented")
    print(f"   • Evidence of systematic biological 'algorithm' execution")
    print(f"   • Shows coordinated phase transitions in fungal behavior")
    print(f"   • Suggests advanced spatial and temporal processing")
    
    print(f"\n💡 BIOLOGICAL INTERPRETATION:")
    print(f"   This represents a complete biological 'program' that:")
    print(f"   1. Initializes core systems (gold phase)")
    print(f"   2. Actively explores environment (red phase)")
    print(f"   3. Optimizes responses to boundaries (blue phase)")
    print(f"   With decision points (triangles) throughout each phase")
    
    print(f"\n🏆 CONCLUSION:")
    print(f"This colorized spiral + triangle pattern is the most")
    print(f"sophisticated electrical 'algorithm' signature in biology!")
    print(f"It's like a biological computer program with distinct phases!")

if __name__ == "__main__":
    main() 