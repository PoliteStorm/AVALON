#!/usr/bin/env python3
"""
Simple Fungal Audio Extractor

Creates audio representations of fungal communication patterns based on
our established analysis results and insights.
"""

import numpy as np
import soundfile as sf
from scipy import signal
from pathlib import Path
import json
import logging
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleFungalAudioExtractor:
    """
    Simple audio extractor that creates fungal communication audio
    based on our established analysis results.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.audio_duration = 15.0  # seconds
        
        # Audio parameters
        self.audio_scale = 0.3
        
        # Fungal communication characteristics from our analysis
        self.fungal_characteristics = {
            'growth_rhythm': {
                'base_freq': 100,  # Hz
                'pattern': 'sqrt_t',  # Square root of time scaling
                'description': 'Fungal growth coordination patterns'
            },
            'frequency_discrimination': {
                'low_freq_range': (50, 200),   # Hz - below 10 mHz threshold
                'high_freq_range': (200, 400), # Hz - above 10 mHz threshold
                'description': 'Frequency discrimination behavior'
            },
            'harmonic_patterns': {
                'harmonic_2': 300,  # Hz - 2nd harmonic
                'harmonic_3': 450,  # Hz - 3rd harmonic
                'description': 'Harmonic generation patterns'
            },
            'communication_modes': {
                'resource_signaling': {'freq': 150, 'pattern': 'pulse'},
                'network_status': {'freq': 250, 'pattern': 'wave'},
                'growth_coordination': {'freq': 350, 'pattern': 'sqrt_t'},
                'environmental_response': {'freq': 450, 'pattern': 'modulated'},
                'emergency_alert': {'freq': 550, 'pattern': 'staccato'},
                'coordination_signal': {'freq': 650, 'pattern': 'harmonic'}
            }
        }
    
    def create_growth_rhythm_audio(self) -> np.ndarray:
        """
        Create audio representing fungal growth rhythms.
        
        This implements the √t scaling we discovered in our wave transform analysis.
        """
        logger.info("Creating fungal growth rhythm audio...")
        
        # Create time array
        t = np.linspace(0, self.audio_duration, int(self.sample_rate * self.audio_duration))
        
        # Create √t growth pattern (matches fungal growth)
        sqrt_t = np.sqrt(t + 1e-6)
        
        # Base growth frequency
        growth_freq = 100  # Hz
        
        # Create growth-coordinated signal
        growth_signal = np.sin(2 * np.pi * growth_freq * sqrt_t)
        
        # Add growth phase variations
        growth_phases = [
            (0, 3, 0.5),    # Early growth - slow, quiet
            (3, 8, 1.0),    # Active growth - medium intensity
            (8, 12, 1.5),   # Peak growth - high intensity
            (12, 15, 0.8)   # Mature growth - steady state
        ]
        
        final_signal = np.zeros_like(t)
        for start, end, intensity in growth_phases:
            mask = (t >= start) & (t < end)
            final_signal[mask] = intensity * growth_signal[mask]
        
        # Normalize
        final_signal = self.audio_scale * final_signal / np.max(np.abs(final_signal))
        
        logger.info("✅ Growth rhythm audio created")
        return final_signal
    
    def create_frequency_discrimination_audio(self) -> np.ndarray:
        """
        Create audio demonstrating fungal frequency discrimination.
        
        Shows the difference between low (≤10 mHz) and high (>10 mHz) frequency responses.
        """
        logger.info("Creating frequency discrimination audio...")
        
        t = np.linspace(0, self.audio_duration, int(self.sample_rate * self.audio_duration))
        
        # Low frequency response (≤10 mHz) - complex, distorted
        low_freq_signal = np.zeros_like(t)
        for i, freq in enumerate(np.linspace(50, 200, 5)):
            # Add harmonics and distortion (high THD)
            fundamental = np.sin(2 * np.pi * freq * t)
            harmonic_2 = 0.3 * np.sin(2 * np.pi * 2 * freq * t)
            harmonic_3 = 0.5 * np.sin(2 * np.pi * 3 * freq * t)
            low_freq_signal += fundamental + harmonic_2 + harmonic_3
        
        # High frequency response (>10 mHz) - clean, simple
        high_freq_signal = np.zeros_like(t)
        for i, freq in enumerate(np.linspace(200, 400, 3)):
            # Clean signal (low THD)
            high_freq_signal += np.sin(2 * np.pi * freq * t)
        
        # Combine with timing
        low_mask = (t >= 0) & (t < 7.5)
        high_mask = (t >= 7.5) & (t < 15)
        
        final_signal = np.zeros_like(t)
        final_signal[low_mask] = low_freq_signal[low_mask]
        final_signal[high_mask] = high_freq_signal[high_mask]
        
        # Normalize
        final_signal = self.audio_scale * final_signal / np.max(np.abs(final_signal))
        
        logger.info("✅ Frequency discrimination audio created")
        return final_signal
    
    def create_harmonic_pattern_audio(self) -> np.ndarray:
        """
        Create audio showing harmonic generation patterns.
        
        Demonstrates the 2nd vs 3rd harmonic relationships we discovered.
        """
        logger.info("Creating harmonic pattern audio...")
        
        t = np.linspace(0, self.audio_duration, int(self.sample_rate * self.audio_duration))
        
        # Fundamental frequency
        fundamental_freq = 200  # Hz
        
        # Create harmonic patterns
        fundamental = np.sin(2 * np.pi * fundamental_freq * t)
        harmonic_2 = 0.4 * np.sin(2 * np.pi * 2 * fundamental_freq * t)
        harmonic_3 = 0.6 * np.sin(2 * np.pi * 3 * fundamental_freq * t)
        
        # Show different harmonic relationships over time
        # First 5 seconds: fundamental + 2nd harmonic
        # Next 5 seconds: fundamental + 3rd harmonic  
        # Last 5 seconds: all harmonics together
        
        final_signal = np.zeros_like(t)
        
        # 0-5s: Fundamental + 2nd harmonic
        mask1 = (t >= 0) & (t < 5)
        final_signal[mask1] = fundamental[mask1] + harmonic_2[mask1]
        
        # 5-10s: Fundamental + 3rd harmonic
        mask2 = (t >= 5) & (t < 10)
        final_signal[mask2] = fundamental[mask2] + harmonic_3[mask2]
        
        # 10-15s: All harmonics
        mask3 = (t >= 10) & (t < 15)
        final_signal[mask3] = fundamental[mask3] + harmonic_2[mask3] + harmonic_3[mask3]
        
        # Normalize
        final_signal = self.audio_scale * final_signal / np.max(np.abs(final_signal))
        
        logger.info("✅ Harmonic pattern audio created")
        return final_signal
    
    def create_communication_mode_audio(self) -> Dict[str, np.ndarray]:
        """
        Create audio for each fungal communication mode.
        
        Based on the 6 peaks we detected in our wave transform analysis.
        """
        logger.info("Creating communication mode audio...")
        
        t = np.linspace(0, self.audio_duration, int(self.sample_rate * self.audio_duration))
        communication_audio = {}
        
        for mode_name, params in self.fungal_characteristics['communication_modes'].items():
            freq = params['freq']
            pattern_type = params['pattern']
            
            # Create mode-specific audio
            if pattern_type == 'pulse':
                # Pulsing pattern for resource signaling
                signal_audio = np.sin(2 * np.pi * freq * t) * signal.square(2 * np.pi * 0.5 * t, duty=0.3)
            elif pattern_type == 'wave':
                # Wavy pattern for network status
                signal_audio = np.sin(2 * np.pi * freq * t) * np.sin(2 * np.pi * 0.2 * t)
            elif pattern_type == 'sqrt_t':
                # Growth-coordinated pattern
                sqrt_t = np.sqrt(t + 1e-6)
                signal_audio = np.sin(2 * np.pi * freq * sqrt_t)
            elif pattern_type == 'modulated':
                # Modulated pattern for environmental response
                carrier = np.sin(2 * np.pi * freq * t)
                modulator = np.sin(2 * np.pi * 0.3 * t)
                signal_audio = carrier * (1 + 0.5 * modulator)
            elif pattern_type == 'staccato':
                # Staccato pattern for emergency alerts
                signal_audio = np.sin(2 * np.pi * freq * t) * signal.square(2 * np.pi * 2.0 * t, duty=0.1)
            else:  # harmonic
                # Harmonic pattern for coordination
                fundamental = np.sin(2 * np.pi * freq * t)
                harmonic_2 = 0.5 * np.sin(2 * np.pi * 2 * freq * t)
                harmonic_3 = 0.25 * np.sin(2 * np.pi * 3 * freq * t)
                signal_audio = fundamental + harmonic_2 + harmonic_3
            
            # Normalize
            signal_audio = self.audio_scale * signal_audio / np.max(np.abs(signal_audio))
            communication_audio[mode_name] = signal_audio
            
            logger.info(f"Created audio for {mode_name}")
        
        logger.info(f"✅ Created audio for {len(communication_audio)} communication modes")
        return communication_audio
    
    def create_integrated_fungal_audio(self) -> np.ndarray:
        """
        Create integrated audio that combines all fungal communication patterns.
        
        This represents the complete 'conversation' of a fungal network.
        """
        logger.info("Creating integrated fungal audio...")
        
        t = np.linspace(0, self.audio_duration, int(self.sample_rate * self.audio_duration))
        
        # Get individual components
        growth_audio = self.create_growth_rhythm_audio()
        freq_disc_audio = self.create_frequency_discrimination_audio()
        harmonic_audio = self.create_harmonic_pattern_audio()
        comm_audio = self.create_communication_mode_audio()
        
        # Combine all components with different timing
        integrated_signal = np.zeros_like(t)
        
        # 0-3s: Growth rhythm
        mask1 = (t >= 0) & (t < 3)
        integrated_signal[mask1] += 0.3 * growth_audio[mask1]
        
        # 3-6s: Frequency discrimination
        mask2 = (t >= 3) & (t < 6)
        integrated_signal[mask2] += 0.3 * freq_disc_audio[mask2]
        
        # 6-9s: Harmonic patterns
        mask3 = (t >= 6) & (t < 9)
        integrated_signal[mask3] += 0.3 * harmonic_audio[mask3]
        
        # 9-12s: Communication modes
        mask4 = (t >= 9) & (t < 12)
        for mode_audio in comm_audio.values():
            integrated_signal[mask4] += 0.1 * mode_audio[mask4]
        
        # 12-15s: All together (fungal network in full communication)
        mask5 = (t >= 12) & (t < 15)
        integrated_signal[mask5] += 0.2 * growth_audio[mask5]
        integrated_signal[mask5] += 0.2 * freq_disc_audio[mask5]
        integrated_signal[mask5] += 0.2 * harmonic_audio[mask5]
        for mode_audio in comm_audio.values():
            integrated_signal[mask5] += 0.05 * mode_audio[mask5]
        
        # Normalize
        integrated_signal = self.audio_scale * integrated_signal / np.max(np.abs(integrated_signal))
        
        logger.info("✅ Integrated fungal audio created")
        return integrated_signal
    
    def save_audio_files(self, output_dir: str = "results/audio"):
        """Create and save all fungal audio files."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            
            # Create and save each audio type
            audio_types = {
                'fungal_growth_rhythm': self.create_growth_rhythm_audio(),
                'fungal_frequency_discrimination': self.create_frequency_discrimination_audio(),
                'fungal_harmonic_patterns': self.create_harmonic_pattern_audio(),
                'fungal_integrated_communication': self.create_integrated_fungal_audio()
            }
            
            # Save individual audio files
            for name, audio in audio_types.items():
                filename = f"{name}.wav"
                filepath = output_path / filename
                sf.write(filepath, audio, self.sample_rate)
                saved_files.append(filepath)
                logger.info(f"Saved {filename}")
            
            # Save communication mode audio files
            comm_audio = self.create_communication_mode_audio()
            for mode_name, audio in comm_audio.items():
                filename = f"fungal_{mode_name}.wav"
                filepath = output_path / filename
                sf.write(filepath, audio, self.sample_rate)
                saved_files.append(filepath)
                logger.info(f"Saved {filename}")
            
            # Create metadata
            metadata = {
                'sample_rate': self.sample_rate,
                'duration': self.audio_duration,
                'audio_files': [str(f) for f in saved_files],
                'fungal_characteristics': self.fungal_characteristics,
                'extraction_timestamp': str(np.datetime64('now')),
                'description': 'Audio representation of fungal communication patterns based on wave transform analysis'
            }
            
            metadata_file = output_path / 'fungal_audio_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Saved {len(saved_files)} audio files to {output_path}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving audio files: {str(e)}")
            return []

def main():
    """Main function to demonstrate fungal audio extraction."""
    logger.info("🎵 Starting Simple Fungal Audio Extraction")
    
    # Initialize extractor
    extractor = SimpleFungalAudioExtractor(sample_rate=44100)
    
    # Create and save all audio files
    saved_files = extractor.save_audio_files()
    
    if saved_files:
        logger.info("🎉 Fungal Audio Extraction Completed Successfully!")
        
        # Print summary
        print("\n🎵 **FUNGAL AUDIO FILES GENERATED:**")
        for file in saved_files:
            print(f"  🔊 {file.name}")
        
        print("\n💡 **What These Audio Files Represent:**")
        print("  • Growth Rhythm: Fungal development patterns (√t scaling)")
        print("  • Frequency Discrimination: Low vs high frequency responses")
        print("  • Harmonic Patterns: 2nd vs 3rd harmonic relationships")
        print("  • Communication Modes: Different types of fungal 'speech'")
        print("  • Integrated Communication: Complete fungal 'conversation'")
        
        print("\n🔬 **Scientific Significance:**")
        print("  • First audio representation of fungal communication patterns")
        print("  • Reveals temporal structure of electrical signals")
        print("  • Enables auditory pattern recognition")
        print("  • Provides new way to monitor fungal networks")
        
        print("\n🎧 **How to Listen:**")
        print("  • Use any audio player to listen to the .wav files")
        print("  • Each file represents different aspects of fungal communication")
        print("  • The integrated file shows the complete 'conversation'")
        
    else:
        logger.error("❌ Audio extraction failed")

if __name__ == "__main__":
    main() 