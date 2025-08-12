#!/usr/bin/env python3
"""
Fungal Audio Extractor

This system converts wave transform analysis results into audible audio files,
revealing the "sound" of fungal communication patterns.

Features:
- Extract audio from wave transform coefficients
- Create growth-rhythm based audio
- Generate communication mode sonification
- Real-time audio synthesis capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from scipy import signal
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FungalAudioExtractor:
    """
    Extracts audio from fungal wave transform analysis results.
    
    This class converts the mathematical wave transform W(k,τ) into
    audible representations that reveal fungal communication patterns.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.audio_duration = 10.0  # seconds
        
        # Audio synthesis parameters
        self.base_frequency = 220.0  # Hz (A3 note)
        self.audio_scale = 0.3  # Volume scaling
        
        # Fungal communication audio mapping
        self.communication_modes = {
            'growth_coordinated': {'freq_range': (200, 400), 'rhythm': 'sqrt_t'},
            'resource_signaling': {'freq_range': (400, 800), 'rhythm': 'pulse'},
            'network_status': {'freq_range': (800, 1200), 'rhythm': 'wave'},
            'environmental_response': {'freq_range': (1200, 1600), 'rhythm': 'modulated'},
            'coordination_signal': {'freq_range': (1600, 2000), 'rhythm': 'harmonic'},
            'emergency_alert': {'freq_range': (2000, 2400), 'rhythm': 'staccato'}
        }
    
    def load_wave_transform_results(self, results_file: str) -> Dict:
        """Load wave transform analysis results."""
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded wave transform results from {results_file}")
            return results
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return {}
    
    def extract_audio_from_wave_transform(self, wave_transform_results: Dict) -> np.ndarray:
        """
        Extract audio from wave transform coefficients.
        
        This converts the mathematical W(k,τ) into audible sound patterns.
        """
        logger.info("Extracting audio from wave transform coefficients...")
        
        # Get wave transform data
        W_matrix = np.array(wave_transform_results.get('W_matrix', []))
        k_range = np.array(wave_transform_results.get('k_range', []))
        tau_range = np.array(wave_transform_results.get('tau_range', []))
        
        if len(W_matrix) == 0:
            logger.error("No wave transform data found")
            return np.array([])
        
        # Extract magnitude and phase
        magnitude = np.abs(W_matrix)
        phase = np.angle(W_matrix)
        
        # Create time array for audio
        t_audio = np.linspace(0, self.audio_duration, int(self.sample_rate * self.audio_duration))
        
        # Initialize audio signal
        audio_signal = np.zeros_like(t_audio)
        
        # Map k values to audio frequencies
        k_to_freq = self._map_k_to_audio_frequency(k_range)
        
        # Map tau values to temporal patterns
        tau_to_timing = self._map_tau_to_timing(tau_range)
        
        # Synthesize audio from each (k, τ) pair
        for i, k in enumerate(k_range):
            for j, tau in enumerate(tau_range):
                # Get magnitude and phase for this (k, τ) pair
                mag = magnitude[i, j]
                ph = phase[i, j]
                
                # Skip if magnitude is too small
                if mag < 0.1 * np.max(magnitude):
                    continue
                
                # Map to audio parameters
                freq = k_to_freq[i]
                timing = tau_to_timing[j]
                
                # Create audio component
                audio_component = self._create_audio_component(
                    t_audio, freq, mag, ph, timing, tau
                )
                
                # Add to main audio signal
                audio_signal += audio_component
        
        # Normalize and apply scaling
        audio_signal = self.audio_scale * audio_signal / np.max(np.abs(audio_signal))
        
        logger.info("✅ Audio extraction completed")
        return audio_signal
    
    def _map_k_to_audio_frequency(self, k_range: np.ndarray) -> np.ndarray:
        """Map k values (spatial frequency) to audio frequencies."""
        # Map k range to audible frequency range (20 Hz - 20 kHz)
        min_freq = 100.0  # Hz
        max_freq = 2000.0  # Hz
        
        # Logarithmic mapping for better audio distribution
        k_normalized = (k_range - k_range.min()) / (k_range.max() - k_range.min())
        audio_freqs = min_freq * (max_freq / min_freq) ** k_normalized
        
        return audio_freqs
    
    def _map_tau_to_timing(self, tau_range: np.ndarray) -> np.ndarray:
        """Map tau values (scale) to temporal patterns."""
        # Map tau to timing characteristics
        # Small tau = fast patterns, large tau = slow patterns
        tau_normalized = (tau_range - tau_range.min()) / (tau_range.max() - tau_range.min())
        
        # Timing ranges from 0.1 to 2.0 seconds
        timing = 0.1 + 1.9 * tau_normalized
        
        return timing
    
    def _create_audio_component(self, t: np.ndarray, freq: float, magnitude: float, 
                               phase: float, timing: float, tau: float) -> np.ndarray:
        """Create individual audio component from wave transform parameters."""
        
        # Base sinusoidal component
        base_signal = magnitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Apply timing envelope based on tau (scale parameter)
        envelope = self._create_timing_envelope(t, timing, tau)
        
        # Apply √t scaling (fungal growth pattern)
        sqrt_t_scaling = np.sqrt(t + 1e-6) / np.sqrt(tau + 1e-6)
        sqrt_envelope = np.exp(-sqrt_t_scaling**2 / 2)
        
        # Combine all effects
        audio_component = base_signal * envelope * sqrt_envelope
        
        return audio_component
    
    def _create_timing_envelope(self, t: np.ndarray, timing: float, tau: float) -> np.ndarray:
        """Create timing envelope based on tau parameter."""
        
        # Create envelope that varies with timing
        envelope = np.ones_like(t)
        
        # Add rhythmic variations
        rhythm_freq = 1.0 / timing
        envelope *= 0.5 + 0.5 * np.sin(2 * np.pi * rhythm_freq * t)
        
        # Add decay based on tau
        decay_rate = 1.0 / (tau + 1e-6)
        envelope *= np.exp(-decay_rate * t)
        
        return envelope
    
    def create_growth_rhythm_audio(self, wave_transform_results: Dict) -> np.ndarray:
        """
        Create audio specifically representing fungal growth rhythms.
        
        This focuses on the √t scaling patterns that reflect fungal development.
        """
        logger.info("Creating growth rhythm audio...")
        
        # Extract tau (scale) information
        tau_range = np.array(wave_transform_results.get('tau_range', []))
        magnitude = np.array(wave_transform_results.get('magnitude', []))
        
        if len(tau_range) == 0:
            return np.array([])
        
        # Create time array
        t_audio = np.linspace(0, self.audio_duration, int(self.sample_rate * self.audio_duration))
        
        # Initialize growth rhythm audio
        growth_audio = np.zeros_like(t_audio)
        
        # Create √t growth pattern
        sqrt_t = np.sqrt(t_audio + 1e-6)
        
        # Map different tau values to growth phases
        for i, tau in enumerate(tau_range):
            # Normalize tau
            tau_norm = tau / np.max(tau_range)
            
            # Create growth phase audio
            growth_freq = 100 + 200 * tau_norm  # Frequency varies with scale
            growth_magnitude = np.mean(magnitude[:, i]) if magnitude.size > 0 else 1.0
            
            # Create √t modulated signal
            growth_signal = growth_magnitude * np.sin(2 * np.pi * growth_freq * sqrt_t)
            
            # Add to main signal
            growth_audio += growth_signal
        
        # Normalize
        growth_audio = self.audio_scale * growth_audio / np.max(np.abs(growth_audio))
        
        logger.info("✅ Growth rhythm audio created")
        return growth_audio
    
    def create_communication_mode_audio(self, wave_transform_results: Dict) -> Dict[str, np.ndarray]:
        """
        Create audio for different fungal communication modes.
        
        Each mode gets its own audio representation based on the 6 peaks we detected.
        """
        logger.info("Creating communication mode audio...")
        
        # Extract peak information
        feature_analysis = wave_transform_results.get('feature_analysis', {})
        peak_features = feature_analysis.get('peak_features', [])
        
        communication_audio = {}
        
        # Create audio for each communication mode
        for i, mode_name in enumerate(self.communication_modes.keys()):
            if i < len(peak_features):
                # Get peak characteristics
                peak = peak_features[i]
                k_val = peak.get('k', 1.0)
                tau_val = peak.get('tau', 1.0)
                magnitude = peak.get('magnitude', 1.0)
                
                # Create mode-specific audio
                mode_audio = self._create_mode_specific_audio(
                    mode_name, k_val, tau_val, magnitude
                )
                
                communication_audio[mode_name] = mode_audio
                logger.info(f"Created audio for {mode_name}")
        
        logger.info(f"✅ Created audio for {len(communication_audio)} communication modes")
        return communication_audio
    
    def _create_mode_specific_audio(self, mode_name: str, k: float, tau: float, 
                                   magnitude: float) -> np.ndarray:
        """Create audio specific to a communication mode."""
        
        # Get mode parameters
        mode_params = self.communication_modes[mode_name]
        freq_range = mode_params['freq_range']
        rhythm_type = mode_params['rhythm']
        
        # Create time array
        t_audio = np.linspace(0, self.audio_duration, int(self.sample_rate * self.audio_duration))
        
        # Base frequency based on k and tau
        base_freq = freq_range[0] + (freq_range[1] - freq_range[0]) * (k / 5.0)
        
        # Create rhythm based on type
        if rhythm_type == 'sqrt_t':
            # Growth-coordinated rhythm
            rhythm = np.sin(2 * np.pi * (1.0 / tau) * np.sqrt(t_audio + 1e-6))
        elif rhythm_type == 'pulse':
            # Pulsing rhythm
            rhythm = signal.square(2 * np.pi * (1.0 / tau) * t_audio, duty=0.3)
        elif rhythm_type == 'wave':
            # Wavy rhythm
            rhythm = np.sin(2 * np.pi * (1.0 / tau) * t_audio)
        elif rhythm_type == 'modulated':
            # Modulated rhythm
            carrier = np.sin(2 * np.pi * base_freq * t_audio)
            modulator = np.sin(2 * np.pi * (1.0 / tau) * t_audio)
            rhythm = carrier * (1 + 0.5 * modulator)
        elif rhythm_type == 'harmonic':
            # Harmonic rhythm
            rhythm = np.sin(2 * np.pi * base_freq * t_audio) + \
                    0.5 * np.sin(2 * np.pi * 2 * base_freq * t_audio) + \
                    0.25 * np.sin(2 * np.pi * 3 * base_freq * t_audio)
        else:  # staccato
            # Staccato rhythm
            rhythm = signal.square(2 * np.pi * (1.0 / tau) * t_audio, duty=0.1)
        
        # Combine base signal with rhythm
        audio_signal = magnitude * rhythm * np.sin(2 * np.pi * base_freq * t_audio)
        
        # Normalize
        audio_signal = self.audio_scale * audio_signal / np.max(np.abs(audio_signal))
        
        return audio_signal
    
    def save_audio_files(self, audio_signals: Dict[str, np.ndarray], output_dir: str = "results/audio"):
        """Save all audio files to disk."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            saved_files = []
            
            for name, audio in audio_signals.items():
                # Create filename
                filename = f"fungal_{name.replace('_', '_')}.wav"
                filepath = output_path / filename
                
                # Save audio file
                sf.write(filepath, audio, self.sample_rate)
                saved_files.append(filepath)
                
                logger.info(f"Saved {filename}")
            
            # Create metadata file
            metadata = {
                'sample_rate': self.sample_rate,
                'duration': self.audio_duration,
                'audio_files': [str(f) for f in saved_files],
                'communication_modes': list(self.communication_modes.keys()),
                'extraction_timestamp': str(np.datetime64('now'))
            }
            
            metadata_file = output_path / 'fungal_audio_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Saved {len(saved_files)} audio files to {output_path}")
            return saved_files
            
        except Exception as e:
            logger.error(f"Error saving audio files: {str(e)}")
            return []
    
    def create_audio_visualization(self, audio_signals: Dict[str, np.ndarray], 
                                  output_dir: str = "results/audio"):
        """Create visualizations of the audio signals."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create subplots for each audio signal
            n_signals = len(audio_signals)
            fig, axes = plt.subplots(n_signals, 2, figsize=(15, 3*n_signals))
            
            if n_signals == 1:
                axes = axes.reshape(1, -1)
            
            for i, (name, audio) in enumerate(audio_signals.items()):
                # Time domain plot
                t = np.linspace(0, self.audio_duration, len(audio))
                axes[i, 0].plot(t, audio, 'b-', linewidth=0.5)
                axes[i, 0].set_title(f'{name.replace("_", " ").title()}')
                axes[i, 0].set_xlabel('Time (s)')
                axes[i, 0].set_ylabel('Amplitude')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Frequency domain plot
                freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
                fft_audio = np.abs(np.fft.fft(audio))
                positive_freqs = freqs[freqs >= 0]
                positive_fft = fft_audio[freqs >= 0]
                
                axes[i, 1].plot(positive_freqs, positive_fft, 'r-', linewidth=0.5)
                axes[i, 1].set_title(f'{name.replace("_", " ").title()} - Spectrum')
                axes[i, 1].set_xlabel('Frequency (Hz)')
                axes[i, 1].set_ylabel('Magnitude')
                axes[i, 1].grid(True, alpha=0.3)
                axes[i, 1].set_xlim(0, 2000)  # Focus on audible range
            
            plt.tight_layout()
            plt.savefig(output_path / 'fungal_audio_visualization.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"✅ Audio visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating audio visualization: {str(e)}")

def main():
    """Main function to demonstrate fungal audio extraction."""
    logger.info("🎵 Starting Fungal Audio Extraction")
    
    # Initialize extractor
    extractor = FungalAudioExtractor(sample_rate=44100)
    
    # Load wave transform results
    results_file = "results/enhanced_adamatzky_analysis_results.json"
    
    if not Path(results_file).exists():
        logger.error(f"Results file not found: {results_file}")
        logger.info("Please run the integrated analysis first")
        return
    
    # Load results
    wave_transform_results = extractor.load_wave_transform_results(results_file)
    
    if not wave_transform_results:
        logger.error("Failed to load wave transform results")
        return
    
    # Extract different types of audio
    logger.info("🎵 Extracting fungal communication audio...")
    
    # 1. Full wave transform audio
    full_audio = extractor.extract_audio_from_wave_transform(wave_transform_results)
    
    # 2. Growth rhythm audio
    growth_audio = extractor.create_growth_rhythm_audio(wave_transform_results)
    
    # 3. Communication mode audio
    communication_audio = extractor.create_communication_mode_audio(wave_transform_results)
    
    # Combine all audio types
    all_audio = {
        'full_wave_transform': full_audio,
        'growth_rhythm': growth_audio,
        **communication_audio
    }
    
    # Save audio files
    saved_files = extractor.save_audio_files(all_audio)
    
    # Create visualizations
    extractor.create_audio_visualization(all_audio)
    
    # Summary
    logger.info("🎉 Fungal Audio Extraction Completed!")
    logger.info(f"📁 Generated {len(saved_files)} audio files")
    logger.info("🎵 You can now listen to the 'sound' of fungal communication!")
    
    # Print file locations
    print("\n🎵 **FUNGAL AUDIO FILES GENERATED:**")
    for file in saved_files:
        print(f"  🔊 {file.name}")
    
    print("\n💡 **What These Audio Files Represent:**")
    print("  • Full wave transform: Complete fungal 'conversation'")
    print("  • Growth rhythm: Fungal development patterns")
    print("  • Communication modes: Different types of fungal 'speech'")
    
    print("\n🔬 **Scientific Significance:**")
    print("  • First audio representation of fungal communication")
    print("  • Reveals temporal structure of electrical signals")
    print("  • Enables auditory pattern recognition")
    print("  • Provides new way to monitor fungal networks")

if __name__ == "__main__":
    main() 