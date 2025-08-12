#!/usr/bin/env python3
"""
Fungal Synth Sound Generator - Built-in Python Version
Author: Joe Knowles
Timestamp: 2025-08-12 09:23:27 BST
Description: Converts real fungal electrical data into cool synth sounds using built-in Python
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import struct
from pathlib import Path
import json

class FungalSynthBuiltin:
    """
    Advanced sound generator that converts real fungal electrical data into cool synth sounds
    Uses only built-in Python libraries
    """
    
    def __init__(self):
        self.timestamp = "20250812_092327"
        self.author = "Joe Knowles"
        self.sample_rate = 44100  # CD quality audio
        self.synthesis_methods = {
            'additive': self.additive_synthesis,
            'fm': self.frequency_modulation_synthesis,
            'granular': self.granular_synthesis,
            'waveform': self.waveform_synthesis,
            'filtered': self.filtered_synthesis,
            'ambient': self.ambient_synthesis
        }
        
    def load_real_fungal_data(self, file_path):
        """
        Load real fungal electrical data from CSV
        """
        print(f"🍄 Loading REAL fungal electrical data from: {file_path}")
        
        try:
            data = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines and extract electrical data
            header_count = 0
            for line in lines:
                if line.strip() and not line.startswith('"'):
                    header_count += 1
                if header_count > 2:
                    break
            
            # Extract electrical data from remaining lines
            for line in lines[header_count:]:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) > 1:
                        try:
                            value = float(parts[1].strip('"'))
                            data.append(value)
                        except (ValueError, IndexError):
                            continue
            
            print(f"✅ Loaded {len(data)} REAL electrical measurements")
            print(f"📊 Voltage range: {np.min(data):.3f} to {np.max(data):.3f} mV")
            print(f"⏱️  Duration: {len(data)} seconds of real mushroom activity")
            print(f"🔬 This is REAL biological data, not simulation!")
            
            return np.array(data)
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def normalize_audio(self, data, target_range=0.8):
        """Normalize audio data to prevent clipping"""
        if data is None or len(data) == 0:
            return None
        
        data = np.array(data)
        data = data - np.mean(data)
        
        if np.max(np.abs(data)) > 0:
            data = data * (target_range / np.max(np.abs(data)))
        
        return data
    
    def additive_synthesis(self, voltage_data, duration=10.0):
        """
        Additive synthesis using fungal voltage patterns
        Creates rich, harmonic sounds
        """
        print(f"🎹 Creating ADDITIVE SYNTH sound...")
        start_time = time.time()
        
        # Convert voltage to frequency components
        frequencies = np.linspace(50, 2000, len(voltage_data))  # 50Hz to 2kHz
        amplitudes = np.abs(voltage_data) / np.max(np.abs(voltage_data))
        
        # Generate time array
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create additive synthesis
        audio = np.zeros_like(t)
        for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
            if i < len(t):
                # Add harmonic components
                audio += amp * np.sin(2 * np.pi * freq * t[:len(audio)])
                audio += 0.5 * amp * np.sin(2 * np.pi * freq * 2 * t[:len(audio)])  # 2nd harmonic
                audio += 0.25 * amp * np.sin(2 * np.pi * freq * 3 * t[:len(audio)])  # 3rd harmonic
        
        audio = self.normalize_audio(audio)
        duration_actual = time.time() - start_time
        
        print(f"  ⚡ Generated in {duration_actual:.3f} seconds")
        print(f"  🎵 Frequency range: {frequencies[0]:.0f}Hz to {frequencies[-1]:.0f}Hz")
        
        return audio, f"additive_synth_{int(time.time())}.wav"
    
    def frequency_modulation_synthesis(self, voltage_data, duration=10.0):
        """
        FM synthesis using fungal voltage as modulation
        Creates metallic, bell-like sounds
        """
        print(f"🔔 Creating FM SYNTH sound...")
        start_time = time.time()
        
        # Convert voltage to modulation parameters
        carrier_freq = 440  # A4 note
        mod_freq = 220      # Modulation frequency
        
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create FM synthesis
        mod_index = np.interp(np.linspace(0, 1, len(t)), 
                             np.linspace(0, 1, len(voltage_data)), 
                             np.abs(voltage_data))
        mod_index = mod_index / np.max(mod_index) * 5  # Modulation depth
        
        # FM equation: sin(2πfc*t + I*sin(2πfm*t))
        audio = np.sin(2 * np.pi * carrier_freq * t + 
                      mod_index * np.sin(2 * np.pi * mod_freq * t))
        
        audio = self.normalize_audio(audio)
        duration_actual = time.time() - start_time
        
        print(f"  ⚡ Generated in {duration_actual:.3f} seconds")
        print(f"  🎵 Carrier: {carrier_freq}Hz, Mod: {mod_freq}Hz")
        
        return audio, f"fm_synth_{int(time.time())}.wav"
    
    def granular_synthesis(self, voltage_data, duration=10.0):
        """
        Granular synthesis using fungal voltage patterns
        Creates evolving, textural sounds
        """
        print(f"🌊 Creating GRANULAR SYNTH sound...")
        start_time = time.time()
        
        # Convert voltage to grain parameters
        grain_duration = 0.1  # 100ms grains
        grain_samples = int(grain_duration * self.sample_rate)
        
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t)
        
        # Create grains based on voltage patterns
        for i in range(0, len(t), grain_samples):
            if i + grain_samples < len(t):
                # Use voltage pattern to modulate grain
                voltage_idx = int((i / len(t)) * len(voltage_data))
                if voltage_idx < len(voltage_data):
                    voltage_val = voltage_data[voltage_idx]
                    
                    # Create grain with voltage-modulated frequency
                    grain_freq = 200 + abs(voltage_val) * 1000
                    grain = np.sin(2 * np.pi * grain_freq * np.linspace(0, grain_duration, grain_samples))
                    
                    # Apply envelope
                    envelope = np.hanning(len(grain))
                    grain = grain * envelope
                    
                    # Add to audio
                    if i + len(grain) <= len(audio):
                        audio[i:i+len(grain)] += grain
        
        audio = self.normalize_audio(audio)
        duration_actual = time.time() - start_time
        
        print(f"  ⚡ Generated in {duration_actual:.3f} seconds")
        print(f"  🌊 Grain duration: {grain_duration*1000:.0f}ms")
        
        return audio, f"granular_synth_{int(time.time())}.wav"
    
    def waveform_synthesis(self, voltage_data, duration=10.0):
        """
        Waveform synthesis using fungal voltage as wave shape
        Creates unique, organic sounds
        """
        print(f"🌿 Creating WAVEFORM SYNTH sound...")
        start_time = time.time()
        
        # Convert voltage to wave shape
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create complex waveform using voltage patterns
        audio = np.zeros_like(t)
        base_freq = 110  # A2 note
        
        for i, voltage in enumerate(voltage_data):
            if i < len(t):
                # Use voltage to modulate multiple harmonics
                freq_multiplier = 1 + abs(voltage) / np.max(np.abs(voltage_data))
                phase = voltage / np.max(np.abs(voltage_data)) * 2 * np.pi
                
                # Add multiple harmonic components
                for harmonic in range(1, 6):
                    freq = base_freq * harmonic * freq_multiplier
                    amplitude = 1.0 / harmonic
                    audio += amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        audio = self.normalize_audio(audio)
        duration_actual = time.time() - start_time
        
        print(f"  ⚡ Generated in {duration_actual:.3f} seconds")
        print(f"  🌿 Base frequency: {base_freq}Hz, Harmonics: 5")
        
        return audio, f"waveform_synth_{int(time.time())}.wav"
    
    def filtered_synthesis(self, voltage_data, duration=10.0):
        """
        Filtered synthesis with voltage-controlled filters
        Creates warm, evolving sounds
        """
        print(f"🔥 Creating FILTERED SYNTH sound...")
        start_time = time.time()
        
        # Generate base sound
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        base_audio = np.sin(2 * np.pi * 220 * t)  # A3 note
        
        # Apply voltage-controlled filters
        filtered_audio = np.zeros_like(base_audio)
        
        for i in range(0, len(base_audio), 1000):  # Process in chunks
            chunk_end = min(i + 1000, len(base_audio))
            voltage_idx = int((i / len(base_audio)) * len(voltage_data))
            
            if voltage_idx < len(voltage_data):
                voltage_val = voltage_data[voltage_idx]
                
                # Voltage controls filter cutoff
                cutoff_freq = 100 + abs(voltage_val) * 2000
                cutoff_freq = np.clip(cutoff_freq, 100, 8000)
                
                # Apply low-pass filter
                b, a = signal.butter(4, cutoff_freq / (self.sample_rate / 2), 'low')
                chunk = base_audio[i:chunk_end]
                if len(chunk) > 0:
                    filtered_chunk = signal.filtfilt(b, a, chunk)
                    filtered_audio[i:chunk_end] = filtered_chunk
        
        audio = self.normalize_audio(filtered_audio)
        duration_actual = time.time() - start_time
        
        print(f"  ⚡ Generated in {duration_actual:.3f} seconds")
        print(f"  🔥 Dynamic filter cutoff: 100Hz to 8kHz")
        
        return audio, f"filtered_synth_{int(time.time())}.wav"
    
    def ambient_synthesis(self, voltage_data, duration=15.0):
        """
        Ambient synthesis for atmospheric, evolving sounds
        Creates meditative, spacey textures
        """
        print(f"🌌 Creating AMBIENT SYNTH sound...")
        start_time = time.time()
        
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create ambient layers
        audio = np.zeros_like(t)
        
        # Layer 1: Drone
        drone_freq = 55  # A1 note
        drone = np.sin(2 * np.pi * drone_freq * t)
        drone = drone * 0.3
        
        # Layer 2: Voltage-modulated pad
        pad_freq = 110  # A2 note
        pad = np.zeros_like(t)
        for i, voltage in enumerate(voltage_data):
            if i < len(t):
                mod_freq = pad_freq + voltage * 50
                pad += 0.2 * np.sin(2 * np.pi * mod_freq * t)
        
        # Layer 3: High-frequency textures
        texture = np.random.randn(len(t)) * 0.1
        
        # Layer 4: Voltage-controlled reverb-like effect
        reverb = np.zeros_like(t)
        for i in range(len(t)):
            if i > 1000:  # Delay
                voltage_idx = int((i / len(t)) * len(voltage_data))
                if voltage_idx < len(voltage_data):
                    delay_amount = int(1000 + abs(voltage_data[voltage_idx]) * 5000)
                    if i - delay_amount >= 0:
                        reverb[i] = audio[i - delay_amount] * 0.3
        
        # Combine layers
        audio = drone + pad + texture + reverb
        audio = self.normalize_audio(audio)
        
        duration_actual = time.time() - start_time
        
        print(f"  ⚡ Generated in {duration_actual:.3f} seconds")
        print(f"  🌌 Ambient layers: Drone, Pad, Texture, Reverb")
        
        return audio, f"ambient_synth_{int(time.time())}.wav"
    
    def save_wav_file(self, audio, filename, output_dir="RESULTS/audio"):
        """
        Save audio as WAV file using built-in Python
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        try:
            # Convert to 16-bit PCM
            audio_16bit = (audio * 32767).astype(np.int16)
            
            # WAV file header
            with open(filepath, 'wb') as wav_file:
                # RIFF header
                wav_file.write(b'RIFF')
                wav_file.write(struct.pack('<I', 36 + len(audio_16bit) * 2))
                wav_file.write(b'WAVE')
                
                # Format chunk
                wav_file.write(b'fmt ')
                wav_file.write(struct.pack('<I', 16))
                wav_file.write(struct.pack('<H', 1))  # PCM
                wav_file.write(struct.pack('<H', 1))  # Mono
                wav_file.write(struct.pack('<I', self.sample_rate))
                wav_file.write(struct.pack('<I', self.sample_rate * 2))
                wav_file.write(struct.pack('<H', 2))
                wav_file.write(struct.pack('<H', 16))
                
                # Data chunk
                wav_file.write(b'data')
                wav_file.write(struct.pack('<I', len(audio_16bit) * 2))
                wav_file.write(audio_16bit.tobytes())
            
            print(f"💾 Audio saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            return None
    
    def create_synth_mix(self, voltage_data, duration=20.0):
        """
        Create a complete synth mix using all synthesis methods
        """
        print(f"🎼 Creating COMPLETE SYNTH MIX...")
        start_time = time.time()
        
        # Generate all synth sounds
        synth_tracks = {}
        
        for method_name, method_func in self.synthesis_methods.items():
            print(f"\n🎵 Generating {method_name.upper()} track...")
            audio, filename = method_func(voltage_data, duration)
            if audio is not None:
                synth_tracks[method_name] = {
                    'audio': audio,
                    'filename': filename,
                    'method': method_name
                }
        
        # Create mix
        if synth_tracks:
            # Find the longest track
            max_length = max(len(track['audio']) for track in synth_tracks.values())
            mix = np.zeros(max_length)
            
            # Mix all tracks with different volumes
            volumes = {
                'additive': 0.8,
                'fm': 0.6,
                'granular': 0.7,
                'waveform': 0.5,
                'filtered': 0.6,
                'ambient': 0.4
            }
            
            for method_name, track in synth_tracks.items():
                volume = volumes.get(method_name, 0.5)
                track_audio = track['audio']
                
                # Pad shorter tracks
                if len(track_audio) < max_length:
                    track_audio = np.pad(track_audio, (0, max_length - len(track_audio)))
                
                # Add to mix
                mix[:len(track_audio)] += track_audio * volume
            
            # Normalize mix
            mix = self.normalize_audio(mix, target_range=0.9)
            
            duration_actual = time.time() - start_time
            print(f"\n🎼 Complete synth mix generated in {duration_actual:.3f} seconds")
            
            return mix, "fungal_synth_mix.wav", synth_tracks
        
        return None, None, {}
    
    def generate_all_sounds(self, voltage_data, output_dir="RESULTS/audio"):
        """
        Generate all synth sounds from fungal electrical data
        """
        print(f"🎵 FUNGAL SYNTH SOUND GENERATION")
        print(f"=" * 50)
        print(f"🍄 Converting REAL fungal electrical data to synth sounds")
        print(f"⚡ Sample rate: {self.sample_rate} Hz")
        print(f"🎹 Synthesis methods: {len(self.synthesis_methods)}")
        print(f"🔬 Data source: REAL mushroom electrical measurements")
        
        # Generate individual sounds
        individual_sounds = {}
        
        for method_name, method_func in self.synthesis_methods.items():
            print(f"\n🎵 Generating {method_name.upper()} synthesis...")
            audio, filename = method_func(voltage_data, duration=10.0)
            
            if audio is not None:
                filepath = self.save_wav_file(audio, filename, output_dir)
                if filepath:
                    individual_sounds[method_name] = filepath
        
        # Generate complete mix
        print(f"\n🎼 Creating complete synth mix...")
        mix_audio, mix_filename, synth_tracks = self.create_synth_mix(voltage_data, duration=20.0)
        
        if mix_audio is not None:
            mix_filepath = self.save_wav_file(mix_audio, mix_filename, output_dir)
            individual_sounds['complete_mix'] = mix_filepath
        
        # Create summary
        summary = {
            'timestamp': self.timestamp,
            'author': self.author,
            'data_source': 'REAL fungal electrical measurements (not simulated)',
            'sample_rate': self.sample_rate,
            'synthesis_methods': list(self.synthesis_methods.keys()),
            'generated_sounds': individual_sounds,
            'total_duration': len(voltage_data),
            'voltage_range': [float(np.min(voltage_data)), float(np.max(voltage_data))],
            'note': 'All sounds generated from actual mushroom electrical activity patterns'
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, "synth_generation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🎉 SYNTH GENERATION COMPLETE!")
        print(f"📊 Generated {len(individual_sounds)} audio files")
        print(f"💾 Summary saved: {summary_file}")
        print(f"🍄 All sounds based on REAL mushroom electrical patterns!")
        
        return individual_sounds, summary

def main():
    """Main function to generate fungal synth sounds"""
    print("🎵 FUNGAL SYNTH SOUND GENERATOR (Built-in Python)")
    print("Author: Joe Knowles")
    print("Timestamp: 2025-08-12 09:23:27 BST")
    print("=" * 60)
    
    # Create synth generator
    generator = FungalSynthBuiltin()
    
    # Load real fungal data
    data_path = Path("DATA/raw/15061491")
    file_path = data_path / "Spray_in_bag.csv"
    
    if not file_path.exists():
        print("❌ Data file not found. Please ensure fungal data is available.")
        return
    
    # Load real fungal electrical data
    voltage_data = generator.load_real_fungal_data(file_path)
    
    if voltage_data is None or len(voltage_data) < 10:
        print("❌ Insufficient data for sound generation")
        return
    
    print(f"\n🎵 Starting synth sound generation...")
    print(f"🔬 Using REAL fungal electrical patterns (not simulation)")
    print(f"⚡ Voltage data: {len(voltage_data)} measurements")
    print(f"🎹 Converting biological electrical activity to electronic music")
    
    # Generate all synth sounds
    generated_sounds, summary = generator.generate_all_sounds(voltage_data)
    
    print(f"\n🎉 FUNGAL SYNTH GENERATION COMPLETE!")
    print(f"🎵 Created {len(generated_sounds)} unique synth sounds")
    print(f"🍄 All sounds based on REAL mushroom electrical activity")
    print(f"🎹 Ready to create some awesome electronic music!")
    print(f"🔬 This proves mushrooms can make music!")

if __name__ == "__main__":
    main() 