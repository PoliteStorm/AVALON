#!/usr/bin/env python3
"""
Fungal Sound Generator
Author: Joe Knowles
Timestamp: 2025-08-12 09:23:27 BST
Description: Converts fungal electrical signals into audible sounds
"""

import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime

class FungalSoundGenerator:
    """Generates audible sounds from fungal electrical signals."""
    
    def __init__(self):
        self.timestamp = "20250812_092327"
        self.author = "Joe Knowles"
        self.sample_rate = 44100  # CD quality audio
        self.output_dir = "RESULTS/audio"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_fungal_data(self, file_path):
        """Load fungal electrical data from CSV files."""
        try:
            data = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines and extract electrical data
            header_count = 0
            for line in lines:
                if line.strip() and not line.startswith('"'):
                    header_count += 1
                if header_count > 2:  # Skip first two header lines
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
            
            print(f"Loaded {len(data)} samples from {file_path}")
            return data
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def normalize_audio(self, data, target_range=0.8):
        """Normalize audio data to prevent clipping."""
        if data is None or len(data) == 0:
            return None
        
        # Convert to numpy array if it isn't already
        data = np.array(data)
        
        # Remove DC offset (mean)
        data = data - np.mean(data)
        
        # Normalize to target range
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data * (target_range / max_val)
        
        return data
    
    def create_audio_waveform(self, data, duration_seconds=10):
        """Create audio waveform from electrical data."""
        if not data or len(data) < 10:
            return None
        
        # Resample data to target duration
        target_samples = int(self.sample_rate * duration_seconds)
        
        if len(data) > target_samples:
            # Downsample by taking every nth sample
            step = len(data) // target_samples
            data = data[::step][:target_samples]
        else:
            # Upsample by repeating data
            repeats = target_samples // len(data) + 1
            data = np.tile(data, repeats)[:target_samples]
        
        # Normalize audio
        audio_data = self.normalize_audio(data)
        
        return audio_data
    
    def generate_basic_sound(self, data, filename, duration=10):
        """Generate basic sound from electrical data."""
        print(f"🎵 Generating basic sound for {filename}...")
        
        audio_data = self.create_audio_waveform(data, duration)
        if audio_data is None:
            print(f"  ❌ Failed to create audio for {filename}")
            return None
        
        # Save as WAV file
        output_file = f"{self.output_dir}/{filename}_basic_sound.wav"
        self.save_wav_file(audio_data, output_file)
        
        print(f"  ✅ Saved: {output_file}")
        return output_file
    
    def generate_frequency_modulated_sound(self, data, filename, duration=10):
        """Generate frequency modulated sound based on signal amplitude."""
        print(f"🎵 Generating frequency modulated sound for {filename}...")
        
        if not data or len(data) < 10:
            print(f"  ❌ Insufficient data for {filename}")
            return None
        
        # Create time array
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Normalize data
        normalized_data = self.normalize_audio(data)
        if normalized_data is None:
            return None
        
        # Resample data to match time array
        if len(normalized_data) > len(t):
            step = len(normalized_data) // len(t)
            normalized_data = normalized_data[::step][:len(t)]
        else:
            repeats = len(t) // len(normalized_data) + 1
            normalized_data = np.tile(normalized_data, repeats)[:len(t)]
        
        # Create frequency modulation based on signal amplitude
        base_freq = 440  # A4 note
        freq_modulation = 200  # Hz range for modulation
        
        # Map signal amplitude to frequency
        frequencies = base_freq + freq_modulation * normalized_data
        
        # Generate FM sound
        phase = np.cumsum(2 * np.pi * frequencies / self.sample_rate)
        audio_data = 0.3 * np.sin(phase)
        
        # Save as WAV file
        output_file = f"{self.output_dir}/{filename}_frequency_modulated.wav"
        self.save_wav_file(audio_data, output_file)
        
        print(f"  ✅ Saved: {output_file}")
        return output_file
    
    def generate_rhythm_based_sound(self, data, filename, duration=10):
        """Generate rhythm-based sound using signal spikes."""
        print(f"🎵 Generating rhythm-based sound for {filename}...")
        
        if not data or len(data) < 10:
            print(f"  ❌ Insufficient data for {filename}")
            return None
        
        # Create time array
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Normalize data
        normalized_data = self.normalize_audio(data)
        if normalized_data is None:
            return None
        
        # Resample data to match time array
        if len(normalized_data) > len(t):
            step = len(normalized_data) // len(t)
            normalized_data = normalized_data[::step][:len(t)]
        else:
            repeats = len(t) // len(normalized_data) + 1
            normalized_data = np.tile(normalized_data, repeats)[:len(t)]
        
        # Detect spikes (peaks above threshold)
        threshold = np.std(normalized_data) * 2
        spike_indices = np.where(np.abs(normalized_data) > threshold)[0]
        
        # Create rhythm pattern
        audio_data = np.zeros(len(t))
        
        # Add drum-like sounds at spike locations
        for idx in spike_indices:
            if idx < len(t):
                # Create a short drum sound
                drum_duration = int(0.1 * self.sample_rate)  # 0.1 seconds
                drum_sound = np.exp(-np.linspace(0, 5, drum_duration)) * 0.5
                
                # Place drum sound at spike location
                end_idx = min(idx + drum_duration, len(audio_data))
                drum_length = end_idx - idx
                audio_data[idx:end_idx] += drum_sound[:drum_length]
        
        # Add background tone
        background_freq = 220  # A3 note
        background = 0.1 * np.sin(2 * np.pi * background_freq * t)
        audio_data += background
        
        # Normalize final audio
        audio_data = self.normalize_audio(audio_data, 0.8)
        
        # Save as WAV file
        output_file = f"{self.output_dir}/{filename}_rhythm_based.wav"
        self.save_wav_file(audio_data, output_file)
        
        print(f"  ✅ Saved: {output_file}")
        return output_file
    
    def generate_relationship_sound(self, data1, data2, filename1, filename2, duration=10):
        """Generate sound showing relationship between two datasets."""
        print(f"🎵 Generating relationship sound: {filename1} ↔ {filename2}")
        
        if not data1 or not data2 or len(data1) < 10 or len(data2) < 10:
            print(f"  ❌ Insufficient data for relationship sound")
            return None
        
        # Create time array
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Normalize both datasets
        norm_data1 = self.normalize_audio(data1)
        norm_data2 = self.normalize_audio(data2)
        
        if norm_data1 is None or norm_data2 is None:
            return None
        
        # Resample data to match time array
        for data, name in [(norm_data1, "data1"), (norm_data2, "data2")]:
            if len(data) > len(t):
                step = len(data) // len(t)
                if name == "data1":
                    norm_data1 = data[::step][:len(t)]
                else:
                    norm_data2 = data[::step][:len(t)]
            else:
                repeats = len(t) // len(data) + 1
                if name == "data1":
                    norm_data1 = np.tile(data, repeats)[:len(t)]
                else:
                    norm_data2 = np.tile(data, repeats)[:len(t)]
        
        # Create stereo sound (left = data1, right = data2)
        left_channel = 0.4 * norm_data1
        right_channel = 0.4 * norm_data2
        
        # Add correlation indicator (center channel)
        correlation = np.correlate(norm_data1, norm_data2, mode='same')
        correlation = self.normalize_audio(correlation, 0.3)
        center_channel = correlation
        
        # Combine channels
        audio_data = np.column_stack([left_channel, right_channel, center_channel])
        
        # Save as WAV file
        output_file = f"{self.output_dir}/{filename1}_vs_{filename2}_relationship.wav"
        self.save_wav_file(audio_data, output_file, stereo=True)
        
        print(f"  ✅ Saved: {output_file}")
        return output_file
    
    def save_wav_file(self, audio_data, filename, stereo=False):
        """Save audio data as WAV file."""
        try:
            # Ensure audio data is in correct format
            if stereo:
                # Multi-channel audio
                audio_data = np.array(audio_data)
                if len(audio_data.shape) == 1:
                    audio_data = audio_data.reshape(-1, 1)
            else:
                # Mono audio
                audio_data = np.array(audio_data)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()
            
            # Convert to 16-bit integers
            audio_data = np.int16(audio_data * 32767)
            
            # Save using numpy (simple WAV format)
            np.save(filename.replace('.wav', '.npy'), audio_data)
            
            # Also save as raw PCM for compatibility
            raw_filename = filename.replace('.wav', '.raw')
            with open(raw_filename, 'wb') as f:
                audio_data.tofile(f)
            
            print(f"    📁 Saved as: {filename.replace('.wav', '.npy')}")
            print(f"    📁 Saved as: {raw_filename}")
            
        except Exception as e:
            print(f"    ❌ Error saving audio: {e}")
    
    def create_audio_summary(self, all_sounds):
        """Create a summary of all generated sounds."""
        summary = {
            'timestamp': '2025-08-12 09:23:27 BST',
            'author': 'Joe Knowles',
            'total_sounds_generated': len(all_sounds),
            'sounds': all_sounds,
            'audio_instructions': {
                'format': 'Numpy arrays (.npy) and raw PCM (.raw)',
                'sample_rate': f'{self.sample_rate} Hz',
                'channels': 'Mono and Stereo available',
                'playback': 'Use audio software or Python to play files'
            }
        }
        
        # Save summary
        summary_file = f"{self.output_dir}/audio_generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Audio summary saved to: {summary_file}")
        return summary_file

def main():
    """Main function to generate fungal sounds."""
    print("🎵 Fungal Sound Generator")
    print("Author: Joe Knowles")
    print("Timestamp: 2025-08-12 09:23:27 BST")
    print("=" * 60)
    
    # Create sound generator
    generator = FungalSoundGenerator()
    
    # Load fungal data
    data_path = Path("DATA/raw/15061491")
    data_files = {}
    
    # Target files for audio generation
    target_files = [
        "Spray_in_bag.csv",
        "Spray_in_bag_crop.csv", 
        "New_Oyster_with spray.csv",
        "New_Oyster_with spray_as_mV.csv"
    ]
    
    print("📁 Loading Fungal Electrical Data for Audio Generation...")
    for filename in target_files:
        file_path = data_path / filename
        if file_path.exists():
            print(f"  Loading {filename}...")
            data = generator.load_fungal_data(file_path)
            
            if data and len(data) > 10:
                data_files[filename] = data
                print(f"    ✓ Loaded {len(data)} samples")
            else:
                print(f"    ✗ Insufficient data")
        else:
            print(f"  ✗ File not found: {filename}")
    
    if not data_files:
        print("❌ No valid fungal data found. Exiting.")
        return
    
    print(f"✅ Loaded {len(data_files)} fungal data files")
    
    # Generate sounds
    all_sounds = []
    
    print(f"\n🎵 Generating Audio from Fungal Signals...")
    print("=" * 60)
    
    for filename, data in data_files.items():
        print(f"\n🎵 Processing: {filename}")
        
        # Generate different types of sounds
        sound_files = []
        
        # 1. Basic sound
        basic_sound = generator.generate_basic_sound(data, filename, duration=15)
        if basic_sound:
            sound_files.append(("Basic Sound", basic_sound))
        
        # 2. Frequency modulated sound
        fm_sound = generator.generate_frequency_modulated_sound(data, filename, duration=15)
        if fm_sound:
            sound_files.append(("Frequency Modulated", fm_sound))
        
        # 3. Rhythm based sound
        rhythm_sound = generator.generate_rhythm_based_sound(data, filename, duration=15)
        if rhythm_sound:
            sound_files.append(("Rhythm Based", rhythm_sound))
        
        all_sounds.append({
            'filename': filename,
            'samples': len(data),
            'sounds_generated': sound_files
        })
    
    # Generate relationship sounds
    print(f"\n🔗 Generating Relationship Audio...")
    print("=" * 60)
    
    filenames = list(data_files.keys())
    for i, filename1 in enumerate(filenames):
        for j, filename2 in enumerate(filenames[i+1:], i+1):
            data1 = data_files[filename1]
            data2 = data_files[filename2]
            
            relationship_sound = generator.generate_relationship_sound(
                data1, data2, filename1, filename2, duration=15
            )
            
            if relationship_sound:
                all_sounds.append({
                    'filename': f"{filename1} ↔ {filename2}",
                    'type': 'relationship',
                    'sounds_generated': [("Relationship", relationship_sound)]
                })
    
    # Create audio summary
    summary_file = generator.create_audio_summary(all_sounds)
    
    print(f"\n🎉 AUDIO GENERATION COMPLETE!")
    print("=" * 60)
    print(f"📁 Audio files saved to: {generator.output_dir}")
    print(f"📄 Summary saved to: {summary_file}")
    print(f"🎵 Total sounds generated: {len(all_sounds)}")
    
    print(f"\n🔊 HOW TO LISTEN TO THE SOUNDS:")
    print("1. Use Python with librosa or soundfile:")
    print("   import numpy as np")
    print("   import soundfile as sf")
    print("   data = np.load('filename.npy')")
    print("   sf.write('output.wav', data, 44100)")
    print("2. Use audio software like Audacity")
    print("3. Convert .raw files to .wav using audio converters")
    
    print(f"\n🎵 WHAT EACH SOUND REPRESENTS:")
    print("• Basic Sound: Raw electrical signals converted to audio")
    print("• Frequency Modulated: Signal amplitude controls pitch")
    print("• Rhythm Based: Signal spikes create drum-like patterns")
    print("• Relationship: Stereo comparison between two datasets")

if __name__ == "__main__":
    main() 