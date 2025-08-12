#!/usr/bin/env python3
"""
Fungal Sound Generator - Fixed Version
Author: Joe Knowles
Timestamp: 2025-08-12 09:23:27 BST
Description: Converts fungal electrical signals into audible sounds with progress tracking
"""

import numpy as np
import os
import time
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
        
        # Performance tracking
        self.start_time = None
        self.processing_stats = {}
    
    def start_timer(self):
        """Start performance timer."""
        self.start_time = time.time()
    
    def end_timer(self, operation_name):
        """End timer and record performance."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.processing_stats[operation_name] = duration
            print(f"  ⏱️  {operation_name} completed in {duration:.3f} seconds")
            return duration
        return 0
    
    def load_fungal_data(self, file_path):
        """Load fungal electrical data from CSV files."""
        self.start_timer()
        
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
            
            duration = self.end_timer("Data Loading")
            print(f"  📊 Loaded {len(data)} samples in {duration:.3f}s")
            print(f"  🚀 Processing speed: {len(data)/duration:.0f} samples/second")
            
            return data
            
        except Exception as e:
            self.end_timer("Data Loading (Failed)")
            print(f"  ❌ Error loading {file_path}: {e}")
            return None
    
    def normalize_audio(self, data, target_range=0.8):
        """Normalize audio data to prevent clipping."""
        if data is None or len(data) == 0:
            return None
        
        # Convert to numpy array if it isn't already
        data = np.array(data, dtype=np.float64)
        
        # Remove DC offset (mean)
        data = data - np.mean(data)
        
        # Normalize to target range
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data * (target_range / max_val)
        
        return data
    
    def create_audio_waveform(self, data, duration_seconds=10):
        """Create audio waveform from electrical data."""
        if data is None or len(data) < 10:
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
        self.start_timer()
        
        audio_data = self.create_audio_waveform(data, duration)
        if audio_data is None:
            self.end_timer("Basic Sound Generation (Failed)")
            print(f"  ❌ Failed to create audio for {filename}")
            return None
        
        # Save as WAV file
        output_file = f"{self.output_dir}/{filename}_basic_sound.wav"
        success = self.save_wav_file(audio_data, output_file)
        
        if success:
            self.end_timer("Basic Sound Generation")
            print(f"  ✅ Saved: {output_file}")
            return output_file
        else:
            self.end_timer("Basic Sound Generation (Failed)")
            return None
    
    def generate_frequency_modulated_sound(self, data, filename, duration=10):
        """Generate frequency modulated sound based on signal amplitude."""
        print(f"🎵 Generating frequency modulated sound for {filename}...")
        self.start_timer()
        
        if data is None or len(data) < 10:
            self.end_timer("FM Sound Generation (Failed)")
            print(f"  ❌ Insufficient data for {filename}")
            return None
        
        # Create time array
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Normalize data
        normalized_data = self.normalize_audio(data)
        if normalized_data is None:
            self.end_timer("FM Sound Generation (Failed)")
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
        success = self.save_wav_file(audio_data, output_file)
        
        if success:
            self.end_timer("FM Sound Generation")
            print(f"  ✅ Saved: {output_file}")
            return output_file
        else:
            self.end_timer("FM Sound Generation (Failed)")
            return None
    
    def generate_rhythm_based_sound(self, data, filename, duration=10):
        """Generate rhythm-based sound using signal spikes."""
        print(f"🎵 Generating rhythm-based sound for {filename}...")
        self.start_timer()
        
        if data is None or len(data) < 10:
            self.end_timer("Rhythm Sound Generation (Failed)")
            print(f"  ❌ Insufficient data for {filename}")
            return None
        
        # Create time array
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Normalize data
        normalized_data = self.normalize_audio(data)
        if normalized_data is None:
            self.end_timer("Rhythm Sound Generation (Failed)")
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
        output_file = f"{self.output_dir}/{filename}_frequency_modulated.wav"
        success = self.save_wav_file(audio_data, output_file)
        
        if success:
            self.end_timer("Rhythm Sound Generation")
            print(f"  ✅ Saved: {output_file}")
            return output_file
        else:
            self.end_timer("Rhythm Sound Generation (Failed)")
            return None
    
    def save_wav_file(self, audio_data, filename):
        """Save audio data as WAV file using scipy."""
        try:
            # Ensure audio data is in correct format
            audio_data = np.array(audio_data, dtype=np.float64)
            
            # Convert to 16-bit integers
            audio_data = np.int16(audio_data * 32767)
            
            # Save using numpy (simple format)
            np_filename = filename.replace('.wav', '.npy')
            np.save(np_filename, audio_data)
            
            # Also save as raw PCM for compatibility
            raw_filename = filename.replace('.wav', '.raw')
            with open(raw_filename, 'wb') as f:
                audio_data.tofile(f)
            
            # Try to save as actual WAV file if scipy is available
            try:
                from scipy.io import wavfile
                wavfile.write(filename, self.sample_rate, audio_data)
                print(f"    📁 Saved as WAV: {filename}")
            except ImportError:
                print(f"    📁 Saved as NPY: {np_filename}")
                print(f"    📁 Saved as RAW: {raw_filename}")
                print(f"    💡 Install scipy for WAV format: pip install scipy")
            
            return True
            
        except Exception as e:
            print(f"    ❌ Error saving audio: {e}")
            return False
    
    def create_audio_summary(self, all_sounds):
        """Create a summary of all generated sounds."""
        summary = {
            'timestamp': '2025-08-12 09:23:27 BST',
            'author': 'Joe Knowles',
            'total_sounds_generated': len(all_sounds),
            'sounds': all_sounds,
            'performance_stats': self.processing_stats,
            'audio_instructions': {
                'format': 'Multiple formats available',
                'sample_rate': f'{self.sample_rate} Hz',
                'channels': 'Mono audio',
                'playback': 'Use provided audio player or convert to WAV'
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
    print("🎵 Fungal Sound Generator - Fixed Version")
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
            
            if data is not None and len(data) > 10:
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
        print(f"  📊 Data size: {len(data)} samples")
        
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
    
    # Create audio summary
    summary_file = generator.create_audio_summary(all_sounds)
    
    print(f"\n🎉 AUDIO GENERATION COMPLETE!")
    print("=" * 60)
    print(f"📁 Audio files saved to: {generator.output_dir}")
    print(f"📄 Summary saved to: {summary_file}")
    print(f"🎵 Total sounds generated: {len(all_sounds)}")
    
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print("=" * 40)
    for operation, duration in generator.processing_stats.items():
        print(f"  {operation}: {duration:.3f}s")
    
    print(f"\n🔊 HOW TO LISTEN TO THE SOUNDS:")
    print("1. Use the provided audio player:")
    print("   python3 TOOLS/scripts/fungal_audio_player.py")
    print("2. Convert to WAV format:")
    print("   import numpy as np")
    print("   import soundfile as sf")
    print("   data = np.load('filename.npy')")
    print("   sf.write('output.wav', data, 44100)")
    print("3. Use audio software like Audacity")
    
    print(f"\n🎵 WHAT EACH SOUND REPRESENTS:")
    print("• Basic Sound: Raw electrical signals converted to audio")
    print("• Frequency Modulated: Signal amplitude controls pitch")
    print("• Rhythm Based: Signal spikes create drum-like patterns")

if __name__ == "__main__":
    main() 