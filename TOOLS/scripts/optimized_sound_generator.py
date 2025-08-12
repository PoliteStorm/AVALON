#!/usr/bin/env python3
"""
Optimized Fungal Sound Generator with Real-Time Progress Tracking
Author: Joe Knowles
Timestamp: 2025-08-12 09:23:27 BST
Description: High-efficiency sound generation with progress bars and timing estimates
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import struct
from pathlib import Path
import json
from datetime import datetime
import threading

class OptimizedSoundGenerator:
    """
    Optimized sound generator with real-time progress tracking
    """
    
    def __init__(self):
        self.timestamp = "20250812_092327"
        self.author = "Joe Knowles"
        self.sample_rate = 44100
        self.results_path = Path("RESULTS")
        
        # Performance tracking
        self.generation_timings = {}
        self.progress_callbacks = []
        
        # Synthesis methods with efficiency ratings
        self.synthesis_methods = {
            'additive': {
                'function': self.additive_synthesis_ultra_fast,
                'efficiency': 'Ultra-Fast',
                'complexity': 'Low',
                'estimated_time': 0.5
            },
            'fm': {
                'function': self.frequency_modulation_synthesis_ultra_fast,
                'efficiency': 'Ultra-Fast',
                'complexity': 'Low',
                'estimated_time': 0.3
            },
            'granular': {
                'function': self.granular_synthesis_ultra_fast,
                'efficiency': 'Fast',
                'complexity': 'Medium',
                'estimated_time': 1.2
            },
            'waveform': {
                'function': self.waveform_synthesis_ultra_fast,
                'efficiency': 'Fast',
                'complexity': 'Medium',
                'estimated_time': 2.0
            },
            'filtered': {
                'function': self.filtered_synthesis_ultra_fast,
                'efficiency': 'Fast',
                'complexity': 'Medium',
                'estimated_time': 1.5
            },
            'ambient': {
                'function': self.ambient_synthesis_ultra_fast,
                'efficiency': 'Fast',
                'complexity': 'Medium',
                'estimated_time': 1.8
            }
        }
    
    def add_progress_callback(self, callback):
        """Add progress callback function"""
        self.progress_callbacks.append(callback)
    
    def update_progress(self, method_name, current, total, estimated_remaining):
        """Update progress for all callbacks"""
        progress_data = {
            'method': method_name,
            'current': current,
            'total': total,
            'percentage': (current / total) * 100,
            'estimated_remaining': estimated_remaining,
            'timestamp': datetime.now().isoformat()
        }
        
        for callback in self.progress_callbacks:
            try:
                callback(progress_data)
            except:
                pass
    
    def print_progress(self, progress_data):
        """Default progress printer"""
        method = progress_data['method']
        current = progress_data['current']
        total = progress_data['total']
        percentage = progress_data['percentage']
        remaining = progress_data['estimated_remaining']
        
        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"\r🎵 {method.upper()}: [{bar}] {percentage:5.1f}% ({current}/{total}) "
              f"⏱️  Est. remaining: {remaining:.1f}s", end='', flush=True)
        
        if current == total:
            print()  # New line when complete
    
    def ultra_fast_normalize(self, data, target_range=0.8):
        """Ultra-fast audio normalization using vectorized operations"""
        if data is None or len(data) == 0:
            return None
        
        data = np.array(data, dtype=np.float32)  # Use float32 for speed
        data = data - np.mean(data)
        
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data * (target_range / max_val)
        
        return data
    
    def additive_synthesis_ultra_fast(self, voltage_data, t, duration=8.0, progress_callback=None):
        """Ultra-fast additive synthesis with progress tracking"""
        start_time = time.time()
        
        # Pre-compute frequency and amplitude arrays
        frequencies = np.linspace(50, 1500, len(voltage_data))
        amplitudes = np.abs(voltage_data) / np.max(np.abs(voltage_data))
        
        # Generate time array
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t_audio, dtype=np.float32)
        
        # Vectorized synthesis with progress tracking
        total_harmonics = len(frequencies)
        
        for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
            if i < len(t_audio):
                # Vectorized harmonic generation
                harmonics = [1.0, 0.5, 0.25]
                for j, harmonic_amp in enumerate(harmonics):
                    if i + j < len(t_audio):
                        audio += harmonic_amp * amp * np.sin(2 * np.pi * freq * (j + 1) * t_audio[:len(audio)])
                
                # Progress update every 10 iterations
                if i % 10 == 0 and progress_callback:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / (i + 1)) * (total_harmonics - i - 1)
                    progress_callback('additive', i + 1, total_harmonics, remaining)
        
        # Final progress update
        if progress_callback:
            progress_callback('additive', total_harmonics, total_harmonics, 0)
        
        audio = self.ultra_fast_normalize(audio)
        generation_time = time.time() - start_time
        
        return audio, f"additive_synth_{int(time.time())}.wav", generation_time
    
    def frequency_modulation_synthesis_ultra_fast(self, voltage_data, t, duration=8.0, progress_callback=None):
        """Ultra-fast FM synthesis with progress tracking"""
        start_time = time.time()
        
        carrier_freq = 440
        mod_freq = 220
        
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Vectorized modulation index computation
        mod_index = np.interp(np.linspace(0, 1, len(t_audio)), 
                             np.linspace(0, 1, len(voltage_data)), 
                             np.abs(voltage_data))
        mod_index = mod_index / np.max(mod_index) * 3
        
        # Vectorized FM synthesis
        audio = np.sin(2 * np.pi * carrier_freq * t_audio + 
                      mod_index * np.sin(2 * np.pi * mod_freq * t_audio))
        
        if progress_callback:
            progress_callback('fm', 1, 1, 0)
        
        audio = self.ultra_fast_normalize(audio)
        generation_time = time.time() - start_time
        
        return audio, f"fm_synth_{int(time.time())}.wav", generation_time
    
    def granular_synthesis_ultra_fast(self, voltage_data, t, duration=8.0, progress_callback=None):
        """Ultra-fast granular synthesis with progress tracking"""
        start_time = time.time()
        
        grain_duration = 0.1
        grain_samples = int(grain_duration * self.sample_rate)
        
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t_audio, dtype=np.float32)
        
        # Pre-compute envelope
        envelope = np.hanning(grain_samples)
        
        # Calculate total grains
        total_grains = len(t_audio) // grain_samples
        
        # Vectorized grain generation with progress tracking
        for i in range(0, len(t_audio), grain_samples):
            if i + grain_samples < len(t_audio):
                voltage_idx = int((i / len(t_audio)) * len(voltage_data))
                if voltage_idx < len(voltage_data):
                    voltage_val = voltage_data[voltage_idx]
                    grain_freq = 200 + abs(voltage_val) * 800
                    
                    # Vectorized grain synthesis
                    grain = np.sin(2 * np.pi * grain_freq * np.linspace(0, grain_duration, grain_samples))
                    grain = grain * envelope
                    
                    if i + len(grain) <= len(audio):
                        audio[i:i+len(grain)] += grain
                
                # Progress update every 50 grains
                grain_count = i // grain_samples
                if grain_count % 50 == 0 and progress_callback:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / (grain_count + 1)) * (total_grains - grain_count - 1)
                    progress_callback('granular', grain_count + 1, total_grains, remaining)
        
        # Final progress update
        if progress_callback:
            progress_callback('granular', total_grains, total_grains, 0)
        
        audio = self.ultra_fast_normalize(audio)
        generation_time = time.time() - start_time
        
        return audio, f"granular_synth_{int(time.time())}.wav", generation_time
    
    def waveform_synthesis_ultra_fast(self, voltage_data, t, duration=8.0, progress_callback=None):
        """Ultra-fast waveform synthesis with progress tracking"""
        start_time = time.time()
        
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t_audio, dtype=np.float32)
        base_freq = 110
        
        total_voltage_points = len(voltage_data)
        
        # Vectorized synthesis with progress tracking
        for i, voltage in enumerate(voltage_data):
            if i < len(t_audio):
                # Use voltage to modulate multiple harmonics
                freq_multiplier = 1 + abs(voltage) / np.max(np.abs(voltage_data))
                phase = voltage / np.max(np.abs(voltage_data)) * 2 * np.pi
                
                # Vectorized harmonic generation
                for harmonic in range(1, 6):
                    freq = base_freq * harmonic * freq_multiplier
                    amplitude = 1.0 / harmonic
                    audio += amplitude * np.sin(2 * np.pi * freq * t_audio + phase)
                
                # Progress update every 20 iterations
                if i % 20 == 0 and progress_callback:
                    elapsed = time.time() - start_time
                    remaining = (elapsed / (i + 1)) * (total_voltage_points - i - 1)
                    progress_callback('waveform', i + 1, total_voltage_points, remaining)
        
        # Final progress update
        if progress_callback:
            progress_callback('waveform', total_voltage_points, total_voltage_points, 0)
        
        audio = self.ultra_fast_normalize(audio)
        generation_time = time.time() - start_time
        
        return audio, f"waveform_synth_{int(time.time())}.wav", generation_time
    
    def filtered_synthesis_ultra_fast(self, voltage_data, t, duration=8.0, progress_callback=None):
        """Ultra-fast filtered synthesis with progress tracking"""
        start_time = time.time()
        
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        base_audio = np.sin(2 * np.pi * 220 * t_audio)
        filtered_audio = np.zeros_like(base_audio, dtype=np.float32)
        
        # Process in larger chunks for efficiency
        chunk_size = 5000
        total_chunks = len(base_audio) // chunk_size
        
        for i in range(0, len(base_audio), chunk_size):
            chunk_end = min(i + chunk_size, len(base_audio))
            voltage_idx = int((i / len(base_audio)) * len(voltage_data))
            
            if voltage_idx < len(voltage_data):
                voltage_val = voltage_data[voltage_idx]
                cutoff_freq = 100 + abs(voltage_val) * 2000
                cutoff_freq = np.clip(cutoff_freq, 100, 8000)
                
                # Apply low-pass filter
                b, a = signal.butter(4, cutoff_freq / (self.sample_rate / 2), 'low')
                chunk = base_audio[i:chunk_end]
                if len(chunk) > 0:
                    filtered_chunk = signal.filtfilt(b, a, chunk)
                    filtered_audio[i:chunk_end] = filtered_chunk
            
            # Progress update
            chunk_count = i // chunk_size
            if progress_callback:
                elapsed = time.time() - start_time
                remaining = (elapsed / (chunk_count + 1)) * (total_chunks - chunk_count - 1)
                progress_callback('filtered', chunk_count + 1, total_chunks, remaining)
        
        # Final progress update
        if progress_callback:
            progress_callback('filtered', total_chunks, total_chunks, 0)
        
        audio = self.ultra_fast_normalize(filtered_audio)
        generation_time = time.time() - start_time
        
        return audio, f"filtered_synth_{int(time.time())}.wav", generation_time
    
    def ambient_synthesis_ultra_fast(self, voltage_data, t, duration=15.0, progress_callback=None):
        """Ultra-fast ambient synthesis with progress tracking"""
        start_time = time.time()
        
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create ambient layers with progress tracking
        if progress_callback:
            progress_callback('ambient', 1, 4, 0)  # Layer 1
        
        # Layer 1: Drone
        drone_freq = 55
        drone = np.sin(2 * np.pi * drone_freq * t_audio)
        drone = drone * 0.3
        
        if progress_callback:
            progress_callback('ambient', 2, 4, 0)  # Layer 2
        
        # Layer 2: Voltage-modulated pad
        pad_freq = 110
        pad = np.zeros_like(t_audio, dtype=np.float32)
        
        # Vectorized pad generation
        for i, voltage in enumerate(voltage_data):
            if i < len(t_audio):
                mod_freq = pad_freq + voltage * 50
                pad += 0.2 * np.sin(2 * np.pi * mod_freq * t_audio)
        
        if progress_callback:
            progress_callback('ambient', 3, 4, 0)  # Layer 3
        
        # Layer 3: High-frequency textures
        texture = np.random.randn(len(t_audio)) * 0.1
        
        if progress_callback:
            progress_callback('ambient', 4, 4, 0)  # Layer 4
        
        # Combine layers
        audio = drone + pad + texture
        audio = self.ultra_fast_normalize(audio)
        
        generation_time = time.time() - start_time
        
        return audio, f"ambient_synth_{int(time.time())}.wav", generation_time
    
    def save_wav_file_optimized(self, audio, filename, output_dir, metadata=None):
        """Optimized WAV file saving"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        try:
            # Use float32 for efficiency, convert to 16-bit at the end
            audio_16bit = (audio * 32767).astype(np.int16)
            
            with open(filepath, 'wb') as wav_file:
                wav_file.write(b'RIFF')
                wav_file.write(struct.pack('<I', 36 + len(audio_16bit) * 2))
                wav_file.write(b'WAVE')
                
                wav_file.write(b'fmt ')
                wav_file.write(struct.pack('<I', 16))
                wav_file.write(struct.pack('<H', 1))
                wav_file.write(struct.pack('<H', 1))
                wav_file.write(struct.pack('<I', self.sample_rate))
                wav_file.write(struct.pack('<I', self.sample_rate * 2))
                wav_file.write(struct.pack('<H', 2))
                wav_file.write(struct.pack('<H', 16))
                
                wav_file.write(b'data')
                wav_file.write(struct.pack('<I', len(audio_16bit) * 2))
                wav_file.write(audio_16bit.tobytes())
            
            # Save metadata if provided
            if metadata:
                metadata_file = filepath.replace('.wav', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            return filepath
            
        except Exception as e:
            print(f"❌ Error saving audio: {e}")
            return None
    
    def generate_all_sounds_optimized(self, voltage_data, t, mushroom_info, output_dir="RESULTS/audio"):
        """Generate all sounds with real-time progress tracking"""
        print(f"🎵 OPTIMIZED SOUND GENERATION WITH PROGRESS TRACKING")
        print(f"🍄 Species: {mushroom_info.get('species', 'Unknown')}")
        print(f"🧬 Strain: {mushroom_info.get('strain', 'Unknown')}")
        print(f"⚡ Sample rate: {self.sample_rate} Hz")
        print(f"=" * 60)
        
        # Add default progress callback
        self.add_progress_callback(self.print_progress)
        
        start_time = time.time()
        generated_sounds = {}
        generation_times = {}
        
        # Estimate total time
        total_estimated_time = sum(method['estimated_time'] for method in self.synthesis_methods.values())
        print(f"⏱️  Estimated total generation time: {total_estimated_time:.1f} seconds")
        print(f"🚀 Using ultra-fast synthesis algorithms")
        print()
        
        for method_name, method_info in self.synthesis_methods.items():
            print(f"🎵 Generating {method_name.upper()} synthesis...")
            print(f"  Efficiency: {method_info['efficiency']}")
            print(f"  Complexity: {method_info['complexity']}")
            print(f"  Estimated time: {method_info['estimated_time']:.1f}s")
            
            # Generate sound with progress tracking
            audio, filename, generation_time = method_info['function'](
                voltage_data, t, progress_callback=self.update_progress
            )
            
            if audio is not None:
                # Save with metadata
                metadata = {
                    'mushroom_info': mushroom_info,
                    'synthesis_method': method_name,
                    'efficiency': method_info['efficiency'],
                    'generation_time': generation_time,
                    'timestamp': self.timestamp,
                    'author': self.author
                }
                
                filepath = self.save_wav_file_optimized(audio, filename, output_dir, metadata)
                if filepath:
                    generated_sounds[method_name] = filepath
                    generation_times[method_name] = generation_time
                    
                    print(f"  ✅ Generated in {generation_time:.3f}s (Est: {method_info['estimated_time']:.1f}s)")
                    print(f"  💾 Saved: {os.path.basename(filepath)}")
                else:
                    print(f"  ❌ Failed to save audio")
            else:
                print(f"  ❌ Failed to generate audio")
            
            print()
        
        # Generate complete mix
        print(f"🎼 Creating complete synth mix...")
        mix_start = time.time()
        
        mix_audio, mix_filename, mix_generation_time = self.create_optimized_mix(
            voltage_data, t, duration=12.0
        )
        
        if mix_audio is not None:
            metadata = {
                'mushroom_info': mushroom_info,
                'synthesis_method': 'complete_mix',
                'efficiency': 'Ultra-Fast',
                'generation_time': mix_generation_time,
                'timestamp': self.timestamp,
                'author': self.author
            }
            
            filepath = self.save_wav_file_optimized(mix_audio, mix_filename, output_dir, metadata)
            if filepath:
                generated_sounds['complete_mix'] = filepath
                generation_times['complete_mix'] = mix_generation_time
                
                print(f"  ✅ Mix generated in {mix_generation_time:.3f}s")
                print(f"  💾 Saved: {os.path.basename(filepath)}")
        
        total_time = time.time() - start_time
        
        # Performance summary
        print(f"\n🎉 SOUND GENERATION COMPLETE!")
        print(f"📊 Performance Summary:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Estimated time: {total_estimated_time:.1f}s")
        print(f"  Speed improvement: {total_estimated_time/total_time:.1f}x faster than estimate")
        print(f"  Sounds generated: {len(generated_sounds)}")
        print(f"  Average time per sound: {total_time/len(generated_sounds):.3f}s")
        
        # Save performance metrics
        self.generation_timings = {
            'total_time': total_time,
            'estimated_time': total_estimated_time,
            'speed_improvement': total_estimated_time/total_time,
            'individual_times': generation_times,
            'timestamp': datetime.now().isoformat()
        }
        
        return generated_sounds, self.generation_timings
    
    def create_optimized_mix(self, voltage_data, t, duration=12.0):
        """Create optimized mix of all sounds"""
        start_time = time.time()
        
        # Generate base sounds efficiently
        base_sounds = {}
        
        # Additive synthesis (fastest)
        audio, _, _ = self.additive_synthesis_ultra_fast(voltage_data, t, duration)
        if audio is not None:
            base_sounds['additive'] = audio * 0.8
        
        # FM synthesis (fastest)
        audio, _, _ = self.frequency_modulation_synthesis_ultra_fast(voltage_data, t, duration)
        if audio is not None:
            base_sounds['fm'] = audio * 0.6
        
        # Granular synthesis
        audio, _, _ = self.granular_synthesis_ultra_fast(voltage_data, t, duration)
        if audio is not None:
            base_sounds['granular'] = audio * 0.7
        
        # Create mix
        if base_sounds:
            max_length = max(len(sound) for sound in base_sounds.values())
            mix = np.zeros(max_length, dtype=np.float32)
            
            for sound in base_sounds.values():
                if len(sound) < max_length:
                    sound = np.pad(sound, (0, max_length - len(sound)))
                mix += sound
            
            mix = self.ultra_fast_normalize(mix, target_range=0.9)
            generation_time = time.time() - start_time
            
            return mix, "fungal_synth_mix.wav", generation_time
        
        return None, None, 0

def main():
    """Test the optimized sound generator"""
    print("🎵 OPTIMIZED FUNGAL SOUND GENERATOR")
    print("Author: Joe Knowles")
    print("Timestamp: 2025-08-12 09:23:27 BST")
    print("=" * 60)
    
    # Create generator
    generator = OptimizedSoundGenerator()
    
    # Test with sample data
    print("🧪 Testing with sample fungal data...")
    
    # Generate sample voltage data
    t = np.linspace(0, 100, 1000)
    voltage_data = np.sin(2 * np.pi * 0.1 * t) + 0.5 * np.random.randn(len(t))
    
    mushroom_info = {
        'species': 'Test Pleurotus ostreatus',
        'strain': 'Test strain',
        'treatment': 'Test conditions',
        'electrode_type': 'Test electrodes'
    }
    
    # Generate sounds with progress tracking
    sounds, timings = generator.generate_all_sounds_optimized(
        voltage_data, t, mushroom_info
    )
    
    print(f"\n🎉 Test complete! Generated {len(sounds)} sounds")
    print(f"⚡ Total time: {timings['total_time']:.3f}s")
    print(f"🚀 Speed improvement: {timings['speed_improvement']:.1f}x")

if __name__ == "__main__":
    main() 