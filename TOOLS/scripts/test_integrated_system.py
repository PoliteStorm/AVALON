#!/usr/bin/env python3
"""
Test Integrated Fungal Analysis System
Author: Joe Knowles
Timestamp: 2025-08-12 09:23:27 BST
Description: Test the integrated system with specific fungal data files
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

class TestIntegratedSystem:
    """
    Test the integrated fungal analysis system
    """
    
    def __init__(self):
        self.timestamp = "20250812_164721"  # Updated to current time
        self.author = "Joe Knowles"
        self.sample_rate = 44100
        self.start_time = time.time()  # Track when analysis started
        
    def load_fungal_data(self, filepath):
        """Load fungal electrical data from CSV file"""
        print(f"🍄 Loading: {filepath}")
        
        try:
            data = []
            with open(filepath, 'r') as f:
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
            
            if len(data) < 10:
                print(f"❌ Insufficient data: {len(data)} measurements")
                return None, None
            
            # Create time array
            t = np.linspace(0, len(data), len(data))
            V_t = np.array(data)
            
            print(f"✅ Loaded {len(data)} electrical measurements")
            print(f"📊 Voltage range: {np.min(V_t):.3f} to {np.max(V_t):.3f} mV")
            print(f"⏱️  Duration: {len(data)} seconds")
            
            return V_t, t
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None
    
    def sqrt_wave_transform_analysis(self, V_t, t, k_range=None, tau_range=None):
        """√t Wave Transform Analysis"""
        print(f"🔬 Computing √t Wave Transform...")
        start_time = time.time()
        
        if k_range is None:
            k_range = np.linspace(0.1, 5.0, 20)
        if tau_range is None:
            signal_duration = t[-1] - t[0]
            tau_range = np.logspace(-1, np.log10(signal_duration), 15)
        
        print(f"  Equation: W(k,τ) = ∫₀^T V(t) · ψ(√t/τ) · e^(-ik√t) dt")
        print(f"  k range: {k_range[0]:.3f} to {k_range[-1]:.3f}")
        print(f"  τ range: {tau_range[0]:.3f} to {tau_range[-1]:.3f}")
        
        # Initialize result matrix
        W_matrix = np.zeros((len(k_range), len(tau_range)), dtype=complex)
        
        # Progress tracking
        total_computations = len(k_range) * len(tau_range)
        completed = 0
        
        for i, k in enumerate(k_range):
            for j, tau in enumerate(tau_range):
                # Compute wave transform for this (k, τ) pair
                W_val = self.compute_single_wave_transform(V_t, t, k, tau)
                W_matrix[i, j] = W_val
                
                completed += 1
                if completed % 30 == 0:
                    progress = (completed / total_computations) * 100
                    print(f"  Progress: {progress:.1f}% ({completed}/{total_computations})")
        
        duration = time.time() - start_time
        print(f"  ✅ Wave transform completed in {duration:.3f} seconds")
        print(f"  🚀 Speed: {total_computations/duration:.1f} computations/second")
        
        return W_matrix, k_range, tau_range
    
    def compute_single_wave_transform(self, V_t, t, k, tau):
        """Compute single wave transform value"""
        max_t = t[-1]
        t_integration = np.logspace(-3, np.log10(max_t), 300)
        V_interp = np.interp(t_integration, t, V_t)
        
        integrand_values = np.zeros(len(t_integration), dtype=complex)
        
        for i, t_val in enumerate(t_integration):
            psi_val = self.mother_wavelet(t_val, tau)
            exp_val = np.exp(-1j * k * np.sqrt(t_val))
            integrand_values[i] = V_interp[i] * psi_val * exp_val
        
        dt = np.diff(t_integration)
        integral = np.sum(0.5 * (integrand_values[:-1] + integrand_values[1:]) * dt)
        
        return integral
    
    def mother_wavelet(self, t, tau):
        """Mother wavelet function ψ(√t/τ)"""
        if t <= 0 or tau <= 0:
            return 0.0
        
        normalized_t = np.sqrt(t) / np.sqrt(tau)
        
        if abs(normalized_t) > 5.0:
            return 0.0
        
        omega_0 = 2.0
        gaussian = np.exp(-normalized_t**2 / 2)
        complex_exp = np.exp(1j * omega_0 * normalized_t)
        norm_factor = 1.0 / np.sqrt(2 * np.pi)
        
        return norm_factor * gaussian * complex_exp
    
    def pattern_recognition_analysis(self, W_matrix, k_range, tau_range):
        """Pattern recognition analysis"""
        print(f"🧠 Performing Pattern Recognition Analysis...")
        
        magnitude = np.abs(W_matrix)
        
        # Find dominant patterns
        max_magnitude_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        max_k = k_range[max_magnitude_idx[0]]
        max_tau = tau_range[max_magnitude_idx[1]]
        max_magnitude = magnitude[max_magnitude_idx]
        
        # Pattern characteristics
        coherence = np.std(magnitude) / np.mean(magnitude)
        pattern_energy = np.sum(magnitude**2)
        
        results = {
            'dominant_pattern': {
                'k_value': max_k,
                'tau_value': max_tau,
                'magnitude': max_magnitude
            },
            'pattern_characteristics': {
                'coherence': coherence,
                'total_energy': pattern_energy,
                'mean_magnitude': np.mean(magnitude)
            }
        }
        
        print(f"  🧠 Dominant pattern: k={max_k:.3f}, τ={max_tau:.3f}")
        print(f"  🧠 Pattern coherence: {coherence:.3f}")
        
        return results
    
    def generate_synth_sounds(self, V_t, t, output_dir="RESULTS/audio"):
        """Generate synth sounds from fungal data"""
        print(f"🎵 Generating Synth Sounds...")
        start_time = time.time()
        
        os.makedirs(output_dir, exist_ok=True)
        generated_sounds = {}
        
        total_sounds = 3  # additive, FM, granular
        current_sound = 0
        
        # Generate additive synthesis
        current_sound += 1
        print(f"  🎹 [{current_sound}/{total_sounds}] Generating additive synthesis...")
        print(f"     📊 Processing {len(V_t)} voltage samples...")
        audio, filename = self.additive_synthesis(V_t, t, duration=8.0)
        if audio is not None:
            print(f"     💾 Saving additive synthesis audio...")
            filepath = self.save_wav_file(audio, filename, output_dir)
            if filepath:
                generated_sounds['additive'] = filepath
                print(f"     ✅ Additive synthesis saved: {os.path.basename(filepath)}")
            else:
                print(f"     ❌ Failed to save additive synthesis")
        else:
            print(f"     ❌ Failed to generate additive synthesis")
        
        # Generate FM synthesis
        current_sound += 1
        print(f"  🔔 [{current_sound}/{total_sounds}] Generating FM synthesis...")
        print(f"     📊 Processing {len(V_t)} voltage samples...")
        audio, filename = self.frequency_modulation_synthesis(V_t, t, duration=8.0)
        if audio is not None:
            print(f"     💾 Saving FM synthesis audio...")
            filepath = self.save_wav_file(audio, filename, output_dir)
            if filepath:
                generated_sounds['fm'] = filepath
                print(f"     ✅ FM synthesis saved: {os.path.basename(filepath)}")
            else:
                print(f"     ❌ Failed to save FM synthesis")
        else:
            print(f"     ❌ Failed to generate FM synthesis")
        
        # Generate granular synthesis
        current_sound += 1
        print(f"  🌊 [{current_sound}/{total_sounds}] Generating granular synthesis...")
        print(f"     📊 Processing {len(V_t)} voltage samples...")
        audio, filename = self.granular_synthesis(V_t, t, duration=8.0)
        if audio is not None:
            print(f"     💾 Saving granular synthesis audio...")
            filepath = self.save_wav_file(audio, filename, output_dir)
            if filepath:
                generated_sounds['granular'] = filepath
                print(f"     ✅ Granular synthesis saved: {os.path.basename(filepath)}")
            else:
                print(f"     ❌ Failed to save granular synthesis")
        else:
            print(f"     ❌ Failed to generate granular synthesis")
        
        duration = time.time() - start_time
        print(f"  ✅ Sound generation completed in {duration:.3f} seconds")
        print(f"  🎵 Generated {len(generated_sounds)} sound files")
        
        return generated_sounds
    
    def additive_synthesis(self, voltage_data, t, duration=8.0):
        """Additive synthesis"""
        print(f"       🔧 Setting up additive synthesis parameters...")
        frequencies = np.linspace(50, 1500, len(voltage_data))
        amplitudes = np.abs(voltage_data) / np.max(np.abs(voltage_data))
        
        print(f"       ⏱️  Creating audio timeline ({duration}s duration)...")
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t_audio)
        
        print(f"       🎵 Generating {len(voltage_data)} frequency components...")
        total_components = len(voltage_data)
        for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
            if i < len(t_audio):
                audio += amp * np.sin(2 * np.pi * freq * t_audio[:len(audio)])
                audio += 0.5 * amp * np.sin(2 * np.pi * freq * 2 * t_audio[:len(audio)])
                
                # Show progress every 10% of components
                if (i + 1) % max(1, total_components // 10) == 0:
                    progress = ((i + 1) / total_components) * 100
                    print(f"         📈 Progress: {progress:.0f}% ({i+1}/{total_components} components)")
        
        print(f"       🔊 Normalizing audio levels...")
        audio = self.normalize_audio(audio)
        # Use actual timestamp for filename
        actual_timestamp = int(time.time())
        filename = f"additive_synth_{actual_timestamp}.wav"
        print(f"       ✅ Additive synthesis complete: {filename}")
        return audio, filename
    
    def frequency_modulation_synthesis(self, voltage_data, t, duration=8.0):
        """FM synthesis"""
        print(f"       🔧 Setting up FM synthesis parameters...")
        carrier_freq = 440
        mod_freq = 220
        
        print(f"       ⏱️  Creating audio timeline ({duration}s duration)...")
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        
        print(f"       📊 Interpolating modulation index from {len(voltage_data)} voltage samples...")
        mod_index = np.interp(np.linspace(0, 1, len(t_audio)), 
                             np.linspace(0, 1, len(voltage_data)), 
                             np.abs(voltage_data))
        mod_index = mod_index / np.max(mod_index) * 3
        
        print(f"       🎵 Generating FM audio with carrier {carrier_freq}Hz, modulator {mod_freq}Hz...")
        audio = np.sin(2 * np.pi * carrier_freq * t_audio + 
                      mod_index * np.sin(2 * np.pi * mod_freq * t_audio))
        
        print(f"       🔊 Normalizing audio levels...")
        audio = self.normalize_audio(audio)
        # Use actual timestamp for filename
        actual_timestamp = int(time.time())
        filename = f"fm_synth_{actual_timestamp}.wav"
        print(f"       ✅ FM synthesis complete: {filename}")
        return audio, filename
    
    def granular_synthesis(self, voltage_data, t, duration=8.0):
        """Granular synthesis"""
        print(f"       🔧 Setting up granular synthesis parameters...")
        grain_duration = 0.1
        grain_samples = int(grain_duration * self.sample_rate)
        
        print(f"       ⏱️  Creating audio timeline ({duration}s duration)...")
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t_audio)
        
        total_grains = len(t_audio) // grain_samples
        print(f"       🌊 Generating {total_grains} audio grains...")
        
        for i in range(0, len(t_audio), grain_samples):
            if i + grain_samples < len(t_audio):
                voltage_idx = int((i / len(t_audio)) * len(voltage_data))
                if voltage_idx < len(voltage_data):
                    voltage_val = voltage_data[voltage_idx]
                    grain_freq = 200 + abs(voltage_val) * 800
                    grain = np.sin(2 * np.pi * grain_freq * np.linspace(0, grain_duration, grain_samples))
                    envelope = np.hanning(len(grain))
                    grain = grain * envelope
                    
                    if i + len(grain) <= len(audio):
                        audio[i:i+len(grain)] += grain
                
                # Show progress every 10% of grains
                grain_count = i // grain_samples
                if grain_count % max(1, total_grains // 10) == 0:
                    progress = (grain_count / total_grains) * 100
                    print(f"         📈 Progress: {progress:.0f}% ({grain_count}/{total_grains} grains)")
        
        print(f"       🔊 Normalizing audio levels...")
        audio = self.normalize_audio(audio)
        # Use actual timestamp for filename
        actual_timestamp = int(time.time())
        filename = f"granular_synth_{actual_timestamp}.wav"
        print(f"       ✅ Granular synthesis complete: {filename}")
        return audio, filename
    
    def normalize_audio(self, data, target_range=0.8):
        """Normalize audio data"""
        if data is None or len(data) == 0:
            return None
        
        data = np.array(data)
        data = data - np.mean(data)
        
        if np.max(np.abs(data)) > 0:
            data = data * (target_range / np.max(np.abs(data)))
        
        return data
    
    def save_wav_file(self, audio, filename, output_dir):
        """Save audio as WAV file"""
        filepath = os.path.join(output_dir, filename)
        
        try:
            print(f"         💾 Converting audio to 16-bit format...")
            audio_16bit = (audio * 32767).astype(np.int16)
            
            print(f"         📁 Writing WAV file: {os.path.basename(filepath)}")
            with open(filepath, 'wb') as wav_file:
                print(f"           📝 Writing RIFF header...")
                wav_file.write(b'RIFF')
                wav_file.write(struct.pack('<I', 36 + len(audio_16bit) * 2))
                wav_file.write(b'WAVE')
                
                print(f"           📝 Writing format chunk...")
                wav_file.write(b'fmt ')
                wav_file.write(struct.pack('<I', 16))
                wav_file.write(struct.pack('<H', 1))
                wav_file.write(struct.pack('<H', 1))
                wav_file.write(struct.pack('<I', self.sample_rate))
                wav_file.write(struct.pack('<I', self.sample_rate * 2))
                wav_file.write(struct.pack('<H', 2))
                wav_file.write(struct.pack('<H', 16))
                
                print(f"           📝 Writing audio data...")
                wav_file.write(b'data')
                wav_file.write(struct.pack('<I', len(audio_16bit) * 2))
                wav_file.write(audio_16bit.tobytes())
            
            print(f"         ✅ Audio saved successfully: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"         ❌ Error saving audio: {e}")
            return None
    
    def run_test_analysis(self):
        """Run test analysis on specific fungal files"""
        print(f"🔬 TESTING INTEGRATED FUNGAL ANALYSIS SYSTEM")
        print(f"Author: {self.author}")
        print(f"Timestamp: {self.timestamp}")
        print(f"=" * 60)
        
        # Test files to analyze
        test_files = [
            "DATA/raw/15061491/Spray_in_bag.csv",
            "DATA/raw/15061491/New_Oyster_with spray.csv",
            "DATA/raw/15061491/Blue_oyster_31_5_22.csv"
        ]
        
        all_results = {}
        total_files = len(test_files)
        
        print(f"📁 Found {total_files} test files to analyze")
        print(f"=" * 60)
        
        for file_idx, filepath in enumerate(test_files, 1):
            if os.path.exists(filepath):
                print(f"\n🔬 [{file_idx}/{total_files}] Analyzing: {os.path.basename(filepath)}")
                print(f"=" * 50)
                
                # Load data
                print(f"📊 Loading fungal electrical data...")
                V_t, t = self.load_fungal_data(filepath)
                if V_t is None:
                    print(f"❌ Skipping {os.path.basename(filepath)} due to loading error")
                    continue
                
                # Perform √t wave transform analysis
                print(f"🔬 Starting wave transform analysis...")
                W_matrix, k_range, tau_range = self.sqrt_wave_transform_analysis(V_t, t)
                
                # Perform pattern recognition
                print(f"🧠 Starting pattern recognition analysis...")
                pattern_results = self.pattern_recognition_analysis(W_matrix, k_range, tau_range)
                
                # Generate synth sounds
                print(f"🎵 Starting sound generation...")
                synth_sounds = self.generate_synth_sounds(V_t, t)
                
                # Store results
                all_results[os.path.basename(filepath)] = {
                    'wave_transform': {
                        'W_matrix': W_matrix,
                        'k_range': k_range,
                        'tau_range': tau_range
                    },
                    'pattern_recognition': pattern_results,
                    'synth_sounds': synth_sounds
                }
                
                print(f"✅ Analysis complete for {os.path.basename(filepath)} ({file_idx}/{total_files})")
                print(f"   🧠 Patterns detected: {len(pattern_results)}")
                print(f"   🎵 Sounds generated: {len(synth_sounds)}")
            else:
                print(f"❌ File not found: {filepath}")
        
        # Save results
        if all_results:
            print(f"\n💾 Saving analysis results...")
            os.makedirs("RESULTS", exist_ok=True)
            results_file = "RESULTS/test_analysis_results.json"
            
            # Convert numpy arrays to lists for JSON
            serializable_results = {}
            for key, value in all_results.items():
                serializable_results[key] = {
                    'wave_transform': {
                        'k_range': value['wave_transform']['k_range'].tolist(),
                        'tau_range': value['wave_transform']['tau_range'].tolist(),
                        'matrix_shape': value['wave_transform']['W_matrix'].shape
                    },
                    'pattern_recognition': value['pattern_recognition'],
                    'synth_sounds': value['synth_sounds']
                }
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"📊 Test results saved: {results_file}")
            
            # Create summary
            print(f"\n📋 ANALYSIS SUMMARY:")
            for filename, results in all_results.items():
                pattern = results['pattern_recognition']['dominant_pattern']
                print(f"  🍄 {filename}:")
                print(f"    • Dominant k: {pattern['k_value']:.3f}")
                print(f"    • Dominant τ: {pattern['tau_value']:.3f}")
                print(f"    • Magnitude: {pattern['magnitude']:.3f}")
                print(f"    • Synth sounds: {len(results['synth_sounds'])}")
                print()
        
        print(f"🎉 TEST ANALYSIS COMPLETE!")
        print(f"🔬 Analyzed {len(all_results)} fungal species")
        print(f"🎵 Generated synth sounds for each")
        print(f"🧠 Ready for new discoveries!")

def main():
    """Main function"""
    tester = TestIntegratedSystem()
    tester.run_test_analysis()

if __name__ == "__main__":
    main() 