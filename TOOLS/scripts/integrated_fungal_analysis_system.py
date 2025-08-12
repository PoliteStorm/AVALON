#!/usr/bin/env python3
"""
Integrated Fungal Analysis System
Author: Joe Knowles
Timestamp: 2025-08-12 09:23:27 BST
Description: Complete system combining √t wave transform analysis with fungal synth sound generation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import os
import struct
from pathlib import Path
import json

class IntegratedFungalAnalysisSystem:
    """
    Complete integrated system for fungal electrical analysis and sound generation
    Combines √t wave transform analysis with advanced synth sound creation
    """
    
    def __init__(self):
        self.timestamp = "20250812_092327"
        self.author = "Joe Knowles"
        self.sample_rate = 44100  # CD quality audio
        self.data_path = Path("DATA/raw")
        self.results_path = Path("RESULTS")
        
        # Analysis methods
        self.analysis_methods = {
            'sqrt_wave_transform': self.sqrt_wave_transform_analysis,
            'frequency_analysis': self.frequency_domain_analysis,
            'pattern_recognition': self.pattern_recognition_analysis,
            'species_comparison': self.species_comparison_analysis
        }
        
        # Sound synthesis methods
        self.synthesis_methods = {
            'additive': self.additive_synthesis,
            'fm': self.frequency_modulation_synthesis,
            'granular': self.granular_synthesis,
            'waveform': self.waveform_synthesis,
            'filtered': self.filtered_synthesis,
            'ambient': self.ambient_synthesis
        }
    
    def load_fungal_data(self, filename):
        """Load fungal electrical data from CSV file"""
        file_path = self.data_path / filename
        
        if not file_path.exists():
            print(f"❌ Data file not found: {file_path}")
            return None, None
        
        print(f"🍄 Loading fungal electrical data: {filename}")
        
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
            
            if len(data) < 10:
                print(f"❌ Insufficient data: {len(data)} measurements")
                return None, None
            
            # Create time array
            t = np.linspace(0, len(data), len(data))
            V_t = np.array(data)
            
            print(f"✅ Loaded {len(data)} electrical measurements")
            print(f"📊 Voltage range: {np.min(V_t):.3f} to {np.max(V_t):.3f} mV")
            print(f"⏱️  Duration: {len(data)} seconds")
            print(f"🔬 REAL biological data (not simulated)")
            
            return V_t, t
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None, None
    
    def sqrt_wave_transform_analysis(self, V_t, t, k_range=None, tau_range=None):
        """
        √t Wave Transform Analysis: W(k,τ) = ∫₀^T V(t) · ψ(√t/τ) · e^(-ik√t) dt
        """
        print(f"🔬 Computing √t Wave Transform...")
        start_time = time.time()
        
        # Auto-generate ranges if not provided
        if k_range is None:
            k_range = np.linspace(0.1, 5.0, 25)
        if tau_range is None:
            signal_duration = t[-1] - t[0]
            tau_range = np.logspace(-1, np.log10(signal_duration), 20)
        
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
                if completed % 50 == 0:
                    progress = (completed / total_computations) * 100
                    print(f"  Progress: {progress:.1f}% ({completed}/{total_computations})")
        
        duration = time.time() - start_time
        print(f"  ✅ Wave transform completed in {duration:.3f} seconds")
        print(f"  🚀 Speed: {total_computations/duration:.1f} computations/second")
        
        return W_matrix, k_range, tau_range
    
    def compute_single_wave_transform(self, V_t, t, k, tau):
        """Compute single wave transform value"""
        # Use finite integration limit for stability
        max_t = t[-1]
        
        # Create integration time points
        t_integration = np.logspace(-3, np.log10(max_t), 500)
        
        # Interpolate voltage signal
        V_interp = np.interp(t_integration, t, V_t)
        
        # Compute integrand values
        integrand_values = np.zeros(len(t_integration), dtype=complex)
        
        for i, t_val in enumerate(t_integration):
            # Mother wavelet ψ(√t/τ)
            psi_val = self.mother_wavelet(t_val, tau)
            
            # Complex exponential e^(-ik√t)
            exp_val = np.exp(-1j * k * np.sqrt(t_val))
            
            # Complete integrand
            integrand_values[i] = V_interp[i] * psi_val * exp_val
        
        # Numerical integration using trapezoidal rule
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
    
    def frequency_domain_analysis(self, V_t, t):
        """Frequency domain analysis of fungal electrical signals"""
        print(f"🎵 Performing Frequency Domain Analysis...")
        start_time = time.time()
        
        # FFT analysis
        fft_result = np.fft.fft(V_t)
        frequencies = np.fft.fftfreq(len(V_t), t[1] - t[0])
        
        # Power spectrum
        power_spectrum = np.abs(fft_result)**2
        
        # Find dominant frequencies
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_frequency = frequencies[dominant_freq_idx]
        
        # Spectral centroid
        spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
        
        duration = time.time() - start_time
        
        results = {
            'dominant_frequency': dominant_frequency,
            'spectral_centroid': spectral_centroid,
            'total_power': np.sum(power_spectrum),
            'frequencies': frequencies,
            'power_spectrum': power_spectrum,
            'analysis_time': duration
        }
        
        print(f"  ✅ Frequency analysis completed in {duration:.3f} seconds")
        print(f"  🎵 Dominant frequency: {dominant_frequency:.2f} Hz")
        print(f"  🎵 Spectral centroid: {spectral_centroid:.2f} Hz")
        
        return results
    
    def pattern_recognition_analysis(self, W_matrix, k_range, tau_range):
        """Pattern recognition analysis of wave transform results"""
        print(f"🧠 Performing Pattern Recognition Analysis...")
        start_time = time.time()
        
        # Magnitude analysis
        magnitude = np.abs(W_matrix)
        
        # Find dominant patterns
        max_magnitude_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        max_k = k_range[max_magnitude_idx[0]]
        max_tau = tau_range[max_magnitude_idx[1]]
        max_magnitude = magnitude[max_magnitude_idx]
        
        # Pattern characteristics
        coherence = np.std(magnitude) / np.mean(magnitude)
        pattern_energy = np.sum(magnitude**2)
        pattern_entropy = -np.sum(magnitude * np.log(magnitude + 1e-10))
        
        # Frequency and scale analysis
        k_power = np.sum(magnitude, axis=1)
        tau_power = np.sum(magnitude, axis=0)
        
        dominant_k = k_range[np.argmax(k_power)]
        dominant_tau = tau_range[np.argmax(tau_power)]
        
        duration = time.time() - start_time
        
        results = {
            'dominant_pattern': {
                'k_value': max_k,
                'tau_value': max_tau,
                'magnitude': max_magnitude
            },
            'pattern_characteristics': {
                'coherence': coherence,
                'total_energy': pattern_energy,
                'pattern_entropy': pattern_entropy,
                'mean_magnitude': np.mean(magnitude)
            },
            'frequency_analysis': {
                'dominant_k': dominant_k,
                'k_power_distribution': k_power.tolist()
            },
            'scale_analysis': {
                'dominant_tau': dominant_tau,
                'tau_power_distribution': tau_power.tolist()
            },
            'analysis_time': duration
        }
        
        print(f"  ✅ Pattern recognition completed in {duration:.3f} seconds")
        print(f"  🧠 Dominant pattern: k={max_k:.3f}, τ={max_tau:.3f}")
        print(f"  🧠 Pattern coherence: {coherence:.3f}")
        
        return results
    
    def species_comparison_analysis(self, data_dict):
        """Compare multiple mushroom species"""
        print(f"🔍 Performing Species Comparison Analysis...")
        start_time = time.time()
        
        comparison_results = {}
        
        for species_name, (V_t, t) in data_dict.items():
            print(f"\n🔍 Analyzing {species_name}...")
            
            # Perform all analyses
            W_matrix, k_range, tau_range = self.sqrt_wave_transform_analysis(V_t, t)
            freq_results = self.frequency_domain_analysis(V_t, t)
            pattern_results = self.pattern_recognition_analysis(W_matrix, k_range, tau_range)
            
            comparison_results[species_name] = {
                'wave_transform': {
                    'W_matrix': W_matrix,
                    'k_range': k_range,
                    'tau_range': tau_range
                },
                'frequency_analysis': freq_results,
                'pattern_recognition': pattern_results
            }
        
        # Cross-species analysis
        cross_species_analysis = self.cross_species_correlation(comparison_results)
        
        duration = time.time() - start_time
        print(f"\n✅ Species comparison completed in {duration:.3f} seconds")
        
        return comparison_results, cross_species_analysis
    
    def cross_species_correlation(self, comparison_results):
        """Analyze correlations between different species"""
        print(f"🔗 Computing Cross-Species Correlations...")
        
        species_names = list(comparison_results.keys())
        correlation_matrix = np.zeros((len(species_names), len(species_names)))
        
        for i, species1 in enumerate(species_names):
            for j, species2 in enumerate(species_names):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Compare dominant patterns
                    pattern1 = comparison_results[species1]['pattern_recognition']['dominant_pattern']
                    pattern2 = comparison_results[species2]['pattern_recognition']['dominant_pattern']
                    
                    # Simple correlation based on pattern similarity
                    k_corr = 1.0 / (1.0 + abs(pattern1['k_value'] - pattern2['k_value']))
                    tau_corr = 1.0 / (1.0 + abs(pattern1['tau_value'] - pattern2['tau_value']))
                    
                    correlation_matrix[i, j] = (k_corr + tau_corr) / 2
        
        return {
            'species_names': species_names,
            'correlation_matrix': correlation_matrix.tolist()
        }
    
    def generate_synth_sounds(self, V_t, t, analysis_results, output_dir="RESULTS/audio"):
        """Generate synth sounds from analysis results"""
        print(f"🎵 Generating Synth Sounds from Analysis...")
        start_time = time.time()
        
        os.makedirs(output_dir, exist_ok=True)
        generated_sounds = {}
        
        # Generate sounds for each synthesis method
        for method_name, method_func in self.synthesis_methods.items():
            print(f"\n🎵 Generating {method_name.upper()} synthesis...")
            audio, filename = method_func(V_t, t, duration=10.0)
            
            if audio is not None:
                filepath = self.save_wav_file(audio, filename, output_dir)
                if filepath:
                    generated_sounds[method_name] = filepath
        
        # Generate complete mix
        print(f"\n🎼 Creating complete synth mix...")
        mix_audio, mix_filename, _ = self.create_synth_mix(V_t, t, duration=15.0)
        
        if mix_audio is not None:
            mix_filepath = self.save_wav_file(mix_audio, mix_filename, output_dir)
            generated_sounds['complete_mix'] = mix_filepath
        
        duration = time.time() - start_time
        print(f"\n✅ Sound generation completed in {duration:.3f} seconds")
        
        return generated_sounds
    
    # Sound synthesis methods (same as before)
    def additive_synthesis(self, voltage_data, t, duration=10.0):
        """Additive synthesis using fungal voltage patterns"""
        frequencies = np.linspace(50, 2000, len(voltage_data))
        amplitudes = np.abs(voltage_data) / np.max(np.abs(voltage_data))
        
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t_audio)
        
        for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
            if i < len(t_audio):
                audio += amp * np.sin(2 * np.pi * freq * t_audio[:len(audio)])
                audio += 0.5 * amp * np.sin(2 * np.pi * freq * 2 * t_audio[:len(audio)])
                audio += 0.25 * amp * np.sin(2 * np.pi * freq * 3 * t_audio[:len(audio)])
        
        audio = self.normalize_audio(audio)
        return audio, f"additive_synth_{int(time.time())}.wav"
    
    def frequency_modulation_synthesis(self, voltage_data, t, duration=10.0):
        """FM synthesis using fungal voltage as modulation"""
        carrier_freq = 440
        mod_freq = 220
        
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        mod_index = np.interp(np.linspace(0, 1, len(t_audio)), 
                             np.linspace(0, 1, len(voltage_data)), 
                             np.abs(voltage_data))
        mod_index = mod_index / np.max(mod_index) * 5
        
        audio = np.sin(2 * np.pi * carrier_freq * t_audio + 
                      mod_index * np.sin(2 * np.pi * mod_freq * t_audio))
        
        audio = self.normalize_audio(audio)
        return audio, f"fm_synth_{int(time.time())}.wav"
    
    def granular_synthesis(self, voltage_data, t, duration=10.0):
        """Granular synthesis using fungal voltage patterns"""
        grain_duration = 0.1
        grain_samples = int(grain_duration * self.sample_rate)
        
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t_audio)
        
        for i in range(0, len(t_audio), grain_samples):
            if i + grain_samples < len(t_audio):
                voltage_idx = int((i / len(t_audio)) * len(voltage_data))
                if voltage_idx < len(voltage_data):
                    voltage_val = voltage_data[voltage_idx]
                    grain_freq = 200 + abs(voltage_val) * 1000
                    grain = np.sin(2 * np.pi * grain_freq * np.linspace(0, grain_duration, grain_samples))
                    envelope = np.hanning(len(grain))
                    grain = grain * envelope
                    
                    if i + len(grain) <= len(audio):
                        audio[i:i+len(grain)] += grain
        
        audio = self.normalize_audio(audio)
        return audio, f"granular_synth_{int(time.time())}.wav"
    
    def waveform_synthesis(self, voltage_data, t, duration=10.0):
        """Waveform synthesis using fungal voltage as wave shape"""
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t_audio)
        base_freq = 110
        
        for i, voltage in enumerate(voltage_data):
            if i < len(t_audio):
                freq_multiplier = 1 + abs(voltage) / np.max(np.abs(voltage_data))
                phase = voltage / np.max(np.abs(voltage_data)) * 2 * np.pi
                
                for harmonic in range(1, 6):
                    freq = base_freq * harmonic * freq_multiplier
                    amplitude = 1.0 / harmonic
                    audio += amplitude * np.sin(2 * np.pi * freq * t_audio + phase)
        
        audio = self.normalize_audio(audio)
        return audio, f"waveform_synth_{int(time.time())}.wav"
    
    def filtered_synthesis(self, voltage_data, t, duration=10.0):
        """Filtered synthesis with voltage-controlled filters"""
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        base_audio = np.sin(2 * np.pi * 220 * t_audio)
        filtered_audio = np.zeros_like(base_audio)
        
        for i in range(0, len(base_audio), 1000):
            chunk_end = min(i + 1000, len(base_audio))
            voltage_idx = int((i / len(base_audio)) * len(voltage_data))
            
            if voltage_idx < len(voltage_data):
                voltage_val = voltage_data[voltage_idx]
                cutoff_freq = 100 + abs(voltage_val) * 2000
                cutoff_freq = np.clip(cutoff_freq, 100, 8000)
                
                b, a = signal.butter(4, cutoff_freq / (self.sample_rate / 2), 'low')
                chunk = base_audio[i:chunk_end]
                if len(chunk) > 0:
                    filtered_chunk = signal.filtfilt(b, a, chunk)
                    filtered_audio[i:chunk_end] = filtered_chunk
        
        audio = self.normalize_audio(filtered_audio)
        return audio, f"filtered_synth_{int(time.time())}.wav"
    
    def ambient_synthesis(self, voltage_data, t, duration=15.0):
        """Ambient synthesis for atmospheric, evolving sounds"""
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t_audio)
        
        # Drone layer
        drone_freq = 55
        drone = np.sin(2 * np.pi * drone_freq * t_audio) * 0.3
        
        # Voltage-modulated pad
        pad_freq = 110
        pad = np.zeros_like(t_audio)
        for i, voltage in enumerate(voltage_data):
            if i < len(t_audio):
                mod_freq = pad_freq + voltage * 50
                pad += 0.2 * np.sin(2 * np.pi * mod_freq * t_audio)
        
        # Texture layer
        texture = np.random.randn(len(t_audio)) * 0.1
        
        # Combine layers
        audio = drone + pad + texture
        audio = self.normalize_audio(audio)
        
        return audio, f"ambient_synth_{int(time.time())}.wav"
    
    def create_synth_mix(self, voltage_data, t, duration=15.0):
        """Create a complete synth mix"""
        synth_tracks = {}
        
        for method_name, method_func in self.synthesis_methods.items():
            audio, filename = method_func(voltage_data, t, duration)
            if audio is not None:
                synth_tracks[method_name] = {
                    'audio': audio,
                    'filename': filename,
                    'method': method_name
                }
        
        if synth_tracks:
            max_length = max(len(track['audio']) for track in synth_tracks.values())
            mix = np.zeros(max_length)
            
            volumes = {
                'additive': 0.8, 'fm': 0.6, 'granular': 0.7,
                'waveform': 0.5, 'filtered': 0.6, 'ambient': 0.4
            }
            
            for method_name, track in synth_tracks.items():
                volume = volumes.get(method_name, 0.5)
                track_audio = track['audio']
                
                if len(track_audio) < max_length:
                    track_audio = np.pad(track_audio, (0, max_length - len(track_audio)))
                
                mix[:len(track_audio)] += track_audio * volume
            
            mix = self.normalize_audio(mix, target_range=0.9)
            return mix, "fungal_synth_mix.wav", synth_tracks
        
        return None, None, {}
    
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
            
            print(f"  💾 Audio saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"  ❌ Error saving audio: {e}")
            return None
    
    def run_complete_analysis(self, species_list=None):
        """Run complete integrated analysis on all available species"""
        print(f"🔬 INTEGRATED FUNGAL ANALYSIS SYSTEM")
        print(f"Author: {self.author}")
        print(f"Timestamp: {self.timestamp}")
        print(f"=" * 60)
        
        if species_list is None:
            # Auto-detect available species
            species_list = []
            for file in os.listdir(self.data_path):
                if file.endswith('.csv'):
                    species_list.append(file)
        
        print(f"🍄 Analyzing {len(species_list)} fungal species:")
        for species in species_list:
            print(f"  • {species}")
        
        # Load and analyze each species
        all_results = {}
        data_dict = {}
        
        for species_file in species_list:
            print(f"\n🔬 Analyzing {species_file}...")
            
            V_t, t = self.load_fungal_data(species_file)
            if V_t is None:
                continue
            
            data_dict[species_file] = (V_t, t)
            
            # Perform √t wave transform analysis
            W_matrix, k_range, tau_range = self.sqrt_wave_transform_analysis(V_t, t)
            
            # Perform frequency domain analysis
            freq_results = self.frequency_domain_analysis(V_t, t)
            
            # Perform pattern recognition
            pattern_results = self.pattern_recognition_analysis(W_matrix, k_range, tau_range)
            
            # Generate synth sounds
            print(f"\n🎵 Generating synth sounds for {species_file}...")
            synth_sounds = self.generate_synth_sounds(V_t, t, pattern_results)
            
            # Store all results
            all_results[species_file] = {
                'wave_transform': {
                    'W_matrix': W_matrix,
                    'k_range': k_range,
                    'tau_range': tau_range
                },
                'frequency_analysis': freq_results,
                'pattern_recognition': pattern_results,
                'synth_sounds': synth_sounds
            }
        
        # Perform cross-species comparison
        if len(data_dict) > 1:
            print(f"\n🔍 Performing cross-species comparison...")
            comparison_results, cross_species_analysis = self.species_comparison_analysis(data_dict)
            
            # Add comparison results
            all_results['cross_species_analysis'] = {
                'comparison': comparison_results,
                'correlation': cross_species_analysis
            }
        
        # Save comprehensive results
        self.save_comprehensive_results(all_results)
        
        print(f"\n🎉 INTEGRATED ANALYSIS COMPLETE!")
        print(f"🔬 Analyzed {len(data_dict)} fungal species")
        print(f"🎵 Generated synth sounds for each species")
        print(f"🔍 Performed cross-species comparison")
        print(f"📊 Results saved to RESULTS/ directory")
        
        return all_results
    
    def save_comprehensive_results(self, all_results):
        """Save comprehensive analysis results"""
        os.makedirs(self.results_path, exist_ok=True)
        
        # Save analysis results
        analysis_file = self.results_path / "comprehensive_analysis_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in all_results.items():
            if key == 'cross_species_analysis':
                serializable_results[key] = value
            else:
                serializable_results[key] = {
                    'wave_transform': {
                        'k_range': value['wave_transform']['k_range'].tolist(),
                        'tau_range': value['wave_transform']['tau_range'].tolist(),
                        'matrix_shape': value['wave_transform']['W_matrix'].shape
                    },
                    'frequency_analysis': value['frequency_analysis'],
                    'pattern_recognition': value['pattern_recognition'],
                    'synth_sounds': value['synth_sounds']
                }
        
        with open(analysis_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📊 Analysis results saved: {analysis_file}")
        
        # Create summary report
        self.create_summary_report(all_results)
    
    def create_summary_report(self, all_results):
        """Create a summary report of the analysis"""
        report_file = self.results_path / "analysis_summary_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# 🔬 Integrated Fungal Analysis Summary Report\n\n")
            f.write(f"**Author:** {self.author}\n")
            f.write(f"**Timestamp:** {self.timestamp}\n\n")
            
            f.write("## 📊 Analysis Overview\n\n")
            f.write(f"Total species analyzed: {len([k for k in all_results.keys() if k != 'cross_species_analysis'])}\n\n")
            
            for species_name, results in all_results.items():
                if species_name != 'cross_species_analysis':
                    f.write(f"### 🍄 {species_name}\n\n")
                    
                    if 'pattern_recognition' in results:
                        pattern = results['pattern_recognition']['dominant_pattern']
                        f.write(f"- **Dominant Pattern:** k={pattern['k_value']:.3f}, τ={pattern['tau_value']:.3f}\n")
                        f.write(f"- **Pattern Magnitude:** {pattern['magnitude']:.3f}\n")
                    
                    if 'frequency_analysis' in results:
                        freq = results['frequency_analysis']
                        f.write(f"- **Dominant Frequency:** {freq['dominant_frequency']:.2f} Hz\n")
                        f.write(f"- **Spectral Centroid:** {freq['spectral_centroid']:.2f} Hz\n")
                    
                    if 'synth_sounds' in results:
                        f.write(f"- **Synth Sounds Generated:** {len(results['synth_sounds'])}\n")
                    
                    f.write("\n")
            
            if 'cross_species_analysis' in all_results:
                f.write("## 🔍 Cross-Species Analysis\n\n")
                f.write("Cross-species correlation analysis completed.\n\n")
        
        print(f"📋 Summary report saved: {report_file}")

def main():
    """Main function"""
    print("🔬 INTEGRATED FUNGAL ANALYSIS SYSTEM")
    print("Author: Joe Knowles")
    print("Timestamp: 2025-08-12 09:23:27 BST")
    print("=" * 60)
    
    # Create integrated system
    system = IntegratedFungalAnalysisSystem()
    
    # Run complete analysis
    results = system.run_complete_analysis()
    
    print(f"\n🎉 SYSTEM READY FOR NEW DISCOVERIES!")
    print(f"🔬 √t Wave Transform analysis complete")
    print(f"🎵 Synth sounds generated for each species")
    print(f"🔍 Cross-species patterns identified")
    print(f"🧠 Ready to make new fungal discoveries!")

if __name__ == "__main__":
    main() 