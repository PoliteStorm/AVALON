#!/usr/bin/env python3
"""
Enhanced Fungal Analysis System with Detailed Labeling and Speed Optimization
Author: Joe Knowles
Timestamp: 2025-08-12 09:23:27 BST
Description: High-speed integrated system with comprehensive mushroom labeling and timing
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

class EnhancedFungalAnalysisSystem:
    """
    Enhanced system with detailed labeling, timing, and speed optimization
    """
    
    def __init__(self):
        self.timestamp = "20250812_092327"
        self.author = "Joe Knowles"
        self.sample_rate = 44100
        self.data_path = Path("DATA/raw")
        self.results_path = Path("RESULTS")
        
        # Performance tracking
        self.performance_metrics = {}
        self.analysis_timings = {}
        
        # Mushroom species database
        self.mushroom_database = {
            'Spray_in_bag.csv': {
                'species': 'Pleurotus ostreatus (Oyster)',
                'strain': 'Spray-treated variant',
                'treatment': 'Water spray stimulation',
                'electrode_type': 'Surface electrodes',
                'measurement_duration': '229 seconds',
                'environmental_conditions': 'Controlled humidity',
                'data_quality': 'High resolution electrical spikes'
            },
            'New_Oyster_with spray.csv': {
                'species': 'Pleurotus ostreatus (Oyster)',
                'strain': 'New culture variant',
                'treatment': 'Water spray stimulation',
                'electrode_type': 'Differential electrodes',
                'measurement_duration': 'Variable',
                'environmental_conditions': 'Humidity controlled',
                'data_quality': 'High sensitivity measurements'
            },
            'Blue_oyster_31_5_22.csv': {
                'species': 'Pleurotus ostreatus (Blue Oyster)',
                'strain': 'Blue variant',
                'treatment': 'Natural growth conditions',
                'electrode_type': 'Substrate electrodes',
                'measurement_duration': 'Extended monitoring',
                'environmental_conditions': 'Natural substrate',
                'data_quality': 'Long-term electrical patterns'
            },
            'Hericium_20_4_22.csv': {
                'species': 'Hericium erinaceus (Lion\'s Mane)',
                'strain': 'Wild type',
                'treatment': 'Natural growth',
                'electrode_type': 'Substrate monitoring',
                'measurement_duration': 'Extended period',
                'environmental_conditions': 'Natural conditions',
                'data_quality': 'Complex neural-like patterns'
            }
        }
    
    def get_mushroom_info(self, filename):
        """Get detailed mushroom information"""
        base_name = os.path.basename(filename)
        if base_name in self.mushroom_database:
            return self.mushroom_database[base_name]
        else:
            return {
                'species': 'Unknown species',
                'strain': 'Unknown strain',
                'treatment': 'Unknown treatment',
                'electrode_type': 'Unknown electrodes',
                'measurement_duration': 'Unknown duration',
                'environmental_conditions': 'Unknown conditions',
                'data_quality': 'Standard measurements'
            }
    
    def timed_operation(self, operation_name, operation_func, *args, **kwargs):
        """Execute operation with timing"""
        start_time = time.time()
        start_cpu = time.process_time()
        
        print(f"⏱️  Starting {operation_name}...")
        result = operation_func(*args, **kwargs)
        
        end_time = time.time()
        end_cpu = time.process_time()
        
        wall_time = end_time - start_time
        cpu_time = end_cpu - start_cpu
        
        self.analysis_timings[operation_name] = {
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ {operation_name} completed in {wall_time:.3f}s (CPU: {cpu_time:.3f}s)")
        
        return result
    
    def load_fungal_data_optimized(self, filepath):
        """Optimized data loading with detailed labeling"""
        mushroom_info = self.get_mushroom_info(filepath)
        
        print(f"🍄 Loading: {os.path.basename(filepath)}")
        print(f"  Species: {mushroom_info['species']}")
        print(f"  Strain: {mushroom_info['strain']}")
        print(f"  Treatment: {mushroom_info['treatment']}")
        print(f"  Electrodes: {mushroom_info['electrode_type']}")
        
        def load_operation():
            try:
                data = []
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                
                # Optimized header detection
                header_count = 0
                for line in lines:
                    if line.strip() and not line.startswith('"'):
                        header_count += 1
                    if header_count > 2:
                        break
                
                # Vectorized data extraction
                data_lines = lines[header_count:]
                data = []
                
                for line in data_lines:
                    if line.strip():
                        parts = line.strip().split(',')
                        if len(parts) > 1:
                            try:
                                value = float(parts[1].strip('"'))
                                data.append(value)
                            except (ValueError, IndexError):
                                continue
                
                if len(data) < 10:
                    return None, None, None
                
                t = np.linspace(0, len(data), len(data))
                V_t = np.array(data)
                
                print(f"  ✅ Loaded {len(data)} electrical measurements")
                print(f"  📊 Voltage range: {np.min(V_t):.3f} to {np.max(V_t):.3f} mV")
                print(f"  ⏱️  Duration: {len(data)} seconds")
                print(f"  🔬 Data quality: {mushroom_info['data_quality']}")
                
                return V_t, t, mushroom_info
                
            except Exception as e:
                print(f"❌ Error loading data: {e}")
                return None, None, None
        
        return self.timed_operation("Data Loading", load_operation)
    
    def sqrt_wave_transform_optimized(self, V_t, t, k_range=None, tau_range=None):
        """Optimized √t wave transform with vectorization"""
        if k_range is None:
            k_range = np.linspace(0.1, 5.0, 20)
        if tau_range is None:
            signal_duration = t[-1] - t[0]
            tau_range = np.logspace(-1, np.log10(signal_duration), 15)
        
        print(f"🔬 Computing √t Wave Transform (Optimized)")
        print(f"  Equation: W(k,τ) = ∫₀^T V(t) · ψ(√t/τ) · e^(-ik√t) dt")
        print(f"  k range: {k_range[0]:.3f} to {k_range[-1]:.3f}")
        print(f"  τ range: {tau_range[0]:.3f} to {tau_range[-1]:.3f}")
        
        def transform_operation():
            # Pre-compute integration points
            max_t = t[-1]
            t_integration = np.logspace(-3, np.log10(max_t), 300)
            
            # Pre-compute voltage interpolation
            V_interp = np.interp(t_integration, t, V_t)
            
            # Initialize result matrix
            W_matrix = np.zeros((len(k_range), len(tau_range)), dtype=complex)
            
            # Vectorized computation
            for i, k in enumerate(k_range):
                for j, tau in enumerate(tau_range):
                    # Optimized integrand computation
                    sqrt_t = np.sqrt(t_integration)
                    normalized_t = sqrt_t / np.sqrt(tau)
                    
                    # Vectorized wavelet computation
                    gaussian_mask = np.abs(normalized_t) <= 5.0
                    psi_vals = np.zeros_like(t_integration, dtype=complex)
                    
                    if np.any(gaussian_mask):
                        gaussian = np.exp(-normalized_t[gaussian_mask]**2 / 2)
                        complex_exp = np.exp(1j * 2.0 * normalized_t[gaussian_mask])
                        psi_vals[gaussian_mask] = gaussian * complex_exp / np.sqrt(2 * np.pi)
                    
                    # Vectorized exponential computation
                    exp_vals = np.exp(-1j * k * sqrt_t)
                    
                    # Complete integrand
                    integrand = V_interp * psi_vals * exp_vals
                    
                    # Efficient integration
                    dt = np.diff(t_integration)
                    integral = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dt)
                    
                    W_matrix[i, j] = integral
            
            return W_matrix, k_range, tau_range
        
        return self.timed_operation("√t Wave Transform", transform_operation)
    
    def pattern_recognition_optimized(self, W_matrix, k_range, tau_range):
        """Optimized pattern recognition with vectorized operations"""
        print(f"🧠 Performing Pattern Recognition Analysis (Optimized)")
        
        def recognition_operation():
            # Vectorized magnitude computation
            magnitude = np.abs(W_matrix)
            
            # Vectorized pattern detection
            max_magnitude_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
            max_k = k_range[max_magnitude_idx[0]]
            max_tau = tau_range[max_magnitude_idx[1]]
            max_magnitude = magnitude[max_magnitude_idx]
            
            # Vectorized power analysis
            k_power = np.sum(magnitude, axis=1)
            tau_power = np.sum(magnitude, axis=0)
            
            dominant_k = k_range[np.argmax(k_power)]
            dominant_tau = tau_range[np.argmax(tau_power)]
            
            # Vectorized statistics
            coherence = np.std(magnitude) / np.mean(magnitude)
            pattern_energy = np.sum(magnitude**2)
            pattern_entropy = -np.sum(magnitude * np.log(magnitude + 1e-10))
            
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
                }
            }
            
            print(f"  🧠 Dominant pattern: k={max_k:.3f}, τ={max_tau:.3f}")
            print(f"  🧠 Pattern coherence: {coherence:.3f}")
            print(f"  🧠 Pattern energy: {pattern_energy:.3f}")
            
            return results
        
        return self.timed_operation("Pattern Recognition", recognition_operation)
    
    def generate_synth_sounds_optimized(self, V_t, t, mushroom_info, output_dir="RESULTS/audio"):
        """Optimized synth sound generation with detailed labeling"""
        print(f"🎵 Generating Synth Sounds (Optimized)")
        print(f"  🍄 Species: {mushroom_info['species']}")
        print(f"  🧬 Strain: {mushroom_info['strain']}")
        
        os.makedirs(output_dir, exist_ok=True)
        generated_sounds = {}
        
        def sound_generation_operation():
            # Generate additive synthesis
            print(f"    🎹 Generating additive synthesis...")
            audio, filename = self.additive_synthesis_optimized(V_t, t, duration=8.0)
            if audio is not None:
                filepath = self.save_wav_file_labeled(audio, filename, output_dir, mushroom_info)
                if filepath:
                    generated_sounds['additive'] = filepath
            
            # Generate FM synthesis
            print(f"    🔔 Generating FM synthesis...")
            audio, filename = self.frequency_modulation_synthesis_optimized(V_t, t, duration=8.0)
            if audio is not None:
                filepath = self.save_wav_file_labeled(audio, filename, output_dir, mushroom_info)
                if filepath:
                    generated_sounds['fm'] = filepath
            
            # Generate granular synthesis
            print(f"    🌊 Generating granular synthesis...")
            audio, filename = self.granular_synthesis_optimized(V_t, t, duration=8.0)
            if audio is not None:
                filepath = self.save_wav_file_labeled(audio, filename, output_dir, mushroom_info)
                if filepath:
                    generated_sounds['granular'] = filepath
            
            return generated_sounds
        
        return self.timed_operation("Sound Generation", sound_generation_operation)
    
    def additive_synthesis_optimized(self, voltage_data, t, duration=8.0):
        """Optimized additive synthesis"""
        frequencies = np.linspace(50, 1500, len(voltage_data))
        amplitudes = np.abs(voltage_data) / np.max(np.abs(voltage_data))
        
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t_audio)
        
        # Vectorized synthesis
        for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
            if i < len(t_audio):
                # Vectorized harmonic generation
                harmonics = [1.0, 0.5, 0.25]
                for j, harmonic_amp in enumerate(harmonics):
                    if i + j < len(t_audio):
                        audio += harmonic_amp * amp * np.sin(2 * np.pi * freq * (j + 1) * t_audio[:len(audio)])
        
        audio = self.normalize_audio(audio)
        return audio, f"additive_synth_{int(time.time())}.wav"
    
    def frequency_modulation_synthesis_optimized(self, voltage_data, t, duration=8.0):
        """Optimized FM synthesis"""
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
        
        audio = self.normalize_audio(audio)
        return audio, f"fm_synth_{int(time.time())}.wav"
    
    def granular_synthesis_optimized(self, voltage_data, t, duration=8.0):
        """Optimized granular synthesis"""
        grain_duration = 0.1
        grain_samples = int(grain_duration * self.sample_rate)
        
        t_audio = np.linspace(0, duration, int(self.sample_rate * duration))
        audio = np.zeros_like(t_audio)
        
        # Pre-compute envelope
        envelope = np.hanning(grain_samples)
        
        # Vectorized grain generation
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
        
        audio = self.normalize_audio(audio)
        return audio, f"granular_synth_{int(time.time())}.wav"
    
    def normalize_audio(self, data, target_range=0.8):
        """Optimized audio normalization"""
        if data is None or len(data) == 0:
            return None
        
        data = np.array(data)
        data = data - np.mean(data)
        
        if np.max(np.abs(data)) > 0:
            data = data * (target_range / np.max(np.abs(data)))
        
        return data
    
    def save_wav_file_labeled(self, audio, filename, output_dir, mushroom_info):
        """Save labeled WAV file with mushroom metadata"""
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
            
            # Create metadata file
            metadata_file = filepath.replace('.wav', '_metadata.json')
            metadata = {
                'audio_file': filename,
                'mushroom_info': mushroom_info,
                'generation_timestamp': self.timestamp,
                'author': self.author,
                'sample_rate': self.sample_rate,
                'duration_seconds': len(audio) / self.sample_rate,
                'processing_notes': 'Generated from real fungal electrical data'
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"      💾 Audio saved: {filepath}")
            print(f"      📋 Metadata saved: {metadata_file}")
            return filepath
            
        except Exception as e:
            print(f"      ❌ Error saving audio: {e}")
            return None
    
    def run_enhanced_analysis(self, species_list=None):
        """Run enhanced analysis with detailed labeling and timing"""
        print(f"🔬 ENHANCED FUNGAL ANALYSIS SYSTEM")
        print(f"Author: {self.author}")
        print(f"Timestamp: {self.timestamp}")
        print(f"=" * 60)
        
        if species_list is None:
            # Auto-detect available species
            species_list = []
            for file in os.listdir(self.data_path / "15061491"):
                if file.endswith('.csv'):
                    species_list.append(f"DATA/raw/15061491/{file}")
        
        print(f"🍄 Analyzing {len(species_list)} fungal species with detailed labeling")
        
        all_results = {}
        total_start_time = time.time()
        
        for filepath in species_list:
            if os.path.exists(filepath):
                filename = os.path.basename(filepath)
                print(f"\n🔬 Analyzing: {filename}")
                print(f"=" * 60)
                
                # Load data with timing
                V_t, t, mushroom_info = self.load_fungal_data_optimized(filepath)
                if V_t is None:
                    continue
                
                # Perform √t wave transform analysis with timing
                W_matrix, k_range, tau_range = self.sqrt_wave_transform_optimized(V_t, t)
                
                # Perform pattern recognition with timing
                pattern_results = self.pattern_recognition_optimized(W_matrix, k_range, tau_range)
                
                # Generate synth sounds with timing
                synth_sounds = self.generate_synth_sounds_optimized(V_t, t, mushroom_info)
                
                # Store all results with detailed labeling
                all_results[filename] = {
                    'mushroom_info': mushroom_info,
                    'wave_transform': {
                        'W_matrix': W_matrix,
                        'k_range': k_range,
                        'tau_range': tau_range
                    },
                    'pattern_recognition': pattern_results,
                    'synth_sounds': synth_sounds,
                    'analysis_timings': self.analysis_timings.copy()
                }
                
                print(f"✅ Enhanced analysis complete for {filename}")
        
        total_duration = time.time() - total_start_time
        
        # Save comprehensive results with detailed labeling
        self.save_enhanced_results(all_results, total_duration)
        
        print(f"\n🎉 ENHANCED ANALYSIS COMPLETE!")
        print(f"🔬 Analyzed {len(all_results)} fungal species")
        print(f"🎵 Generated labeled synth sounds for each")
        print(f"⏱️  Total analysis time: {total_duration:.3f} seconds")
        print(f"📊 Results saved with detailed mushroom labeling")
        
        return all_results
    
    def save_enhanced_results(self, all_results, total_duration):
        """Save enhanced results with comprehensive labeling"""
        os.makedirs(self.results_path, exist_ok=True)
        
        # Save detailed analysis results
        results_file = self.results_path / "enhanced_analysis_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for filename, results in all_results.items():
            serializable_results[filename] = {
                'mushroom_info': results['mushroom_info'],
                'wave_transform': {
                    'k_range': results['wave_transform']['k_range'].tolist(),
                    'tau_range': results['wave_transform']['tau_range'].tolist(),
                    'matrix_shape': results['wave_transform']['W_matrix'].shape
                },
                'pattern_recognition': results['pattern_recognition'],
                'synth_sounds': results['synth_sounds'],
                'analysis_timings': results['analysis_timings']
            }
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📊 Enhanced results saved: {results_file}")
        
        # Create detailed summary report
        self.create_enhanced_summary_report(all_results, total_duration)
    
    def create_enhanced_summary_report(self, all_results, total_duration):
        """Create comprehensive summary report with detailed labeling"""
        report_file = self.results_path / "enhanced_analysis_summary_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# 🔬 Enhanced Fungal Analysis Summary Report\n\n")
            f.write(f"**Author:** {self.author}\n")
            f.write(f"**Timestamp:** {self.timestamp}\n")
            f.write(f"**Total Analysis Time:** {total_duration:.3f} seconds\n\n")
            
            f.write("## 📊 Analysis Overview\n\n")
            f.write(f"**Total species analyzed:** {len(all_results)}\n")
            f.write(f"**Analysis completion time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for filename, results in all_results.items():
                mushroom_info = results['mushroom_info']
                pattern = results['pattern_recognition']['dominant_pattern']
                timings = results['analysis_timings']
                
                f.write(f"### 🍄 {filename}\n\n")
                f.write(f"**Species Information:**\n")
                f.write(f"- **Species:** {mushroom_info['species']}\n")
                f.write(f"- **Strain:** {mushroom_info['strain']}\n")
                f.write(f"- **Treatment:** {mushroom_info['treatment']}\n")
                f.write(f"- **Electrode Type:** {mushroom_info['electrode_type']}\n")
                f.write(f"- **Environmental Conditions:** {mushroom_info['environmental_conditions']}\n")
                f.write(f"- **Data Quality:** {mushroom_info['data_quality']}\n\n")
                
                f.write(f"**Analysis Results:**\n")
                f.write(f"- **Dominant Pattern:** k={pattern['k_value']:.3f}, τ={pattern['tau_value']:.3f}\n")
                f.write(f"- **Pattern Magnitude:** {pattern['magnitude']:.3f}\n")
                f.write(f"- **Synth Sounds Generated:** {len(results['synth_sounds'])}\n\n")
                
                f.write(f"**Processing Times:**\n")
                for operation, timing in timings.items():
                    f.write(f"- **{operation}:** {timing['wall_time']:.3f}s (CPU: {timing['cpu_time']:.3f}s)\n")
                f.write("\n")
            
            f.write("## ⚡ Performance Summary\n\n")
            f.write(f"**Total Wall Time:** {total_duration:.3f} seconds\n")
            f.write(f"**Average Time per Species:** {total_duration/len(all_results):.3f} seconds\n")
            f.write(f"**Species per Second:** {len(all_results)/total_duration:.2f}\n\n")
            
            f.write("## 🎵 Sound Generation Summary\n\n")
            total_sounds = sum(len(results['synth_sounds']) for results in all_results.values())
            f.write(f"**Total Synth Sounds Generated:** {total_sounds}\n")
            f.write(f"**Sounds per Species:** {total_sounds/len(all_results):.1f}\n\n")
        
        print(f"📋 Enhanced summary report saved: {report_file}")

def main():
    """Main function"""
    print("🔬 ENHANCED FUNGAL ANALYSIS SYSTEM")
    print("Author: Joe Knowles")
    print("Timestamp: 2025-08-12 09:23:27 BST")
    print("=" * 60)
    
    # Create enhanced system
    system = EnhancedFungalAnalysisSystem()
    
    # Run enhanced analysis
    results = system.run_enhanced_analysis()
    
    print(f"\n🎉 SYSTEM READY FOR NEW DISCOVERIES!")
    print(f"🔬 √t Wave Transform analysis complete with detailed labeling")
    print(f"🎵 Labeled synth sounds generated for each species")
    print(f"⏱️  All processing times tracked and optimized")
    print(f"🧠 Ready to make new fungal discoveries!")

if __name__ == "__main__":
    main() 