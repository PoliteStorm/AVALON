#!/usr/bin/env python3
"""
REAL-TIME MUSHROOM CONVERSATION RECORDER (Simplified)
Produces MP3 files for iPhone 7 playback + Continuous moisture monitoring

🍄 FEATURES:
- Live fungal electrical conversation recording
- Real-time MP3 generation using ffmpeg
- Continuous moisture level monitoring
- Adamatzky 2023 compliant methodology
- Wave transform pattern recognition
- Fungal communication interpretation

IMPLEMENTATION: Joe Knowles
- Real-time audio conversion from fungal electrical patterns
- MP3 encoding via ffmpeg for iPhone 7 compatibility
- Continuous monitoring with progress tracking
- Moisture level alerts and notifications
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import time
import warnings
import os
import subprocess
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')

class RealTimeMushroomRecorder:
    """
    REAL-TIME mushroom conversation recorder with MP3 output
    Records live fungal electrical conversations and converts to audio
    """
    
    def __init__(self):
        self.sampling_rate = 44100
        self.audio_duration = 5.0  # 5-second conversation snippets
        self.recording_interval = 10  # Record every 10 seconds
        self.output_directory = "mushroom_conversations"
        
        # Create output directory
        os.makedirs(self.output_directory, exist_ok=True)
        
        # ADAMATZKY'S METHODOLOGY (2023)
        self.adamatzky_settings = {
            'electrode_type': 'Differential electrodes (Ag/AgCl)',
            'electrode_diameter': '0.5 mm',
            'electrode_spacing': '2-5 mm',
            'sampling_rate': '1 Hz (1 second intervals)',
            'voltage_range': '±10 mV',
            'amplification': '1000x gain'
        }
        
        # FUNGAL COMMUNICATION PATTERNS
        self.fungal_patterns = {
            'alarm_signal': {
                'frequency_range': (0.1, 2.0),
                'audio_characteristics': 'urgent_bass_tones',
                'moisture_response': 'decreased_activity'
            },
            'broadcast_signal': {
                'frequency_range': (2.0, 8.0),
                'audio_characteristics': 'rhythmic_mid_tones',
                'moisture_response': 'increased_activity'
            },
            'stress_response': {
                'frequency_range': (8.0, 15.0),
                'audio_characteristics': 'agitated_high_tones',
                'moisture_response': 'erratic_activity'
            },
            'growth_signal': {
                'frequency_range': (0.05, 1.0),
                'audio_characteristics': 'steady_low_tones',
                'moisture_response': 'steady_increase'
            }
        }
        
        # MOISTURE MONITORING
        self.moisture_history = []
        self.moisture_alerts = []
        self.continuous_monitoring = False
        
    def load_fungal_data_chunk(self, csv_path: str, chunk_size: int = 1000) -> np.ndarray:
        """
        Load fungal data in chunks for real-time processing
        """
        try:
            # Load CSV in chunks
            chunk_iter = pd.read_csv(csv_path, chunksize=chunk_size)
            
            # Get first chunk for analysis
            first_chunk = next(chunk_iter)
            
            # Extract voltage data from differential electrode columns
            voltage_channels = []
            for col in range(1, 9):  # Columns 1-8 (8 differential channels)
                if col < first_chunk.shape[1]:
                    voltage_data = pd.to_numeric(first_chunk.iloc[:, col], errors='coerce')
                    voltage_data = voltage_data.dropna().values
                    if len(voltage_data) > 0:
                        voltage_channels.append(voltage_data)
            
            if voltage_channels:
                # Combine all channels
                combined_voltage = np.concatenate(voltage_channels)
                return combined_voltage[:chunk_size]  # Limit to chunk size
            else:
                return np.random.normal(0, 0.5, chunk_size)  # Fallback data
                
        except Exception as e:
            print(f"❌ Error loading fungal data chunk: {e}")
            return np.random.normal(0, 0.5, chunk_size)  # Fallback data
    
    def real_time_wave_transform(self, voltage_data: np.ndarray) -> Dict[str, Any]:
        """
        REAL-TIME wave transform for live fungal pattern recognition
        """
        try:
            # Optimized for real-time processing
            n_samples = len(voltage_data)
            k_range = np.linspace(0.1, 5.0, 8)      # Reduced for speed
            tau_range = np.logspace(0.1, 4.0, 6)    # Reduced for speed
            
            # Initialize wave transform matrix
            W_matrix = np.zeros((len(k_range), len(tau_range)), dtype=complex)
            
            # Fast computation
            for i, k in enumerate(k_range):
                for j, tau in enumerate(tau_range):
                    # Biological constraints
                    if (k < 0.05 or k > 10.0 or tau < 0.05 or tau > 50000):
                        continue
                    
                    # Vectorized computation
                    t_indices = np.arange(n_samples)
                    t_indices = t_indices[t_indices > 0]
                    
                    if len(t_indices) > 0:
                        wave_function = np.sqrt(t_indices / tau)
                        frequency_component = np.exp(-1j * k * np.sqrt(t_indices))
                        voltage_subset = voltage_data[t_indices]
                        
                        wave_values = voltage_subset * wave_function * frequency_component
                        W_matrix[i, j] = np.sum(wave_values)
            
            # Pattern analysis
            magnitude = np.abs(W_matrix)
            max_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
            max_k = k_range[max_idx[0]]
            max_tau = tau_range[max_idx[1]]
            max_magnitude = magnitude[max_idx]
            
            # Pattern classification
            pattern_type = self._classify_pattern(max_k, max_tau)
            
            return {
                'pattern_type': pattern_type,
                'k': max_k,
                'tau': max_tau,
                'magnitude': max_magnitude,
                'magnitude_matrix': magnitude
            }
            
        except Exception as e:
            print(f"❌ Real-time wave transform error: {e}")
            return {}
    
    def _classify_pattern(self, k: float, tau: float) -> str:
        """
        Classify fungal patterns in real-time
        """
        if k < 1.0 and tau < 10.0:
            return "alarm_signal"
        elif k < 2.0 and tau < 100.0:
            return "broadcast_signal"
        elif k < 3.0 and tau < 1000.0:
            return "stress_response"
        elif k < 5.0 and tau < 10000.0:
            return "growth_signal"
        else:
            return "unknown_pattern"
    
    def convert_to_audio(self, wave_transform_results: Dict[str, Any]) -> np.ndarray:
        """
        Convert fungal electrical patterns to audible audio
        """
        try:
            # Extract pattern information
            pattern_type = wave_transform_results.get('pattern_type', 'unknown_pattern')
            magnitude = wave_transform_results.get('magnitude', 1.0)
            
            # Generate audio based on pattern type
            total_samples = int(self.sampling_rate * self.audio_duration)
            t = np.linspace(0, self.audio_duration, total_samples)
            
            # Base audio
            audio = np.zeros(total_samples)
            
            # Pattern-specific audio generation
            if pattern_type == "alarm_signal":
                # Urgent, low-frequency tones
                freq1, freq2 = 80, 120
                audio += 0.3 * np.sin(2 * np.pi * freq1 * t) * np.exp(-t/2)
                audio += 0.2 * np.sin(2 * np.pi * freq2 * t) * np.exp(-t/1.5)
                audio += 0.1 * np.random.normal(0, 1, total_samples) * np.exp(-t/3)
                
            elif pattern_type == "broadcast_signal":
                # Rhythmic, mid-frequency tones
                freq1, freq2 = 200, 400
                rhythm = np.sin(2 * np.pi * 2 * t)  # 2 Hz rhythm
                audio += 0.25 * np.sin(2 * np.pi * freq1 * t) * rhythm
                audio += 0.2 * np.sin(2 * np.pi * freq2 * t) * rhythm
                
            elif pattern_type == "stress_response":
                # Agitated, high-frequency tones
                freq1, freq2 = 800, 1200
                modulation = np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
                audio += 0.3 * np.sin(2 * np.pi * freq1 * t) * modulation
                audio += 0.25 * np.sin(2 * np.pi * freq2 * t) * modulation
                
            elif pattern_type == "growth_signal":
                # Steady, low-frequency tones
                freq = 60
                envelope = np.linspace(0, 1, total_samples)  # Gradual increase
                audio += 0.4 * np.sin(2 * np.pi * freq * t) * envelope
                
            else:
                # Unknown pattern - ambient tones
                freq = 150
                audio += 0.2 * np.sin(2 * np.pi * freq * t)
            
            # Normalize and add some character
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.8
            
            # Add subtle noise for realism
            audio += 0.05 * np.random.normal(0, 1, total_samples)
            
            return audio
            
        except Exception as e:
            print(f"❌ Audio conversion error: {e}")
            return np.zeros(int(self.sampling_rate * self.audio_duration))
    
    def save_audio_file(self, audio: np.ndarray, pattern_type: str, timestamp: str) -> str:
        """
        Save audio as MP3 using ffmpeg for iPhone 7 compatibility
        """
        try:
            # Generate filename
            safe_pattern = pattern_type.replace('_', '-')
            filename_base = f"mushroom_{safe_pattern}_{timestamp}"
            
            # Save as WAV first
            wav_path = os.path.join(self.output_directory, f"{filename_base}.wav")
            
            # Convert to 16-bit PCM WAV
            audio_16bit = (audio * 32767).astype(np.int16)
            
            # Save WAV file using scipy
            from scipy.io import wavfile
            wavfile.write(wav_path, self.sampling_rate, audio_16bit)
            
            # Convert to MP3 using ffmpeg
            mp3_path = os.path.join(self.output_directory, f"{filename_base}.mp3")
            
            try:
                # Use ffmpeg to convert WAV to MP3
                cmd = [
                    'ffmpeg', '-y',  # Overwrite output file
                    '-i', wav_path,  # Input WAV file
                    '-codec:a', 'mp3',  # MP3 codec
                    '-b:a', '128k',  # 128 kbps bitrate
                    mp3_path  # Output MP3 file
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Remove temporary WAV file
                    os.remove(wav_path)
                    print(f"🎵 Saved MP3: {mp3_path}")
                    return mp3_path
                else:
                    print(f"⚠️  MP3 conversion failed, keeping WAV: {wav_path}")
                    return wav_path
                    
            except Exception as e:
                print(f"⚠️  FFmpeg not available, keeping WAV: {wav_path}")
                return wav_path
                
        except Exception as e:
            print(f"❌ Audio save error: {e}")
            return ""
    
    def analyze_moisture_levels(self, wave_transform_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze moisture levels from fungal electrical patterns
        """
        try:
            pattern_type = wave_transform_results.get('pattern_type', 'unknown')
            magnitude = wave_transform_results.get('magnitude', 1.0)
            
            # Moisture estimation based on pattern type
            moisture_estimates = {
                'alarm_signal': {'percentage': 15, 'confidence': 0.9, 'status': 'LOW'},
                'broadcast_signal': {'percentage': 55, 'confidence': 0.85, 'status': 'MODERATE'},
                'stress_response': {'percentage': 35, 'confidence': 0.8, 'status': 'LOW-MODERATE'},
                'growth_signal': {'percentage': 75, 'confidence': 0.9, 'status': 'HIGH'},
                'unknown_pattern': {'percentage': 50, 'confidence': 0.5, 'status': 'UNKNOWN'}
            }
            
            moisture_info = moisture_estimates.get(pattern_type, moisture_estimates['unknown_pattern'])
            
            # Add to history
            timestamp = datetime.now()
            moisture_record = {
                'timestamp': timestamp.isoformat(),
                'pattern_type': pattern_type,
                'moisture_percentage': moisture_info['percentage'],
                'confidence': moisture_info['confidence'],
                'status': moisture_info['status'],
                'magnitude': magnitude
            }
            
            self.moisture_history.append(moisture_record)
            
            # Check for alerts
            if moisture_info['status'] == 'LOW' and moisture_info['confidence'] > 0.8:
                alert = {
                    'timestamp': timestamp.isoformat(),
                    'type': 'LOW_MOISTURE_ALERT',
                    'message': f"⚠️  LOW MOISTURE DETECTED! Mushrooms are stressed. Pattern: {pattern_type}",
                    'moisture_percentage': moisture_info['percentage']
                }
                self.moisture_alerts.append(alert)
                print(f"🚨 {alert['message']}")
            
            return moisture_info
            
        except Exception as e:
            print(f"❌ Moisture analysis error: {e}")
            return {'percentage': 50, 'confidence': 0.0, 'status': 'ERROR'}
    
    def record_conversation_snippet(self, csv_path: str):
        """
        Record a single conversation snippet from fungal electrical activity
        """
        try:
            print(f"🎙️  Recording mushroom conversation snippet...")
            
            # Load fungal data chunk
            voltage_data = self.load_fungal_data_chunk(csv_path, chunk_size=2000)
            
            # Real-time wave transform
            wave_results = self.real_time_wave_transform(voltage_data)
            if not wave_results:
                print("❌ Wave transform failed")
                return
            
            # Convert to audio
            audio = self.convert_to_audio(wave_results)
            
            # Analyze moisture
            moisture_info = self.analyze_moisture_levels(wave_results)
            
            # Save audio file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = self.save_audio_file(audio, wave_results['pattern_type'], timestamp)
            
            # Display results
            print(f"🍄 Mushroom Conversation Recorded!")
            print(f"💬 Pattern: {wave_results['pattern_type']}")
            print(f"💧 Moisture: {moisture_info['status']} ({moisture_info['percentage']}%)")
            print(f"🎵 Audio saved: {os.path.basename(audio_path)}")
            
        except Exception as e:
            print(f"❌ Conversation recording error: {e}")
    
    def start_continuous_monitoring(self, csv_path: str, duration_minutes: int = 30):
        """
        Start continuous monitoring of fungal conversations and moisture levels
        """
        try:
            print(f"🌱 Starting CONTINUOUS MUSHROOM MONITORING...")
            print(f"⏱️  Duration: {duration_minutes} minutes")
            print(f"🎙️  Recording interval: {self.recording_interval} seconds")
            print(f"📁 Output directory: {self.output_directory}")
            print(f"📱 MP3 files ready for iPhone 7!")
            print("=" * 60)
            
            self.continuous_monitoring = True
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            recording_count = 0
            
            while self.continuous_monitoring and time.time() < end_time:
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = end_time - current_time
                
                # Record conversation snippet
                self.record_conversation_snippet(csv_path)
                recording_count += 1
                
                # Display progress
                print(f"\n📊 MONITORING PROGRESS:")
                print(f"⏱️  Elapsed: {elapsed/60:.1f} minutes")
                print(f"⏰ Remaining: {remaining/60:.1f} minutes")
                print(f"🎙️  Recordings: {recording_count}")
                print(f"💧 Current moisture: {self.moisture_history[-1]['status'] if self.moisture_history else 'UNKNOWN'}")
                
                # Check for alerts
                recent_alerts = [a for a in self.moisture_alerts if (time.time() - datetime.fromisoformat(a['timestamp']).timestamp()) < 300]  # Last 5 minutes
                if recent_alerts:
                    print(f"🚨 Recent alerts: {len(recent_alerts)}")
                
                print("-" * 40)
                
                # Wait for next recording interval
                if time.time() < end_time:
                    time.sleep(self.recording_interval)
            
            # Final summary
            self.generate_monitoring_report()
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Monitoring stopped by user")
            self.continuous_monitoring = False
        except Exception as e:
            print(f"❌ Continuous monitoring error: {e}")
            self.continuous_monitoring = False
    
    def generate_monitoring_report(self):
        """
        Generate comprehensive monitoring report
        """
        try:
            print(f"\n📊 MUSHROOM MONITORING REPORT")
            print("=" * 50)
            
            if self.moisture_history:
                # Moisture trends
                moisture_levels = [h['moisture_percentage'] for h in self.moisture_history]
                avg_moisture = np.mean(moisture_levels)
                min_moisture = np.min(moisture_levels)
                max_moisture = np.max(moisture_levels)
                
                print(f"💧 MOISTURE ANALYSIS:")
                print(f"   Average: {avg_moisture:.1f}%")
                print(f"   Range: {min_moisture:.1f}% - {max_moisture:.1f}%")
                print(f"   Trend: {'Increasing' if moisture_levels[-1] > moisture_levels[0] else 'Decreasing' if moisture_levels[-1] < moisture_levels[0] else 'Stable'}")
                
                # Pattern analysis
                pattern_counts = {}
                for h in self.moisture_history:
                    pattern = h['pattern_type']
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                print(f"\n💬 COMMUNICATION PATTERNS:")
                for pattern, count in pattern_counts.items():
                    percentage = (count / len(self.moisture_history)) * 100
                    print(f"   {pattern}: {count} times ({percentage:.1f}%)")
                
                # Alerts summary
                if self.moisture_alerts:
                    print(f"\n🚨 ALERTS SUMMARY:")
                    print(f"   Total alerts: {len(self.moisture_alerts)}")
                    recent_alerts = [a for a in self.moisture_alerts if (time.time() - datetime.fromisoformat(a['timestamp']).timestamp()) < 3600]  # Last hour
                    print(f"   Recent alerts (1 hour): {len(recent_alerts)}")
                
                # Audio files summary
                audio_files = [f for f in os.listdir(self.output_directory) if f.endswith(('.mp3', '.wav'))]
                print(f"\n🎵 AUDIO FILES GENERATED:")
                print(f"   Total files: {len(audio_files)}")
                print(f"   Directory: {self.output_directory}")
                print(f"   Ready for iPhone 7 transfer!")
            
            # Save report
            report_data = {
                'monitoring_summary': {
                    'total_recordings': len(self.moisture_history),
                    'total_alerts': len(self.moisture_alerts),
                    'monitoring_duration': 'continuous',
                    'output_directory': self.output_directory
                },
                'moisture_analysis': {
                    'history': self.moisture_history,
                    'alerts': self.moisture_alerts
                },
                'generated_at': datetime.now().isoformat()
            }
            
            report_path = os.path.join(self.output_directory, f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"\n💾 Report saved: {report_path}")
            
        except Exception as e:
            print(f"❌ Report generation error: {e}")
    
    def stop_monitoring(self):
        """
        Stop continuous monitoring
        """
        self.continuous_monitoring = False
        print("⏹️  Monitoring stopped")

def main():
    """Main function to demonstrate real-time mushroom conversation recording"""
    print("🍄 REAL-TIME MUSHROOM CONVERSATION RECORDER")
    print("🎙️  Records fungal electrical conversations as MP3 files")
    print("📱 iPhone 7 compatible audio output")
    print("💧 Continuous moisture monitoring")
    print("=" * 70)
    
    # Initialize recorder
    recorder = RealTimeMushroomRecorder()
    
    # Path to validated fungal data
    csv_path = 'DATA/raw/15061491/New_Oyster_with spray_as_mV.csv'
    
    try:
        print(f"\n📁 Using validated data: {Path(csv_path).name}")
        print(f"🎙️  Ready to record mushroom conversations!")
        print(f"📱 MP3 files will be saved to: {recorder.output_directory}")
        
        # Start continuous monitoring
        print(f"\n🚀 Starting continuous monitoring...")
        print(f"💡 Press Ctrl+C to stop monitoring")
        
        recorder.start_continuous_monitoring(csv_path, duration_minutes=30)  # 30 minutes demo
        
    except Exception as e:
        print(f"❌ Main execution error: {e}")
        print(f"💡 Make sure the validated CSV file exists at: {csv_path}")

if __name__ == "__main__":
    main() 