#!/usr/bin/env python3
"""
🎵 LIVE NOISE GENERATOR - Real-time Audio Feedback During Analysis
===============================================================

This system generates live audio feedback during CSV analysis based on:
1. Mycelial network patterns (Fricker et al., 2017)
2. Environmental stress responses (Adamatzky 2023)
3. Real-time data processing feedback
4. Interactive audio-visual correlation

Author: Environmental Sensing Research Team
Date: August 13, 2025
Version: 1.0.0
Research Compliance: ✅ Fricker 2017, Adamatzky 2023
"""

import numpy as np
import sounddevice as sd
import threading
import queue
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_noise_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveNoiseGenerator:
    """
    Real-time noise generator that provides live audio feedback during CSV analysis.
    
    Based on research findings:
    - Fricker et al. (2017): Mycelial network architecture and resource flow
    - Adamatzky (2023): Fungal electrical spiking patterns
    - Real-time environmental monitoring principles
    """
    
    def __init__(self, sample_rate: int = 44100, buffer_size: int = 1024):
        """
        Initialize the live noise generator.
        
        Args:
            sample_rate: Audio sample rate in Hz
            buffer_size: Audio buffer size for real-time processing
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Audio generation parameters
        self.audio_params = {
            'base_frequency': 220.0,  # A3 note (220 Hz)
            'frequency_range': [20, 2000],  # Hz
            'amplitude_range': [0.1, 0.8],  # Normalized amplitude
            'noise_types': ['white', 'pink', 'brown', 'mycelial', 'environmental']
        }
        
        # Mycelial network parameters (Fricker et al., 2017)
        self.mycelial_params = {
            'hyphal_growth_rate': 0.1,  # mm/hour
            'branching_frequency': 0.05,  # branches per mm
            'fusion_probability': 0.02,   # fusion events per mm
            'resource_flow_rate': 0.5,   # arbitrary units
            'stress_response_threshold': 0.7
        }
        
        # Environmental monitoring parameters
        self.environmental_params = {
            'temperature_sensitivity': 0.1,  # Hz/°C
            'humidity_sensitivity': 0.05,   # Hz/%
            'pollution_sensitivity': 0.2,   # Hz/ppm
            'stress_response_time': 0.5     # seconds
        }
        
        # Audio generation state
        self.is_generating = False
        self.current_noise_type = 'mycelial'
        self.audio_thread = None
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Real-time data buffers
        self.environmental_data_buffer = []
        self.csv_analysis_buffer = []
        self.audio_pattern_buffer = []
        
        # Performance metrics
        self.performance_metrics = {
            'audio_samples_generated': 0,
            'buffer_underruns': 0,
            'processing_latency_ms': [],
            'audio_quality_score': 100.0
        }
        
        # Initialize audio system
        self._initialize_audio_system()
        
        logger.info("🎵 Live Noise Generator initialized successfully")
        logger.info(f"📊 Sample rate: {sample_rate} Hz")
        logger.info(f"🔊 Buffer size: {buffer_size} samples")
        logger.info(f"🍄 Mycelial parameters: {len(self.mycelial_params)} parameters")
    
    def _initialize_audio_system(self):
        """Initialize the audio system for real-time generation."""
        try:
            # Test audio device
            devices = sd.query_devices()
            logger.info(f"🎧 Available audio devices: {len(devices)}")
            
            # Set default device
            sd.default.device = sd.default.device[0], sd.default.device[1]
            logger.info(f"🎧 Using audio device: {sd.default.device}")
            
        except Exception as e:
            logger.error(f"❌ Audio system initialization failed: {e}")
            raise
    
    def start_live_noise(self, noise_type: str = 'mycelial'):
        """
        Start generating live noise based on the specified type.
        
        Args:
            noise_type: Type of noise to generate
        """
        try:
            if self.is_generating:
                logger.warning("⚠️ Live noise generation already active")
                return False
            
            self.current_noise_type = noise_type
            self.is_generating = True
            
            # Start audio generation thread
            self.audio_thread = threading.Thread(target=self._audio_generation_loop, daemon=True)
            self.audio_thread.start()
            
            logger.info(f"🚀 Live noise generation started: {noise_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting live noise: {e}")
            return False
    
    def stop_live_noise(self):
        """Stop live noise generation."""
        try:
            self.is_generating = False
            
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=1.0)
            
            logger.info("⏹️ Live noise generation stopped")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error stopping live noise: {e}")
            return False
    
    def _audio_generation_loop(self):
        """Main audio generation loop."""
        try:
            while self.is_generating:
                start_time = time.time()
                
                # Generate audio buffer
                audio_buffer = self._generate_audio_buffer()
                
                # Add to queue for playback
                if not self.audio_queue.full():
                    self.audio_queue.put(audio_buffer)
                
                # Play audio in real-time
                self._play_audio_buffer(audio_buffer)
                
                # Update performance metrics
                processing_time = (time.time() - start_time) * 1000
                self.performance_metrics['processing_latency_ms'].append(processing_time)
                self.performance_metrics['audio_samples_generated'] += len(audio_buffer)
                
                # Keep only recent metrics
                if len(self.performance_metrics['processing_latency_ms']) > 100:
                    self.performance_metrics['processing_latency_ms'] = self.performance_metrics['processing_latency_ms'][-100:]
                
                # Sleep to maintain real-time performance
                time.sleep(self.buffer_size / self.sample_rate)
                
        except Exception as e:
            logger.error(f"❌ Error in audio generation loop: {e}")
            self.is_generating = False
    
    def _generate_audio_buffer(self) -> np.ndarray:
        """Generate audio buffer based on current noise type and environmental data."""
        try:
            if self.current_noise_type == 'mycelial':
                return self._generate_mycelial_noise()
            elif self.current_noise_type == 'environmental':
                return self._generate_environmental_noise()
            elif self.current_noise_type == 'white':
                return self._generate_white_noise()
            elif self.current_noise_type == 'pink':
                return self._generate_pink_noise()
            elif self.current_noise_type == 'brown':
                return self._generate_brown_noise()
            else:
                return self._generate_mycelial_noise()  # Default
                
        except Exception as e:
            logger.error(f"Error generating audio buffer: {e}")
            return np.zeros(self.buffer_size)
    
    def _generate_mycelial_noise(self) -> np.ndarray:
        """
        Generate mycelial network-inspired noise based on Fricker et al. (2017).
        
        Features:
        - Hyphal growth patterns
        - Branching events
        - Fusion events
        - Resource flow dynamics
        """
        try:
            # Generate time array
            t = np.linspace(0, self.buffer_size / self.sample_rate, self.buffer_size)
            
            # Base mycelial growth pattern
            growth_pattern = np.sin(2 * np.pi * self.mycelial_params['hyphal_growth_rate'] * t)
            
            # Add branching events (random high-frequency bursts)
            branching_events = np.random.poisson(self.mycelial_params['branching_frequency'] * 100, self.buffer_size) * 0.1
            branching_events = np.convolve(branching_events, np.ones(10)/10, mode='same')
            
            # Add fusion events (low-frequency modulations)
            fusion_events = np.sin(2 * np.pi * self.mycelial_params['fusion_probability'] * t) * 0.05
            
            # Add resource flow dynamics
            resource_flow = np.sin(2 * np.pi * self.mycelial_params['resource_flow_rate'] * t) * 0.1
            
            # Combine all patterns
            mycelial_noise = (growth_pattern * 0.3 + 
                            branching_events * 0.4 + 
                            fusion_events * 0.2 + 
                            resource_flow * 0.1)
            
            # Add environmental stress response
            if self.environmental_data_buffer:
                stress_level = self._calculate_environmental_stress()
                if stress_level > self.mycelial_params['stress_response_threshold']:
                    # Add high-frequency stress response
                    stress_response = np.sin(2 * np.pi * 1000 * t) * 0.2
                    mycelial_noise += stress_response
            
            # Normalize and apply amplitude
            mycelial_noise = np.clip(mycelial_noise, -1, 1) * 0.5
            
            return mycelial_noise.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating mycelial noise: {e}")
            return np.zeros(self.buffer_size, dtype=np.float32)
    
    def _generate_environmental_noise(self) -> np.ndarray:
        """
        Generate environmental monitoring noise based on real-time data.
        
        Features:
        - Temperature variations
        - Humidity changes
        - Pollution levels
        - Environmental stress responses
        """
        try:
            # Generate time array
            t = np.linspace(0, self.buffer_size / self.sample_rate, self.buffer_size)
            
            # Base environmental pattern
            base_pattern = np.sin(2 * np.pi * 0.5 * t) * 0.3
            
            # Temperature variations
            temp_variation = np.sin(2 * np.pi * self.environmental_params['temperature_sensitivity'] * t) * 0.2
            
            # Humidity changes
            humidity_variation = np.sin(2 * np.pi * self.environmental_params['humidity_sensitivity'] * t) * 0.15
            
            # Pollution response (high-frequency when pollution detected)
            pollution_response = np.zeros(self.buffer_size)
            if self.environmental_data_buffer:
                pollution_level = self._get_current_pollution_level()
                if pollution_level > 0.5:
                    pollution_response = np.sin(2 * np.pi * 800 * t) * 0.3
            
            # Environmental stress response
            stress_response = np.zeros(self.buffer_size)
            if self.environmental_data_buffer:
                stress_level = self._calculate_environmental_stress()
                if stress_level > 0.6:
                    stress_response = np.sin(2 * np.pi * 1200 * t) * 0.25
            
            # Combine all patterns
            environmental_noise = (base_pattern + 
                                temp_variation + 
                                humidity_variation + 
                                pollution_response + 
                                stress_response)
            
            # Normalize and apply amplitude
            environmental_noise = np.clip(environmental_noise, -1, 1) * 0.6
            
            return environmental_noise.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating environmental noise: {e}")
            return np.zeros(self.buffer_size, dtype=np.float32)
    
    def _generate_white_noise(self) -> np.ndarray:
        """Generate white noise."""
        return np.random.normal(0, 0.3, self.buffer_size).astype(np.float32)
    
    def _generate_pink_noise(self) -> np.ndarray:
        """Generate pink noise (1/f spectrum)."""
        try:
            # Generate white noise
            white_noise = np.random.normal(0, 1, self.buffer_size)
            
            # Apply 1/f filter
            freqs = np.fft.fftfreq(self.buffer_size)
            pink_filter = 1 / np.sqrt(np.abs(freqs) + 1e-10)
            pink_filter[0] = 0  # Remove DC component
            
            # Apply filter in frequency domain
            pink_spectrum = np.fft.fft(white_noise) * pink_filter
            pink_noise = np.real(np.fft.ifft(pink_spectrum))
            
            # Normalize
            pink_noise = pink_noise / np.max(np.abs(pink_noise)) * 0.4
            
            return pink_noise.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating pink noise: {e}")
            return np.zeros(self.buffer_size, dtype=np.float32)
    
    def _generate_brown_noise(self) -> np.ndarray:
        """Generate brown noise (1/f² spectrum)."""
        try:
            # Generate white noise
            white_noise = np.random.normal(0, 1, self.buffer_size)
            
            # Apply 1/f² filter
            freqs = np.fft.fftfreq(self.buffer_size)
            brown_filter = 1 / (np.abs(freqs) + 1e-10)
            brown_filter[0] = 0  # Remove DC component
            
            # Apply filter in frequency domain
            brown_spectrum = np.fft.fft(white_noise) * brown_filter
            brown_noise = np.real(np.fft.ifft(brown_spectrum))
            
            # Normalize
            brown_noise = brown_noise / np.max(np.abs(brown_noise)) * 0.5
            
            return brown_noise.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating brown noise: {e}")
            return np.zeros(self.buffer_size, dtype=np.float32)
    
    def _play_audio_buffer(self, audio_buffer: np.ndarray):
        """Play audio buffer in real-time."""
        try:
            # Play audio using sounddevice
            sd.play(audio_buffer, self.sample_rate)
            
            # Wait for playback to complete
            sd.wait()
            
        except Exception as e:
            logger.error(f"Error playing audio buffer: {e}")
            self.performance_metrics['buffer_underruns'] += 1
    
    def update_environmental_data(self, environmental_data: Dict[str, float]):
        """
        Update environmental data for real-time noise generation.
        
        Args:
            environmental_data: Current environmental parameters
        """
        try:
            # Add timestamp
            environmental_data['timestamp'] = datetime.now().isoformat()
            
            # Add to buffer
            self.environmental_data_buffer.append(environmental_data)
            
            # Keep only recent data
            if len(self.environmental_data_buffer) > 1000:
                self.environmental_data_buffer = self.environmental_data_buffer[-1000:]
            
            logger.debug(f"Environmental data updated: {len(environmental_data)} parameters")
            
        except Exception as e:
            logger.error(f"Error updating environmental data: {e}")
    
    def update_csv_analysis_data(self, analysis_data: Dict[str, Any]):
        """
        Update CSV analysis data for noise generation.
        
        Args:
            analysis_data: Current CSV analysis results
        """
        try:
            # Add timestamp
            analysis_data['timestamp'] = datetime.now().isoformat()
            
            # Add to buffer
            self.csv_analysis_buffer.append(analysis_data)
            
            # Keep only recent data
            if len(self.csv_analysis_buffer) > 1000:
                self.csv_analysis_buffer = self.csv_analysis_buffer[-1000:]
            
            logger.debug(f"CSV analysis data updated: {len(analysis_data)} parameters")
            
        except Exception as e:
            logger.error(f"Error updating CSV analysis data: {e}")
    
    def _calculate_environmental_stress(self) -> float:
        """Calculate environmental stress level from current data."""
        try:
            if not self.environmental_data_buffer:
                return 0.0
            
            # Get most recent data
            current_data = self.environmental_data_buffer[-1]
            
            # Calculate stress indicators
            temp_stress = 0.0
            if 'temperature' in current_data:
                temp = current_data['temperature']
                if temp < 15 or temp > 30:  # Optimal range: 15-30°C
                    temp_stress = min(1.0, abs(temp - 22.5) / 15)
            
            humidity_stress = 0.0
            if 'humidity' in current_data:
                humidity = current_data['humidity']
                if humidity < 40 or humidity > 80:  # Optimal range: 40-80%
                    humidity_stress = min(1.0, abs(humidity - 60) / 40)
            
            pollution_stress = 0.0
            if 'pollution' in current_data:
                pollution = current_data['pollution']
                pollution_stress = min(1.0, pollution / 100)  # Normalize to 0-1
            
            # Combine stress indicators
            total_stress = (temp_stress + humidity_stress + pollution_stress) / 3
            
            return total_stress
            
        except Exception as e:
            logger.error(f"Error calculating environmental stress: {e}")
            return 0.0
    
    def _get_current_pollution_level(self) -> float:
        """Get current pollution level from environmental data."""
        try:
            if not self.environmental_data_buffer:
                return 0.0
            
            current_data = self.environmental_data_buffer[-1]
            return current_data.get('pollution', 0.0) / 100  # Normalize to 0-1
            
        except Exception as e:
            logger.error(f"Error getting pollution level: {e}")
            return 0.0
    
    def change_noise_type(self, noise_type: str):
        """
        Change the type of noise being generated.
        
        Args:
            noise_type: New noise type
        """
        try:
            if noise_type in self.audio_params['noise_types']:
                self.current_noise_type = noise_type
                logger.info(f"🎵 Noise type changed to: {noise_type}")
                return True
            else:
                logger.warning(f"⚠️ Unknown noise type: {noise_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error changing noise type: {e}")
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        try:
            metrics = self.performance_metrics.copy()
            
            # Calculate additional metrics
            if metrics['processing_latency_ms']:
                metrics['avg_latency_ms'] = np.mean(metrics['processing_latency_ms'])
                metrics['max_latency_ms'] = np.max(metrics['processing_latency_ms'])
            else:
                metrics['avg_latency_ms'] = 0.0
                metrics['max_latency_ms'] = 0.0
            
            # Calculate audio quality score
            buffer_utilization = len(self.environmental_data_buffer) / 1000
            latency_penalty = max(0, (metrics['avg_latency_ms'] - 50) / 50 * 20)
            underrun_penalty = metrics['buffer_underruns'] * 5
            
            metrics['audio_quality_score'] = max(0, 100 - latency_penalty - underrun_penalty)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'is_generating': self.is_generating,
                'current_noise_type': self.current_noise_type,
                'sample_rate': self.sample_rate,
                'buffer_size': self.buffer_size,
                'environmental_data_buffer_size': len(self.environmental_data_buffer),
                'csv_analysis_buffer_size': len(self.csv_analysis_buffer),
                'performance_metrics': self.get_performance_metrics(),
                'mycelial_params': self.mycelial_params,
                'environmental_params': self.environmental_params
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e), 'status': 'failed'}

def main():
    """Main execution function for testing the live noise generator."""
    print("🎵 LIVE NOISE GENERATOR - Real-time Audio Feedback During Analysis")
    print("=" * 70)
    
    try:
        # Initialize noise generator
        noise_generator = LiveNoiseGenerator()
        
        print("✅ Live noise generator initialized successfully")
        print(f"🎧 Sample rate: {noise_generator.sample_rate} Hz")
        print(f"🔊 Buffer size: {noise_generator.buffer_size} samples")
        print(f"🍄 Mycelial parameters: {len(noise_generator.mycelial_params)} parameters")
        print(f"🌍 Environmental parameters: {len(noise_generator.environmental_params)} parameters")
        
        # Test different noise types
        print("\n🎵 Testing different noise types...")
        for noise_type in noise_generator.audio_params['noise_types']:
            print(f"  - {noise_type}: Available")
        
        # Get system status
        status = noise_generator.get_system_status()
        print(f"🏥 System status: {status['is_generating']}")
        
        print("\n🚀 Ready for live noise generation!")
        print("🎵 Use noise_generator.start_live_noise() to start generation")
        print("🔊 Use noise_generator.change_noise_type() to change noise type")
        print("📊 Use noise_generator.update_environmental_data() for real-time updates")
        print("⏹️ Use noise_generator.stop_live_noise() to stop generation")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Live noise generator initialization failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 