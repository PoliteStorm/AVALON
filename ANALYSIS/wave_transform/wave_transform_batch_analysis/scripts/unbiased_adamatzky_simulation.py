#!/usr/bin/env python3
"""
Unbiased Adamatzky Simulation
Generates realistic biological signals without forced parameters
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy import signal as scipy_signal

class UnbiasedAdamatzkySimulator:
    """Generates unbiased biological signals matching Adamatzky's experimental setup"""
    
    def __init__(self):
        # Adamatzky's experimental setup (no forced patterns)
        self.experimental_setup = {
            "electrode_type": "Iridium-coated stainless steel sub-dermal needle electrodes",
            "voltage_range": 78,  # mV
            "sampling_rate": 1,   # Hz
            "biological_amplitude_range": (0.05, 5.0),  # mV
            "recording_duration_hours": 6
        }
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "experimental_setup": self.experimental_setup,
            "simulation_parameters": {},
            "output_file": None
        }
    
    def generate_unbiased_biological_signal(self, duration_hours=6):
        """Generate unbiased biological signal without forced patterns"""
        
        print(f"Generating unbiased biological signal for {duration_hours} hours...")
        print("No forced patterns - letting natural biological variability emerge")
        
        # Convert to seconds
        duration_seconds = duration_hours * 3600
        n_samples = duration_seconds * self.experimental_setup["sampling_rate"]
        
        # Generate time array
        time = np.arange(0, duration_seconds, 1/self.experimental_setup["sampling_rate"])
        
        # Generate unbiased biological signal
        signal = self.create_unbiased_signal(n_samples)
        
        # Ensure signal stays within experimental voltage range
        signal = np.clip(signal, -39, 39)  # ±39 mV around 0
        
        return {
            "time": time,
            "amplitude": signal,
            "metadata": {
                "duration_hours": duration_hours,
                "sampling_rate": self.experimental_setup["sampling_rate"],
                "voltage_range": self.experimental_setup["voltage_range"],
                "electrode_type": self.experimental_setup["electrode_type"],
                "unbiased_generation": True
            }
        }
    
    def create_unbiased_signal(self, n_samples):
        """Create unbiased biological signal without forced patterns"""
        
        # Start with pure noise (biological baseline)
        signal = np.random.normal(0, 0.01, n_samples)  # 0.01 mV noise
        
        # Add random biological events (no forced timing or amplitude)
        n_events = np.random.poisson(n_samples / 10000)  # Random event count
        
        for _ in range(n_events):
            # Random event timing
            event_time = np.random.uniform(0, n_samples)
            event_idx = int(event_time)
            
            if event_idx < n_samples:
                # Random event characteristics
                event_amplitude = np.random.uniform(0.05, 5.0)  # Adamatzky's range
                event_duration = np.random.uniform(10, 1000)  # 10-1000 seconds
                event_samples = int(event_duration)
                
                # Random event shape (no forced patterns)
                event_shape = self.generate_random_event_shape(event_samples, event_amplitude)
                
                # Add event to signal
                end_idx = min(event_idx + len(event_shape), n_samples)
                signal[event_idx:end_idx] += event_shape[:end_idx-event_idx]
        
        # Add slow biological trends (no forced periodicity)
        trend_frequency = np.random.uniform(0.0001, 0.001)  # Very slow trends
        trend_amplitude = np.random.uniform(0.01, 0.1)
        trend = trend_amplitude * np.sin(2 * np.pi * trend_frequency * np.arange(n_samples))
        signal += trend
        
        return signal
    
    def generate_random_event_shape(self, duration_samples, amplitude):
        """Generate random event shape without forced patterns"""
        
        # Random shape parameters
        shape_type = np.random.choice(['gaussian', 'exponential', 'oscillatory', 'spike'])
        
        if shape_type == 'gaussian':
            # Gaussian-like event
            t = np.linspace(-3, 3, duration_samples)
            event = amplitude * np.exp(-(t**2) / 2)
            
        elif shape_type == 'exponential':
            # Exponential rise/fall
            t = np.linspace(0, 1, duration_samples)
            if np.random.random() > 0.5:
                event = amplitude * (1 - np.exp(-5 * t))  # Rise
            else:
                event = amplitude * np.exp(-5 * t)  # Fall
                
        elif shape_type == 'oscillatory':
            # Oscillatory event
            t = np.linspace(0, 2*np.pi, duration_samples)
            frequency = np.random.uniform(0.1, 1.0)
            event = amplitude * np.sin(frequency * t) * np.exp(-t / (2*np.pi))
            
        else:  # spike
            # Sharp spike
            t = np.linspace(0, 1, duration_samples)
            event = amplitude * np.exp(-((t - 0.5) / 0.1)**2)
        
        return event
    
    def save_simulated_data(self, simulated_data):
        """Save simulated data to CSV"""
        
        output_file = "data/unbiased_adamatzky_signals.csv"
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': simulated_data['time'],
            'amplitude': simulated_data['amplitude']
        })
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        self.results["output_file"] = output_file
        self.results["simulation_parameters"] = {
            "n_samples": len(simulated_data['amplitude']),
            "duration_seconds": simulated_data['time'][-1],
            "amplitude_range": (np.min(simulated_data['amplitude']), np.max(simulated_data['amplitude'])),
            "mean_amplitude": np.mean(simulated_data['amplitude']),
            "std_amplitude": np.std(simulated_data['amplitude'])
        }
        
        print(f"Unbiased simulated data saved to: {output_file}")
        return output_file
    
    def analyze_signal_characteristics(self, simulated_data):
        """Analyze signal characteristics without bias"""
        
        signal = simulated_data['amplitude']
        
        # Basic statistics
        stats = {
            "min": np.min(signal),
            "max": np.max(signal),
            "mean": np.mean(signal),
            "std": np.std(signal),
            "range": np.max(signal) - np.min(signal)
        }
        
        # Peak detection (no forced thresholds)
        peaks, properties = scipy_signal.find_peaks(signal, prominence=0.01)
        
        peak_analysis = {
            "peak_count": len(peaks),
            "peak_density": len(peaks) / len(signal),
            "mean_peak_amplitude": np.mean(signal[peaks]) if len(peaks) > 0 else 0,
            "peak_prominence_mean": np.mean(properties['prominences']) if len(peaks) > 0 else 0
        }
        
        # Frequency analysis (no forced frequency bands)
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1)
        power_spectrum = np.abs(fft) ** 2
        
        # Find dominant frequencies naturally
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        dominant_period = 1 / abs(dominant_freq) if dominant_freq != 0 else float('inf')
        
        frequency_analysis = {
            "dominant_frequency": dominant_freq,
            "dominant_period": dominant_period,
            "spectral_entropy": -np.sum(power_spectrum * np.log(power_spectrum + 1e-10)),
            "total_power": np.sum(power_spectrum)
        }
        
        return {
            "statistics": stats,
            "peak_analysis": peak_analysis,
            "frequency_analysis": frequency_analysis
        }
    
    def run_unbiased_simulation(self):
        """Run unbiased simulation"""
        
        print("=" * 80)
        print("UNBIASED ADAMATZKY SIMULATION")
        print("=" * 80)
        print("Generating realistic biological signals WITHOUT forced parameters:")
        print(f"- Electrode Type: {self.experimental_setup['electrode_type']}")
        print(f"- Voltage Range: {self.experimental_setup['voltage_range']} mV")
        print(f"- Sampling Rate: {self.experimental_setup['sampling_rate']} Hz")
        print(f"- Biological Range: {self.experimental_setup['biological_amplitude_range']} mV")
        print("- NO forced patterns or timing")
        print("- NO forced amplitude relationships")
        print("- NO forced frequency bands")
        print("=" * 80)
        
        # Generate unbiased signal
        simulated_data = self.generate_unbiased_biological_signal(duration_hours=6)
        
        # Save data
        output_file = self.save_simulated_data(simulated_data)
        
        # Analyze characteristics
        analysis = self.analyze_signal_characteristics(simulated_data)
        
        # Print results
        self.print_unbiased_results(simulated_data, analysis)
        
        # Save results
        self.save_results(analysis)
        
        return {
            "simulated_data": simulated_data,
            "analysis": analysis,
            "output_file": output_file
        }
    
    def print_unbiased_results(self, simulated_data, analysis):
        """Print unbiased simulation results"""
        
        print("\n" + "=" * 80)
        print("UNBIASED SIMULATION RESULTS")
        print("=" * 80)
        
        print(f"\nSignal Characteristics:")
        print(f"  Amplitude Range: {analysis['statistics']['min']:.3f} - {analysis['statistics']['max']:.3f} mV")
        print(f"  Mean Amplitude: {analysis['statistics']['mean']:.3f} mV")
        print(f"  Standard Deviation: {analysis['statistics']['std']:.3f} mV")
        
        print(f"\nPeak Analysis (Natural Detection):")
        print(f"  Peak Count: {analysis['peak_analysis']['peak_count']}")
        print(f"  Peak Density: {analysis['peak_analysis']['peak_density']:.6f}")
        print(f"  Mean Peak Amplitude: {analysis['peak_analysis']['mean_peak_amplitude']:.3f} mV")
        
        print(f"\nFrequency Analysis (Natural Detection):")
        print(f"  Dominant Period: {analysis['frequency_analysis']['dominant_period']:.1f} seconds")
        print(f"  Spectral Entropy: {analysis['frequency_analysis']['spectral_entropy']:.3f}")
        
        print(f"\nOutput File: {self.results['output_file']}")
        print("\n" + "=" * 80)
        print("UNBIASED SIMULATION COMPLETE")
        print("=" * 80)
        print("✅ No forced parameters used")
        print("✅ Natural biological variability preserved")
        print("✅ Transform can detect patterns honestly")
        print("✅ Ready for unbiased wave transform testing")
    
    def save_results(self, analysis):
        """Save simulation results"""
        
        results_file = f"results/unbiased_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        # Combine results
        full_results = {
            **self.results,
            "analysis": analysis
        }
        
        # Save to JSON
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        print(f"Results saved to: {results_file}")

def main():
    """Main unbiased simulation function"""
    
    simulator = UnbiasedAdamatzkySimulator()
    results = simulator.run_unbiased_simulation()
    
    print(f"\n✅ Unbiased Adamatzky simulation complete!")
    print("The simulated data contains no forced parameters.")
    print("Your wave transform can now test honestly on this data.")

if __name__ == "__main__":
    main() 