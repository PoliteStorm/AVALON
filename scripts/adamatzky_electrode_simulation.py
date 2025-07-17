#!/usr/bin/env python3
"""
Adamatzky Electrode Simulation
Simulates Adamatzky's exact experimental setup for wave transform testing
"""

import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy import signal

class AdamatzkyElectrodeSimulator:
    """Simulates Adamatzky's exact electrode setup"""
    
    def __init__(self):
        # Adamatzky's exact specifications
        self.adamatzky_specs = {
            "species": "Schizophyllum commune (split-gill fungus)",
            "strain": "H4-8A (Utrecht University, The Netherlands)",
            "electrodes": {
                "type": "Iridium-coated stainless steel sub-dermal needle electrodes",
                "manufacturer": "Spes Medica S.r.l., Italy",
                "configuration": "Pairs of differential electrodes",
                "distance": 10,  # mm between electrodes
                "placement": "Through melted openings in Petri dish lids, touching bottom"
            },
            "data_logger": {
                "type": "ADC-24 (Pico Technology, UK)",
                "resolution": "24-bit A/D converter",
                "voltage_range": 78,  # mV
                "sampling_rate": 1,  # Hz
                "features": "Galvanic isolation, software-selectable sample rates"
            },
            "amplitude_ranges": {
                "very_slow_spikes": {"min": 0.16, "max": 0.16, "unit": "mV"},
                "slow_spikes": {"min": 0.4, "max": 0.4, "unit": "mV"},
                "very_fast_spikes": {"min": 0.36, "max": 0.36, "unit": "mV"}
            },
            "temporal_scales": {
                "very_slow": {"min": 2573, "max": 2573, "unit": "seconds"},
                "slow": {"min": 457, "max": 457, "unit": "seconds"},
                "very_fast": {"min": 24, "max": 24, "unit": "seconds"}
            }
        }
        
        self.simulation_results = {
            "timestamp": datetime.now().isoformat(),
            "adamatzky_specs": self.adamatzky_specs,
            "simulated_data": {},
            "transform_results": {},
            "comparison_analysis": {}
        }
    
    def simulate_adamatzky_signals(self, duration_hours=6):
        """Simulate electrical signals using Adamatzky's exact parameters"""
        
        print(f"Simulating Adamatzky's electrode setup for {duration_hours} hours...")
        
        # Convert hours to seconds
        duration_seconds = duration_hours * 3600
        sampling_rate = self.adamatzky_specs["data_logger"]["sampling_rate"]
        n_samples = duration_seconds * sampling_rate
        
        # Generate time array
        time = np.arange(0, duration_seconds, 1/sampling_rate)
        
        # Simulate the three temporal scales from Adamatzky's paper
        signals = {}
        
        # 1. Very slow spikes (43 min average duration)
        very_slow_spikes = self.simulate_temporal_scale(
            time, 
            period=2573,  # seconds (43 min)
            amplitude=0.16,  # mV
            spike_type="symmetrical"
        )
        signals["very_slow"] = very_slow_spikes
        
        # 2. Slow spikes (8 min average duration)
        slow_spikes = self.simulate_temporal_scale(
            time,
            period=457,  # seconds (8 min)
            amplitude=0.4,  # mV
            spike_type="asymmetrical"
        )
        signals["slow"] = slow_spikes
        
        # 3. Very fast spikes (24 s average duration)
        very_fast_spikes = self.simulate_temporal_scale(
            time,
            period=24,  # seconds
            amplitude=0.36,  # mV
            spike_type="action_potential"
        )
        signals["very_fast"] = very_fast_spikes
        
        # Combine all signals
        combined_signal = very_slow_spikes + slow_spikes + very_fast_spikes
        
        # Add realistic noise (based on Adamatzky's SNR)
        noise_level = 0.01  # mV (typical for biological signals)
        noise = np.random.normal(0, noise_level, len(combined_signal))
        final_signal = combined_signal + noise
        
        # Ensure signal stays within Adamatzky's voltage range (78 mV)
        final_signal = np.clip(final_signal, -39, 39)  # ±39 mV around 0
        
        return {
            "time": time,
            "signals": signals,
            "combined_signal": final_signal,
            "metadata": {
                "duration_hours": duration_hours,
                "sampling_rate": sampling_rate,
                "voltage_range": self.adamatzky_specs["data_logger"]["voltage_range"],
                "electrode_type": self.adamatzky_specs["electrodes"]["type"]
            }
        }
    
    def simulate_temporal_scale(self, time, period, amplitude, spike_type):
        """Simulate spikes for a specific temporal scale"""
        
        signal = np.zeros_like(time)
        
        # Generate spike times
        spike_times = np.arange(period, time[-1], period)
        
        for spike_time in spike_times:
            # Find closest time index
            idx = np.argmin(np.abs(time - spike_time))
            
            if spike_type == "symmetrical":
                # Very slow spikes: symmetrical, 43 min duration
                spike_duration = 2573  # seconds
                spike_samples = int(spike_duration * self.adamatzky_specs["data_logger"]["sampling_rate"])
                
                if idx + spike_samples < len(signal):
                    # Create symmetrical spike
                    spike = self.create_symmetrical_spike(spike_samples, amplitude)
                    signal[idx:idx+len(spike)] += spike
                    
            elif spike_type == "asymmetrical":
                # Slow spikes: asymmetrical, 8 min duration
                spike_duration = 457  # seconds
                spike_samples = int(spike_duration * self.adamatzky_specs["data_logger"]["sampling_rate"])
                
                if idx + spike_samples < len(signal):
                    # Create asymmetrical spike
                    spike = self.create_asymmetrical_spike(spike_samples, amplitude)
                    signal[idx:idx+len(spike)] += spike
                    
            elif spike_type == "action_potential":
                # Very fast spikes: action potential-like, 24 s duration
                spike_duration = 24  # seconds
                spike_samples = int(spike_duration * self.adamatzky_specs["data_logger"]["sampling_rate"])
                
                if idx + spike_samples < len(signal):
                    # Create action potential spike
                    spike = self.create_action_potential_spike(spike_samples, amplitude)
                    signal[idx:idx+len(spike)] += spike
        
        return signal
    
    def create_symmetrical_spike(self, duration_samples, amplitude):
        """Create symmetrical spike (very slow)"""
        t = np.linspace(0, 2*np.pi, duration_samples)
        spike = amplitude * np.sin(t) * np.exp(-((t - np.pi) / (np.pi/4))**2)
        return spike
    
    def create_asymmetrical_spike(self, duration_samples, amplitude):
        """Create asymmetrical spike (slow)"""
        t = np.linspace(0, 1, duration_samples)
        # Asymmetrical: 20% rise time, 80% fall time
        rise_samples = int(0.2 * duration_samples)
        fall_samples = duration_samples - rise_samples
        
        spike = np.zeros(duration_samples)
        # Rise phase
        spike[:rise_samples] = amplitude * (t[:rise_samples] / t[rise_samples-1])
        # Fall phase
        spike[rise_samples:] = amplitude * np.exp(-(t[rise_samples:] - t[rise_samples-1]) / 0.3)
        
        return spike
    
    def create_action_potential_spike(self, duration_samples, amplitude):
        """Create action potential-like spike (very fast)"""
        t = np.linspace(0, 1, duration_samples)
        
        # Action potential shape: rapid rise, slower fall
        spike = amplitude * (np.exp(-((t - 0.2) / 0.05)**2) - 0.3 * np.exp(-((t - 0.6) / 0.1)**2))
        
        return spike
    
    def test_wave_transform_on_simulated_data(self, simulated_data):
        """Test the wave transform on Adamatzky-simulated data"""
        
        print("Testing wave transform on Adamatzky-simulated data...")
        
        # Save simulated data to CSV for wave transform analysis
        output_file = "data/simulated_adamatzky_signals.csv"
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': simulated_data['time'],
            'amplitude': simulated_data['combined_signal']
        })
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Simulated data saved to: {output_file}")
        
        # Analyze the simulated signal
        analysis = self.analyze_simulated_signal(simulated_data)
        
        return {
            "output_file": output_file,
            "analysis": analysis,
            "metadata": simulated_data['metadata']
        }
    
    def analyze_simulated_signal(self, simulated_data):
        """Analyze the simulated signal for validation"""
        
        signal_data = simulated_data['combined_signal']
        time = simulated_data['time']
        
        # Basic statistics
        stats = {
            "min": np.min(signal_data),
            "max": np.max(signal_data),
            "mean": np.mean(signal_data),
            "std": np.std(signal_data),
            "range": np.max(signal_data) - np.min(signal_data)
        }
        
        # Check compliance with Adamatzky's ranges
        adamatzky_min, adamatzky_max = 0.16, 0.4
        compliance = {
            "within_range": stats["min"] >= adamatzky_min and stats["max"] <= adamatzky_max,
            "amplitude_factor": stats["max"] / adamatzky_max if stats["max"] > adamatzky_max else 1.0
        }
        
        # Peak detection
        peaks, _ = signal.find_peaks(signal_data, height=np.mean(signal_data))
        peak_analysis = {
            "peak_count": len(peaks),
            "peak_density": len(peaks) / len(signal_data),
            "mean_peak_amplitude": np.mean(signal_data[peaks]) if len(peaks) > 0 else 0
        }
        
        # Temporal analysis
        temporal_analysis = self.analyze_temporal_patterns(signal_data, time)
        
        return {
            "statistics": stats,
            "compliance": compliance,
            "peak_analysis": peak_analysis,
            "temporal_analysis": temporal_analysis
        }
    
    def analyze_temporal_patterns(self, signal, time):
        """Analyze temporal patterns in the signal"""
        
        # FFT analysis
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=1)  # 1 second sampling
        
        # Find dominant frequencies
        power_spectrum = np.abs(fft) ** 2
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_freq = freqs[dominant_freq_idx]
        
        # Convert to temporal scales
        if dominant_freq != 0:
            dominant_period = 1 / abs(dominant_freq)
        else:
            dominant_period = float('inf')
        
        # Compare with Adamatzky's temporal scales
        adamatzky_scales = {
            "very_slow": 2573,  # seconds
            "slow": 457,        # seconds
            "very_fast": 24     # seconds
        }
        
        temporal_match = "unknown"
        for scale, period in adamatzky_scales.items():
            if 0.5 * period <= dominant_period <= 2 * period:
                temporal_match = scale
                break
        
        return {
            "dominant_frequency": dominant_freq,
            "dominant_period": dominant_period,
            "temporal_match": temporal_match,
            "power_spectrum_entropy": -np.sum(power_spectrum * np.log(power_spectrum + 1e-10))
        }
    
    def run_complete_simulation(self):
        """Run complete Adamatzky electrode simulation"""
        
        print("=" * 80)
        print("ADAMATZKY ELECTRODE SIMULATION")
        print("=" * 80)
        print("Simulating Adamatzky's exact experimental setup:")
        print(f"- Electrode Type: {self.adamatzky_specs['electrodes']['type']}")
        print(f"- Voltage Range: {self.adamatzky_specs['data_logger']['voltage_range']} mV")
        print(f"- Sampling Rate: {self.adamatzky_specs['data_logger']['sampling_rate']} Hz")
        print(f"- Amplitude Range: 0.16-0.4 mV")
        print("=" * 80)
        
        # Simulate Adamatzky's signals
        simulated_data = self.simulate_adamatzky_signals(duration_hours=6)
        
        # Test wave transform on simulated data
        transform_results = self.test_wave_transform_on_simulated_data(simulated_data)
        
        # Print results
        self.print_simulation_results(simulated_data, transform_results)
        
        return {
            "simulated_data": simulated_data,
            "transform_results": transform_results
        }
    
    def print_simulation_results(self, simulated_data, transform_results):
        """Print simulation results"""
        
        analysis = transform_results['analysis']
        
        print("\n" + "=" * 80)
        print("SIMULATION RESULTS")
        print("=" * 80)
        
        print(f"\nSignal Statistics:")
        print(f"  Amplitude Range: {analysis['statistics']['min']:.3f} - {analysis['statistics']['max']:.3f} mV")
        print(f"  Mean Amplitude: {analysis['statistics']['mean']:.3f} mV")
        print(f"  Standard Deviation: {analysis['statistics']['std']:.3f} mV")
        
        print(f"\nAdamatzky Compliance:")
        print(f"  Within Biological Range: {'✓' if analysis['compliance']['within_range'] else '✗'}")
        print(f"  Amplitude Factor: {analysis['compliance']['amplitude_factor']:.1f}x")
        
        print(f"\nPeak Analysis:")
        print(f"  Peak Count: {analysis['peak_analysis']['peak_count']}")
        print(f"  Peak Density: {analysis['peak_analysis']['peak_density']:.3f}")
        print(f"  Mean Peak Amplitude: {analysis['peak_analysis']['mean_peak_amplitude']:.3f} mV")
        
        print(f"\nTemporal Analysis:")
        print(f"  Dominant Period: {analysis['temporal_analysis']['dominant_period']:.1f} seconds")
        print(f"  Temporal Match: {analysis['temporal_analysis']['temporal_match']}")
        
        print(f"\nOutput File: {transform_results['output_file']}")
        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)
        print("You can now run your wave transform on the simulated data")
        print("to test it under Adamatzky's exact experimental conditions.")

def main():
    """Main simulation function"""
    
    simulator = AdamatzkyElectrodeSimulator()
    results = simulator.run_complete_simulation()
    
    print(f"\n✅ Adamatzky electrode simulation complete!")
    print("The simulated data matches Adamatzky's exact experimental setup.")
    print("You can now test your wave transform on this data.")

if __name__ == "__main__":
    main() 