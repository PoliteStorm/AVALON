#!/usr/bin/env python3
"""
Comprehensive Electrical Activity Data Extraction and Analysis
Extracts electrical activity data from coordinate files, direct recordings, and moisture data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ElectricalActivityExtractor:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        self.results = {}
        
    def extract_coordinate_electrical_data(self, csv_file):
        """Extract electrical-like signals from coordinate data by calculating movement metrics"""
        try:
            # Read coordinate data
            df = pd.read_csv(csv_file, header=None, names=['x', 'y'])
            
            # Calculate movement-based electrical signals
            # Distance from origin (radial distance)
            distance = np.sqrt(df['x']**2 + df['y']**2)
            
            # Velocity (rate of change of distance)
            velocity = np.gradient(distance)
            
            # Acceleration (rate of change of velocity)
            acceleration = np.gradient(velocity)
            
            # Angular velocity (rate of change of angle)
            angle = np.arctan2(df['y'], df['x'])
            angular_velocity = np.gradient(angle)
            
            # Curvature (rate of change of angle with respect to distance)
            curvature = np.gradient(angular_velocity) / (np.abs(velocity) + 1e-10)
            
            # Create electrical-like signals
            signals = {
                'distance': distance,
                'velocity': velocity,
                'acceleration': acceleration,
                'angular_velocity': angular_velocity,
                'curvature': curvature,
                'x_position': df['x'],
                'y_position': df['y']
            }
            
            return signals, len(df)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            return None, 0
    
    def extract_direct_electrical_data(self, csv_file):
        """Extract direct electrical recordings from voltage/time data"""
        try:
            # Try different reading strategies
            strategies = [
                lambda f: pd.read_csv(f, header=0),
                lambda f: pd.read_csv(f, header=None),
                lambda f: pd.read_csv(f, skiprows=1),
                lambda f: pd.read_csv(f, skiprows=2)
            ]
            
            df = None
            for strategy in strategies:
                try:
                    df = strategy(csv_file)
                    if len(df.columns) >= 1:
                        break
                except:
                    continue
            
            if df is None or len(df) == 0:
                return None, 0
            
            # Find the most likely voltage column
            voltage_col = None
            for col in df.columns:
                if isinstance(col, str):
                    if any(keyword in col.lower() for keyword in ['mv', 'voltage', 'differential', 'unnamed']):
                        voltage_col = col
                        break
            
            if voltage_col is None:
                # Use the first numeric column
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        voltage_col = col
                        break
            
            if voltage_col is None:
                return None, 0
            
            # Extract voltage signal
            voltage = df[voltage_col].values
            
            # Remove any NaN values
            voltage = voltage[~np.isnan(voltage)]
            
            if len(voltage) == 0:
                return None, 0
            
            # Calculate electrical metrics
            signals = {
                'voltage': voltage,
                'voltage_gradient': np.gradient(voltage),
                'voltage_acceleration': np.gradient(np.gradient(voltage)),
                'voltage_envelope': np.abs(voltage),
                'voltage_filtered': self._bandpass_filter(voltage, 0.1, 10.0)
            }
            
            return signals, len(voltage)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            return None, 0
    
    def extract_moisture_data(self, csv_file):
        """Extract moisture data which may correlate with electrical activity"""
        try:
            df = pd.read_csv(csv_file)
            
            # Find moisture column
            moisture_col = None
            for col in df.columns:
                if isinstance(col, str) and any(keyword in col.lower() for keyword in ['moisture', 'water', 'content']):
                    moisture_col = col
                    break
            
            if moisture_col is None:
                # Use the last numeric column (usually the moisture value)
                for col in reversed(df.columns):
                    if pd.api.types.is_numeric_dtype(df[col]):
                        moisture_col = col
                        break
            
            if moisture_col is None:
                return None, 0
            
            moisture = df[moisture_col].values
            moisture = moisture[~np.isnan(moisture)]
            
            if len(moisture) == 0:
                return None, 0
            
            signals = {
                'moisture': moisture,
                'moisture_gradient': np.gradient(moisture),
                'moisture_acceleration': np.gradient(np.gradient(moisture))
            }
            
            return signals, len(moisture)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            return None, 0
    
    def _bandpass_filter(self, signal, low_freq, high_freq, sample_rate=1.0):
        """Simple bandpass filter for electrical signals"""
        try:
            from scipy import signal as scipy_signal
            
            # Design bandpass filter
            nyquist = sample_rate / 2
            low = low_freq / nyquist
            high = high_freq / nyquist
            
            if low >= high:
                return signal
            
            b, a = scipy_signal.butter(4, [low, high], btype='band')
            filtered = scipy_signal.filtfilt(b, a, signal)
            return filtered
        except:
            return signal
    
    def calculate_electrical_metrics(self, signals):
        """Calculate comprehensive electrical activity metrics"""
        metrics = {}
        
        for signal_name, signal_data in signals.items():
            if len(signal_data) == 0:
                continue
                
            # Basic statistics
            metrics[f"{signal_name}_mean"] = float(np.mean(signal_data))
            metrics[f"{signal_name}_std"] = float(np.std(signal_data))
            metrics[f"{signal_name}_min"] = float(np.min(signal_data))
            metrics[f"{signal_name}_max"] = float(np.max(signal_data))
            metrics[f"{signal_name}_range"] = float(np.max(signal_data) - np.min(signal_data))
            
            # Signal power
            metrics[f"{signal_name}_power"] = float(np.mean(signal_data**2))
            metrics[f"{signal_name}_rms"] = float(np.sqrt(np.mean(signal_data**2)))
            
            # Zero crossings (spike-like activity)
            zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)
            metrics[f"{signal_name}_zero_crossings"] = int(zero_crossings)
            metrics[f"{signal_name}_zero_crossing_rate"] = float(zero_crossings / len(signal_data))
            
            # Peak detection
            peaks = self._find_peaks(signal_data)
            metrics[f"{signal_name}_peak_count"] = len(peaks)
            metrics[f"{signal_name}_peak_rate"] = len(peaks) / len(signal_data)
            
            if len(peaks) > 0:
                metrics[f"{signal_name}_peak_amplitude_mean"] = float(np.mean(np.abs(signal_data[peaks])))
                metrics[f"{signal_name}_peak_amplitude_std"] = float(np.std(np.abs(signal_data[peaks])))
            
            # Spectral analysis
            try:
                fft = np.fft.fft(signal_data)
                power_spectrum = np.abs(fft)**2
                dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                dominant_freq = dominant_freq_idx / len(signal_data)
                metrics[f"{signal_name}_dominant_frequency"] = float(dominant_freq)
                metrics[f"{signal_name}_spectral_power"] = float(np.sum(power_spectrum))
            except:
                metrics[f"{signal_name}_dominant_frequency"] = 0.0
                metrics[f"{signal_name}_spectral_power"] = 0.0
        
        return metrics
    
    def _find_peaks(self, signal, threshold_factor=2.0):
        """Find peaks in signal above threshold"""
        threshold = threshold_factor * np.std(signal)
        peaks = []
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
                peaks.append(i)
        
        return peaks
    
    def process_all_data(self):
        """Process all available CSV files and extract electrical activity data"""
        print("Extracting electrical activity data from all CSV files...")
        
        # Process coordinate data
        csv_data_dir = self.base_path / "csv_data"
        if csv_data_dir.exists():
            print(f"\nProcessing coordinate data from {csv_data_dir}...")
            coordinate_files = list(csv_data_dir.glob("*.csv"))
            
            for file_path in coordinate_files:
                print(f"Processing {file_path.name}...")
                signals, sample_count = self.extract_coordinate_electrical_data(file_path)
                
                if signals and sample_count > 0:
                    metrics = self.calculate_electrical_metrics(signals)
                    metrics['file_type'] = 'coordinate'
                    metrics['sample_count'] = sample_count
                    metrics['file_name'] = file_path.name
                    
                    self.results[file_path.stem] = metrics
        
        # Process direct electrical recordings
        recordings_dir = self.base_path / "15061491"
        if recordings_dir.exists():
            print(f"\nProcessing direct electrical recordings from {recordings_dir}...")
            
            # Process main recordings
            for file_path in recordings_dir.glob("*.csv"):
                if file_path.stat().st_size < 50 * 1024 * 1024:  # Skip files > 50MB for now
                    print(f"Processing {file_path.name}...")
                    
                    # Try as direct electrical data first
                    signals, sample_count = self.extract_direct_electrical_data(file_path)
                    
                    if signals and sample_count > 0:
                        metrics = self.calculate_electrical_metrics(signals)
                        metrics['file_type'] = 'direct_electrical'
                        metrics['sample_count'] = sample_count
                        metrics['file_name'] = file_path.name
                        
                        self.results[file_path.stem] = metrics
                    else:
                        # Try as moisture data
                        signals, sample_count = self.extract_moisture_data(file_path)
                        
                        if signals and sample_count > 0:
                            metrics = self.calculate_electrical_metrics(signals)
                            metrics['file_type'] = 'moisture'
                            metrics['sample_count'] = sample_count
                            metrics['file_name'] = file_path.name
                            
                            self.results[file_path.stem] = metrics
            
            # Process fungal spikes recordings
            spikes_dir = recordings_dir / "fungal_spikes" / "good_recordings"
            if spikes_dir.exists():
                print(f"\nProcessing fungal spike recordings from {spikes_dir}...")
                
                for file_path in spikes_dir.glob("*.csv"):
                    if file_path.stat().st_size < 50 * 1024 * 1024:  # Skip very large files
                        print(f"Processing {file_path.name}...")
                        
                        signals, sample_count = self.extract_direct_electrical_data(file_path)
                        
                        if signals and sample_count > 0:
                            metrics = self.calculate_electrical_metrics(signals)
                            metrics['file_type'] = 'fungal_spike'
                            metrics['sample_count'] = sample_count
                            metrics['file_name'] = file_path.name
                            
                            self.results[file_path.stem] = metrics
        
        print(f"\nProcessed {len(self.results)} files successfully!")
        return self.results
    
    def save_results(self, output_file="electrical_activity_results.json"):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"electrical_activity_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        return output_file
    
    def generate_summary_report(self):
        """Generate a summary report of the electrical activity data"""
        if not self.results:
            print("No results to summarize!")
            return
        
        print("\n" + "="*60)
        print("ELECTRICAL ACTIVITY DATA SUMMARY REPORT")
        print("="*60)
        
        # Group by file type
        by_type = {}
        for file_name, data in self.results.items():
            file_type = data.get('file_type', 'unknown')
            if file_type not in by_type:
                by_type[file_type] = []
            by_type[file_type].append((file_name, data))
        
        for file_type, files in by_type.items():
            print(f"\n{file_type.upper()} FILES ({len(files)} files):")
            print("-" * 40)
            
            for file_name, data in files:
                sample_count = data.get('sample_count', 0)
                print(f"  {file_name}: {sample_count:,} samples")
                
                # Show key metrics
                if 'voltage' in data:
                    voltage_metrics = {k: v for k, v in data.items() if 'voltage' in k}
                    if voltage_metrics:
                        print(f"    Voltage RMS: {data.get('voltage_rms', 0):.4f}")
                        print(f"    Peak Rate: {data.get('voltage_peak_rate', 0):.4f}")
                        print(f"    Dominant Freq: {data.get('voltage_dominant_frequency', 0):.4f}")
                
                if 'distance' in data:
                    distance_metrics = {k: v for k, v in data.items() if 'distance' in k}
                    if distance_metrics:
                        print(f"    Distance Range: {data.get('distance_range', 0):.4f}")
                        print(f"    Velocity RMS: {data.get('velocity_rms', 0):.4f}")
        
        # Overall statistics
        print(f"\nOVERALL STATISTICS:")
        print("-" * 40)
        print(f"Total files processed: {len(self.results)}")
        print(f"Total samples across all files: {sum(data.get('sample_count', 0) for data in self.results.values()):,}")
        
        # Find files with highest electrical activity
        if any('voltage' in data for data in self.results.values()):
            voltage_files = [(name, data) for name, data in self.results.items() 
                           if 'voltage_rms' in data]
            if voltage_files:
                voltage_files.sort(key=lambda x: x[1].get('voltage_rms', 0), reverse=True)
                print(f"\nTop 5 files by voltage activity:")
                for i, (name, data) in enumerate(voltage_files[:5]):
                    print(f"  {i+1}. {name}: RMS={data.get('voltage_rms', 0):.4f}")

def main():
    """Main function to run the electrical activity extraction"""
    extractor = ElectricalActivityExtractor()
    
    # Process all data
    results = extractor.process_all_data()
    
    # Save results
    output_file = extractor.save_results()
    
    # Generate summary report
    extractor.generate_summary_report()
    
    print(f"\nElectrical activity extraction complete!")
    print(f"Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main() 