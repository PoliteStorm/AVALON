import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

class FileAnalyzer:
    def __init__(self, data_dir: str = "/home/kronos/AVALON/15061491"):
        self.data_dir = Path(data_dir)
        
    def _convert_to_numeric(self, series):
        """Convert a series to numeric, handling various formats"""
        # Try direct conversion first
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().any():
            return numeric
        
        # Try cleaning the string and converting
        cleaned = series.str.replace(',', '.').str.extract('([-+]?\d*\.?\d+)')[0]
        return pd.to_numeric(cleaned, errors='coerce')
        
    def analyze_file(self, filename: str) -> Dict:
        """
        Analyze a single CSV file for potential spiking activity
        
        Args:
            filename: Name of the file to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        file_path = self.data_dir / filename
        try:
            # Read the CSV file with all columns as string initially
            df = pd.read_csv(file_path, dtype=str, low_memory=False)
            
            # Try to identify the data column
            potential_cols = []
            for col in df.columns:
                if any(term in str(col).lower() for term in ['mv', 'voltage', 'signal', 'potential', 'differential', 'v)']):
                    potential_cols.append(col)
            
            # If no voltage columns found, try numerical columns
            if not potential_cols:
                for col in df.columns[1:]:  # Skip first column as it's often time/index
                    numeric_data = self._convert_to_numeric(df[col])
                    if numeric_data.notna().sum() > len(df) * 0.5:  # More than 50% numeric
                        potential_cols.append(col)
            
            # If still no columns found, use the second column or first as fallback
            if not potential_cols and len(df.columns) > 1:
                potential_cols = [df.columns[1]]
            elif not potential_cols:
                potential_cols = [df.columns[0]]
            
            # Use the first identified column
            data_col = potential_cols[0]
            signal = self._convert_to_numeric(df[data_col])
            
            # Remove NaN values
            signal = signal.dropna().values
            
            if len(signal) < 2:
                raise ValueError("No valid numerical data found")
            
            # Basic statistics
            mean = np.mean(signal)
            std = np.std(signal)
            
            # Calculate rapid changes (potential spikes)
            diff = np.diff(signal)
            std_diff = np.std(diff)
            rapid_changes = np.sum(np.abs(diff) > (3 * std_diff))
            
            # Estimate noise level using median absolute deviation
            noise_level = np.median(np.abs(diff - np.median(diff))) * 1.4826
            
            # Calculate signal-to-noise ratio
            signal_range = np.percentile(signal, 95) - np.percentile(signal, 5)
            snr = signal_range / noise_level if noise_level > 0 else 0
            
            # Detect periodic components
            if len(signal) > 100:
                fft = np.abs(np.fft.fft(signal - np.mean(signal)))
                freqs = np.fft.fftfreq(len(signal))
                dominant_freq = np.abs(freqs[np.argmax(fft[1:]) + 1])
                has_periodicity = np.max(fft[1:]) > np.mean(fft[1:]) * 5
            else:
                dominant_freq = 0
                has_periodicity = False
            
            return {
                'filename': filename,
                'samples': len(signal),
                'mean': mean,
                'std': std,
                'rapid_changes': rapid_changes,
                'rapid_changes_ratio': rapid_changes / len(signal),
                'noise_level': noise_level,
                'snr': snr,
                'data_column': data_col,
                'has_periodicity': has_periodicity,
                'dominant_freq': dominant_freq,
                'has_spikes': (rapid_changes / len(signal) > 0.001 and snr > 2.0) or 
                             (snr > 10 and rapid_changes > 10),
                'error': None
            }
            
        except Exception as e:
            return {
                'filename': filename,
                'error': str(e)
            }
    
    def analyze_all_files(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Analyze all CSV files in the directory
        
        Returns:
            Tuple of (spiking_files, non_spiking_files, error_files)
        """
        spiking_files = []
        non_spiking_files = []
        error_files = []
        
        # Get all CSV files
        csv_files = [f.name for f in self.data_dir.glob('*.csv')]
        
        print(f"\nAnalyzing {len(csv_files)} CSV files...")
        
        for filename in csv_files:
            print(f"\nAnalyzing {filename}...")
            result = self.analyze_file(filename)
            
            if 'error' in result and result['error'] is not None:
                print(f"Error: {result['error']}")
                error_files.append(filename)
                continue
                
            print(f"Samples: {result['samples']}")
            print(f"Data column: {result['data_column']}")
            print(f"Signal-to-noise ratio: {result['snr']:.2f}")
            print(f"Rapid changes: {result['rapid_changes']} ({result['rapid_changes_ratio']*100:.2f}%)")
            if result['has_periodicity']:
                print(f"Dominant frequency: {result['dominant_freq']:.4f}")
            
            if result['has_spikes']:
                print("Status: Contains spiking activity")
                spiking_files.append(filename)
            else:
                print("Status: No significant spiking activity")
                non_spiking_files.append(filename)
                
        return spiking_files, non_spiking_files, error_files
        
def main():
    analyzer = FileAnalyzer()
    spiking, non_spiking, errors = analyzer.analyze_all_files()
    
    print("\n=== ANALYSIS SUMMARY ===")
    
    print("\nFiles with spiking activity:")
    for f in sorted(spiking):
        print(f"- {f}")
        
    print("\nFiles without significant spiking activity:")
    for f in sorted(non_spiking):
        print(f"- {f}")
        
    print("\nFiles with errors:")
    for f in sorted(errors):
        print(f"- {f}")
        
    # Save results to files
    with open('spiking_files.txt', 'w') as f:
        f.write('\n'.join(sorted(spiking)))
        
    with open('non_spiking_files.txt', 'w') as f:
        f.write('\n'.join(sorted(non_spiking)))
        
    with open('error_files.txt', 'w') as f:
        f.write('\n'.join(sorted(errors)))
        
if __name__ == "__main__":
    main() 