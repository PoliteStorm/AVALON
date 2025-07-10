import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os

class FungalSpikeAnalyzer:
    def __init__(self, data_dir: str = "/home/kronos/AVALON/15061491"):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "fungal_spikes"
        
    def _convert_to_numeric(self, series):
        """Convert a series to numeric, handling various formats"""
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().any():
            return numeric
        cleaned = series.str.replace(',', '.').str.extract('([-+]?\d*\.?\d+)')[0]
        return pd.to_numeric(cleaned, errors='coerce')
        
    def is_fungal_recording(self, filename: str, data: pd.DataFrame) -> bool:
        """
        Determine if the file is a fungal recording rather than environmental data
        """
        # Check filename indicators
        env_indicators = ['moisture_logger', 'temperature', 'humidity']  # Allow some moisture files
        if any(ind.lower() in filename.lower() for ind in env_indicators):
            return False
            
        # Check column names
        voltage_indicators = ['mv', 'v)', 'voltage', 'differential', 'potential']
        has_voltage_col = any(any(ind in str(col).lower() for ind in voltage_indicators) 
                            for col in data.columns)
                            
        # Special cases - known fungal recordings
        fungal_indicators = ['oyster', 'hericium', 'spray', 'ch1-2', 'differential']
        is_known_fungal = any(ind.lower() in filename.lower() for ind in fungal_indicators)
        
        return has_voltage_col or is_known_fungal
        
    def analyze_file(self, filename: str) -> dict:
        """Analyze a single file for fungal spiking activity"""
        file_path = self.data_dir / filename
        try:
            # Read data
            df = pd.read_csv(file_path, dtype=str, low_memory=False)
            
            # Check if it's a fungal recording
            if not self.is_fungal_recording(filename, df):
                return {'filename': filename, 'is_fungal': False}
            
            # Find voltage column
            voltage_col = None
            for col in df.columns:
                if any(term in str(col).lower() for term in ['mv', 'v)', 'voltage', 'differential', 'potential']):
                    voltage_col = col
                    break
            
            if not voltage_col:
                # Try to find a numeric column that might contain voltage data
                for col in df.columns[1:]:  # Skip first column as it's often time
                    numeric_data = self._convert_to_numeric(df[col])
                    if numeric_data.notna().sum() > len(df) * 0.5:  # More than 50% numeric
                        voltage_col = col
                        break
                        
            if not voltage_col and len(df.columns) > 1:
                voltage_col = df.columns[1]  # Use second column as fallback
            else:
                voltage_col = df.columns[0]
                
            # Process signal
            signal = self._convert_to_numeric(df[voltage_col])
            signal = signal.dropna().values
            
            if len(signal) < 100:  # Too short for reliable analysis
                return {'filename': filename, 'is_fungal': False}
                
            # Calculate metrics
            diff = np.diff(signal)
            std_diff = np.std(diff)
            rapid_changes = np.sum(np.abs(diff) > (3 * std_diff))
            noise_level = np.median(np.abs(diff - np.median(diff))) * 1.4826
            signal_range = np.percentile(signal, 95) - np.percentile(signal, 5)
            snr = signal_range / noise_level if noise_level > 0 else 0
            
            # Calculate additional features
            if len(signal) > 1000:
                # Look for rhythmic patterns
                fft = np.abs(np.fft.fft(signal - np.mean(signal)))
                freqs = np.fft.fftfreq(len(signal))
                dominant_freq = np.abs(freqs[np.argmax(fft[1:]) + 1])
                has_rhythm = np.max(fft[1:]) > np.mean(fft[1:]) * 5
            else:
                dominant_freq = 0
                has_rhythm = False
            
            # More flexible criteria for good recording:
            # 1. Either high SNR or clear rhythmic pattern
            # 2. Reasonable spike frequency
            # 3. Sufficient length
            is_good_recording = (
                len(signal) > 1000 and  # Long enough for analysis
                (snr > 5 or has_rhythm) and  # Either good SNR or clear rhythm
                (0.0001 < rapid_changes / len(signal) < 0.2) and  # Reasonable spike frequency
                signal_range > 0  # Has some variation
            )
            
            quality_score = (
                min(snr / 10, 10) +  # SNR contribution (capped at 10)
                (5 if has_rhythm else 0) +  # Rhythm bonus
                min(5, np.log10(len(signal)) - 2)  # Length bonus
            )
            
            return {
                'filename': filename,
                'is_fungal': True,
                'is_good_recording': is_good_recording,
                'snr': snr,
                'rapid_changes_ratio': rapid_changes / len(signal),
                'samples': len(signal),
                'voltage_column': voltage_col,
                'has_rhythm': has_rhythm,
                'dominant_freq': dominant_freq,
                'quality_score': quality_score
            }
            
        except Exception as e:
            return {'filename': filename, 'is_fungal': False, 'error': str(e)}
            
    def organize_files(self):
        """Analyze all files and organize them into categories"""
        # Create output directory
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        good_dir = self.output_dir / "good_recordings"
        good_dir.mkdir(exist_ok=True)
        
        results = []
        print("\nAnalyzing files for fungal spiking activity...")
        
        for file in self.data_dir.glob('*.csv'):
            print(f"Analyzing {file.name}...")
            result = self.analyze_file(file.name)
            if result.get('is_fungal', False) and result.get('is_good_recording', False):
                results.append(result)
                # Copy good files to output directory
                shutil.copy2(file, good_dir / file.name)
                
        # Sort results by quality score
        results.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Generate report
        report = ["=== Fungal Spiking Activity Analysis ===\n"]
        report.append("High-Quality Fungal Recordings (sorted by overall quality):\n")
        
        for r in results:
            report.append(f"File: {r['filename']}")
            report.append(f"- Quality Score: {r['quality_score']:.1f}")
            report.append(f"- Signal-to-Noise Ratio: {r['snr']:.2f}")
            report.append(f"- Spike Frequency: {r['rapid_changes_ratio']*100:.2f}%")
            report.append(f"- Samples: {r['samples']}")
            report.append(f"- Data Column: {r['voltage_column']}")
            if r['has_rhythm']:
                report.append(f"- Dominant Frequency: {r['dominant_freq']:.4f} Hz")
            report.append("")
            
        # Save report
        with open(self.output_dir / "analysis_report.txt", 'w') as f:
            f.write('\n'.join(report))
            
        print(f"\nAnalysis complete! {len(results)} high-quality fungal recordings found.")
        print(f"Files have been organized in: {self.output_dir}")
        print("Check analysis_report.txt for detailed information.")
        
def main():
    analyzer = FungalSpikeAnalyzer()
    analyzer.organize_files()
    
if __name__ == "__main__":
    main() 