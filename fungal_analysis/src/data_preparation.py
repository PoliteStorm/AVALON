import pandas as pd
import numpy as np
from pathlib import Path
import shutil

def prepare_data(source_dir: str = "/home/kronos/AVALON/15061491", 
                output_dir: str = "/home/kronos/AVALON/fungal_analysis/data"):
    """
    Prepare fungal recording data for analysis.
    
    Args:
        source_dir: Directory containing original CSV files
        output_dir: Directory to save processed numpy arrays
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    def _convert_to_numeric(series):
        """Convert a series to numeric, handling various formats."""
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().any():
            return numeric
        cleaned = series.str.replace(',', '.').str.extract('([-+]?\d*\.?\d+)')[0]
        return pd.to_numeric(cleaned, errors='coerce')
    
    def process_file(filepath):
        """Process a single CSV file."""
        try:
            # Read data
            df = pd.read_csv(filepath, dtype=str)
            
            # Find voltage column
            voltage_col = None
            for col in df.columns:
                if any(term in str(col).lower() for term in 
                      ['mv', 'v)', 'voltage', 'differential', 'potential']):
                    voltage_col = col
                    break
            
            if not voltage_col and len(df.columns) > 1:
                voltage_col = df.columns[1]  # Use second column as fallback
            else:
                voltage_col = df.columns[0]
            
            # Convert to numeric and clean
            signal = _convert_to_numeric(df[voltage_col])
            signal = signal.dropna().values
            
            if len(signal) < 100:  # Too short for analysis
                return None
                
            # Normalize
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
            
            return signal
            
        except Exception as e:
            print(f"Error processing {filepath.name}: {str(e)}")
            return None
    
    # Process all CSV files in source directory and its subdirectories
    processed_count = 0
    for filepath in source_path.rglob('*.csv'):
        # Skip files that look like environmental data
        if any(term in filepath.name.lower() for term in 
               ['moisture_logger', 'temperature', 'humidity']):
            continue
            
        # Process if it looks like a fungal recording
        if any(term in filepath.name.lower() for term in 
               ['oyster', 'hericium', 'spray', 'ch1-2', 'differential']):
            signal = process_file(filepath)
            
            if signal is not None:
                # Save as numpy array
                output_file = output_path / f"{filepath.stem}.npy"
                np.save(output_file, signal)
                processed_count += 1
                print(f"Processed: {filepath.name} → {output_file.name}")
    
    print(f"\nTotal files processed: {processed_count}")

if __name__ == '__main__':
    prepare_data() 