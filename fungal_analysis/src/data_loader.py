import pandas as pd
import numpy as np
from pathlib import Path
import re

class FungalDataLoader:
    """Load and preprocess fungal electrical data from various formats."""
    
    @staticmethod
    def detect_format(fp: Path) -> str:
        """Detect the format of the input file."""
        # Read first few lines
        with open(fp, 'r') as f:
            header = [next(f) for _ in range(5)]
        
        # Check for SigView format (single column of numbers)
        try:
            all(float(line.strip()) for line in header)
            return 'sigview'
        except ValueError:
            pass
            
        # Check for moisture logger format
        if any('moisture' in line.lower() or 'humidity' in line.lower() for line in header):
            return 'moisture'
            
        # Check for multi-channel format
        if any('differential' in line.lower() for line in header):
            return 'multi_channel'
            
        # Default to standard CSV format
        return 'standard'
    
    @staticmethod
    def convert_sigview(fp: Path) -> pd.DataFrame:
        """Convert SigView single-column format to standard format."""
        # Read raw values
        values = pd.read_csv(fp, header=None, names=['voltage'])
        
        # Create time column (assuming 1 second sampling)
        values['time'] = np.arange(len(values)) / 1.0
        
        return values[['time', 'voltage']]
    
    @staticmethod
    def convert_moisture(fp: Path) -> pd.DataFrame:
        """Convert moisture logger format to standard format."""
        # Skip metadata rows until we find the header
        skiprows = 0
        with open(fp, 'r') as f:
            for i, line in enumerate(f):
                if any(x in line.lower() for x in ['time', 'datetime', 'timestamp']):
                    skiprows = i
                    break
        
        # Read data with proper header row
        df = pd.read_csv(fp, skiprows=skiprows)
        
        # Find time and measurement columns
        time_col = next(col for col in df.columns if any(x in col.lower() for x in ['time', 'datetime', 'timestamp']))
        data_col = next(col for col in df.columns if any(x in col.lower() for x in ['moisture', 'humidity', 'measurement']))
        
        # Convert to standard format
        df = df[[time_col, data_col]]
        df.columns = ['time', 'voltage']
        
        return df
        
    @staticmethod
    def convert_multi_channel(fp: Path) -> pd.DataFrame:
        """Convert multi-channel differential format to standard format."""
        # Read data
        df = pd.read_csv(fp)
        
        # Convert time column if it exists
        if 'time' not in df.columns and df.columns[0] == '':
            # Convert time strings to seconds
            time_strings = df.iloc[:, 0]
            seconds = []
            for time_str in time_strings:
                try:
                    h, m, s = map(int, time_str.split(':'))
                    seconds.append(h * 3600 + m * 60 + s)
                except:
                    seconds.append(np.nan)
            df['time'] = seconds
        
        # Find voltage columns (differential measurements)
        voltage_cols = [col for col in df.columns if 'differential' in col.lower()]
        
        # If no voltage columns found, try other patterns
        if not voltage_cols:
            voltage_cols = [col for col in df.columns if any(x in col.lower() for x in ['mv', 'volt', 'v)', 'signal'])]
        
        if not voltage_cols:
            # Use all numeric columns except time
            voltage_cols = [col for col in df.columns if col != 'time' and df[col].dtype in ['float64', 'int64']]
        
        # Combine channels into a single voltage series (mean of absolute values)
        df['voltage'] = df[voltage_cols].abs().mean(axis=1)
        
        return df[['time', 'voltage']]
    
    @staticmethod
    def load_data(fp: Path) -> pd.DataFrame:
        """Load data from file, detecting and converting format as needed."""
        format_type = FungalDataLoader.detect_format(fp)
        
        if format_type == 'sigview':
            return FungalDataLoader.convert_sigview(fp)
        elif format_type == 'moisture':
            return FungalDataLoader.convert_moisture(fp)
        elif format_type == 'multi_channel':
            return FungalDataLoader.convert_multi_channel(fp)
        else:
            # Standard CSV format - find appropriate columns
            df = pd.read_csv(fp)
            
            # Find voltage/signal column
            voltage_col = None
            for col in df.columns:
                if any(k in col.lower() for k in ['mv', 'volt', 'v)', 'signal', 'potential']):
                    voltage_col = col
                    break
            if voltage_col is None:
                voltage_col = df.columns[1]  # Fallback to second column
                
            # Find or create time column
            if 'time' in df.columns:
                time_col = 'time'
            else:
                # Create time column based on sampling rate (default 1 Hz)
                df['time'] = np.arange(len(df)) / 1.0
                time_col = 'time'
            
            return df[[time_col, voltage_col]].rename(columns={voltage_col: 'voltage'}) 