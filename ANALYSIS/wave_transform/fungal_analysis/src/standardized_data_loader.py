import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass

class StandardizedDataLoader:
    """
    Standardized data loader with rigorous validation and error handling.
    
    Features:
    - Robust CSV parsing with multiple decimal separator handling
    - Automatic sampling rate detection
    - Signal quality validation
    - Comprehensive metadata extraction
    - Error propagation
    - Detailed logging
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.metadata_cache = {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup detailed logging for data loading process."""
        log_dir = self.data_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Add file handler
        fh = logging.FileHandler(log_dir / f"data_loading_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    def _validate_signal(self, signal: np.ndarray) -> Tuple[bool, Dict]:
        """
        Validate signal quality and characteristics.
        
        Returns:
        - bool: Whether signal passes validation
        - dict: Validation metrics
        """
        metrics = {
            'length': len(signal),
            'mean': np.mean(signal),
            'std': np.std(signal),
            'nan_count': np.isnan(signal).sum(),
            'inf_count': np.isinf(signal).sum(),
            'min': np.nanmin(signal),
            'max': np.nanmax(signal),
            'zero_runs': len([x for x in np.split(signal, np.where(np.diff(signal != 0))[0] + 1) 
                            if len(x) > 1 and (x == 0).all()]),
        }
        
        # Define validation criteria
        valid = (
            metrics['length'] >= 100 and  # Minimum length for analysis
            metrics['nan_count'] == 0 and  # No NaN values
            metrics['inf_count'] == 0 and  # No infinite values
            metrics['std'] > 0 and  # Non-zero variance
            metrics['zero_runs'] < metrics['length'] * 0.1  # Less than 10% constant runs
        )
        
        return valid, metrics
    
    def _detect_sampling_rate(self, df: pd.DataFrame, filename: str) -> float:
        """
        Detect sampling rate using multiple methods.
        
        1. From time column if present
        2. From filename indicators
        3. From data characteristics
        """
        # Method 1: From time column
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        if time_cols:
            time_data = pd.to_numeric(df[time_cols[0]], errors='coerce')
            if not time_data.isna().all():
                time_diff = np.diff(time_data.dropna())
                if len(time_diff) > 0:
                    median_diff = np.median(time_diff)
                    if median_diff > 0:
                        return 1.0 / median_diff
        
        # Method 2: From filename
        if 'second' in filename.lower():
            return 1.0
        if any(x in filename.lower() for x in ['10hz', '10_hz']):
            return 10.0
            
        # Method 3: Data characteristics (default to 10 Hz if uncertain)
        logger.warning(f"Could not detect sampling rate for {filename}, using default 10 Hz")
        return 10.0
    
    def _extract_metadata(self, filepath: Path, df: pd.DataFrame) -> Dict:
        """Extract and validate metadata from file and contents."""
        metadata = {
            'filename': filepath.name,
            'timestamp': datetime.fromtimestamp(filepath.stat().st_mtime),
            'size_bytes': filepath.stat().st_size,
            'columns': list(df.columns),
        }
        
        # Extract experiment type from filename
        exp_indicators = {
            'moisture': ['moisture', 'humidity'],
            'electrical': ['mv', 'voltage', 'potential'],
            'growth': ['growth', 'colony'],
            'network': ['network', 'mycelium']
        }
        
        for exp_type, indicators in exp_indicators.items():
            if any(ind in filepath.name.lower() for ind in indicators):
                metadata['experiment_type'] = exp_type
                break
        else:
            metadata['experiment_type'] = 'unknown'
            
        return metadata
    
    def load_csv(self, filepath: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Load and validate CSV data with comprehensive error handling.
        
        Returns:
        - numpy.ndarray: Validated signal data
        - dict: Metadata including validation metrics
        """
        filepath = Path(filepath)
        logger.info(f"Loading {filepath.name}")
        
        try:
            # Try reading with different decimal separators
            try:
                df = pd.read_csv(filepath)
            except:
                # Try with comma decimal separator
                df = pd.read_csv(filepath, decimal=',')
            
            # Find data column
            data_col = None
            for col in df.columns:
                if any(term in str(col).lower() for term in 
                      ['mv', 'v)', 'voltage', 'differential', 'potential']):
                    data_col = col
                    break
            
            if not data_col:
                # Try to identify numeric column
                for col in df.columns[1:]:  # Skip first column (usually time)
                    numeric = pd.to_numeric(df[col], errors='coerce')
                    if numeric.notna().sum() > len(df) * 0.5:  # >50% numeric
                        data_col = col
                        break
            
            if not data_col:
                raise DataValidationError("No suitable data column found")
            
            # Convert to numeric with comprehensive error handling
            signal = pd.to_numeric(df[data_col], errors='coerce')
            
            # Basic cleaning
            signal = signal.dropna()
            if len(signal) < 100:
                raise DataValidationError("Signal too short after cleaning")
            
            # Validate signal
            valid, metrics = self._validate_signal(signal.values)
            if not valid:
                raise DataValidationError(f"Signal validation failed: {metrics}")
            
            # Extract metadata
            metadata = self._extract_metadata(filepath, df)
            metadata.update({
                'sampling_rate': self._detect_sampling_rate(df, filepath.name),
                'validation_metrics': metrics
            })
            
            # Cache metadata
            self.metadata_cache[str(filepath)] = metadata
            
            # Save validation report
            report_path = filepath.parent / "validation_reports"
            report_path.mkdir(exist_ok=True)
            with open(report_path / f"{filepath.stem}_validation.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return signal.values, metadata
            
        except Exception as e:
            logger.error(f"Error loading {filepath.name}: {str(e)}")
            return None, None
    
    def load_directory(self, pattern: str = "*.csv") -> Dict[str, Tuple[np.ndarray, Dict]]:
        """Load all matching files in directory with validation."""
        results = {}
        for filepath in self.data_dir.rglob(pattern):
            signal, metadata = self.load_csv(filepath)
            if signal is not None:
                results[str(filepath)] = (signal, metadata)
        return results 