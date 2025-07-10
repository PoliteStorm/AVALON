import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class ExperimentMetadata:
    species: str
    condition: str
    duration: str
    replicate: int
    electrode_type: str

class EnhancedAnalysis:
    """Enhanced analysis incorporating multiple species and conditions"""
    
    def __init__(self):
        self.species_codes = {
            'Pv': 'Pleurotus',
            'Rb': 'Reishi',
            'Pi': 'Pioppino',
            'Pp': 'Unknown_Species'
        }
        
    def parse_filename(self, filename: str) -> ExperimentMetadata:
        """Parse metadata from standardized filename format"""
        parts = filename.split('_')
        
        # Handle coordinate files differently
        if 'coordinates' in filename:
            # Extract replicate number from before 'coordinates'
            replicate_str = parts[-2] if len(parts) > 1 else '1'
            try:
                replicate = int(replicate_str)
            except ValueError:
                replicate = 1
        else:
            # Standard filename format
            try:
                replicate = int(parts[-1].split('.')[0])
            except (ValueError, IndexError):
                replicate = 1
        
        return ExperimentMetadata(
            species=self.species_codes.get(parts[0], 'Unknown'),
            condition='_'.join([p for p in parts if any(x in p for x in ['I', 'Fc', 'U'])]),
            duration=next((p for p in parts if any(x in p for x in ['d', 'h'])), 'Unknown'),
            replicate=replicate,
            electrode_type='_'.join([p for p in parts if p in ['Fc', 'U', 'Hf']])
        )
    
    def comparative_analysis(self, base_dir: Path, output_dir: Path):
        """Perform comparative analysis across species and conditions"""
        results = []
        
        # Process each species group
        for species in self.species_codes.keys():
            species_files = list(base_dir.glob(f"{species}*.csv"))
            for fp in species_files:
                metadata = self.parse_filename(fp.name)
                data = pd.read_csv(fp)
                
                # Calculate key metrics
                metrics = self.calculate_metrics(data)
                results.append({
                    'species': metadata.species,
                    'condition': metadata.condition,
                    'duration': metadata.duration,
                    **metrics
                })
        
        # Generate comparative visualizations
        self.plot_comparative_results(results, output_dir)
    
    def calculate_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate key metrics for comparison"""
        return {
            'signal_complexity': self.calculate_signal_complexity(data),
            'response_latency': self.calculate_response_latency(data),
            'pattern_consistency': self.calculate_pattern_consistency(data)
        }
    
    def calculate_signal_complexity(self, data: pd.DataFrame) -> float:
        """Calculate signal complexity using wavelet entropy"""
        if len(data) < 10:
            return 0.0
        
        # Use first numeric column as signal
        signal_col = None
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                signal_col = col
                break
        
        if signal_col is None:
            return 0.0
            
        signal = data[signal_col].values
        # Simple complexity measure: coefficient of variation
        return np.std(signal) / (np.mean(np.abs(signal)) + 1e-10)
    
    def calculate_response_latency(self, data: pd.DataFrame) -> float:
        """Calculate response latency to environmental changes"""
        if len(data) < 10:
            return 0.0
        
        # Simple measure: time to first significant change
        signal_col = None
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                signal_col = col
                break
        
        if signal_col is None:
            return 0.0
            
        signal = data[signal_col].values
        baseline = np.mean(signal[:min(100, len(signal)//4)])
        threshold = baseline + 2 * np.std(signal[:min(100, len(signal)//4)])
        
        for i, val in enumerate(signal):
            if abs(val - baseline) > threshold:
                return i
        return len(signal)
    
    def calculate_pattern_consistency(self, data: pd.DataFrame) -> float:
        """Calculate consistency of patterns across time"""
        if len(data) < 10:
            return 0.0
        
        # Simple measure: autocorrelation at lag 1
        signal_col = None
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                signal_col = col
                break
        
        if signal_col is None:
            return 0.0
            
        signal = data[signal_col].values
        if len(signal) < 2:
            return 0.0
        
        return np.corrcoef(signal[:-1], signal[1:])[0, 1] if len(signal) > 1 else 0.0
    
    def plot_comparative_results(self, results: List[Dict], output_dir: Path):
        """Generate comparative visualizations"""
        if not results:
            print("No results to plot")
            return
            
        df = pd.DataFrame(results)
        
        # Filter out invalid data
        df = df.dropna()
        if len(df) == 0:
            print("No valid data to plot")
            return
        
        # Species comparison plot
        if 'signal_complexity' in df.columns and len(df) > 0:
            plt.figure(figsize=(12, 8))
            try:
                sns.boxplot(data=df, x='species', y='signal_complexity')
                plt.title('Signal Complexity Across Species')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / 'species_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("✓ Species comparison plot saved")
            except Exception as e:
                print(f"Warning: Could not create species comparison plot: {e}")
                plt.close()
        
        # Time evolution plot
        if 'pattern_consistency' in df.columns and len(df) > 0:
            plt.figure(figsize=(12, 8))
            try:
                sns.scatterplot(data=df, x='duration', y='pattern_consistency', 
                               hue='species', style='condition')
                plt.title('Pattern Consistency Over Time')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / 'time_evolution.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("✓ Time evolution plot saved")
            except Exception as e:
                print(f"Warning: Could not create time evolution plot: {e}")
                plt.close()
        
        # Condition response plot
        if 'response_latency' in df.columns and len(df) > 0:
            plt.figure(figsize=(12, 8))
            try:
                sns.barplot(data=df, x='condition', y='response_latency', 
                           hue='species')
                plt.title('Response Latency by Condition')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(output_dir / 'condition_response.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("✓ Condition response plot saved")
            except Exception as e:
                print(f"Warning: Could not create condition response plot: {e}")
                plt.close() 