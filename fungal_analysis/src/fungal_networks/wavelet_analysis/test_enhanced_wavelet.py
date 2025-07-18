"""
Test script for enhanced wavelet analysis framework
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import os
import pandas as pd
from enhanced_wavelet import EnhancedWaveletAnalysis
from typing import Dict, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_mat_file(filepath: str) -> Optional[np.ndarray]:
    """Load .mat file and extract relevant data"""
    try:
        mat_data = io.loadmat(filepath)
        # Extract the signal data - try common keys
        signal_data = None
        potential_keys = ['data', 'signal', 'timeseries', 'voltage', 'potential']
        
        for key in potential_keys:
            if key in mat_data and isinstance(mat_data[key], np.ndarray):
                signal_data = mat_data[key]
                break
                
        if signal_data is None:
            # Try any array that could be a signal
            for key in mat_data.keys():
                if isinstance(mat_data[key], np.ndarray) and len(mat_data[key].shape) >= 1:
                    signal_data = mat_data[key]
                    break
                    
        if signal_data is None:
            logger.warning(f"No suitable data found in {filepath}")
            return None
            
        # Ensure 1D array
        if len(signal_data.shape) > 1:
            signal_data = signal_data.flatten()
            
        return signal_data
        
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return None

def load_csv_file(filepath: str) -> Optional[Dict]:
    """Load CSV file containing coordinate data"""
    try:
        df = pd.read_csv(filepath)
        # Assuming coordinates are in columns named 'x' and 'y'
        if 'x' in df.columns and 'y' in df.columns:
            coordinates = {
                'x': df['x'].values,
                'y': df['y'].values
            }
            return coordinates
        else:
            logger.warning(f"No coordinate columns found in {filepath}")
            return None
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        return None

def extract_metadata_from_filename(filepath: str) -> Dict:
    """Extract metadata from standardized filename"""
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    
    metadata = {
        'species': parts[0],
        'scale': parts[1],
        'treatment': '_'.join(parts[2:-3]),
        'time': parts[-3],
        'replicate': parts[-2]
    }
    
    return metadata

def test_single_file(filepath: str, analyzer: EnhancedWaveletAnalysis) -> Optional[Dict]:
    """Test wavelet analysis on a single file"""
    # Extract metadata
    metadata = extract_metadata_from_filename(filepath)
    logger.info(f"Processing {os.path.basename(filepath)}")
    logger.info(f"Metadata: {metadata}")
    
    # Load data based on file type
    if filepath.endswith('.mat'):
        data = load_mat_file(filepath)
        if data is None:
            return None
            
        # Perform analysis
        results = analyzer.analyze_signal(
            data,
            detrend=True,
            denoise=True
        )
        
        # Create visualization
        fig = analyzer.plot_analysis(
            results,
            title=f'Analysis of {os.path.basename(filepath)}'
        )
        
        # Save plot
        output_path = filepath.replace('.mat', '_enhanced_wavelet.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    elif filepath.endswith('.csv'):
        data = load_csv_file(filepath)
        if data is None:
            return None
            
        # For coordinate data, we'll analyze the spatial distribution
        # Convert coordinates to a time series using distance from origin
        distances = np.sqrt(data['x']**2 + data['y']**2)
        
        # Perform analysis on the distance series
        results = analyzer.analyze_signal(
            distances,
            detrend=True,
            denoise=True
        )
        
        # Create visualization
        fig = analyzer.plot_analysis(
            results,
            title=f'Spatial Analysis of {os.path.basename(filepath)}'
        )
        
        # Save plot
        output_path = filepath.replace('.csv', '_enhanced_wavelet.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    else:
        logger.warning(f"Unsupported file type: {filepath}")
        return None
        
    # Add metadata to results
    results['metadata'] = metadata
    return results

def analyze_by_species(results: Dict) -> Dict:
    """Analyze patterns grouped by species"""
    species_patterns = {}
    
    for filepath, result in results.items():
        if result is None:
            continue
            
        species = result['metadata']['species']
        if species not in species_patterns:
            species_patterns[species] = {
                'peak_frequencies': [],
                'temporal_patterns': [],
                'power_distributions': []
            }
            
        species_patterns[species]['peak_frequencies'].append(
            len(result['patterns']['peak_frequencies'])
        )
        species_patterns[species]['temporal_patterns'].append(
            len(result['patterns']['peak_times'])
        )
        species_patterns[species]['power_distributions'].append(
            result['patterns']['power_distribution']
        )
        
    # Compute statistics for each species
    for species in species_patterns:
        species_patterns[species]['avg_peaks'] = np.mean(
            species_patterns[species]['peak_frequencies']
        )
        species_patterns[species]['avg_patterns'] = np.mean(
            species_patterns[species]['temporal_patterns']
        )
        species_patterns[species]['power_std'] = np.std(
            species_patterns[species]['power_distributions'], axis=0
        )
        
    return species_patterns

def test_batch(data_dir: str = None, max_files: int = None) -> Dict:
    """Test wavelet analysis on multiple files"""
    if data_dir is None:
        # Use the fungal_networks directory in the workspace root
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), 'fungal_networks')
        
    # Initialize analyzer
    analyzer = EnhancedWaveletAnalysis(
        wavelet_type='morl',
        sampling_rate=1.0
    )
    
    # Get list of data files
    data_files = []
    
    # Add .mat files from root directory
    mat_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                 if f.endswith('.mat')]
    data_files.extend(mat_files)
    
    # Add CSV files from csv_data directory
    csv_dir = os.path.join(data_dir, 'csv_data')
    if os.path.exists(csv_dir):
        csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir)
                    if f.endswith('.csv')]
        data_files.extend(csv_files)
    
    # Sort files to ensure reproducible order
    data_files.sort()
    
    # Limit number of files if specified
    if max_files:
        data_files = data_files[:max_files]
        
    # Process files
    results = {}
    for filepath in data_files:
        results[filepath] = test_single_file(filepath, analyzer)
        
    # Analyze patterns by species
    species_patterns = analyze_by_species(results)
    
    return results, species_patterns

if __name__ == "__main__":
    # Run batch test
    logger.info("Starting batch analysis...")
    results, species_patterns = test_batch(max_files=10)  # Start with 10 files as a test
    logger.info("Analysis complete!")
    
    # Print species-level statistics
    logger.info("\nSpecies-level Statistics:")
    for species, patterns in species_patterns.items():
        logger.info(f"\n{species}:")
        logger.info(f"- Average number of significant frequencies: {patterns['avg_peaks']:.2f}")
        logger.info(f"- Average number of temporal patterns: {patterns['avg_patterns']:.2f}")
        logger.info(f"- Power distribution std: {np.mean(patterns['power_std']):.2f}")
        
    # Print individual file results
    logger.info("\nIndividual File Results:")
    for filepath, result in results.items():
        if result is not None:
            filename = os.path.basename(filepath)
            logger.info(f"\n{filename}:")
            logger.info(f"- Detected {len(result['patterns']['peak_frequencies'])} significant frequencies")
            logger.info(f"- Found {len(result['patterns']['peak_times'])} temporal patterns") 