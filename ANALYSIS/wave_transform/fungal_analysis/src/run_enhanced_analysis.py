"""
Run enhanced analysis on fungal electrical signal datasets.
"""

import numpy as np
from pathlib import Path
import json
from enhanced_analysis import EnhancedAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm
import psutil
import time
import sys

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_resources():
    """Check if system has enough resources to proceed."""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    logger.info(f"Available memory: {memory.available / 1024 / 1024:.1f} MB")
    logger.info(f"Current CPU usage: {cpu_percent}%")
    
    if memory.available < 500 * 1024 * 1024:  # Less than 500MB available
        logger.warning("Low memory available. Analysis may be slow or fail.")
        return False
        
    if cpu_percent > 90:
        logger.warning("High CPU usage. Analysis may be slow.")
        return False
        
    return True

def plot_linguistic_results(results: dict, output_dir: Path, filename: str):
    """Plot linguistic analysis results."""
    try:
        plt.figure(figsize=(12, 8))  # Reduced figure size
        
        for theta in ['theta_1.0', 'theta_2.0']:
            dist = results['linguistic'][theta]['word_stats']['word_length_distribution']
            plt.subplot(2, 2, 1 if theta == 'theta_1.0' else 2)
            plt.bar(dist.keys(), dist.values(), alpha=0.6, 
                   label=f'θ = {theta.split("_")[1]}')
            plt.title(f'Word Length Distribution ({theta})')
            plt.xlabel('Word Length')
            plt.ylabel('Probability')
            plt.legend()
            
            # Transition matrix heatmap
            trans_matrix = np.array(results['linguistic'][theta]['syntax']['transition_matrix'])
            if trans_matrix.size > 0:
                plt.subplot(2, 2, 3 if theta == 'theta_1.0' else 4)
                sns.heatmap(trans_matrix, cmap='viridis', 
                           xticklabels=False, yticklabels=False)
                plt.title(f'Word Transitions ({theta})')
                
        plt.tight_layout()
        plt.savefig(output_dir / f'{filename}_linguistic.png', dpi=100)  # Reduced DPI
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting results for {filename}: {str(e)}")

def analyze_recording(filepath: Path, output_dir: Path) -> bool:
    """Analyze a single recording with enhanced analysis including linguistics."""
    try:
        # Load data
        data = np.load(filepath)
        logger.info(f"\nAnalyzing recording: {filepath.name}")
        logger.info(f"Signal length: {len(data)} samples")
        
        # Check file size
        if len(data) > 1000000:  # Very large file
            logger.warning("Large file detected. Analysis may take longer.")
            
        # Initialize analyzer with conservative CPU limit
        analyzer = EnhancedAnalysis(max_cpu_percent=70.0)
        
        # Run analysis
        start_time = time.time()
        results = analyzer.comprehensive_analysis(data)
        elapsed = time.time() - start_time
        
        # Save results
        results_file = output_dir / f'{filepath.stem}_analysis.json'
        with open(results_file, 'w') as f:
            json_results = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in results.items()
            }
            json.dump(json_results, f, indent=2)
            
        # Generate plots if results are valid
        if results['spikes']['count'] > 0:
            plot_linguistic_results(results, output_dir, filepath.stem)
        
        # Print summary
        logger.info(f"\nAnalysis completed in {elapsed:.1f} seconds")
        logger.info(f"Spikes detected: {results['spikes']['count']}")
        
        for theta in ['theta_1.0', 'theta_2.0']:
            ling_results = results['linguistic'][theta]
            logger.info(f"\nLinguistic analysis (θ = {theta.split('_')[1]}):")
            logger.info(f"Average word length: {ling_results['word_stats']['avg_word_length']:.2f}")
            logger.info(f"Vocabulary size: {ling_results['word_stats']['vocabulary_size']}")
            logger.info(f"Complexity: {ling_results['complexity']['normalized_complexity']:.2f}")
            logger.info(f"Entropy: {ling_results['complexity']['entropy']:.2f}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error analyzing {filepath.name}: {str(e)}")
        return False
    
def main():
    """Run enhanced analysis on all recordings in the dataset."""
    # Check system resources
    if not check_system_resources():
        logger.warning("System resources are limited. Consider closing other applications.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    data_dir = Path('/home/kronos/AVALON/fungal_analysis/data')
    output_dir = data_dir.parent / 'enhanced_analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    # Get list of files and sort by size
    files = list(data_dir.glob('*.npy'))
    files.sort(key=lambda x: x.stat().st_size)  # Process smaller files first
    
    successful = 0
    failed = 0
    
    logger.info(f"\nFound {len(files)} files to process")
    
    for filepath in tqdm(files, desc="Processing recordings"):
        try:
            # Check resources before each file
            if not check_system_resources():
                logger.warning("System resources low. Pausing for 30 seconds...")
                time.sleep(30)
                
            if analyze_recording(filepath, output_dir):
                successful += 1
            else:
                failed += 1
                
            # Brief pause between files
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error processing {filepath.name}: {str(e)}")
            failed += 1
            
    logger.info(f"\nAnalysis complete.")
    logger.info(f"Successful: {successful}, Failed: {failed}")
            
if __name__ == '__main__':
    main() 