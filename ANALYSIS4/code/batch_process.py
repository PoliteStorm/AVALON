#!/usr/bin/env python3

import argparse
from pathlib import Path
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
from tqdm import tqdm
from process_single_file import process_file

def find_csv_files(input_dir: str, pattern: str = "*.csv") -> list:
    """Find all CSV files matching pattern in input directory"""
    input_path = Path(input_dir)
    return list(input_path.glob(pattern))

def process_file_wrapper(args):
    """Wrapper for process_file to handle multiprocessing"""
    file_path, output_dir, num_scales, chunk_size = args
    try:
        result = process_file(
            str(file_path),
            num_scales=num_scales,
            chunk_size=chunk_size,
            output_dir=str(output_dir / file_path.stem)
        )
        return {
            'file': str(file_path),
            'status': 'success',
            'output_dir': str(output_dir / file_path.stem)
        }
    except Exception as e:
        return {
            'file': str(file_path),
            'status': 'error',
            'error': str(e)
        }

def batch_process(input_dir: str, output_dir: str, pattern: str = "*.csv",
                 num_scales: int = 32, chunk_size: int = 5000,
                 max_workers: int = 4) -> dict:
    """
    Process multiple CSV files in parallel with progress tracking
    
    Args:
        input_dir: Directory containing CSV files
        output_dir: Base directory for results
        pattern: Glob pattern for finding CSV files
        num_scales: Number of wavelet scales
        chunk_size: Size of processing chunks
        max_workers: Number of parallel processes
        
    Returns:
        Dictionary with processing results and statistics
    """
    
    # Find all CSV files
    csv_files = find_csv_files(input_dir, pattern)
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir} matching pattern {pattern}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for parallel processing
    process_args = [
        (f, output_path, num_scales, chunk_size)
        for f in csv_files
    ]
    
    # Process files in parallel with progress bar
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(csv_files), desc="Processing files") as pbar:
            for result in executor.map(process_file_wrapper, process_args):
                results.append(result)
                pbar.update(1)
    
    # Generate summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': input_dir,
        'pattern': pattern,
        'total_files': len(csv_files),
        'successful': len(successful),
        'failed': len(failed),
        'parameters': {
            'num_scales': num_scales,
            'chunk_size': chunk_size,
            'max_workers': max_workers
        },
        'results': results
    }
    
    # Save summary
    summary_file = output_path / 'batch_processing_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Batch process CSV files')
    parser.add_argument('input_dir', help='Directory containing CSV files')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--pattern', default='*.csv',
                       help='Glob pattern for CSV files (default: *.csv)')
    parser.add_argument('--num-scales', type=int, default=32,
                       help='Number of wavelet scales (default: 32)')
    parser.add_argument('--chunk-size', type=int, default=5000,
                       help='Processing chunk size (default: 5000)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel processes (default: 4)')
    
    args = parser.parse_args()
    
    try:
        summary = batch_process(
            args.input_dir,
            args.output_dir,
            args.pattern,
            args.num_scales,
            args.chunk_size,
            args.max_workers
        )
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total files: {summary['total_files']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        
        if summary['failed'] > 0:
            print("\nFailed files:")
            for result in summary['results']:
                if result['status'] == 'error':
                    print(f"- {result['file']}: {result['error']}")
        
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 