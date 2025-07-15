#!/usr/bin/env python3
"""
Main analysis script for fungal electrophysiology data using √t transform.
This script runs the complete analysis pipeline with configurable parameters.
"""

import sys
import os
import json
from datetime import datetime
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def run_analysis(data_dir, voltage_dir, output_dir, config_path=None):
    """
    Run the complete fungal analysis pipeline.
    
    Args:
        data_dir: Directory containing coordinate CSV files
        voltage_dir: Directory containing voltage recording files
        output_dir: Directory to save results
        config_path: Path to configuration file (optional)
    """
    print("="*80)
    print("FUNGAL ELECTROPHYSIOLOGY ANALYSIS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data directory: {data_dir}")
    print(f"Voltage directory: {voltage_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load configuration if provided
    config = None
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Loaded configuration from: {config_path}")
    
    # Initialize analyzer
    analyzer = RigorousFungalAnalyzer(data_dir, voltage_dir)
    
    # Run comprehensive analysis
    print("\nRunning comprehensive analysis...")
    results = analyzer.run_comprehensive_analysis()
    
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return results

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Run fungal electrophysiology analysis')
    parser.add_argument('--data-dir', default='../../data/raw/csv_data',
                       help='Directory containing coordinate CSV files')
    parser.add_argument('--voltage-dir', default='../../data/raw/15061491/fungal_spikes/good_recordings',
                       help='Directory containing voltage recording files')
    parser.add_argument('--output-dir', default='../../results/analysis',
                       help='Directory to save results')
    parser.add_argument('--config', default='../../config/parameters/species_config.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1
    
    if not os.path.exists(args.voltage_dir):
        print(f"Error: Voltage directory not found: {args.voltage_dir}")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis
    try:
        results = run_analysis(args.data_dir, args.voltage_dir, args.output_dir, args.config)
        print("Analysis completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 