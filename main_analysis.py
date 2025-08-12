#!/usr/bin/env python3
"""
Main analysis script for fungal electrical activity analysis using √t transform.
This script analyzes voltage data (electrical recordings) only.
FOCUS: Electrical activity only - no coordinate data analysis.
"""

import sys
import os
import json
from datetime import datetime
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

def load_config(config_path):
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: Configuration file not found: {config_path}")
        return {}

def run_analysis(voltage_dir, output_dir, config_path=None):
    """
    Run the complete fungal electrical analysis pipeline.
    
    Args:
        voltage_dir: Directory containing voltage recording files
        output_dir: Directory to save results
        config_path: Path to configuration file (optional)
    """
    print("="*80)
    print("FUNGAL ELECTRICAL ACTIVITY ANALYSIS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Voltage data directory: {voltage_dir}")
    print(f"Output directory: {output_dir}")
    print()
    print("ANALYSIS TYPE:")
    print("  - Voltage data: Electrical signal analysis")
    print("  - Focus: Electrical spiking patterns")
    print("  - Method: √t transform for electrical signals")
    print()
    
    # Load configuration if provided
    config = None
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Loaded configuration from: {config_path}")
    else:
        print("Using default configuration")
    
    # Initialize analyzer with voltage data only
    analyzer = RigorousFungalAnalyzer(None, voltage_dir)
    
    # Run comprehensive electrical analysis
    print("\nRunning comprehensive electrical analysis...")
    print("This will analyze:")
    print("  ✓ Voltage data for electrical patterns")
    print("  ✓ Electrical spike detection")
    print("  ✓ Signal processing and filtering")
    print("  ✓ Biological plausibility assessment")
    print("  ✓ Electrical activity statistics")
    
    results = analyzer.run_comprehensive_analysis()
    
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return results

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run fungal electrical activity analysis (voltage data only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python main_analysis.py
  python main_analysis.py --voltage-dir ./voltage_data
  python main_analysis.py --config ./config.json --output-dir ./results
        """
    )
    
    parser.add_argument('--voltage-dir', default='15061491/fungal_spikes/good_recordings',
                       help='Directory containing voltage recording files (default: 15061491/fungal_spikes/good_recordings)')
    parser.add_argument('--output-dir', default='results/analysis',
                       help='Directory to save results (default: results/analysis)')
    parser.add_argument('--config', default=None,
                       help='Path to configuration file (optional)')
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.exists(args.voltage_dir):
        print(f"Error: Voltage data directory not found: {args.voltage_dir}")
        print("Please check the path or use --voltage-dir to specify the correct location.")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Run analysis
    try:
        results = run_analysis(args.voltage_dir, args.output_dir, args.config)
        print("\n✅ Electrical analysis completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        return 0
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Please check the error message above and ensure all dependencies are installed.")
        return 1

if __name__ == "__main__":
    exit(main()) 