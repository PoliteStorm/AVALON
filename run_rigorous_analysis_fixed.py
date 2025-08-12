#!/usr/bin/env python3
"""
Runner script for the rigorous fungal analysis.
This script can be run from the fungal_analysis_project directory.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the analysis
from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

if __name__ == "__main__":
    print("=== Running Rigorous Fungal Analysis ===")
    
    # Get the correct paths relative to the project root
    project_root = Path(__file__).parent.parent  # Go up one level from fungal_analysis_project
    csv_data_dir = project_root / "csv_data"
    voltage_data_dir = project_root / "15061491" / "fungal_spikes" / "good_recordings"
    
    print("Data directories:")
    print(f"  Coordinate data: {csv_data_dir}")
    print(f"  Voltage data: {voltage_data_dir}")
    print()
    
    # Check if directories exist
    if not csv_data_dir.exists():
        print(f"ERROR: CSV data directory not found: {csv_data_dir}")
        sys.exit(1)
    
    if not voltage_data_dir.exists():
        print(f"WARNING: Voltage data directory not found: {voltage_data_dir}")
        print("Analysis will proceed with coordinate data only.")
    
    # Initialize analyzer with correct paths
    analyzer = RigorousFungalAnalyzer(str(csv_data_dir), str(voltage_data_dir))
    
    try:
        # Run comprehensive analysis
        print("Starting comprehensive analysis...")
        results = analyzer.run_comprehensive_analysis()
        
        print("\n=== Analysis Complete ===")
        print("Results saved to: results/rigorous_analysis_results/")
        
    except Exception as e:
        print(f"ERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 