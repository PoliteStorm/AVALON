#!/usr/bin/env python3
"""
Runner script for the rigorous fungal analysis.
This script can be run from the project root directory.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the analysis
from analysis.rigorous_fungal_analysis import RigorousFungalAnalyzer

if __name__ == "__main__":
    print("=== Running Rigorous Fungal Analysis ===")
    print("Data directories:")
    print("  Coordinate data: data/csv_data")
    print("  Voltage data: data/15061491/fungal_spikes/good_recordings")
    print()
    
    # Initialize analyzer with correct paths
    analyzer = RigorousFungalAnalyzer("data/csv_data", "data/15061491/fungal_spikes/good_recordings")
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    print("\n=== Analysis Complete ===")
    print("Results saved to: results/rigorous_analysis_results/") 