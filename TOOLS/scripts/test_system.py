#!/usr/bin/env python3
"""
Test script for Fungal Electrical Activity Monitoring System
"""

import os
import sys
import json
from datetime import datetime

def test_data_files():
    """Test that data files exist and are readable"""
    print("Testing data files...")
    data_files = [
        "data/Norm_vs_deep_tip_crop.csv",
        "data/New_Oyster_with spray_as_mV_seconds_SigView.csv",
        "data/Ch1-2_1second_sampling.csv"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            return False
    return True

def test_scripts():
    """Test that key scripts exist"""
    print("Testing scripts...")
    script_files = [
        "scripts/ultra_optimized_fungal_monitoring_simple.py",
        "scripts/fungal_electrical_monitoring_with_wave_transform.py",
        "scripts/batch_fungal_csv_analysis.py"
    ]
    
    for script_path in script_files:
        if os.path.exists(script_path):
            print(f"✓ {script_path} exists")
        else:
            print(f"✗ {script_path} missing")
            return False
    return True

def test_documentation():
    """Test that documentation exists"""
    print("Testing documentation...")
    doc_files = [
        "docs/fungal_electrical_parameters.md",
        "docs/parameter_requirements_summary.md",
        "README.md"
    ]
    
    for doc_path in doc_files:
        if os.path.exists(doc_path):
            print(f"✓ {doc_path} exists")
        else:
            print(f"✗ {doc_path} missing")
            return False
    return True

def test_quick_analysis():
    """Test a quick analysis on one file"""
    print("Testing quick analysis...")
    try:
        # Import the analysis function
        sys.path.append('scripts')
        from ultra_optimized_fungal_monitoring_simple import analyze_file
        
        # Test with one file
        test_file = "data/Norm_vs_deep_tip_crop.csv"
        if os.path.exists(test_file):
            results = analyze_file(test_file)
            if results and 'spike_count' in results:
                print(f"✓ Quick analysis successful - {results['spike_count']} spikes detected")
                return True
            else:
                print("✗ Quick analysis failed - no results returned")
                return False
        else:
            print("✗ Test file not found")
            return False
    except Exception as e:
        print(f"✗ Quick analysis error: {e}")
        return False

def main():
    print("Testing Fungal Electrical Activity Monitoring System...")
    print("=" * 60)
    
    tests = [
        ("Data Files", test_data_files),
        ("Scripts", test_scripts),
        ("Documentation", test_documentation),
        ("Quick Analysis", test_quick_analysis)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            print(f"✓ {test_name} test passed")
        else:
            print(f"✗ {test_name} test failed")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run: python scripts/ultra_optimized_fungal_monitoring_simple.py data/")
        print("2. Run: python scripts/fungal_electrical_monitoring_with_wave_transform.py data/")
        print("3. Check results/ directory for outputs")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main() 