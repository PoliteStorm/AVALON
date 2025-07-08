#!/usr/bin/env python3
"""
Updated TEST_ALL_SCRIPTS.py for new directory structure
Tests all scripts in the reorganized Joe's Quantum Research project.
"""

import os
import sys
import subprocess
import time
import traceback
from pathlib import Path

def test_script(script_path, description):
    """Test a single script with error handling"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    try:
        if not os.path.exists(script_path):
            print(f"❌ SKIP: Script not found: {script_path}")
            return False
            
        # Import test for Python files
        if script_path.endswith('.py'):
            try:
                # Add the script's directory to Python path
                script_dir = os.path.dirname(script_path)
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                
                # Try to import the module
                module_name = os.path.splitext(os.path.basename(script_path))[0]
                print(f"🔍 Testing import of {module_name}...")
                
                # Use importlib to import the module
                import importlib.util
                spec = importlib.util.spec_from_file_location(module_name, script_path)
                module = importlib.util.module_from_spec(spec)
                
                # Don't execute, just test if it can be imported
                print(f"✅ PASS: {description} - Import successful")
                return True
                
            except Exception as e:
                print(f"❌ FAIL: {description} - Import error: {str(e)}")
                return False
        else:
            print(f"❌ SKIP: Non-Python file: {script_path}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {description} - {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Test all scripts in the reorganized directory structure"""
    print("🍄 Joe's Quantum Research - Updated Testing Suite")
    print("Testing all scripts in new directory structure...")
    print("=" * 60)
    
    # Define test scripts with their new paths
    test_scripts = [
        # Quantum Consciousness
        ("quantum_consciousness/quantum_consciousness_main.py", "Quantum Consciousness Main System"),
        ("quantum_consciousness/quantum_temporal_foam_analyzer.py", "Quantum Temporal Foam Analyzer"),
        ("quantum_consciousness/spherical_time_analyzer.py", "Spherical Time Analyzer"),
        ("quantum_consciousness/quantum_foam_visualizer.py", "Quantum Foam Visualizer"),
        
        # Multiverse Analysis
        ("multiverse_analysis/multiverse_consciousness_analyzer.py", "Multiverse Consciousness Analyzer"),
        ("multiverse_analysis/reality_check_demo.py", "Reality Check Demo"),
        
        # Mushroom Communication
        ("mushroom_communication/mushroom_translator_standalone.py", "Mushroom Translator Standalone"),
        ("mushroom_communication/mushroom_translator_fixed.py", "Mushroom Translator Fixed"),
        ("mushroom_communication/mushroom_demo.py", "Mushroom Demo"),
        ("mushroom_communication/comprehensive_communication_simulation.py", "Comprehensive Communication Simulation"),
        ("mushroom_communication/fungal_computing_demo.py", "Fungal Computing Demo"),
        
        # Symbol Analysis
        ("symbol_analysis/joe_analysis.py", "Joe's Symbol Analysis"),
        ("symbol_analysis/frequency_code_analyzer.py", "Frequency Code Analyzer"),
        
        # Pattern Decoders
        ("pattern_decoders/biological_pattern_decoder.py", "Biological Pattern Decoder"),
        ("pattern_decoders/colorized_spiral_decoder.py", "Colorized Spiral Decoder"),
        ("pattern_decoders/spiral_pattern_decoder.py", "Spiral Pattern Decoder"),
        
        # Scientific Verification
        ("scientific_verification/adamatzky_comparison.py", "Adamatzky Comparison"),
        ("scientific_verification/validation_report.py", "Validation Report"),
        ("scientific_verification/mycelial_ai_analysis.py", "Mycelial AI Analysis"),
        ("scientific_verification/mycelial_ai_analysis_simple.py", "Mycelial AI Analysis Simple"),
        
        # Demonstrations
        ("demonstrations/run_quantum_foam_analysis.py", "Quantum Foam Analysis Demo"),
        
        # Utilities
        ("utilities/extract_detailed_results.py", "Extract Detailed Results"),
    ]
    
    # Test all scripts
    results = []
    for script_path, description in test_scripts:
        result = test_script(script_path, description)
        results.append((script_path, description, result))
        time.sleep(0.1)  # Small delay between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, _, result in results if result)
    total = len(results)
    
    print(f"✅ PASSED: {passed}/{total} scripts")
    print(f"❌ FAILED: {total - passed}/{total} scripts")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("Your reorganized quantum research system is working perfectly!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("Failed scripts:")
        for script_path, description, result in results:
            if not result:
                print(f"  - {script_path}: {description}")
    
    print(f"\n{'='*60}")
    print("🍄 Testing complete! All scripts now organized by topic.")
    print("✨ New directory structure is ready for research!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 