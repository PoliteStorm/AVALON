#!/usr/bin/env python3
"""
🚀 QUICK START - Phase 1 Analysis
==================================

Quick start script to run the complete Phase 1 analysis pipeline.

Author: Environmental Sensing Research Team
Date: August 12, 2025
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add Phase 1 directory to path
phase1_dir = Path(__file__).parent / "PHASE_1_DATA_INFRASTRUCTURE"
sys.path.append(str(phase1_dir))

def main():
    """Run Phase 1 analysis."""
    print("🌍 ENVIRONMENTAL SENSING SYSTEM - PHASE 1")
    print("=" * 50)
    
    try:
        # Import and run the main Phase 1 script
        from environmental_sensing_phase1_data_infrastructure import main as phase1_main
        
        print("🚀 Starting Phase 1 analysis...")
        result = phase1_main()
        
        if result == 0:
            print("\n✅ Phase 1 analysis completed successfully!")
            print("📊 Check results in: ENVIRONMENTAL_SENSING_SYSTEM/RESULTS/baseline_analysis/")
            print("🚀 Ready for Phase 2: Audio Synthesis & Environmental Correlation!")
        else:
            print(f"\n❌ Phase 1 analysis failed with exit code: {result}")
            return 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all Phase 1 scripts are in place.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 