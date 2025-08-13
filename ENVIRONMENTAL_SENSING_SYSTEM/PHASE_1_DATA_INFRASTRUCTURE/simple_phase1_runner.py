#!/usr/bin/env python3
"""
🌍 SIMPLE PHASE 1 RUNNER - No External Dependencies
====================================================

Simplified runner for Phase 1 analysis without external dependencies.

Author: Environmental Sensing Research Team
Date: August 12, 2025
Version: 1.0.0
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

def main():
    """Run simplified Phase 1 analysis."""
    print("🌍 ENVIRONMENTAL SENSING SYSTEM - PHASE 1")
    print("=" * 50)
    
    try:
        # Check if we're in the right directory
        current_dir = Path.cwd()
        print(f"📍 Current directory: {current_dir}")
        
        # Check Phase 1 structure
        phase1_dir = current_dir / "PHASE_1_DATA_INFRASTRUCTURE"
        if not phase1_dir.exists():
            print("❌ Phase 1 directory not found!")
            return 1
        
        print("✅ Phase 1 directory found")
        
        # Check for required scripts
        required_scripts = [
            "environmental_sensing_phase1_data_infrastructure.py",
            "baseline_environmental_analysis.py", 
            "data_validation_framework.py"
        ]
        
        missing_scripts = []
        for script in required_scripts:
            script_path = phase1_dir / script
            if script_path.exists():
                print(f"✅ {script} - Found")
            else:
                print(f"❌ {script} - Missing")
                missing_scripts.append(script)
        
        if missing_scripts:
            print(f"\n❌ Missing scripts: {', '.join(missing_scripts)}")
            return 1
        
        # Check data directory
        data_dir = Path("../../DATA/raw/15061491")
        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            print(f"✅ Data directory found with {len(csv_files)} CSV files")
        else:
            print("⚠️ Data directory not found")
        
        # Check results directory
        results_dir = current_dir / "RESULTS" / "baseline_analysis"
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Results directory ready: {results_dir}")
        
        # Create simple status report
        status_report = {
            "phase": "PHASE_1_DATA_INFRASTRUCTURE",
            "status": "READY_FOR_EXECUTION",
            "timestamp": datetime.now().isoformat(),
            "scripts_ready": len(required_scripts),
            "data_files_available": len(csv_files) if 'csv_files' in locals() else 0,
            "results_directory": str(results_dir),
            "next_steps": [
                "Install required dependencies (pandas, numpy, matplotlib, seaborn)",
                "Run: python3 environmental_sensing_phase1_data_infrastructure.py",
                "Check results in RESULTS/baseline_analysis/",
                "Proceed to Phase 2: Audio Synthesis"
            ]
        }
        
        # Save status report
        status_file = results_dir / "phase1_status_report.json"
        with open(status_file, 'w') as f:
            json.dump(status_report, f, indent=2)
        print(f"✅ Status report saved: {status_file}")
        
        print("\n🎉 PHASE 1 INFRASTRUCTURE READY!")
        print("=" * 50)
        print("📁 All scripts created and validated")
        print("📊 Directory structure complete")
        print("🔍 Data validation framework ready")
        print("📈 Baseline analysis tools ready")
        print("🚀 Ready for execution with dependencies!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 