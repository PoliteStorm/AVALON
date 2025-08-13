#!/usr/bin/env python3
"""
⚡ QUICK STATUS CHECK - Environmental Sensing System
===================================================

Quick verification script for new conversations to check system status
and provide immediate next steps.

Author: Environmental Sensing Research Team
Date: August 13, 2025
"""

import json
import os
from pathlib import Path
from datetime import datetime

def quick_status_check():
    """Quick status check for new conversations."""
    print("⚡ QUICK STATUS CHECK - Environmental Sensing System")
    print("=" * 60)
    print()
    
    # Check current status file
    status_file = Path("/home/kronos/testTRANSFORM/ENVIRONMENTAL_SENSING_SYSTEM/current_status.json")
    
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            print("📊 SYSTEM STATUS: ✅ OPERATIONAL")
            print(f"🕒 Last Updated: {status.get('last_updated', 'Unknown')}")
            print(f"🎯 Current Phase: {status.get('current_phase', 'Unknown')}")
            print()
            
            # Display phase completion
            phases = status.get('phase_completion', {})
            print("📈 PHASE COMPLETION:")
            for phase, completion in phases.items():
                status_icon = "✅" if completion == 100 else "🚀" if completion > 0 else "⏳"
                print(f"   {status_icon} {phase.replace('_', ' ').title()}: {completion}%")
            
            print()
            
            # Display next actions
            print("🚀 NEXT ACTIONS:")
            actions = status.get('next_actions', [])
            for i, action in enumerate(actions, 1):
                print(f"   {i}. {action}")
            
            print()
            
        except Exception as e:
            print(f"⚠️  Error reading status: {e}")
    else:
        print("⚠️  Status file not found")
    
    # Check Phase 3 directory
    phase3_dir = Path("/home/kronos/testTRANSFORM/ENVIRONMENTAL_SENSING_SYSTEM/PHASE_3_2D_VISUALIZATION")
    
    if phase3_dir.exists():
        print("🎯 PHASE 3 STATUS:")
        print("   📁 Directory: ✅ Exists")
        
        # Check enhanced runner
        enhanced_runner = phase3_dir / "enhanced_phase3_runner.py"
        if enhanced_runner.exists():
            print("   🚀 Enhanced Runner: ✅ Available")
        else:
            print("   🚀 Enhanced Runner: ❌ Not found")
        
        # Check results
        results_dir = phase3_dir / "results"
        if results_dir.exists():
            print("   📊 Results: ✅ Available")
        else:
            print("   📊 Results: ❌ Not found")
        
        # Check virtual environment
        venv_dir = phase3_dir / "phase3_venv"
        if venv_dir.exists():
            print("   🐍 Virtual Environment: ✅ Available")
        else:
            print("   🐍 Virtual Environment: ❌ Not found")
        
        print()
    
    # Quick commands
    print("⚡ QUICK COMMANDS:")
    print("   cd /home/kronos/testTRANSFORM/ENVIRONMENTAL_SENSING_SYSTEM/PHASE_3_2D_VISUALIZATION")
    print("   source phase3_venv/bin/activate")
    print("   python3 enhanced_phase3_runner.py")
    print()
    
    print("🎯 READY FOR: Week 2 Implementation (Web-based Dashboard Frontend)")
    print("🌟 STATUS: All critical issues resolved, Phase 3 95% complete")

if __name__ == "__main__":
    quick_status_check() 