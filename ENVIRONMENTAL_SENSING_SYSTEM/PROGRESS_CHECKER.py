#!/usr/bin/env python3
"""
🔄 AUTOMATIC PROGRESS CHECKER - Environmental Sensing System
============================================================

This script automatically detects current progress, validates status,
and provides smart recommendations for next steps.

Author: Environmental Sensing Research Team
Date: August 12, 2025
Version: 1.0.0
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import subprocess
from typing import Dict, List, Any, Tuple

class ProgressChecker:
    """Automatic progress checker and status validator."""
    
    def __init__(self, base_directory: str = "."):
        self.base_directory = Path(base_directory)
        self.current_status = {}
        self.progress_metrics = {}
        self.recommendations = []
        self.troubleshooting_needs = []
        
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive progress and status check."""
        print("🔄 ENVIRONMENTAL SENSING SYSTEM - PROGRESS CHECK")
        print("=" * 60)
        
        # Check all phases
        self._check_phase1_status()
        self._check_phase2_readiness()
        self._check_system_integrity()
        self._check_data_availability()
        self._check_results_generation()
        
        # Generate comprehensive report
        self._generate_progress_report()
        self._identify_next_actions()
        self._detect_troubleshooting_needs()
        
        return self.current_status
    
    def _check_phase1_status(self):
        """Check Phase 1 completion status."""
        print("🔍 Checking Phase 1: Data Infrastructure...")
        
        phase1_dir = self.base_directory / "PHASE_1_DATA_INFRASTRUCTURE"
        if not phase1_dir.exists():
            self.current_status['phase1'] = {'status': 'MISSING', 'completion': 0}
            self.troubleshooting_needs.append("Phase 1 directory not found")
            return
        
        # Check required scripts
        required_scripts = [
            "environmental_sensing_phase1_data_infrastructure.py",
            "baseline_environmental_analysis.py",
            "data_validation_framework.py",
            "simple_phase1_runner.py"
        ]
        
        scripts_found = 0
        for script in required_scripts:
            if (phase1_dir / script).exists():
                scripts_found += 1
        
        completion_percentage = (scripts_found / len(required_scripts)) * 100
        
        if completion_percentage == 100:
            self.current_status['phase1'] = {
                'status': 'COMPLETE',
                'completion': 100,
                'scripts_ready': scripts_found,
                'total_scripts': len(required_scripts)
            }
            print(f"✅ Phase 1: COMPLETE ({scripts_found}/{len(required_scripts)} scripts)")
        else:
            self.current_status['phase1'] = {
                'status': 'INCOMPLETE',
                'completion': completion_percentage,
                'scripts_ready': scripts_found,
                'total_scripts': len(required_scripts)
            }
            print(f"⚠️ Phase 1: INCOMPLETE ({scripts_found}/{len(required_scripts)} scripts)")
    
    def _check_phase2_readiness(self):
        """Check Phase 2 readiness and requirements."""
        print("🔍 Checking Phase 2: Audio Synthesis Readiness...")
        
        # Check if Phase 1 is complete
        if self.current_status.get('phase1', {}).get('status') == 'COMPLETE':
            phase2_dir = self.base_directory / "PHASE_2_AUDIO_SYNTHESIS"
            
            if not phase2_dir.exists():
                phase2_dir.mkdir(parents=True, exist_ok=True)
                print("📁 Phase 2 directory created")
            
            # Check for existing Phase 2 files
            existing_files = list(phase2_dir.glob("*.py")) + list(phase2_dir.glob("*.md"))
            
            if existing_files:
                self.current_status['phase2'] = {
                    'status': 'IN_PROGRESS',
                    'completion': len(existing_files) * 10,  # Estimate
                    'existing_files': len(existing_files)
                }
                print(f"🔄 Phase 2: IN PROGRESS ({len(existing_files)} files)")
            else:
                self.current_status['phase2'] = {
                    'status': 'READY_TO_BEGIN',
                    'completion': 0,
                    'dependencies_met': True
                }
                print("🚀 Phase 2: READY TO BEGIN")
        else:
            self.current_status['phase2'] = {
                'status': 'BLOCKED',
                'completion': 0,
                'blocking_issue': 'Phase 1 not complete'
            }
            print("❌ Phase 2: BLOCKED (Phase 1 incomplete)")
    
    def _check_system_integrity(self):
        """Check overall system integrity and structure."""
        print("🔍 Checking System Integrity...")
        
        # Check directory structure
        required_dirs = [
            "PHASE_1_DATA_INFRASTRUCTURE",
            "PHASE_2_AUDIO_SYNTHESIS", 
            "PHASE_3_2D_VISUALIZATION",
            "PHASE_4_3D_VISUALIZATION",
            "PHASE_5_JSON_ARCHITECTURE",
            "PHASE_6_CSV_PROCESSING",
            "PHASE_7_ADVANCED_AUDIO",
            "PHASE_8_INTEGRATION",
            "RESULTS",
            "DOCUMENTATION",
            "TESTS"
        ]
        
        dirs_exist = 0
        for dir_name in required_dirs:
            if (self.base_directory / dir_name).exists():
                dirs_exist += 1
        
        integrity_score = (dirs_exist / len(required_dirs)) * 100
        
        self.current_status['system_integrity'] = {
            'score': integrity_score,
            'directories_ready': dirs_exist,
            'total_directories': len(required_dirs)
        }
        
        if integrity_score >= 90:
            print(f"✅ System Integrity: EXCELLENT ({integrity_score:.1f}%)")
        elif integrity_score >= 70:
            print(f"🟡 System Integrity: GOOD ({integrity_score:.1f}%)")
        else:
            print(f"⚠️ System Integrity: NEEDS ATTENTION ({integrity_score:.1f}%)")
    
    def _check_data_availability(self):
        """Check data availability and accessibility."""
        print("🔍 Checking Data Availability...")
        
        # Check for CSV data
        data_path = Path("../../DATA/raw/15061491")
        if data_path.exists():
            csv_files = list(data_path.glob("*.csv"))
            self.current_status['data_availability'] = {
                'status': 'AVAILABLE',
                'csv_files_found': len(csv_files),
                'data_path': str(data_path)
            }
            print(f"✅ Data: AVAILABLE ({len(csv_files)} CSV files)")
        else:
            self.current_status['data_availability'] = {
                'status': 'NOT_FOUND',
                'csv_files_found': 0,
                'data_path': str(data_path)
            }
            print(f"❌ Data: NOT FOUND at {data_path}")
            self.troubleshooting_needs.append("CSV data directory not accessible")
    
    def _check_results_generation(self):
        """Check for existing results and outputs."""
        print("🔍 Checking Results Generation...")
        
        results_dir = self.base_directory / "RESULTS" / "baseline_analysis"
        if results_dir.exists():
            result_files = list(results_dir.glob("*"))
            self.current_status['results_generation'] = {
                'status': 'EXISTS',
                'result_files': len(result_files),
                'results_directory': str(results_dir)
            }
            print(f"✅ Results: EXISTS ({len(result_files)} files)")
        else:
            self.current_status['results_generation'] = {
                'status': 'NOT_GENERATED',
                'result_files': 0,
                'results_directory': str(results_dir)
            }
            print(f"⚠️ Results: NOT GENERATED")
    
    def _generate_progress_report(self):
        """Generate comprehensive progress report."""
        print("\n📊 GENERATING PROGRESS REPORT...")
        
        # Calculate overall progress
        phase1_completion = self.current_status.get('phase1', {}).get('completion', 0)
        phase2_completion = self.current_status.get('phase2', {}).get('completion', 0)
        
        # Weight Phase 1 more heavily as it's foundational
        overall_progress = (phase1_completion * 0.7) + (phase2_completion * 0.3)
        
        self.progress_metrics = {
            'overall_progress': overall_progress,
            'phase1_completion': phase1_completion,
            'phase2_completion': phase2_completion,
            'system_integrity': self.current_status.get('system_integrity', {}).get('score', 0),
            'data_availability': self.current_status.get('data_availability', {}).get('status', 'UNKNOWN'),
            'results_generated': self.current_status.get('results_generation', {}).get('status', 'UNKNOWN')
        }
        
        print(f"📈 Overall Progress: {overall_progress:.1f}%")
    
    def _identify_next_actions(self):
        """Identify next actions and recommendations."""
        print("🎯 IDENTIFYING NEXT ACTIONS...")
        
        # Phase 1 recommendations
        if self.current_status.get('phase1', {}).get('status') != 'COMPLETE':
            self.recommendations.append("Complete Phase 1: Data Infrastructure")
            self.recommendations.append("Run data validation and baseline analysis")
        
        # Phase 2 recommendations
        if self.current_status.get('phase2', {}).get('status') == 'READY_TO_BEGIN':
            self.recommendations.append("Begin Phase 2: Audio Synthesis & Environmental Correlation")
            self.recommendations.append("Implement environmental audio synthesis engine")
            self.recommendations.append("Create pollution audio signature database")
        
        # System recommendations
        if self.current_status.get('system_integrity', {}).get('score', 0) < 90:
            self.recommendations.append("Complete directory structure setup")
        
        # Data recommendations
        if self.current_status.get('data_availability', {}).get('status') != 'AVAILABLE':
            self.recommendations.append("Verify CSV data accessibility")
            self.recommendations.append("Check data directory permissions")
        
        # Results recommendations
        if self.current_status.get('results_generation', {}).get('status') != 'EXISTS':
            self.recommendations.append("Generate Phase 1 results")
            self.recommendations.append("Run baseline environmental analysis")
    
    def _detect_troubleshooting_needs(self):
        """Detect issues requiring troubleshooting."""
        print("🔧 DETECTING TROUBLESHOOTING NEEDS...")
        
        # Check for common issues
        if self.current_status.get('phase1', {}).get('status') != 'COMPLETE':
            self.troubleshooting_needs.append("Phase 1 incomplete - check script creation")
        
        if self.current_status.get('data_availability', {}).get('status') != 'AVAILABLE':
            self.troubleshooting_needs.append("Data not accessible - check paths and permissions")
        
        if self.current_status.get('system_integrity', {}).get('score', 0) < 70:
            self.troubleshooting_needs.append("System structure incomplete - create missing directories")
        
        if not self.troubleshooting_needs:
            print("✅ No troubleshooting needed")
        else:
            print(f"⚠️ {len(self.troubleshooting_needs)} issues detected")
    
    def display_comprehensive_report(self):
        """Display comprehensive progress report."""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE PROGRESS REPORT")
        print("="*60)
        
        # Overall Status
        print(f"\n🎯 OVERALL STATUS:")
        print(f"   Progress: {self.progress_metrics['overall_progress']:.1f}%")
        print(f"   System Integrity: {self.progress_metrics['system_integrity']:.1f}%")
        print(f"   Data: {self.progress_metrics['data_availability']}")
        print(f"   Results: {self.progress_metrics['results_generated']}")
        
        # Phase Status
        print(f"\n📋 PHASE STATUS:")
        print(f"   Phase 1: {self.current_status.get('phase1', {}).get('status', 'UNKNOWN')} ({self.progress_metrics['phase1_completion']:.1f}%)")
        print(f"   Phase 2: {self.current_status.get('phase2', {}).get('status', 'UNKNOWN')} ({self.progress_metrics['phase2_completion']:.1f}%)")
        
        # Next Actions
        if self.recommendations:
            print(f"\n🚀 NEXT ACTIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Troubleshooting
        if self.troubleshooting_needs:
            print(f"\n🔧 TROUBLESHOOTING NEEDED:")
            for i, issue in enumerate(self.troubleshooting_needs, 1):
                print(f"   {i}. {issue}")
        
        # Quick Commands
        print(f"\n⚡ QUICK COMMANDS:")
        if self.current_status.get('phase1', {}).get('status') == 'COMPLETE':
            print(f"   Run Phase 1: python3 PHASE_1_DATA_INFRASTRUCTURE/simple_phase1_runner.py")
        if self.current_status.get('phase2', {}).get('status') == 'READY_TO_BEGIN':
            print(f"   Begin Phase 2: Start implementing audio synthesis engine")
        
        print("\n" + "="*60)
    
    def save_status_report(self):
        """Save current status to file for conversation continuity."""
        status_report = {
            'timestamp': datetime.now().isoformat(),
            'current_status': self.current_status,
            'progress_metrics': self.progress_metrics,
            'recommendations': self.recommendations,
            'troubleshooting_needs': self.troubleshooting_needs
        }
        
        # Save to multiple locations for easy access
        locations = [
            "PROGRESS_TRACKER.md",
            "RESULTS/current_status.json",
            "current_status.json"
        ]
        
        for location in locations:
            try:
                file_path = self.base_directory / location
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                if location.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(status_report, f, indent=2)
                else:
                    # Update markdown file
                    self._update_progress_tracker_md(file_path, status_report)
                
                print(f"✅ Status saved to: {file_path}")
            except Exception as e:
                print(f"⚠️ Could not save to {location}: {e}")
    
    def _update_progress_tracker_md(self, file_path: Path, status_report: Dict):
        """Update the progress tracker markdown file."""
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Update timestamp
            content = content.replace(
                "**Last Updated**: August 12, 2025",
                f"**Last Updated**: {datetime.now().strftime('%B %d, %Y')}"
            )
            
            # Update current phase
            current_phase = "PHASE_2_AUDIO_SYNTHESIS" if status_report['progress_metrics']['phase1_completion'] == 100 else "PHASE_1_DATA_INFRASTRUCTURE"
            content = content.replace(
                "**Current Phase**: PHASE_1_DATA_INFRASTRUCTURE ✅ **COMPLETE**",
                f"**Current Phase**: {current_phase} {'✅ **COMPLETE**' if status_report['progress_metrics']['phase1_completion'] == 100 else '🔄 **IN PROGRESS**'}"
            )
            
            # Update next phase
            next_phase = "PHASE_3_2D_VISUALIZATION" if status_report['progress_metrics']['phase2_completion'] >= 80 else "PHASE_2_AUDIO_SYNTHESIS"
            content = content.replace(
                "**Next Phase**: PHASE_2_AUDIO_SYNTHESIS 🚀 **READY TO BEGIN**",
                f"**Next Phase**: {next_phase} {'🚀 **READY TO BEGIN**' if status_report['progress_metrics']['phase2_completion'] >= 80 else '⏳ **IN PROGRESS**'}"
            )
            
            with open(file_path, 'w') as f:
                f.write(content)

def main():
    """Main execution function."""
    print("🔄 ENVIRONMENTAL SENSING SYSTEM - AUTOMATIC PROGRESS CHECKER")
    print("=" * 70)
    
    try:
        # Initialize checker
        checker = ProgressChecker()
        
        # Run comprehensive check
        status = checker.run_comprehensive_check()
        
        # Display report
        checker.display_comprehensive_report()
        
        # Save status for conversation continuity
        checker.save_status_report()
        
        print("\n🎉 Progress check complete! Status saved for conversation continuity.")
        print("📊 No manual file attachments needed - system automatically tracks progress!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Progress check failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 