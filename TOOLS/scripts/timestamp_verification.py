#!/usr/bin/env python3
"""
Timestamp Verification Script
Author: Joe Knowles
Timestamp: 2025-08-12 09:23:27 BST
Description: Verifies all timestamps are consistent across the project
"""

import os
import re
from pathlib import Path

def verify_timestamps():
    """Verify all timestamps are consistent across the project."""
    print("🕐 Timestamp Verification Script")
    print("Author: Joe Knowles")
    print("Timestamp: 2025-08-12 09:23:27 BST")
    print("=" * 60)
    
    # Expected timestamps
    expected_human = "2025-08-12 09:23:27 BST"
    expected_machine = "20250812_092327"
    
    # Directories to check
    directories = [
        "TOOLS/scripts/",
        "DOCUMENTATION/",
        "RESULTS/analysis/",
        "."
    ]
    
    # File patterns to check
    file_patterns = [
        "*.py",
        "*.md",
        "*.txt",
        "*.json"
    ]
    
    # Results tracking
    total_files = 0
    verified_files = 0
    issues_found = []
    
    print(f"🔍 Checking for expected timestamps:")
    print(f"  Human readable: {expected_human}")
    print(f"  Machine format: {expected_machine}")
    print()
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        print(f"📁 Checking directory: {directory}")
        
        for pattern in file_patterns:
            for file_path in Path(directory).glob(pattern):
                if file_path.is_file():
                    total_files += 1
                    file_issues = check_file_timestamps(file_path, expected_human, expected_machine)
                    
                    if file_issues:
                        issues_found.extend(file_issues)
                    else:
                        verified_files += 1
                        print(f"  ✅ {file_path.name}")
    
    # Summary
    print(f"\n📊 VERIFICATION SUMMARY")
    print(f"=" * 40)
    print(f"Total files checked: {total_files}")
    print(f"Files verified: {verified_files}")
    print(f"Issues found: {len(issues_found)}")
    
    if issues_found:
        print(f"\n❌ ISSUES FOUND:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print(f"\n🎉 ALL TIMESTAMPS VERIFIED SUCCESSFULLY!")
        print(f"✅ All {total_files} files have consistent timestamps")
    
    return len(issues_found) == 0

def check_file_timestamps(file_path, expected_human, expected_machine):
    """Check timestamps in a single file."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for human readable timestamp
        if expected_human in content:
            # Count occurrences
            count = content.count(expected_human)
            if count == 0:
                issues.append(f"{file_path}: Missing human readable timestamp")
        else:
            # Check if it's a file that should have timestamps
            if should_have_timestamps(file_path):
                issues.append(f"{file_path}: Missing human readable timestamp")
        
        # Check for machine format timestamp
        if expected_machine in content:
            # Count occurrences
            count = content.count(expected_machine)
            if count == 0:
                issues.append(f"{file_path}: Missing machine format timestamp")
        
        # Check for old timestamps
        old_timestamps = [
            "2025-01-27",
            "20250127",
            "2025-01-27T",
            "2025-01-27 "
        ]
        
        for old_timestamp in old_timestamps:
            if old_timestamp in content:
                issues.append(f"{file_path}: Contains old timestamp: {old_timestamp}")
        
    except Exception as e:
        issues.append(f"{file_path}: Error reading file: {e}")
    
    return issues

def should_have_timestamps(file_path):
    """Determine if a file should have timestamps."""
    filename = file_path.name.lower()
    
    # Files that should have timestamps
    timestamp_files = [
        "advanced_fungal_communication_analyzer.py",
        "test_advanced_analysis.py",
        "data_driven_advanced_analysis_test.py",
        "simple_advanced_analysis_test.py",
        "optimized_relationship_analyzer.py",
        "speed_improvement_visualizer.py",
        "requirements_advanced_analysis.txt",
        "mushroom_computing_possibilities.md",
        "relationship_analysis_summary.md"
    ]
    
    return any(name in filename for name in timestamp_files)

def main():
    """Main verification function."""
    success = verify_timestamps()
    
    if success:
        print(f"\n🎯 VERIFICATION COMPLETE: ALL TIMESTAMPS CONSISTENT")
        return 0
    else:
        print(f"\n❌ VERIFICATION FAILED: TIMESTAMP ISSUES DETECTED")
        return 1

if __name__ == "__main__":
    exit(main()) 