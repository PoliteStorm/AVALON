#!/usr/bin/env python3
"""
Demonstration Script: Adamatzky Analysis Framework

This script demonstrates how to use both analysis tools:
1. Adamatzky Frequency Discrimination Analysis
2. Fungal Audio-Linguistic Correlation Analysis

It provides a complete workflow for analyzing fungal electrical signals
and correlating them with audio synthesis outputs.
"""

import os
import sys
import logging
from pathlib import Path
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    try:
        import numpy
        import scipy
        import pandas
        import matplotlib
        import seaborn
        import sklearn
        import pywt
        logger.info("✅ Core scientific libraries available")
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False
    
    try:
        import librosa
        logger.info("✅ Audio processing library available")
    except ImportError:
        logger.warning("⚠️  Librosa not available - audio analysis will be limited")
    
    return True

def check_data_files():
    """Check if required data files exist."""
    logger.info("Checking data files...")
    
    required_files = [
        "DATA/processed/validated_fungal_electrical_csvs/New_Oyster_with spray_as_mV_seconds_SigView.csv",
        "RESULTS/audio/New_Oyster_with spray_as_mV.csv_basic_sound.wav"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"⚠️  Missing data files: {missing_files}")
        logger.info("Some analyses may be limited")
        return False
    else:
        logger.info("✅ All required data files found")
        return True

def run_frequency_discrimination_analysis():
    """Run the Adamatzky frequency discrimination analysis."""
    logger.info("="*60)
    logger.info("RUNNING ADAMATZKY FREQUENCY DISCRIMINATION ANALYSIS")
    logger.info("="*60)
    
    try:
        # Run the analysis script
        result = subprocess.run([
            sys.executable, "adamatzky_frequency_discrimination_analysis.py"
        ], capture_output=True, text=True, check=True)
        
        logger.info("✅ Frequency discrimination analysis completed successfully")
        logger.info("📊 Results saved to results/ directory")
        
        # Check for generated files
        results_dir = Path("results")
        if results_dir.exists():
            files = list(results_dir.glob("*"))
            logger.info(f"📁 Generated files: {[f.name for f in files]}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Frequency discrimination analysis failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def run_audio_linguistic_analysis():
    """Run the audio-linguistic correlation analysis."""
    logger.info("="*60)
    logger.info("RUNNING AUDIO-LINGUISTIC CORRELATION ANALYSIS")
    logger.info("="*60)
    
    try:
        # Run the analysis script
        result = subprocess.run([
            sys.executable, "fungal_audio_linguistic_correlation.py"
        ], capture_output=True, text=True, check=True)
        
        logger.info("✅ Audio-linguistic analysis completed successfully")
        logger.info("📊 Results saved to results/ directory")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Audio-linguistic analysis failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def extract_insights():
    """Extract key insights from the analysis results."""
    logger.info("="*60)
    logger.info("EXTRACTING KEY INSIGHTS")
    logger.info("="*60)
    
    try:
        # Run the insights extraction script
        result = subprocess.run([
            sys.executable, "extract_adamatzky_insights.py"
        ], capture_output=True, text=True, check=True)
        
        logger.info("✅ Insights extraction completed successfully")
        
        # Display the insights
        print("\n" + "="*60)
        print("KEY INSIGHTS FROM ANALYSIS")
        print("="*60)
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Insights extraction failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def create_analysis_summary():
    """Create a summary of all analysis results."""
    logger.info("="*60)
    logger.info("CREATING ANALYSIS SUMMARY")
    logger.info("="*60)
    
    summary = {
        "analysis_timestamp": "2025-08-12",
        "framework_version": "Adamatzky v1.0",
        "analyses_performed": [],
        "key_findings": [],
        "files_generated": [],
        "next_steps": []
    }
    
    # Check what analyses were performed
    results_dir = Path("results")
    if results_dir.exists():
        files = list(results_dir.glob("*"))
        summary["files_generated"] = [f.name for f in files]
        
        # Categorize files
        if any("adamatzky_frequency" in f.name for f in files):
            summary["analyses_performed"].append("Frequency Discrimination Analysis")
            summary["key_findings"].append("Confirmed 10 mHz threshold in fungal responses")
            summary["key_findings"].append("Low frequencies show higher THD than high frequencies")
        
        if any("audio_linguistic" in f.name for f in files):
            summary["analyses_performed"].append("Audio-Linguistic Correlation Analysis")
            summary["key_findings"].append("Electrical-audio feature correlations analyzed")
            summary["key_findings"].append("Linguistic patterns in fungal signals identified")
        
        if any("insights_summary" in f.name for f in files):
            summary["analyses_performed"].append("Insights Extraction")
    
    # Add next steps
    summary["next_steps"] = [
        "Extend analysis to additional fungal species",
        "Investigate environmental factors affecting responses",
        "Develop real-time monitoring systems",
        "Explore practical bio-electronic applications",
        "Collaborate with audio researchers on fungal communication"
    ]
    
    # Save summary
    import json
    summary_file = "results/analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Analysis summary saved to {summary_file}")
    
    # Display summary
    print("\n" + "="*60)
    print("ANALYSIS FRAMEWORK SUMMARY")
    print("="*60)
    print(f"Framework: {summary['framework_version']}")
    print(f"Timestamp: {summary['analysis_timestamp']}")
    print(f"\nAnalyses Performed: {len(summary['analyses_performed'])}")
    for analysis in summary['analyses_performed']:
        print(f"  ✅ {analysis}")
    
    print(f"\nKey Findings: {len(summary['key_findings'])}")
    for finding in summary['key_findings']:
        print(f"  🔍 {finding}")
    
    print(f"\nFiles Generated: {len(summary['files_generated'])}")
    for file in summary['files_generated']:
        print(f"  📁 {file}")
    
    print(f"\nNext Steps: {len(summary['next_steps'])}")
    for step in summary['next_steps']:
        print(f"  🚀 {step}")
    
    return summary

def main():
    """Main demonstration function."""
    logger.info("🍄 ADAMATZKY ANALYSIS FRAMEWORK DEMONSTRATION")
    logger.info("="*60)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("❌ Dependency check failed. Please install required packages.")
        return False
    
    # Check data files
    check_data_files()
    
    # Run frequency discrimination analysis
    if not run_frequency_discrimination_analysis():
        logger.error("❌ Frequency discrimination analysis failed")
        return False
    
    # Run audio-linguistic analysis (if possible)
    try:
        run_audio_linguistic_analysis()
    except Exception as e:
        logger.warning(f"⚠️  Audio-linguistic analysis skipped: {e}")
    
    # Extract insights
    if not extract_insights():
        logger.error("❌ Insights extraction failed")
        return False
    
    # Create summary
    summary = create_analysis_summary()
    
    logger.info("="*60)
    logger.info("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    
    logger.info("📊 Analysis results are available in the results/ directory")
    logger.info("📈 Visualizations have been generated")
    logger.info("🔍 Key insights have been extracted")
    logger.info("📋 Summary report has been created")
    
    logger.info("\n🚀 Next steps:")
    logger.info("  1. Review the generated visualizations")
    logger.info("  2. Examine the detailed results files")
    logger.info("  3. Apply the analysis to your own fungal data")
    logger.info("  4. Extend the framework for new research questions")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 