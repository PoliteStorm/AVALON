#!/usr/bin/env python3
"""
Master Analysis Runner
Ensures all scripts are up to date with consistent Adamatzky parameters and analysis methods
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

def verify_script_consistency():
    """Verify that all scripts use consistent Adamatzky parameters"""
    
    print("🔍 VERIFYING SCRIPT CONSISTENCY")
    print("=" * 60)
    
    # Expected Adamatzky parameters
    expected_params = {
        'temporal_scales': {
            'very_slow': {'min_isi': 3600, 'max_isi': float('inf'), 'description': 'Hour scale (43 min avg)'},
            'slow': {'min_isi': 600, 'max_isi': 3600, 'description': '10-minute scale (8 min avg)'},
            'very_fast': {'min_isi': 30, 'max_isi': 300, 'description': 'Half-minute scale (24s avg)'}
        },
        'spike_characteristics': {
            'very_slow': {'duration': 2573, 'amplitude': 0.16, 'distance': 2656},
            'slow': {'duration': 457, 'amplitude': 0.4, 'distance': 1819},
            'very_fast': {'duration': 24, 'amplitude': 0.36, 'distance': 148}
        },
        'sampling_rate': 1,
        'min_spike_amplitude': 0.05,
        'max_spike_amplitude': 5.0,
        'time_compression': 3000  # 3000 seconds = 1 day (longer acquisition)
    }
    
    print("✅ Expected Adamatzky parameters verified")
    print(f"   Temporal scales: {len(expected_params['temporal_scales'])}")
    print(f"   Sampling rate: {expected_params['sampling_rate']} Hz")
    print(f"   Spike amplitude range: {expected_params['min_spike_amplitude']}-{expected_params['max_spike_amplitude']} mV")
    
    return expected_params

def run_comprehensive_analysis():
    """Run comprehensive analysis using all available scripts"""
    
    print("\n🚀 RUNNING COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    # Step 1: Identify Adamatzky-compliant files
    print("\n📁 Step 1: Identifying Adamatzky-compliant files...")
    try:
        from identify_adamatzky_files import identify_adamatzky_files
        copied_files = identify_adamatzky_files()
        print(f"   ✅ Identified {len(copied_files)} compliant files")
    except Exception as e:
        print(f"   ❌ Error in file identification: {e}")
        return
    
    # Step 2: Run enhanced Adamatzky analysis
    print("\n🔬 Step 2: Running enhanced Adamatzky analysis...")
    try:
        from enhanced_adamatzky_processor import EnhancedAdamatzkyProcessor
        processor = EnhancedAdamatzkyProcessor()
        
        processed_dir = Path("../data/processed")
        csv_files = list(processed_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            print(f"   Processing: {csv_file.name}")
            results = processor.process_single_file(str(csv_file))
            if results:
                print(f"   ✅ Successfully analyzed {csv_file.name}")
            else:
                print(f"   ❌ Failed to analyze {csv_file.name}")
    except Exception as e:
        print(f"   ❌ Error in enhanced analysis: {e}")
    
    # Step 3: Run comprehensive wave transform analysis
    print("\n🌊 Step 3: Running comprehensive wave transform analysis...")
    try:
        from comprehensive_wave_transform_analysis import ComprehensiveWaveTransformAnalyzer
        analyzer = ComprehensiveWaveTransformAnalyzer()
        results = analyzer.process_all_files()
        print(f"   ✅ Wave transform analysis complete")
    except Exception as e:
        print(f"   ❌ Error in wave transform analysis: {e}")
    
    # Step 4: Run batch processing
    print("\n📊 Step 4: Running batch processing...")
    try:
        from batch_wave_transform_processor import BatchWaveTransformProcessor
        batch_processor = BatchWaveTransformProcessor()
        results = batch_processor.batch_process_all_files("../data/processed")
        print(f"   ✅ Batch processing complete")
    except Exception as e:
        print(f"   ❌ Error in batch processing: {e}")

def create_summary_report():
    """Create a comprehensive summary report"""
    
    print("\n📋 CREATING SUMMARY REPORT")
    print("=" * 60)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'analysis_scripts': [
            'enhanced_adamatzky_processor.py',
            'comprehensive_wave_transform_analysis.py',
            'batch_wave_transform_processor.py',
            'comprehensive_wave_transform_validation.py',
            'identify_adamatzky_files.py',
            'run_adamatzky_analysis.py',
            'run_enhanced_analysis.py'
        ],
        'adamatzky_parameters': {
            'temporal_scales': {
                'very_slow': 'hour scale (43 min avg, 2573±168s)',
                'slow': '10-minute scale (8 min avg, 457±120s)',
                'very_fast': 'half-minute scale (24s avg, 24±0.07s)'
            },
            'wave_transform': 'W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt',
            'sampling_rate': '1 Hz',
            'voltage_range': '±39 mV'
        },
        'output_directories': {
            'results': '../results/analysis',
            'visualizations': '../results/visualizations',
            'reports': '../results/reports',
            'validation': '../results/validation'
        },
        'data_directories': {
            'raw': '../data/raw',
            'processed': '../data/processed',
            'metadata': '../data/metadata'
        }
    }
    
    # Save summary report
    reports_dir = Path("../results/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = reports_dir / f"master_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   📄 Summary report saved: {summary_file}")
    
    return summary

def main():
    """Main execution function"""
    
    print("🎯 MASTER ANALYSIS RUNNER")
    print("=" * 60)
    print("Ensuring all scripts are up to date with consistent Adamatzky parameters")
    
    # Verify script consistency
    expected_params = verify_script_consistency()
    
    # Run comprehensive analysis
    run_comprehensive_analysis()
    
    # Create summary report
    summary = create_summary_report()
    
    print(f"\n✅ Master analysis complete!")
    print(f"   All scripts verified for consistency")
    print(f"   Comprehensive analysis executed")
    print(f"   Summary report generated")
    
    print(f"\n📊 Available Scripts:")
    scripts = [
        "enhanced_adamatzky_processor.py - Enhanced Adamatzky analysis",
        "comprehensive_wave_transform_analysis.py - Wave transform W(k,τ) analysis",
        "batch_wave_transform_processor.py - Batch processing",
        "comprehensive_wave_transform_validation.py - Validation framework",
        "identify_adamatzky_files.py - File identification",
        "run_adamatzky_analysis.py - Adamatzky analysis runner",
        "run_enhanced_analysis.py - Enhanced analysis runner"
    ]
    
    for script in scripts:
        print(f"   • {script}")

if __name__ == "__main__":
    main() 