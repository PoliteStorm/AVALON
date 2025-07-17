#!/usr/bin/env python3
"""
Enhanced Adamatzky Analysis Test Runner
Executes comprehensive testing of fungal electrical activity analysis
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from enhanced_adamatzky_processor import EnhancedAdamatzkyProcessor

def run_comprehensive_test():
    """Run comprehensive test of the enhanced Adamatzky processor"""
    
    print("🚀 Enhanced Adamatzky Analysis Test Runner")
    print("=" * 60)
    
    # Initialize processor
    processor = EnhancedAdamatzkyProcessor()
    
    # Test data sources
    test_data_sources = [
        "../data/processed",  # Processed data directory
        "../data/raw",        # Raw data directory
        "../../validated_fungal_electrical_csvs",  # Validated data
        "../../csv_data"  # Additional CSV data
    ]
    
    all_results = {}
    
    for data_source in test_data_sources:
        data_path = Path(data_source)
        if data_path.exists():
            print(f"\n📁 Testing data source: {data_source}")
            print("-" * 40)
            
            # Find CSV files
            csv_files = list(data_path.glob("*.csv"))
            print(f"   Found {len(csv_files)} CSV files")
            
            if csv_files:
                # Process first few files for testing
                test_files = csv_files[:5]  # Limit to 5 files for testing
                
                for csv_file in test_files:
                    try:
                        print(f"\n🔬 Processing: {csv_file.name}")
                        results = processor.process_single_file(str(csv_file))
                        
                        if results:
                            all_results[csv_file.name] = results
                            print(f"   ✅ Successfully processed")
                            
                            # Display key metrics
                            wave_features = results['wave_features']
                            print(f"   📊 Features: {wave_features['n_features']}")
                            print(f"   📊 Max magnitude: {wave_features['max_magnitude']:.3f}")
                            
                            # Show temporal scale distribution
                            temporal_dist = pd.Series(wave_features['temporal_scale_distribution']).value_counts()
                            print(f"   ⏰ Temporal scales:")
                            for scale, count in temporal_dist.items():
                                percentage = (count / len(wave_features['temporal_scale_distribution'])) * 100
                                print(f"      {scale}: {percentage:.1f}%")
                        
                    except Exception as e:
                        print(f"   ❌ Error processing {csv_file.name}: {e}")
            else:
                print(f"   ⚠️  No CSV files found in {data_source}")
        else:
            print(f"   ⚠️  Data source not found: {data_source}")
    
    # Create comprehensive summary
    if all_results:
        print(f"\n📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        print(f"   Files processed: {len(all_results)}")
        
        # Aggregate statistics
        total_features = sum(r['wave_features']['n_features'] for r in all_results.values())
        avg_magnitude = np.mean([r['wave_features']['max_magnitude'] for r in all_results.values()])
        
        print(f"   Total features detected: {total_features}")
        print(f"   Average max magnitude: {avg_magnitude:.3f}")
        
        # Temporal scale summary
        all_temporal_scales = []
        for results in all_results.values():
            all_temporal_scales.extend(results['wave_features']['temporal_scale_distribution'])
        
        if all_temporal_scales:
            scale_dist = pd.Series(all_temporal_scales).value_counts()
            print(f"\n   ⏰ Overall Temporal Scale Distribution:")
            for scale, count in scale_dist.items():
                percentage = (count / len(all_temporal_scales)) * 100
                print(f"      {scale}: {percentage:.1f}%")
        
        # Validation summary
        validation_scores = []
        for results in all_results.values():
            if 'validation_results' in results:
                validation_scores.append(results['validation_results'])
        
        if validation_scores:
            print(f"\n   ✅ Validation Summary:")
            # Calculate average validation scores
            avg_temporal_alignment = np.mean([v.get('temporal_alignment', 0) for v in validation_scores])
            print(f"      Average temporal alignment: {avg_temporal_alignment:.3f}")
        
        # Save test results
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'files_processed': len(all_results),
            'file_names': list(all_results.keys()),
            'total_features': total_features,
            'avg_max_magnitude': avg_magnitude,
            'temporal_scale_distribution': scale_dist.to_dict() if all_temporal_scales else {},
            'validation_summary': {
                'avg_temporal_alignment': avg_temporal_alignment if validation_scores else 0
            }
        }
        
        # Save test results
        test_results_file = processor.dirs['reports'] / f"comprehensive_test_{processor.timestamp}.json"
        with open(test_results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n💾 Test results saved to: {test_results_file}")
        
    else:
        print(f"\n❌ No files were successfully processed")
    
    print(f"\n✅ Enhanced Adamatzky analysis test complete!")
    print(f"   Results directory: {processor.output_dir}")
    print(f"   Visualizations: {processor.dirs['visualizations']}")
    print(f"   Reports: {processor.dirs['reports']}")

def run_single_file_test():
    """Run test on a single file"""
    
    print("🔬 Single File Test")
    print("=" * 40)
    
    # Find a test file
    test_file = None
    for data_source in ["../data/processed", "../data/raw", "../../validated_fungal_electrical_csvs", "../../csv_data"]:
        data_path = Path(data_source)
        if data_path.exists():
            csv_files = list(data_path.glob("*.csv"))
            if csv_files:
                test_file = csv_files[0]
                break
    
    if test_file:
        print(f"Testing with file: {test_file}")
        
        processor = EnhancedAdamatzkyProcessor()
        results = processor.process_single_file(str(test_file))
        
        if results:
            print(f"✅ Successfully processed {test_file.name}")
            print(f"   Features detected: {results['wave_features']['n_features']}")
            print(f"   Max magnitude: {results['wave_features']['max_magnitude']:.3f}")
        else:
            print(f"❌ Failed to process {test_file.name}")
    else:
        print("❌ No test files found")

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Run comprehensive test
    run_comprehensive_test()
    
    # Optionally run single file test
    # run_single_file_test() 