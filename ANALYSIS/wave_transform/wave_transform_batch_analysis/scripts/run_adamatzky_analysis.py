#!/usr/bin/env python3
"""
Analyze Adamatzky-Compliant Files
Run enhanced Adamatzky analysis on the identified compliant files
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from enhanced_adamatzky_processor import EnhancedAdamatzkyProcessor

def main():
    """Run analysis on Adamatzky-compliant files"""
    
    processor = EnhancedAdamatzkyProcessor()
    
    # Analyze files in the processed directory
    processed_dir = Path("../data/processed")
    csv_files = list(processed_dir.glob("*.csv"))
    
    print(f"🔬 Analyzing {len(csv_files)} Adamatzky-compliant files...")
    print("=" * 60)
    
    for csv_file in csv_files:
        print(f"📊 Processing: {csv_file.name}")
        results = processor.process_single_file(str(csv_file))
        
        if results:
            print(f"   ✅ Successfully analyzed")
            wave_features = results['wave_features']
            print(f"   📊 Features: {wave_features['n_features']}")
            print(f"   📊 Max magnitude: {wave_features['max_magnitude']:.3f}")
            
            # Show temporal scale distribution
            if 'temporal_scale_distribution' in wave_features:
                temporal_dist = pd.Series(wave_features['temporal_scale_distribution']).value_counts()
                print(f"   ⏰ Temporal scales:")
                for scale, count in temporal_dist.items():
                    percentage = (count / len(wave_features['temporal_scale_distribution'])) * 100
                    print(f"      {scale}: {percentage:.1f}%")
        else:
            print(f"   ❌ Failed to analyze")
    
    print(f"✅ Adamatzky analysis complete!")

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    main()
