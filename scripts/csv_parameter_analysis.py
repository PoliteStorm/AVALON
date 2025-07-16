#!/usr/bin/env python3
"""
CSV Parameter Analysis for Fungal Electrical Activity
Identifies CSV files that meet parameters for Adamatzky's research and wave transform
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def analyze_csv_file(filepath):
    """Analyze a single CSV file for electrical activity parameters"""
    try:
        # Try different reading strategies
        df = None
        
        # Strategy 1: Standard CSV reading
        try:
            df = pd.read_csv(filepath, header=None, low_memory=False)
        except:
            pass
            
        # Strategy 2: Try with comma separator
        if df is None:
            try:
                df = pd.read_csv(filepath, sep=',', header=None, low_memory=False)
            except:
                pass
                
        # Strategy 3: Try with tab separator
        if df is None:
            try:
                df = pd.read_csv(filepath, sep='\t', header=None, low_memory=False)
            except:
                pass
        
        if df is None:
            return None
            
        # Convert to numeric, handling errors
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) == 0:
            return None
            
        # Analyze the first column (voltage data)
        voltage_data = df.iloc[:, 0]
        
        # Calculate parameters
        mean_voltage = float(voltage_data.mean())
        std_voltage = float(voltage_data.std())
        voltage_range = float(voltage_data.max() - voltage_data.min())
        n_samples = int(len(voltage_data))
        
        # Estimate sample rate (assuming 1 second intervals for most files)
        sample_rate = float(n_samples / 60.0)  # Rough estimate
        
        # Calculate signal quality metrics
        signal_variance = float(voltage_data.var())
        peak_to_peak = float(voltage_data.max() - voltage_data.min())
        
        # Check if data meets basic electrical activity criteria
        has_electrical_activity = (
            signal_variance > 0.001 and  # Some signal variation
            peak_to_peak > 0.01 and      # Reasonable voltage range
            n_samples > 1000 and         # Sufficient data points
            std_voltage > 0.01           # Some signal variability
        )
        
        # Check if data meets Adamatzky's parameters
        meets_adamatzky = (
            has_electrical_activity and
            mean_voltage > -1.0 and mean_voltage < 1.0 and  # Reasonable voltage range
            std_voltage > 0.01 and std_voltage < 1.0 and     # Signal variability
            n_samples > 5000              # Sufficient data for analysis
        )
        
        # Check if data meets wave transform parameters
        meets_wave_transform = (
            has_electrical_activity and
            n_samples > 10000 and        # More data needed for wave analysis
            std_voltage > 0.02 and       # Higher signal variability for wave detection
            peak_to_peak > 0.05          # Larger voltage range for wave patterns
        )
        
        return {
            'file': filepath,
            'filename': os.path.basename(filepath),
            'n_samples': n_samples,
            'n_columns': int(len(df.columns)),
            'mean_voltage': mean_voltage,
            'std_voltage': std_voltage,
            'voltage_range': voltage_range,
            'peak_to_peak': peak_to_peak,
            'signal_variance': signal_variance,
            'estimated_sample_rate': sample_rate,
            'has_electrical_activity': has_electrical_activity,
            'meets_adamatzky': meets_adamatzky,
            'meets_wave_transform': meets_wave_transform,
            'quality_score': float(calculate_quality_score(voltage_data))
        }
        
    except Exception as e:
        return {
            'file': filepath,
            'filename': os.path.basename(filepath),
            'error': str(e),
            'has_electrical_activity': False,
            'meets_adamatzky': False,
            'meets_wave_transform': False
        }

def calculate_quality_score(voltage_data):
    """Calculate a quality score for the electrical signal"""
    try:
        # Signal-to-noise ratio approximation
        signal_power = np.var(voltage_data)
        noise_power = np.var(np.diff(voltage_data)) / 2
        snr = signal_power / (noise_power + 1e-10)
        
        # Normalize SNR to 0-1 range
        snr_score = min(1.0, snr / 10.0)
        
        # Data length score
        length_score = min(1.0, len(voltage_data) / 50000.0)
        
        # Signal variability score
        variability_score = min(1.0, np.std(voltage_data) / 0.5)
        
        # Overall quality score
        quality_score = (snr_score + length_score + variability_score) / 3.0
        
        return quality_score
    except:
        return 0.0

def main():
    """Main analysis function"""
    print("🔍 ANALYZING CSV FILES FOR FUNGAL ELECTRICAL ACTIVITY PARAMETERS")
    print("=" * 80)
    
    # Define directories to search
    directories = [
        '15061491/',
        '15061491/fungal_spikes/good_recordings/',
        'csv_data/',
        'fungal_analysis_project/data/'
    ]
    
    all_results = []
    valid_files = []
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\n📁 Scanning directory: {directory}")
            
            for filename in os.listdir(directory):
                if filename.endswith('.csv'):
                    filepath = os.path.join(directory, filename)
                    print(f"  Analyzing: {filename}")
                    
                    result = analyze_csv_file(filepath)
                    if result:
                        all_results.append(result)
                        
                        if result.get('has_electrical_activity', False):
                            valid_files.append(result)
    
    # Sort results by quality score
    valid_files.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\n📈 Total files analyzed: {len(all_results)}")
    print(f"✅ Files with electrical activity: {len([f for f in all_results if f.get('has_electrical_activity', False)])}")
    print(f"🎯 Files meeting Adamatzky parameters: {len([f for f in all_results if f.get('meets_adamatzky', False)])}")
    print(f"🌊 Files meeting wave transform parameters: {len([f for f in all_results if f.get('meets_wave_transform', False)])}")
    
    # Show top files
    print(f"\n🏆 TOP 10 FILES BY QUALITY SCORE:")
    print("-" * 80)
    
    for i, file in enumerate(valid_files[:10]):
        print(f"{i+1:2d}. {file['filename']}")
        print(f"    Quality Score: {file.get('quality_score', 0):.3f}")
        print(f"    Samples: {file.get('n_samples', 0):,}")
        print(f"    Mean Voltage: {file.get('mean_voltage', 0):.4f}V")
        print(f"    Std Voltage: {file.get('std_voltage', 0):.4f}V")
        print(f"    Adamatzky: {'✅' if file.get('meets_adamatzky', False) else '❌'}")
        print(f"    Wave Transform: {'✅' if file.get('meets_wave_transform', False) else '❌'}")
        print()
    
    # Show files meeting both criteria
    both_criteria = [f for f in valid_files if f.get('meets_adamatzky', False) and f.get('meets_wave_transform', False)]
    
    print(f"\n🎯 FILES MEETING BOTH ADAMATZKY AND WAVE TRANSFORM PARAMETERS:")
    print("-" * 80)
    
    if both_criteria:
        for i, file in enumerate(both_criteria):
            print(f"{i+1:2d}. {file['filename']}")
            print(f"    Quality Score: {file.get('quality_score', 0):.3f}")
            print(f"    Samples: {file.get('n_samples', 0):,}")
            print(f"    Voltage Range: {file.get('voltage_range', 0):.4f}V")
            print(f"    Signal Variance: {file.get('signal_variance', 0):.6f}")
            print()
    else:
        print("❌ No files meet both criteria")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"csv_parameter_analysis_{timestamp}.json"
    
    # Convert numpy types before saving
    results_data = {
        'analysis_timestamp': timestamp,
        'total_files': len(all_results),
        'valid_files': len(valid_files),
        'meets_adamatzky': len([f for f in all_results if f.get('meets_adamatzky', False)]),
        'meets_wave_transform': len([f for f in all_results if f.get('meets_wave_transform', False)]),
        'meets_both': len(both_criteria),
        'top_files': convert_numpy_types(valid_files[:10]),
        'both_criteria_files': convert_numpy_types(both_criteria),
        'all_results': convert_numpy_types(all_results)
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return both_criteria

if __name__ == "__main__":
    main() 