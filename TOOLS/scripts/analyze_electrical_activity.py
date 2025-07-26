#!/usr/bin/env python3
"""
Comprehensive Electrical Activity Analysis
Analyzes extracted electrical activity data to identify patterns and insights.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
from collections import defaultdict
import re

warnings.filterwarnings('ignore')

class ElectricalActivityAnalyzer:
    def __init__(self, results_file):
        self.results_file = results_file
        self.results = None
        self.analysis_results = {}
        
    def load_results(self):
        """Load the electrical activity results"""
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        print(f"Loaded {len(self.results)} files from {self.results_file}")
        
    def analyze_file_types(self):
        """Analyze patterns by file type"""
        file_types = defaultdict(list)
        
        for file_name, data in self.results.items():
            file_type = data.get('file_type', 'unknown')
            file_types[file_type].append((file_name, data))
        
        print("\n" + "="*60)
        print("FILE TYPE ANALYSIS")
        print("="*60)
        
        for file_type, files in file_types.items():
            print(f"\n{file_type.upper()} ({len(files)} files):")
            print("-" * 40)
            
            # Calculate statistics for this file type
            sample_counts = [data.get('sample_count', 0) for _, data in files]
            total_samples = sum(sample_counts)
            avg_samples = np.mean(sample_counts)
            
            print(f"Total samples: {total_samples:,}")
            print(f"Average samples per file: {avg_samples:,.0f}")
            print(f"Sample range: {min(sample_counts):,} - {max(sample_counts):,}")
            
            # Show top files by sample count
            files_sorted = sorted(files, key=lambda x: x[1].get('sample_count', 0), reverse=True)
            print(f"Top 5 files by sample count:")
            for i, (name, data) in enumerate(files_sorted[:5]):
                samples = data.get('sample_count', 0)
                print(f"  {i+1}. {name}: {samples:,} samples")
        
        return file_types
    
    def analyze_coordinate_data(self):
        """Analyze coordinate-based electrical signals"""
        coordinate_files = [(name, data) for name, data in self.results.items() 
                           if data.get('file_type') == 'coordinate']
        
        if not coordinate_files:
            return
        
        print("\n" + "="*60)
        print("COORDINATE DATA ELECTRICAL ANALYSIS")
        print("="*60)
        
        # Extract movement-based signals
        movement_metrics = []
        for name, data in coordinate_files:
            metrics = {
                'file': name,
                'samples': data.get('sample_count', 0),
                'distance_range': data.get('distance_range', 0),
                'velocity_rms': data.get('velocity_rms', 0),
                'acceleration_rms': data.get('acceleration_rms', 0),
                'angular_velocity_rms': data.get('angular_velocity_rms', 0),
                'curvature_rms': data.get('curvature_rms', 0)
            }
            movement_metrics.append(metrics)
        
        df = pd.DataFrame(movement_metrics)
        
        # Analyze movement patterns
        print(f"\nMovement Analysis ({len(df)} files):")
        print(f"Average distance range: {df['distance_range'].mean():.4f}")
        print(f"Average velocity RMS: {df['velocity_rms'].mean():.4f}")
        print(f"Average acceleration RMS: {df['acceleration_rms'].mean():.4f}")
        print(f"Average angular velocity RMS: {df['angular_velocity_rms'].mean():.4f}")
        print(f"Average curvature RMS: {df['curvature_rms'].mean():.4f}")
        
        # Find most active files
        print(f"\nTop 10 most active files (by velocity):")
        top_velocity = df.nlargest(10, 'velocity_rms')
        for _, row in top_velocity.iterrows():
            print(f"  {row['file']}: velocity_rms={row['velocity_rms']:.4f}")
        
        # Find files with highest curvature (complex movement)
        print(f"\nTop 10 files with highest curvature (complex movement):")
        top_curvature = df.nlargest(10, 'curvature_rms')
        for _, row in top_curvature.iterrows():
            print(f"  {row['file']}: curvature_rms={row['curvature_rms']:.4f}")
        
        return df
    
    def analyze_direct_electrical_data(self):
        """Analyze direct electrical recordings"""
        electrical_files = [(name, data) for name, data in self.results.items() 
                           if data.get('file_type') in ['direct_electrical', 'fungal_spike']]
        
        if not electrical_files:
            return
        
        print("\n" + "="*60)
        print("DIRECT ELECTRICAL RECORDING ANALYSIS")
        print("="*60)
        
        # Extract voltage metrics
        voltage_metrics = []
        for name, data in electrical_files:
            metrics = {
                'file': name,
                'file_type': data.get('file_type'),
                'samples': data.get('sample_count', 0),
                'voltage_rms': data.get('voltage_rms', 0),
                'voltage_power': data.get('voltage_power', 0),
                'voltage_peak_rate': data.get('voltage_peak_rate', 0),
                'voltage_zero_crossing_rate': data.get('voltage_zero_crossing_rate', 0),
                'voltage_dominant_frequency': data.get('voltage_dominant_frequency', 0),
                'voltage_spectral_power': data.get('voltage_spectral_power', 0)
            }
            voltage_metrics.append(metrics)
        
        df = pd.DataFrame(voltage_metrics)
        
        print(f"\nElectrical Analysis ({len(df)} files):")
        print(f"Average voltage RMS: {df['voltage_rms'].mean():.4f}")
        print(f"Average voltage power: {df['voltage_power'].mean():.4f}")
        print(f"Average peak rate: {df['voltage_peak_rate'].mean():.4f}")
        print(f"Average zero crossing rate: {df['voltage_zero_crossing_rate'].mean():.4f}")
        print(f"Average dominant frequency: {df['voltage_dominant_frequency'].mean():.4f}")
        
        # Find files with highest electrical activity
        print(f"\nTop 10 files by voltage RMS:")
        top_voltage = df.nlargest(10, 'voltage_rms')
        for _, row in top_voltage.iterrows():
            print(f"  {row['file']}: RMS={row['voltage_rms']:.4f}, peaks={row['voltage_peak_rate']:.4f}")
        
        # Find files with highest spike activity
        print(f"\nTop 10 files by peak rate (spike activity):")
        top_peaks = df.nlargest(10, 'voltage_peak_rate')
        for _, row in top_peaks.iterrows():
            print(f"  {row['file']}: peak_rate={row['voltage_peak_rate']:.4f}, RMS={row['voltage_rms']:.4f}")
        
        # Analyze by file type
        print(f"\nAnalysis by file type:")
        for file_type in df['file_type'].unique():
            type_data = df[df['file_type'] == file_type]
            print(f"  {file_type} ({len(type_data)} files):")
            print(f"    Avg RMS: {type_data['voltage_rms'].mean():.4f}")
            print(f"    Avg peak rate: {type_data['voltage_peak_rate'].mean():.4f}")
            print(f"    Avg dominant freq: {type_data['voltage_dominant_frequency'].mean():.4f}")
        
        return df
    
    def analyze_moisture_data(self):
        """Analyze moisture data correlations"""
        moisture_files = [(name, data) for name, data in self.results.items() 
                         if data.get('file_type') == 'moisture']
        
        if not moisture_files:
            return
        
        print("\n" + "="*60)
        print("MOISTURE DATA ANALYSIS")
        print("="*60)
        
        # Extract moisture metrics
        moisture_metrics = []
        for name, data in moisture_files:
            metrics = {
                'file': name,
                'samples': data.get('sample_count', 0),
                'moisture_mean': data.get('moisture_mean', 0),
                'moisture_std': data.get('moisture_std', 0),
                'moisture_range': data.get('moisture_range', 0),
                'moisture_gradient_rms': data.get('moisture_gradient_rms', 0),
                'moisture_acceleration_rms': data.get('moisture_acceleration_rms', 0)
            }
            moisture_metrics.append(metrics)
        
        df = pd.DataFrame(moisture_metrics)
        
        print(f"\nMoisture Analysis ({len(df)} files):")
        print(f"Average moisture level: {df['moisture_mean'].mean():.4f}")
        print(f"Average moisture variability: {df['moisture_std'].mean():.4f}")
        print(f"Average moisture range: {df['moisture_range'].mean():.4f}")
        print(f"Average moisture gradient RMS: {df['moisture_gradient_rms'].mean():.4f}")
        
        # Find files with highest moisture variability
        print(f"\nTop 10 files by moisture variability:")
        top_variability = df.nlargest(10, 'moisture_std')
        for _, row in top_variability.iterrows():
            print(f"  {row['file']}: std={row['moisture_std']:.4f}, range={row['moisture_range']:.4f}")
        
        return df
    
    def analyze_species_patterns(self):
        """Analyze patterns by fungal species"""
        print("\n" + "="*60)
        print("SPECIES PATTERN ANALYSIS")
        print("="*60)
        
        # Extract species from filenames
        species_data = defaultdict(list)
        
        for file_name, data in self.results.items():
            # Extract species code from filename
            species_match = re.match(r'([A-Za-z]+)_', file_name)
            if species_match:
                species = species_match.group(1)
                species_data[species].append((file_name, data))
        
        print(f"\nSpecies Analysis ({len(species_data)} species):")
        
        for species, files in species_data.items():
            print(f"\n{species} ({len(files)} files):")
            
            # Calculate species statistics
            sample_counts = [data.get('sample_count', 0) for _, data in files]
            total_samples = sum(sample_counts)
            
            # Get electrical metrics if available
            electrical_files = [(name, data) for name, data in files 
                              if data.get('file_type') in ['direct_electrical', 'fungal_spike']]
            
            if electrical_files:
                voltage_rms_values = [data.get('voltage_rms', 0) for _, data in electrical_files]
                peak_rates = [data.get('voltage_peak_rate', 0) for _, data in electrical_files]
                
                print(f"  Total samples: {total_samples:,}")
                print(f"  Electrical files: {len(electrical_files)}")
                print(f"  Avg voltage RMS: {np.mean(voltage_rms_values):.4f}")
                print(f"  Avg peak rate: {np.mean(peak_rates):.4f}")
                
                # Show top electrical files for this species
                electrical_files_sorted = sorted(electrical_files, 
                                              key=lambda x: x[1].get('voltage_rms', 0), 
                                              reverse=True)
                print(f"  Top electrical files:")
                for name, data in electrical_files_sorted[:3]:
                    print(f"    {name}: RMS={data.get('voltage_rms', 0):.4f}")
            else:
                print(f"  Total samples: {total_samples:,}")
                print(f"  No direct electrical recordings")
    
    def generate_correlation_analysis(self):
        """Generate correlation analysis between different metrics"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Create combined dataset
        all_metrics = []
        
        for file_name, data in self.results.items():
            metrics = {
                'file': file_name,
                'file_type': data.get('file_type'),
                'samples': data.get('sample_count', 0),
                'voltage_rms': data.get('voltage_rms', 0),
                'voltage_peak_rate': data.get('voltage_peak_rate', 0),
                'voltage_zero_crossing_rate': data.get('voltage_zero_crossing_rate', 0),
                'voltage_dominant_frequency': data.get('voltage_dominant_frequency', 0),
                'distance_range': data.get('distance_range', 0),
                'velocity_rms': data.get('velocity_rms', 0),
                'acceleration_rms': data.get('acceleration_rms', 0),
                'moisture_mean': data.get('moisture_mean', 0),
                'moisture_std': data.get('moisture_std', 0)
            }
            all_metrics.append(metrics)
        
        df = pd.DataFrame(all_metrics)
        
        # Calculate correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        print(f"\nCorrelation Matrix (top correlations):")
        
        # Find strongest correlations
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.3:  # Only show significant correlations
                    correlations.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_value
                    ))
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for var1, var2, corr in correlations[:10]:
            print(f"  {var1} ↔ {var2}: {corr:.3f}")
        
        return df, correlation_matrix
    
    def generate_insights_report(self):
        """Generate comprehensive insights report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE INSIGHTS REPORT")
        print("="*60)
        
        insights = []
        
        # File type distribution
        file_types = defaultdict(int)
        for data in self.results.values():
            file_types[data.get('file_type', 'unknown')] += 1
        
        insights.append(f"Data Distribution:")
        insights.append(f"  - Coordinate files: {file_types['coordinate']} files")
        insights.append(f"  - Direct electrical: {file_types['direct_electrical']} files")
        insights.append(f"  - Fungal spike: {file_types['fungal_spike']} files")
        insights.append(f"  - Moisture: {file_types['moisture']} files")
        
        # Sample size analysis
        sample_counts = [data.get('sample_count', 0) for data in self.results.values()]
        insights.append(f"\nSample Size Analysis:")
        insights.append(f"  - Total samples: {sum(sample_counts):,}")
        insights.append(f"  - Average samples per file: {np.mean(sample_counts):,.0f}")
        insights.append(f"  - Largest file: {max(sample_counts):,} samples")
        insights.append(f"  - Smallest file: {min(sample_counts):,} samples")
        
        # Electrical activity insights
        electrical_files = [data for data in self.results.values() 
                          if data.get('file_type') in ['direct_electrical', 'fungal_spike']]
        
        if electrical_files:
            voltage_rms_values = [data.get('voltage_rms', 0) for data in electrical_files]
            peak_rates = [data.get('voltage_peak_rate', 0) for data in electrical_files]
            
            insights.append(f"\nElectrical Activity Insights:")
            insights.append(f"  - Files with electrical data: {len(electrical_files)}")
            insights.append(f"  - Average voltage RMS: {np.mean(voltage_rms_values):.4f}")
            insights.append(f"  - Average spike rate: {np.mean(peak_rates):.4f}")
            insights.append(f"  - Highest voltage RMS: {max(voltage_rms_values):.4f}")
            insights.append(f"  - Highest spike rate: {max(peak_rates):.4f}")
        
        # Movement insights
        coordinate_files = [data for data in self.results.values() 
                          if data.get('file_type') == 'coordinate']
        
        if coordinate_files:
            velocity_rms_values = [data.get('velocity_rms', 0) for data in coordinate_files]
            curvature_rms_values = [data.get('curvature_rms', 0) for data in coordinate_files]
            
            insights.append(f"\nMovement Pattern Insights:")
            insights.append(f"  - Coordinate files: {len(coordinate_files)}")
            insights.append(f"  - Average velocity RMS: {np.mean(velocity_rms_values):.4f}")
            insights.append(f"  - Average curvature RMS: {np.mean(curvature_rms_values):.4f}")
            insights.append(f"  - Most active movement: {max(velocity_rms_values):.4f}")
            insights.append(f"  - Most complex movement: {max(curvature_rms_values):.4f}")
        
        # Print insights
        for insight in insights:
            print(insight)
        
        return insights
    
    def save_analysis_results(self, output_file=None):
        """Save analysis results to JSON file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"electrical_analysis_results_{timestamp}.json"
        
        analysis_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_files': len(self.results),
            'file_type_distribution': {},
            'top_electrical_files': [],
            'top_movement_files': [],
            'species_analysis': {},
            'correlation_insights': []
        }
        
        # File type distribution
        file_types = defaultdict(int)
        for data in self.results.values():
            file_types[data.get('file_type', 'unknown')] += 1
        analysis_data['file_type_distribution'] = dict(file_types)
        
        # Top electrical files
        electrical_files = [(name, data) for name, data in self.results.items() 
                           if data.get('file_type') in ['direct_electrical', 'fungal_spike']]
        electrical_files.sort(key=lambda x: x[1].get('voltage_rms', 0), reverse=True)
        analysis_data['top_electrical_files'] = [
            {'file': name, 'voltage_rms': data.get('voltage_rms', 0), 
             'peak_rate': data.get('voltage_peak_rate', 0)}
            for name, data in electrical_files[:10]
        ]
        
        # Top movement files
        coordinate_files = [(name, data) for name, data in self.results.items() 
                           if data.get('file_type') == 'coordinate']
        coordinate_files.sort(key=lambda x: x[1].get('velocity_rms', 0), reverse=True)
        analysis_data['top_movement_files'] = [
            {'file': name, 'velocity_rms': data.get('velocity_rms', 0), 
             'curvature_rms': data.get('curvature_rms', 0)}
            for name, data in coordinate_files[:10]
        ]
        
        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\nAnalysis results saved to: {output_file}")
        return output_file
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive electrical activity analysis...")
        
        # Load results
        self.load_results()
        
        # Run all analyses
        file_types = self.analyze_file_types()
        coordinate_df = self.analyze_coordinate_data()
        electrical_df = self.analyze_direct_electrical_data()
        moisture_df = self.analyze_moisture_data()
        self.analyze_species_patterns()
        correlation_df, correlation_matrix = self.generate_correlation_analysis()
        insights = self.generate_insights_report()
        
        # Save results
        output_file = self.save_analysis_results()
        
        print(f"\nAnalysis complete! Results saved to: {output_file}")
        
        return {
            'file_types': file_types,
            'coordinate_df': coordinate_df,
            'electrical_df': electrical_df,
            'moisture_df': moisture_df,
            'correlation_df': correlation_df,
            'correlation_matrix': correlation_matrix,
            'insights': insights,
            'output_file': output_file
        }

def main():
    """Main function to run the analysis"""
    # Find the most recent results file
    results_files = list(Path('.').glob('electrical_activity_results_*.json'))
    if not results_files:
        print("No electrical activity results files found!")
        return
    
    # Use the most recent file
    latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
    print(f"Using results file: {latest_file}")
    
    # Run analysis
    analyzer = ElectricalActivityAnalyzer(latest_file)
    results = analyzer.run_complete_analysis()
    
    return results

if __name__ == "__main__":
    main() 