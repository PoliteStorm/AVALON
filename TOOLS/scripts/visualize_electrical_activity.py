#!/usr/bin/env python3
"""
Electrical Activity Visualization
Generates comprehensive plots and charts from electrical activity analysis.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import re

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ElectricalActivityVisualizer:
    def __init__(self, results_file, analysis_file=None):
        self.results_file = results_file
        self.analysis_file = analysis_file
        self.results = None
        self.analysis = None
        
    def load_data(self):
        """Load the electrical activity data"""
        # Load results
        with open(self.results_file, 'r') as f:
            self.results = json.load(f)
        
        # Load analysis if available
        if self.analysis_file and Path(self.analysis_file).exists():
            with open(self.analysis_file, 'r') as f:
                self.analysis = json.load(f)
        
        print(f"Loaded {len(self.results)} files for visualization")
    
    def create_file_type_distribution_plot(self):
        """Create file type distribution plot"""
        file_types = {}
        for data in self.results.values():
            file_type = data.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        labels = list(file_types.keys())
        sizes = list(file_types.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('File Type Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(labels, sizes, color=colors)
        ax2.set_title('File Count by Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Files')
        
        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(size), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('file_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_sample_size_analysis(self):
        """Create sample size analysis plots"""
        sample_counts = [data.get('sample_count', 0) for data in self.results.values()]
        file_types = [data.get('file_type', 'unknown') for data in self.results.values()]
        
        df = pd.DataFrame({
            'sample_count': sample_counts,
            'file_type': file_types
        })
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram of sample counts
        ax1.hist(sample_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Sample Count')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Sample Counts', fontweight='bold')
        ax1.set_xscale('log')
        
        # Box plot by file type
        df.boxplot(column='sample_count', by='file_type', ax=ax2)
        ax2.set_title('Sample Count by File Type', fontweight='bold')
        ax2.set_xlabel('File Type')
        ax2.set_ylabel('Sample Count (log scale)')
        ax2.set_yscale('log')
        
        # Top 20 files by sample count
        file_sample_pairs = [(name, data.get('sample_count', 0)) 
                            for name, data in self.results.items()]
        file_sample_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_files = file_sample_pairs[:20]
        file_names = [name[:30] + '...' if len(name) > 30 else name for name, _ in top_files]
        sample_counts_top = [count for _, count in top_files]
        
        bars = ax3.barh(range(len(file_names)), sample_counts_top, color='lightcoral')
        ax3.set_yticks(range(len(file_names)))
        ax3.set_yticklabels(file_names)
        ax3.set_xlabel('Sample Count')
        ax3.set_title('Top 20 Files by Sample Count', fontweight='bold')
        ax3.set_xscale('log')
        
        # Sample count statistics
        stats_text = f"""
        Total Files: {len(self.results)}
        Total Samples: {sum(sample_counts):,}
        Average Samples: {np.mean(sample_counts):,.0f}
        Median Samples: {np.median(sample_counts):,.0f}
        Max Samples: {max(sample_counts):,}
        Min Samples: {min(sample_counts):,}
        """
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightblue", alpha=0.7))
        ax4.set_title('Sample Count Statistics', fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_size_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_electrical_activity_plots(self):
        """Create electrical activity analysis plots"""
        electrical_files = [(name, data) for name, data in self.results.items() 
                           if data.get('file_type') in ['direct_electrical', 'fungal_spike']]
        
        if not electrical_files:
            print("No electrical files found for plotting")
            return
        
        # Extract electrical metrics
        electrical_data = []
        for name, data in electrical_files:
            electrical_data.append({
                'file': name,
                'file_type': data.get('file_type'),
                'samples': data.get('sample_count', 0),
                'voltage_rms': data.get('voltage_rms', 0),
                'voltage_power': data.get('voltage_power', 0),
                'voltage_peak_rate': data.get('voltage_peak_rate', 0),
                'voltage_zero_crossing_rate': data.get('voltage_zero_crossing_rate', 0),
                'voltage_dominant_frequency': data.get('voltage_dominant_frequency', 0)
            })
        
        df = pd.DataFrame(electrical_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Voltage RMS distribution
        ax1.hist(df['voltage_rms'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        ax1.set_xlabel('Voltage RMS')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Voltage RMS', fontweight='bold')
        ax1.set_xscale('log')
        
        # Peak rate vs Voltage RMS scatter
        ax2.scatter(df['voltage_rms'], df['voltage_peak_rate'], 
                   c=df['file_type'].map({'direct_electrical': 'blue', 'fungal_spike': 'red'}),
                   alpha=0.7, s=100)
        ax2.set_xlabel('Voltage RMS')
        ax2.set_ylabel('Peak Rate')
        ax2.set_title('Peak Rate vs Voltage RMS', fontweight='bold')
        ax2.set_xscale('log')
        ax2.legend(['Direct Electrical', 'Fungal Spike'])
        
        # Top electrical files
        top_electrical = df.nlargest(10, 'voltage_rms')
        bars = ax3.barh(range(len(top_electrical)), top_electrical['voltage_rms'], 
                        color='gold', alpha=0.7)
        ax3.set_yticks(range(len(top_electrical)))
        ax3.set_yticklabels([name[:25] + '...' if len(name) > 25 else name 
                            for name in top_electrical['file']])
        ax3.set_xlabel('Voltage RMS')
        ax3.set_title('Top 10 Files by Voltage RMS', fontweight='bold')
        ax3.set_xscale('log')
        
        # Peak rate distribution
        ax4.hist(df['voltage_peak_rate'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xlabel('Peak Rate')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Peak Rates', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('electrical_activity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_movement_analysis_plots(self):
        """Create movement pattern analysis plots"""
        coordinate_files = [(name, data) for name, data in self.results.items() 
                           if data.get('file_type') == 'coordinate']
        
        if not coordinate_files:
            print("No coordinate files found for plotting")
            return
        
        # Extract movement metrics
        movement_data = []
        for name, data in coordinate_files:
            movement_data.append({
                'file': name,
                'samples': data.get('sample_count', 0),
                'distance_range': data.get('distance_range', 0),
                'velocity_rms': data.get('velocity_rms', 0),
                'acceleration_rms': data.get('acceleration_rms', 0),
                'angular_velocity_rms': data.get('angular_velocity_rms', 0),
                'curvature_rms': data.get('curvature_rms', 0)
            })
        
        df = pd.DataFrame(movement_data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Velocity distribution
        ax1.hist(df['velocity_rms'], bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_xlabel('Velocity RMS')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Movement Velocity', fontweight='bold')
        
        # Velocity vs Acceleration scatter
        ax2.scatter(df['velocity_rms'], df['acceleration_rms'], alpha=0.6, color='purple')
        ax2.set_xlabel('Velocity RMS')
        ax2.set_ylabel('Acceleration RMS')
        ax2.set_title('Velocity vs Acceleration', fontweight='bold')
        
        # Top movement files
        top_movement = df.nlargest(10, 'velocity_rms')
        bars = ax3.barh(range(len(top_movement)), top_movement['velocity_rms'], 
                        color='lightcoral', alpha=0.7)
        ax3.set_yticks(range(len(top_movement)))
        ax3.set_yticklabels([name[:25] + '...' if len(name) > 25 else name 
                            for name in top_movement['file']])
        ax3.set_xlabel('Velocity RMS')
        ax3.set_title('Top 10 Files by Movement Velocity', fontweight='bold')
        
        # Curvature distribution
        ax4.hist(df['curvature_rms'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_xlabel('Curvature RMS')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Movement Curvature', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('movement_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_species_analysis_plots(self):
        """Create species-specific analysis plots"""
        # Extract species from filenames
        species_data = {}
        
        for file_name, data in self.results.items():
            species_match = re.match(r'([A-Za-z]+)_', file_name)
            if species_match:
                species = species_match.group(1)
                if species not in species_data:
                    species_data[species] = []
                species_data[species].append(data)
        
        if not species_data:
            print("No species data found for plotting")
            return
        
        # Calculate species statistics
        species_stats = []
        for species, files in species_data.items():
            if len(files) >= 3:  # Only include species with 3+ files
                sample_counts = [f.get('sample_count', 0) for f in files]
                voltage_rms_values = [f.get('voltage_rms', 0) for f in files 
                                    if f.get('file_type') in ['direct_electrical', 'fungal_spike']]
                velocity_rms_values = [f.get('velocity_rms', 0) for f in files 
                                     if f.get('file_type') == 'coordinate']
                
                species_stats.append({
                    'species': species,
                    'file_count': len(files),
                    'total_samples': sum(sample_counts),
                    'avg_samples': np.mean(sample_counts),
                    'avg_voltage_rms': np.mean(voltage_rms_values) if voltage_rms_values else 0,
                    'avg_velocity_rms': np.mean(velocity_rms_values) if velocity_rms_values else 0
                })
        
        df = pd.DataFrame(species_stats)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # File count by species
        bars = ax1.bar(df['species'], df['file_count'], color='skyblue', alpha=0.7)
        ax1.set_xlabel('Species')
        ax1.set_ylabel('Number of Files')
        ax1.set_title('File Count by Species', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Total samples by species
        bars = ax2.bar(df['species'], df['total_samples'], color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Species')
        ax2.set_ylabel('Total Samples')
        ax2.set_title('Total Samples by Species', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Average voltage RMS by species
        voltage_data = df[df['avg_voltage_rms'] > 0]
        if len(voltage_data) > 0:
            bars = ax3.bar(voltage_data['species'], voltage_data['avg_voltage_rms'], 
                          color='lightgreen', alpha=0.7)
            ax3.set_xlabel('Species')
            ax3.set_ylabel('Average Voltage RMS')
            ax3.set_title('Average Voltage RMS by Species', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.set_yscale('log')
        
        # Average velocity RMS by species
        velocity_data = df[df['avg_velocity_rms'] > 0]
        if len(velocity_data) > 0:
            bars = ax4.bar(velocity_data['species'], velocity_data['avg_velocity_rms'], 
                          color='gold', alpha=0.7)
            ax4.set_xlabel('Species')
            ax4.set_ylabel('Average Velocity RMS')
            ax4.set_title('Average Velocity RMS by Species', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('species_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap"""
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
        
        # Calculate correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Correlation Matrix of Electrical Activity Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with all key metrics"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. File type distribution (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        file_types = {}
        for data in self.results.values():
            file_type = data.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        labels = list(file_types.keys())
        sizes = list(file_types.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('File Type Distribution', fontweight='bold')
        
        # 2. Sample count histogram (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        sample_counts = [data.get('sample_count', 0) for data in self.results.values()]
        ax2.hist(sample_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Sample Count')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Sample Count Distribution', fontweight='bold')
        ax2.set_xscale('log')
        
        # 3. Electrical activity summary (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        electrical_files = [data for data in self.results.values() 
                          if data.get('file_type') in ['direct_electrical', 'fungal_spike']]
        
        if electrical_files:
            voltage_rms_values = [data.get('voltage_rms', 0) for data in electrical_files]
            peak_rates = [data.get('voltage_peak_rate', 0) for data in electrical_files]
            
            ax3.scatter(voltage_rms_values, peak_rates, alpha=0.6, color='red')
            ax3.set_xlabel('Voltage RMS')
            ax3.set_ylabel('Peak Rate')
            ax3.set_title('Electrical Activity: Voltage vs Peak Rate', fontweight='bold')
            ax3.set_xscale('log')
        else:
            ax3.text(0.5, 0.5, 'No electrical data available', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Electrical Activity', fontweight='bold')
        
        # 4. Movement activity summary (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        coordinate_files = [data for data in self.results.values() 
                          if data.get('file_type') == 'coordinate']
        
        if coordinate_files:
            velocity_rms_values = [data.get('velocity_rms', 0) for data in coordinate_files]
            curvature_rms_values = [data.get('curvature_rms', 0) for data in coordinate_files]
            
            ax4.scatter(velocity_rms_values, curvature_rms_values, alpha=0.6, color='blue')
            ax4.set_xlabel('Velocity RMS')
            ax4.set_ylabel('Curvature RMS')
            ax4.set_title('Movement Activity: Velocity vs Curvature', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No coordinate data available', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Movement Activity', fontweight='bold')
        
        # 5. Summary statistics (bottom)
        ax5 = fig.add_subplot(gs[2:, :])
        
        stats_text = f"""
        COMPREHENSIVE ELECTRICAL ACTIVITY DASHBOARD
        
        DATA OVERVIEW:
        • Total Files: {len(self.results):,}
        • Total Samples: {sum(sample_counts):,}
        • Average Samples per File: {np.mean(sample_counts):,.0f}
        
        FILE TYPE BREAKDOWN:
        • Coordinate Files: {file_types.get('coordinate', 0)} files
        • Direct Electrical: {file_types.get('direct_electrical', 0)} files
        • Fungal Spike: {file_types.get('fungal_spike', 0)} files
        • Moisture: {file_types.get('moisture', 0)} files
        
        ELECTRICAL ACTIVITY:
        • Files with Electrical Data: {len(electrical_files)}
        • Average Voltage RMS: {np.mean([data.get('voltage_rms', 0) for data in electrical_files]):.2f}
        • Average Peak Rate: {np.mean([data.get('voltage_peak_rate', 0) for data in electrical_files]):.4f}
        
        MOVEMENT PATTERNS:
        • Coordinate Files: {len(coordinate_files)}
        • Average Velocity RMS: {np.mean([data.get('velocity_rms', 0) for data in coordinate_files]):.2f}
        • Average Curvature RMS: {np.mean([data.get('curvature_rms', 0) for data in coordinate_files]):.2f}
        
        ANALYSIS TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightblue", alpha=0.8))
        ax5.set_title('Electrical Activity Analysis Summary', fontsize=16, fontweight='bold')
        ax5.axis('off')
        
        plt.tight_layout()
        plt.savefig('electrical_activity_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_visualizations(self):
        """Generate all visualization plots"""
        print("Generating comprehensive electrical activity visualizations...")
        
        self.load_data()
        
        # Generate all plots
        self.create_file_type_distribution_plot()
        self.create_sample_size_analysis()
        self.create_electrical_activity_plots()
        self.create_movement_analysis_plots()
        self.create_species_analysis_plots()
        self.create_correlation_heatmap()
        self.create_comprehensive_dashboard()
        
        print("All visualizations completed and saved!")

def main():
    """Main function to run the visualization"""
    # Find the most recent results file
    results_files = list(Path('.').glob('electrical_activity_results_*.json'))
    if not results_files:
        print("No electrical activity results files found!")
        return
    
    # Use the most recent file
    latest_file = max(results_files, key=lambda x: x.stat().st_mtime)
    print(f"Using results file: {latest_file}")
    
    # Find analysis file if available
    analysis_files = list(Path('.').glob('electrical_analysis_results_*.json'))
    latest_analysis = max(analysis_files, key=lambda x: x.stat().st_mtime) if analysis_files else None
    
    # Run visualization
    visualizer = ElectricalActivityVisualizer(latest_file, latest_analysis)
    visualizer.generate_all_visualizations()
    
    return visualizer

if __name__ == "__main__":
    main() 