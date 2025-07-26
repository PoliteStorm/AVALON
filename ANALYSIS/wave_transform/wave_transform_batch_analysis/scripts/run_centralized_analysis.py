#!/usr/bin/env python3
"""
Centralized Analysis Runner
Master script that uses centralized configuration to eliminate forced parameters

This script ensures:
- All parameters come from centralized configuration
- Dynamic adaptation based on data characteristics
- Consistent Adamatzky parameters across all analyses
- Professional documentation and reporting
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import warnings

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

# Import centralized configuration
sys.path.append(str(Path(__file__).parent.parent / "config"))
from analysis_config import config

# Import analysis modules
from enhanced_adamatzky_processor import EnhancedAdamatzkyProcessor
from comprehensive_wave_transform_analysis import ComprehensiveWaveTransformAnalyzer
from comprehensive_wave_transform_validation import ComprehensiveWaveTransformValidator

warnings.filterwarnings('ignore')

class CentralizedAnalysisRunner:
    """
    Centralized analysis runner using configuration-driven parameters
    """
    
    def __init__(self):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.processor = EnhancedAdamatzkyProcessor()
        self.analyzer = ComprehensiveWaveTransformAnalyzer()
        self.validator = ComprehensiveWaveTransformValidator()
        
        # Get output directories
        self.output_dirs = self.config.get_output_dirs()
        
        print("🎯 CENTRALIZED ANALYSIS RUNNER")
        print("=" * 60)
        print("Using configuration-driven parameters (no forced parameters)")
        
        # Validate configuration
        validation_results = self.config.validate_config()
        if not validation_results['is_valid']:
            print("❌ Configuration validation failed:")
            for issue in validation_results['issues']:
                print(f"   - {issue}")
            return
        else:
            print("✅ Configuration validation passed")
    
    def run_comprehensive_analysis(self, data_dir: str = "../data/processed"):
        """
        Run comprehensive analysis using centralized configuration
        
        Args:
            data_dir: Directory containing CSV files to analyze
        """
        print(f"\n🚀 RUNNING COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"❌ Data directory not found: {data_dir}")
            return
        
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            print(f"❌ No CSV files found in {data_dir}")
            return
        
        print(f"📁 Found {len(csv_files)} CSV files to analyze")
        
        all_results = {}
        
        for csv_file in csv_files:
            print(f"\n🔬 Processing: {csv_file.name}")
            print("-" * 40)
            
            try:
                # Process with enhanced Adamatzky processor
                results = self.processor.process_single_file(str(csv_file))
                if results:
                    all_results[csv_file.name] = results
                    print(f"✅ Successfully processed {csv_file.name}")
                else:
                    print(f"❌ Failed to process {csv_file.name}")
                    
            except Exception as e:
                print(f"❌ Error processing {csv_file.name}: {e}")
        
        # Create comprehensive summary
        if all_results:
            summary = self.create_comprehensive_summary(all_results)
            self.save_summary_report(summary)
            print(f"\n📊 Analysis complete! Processed {len(all_results)} files")
        else:
            print("\n❌ No files were successfully processed")
    
    def create_comprehensive_summary(self, all_results: dict) -> dict:
        """
        Create comprehensive summary of all analysis results
        
        Args:
            all_results: Dictionary of analysis results
            
        Returns:
            Comprehensive summary dictionary
        """
        print(f"\n📋 CREATING COMPREHENSIVE SUMMARY")
        print("=" * 60)
        
        summary = {
            'timestamp': self.timestamp,
            'configuration': {
                'adamatzky_parameters': self.config.get_adamatzky_params(),
                'validation_thresholds': self.config.get_validation_thresholds(),
                'wave_transform_parameters': self.config.get_wave_transform_params(),
                'compression_settings': self.config.config['compression_settings']
            },
            'analysis_summary': {
                'total_files': len(all_results),
                'files_processed': list(all_results.keys()),
                'total_features': 0,
                'average_validation_score': 0.0,
                'compression_factors_used': [],
                'temporal_scale_distribution': {}
            },
            'file_results': {}
        }
        
        # Aggregate results
        total_features = 0
        validation_scores = []
        compression_factors = []
        all_temporal_scales = []
        
        for filename, results in all_results.items():
            # Extract file-specific results
            file_summary = {
                'filename': filename,
                'metadata': results.get('metadata', {}),
                'wave_features': results.get('wave_features', {}),
                'validation_results': results.get('validation_results', {}),
                'visualization_paths': results.get('plot_paths', {})
            }
            
            summary['file_results'][filename] = file_summary
            
            # Aggregate statistics
            if 'wave_features' in results:
                features = results['wave_features']
                total_features += features.get('n_features', 0)
                all_temporal_scales.extend(features.get('temporal_scale_distribution', []))
            
            if 'metadata' in results:
                compression_factors.append(results['metadata'].get('compression_factor', 0))
            
            if 'validation_results' in results:
                validation_scores.append(results['validation_results'].get('overall_score', 0.0))
        
        # Calculate summary statistics
        summary['analysis_summary']['total_features'] = total_features
        summary['analysis_summary']['average_validation_score'] = (
            sum(validation_scores) / len(validation_scores) if validation_scores else 0.0
        )
        summary['analysis_summary']['compression_factors_used'] = list(set(compression_factors))
        
        # Temporal scale distribution
        if all_temporal_scales:
            from collections import Counter
            scale_counts = Counter(all_temporal_scales)
            total_scales = len(all_temporal_scales)
            summary['analysis_summary']['temporal_scale_distribution'] = {
                scale: {
                    'count': count,
                    'percentage': (count / total_scales) * 100
                }
                for scale, count in scale_counts.items()
            }
        
        return summary
    
    def save_summary_report(self, summary: dict):
        """
        Save comprehensive summary report
        
        Args:
            summary: Summary dictionary to save
        """
        # Save JSON summary
        summary_file = self.output_dirs['reports'] / f"comprehensive_summary_{self.timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create markdown report
        md_file = self.output_dirs['reports'] / f"comprehensive_summary_{self.timestamp}.md"
        self.create_markdown_report(summary, md_file)
        
        print(f"📄 Summary reports saved:")
        print(f"   JSON: {summary_file}")
        print(f"   Markdown: {md_file}")
    
    def create_markdown_report(self, summary: dict, output_file: Path):
        """
        Create markdown report from summary
        
        Args:
            summary: Summary dictionary
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            f.write("# Comprehensive Wave Transform Analysis Report\n\n")
            f.write(f"**Generated:** {summary['timestamp']}\n\n")
            f.write("## Configuration Summary\n\n")
            
            # Configuration details
            config = summary['configuration']
            f.write("### Adamatzky Parameters\n")
            f.write(f"- Sampling Rate: {config['adamatzky_parameters']['sampling_rate']} Hz\n")
            f.write(f"- Voltage Range: {config['adamatzky_parameters']['voltage_range']['min']} to {config['adamatzky_parameters']['voltage_range']['max']} mV\n")
            f.write(f"- Wave Transform Formula: {config['adamatzky_parameters']['wave_transform_formula']}\n\n")
            
            f.write("### Compression Settings\n")
            comp_settings = config['compression_settings']
            f.write(f"- Adaptive Compression: {comp_settings['adaptive_compression']}\n")
            f.write(f"- Target Samples: {comp_settings['target_samples']}\n")
            f.write(f"- Fallback Compression: {comp_settings['fallback_compression']}x\n\n")
            
            f.write("### Validation Thresholds\n")
            thresholds = config['validation_thresholds']
            for threshold_name, threshold_value in thresholds.items():
                f.write(f"- {threshold_name.replace('_', ' ').title()}: {threshold_value}\n")
            f.write("\n")
            
            # Analysis results
            analysis = summary['analysis_summary']
            f.write("## Analysis Results\n\n")
            f.write(f"- **Total Files Processed:** {analysis['total_files']}\n")
            f.write(f"- **Total Features Detected:** {analysis['total_features']}\n")
            f.write(f"- **Average Validation Score:** {analysis['average_validation_score']:.3f}\n")
            f.write(f"- **Compression Factors Used:** {analysis['compression_factors_used']}\n\n")
            
            # Temporal scale distribution
            if analysis['temporal_scale_distribution']:
                f.write("### Temporal Scale Distribution\n\n")
                for scale, stats in analysis['temporal_scale_distribution'].items():
                    f.write(f"- **{scale.replace('_', ' ').title()}:** {stats['count']} features ({stats['percentage']:.1f}%)\n")
                f.write("\n")
            
            # File results
            f.write("## Individual File Results\n\n")
            for filename, file_results in summary['file_results'].items():
                f.write(f"### {filename}\n")
                metadata = file_results.get('metadata', {})
                f.write(f"- Original Samples: {metadata.get('original_samples', 'N/A')}\n")
                f.write(f"- Compressed Samples: {metadata.get('compressed_samples', 'N/A')}\n")
                f.write(f"- Compression Factor: {metadata.get('compression_factor', 'N/A')}x\n")
                
                wave_features = file_results.get('wave_features', {})
                f.write(f"- Features Detected: {wave_features.get('n_features', 0)}\n")
                f.write(f"- Max Magnitude: {wave_features.get('max_magnitude', 0):.3f}\n")
                f.write(f"- Avg Magnitude: {wave_features.get('avg_magnitude', 0):.3f}\n\n")
    
    def print_configuration_summary(self):
        """Print current configuration summary"""
        print(f"\n📋 CONFIGURATION SUMMARY")
        print("=" * 60)
        
        adamatzky_params = self.config.get_adamatzky_params()
        print(f"Adamatzky Parameters:")
        print(f"  - Sampling Rate: {adamatzky_params['sampling_rate']} Hz")
        print(f"  - Voltage Range: {adamatzky_params['voltage_range']['min']} to {adamatzky_params['voltage_range']['max']} mV")
        print(f"  - Wave Transform: {adamatzky_params['wave_transform_formula']}")
        
        compression_settings = self.config.config['compression_settings']
        print(f"\nCompression Settings:")
        print(f"  - Adaptive Compression: {compression_settings['adaptive_compression']}")
        print(f"  - Target Samples: {compression_settings['target_samples']}")
        print(f"  - Fallback Compression: {compression_settings['fallback_compression']}x")
        
        validation_thresholds = self.config.get_validation_thresholds()
        print(f"\nValidation Thresholds:")
        for threshold_name, threshold_value in validation_thresholds.items():
            print(f"  - {threshold_name.replace('_', ' ').title()}: {threshold_value}")

def main():
    """Main execution function"""
    runner = CentralizedAnalysisRunner()
    
    # Print configuration summary
    runner.print_configuration_summary()
    
    # Run comprehensive analysis
    runner.run_comprehensive_analysis()
    
    print(f"\n✅ Centralized analysis complete!")
    print(f"   All parameters from centralized configuration")
    print(f"   No forced parameters used")
    print(f"   Results saved to output directories")

if __name__ == "__main__":
    main() 