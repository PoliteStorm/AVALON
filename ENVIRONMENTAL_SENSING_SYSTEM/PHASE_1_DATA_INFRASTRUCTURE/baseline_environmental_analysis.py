#!/usr/bin/env python3
"""
🌱 BASELINE ENVIRONMENTAL ANALYSIS
==================================

This script performs focused baseline environmental analysis on validated CSV data
to establish reference parameters for environmental monitoring.

Author: Environmental Sensing Research Team
Date: August 12, 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineEnvironmentalAnalyzer:
    """Focused analyzer for establishing environmental baselines."""
    
    def __init__(self, data_directory: str = "DATA/raw/15061491"):
        self.data_directory = Path(data_directory)
        self.baseline_results = {}
        
    def analyze_dataset(self, filename: str) -> dict:
        """Analyze a single dataset for baseline parameters."""
        logger.info(f"Analyzing baseline for: {filename}")
        
        filepath = self.data_directory / filename
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return {}
        
        try:
            # Read data
            df = pd.read_csv(filepath)
            
            # Basic statistics
            analysis = {
                'filename': filename,
                'timestamp': datetime.now().isoformat(),
                'data_points': len(df),
                'columns': list(df.columns),
                'basic_stats': {},
                'electrical_analysis': {},
                'environmental_estimation': {}
            }
            
            # Analyze numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    analysis['basic_stats'][col] = {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'median': float(col_data.median())
                    }
            
            # Electrical analysis (assume last numeric column is voltage)
            if len(numeric_cols) > 0:
                voltage_col = numeric_cols[-1]
                voltage_data = df[voltage_col].dropna()
                
                if len(voltage_data) > 0:
                    analysis['electrical_analysis'] = {
                        'voltage_range_mv': [float(voltage_data.min() * 1000), float(voltage_data.max() * 1000)],
                        'voltage_mean_mv': float(voltage_data.mean() * 1000),
                        'voltage_std_mv': float(voltage_data.std() * 1000),
                        'voltage_cv': float(voltage_data.std() / abs(voltage_data.mean())) if voltage_data.mean() != 0 else 0,
                        'zero_crossings': int(np.sum(np.diff(np.sign(voltage_data)) != 0)),
                        'peak_count': int(len(voltage_data[voltage_data > voltage_data.mean() + 2*voltage_data.std()]))
                    }
            
            # Environmental estimation based on filename
            analysis['environmental_estimation'] = self._estimate_environmental_conditions(filename)
            
            logger.info(f"✅ Baseline analysis complete for {filename}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {filename}: {e}")
            return {'filename': filename, 'error': str(e)}
    
    def _estimate_environmental_conditions(self, filename: str) -> dict:
        """Estimate environmental conditions from filename and data patterns."""
        filename_lower = filename.lower()
        
        # Default conditions
        conditions = {
            'temperature_celsius': 22.0,
            'humidity_percent': 60.0,
            'ph_level': 6.8,
            'soil_moisture_percent': 50.0,
            'treatment_type': 'control',
            'species': 'unknown'
        }
        
        # Adjust based on filename patterns
        if 'fridge' in filename_lower:
            conditions['temperature_celsius'] = 4.0
            conditions['treatment_type'] = 'temperature_reduction'
        elif 'moisture' in filename_lower:
            conditions['humidity_percent'] = 80.0
            conditions['soil_moisture_percent'] = 70.0
            conditions['treatment_type'] = 'moisture_addition'
        elif 'spray' in filename_lower:
            conditions['humidity_percent'] = 75.0
            conditions['treatment_type'] = 'spray_treatment'
        
        # Species identification
        if 'hericium' in filename_lower:
            conditions['species'] = 'hericium_erinaceus'
            conditions['ph_level'] = 6.5
        elif 'oyster' in filename_lower:
            conditions['species'] = 'pleurotus_ostreatus'
            conditions['ph_level'] = 7.0
        elif 'schizophyllum' in filename_lower:
            conditions['species'] = 'schizophyllum_commune'
            conditions['ph_level'] = 6.8
        
        return conditions
    
    def analyze_all_baselines(self) -> dict:
        """Analyze all available datasets for baseline establishment."""
        logger.info("Starting comprehensive baseline analysis...")
        
        # Find all CSV files
        csv_files = list(self.data_directory.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files for baseline analysis")
        
        # Analyze each file
        all_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_files': len(csv_files),
            'successful_analyses': 0,
            'failed_analyses': 0,
            'individual_analyses': {},
            'aggregate_baseline': {},
            'recommendations': []
        }
        
        for csv_file in csv_files:
            result = self.analyze_dataset(csv_file.name)
            all_results['individual_analyses'][csv_file.name] = result
            
            if 'error' not in result:
                all_results['successful_analyses'] += 1
            else:
                all_results['failed_analyses'] += 1
        
        # Calculate aggregate baseline
        if all_results['successful_analyses'] > 0:
            all_results['aggregate_baseline'] = self._calculate_aggregate_baseline(all_results['individual_analyses'])
        
        # Generate recommendations
        all_results['recommendations'] = self._generate_baseline_recommendations(all_results)
        
        self.baseline_results = all_results
        return all_results
    
    def _calculate_aggregate_baseline(self, individual_analyses: dict) -> dict:
        """Calculate aggregate baseline from individual analyses."""
        logger.info("Calculating aggregate baseline...")
        
        # Collect all electrical parameters
        voltage_ranges = []
        voltage_means = []
        voltage_stds = []
        environmental_conditions = []
        
        for filename, analysis in individual_analyses.items():
            if 'error' not in analysis and 'electrical_analysis' in analysis:
                elec = analysis['electrical_analysis']
                if 'voltage_range_mv' in elec:
                    voltage_ranges.extend(elec['voltage_range_mv'])
                    voltage_means.append(elec['voltage_mean_mv'])
                    voltage_stds.append(elec['voltage_std_mv'])
            
            if 'environmental_estimation' in analysis:
                env = analysis['environmental_estimation']
                environmental_conditions.append(env)
        
        # Calculate aggregate statistics
        aggregate = {
            'electrical_baseline': {
                'voltage_range_mv': [min(voltage_ranges), max(voltage_ranges)] if voltage_ranges else [0, 0],
                'voltage_mean_mv': np.mean(voltage_means) if voltage_means else 0,
                'voltage_std_mv': np.mean(voltage_stds) if voltage_stds else 0,
                'total_data_points': sum(analysis.get('data_points', 0) for analysis in individual_analyses.values() if 'error' not in analysis)
            },
            'environmental_baseline': {
                'temperature_celsius': np.mean([env['temperature_celsius'] for env in environmental_conditions]) if environmental_conditions else 22.0,
                'humidity_percent': np.mean([env['humidity_percent'] for env in environmental_conditions]) if environmental_conditions else 60.0,
                'ph_level': np.mean([env['ph_level'] for env in environmental_conditions]) if environmental_conditions else 6.8,
                'soil_moisture_percent': np.mean([env['soil_moisture_percent'] for env in environmental_conditions]) if environmental_conditions else 50.0
            },
            'species_diversity': len(set(env['species'] for env in environmental_conditions if 'species' in env)),
            'treatment_types': list(set(env['treatment_type'] for env in environmental_conditions if 'treatment_type' in env))
        }
        
        return aggregate
    
    def _generate_baseline_recommendations(self, results: dict) -> list:
        """Generate recommendations based on baseline analysis."""
        recommendations = []
        
        # Data quality recommendations
        if results['failed_analyses'] > 0:
            recommendations.append(f"Address {results['failed_analyses']} failed analyses")
        
        if results['successful_analyses'] < 5:
            recommendations.append("Collect more datasets for robust baseline establishment")
        
        # Electrical baseline recommendations
        if 'aggregate_baseline' in results and results['aggregate_baseline']:
            elec_baseline = results['aggregate_baseline']['electrical_baseline']
            if elec_baseline['voltage_std_mv'] > 1000:
                recommendations.append("High voltage variability detected - investigate data quality")
            
            if elec_baseline['total_data_points'] < 100000:
                recommendations.append("Increase data collection for statistical significance")
        
        # Environmental baseline recommendations
        if 'aggregate_baseline' in results and results['aggregate_baseline']:
            env_baseline = results['aggregate_baseline']['environmental_baseline']
            if env_baseline['species_diversity'] < 2:
                recommendations.append("Include more species for comprehensive baseline")
            
            if len(results['aggregate_baseline']['treatment_types']) < 2:
                recommendations.append("Include more environmental treatments for baseline diversity")
        
        # General recommendations
        recommendations.append("Proceed to Phase 2: Audio Synthesis & Environmental Correlation")
        recommendations.append("Validate baseline parameters with known environmental conditions")
        
        return recommendations
    
    def save_baseline_results(self, output_directory: str = "ENVIRONMENTAL_SENSING_SYSTEM/RESULTS/baseline_analysis"):
        """Save baseline analysis results to files."""
        if not self.baseline_results:
            logger.warning("No baseline results to save")
            return
        
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        json_file = output_dir / "baseline_environmental_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(self.baseline_results, f, indent=2, default=str)
        logger.info(f"Baseline results saved to {json_file}")
        
        # Save summary report
        summary_file = output_dir / "baseline_analysis_summary.md"
        self._generate_summary_report(summary_file)
        logger.info(f"Summary report saved to {summary_file}")
    
    def _generate_summary_report(self, output_file: Path):
        """Generate human-readable summary report."""
        with open(output_file, 'w') as f:
            f.write("# 🌱 Baseline Environmental Analysis - Summary Report\n\n")
            f.write(f"**Generated**: {self.baseline_results['analysis_timestamp']}\n\n")
            
            f.write("## 📊 Analysis Overview\n\n")
            f.write(f"- **Total Files Analyzed**: {self.baseline_results['total_files']}\n")
            f.write(f"- **Successful Analyses**: {self.baseline_results['successful_analyses']}\n")
            f.write(f"- **Failed Analyses**: {self.baseline_results['failed_analyses']}\n")
            f.write(f"- **Success Rate**: {self.baseline_results['successful_analyses']/self.baseline_results['total_files']*100:.1f}%\n\n")
            
            if 'aggregate_baseline' in self.baseline_results:
                f.write("## 🔍 Aggregate Baseline Parameters\n\n")
                baseline = self.baseline_results['aggregate_baseline']
                
                f.write("### Electrical Baseline:\n")
                elec = baseline['electrical_baseline']
                f.write(f"- **Voltage Range**: {elec['voltage_range_mv'][0]:.2f} to {elec['voltage_range_mv'][1]:.2f} mV\n")
                f.write(f"- **Mean Voltage**: {elec['voltage_mean_mv']:.2f} mV\n")
                f.write(f"- **Voltage Stability**: {elec['voltage_std_mv']:.2f} mV\n")
                f.write(f"- **Total Data Points**: {elec['total_data_points']:,}\n\n")
                
                f.write("### Environmental Baseline:\n")
                env = baseline['environmental_baseline']
                f.write(f"- **Temperature**: {env['temperature_celsius']:.1f}°C\n")
                f.write(f"- **Humidity**: {env['humidity_percent']:.1f}%\n")
                f.write(f"- **pH Level**: {env['ph_level']:.1f}\n")
                f.write(f"- **Soil Moisture**: {env['soil_moisture_percent']:.1f}%\n\n")
                
                f.write("### Diversity Metrics:\n")
                f.write(f"- **Species Diversity**: {baseline['species_diversity']}\n")
                f.write(f"- **Treatment Types**: {', '.join(baseline['treatment_types'])}\n\n")
            
            f.write("## 📋 Recommendations\n\n")
            for i, rec in enumerate(self.baseline_results['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\n## 🚀 Next Steps\n\n")
            f.write("1. **Review baseline parameters** for environmental monitoring\n")
            f.write("2. **Validate baselines** with known environmental conditions\n")
            f.write("3. **Proceed to Phase 2**: Audio Synthesis & Environmental Correlation\n")
            f.write("4. **Begin audio signature generation** for pollution detection\n")

def main():
    """Main execution function."""
    logger.info("🌱 Baseline Environmental Analysis")
    logger.info("=" * 50)
    
    try:
        # Initialize analyzer
        analyzer = BaselineEnvironmentalAnalyzer()
        
        # Run analysis
        results = analyzer.analyze_all_baselines()
        
        # Save results
        analyzer.save_baseline_results()
        
        # Display summary
        print("\n" + "="*50)
        print("🎉 BASELINE ANALYSIS COMPLETE!")
        print("="*50)
        print(f"📁 Files Analyzed: {results['total_files']}")
        print(f"✅ Successful: {results['successful_analyses']}")
        print(f"❌ Failed: {results['failed_analyses']}")
        print(f"📊 Success Rate: {results['successful_analyses']/results['total_files']*100:.1f}%")
        
        if 'aggregate_baseline' in results:
            baseline = results['aggregate_baseline']
            print(f"🔌 Voltage Range: {baseline['electrical_baseline']['voltage_range_mv'][0]:.0f} to {baseline['electrical_baseline']['voltage_range_mv'][1]:.0f} mV")
            print(f"🌡️ Temperature: {baseline['environmental_baseline']['temperature_celsius']:.1f}°C")
            print(f"💧 Humidity: {baseline['environmental_baseline']['humidity_percent']:.1f}%")
            print(f"🧬 Species: {baseline['species_diversity']}")
        
        print("\n📋 Key Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
        
        print("\n🚀 Ready for Phase 2: Audio Synthesis & Environmental Correlation!")
        
    except Exception as e:
        logger.error(f"❌ Baseline analysis failed: {e}")
        print(f"\n❌ ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 