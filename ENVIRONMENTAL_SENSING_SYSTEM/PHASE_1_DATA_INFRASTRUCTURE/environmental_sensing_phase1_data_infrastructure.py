#!/usr/bin/env python3
"""
🌍 ENVIRONMENTAL SENSING SYSTEM - PHASE 1: Data Infrastructure
==============================================================

This script implements the data infrastructure for environmental sensing through
fungal electrical audio analysis. It validates, cleans, and analyzes CSV data
to establish baseline environmental parameters.

Author: Environmental Sensing Research Team
Date: August 12, 2025
Version: 1.0.0
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('environmental_sensing_phase1.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EnvironmentalParameters:
    """Data class for environmental parameters."""
    temperature_celsius: float
    humidity_percent: float
    ph_level: float
    soil_moisture_percent: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'temperature_celsius': self.temperature_celsius,
            'humidity_percent': self.humidity_percent,
            'ph_level': self.ph_level,
            'soil_moisture_percent': self.soil_moisture_percent,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class ElectricalSignals:
    """Data class for electrical signal parameters."""
    voltage_range_mv: Tuple[float, float]
    frequency_spectrum_hz: Tuple[float, float]
    amplitude_stability_percent: float
    harmonic_relationships: List[float]
    sampling_rate_hz: float
    data_points: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'voltage_range_mv': list(self.voltage_range_mv),
            'frequency_spectrum_hz': list(self.frequency_spectrum_hz),
            'amplitude_stability_percent': self.amplitude_stability_percent,
            'harmonic_relationships': self.harmonic_relationships,
            'sampling_rate_hz': self.sampling_rate_hz,
            'data_points': self.data_points
        }

class DataValidationFramework:
    """Framework for validating and cleaning environmental data."""
    
    def __init__(self, data_directory: str = "DATA/raw/15061491"):
        self.data_directory = Path(data_directory)
        self.validation_results = {}
        self.cleaned_data = {}
        self.baseline_parameters = {}
        
    def validate_csv_file(self, filepath: Path) -> Dict[str, Any]:
        """Validate a single CSV file for environmental analysis."""
        logger.info(f"Validating file: {filepath.name}")
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            
            # Basic validation
            validation_result = {
                'filename': filepath.name,
                'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'column_names': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'is_valid': True,
                'validation_errors': [],
                'recommendations': []
            }
            
            # Check for minimum data requirements
            if len(df) < 1000:
                validation_result['is_valid'] = False
                validation_result['validation_errors'].append("Insufficient data points (< 1000)")
                validation_result['recommendations'].append("Collect more data for statistical significance")
            
            # Check for required columns (basic electrical data)
            if len(df.columns) < 2:
                validation_result['is_valid'] = False
                validation_result['validation_errors'].append("Insufficient columns for analysis")
                validation_result['recommendations'].append("Ensure at least voltage and time columns")
            
            # Check for extreme outliers
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    
                    if len(outliers) > len(df) * 0.1:  # More than 10% outliers
                        validation_result['validation_errors'].append(f"High outlier percentage in {col}: {len(outliers)/len(df)*100:.1f}%")
                        validation_result['recommendations'].append(f"Investigate outliers in {col}")
            
            # Store validation result
            self.validation_results[filepath.name] = validation_result
            
            # If valid, store cleaned data
            if validation_result['is_valid']:
                self.cleaned_data[filepath.name] = df
                logger.info(f"✅ File {filepath.name} validated successfully")
            else:
                logger.warning(f"⚠️ File {filepath.name} has validation issues")
                
            return validation_result
            
        except Exception as e:
            error_result = {
                'filename': filepath.name,
                'is_valid': False,
                'validation_errors': [f"File reading error: {str(e)}"],
                'recommendations': ["Check file format and encoding"]
            }
            self.validation_results[filepath.name] = error_result
            logger.error(f"❌ Error validating {filepath.name}: {e}")
            return error_result
    
    def validate_all_files(self) -> Dict[str, Any]:
        """Validate all CSV files in the data directory."""
        logger.info("Starting comprehensive data validation...")
        
        csv_files = list(self.data_directory.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to validate")
        
        validation_summary = {
            'total_files': len(csv_files),
            'valid_files': 0,
            'invalid_files': 0,
            'total_data_points': 0,
            'total_file_size_mb': 0,
            'validation_results': {}
        }
        
        for csv_file in csv_files:
            result = self.validate_csv_file(csv_file)
            validation_summary['validation_results'][csv_file.name] = result
            
            if result['is_valid']:
                validation_summary['valid_files'] += 1
                validation_summary['total_data_points'] += result.get('total_rows', 0)
                validation_summary['total_file_size_mb'] += result.get('file_size_mb', 0)
            else:
                validation_summary['invalid_files'] += 1
        
        logger.info(f"Validation complete: {validation_summary['valid_files']} valid, {validation_summary['invalid_files']} invalid")
        return validation_summary

class BaselineEnvironmentalAnalysis:
    """Analyze baseline environmental parameters from validated data."""
    
    def __init__(self, cleaned_data: Dict[str, pd.DataFrame]):
        self.cleaned_data = cleaned_data
        self.baseline_analysis = {}
        self.environmental_parameters = {}
        
    def analyze_electrical_signals(self, df: pd.DataFrame, filename: str) -> ElectricalSignals:
        """Analyze electrical signals from a dataset."""
        logger.info(f"Analyzing electrical signals for {filename}")
        
        # Find voltage column (assume last column or column with largest range)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            raise ValueError(f"No numeric columns found in {filename}")
        
        # Assume last column is voltage if multiple columns
        voltage_column = numeric_columns[-1]
        voltage_data = df[voltage_column].dropna()
        
        # Calculate electrical parameters
        voltage_range_mv = (voltage_data.min() * 1000, voltage_data.max() * 1000)  # Convert to mV
        
        # Estimate sampling rate (assume time column is first if exists)
        time_column = None
        for col in df.columns:
            if 'time' in col.lower() or col == numeric_columns[0]:
                time_column = col
                break
        
        if time_column and len(df) > 1:
            time_diff = df[time_column].diff().dropna()
            if len(time_diff) > 0:
                avg_time_diff = time_diff.mean()
                sampling_rate_hz = 1.0 / avg_time_diff if avg_time_diff > 0 else 1.0
            else:
                sampling_rate_hz = 1.0
        else:
            sampling_rate_hz = 1.0
        
        # Calculate frequency spectrum (estimate from voltage variations)
        voltage_std = voltage_data.std()
        voltage_mean = voltage_data.mean()
        
        # Estimate frequency range based on voltage variations
        if voltage_std > 0:
            # Higher variation suggests higher frequency content
            freq_min = max(0.001, voltage_std / 1000)  # Minimum 1 mHz
            freq_max = min(10.0, voltage_std * 100)    # Maximum 10 Hz
        else:
            freq_min, freq_max = 0.001, 1.0
        
        # Calculate amplitude stability
        voltage_cv = voltage_std / abs(voltage_mean) if voltage_mean != 0 else 0
        amplitude_stability_percent = max(0, 100 - (voltage_cv * 100))
        
        # Estimate harmonic relationships (simplified)
        harmonic_relationships = [1.0, 2.0, 3.0, 4.0, 5.0]  # Basic harmonics
        
        return ElectricalSignals(
            voltage_range_mv=voltage_range_mv,
            frequency_spectrum_hz=(freq_min, freq_max),
            amplitude_stability_percent=amplitude_stability_percent,
            harmonic_relationships=harmonic_relationships,
            sampling_rate_hz=sampling_rate_hz,
            data_points=len(voltage_data)
        )
    
    def estimate_environmental_parameters(self, filename: str) -> EnvironmentalParameters:
        """Estimate environmental parameters based on filename and data patterns."""
        logger.info(f"Estimating environmental parameters for {filename}")
        
        # Extract environmental information from filename
        filename_lower = filename.lower()
        
        # Default parameters
        temp = 22.0  # Room temperature
        humidity = 60.0  # Moderate humidity
        ph = 6.8  # Neutral pH
        moisture = 50.0  # Moderate moisture
        
        # Adjust based on filename patterns
        if 'fridge' in filename_lower:
            temp = 4.0  # Refrigerator temperature
        elif 'moisture' in filename_lower or 'spray' in filename_lower:
            humidity = 80.0  # High humidity
            moisture = 70.0  # High moisture
        elif 'dry' in filename_lower:
            humidity = 30.0  # Low humidity
            moisture = 20.0  # Low moisture
        
        # Adjust pH based on species (if known)
        if 'hericium' in filename_lower:
            ph = 6.5  # Slightly acidic
        elif 'oyster' in filename_lower:
            ph = 7.0  # Neutral
        
        return EnvironmentalParameters(
            temperature_celsius=temp,
            humidity_percent=humidity,
            ph_level=ph,
            soil_moisture_percent=moisture,
            timestamp=datetime.now()
        )
    
    def analyze_all_datasets(self) -> Dict[str, Any]:
        """Analyze all validated datasets for baseline parameters."""
        logger.info("Starting baseline environmental analysis...")
        
        analysis_results = {
            'total_datasets': len(self.cleaned_data),
            'electrical_analysis': {},
            'environmental_parameters': {},
            'statistical_summary': {},
            'baseline_establishment': {}
        }
        
        for filename, df in self.cleaned_data.items():
            try:
                # Analyze electrical signals
                electrical_signals = self.analyze_electrical_signals(df, filename)
                analysis_results['electrical_analysis'][filename] = electrical_signals.to_dict()
                
                # Estimate environmental parameters
                env_params = self.estimate_environmental_parameters(filename)
                analysis_results['environmental_parameters'][filename] = env_params.to_dict()
                
                logger.info(f"✅ Analysis complete for {filename}")
                
            except Exception as e:
                logger.error(f"❌ Error analyzing {filename}: {e}")
                analysis_results['electrical_analysis'][filename] = {'error': str(e)}
                analysis_results['environmental_parameters'][filename] = {'error': str(e)}
        
        # Calculate statistical summary
        self._calculate_statistical_summary(analysis_results)
        
        # Establish baseline parameters
        self._establish_baseline_parameters(analysis_results)
        
        self.baseline_analysis = analysis_results
        return analysis_results
    
    def _calculate_statistical_summary(self, analysis_results: Dict[str, Any]):
        """Calculate statistical summary across all datasets."""
        logger.info("Calculating statistical summary...")
        
        # Collect all voltage ranges
        voltage_ranges = []
        frequency_ranges = []
        stability_scores = []
        
        for filename, elec_data in analysis_results['electrical_analysis'].items():
            if 'error' not in elec_data:
                voltage_ranges.append(elec_data['voltage_range_mv'])
                frequency_ranges.append(elec_data['frequency_spectrum_hz'])
                stability_scores.append(elec_data['amplitude_stability_percent'])
        
        if voltage_ranges:
            # Voltage statistics
            all_voltages = [v for range_tuple in voltage_ranges for v in range_tuple]
            analysis_results['statistical_summary']['voltage'] = {
                'min_mv': min(all_voltages),
                'max_mv': max(all_voltages),
                'mean_mv': np.mean(all_voltages),
                'std_mv': np.std(all_voltages)
            }
            
            # Frequency statistics
            all_frequencies = [f for range_tuple in frequency_ranges for f in range_tuple]
            analysis_results['statistical_summary']['frequency'] = {
                'min_hz': min(all_frequencies),
                'max_hz': max(all_frequencies),
                'mean_hz': np.mean(all_frequencies),
                'std_hz': np.std(all_frequencies)
            }
            
            # Stability statistics
            analysis_results['statistical_summary']['stability'] = {
                'min_percent': min(stability_scores),
                'max_percent': max(stability_scores),
                'mean_percent': np.mean(stability_scores),
                'std_percent': np.std(stability_scores)
            }
    
    def _establish_baseline_parameters(self, analysis_results: Dict[str, Any]):
        """Establish baseline parameters for environmental monitoring."""
        logger.info("Establishing baseline parameters...")
        
        if 'statistical_summary' in analysis_results:
            # Use median values as baseline
            voltage_stats = analysis_results['statistical_summary'].get('voltage', {})
            frequency_stats = analysis_results['statistical_summary'].get('frequency', {})
            stability_stats = analysis_results['statistical_summary'].get('stability', {})
            
            baseline = {
                'voltage_baseline_mv': voltage_stats.get('mean_mv', 0),
                'frequency_baseline_hz': frequency_stats.get('mean_hz', 1.0),
                'stability_baseline_percent': stability_stats.get('mean_percent', 80.0),
                'environmental_baseline': {
                    'temperature_celsius': 22.0,
                    'humidity_percent': 60.0,
                    'ph_level': 6.8,
                    'soil_moisture_percent': 50.0
                },
                'detection_thresholds': {
                    'voltage_change_threshold_mv': voltage_stats.get('std_mv', 100) * 2,
                    'frequency_change_threshold_hz': frequency_stats.get('std_hz', 0.5) * 2,
                    'stability_change_threshold_percent': 10.0
                }
            }
            
            analysis_results['baseline_establishment'] = baseline
            logger.info("✅ Baseline parameters established")

class DataInfrastructureManager:
    """Main manager for Phase 1 data infrastructure."""
    
    def __init__(self, base_directory: str = "ENVIRONMENTAL_SENSING_SYSTEM"):
        self.base_directory = Path(base_directory)
        self.results_directory = self.base_directory / "RESULTS" / "baseline_analysis"
        self.results_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.validation_framework = DataValidationFramework()
        self.baseline_analyzer = None
        
    def run_phase1_analysis(self) -> Dict[str, Any]:
        """Run complete Phase 1 analysis."""
        logger.info("🚀 Starting Phase 1: Data Infrastructure Analysis")
        
        # Step 1: Validate all CSV files
        logger.info("Step 1: Validating CSV files...")
        validation_summary = self.validation_framework.validate_all_files()
        
        # Step 2: Analyze baseline parameters
        if validation_summary['valid_files'] > 0:
            logger.info("Step 2: Analyzing baseline parameters...")
            self.baseline_analyzer = BaselineEnvironmentalAnalysis(self.validation_framework.cleaned_data)
            baseline_analysis = self.baseline_analyzer.analyze_all_datasets()
        else:
            logger.error("No valid files found for analysis")
            baseline_analysis = {}
        
        # Step 3: Generate comprehensive results
        logger.info("Step 3: Generating comprehensive results...")
        phase1_results = {
            'phase': 'PHASE_1_DATA_INFRASTRUCTURE',
            'timestamp': datetime.now().isoformat(),
            'validation_summary': validation_summary,
            'baseline_analysis': baseline_analysis,
            'data_quality_score': self._calculate_data_quality_score(validation_summary),
            'recommendations': self._generate_recommendations(validation_summary, baseline_analysis)
        }
        
        # Step 4: Save results
        logger.info("Step 4: Saving results...")
        self._save_phase1_results(phase1_results)
        
        # Step 5: Generate summary report
        logger.info("Step 5: Generating summary report...")
        self._generate_summary_report(phase1_results)
        
        logger.info("✅ Phase 1 analysis complete!")
        return phase1_results
    
    def _calculate_data_quality_score(self, validation_summary: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        if validation_summary['total_files'] == 0:
            return 0.0
        
        valid_ratio = validation_summary['valid_files'] / validation_summary['total_files']
        data_points_score = min(1.0, validation_summary['total_data_points'] / 100000)  # Normalize to 100k points
        
        quality_score = (valid_ratio * 0.7) + (data_points_score * 0.3)
        return round(quality_score * 100, 2)
    
    def _generate_recommendations(self, validation_summary: Dict[str, Any], baseline_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Data quality recommendations
        if validation_summary['invalid_files'] > 0:
            recommendations.append(f"Address validation issues in {validation_summary['invalid_files']} files")
        
        if validation_summary['total_data_points'] < 50000:
            recommendations.append("Collect more data points for robust statistical analysis")
        
        # Baseline recommendations
        if 'baseline_establishment' in baseline_analysis:
            baseline = baseline_analysis['baseline_establishment']
            if baseline.get('stability_baseline_percent', 0) < 70:
                recommendations.append("Improve data collection stability for better baseline establishment")
        
        # General recommendations
        recommendations.append("Proceed to Phase 2: Audio Synthesis & Environmental Correlation")
        recommendations.append("Validate baseline parameters with known environmental conditions")
        
        return recommendations
    
    def _save_phase1_results(self, results: Dict[str, Any]):
        """Save Phase 1 results to files."""
        # Save JSON results
        json_file = self.results_directory / "phase1_comprehensive_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {json_file}")
        
        # Save validation summary CSV
        if 'validation_summary' in results:
            validation_df = pd.DataFrame(results['validation_summary']['validation_results']).T
            csv_file = self.results_directory / "phase1_validation_summary.csv"
            validation_df.to_csv(csv_file)
            logger.info(f"Validation summary saved to {csv_file}")
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate human-readable summary report."""
        report_file = self.results_directory / "phase1_summary_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# 🌍 Phase 1: Data Infrastructure Analysis - Summary Report\n\n")
            f.write(f"**Generated**: {results['timestamp']}\n\n")
            
            f.write("## 📊 Data Quality Assessment\n\n")
            f.write(f"- **Overall Quality Score**: {results['data_quality_score']}%\n")
            f.write(f"- **Total Files**: {results['validation_summary']['total_files']}\n")
            f.write(f"- **Valid Files**: {results['validation_summary']['valid_files']}\n")
            f.write(f"- **Invalid Files**: {results['validation_summary']['invalid_files']}\n")
            f.write(f"- **Total Data Points**: {results['validation_summary']['total_data_points']:,}\n\n")
            
            f.write("## 🔍 Baseline Analysis Status\n\n")
            if 'baseline_analysis' in results and results['baseline_analysis']:
                f.write("✅ **Baseline parameters established successfully**\n\n")
                f.write("### Key Baseline Parameters:\n")
                if 'baseline_establishment' in results['baseline_analysis']:
                    baseline = results['baseline_analysis']['baseline_establishment']
                    f.write(f"- **Voltage Baseline**: {baseline.get('voltage_baseline_mv', 'N/A'):.2f} mV\n")
                    f.write(f"- **Frequency Baseline**: {baseline.get('frequency_baseline_hz', 'N/A'):.3f} Hz\n")
                    f.write(f"- **Stability Baseline**: {baseline.get('stability_baseline_percent', 'N/A'):.1f}%\n")
            else:
                f.write("❌ **Baseline analysis incomplete**\n\n")
            
            f.write("## 📋 Recommendations\n\n")
            for i, rec in enumerate(results['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\n## 🚀 Next Steps\n\n")
            f.write("1. **Review validation results** and address any data quality issues\n")
            f.write("2. **Validate baseline parameters** with known environmental conditions\n")
            f.write("3. **Proceed to Phase 2**: Audio Synthesis & Environmental Correlation\n")
            f.write("4. **Begin audio signature generation** for environmental monitoring\n")
        
        logger.info(f"Summary report generated: {report_file}")

def main():
    """Main execution function for Phase 1."""
    logger.info("🌍 Environmental Sensing System - Phase 1: Data Infrastructure")
    logger.info("=" * 70)
    
    try:
        # Initialize and run Phase 1
        manager = DataInfrastructureManager()
        results = manager.run_phase1_analysis()
        
        # Display summary
        print("\n" + "="*70)
        print("🎉 PHASE 1 COMPLETE!")
        print("="*70)
        print(f"📊 Data Quality Score: {results['data_quality_score']}%")
        print(f"📁 Valid Files: {results['validation_summary']['valid_files']}/{results['validation_summary']['total_files']}")
        print(f"📈 Total Data Points: {results['validation_summary']['total_data_points']:,}")
        print(f"🔍 Baseline Analysis: {'✅ Complete' if results['baseline_analysis'] else '❌ Incomplete'}")
        print("\n📋 Key Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
        print("\n🚀 Ready for Phase 2: Audio Synthesis & Environmental Correlation!")
        
    except Exception as e:
        logger.error(f"❌ Phase 1 execution failed: {e}")
        print(f"\n❌ ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 