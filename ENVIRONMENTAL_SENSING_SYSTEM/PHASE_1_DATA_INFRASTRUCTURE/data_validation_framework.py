#!/usr/bin/env python3
"""
🔍 DATA VALIDATION FRAMEWORK
============================

This script provides a comprehensive framework for validating CSV data
for environmental sensing applications.

Author: Environmental Sensing Research Team
Date: August 12, 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validator for environmental sensing data."""
    
    def __init__(self, data_directory: str = "DATA/raw/15061491"):
        self.data_directory = Path(data_directory)
        self.validation_results = {}
        self.cleaned_data = {}
        self.data_quality_metrics = {}
        
    def validate_single_file(self, filepath: Path) -> Dict[str, Any]:
        """Validate a single CSV file comprehensively."""
        logger.info(f"Validating: {filepath.name}")
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            
            # Initialize validation result
            result = {
                'filename': filepath.name,
                'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'column_names': list(df.columns),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'validation_timestamp': datetime.now().isoformat(),
                'is_valid': True,
                'validation_errors': [],
                'warnings': [],
                'recommendations': [],
                'data_quality_score': 0.0
            }
            
            # Perform comprehensive validation
            self._validate_data_structure(df, result)
            self._validate_data_content(df, result)
            self._validate_statistical_properties(df, result)
            self._validate_environmental_relevance(df, result)
            
            # Calculate overall quality score
            result['data_quality_score'] = self._calculate_quality_score(result)
            
            # Determine if file is valid
            result['is_valid'] = len(result['validation_errors']) == 0
            
            # Store results
            self.validation_results[filepath.name] = result
            
            # If valid, store cleaned data
            if result['is_valid']:
                self.cleaned_data[filepath.name] = df
                logger.info(f"✅ {filepath.name} validated successfully (Score: {result['data_quality_score']:.1f}%)")
            else:
                logger.warning(f"⚠️ {filepath.name} has validation issues (Score: {result['data_quality_score']:.1f}%)")
            
            return result
            
        except Exception as e:
            error_result = {
                'filename': filepath.name,
                'is_valid': False,
                'validation_errors': [f"File reading error: {str(e)}"],
                'recommendations': ["Check file format and encoding"],
                'data_quality_score': 0.0
            }
            self.validation_results[filepath.name] = error_result
            logger.error(f"❌ Error validating {filepath.name}: {e}")
            return error_result
    
    def _validate_data_structure(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Validate basic data structure."""
        # Check minimum data requirements
        if len(df) < 100:
            result['validation_errors'].append("Insufficient data points (< 100)")
            result['recommendations'].append("Collect more data for meaningful analysis")
        elif len(df) < 1000:
            result['warnings'].append("Limited data points (< 1000) - may affect statistical significance")
        
        # Check column requirements
        if len(df.columns) < 2:
            result['validation_errors'].append("Insufficient columns for analysis (< 2)")
            result['recommendations'].append("Ensure at least voltage and time columns")
        
        # Check for required column types
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) == 0:
            result['validation_errors'].append("No numeric columns found")
            result['recommendations'].append("Include numeric data for electrical analysis")
        elif len(numeric_columns) < 2:
            result['warnings'].append("Limited numeric columns - may restrict analysis capabilities")
    
    def _validate_data_content(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Validate data content quality."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                result['validation_errors'].append(f"Column {col} has no valid numeric data")
                continue
            
            # Check for extreme outliers
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 3 * IQR  # 3 IQR rule for extreme outliers
                upper_bound = Q3 + 3 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                outlier_percentage = len(outliers) / len(col_data) * 100
                
                if outlier_percentage > 20:
                    result['validation_errors'].append(f"Column {col}: {outlier_percentage:.1f}% extreme outliers")
                    result['recommendations'].append(f"Investigate and clean outliers in {col}")
                elif outlier_percentage > 10:
                    result['warnings'].append(f"Column {col}: {outlier_percentage:.1f}% outliers detected")
            
            # Check for constant values
            if col_data.std() == 0:
                result['warnings'].append(f"Column {col} has constant values - no variation detected")
            
            # Check for reasonable ranges (for voltage data)
            if 'voltage' in col.lower() or col == numeric_columns[-1]:  # Assume last column is voltage
                voltage_range = col_data.max() - col_data.min()
                if voltage_range > 1000:  # More than 1000V
                    result['warnings'].append(f"Column {col}: Large voltage range ({voltage_range:.1f}V) - verify units")
                elif voltage_range < 0.001:  # Less than 1mV
                    result['warnings'].append(f"Column {col}: Small voltage range ({voltage_range*1000:.3f}mV) - verify sensitivity")
    
    def _validate_statistical_properties(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Validate statistical properties of the data."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            col_data = df[col].dropna()
            
            if len(col_data) < 10:  # Need minimum data for statistical validation
                continue
            
            # Check for normal distribution approximation
            skewness = col_data.skew()
            kurtosis = col_data.kurtosis()
            
            if abs(skewness) > 3:
                result['warnings'].append(f"Column {col}: High skewness ({skewness:.2f}) - data may not be normally distributed")
            
            if abs(kurtosis) > 10:
                result['warnings'].append(f"Column {col}: High kurtosis ({kurtosis:.2f}) - data may have heavy tails")
            
            # Check for autocorrelation (time series properties)
            if len(col_data) > 100:
                autocorr = col_data.autocorr()
                if abs(autocorr) > 0.8:
                    result['warnings'].append(f"Column {col}: High autocorrelation ({autocorr:.3f}) - data may not be independent")
    
    def _validate_environmental_relevance(self, df: pd.DataFrame, result: Dict[str, Any]):
        """Validate environmental relevance of the data."""
        filename = result['filename'].lower()
        
        # Check for environmental treatment indicators
        if 'moisture' in filename or 'spray' in filename:
            # Should have some variation in data
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                col_data = df[numeric_columns[-1]].dropna()
                if len(col_data) > 0 and col_data.std() < 0.001:
                    result['warnings'].append("Moisture treatment detected but low data variation - verify treatment effectiveness")
        
        if 'fridge' in filename:
            # Should show temperature-related changes
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                col_data = df[numeric_columns[-1]].dropna()
                if len(col_data) > 0:
                    # Check for gradual changes that might indicate temperature effects
                    diff_data = col_data.diff().dropna()
                    if len(diff_data) > 0 and diff_data.std() < 0.0001:
                        result['warnings'].append("Temperature treatment detected but minimal change - verify temperature effects")
    
    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        base_score = 100.0
        
        # Deduct points for errors
        error_penalty = len(result['validation_errors']) * 20
        warning_penalty = len(result['warnings']) * 5
        
        # Deduct points for data limitations
        if result['total_rows'] < 1000:
            data_penalty = 10
        elif result['total_rows'] < 10000:
            data_penalty = 5
        else:
            data_penalty = 0
        
        # Deduct points for missing data
        total_cells = result['total_rows'] * result['total_columns']
        missing_cells = sum(result['missing_values'].values())
        missing_penalty = (missing_cells / total_cells) * 20 if total_cells > 0 else 0
        
        # Calculate final score
        final_score = base_score - error_penalty - warning_penalty - data_penalty - missing_penalty
        
        return max(0.0, final_score)
    
    def validate_all_files(self) -> Dict[str, Any]:
        """Validate all CSV files in the data directory."""
        logger.info("Starting comprehensive data validation...")
        
        csv_files = list(self.data_directory.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to validate")
        
        # Validate each file
        for csv_file in csv_files:
            self.validate_single_file(csv_file)
        
        # Generate validation summary
        summary = self._generate_validation_summary()
        
        logger.info("Data validation complete!")
        return summary
    
    def _generate_validation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive validation summary."""
        total_files = len(self.validation_results)
        valid_files = sum(1 for r in self.validation_results.values() if r['is_valid'])
        invalid_files = total_files - valid_files
        
        # Calculate aggregate statistics
        total_data_points = sum(r.get('total_rows', 0) for r in self.validation_results.values() if r['is_valid'])
        total_file_size = sum(r.get('file_size_mb', 0) for r in self.validation_results.values() if r['is_valid'])
        avg_quality_score = np.mean([r.get('data_quality_score', 0) for r in self.validation_results.values()])
        
        # Collect all validation errors and warnings
        all_errors = []
        all_warnings = []
        all_recommendations = []
        
        for result in self.validation_results.values():
            all_errors.extend(result.get('validation_errors', []))
            all_warnings.extend(result.get('warnings', []))
            all_recommendations.extend(result.get('recommendations', []))
        
        summary = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_files': total_files,
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'success_rate': (valid_files / total_files * 100) if total_files > 0 else 0,
            'total_data_points': total_data_points,
            'total_file_size_mb': total_file_size,
            'average_quality_score': round(avg_quality_score, 2),
            'validation_results': self.validation_results,
            'aggregate_issues': {
                'total_errors': len(all_errors),
                'total_warnings': len(all_warnings),
                'total_recommendations': len(all_recommendations),
                'common_errors': self._get_common_issues(all_errors),
                'common_warnings': self._get_common_issues(all_warnings),
                'common_recommendations': self._get_common_issues(all_recommendations)
            }
        }
        
        return summary
    
    def _get_common_issues(self, issues: List[str]) -> Dict[str, int]:
        """Get frequency count of common issues."""
        issue_counts = {}
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Sort by frequency and return top 5
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_issues[:5])
    
    def save_validation_results(self, output_directory: str = "ENVIRONMENTAL_SENSING_SYSTEM/RESULTS/baseline_analysis"):
        """Save validation results to files."""
        if not self.validation_results:
            logger.warning("No validation results to save")
            return
        
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed validation results
        json_file = output_dir / "data_validation_results.json"
        with open(json_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        logger.info(f"Validation results saved to {json_file}")
        
        # Save validation summary
        summary = self._generate_validation_summary()
        summary_file = output_dir / "data_validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Validation summary saved to {summary_file}")
        
        # Save summary report
        report_file = output_dir / "data_validation_report.md"
        self._generate_validation_report(report_file, summary)
        logger.info(f"Validation report saved to {report_file}")
    
    def _generate_validation_report(self, output_file: Path, summary: Dict[str, Any]):
        """Generate human-readable validation report."""
        with open(output_file, 'w') as f:
            f.write("# 🔍 Data Validation Report - Environmental Sensing System\n\n")
            f.write(f"**Generated**: {summary['validation_timestamp']}\n\n")
            
            f.write("## 📊 Validation Overview\n\n")
            f.write(f"- **Total Files**: {summary['total_files']}\n")
            f.write(f"- **Valid Files**: {summary['valid_files']}\n")
            f.write(f"- **Invalid Files**: {summary['invalid_files']}\n")
            f.write(f"- **Success Rate**: {summary['success_rate']:.1f}%\n")
            f.write(f"- **Total Data Points**: {summary['total_data_points']:,}\n")
            f.write(f"- **Total File Size**: {summary['total_file_size_mb']:.1f} MB\n")
            f.write(f"- **Average Quality Score**: {summary['average_quality_score']:.1f}%\n\n")
            
            f.write("## ⚠️ Common Issues\n\n")
            f.write("### Validation Errors:\n")
            for issue, count in summary['aggregate_issues']['common_errors'].items():
                f.write(f"- **{issue}**: {count} occurrences\n")
            
            f.write("\n### Warnings:\n")
            for issue, count in summary['aggregate_issues']['common_warnings'].items():
                f.write(f"- **{issue}**: {count} occurrences\n")
            
            f.write("\n### Recommendations:\n")
            for issue, count in summary['aggregate_issues']['common_recommendations'].items():
                f.write(f"- **{issue}**: {count} occurrences\n")
            
            f.write("\n## 📋 File-by-File Results\n\n")
            for filename, result in summary['validation_results'].items():
                status = "✅" if result['is_valid'] else "❌"
                score = result.get('data_quality_score', 0)
                f.write(f"- **{status} {filename}**: {score:.1f}% quality score\n")
            
            f.write("\n## 🚀 Next Steps\n\n")
            f.write("1. **Review validation results** and address critical issues\n")
            f.write("2. **Clean invalid data** or collect replacement datasets\n")
            f.write("3. **Proceed to baseline analysis** with validated data\n")
            f.write("4. **Begin Phase 2**: Audio Synthesis & Environmental Correlation\n")

def main():
    """Main execution function."""
    logger.info("🔍 Data Validation Framework")
    logger.info("=" * 40)
    
    try:
        # Initialize validator
        validator = DataValidator()
        
        # Run validation
        summary = validator.validate_all_files()
        
        # Save results
        validator.save_validation_results()
        
        # Display summary
        print("\n" + "="*40)
        print("🎉 DATA VALIDATION COMPLETE!")
        print("="*40)
        print(f"📁 Total Files: {summary['total_files']}")
        print(f"✅ Valid: {summary['valid_files']}")
        print(f"❌ Invalid: {summary['invalid_files']}")
        print(f"📊 Success Rate: {summary['success_rate']:.1f}%")
        print(f"📈 Total Data Points: {summary['total_data_points']:,}")
        print(f"🎯 Average Quality: {summary['average_quality_score']:.1f}%")
        
        if summary['aggregate_issues']['total_errors'] > 0:
            print(f"\n⚠️ Total Errors: {summary['aggregate_issues']['total_errors']}")
            print("Top Issues:")
            for issue, count in list(summary['aggregate_issues']['common_errors'].items())[:3]:
                print(f"   • {issue}: {count} occurrences")
        
        print("\n🚀 Ready for baseline environmental analysis!")
        
    except Exception as e:
        logger.error(f"❌ Data validation failed: {e}")
        print(f"\n❌ ERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 