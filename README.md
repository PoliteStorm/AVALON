# Fungal Electrical Activity Monitoring System

A comprehensive system for analyzing fungal electrical activity using Adamatzky's method combined with wave transform analysis.

## Directory Structure

```
fungal_electrical_monitoring_system/
├── scripts/                    # All Python analysis scripts
├── data/                       # Validated CSV files for analysis
├── results/                    # Analysis results and visualizations
├── docs/                       # Documentation and parameter guides
└── requirements/               # Dependencies and requirements
```

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements/requirements.txt
   ```

2. **Run Basic Analysis:**
   ```bash
   python scripts/ultra_optimized_fungal_monitoring_simple.py data/
   ```

3. **Run Comprehensive Analysis:**
   ```bash
   python scripts/fungal_electrical_monitoring_with_wave_transform.py data/
   ```

## Scripts Overview

### Core Analysis Scripts
- `fungal_electrical_monitoring_with_wave_transform.py` - Full integrated analysis with wave transform
- `optimized_fungal_electrical_monitoring.py` - Optimized version with vectorized operations
- `ultra_optimized_fungal_monitoring.py` - Ultra-fast version with Numba JIT compilation
- `ultra_optimized_fungal_monitoring_simple.py` - Simplified ultra-fast version

### Batch Processing
- `batch_fungal_csv_analysis.py` - Process multiple CSV files
- `csv_parameter_analysis.py` - Analyze CSV parameters and quality

### Validation and Testing
- `comprehensive_false_positive_testing.py` - Test for false positives
- `false_positive_validation.py` - Validate results against controls
- `advanced_pattern_validation.py` - Advanced pattern validation
- `validate_results.py` - Validate analysis results

### Visualization
- `detailed_pattern_visualization.py` - Create detailed visualizations
- `test_best_files.py` - Test and visualize best quality files
- `show_fungal_data.py` - Display fungal data summaries

### Results Analysis
- `show_all_json_results.py` - Display all JSON results
- `test_results_summary.py` - Summarize test results
- `final_summary.py` - Create final analysis summary

## Data Files

The `data/` directory contains validated CSV files that meet Adamatzky's research parameters:
- `Norm_vs_deep_tip_crop.csv` - High-quality electrical recordings
- `New_Oyster_with spray_as_mV_seconds_SigView.csv` - Oyster mushroom data
- `Ch1-2_1second_sampling.csv` - Channel 1-2 sampling data

## Results

The `results/` directory contains:
- JSON analysis results
- CSV summary files
- Visualization plots (PNG files)
- Performance comparison data

## Documentation

The `docs/` directory contains:
- `fungal_electrical_parameters.md` - Complete parameter requirements
- `parameter_requirements_summary.md` - Parameter summary
- `json_files_explanation.md` - JSON file explanations
- `electrical_activity_summary_report.md` - Activity summary
- `comprehensive_sqrt_transform_report.md` - Wave transform analysis
- `sqrt_transform_research_report.md` - Research findings

## Key Features

### Adamatzky's Method Implementation
- Spike detection with configurable thresholds
- Inter-spike interval analysis
- Amplitude distribution analysis
- Biological activity scoring

### Wave Transform Analysis
- Multi-scale pattern detection
- Rhythmic component analysis
- Power spectral density analysis
- Pattern complexity assessment

### Performance Optimizations
- Vectorized operations using NumPy
- Parallel processing with multiprocessing
- Numba JIT compilation for critical functions
- Caching for repeated calculations
- Memory-efficient data processing

### Validation Framework
- False positive detection
- Synthetic control testing
- Parameter independence validation
- Biological correlation analysis

## Usage Examples

### Basic Analysis
```python
from scripts.ultra_optimized_fungal_monitoring_simple import analyze_file

# Analyze a single file
results = analyze_file("data/Norm_vs_deep_tip_crop.csv")
print(f"Spike count: {results['spike_count']}")
print(f"Biological activity score: {results['biological_activity_score']}")
```

### Batch Analysis
```python
from scripts.batch_fungal_csv_analysis import analyze_directory

# Analyze all files in directory
results = analyze_directory("data/")
print(f"Processed {len(results)} files")
```

### Comprehensive Analysis
```python
from scripts.fungal_electrical_monitoring_with_wave_transform import integrated_analysis

# Run full integrated analysis
results = integrated_analysis("data/")
print(f"Wave transform features: {results['wave_features']}")
print(f"Alignment ratio: {results['adamatzky_alignment_ratio']}")
```

## Performance

The system includes multiple optimization levels:
- **Standard**: Basic implementation (~30 seconds per file)
- **Optimized**: Vectorized operations (~10 seconds per file)
- **Ultra-optimized**: Numba JIT compilation (~2 seconds per file)

## Validation Results

The system has been validated against:
- Adamatzky's 2023 research parameters
- Synthetic control data
- False positive detection
- Pattern complexity analysis

Results show excellent alignment with published research and robust pattern detection capabilities.

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Pandas
- Matplotlib
- Numba (for ultra-optimized version)
- Multiprocessing (for parallel processing)

## License

This system is designed for research purposes and implements methods from Adamatzky's fungal electrical activity research. 