# Fungal Analysis Project Structure

## Overview
This document describes the organized structure of the fungal electrophysiology analysis project, designed for easy testing, maintenance, and further development.

## Directory Structure

```
fungal_analysis_project/
├── README.md                           # Main project documentation
├── requirements.txt                     # Python dependencies
├── PROJECT_STRUCTURE.md               # This file
│
├── src/                               # Source code
│   ├── transforms/                    # Transform implementations
│   │   └── improved_sqrt_transform.py # Main √t transform
│   ├── validation/                    # Validation frameworks
│   │   ├── transform_validation_framework.py
│   │   └── proper_validation_framework.py
│   ├── analysis/                      # Analysis modules
│   │   └── rigorous_fungal_analysis.py # Main analysis pipeline
│   └── utils/                         # Utility functions
│
├── data/                              # Data files
│   ├── raw/                          # Original data
│   │   ├── csv_data/                 # 270 coordinate files
│   │   └── 15061491/                 # Voltage recording data
│   ├── processed/                     # Processed data (future)
│   └── metadata/                      # Data metadata (future)
│
├── results/                           # Analysis results
│   ├── analysis/                      # Analysis results
│   │   ├── rigorous_analysis_results/ # Complete analysis results
│   │   └── validation_results/        # Validation framework results
│   ├── visualizations/                # Generated plots (future)
│   └── reports/                       # Analysis reports (future)
│
├── docs/                              # Documentation
│   ├── api/                           # API documentation (future)
│   ├── research/                      # Research notes
│   │   └── adamatzky_analysis.md     # Analysis of Adamatzky's work
│   └── user_guide/                    # User guides (future)
│
├── scripts/                           # Analysis scripts
│   ├── data_processing/               # Data processing scripts (future)
│   ├── analysis/                      # Analysis scripts
│   │   ├── run_analysis.py           # Main analysis script
│   │   └── [various test scripts]    # Original test scripts
│   └── visualization/                 # Visualization scripts (future)
│
├── tests/                             # Test files
│   ├── unit/                          # Unit tests (future)
│   ├── integration/                   # Integration tests (future)
│   └── validation/                    # Validation tests
│       └── test_transform_validation.py # Transform validation tests
│
└── config/                            # Configuration files
    ├── parameters/                    # Parameter configurations
    │   └── species_config.json       # Species-specific parameters
    └── species_profiles/              # Species profiles (future)
```

## Key Files and Their Purpose

### Core Analysis Files
- **`src/analysis/rigorous_fungal_analysis.py`**: Main analysis pipeline
- **`src/transforms/improved_sqrt_transform.py`**: √t transform implementation
- **`src/validation/transform_validation_framework.py`**: Validation framework

### Configuration Files
- **`config/parameters/species_config.json`**: Species-specific parameters based on Adamatzky's research
- **`requirements.txt`**: Python dependencies

### Documentation
- **`README.md`**: Main project documentation with key findings
- **`docs/research/adamatzky_analysis.md`**: Analysis of results in context of Adamatzky's work

### Test Files
- **`tests/validation/test_transform_validation.py`**: Comprehensive transform validation tests
- **`scripts/analysis/run_analysis.py`**: Main analysis script with command-line interface

### Results
- **`results/analysis/rigorous_analysis_results/`**: Complete analysis results including:
  - `summary_results_20250715_172451.json`: Latest analysis summary
  - `analysis_visualizations_20250715_172451.png`: Visualization plots
  - `detailed_results_*.json`: Detailed analysis results

## Key Findings Summary

### Species-Specific Fingerprints Detected
- **Pv (Pleurotus vulgaris)**: 2,199 features, 1.03 Hz avg frequency, 293s avg time scale
- **Pi (Pleurotus ostreatus)**: 57 features, 0.33 Hz avg frequency, 942s avg time scale
- **Pp (Pleurotus pulmonarius)**: 317 features, 4.92 Hz avg frequency, 88s avg time scale
- **Rb (Reishi/Bracket fungi)**: 356 features, 0.30 Hz avg frequency, 2,971s avg time scale

### Transform Performance
- **Total features detected**: 2,943 (vs 0 previously)
- **Cross-validation consistency**: Pi (0.070), Pp (0.082) - highest reliability
- **Pattern clustering**: Strong clustering across all species

## Quick Start for Testing

### 1. Setup Environment
```bash
cd fungal_analysis_project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Tests
```bash
# Run validation tests
python tests/validation/test_transform_validation.py

# Run main analysis
python scripts/analysis/run_analysis.py
```

### 3. View Results
- Check `results/analysis/rigorous_analysis_results/` for latest results
- View `docs/research/adamatzky_analysis.md` for research significance

## Configuration

### Species Parameters
Each species has optimized k,τ parameters:
- **Pv**: k_range (0.1-10 Hz), tau_range (1-100s)
- **Pi**: k_range (0.03-3 Hz), tau_range (3-300s)
- **Pp**: k_range (0.3-30 Hz), tau_range (0.3-30s)
- **Rb**: k_range (0.01-1 Hz), tau_range (10-1000s)

### Validation Criteria
- Frequency range: 0.001-10 Hz
- Time scale range: 0.1-100,000 seconds
- Amplitude thresholds: Species-specific

## Research Significance

This work confirms Adamatzky's predictions of species-specific electrical fingerprints:
- ✅ Different species show distinct frequency patterns
- ✅ Time scale differentiation matches biological characteristics
- ✅ Cross-validation confirms reliable species identification
- ✅ Pattern clustering reveals structured electrical activity

## Next Steps

1. **Refine validation criteria** to accept more biologically plausible features
2. **Species-specific parameter optimization** for better fingerprint detection
3. **Environmental response analysis** to study moisture/light effects
4. **Real-time analysis pipeline** for live data processing

## File Organization Benefits

### 1. Modularity
- Separate concerns: transforms, validation, analysis
- Easy to test individual components
- Clear dependencies between modules

### 2. Reproducibility
- Configuration files for consistent parameters
- Test scripts for validation
- Documented analysis pipeline

### 3. Extensibility
- Easy to add new species
- Simple to modify validation criteria
- Clear structure for new features

### 4. Documentation
- Research context in docs/
- API documentation structure
- User guides for different audiences

This organized structure makes it easy to:
- **Test** the transform with different parameters
- **Validate** results against known biological patterns
- **Extend** the analysis to new species or conditions
- **Document** findings for research publication 