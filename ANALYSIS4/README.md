# Enhanced Fungal Data Analysis (ANALYSIS4)

This directory contains an improved analysis pipeline for processing real fungal data, incorporating enhanced wavelet analysis and comprehensive metrics calculation.

## Features

- Memory-efficient chunked processing for large datasets
- Enhanced wavelet metrics including:
  - Signal complexity analysis
  - Pattern consistency measurement
  - Response latency detection
  - Characteristic frequency extraction
- Multi-modal analysis:
  - Electrical signals
  - Acoustic measurements
  - Spatial network properties
- Comprehensive visualization suite
- Automated batch processing
- Detailed results reporting

## Directory Structure

```
ANALYSIS4/
├── code/
│   ├── process_real_data.py     # Main processing script
│   ├── sqrt_wavelet.py          # Wavelet transform implementation
│   ├── wavelet_metrics.py       # Analysis metrics
│   └── batch_process.py         # Batch processing utilities
├── data/                        # Input data directory
├── results/                     # Analysis results
└── visualizations/              # Generated visualizations
```

## Usage

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the analysis:
   ```bash
   python code/process_real_data.py --data-dir path/to/data --output-dir path/to/output
   ```

3. Results will be organized by replicate in the output directory, including:
   - Wavelet analysis visualizations
   - Metrics summaries
   - JSON results files

## Data Format

The analysis expects the following structure for input data:

```
data/
├── replicate_1/
│   ├── electro.json    # Electrical measurements
│   ├── acoustic.json   # Acoustic measurements
│   └── spatial.json    # Spatial network data
├── replicate_2/
└── replicate_3/
```

Each JSON file should contain:
- Time series data
- Event markers (if applicable)
- Metadata about the measurements

## Output Format

The analysis generates:
1. Per-replicate visualizations
2. Comprehensive metrics in JSON format
3. Summary reports
4. Interactive visualizations (where applicable)

## Analysis Pipeline

1. Data Loading & Validation
2. Signal Processing
   - Wavelet transformation
   - Feature extraction
   - Anomaly detection
3. Metrics Calculation
   - Complexity measures
   - Pattern analysis
   - Response characteristics
4. Visualization Generation
5. Results Compilation

## Contributing

When adding new features or modifications:
1. Follow the existing code structure
2. Add appropriate error handling
3. Update documentation
4. Add tests for new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details. 