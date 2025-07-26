# Enhanced Fungal Network Analysis Framework

This framework provides advanced tools for analyzing fungal network patterns using wavelet analysis. It is specifically designed to detect and characterize temporal patterns in fungal electrical signals.

## Features

- Multi-scale wavelet analysis
- Adaptive signal denoising
- Pattern detection with statistical validation
- Confidence interval computation
- Comprehensive visualization tools

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The framework provides a Python class `EnhancedWaveletAnalysis` that implements various analysis methods:

```python
from fungal_networks.wavelet_analysis.enhanced_wavelet import EnhancedWaveletAnalysis

# Initialize analyzer
analyzer = EnhancedWaveletAnalysis(
    wavelet_type='morl',
    sampling_rate=1.0
)

# Analyze signal
results = analyzer.analyze_signal(
    signal_data,
    detrend=True,
    denoise=True
)

# Visualize results
analyzer.plot_analysis(results)
```

## Key Components

1. Signal Preprocessing
   - Detrending
   - Wavelet-based denoising
   - Signal normalization

2. Wavelet Analysis
   - Continuous wavelet transform
   - Multi-scale analysis
   - Pattern detection

3. Statistical Analysis
   - Bootstrap-based confidence intervals
   - Significance testing
   - Pattern validation

4. Visualization
   - Time-series plots
   - Wavelet coefficient heatmaps
   - Power spectrum analysis

## Data Structure

The framework expects input data in the following formats:
- `.mat` files containing time series data
- Numpy arrays of signal measurements

## Testing

Run the test script to validate the implementation:
```bash
cd fungal_networks/wavelet_analysis
python test_enhanced_wavelet.py
```

## Contributing

When contributing to this framework:
1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass before submitting changes

 
