# Fungal Language Analysis Project

This project implements an enhanced decoder for analyzing fungal communication patterns based on Adamatzky's research. It analyzes multiple aspects of fungal signaling including temporal patterns, amplitude variations, frequency modulations, and spatial coordination.

## Features

- Multi-modal analysis of fungal communication signals
- Temporal pattern recognition
- Amplitude-based vocabulary analysis
- Frequency modulation analysis
- Spatial coordination pattern detection
- Comprehensive reporting and visualization

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

Run the main analysis script:
```bash
python main.py
```

This will:
1. Generate sample data for testing
2. Analyze temporal, amplitude, and frequency patterns
3. Build a fungal communication vocabulary
4. Generate analysis reports and visualizations
5. Save results in the `language_analysis_outputs` directory

## Project Structure

- `enhanced_fungal_language_decoder.py`: Core analysis implementation
- `main.py`: Main script for running analyses
- `requirements.txt`: Project dependencies
- `language_analysis_outputs/`: Directory for analysis outputs

## Dependencies

- numpy
- matplotlib
- scipy
- scikit-learn
- seaborn
- statsmodels
- PyYAML

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License 