# AVALON: Fungal Electrical Activity Analysis

## Overview
This repository contains code and analysis results for studying electrical activity in fungi using wavelet transform analysis and advanced signal processing techniques.

## Key Features
- Square root wavelet transform implementation (W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt)
- 2D and 3D visualization of fungal electrical activity
- Multi-scale temporal pattern analysis
- Interactive visualization tools
- Resource-optimized processing for large datasets

## Repository Structure
```
AVALON/
├── fungal_analysis/          # Core analysis code
│   ├── src/                  # Source code
│   ├── visualizations/       # Generated visualizations
│   └── sqrt_wavelet_results/ # Wavelet analysis results
├── data/                     # Raw data files
└── docs/                     # Documentation
```

## Installation
```bash
# Clone the repository
git clone https://github.com/politestorm/AVALON.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. Data Processing:
```python
python fungal_analysis/src/run_analysis.py
```

2. Visualization:
```python
python fungal_analysis/src/wavelet_visualizer.py
```

## Results
The analysis reveals several key findings:
1. Multi-scale communication architecture in fungi
2. Hierarchical organization of electrical signals
3. Complex temporal patterns and rhythms
4. Environmental response patterns

Detailed results can be found in the visualizations directory.

## Contact
- Email: knowsj2@gmail.com
- GitHub: @politestorm

## License
MIT License 