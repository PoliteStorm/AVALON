# Adamatzky Frequency Discrimination Analysis for Fungal Mycelium Networks

This repository implements the methodology from **Adamatzky et al. (2022)** for analyzing frequency discrimination in Pleurotus ostreatus mycelium networks, with extensions for audio correlation and linguistic pattern analysis.

## 🍄 Research Background

Based on the paper: **"Fungal electronics: frequency discrimination in mycelium networks"**

### Key Findings from Adamatzky's Research:
- **Frequency Discrimination**: Fungal networks can discriminate between frequencies in a fuzzy/threshold-based manner
- **Critical Threshold**: 10 mHz marks a transition in electrical response behavior
- **Harmonic Generation**: Below 10 mHz, 3rd harmonics are amplified while 2nd harmonics are damped
- **THD Analysis**: Total Harmonic Distortion shows distinct patterns below vs. above 10 mHz
- **Fuzzy Logic**: Continuous frequency classification using fuzzy set theory
- **Linguistic Complexity**: Fungal electrical patterns exhibit language-like complexity

## 🔬 Analysis Components

### 1. Frequency Discrimination Analysis (`adamatzky_frequency_discrimination_analysis.py`)
- **Frequency Range**: 1-100 mHz (covering fungal action potential frequencies)
- **Sinusoidal Stimulation**: Applies electrical signals following Adamatzky's protocol
- **FFT Analysis**: Uses Blackman window function (Adamatzky's choice)
- **Harmonic Analysis**: 2nd vs 3rd harmonic amplitude ratios
- **THD Calculation**: Total Harmonic Distortion analysis
- **Fuzzy Classification**: Linguistic classification using fuzzy sets

### 2. Audio-Linguistic Correlation (`fungal_audio_linguistic_correlation.py`)
- **Audio Feature Extraction**: Spectral, rhythmic, and harmonic features
- **Linguistic Analysis**: Adamatzky's word-based language analysis
- **Correlation Analysis**: Electrical signal vs. audio feature correlations
- **Spike Detection**: Adaptive thresholding for electrical spikes
- **Complexity Metrics**: Algorithmic complexity and entropy measures

## 📊 Key Metrics

### Frequency Response Metrics:
- **THD (Total Harmonic Distortion)**: Signal distortion measure
- **Harmonic Ratios**: 2nd/3rd harmonic amplitude relationships
- **Frequency Discrimination Threshold**: 10 mHz boundary
- **Fuzzy Classification**: Very Low, Low, Medium, High, Very High

### Linguistic Metrics:
- **Word Length Distribution**: Statistical analysis of fungal "words"
- **Vocabulary Size**: Unique pattern count
- **Transition Matrix**: Word-to-word transition probabilities
- **Complexity Index**: Algorithmic complexity measures

## 🚀 Installation & Setup

### 1. Install Dependencies:
```bash
pip install -r requirements_adamatzky_analysis.txt
```

### 2. Verify Data Structure:
```
DATA/
├── processed/validated_fungal_electrical_csvs/
│   ├── New_Oyster_with spray_as_mV_seconds_SigView.csv
│   ├── Ch1-2_1second_sampling.csv
│   └── Norm_vs_deep_tip_crop.csv
└── raw/csv_data/
    └── [various fungal data files]

RESULTS/
├── audio/
│   ├── New_Oyster_with spray_as_mV.csv_basic_sound.wav
│   ├── New_Oyster_with spray_as_mV.csv_frequency_modulated.wav
│   └── [other audio files]
└── adamatzky_analysis/
    └── [previous analysis results]
```

## 🔍 Usage Examples

### Basic Frequency Discrimination Analysis:
```bash
python adamatzky_frequency_discrimination_analysis.py
```

### Audio-Linguistic Correlation Analysis:
```bash
python fungal_audio_linguistic_correlation.py
```

### Custom Analysis:
```python
from adamatzky_frequency_discrimination_analysis import AdamatzkyFrequencyAnalyzer

# Initialize analyzer
analyzer = AdamatzkyFrequencyAnalyzer(sampling_rate=1.0)

# Load your fungal data
signal_data = analyzer.load_fungal_data("path/to/your/data.csv")

# Perform analysis
results = analyzer.analyze_frequency_discrimination(signal_data)

# Create visualizations
analyzer.create_visualizations(results)

# Save results
analyzer.save_results(results)
```

## 📈 Expected Results

### Frequency Discrimination:
- **Low Frequencies (≤10 mHz)**: High THD (up to 45.9%), strong harmonic distortion
- **High Frequencies (>10 mHz)**: Low THD (<10%), minimal distortion
- **Threshold Behavior**: Clear transition around 10 mHz

### Harmonic Patterns:
- **Below 10 mHz**: 3rd harmonics > 2nd harmonics
- **Above 10 mHz**: Reduced harmonic generation
- **Frequency Mixing**: Complex intermodulation products

### Linguistic Complexity:
- **Word Length Distribution**: Similar to human languages
- **Vocabulary Richness**: High pattern diversity
- **Syntax Patterns**: Non-random transition structures

## 🎵 Audio Correlation Insights

The analysis reveals connections between:
- **Electrical Spiking Patterns** ↔ **Audio Rhythmic Features**
- **Frequency Content** ↔ **Spectral Audio Characteristics**
- **Linguistic Complexity** ↔ **Audio Complexity Measures**

This suggests fungal networks may encode information in ways that translate to audible patterns.

## 🔬 Advanced Analysis Options

### Custom Frequency Ranges:
```python
# Modify frequency range for specific fungal species
analyzer.frequencies_mhz = np.array([0.5, 1, 2, 5, 10, 20, 50, 100])
```

### Enhanced Fuzzy Logic:
```python
# Install advanced fuzzy logic package
# pip install fuzzylogic

# Use more sophisticated membership functions
from fuzzylogic import FuzzySet, FuzzyRule
```

### Batch Analysis:
```python
# Analyze multiple files
import glob
data_files = glob.glob("DATA/processed/*.csv")

for file in data_files:
    results = analyzer.analyze_frequency_discrimination(
        analyzer.load_fungal_data(file)
    )
    # Process results...
```

## 📊 Visualization Outputs

### 1. Frequency Analysis Plots:
- THD vs. Frequency curves
- Harmonic ratio analysis
- Fuzzy classification heatmaps
- Summary statistics

### 2. Correlation Analysis:
- Electrical-audio feature correlations
- Linguistic pattern distributions
- Vocabulary analysis charts

### 3. Data Files:
- JSON results with full analysis data
- PNG visualization files
- CSV export options

## 🧪 Experimental Protocol

### Adamatzky's Original Setup:
- **Substrate**: 200g grains + hemp colonized by Pleurotus ostreatus
- **Electrodes**: Iridium-coated stainless steel sub-dermal needles
- **Signal Generator**: 4050B Series Dual Channel Function Generator
- **Data Logger**: ADC-24 with 24-bit resolution
- **Frequency Range**: 1-100 mHz (1 mHz steps, 10 mHz steps)

### Key Parameters:
- **Sampling Rate**: 1 Hz (1 second intervals)
- **Voltage Range**: ±10 V peak-to-peak
- **Window Function**: Blackman (optimal for amplitude representation)
- **Analysis Method**: Fast Fourier Transform (FFT)

## 🔬 Research Applications

### 1. Fungal Electronics:
- **Memristor Design**: Non-linear electrical properties
- **Oscillator Circuits**: Natural oscillation frequencies
- **Logic Gates**: Boolean operations via spike patterns

### 2. Bio-Inspired Computing:
- **Neuromorphic Systems**: Spike-based information processing
- **Adaptive Networks**: Self-modifying circuit topologies
- **Living Electronics**: Self-repairing, fault-tolerant systems

### 3. Communication Studies:
- **Fungal Language**: Pattern complexity analysis
- **Information Theory**: Entropy and complexity measures
- **Network Dynamics**: Collective behavior analysis

## 📚 References

1. **Adamatzky, A., et al. (2022)**. "Fungal electronics: frequency discrimination in mycelium networks"
2. **Adamatzky, A. (2018)**. "On spiking behaviour of oyster fungi Pleurotus djamor"
3. **Adamatzky, A. (2022)**. "Language of fungi derived from their electrical spiking activity"
4. **Beasley, A.E., et al. (2022)**. "Mem-fractive properties of mushrooms"

## 🤝 Contributing

This analysis framework is designed to be extensible. Key areas for enhancement:
- **Additional Frequency Ranges**: Extend beyond 100 mHz
- **Multi-Species Analysis**: Compare different fungal species
- **Environmental Factors**: Temperature, humidity, substrate effects
- **Real-time Analysis**: Live monitoring and analysis
- **Machine Learning**: Pattern recognition and prediction

## 📄 License

This analysis framework is provided for research purposes. Please cite the original Adamatzky papers when using these methods in your research.

## 🆘 Troubleshooting

### Common Issues:
1. **Missing Dependencies**: Install all requirements
2. **Data Format**: Ensure CSV files have voltage data in expected columns
3. **Memory Issues**: Reduce chunk sizes for large datasets
4. **Audio Loading**: Verify audio file formats are supported by librosa

### Performance Tips:
- Use smaller data chunks for large signals
- Reduce FFT resolution for faster analysis
- Enable parallel processing where available
- Monitor system resources during analysis

---

**Note**: This analysis implements the exact methodology from Adamatzky's research, providing a foundation for advancing fungal electronics and bio-inspired computing research. 