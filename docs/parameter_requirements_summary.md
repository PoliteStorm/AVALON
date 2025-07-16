# Complete Parameter Requirements for Fungal Electrical Activity Monitoring

## 🎯 **Core Adamatzky Method Parameters**

### **Threshold Parameters**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `baseline_threshold` | 0.1 mV | 0.05-0.5 mV | Minimum voltage for baseline |
| `threshold_multiplier` | 1.0 | 0.5-2.0 | Spike detection sensitivity |
| `adaptive_threshold` | True | True/False | Dynamic threshold adjustment |

### **Temporal Parameters**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `min_isi` | 0.1 s | 0.05-1.0 s | Minimum inter-spike interval |
| `max_isi` | 10.0 s | 5.0-60.0 s | Maximum ISI for burst detection |
| `spike_duration` | 0.05 s | 0.01-0.1 s | Expected spike duration |

### **Amplitude Parameters**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `min_spike_amplitude` | 0.05 mV | 0.01-0.2 mV | Minimum spike amplitude |
| `max_spike_amplitude` | 5.0 mV | 1.0-10.0 mV | Maximum spike amplitude |

### **Quality Control Parameters**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `min_snr` | 3.0 | 2.0-10.0 | Minimum signal-to-noise ratio |
| `baseline_stability` | 0.1 mV | 0.05-0.5 mV | Maximum baseline drift |

## 📊 **Data Acquisition Parameters**

### **Sampling Parameters**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `sampling_rate` | 1000 Hz | 100-10000 Hz | Data acquisition frequency |
| `recording_duration` | 3600 s | 60-86400 s | Total recording time |
| `buffer_size` | 10000 samples | 1000-100000 | Real-time processing buffer |

### **Hardware Parameters**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `electrode_impedance` | 1e6 Ω | 1e5-1e7 Ω | Electrode resistance |
| `amplifier_gain` | 1000 | 100-10000 | Signal amplification |
| `filter_bandwidth` | [0.1, 100] Hz | [0.01, 1000] Hz | Bandpass filter range |

## 🌱 **Species-Specific Parameters**

### **Pleurotus ostreatus (Oyster Mushroom)**
```python
pleurotus_params = {
    'baseline_threshold': 0.15,    # mV
    'spike_threshold': 0.2,        # mV
    'min_isi': 0.2,               # seconds
    'max_amplitude': 3.0,          # mV
    'typical_frequency': 0.5       # Hz
}
```

### **Hericium erinaceus (Lion's Mane)**
```python
hericium_params = {
    'baseline_threshold': 0.1,     # mV
    'spike_threshold': 0.15,       # mV
    'min_isi': 0.1,               # seconds
    'max_amplitude': 2.0,          # mV
    'typical_frequency': 1.0       # Hz
}
```

### **Rhizopus (Bread Mold)**
```python
rhizopus_params = {
    'baseline_threshold': 0.2,     # mV
    'spike_threshold': 0.25,       # mV
    'min_isi': 0.3,               # seconds
    'max_amplitude': 4.0,          # mV
    'typical_frequency': 0.3       # Hz
}
```

## 🌍 **Environmental Parameters**

### **Growth Conditions**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `temperature` | 25 °C | 20-30 °C | Growth chamber temperature |
| `humidity` | 80 % | 60-95 % | Relative humidity |
| `ph` | 6.5 | 5.0-8.0 | Substrate pH |

### **Substrate Parameters**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `moisture_content` | 70 % | 50-90 % | Substrate moisture level |
| `nutrient_concentration` | 1.0 g/L | 0.1-10.0 g/L | Nutrient concentration |

## 📈 **Analysis Parameters**

### **Statistical Parameters**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `confidence_level` | 0.95 | 0.90-0.99 | Statistical significance |
| `p_value_threshold` | 0.05 | 0.01-0.10 | Significance level |

### **Quality Control**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `min_spikes` | 10 | 1-100 | Minimum spikes for analysis |
| `max_spikes` | 10000 | 1000-100000 | Maximum spikes before saturation |
| `min_quality_score` | 0.7 | 0.0-1.0 | Minimum quality score |

### **Validation Parameters**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `validation_split` | 0.2 | 0.1-0.5 | Fraction for validation |
| `cv_folds` | 5 | 3-10 | Cross-validation folds |

## ⚡ **Real-Time Processing Parameters**

### **Performance Parameters**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `processing_window` | 1000 samples | 100-10000 | Real-time buffer |
| `update_rate` | 10 Hz | 1-100 Hz | Analysis update frequency |

### **Memory Parameters**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `cache_size` | 1000000 samples | 100000-10000000 | Memory cache |

## 🔍 **Wave Transform Parameters**

### **Core Transform Parameters**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `time_scales` | [1, 10, 100, 1000] | 1-10000 | Multi-scale analysis |
| `frequency_bands` | [0.1, 1.0, 10.0] Hz | 0.01-100 Hz | Frequency analysis |
| `wavelet_type` | 'morlet' | ['morlet', 'gaussian', 'ricker'] | Wavelet function |

### **Feature Extraction**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `magnitude_threshold` | 0.1 | 0.01-1.0 | Feature detection threshold |
| `freq_resolution` | 0.1 Hz | 0.01-1.0 Hz | Frequency precision |
| `time_resolution` | 1.0 s | 0.1-10.0 s | Temporal precision |

## 🚨 **False Positive Prevention Parameters**

### **Data Quality Checks**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `min_voltage_range` | 0.1 V | 0.05-1.0 V | Minimum voltage variation |
| `min_voltage_std` | 0.05 V | 0.01-0.5 V | Minimum voltage variability |
| `min_zero_crossings` | 0.001 | 0.0001-0.01 | Minimum zero crossings |

### **Coordinate Validation**
| Parameter | Value | Range | Purpose |
|-----------|-------|-------|---------|
| `max_velocity` | 100 units | 10-1000 | Maximum realistic velocity |
| `min_curvature` | 0.1 | 0.01-1.0 | Minimum movement complexity |

## 📋 **Complete Configuration Example**

```python
fungal_electrical_config = {
    # Adamatzky Method
    'adamatzky': {
        'baseline_threshold': 0.1,        # mV
        'threshold_multiplier': 1.0,
        'adaptive_threshold': True,
        'min_isi': 0.1,                   # seconds
        'max_isi': 10.0,                  # seconds
        'spike_duration': 0.05,           # seconds
        'min_spike_amplitude': 0.05,      # mV
        'max_spike_amplitude': 5.0,       # mV
        'min_snr': 3.0,
        'baseline_stability': 0.1         # mV
    },
    
    # Data Acquisition
    'acquisition': {
        'sampling_rate': 1000,            # Hz
        'recording_duration': 3600,       # seconds
        'buffer_size': 10000,             # samples
        'electrode_impedance': 1e6,       # Ω
        'amplifier_gain': 1000,
        'filter_bandwidth': [0.1, 100]    # Hz
    },
    
    # Environment
    'environment': {
        'temperature': 25,                 # °C
        'humidity': 80,                   # %
        'ph': 6.5,
        'moisture_content': 70,           # %
        'nutrient_concentration': 1.0     # g/L
    },
    
    # Analysis
    'analysis': {
        'confidence_level': 0.95,
        'p_value_threshold': 0.05,
        'min_spikes': 10,
        'max_spikes': 10000,
        'min_quality_score': 0.7,
        'validation_split': 0.2,
        'cv_folds': 5
    },
    
    # Real-time
    'realtime': {
        'processing_window': 1000,
        'update_rate': 10,                # Hz
        'cache_size': 1000000
    }
}
```

## ✅ **Parameter Validation Checklist**

### **Pre-Analysis Validation**
- [ ] **Voltage Range**: > 0.1V for meaningful signals
- [ ] **Voltage Variability**: > 0.05V std for biological activity
- [ ] **Zero Crossings**: > 0.1% of samples for oscillatory activity
- [ ] **Sampling Rate**: > 100Hz for Nyquist compliance
- [ ] **Electrode Impedance**: 100kΩ - 10MΩ for optimal coupling
- [ ] **Amplifier Gain**: > 100 for sufficient amplification

### **Species-Specific Validation**
- [ ] **Baseline Threshold**: Appropriate for species
- [ ] **Spike Amplitude Range**: Within species limits
- [ ] **Typical Frequency**: Matches species characteristics
- [ ] **Environmental Conditions**: Optimal for species growth

### **Quality Control Validation**
- [ ] **SNR**: > 3.0 for reliable detection
- [ ] **Spike Count**: Between min and max thresholds
- [ ] **Quality Score**: > 0.7 for high-quality data
- [ ] **Baseline Stability**: < 0.1V drift

## 🎯 **Key Requirements Summary**

### **Essential Parameters (Must Have)**
1. **Baseline Threshold**: 0.1 mV (fungal resting potential)
2. **Minimum ISI**: 0.1 s (refractory period)
3. **Sampling Rate**: 1000 Hz (Nyquist compliance)
4. **SNR**: > 3.0 (reliable detection)
5. **Quality Score**: > 0.7 (data quality)

### **Species-Specific Requirements**
1. **Pleurotus**: Higher baseline, slower spikes
2. **Hericium**: Lower baseline, faster spikes
3. **Rhizopus**: Highest baseline, slowest spikes

### **Environmental Requirements**
1. **Temperature**: 20-30°C (optimal growth)
2. **Humidity**: 60-95% (moisture needs)
3. **pH**: 5.0-8.0 (substrate conditions)

### **False Positive Prevention**
1. **Voltage Range**: > 0.1V (meaningful signals)
2. **Zero Crossings**: > 0.1% (oscillatory activity)
3. **Coordinate Scaling**: Realistic velocities
4. **Data Quality**: Comprehensive validation

This comprehensive parameter set ensures reliable fungal electrical activity monitoring across different species and experimental conditions, following Adamatzky's methodology while incorporating modern signal processing techniques and false positive prevention measures. 