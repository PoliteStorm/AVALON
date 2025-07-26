# Fungal Electrical Activity Monitoring Parameters
## Comprehensive Parameter Guide for Adamatzky's Method

This document provides a complete parameter reference for monitoring fungal electrical activity using Adamatzky's spike detection method. The current implementation focuses on traditional spike detection and does not include the wave transform analysis.

### 1. Adamatzky Method Parameters

#### Core Detection Parameters
- **baseline_threshold**: 0.1 mV (minimum voltage change to consider as activity)
- **threshold_multiplier**: 1.0 (multiplier for adaptive threshold calculation)
- **adaptive_threshold**: True (use signal variability for dynamic threshold)
- **min_isi**: 0.1 seconds (minimum inter-spike interval)
- **max_isi**: 10.0 seconds (maximum inter-spike interval)
- **spike_duration**: 0.05 seconds (expected spike duration)
- **min_spike_amplitude**: 0.05 mV (minimum spike amplitude)
- **max_spike_amplitude**: 5.0 mV (maximum spike amplitude)
- **min_snr**: 3.0 (minimum signal-to-noise ratio)
- **baseline_stability**: 0.1 mV (maximum baseline variation)

#### Quality Control Parameters
- **baseline_calculation**: 'median' (robust baseline estimation)
- **spike_validation**: True (validate detected spikes)
- **amplitude_constraints**: True (enforce amplitude limits)
- **isi_constraints**: True (enforce inter-spike interval limits)

### 2. Data Acquisition Parameters

#### Hardware Parameters
- **sampling_rate**: 1000 Hz (minimum recommended: 100 Hz)
- **recording_duration**: 3600 seconds (1 hour default)
- **buffer_size**: 10000 samples
- **electrode_impedance**: 1e6 Ω (1 MΩ minimum)
- **amplifier_gain**: 1000 (minimum: 100)
- **filter_bandwidth**: [0.1, 100] Hz (bandpass filter)

#### Environmental Parameters
- **temperature_range**: [18, 25] °C (optimal fungal growth)
- **humidity_range**: [60, 80] % (relative humidity)
- **light_conditions**: 'controlled' or 'natural'
- **substrate_moisture**: [70, 90] % (substrate water content)

### 3. Analysis Parameters

#### Statistical Parameters
- **confidence_level**: 0.95 (95% confidence intervals)
- **p_value_threshold**: 0.05 (statistical significance)
- **min_spikes**: 10 (minimum spikes for analysis)
- **max_spikes**: 10000 (maximum spikes to process)
- **min_quality_score**: 0.7 (minimum recording quality)

#### Validation Parameters
- **validation_split**: 0.2 (20% for validation)
- **cv_folds**: 5 (cross-validation folds)
- **bootstrap_samples**: 1000 (bootstrap iterations)

### 4. Species-Specific Parameters

#### Pleurotus ostreatus (Oyster Mushroom)
- **baseline_threshold**: 0.15 mV
- **spike_threshold**: 0.2 mV
- **min_isi**: 0.2 seconds
- **max_amplitude**: 3.0 mV
- **typical_frequency**: 0.5 Hz

#### Hericium erinaceus (Lion's Mane)
- **baseline_threshold**: 0.1 mV
- **spike_threshold**: 0.15 mV
- **min_isi**: 0.1 seconds
- **max_amplitude**: 2.0 mV
- **typical_frequency**: 1.0 Hz

#### Rhizopus oryzae (Bread Mold)
- **baseline_threshold**: 0.2 mV
- **spike_threshold**: 0.25 mV
- **min_isi**: 0.3 seconds
- **max_amplitude**: 4.0 mV
- **typical_frequency**: 0.3 Hz

### 5. False Positive Prevention Parameters

#### Data Quality Checks
- **min_voltage_range**: 0.05 mV (minimum voltage variation)
- **max_voltage_range**: 10.0 mV (maximum voltage variation)
- **baseline_drift_limit**: 0.5 mV (maximum baseline drift)
- **noise_floor**: 0.01 mV (minimum detectable signal)

#### Spike Validation
- **min_spike_width**: 0.01 seconds (minimum spike duration)
- **max_spike_width**: 0.2 seconds (maximum spike duration)
- **spike_symmetry_threshold**: 0.8 (spike shape validation)
- **refractory_period**: 0.05 seconds (minimum time between spikes)

### 6. Real-Time Processing Parameters

#### Performance Parameters
- **processing_window**: 10 seconds (analysis window size)
- **overlap**: 0.5 (50% overlap between windows)
- **update_rate**: 1 Hz (analysis update frequency)
- **max_latency**: 0.1 seconds (maximum processing delay)

#### Alert Parameters
- **spike_rate_threshold**: 0.1 Hz (minimum spike rate for alert)
- **burst_detection**: True (detect burst firing patterns)
- **anomaly_detection**: True (detect unusual patterns)
- **alert_cooldown**: 60 seconds (minimum time between alerts)

### 7. Output and Reporting Parameters

#### Data Export
- **save_raw_data**: True (save raw voltage recordings)
- **save_processed_data**: True (save processed signals)
- **save_spike_times**: True (save spike timing data)
- **save_statistics**: True (save statistical summaries)

#### Visualization Parameters
- **plot_spikes**: True (generate spike plots)
- **plot_baseline**: True (show baseline analysis)
- **plot_spectrum**: True (show frequency spectrum)
- **plot_quality_metrics**: True (show quality indicators)

### 8. Integration with Wave Transform (Future Enhancement)

**Note**: The current implementation only uses Adamatzky's spike detection method. To integrate the wave transform for enhanced pattern detection, additional parameters would be needed:

#### Wave Transform Parameters (Proposed)
- **wave_scale_range**: [0.1, 10.0] (scale parameter range)
- **wave_shift_range**: [0, 100] (shift parameter range)
- **wave_threshold**: 0.1 (wave transform threshold)
- **wave_confidence**: 0.8 (wave transform confidence)
- **integration_weight**: 0.5 (weight for combining methods)

#### Combined Analysis Parameters
- **method_combination**: 'weighted_average' or 'voting'
- **spike_wave_alignment**: True (align spike and wave features)
- **cross_validation**: True (validate both methods)
- **ensemble_threshold**: 0.7 (threshold for ensemble detection)

### 9. Implementation Notes

#### Current Limitations
- Only implements Adamatzky's spike detection method
- No wave transform integration
- Limited to single-channel recordings
- No real-time processing optimization

#### Recommended Enhancements
- Integrate wave transform for multi-scale pattern detection
- Add multi-channel recording support
- Implement real-time processing pipeline
- Add machine learning-based spike classification
- Include environmental correlation analysis

### 10. Usage Example

```python
# Initialize monitor with default parameters
monitor = FungalElectricalMonitor()

# Set species-specific parameters
monitor.get_species_parameters('pleurotus')

# Analyze recording
results = monitor.analyze_recording(voltage_data, sampling_rate)

# Access results
spike_count = results['stats']['n_spikes']
quality_score = results['stats']['quality_score']
spike_rate = results['stats']['spike_rate']
```

This parameter guide ensures reliable fungal electrical activity monitoring using Adamatzky's established method. For enhanced pattern detection, integration with the wave transform would provide multi-scale analysis capabilities. 