# Adamatzky Fungal Electrical Activity Analysis Guide

## Overview

This guide documents the implementation of Adamatzky's 2023 methodology for analyzing electrical activity in fungal colonies, specifically focusing on the split-gill fungus *Schizophyllum commune*.

## Key Findings from Adamatzky 2023

### Three Families of Oscillatory Patterns

1. **Very Slow Activity (Hour Scale)**
   - Duration: ~43 minutes (2573 ± 168 seconds)
   - Amplitude: 0.16 ± 0.02 mV
   - Distance between spikes: 2656 ± 278 seconds
   - Characteristics: Symmetrical spikes

2. **Slow Activity (10-Minute Scale)**
   - Duration: ~8 minutes (457 ± 120 seconds)
   - Amplitude: 0.4 ± 0.10 mV
   - Distance between spikes: 1819 ± 532 seconds
   - Characteristics: Temporally asymmetric (20% rise time)

3. **Very Fast Activity (Half-Minute Scale)**
   - Duration: ~24 seconds (24 ± 0.07 seconds)
   - Amplitude: 0.36 ± 0.06 mV
   - Distance between spikes: 148 ± 68 seconds
   - Characteristics: Action potential-like spikes

## Implementation Details

### Wave Transform Parameters

The wave transform W(k,τ) is implemented as:

```
W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
```

Where:
- V(t) is the voltage signal
- ψ(√t/τ) is the wave function
- k is the frequency parameter
- τ is the time scale parameter

### Temporal Scale Classification

```python
def classify_temporal_scale(scale):
    if scale <= 300:      # 30-300 seconds
        return 'very_fast'
    elif scale <= 3600:   # 600-3600 seconds (10 minutes)
        return 'slow'
    else:                 # >3600 seconds (hour scale)
        return 'very_slow'
```

### FitzHugh-Nagumo Model Integration

The FitzHugh-Nagumo model is used to simulate spiking behavior:

```
∂u/∂t = c₁u(u-a)(1-u) - c₂uv + I + Dᵤ∇²u
∂v/∂t = b(u-v)
```

Parameters:
- Dᵤ = 1.0 (diffusion coefficient)
- a = 0.13 (threshold parameter)
- b = 0.013 (recovery rate)
- c₁ = 0.26 (excitability parameter)
- c₂ = 0.015-0.05 (excitability range)

## Biological Interpretation

### Very Slow Spikes
- Associated with metabolite translocation
- Speed: ~9 mm/h (matches very slow spikes)
- Related to nutrient transport and distribution

### Slow Spikes
- Associated with calcium wave propagation
- Speed: ~0.03 mm/s (matches slow spikes)
- Related to intracellular signaling

### Very Fast Spikes
- Associated with pulsating growth
- Speed: ~1500 mm/h
- Related to hyphal tip growth dynamics

## Validation Methods

### 1. Temporal Scale Alignment
- Verify that detected spikes align with Adamatzky's three families
- Check distribution ratios (expected: 60% slow/very slow)

### 2. Mathematical Consistency
- Energy conservation check
- Orthogonality validation
- Scale invariance verification

### 3. False Positive Detection
- Uniformity testing
- Randomness assessment
- Biological plausibility check

### 4. Signal Quality Assessment
- Signal-to-noise ratio calculation
- Baseline stability analysis
- Amplitude range validation

## Usage

### Basic Analysis
```python
from enhanced_adamatzky_processor import EnhancedAdamatzkyProcessor

processor = EnhancedAdamatzkyProcessor()
results = processor.process_single_file("data/fungal_signal.csv")
```

### Batch Processing
```python
results = processor.batch_process_all_files("data/")
```

### Output Structure
```
results/
├── analysis/          # Individual file analysis results
├── validation/        # Validation reports
├── visualizations/    # Generated plots and heatmaps
└── reports/          # Summary reports and documentation
```

## Expected Results

### Temporal Scale Distribution
- Very Fast: 10-20%
- Slow: 40-60%
- Very Slow: 20-40%

### Validation Scores
- Temporal Alignment: >0.7
- Mathematical Consistency: >0.8
- False Positive Rate: <0.1
- Signal Quality: >0.6

## References

1. Adamatzky, A., et al. (2023). Growing colonies of the split-gill fungus Schizophyllum commune show action potential-like spikes of extracellular electrical potential. *Nature Scientific Reports*.

2. FitzHugh, R. (1961). Impulses and physiological states in theoretical models of nerve membrane. *Biophysical Journal*.

3. Nagumo, J., et al. (1962). An active pulse transmission line simulating nerve axon. *Proceedings of the IRE*.

## Troubleshooting

### Common Issues

1. **Low Temporal Alignment Score**
   - Check signal quality and preprocessing
   - Verify sampling rate (should be 1 Hz)
   - Ensure proper time compression

2. **Poor Mathematical Consistency**
   - Review wave transform parameters
   - Check for signal artifacts
   - Validate energy conservation

3. **High False Positive Rate**
   - Apply additional filtering
   - Check baseline stability
   - Verify biological plausibility

### Performance Optimization

- Use appropriate compression factors for large datasets
- Implement parallel processing for batch analysis
- Optimize memory usage for long recordings

## Future Enhancements

1. **Machine Learning Integration**
   - Automated spike classification
   - Pattern recognition algorithms
   - Predictive modeling

2. **Real-time Analysis**
   - Streaming data processing
   - Live visualization
   - Continuous monitoring

3. **Advanced Validation**
   - Cross-correlation analysis
   - Phase synchronization metrics
   - Network connectivity analysis 