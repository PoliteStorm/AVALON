# Anomaly Detection Report

Generated: 2025-07-18 02:15:00

## Summary

- **Critical Anomalies**: 11
- **Warnings**: 23
- **Medium Issues**: 23
- **Low Issues**: 270
- **Data Quality Issues**: 2
- **Simulation Compromises**: 2

## Critical Anomalies

### EXTREME_AMPLITUDE
- **File**: Activity_pause_spray (1).csv
- **Value**: -0.108786 to 159786.000000 mV
- **Description**: Amplitude range -0.108786 to 159786.000000 mV is biologically impossible

### EXTREME_AMPLITUDE
- **File**: Norm_vs_deep_tip.csv
- **Value**: 0.304957 to 803714.000000 mV
- **Description**: Amplitude range 0.304957 to 803714.000000 mV is biologically impossible

### EXTREME_AMPLITUDE
- **File**: Fridge_substrate_21_1_22.csv
- **Value**: -0.000305 to 1642594603.000000 mV
- **Description**: Amplitude range -0.000305 to 1642594603.000000 mV is biologically impossible

### EXTREME_AMPLITUDE
- **File**: Analysis_recording.csv
- **Value**: -0.000305 to 1642594603.000000 mV
- **Description**: Amplitude range -0.000305 to 1642594603.000000 mV is biologically impossible

### EXTREME_AMPLITUDE
- **File**: Activity_pause_spray.csv
- **Value**: -0.108786 to 159786.000000 mV
- **Description**: Amplitude range -0.108786 to 159786.000000 mV is biologically impossible

### EXTREME_AMPLITUDE
- **File**: Activity_time_part2.csv
- **Value**: -0.005000 to 500009.000000 mV
- **Description**: Amplitude range -0.005000 to 500009.000000 mV is biologically impossible

### EXTREME_AMPLITUDE
- **File**: Activity_time_part3.csv
- **Value**: 0.000000 to 1000009.000000 mV
- **Description**: Amplitude range 0.000000 to 1000009.000000 mV is biologically impossible

### EXTREME_AMPLITUDE
- **File**: Analysis_recording_part1.csv
- **Value**: -0.000305 to 1642594603.000000 mV
- **Description**: Amplitude range -0.000305 to 1642594603.000000 mV is biologically impossible

### EXTREME_AMPLITUDE
- **File**: Fridge_substrate_21_1_22_part1.csv
- **Value**: -0.000305 to 1642594603.000000 mV
- **Description**: Amplitude range -0.000305 to 1642594603.000000 mV is biologically impossible

### EXTREME_AMPLITUDE
- **File**: Analysis_recording_part1_part1.csv
- **Value**: -0.000305 to 1642594603.000000 mV
- **Description**: Amplitude range -0.000305 to 1642594603.000000 mV is biologically impossible

### EXTREME_AMPLITUDE
- **File**: Fridge_substrate_21_1_22_part1_part1.csv
- **Value**: -0.000305 to 1642594603.000000 mV
- **Description**: Amplitude range -0.000305 to 1642594603.000000 mV is biologically impossible

## Simulation Compromises

### NO_BIOLOGICALLY_PLAUSIBLE_ELECTRICAL
- **Impact**: Simulation will not reflect real fungal electrical activity
- **Solution**: Amplitude normalization required for all electrical data

### EXTREME_AMPLITUDE_RANGE
- **Impact**: Simulation parameters will be unrealistic
- **Solution**: Implement amplitude clipping and normalization

