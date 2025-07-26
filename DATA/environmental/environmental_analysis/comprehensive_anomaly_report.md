# Comprehensive Anomaly Detection Report

Generated: 2025-07-18 02:16:55

## Summary

- **Total Additional Issues**: 23
- **Issue Categories**: 7

## Issue Categories

### INCONSISTENT_SPECIES_NAMING
**Count**: 1

- **Description**: Same species 'Unknown' with inconsistent naming patterns

### MISSING_ELECTRODE_INFO
**Count**: 1

- **Description**: 9/22 files missing electrode information

### MISSING_COORDINATE_REPLICATES
**Count**: 2

- **Description**: Missing coordinate replicates: [3, 4, 5]

- **Description**: Missing coordinate replicates: [3, 5, 8, 10]

### LARGE_MOISTURE_RANGE
**Count**: 1

- **Description**: Large moisture range across Moisture logger sensors may indicate calibration issues

### MISSING_TEMPORAL_PARTS
**Count**: 6

- **Description**: Expected temporal series parts missing: ['Analysis_recording_part2.csv', 'Analysis_recording_part3.csv']
  - **File**: Analysis_recording_part1.csv

- **Description**: Expected temporal series parts missing: ['Fridge_substrate_21_1_22_part2.csv', 'Fridge_substrate_21_1_22_part3.csv']
  - **File**: Fridge_substrate_21_1_22_part1.csv

- **Description**: Expected temporal series parts missing: ['Analysis_recording_part1_part2.csv', 'Analysis_recording_part1_part3.csv']
  - **File**: Analysis_recording_part1_part1.csv

- **Description**: Expected temporal series parts missing: ['Fridge_substrate_21_1_22_part1_part2.csv', 'Fridge_substrate_21_1_22_part1_part3.csv']
  - **File**: Fridge_substrate_21_1_22_part1_part1.csv

- **Description**: Expected temporal series parts missing: ['Hericium_20_4_22_part1_part2.csv', 'Hericium_20_4_22_part1_part3.csv']
  - **File**: Hericium_20_4_22_part1_part1.csv

- **Description**: Expected temporal series parts missing: ['Hericium_20_4_22_part2_part2.csv', 'Hericium_20_4_22_part2_part3.csv']
  - **File**: Hericium_20_4_22_part2_part1.csv

### ENCODING_ISSUE
**Count**: 10

- **Description**: Contains encoding characters () indicating potential encoding problems
  - **File**: 15061491/Ch1-2.csv

- **Description**: Contains encoding characters () indicating potential encoding problems
  - **File**: 15061491/Spray_in_bag_crop.csv

- **Description**: Contains encoding characters () indicating potential encoding problems
  - **File**: 15061491/Activity_pause_spray (1).csv

- **Description**: Contains encoding characters () indicating potential encoding problems
  - **File**: 15061491/New_Oyster_with spray_as_mV_seconds.csv

- **Description**: Contains encoding characters () indicating potential encoding problems
  - **File**: 15061491/Norm_vs_deep_tip.csv

- **Description**: Contains encoding characters () indicating potential encoding problems
  - **File**: 15061491/Fridge_substrate_21_1_22.csv

- **Description**: Contains encoding characters () indicating potential encoding problems
  - **File**: 15061491/GL2_dry.csv

- **Description**: Contains encoding characters () indicating potential encoding problems
  - **File**: 15061491/GL1.csv

- **Description**: Contains encoding characters () indicating potential encoding problems
  - **File**: 15061491/Hericium_20_4_22.csv

- **Description**: Contains encoding characters () indicating potential encoding problems
  - **File**: 15061491/Blue_oyster_31_5_22.csv

### UNUSUAL_BYTES_PER_LINE
**Count**: 2

- **Description**: Unusually high bytes per line (1050.7) - may indicate data corruption
  - **File**: Hericium_20_4_22.csv

- **Description**: Unusually high bytes per line (110175248.0) - may indicate data corruption
  - **File**: Spray_in_bag.csv

