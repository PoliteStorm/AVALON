#!/usr/bin/env python3
"""
Centralized Configuration for Wave Transform Analysis
Eliminates forced parameters and ensures consistency across all scripts

This configuration system allows for:
- Dynamic parameter adjustment based on data characteristics
- Consistent Adamatzky parameters across all scripts
- Easy modification of analysis parameters
- Validation of parameter consistency
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np # Added for adaptive parameter methods

class AnalysisConfig:
    """
    Centralized configuration management for wave transform analysis
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration with default Adamatzky parameters
        
        Args:
            config_file: Optional path to custom configuration file
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Default Adamatzky 2023 validated parameters
        self.default_config = {
            'adamatzky_parameters': {
                'temporal_scales': {
                    'very_slow': {
                        'min_isi': 3600, 
                        'max_isi': float('inf'), 
                        'description': 'Hour scale (43 min avg, 2573±168s)',
                        'duration': 2573,
                        'amplitude': 0.16,
                        'distance': 2656
                    },
                    'slow': {
                        'min_isi': 600, 
                        'max_isi': 3600, 
                        'description': '10-minute scale (8 min avg, 457±120s)',
                        'duration': 457,
                        'amplitude': 0.4,
                        'distance': 1819
                    },
                    'very_fast': {
                        'min_isi': 30, 
                        'max_isi': 300, 
                        'description': 'Half-minute scale (24s avg, 24±0.07s)',
                        'duration': 24,
                        'amplitude': 0.36,
                        'distance': 148
                    }
                },
                'sampling_rate': 1,  # Hz
                'min_spike_amplitude': 0.05,  # mV
                'max_spike_amplitude': 5.0,   # mV
                'voltage_range': {'min': -39, 'max': 39},  # mV
                'wave_transform_formula': 'W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt'
            },
            'compression_settings': {
                'adaptive_compression': True,
                'min_samples': 100,
                'max_samples': 1000,
                'target_samples': 500,
                'fallback_compression': 360,  # Based on recent successful outputs
                'compression_options': [180, 360, 720, 1440, 3000, 86400]
            },
            'validation_thresholds': {
                'biological_plausibility': 0.7,
                'mathematical_consistency': 0.8,
                'false_positive_rate': 0.1,
                'signal_quality': 0.6,
                'energy_conservation': 0.8,
                'orthogonality': 0.6,
                'scale_invariance': 0.5
            },
            'wave_transform_parameters': {
                'k_values': {
                    'min': 0.1,
                    'max': 10.0,
                    'steps': 20
                },
                'tau_values': {
                    'very_fast_range': [30, 300],
                    'slow_range': [600, 3600],
                    'very_slow_range': [3600, 86400],
                    'steps_per_range': [10, 15, 10]
                },
                'magnitude_threshold': 0.01
            },
            'fitzhugh_nagumo_parameters': {
                'Du': 1.0,      # Diffusion coefficient
                'a': 0.13,       # Threshold parameter
                'b': 0.013,      # Recovery rate
                'c1': 0.26,      # Excitability parameter
                'c2_range': [0.015, 0.05]  # Excitability range
            },
            'output_directories': {
                'results': '../results/analysis',
                'visualizations': '../results/visualizations',
                'reports': '../results/reports',
                'validation': '../results/validation',
                'comparisons': '../results/comparisons',
                'docs': '../docs'
            },
            'data_directories': {
                'raw': '../data/raw',
                'processed': '../data/processed',
                'metadata': '../data/metadata'
            }
        }
        
        # Load custom config if provided
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
                self.config = self._merge_configs(self.default_config, custom_config)
        else:
            self.config = self.default_config.copy()
    
    def _merge_configs(self, default: Dict, custom: Dict) -> Dict:
        """Merge custom configuration with defaults"""
        merged = default.copy()
        for key, value in custom.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    def get_compression_factor(self, data_length: int) -> int:
        """
        Dynamically determine optimal compression factor based on data length
        
        Args:
            data_length: Number of samples in the original data
            
        Returns:
            Optimal compression factor
        """
        if not self.config['compression_settings']['adaptive_compression']:
            return self.config['compression_settings']['fallback_compression']
        
        target_samples = self.config['compression_settings']['target_samples']
        min_samples = self.config['compression_settings']['min_samples']
        max_samples = self.config['compression_settings']['max_samples']
        
        # Calculate optimal compression factor
        optimal_compression = data_length // target_samples
        
        # Ensure we stay within reasonable bounds
        if optimal_compression < 1:
            optimal_compression = 1
        elif optimal_compression > data_length:
            optimal_compression = data_length
        
        # Find the closest compression option
        compression_options = self.config['compression_settings']['compression_options']
        closest_compression = min(compression_options, 
                                key=lambda x: abs(x - optimal_compression))
        
        # Validate the result
        compressed_samples = data_length // closest_compression
        if compressed_samples < min_samples:
            # Use more aggressive compression
            for option in reversed(compression_options):
                if data_length // option >= min_samples:
                    closest_compression = option
                    break
        elif compressed_samples > max_samples:
            # Use less aggressive compression
            for option in compression_options:
                if data_length // option <= max_samples:
                    closest_compression = option
                    break
        
        return closest_compression
    
    def get_adamatzky_params(self) -> Dict[str, Any]:
        """Get Adamatzky parameters"""
        return self.config['adamatzky_parameters']
    
    def get_validation_thresholds(self) -> Dict[str, float]:
        """Get validation thresholds"""
        return self.config['validation_thresholds']
    
    def get_wave_transform_params(self) -> Dict[str, Any]:
        """Get wave transform parameters"""
        return self.config['wave_transform_parameters']
    
    def get_output_dirs(self) -> Dict[str, Path]:
        """Get output directory paths"""
        return {k: Path(v) for k, v in self.config['output_directories'].items()}
    
    def get_data_dirs(self) -> Dict[str, Path]:
        """Get data directory paths"""
        return {k: Path(v) for k, v in self.config['data_directories'].items()}
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate configuration for consistency and completeness
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check temporal scales consistency
        temporal_scales = self.config['adamatzky_parameters']['temporal_scales']
        for scale_name, scale_params in temporal_scales.items():
            if scale_params['min_isi'] >= scale_params['max_isi']:
                validation_results['is_valid'] = False
                validation_results['issues'].append(
                    f"Invalid temporal scale {scale_name}: min_isi >= max_isi"
                )
        
        # Check compression settings
        compression_settings = self.config['compression_settings']
        if compression_settings['min_samples'] >= compression_settings['max_samples']:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                "Invalid compression settings: min_samples >= max_samples"
            )
        
        # Check validation thresholds
        thresholds = self.config['validation_thresholds']
        for threshold_name, threshold_value in thresholds.items():
            if not (0 <= threshold_value <= 1):
                validation_results['is_valid'] = False
                validation_results['issues'].append(
                    f"Invalid threshold {threshold_name}: {threshold_value} (should be 0-1)"
                )
        
        # Check wave transform parameters
        wt_params = self.config['wave_transform_parameters']
        if wt_params['k_values']['min'] >= wt_params['k_values']['max']:
            validation_results['is_valid'] = False
            validation_results['issues'].append(
                "Invalid k_values: min >= max"
            )
        
        return validation_results
    
    def save_config(self, output_path: str = None) -> str:
        """
        Save current configuration to file
        
        Args:
            output_path: Optional output path
            
        Returns:
            Path to saved configuration file
        """
        if output_path is None:
            config_dir = Path("../config")
            config_dir.mkdir(exist_ok=True)
            output_path = config_dir / f"analysis_config_{self.timestamp}.json"
        
        config_to_save = {
            'timestamp': self.timestamp,
            'description': 'Wave Transform Analysis Configuration',
            'config': self.config
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        return str(output_path)
    
    def create_config_documentation(self) -> str:
        """
        Create comprehensive documentation for the configuration
        
        Returns:
            Documentation string
        """
        doc = f"""# Wave Transform Analysis Configuration

**Generated:** {self.timestamp}

## Overview
This configuration ensures consistent parameters across all wave transform analysis scripts, eliminating forced parameters and enabling dynamic adaptation based on data characteristics.

## Adamatzky Parameters

### Temporal Scales
"""
        
        for scale_name, scale_params in self.config['adamatzky_parameters']['temporal_scales'].items():
            doc += f"""
**{scale_name.replace('_', ' ').title()}:**
- Range: {scale_params['min_isi']}-{scale_params['max_isi']} seconds
- Description: {scale_params['description']}
- Duration: {scale_params['duration']} ± {scale_params['duration']*0.1:.0f} seconds
- Amplitude: {scale_params['amplitude']} mV
- Distance: {scale_params['distance']} seconds
"""
        
        doc += f"""
### Signal Parameters
- Sampling Rate: {self.config['adamatzky_parameters']['sampling_rate']} Hz
- Voltage Range: {self.config['adamatzky_parameters']['voltage_range']['min']} to {self.config['adamatzky_parameters']['voltage_range']['max']} mV
- Spike Amplitude Range: {self.config['adamatzky_parameters']['min_spike_amplitude']} to {self.config['adamatzky_parameters']['max_spike_amplitude']} mV

### Wave Transform Formula
```
{self.config['adamatzky_parameters']['wave_transform_formula']}
```

## Compression Settings
- Adaptive Compression: {self.config['compression_settings']['adaptive_compression']}
- Target Samples: {self.config['compression_settings']['target_samples']}
- Min Samples: {self.config['compression_settings']['min_samples']}
- Max Samples: {self.config['compression_settings']['max_samples']}
- Fallback Compression: {self.config['compression_settings']['fallback_compression']}x

## Validation Thresholds
"""
        
        for threshold_name, threshold_value in self.config['validation_thresholds'].items():
            doc += f"- {threshold_name.replace('_', ' ').title()}: {threshold_value}\n"
        
        doc += f"""
## Wave Transform Parameters
- k Range: {self.config['wave_transform_parameters']['k_values']['min']} to {self.config['wave_transform_parameters']['k_values']['max']}
- k Steps: {self.config['wave_transform_parameters']['k_values']['steps']}
- Magnitude Threshold: {self.config['wave_transform_parameters']['magnitude_threshold']}

## Output Directories
"""
        
        for dir_name, dir_path in self.config['output_directories'].items():
            doc += f"- {dir_name.replace('_', ' ').title()}: {dir_path}\n"
        
        return doc

    def get_adaptive_temporal_ranges(self, signal_data: np.ndarray = None) -> Dict[str, Dict]:
        """
        Get adaptive temporal ranges based on signal characteristics and Adamatzky 2023
        
        Args:
            signal_data: Optional signal data for adaptive adjustment
            
        Returns:
            Dictionary of temporal ranges with adaptive boundaries
        """
        # Base Adamatzky 2023 ranges
        base_ranges = self.config['adamatzky_parameters']['temporal_scales']
        
        if signal_data is not None:
            # Adaptive adjustment based on signal characteristics
            signal_duration = len(signal_data)  # seconds (1 Hz sampling)
            signal_std = np.std(signal_data)
            signal_range = np.max(signal_data) - np.min(signal_data)
            
            # Adjust ranges based on signal duration
            if signal_duration < 3600:  # Less than 1 hour
                # Focus on faster scales
                base_ranges['very_slow']['min_isi'] = signal_duration // 2
                base_ranges['slow']['max_isi'] = min(3600, signal_duration)
                base_ranges['very_fast']['max_isi'] = min(300, signal_duration // 4)
            
            # Adjust based on signal quality
            if signal_std < 0.01:  # Low amplitude signal
                # More sensitive thresholds
                for scale in base_ranges.values():
                    scale['amplitude'] *= 0.5
            elif signal_std > 1.0:  # High amplitude signal
                # Less sensitive thresholds
                for scale in base_ranges.values():
                    scale['amplitude'] *= 2.0
        
        return base_ranges
    
    def get_adaptive_thresholds(self, signal_data: np.ndarray = None) -> Dict[str, float]:
        """
        Get adaptive thresholds based on signal characteristics
        
        Args:
            signal_data: Optional signal data for adaptive adjustment
            
        Returns:
            Dictionary of adaptive thresholds
        """
        base_thresholds = self.config['validation_thresholds'].copy()
        
        if signal_data is not None:
            # Calculate signal characteristics
            signal_std = np.std(signal_data)
            signal_range = np.max(signal_data) - np.min(signal_data)
            signal_duration = len(signal_data)
            
            # Adaptive biological plausibility based on signal quality
            if signal_std > 0.1 and signal_range > 0.5:
                base_thresholds['biological_plausibility'] = 0.8  # High quality signal
            elif signal_std < 0.01 or signal_range < 0.1:
                base_thresholds['biological_plausibility'] = 0.5  # Low quality signal
            else:
                base_thresholds['biological_plausibility'] = 0.7  # Default
            
            # Adaptive mathematical consistency based on signal duration
            if signal_duration > 86400:  # More than 1 day
                base_thresholds['mathematical_consistency'] = 0.9  # Long recording
            elif signal_duration < 3600:  # Less than 1 hour
                base_thresholds['mathematical_consistency'] = 0.6  # Short recording
            else:
                base_thresholds['mathematical_consistency'] = 0.8  # Default
        
        return base_thresholds
    
    def get_adaptive_percentile(self, signal_data: np.ndarray) -> float:
        """
        Get adaptive percentile threshold based on signal characteristics
        
        Args:
            signal_data: Signal data for analysis
            
        Returns:
            Adaptive percentile (0-100)
        """
        signal_std = np.std(signal_data)
        signal_range = np.max(signal_data) - np.min(signal_data)
        
        # Adaptive percentile based on signal quality
        if signal_std > 0.5 and signal_range > 2.0:
            return 95.0  # High quality signal - more selective
        elif signal_std < 0.01 or signal_range < 0.1:
            return 70.0  # Low quality signal - less selective
        else:
            return 90.0  # Default
        
    def get_adaptive_multiplier(self, signal_data: np.ndarray) -> float:
        """
        Get adaptive multiplier based on signal characteristics
        
        Args:
            signal_data: Signal data for analysis
            
        Returns:
            Adaptive multiplier (0.01-0.5)
        """
        signal_std = np.std(signal_data)
        signal_range = np.max(signal_data) - np.min(signal_data)
        
        # Adaptive multiplier based on signal quality
        if signal_std > 0.5 and signal_range > 2.0:
            return 0.02  # High quality signal - more sensitive
        elif signal_std < 0.01 or signal_range < 0.1:
            return 0.3   # Low quality signal - less sensitive
        else:
            return 0.05  # Default
        
    def get_adaptive_scale_limits(self, signal_data: np.ndarray) -> Dict[str, int]:
        """
        Get adaptive scale limits based on signal complexity
        
        Args:
            signal_data: Signal data for analysis
            
        Returns:
            Dictionary with adaptive limits
        """
        signal_duration = len(signal_data)
        signal_std = np.std(signal_data)
        
        # Adaptive scale limits based on signal characteristics
        if signal_duration > 86400:  # More than 1 day
            max_scales = 20
            max_segments = 10
        elif signal_duration < 3600:  # Less than 1 hour
            max_scales = 5
            max_segments = 3
        else:
            max_scales = 10
            max_segments = 5
        
        # Adjust based on signal complexity
        if signal_std > 0.5:
            max_scales = min(max_scales + 5, 25)  # More complex signal
        elif signal_std < 0.01:
            max_scales = max(max_scales - 3, 3)    # Less complex signal
        
        return {
            'max_scales': max_scales,
            'max_segments': max_segments,
            'min_scale_distance': max(1, signal_duration // 1000)
        }
    
    def get_fitzhugh_nagumo_params(self) -> Dict[str, Any]:
        """Get FitzHugh-Nagumo parameters from Adamatzky 2023"""
        return self.config['fitzhugh_nagumo_parameters']
    
    def validate_adamatzky_compliance(self, detected_scales: list, detected_amplitudes: list) -> Dict[str, Any]:
        """
        Validate detected features against Adamatzky 2023 findings
        
        Args:
            detected_scales: List of detected temporal scales
            detected_amplitudes: List of detected amplitudes
            
        Returns:
            Validation results dictionary
        """
        adamatzky_params = self.config['adamatzky_parameters']['temporal_scales']
        
        validation = {
            'compliant': True,
            'issues': [],
            'scale_analysis': {},
            'amplitude_analysis': {},
            'recommendations': []
        }
        
        # Analyze temporal scale distribution
        very_fast_count = sum(1 for s in detected_scales if 30 <= s <= 300)
        slow_count = sum(1 for s in detected_scales if 600 <= s <= 3600)
        very_slow_count = sum(1 for s in detected_scales if s >= 3600)
        
        validation['scale_analysis'] = {
            'very_fast': {'count': very_fast_count, 'expected_range': [1, 10]},
            'slow': {'count': slow_count, 'expected_range': [1, 15]},
            'very_slow': {'count': very_slow_count, 'expected_range': [0, 5]}
        }
        
        # Check for Adamatzky compliance
        if very_fast_count == 0:
            validation['compliant'] = False
            validation['issues'].append("No very fast spikes detected (expected 24s scale)")
            validation['recommendations'].append("Check signal quality and detection sensitivity")
        
        if slow_count == 0:
            validation['compliant'] = False
            validation['issues'].append("No slow spikes detected (expected 8min scale)")
            validation['recommendations'].append("Extend recording duration or improve detection")
        
        # Analyze amplitude distribution
        if detected_amplitudes:
            mean_amplitude = np.mean(detected_amplitudes)
            std_amplitude = np.std(detected_amplitudes)
            
            validation['amplitude_analysis'] = {
                'mean': mean_amplitude,
                'std': std_amplitude,
                'expected_range': [0.05, 5.0],  # Adamatzky 2023 range
                'in_range': 0.05 <= mean_amplitude <= 5.0
            }
            
            if not validation['amplitude_analysis']['in_range']:
                validation['compliant'] = False
                validation['issues'].append(f"Amplitude {mean_amplitude:.3f} mV outside expected range [0.05, 5.0] mV")
        
        return validation

# Global configuration instance
config = AnalysisConfig() 