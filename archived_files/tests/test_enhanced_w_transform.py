"""
Tests for Enhanced W-Transform Analyzer
=====================================

This module contains comprehensive tests for the enhanced W-transform analyzer,
validating its functionality against known patterns and research findings.
"""

import numpy as np
import pytest
from fungal_communication_github.enhanced_w_transform_analyzer import (
    EnhancedWTransformAnalyzer,
    WTransformConfig
)

def generate_test_signal(duration: float = 10.0, 
                        sampling_rate: float = 100.0,
                        frequencies: list = None,
                        noise_level: float = 0.1) -> tuple:
    """Generate a test signal with known components"""
    if frequencies is None:
        frequencies = [0.5, 2.0]  # Hz
    
    time = np.linspace(0, duration, int(duration * sampling_rate))
    signal = np.zeros_like(time)
    
    # Add frequency components
    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * time)
    
    # Add noise
    signal += noise_level * np.random.randn(len(time))
    
    return time, signal

def test_w_transform_basic():
    """Test basic W-transform computation"""
    # Create analyzer
    analyzer = EnhancedWTransformAnalyzer()
    
    # Generate simple test signal
    time, signal = generate_test_signal(
        duration=5.0,
        sampling_rate=100.0,
        frequencies=[1.0],
        noise_level=0.05
    )
    
    # Compute transform
    results = analyzer.compute_w_transform(signal, time)
    
    # Basic validation
    assert 'transform' in results
    assert 'power' in results
    assert 'significance' in results
    assert 'patterns' in results
    assert 'ridges' in results
    assert 'biological_interpretation' in results
    assert 'validation_metrics' in results

def test_pattern_detection():
    """Test pattern detection capabilities"""
    analyzer = EnhancedWTransformAnalyzer()
    
    # Generate multi-component signal
    time, signal = generate_test_signal(
        duration=10.0,
        frequencies=[0.5, 2.0],
        noise_level=0.1
    )
    
    results = analyzer.compute_w_transform(signal, time)
    
    # Validate pattern detection
    patterns = results['patterns']
    assert len(patterns) > 0
    
    # Check for multi-modal timescales
    multi_modal = [p for p in patterns if p['type'] == 'multi_modal_timescales']
    assert len(multi_modal) > 0
    assert multi_modal[0]['confidence'] > 0.5

def test_statistical_validation():
    """Test statistical validation framework with enhanced metrics"""
    analyzer = EnhancedWTransformAnalyzer()
    
    # Generate signal with known properties
    time = np.linspace(0, 10, 1000)
    
    # Test Case 1: Strong signal
    strong_signal = np.sin(2 * np.pi * time) + 0.01 * np.random.randn(len(time))
    results_strong = analyzer.compute_w_transform(strong_signal, time)
    
    # Validate significance results
    significance = results_strong['significance']
    assert 'significant_coefficients' in significance
    assert 'p_values' in significance
    assert 'z_scores' in significance
    assert 'normalized_p_values' in significance
    
    # Check that strong signal has significant components
    assert np.any(significance['significant_coefficients'])
    assert np.min(significance['p_values']) < 0.05
    
    # Test Case 2: Pure noise
    noise = np.random.randn(len(time))
    results_noise = analyzer.compute_w_transform(noise, time)
    
    # Validate noise results
    noise_sig = results_noise['significance']
    assert np.mean(noise_sig['significant_coefficients']) < 0.1  # Few false positives
    
    # Test Case 3: Edge case - constant signal
    constant = np.ones(len(time))
    results_constant = analyzer.compute_w_transform(constant, time)
    
    # Should handle constant signal gracefully
    assert 'error' not in results_constant['significance']
    
    # Test Case 4: Edge case - very short signal
    short_signal = np.sin(2 * np.pi * time[:10])
    results_short = analyzer.compute_w_transform(short_signal, time[:10])
    
    # Should handle short signal appropriately
    assert 'error' not in results_short['significance']

def test_z_score_computation():
    """Test Z-score computation and normalization"""
    analyzer = EnhancedWTransformAnalyzer()
    
    # Generate test signal
    time = np.linspace(0, 5, 500)
    signal = np.sin(2 * np.pi * time) + 0.1 * np.random.randn(len(time))
    
    results = analyzer.compute_w_transform(signal, time)
    significance = results['significance']
    
    # Check Z-score properties
    z_scores = significance['z_scores']
    assert np.abs(np.mean(z_scores)) < 0.1  # Should be approximately zero-mean
    assert 0.9 < np.std(z_scores) < 1.1  # Should be approximately unit variance
    
    # Check normalized p-values
    norm_p_values = significance['normalized_p_values']
    assert np.all(norm_p_values >= 0)
    assert np.all(norm_p_values <= 1)

def test_biological_interpretation():
    """Test biological interpretation capabilities"""
    config = WTransformConfig(
        min_scale=0.1,
        max_scale=10.0,
        num_scales=32
    )
    analyzer = EnhancedWTransformAnalyzer(config)
    
    # Generate biologically-relevant signal
    time, signal = generate_test_signal(
        duration=20.0,
        frequencies=[0.1, 0.5],  # Biological frequencies
        noise_level=0.1
    )
    
    results = analyzer.compute_w_transform(
        signal, time, species='Cordyceps_militaris'
    )
    
    # Check biological interpretation
    bio = results['biological_interpretation']
    assert 'process_type' in bio
    assert 'complexity_metrics' in bio
    assert 'species_analysis' in bio
    
    # Validate species-specific analysis
    species_analysis = bio['species_analysis']
    assert abs(species_analysis['expected_timing'] - 116.0) < 1e-6

def test_ridge_extraction():
    """Test ridge extraction functionality"""
    analyzer = EnhancedWTransformAnalyzer()
    
    # Generate signal with clear ridges
    time, signal = generate_test_signal(
        duration=10.0,
        frequencies=[1.0],
        noise_level=0.05
    )
    
    results = analyzer.compute_w_transform(signal, time)
    
    # Check ridge extraction
    ridges = results['ridges']
    assert len(ridges) > 0
    
    # Validate ridge structure
    ridge = ridges[0]
    assert 'scale_indices' in ridge
    assert 'k_indices' in ridge
    assert 'amplitude' in ridge
    assert len(ridge['scale_indices']) >= analyzer.config.min_ridge_length

def test_validation_metrics():
    """Test validation metrics computation"""
    analyzer = EnhancedWTransformAnalyzer()
    
    # Generate high-quality signal
    time, signal = generate_test_signal(
        duration=10.0,
        frequencies=[1.0],
        noise_level=0.01
    )
    
    results = analyzer.compute_w_transform(signal, time)
    
    # Check validation metrics
    metrics = results['validation_metrics']
    assert 'pattern_confidence' in metrics
    assert 'ridge_quality' in metrics
    assert 'statistical_strength' in metrics
    assert 'overall_quality' in metrics
    
    # Validate quality scores
    assert 0 <= metrics['overall_quality'] <= 1
    assert metrics['overall_quality'] > 0.5  # High-quality signal

def test_error_handling():
    """Test error handling in statistical computations"""
    analyzer = EnhancedWTransformAnalyzer()
    
    # Test Case 1: Invalid input shapes
    time = np.linspace(0, 1, 100)
    signal = np.random.randn(50)  # Mismatched length
    
    with pytest.raises(ValueError):
        analyzer.compute_w_transform(signal, time)
    
    # Test Case 2: NaN values
    signal_with_nan = np.full(100, np.nan)
    results_nan = analyzer.compute_w_transform(signal_with_nan, time)
    
    # Should handle NaN gracefully
    assert 'error' in results_nan['significance']
    
    # Test Case 3: Infinite values
    signal_with_inf = np.full(100, np.inf)
    results_inf = analyzer.compute_w_transform(signal_with_inf, time)
    
    # Should handle infinity gracefully
    assert 'error' in results_inf['significance']

def test_monte_carlo_stability():
    """Test stability of Monte Carlo significance testing"""
    config = WTransformConfig(
        monte_carlo_iterations=100,  # Reduced for testing
        significance_level=0.05
    )
    analyzer = EnhancedWTransformAnalyzer(config)
    
    # Generate test signal
    time = np.linspace(0, 5, 500)
    signal = np.sin(2 * np.pi * time) + 0.1 * np.random.randn(len(time))
    
    # Run multiple analyses
    results_list = []
    for _ in range(5):
        results = analyzer.compute_w_transform(signal, time)
        results_list.append(results['significance']['significant_coefficients'])
    
    # Check consistency across runs
    results_array = np.array(results_list)
    consistency = np.mean(np.std(results_array, axis=0) < 0.5)
    assert consistency > 0.9  # At least 90% of results should be consistent

def test_custom_configuration():
    """Test analyzer with custom configuration"""
    custom_config = WTransformConfig(
        min_scale=0.2,
        max_scale=5.0,
        num_scales=16,
        k_range=(-5, 5),
        num_k=32,
        monte_carlo_iterations=500,
        min_pattern_confidence=0.8
    )
    
    analyzer = EnhancedWTransformAnalyzer(custom_config)
    
    # Generate test signal
    time, signal = generate_test_signal()
    results = analyzer.compute_w_transform(signal, time)
    
    # Validate custom configuration effects
    assert len(analyzer.scales) == custom_config.num_scales
    assert len(analyzer.k_values) == custom_config.num_k
    assert min(analyzer.k_values) >= custom_config.k_range[0]
    assert max(analyzer.k_values) <= custom_config.k_range[1] 