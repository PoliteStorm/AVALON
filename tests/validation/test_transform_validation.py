#!/usr/bin/env python3
"""
Test script for validating the √t transform functionality.
This script runs various tests to ensure the transform is working correctly.
"""

import sys
import os
import numpy as np
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from transforms.improved_sqrt_transform import ImprovedSqrtTTransform
from validation.transform_validation_framework import ValidationFramework

def test_basic_transform():
    """Test basic transform functionality."""
    print("Testing basic transform functionality...")
    
    # Create test signal with √t scaling
    t = np.linspace(0, 100, 1000)
    sqrt_t = np.sqrt(t)
    test_signal = np.sin(2 * np.pi * 0.1 * sqrt_t) + 0.1 * np.random.normal(0, 1, 1000)
    
    # Initialize transform
    transform = ImprovedSqrtTTransform()
    
    # Test parameter ranges
    param_ranges = transform.alternative_parameter_ranges()
    
    # Test each parameter range
    for name, (k_range, tau_range) in param_ranges.items():
        print(f"  Testing {name} parameter range...")
        
        # Apply transform
        W = transform.transform_with_window(test_signal, k_range, tau_range, 'gaussian')
        
        # Check that transform produces results
        assert W.shape == (len(k_range), len(tau_range)), f"Wrong shape for {name}"
        assert np.any(np.abs(W) > 0), f"No features detected for {name}"
        
        print(f"    ✓ {name}: {W.shape}, max magnitude: {np.max(np.abs(W)):.4f}")
    
    print("  ✓ Basic transform tests passed")

def test_species_specific_parameters():
    """Test species-specific parameter optimization."""
    print("Testing species-specific parameters...")
    
    # Load configuration
    config_path = '../../config/parameters/species_config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Test each species
    for species, params in config['species_parameters'].items():
        print(f"  Testing {species} ({params['name']})...")
        
        # Create species-appropriate test signal
        k_range = params['k_range']
        tau_range = params['tau_range']
        
        # Generate test signal with expected characteristics
        t = np.linspace(0, 1000, 5000)
        sqrt_t = np.sqrt(t)
        
        # Use species-specific frequency
        freq = np.mean(k_range) / (2 * np.pi)
        test_signal = np.sin(2 * np.pi * freq * sqrt_t) + 0.1 * np.random.normal(0, 1, 5000)
        
        # Apply transform
        transform = ImprovedSqrtTTransform()
        W = transform.transform_with_window(test_signal, k_range, tau_range, 'gaussian')
        
        # Check results
        magnitude = np.abs(W)
        max_mag = np.max(magnitude)
        mean_mag = np.mean(magnitude)
        
        print(f"    ✓ {species}: max={max_mag:.4f}, mean={mean_mag:.4f}")
        
        # Verify that we detect features
        assert max_mag > mean_mag * 2, f"No significant features for {species}"
    
    print("  ✓ Species-specific parameter tests passed")

def test_validation_framework():
    """Test validation framework functionality."""
    print("Testing validation framework...")
    
    # Create test data
    t = np.linspace(0, 100, 1000)
    sqrt_t = np.sqrt(t)
    test_signal = np.sin(2 * np.pi * 0.1 * sqrt_t) + 0.1 * np.random.normal(0, 1, 1000)
    
    # Initialize validation framework
    transform = ImprovedSqrtTTransform()
    validator = ValidationFramework(transform)
    
    # Test mathematical properties
    print("  Testing mathematical properties...")
    math_results = validator.mathematical_property_validation()
    
    # Check results
    assert math_results['linearity']['is_linear'], "Transform should be linear"
    assert math_results['scale_invariance']['is_scale_invariant'], "Transform should be scale invariant"
    
    print("    ✓ Mathematical properties validated")
    
    # Test biological plausibility
    print("  Testing biological plausibility...")
    k_values = np.logspace(-1, 1, 10)
    tau_values = np.logspace(0, 2, 10)
    W = transform.transform_with_window(test_signal, k_values, tau_values)
    
    features = transform.refined_detection_methods(W, k_values, tau_values, 'statistical')
    plausible_features = validator.biological_plausibility_check(features)
    
    print(f"    ✓ Found {len(plausible_features)} biologically plausible features")
    
    print("  ✓ Validation framework tests passed")

def test_cross_validation():
    """Test cross-validation functionality."""
    print("Testing cross-validation...")
    
    # Create multiple test signals
    signals = []
    for i in range(5):
        t = np.linspace(0, 100, 1000)
        sqrt_t = np.sqrt(t)
        freq = 0.1 + 0.05 * i
        signal = np.sin(2 * np.pi * freq * sqrt_t) + 0.1 * np.random.normal(0, 1, 1000)
        signals.append(signal)
    
    # Test cross-validation
    transform = ImprovedSqrtTTransform()
    k_values = np.logspace(-1, 1, 10)
    tau_values = np.logspace(0, 2, 10)
    
    all_features = []
    for i, signal in enumerate(signals):
        W = transform.transform_with_window(signal, k_values, tau_values)
        features = transform.refined_detection_methods(W, k_values, tau_values, 'statistical')
        all_features.append(features)
        print(f"    Signal {i+1}: {len(features)} features")
    
    # Check consistency
    feature_counts = [len(f) for f in all_features]
    mean_count = np.mean(feature_counts)
    std_count = np.std(feature_counts)
    
    print(f"    Mean features: {mean_count:.2f} ± {std_count:.2f}")
    
    # Verify reasonable consistency
    assert std_count < mean_count * 0.5, "Feature detection should be consistent"
    
    print("  ✓ Cross-validation tests passed")

def main():
    """Run all tests."""
    print("="*80)
    print("FUNGAL ANALYSIS TRANSFORM VALIDATION TESTS")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        test_basic_transform()
        test_species_specific_parameters()
        test_validation_framework()
        test_cross_validation()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 