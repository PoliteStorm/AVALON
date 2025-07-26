#!/usr/bin/env python3
"""
Compare Improved √t Transform on Biological-like vs Control Signals
"""

import numpy as np
from improved_sqrt_transform import ImprovedSqrtTTransform

# Create test signals
length = 10000

t = np.arange(length)
sqrt_t = np.sqrt(t)

# Biological-like signal (√t structure + noise)
bio_signal = np.sin(2 * np.pi * 0.1 * sqrt_t) + 0.1 * np.random.normal(0, 1, length)

# Diffusion signal (as another biological-like example)
diffusion_signal = np.zeros(length)
for i in range(1, length):
    diffusion_signal[i] = diffusion_signal[i-1] + np.random.normal(0, np.sqrt(0.1))

# Control signals
random_noise = np.random.normal(0, 1, length)
linear_trend = 0.01 * t + np.random.normal(0, 1, length)
oscillatory = np.sin(2 * np.pi * 0.01 * t) + 0.1 * np.random.normal(0, 1, length)

signals = {
    'bio_sqrt': bio_signal,
    'bio_diffusion': diffusion_signal,
    'control_noise': random_noise,
    'control_trend': linear_trend,
    'control_oscillatory': oscillatory
}

# Best settings from previous results
window_func = 'morlet'
param_name = 'adaptive'
det_method = 'multi_scale'

# Get parameter ranges
improved_transform = ImprovedSqrtTTransform()
param_ranges = improved_transform.alternative_parameter_ranges()
k_vals, tau_vals = param_ranges[param_name]

results = {}

for name, signal in signals.items():
    print(f"\nTesting: {name}")
    W = improved_transform.transform_with_window(signal, k_vals, tau_vals, window_func)
    features = improved_transform.refined_detection_methods(W, k_vals, tau_vals, det_method)
    results[name] = {
        'features': features,
        'count': len(features)
    }
    print(f"  Features detected: {len(features)}")

# Summary comparison
print("\n=== SUMMARY COMPARISON ===")
for name, result in results.items():
    print(f"{name}: {result['count']} features")

bio_total = results['bio_sqrt']['count'] + results['bio_diffusion']['count']
control_total = results['control_noise']['count'] + results['control_trend']['count'] + results['control_oscillatory']['count']
print(f"\nBiological-like total: {bio_total}")
print(f"Control total: {control_total}")

if bio_total > control_total:
    print("\n✓ The transform finds more features in biological-like signals than in controls. It is promising for biological tests.")
else:
    print("\n✗ The transform does NOT distinguish biological-like signals from controls. It is NOT reliable for biological tests.") 