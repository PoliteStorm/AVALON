#!/usr/bin/env python3
"""
Performance Comparison Script
Compare original, optimized, and ultra-optimized fungal monitoring implementations
"""

import time
import numpy as np
import psutil
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Import the different implementations
try:
    from optimized_fungal_electrical_monitoring import OptimizedFungalMonitor
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False
    print("Optimized implementation not available")

try:
    from ultra_optimized_fungal_monitoring import UltraOptimizedFungalMonitor
    ULTRA_AVAILABLE = True
except ImportError:
    ULTRA_AVAILABLE = False
    print("Ultra-optimized implementation not available")

def generate_test_signal(sampling_rate=1000, duration=60, complexity='medium'):
    """Generate test signals of different complexities"""
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Base signal
    baseline = 0.5 + 0.1 * np.sin(2 * np.pi * 0.01 * t)
    
    # Add spikes based on complexity
    if complexity == 'low':
        n_spikes = 10
    elif complexity == 'medium':
        n_spikes = 30
    else:  # high
        n_spikes = 50
    
    spike_times = np.random.exponential(2.0, n_spikes)
    spike_times = np.cumsum(spike_times)
    spike_times = spike_times[spike_times < duration]
    
    signal = baseline.copy()
    for spike_time in spike_times:
        spike_idx = int(spike_time * sampling_rate)
        if spike_idx < len(signal):
            spike_duration = int(0.05 * sampling_rate)
            for i in range(min(spike_duration, len(signal) - spike_idx)):
                signal[spike_idx + i] += 0.5 * np.exp(-i / (0.01 * sampling_rate))
    
    # Add noise
    noise_level = 0.02 if complexity == 'low' else 0.03 if complexity == 'medium' else 0.04
    noise = np.random.normal(0, noise_level, len(signal))
    signal += noise
    
    return signal, sampling_rate

def benchmark_implementation(monitor_class, signal, sampling_rate, name):
    """Benchmark a specific implementation"""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Warm up
    monitor = monitor_class()
    monitor.get_species_parameters('pleurotus')
    
    # Benchmark
    start_time = time.time()
    
    if hasattr(monitor, 'analyze_recording_ultra_optimized'):
        results = monitor.analyze_recording_ultra_optimized(signal, sampling_rate)
    elif hasattr(monitor, 'analyze_recording_optimized'):
        results = monitor.analyze_recording_optimized(signal, sampling_rate)
    else:
        results = monitor.analyze_recording(signal, sampling_rate)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Memory usage
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = final_memory - initial_memory
    
    return {
        'name': name,
        'processing_time': processing_time,
        'memory_used': memory_used,
        'spikes_detected': results['stats']['n_spikes'],
        'quality_score': results['stats']['quality_score'],
        'snr': results['stats']['snr']
    }

def run_performance_comparison():
    """Run comprehensive performance comparison"""
    print("=== Fungal Electrical Monitoring Performance Comparison ===")
    
    # Test signal complexities
    complexities = ['low', 'medium', 'high']
    results_summary = {}
    
    for complexity in complexities:
        print(f"\nTesting {complexity} complexity signal...")
        
        # Generate test signal
        signal, sampling_rate = generate_test_signal(complexity=complexity)
        print(f"Signal length: {len(signal)} samples ({len(signal)/sampling_rate:.1f} seconds)")
        
        complexity_results = []
        
        # Test ultra-optimized implementation
        if ULTRA_AVAILABLE:
            try:
                result = benchmark_implementation(
                    UltraOptimizedFungalMonitor, 
                    signal, 
                    sampling_rate, 
                    'Ultra-Optimized'
                )
                complexity_results.append(result)
                print(f"Ultra-Optimized: {result['processing_time']:.3f}s, {result['memory_used']:.2f}MB")
            except Exception as e:
                print(f"Ultra-Optimized failed: {e}")
        
        # Test optimized implementation
        if OPTIMIZED_AVAILABLE:
            try:
                result = benchmark_implementation(
                    OptimizedFungalMonitor, 
                    signal, 
                    sampling_rate, 
                    'Optimized'
                )
                complexity_results.append(result)
                print(f"Optimized: {result['processing_time']:.3f}s, {result['memory_used']:.2f}MB")
            except Exception as e:
                print(f"Optimized failed: {e}")
        
        results_summary[complexity] = complexity_results
    
    # Create performance visualization
    create_performance_visualization(results_summary)
    
    # Save detailed results
    save_performance_results(results_summary)
    
    return results_summary

def create_performance_visualization(results_summary):
    """Create performance comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fungal Electrical Monitoring Performance Comparison', fontsize=16)
    
    complexities = list(results_summary.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Processing time comparison
    ax1 = axes[0, 0]
    for i, complexity in enumerate(complexities):
        if complexity in results_summary:
            times = [r['processing_time'] for r in results_summary[complexity]]
            names = [r['name'] for r in results_summary[complexity]]
            ax1.bar([f"{name}\n({complexity})" for name in names], times, 
                   color=colors[i], alpha=0.7, label=complexity)
    ax1.set_ylabel('Processing Time (seconds)')
    ax1.set_title('Processing Time Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Memory usage comparison
    ax2 = axes[0, 1]
    for i, complexity in enumerate(complexities):
        if complexity in results_summary:
            memory = [r['memory_used'] for r in results_summary[complexity]]
            names = [r['name'] for r in results_summary[complexity]]
            ax2.bar([f"{name}\n({complexity})" for name in names], memory, 
                   color=colors[i], alpha=0.7, label=complexity)
    ax2.set_ylabel('Memory Usage (MB)')
    ax2.set_title('Memory Usage Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    # Speedup comparison (relative to slowest method)
    ax3 = axes[1, 0]
    for i, complexity in enumerate(complexities):
        if complexity in results_summary and len(results_summary[complexity]) > 1:
            times = [r['processing_time'] for r in results_summary[complexity]]
            names = [r['name'] for r in results_summary[complexity]]
            max_time = max(times)
            speedups = [max_time / t for t in times]
            ax3.bar([f"{name}\n({complexity})" for name in names], speedups, 
                   color=colors[i], alpha=0.7, label=complexity)
    ax3.set_ylabel('Speedup Factor')
    ax3.set_title('Speedup Comparison')
    ax3.tick_params(axis='x', rotation=45)
    
    # Quality comparison
    ax4 = axes[1, 1]
    for i, complexity in enumerate(complexities):
        if complexity in results_summary:
            qualities = [r['quality_score'] for r in results_summary[complexity]]
            names = [r['name'] for r in results_summary[complexity]]
            ax4.bar([f"{name}\n({complexity})" for name in names], qualities, 
                   color=colors[i], alpha=0.7, label=complexity)
    ax4.set_ylabel('Quality Score')
    ax4.set_title('Quality Score Comparison')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Performance visualization saved to {filename}")
    
    plt.show()

def save_performance_results(results_summary):
    """Save detailed performance results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_results_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Prepare results for JSON
    json_results = {}
    for complexity, results in results_summary.items():
        json_results[complexity] = []
        for result in results:
            json_result = {}
            for key, value in result.items():
                json_result[key] = convert_numpy(value)
            json_results[complexity].append(json_result)
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Detailed performance results saved to {filename}")
    
    # Print summary
    print("\n=== Performance Summary ===")
    for complexity, results in results_summary.items():
        print(f"\n{complexity.upper()} Complexity:")
        for result in results:
            print(f"  {result['name']}:")
            print(f"    Time: {result['processing_time']:.3f}s")
            print(f"    Memory: {result['memory_used']:.2f}MB")
            print(f"    Spikes: {result['spikes_detected']}")
            print(f"    Quality: {result['quality_score']:.3f}")

def main():
    """Main performance comparison function"""
    print("Starting performance comparison...")
    
    # Check available implementations
    print(f"Available implementations:")
    print(f"- Ultra-Optimized: {ULTRA_AVAILABLE}")
    print(f"- Optimized: {OPTIMIZED_AVAILABLE}")
    
    if not ULTRA_AVAILABLE and not OPTIMIZED_AVAILABLE:
        print("No optimized implementations available for comparison!")
        return
    
    # Run performance comparison
    results = run_performance_comparison()
    
    print("\nPerformance comparison completed!")

if __name__ == "__main__":
    main() 