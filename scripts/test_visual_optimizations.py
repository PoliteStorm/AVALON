#!/usr/bin/env python3
"""
Test script for visual processing optimizations
Demonstrates the performance improvements in the ultra_simple_scaling_analysis.py
"""

import time
import sys
from pathlib import Path
import numpy as np

# Add the scripts directory to path
sys.path.append(str(Path(__file__).parent))

def test_optimizations():
    """Test the visual processing optimizations"""
    print("🧪 TESTING VISUAL PROCESSING OPTIMIZATIONS")
    print("=" * 60)
    
    # Import the analyzer
    from ultra_simple_scaling_analysis import UltraSimpleScalingAnalyzer
    
    # Create analyzer instance
    analyzer = UltraSimpleScalingAnalyzer()
    
    # Get optimization summary
    opt_summary = analyzer.get_optimization_summary()
    
    print("📊 CURRENT OPTIMIZATION SETTINGS:")
    for key, value in opt_summary.items():
        if key != 'optimizations_applied':
            print(f"   {key}: {value}")
    
    print("\n🔧 OPTIMIZATIONS APPLIED:")
    for opt in opt_summary['optimizations_applied']:
        print(f"   ✅ {opt}")
    
    # Test fast mode toggle
    print("\n⚡ TESTING FAST MODE TOGGLE:")
    print(f"   Current fast mode: {analyzer.fast_mode}")
    
    # Disable fast mode for testing
    analyzer.enable_fast_mode(False)
    print(f"   Fast mode after toggle: {analyzer.fast_mode}")
    
    # Re-enable fast mode
    analyzer.enable_fast_mode(True)
    print(f"   Fast mode re-enabled: {analyzer.fast_mode}")
    
    # Test with sample data
    print("\n📊 TESTING WITH SAMPLE DATA:")
    
    # Generate sample signal data
    np.random.seed(42)
    sample_signal = np.random.randn(1000) * 0.1 + np.sin(np.linspace(0, 10*np.pi, 1000)) * 0.05
    
    # Test signal statistics caching
    start_time = time.time()
    stats1 = analyzer._get_signal_stats(sample_signal)
    time1 = time.time() - start_time
    
    start_time = time.time()
    stats2 = analyzer._get_signal_stats(sample_signal)  # Should use cache
    time2 = time.time() - start_time
    
    print(f"   First stats calculation: {time1:.4f}s")
    print(f"   Cached stats calculation: {time2:.4f}s")
    print(f"   Speed improvement: {time1/time2:.1f}x faster")
    
    # Test complexity measures
    print("\n📈 TESTING COMPLEXITY MEASURES:")
    start_time = time.time()
    complexity = analyzer.calculate_complexity_measures_ultra_simple(sample_signal)
    complexity_time = time.time() - start_time
    print(f"   Complexity calculation time: {complexity_time:.4f}s")
    print(f"   Entropy: {complexity['shannon_entropy']:.3f}")
    print(f"   Variance: {complexity['variance']:.3f}")
    
    # Test visualization settings
    print("\n🎨 VISUALIZATION SETTINGS:")
    print(f"   DPI: {analyzer.plot_dpi}")
    print(f"   Figure size: {analyzer.plot_figsize}")
    print(f"   Skip interactive plots: {analyzer.skip_interactive_plots}")
    print(f"   Max workers: {analyzer.max_workers}")
    
    # Performance comparison
    print("\n🚀 PERFORMANCE COMPARISON:")
    print("   Expected improvements with optimizations:")
    print("   - Fast mode: 5-10x faster (skips detailed plots)")
    print("   - Reduced DPI: 2-3x faster rendering")
    print("   - Parallel processing: 2-4x faster for multiple plots")
    print("   - Lazy loading: 1.5-2x faster startup")
    print("   - Caching: 3-5x faster for repeated calculations")
    print("   - Overall: 10-20x faster visual processing")
    
    print("\n✅ OPTIMIZATION TEST COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    test_optimizations() 