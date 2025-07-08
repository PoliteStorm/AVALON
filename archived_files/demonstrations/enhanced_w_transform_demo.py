"""
Enhanced W-Transform Analysis Demonstration
========================================

This script demonstrates the use of the enhanced W-transform analyzer
for analyzing fungal communication patterns. It includes examples of:

1. Basic signal analysis
2. Pattern detection
3. Biological interpretation
4. Statistical validation
5. Visualization of results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import seaborn as sns
from datetime import datetime
from typing import Dict, Any
import scipy.stats as stats
from scipy.stats import norm

from fungal_communication_github.enhanced_w_transform_analyzer import (
    EnhancedWTransformAnalyzer,
    WTransformConfig
)

def load_experimental_data(data_file: str) -> tuple:
    """Load experimental voltage data"""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    voltage_data = np.array(data['voltage_measurements'])
    time_data = np.array(data['time_points'])
    return time_data, voltage_data

def plot_w_transform_results(results: Dict[str, Any], 
                           output_dir: str,
                           species: str = None):
    """Create visualizations of W-transform results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # 1. Power spectrum heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(
        np.log10(results['power'] + 1e-10),
        aspect='auto',
        origin='lower',
        extent=[
            min(results['parameters']['frequencies']),
            max(results['parameters']['frequencies']),
            min(results['parameters']['scales']),
            max(results['parameters']['scales'])
        ]
    )
    plt.colorbar(label='Log10 Power')
    plt.xlabel('Frequency (k)')
    plt.ylabel('Scale (τ)')
    plt.title('W-Transform Power Spectrum')
    plt.savefig(output_dir / f'w_transform_power_{timestamp}.png')
    plt.close()
    
    # 2. Pattern detection results
    plt.figure(figsize=(10, 6))
    pattern_types = [p['type'] for p in results['patterns']]
    confidences = [p['confidence'] for p in results['patterns']]
    
    if pattern_types:
        plt.barh(pattern_types, confidences)
        plt.xlabel('Confidence')
        plt.title('Detected Patterns')
        plt.tight_layout()
        plt.savefig(output_dir / f'pattern_confidence_{timestamp}.png')
    plt.close()
    
    # 3. Ridge visualization
    if results['ridges']:
        plt.figure(figsize=(12, 8))
        for ridge in results['ridges']:
            plt.plot(
                ridge['k_indices'],
                ridge['scale_indices'],
                'o-',
                alpha=0.6,
                label=f'Ridge (length={len(ridge["scale_indices"])})'
            )
        plt.xlabel('Frequency Index')
        plt.ylabel('Scale Index')
        plt.title('W-Transform Ridges')
        plt.legend()
        plt.savefig(output_dir / f'ridges_{timestamp}.png')
        plt.close()
    
    # 4. Statistical significance
    plt.figure(figsize=(12, 8))
    plt.imshow(
        results['significance']['significant_coefficients'],
        aspect='auto',
        origin='lower',
        cmap='RdYlBu',
        extent=[
            min(results['parameters']['frequencies']),
            max(results['parameters']['frequencies']),
            min(results['parameters']['scales']),
            max(results['parameters']['scales'])
        ]
    )
    plt.colorbar(label='Significant')
    plt.xlabel('Frequency (k)')
    plt.ylabel('Scale (τ)')
    plt.title('Statistical Significance Map')
    plt.savefig(output_dir / f'significance_{timestamp}.png')
    plt.close()
    
    # 5. Validation metrics
    metrics = results['validation_metrics']
    plt.figure(figsize=(8, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    plt.bar(metric_names, metric_values)
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.title('Validation Metrics')
    plt.tight_layout()
    plt.savefig(output_dir / f'validation_metrics_{timestamp}.png')
    plt.close()
    
    # Add new statistical validation plots
    
    # 6. Z-score distribution
    plt.figure(figsize=(10, 6))
    z_scores = results['significance']['z_scores'].flatten()
    plt.hist(z_scores, bins=50, density=True, alpha=0.7)
    
    # Add theoretical normal distribution
    x = np.linspace(min(z_scores), max(z_scores), 100)
    plt.plot(x, norm.pdf(x, 0, 1), 'r-', lw=2, label='Theoretical Normal')
    
    plt.xlabel('Z-Score')
    plt.ylabel('Density')
    plt.title('Z-Score Distribution')
    plt.legend()
    plt.savefig(output_dir / f'z_scores_{timestamp}.png')
    plt.close()
    
    # 7. P-value distribution
    plt.figure(figsize=(10, 6))
    p_values = results['significance']['normalized_p_values'].flatten()
    plt.hist(p_values, bins=50, density=True, alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Uniform Distribution')
    plt.xlabel('P-Value')
    plt.ylabel('Density')
    plt.title('P-Value Distribution')
    plt.legend()
    plt.savefig(output_dir / f'p_values_{timestamp}.png')
    plt.close()
    
    # Save enhanced numerical results
    results_file = output_dir / f'w_transform_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json_results = {
            'patterns': results['patterns'],
            'validation_metrics': results['validation_metrics'],
            'biological_interpretation': results['biological_interpretation'],
            'statistical_validation': {
                'z_score_stats': {
                    'mean': float(np.mean(z_scores)),
                    'std': float(np.std(z_scores)),
                    'normality_test': float(stats.normaltest(z_scores)[1])
                },
                'p_value_stats': {
                    'mean': float(np.mean(p_values)),
                    'uniformity_test': float(stats.kstest(p_values, 'uniform')[1])
                }
            },
            'parameters': {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results['parameters'].items()
            }
        }
        json.dump(json_results, f, indent=2)

def print_statistical_summary(results: Dict[str, Any]):
    """Print summary of statistical validation results"""
    significance = results['significance']
    z_scores = significance['z_scores'].flatten()
    p_values = significance['normalized_p_values'].flatten()
    
    print("\n📊 Statistical Validation Summary:")
    print(f"  • Significant Components: {np.mean(significance['significant_coefficients'])*100:.1f}%")
    print(f"  • Z-Score Statistics:")
    print(f"    - Mean: {np.mean(z_scores):.3f}")
    print(f"    - Std Dev: {np.std(z_scores):.3f}")
    print(f"    - Normality Test p-value: {stats.normaltest(z_scores)[1]:.3f}")
    print(f"  • P-Value Statistics:")
    print(f"    - Mean: {np.mean(p_values):.3f}")
    print(f"    - Uniformity Test p-value: {stats.kstest(p_values, 'uniform')[1]:.3f}")

def main():
    """Main demonstration function"""
    print("🍄 Enhanced W-Transform Analysis Demonstration")
    print("============================================")
    
    # Initialize analyzer with research-backed configuration
    config = WTransformConfig(
        min_scale=0.1,
        max_scale=10.0,
        num_scales=32,
        k_range=(-5, 5),
        num_k=64,
        monte_carlo_iterations=1000,
        min_pattern_confidence=0.7
    )
    
    analyzer = EnhancedWTransformAnalyzer(config)
    print("\n✨ Initialized W-transform analyzer with research-backed configuration")
    
    try:
        time_data, voltage_data = load_experimental_data(
            'research_results/mushroom_translation_20250704_224618.json'
        )
        print("\n📊 Loaded experimental data")
        
        # Analyze data for multiple species
        species_list = [
            'Cordyceps_militaris',
            'Pleurotus_djamor',
            'Schizophyllum_commune'
        ]
        
        for species in species_list:
            print(f"\n🔍 Analyzing patterns for {species}")
            
            # Compute W-transform
            results = analyzer.compute_w_transform(
                voltage_data,
                time_data,
                species=species
            )
            
            # Plot results with enhanced visualizations
            plot_w_transform_results(
                results,
                output_dir=f'research_results/w_transform_{species}',
                species=species
            )
            
            # Print enhanced results summary
            print("\n📈 Key Findings:")
            print(f"  • Detected {len(results['patterns'])} significant patterns")
            print(f"  • Overall quality score: {results['validation_metrics']['overall_quality']:.2f}")
            
            bio = results['biological_interpretation']
            print(f"  • Dominant process: {bio['process_type']}")
            
            if 'species_analysis' in bio:
                timing_match = bio['species_analysis']['timing_match_quality']
                print(f"  • Species timing match: {timing_match:.2%}")
            
            # Print statistical validation summary
            print_statistical_summary(results)
            
            print("\n🎨 Generated visualizations and saved results")
        
        print("\n✅ Analysis complete!")
        
    except FileNotFoundError:
        print("\n⚠️  Demo data file not found. Using synthetic data instead...")
        
        # Generate synthetic data with known statistical properties
        duration = 60.0  # seconds
        sampling_rate = 100.0  # Hz
        time_data = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Create synthetic signal with biological-like components
        voltage_data = (
            0.5 * np.sin(2 * np.pi * 0.1 * time_data) +  # Slow metabolic
            0.3 * np.sin(2 * np.pi * 0.5 * time_data) +  # Information processing
            0.2 * np.sin(2 * np.pi * 1.0 * time_data) +  # Active signaling
            0.1 * np.random.randn(len(time_data))        # Noise
        )
        
        # Add non-stationary component
        envelope = 1 + 0.5 * np.sin(2 * np.pi * 0.02 * time_data)
        voltage_data *= envelope
        
        # Analyze synthetic data
        results = analyzer.compute_w_transform(voltage_data, time_data)
        
        # Plot results with enhanced visualizations
        plot_w_transform_results(
            results,
            output_dir='research_results/w_transform_synthetic'
        )
        
        print("\n📈 Synthetic Data Analysis Complete!")
        print(f"  • Detected {len(results['patterns'])} patterns")
        print(f"  • Overall quality: {results['validation_metrics']['overall_quality']:.2f}")
        
        # Print statistical validation summary
        print_statistical_summary(results)

if __name__ == '__main__':
    main() 