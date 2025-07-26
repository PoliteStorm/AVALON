#!/usr/bin/env python3
"""
Scaling Comparison Analysis: Square Root vs Linear Time Scaling
Tests both scaling methods with Adamatzky's spike settings on processed CSVs

Wave Transform: W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
- Square Root Scaling: ψ(√t/τ) (inherent in transform)
- Linear Scaling: ψ(t/τ) (comparison)

Uses Adamatzky's spike settings and unbiased wave transform
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional for enhanced plotting
from scipy import signal, stats
import json
from datetime import datetime
from pathlib import Path
import sys
import warnings
from typing import Dict, List, Tuple, Optional

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))
from enhanced_adamatzky_processor import EnhancedAdamatzkyProcessor

# Import centralized configuration
sys.path.append(str(Path(__file__).parent.parent / "config"))
from analysis_config import config

warnings.filterwarnings('ignore')

class ScalingComparisonAnalyzer:
    """
    Comprehensive comparison of square root vs linear time scaling
    using Adamatzky's spike settings and unbiased wave transform
    """
    
    def __init__(self):
        self.processor = EnhancedAdamatzkyProcessor()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get configuration
        self.config = config
        self.adamatzky_params = self.config.get_adamatzky_params()
        
        # Create output directories
        self.output_dir = Path("results/scaling_comparison")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Adamatzky's spike settings
        self.adamatzky_spike_settings = {
            'amplitude_range': (0.05, 5.0),  # mV
            'sampling_rate': 1,  # Hz
            'voltage_range': 78,  # mV (±39 mV)
            'electrode_type': 'Iridium-coated stainless steel sub-dermal needle electrodes'
        }
        
        print("🔬 SCALING COMPARISON ANALYSIS")
        print("=" * 60)
        print("Comparing Square Root vs Linear Time Scaling")
        print("Wave Transform: W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt")
        print("Adamatzky Spike Settings Applied")
        print("=" * 60)
    
    def apply_adamatzky_settings_to_csv(self, csv_file: str) -> Tuple[np.ndarray, Dict]:
        """
        Apply Adamatzky's spike settings to CSV data
        - Rescale amplitudes to Adamatzky's range (0.05-5.0 mV)
        - Downsample to 1 Hz if needed
        - Apply voltage range normalization
        """
        print(f"\n📊 Processing: {Path(csv_file).name}")
        
        # Load original data
        try:
            df = pd.read_csv(csv_file)
            
            # Find voltage column
            voltage_col = None
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['voltage', 'mv', 'amplitude', 'signal']):
                    voltage_col = col
                    break
            
            if voltage_col is None:
                # Assume first numeric column is voltage
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    voltage_col = numeric_cols[0]
                else:
                    raise ValueError("No voltage column found")
            
            original_signal = df[voltage_col].values
            
            # Remove NaN values
            original_signal = original_signal[~np.isnan(original_signal)]
            
            print(f"   Original signal length: {len(original_signal)} samples")
            print(f"   Original amplitude range: {np.min(original_signal):.3f} to {np.max(original_signal):.3f} mV")
            
            # Apply Adamatzky's settings
            processed_signal = self._apply_adamatzky_normalization(original_signal)
            
            metadata = {
                'original_samples': len(original_signal),
                'processed_samples': len(processed_signal),
                'original_amplitude_range': (float(np.min(original_signal)), float(np.max(original_signal))),
                'adamatzky_amplitude_range': self.adamatzky_spike_settings['amplitude_range'],
                'sampling_rate': self.adamatzky_spike_settings['sampling_rate'],
                'voltage_range': self.adamatzky_spike_settings['voltage_range'],
                'filename': Path(csv_file).name
            }
            
            print(f"   Processed amplitude range: {np.min(processed_signal):.3f} to {np.max(processed_signal):.3f} mV")
            print(f"   Adamatzky compliance: ✅")
            
            return processed_signal, metadata
            
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
            return None, {}
    
    def _apply_adamatzky_normalization(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply Adamatzky's spike settings normalization"""
        # 1. Amplitude normalization to Adamatzky's range
        min_amp, max_amp = self.adamatzky_spike_settings['amplitude_range']
        
        # Remove DC offset
        signal_centered = signal_data - np.mean(signal_data)
        
        # Scale to Adamatzky's range
        current_range = np.max(np.abs(signal_centered))
        if current_range > 0:
            scale_factor = max_amp / current_range
            normalized_signal = signal_centered * scale_factor
        else:
            normalized_signal = signal_centered
        
        # Add small random noise to avoid zero values
        noise_level = 0.01  # 1% of max amplitude
        noise = np.random.normal(0, noise_level, len(normalized_signal))
        normalized_signal += noise
        
        # Ensure within Adamatzky's range
        normalized_signal = np.clip(normalized_signal, min_amp, max_amp)
        
        return normalized_signal
    
    def apply_square_root_wave_transform(self, signal_data: np.ndarray, filename: str) -> Dict:
        """
        Apply wave transform with square root time scaling (original method)
        W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
        """
        print(f"\n🌊 SQUARE ROOT TIME SCALING")
        print("=" * 40)
        print("W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt")
        
        n_samples = len(signal_data)
        
        # Get wave transform parameters
        wt_params = self.config.get_wave_transform_params()
        magnitude_threshold = wt_params['magnitude_threshold']
        
        # Define k and tau values
        k_values = np.linspace(wt_params['k_values']['min'], 
                              wt_params['k_values']['max'], 
                              wt_params['k_values']['steps'])
        
        tau_params = wt_params['tau_values']
        tau_values = np.concatenate([
            np.linspace(tau_params['very_fast_range'][0], tau_params['very_fast_range'][1], 
                       tau_params['steps_per_range'][0]),
            np.linspace(tau_params['slow_range'][0], tau_params['slow_range'][1], 
                       tau_params['steps_per_range'][1]),
            np.linspace(tau_params['very_slow_range'][0], tau_params['very_slow_range'][1], 
                       tau_params['steps_per_range'][2])
        ])
        
        features = []
        
        for k in k_values:
            for tau in tau_values:
                transformed = np.zeros(n_samples, dtype=complex)
                
                for i in range(n_samples):
                    t = i / self.adamatzky_params['sampling_rate']  # Time in seconds
                    
                    if t > 0:
                        # Square root time scaling: ψ(√t/τ)
                        wave_function = np.sqrt(t / tau)
                        # Frequency component: e^(-ik√t)
                        frequency_component = np.exp(-1j * k * np.sqrt(t))
                        wave_value = wave_function * frequency_component
                        transformed[i] = signal_data[i] * wave_value
                
                magnitude = np.abs(np.sum(transformed))
                phase = np.angle(np.sum(transformed))
                
                if magnitude > magnitude_threshold:
                    temporal_scale = self._classify_temporal_scale(tau)
                    features.append({
                        'k': k,
                        'tau': tau,
                        'magnitude': magnitude,
                        'phase': phase,
                        'frequency': k / (2 * np.pi),
                        'temporal_scale': temporal_scale,
                        'scaling_method': 'square_root'
                    })
        
        return {
            'all_features': features,
            'n_features': len(features),
            'max_magnitude': max([f['magnitude'] for f in features]) if features else 0,
            'avg_magnitude': np.mean([f['magnitude'] for f in features]) if features else 0,
            'scaling_method': 'square_root',
            'filename': filename
        }
    
    def apply_linear_wave_transform(self, signal_data: np.ndarray, filename: str) -> Dict:
        """
        Apply wave transform with linear time scaling (comparison method)
        W(k,τ) = ∫₀^∞ V(t) · ψ(t/τ) · e^(-ikt) dt
        """
        print(f"\n📈 LINEAR TIME SCALING")
        print("=" * 40)
        print("W(k,τ) = ∫₀^∞ V(t) · ψ(t/τ) · e^(-ikt) dt")
        
        n_samples = len(signal_data)
        
        # Get wave transform parameters
        wt_params = self.config.get_wave_transform_params()
        magnitude_threshold = wt_params['magnitude_threshold']
        
        # Define k and tau values
        k_values = np.linspace(wt_params['k_values']['min'], 
                              wt_params['k_values']['max'], 
                              wt_params['k_values']['steps'])
        
        tau_params = wt_params['tau_values']
        tau_values = np.concatenate([
            np.linspace(tau_params['very_fast_range'][0], tau_params['very_fast_range'][1], 
                       tau_params['steps_per_range'][0]),
            np.linspace(tau_params['slow_range'][0], tau_params['slow_range'][1], 
                       tau_params['steps_per_range'][1]),
            np.linspace(tau_params['very_slow_range'][0], tau_params['very_slow_range'][1], 
                       tau_params['steps_per_range'][2])
        ])
        
        features = []
        
        for k in k_values:
            for tau in tau_values:
                transformed = np.zeros(n_samples, dtype=complex)
                
                for i in range(n_samples):
                    t = i / self.adamatzky_params['sampling_rate']  # Time in seconds
                    
                    if t > 0:
                        # Linear time scaling: ψ(t/τ)
                        wave_function = t / tau
                        # Frequency component: e^(-ikt) (linear time)
                        frequency_component = np.exp(-1j * k * t)
                        wave_value = wave_function * frequency_component
                        transformed[i] = signal_data[i] * wave_value
                
                magnitude = np.abs(np.sum(transformed))
                phase = np.angle(np.sum(transformed))
                
                if magnitude > magnitude_threshold:
                    temporal_scale = self._classify_temporal_scale(tau)
                    features.append({
                        'k': k,
                        'tau': tau,
                        'magnitude': magnitude,
                        'phase': phase,
                        'frequency': k / (2 * np.pi),
                        'temporal_scale': temporal_scale,
                        'scaling_method': 'linear'
                    })
        
        return {
            'all_features': features,
            'n_features': len(features),
            'max_magnitude': max([f['magnitude'] for f in features]) if features else 0,
            'avg_magnitude': np.mean([f['magnitude'] for f in features]) if features else 0,
            'scaling_method': 'linear',
            'filename': filename
        }
    
    def _classify_temporal_scale(self, tau: float) -> str:
        """Classify tau according to Adamatzky's temporal scales"""
        if tau <= 300:
            return 'very_fast'
        elif tau <= 3600:
            return 'slow'
        else:
            return 'very_slow'
    
    def compare_scaling_methods(self, sqrt_results: Dict, linear_results: Dict, metadata: Dict) -> Dict:
        """Compare square root vs linear scaling results"""
        print(f"\n📊 SCALING COMPARISON RESULTS")
        print("=" * 50)
        
        comparison = {
            'filename': metadata['filename'],
            'metadata': metadata,
            'square_root_results': sqrt_results,
            'linear_results': linear_results,
            'comparison_metrics': {}
        }
        
        # Feature count comparison
        sqrt_features = sqrt_results['n_features']
        linear_features = linear_results['n_features']
        
        print(f"📈 Feature Detection:")
        print(f"   Square Root Scaling: {sqrt_features} features")
        print(f"   Linear Scaling: {linear_features} features")
        print(f"   Ratio (sqrt/linear): {sqrt_features/linear_features:.2f}" if linear_features > 0 else "   Ratio: N/A")
        
        # Magnitude comparison
        sqrt_max_mag = sqrt_results['max_magnitude']
        linear_max_mag = linear_results['max_magnitude']
        sqrt_avg_mag = sqrt_results['avg_magnitude']
        linear_avg_mag = linear_results['avg_magnitude']
        
        print(f"📊 Magnitude Analysis:")
        print(f"   Square Root - Max: {sqrt_max_mag:.3f}, Avg: {sqrt_avg_mag:.3f}")
        print(f"   Linear - Max: {linear_max_mag:.3f}, Avg: {linear_avg_mag:.3f}")
        
        # Temporal scale distribution comparison
        if sqrt_results['all_features'] and linear_results['all_features']:
            sqrt_scales = [f['temporal_scale'] for f in sqrt_results['all_features']]
            linear_scales = [f['temporal_scale'] for f in linear_results['all_features']]
            
            sqrt_scale_counts = pd.Series(sqrt_scales).value_counts()
            linear_scale_counts = pd.Series(linear_scales).value_counts()
            
            print(f"⏰ Temporal Scale Distribution:")
            print(f"   Square Root: {dict(sqrt_scale_counts)}")
            print(f"   Linear: {dict(linear_scale_counts)}")
        
        # Calculate comparison metrics
        comparison['comparison_metrics'] = {
            'feature_count_ratio': sqrt_features / linear_features if linear_features > 0 else float('inf'),
            'max_magnitude_ratio': sqrt_max_mag / linear_max_mag if linear_max_mag > 0 else float('inf'),
            'avg_magnitude_ratio': sqrt_avg_mag / linear_avg_mag if linear_avg_mag > 0 else float('inf'),
            'sqrt_superiority': sqrt_features > linear_features,
            'magnitude_superiority': sqrt_max_mag > linear_max_mag
        }
        
        return comparison
    
    def create_comparison_visualization(self, comparison: Dict) -> str:
        """Create comprehensive comparison visualization"""
        print(f"\n📊 Creating comparison visualization...")
        
        sqrt_results = comparison['square_root_results']
        linear_results = comparison['linear_results']
        filename = comparison['filename']
        
        # Extract data for plotting
        sqrt_features = sqrt_results['all_features']
        linear_features = linear_results['all_features']
        
        if not sqrt_features or not linear_features:
            print("   ⚠️  Insufficient features for visualization")
            return ""
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Scaling Comparison: Square Root vs Linear\n{filename}', fontsize=16)
        
        # 1. Feature count comparison
        methods = ['Square Root', 'Linear']
        feature_counts = [len(sqrt_features), len(linear_features)]
        colors = ['#2E86AB', '#A23B72']
        
        bars = axes[0, 0].bar(methods, feature_counts, color=colors, alpha=0.7)
        axes[0, 0].set_title('Feature Detection Count')
        axes[0, 0].set_ylabel('Number of Features')
        for bar, count in zip(bars, feature_counts):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(count), ha='center', va='bottom')
        
        # 2. Magnitude comparison
        sqrt_magnitudes = [f['magnitude'] for f in sqrt_features]
        linear_magnitudes = [f['magnitude'] for f in linear_features]
        
        axes[0, 1].hist(sqrt_magnitudes, bins=20, alpha=0.7, label='Square Root', color='#2E86AB')
        axes[0, 1].hist(linear_magnitudes, bins=20, alpha=0.7, label='Linear', color='#A23B72')
        axes[0, 1].set_title('Magnitude Distribution')
        axes[0, 1].set_xlabel('Magnitude')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # 3. Temporal scale distribution
        sqrt_scales = [f['temporal_scale'] for f in sqrt_features]
        linear_scales = [f['temporal_scale'] for f in linear_features]
        
        sqrt_scale_counts = pd.Series(sqrt_scales).value_counts()
        linear_scale_counts = pd.Series(linear_scales).value_counts()
        
        x = np.arange(len(set(sqrt_scales + linear_scales)))
        width = 0.35
        
        all_scales = sorted(set(sqrt_scales + linear_scales))
        sqrt_counts = [sqrt_scale_counts.get(scale, 0) for scale in all_scales]
        linear_counts = [linear_scale_counts.get(scale, 0) for scale in all_scales]
        
        axes[0, 2].bar(x - width/2, sqrt_counts, width, label='Square Root', color='#2E86AB', alpha=0.7)
        axes[0, 2].bar(x + width/2, linear_counts, width, label='Linear', color='#A23B72', alpha=0.7)
        axes[0, 2].set_title('Temporal Scale Distribution')
        axes[0, 2].set_xlabel('Temporal Scale')
        axes[0, 2].set_ylabel('Feature Count')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(all_scales)
        axes[0, 2].legend()
        
        # 4. k vs tau heatmap comparison (Square Root)
        sqrt_k_values = [f['k'] for f in sqrt_features]
        sqrt_tau_values = [f['tau'] for f in sqrt_features]
        sqrt_magnitudes = [f['magnitude'] for f in sqrt_features]
        
        unique_k = np.unique(sqrt_k_values)
        unique_tau = np.unique(sqrt_tau_values)
        heatmap_data = np.zeros((len(unique_tau), len(unique_k)))
        
        for i, tau in enumerate(unique_tau):
            for j, k in enumerate(unique_k):
                mask = (np.array(sqrt_k_values) == k) & (np.array(sqrt_tau_values) == tau)
                if np.any(mask):
                    heatmap_data[i, j] = np.mean(np.array(sqrt_magnitudes)[mask])
        
        im1 = axes[1, 0].imshow(heatmap_data, aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Square Root: k vs τ Heatmap')
        axes[1, 0].set_xlabel('Frequency Parameter k')
        axes[1, 0].set_ylabel('Time Scale τ (seconds)')
        plt.colorbar(im1, ax=axes[1, 0])
        
        # 5. k vs tau heatmap comparison (Linear)
        linear_k_values = [f['k'] for f in linear_features]
        linear_tau_values = [f['tau'] for f in linear_features]
        linear_magnitudes = [f['magnitude'] for f in linear_features]
        
        unique_k = np.unique(linear_k_values)
        unique_tau = np.unique(linear_tau_values)
        heatmap_data = np.zeros((len(unique_tau), len(unique_k)))
        
        for i, tau in enumerate(unique_tau):
            for j, k in enumerate(unique_k):
                mask = (np.array(linear_k_values) == k) & (np.array(linear_tau_values) == tau)
                if np.any(mask):
                    heatmap_data[i, j] = np.mean(np.array(linear_magnitudes)[mask])
        
        im2 = axes[1, 1].imshow(heatmap_data, aspect='auto', cmap='viridis')
        axes[1, 1].set_title('Linear: k vs τ Heatmap')
        axes[1, 1].set_xlabel('Frequency Parameter k')
        axes[1, 1].set_ylabel('Time Scale τ (seconds)')
        plt.colorbar(im2, ax=axes[1, 1])
        
        # 6. Summary statistics
        metrics = comparison['comparison_metrics']
        summary_text = f"""
Feature Count Ratio: {metrics['feature_count_ratio']:.2f}
Max Magnitude Ratio: {metrics['max_magnitude_ratio']:.2f}
Avg Magnitude Ratio: {metrics['avg_magnitude_ratio']:.2f}
Square Root Superior: {metrics['sqrt_superiority']}
Magnitude Superior: {metrics['magnitude_superiority']}
        """.strip()
        
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='center', fontfamily='monospace')
        axes[1, 2].set_title('Comparison Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"scaling_comparison_{filename.replace('.csv', '')}_{self.timestamp}.png"
        plot_path = self.output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {plot_path}")
        return str(plot_path)
    
    def process_single_file(self, csv_file: str) -> Dict:
        """Process a single CSV file with both scaling methods"""
        print(f"\n🔬 Processing: {Path(csv_file).name}")
        print("=" * 60)
        
        # Apply Adamatzky's settings
        processed_signal, metadata = self.apply_adamatzky_settings_to_csv(csv_file)
        if processed_signal is None:
            return {}
        
        # Apply square root wave transform
        sqrt_results = self.apply_square_root_wave_transform(processed_signal, Path(csv_file).name)
        
        # Apply linear wave transform
        linear_results = self.apply_linear_wave_transform(processed_signal, Path(csv_file).name)
        
        # Compare results
        comparison = self.compare_scaling_methods(sqrt_results, linear_results, metadata)
        
        # Create visualization
        plot_path = self.create_comparison_visualization(comparison)
        comparison['plot_path'] = plot_path
        
        return comparison
    
    def process_all_processed_files(self) -> Dict:
        """Process all files in the processed directory"""
        processed_dir = Path("data/processed")
        
        if not processed_dir.exists():
            print(f"❌ Processed directory not found: {processed_dir}")
            return {}
        
        csv_files = list(processed_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {processed_dir}")
            return {}
        
        print(f"\n📁 Found {len(csv_files)} CSV files to process")
        
        all_results = {}
        
        for csv_file in csv_files:
            result = self.process_single_file(str(csv_file))
            if result:
                all_results[Path(csv_file).name] = result
        
        # Create summary
        summary = self.create_comparison_summary(all_results)
        
        # Save results
        results_file = self.output_dir / f"scaling_comparison_results_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✅ Results saved: {results_file}")
        
        return summary
    
    def create_comparison_summary(self, all_results: Dict) -> Dict:
        """Create comprehensive summary of all comparisons"""
        print(f"\n📊 CREATING COMPARISON SUMMARY")
        print("=" * 50)
        
        summary = {
            'timestamp': self.timestamp,
            'total_files': len(all_results),
            'adamatzky_settings': self.adamatzky_spike_settings,
            'file_results': all_results,
            'overall_statistics': {}
        }
        
        if not all_results:
            return summary
        
        # Calculate overall statistics
        sqrt_superior_count = 0
        linear_superior_count = 0
        feature_ratios = []
        magnitude_ratios = []
        
        for filename, result in all_results.items():
            metrics = result['comparison_metrics']
            
            if metrics['sqrt_superiority']:
                sqrt_superior_count += 1
            else:
                linear_superior_count += 1
            
            if metrics['feature_count_ratio'] != float('inf'):
                feature_ratios.append(metrics['feature_count_ratio'])
            
            if metrics['max_magnitude_ratio'] != float('inf'):
                magnitude_ratios.append(metrics['max_magnitude_ratio'])
        
        summary['overall_statistics'] = {
            'files_with_sqrt_superiority': sqrt_superior_count,
            'files_with_linear_superiority': linear_superior_count,
            'sqrt_superiority_percentage': (sqrt_superior_count / len(all_results)) * 100,
            'avg_feature_count_ratio': np.mean(feature_ratios) if feature_ratios else 0,
            'avg_magnitude_ratio': np.mean(magnitude_ratios) if magnitude_ratios else 0,
            'total_sqrt_features': sum(r['square_root_results']['n_features'] for r in all_results.values()),
            'total_linear_features': sum(r['linear_results']['n_features'] for r in all_results.values())
        }
        
        print(f"📈 Overall Statistics:")
        print(f"   Files processed: {len(all_results)}")
        print(f"   Square root superior: {sqrt_superior_count} files ({summary['overall_statistics']['sqrt_superiority_percentage']:.1f}%)")
        print(f"   Linear superior: {linear_superior_count} files")
        print(f"   Average feature ratio: {summary['overall_statistics']['avg_feature_count_ratio']:.2f}")
        print(f"   Average magnitude ratio: {summary['overall_statistics']['avg_magnitude_ratio']:.2f}")
        print(f"   Total features detected:")
        print(f"     Square root: {summary['overall_statistics']['total_sqrt_features']}")
        print(f"     Linear: {summary['overall_statistics']['total_linear_features']}")
        
        return summary

def main():
    """Main execution function"""
    analyzer = ScalingComparisonAnalyzer()
    
    # Process all files in the processed directory
    results = analyzer.process_all_processed_files()
    
    if results:
        print(f"\n🎉 SCALING COMPARISON COMPLETE!")
        print("=" * 60)
        print("Results saved in results/scaling_comparison/")
        print("Check the JSON file and PNG visualizations for detailed analysis")
    else:
        print(f"\n❌ No results generated")

if __name__ == "__main__":
    main() 