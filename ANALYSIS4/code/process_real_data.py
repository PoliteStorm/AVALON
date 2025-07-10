import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
from sqrt_wavelet import SqrtWaveletTransform
from wavelet_metrics import WaveletMetrics

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.complex):
            return str(obj)
        return super().default(obj)

class RealDataProcessor:
    """Process real fungal data with enhanced wavelet analysis"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.wavelet = SqrtWaveletTransform(num_scales=32, chunk_size=5000)
        self.metrics = WaveletMetrics()
        
    def load_json_data(self, file_path: Path) -> Dict:
        """Load and validate JSON data file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return {}
            
    def process_replicate(self, replicate_dir: Path, output_dir: Path) -> Dict:
        """Process a single replicate directory"""
        results = {
            'replicate': replicate_dir.name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Load data files
        electro_data = self.load_json_data(replicate_dir / 'electro.json')
        acoustic_data = self.load_json_data(replicate_dir / 'acoustic.json')
        spatial_data = self.load_json_data(replicate_dir / 'spatial.json')
        
        if not electro_data and not acoustic_data and not spatial_data:
            return {'status': 'error', 'message': 'No valid data files found'}
            
        # Process electrical data
        if electro_data:
            try:
                electro_results = self.analyze_electrical_data(electro_data)
                results['metrics']['electrical'] = electro_results
                
                # Generate visualizations
                self.plot_electrical_analysis(
                    electro_data,
                    electro_results,
                    output_dir / f"{replicate_dir.name}_electrical_analysis.png"
                )
            except Exception as e:
                print(f"Error processing electrical data: {str(e)}")
                
        # Process acoustic data
        if acoustic_data:
            try:
                acoustic_results = self.analyze_acoustic_data(acoustic_data)
                results['metrics']['acoustic'] = acoustic_results
                
                # Generate visualizations
                self.plot_acoustic_analysis(
                    acoustic_data,
                    acoustic_results,
                    output_dir / f"{replicate_dir.name}_acoustic_analysis.png"
                )
            except Exception as e:
                print(f"Error processing acoustic data: {str(e)}")
                
        # Process spatial data
        if spatial_data:
            try:
                spatial_results = self.analyze_spatial_data(spatial_data)
                results['metrics']['spatial'] = spatial_results
                
                # Generate visualizations
                self.plot_spatial_analysis(
                    spatial_data,
                    spatial_results,
                    output_dir / f"{replicate_dir.name}_spatial_analysis.png"
                )
            except Exception as e:
                print(f"Error processing spatial data: {str(e)}")
                
        return results
    
    def analyze_electrical_data(self, data: Dict) -> Dict:
        """Analyze electrical signal data"""
        if not data or 'voltage' not in data:
            return {}
            
        signal = np.array(data['voltage'])
        time = np.array(data.get('time', range(len(signal))))
        
        # Perform wavelet transform
        coeffs = self.wavelet.transform(signal)
        
        # Calculate metrics
        metrics = {
            'complexity': self.metrics.calculate_signal_complexity(signal),
            'pattern_consistency': self.metrics.calculate_pattern_consistency(signal),
            'characteristic_frequencies': self.metrics.extract_characteristic_frequencies(signal)
        }
        
        # If event markers exist, calculate response metrics
        if 'events' in data:
            for event in data['events']:
                event_time = event['time']
                metrics[f"response_latency_{event['type']}"] = \
                    self.metrics.calculate_response_latency(signal, event_time)
                    
        return metrics
    
    def analyze_acoustic_data(self, data: Dict) -> Dict:
        """Analyze acoustic signal data"""
        if not data or 'amplitude' not in data:
            return {}
            
        signal = np.array(data['amplitude'])
        time = np.array(data.get('time', range(len(signal))))
        
        # Perform wavelet transform
        coeffs = self.wavelet.transform(signal)
        
        # Calculate metrics
        metrics = {
            'complexity': self.metrics.calculate_signal_complexity(signal),
            'pattern_consistency': self.metrics.calculate_pattern_consistency(signal),
            'characteristic_frequencies': self.metrics.extract_characteristic_frequencies(signal)
        }
        
        return metrics
    
    def analyze_spatial_data(self, data: Dict) -> Dict:
        """Analyze spatial growth and network data"""
        if not data or 'coordinates' not in data:
            return {}
            
        # Extract network properties
        coordinates = np.array(data['coordinates'])
        
        metrics = {
            'num_nodes': len(coordinates),
            'network_density': self.calculate_network_density(coordinates),
            'growth_rate': self.calculate_growth_rate(data)
        }
        
        return metrics
    
    def calculate_network_density(self, coordinates: np.ndarray) -> float:
        """Calculate network density from node coordinates"""
        if len(coordinates) < 2:
            return 0.0
            
        # Calculate average distance between points
        distances = []
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distances.append(dist)
                
        return 1.0 / (np.mean(distances) + 1e-10)
    
    def calculate_growth_rate(self, data: Dict) -> float:
        """Calculate growth rate from temporal data"""
        if 'time_series' not in data:
            return 0.0
            
        time_series = data['time_series']
        if not time_series or len(time_series) < 2:
            return 0.0
            
        # Calculate average change in network size
        sizes = [ts['size'] for ts in time_series]
        times = [ts['time'] for ts in time_series]
        
        if len(sizes) < 2:
            return 0.0
            
        return (sizes[-1] - sizes[0]) / (times[-1] - times[0] + 1e-10)
    
    def plot_electrical_analysis(self, data: Dict, metrics: Dict, output_file: Path):
        """Generate visualization for electrical analysis"""
        if not data or 'voltage' not in data:
            return
            
        signal = np.array(data['voltage'])
        time = np.array(data.get('time', range(len(signal))))
        
        plt.figure(figsize=(15, 10))
        
        # Raw signal
        plt.subplot(3, 1, 1)
        plt.plot(time, signal, 'b-', label='Voltage')
        plt.title('Electrical Signal Analysis')
        plt.xlabel('Time')
        plt.ylabel('Voltage')
        plt.grid(True)
        
        # Wavelet transform
        coeffs = self.wavelet.transform(signal)
        plt.subplot(3, 1, 2)
        plt.imshow(np.abs(coeffs), aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude')
        plt.title('Wavelet Transform')
        plt.xlabel('Time')
        plt.ylabel('Scale')
        
        # Metrics summary
        plt.subplot(3, 1, 3)
        metrics_text = '\n'.join([
            f"{k}: {v:.4f}" if isinstance(v, (int, float))
            else f"{k}: {v}"
            for k, v in metrics.items()
        ])
        plt.text(0.1, 0.5, metrics_text, fontsize=10, transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title('Analysis Metrics')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_acoustic_analysis(self, data: Dict, metrics: Dict, output_file: Path):
        """Generate visualization for acoustic analysis"""
        if not data or 'amplitude' not in data:
            return
            
        signal = np.array(data['amplitude'])
        time = np.array(data.get('time', range(len(signal))))
        
        plt.figure(figsize=(15, 10))
        
        # Raw signal
        plt.subplot(3, 1, 1)
        plt.plot(time, signal, 'g-', label='Amplitude')
        plt.title('Acoustic Signal Analysis')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # Wavelet transform
        coeffs = self.wavelet.transform(signal)
        plt.subplot(3, 1, 2)
        plt.imshow(np.abs(coeffs), aspect='auto', cmap='jet')
        plt.colorbar(label='Magnitude')
        plt.title('Wavelet Transform')
        plt.xlabel('Time')
        plt.ylabel('Scale')
        
        # Metrics summary
        plt.subplot(3, 1, 3)
        metrics_text = '\n'.join([
            f"{k}: {v:.4f}" if isinstance(v, (int, float))
            else f"{k}: {v}"
            for k, v in metrics.items()
        ])
        plt.text(0.1, 0.5, metrics_text, fontsize=10, transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title('Analysis Metrics')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_spatial_analysis(self, data: Dict, metrics: Dict, output_file: Path):
        """Generate visualization for spatial analysis"""
        if not data or 'coordinates' not in data:
            return
            
        coordinates = np.array(data['coordinates'])
        
        plt.figure(figsize=(15, 10))
        
        # Network visualization
        plt.subplot(2, 1, 1)
        plt.scatter(coordinates[:, 0], coordinates[:, 1], c='r', alpha=0.6)
        plt.title('Spatial Network Analysis')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        
        # Growth time series if available
        if 'time_series' in data:
            plt.subplot(2, 1, 2)
            time_series = data['time_series']
            times = [ts['time'] for ts in time_series]
            sizes = [ts['size'] for ts in time_series]
            plt.plot(times, sizes, 'b-o')
            plt.title('Network Growth Over Time')
            plt.xlabel('Time')
            plt.ylabel('Network Size')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process real fungal data')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Base directory containing real data')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory for output files')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = RealDataProcessor(data_dir)
    
    # Process each replicate
    all_results = []
    for replicate_dir in data_dir.glob('replicate_*'):
        if replicate_dir.is_dir():
            print(f"\nProcessing {replicate_dir.name}...")
            
            # Create output subdirectory for this replicate
            replicate_output_dir = output_dir / replicate_dir.name
            replicate_output_dir.mkdir(exist_ok=True)
            
            # Process replicate
            results = processor.process_replicate(replicate_dir, replicate_output_dir)
            all_results.append(results)
            
            print(f"✓ Completed processing {replicate_dir.name}")
    
    # Save overall results using the custom encoder
    summary_file = output_dir / 'analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_replicates': len(all_results),
            'results': all_results
        }, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 