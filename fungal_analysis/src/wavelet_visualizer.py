import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Optional
import json
import os

# Get absolute path to AVALON root directory
AVALON_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class WaveletVisualizer:
    def __init__(self):
        self.results_dir = AVALON_ROOT / "sqrt_wavelet_results"
        self.output_dir = AVALON_ROOT / "visualizations"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def load_wavelet_data(self, recording_dir: Path) -> Dict:
        """Load wavelet analysis results from a recording directory."""
        try:
            coeffs = np.load(recording_dir / "coefficients.npy")
            magnitude = np.load(recording_dir / "magnitude.npy")
            phase = np.load(recording_dir / "phase.npy")
            
            return {
                'coefficients': coeffs,
                'magnitude': magnitude,
                'phase': phase
            }
        except Exception as e:
            print(f"Error loading data from {recording_dir}: {e}")
            return None

    def plot_2d_wavelet(self, data: Dict, recording_name: str):
        """Generate 2D visualizations of wavelet analysis."""
        plt.figure(figsize=(20, 15))
        
        # Plot 1: Magnitude Heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(data['magnitude'], cmap='viridis')
        plt.title('Wavelet Magnitude')
        plt.xlabel('Time')
        plt.ylabel('Scale')
        
        # Plot 2: Phase Heatmap
        plt.subplot(2, 2, 2)
        sns.heatmap(data['phase'], cmap='twilight')
        plt.title('Wavelet Phase')
        plt.xlabel('Time')
        plt.ylabel('Scale')
        
        # Plot 3: Scale-averaged Power
        plt.subplot(2, 2, 3)
        scale_power = np.mean(data['magnitude'], axis=1)
        plt.plot(scale_power)
        plt.title('Scale-averaged Power')
        plt.xlabel('Scale')
        plt.ylabel('Power')
        
        # Plot 4: Time-averaged Power
        plt.subplot(2, 2, 4)
        time_power = np.mean(data['magnitude'], axis=0)
        plt.plot(time_power)
        plt.title('Time-averaged Power')
        plt.xlabel('Time')
        plt.ylabel('Power')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{recording_name}_2d_analysis.png", dpi=300)
        plt.close()

    def plot_3d_wavelet(self, data: Dict, recording_name: str):
        """Generate 3D visualizations of wavelet analysis."""
        # Create time and scale meshgrid
        scales = np.arange(data['magnitude'].shape[0])
        times = np.arange(data['magnitude'].shape[1])
        T, S = np.meshgrid(times, scales)
        
        # Plot 1: 3D Surface of Magnitude
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(T, S, data['magnitude'], cmap='viridis')
        ax.set_title('3D Wavelet Magnitude Surface')
        ax.set_xlabel('Time')
        ax.set_ylabel('Scale')
        ax.set_zlabel('Magnitude')
        plt.colorbar(surf)
        plt.savefig(self.output_dir / f"{recording_name}_3d_magnitude.png", dpi=300)
        plt.close()
        
        # Plot 2: 3D Surface of Phase
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(T, S, data['phase'], cmap='twilight')
        ax.set_title('3D Wavelet Phase Surface')
        ax.set_xlabel('Time')
        ax.set_ylabel('Scale')
        ax.set_zlabel('Phase')
        plt.colorbar(surf)
        plt.savefig(self.output_dir / f"{recording_name}_3d_phase.png", dpi=300)
        plt.close()
        
        # Plot 3: 3D Scatter of Significant Points
        threshold = np.percentile(data['magnitude'], 95)  # Top 5% points
        significant_points = data['magnitude'] > threshold
        Y, X = np.where(significant_points)
        Z = data['magnitude'][Y, X]
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis', alpha=0.6)
        ax.set_title('3D Scatter of Significant Points')
        ax.set_xlabel('Time')
        ax.set_ylabel('Scale')
        ax.set_zlabel('Magnitude')
        plt.colorbar(scatter)
        plt.savefig(self.output_dir / f"{recording_name}_3d_scatter.png", dpi=300)
        plt.close()

    def generate_visualizations(self):
        """Generate visualizations for all recordings."""
        print("Generating wavelet visualizations...")
        print(f"Reading from: {self.results_dir}")
        print(f"Saving to: {self.output_dir}")
        
        for recording_dir in self.results_dir.iterdir():
            if not recording_dir.is_dir():
                continue
                
            print(f"Processing {recording_dir.name}...")
            data = self.load_wavelet_data(recording_dir)
            if data is None:
                continue
            
            try:
                # Generate 2D visualizations
                self.plot_2d_wavelet(data, recording_dir.name)
                print(f"  ✓ Generated 2D visualizations")
                
                # Generate 3D visualizations
                self.plot_3d_wavelet(data, recording_dir.name)
                print(f"  ✓ Generated 3D visualizations")
                
            except Exception as e:
                print(f"  ✗ Error generating visualizations for {recording_dir.name}: {e}")
                continue
                
        print("\nVisualization generation complete!")
        print(f"Output directory: {self.output_dir}")

def main():
    visualizer = WaveletVisualizer()
    visualizer.generate_visualizations()

if __name__ == "__main__":
    main() 