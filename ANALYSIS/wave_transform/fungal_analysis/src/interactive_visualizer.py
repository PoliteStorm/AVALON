import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional, Tuple
import pandas as pd

class InteractiveVisualizer:
    """Generate interactive 3D visualizations of wavelet analysis results."""
    
    def __init__(self, results_dir: str = "fungal_analysis/sqrt_wavelet_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir.parent / "visualizations/interactive"
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
            print(f"Error loading data from {recording_dir}: {str(e)}")
            return None

    def create_3d_surface(self, data: np.ndarray, title: str, 
                         x_label: str = "Time", 
                         y_label: str = "Scale",
                         z_label: str = "Magnitude") -> go.Figure:
        """Create an interactive 3D surface plot."""
        # Create meshgrid for proper 3D visualization
        x = np.arange(data.shape[1])
        y = np.arange(data.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Create the 3D surface
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=data)])
        
        # Update layout with better 3D presentation
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label,
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def create_3d_scatter(self, data: np.ndarray, threshold: float = 95,
                         title: str = "Significant Points") -> go.Figure:
        """Create 3D scatter plot of significant points."""
        # Find significant points (above threshold percentile)
        threshold_value = np.percentile(data, threshold)
        significant_points = np.where(data > threshold_value)
        
        # Create scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=significant_points[1],  # Time points
            y=significant_points[0],  # Scale points
            z=data[significant_points],
            mode='markers',
            marker=dict(
                size=5,
                color=data[significant_points],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Magnitude")
            )
        )])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Time",
                yaxis_title="Scale",
                zaxis_title="Magnitude"
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def create_phase_volume(self, magnitude: np.ndarray, phase: np.ndarray,
                          title: str = "Phase Volume") -> go.Figure:
        """Create 3D volume visualization combining magnitude and phase."""
        # Create a 3D grid
        x = np.arange(magnitude.shape[1])
        y = np.arange(magnitude.shape[0])
        z = np.linspace(0, 1, 50)  # Create 50 slices
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Create volume data
        volume = np.zeros((magnitude.shape[1], magnitude.shape[0], 50))
        for i in range(50):
            volume[:,:,i] = magnitude.T * (1 - i/50)  # Fade with depth
        
        # Create the figure
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=volume.flatten(),
            opacity=0.1,  # Single opacity value
            surface_count=20,  # Number of iso-surfaces
            colorscale='Viridis',
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        
        # Add phase information as surface
        phase_surface = go.Surface(
            x=X[:,:,0],
            y=Y[:,:,0],
            z=np.zeros_like(X[:,:,0]),
            surfacecolor=phase.T,
            colorscale='phase',
            opacity=0.5,
            showscale=True,
            colorbar=dict(title="Phase")
        )
        fig.add_trace(phase_surface)
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Time",
                yaxis_title="Scale",
                zaxis_title="Magnitude",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def visualize_recording(self, recording_name: str) -> None:
        """Create all interactive visualizations for a recording."""
        recording_dir = self.results_dir / recording_name
        if not recording_dir.exists():
            print(f"Recording directory not found: {recording_dir}")
            return
            
        # Load data
        data = self.load_wavelet_data(recording_dir)
        if data is None:
            return
            
        # Create output directory for this recording
        output_dir = self.output_dir / recording_name
        output_dir.mkdir(exist_ok=True)
        
        # Generate visualizations
        
        # 1. Magnitude Surface
        mag_fig = self.create_3d_surface(
            data['magnitude'],
            title=f"Wavelet Magnitude - {recording_name}",
            z_label="Magnitude"
        )
        mag_fig.write_html(output_dir / "magnitude_surface.html")
        
        # 2. Phase Surface
        phase_fig = self.create_3d_surface(
            data['phase'],
            title=f"Wavelet Phase - {recording_name}",
            z_label="Phase"
        )
        phase_fig.write_html(output_dir / "phase_surface.html")
        
        # 3. Significant Points
        scatter_fig = self.create_3d_scatter(
            data['magnitude'],
            title=f"Significant Points - {recording_name}"
        )
        scatter_fig.write_html(output_dir / "significant_points.html")
        
        # 4. Phase Volume
        volume_fig = self.create_phase_volume(
            data['magnitude'],
            data['phase'],
            title=f"Phase Volume - {recording_name}"
        )
        volume_fig.write_html(output_dir / "phase_volume.html")
        
        print(f"Created interactive visualizations for {recording_name}")
    
    def visualize_all(self) -> None:
        """Create visualizations for all recordings."""
        # Find all recording directories
        recording_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
        
        for recording_dir in recording_dirs:
            print(f"\nProcessing {recording_dir.name}...")
            self.visualize_recording(recording_dir.name)

if __name__ == "__main__":
    visualizer = InteractiveVisualizer()
    visualizer.visualize_all() 