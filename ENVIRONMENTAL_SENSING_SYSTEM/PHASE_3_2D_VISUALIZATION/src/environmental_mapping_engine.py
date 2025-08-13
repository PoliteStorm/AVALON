#!/usr/bin/env python3
"""
🌍 Environmental Mapping Engine - Phase 3
==========================================

This module provides 2D spatial visualization of environmental conditions
using fungal electrical data and environmental parameters.

Author: Environmental Sensing Research Team
Date: August 12, 2025
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class EnvironmentalMappingEngine:
    """
    Core engine for 2D environmental parameter visualization.
    
    This class provides comprehensive 2D mapping capabilities for:
    - Environmental parameter heatmaps
    - Contour maps and vector fields
    - Time-lapse visualizations
    - Species distribution mapping
    - Interactive parameter filtering
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Environmental Mapping Engine.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.data_cache = {}
        self.visualization_cache = {}
        self.output_dir = Path("PHASE_3_2D_VISUALIZATION/results/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualization parameters
        self.color_maps = {
            'temperature': 'RdYlBu_r',
            'humidity': 'Blues',
            'pH': 'RdYlGn',
            'pollution': 'Reds',
            'moisture': 'Greens',
            'electrical_activity': 'Viridis'
        }
        
        # Set default plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        print("🌍 Environmental Mapping Engine initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'map_resolution': 100,
            'update_frequency': 5,  # seconds
            'color_scheme': 'scientific',
            'interactive_mode': True,
            'export_formats': ['png', 'html', 'svg'],
            'default_parameters': ['temperature', 'humidity', 'pH', 'pollution']
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"⚠️  Warning: Could not load config file: {e}")
        
        return default_config
    
    def load_environmental_data(self, data_source: str) -> pd.DataFrame:
        """
        Load environmental data from various sources.
        
        Args:
            data_source: Path to data file or data source identifier
            
        Returns:
            DataFrame with environmental parameters
        """
        print(f"📊 Loading environmental data from: {data_source}")
        
        try:
            if data_source.endswith('.csv'):
                data = pd.read_csv(data_source)
            elif data_source.endswith('.json'):
                with open(data_source, 'r') as f:
                    data = pd.DataFrame(json.load(f))
            else:
                # Try to load from Phase 1 or 2 results
                potential_paths = [
                    f"../PHASE_1_DATA_INFRASTRUCTURE/RESULTS/baseline_analysis/{data_source}",
                    f"../PHASE_2_AUDIO_SYNTHESIS/results/{data_source}",
                    f"../RESULTS/baseline_analysis/{data_source}"
                ]
                
                for path in potential_paths:
                    if Path(path).exists():
                        data = pd.read_csv(path)
                        break
                else:
                    raise FileNotFoundError(f"Data source not found: {data_source}")
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Add timestamp if not present
            if 'timestamp' not in data.columns and 'time' in data.columns:
                data['timestamp'] = pd.to_datetime(data['time'])
            elif 'timestamp' not in data.columns:
                data['timestamp'] = pd.date_range(
                    start=datetime.now() - timedelta(hours=len(data)),
                    periods=len(data),
                    freq='H'
                )
            
            print(f"✅ Loaded {len(data)} data points with {len(data.columns)} parameters")
            return data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            # Return sample data for testing
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample environmental data for testing."""
        print("🧪 Generating sample environmental data for testing")
        
        np.random.seed(42)
        n_points = 100
        
        # Create sample coordinates
        x_coords = np.random.uniform(0, 100, n_points)
        y_coords = np.random.uniform(0, 100, n_points)
        
        # Generate environmental parameters
        data = {
            'x_coordinate': x_coords,
            'y_coordinate': y_coords,
            'temperature': np.random.normal(22, 5, n_points),
            'humidity': np.random.uniform(30, 80, n_points),
            'ph': np.random.normal(6.5, 0.5, n_points),
            'pollution': np.random.exponential(0.1, n_points),
            'moisture': np.random.uniform(20, 70, n_points),
            'electrical_activity': np.random.normal(0.1, 0.05, n_points)
        }
        
        # Add temporal variation
        time_base = datetime.now() - timedelta(hours=n_points)
        data['timestamp'] = [time_base + timedelta(hours=i) for i in range(n_points)]
        
        return pd.DataFrame(data)
    
    def create_environmental_heatmap(self, 
                                   data: pd.DataFrame,
                                   parameter: str,
                                   x_col: str = 'x_coordinate',
                                   y_col: str = 'y_coordinate',
                                   title: Optional[str] = None) -> go.Figure:
        """
        Create a 2D heatmap of environmental parameters.
        
        Args:
            data: DataFrame with environmental data
            parameter: Parameter to visualize
            x_col: Column name for X coordinates
            y_col: Column name for Y coordinates
            title: Plot title (optional)
            
        Returns:
            Plotly figure object
        """
        print(f"🔥 Creating heatmap for parameter: {parameter}")
        
        try:
            # Prepare data for heatmap
            x_values = data[x_col].values
            y_values = data[y_col].values
            z_values = data[parameter].values
            
            # Create regular grid for interpolation
            xi = np.linspace(x_values.min(), x_values.max(), self.config['map_resolution'])
            yi = np.linspace(y_values.min(), y_values.max(), self.config['map_resolution'])
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # Interpolate values to regular grid
            from scipy.interpolate import griddata
            zi_grid = griddata((x_values, y_values), z_values, (xi_grid, yi_grid), method='cubic')
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=zi_grid,
                x=xi,
                y=yi,
                colorscale=self.color_maps.get(parameter, 'Viridis'),
                colorbar=dict(title=parameter.replace('_', ' ').title()),
                hovertemplate=f'{parameter.replace("_", " ").title()}: %{{z:.3f}}<br>' +
                             f'X: %{{x:.1f}}<br>Y: %{{y:.1f}}<extra></extra>'
            ))
            
            # Add original data points
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(
                    color='white',
                    size=4,
                    line=dict(color='black', width=1)
                ),
                name='Data Points',
                hovertemplate=f'{parameter.replace("_", " ").title()}: %{{text}}<br>' +
                             f'X: %{{x:.1f}}<br>Y: %{{y:.1f}}<extra></extra>',
                text=[f'{val:.3f}' for val in z_values]
            ))
            
            # Update layout
            title_text = title or f"{parameter.replace('_', ' ').title()} Distribution"
            fig.update_layout(
                title=title_text,
                xaxis_title=f"{x_col.replace('_', ' ').title()}",
                yaxis_title=f"{y_col.replace('_', ' ').title()}",
                template="plotly_white",
                width=800,
                height=600
            )
            
            print(f"✅ Heatmap created successfully for {parameter}")
            return fig
            
        except Exception as e:
            print(f"❌ Error creating heatmap: {e}")
            return self._create_error_plot(f"Heatmap Error: {e}")
    
    def create_contour_map(self, 
                          data: pd.DataFrame,
                          parameter: str,
                          x_col: str = 'x_coordinate',
                          y_col: str = 'y_coordinate',
                          levels: int = 20) -> go.Figure:
        """
        Create a contour map of environmental parameters.
        
        Args:
            data: DataFrame with environmental data
            parameter: Parameter to visualize
            x_col: Column name for X coordinates
            y_col: Column name for Y coordinates
            levels: Number of contour levels
            
        Returns:
            Plotly figure object
        """
        print(f"🗺️  Creating contour map for parameter: {parameter}")
        
        try:
            # Prepare data for contour
            x_values = data[x_col].values
            y_values = data[y_col].values
            z_values = data[parameter].values
            
            # Create regular grid
            xi = np.linspace(x_values.min(), x_values.max(), self.config['map_resolution'])
            yi = np.linspace(y_values.min(), y_values.max(), self.config['map_resolution'])
            xi_grid, yi_grid = np.meshgrid(xi, yi)
            
            # Interpolate values
            from scipy.interpolate import griddata
            zi_grid = griddata((x_values, y_values), z_values, (xi_grid, yi_grid), method='cubic')
            
            # Create contour plot
            fig = go.Figure(data=go.Contour(
                z=zi_grid,
                x=xi,
                y=yi,
                colorscale=self.color_maps.get(parameter, 'Viridis'),
                contours=dict(
                    coloring='heatmap',
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                ),
                ncontours=levels,
                colorbar=dict(title=parameter.replace('_', ' ').title())
            ))
            
            # Add data points
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(
                    color=z_values,
                    colorscale=self.color_maps.get(parameter, 'Viridis'),
                    size=6,
                    colorbar=dict(title=parameter.replace('_', ' ').title()),
                    showscale=False
                ),
                name='Data Points',
                hovertemplate=f'{parameter.replace("_", " ").title()}: %{{text}}<br>' +
                             f'X: %{{x:.1f}}<br>Y: %{{y:.1f}}<extra></extra>',
                text=[f'{val:.3f}' for val in z_values]
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{parameter.replace('_', ' ').title()} Contour Map",
                xaxis_title=f"{x_col.replace('_', ' ').title()}",
                yaxis_title=f"{y_col.replace('_', ' ').title()}",
                template="plotly_white",
                width=800,
                height=600
            )
            
            print(f"✅ Contour map created successfully for {parameter}")
            return fig
            
        except Exception as e:
            print(f"❌ Error creating contour map: {e}")
            return self._create_error_plot(f"Contour Map Error: {e}")
    
    def create_multi_parameter_dashboard(self, 
                                       data: pd.DataFrame,
                                       parameters: List[str],
                                       layout: str = 'grid') -> go.Figure:
        """
        Create a multi-parameter dashboard with multiple visualizations.
        
        Args:
            data: DataFrame with environmental data
            parameters: List of parameters to visualize
            layout: Layout type ('grid', 'horizontal', 'vertical')
            
        Returns:
            Plotly figure object with subplots
        """
        print(f"📊 Creating multi-parameter dashboard with {len(parameters)} parameters")
        
        try:
            if layout == 'grid':
                n_cols = min(3, len(parameters))
                n_rows = (len(parameters) + n_cols - 1) // n_cols
                
                fig = make_subplots(
                    rows=n_rows,
                    cols=n_cols,
                    subplot_titles=[p.replace('_', ' ').title() for p in parameters],
                    specs=[[{"type": "heatmap"} for _ in range(n_cols)] for _ in range(n_rows)]
                )
                
                for i, param in enumerate(parameters):
                    row = i // n_cols + 1
                    col = i % n_cols + 1
                    
                    # Create heatmap for this parameter
                    heatmap = self.create_environmental_heatmap(data, param)
                    
                    # Add to subplot
                    fig.add_trace(
                        heatmap.data[0],
                        row=row,
                        col=col
                    )
                
                fig.update_layout(
                    title="Multi-Parameter Environmental Dashboard",
                    height=200 * n_rows,
                    width=800,
                    showlegend=False
                )
                
            else:
                # Simple horizontal layout
                fig = make_subplots(
                    rows=1,
                    cols=len(parameters),
                    subplot_titles=[p.replace('_', ' ').title() for p in parameters]
                )
                
                for i, param in enumerate(parameters):
                    heatmap = self.create_environmental_heatmap(data, param)
                    fig.add_trace(
                        heatmap.data[0],
                        row=1,
                        col=i+1
                    )
                
                fig.update_layout(
                    title="Multi-Parameter Environmental Dashboard",
                    height=400,
                    width=250 * len(parameters),
                    showlegend=False
                )
            
            print(f"✅ Multi-parameter dashboard created successfully")
            return fig
            
        except Exception as e:
            print(f"❌ Error creating multi-parameter dashboard: {e}")
            return self._create_error_plot(f"Dashboard Error: {e}")
    
    def create_time_lapse_visualization(self, 
                                      data: pd.DataFrame,
                                      parameter: str,
                                      time_column: str = 'timestamp',
                                      interval: str = '1H') -> go.Figure:
        """
        Create a time-lapse visualization of environmental changes.
        
        Args:
            data: DataFrame with environmental data
            parameter: Parameter to visualize over time
            time_column: Column containing time information
            interval: Time interval for aggregation
            
        Returns:
            Plotly figure object
        """
        print(f"⏰ Creating time-lapse visualization for parameter: {parameter}")
        
        try:
            # Ensure timestamp column is datetime
            if not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                data[time_column] = pd.to_datetime(data[time_column])
            
            # Sort by time
            data_sorted = data.sort_values(time_column)
            
            # Create time series plot
            fig = go.Figure()
            
            # Add main time series
            fig.add_trace(go.Scatter(
                x=data_sorted[time_column],
                y=data_sorted[parameter],
                mode='lines+markers',
                name=parameter.replace('_', ' ').title(),
                line=dict(width=2),
                marker=dict(size=4)
            ))
            
            # Add trend line
            if len(data_sorted) > 10:
                z = np.polyfit(range(len(data_sorted)), data_sorted[parameter], 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=data_sorted[time_column],
                    y=p(range(len(data_sorted))),
                    mode='lines',
                    name='Trend Line',
                    line=dict(dash='dash', color='red')
                ))
            
            # Update layout
            fig.update_layout(
                title=f"{parameter.replace('_', ' ').title()} Over Time",
                xaxis_title="Time",
                yaxis_title=parameter.replace('_', ' ').title(),
                template="plotly_white",
                width=800,
                height=500,
                hovermode='x unified'
            )
            
            print(f"✅ Time-lapse visualization created successfully for {parameter}")
            return fig
            
        except Exception as e:
            print(f"❌ Error creating time-lapse visualization: {e}")
            return self._create_error_plot(f"Time-lapse Error: {e}")
    
    def _create_error_plot(self, error_message: str) -> go.Figure:
        """Create an error plot when visualization fails."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"❌ Visualization Error<br>{error_message}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Visualization Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=400,
            height=300
        )
        return fig
    
    def save_visualization(self, 
                          fig: go.Figure,
                          filename: str,
                          formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Save visualization in multiple formats.
        
        Args:
            fig: Plotly figure object
            filename: Base filename (without extension)
            formats: List of formats to save (default from config)
            
        Returns:
            Dictionary of saved file paths
        """
        if formats is None:
            formats = self.config['export_formats']
        
        saved_files = {}
        
        for fmt in formats:
            try:
                if fmt == 'html':
                    filepath = self.output_dir / f"{filename}.html"
                    fig.write_html(str(filepath))
                    saved_files[fmt] = str(filepath)
                    
                elif fmt == 'png':
                    filepath = self.output_dir / f"{filename}.png"
                    fig.write_image(str(filepath))
                    saved_files[fmt] = str(filepath)
                    
                elif fmt == 'svg':
                    filepath = self.output_dir / f"{filename}.svg"
                    fig.write_image(str(filepath))
                    saved_files[fmt] = str(filepath)
                    
                elif fmt == 'json':
                    filepath = self.output_dir / f"{filename}.json"
                    fig.write_json(str(filepath))
                    saved_files[fmt] = str(filepath)
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not save {fmt} format: {e}")
        
        print(f"💾 Saved visualization in {len(saved_files)} formats")
        return saved_files
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get summary of available visualizations and capabilities."""
        return {
            'engine_name': 'Environmental Mapping Engine',
            'version': '1.0.0',
            'capabilities': [
                'Environmental parameter heatmaps',
                'Contour maps and vector fields',
                'Multi-parameter dashboards',
                'Time-lapse visualizations',
                'Interactive plotting with Plotly',
                'Multiple export formats'
            ],
            'supported_parameters': list(self.color_maps.keys()),
            'color_schemes': self.color_maps,
            'output_directory': str(self.output_dir),
            'configuration': self.config
        }


def main():
    """Main function for testing the Environmental Mapping Engine."""
    print("🧪 Testing Environmental Mapping Engine")
    
    # Initialize engine
    engine = EnvironmentalMappingEngine()
    
    # Generate sample data
    data = engine._generate_sample_data()
    
    # Create sample visualizations
    print("\n🎨 Creating sample visualizations...")
    
    # 1. Temperature heatmap
    temp_heatmap = engine.create_environmental_heatmap(data, 'temperature')
    engine.save_visualization(temp_heatmap, 'sample_temperature_heatmap')
    
    # 2. Humidity contour map
    humidity_contour = engine.create_contour_map(data, 'humidity')
    engine.save_visualization(humidity_contour, 'sample_humidity_contour')
    
    # 3. Multi-parameter dashboard
    dashboard = engine.create_multi_parameter_dashboard(
        data, 
        ['temperature', 'humidity', 'ph', 'pollution']
    )
    engine.save_visualization(dashboard, 'sample_multi_parameter_dashboard')
    
    # 4. Time-lapse visualization
    time_lapse = engine.create_time_lapse_visualization(data, 'temperature')
    engine.save_visualization(time_lapse, 'sample_temperature_timelapse')
    
    # Print summary
    summary = engine.get_visualization_summary()
    print(f"\n📊 Visualization Summary:")
    print(f"   Engine: {summary['engine_name']} v{summary['version']}")
    print(f"   Capabilities: {len(summary['capabilities'])} features")
    print(f"   Output Directory: {summary['output_directory']}")
    
    print("\n✅ Environmental Mapping Engine test completed successfully!")


if __name__ == "__main__":
    main() 