import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import webbrowser
import os

def create_substrate_3d_model(params):
    """Create a 3D model of the substrate with mycelium"""
    # Create a figure
    fig = go.Figure()
    
    # Add substrate (cube)
    # Define vertices of a cube
    vertices = np.array([
        [-2.5, -2.5, 0], [2.5, -2.5, 0], [2.5, 2.5, 0], [-2.5, 2.5, 0],
        [-2.5, -2.5, 2], [2.5, -2.5, 2], [2.5, 2.5, 2], [-2.5, 2.5, 2]
    ])
    
    # Define faces using indices
    I = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]
    J = [1, 2, 3, 7, 5, 6, 7, 5, 6, 7, 4, 0, 1, 2, 3]
    K = [2, 3, 0, 4, 6, 7, 4, 1, 2, 3, 0, 1, 2, 3, 0]
    
    # Calculate substrate color based on parameters
    r = 0.4 + params.lignin_content * 0.2
    g = 0.3 + params.cellulose_content * 0.2
    b = 0.1 + params.moisture_content * 0.2
    
    # Add substrate as a mesh3d
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=I, j=J, k=K,
        color=f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})',
        opacity=0.8,
        name='Substrate'
    ))
    
    # Add mycelium (spheres)
    num_points = int(50 * params.mycelium_density)
    x = np.random.uniform(-2.3, 2.3, num_points)
    y = np.random.uniform(-2.3, 2.3, num_points)
    z = np.random.uniform(0.1, 1.9, num_points)
    
    # Add mycelium as scatter3d
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color='white',
            opacity=0.8
        ),
        name='Mycelium'
    ))
    
    # Add electrodes (cylinders)
    # Create cylinder points
    theta = np.linspace(0, 2*np.pi, 20)
    z_cyl = np.linspace(0, 2, 20)
    theta_grid, z_grid = np.meshgrid(theta, z_cyl)
    x_cyl = 0.5 * np.cos(theta_grid)
    y_cyl = 0.5 * np.sin(theta_grid)
    
    # Left electrode
    fig.add_trace(go.Surface(
        x=-6 + x_cyl,
        y=y_cyl,
        z=z_grid,
        colorscale=[[0, 'rgb(200, 200, 200)'], [1, 'rgb(200, 200, 200)']],
        showscale=False,
        name='Left Electrode'
    ))
    
    # Right electrode
    fig.add_trace(go.Surface(
        x=6 + x_cyl,
        y=y_cyl,
        z=z_grid,
        colorscale=[[0, 'rgb(200, 200, 200)'], [1, 'rgb(200, 200, 200)']],
        showscale=False,
        name='Right Electrode'
    ))
    
    # Add electric field lines (if conductivity is high enough)
    conductivity = params.base_conductivity + (params.moisture_content * params.moisture_conductivity_factor * params.base_conductivity) + (params.mycelium_density * params.mycelium_conductivity_factor * params.base_conductivity)
    
    if conductivity > 0.05:
        # Number of field lines based on conductivity
        num_lines = int(conductivity * 100)
        num_lines = min(max(5, num_lines), 30)  # Limit between 5 and 30 lines
        
        for _ in range(num_lines):
            # Random starting point on left electrode
            theta_start = np.random.uniform(0, 2*np.pi)
            z_start = np.random.uniform(0.2, 1.8)
            x_start = -6 + 0.5 * np.cos(theta_start)
            y_start = 0.5 * np.sin(theta_start)
            
            # Random ending point on right electrode
            theta_end = np.random.uniform(0, 2*np.pi)
            z_end = np.random.uniform(0.2, 1.8)
            x_end = 6 + 0.5 * np.cos(theta_end)
            y_end = 0.5 * np.sin(theta_end)
            
            # Create a curved path with some randomness
            t = np.linspace(0, 1, 20)
            # Add some random variation to the path
            rand_y = np.random.uniform(-1, 1)
            rand_z = np.random.uniform(-0.5, 0.5)
            
            # Parametric curve
            x_curve = x_start + (x_end - x_start) * t
            y_curve = y_start + (y_end - y_start) * t + rand_y * np.sin(np.pi * t)
            z_curve = z_start + (z_end - z_start) * t + rand_z * np.sin(np.pi * t)
            
            # Add the field line
            fig.add_trace(go.Scatter3d(
                x=x_curve,
                y=y_curve,
                z=z_curve,
                mode='lines',
                line=dict(
                    color='cyan',
                    width=2
                ),
                opacity=0.7,
                name='Electric Field'
            ))
    
    # Update layout
    fig.update_layout(
        title="3D Substrate Model",
        scene=dict(
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2)
            )
        ),
        showlegend=True
    )
    
    return fig

def create_cross_section_view(params):
    """Create a cross-section view of the substrate"""
    # Create a figure
    fig = go.Figure()
    
    # Create a grid for the cross-section
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create a heatmap for moisture distribution
    Z_moisture = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            # Distance from center
            dist = np.sqrt(X[i, j]**2 + Y[i, j]**2)
            # More moisture in center, less at edges
            if dist < 2.5:
                Z_moisture[i, j] = params.moisture_content * (1 - 0.3 * dist/2.5)
            else:
                Z_moisture[i, j] = 0
    
    # Add moisture heatmap
    fig.add_trace(go.Heatmap(
        z=Z_moisture,
        x=x,
        y=y,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Moisture"),
        name='Moisture Distribution'
    ))
    
    # Add mycelium points (top view)
    num_points = int(100 * params.mycelium_density)
    x_myc = np.random.uniform(-2.3, 2.3, num_points)
    y_myc = np.random.uniform(-2.3, 2.3, num_points)
    
    fig.add_trace(go.Scatter(
        x=x_myc,
        y=y_myc,
        mode='markers',
        marker=dict(
            size=5,
            color='white',
            line=dict(width=1, color='gray')
        ),
        name='Mycelium'
    ))
    
    # Update layout
    fig.update_layout(
        title="Substrate Cross-Section View",
        xaxis=dict(title="X (cm)"),
        yaxis=dict(title="Y (cm)"),
        showlegend=True
    )
    
    return fig

def create_combined_visualization(params):
    """Create a combined visualization with 3D model and cross-section"""
    # Create a figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'xy'}]],
        subplot_titles=("3D Substrate Model", "Cross-Section View")
    )
    
    # Add substrate (cube)
    # Define vertices of a cube
    vertices = np.array([
        [-2.5, -2.5, 0], [2.5, -2.5, 0], [2.5, 2.5, 0], [-2.5, 2.5, 0],
        [-2.5, -2.5, 2], [2.5, -2.5, 2], [2.5, 2.5, 2], [-2.5, 2.5, 2]
    ])
    
    # Define faces using indices
    I = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]
    J = [1, 2, 3, 7, 5, 6, 7, 5, 6, 7, 4, 0, 1, 2, 3]
    K = [2, 3, 0, 4, 6, 7, 4, 1, 2, 3, 0, 1, 2, 3, 0]
    
    r = 0.4 + params.lignin_content * 0.2
    g = 0.3 + params.cellulose_content * 0.2
    b = 0.1 + params.moisture_content * 0.2
    
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=I, j=J, k=K,
        color=f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})',
        opacity=0.8,
        name='Substrate'
    ), row=1, col=1)
    
    # Add mycelium (spheres)
    num_points = int(50 * params.mycelium_density)
    x = np.random.uniform(-2.3, 2.3, num_points)
    y = np.random.uniform(-2.3, 2.3, num_points)
    z = np.random.uniform(0.1, 1.9, num_points)
    
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color='white',
            opacity=0.8
        ),
        name='Mycelium'
    ), row=1, col=1)
    
    # Add cross-section view
    # Create a grid for the cross-section
    x_grid = np.linspace(-2.5, 2.5, 100)
    y_grid = np.linspace(-2.5, 2.5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Create a heatmap for moisture distribution
    Z_moisture = np.zeros_like(X)
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            # Distance from center
            dist = np.sqrt(X[i, j]**2 + Y[i, j]**2)
            # More moisture in center, less at edges
            if dist < 2.5:
                Z_moisture[i, j] = params.moisture_content * (1 - 0.3 * dist/2.5)
            else:
                Z_moisture[i, j] = 0
    
    fig.add_trace(go.Heatmap(
        z=Z_moisture,
        x=x_grid,
        y=y_grid,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Moisture"),
        name='Moisture Distribution'
    ), row=1, col=2)
    
    # Add mycelium points (top view)
    x_myc = np.random.uniform(-2.3, 2.3, num_points)
    y_myc = np.random.uniform(-2.3, 2.3, num_points)
    
    fig.add_trace(go.Scatter(
        x=x_myc,
        y=y_myc,
        mode='markers',
        marker=dict(
            size=5,
            color='white',
            line=dict(width=1, color='gray')
        ),
        name='Mycelium'
    ), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="Mushroom Substrate Visualization",
        height=600,
        width=1200,
        scene=dict(
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.2)
            )
        ),
        xaxis=dict(range=[-2.5, 2.5]),
        yaxis=dict(range=[-2.5, 2.5])
    )
    
    return fig

def save_and_open_visualization(fig, filename="substrate_visualization.html"):
    """Save the visualization to an HTML file and open it in a browser"""
    # Save to the current working directory
    file_path = os.path.abspath(filename)
    
    # Save the figure
    fig.write_html(file_path)
    
    print(f"Visualization saved to: {file_path}")
    
    # Open in default browser
    webbrowser.open(f'file://{file_path}')
    
    return file_path 