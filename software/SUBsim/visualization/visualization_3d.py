import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from simulation.substrate import calculate_conductivity, calculate_ph, calculate_contaminant_resistance, calculate_electrical_properties, calculate_growth_potential
import os
import tempfile
import webbrowser
from itertools import combinations, product
import subprocess
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_substrate_3d_model(params):
    """Create a 3D visualization of the substrate with mycelium growth"""
    # Create a figure with a 3D projection
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a denser grid of points for better visualization
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    z = np.linspace(0, 1, 30)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate properties
    conductivity = calculate_conductivity(params)
    ph = calculate_ph(params)
    resistance = calculate_contaminant_resistance(params)
    electrical_props = calculate_electrical_properties(params)
    
    # Create a density field representing mycelium growth
    # Higher density near the center, affected by moisture and lignin content
    density = params.mycelium_density * (1 - 0.8 * ((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2))
    
    # Add some randomness for more natural appearance
    np.random.seed(42)  # For reproducibility
    density += np.random.normal(0, 0.05, density.shape) * params.mycelium_density
    
    # Adjust density based on substrate properties
    density *= (0.5 + 0.5 * params.moisture_content)  # More moisture = better growth
    density *= (1 - 0.3 * params.lignin_content)      # Less lignin = easier growth
    density *= (0.7 + 0.3 * resistance)               # Higher resistance = better growth
    
    # Create a mask for points with significant mycelium presence
    mask = density > 0.2
    
    # Create color map based on electrical conductivity
    # Blue (low) to red (high)
    conductivity_normalized = np.clip(density * electrical_props['conductivity'] * 100, 0, 1)
    colors = plt.cm.coolwarm(conductivity_normalized.flatten())
    
    # Plot the points with mycelium
    scatter = ax.scatter(X[mask], Y[mask], Z[mask], c=colors[mask.flatten()], s=30, alpha=0.6, edgecolors='none')
    
    # Add substrate box
    r = [0, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s - e)) == 1:
            ax.plot3D(*zip(s, e), color="gray", alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'3D Substrate Model\nMycelium Density: {params.mycelium_density:.2f}, Conductivity: {conductivity:.6f} S/m', fontsize=14)
    
    # Add a colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.coolwarm), ax=ax, pad=0.1)
    cbar.set_label('Electrical Conductivity (relative)', fontsize=12)
    
    # Add text with key parameters
    plt.figtext(0.02, 0.02, 
                f"Substrate: {params.substrate_type if hasattr(params, 'substrate_type') else 'Mixed'}\n"
                f"Moisture: {params.moisture_content:.2f}\n"
                f"pH: {ph:.1f}\n"
                f"Resistivity: {electrical_props['resistivity']:.2f} Ω·m\n"
                f"Dielectric: {electrical_props['dielectric']:.1f}",
                bbox=dict(facecolor='white', alpha=0.7), fontsize=12)
    
    # Set the initial view angle for better visualization
    ax.view_init(elev=30, azim=45)
    
    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])
    
    # Add grid lines for better spatial reference
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Make the figure tight and add some padding
    plt.tight_layout(pad=3.0)
    
    # Make the figure interactive
    plt.rcParams['figure.figsize'] = [12, 10]
    plt.rcParams['figure.dpi'] = 100
    
    return fig

def create_cross_section_view(params):
    """Create a cross-section visualization of the substrate with electrical properties"""
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Create a grid of points for a 2D slice
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate properties
    conductivity = calculate_conductivity(params)
    ph = calculate_ph(params)
    resistance = calculate_contaminant_resistance(params)
    electrical_props = calculate_electrical_properties(params)
    
    # Create density fields for different cross-sections
    # Center slice
    center_density = params.mycelium_density * (1 - 0.8 * ((X - 0.5)**2 + (Y - 0.5)**2))
    center_density *= (0.5 + 0.5 * params.moisture_content)
    center_density *= (1 - 0.3 * params.lignin_content)
    center_density *= (0.7 + 0.3 * resistance)
    
    # Electrical conductivity field
    conductivity_field = center_density * electrical_props['conductivity'] * 100
    
    # Dielectric constant field
    dielectric_field = np.ones_like(center_density) * electrical_props['dielectric']
    dielectric_field *= (0.8 + 0.4 * center_density)  # Mycelium affects dielectric properties
    
    # Signal propagation field (inverse of dielectric)
    propagation_field = 1.0 / np.sqrt(dielectric_field)
    
    # Plot mycelium density
    im1 = axs[0, 0].imshow(center_density, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
    axs[0, 0].set_title('Mycelium Density')
    axs[0, 0].set_xlabel('X')
    axs[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axs[0, 0])
    
    # Plot electrical conductivity
    im2 = axs[0, 1].imshow(conductivity_field, cmap='coolwarm', origin='lower', extent=[0, 1, 0, 1])
    axs[0, 1].set_title('Electrical Conductivity')
    axs[0, 1].set_xlabel('X')
    axs[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axs[0, 1])
    
    # Plot dielectric constant
    im3 = axs[1, 0].imshow(dielectric_field, cmap='Greens', origin='lower', extent=[0, 1, 0, 1])
    axs[1, 0].set_title('Dielectric Constant')
    axs[1, 0].set_xlabel('X')
    axs[1, 0].set_ylabel('Y')
    plt.colorbar(im3, ax=axs[1, 0])
    
    # Plot signal propagation speed
    im4 = axs[1, 1].imshow(propagation_field, cmap='plasma', origin='lower', extent=[0, 1, 0, 1])
    axs[1, 1].set_title('Signal Propagation Speed (relative to c)')
    axs[1, 1].set_xlabel('X')
    axs[1, 1].set_ylabel('Y')
    plt.colorbar(im4, ax=axs[1, 1])
    
    # Add overall title
    plt.suptitle(f'Substrate Cross-Section with Electrical Properties\n{params.substrate_type if hasattr(params, "substrate_type") else "Mixed"} Substrate, pH {ph:.1f}', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def create_combined_visualization(params):
    """Create a combined visualization with multiple views and electrical properties"""
    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 3D model in top-left
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Create a grid of points for 3D
    x = np.linspace(0, 1, 15)
    y = np.linspace(0, 1, 15)
    z = np.linspace(0, 1, 15)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Calculate properties
    conductivity = calculate_conductivity(params)
    ph = calculate_ph(params)
    resistance = calculate_contaminant_resistance(params)
    electrical_props = calculate_electrical_properties(params)
    
    # Create a density field for 3D
    density_3d = params.mycelium_density * (1 - 0.8 * ((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2))
    density_3d *= (0.5 + 0.5 * params.moisture_content)
    density_3d *= (1 - 0.3 * params.lignin_content)
    density_3d *= (0.7 + 0.3 * resistance)
    
    # Create a mask for points with significant mycelium presence
    mask_3d = density_3d > 0.2
    
    # Create color map based on electrical conductivity
    conductivity_normalized = np.clip(density_3d * electrical_props['conductivity'] * 100, 0, 1)
    colors = plt.cm.coolwarm(conductivity_normalized.flatten())
    
    # Plot the points with mycelium in 3D
    ax1.scatter(X[mask_3d], Y[mask_3d], Z[mask_3d], c=colors[mask_3d.flatten()], s=30, alpha=0.6)
    ax1.set_title('3D Substrate Model')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Create a grid for 2D plots
    x_2d = np.linspace(0, 1, 100)
    y_2d = np.linspace(0, 1, 100)
    X_2d, Y_2d = np.meshgrid(x_2d, y_2d)
    
    # Create density field for 2D
    density_2d = params.mycelium_density * (1 - 0.8 * ((X_2d - 0.5)**2 + (Y_2d - 0.5)**2))
    density_2d *= (0.5 + 0.5 * params.moisture_content)
    density_2d *= (1 - 0.3 * params.lignin_content)
    density_2d *= (0.7 + 0.3 * resistance)
    
    # Electrical conductivity field
    conductivity_field = density_2d * electrical_props['conductivity'] * 100
    
    # Dielectric constant field
    dielectric_field = np.ones_like(density_2d) * electrical_props['dielectric']
    dielectric_field *= (0.8 + 0.4 * density_2d)
    
    # Signal propagation field
    propagation_field = 1.0 / np.sqrt(dielectric_field)
    
    # Plot mycelium density in top-right
    ax2 = fig.add_subplot(222)
    im2 = ax2.imshow(density_2d, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
    ax2.set_title('Mycelium Density')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2)
    
    # Plot electrical conductivity in bottom-left
    ax3 = fig.add_subplot(223)
    im3 = ax3.imshow(conductivity_field, cmap='coolwarm', origin='lower', extent=[0, 1, 0, 1])
    ax3.set_title('Electrical Conductivity')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    plt.colorbar(im3, ax=ax3)
    
    # Plot signal propagation in bottom-right
    ax4 = fig.add_subplot(224)
    im4 = ax4.imshow(propagation_field, cmap='plasma', origin='lower', extent=[0, 1, 0, 1])
    ax4.set_title('Signal Propagation Speed')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    plt.colorbar(im4, ax=ax4)
    
    # Add overall title
    plt.suptitle(f'Substrate Visualization for Unconventional Computing\n{params.substrate_type if hasattr(params, "substrate_type") else "Mixed"} Substrate, pH {ph:.1f}, Conductivity {conductivity:.6f} S/m', fontsize=16)
    
    # Add text with key electrical parameters
    plt.figtext(0.02, 0.02, 
                f"Electrical Properties:\n"
                f"Conductivity: {electrical_props['conductivity']:.6f} S/m\n"
                f"Resistivity: {electrical_props['resistivity']:.2f} Ω·m\n"
                f"Dielectric Constant: {electrical_props['dielectric']:.1f}\n"
                f"Signal Propagation: {electrical_props['propagation_speed']:.3f}c\n"
                f"Signal Delay: {(1/electrical_props['propagation_speed'])*3.33:.2f} ns/cm",
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def save_and_open_visualization(fig, params, filename="substrate_visualization.png"):
    """Save the visualization to a PNG file in the project directory"""
    try:
        # Create a visualizations directory in the project folder if it doesn't exist
        vis_dir = os.path.join(os.getcwd(), "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Change the extension to PNG if it's not already
        if not filename.lower().endswith('.png'):
            filename = os.path.splitext(filename)[0] + '.png'
        
        # Full path to the file
        filepath = os.path.join(vis_dir, filename)
        print(f"Saving visualization to: {filepath}")
        
        # Save the figure directly as PNG
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved PNG to: {filepath}")
        
        # Verify PNG file exists
        if not os.path.exists(filepath):
            print(f"ERROR: PNG file was not created at {filepath}")
            return None
        else:
            print(f"PNG file created successfully at {filepath}")
        
        return filepath
    
    except Exception as e:
        print(f"Error in save_and_open_visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_interactive_3d_model(params):
    """Create an interactive 3D visualization of the substrate with detailed mycelium structure using Plotly"""
    # Calculate properties
    conductivity = calculate_conductivity(params)
    ph = calculate_ph(params)
    resistance = calculate_contaminant_resistance(params)
    electrical_props = calculate_electrical_properties(params)
    growth_potential = calculate_growth_potential(params)  # Add this line to calculate growth potential
    
    # Create a figure
    fig = go.Figure()
    
    # Set dimensions in cm
    width, length, height = 10, 10, 10
    
    # Create a more detailed mycelium network
    # 1. Generate primary hyphae (main branches)
    np.random.seed(42)  # For reproducibility
    
    # Number of primary hyphae based on mycelium density
    num_primary = int(30 * params.mycelium_density)
    
    # Track all points for later coloring
    all_x, all_y, all_z = [], [], []
    all_types = []  # 0: primary, 1: secondary, 2: tertiary, 3: hyphal tips
    all_diameters = []  # Hyphal diameters in micrometers
    all_ages = []  # Relative age (0-1)
    
    # Generate primary hyphae starting points (mostly from bottom of substrate)
    primary_starts = []
    for _ in range(num_primary):
        if np.random.random() < 0.7:  # 70% start from bottom
            x = np.random.uniform(1, width-1)
            y = np.random.uniform(1, length-1)
            z = 0.1  # Just above bottom
        else:  # 30% start from sides
            side = np.random.choice(['x', 'y'])
            if side == 'x':
                x = np.random.choice([0.1, width-0.1])
                y = np.random.uniform(1, length-1)
            else:
                x = np.random.uniform(1, width-1)
                y = np.random.choice([0.1, length-0.1])
            z = np.random.uniform(0.5, height-1)
        primary_starts.append((x, y, z))
    
    # Growth parameters affected by substrate properties
    growth_rate = 0.5 + 0.5 * params.moisture_content
    growth_rate *= (1 - 0.3 * params.lignin_content)
    growth_rate *= (0.7 + 0.3 * resistance)
    
    # Branching probability affected by nitrogen content
    branching_prob = 0.3 + 0.7 * params.nitrogen_content
    
    # Direction bias based on substrate type
    vertical_bias = 0.6 if params.substrate_type == "STRAW" else 0.3
    
    # Generate primary hyphae paths with growth towards center and up
    for i, (x, y, z) in enumerate(primary_starts):
        # Length of this primary hypha
        primary_length = int(15 + 20 * growth_rate * np.random.random())
        
        # Starting direction - tend to grow upward and toward center
        dx = 0.1 * (width/2 - x) + 0.05 * np.random.randn()
        dy = 0.1 * (length/2 - y) + 0.05 * np.random.randn()
        dz = vertical_bias + 0.1 * np.random.randn()
        
        # Normalize direction vector
        mag = np.sqrt(dx**2 + dy**2 + dz**2)
        dx, dy, dz = dx/mag, dy/mag, dz/mag
        
        # Generate points along primary hypha
        prev_x, prev_y, prev_z = x, y, z
        primary_points = [(x, y, z)]
        
        for j in range(primary_length):
            # Add some randomness to direction
            dx += 0.1 * np.random.randn()
            dy += 0.1 * np.random.randn()
            dz += 0.1 * np.random.randn()
            
            # Normalize again
            mag = np.sqrt(dx**2 + dy**2 + dz**2)
            dx, dy, dz = dx/mag, dy/mag, dz/mag
            
            # Calculate new point
            new_x = prev_x + 0.3 * dx
            new_y = prev_y + 0.3 * dy
            new_z = prev_z + 0.3 * dz
            
            # Keep within bounds
            new_x = max(0, min(width, new_x))
            new_y = max(0, min(length, new_y))
            new_z = max(0, min(height, new_z))
            
            # Add point
            primary_points.append((new_x, new_y, new_z))
            prev_x, prev_y, prev_z = new_x, new_y, new_z
            
            # Add to overall tracking
            all_x.extend([prev_x, new_x])
            all_y.extend([prev_y, new_y])
            all_z.extend([prev_z, new_z])
            all_types.extend([0, 0])  # Primary type
            
            # Primary hyphae are thicker (3-10 µm)
            diameter = 3 + 7 * (1 - j/primary_length)  # Thinner as they extend
            all_diameters.extend([diameter, diameter])
            
            # Age decreases along length
            age = 1 - j/primary_length
            all_ages.extend([age, age])
            
            # Generate secondary branches with some probability
            if np.random.random() < branching_prob and j > 2:
                # Create a secondary branch
                sec_x, sec_y, sec_z = new_x, new_y, new_z
                
                # Direction for secondary - tend to grow perpendicular to primary
                cross_x = -dy + 0.2 * np.random.randn()
                cross_y = dx + 0.2 * np.random.randn()
                cross_z = 0.3 * np.random.randn()
                
                # Normalize
                mag = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
                cross_x, cross_y, cross_z = cross_x/mag, cross_y/mag, cross_z/mag
                
                # Length of secondary branch
                sec_length = int(5 + 10 * growth_rate * np.random.random())
                
                # Generate secondary branch
                sec_prev_x, sec_prev_y, sec_prev_z = sec_x, sec_y, sec_z
                
                for k in range(sec_length):
                    # Add randomness to direction
                    cross_x += 0.15 * np.random.randn()
                    cross_y += 0.15 * np.random.randn()
                    cross_z += 0.15 * np.random.randn()
                    
                    # Normalize
                    mag = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
                    cross_x, cross_y, cross_z = cross_x/mag, cross_y/mag, cross_z/mag
                    
                    # Calculate new point
                    sec_new_x = sec_prev_x + 0.2 * cross_x
                    sec_new_y = sec_prev_y + 0.2 * cross_y
                    sec_new_z = sec_prev_z + 0.2 * cross_z
                    
                    # Keep within bounds
                    sec_new_x = max(0, min(width, sec_new_x))
                    sec_new_y = max(0, min(length, sec_new_y))
                    sec_new_z = max(0, min(height, sec_new_z))
                    
                    # Add to tracking
                    all_x.extend([sec_prev_x, sec_new_x])
                    all_y.extend([sec_prev_y, sec_new_y])
                    all_z.extend([sec_prev_z, sec_new_z])
                    all_types.extend([1, 1])  # Secondary type
                    
                    # Secondary hyphae are thinner (2-5 µm)
                    sec_diameter = 2 + 3 * (1 - k/sec_length)
                    all_diameters.extend([sec_diameter, sec_diameter])
                    
                    # Age for secondary
                    sec_age = (1 - j/primary_length) * (1 - k/sec_length)
                    all_ages.extend([sec_age, sec_age])
                    
                    sec_prev_x, sec_prev_y, sec_prev_z = sec_new_x, sec_new_y, sec_new_z
                    
                    # Generate tertiary branches with lower probability
                    if k > 2 and np.random.random() < 0.3 * branching_prob:
                        # Create tertiary branch
                        tert_x, tert_y, tert_z = sec_new_x, sec_new_y, sec_new_z
                        
                        # Direction - more random
                        tert_dx = np.random.randn()
                        tert_dy = np.random.randn()
                        tert_dz = np.random.randn()
                        
                        # Normalize
                        mag = np.sqrt(tert_dx**2 + tert_dy**2 + tert_dz**2)
                        tert_dx, tert_dy, tert_dz = tert_dx/mag, tert_dy/mag, tert_dz/mag
                        
                        # Length of tertiary branch
                        tert_length = int(3 + 5 * growth_rate * np.random.random())
                        
                        # Generate tertiary branch
                        tert_prev_x, tert_prev_y, tert_prev_z = tert_x, tert_y, tert_z
                        
                        for m in range(tert_length):
                            # Add randomness
                            tert_dx += 0.2 * np.random.randn()
                            tert_dy += 0.2 * np.random.randn()
                            tert_dz += 0.2 * np.random.randn()
                            
                            # Normalize
                            mag = np.sqrt(tert_dx**2 + tert_dy**2 + tert_dz**2)
                            tert_dx, tert_dy, tert_dz = tert_dx/mag, tert_dy/mag, tert_dz/mag
                            
                            # Calculate new point
                            tert_new_x = tert_prev_x + 0.1 * tert_dx
                            tert_new_y = tert_prev_y + 0.1 * tert_dy
                            tert_new_z = tert_prev_z + 0.1 * tert_dz
                            
                            # Keep within bounds
                            tert_new_x = max(0, min(width, tert_new_x))
                            tert_new_y = max(0, min(length, tert_new_y))
                            tert_new_z = max(0, min(height, tert_new_z))
                            
                            # Add to tracking
                            all_x.extend([tert_prev_x, tert_new_x])
                            all_y.extend([tert_prev_y, tert_new_y])
                            all_z.extend([tert_prev_z, tert_new_z])
                            all_types.extend([2, 2])  # Tertiary type
                            
                            # Tertiary hyphae are thinnest (1-3 µm)
                            tert_diameter = 1 + 2 * (1 - m/tert_length)
                            all_diameters.extend([tert_diameter, tert_diameter])
                            
                            # Age for tertiary
                            tert_age = sec_age * (1 - m/tert_length)
                            all_ages.extend([tert_age, tert_age])
                            
                            tert_prev_x, tert_prev_y, tert_prev_z = tert_new_x, tert_new_y, tert_new_z
                            
                            # Add hyphal tips at the ends
                            if m == tert_length - 1:
                                all_x.append(tert_new_x)
                                all_y.append(tert_new_y)
                                all_z.append(tert_new_z)
                                all_types.append(3)  # Hyphal tip
                                all_diameters.append(1.5)  # Tips are thin
                                all_ages.append(0.1)  # Tips are new
    
    # Convert lists to arrays
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    all_types = np.array(all_types)
    all_diameters = np.array(all_diameters)
    all_ages = np.array(all_ages)
    
    # Calculate electrical conductivity for each point
    # Higher in older, thicker hyphae and affected by substrate properties
    conductivity_values = electrical_props['conductivity'] * 100 * all_ages * (all_diameters / 10)
    
    # Create separate traces for different hyphal types for better visualization
    # Instead of using line width arrays, we'll use markers with size based on diameter
    
    # Primary hyphae (type 0)
    mask_primary = all_types == 0
    if np.sum(mask_primary) > 0:
        fig.add_trace(go.Scatter3d(
            x=all_x[mask_primary],
            y=all_y[mask_primary],
            z=all_z[mask_primary],
            mode='markers',
            marker=dict(
                size=all_diameters[mask_primary] * 0.8,  # Scale for visualization
                color='#FF3333',  # Bright red
                opacity=0.8
            ),
            name='Primary Hyphae',
            hovertemplate=(
                'X: %{x:.1f} cm<br>'
                'Y: %{y:.1f} cm<br>'
                'Z: %{z:.1f} cm<br>'
                'Diameter: %{marker.size:.1f} µm<br>'
                'Age: %{marker.color:.2f}<extra>Primary Hypha</extra>'
            )
        ))
    
    # Secondary hyphae (type 1)
    mask_secondary = all_types == 1
    if np.sum(mask_secondary) > 0:
        fig.add_trace(go.Scatter3d(
            x=all_x[mask_secondary],
            y=all_y[mask_secondary],
            z=all_z[mask_secondary],
            mode='markers',
            marker=dict(
                size=all_diameters[mask_secondary] * 0.8,
                color='#3366FF',  # Bright blue
                opacity=0.8
            ),
            name='Secondary Hyphae',
            hovertemplate=(
                'X: %{x:.1f} cm<br>'
                'Y: %{y:.1f} cm<br>'
                'Z: %{z:.1f} cm<br>'
                'Diameter: %{marker.size:.1f} µm<br>'
                'Age: %{marker.color:.2f}<extra>Secondary Hypha</extra>'
            )
        ))
    
    # Tertiary hyphae (type 2)
    mask_tertiary = all_types == 2
    if np.sum(mask_tertiary) > 0:
        fig.add_trace(go.Scatter3d(
            x=all_x[mask_tertiary],
            y=all_y[mask_tertiary],
            z=all_z[mask_tertiary],
            mode='markers',
            marker=dict(
                size=all_diameters[mask_tertiary] * 0.8,
                color='#33CC33',  # Bright green
                opacity=0.8
            ),
            name='Tertiary Hyphae',
            hovertemplate=(
                'X: %{x:.1f} cm<br>'
                'Y: %{y:.1f} cm<br>'
                'Z: %{z:.1f} cm<br>'
                'Diameter: %{marker.size:.1f} µm<br>'
                'Age: %{marker.color:.2f}<extra>Tertiary Hypha</extra>'
            )
        ))
    
    # Hyphal tips (type 3)
    mask_tips = all_types == 3
    if np.sum(mask_tips) > 0:
        fig.add_trace(go.Scatter3d(
            x=all_x[mask_tips],
            y=all_y[mask_tips],
            z=all_z[mask_tips],
            mode='markers',
            marker=dict(
                size=4,  # Fixed size for tips
                color='#FF55FF',  # Bright purple
                symbol='circle',
                line=dict(color='#CC55CC', width=1)
            ),
            name='Hyphal Tips',
            hovertemplate=(
                'X: %{x:.1f} cm<br>'
                'Y: %{y:.1f} cm<br>'
                'Z: %{z:.1f} cm<br>'
                'Diameter: 1-2 µm<br>'
                'Growing Point<extra>Hyphal Tip</extra>'
            )
        ))
    
    # Add substrate box with semi-transparent walls
    for i, j, k in product([0, width], [0, length], [0, height]):
        for i2, j2, k2 in product([0, width], [0, length], [0, height]):
            if sum(abs(n1-n2) for n1, n2 in zip([i, j, k], [i2, j2, k2])) == width:
                fig.add_trace(go.Scatter3d(
                    x=[i, i2],
                    y=[j, j2],
                    z=[k, k2],
                    mode='lines',
                    line=dict(color='rgba(200,200,200,0.5)', width=2),  # Fixed width as a single number
                    showlegend=False
                ))
    
    # Add measurement axes with ticks
    # X-axis
    fig.add_trace(go.Scatter3d(
        x=[0, 11],
        y=[0, 0],
        z=[0, 0],
        mode='lines+text',
        line=dict(color='white', width=3),  # Fixed width as a single number
        text=['', 'X (cm)'],
        textposition='middle right',
        showlegend=False
    ))
    
    # Y-axis
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 11],
        z=[0, 0],
        mode='lines+text',
        line=dict(color='white', width=3),  # Fixed width as a single number
        text=['', 'Y (cm)'],
        textposition='middle right',
        showlegend=False
    ))
    
    # Z-axis
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, 11],
        mode='lines+text',
        line=dict(color='white', width=3),  # Fixed width as a single number
        text=['', 'Z (cm)'],
        textposition='middle right',
        showlegend=False
    ))
    
    # Add ticks on axes
    for i in range(2, 11, 2):
        # X-axis ticks
        fig.add_trace(go.Scatter3d(
            x=[i, i],
            y=[0, -0.3],
            z=[0, 0],
            mode='lines+text',
            line=dict(color='white', width=2),  # Fixed width as a single number
            text=['', str(i)],
            textposition='bottom center',
            showlegend=False
        ))
        
        # Y-axis ticks
        fig.add_trace(go.Scatter3d(
            x=[0, -0.3],
            y=[i, i],
            z=[0, 0],
            mode='lines+text',
            line=dict(color='white', width=2),  # Fixed width as a single number
            text=['', str(i)],
            textposition='bottom center',
            showlegend=False
        ))
        
        # Z-axis ticks
        fig.add_trace(go.Scatter3d(
            x=[0, -0.3],
            y=[0, 0],
            z=[i, i],
            mode='lines+text',
            line=dict(color='white', width=2),  # Fixed width as a single number
            text=['', str(i)],
            textposition='bottom center',
            showlegend=False
        ))
    
    # Update layout with dark background
    fig.update_layout(
        title=dict(
            text=f'Interactive 3D Substrate Model (10×10×10 cm)<br>Mycelium Density: {params.mycelium_density:.2f}, Conductivity: {conductivity:.6e} S/m',
            x=0.5,
            font=dict(size=18, color='white')
        ),
        scene=dict(
            xaxis_title='X (cm)',
            yaxis_title='Y (cm)',
            zaxis_title='Z (cm)',
            xaxis=dict(
                range=[0, 10], 
                dtick=2,
                gridcolor='rgba(100,100,100,0.2)',
                showbackground=True,
                backgroundcolor='rgba(20, 20, 40, 1)'
            ),
            yaxis=dict(
                range=[0, 10], 
                dtick=2,
                gridcolor='rgba(100,100,100,0.2)',
                showbackground=True,
                backgroundcolor='rgba(20, 20, 40, 1)'
            ),
            zaxis=dict(
                range=[0, 10], 
                dtick=2,
                gridcolor='rgba(100,100,100,0.2)',
                showbackground=True,
                backgroundcolor='rgba(20, 20, 40, 1)'
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='rgb(10, 10, 30)'  # Dark blue background
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=800,
        width=1000,
        paper_bgcolor='rgb(10, 10, 30)',  # Dark blue background
        plot_bgcolor='rgb(10, 10, 30)',   # Dark blue background
        font=dict(color='white')          # White text for better contrast
    )
    
    # Add detailed annotations with key parameters and units
    substrate_info = (
        f"<b>Substrate Properties:</b><br>"
        f"Type: {params.substrate_type if hasattr(params, 'substrate_type') else 'Mixed'}<br>"
        f"Moisture: {params.moisture_content:.2f} (v/v)<br>"
        f"Lignin: {params.lignin_content:.2f} (w/w)<br>"
        f"Cellulose: {params.cellulose_content:.2f} (w/w)<br>"
        f"pH: {ph:.1f}<br><br>"
        f"<b>Electrical Properties:</b><br>"
        f"Conductivity: {conductivity:.2e} S/m<br>"
        f"Resistivity: {electrical_props['resistivity']:.2e} Ω·m<br>"
        f"Dielectric Constant: {electrical_props['dielectric']:.1f}<br>"
        f"Signal Propagation: {electrical_props['propagation_speed']:.2f}c<br><br>"
        f"<b>Growth Metrics:</b><br>"
        f"Mycelium Density: {params.mycelium_density:.2f} (v/v)<br>"
        f"Contaminant Resistance: {resistance:.2f}<br>"
        f"Growth Potential: {growth_potential:.2f}"
    )
    
    fig.add_annotation(
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        text=substrate_info,
        showarrow=False,
        font=dict(size=12, color='white'),
        bgcolor="rgba(30, 30, 60, 0.8)",
        opacity=0.9,
        bordercolor="rgba(100, 100, 200, 0.8)",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    # Add a legend for the color scale
    fig.add_annotation(
        x=0.99,
        y=0.5,
        xref="paper",
        yref="paper",
        text="<b>Mycelium Structure</b><br>Red: Primary hyphae (3-10 µm)<br>Blue: Secondary hyphae (2-5 µm)<br>Green: Tertiary hyphae (1-3 µm)<br>Purple: Hyphal tips (1-2 µm)",
        showarrow=False,
        font=dict(size=12, color='white'),
        bgcolor="rgba(30, 30, 60, 0.8)",
        opacity=0.9,
        bordercolor="rgba(100, 100, 200, 0.8)",
        borderwidth=1,
        borderpad=4,
        align="right"
    )
    
    return fig

def save_and_open_interactive_visualization(fig, params, filename="interactive_substrate_model.html"):
    """Save the interactive visualization to an HTML file and open it in a browser"""
    try:
        # Create a visualizations directory in the project folder if it doesn't exist
        vis_dir = os.path.join(os.getcwd(), "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Change the extension to HTML if it's not already
        if not filename.lower().endswith('.html'):
            filename = os.path.splitext(filename)[0] + '.html'
        
        # Full path to the file
        filepath = os.path.join(vis_dir, filename)
        print(f"Saving interactive visualization to: {filepath}")
        
        # Save the figure as HTML
        fig.write_html(
            filepath,
            include_plotlyjs='cdn',  # Use CDN for plotly.js
            full_html=True,
            include_mathjax='cdn',   # Use CDN for MathJax
            config={
                'displayModeBar': True,
                'editable': True,
                'scrollZoom': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'substrate_model',
                    'height': 800,
                    'width': 1000,
                    'scale': 1
                }
            }
        )
        
        # Verify HTML file exists
        if not os.path.exists(filepath):
            print(f"ERROR: HTML file was not created at {filepath}")
            return None
        else:
            print(f"HTML file created successfully at {filepath}")
        
        # Try to open the file in a browser
        print(f"Attempting to open file in browser: {filepath}")
        try:
            # Fix the file URL format - there was an extra slash causing issues
            file_url = 'file://' + os.path.abspath(filepath).replace('\\', '/')
            print(f"Opening URL: {file_url}")
            
            # Try to open the URL
            webbrowser.open(file_url)
        except Exception as e:
            print(f"Error opening file in browser: {e}")
            print("Please open the file manually at:", filepath)
        
        return filepath
    
    except Exception as e:
        print(f"Error in save_and_open_interactive_visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_interactive_cross_section(params):
    """Create an interactive cross-section visualization using Plotly"""
    # Calculate properties
    conductivity = calculate_conductivity(params)
    ph = calculate_ph(params)
    resistance = calculate_contaminant_resistance(params)
    electrical_props = calculate_electrical_properties(params)
    
    # Create a grid of points for a 2D slice
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create density field for center slice
    center_density = params.mycelium_density * (1 - 0.8 * ((X - 0.5)**2 + (Y - 0.5)**2))
    center_density *= (0.5 + 0.5 * params.moisture_content)
    center_density *= (1 - 0.3 * params.lignin_content)
    center_density *= (0.7 + 0.3 * resistance)
    
    # Electrical conductivity field
    conductivity_field = center_density * electrical_props['conductivity'] * 100
    
    # Dielectric constant field
    dielectric_field = np.ones_like(center_density) * electrical_props['dielectric']
    dielectric_field *= (0.8 + 0.4 * center_density)  # Mycelium affects dielectric properties
    
    # Signal propagation field (inverse of dielectric)
    propagation_field = 1.0 / np.sqrt(dielectric_field)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Mycelium Density', 
            'Electrical Conductivity', 
            'Dielectric Constant', 
            'Signal Propagation Speed'
        )
    )
    
    # Add heatmaps
    fig.add_trace(
        go.Heatmap(
            z=center_density, 
            x=x, 
            y=y,
            colorscale='Viridis',
            colorbar=dict(title='Density', x=0.46, y=0.8, len=0.4),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Density: %{z:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=conductivity_field, 
            x=x, 
            y=y,
            colorscale='Jet',
            colorbar=dict(title='Conductivity (S/m)', x=0.96, y=0.8, len=0.4),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Conductivity: %{z:.6f} S/m<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Heatmap(
            z=dielectric_field, 
            x=x, 
            y=y,
            colorscale='Greens',
            colorbar=dict(title='Dielectric Constant', x=0.46, y=0.3, len=0.4),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Dielectric: %{z:.2f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=propagation_field, 
            x=x, 
            y=y,
            colorscale='Plasma',
            colorbar=dict(title='Propagation Speed (c)', x=0.96, y=0.3, len=0.4),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Speed: %{z:.3f}c<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Substrate Cross-Section with Electrical Properties<br>{params.substrate_type if hasattr(params, "substrate_type") else "Mixed"} Substrate, pH {ph:.1f}',
            x=0.5,
            font=dict(size=18)
        ),
        height=800,
        width=1000,
        showlegend=False
    )
    
    # Update axes labels
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="X", row=i, col=j)
            fig.update_yaxes(title_text="Y", row=i, col=j)
    
    return fig

def create_interactive_combined_visualization(params):
    """Create a comprehensive interactive visualization with 3D model and property plots"""
    # Calculate properties
    conductivity = calculate_conductivity(params)
    ph = calculate_ph(params)
    resistance = calculate_contaminant_resistance(params)
    electrical_props = calculate_electrical_properties(params)
    
    # Create a grid of points for 3D (less dense for performance)
    x_3d = np.linspace(0, 1, 15)
    y_3d = np.linspace(0, 1, 15)
    z_3d = np.linspace(0, 1, 15)
    X_3d, Y_3d, Z_3d = np.meshgrid(x_3d, y_3d, z_3d, indexing='ij')
    
    # Create a density field for 3D
    density_3d = params.mycelium_density * (1 - 0.8 * ((X_3d - 0.5)**2 + (Y_3d - 0.5)**2 + (Z_3d - 0.5)**2))
    density_3d *= (0.5 + 0.5 * params.moisture_content)
    density_3d *= (1 - 0.3 * params.lignin_content)
    density_3d *= (0.7 + 0.3 * resistance)
    
    # Create a mask for points with significant mycelium presence
    mask_3d = density_3d > 0.2
    
    # Create color map based on electrical conductivity
    conductivity_normalized = np.clip(density_3d * electrical_props['conductivity'] * 100, 0, 1)
    
    # Extract points that meet the threshold
    x_points = X_3d[mask_3d].flatten()
    y_points = Y_3d[mask_3d].flatten()
    z_points = Z_3d[mask_3d].flatten()
    colors = conductivity_normalized[mask_3d].flatten()
    
    # Create a grid for 2D plots
    x_2d = np.linspace(0, 1, 100)
    y_2d = np.linspace(0, 1, 100)
    X_2d, Y_2d = np.meshgrid(x_2d, y_2d)
    
    # Create density field for 2D (center slice)
    center_density = params.mycelium_density * (1 - 0.8 * ((X_2d - 0.5)**2 + (Y_2d - 0.5)**2))
    center_density *= (0.5 + 0.5 * params.moisture_content)
    center_density *= (1 - 0.3 * params.lignin_content)
    center_density *= (0.7 + 0.3 * resistance)
    
    # Electrical conductivity field
    conductivity_field = center_density * electrical_props['conductivity'] * 100
    
    # Dielectric constant field
    dielectric_field = np.ones_like(center_density) * electrical_props['dielectric']
    dielectric_field *= (0.8 + 0.4 * center_density)  # Mycelium affects dielectric properties
    
    # Signal propagation field
    propagation_field = 1.0 / np.sqrt(dielectric_field)
    
    # Create subplots with 3D model and 2D plots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'type': 'scene', 'rowspan': 2}, {'type': 'xy'}],
            [None, {'type': 'xy'}]
        ],
        subplot_titles=(
            '3D Substrate Model', 
            'Mycelium Density',
            '',
            'Electrical Conductivity'
        ),
        vertical_spacing=0.05,
        horizontal_spacing=0.05
    )
    
    # Add 3D scatter plot
    fig.add_trace(
        go.Scatter3d(
            x=x_points,
            y=y_points,
            z=z_points,
            mode='markers',
            marker=dict(
                size=4,
                color=colors,
                colorscale='Jet',
                opacity=0.8,
                colorbar=dict(
                    title="Conductivity",
                    thickness=20,
                    len=0.5,
                    x=0.45
                )
            ),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>Conductivity: %{marker.color:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add substrate box
    for i, j, k in product([0, 1], [0, 1], [0, 1]):
        for i2, j2, k2 in product([0, 1], [0, 1], [0, 1]):
            if sum(abs(n1-n2) for n1, n2 in zip([i, j, k], [i2, j2, k2])) == 1:
                fig.add_trace(
                    go.Scatter3d(
                        x=[i, i2],
                        y=[j, j2],
                        z=[k, k2],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )
    
    # Add density heatmap
    fig.add_trace(
        go.Heatmap(
            z=center_density, 
            x=x_2d, 
            y=y_2d,
            colorscale='Viridis',
            colorbar=dict(title='Density', x=1.0, y=0.8, len=0.4),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Density: %{z:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add conductivity heatmap
    fig.add_trace(
        go.Heatmap(
            z=conductivity_field, 
            x=x_2d, 
            y=y_2d,
            colorscale='Jet',
            colorbar=dict(title='Conductivity (S/m)', x=1.0, y=0.3, len=0.4),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Conductivity: %{z:.6f} S/m<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update 3D scene
    fig.update_scenes(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube',
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
    
    # Update 2D axes
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_yaxes(title_text="Y", row=1, col=2)
    fig.update_xaxes(title_text="X", row=2, col=2)
    fig.update_yaxes(title_text="Y", row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Comprehensive Substrate Visualization<br>{params.substrate_type if hasattr(params, "substrate_type") else "Mixed"} Substrate, pH {ph:.1f}, Conductivity {conductivity:.6f} S/m',
            x=0.5,
            font=dict(size=18)
        ),
        height=900,
        width=1200,
        margin=dict(l=10, r=10, t=80, b=10)
    )
    
    # Add annotations with key parameters
    substrate_info = (
        f"Substrate: {params.substrate_type if hasattr(params, 'substrate_type') else 'Mixed'}<br>"
        f"Moisture: {params.moisture_content:.2f}<br>"
        f"pH: {ph:.1f}<br>"
        f"Resistivity: {electrical_props['resistivity']:.2f} Ω·m<br>"
        f"Dielectric: {electrical_props['dielectric']:.1f}<br>"
        f"Mycelium Density: {params.mycelium_density:.2f}"
    )
    
    fig.add_annotation(
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        text=substrate_info,
        showarrow=False,
        font=dict(size=14),
        bgcolor="white",
        opacity=0.8,
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    return fig

def create_multi_tab_visualization(params):
    """Create a multi-tab interactive visualization with different views of the substrate"""
    # Calculate properties
    conductivity = calculate_conductivity(params)
    ph = calculate_ph(params)
    resistance = calculate_contaminant_resistance(params)
    electrical_props = calculate_electrical_properties(params)
    growth_potential = calculate_growth_potential(params)
    
    # Create a figure with tabs
    fig = go.Figure()
    
    # Add tabs for different visualizations
    # Tab 1: Mycelium Structure
    add_mycelium_structure_tab(fig, params)
    
    # Tab 2: Electrical Conductivity
    add_conductivity_tab(fig, params)
    
    # Tab 3: Growth Potential
    add_growth_potential_tab(fig, params)
    
    # Tab 4: Dielectric Properties
    add_dielectric_tab(fig, params)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Interactive Substrate Visualization - {params.substrate_type}',
            x=0.5,
            font=dict(size=20, color='#333333')
        ),
        height=900,
        width=1200,
        margin=dict(l=10, r=10, t=80, b=10),
        scene=dict(
            xaxis_title='X (cm)',
            yaxis_title='Y (cm)',
            zaxis_title='Z (cm)',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.5)
            )
        ),
        updatemenus=[{
            'type': 'buttons',
            'direction': 'right',
            'showactive': True,
            'x': 0.1,
            'y': 1.15,
            'buttons': [
                {'label': 'Mycelium Structure', 
                 'method': 'update', 
                 'args': [{'visible': [True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False]},
                          {'title': 'Mycelium Structure Visualization'}]},
                {'label': 'Electrical Conductivity', 
                 'method': 'update', 
                 'args': [{'visible': [False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False]},
                          {'title': 'Electrical Conductivity Visualization'}]},
                {'label': 'Growth Potential', 
                 'method': 'update', 
                 'args': [{'visible': [False, False, False, False, False, False, False, False, False, True, True, True, True, False, False, False]},
                          {'title': 'Growth Potential Visualization'}]},
                {'label': 'Dielectric Properties', 
                 'method': 'update', 
                 'args': [{'visible': [False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True]},
                          {'title': 'Dielectric Properties Visualization'}]}
            ]
        }]
    )
    
    # Add a summary annotation with key parameters
    summary_info = (
        f"<b>Substrate: {params.substrate_type}</b><br>"
        f"Moisture: {params.moisture_content:.2f} (v/v)<br>"
        f"Lignin: {params.lignin_content:.2f} (w/w)<br>"
        f"pH: {ph:.1f}<br>"
        f"Conductivity: {conductivity:.2e} S/m<br>"
        f"Growth Potential: {growth_potential:.2f}<br>"
        f"<i>Use buttons above to switch between views</i>"
    )
    
    fig.add_annotation(
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        text=summary_info,
        showarrow=False,
        font=dict(size=14),
        bgcolor="white",
        opacity=0.8,
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="center"
    )
    
    return fig

def save_and_open_multi_tab_visualization(fig, params, filename="multi_tab_substrate_visualization.html"):
    """Save the multi-tab visualization to an HTML file and open it in a browser"""
    try:
        # Create a visualizations directory in the project folder if it doesn't exist
        vis_dir = os.path.join(os.getcwd(), "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Change the extension to HTML if it's not already
        if not filename.lower().endswith('.html'):
            filename = os.path.splitext(filename)[0] + '.html'
        
        # Full path to the file
        filepath = os.path.join(vis_dir, filename)
        print(f"Saving multi-tab visualization to: {filepath}")
        
        # Save the figure as HTML
        fig.write_html(
            filepath,
            include_plotlyjs='cdn',  # Use CDN for plotly.js
            full_html=True,
            include_mathjax='cdn',   # Use CDN for MathJax
            config={
                'displayModeBar': True,
                'editable': True,
                'scrollZoom': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'substrate_visualization',
                    'height': 900,
                    'width': 1200,
                    'scale': 1
                }
            }
        )
        
        # Verify HTML file exists
        if not os.path.exists(filepath):
            print(f"ERROR: HTML file was not created at {filepath}")
            return None
        else:
            print(f"HTML file created successfully at {filepath}")
        
        # Try to open the file in a browser
        print(f"Attempting to open file in browser: {filepath}")
        try:
            # Fix the file URL format
            file_url = 'file://' + os.path.abspath(filepath).replace('\\', '/')
            print(f"Opening URL: {file_url}")
            
            # Try to open the URL
            webbrowser.open(file_url)
        except Exception as e:
            print(f"Error opening file in browser: {e}")
            print("Please open the file manually at:", filepath)
        
        return filepath
    
    except Exception as e:
        print(f"Error in save_and_open_multi_tab_visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def add_mycelium_structure_tab(fig, params):
    """Add mycelium structure visualization to the figure"""
    # Calculate properties
    conductivity = calculate_conductivity(params)
    ph = calculate_ph(params)
    resistance = calculate_contaminant_resistance(params)
    electrical_props = calculate_electrical_properties(params)
    
    # Set dimensions in cm
    width, length, height = 10, 10, 10
    
    # Create a more detailed mycelium network
    # 1. Generate primary hyphae (main branches)
    np.random.seed(42)  # For reproducibility
    
    # Number of primary hyphae based on mycelium density
    num_primary = int(30 * params.mycelium_density)
    
    # Track all points for later coloring
    all_x, all_y, all_z = [], [], []
    all_types = []  # 0: primary, 1: secondary, 2: tertiary, 3: hyphal tips
    all_diameters = []  # Hyphal diameters in micrometers
    all_ages = []  # Relative age (0-1)
    
    # Growth parameters affected by substrate properties
    growth_rate = 0.5 + 0.5 * params.moisture_content
    growth_rate *= (1 - 0.3 * params.lignin_content)
    growth_rate *= (0.7 + 0.3 * resistance)
    
    # Branching probability affected by nitrogen content
    branching_prob = 0.3 + 0.7 * params.nitrogen_content
    
    # Direction bias based on substrate type
    vertical_bias = 0.6 if params.substrate_type == "STRAW" else 0.3
    
    # Generate primary hyphae paths with growth towards center and up
    for i, (x, y, z) in enumerate(primary_starts):
        # Length of this primary hypha
        primary_length = int(15 + 20 * growth_rate * np.random.random())
        
        # Starting direction - tend to grow upward and toward center
        dx = 0.1 * (width/2 - x) + 0.05 * np.random.randn()
        dy = 0.1 * (length/2 - y) + 0.05 * np.random.randn()
        dz = vertical_bias + 0.1 * np.random.randn()
        
        # Normalize direction vector
        mag = np.sqrt(dx**2 + dy**2 + dz**2)
        dx, dy, dz = dx/mag, dy/mag, dz/mag
        
        # Generate points along primary hypha
        prev_x, prev_y, prev_z = x, y, z
        primary_points = [(x, y, z)]
        
        for j in range(primary_length):
            # Add some randomness to direction
            dx += 0.1 * np.random.randn()
            dy += 0.1 * np.random.randn()
            dz += 0.1 * np.random.randn()
            
            # Normalize again
            mag = np.sqrt(dx**2 + dy**2 + dz**2)
            dx, dy, dz = dx/mag, dy/mag, dz/mag
            
            # Calculate new point
            new_x = prev_x + 0.3 * dx
            new_y = prev_y + 0.3 * dy
            new_z = prev_z + 0.3 * dz
            
            # Keep within bounds
            new_x = max(0, min(width, new_x))
            new_y = max(0, min(length, new_y))
            new_z = max(0, min(height, new_z))
            
            # Add point
            primary_points.append((new_x, new_y, new_z))
            prev_x, prev_y, prev_z = new_x, new_y, new_z
            
            # Add to overall tracking
            all_x.extend([prev_x, new_x])
            all_y.extend([prev_y, new_y])
            all_z.extend([prev_z, new_z])
            all_types.extend([0, 0])  # Primary type
            
            # Primary hyphae are thicker (3-10 µm)
            diameter = 3 + 7 * (1 - j/primary_length)  # Thinner as they extend
            all_diameters.extend([diameter, diameter])
            
            # Age decreases along length
            age = 1 - j/primary_length
            all_ages.extend([age, age])
            
            # Generate secondary branches with some probability
            if np.random.random() < branching_prob and j > 2:
                # Create a secondary branch
                sec_x, sec_y, sec_z = new_x, new_y, new_z
                
                # Direction for secondary - tend to grow perpendicular to primary
                cross_x = -dy + 0.2 * np.random.randn()
                cross_y = dx + 0.2 * np.random.randn()
                cross_z = 0.3 * np.random.randn()
                
                # Normalize
                mag = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
                cross_x, cross_y, cross_z = cross_x/mag, cross_y/mag, cross_z/mag
                
                # Length of secondary branch
                sec_length = int(5 + 10 * growth_rate * np.random.random())
                
                # Generate secondary branch
                sec_prev_x, sec_prev_y, sec_prev_z = sec_x, sec_y, sec_z
                
                for k in range(sec_length):
                    # Add randomness to direction
                    cross_x += 0.15 * np.random.randn()
                    cross_y += 0.15 * np.random.randn()
                    cross_z += 0.15 * np.random.randn()
                    
                    # Normalize
                    mag = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
                    cross_x, cross_y, cross_z = cross_x/mag, cross_y/mag, cross_z/mag
                    
                    # Calculate new point
                    sec_new_x = sec_prev_x + 0.2 * cross_x
                    sec_new_y = sec_prev_y + 0.2 * cross_y
                    sec_new_z = sec_prev_z + 0.2 * cross_z
                    
                    # Keep within bounds
                    sec_new_x = max(0, min(width, sec_new_x))
                    sec_new_y = max(0, min(length, sec_new_y))
                    sec_new_z = max(0, min(height, sec_new_z))
                    
                    # Add to tracking
                    all_x.extend([sec_prev_x, sec_new_x])
                    all_y.extend([sec_prev_y, sec_new_y])
                    all_z.extend([sec_prev_z, sec_new_z])
                    all_types.extend([1, 1])  # Secondary type
                    
                    # Secondary hyphae are thinner (2-5 µm)
                    sec_diameter = 2 + 3 * (1 - k/sec_length)
                    all_diameters.extend([sec_diameter, sec_diameter])
                    
                    # Age for secondary
                    sec_age = (1 - j/primary_length) * (1 - k/sec_length)
                    all_ages.extend([sec_age, sec_age])
                    
                    sec_prev_x, sec_prev_y, sec_prev_z = sec_new_x, sec_new_y, sec_new_z
                    
                    # Generate tertiary branches with lower probability
                    if k > 2 and np.random.random() < 0.3 * branching_prob:
                        # Create tertiary branch
                        tert_x, tert_y, tert_z = sec_new_x, sec_new_y, sec_new_z
                        
                        # Direction - more random
                        tert_dx = np.random.randn()
                        tert_dy = np.random.randn()
                        tert_dz = np.random.randn()
                        
                        # Normalize
                        mag = np.sqrt(tert_dx**2 + tert_dy**2 + tert_dz**2)
                        tert_dx, tert_dy, tert_dz = tert_dx/mag, tert_dy/mag, tert_dz/mag
                        
                        # Length of tertiary branch
                        tert_length = int(3 + 5 * growth_rate * np.random.random())
                        
                        # Generate tertiary branch
                        tert_prev_x, tert_prev_y, tert_prev_z = tert_x, tert_y, tert_z
                        
                        for m in range(tert_length):
                            # Add randomness
                            tert_dx += 0.2 * np.random.randn()
                            tert_dy += 0.2 * np.random.randn()
                            tert_dz += 0.2 * np.random.randn()
                            
                            # Normalize
                            mag = np.sqrt(tert_dx**2 + tert_dy**2 + tert_dz**2)
                            tert_dx, tert_dy, tert_dz = tert_dx/mag, tert_dy/mag, tert_dz/mag
                            
                            # Calculate new point
                            tert_new_x = tert_prev_x + 0.1 * tert_dx
                            tert_new_y = tert_prev_y + 0.1 * tert_dy
                            tert_new_z = tert_prev_z + 0.1 * tert_dz
                            
                            # Keep within bounds
                            tert_new_x = max(0, min(width, tert_new_x))
                            tert_new_y = max(0, min(length, tert_new_y))
                            tert_new_z = max(0, min(height, tert_new_z))
                            
                            # Add to tracking
                            all_x.extend([tert_prev_x, tert_new_x])
                            all_y.extend([tert_prev_y, tert_new_y])
                            all_z.extend([tert_prev_z, tert_new_z])
                            all_types.extend([2, 2])  # Tertiary type
                            
                            # Tertiary hyphae are thinnest (1-3 µm)
                            tert_diameter = 1 + 2 * (1 - m/tert_length)
                            all_diameters.extend([tert_diameter, tert_diameter])
                            
                            # Age for tertiary
                            tert_age = sec_age * (1 - m/tert_length)
                            all_ages.extend([tert_age, tert_age])
                            
                            tert_prev_x, tert_prev_y, tert_prev_z = tert_new_x, tert_new_y, tert_new_z
                            
                            # Add hyphal tips at the ends
                            if m == tert_length - 1:
                                all_x.append(tert_new_x)
                                all_y.append(tert_new_y)
                                all_z.append(tert_new_z)
                                all_types.append(3)  # Hyphal tip
                                all_diameters.append(1.5)  # Tips are thin
                                all_ages.append(0.1)  # Tips are new
    
    # Convert lists to arrays
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_z = np.array(all_z)
    all_types = np.array(all_types)
    all_diameters = np.array(all_diameters)
    all_ages = np.array(all_ages)
    
    # Calculate electrical conductivity for each point
    # Higher in older, thicker hyphae and affected by substrate properties
    conductivity_values = electrical_props['conductivity'] * 100 * all_ages * (all_diameters / 10)
    
    # Create separate traces for different hyphal types for better visualization
    # Instead of using line width arrays, we'll use markers with size based on diameter
    
    # Primary hyphae (type 0)
    mask_primary = all_types == 0
    if np.sum(mask_primary) > 0:
        fig.add_trace(go.Scatter3d(
            x=all_x[mask_primary],
            y=all_y[mask_primary],
            z=all_z[mask_primary],
            mode='markers',
            marker=dict(
                size=all_diameters[mask_primary] * 0.8,  # Scale for visualization
                color='#FF3333',  # Bright red
                opacity=0.8
            ),
            name='Primary Hyphae',
            hovertemplate=(
                'X: %{x:.1f} cm<br>'
                'Y: %{y:.1f} cm<br>'
                'Z: %{z:.1f} cm<br>'
                'Diameter: %{marker.size:.1f} µm<br>'
                'Age: %{marker.color:.2f}<extra>Primary Hypha</extra>'
            )
        ))
    
    # Secondary hyphae (type 1)
    mask_secondary = all_types == 1
    if np.sum(mask_secondary) > 0:
        fig.add_trace(go.Scatter3d(
            x=all_x[mask_secondary],
            y=all_y[mask_secondary],
            z=all_z[mask_secondary],
            mode='markers',
            marker=dict(
                size=all_diameters[mask_secondary] * 0.8,
                color='#3366FF',  # Bright blue
                opacity=0.8
            ),
            name='Secondary Hyphae',
            hovertemplate=(
                'X: %{x:.1f} cm<br>'
                'Y: %{y:.1f} cm<br>'
                'Z: %{z:.1f} cm<br>'
                'Diameter: %{marker.size:.1f} µm<br>'
                'Age: %{marker.color:.2f}<extra>Secondary Hypha</extra>'
            )
        ))
    
    # Tertiary hyphae (type 2)
    mask_tertiary = all_types == 2
    if np.sum(mask_tertiary) > 0:
        fig.add_trace(go.Scatter3d(
            x=all_x[mask_tertiary],
            y=all_y[mask_tertiary],
            z=all_z[mask_tertiary],
            mode='markers',
            marker=dict(
                size=all_diameters[mask_tertiary] * 0.8,
                color='#33CC33',  # Bright green
                opacity=0.8
            ),
            name='Tertiary Hyphae',
            hovertemplate=(
                'X: %{x:.1f} cm<br>'
                'Y: %{y:.1f} cm<br>'
                'Z: %{z:.1f} cm<br>'
                'Diameter: %{marker.size:.1f} µm<br>'
                'Age: %{marker.color:.2f}<extra>Tertiary Hypha</extra>'
            )
        ))
    
    # Hyphal tips (type 3)
    mask_tips = all_types == 3
    if np.sum(mask_tips) > 0:
        fig.add_trace(go.Scatter3d(
            x=all_x[mask_tips],
            y=all_y[mask_tips],
            z=all_z[mask_tips],
            mode='markers',
            marker=dict(
                size=4,  # Fixed size for tips
                color='#FF55FF',  # Bright purple
                symbol='circle',
                line=dict(color='#CC55CC', width=1)
            ),
            name='Hyphal Tips',
            hovertemplate=(
                'X: %{x:.1f} cm<br>'
                'Y: %{y:.1f} cm<br>'
                'Z: %{z:.1f} cm<br>'
                'Diameter: 1-2 µm<br>'
                'Growing Point<extra>Hyphal Tip</extra>'
            )
        ))
    
    # Add substrate box with semi-transparent walls
    for i, j, k in product([0, width], [0, length], [0, height]):
        for i2, j2, k2 in product([0, width], [0, length], [0, height]):
            if sum(abs(n1-n2) for n1, n2 in zip([i, j, k], [i2, j2, k2])) == width:
                fig.add_trace(go.Scatter3d(
                    x=[i, i2],
                    y=[j, j2],
                    z=[k, k2],
                    mode='lines',
                    line=dict(color='rgba(200,200,200,0.5)', width=2),  # Fixed width as a single number
                    showlegend=False
                ))
    
    # Add measurement axes with ticks
    # X-axis
    fig.add_trace(go.Scatter3d(
        x=[0, 11],
        y=[0, 0],
        z=[0, 0],
        mode='lines+text',
        line=dict(color='white', width=3),  # Fixed width as a single number
        text=['', 'X (cm)'],
        textposition='middle right',
        showlegend=False
    ))
    
    # Y-axis
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 11],
        z=[0, 0],
        mode='lines+text',
        line=dict(color='white', width=3),  # Fixed width as a single number
        text=['', 'Y (cm)'],
        textposition='middle right',
        showlegend=False
    ))
    
    # Z-axis
    fig.add_trace(go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[0, 11],
        mode='lines+text',
        line=dict(color='white', width=3),  # Fixed width as a single number
        text=['', 'Z (cm)'],
        textposition='middle right',
        showlegend=False
    ))
    
    # Add ticks on axes
    for i in range(2, 11, 2):
        # X-axis ticks
        fig.add_trace(go.Scatter3d(
            x=[i, i],
            y=[0, -0.3],
            z=[0, 0],
            mode='lines+text',
            line=dict(color='white', width=2),  # Fixed width as a single number
            text=['', str(i)],
            textposition='bottom center',
            showlegend=False
        ))
        
        # Y-axis ticks
        fig.add_trace(go.Scatter3d(
            x=[0, -0.3],
            y=[i, i],
            z=[0, 0],
            mode='lines+text',
            line=dict(color='white', width=2),  # Fixed width as a single number
            text=['', str(i)],
            textposition='bottom center',
            showlegend=False
        ))
        
        # Z-axis ticks
        fig.add_trace(go.Scatter3d(
            x=[0, -0.3],
            y=[0, 0],
            z=[i, i],
            mode='lines+text',
            line=dict(color='white', width=2),  # Fixed width as a single number
            text=['', str(i)],
            textposition='bottom center',
            showlegend=False
        ))
    
    # Update layout with dark background
    fig.update_layout(
        title=dict(
            text=f'Interactive 3D Substrate Model (10×10×10 cm)<br>Mycelium Density: {params.mycelium_density:.2f}, Conductivity: {conductivity:.6e} S/m',
            x=0.5,
            font=dict(size=18, color='white')
        ),
        scene=dict(
            xaxis_title='X (cm)',
            yaxis_title='Y (cm)',
            zaxis_title='Z (cm)',
            xaxis=dict(
                range=[0, 10], 
                dtick=2,
                gridcolor='rgba(100,100,100,0.2)',
                showbackground=True,
                backgroundcolor='rgba(20, 20, 40, 1)'
            ),
            yaxis=dict(
                range=[0, 10], 
                dtick=2,
                gridcolor='rgba(100,100,100,0.2)',
                showbackground=True,
                backgroundcolor='rgba(20, 20, 40, 1)'
            ),
            zaxis=dict(
                range=[0, 10], 
                dtick=2,
                gridcolor='rgba(100,100,100,0.2)',
                showbackground=True,
                backgroundcolor='rgba(20, 20, 40, 1)'
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='rgb(10, 10, 30)'  # Dark blue background
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=800,
        width=1000,
        paper_bgcolor='rgb(10, 10, 30)',  # Dark blue background
        plot_bgcolor='rgb(10, 10, 30)',   # Dark blue background
        font=dict(color='white')          # White text for better contrast
    )
    
    # Add detailed annotations with key parameters and units
    substrate_info = (
        f"<b>Substrate Properties:</b><br>"
        f"Type: {params.substrate_type if hasattr(params, 'substrate_type') else 'Mixed'}<br>"
        f"Moisture: {params.moisture_content:.2f} (v/v)<br>"
        f"Lignin: {params.lignin_content:.2f} (w/w)<br>"
        f"Cellulose: {params.cellulose_content:.2f} (w/w)<br>"
        f"pH: {ph:.1f}<br><br>"
        f"<b>Electrical Properties:</b><br>"
        f"Conductivity: {conductivity:.2e} S/m<br>"
        f"Resistivity: {electrical_props['resistivity']:.2e} Ω·m<br>"
        f"Dielectric Constant: {electrical_props['dielectric']:.1f}<br>"
        f"Signal Propagation: {electrical_props['propagation_speed']:.2f}c<br><br>"
        f"<b>Growth Metrics:</b><br>"
        f"Mycelium Density: {params.mycelium_density:.2f} (v/v)<br>"
        f"Contaminant Resistance: {resistance:.2f}<br>"
        f"Growth Potential: {growth_potential:.2f}"
    )
    
    fig.add_annotation(
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        text=substrate_info,
        showarrow=False,
        font=dict(size=12, color='white'),
        bgcolor="rgba(30, 30, 60, 0.8)",
        opacity=0.9,
        bordercolor="rgba(100, 100, 200, 0.8)",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    # Add a legend for the color scale
    fig.add_annotation(
        x=0.99,
        y=0.5,
        xref="paper",
        yref="paper",
        text="<b>Mycelium Structure</b><br>Red: Primary hyphae (3-10 µm)<br>Blue: Secondary hyphae (2-5 µm)<br>Green: Tertiary hyphae (1-3 µm)<br>Purple: Hyphal tips (1-2 µm)",
        showarrow=False,
        font=dict(size=12, color='white'),
        bgcolor="rgba(30, 30, 60, 0.8)",
        opacity=0.9,
        bordercolor="rgba(100, 100, 200, 0.8)",
        borderwidth=1,
        borderpad=4,
        align="right"
    )
    
    return fig

def add_conductivity_tab(fig, params):
    """Add electrical conductivity visualization to the figure"""
    # Calculate properties
    conductivity = calculate_conductivity(params)
    electrical_props = calculate_electrical_properties(params)
    
    # Create a grid of points for visualization
    x = np.linspace(0, 10, 30)  # 10 cm width
    y = np.linspace(0, 10, 30)  # 10 cm length
    z = np.linspace(0, 10, 30)  # 10 cm height
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create a density field representing mycelium growth
    density = params.mycelium_density * (1 - 0.8 * ((X/10 - 0.5)**2 + (Y/10 - 0.5)**2 + (Z/10 - 0.5)**2))
    
    # Add some randomness for more natural appearance
    np.random.seed(42)  # For reproducibility
    density += np.random.normal(0, 0.05, density.shape) * params.mycelium_density
    
    # Adjust density based on substrate properties
    density *= (0.5 + 0.5 * params.moisture_content)
    density *= (1 - 0.3 * params.lignin_content)
    
    # Calculate conductivity field
    conductivity_field = density * electrical_props['conductivity'] * 100
    
    # Create a mask for points with significant conductivity
    mask = conductivity_field > 0.0001
    
    # Extract points that meet the threshold
    x_points = X[mask].flatten()
    y_points = Y[mask].flatten()
    z_points = Z[mask].flatten()
    cond_values = conductivity_field[mask].flatten()
    
    # Add the conductivity points
    fig.add_trace(go.Scatter3d(
        x=x_points,
        y=y_points,
        z=z_points,
        mode='markers',
        marker=dict(
            size=5,
            color=cond_values,
            colorscale='Plasma',  # Vibrant colorscale
            opacity=0.8,
            colorbar=dict(
                title="Conductivity (S/m)",
                thickness=20,
                len=0.5,
                tickformat='.2e'
            )
        ),
        name='Conductivity',
        hovertemplate=(
            'X: %{x:.1f} cm<br>'
            'Y: %{y:.1f} cm<br>'
            'Z: %{z:.1f} cm<br>'
            'Conductivity: %{marker.color:.2e} S/m<extra></extra>'
        ),
        visible=False  # Initially hidden, shown when tab is selected
    ))
    
    # Add substrate box
    for i, j, k in product([0, 10], [0, 10], [0, 10]):
        for i2, j2, k2 in product([0, 10], [0, 10], [0, 10]):
            if sum(abs(n1-n2) for n1, n2 in zip([i, j, k], [i2, j2, k2])) == 10:
                fig.add_trace(go.Scatter3d(
                    x=[i, i2],
                    y=[j, j2],
                    z=[k, k2],
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.4)', width=2),
                    showlegend=False,
                    visible=False  # Initially hidden
                ))
    
    # Add a slice through the middle to show conductivity gradient
    # X-Y plane at middle Z
    z_mid = 5
    x_slice = np.linspace(0, 10, 50)
    y_slice = np.linspace(0, 10, 50)
    X_slice, Y_slice = np.meshgrid(x_slice, y_slice)
    
    # Calculate conductivity at the slice
    Z_slice = np.ones_like(X_slice) * z_mid
    slice_density = params.mycelium_density * (1 - 0.8 * ((X_slice/10 - 0.5)**2 + (Y_slice/10 - 0.5)**2 + (z_mid/10 - 0.5)**2))
    slice_density *= (0.5 + 0.5 * params.moisture_content)
    slice_density *= (1 - 0.3 * params.lignin_content)
    slice_conductivity = slice_density * electrical_props['conductivity'] * 100
    
    # Add the slice
    fig.add_trace(go.Surface(
        x=X_slice,
        y=Y_slice,
        z=Z_slice,
        surfacecolor=slice_conductivity,
        colorscale='Plasma',
        opacity=0.8,
        showscale=False,
        name='Conductivity Slice',
        hovertemplate=(
            'X: %{x:.1f} cm<br>'
            'Y: %{y:.1f} cm<br>'
            'Z: %{z:.1f} cm<br>'
            'Conductivity: %{surfacecolor:.2e} S/m<extra>Slice</extra>'
        ),
        visible=False  # Initially hidden
    ))

def add_growth_potential_tab(fig, params):
    """Add growth potential visualization to the figure"""
    # Calculate properties
    growth_potential = calculate_growth_potential(params)
    ph = calculate_ph(params)
    
    # Create a grid for visualization
    x = np.linspace(0, 10, 30)  # 10 cm width
    y = np.linspace(0, 10, 30)  # 10 cm length
    z = np.linspace(0, 10, 30)  # 10 cm height
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create a growth potential field
    # Base growth potential affected by position (better near surface)
    growth_field = growth_potential * (1 - 0.5 * ((X/10 - 0.5)**2 + (Y/10 - 0.5)**2))
    
    # Add depth effect - growth decreases with depth
    growth_field *= (1.2 - 0.8 * (Z / 10))
    
    # Add some randomness for more natural appearance
    np.random.seed(42)  # For reproducibility
    growth_field += np.random.normal(0, 0.05, growth_field.shape) * growth_potential
    
    # Clip values to valid range
    growth_field = np.clip(growth_field, 0, 1)
    
    # Create a mask for points with significant growth potential
    mask = growth_field > 0.3
    
    # Extract points that meet the threshold
    x_points = X[mask].flatten()
    y_points = Y[mask].flatten()
    z_points = Z[mask].flatten()
    growth_values = growth_field[mask].flatten()
    
    # Add the growth potential points
    fig.add_trace(go.Scatter3d(
        x=x_points,
        y=y_points,
        z=z_points,
        mode='markers',
        marker=dict(
            size=5,
            color=growth_values,
            colorscale='Viridis',  # Green-blue colorscale
            opacity=0.8,
            colorbar=dict(
                title="Growth Potential",
                thickness=20,
                len=0.5
            )
        ),
        name='Growth Potential',
        hovertemplate=(
            'X: %{x:.1f} cm<br>'
            'Y: %{y:.1f} cm<br>'
            'Z: %{z:.1f} cm<br>'
            'Growth: %{marker.color:.2f}<extra></extra>'
        ),
        visible=False  # Initially hidden
    ))
    
    # Add substrate box
    for i, j, k in product([0, 10], [0, 10], [0, 10]):
        for i2, j2, k2 in product([0, 10], [0, 10], [0, 10]):
            if sum(abs(n1-n2) for n1, n2 in zip([i, j, k], [i2, j2, k2])) == 10:
                fig.add_trace(go.Scatter3d(
                    x=[i, i2],
                    y=[j, j2],
                    z=[k, k2],
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.4)', width=2),
                    showlegend=False,
                    visible=False  # Initially hidden
                ))
    
    # Add slices through the middle to show growth potential gradient
    # X-Y plane at different Z levels
    for z_level in [2, 5, 8]:
        x_slice = np.linspace(0, 10, 50)
        y_slice = np.linspace(0, 10, 50)
        X_slice, Y_slice = np.meshgrid(x_slice, y_slice)
        
        # Calculate growth potential at the slice
        Z_slice = np.ones_like(X_slice) * z_level
        slice_growth = growth_potential * (1 - 0.5 * ((X_slice/10 - 0.5)**2 + (Y_slice/10 - 0.5)**2))
        slice_growth *= (1.2 - 0.8 * (z_level / 10))
        
        # Add randomness
        np.random.seed(42 + z_level)  # Different seed for each slice
        slice_growth += np.random.normal(0, 0.05, slice_growth.shape) * growth_potential
        slice_growth = np.clip(slice_growth, 0, 1)
        
        # Add the slice
        fig.add_trace(go.Surface(
            x=X_slice,
            y=Y_slice,
            z=Z_slice,
            surfacecolor=slice_growth,
            colorscale='Viridis',
            opacity=0.7,
            showscale=False,
            name=f'Growth Slice Z={z_level}',
            hovertemplate=(
                'X: %{x:.1f} cm<br>'
                'Y: %{y:.1f} cm<br>'
                'Z: {z_level} cm<br>'
                'Growth: %{surfacecolor:.2f}<extra>Slice</extra>'
            ).format(z_level=z_level),
            visible=False  # Initially hidden
        ))

def add_dielectric_tab(fig, params):
    """Add dielectric properties visualization to the figure"""
    # Calculate properties
    electrical_props = calculate_electrical_properties(params)
    
    # Create a grid for visualization
    x = np.linspace(0, 10, 30)  # 10 cm width
    y = np.linspace(0, 10, 30)  # 10 cm length
    z = np.linspace(0, 10, 30)  # 10 cm height
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create a density field representing mycelium growth
    density = params.mycelium_density * (1 - 0.8 * ((X/10 - 0.5)**2 + (Y/10 - 0.5)**2 + (Z/10 - 0.5)**2))
    
    # Add some randomness for more natural appearance
    np.random.seed(42)  # For reproducibility
    density += np.random.normal(0, 0.05, density.shape) * params.mycelium_density
    
    # Adjust density based on substrate properties
    density *= (0.5 + 0.5 * params.moisture_content)
    density *= (1 - 0.3 * params.lignin_content)
    
    # Calculate dielectric field
    # Base dielectric constant from substrate
    dielectric_field = np.ones_like(density) * electrical_props['dielectric']
    
    # Adjust based on mycelium density (mycelium slightly increases dielectric constant)
    dielectric_field *= (1 + 0.2 * density)
    
    # Calculate signal propagation speed (relative to c)
    propagation_field = 1.0 / np.sqrt(dielectric_field)
    
    # Create a mask for visualization
    mask = density > 0.1
    
    # Extract points that meet the threshold
    x_points = X[mask].flatten()
    y_points = Y[mask].flatten()
    z_points = Z[mask].flatten()
    dielectric_values = dielectric_field[mask].flatten()
    propagation_values = propagation_field[mask].flatten()
    
    # Add the dielectric constant points
    fig.add_trace(go.Scatter3d(
        x=x_points,
        y=y_points,
        z=z_points,
        mode='markers',
        marker=dict(
            size=5,
            color=dielectric_values,
            colorscale='Cividis',  # Yellow-purple colorscale
            opacity=0.8,
            colorbar=dict(
                title="Dielectric Constant",
                thickness=20,
                len=0.5
            )
        ),
        name='Dielectric Constant',
        hovertemplate=(
            'X: %{x:.1f} cm<br>'
            'Y: %{y:.1f} cm<br>'
            'Z: %{z:.1f} cm<br>'
            'Dielectric: %{marker.color:.1f}<extra></extra>'
        ),
        visible=False  # Initially hidden
    ))
    
    # Add substrate box
    for i, j, k in product([0, 10], [0, 10], [0, 10]):
        for i2, j2, k2 in product([0, 10], [0, 10], [0, 10]):
            if sum(abs(n1-n2) for n1, n2 in zip([i, j, k], [i2, j2, k2])) == 10:
                fig.add_trace(go.Scatter3d(
                    x=[i, i2],
                    y=[j, j2],
                    z=[k, k2],
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.4)', width=2),
                    showlegend=False,
                    visible=False  # Initially hidden
                ))
    
    # Add a slice through the middle to show dielectric gradient
    # X-Y plane at middle Z
    z_mid = 5
    x_slice = np.linspace(0, 10, 50)
    y_slice = np.linspace(0, 10, 50)
    X_slice, Y_slice = np.meshgrid(x_slice, y_slice)
    
    # Calculate dielectric at the slice
    Z_slice = np.ones_like(X_slice) * z_mid
    slice_density = params.mycelium_density * (1 - 0.8 * ((X_slice/10 - 0.5)**2 + (Y_slice/10 - 0.5)**2 + (z_mid/10 - 0.5)**2))
    slice_density *= (0.5 + 0.5 * params.moisture_content)
    slice_density *= (1 - 0.3 * params.lignin_content)
    
    slice_dielectric = np.ones_like(slice_density) * electrical_props['dielectric']
    slice_dielectric *= (1 + 0.2 * slice_density)
    
    # Add the slice
    fig.add_trace(go.Surface(
        x=X_slice,
        y=Y_slice,
        z=Z_slice,
        surfacecolor=slice_dielectric,
        colorscale='Cividis',
        opacity=0.7,
        showscale=False,
        name='Dielectric Slice',
        hovertemplate=(
            'X: %{x:.1f} cm<br>'
            'Y: %{y:.1f} cm<br>'
            'Z: %{z:.1f} cm<br>'
            'Dielectric: %{surfacecolor:.1f}<extra>Slice</extra>'
        ),
        visible=False  # Initially hidden
    ))

def add_substrate_info_annotation(fig, params):
    """Add detailed substrate information annotation to the figure"""
    # Calculate properties
    ph = calculate_ph(params)
    conductivity = calculate_conductivity(params)
    resistance = calculate_contaminant_resistance(params)
    growth_potential = calculate_growth_potential(params)
    
    # Create substrate info text
    substrate_info = (
        f"<b>Substrate Properties:</b><br>"
        f"Type: {params.substrate_type.replace('_', ' ')}<br>"
        f"Moisture: {params.moisture_content:.2f} (v/v)<br>"
        f"Lignin: {params.lignin_content:.2f} (w/w)<br>"
        f"Cellulose: {params.cellulose_content:.2f} (w/w)<br>"
        f"Nitrogen: {params.nitrogen_content:.3f} (w/w)<br>"
        f"pH: {ph:.1f}<br>"
    )
    
    # Add calcium supplement info if used
    if params.calcium_supplement != 'NONE':
        calcium_type = params.calcium_supplement.replace('_', ' ').title()
        calcination_status = "Calcinated" if params.calcium_calcination else "Raw"
        substrate_info += (
            f"<br><b>Calcium Supplementation:</b><br>"
            f"Type: {calcium_type}<br>"
            f"Status: {calcination_status}<br>"
            f"Content: {params.calcium_content:.2f} (w/w)<br>"
        )
        
        # Add specific info for oyster shell
        if params.calcium_supplement == 'OYSTER_SHELL':
            substrate_info += (
                f"<i>Oyster shell provides pH buffering and calcium,<br>"
                f"supporting mycelial growth and fruiting body development.</i><br>"
            )
    
    # Add mycelium info
    substrate_info += (
        f"<br><b>Mycelium Properties:</b><br>"
        f"Species: {params.mycelium_species.replace('_', ' ')}<br>"
        f"Density: {params.mycelium_density:.2f}<br>"
        f"Age: {params.mycelium_age:.1f} days<br>"
        f"Growth Potential: {growth_potential:.2f}<br>"
    )
    
    # Add the annotation to the figure
    fig.add_annotation(
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        text=substrate_info,
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        opacity=0.8,
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    return fig