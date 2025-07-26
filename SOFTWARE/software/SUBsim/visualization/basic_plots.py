import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from simulation.substrate import calculate_ph, calculate_conductivity, calculate_contaminant_resistance, calculate_growth_potential, calculate_electrical_properties

def create_ph_bar_chart(params):
    """Create a bar chart showing pH levels for different pasteurization methods"""
    methods = ['HEAT', 'LIME', 'HYDROGEN_PEROXIDE', 'STEAM', 'BLEACH', 'NONE']
    ph_values = []
    
    # Store original method
    original_method = params.pasteurization_method
    
    # Calculate pH for each method
    for method in methods:
        params.pasteurization_method = method
        ph_values.append(calculate_ph(params))
    
    # Restore original method
    params.pasteurization_method = original_method
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create bars
    bars = ax.bar(methods, ph_values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#95a5a6'])
    
    # Add pH value labels on top of bars
    for bar, ph in zip(bars, ph_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{ph:.1f}', ha='center', va='bottom')
    
    # Add horizontal lines for optimal pH ranges
    ax.axhspan(5.5, 6.5, alpha=0.2, color='green', label='Optimal for most mushrooms')
    
    # Add labels and title
    ax.set_xlabel('Pasteurization Method')
    ax.set_ylabel('pH Level')
    ax.set_title('pH Levels by Pasteurization Method')
    ax.set_ylim(0, 14)
    
    # Add a legend
    ax.legend()
    
    # Add pH scale reference
    ax.text(0.02, 0.02, 'Acidic < 7.0 < Basic', transform=ax.transAxes, fontsize=10)
    
    # Highlight current method
    current_method_index = methods.index(original_method) if original_method in methods else -1
    if current_method_index >= 0:
        bars[current_method_index].set_edgecolor('black')
        bars[current_method_index].set_linewidth(2)
    
    plt.tight_layout()
    return fig

def create_conductivity_moisture_plot(params):
    """Create a line plot showing how conductivity changes with moisture content"""
    moisture_values = np.linspace(0.1, 0.9, 20)
    conductivity_values = []
    
    # Store original moisture
    original_moisture = params.moisture_content
    
    # Calculate conductivity for each moisture level
    for moisture in moisture_values:
        params.moisture_content = moisture
        conductivity_values.append(calculate_conductivity(params))
    
    # Restore original moisture
    params.moisture_content = original_moisture
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot conductivity vs moisture
    ax.plot(moisture_values, conductivity_values, 'b-', linewidth=2)
    
    # Mark current moisture level
    current_conductivity = calculate_conductivity(params)
    ax.plot(original_moisture, current_conductivity, 'ro', markersize=8, 
            label=f'Current: {original_moisture:.2f} moisture, {current_conductivity:.6f} S/m')
    
    # Add optimal range for unconventional computing
    ax.axhspan(0.001, 0.01, alpha=0.2, color='green', label='Optimal for computing')
    
    # Add labels and title
    ax.set_xlabel('Moisture Content (ratio)')
    ax.set_ylabel('Conductivity (S/m)')
    ax.set_title('Electrical Conductivity vs. Moisture Content')
    
    # Add a legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def create_resistance_comparison(params):
    """Create a bar chart comparing contaminant resistance for different substrate types"""
    substrate_types = ['HARDWOOD', 'STRAW', 'COFFEE_GROUNDS', 'SAWDUST', 'COMPOST', 'MIXED']
    resistance_values = []
    
    # Store original substrate type
    original_type = params.substrate_type if hasattr(params, 'substrate_type') else "HARDWOOD"
    
    # Calculate resistance for each substrate type
    for substrate_type in substrate_types:
        params.substrate_type = substrate_type
        resistance_values.append(calculate_contaminant_resistance(params))
    
    # Restore original substrate type
    params.substrate_type = original_type
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create bars
    bars = ax.bar(substrate_types, resistance_values, color=['#8B4513', '#F0E68C', '#6F4E37', '#DEB887', '#556B2F', '#A0522D'])
    
    # Add resistance value labels on top of bars
    for bar, resistance in zip(bars, resistance_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{resistance:.2f}', ha='center', va='bottom')
    
    # Add labels and title
    ax.set_xlabel('Substrate Type')
    ax.set_ylabel('Contaminant Resistance (0-1)')
    ax.set_title('Contaminant Resistance by Substrate Type')
    ax.set_ylim(0, 1.1)
    
    # Highlight current substrate type
    current_type_index = substrate_types.index(original_type) if original_type in substrate_types else -1
    if current_type_index >= 0:
        bars[current_type_index].set_edgecolor('black')
        bars[current_type_index].set_linewidth(2)
    
    plt.tight_layout()
    return fig

def create_growth_potential_heatmap(params):
    """Create a heatmap showing growth potential based on temperature and moisture"""
    # Create a grid of temperature and moisture values
    temperatures = np.linspace(15, 30, 16)  # 15-30°C
    moistures = np.linspace(0.4, 0.8, 16)   # 40-80%
    
    # Store original values
    original_temp = params.incubation_temperature if hasattr(params, 'incubation_temperature') else 24.0
    original_moisture = params.moisture_content
    
    # Calculate growth potential for each combination
    growth_data = np.zeros((len(moistures), len(temperatures)))
    
    for i, moisture in enumerate(moistures):
        for j, temp in enumerate(temperatures):
            params.moisture_content = moisture
            params.incubation_temperature = temp
            growth_data[i, j] = calculate_growth_potential(params)
    
    # Restore original values
    params.incubation_temperature = original_temp
    params.moisture_content = original_moisture
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(growth_data, cmap='viridis', origin='lower', aspect='auto',
                  extent=[temperatures[0], temperatures[-1], moistures[0], moistures[-1]])
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Growth Potential (0-1)')
    
    # Mark current point
    ax.plot(original_temp, original_moisture, 'ro', markersize=10, 
            label=f'Current: {original_temp:.1f}°C, {original_moisture:.2f} moisture')
    
    # Add labels and title
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Moisture Content (ratio)')
    ax.set_title(f'Growth Potential Heatmap for {params.mushroom_species if hasattr(params, "mushroom_species") else "Mushroom"}')
    
    # Add a legend
    ax.legend(loc='upper left')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_electrical_properties_plot(params):
    """Create a plot showing electrical properties for different mycelium densities"""
    # Create a range of mycelium densities
    densities = np.linspace(0.1, 0.9, 20)
    
    # Store original density
    original_density = params.mycelium_density
    
    # Calculate electrical properties for each density
    conductivities = []
    impedances = []
    
    for density in densities:
        params.mycelium_density = density
        props = calculate_electrical_properties(params)
        conductivities.append(props['conductivity'])
        impedances.append(props['impedance'])
    
    # Restore original density
    params.mycelium_density = original_density
    
    # Get current properties
    current_props = calculate_electrical_properties(params)
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot conductivity on left axis
    line1, = ax1.plot(densities, conductivities, 'b-', linewidth=2, label='Conductivity')
    ax1.set_xlabel('Mycelium Density (ratio)')
    ax1.set_ylabel('Conductivity (S/m)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot impedance on right axis
    line2, = ax2.plot(densities, impedances, 'r-', linewidth=2, label='Impedance')
    ax2.set_ylabel('Impedance (Ω·m)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Mark current point on both curves
    ax1.plot(original_density, current_props['conductivity'], 'bo', markersize=8)
    ax2.plot(original_density, current_props['impedance'], 'ro', markersize=8)
    
    # Add title
    plt.title('Electrical Properties vs. Mycelium Density')
    
    # Add legend
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center')
    
    # Add annotation with current values
    plt.figtext(0.15, 0.02, 
                f"Current values at {original_density:.2f} density:\n"
                f"Conductivity: {current_props['conductivity']:.6f} S/m\n"
                f"Impedance: {current_props['impedance']:.2f} Ω·m\n"
                f"Dielectric Constant: {current_props['dielectric']:.1f}\n"
                f"Signal Propagation: {current_props['propagation_speed']:.3f}c",
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def create_dielectric_moisture_plot(params):
    """Create a plot showing how dielectric constant changes with moisture content"""
    # Create a range of moisture values
    moistures = np.linspace(0.1, 0.9, 20)
    
    # Store original moisture
    original_moisture = params.moisture_content
    
    # Calculate dielectric constant for each moisture level
    dielectrics = []
    propagation_speeds = []
    
    for moisture in moistures:
        params.moisture_content = moisture
        props = calculate_electrical_properties(params)
        dielectrics.append(props['dielectric'])
        propagation_speeds.append(props['propagation_speed'])
    
    # Restore original moisture
    params.moisture_content = original_moisture
    
    # Get current properties
    current_props = calculate_electrical_properties(params)
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot dielectric constant on left axis
    line1, = ax1.plot(moistures, dielectrics, 'g-', linewidth=2, label='Dielectric Constant')
    ax1.set_xlabel('Moisture Content (ratio)')
    ax1.set_ylabel('Dielectric Constant', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    
    # Plot propagation speed on right axis
    line2, = ax2.plot(moistures, propagation_speeds, 'm-', linewidth=2, label='Signal Propagation Speed')
    ax2.set_ylabel('Propagation Speed (relative to c)', color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    
    # Mark current point on both curves
    ax1.plot(original_moisture, current_props['dielectric'], 'go', markersize=8)
    ax2.plot(original_moisture, current_props['propagation_speed'], 'mo', markersize=8)
    
    # Add title
    plt.title('Dielectric Properties vs. Moisture Content')
    
    # Add legend
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='center right')
    
    # Add annotation with current values
    plt.figtext(0.15, 0.02, 
                f"Current values at {original_moisture:.2f} moisture:\n"
                f"Dielectric Constant: {current_props['dielectric']:.1f}\n"
                f"Signal Propagation: {current_props['propagation_speed']:.3f}c\n"
                f"Signal Delay: {(1/current_props['propagation_speed'])*3.33:.2f} ns/cm",
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def show_all_plots(params):
    """Create and display all plots"""
    ph_chart = create_ph_bar_chart(params)
    conductivity_plot = create_conductivity_moisture_plot(params)
    resistance_chart = create_resistance_comparison(params)
    growth_heatmap = create_growth_potential_heatmap(params)
    
    plt.tight_layout()
    plt.show() 