import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from simulation.substrate import calculate_ph, calculate_conductivity, calculate_contaminant_resistance

def create_ph_bar_chart(params):
    """Create a bar chart showing pH for different pasteurization methods"""
    methods = ['LIME', 'ASH', 'SOAP', 'BLEACH', 'HEAT']
    ph_values = []
    
    # Calculate pH for each method
    for method in methods:
        params_copy = params.__class__()
        for attr, value in vars(params).items():
            setattr(params_copy, attr, value)
        params_copy.pasteurization_method = method
        ph_values.append(calculate_ph(params_copy))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, ph_values, color='skyblue')
    
    # Highlight current method
    current_index = methods.index(params.pasteurization_method)
    bars[current_index].set_color('orange')
    
    # Add neutral pH line
    ax.axhline(y=7, color='r', linestyle='--', alpha=0.5)
    
    # Add labels and title
    ax.set_ylabel('pH Value')
    ax.set_title('pH by Pasteurization Method')
    ax.set_ylim(5, 13)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    return fig

def create_conductivity_moisture_plot(params):
    """Create a line plot showing conductivity vs moisture content"""
    moisture_range = np.linspace(0, 1, 20)
    conductivity_values = []
    
    # Calculate conductivity for different moisture levels
    for moisture in moisture_range:
        params_copy = params.__class__()
        for attr, value in vars(params).items():
            setattr(params_copy, attr, value)
        params_copy.moisture_content = moisture
        conductivity_values.append(calculate_conductivity(params_copy))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(moisture_range, conductivity_values, 'g-', linewidth=2)
    
    # Mark current moisture level
    current_conductivity = calculate_conductivity(params)
    ax.plot(params.moisture_content, current_conductivity, 'ro', markersize=8)
    
    # Add labels and title
    ax.set_xlabel('Moisture Content')
    ax.set_ylabel('Conductivity (S/m)')
    ax.set_title('Conductivity vs Moisture Content')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation for current point
    ax.annotate(f'Current: ({params.moisture_content:.2f}, {current_conductivity:.4f})',
                xy=(params.moisture_content, current_conductivity),
                xytext=(10, -20),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    return fig

def create_resistance_comparison(params):
    """Create a bar chart showing contaminant resistance for different methods"""
    methods = ['LIME', 'ASH', 'SOAP', 'BLEACH', 'HEAT']
    resistance_values = []
    
    # Calculate resistance for each method
    for method in methods:
        params_copy = params.__class__()
        for attr, value in vars(params).items():
            setattr(params_copy, attr, value)
        params_copy.pasteurization_method = method
        resistance_values.append(calculate_contaminant_resistance(params_copy))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, resistance_values, color='lightgreen')
    
    # Highlight current method
    current_index = methods.index(params.pasteurization_method)
    bars[current_index].set_color('orange')
    
    # Add labels and title
    ax.set_ylabel('Contaminant Resistance')
    ax.set_title('Contaminant Resistance by Pasteurization Method')
    ax.set_ylim(0, 1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Add resistance level indicators
    ax.axhspan(0.8, 1.0, alpha=0.2, color='green', label='Excellent')
    ax.axhspan(0.6, 0.8, alpha=0.2, color='lightgreen', label='Good')
    ax.axhspan(0.4, 0.6, alpha=0.2, color='yellow', label='Moderate')
    ax.axhspan(0, 0.4, alpha=0.2, color='red', label='Poor')
    ax.legend(loc='lower right')
    
    return fig

def create_growth_potential_heatmap(params):
    """Create a heatmap showing mycelium growth potential based on pH and moisture"""
    # Create a grid of pH and moisture values
    ph_range = np.linspace(5, 13, 100)
    moisture_range = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(ph_range, moisture_range)
    
    # Growth potential model (conceptual)
    # Optimal conditions: pH around 7.5, moisture around 0.7
    Z = -((X - 7.5)**2) / 5 - ((Y - 0.7)**2) / 0.1 + 1
    Z = np.clip(Z, 0, 1)  # Normalize to 0-1 range
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, Z, 20, cmap='viridis')
    
    # Mark current conditions
    current_ph = calculate_ph(params)
    ax.plot(current_ph, params.moisture_content, 'ro', markersize=8)
    
    # Add labels and title
    ax.set_xlabel('pH')
    ax.set_ylabel('Moisture Content')
    ax.set_title('Mycelium Growth Potential')
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Growth Potential')
    
    # Add annotation for current point
    current_growth = 1 - ((current_ph - 7.5)**2) / 5 - ((params.moisture_content - 0.7)**2) / 0.1
    current_growth = max(0, min(1, current_growth))  # Clip to 0-1
    
    growth_text = 'Excellent' if current_growth > 0.8 else 'Good' if current_growth > 0.6 else 'Moderate' if current_growth > 0.4 else 'Poor'
    
    ax.annotate(f'Current: pH={current_ph:.1f}, MC={params.moisture_content:.2f}\nGrowth Potential: {current_growth:.2f} ({growth_text})',
                xy=(current_ph, params.moisture_content),
                xytext=(20, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    return fig

def show_all_plots(params):
    """Create and display all plots"""
    ph_chart = create_ph_bar_chart(params)
    conductivity_plot = create_conductivity_moisture_plot(params)
    resistance_chart = create_resistance_comparison(params)
    growth_heatmap = create_growth_potential_heatmap(params)
    
    plt.tight_layout()
    plt.show() 