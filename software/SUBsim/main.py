import tkinter as tk
import customtkinter as ctk
from simulation.substrate import SubstrateParameters, calculate_conductivity, calculate_ph, calculate_contaminant_resistance
from visualization.gui import SimulationApp

def run_console_simulation():
    """Run the simulation in console mode"""
    # Initialize parameters
    params = SubstrateParameters()
    
    # Calculate key properties
    conductivity = calculate_conductivity(params)
    ph = calculate_ph(params)
    contaminant_resistance = calculate_contaminant_resistance(params)
    
    # Print results
    print("\n" + "="*50)
    print("MUSHROOM SUBSTRATE SIMULATION RESULTS")
    print("="*50)
    print(f"Substrate Composition:")
    print(f"  Lignin Content: {params.lignin_content:.2f}")
    print(f"  Cellulose Content: {params.cellulose_content:.2f}")
    print(f"  Moisture Content: {params.moisture_content:.2f}")
    print(f"\nPasteurization Method: {params.pasteurization_method}")
    print(f"  Duration: {params.pasteurization_duration} hours")
    print(f"  Concentration: {params.pasteurization_concentration:.2f}")
    print(f"\nMycelium Density: {params.mycelium_density:.2f}")
    print("\nResults:")
    print(f"  pH Level: {ph:.2f}")
    print(f"  Electrical Conductivity: {conductivity:.6f} S/m")
    print(f"  Contaminant Resistance: {contaminant_resistance:.2f}")
    print("="*50)
    
    return params, conductivity, ph, contaminant_resistance

def run_gui():
    """Run the simulation with GUI"""
    root = ctk.CTk()
    app = SimulationApp(root)
    root.mainloop()

if __name__ == "__main__":
    # Uncomment the line below to run in console mode
    # run_console_simulation()
    
    # Run with GUI
    run_gui()
