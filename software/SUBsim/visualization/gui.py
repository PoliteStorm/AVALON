import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from simulation.substrate import SubstrateParameters, calculate_conductivity, calculate_ph, calculate_contaminant_resistance, calculate_growth_potential, calculate_electrical_properties
from visualization.basic_plots import (create_ph_bar_chart, create_conductivity_moisture_plot, 
                                      create_resistance_comparison, create_growth_potential_heatmap,
                                      create_electrical_properties_plot, create_dielectric_moisture_plot)
from visualization.visualization_3d import (create_substrate_3d_model, create_cross_section_view, 
                                           create_combined_visualization, save_and_open_visualization,
                                           create_interactive_3d_model, save_and_open_interactive_visualization,
                                           create_interactive_cross_section, create_interactive_combined_visualization,
                                           create_multi_tab_visualization, save_and_open_multi_tab_visualization)
import webbrowser
import os
import tempfile
import subprocess

class SimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mushroom Substrate Simulation")
        self.root.geometry("1200x800")
        
        # Set the theme
        ctk.set_appearance_mode("dark")  # Options: "dark", "light"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"
        
        # Initialize parameters
        self.params = SubstrateParameters()
        
        # Create main layout
        self.create_layout()
        
        # Run initial simulation
        self.run_simulation()
    
    def create_layout(self):
        """Create the main application layout"""
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create sidebar for parameters
        self.create_sidebar()
        
        # Create content area for results and visualizations
        self.create_content_area()
    
    def create_sidebar(self):
        """Create the sidebar with parameter controls"""
        sidebar = ctk.CTkFrame(self.main_frame, width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Title
        title = ctk.CTkLabel(sidebar, text="Substrate Parameters", font=("Helvetica", 16, "bold"))
        title.pack(pady=10)
        
        # Create a scrollable frame for parameters
        scrollable_params = ctk.CTkScrollableFrame(sidebar)
        scrollable_params.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Substrate composition section
        self.create_substrate_frame()
        
        # Lignin content
        lignin_label = ctk.CTkLabel(scrollable_params, text="Lignin Content (ratio 0-1)")
        lignin_label.pack(anchor="w", pady=(10, 0))
        
        self.lignin_var = tk.DoubleVar(value=self.params.lignin_content)
        self.lignin_slider = ctk.CTkSlider(scrollable_params, from_=0, to=1, number_of_steps=100, variable=self.lignin_var)
        self.lignin_slider.pack(fill=tk.X, pady=5)
        
        self.lignin_value_label = ctk.CTkLabel(scrollable_params, text=f"{self.params.lignin_content:.2f} (≈{self.params.lignin_content*100:.1f}%)")
        self.lignin_value_label.pack()
        
        # Update value label when slider changes
        def update_lignin(value):
            self.params.lignin_content = float(value)
            self.lignin_value_label.configure(text=f"{value:.2f} (≈{value*100:.1f}%)")
        
        self.lignin_slider.configure(command=update_lignin)
        
        # Cellulose content
        cellulose_label = ctk.CTkLabel(scrollable_params, text="Cellulose Content (ratio 0-1)")
        cellulose_label.pack(anchor="w", pady=(10, 0))
        
        self.cellulose_var = tk.DoubleVar(value=self.params.cellulose_content)
        self.cellulose_slider = ctk.CTkSlider(scrollable_params, from_=0, to=1, number_of_steps=100, variable=self.cellulose_var)
        self.cellulose_slider.pack(fill=tk.X, pady=5)
        
        self.cellulose_value_label = ctk.CTkLabel(scrollable_params, text=f"{self.params.cellulose_content:.2f} (≈{self.params.cellulose_content*100:.1f}%)")
        self.cellulose_value_label.pack()
        
        # Update value label when slider changes
        def update_cellulose(value):
            self.params.cellulose_content = float(value)
            self.cellulose_value_label.configure(text=f"{value:.2f} (≈{value*100:.1f}%)")
        
        self.cellulose_slider.configure(command=update_cellulose)
        
        # Moisture content
        moisture_label = ctk.CTkLabel(scrollable_params, text="Moisture Content (ratio 0-1)")
        moisture_label.pack(anchor="w", pady=(10, 0))
        
        self.moisture_var = tk.DoubleVar(value=self.params.moisture_content)
        self.moisture_slider = ctk.CTkSlider(scrollable_params, from_=0, to=1, number_of_steps=100, variable=self.moisture_var)
        self.moisture_slider.pack(fill=tk.X, pady=5)
        
        self.moisture_value_label = ctk.CTkLabel(scrollable_params, text=f"{self.params.moisture_content:.2f} (≈{self.params.moisture_content*100:.1f}%)")
        self.moisture_value_label.pack()
        
        # Update value label when slider changes
        def update_moisture(value):
            self.params.moisture_content = float(value)
            self.moisture_value_label.configure(text=f"{value:.2f} (≈{value*100:.1f}%)")
        
        self.moisture_slider.configure(command=update_moisture)
        
        # pH level
        ph_label = ctk.CTkLabel(scrollable_params, text="Initial pH Level (4.5-8.0)")
        ph_label.pack(anchor="w", pady=(10, 0))
        
        self.ph_var = tk.DoubleVar(value=self.params.initial_ph if hasattr(self.params, 'initial_ph') else 6.0)
        self.ph_slider = ctk.CTkSlider(scrollable_params, from_=4.5, to=8.0, number_of_steps=35, variable=self.ph_var)
        self.ph_slider.pack(fill=tk.X, pady=5)
        
        self.ph_value_label = ctk.CTkLabel(scrollable_params, text=f"{self.ph_var.get():.1f}")
        self.ph_value_label.pack()
        
        # Update value label when slider changes
        def update_ph(value):
            self.params.initial_ph = float(value)
            self.ph_value_label.configure(text=f"{value:.1f}")
        
        self.ph_slider.configure(command=update_ph)
        
        # Nitrogen content
        nitrogen_label = ctk.CTkLabel(scrollable_params, text="Nitrogen Content (ratio 0-0.05)")
        nitrogen_label.pack(anchor="w", pady=(10, 0))
        
        self.nitrogen_var = tk.DoubleVar(value=self.params.nitrogen_content if hasattr(self.params, 'nitrogen_content') else 0.02)
        self.nitrogen_slider = ctk.CTkSlider(scrollable_params, from_=0, to=0.05, number_of_steps=50, variable=self.nitrogen_var)
        self.nitrogen_slider.pack(fill=tk.X, pady=5)
        
        self.nitrogen_value_label = ctk.CTkLabel(scrollable_params, text=f"{self.nitrogen_var.get():.3f} (≈{self.nitrogen_var.get()*100:.1f}%)")
        self.nitrogen_value_label.pack()
        
        # Update value label when slider changes
        def update_nitrogen(value):
            self.params.nitrogen_content = float(value)
            self.nitrogen_value_label.configure(text=f"{value:.3f} (≈{value*100:.1f}%)")
        
        self.nitrogen_slider.configure(command=update_nitrogen)
        
        # Pasteurization section
        pasteurization_frame = ctk.CTkFrame(scrollable_params)
        pasteurization_frame.pack(fill=tk.X, padx=10, pady=10)
        
        pasteurization_label = ctk.CTkLabel(pasteurization_frame, text="Pasteurization Method", font=("Helvetica", 14))
        pasteurization_label.pack(anchor="w", pady=5)
        
        # Method selection
        self.method_var = tk.StringVar(value=self.params.pasteurization_method)
        methods = ["HEAT", "LIME", "HYDROGEN_PEROXIDE", "STEAM", "BLEACH", "NONE"]
        
        for method in methods:
            method_radio = ctk.CTkRadioButton(pasteurization_frame, text=method, variable=self.method_var, value=method)
            method_radio.pack(anchor="w", pady=2)
        
        # Duration
        duration_label = ctk.CTkLabel(pasteurization_frame, text="Duration (hours)")
        duration_label.pack(anchor="w", pady=(10, 0))
        
        self.duration_var = tk.DoubleVar(value=self.params.pasteurization_duration)
        self.duration_slider = ctk.CTkSlider(pasteurization_frame, from_=0, to=24, number_of_steps=48, variable=self.duration_var)
        self.duration_slider.pack(fill=tk.X, pady=5)
        
        self.duration_value_label = ctk.CTkLabel(pasteurization_frame, text=f"{self.params.pasteurization_duration:.1f} hours")
        self.duration_value_label.pack()
        
        # Update value label when slider changes
        def update_duration(value):
            self.params.pasteurization_duration = float(value)
            self.duration_value_label.configure(text=f"{value:.1f} hours")
        
        self.duration_slider.configure(command=update_duration)
        
        # Concentration
        concentration_label = ctk.CTkLabel(pasteurization_frame, text="Concentration (ratio 0-1)")
        concentration_label.pack(anchor="w", pady=(10, 0))
        
        self.concentration_var = tk.DoubleVar(value=self.params.pasteurization_concentration)
        self.concentration_slider = ctk.CTkSlider(pasteurization_frame, from_=0, to=1, number_of_steps=100, variable=self.concentration_var)
        self.concentration_slider.pack(fill=tk.X, pady=5)
        
        self.concentration_value_label = ctk.CTkLabel(pasteurization_frame, text=f"{self.params.pasteurization_concentration:.2f} (≈{self.params.pasteurization_concentration*100:.1f}%)")
        self.concentration_value_label.pack()
        
        # Update value label when slider changes
        def update_concentration(value):
            self.params.pasteurization_concentration = float(value)
            self.concentration_value_label.configure(text=f"{value:.2f} (≈{value*100:.1f}%)")
        
        self.concentration_slider.configure(command=update_concentration)
        
        # Temperature (for heat pasteurization)
        temperature_label = ctk.CTkLabel(pasteurization_frame, text="Temperature (°C) - for heat methods")
        temperature_label.pack(anchor="w", pady=(10, 0))
        
        self.temperature_var = tk.DoubleVar(value=self.params.pasteurization_temperature if hasattr(self.params, 'pasteurization_temperature') else 65.0)
        self.temperature_slider = ctk.CTkSlider(pasteurization_frame, from_=50, to=100, number_of_steps=50, variable=self.temperature_var)
        self.temperature_slider.pack(fill=tk.X, pady=5)
        
        self.temperature_value_label = ctk.CTkLabel(pasteurization_frame, text=f"{self.temperature_var.get():.1f}°C ({(self.temperature_var.get()*9/5)+32:.1f}°F)")
        self.temperature_value_label.pack()
        
        # Update value label when slider changes
        def update_temperature(value):
            self.params.pasteurization_temperature = float(value)
            self.temperature_value_label.configure(text=f"{value:.1f}°C ({(value*9/5)+32:.1f}°F)")
        
        self.temperature_slider.configure(command=update_temperature)
        
        # Mycelium section
        mycelium_frame = ctk.CTkFrame(scrollable_params)
        mycelium_frame.pack(fill=tk.X, padx=10, pady=10)
        
        mycelium_label = ctk.CTkLabel(mycelium_frame, text="Mycelium Properties", font=("Helvetica", 14))
        mycelium_label.pack(anchor="w", pady=5)
        
        # Mushroom species selection
        species_label = ctk.CTkLabel(mycelium_frame, text="Mushroom Species")
        species_label.pack(anchor="w", pady=(5, 0))
        
        self.species_var = tk.StringVar(value=self.params.mushroom_species if hasattr(self.params, 'mushroom_species') else "OYSTER")
        species_types = ["OYSTER", "SHIITAKE", "LIONS_MANE", "REISHI", "BUTTON", "ENOKI", "MAITAKE", "CORDYCEPS"]
        self.species_menu = ctk.CTkOptionMenu(
            mycelium_frame, 
            values=species_types,
            variable=self.species_var,
            command=self.update_species_defaults
        )
        self.species_menu.pack(fill=tk.X, pady=5)
        
        # Density
        density_label = ctk.CTkLabel(mycelium_frame, text="Mycelium Density (ratio 0-1)")
        density_label.pack(anchor="w")
        
        self.density_var = tk.DoubleVar(value=self.params.mycelium_density)
        self.density_slider = ctk.CTkSlider(mycelium_frame, from_=0, to=1, number_of_steps=100, variable=self.density_var)
        self.density_slider.pack(fill=tk.X, pady=5)
        
        self.density_value_label = ctk.CTkLabel(mycelium_frame, text=f"{self.params.mycelium_density:.2f} (≈{self.params.mycelium_density*100:.1f}%)")
        self.density_value_label.pack()
        
        # Update value label when slider changes
        def update_density(value):
            self.params.mycelium_density = float(value)
            self.density_value_label.configure(text=f"{value:.2f} (≈{value*100:.1f}%)")
        
        self.density_slider.configure(command=update_density)
        
        # Spawn rate
        spawn_label = ctk.CTkLabel(mycelium_frame, text="Spawn Rate (ratio 0-0.5)")
        spawn_label.pack(anchor="w", pady=(10, 0))
        
        self.spawn_var = tk.DoubleVar(value=self.params.spawn_rate if hasattr(self.params, 'spawn_rate') else 0.1)
        self.spawn_slider = ctk.CTkSlider(mycelium_frame, from_=0, to=0.5, number_of_steps=50, variable=self.spawn_var)
        self.spawn_slider.pack(fill=tk.X, pady=5)
        
        self.spawn_value_label = ctk.CTkLabel(mycelium_frame, text=f"{self.spawn_var.get():.2f} (≈{self.spawn_var.get()*100:.1f}%)")
        self.spawn_value_label.pack()
        
        # Update value label when slider changes
        def update_spawn(value):
            self.params.spawn_rate = float(value)
            self.spawn_value_label.configure(text=f"{value:.2f} (≈{value*100:.1f}%)")
        
        self.spawn_slider.configure(command=update_spawn)
        
        # Environmental conditions section
        environment_frame = ctk.CTkFrame(scrollable_params)
        environment_frame.pack(fill=tk.X, padx=10, pady=10)
        
        environment_label = ctk.CTkLabel(environment_frame, text="Environmental Conditions", font=("Helvetica", 14))
        environment_label.pack(anchor="w", pady=5)
        
        # Temperature
        env_temp_label = ctk.CTkLabel(environment_frame, text="Incubation Temperature (°C)")
        env_temp_label.pack(anchor="w", pady=(5, 0))
        
        self.env_temp_var = tk.DoubleVar(value=self.params.incubation_temperature if hasattr(self.params, 'incubation_temperature') else 24.0)
        self.env_temp_slider = ctk.CTkSlider(environment_frame, from_=15, to=30, number_of_steps=30, variable=self.env_temp_var)
        self.env_temp_slider.pack(fill=tk.X, pady=5)
        
        self.env_temp_value_label = ctk.CTkLabel(environment_frame, text=f"{self.env_temp_var.get():.1f}°C ({(self.env_temp_var.get()*9/5)+32:.1f}°F)")
        self.env_temp_value_label.pack()
        
        # Update value label when slider changes
        def update_env_temp(value):
            self.params.incubation_temperature = float(value)
            self.env_temp_value_label.configure(text=f"{value:.1f}°C ({(value*9/5)+32:.1f}°F)")
        
        self.env_temp_slider.configure(command=update_env_temp)
        
        # Humidity
        humidity_label = ctk.CTkLabel(environment_frame, text="Relative Humidity (%)")
        humidity_label.pack(anchor="w", pady=(10, 0))
        
        self.humidity_var = tk.DoubleVar(value=self.params.humidity if hasattr(self.params, 'humidity') else 85.0)
        self.humidity_slider = ctk.CTkSlider(environment_frame, from_=50, to=100, number_of_steps=50, variable=self.humidity_var)
        self.humidity_slider.pack(fill=tk.X, pady=5)
        
        self.humidity_value_label = ctk.CTkLabel(environment_frame, text=f"{self.humidity_var.get():.1f}%")
        self.humidity_value_label.pack()
        
        # Update value label when slider changes
        def update_humidity(value):
            self.params.humidity = float(value)
            self.humidity_value_label.configure(text=f"{value:.1f}%")
        
        self.humidity_slider.configure(command=update_humidity)
        
        # Buttons
        button_frame = ctk.CTkFrame(sidebar)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        run_button = ctk.CTkButton(button_frame, text="Run Simulation", command=self.run_simulation)
        run_button.pack(fill=tk.X, pady=5)
        
        reset_button = ctk.CTkButton(button_frame, text="Reset Parameters", command=self.reset_parameters)
        reset_button.pack(fill=tk.X, pady=5)
        
        # Preview label
        self.preview_label = ctk.CTkLabel(sidebar, text="Adjust parameters and click 'Run Simulation'", wraplength=280)
        self.preview_label.pack(pady=10)

    def create_substrate_frame(self):
        """Create the frame for substrate parameters"""
        substrate_frame = ctk.CTkFrame(self.main_frame)
        substrate_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Title for the section
        substrate_label = ctk.CTkLabel(substrate_frame, text="Substrate Parameters", font=ctk.CTkFont(size=16, weight="bold"))
        substrate_label.pack(pady=5)
        
        # Substrate type selection
        substrate_type_frame = ctk.CTkFrame(substrate_frame)
        substrate_type_frame.pack(fill=tk.X, padx=10, pady=5)
        
        substrate_type_label = ctk.CTkLabel(substrate_type_frame, text="Substrate Type:", width=120)
        substrate_type_label.pack(side=tk.LEFT, padx=5)
        
        substrate_types = ["HARDWOOD", "SOFTWOOD", "STRAW", "COFFEE_GROUNDS", "COMPOST", "COCO_COIR", "OYSTER_SHELL_MIX"]
        self.substrate_type_var = tk.StringVar(value=self.params.substrate_type)
        substrate_type_menu = ctk.CTkOptionMenu(
            substrate_type_frame, 
            variable=self.substrate_type_var,
            values=substrate_types,
            command=self.update_substrate_type
        )
        substrate_type_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Calcium supplementation
        calcium_frame = ctk.CTkFrame(substrate_frame)
        calcium_frame.pack(fill=tk.X, padx=10, pady=5)
        
        calcium_label = ctk.CTkLabel(calcium_frame, text="Calcium Supplement:", width=120)
        calcium_label.pack(side=tk.LEFT, padx=5)
        
        calcium_types = ["NONE", "OYSTER_SHELL", "LIMESTONE", "GYPSUM"]
        self.calcium_type_var = tk.StringVar(value=self.params.calcium_supplement)
        calcium_type_menu = ctk.CTkOptionMenu(
            calcium_frame, 
            variable=self.calcium_type_var,
            values=calcium_types,
            command=self.update_calcium_type
        )
        calcium_type_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Calcium content slider
        calcium_content_frame = ctk.CTkFrame(substrate_frame)
        calcium_content_frame.pack(fill=tk.X, padx=10, pady=5)
        
        calcium_content_label = ctk.CTkLabel(calcium_content_frame, text="Calcium Content:", width=120)
        calcium_content_label.pack(side=tk.LEFT, padx=5)
        
        self.calcium_content_var = tk.DoubleVar(value=self.params.calcium_content)
        calcium_content_slider = ctk.CTkSlider(
            calcium_content_frame, 
            from_=0.0, 
            to=0.1, 
            variable=self.calcium_content_var,
            command=self.update_params
        )
        calcium_content_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        calcium_content_value = ctk.CTkLabel(calcium_content_frame, text=f"{self.params.calcium_content:.2f}", width=40)
        calcium_content_value.pack(side=tk.LEFT, padx=5)
        
        # Calcination checkbox
        calcination_frame = ctk.CTkFrame(substrate_frame)
        calcination_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.calcination_var = tk.BooleanVar(value=self.params.calcium_calcination)
        calcination_checkbox = ctk.CTkCheckBox(
            calcination_frame, 
            text="Calcinated Calcium",
            variable=self.calcination_var,
            command=self.update_params
        )
        calcination_checkbox.pack(padx=5, pady=5)
        
        # Add tooltip explaining calcination
        self.add_tooltip(calcination_checkbox, 
                        "Calcination is the process of heating calcium sources to high temperatures (620°C+)\n"
                        "to convert calcium carbonate (CaCO₃) to calcium oxide (CaO), improving bioavailability.")

    def create_content_area(self):
        """Create the main content area with tabs for different visualizations"""
        content = ctk.CTkFrame(self.main_frame)
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabview
        self.tabview = ctk.CTkTabview(content)
        self.tabview.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.tab_results = self.tabview.add("Results")
        self.tab_charts = self.tabview.add("Charts")
        self.tab_3d = self.tabview.add("3D View")
        
        # Setup results tab
        self.setup_results_tab()
        
        # Setup charts tab
        self.setup_charts_tab()
        
        # Setup 3D visualization tab
        self.setup_3d_tab()
    
    def setup_results_tab(self):
        """Setup the results tab with metrics and summary"""
        # Create a scrollable frame for the entire tab
        scrollable_frame = ctk.CTkScrollableFrame(self.tab_results)
        scrollable_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a frame for the metrics
        metrics_frame = ctk.CTkFrame(scrollable_frame)
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create a grid of metrics
        # pH
        ph_frame = ctk.CTkFrame(metrics_frame)
        ph_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ph_label = ctk.CTkLabel(ph_frame, text="pH Level", font=("Helvetica", 14, "bold"))
        ph_label.pack(pady=5)
        
        self.ph_value = ctk.CTkLabel(ph_frame, text="--", font=("Helvetica", 24))
        self.ph_value.pack(pady=10)
        
        ph_unit = ctk.CTkLabel(ph_frame, text="Scale: 0-14 (7 is neutral)", font=("Helvetica", 10))
        ph_unit.pack(pady=2)
        
        # Conductivity
        conductivity_frame = ctk.CTkFrame(metrics_frame)
        conductivity_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        conductivity_label = ctk.CTkLabel(conductivity_frame, text="Conductivity", font=("Helvetica", 14, "bold"))
        conductivity_label.pack(pady=5)
        
        self.conductivity_value = ctk.CTkLabel(conductivity_frame, text="--", font=("Helvetica", 24))
        self.conductivity_value.pack(pady=10)
        
        conductivity_unit = ctk.CTkLabel(conductivity_frame, text="Siemens/meter (S/m)", font=("Helvetica", 10))
        conductivity_unit.pack(pady=2)
        
        # Contaminant Resistance
        resistance_frame = ctk.CTkFrame(metrics_frame)
        resistance_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        
        resistance_label = ctk.CTkLabel(resistance_frame, text="Contaminant Resistance", font=("Helvetica", 14, "bold"))
        resistance_label.pack(pady=5)
        
        self.resistance_value = ctk.CTkLabel(resistance_frame, text="--", font=("Helvetica", 24))
        self.resistance_value.pack(pady=10)
        
        resistance_unit = ctk.CTkLabel(resistance_frame, text="Scale: 0-1 (higher is better)", font=("Helvetica", 10))
        resistance_unit.pack(pady=2)
        
        # Configure grid
        metrics_frame.grid_columnconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(1, weight=1)
        metrics_frame.grid_columnconfigure(2, weight=1)
        
        # Create a frame for the summary
        summary_frame = ctk.CTkFrame(scrollable_frame)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        summary_label = ctk.CTkLabel(summary_frame, text="Simulation Summary", font=("Helvetica", 16, "bold"))
        summary_label.pack(pady=10)
        
        # Create a text widget for the summary
        self.summary_text = ctk.CTkTextbox(summary_frame, wrap="word", font=("Helvetica", 12), height=300)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add initial message
        self.summary_text.insert("1.0", "Click 'Run Simulation' to generate results.")
    
    def setup_charts_tab(self):
        """Setup the charts tab with various visualizations"""
        # Create a notebook for different charts
        charts_notebook = ctk.CTkTabview(self.tab_charts)
        charts_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add tabs for different charts
        charts_notebook.add("pH")
        charts_notebook.add("Conductivity")
        charts_notebook.add("Resistance")
        charts_notebook.add("Growth")
        charts_notebook.add("Electrical")
        charts_notebook.add("Dielectric")
        
        # Create frames for each chart
        self.ph_chart_frame = ctk.CTkFrame(charts_notebook.tab("pH"))
        self.ph_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.conductivity_chart_frame = ctk.CTkFrame(charts_notebook.tab("Conductivity"))
        self.conductivity_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.resistance_chart_frame = ctk.CTkFrame(charts_notebook.tab("Resistance"))
        self.resistance_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.growth_chart_frame = ctk.CTkFrame(charts_notebook.tab("Growth"))
        self.growth_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.electrical_chart_frame = ctk.CTkFrame(charts_notebook.tab("Electrical"))
        self.electrical_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.dielectric_chart_frame = ctk.CTkFrame(charts_notebook.tab("Dielectric"))
        self.dielectric_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add placeholder text to chart frames
        for frame, title in [(self.ph_chart_frame, "pH Levels"), 
                            (self.conductivity_chart_frame, "Conductivity"),
                            (self.resistance_chart_frame, "Resistance"),
                            (self.growth_chart_frame, "Growth Potential"),
                            (self.electrical_chart_frame, "Electrical Properties"),
                            (self.dielectric_chart_frame, "Dielectric Properties")]:
            placeholder = ctk.CTkLabel(frame, text=f"Run simulation to generate {title} chart")
            placeholder.pack(expand=True, pady=50)
    
    def setup_3d_tab(self):
        """Setup the 3D visualization tab"""
        # Create a scrollable frame for the 3D visualization
        scrollable_frame = ctk.CTkScrollableFrame(self.tab_3d)
        scrollable_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a frame for the 3D visualization
        self.visualization_frame = ctk.CTkFrame(scrollable_frame)
        self.visualization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add a label
        title_label = ctk.CTkLabel(
            self.visualization_frame, 
            text="3D Substrate Visualization", 
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Add description
        description = ctk.CTkLabel(
            self.visualization_frame,
            text="Click the buttons below to view interactive 3D visualizations of the substrate.",
            wraplength=600
        )
        description.pack(pady=10)
        
        # Add buttons for different visualizations
        button_frame = ctk.CTkFrame(self.visualization_frame)
        button_frame.pack(pady=20)
        
        view_3d_button = ctk.CTkButton(
            button_frame,
            text="View 3D Model",
            command=self.show_3d_model
        )
        view_3d_button.pack(side=tk.LEFT, padx=10)
        
        view_cross_section_button = ctk.CTkButton(
            button_frame,
            text="View Cross-Section",
            command=self.show_cross_section
        )
        view_cross_section_button.pack(side=tk.LEFT, padx=10)
        
        view_combined_button = ctk.CTkButton(
            button_frame,
            text="View Combined Visualization",
            command=self.show_combined_visualization
        )
        view_combined_button.pack(side=tk.LEFT, padx=10)
        
        # Add a frame for the preview image
        self.preview_frame = ctk.CTkFrame(self.visualization_frame)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Add a label for the preview
        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="Run the simulation to generate visualizations",
            font=("Helvetica", 14)
        )
        self.preview_label.pack(pady=50)
    
    def update_method(self):
        """Update the pasteurization method"""
        self.params.pasteurization_method = self.method_var.get()
    
    def run_simulation(self):
        """Run the simulation with current parameters"""
        try:
            # Update parameters from UI
            self.params.lignin_content = self.lignin_var.get()
            self.params.cellulose_content = self.cellulose_var.get()
            self.params.moisture_content = self.moisture_var.get()
            self.params.pasteurization_method = self.method_var.get()
            self.params.pasteurization_duration = self.duration_var.get()
            self.params.pasteurization_concentration = self.concentration_var.get()
            self.params.mycelium_density = self.density_var.get()
            
            # Update new parameters
            if hasattr(self, 'substrate_type_var'):
                self.params.substrate_type = self.substrate_type_var.get()
            if hasattr(self, 'ph_var'):
                self.params.initial_ph = self.ph_var.get()
            if hasattr(self, 'nitrogen_var'):
                self.params.nitrogen_content = self.nitrogen_var.get()
            if hasattr(self, 'temperature_var'):
                self.params.pasteurization_temperature = self.temperature_var.get()
            if hasattr(self, 'species_var'):
                self.params.mushroom_species = self.species_var.get()
            if hasattr(self, 'spawn_var'):
                self.params.spawn_rate = self.spawn_var.get()
            if hasattr(self, 'env_temp_var'):
                self.params.incubation_temperature = self.env_temp_var.get()
            if hasattr(self, 'humidity_var'):
                self.params.humidity = self.humidity_var.get()
            
            # Show a "calculating" message
            self.preview_label.configure(text="Calculating simulation results...")
            self.root.update()  # Force update the UI
            
            # Calculate results
            conductivity = calculate_conductivity(self.params)
            ph = calculate_ph(self.params)
            contaminant_resistance = calculate_contaminant_resistance(self.params)
            
            # Update display
            self.update_results_display(conductivity, ph, contaminant_resistance)
            self.update_charts()
            
            # Show success message
            self.preview_label.configure(text="Simulation completed successfully.\nView results in the tabs above.")
            
            print("Simulation run successfully")
        except Exception as e:
            self.preview_label.configure(text=f"Error running simulation: {str(e)}")
            print(f"Error running simulation: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_results_display(self, conductivity, ph, contaminant_resistance):
        """Update the results display with calculated values and detailed information"""
        # Update metric values
        self.ph_value.configure(text=f"{ph:.2f}")
        self.conductivity_value.configure(text=f"{conductivity:.6f} S/m")
        self.resistance_value.configure(text=f"{contaminant_resistance:.2f}")
        
        # Calculate additional properties
        growth_potential = calculate_growth_potential(self.params)
        electrical_props = calculate_electrical_properties(self.params)
        
        # Calculate substrate weight based on standard 5kg block
        standard_weight = 5.0  # kg
        dry_weight = standard_weight * (1 - self.params.moisture_content)
        wet_weight = standard_weight
        
        # Create a detailed summary with units and explanations
        summary = f"""
SUBSTRATE COMPOSITION DETAILS:
• Base Substrate Type: {self.params.substrate_type if hasattr(self.params, 'substrate_type') else "Mixed"}
• Lignin Content: {self.params.lignin_content:.2f} (ratio - {self.params.lignin_content*100:.1f}%)
• Cellulose Content: {self.params.cellulose_content:.2f} (ratio - {self.params.cellulose_content*100:.1f}%)
• Hemicellulose Content: {(1-self.params.lignin_content-self.params.cellulose_content):.2f} (ratio - {(1-self.params.lignin_content-self.params.cellulose_content)*100:.1f}%)
• Moisture Content: {self.params.moisture_content:.2f} (ratio - {self.params.moisture_content*100:.1f}%)
• Nitrogen Content: {self.params.nitrogen_content if hasattr(self.params, 'nitrogen_content') else 0.02:.3f} (ratio - {(self.params.nitrogen_content if hasattr(self.params, 'nitrogen_content') else 0.02)*100:.1f}%)
• C:N Ratio: {(0.5/(self.params.nitrogen_content if hasattr(self.params, 'nitrogen_content') else 0.02)):.1f}:1

SUBSTRATE WEIGHT CALCULATIONS:
• Standard Block Weight: {wet_weight:.2f} kg
• Dry Matter: {dry_weight:.2f} kg ({(1-self.params.moisture_content)*100:.1f}%)
• Water Content: {wet_weight - dry_weight:.2f} kg ({self.params.moisture_content*100:.1f}%)

PASTEURIZATION METHOD:
• Method: {self.params.pasteurization_method}
• Duration: {self.params.pasteurization_duration:.1f} hours
• Temperature: {self.params.pasteurization_temperature if hasattr(self.params, 'pasteurization_temperature') else 65.0:.1f}°C ({((self.params.pasteurization_temperature if hasattr(self.params, 'pasteurization_temperature') else 65.0)*9/5)+32:.1f}°F)
• Concentration: {self.params.pasteurization_concentration:.2f} (ratio - {self.params.pasteurization_concentration*100:.1f}%)

MYCELIUM PROPERTIES:
• Species: {self.params.mushroom_species if hasattr(self.params, 'mushroom_species') else "Oyster"}
• Mycelium Density: {self.params.mycelium_density:.2f} (ratio - {self.params.mycelium_density*100:.1f}%)
• Spawn Rate: {self.params.spawn_rate if hasattr(self.params, 'spawn_rate') else 0.1:.2f} (ratio - {(self.params.spawn_rate if hasattr(self.params, 'spawn_rate') else 0.1)*100:.1f}%)

ENVIRONMENTAL CONDITIONS:
• Incubation Temperature: {self.params.incubation_temperature if hasattr(self.params, 'incubation_temperature') else 24.0:.1f}°C ({((self.params.incubation_temperature if hasattr(self.params, 'incubation_temperature') else 24.0)*9/5)+32:.1f}°F)
• Relative Humidity: {self.params.humidity if hasattr(self.params, 'humidity') else 85.0:.1f}%

SIMULATION RESULTS:
• pH Level: {ph:.2f} (optimal range for most mushrooms: 5.5-6.5)
• Electrical Conductivity: {conductivity:.6f} S/m
• Contaminant Resistance: {contaminant_resistance:.2f} (scale 0-1)
• Growth Potential: {growth_potential:.2f} (scale 0-1)

ELECTRICAL PROPERTIES (for unconventional computing):
• Electrical Conductivity: {electrical_props['conductivity']:.6f} S/m
• Resistivity: {electrical_props['resistivity']:.2f} Ω·m
• Impedance (est.): {electrical_props['impedance']:.2f} Ω·m
• Dielectric Constant: {electrical_props['dielectric']:.1f}
• Signal Propagation Speed: {electrical_props['propagation_speed']:.3f}c

SUBSTRATE PREPARATION RECIPE:
"""

        # Add specific preparation instructions based on pasteurization method
        if self.params.pasteurization_method == "HEAT":
            summary += f"• Heat water to {self.params.pasteurization_temperature if hasattr(self.params, 'pasteurization_temperature') else 65.0:.1f}°C ({((self.params.pasteurization_temperature if hasattr(self.params, 'pasteurization_temperature') else 65.0)*9/5)+32:.1f}°F)\n"
            summary += f"• Submerge substrate for {self.params.pasteurization_duration:.1f} hours\n"
            summary += f"• Drain and cool to {self.params.incubation_temperature if hasattr(self.params, 'incubation_temperature') else 24.0:.1f}°C before inoculation\n"
        
        elif self.params.pasteurization_method == "LIME":
            lime_amount = self.params.pasteurization_concentration * 0.01 * wet_weight
            summary += f"• Hydrated Lime Required: {lime_amount:.2f} kg for {wet_weight:.2f} kg substrate\n"
            summary += f"• Water for Soaking: {wet_weight * 2:.2f} liters\n"
            summary += f"• Soak Time: {self.params.pasteurization_duration:.1f} hours\n"
            summary += f"• pH After Treatment: ~12.0 (will decrease during incubation)\n"
        
        elif self.params.pasteurization_method == "HYDROGEN_PEROXIDE":
            h2o2_amount = self.params.pasteurization_concentration * wet_weight
            summary += f"• 3% Hydrogen Peroxide Required: {h2o2_amount:.2f} liters for {wet_weight:.2f} kg substrate\n"
            summary += f"• Water for Dilution: {wet_weight * 2:.2f} liters\n"
            summary += f"• Soak Time: {self.params.pasteurization_duration:.1f} hours\n"
        
        elif self.params.pasteurization_method == "STEAM":
            summary += f"• Steam at 100°C (212°F) for {self.params.pasteurization_duration:.1f} hours\n"
            summary += f"• Ensure even steam distribution throughout substrate\n"
            summary += f"• Cool to {self.params.incubation_temperature if hasattr(self.params, 'incubation_temperature') else 24.0:.1f}°C before inoculation\n"
        
        elif self.params.pasteurization_method == "BLEACH":
            bleach_amount = self.params.pasteurization_concentration * 0.01 * wet_weight
            summary += f"• Bleach Solution (5%): {bleach_amount:.2f} liters for {wet_weight:.2f} kg substrate\n"
            summary += f"• Water for Soaking: {wet_weight * 2:.2f} liters\n"
            summary += f"• Soak Time: {self.params.pasteurization_duration:.1f} hours\n"
            summary += f"• Rinse thoroughly after treatment to remove bleach residue\n"
        
        # Add notes for unconventional computing
        summary += f"""
NOTES FOR UNCONVENTIONAL COMPUTING:
• Higher moisture content increases conductivity
• Mycelium networks form natural computing structures
• Optimal conductivity for computing: 0.001-0.01 S/m
• Lignin content affects signal propagation characteristics
• pH influences ion mobility and electrical properties
• Estimated network complexity: {int(self.params.mycelium_density * 1000)} nodes per cm³
• Signal transmission time across 1cm: {(1/electrical_props['propagation_speed'])*3.33:.2f} ns
"""
        
        # Update the text widget
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", summary)
    
    def update_charts(self):
        """Update all charts with current parameters"""
        try:
            # Clear existing charts
            for frame in [self.ph_chart_frame, self.conductivity_chart_frame, 
                          self.resistance_chart_frame, self.growth_chart_frame,
                          self.electrical_chart_frame, self.dielectric_chart_frame]:
                for widget in frame.winfo_children():
                    widget.destroy()
            
            # Create pH chart
            ph_fig = create_ph_bar_chart(self.params)
            ph_canvas = FigureCanvasTkAgg(ph_fig, master=self.ph_chart_frame)
            ph_canvas.draw()
            ph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Create conductivity chart
            conductivity_fig = create_conductivity_moisture_plot(self.params)
            conductivity_canvas = FigureCanvasTkAgg(conductivity_fig, master=self.conductivity_chart_frame)
            conductivity_canvas.draw()
            conductivity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Create resistance chart
            resistance_fig = create_resistance_comparison(self.params)
            resistance_canvas = FigureCanvasTkAgg(resistance_fig, master=self.resistance_chart_frame)
            resistance_canvas.draw()
            resistance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Create growth potential chart
            growth_fig = create_growth_potential_heatmap(self.params)
            growth_canvas = FigureCanvasTkAgg(growth_fig, master=self.growth_chart_frame)
            growth_canvas.draw()
            growth_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Create electrical properties chart
            electrical_fig = create_electrical_properties_plot(self.params)
            electrical_canvas = FigureCanvasTkAgg(electrical_fig, master=self.electrical_chart_frame)
            electrical_canvas.draw()
            electrical_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Create dielectric properties chart
            dielectric_fig = create_dielectric_moisture_plot(self.params)
            dielectric_canvas = FigureCanvasTkAgg(dielectric_fig, master=self.dielectric_chart_frame)
            dielectric_canvas.draw()
            dielectric_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            print(f"Error updating charts: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def show_3d_model(self):
        """Show a 3D model of the substrate"""
        try:
            # Update the preview label to show we're working
            self.preview_label.configure(text="Generating 3D model... please wait")
            self.root.update()  # Force update the UI
            
            # Create the interactive visualization
            fig = create_interactive_3d_model(self.params)
            
            # Save and try to open it
            file_path = save_and_open_interactive_visualization(fig, self.params, "substrate_3d_model.html")
            
            if file_path and os.path.exists(file_path):
                # Update the preview label
                self.preview_label.configure(text=f"3D model saved to:\n{file_path}\n\nIf it doesn't open automatically, please open this file in your browser.")
                
                # Add a button to manually open the file
                if hasattr(self, 'open_file_button'):
                    self.open_file_button.destroy()
                
                self.open_file_button = ctk.CTkButton(
                    self.preview_frame,
                    text="Open in Browser",
                    command=lambda: self.open_file(file_path)
                )
                self.open_file_button.pack(pady=10)
            else:
                self.preview_label.configure(text="Error: Could not create 3D model")
        except Exception as e:
            self.preview_label.configure(text=f"Error creating 3D model: {str(e)}")
            print(f"Error creating 3D model: {str(e)}")
            import traceback
            traceback.print_exc()

    def show_cross_section(self):
        """Show the cross-section visualization"""
        try:
            # Update the preview label to show we're working
            self.preview_label.configure(text="Generating cross-section view... please wait")
            self.root.update()  # Force update the UI
            
            # Create the interactive visualization
            fig = create_interactive_cross_section(self.params)
            
            # Save and try to open it
            file_path = save_and_open_interactive_visualization(fig, self.params, "substrate_cross_section.html")
            
            if file_path and os.path.exists(file_path):
                # Update the preview label
                self.preview_label.configure(text=f"Cross-section view saved to:\n{file_path}\n\nIf it doesn't open automatically, please open this file in your browser.")
                
                # Add a button to manually open the file
                if hasattr(self, 'open_file_button'):
                    self.open_file_button.destroy()
                
                self.open_file_button = ctk.CTkButton(
                    self.preview_frame,
                    text="Open in Browser",
                    command=lambda: self.open_file(file_path)
                )
                self.open_file_button.pack(pady=10)
            else:
                self.preview_label.configure(text="Error: Could not create cross-section view")
        except Exception as e:
            self.preview_label.configure(text=f"Error creating cross-section visualization:\n{str(e)}")
            print(f"Error creating cross-section view: {str(e)}")
            import traceback
            traceback.print_exc()

    def show_combined_visualization(self):
        """Show the combined visualization"""
        try:
            # Update the preview label to show we're working
            self.preview_label.configure(text="Generating combined visualization... please wait")
            self.root.update()  # Force update the UI
            
            # Create the interactive visualization
            fig = create_interactive_combined_visualization(self.params)
            
            # Save and try to open it
            file_path = save_and_open_interactive_visualization(fig, self.params, "substrate_combined_view.html")
            
            if file_path and os.path.exists(file_path):
                # Update the preview label
                self.preview_label.configure(text=f"Combined visualization saved to:\n{file_path}\n\nIf it doesn't open automatically, please open this file in your browser.")
                
                # Add a button to manually open the file
                if hasattr(self, 'open_file_button'):
                    self.open_file_button.destroy()
                
                self.open_file_button = ctk.CTkButton(
                    self.preview_frame,
                    text="Open in Browser",
                    command=lambda: self.open_file(file_path)
                )
                self.open_file_button.pack(pady=10)
            else:
                self.preview_label.configure(text="Error: Could not create combined visualization")
        except Exception as e:
            self.preview_label.configure(text=f"Error creating combined visualization:\n{str(e)}")
            print(f"Error creating combined view: {str(e)}")
            import traceback
            traceback.print_exc()

    def open_file(self, filepath):
        """Open a file with the default application"""
        try:
            print(f"Opening file: {filepath}")
            if os.path.exists(filepath):
                if os.name == 'nt':  # Windows
                    os.startfile(filepath)
                elif os.name == 'posix':  # Linux, Mac
                    if 'darwin' in os.sys.platform:  # Mac
                        subprocess.call(('open', filepath))
                    else:  # Linux
                        subprocess.call(('xdg-open', filepath))
                return True
            else:
                print(f"File not found: {filepath}")
                return False
        except Exception as e:
            print(f"Error opening file: {e}")
            return False

    def reset_parameters(self):
        """Reset all parameters to default values"""
        try:
            # Create a new parameters object with default values
            default_params = SubstrateParameters()
            
            # Update all sliders and variables
            self.lignin_var.set(default_params.lignin_content)
            self.cellulose_var.set(default_params.cellulose_content)
            self.moisture_var.set(default_params.moisture_content)
            self.method_var.set(default_params.pasteurization_method)
            self.duration_var.set(default_params.pasteurization_duration)
            self.concentration_var.set(default_params.pasteurization_concentration)
            self.density_var.set(default_params.mycelium_density)
            
            # Update the parameter object
            self.params = default_params
            
            # Update all value labels
            self.lignin_value_label.configure(text=f"{default_params.lignin_content:.2f} (≈{default_params.lignin_content*100:.1f}%)")
            self.cellulose_value_label.configure(text=f"{default_params.cellulose_content:.2f} (≈{default_params.cellulose_content*100:.1f}%)")
            self.moisture_value_label.configure(text=f"{default_params.moisture_content:.2f} (≈{default_params.moisture_content*100:.1f}%)")
            self.duration_value_label.configure(text=f"{default_params.pasteurization_duration:.1f} hours")
            self.concentration_value_label.configure(text=f"{default_params.pasteurization_concentration:.2f} (≈{default_params.pasteurization_concentration*100:.1f}%)")
            self.density_value_label.configure(text=f"{default_params.mycelium_density:.2f} (≈{default_params.mycelium_density*100:.1f}%)")
            
            # Show confirmation message
            self.preview_label.configure(text="Parameters reset to default values.\nClick 'Run Simulation' to see results with default parameters.")
            
            # Clear current results
            self.ph_value.configure(text="--")
            self.conductivity_value.configure(text="--")
            self.resistance_value.configure(text="--")
            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert("1.0", "Click 'Run Simulation' to generate results.")
            
            # Clear charts
            for frame in [self.ph_chart_frame, self.conductivity_chart_frame, 
                          self.resistance_chart_frame, self.growth_chart_frame,
                          self.electrical_chart_frame, self.dielectric_chart_frame]:
                for widget in frame.winfo_children():
                    widget.destroy()
                
            # Add placeholder text to chart frames
            for frame, title in [(self.ph_chart_frame, "pH Levels"), 
                                (self.conductivity_chart_frame, "Conductivity"),
                                (self.resistance_chart_frame, "Resistance"),
                                (self.growth_chart_frame, "Growth Potential"),
                                (self.electrical_chart_frame, "Electrical Properties"),
                                (self.dielectric_chart_frame, "Dielectric Properties")]:
                placeholder = ctk.CTkLabel(frame, text=f"Run simulation to generate {title} chart")
                placeholder.pack(expand=True, pady=50)
                
            print("Parameters reset successfully")
        except Exception as e:
            print(f"Error resetting parameters: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_substrate_type(self, substrate_type):
        """Update default parameters based on substrate type"""
        # Set default values based on substrate type
        if substrate_type == "HARDWOOD":
            self.lignin_var.set(0.25)
            self.cellulose_var.set(0.45)
            self.moisture_var.set(0.60)
            self.nitrogen_var.set(0.01)
            self.ph_var.set(5.5)
        elif substrate_type == "SOFTWOOD":
            self.lignin_var.set(0.15)
            self.cellulose_var.set(0.35)
            self.moisture_var.set(0.70)
            self.nitrogen_var.set(0.015)
            self.ph_var.set(6.0)
        elif substrate_type == "STRAW":
            self.lignin_var.set(0.15)
            self.cellulose_var.set(0.35)
            self.moisture_var.set(0.70)
            self.nitrogen_var.set(0.015)
            self.ph_var.set(6.0)
        elif substrate_type == "COFFEE_GROUNDS":
            self.lignin_var.set(0.20)
            self.cellulose_var.set(0.30)
            self.moisture_var.set(0.65)
            self.nitrogen_var.set(0.03)
            self.ph_var.set(6.2)
        elif substrate_type == "COMPOST":
            self.lignin_var.set(0.10)
            self.cellulose_var.set(0.30)
            self.moisture_var.set(0.70)
            self.nitrogen_var.set(0.01)
            self.ph_var.set(5.0)
        elif substrate_type == "COCO_COIR":
            self.lignin_var.set(0.10)
            self.cellulose_var.set(0.30)
            self.moisture_var.set(0.70)
            self.nitrogen_var.set(0.01)
            self.ph_var.set(5.0)
        elif substrate_type == "OYSTER_SHELL_MIX":
            self.lignin_var.set(0.20)
            self.cellulose_var.set(0.40)
            self.moisture_var.set(0.60)
            self.nitrogen_var.set(0.02)
            self.ph_var.set(5.5)
        else:  # MIXED
            self.lignin_var.set(0.20)
            self.cellulose_var.set(0.40)
            self.moisture_var.set(0.60)
            self.nitrogen_var.set(0.02)
            self.ph_var.set(5.5)

    def update_species_defaults(self, species):
        """Update default parameters based on mushroom species"""
        # Set default values based on species
        if species == "OYSTER":
            self.lignin_var.set(0.25)
            self.cellulose_var.set(0.45)
            self.moisture_var.set(0.60)
            self.nitrogen_var.set(0.01)
            self.ph_var.set(5.5)
        elif species == "SHIITAKE":
            self.lignin_var.set(0.20)
            self.cellulose_var.set(0.35)
            self.moisture_var.set(0.70)
            self.nitrogen_var.set(0.015)
            self.ph_var.set(6.0)
        elif species == "LIONS_MANE":
            self.lignin_var.set(0.30)
            self.cellulose_var.set(0.50)
            self.moisture_var.set(0.55)
            self.nitrogen_var.set(0.02)
            self.ph_var.set(6.5)
        elif species == "REISHI":
            self.lignin_var.set(0.25)
            self.cellulose_var.set(0.40)
            self.moisture_var.set(0.65)
            self.nitrogen_var.set(0.01)
            self.ph_var.set(5.5)
        elif species == "BUTTON":
            self.lignin_var.set(0.15)
            self.cellulose_var.set(0.30)
            self.moisture_var.set(0.75)
            self.nitrogen_var.set(0.01)
            self.ph_var.set(6.0)
        elif species == "ENOKI":
            self.lignin_var.set(0.20)
            self.cellulose_var.set(0.35)
            self.moisture_var.set(0.70)
            self.nitrogen_var.set(0.015)
            self.ph_var.set(6.2)
        elif species == "MAITAKE":
            self.lignin_var.set(0.25)
            self.cellulose_var.set(0.40)
            self.moisture_var.set(0.65)
            self.nitrogen_var.set(0.01)
            self.ph_var.set(5.5)
        elif species == "CORDYCEPS":
            self.lignin_var.set(0.30)
            self.cellulose_var.set(0.50)
            self.moisture_var.set(0.55)
            self.nitrogen_var.set(0.02)
            self.ph_var.set(6.5)

    def show_multi_tab_visualization(self):
        """Show the multi-tab visualization with different views"""
        try:
            # Update the preview label to show we're working
            self.preview_label.configure(text="Generating multi-tab visualization... please wait")
            self.root.update()  # Force update the UI
            
            # Create the interactive visualization
            fig = create_multi_tab_visualization(self.params)
            
            # Save and try to open it
            file_path = save_and_open_multi_tab_visualization(fig, self.params, "substrate_multi_tab_view.html")
            
            if file_path and os.path.exists(file_path):
                # Update the preview label
                self.preview_label.configure(text=f"Multi-tab visualization saved to:\n{file_path}\n\nIf it doesn't open automatically, please open this file in your browser.")
                
                # Add a button to manually open the file
                if hasattr(self, 'open_file_button'):
                    self.open_file_button.destroy()
                
                self.open_file_button = ctk.CTkButton(
                    self.preview_frame,
                    text="Open in Browser",
                    command=lambda: self.open_file(file_path)
                )
                self.open_file_button.pack(pady=10)
            else:
                self.preview_label.configure(text="Error: Could not create multi-tab visualization")
        except Exception as e:
            self.preview_label.configure(text=f"Error creating multi-tab visualization:\n{str(e)}")
            print(f"Error creating multi-tab view: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_calcium_type(self, calcium_type):
        """Update parameters based on calcium supplement type"""
        self.params.calcium_supplement = calcium_type
        
        # Enable/disable calcium content slider based on selection
        if calcium_type == "NONE":
            self.calcium_content_var.set(0.0)
            self.calcination_var.set(False)
        else:
            # Set default values based on calcium type
            if calcium_type == "OYSTER_SHELL":
                self.calcium_content_var.set(0.05)
            elif calcium_type == "LIMESTONE":
                self.calcium_content_var.set(0.03)
            elif calcium_type == "GYPSUM":
                self.calcium_content_var.set(0.02)
        
        # Update parameters
        self.update_params()

    def update_params(self):
        """Update all parameters from UI controls"""
        try:
            # Update substrate parameters
            self.params.substrate_type = self.substrate_type_var.get()
            self.params.lignin_content = self.lignin_var.get()
            self.params.cellulose_content = self.cellulose_var.get()
            self.params.moisture_content = self.moisture_var.get()
            self.params.nitrogen_content = self.nitrogen_var.get()
            self.params.initial_ph = self.ph_var.get()
            
            # Update calcium parameters
            self.params.calcium_supplement = self.calcium_type_var.get()
            self.params.calcium_content = self.calcium_content_var.get()
            self.params.calcium_calcination = self.calcination_var.get()
            
            # Update pasteurization parameters
            if hasattr(self, 'pasteurization_method_var'):
                self.params.pasteurization_method = self.pasteurization_method_var.get()
                self.params.pasteurization_duration = self.pasteurization_duration_var.get()
                self.params.pasteurization_concentration = self.pasteurization_concentration_var.get()
                self.params.pasteurization_temperature = self.pasteurization_temperature_var.get()
            
            # Update mycelium parameters
            if hasattr(self, 'mycelium_species_var'):
                self.params.mycelium_species = self.mycelium_species_var.get()
                self.params.mycelium_density = self.mycelium_density_var.get()
                self.params.mycelium_age = self.mycelium_age_var.get()
            
            # Update environmental parameters
            if hasattr(self, 'incubation_temperature_var'):
                self.params.incubation_temperature = self.incubation_temperature_var.get()
                self.params.humidity = self.humidity_var.get()
            
            print("Parameters updated successfully")
        except Exception as e:
            print(f"Error updating parameters: {str(e)}")
            import traceback
            traceback.print_exc()

    def add_tooltip(self, widget, text):
        """Add a tooltip to a widget"""
        def show_tooltip(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            # Create a toplevel window
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            # Create label in the toplevel
            label = tk.Label(self.tooltip, text=text, justify=tk.LEFT,
                             background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                             font=("Arial", "10", "normal"), padx=5, pady=5)
            label.pack(ipadx=1)
        
        def hide_tooltip(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
                delattr(self, 'tooltip')
        
        # Bind events to the widget
        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)
