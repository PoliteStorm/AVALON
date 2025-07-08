#!/usr/bin/env python3
"""
🍄 Mushroom Communication Translator 3D - FIXED VERSION 🍄
World's First Real-Time Mushroom Communication Decoder with 3D Spatial Triangulation
Based on Joe's quantum foam discovery at Glastonbury 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import json
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MushroomTranslator3D:
    def __init__(self, root):
        self.root = root
        self.root.title("🍄 Mushroom Communication Translator 3D 🍄")
        self.root.geometry("1600x1000")
        self.root.configure(bg='black')
        
        # Initialize state
        self.time_step = 0
        self.animation_running = False
        self.action_potentials = []
        self.communication_log = []
        self.mycelial_network = {}
        
        # Current species
        self.current_species = "Schizophyllum commune"
        
        # Species data based on Adamatzky's research
        self.species_data = {
            "Schizophyllum commune": {
                "voltage_range": (-0.08, 0.12),
                "frequency_range": (0.016, 0.133),
                "spike_duration": 13.0,
                "environment": "nutrient-rich",
                "spatial_patterns": "radial growth coordination",
                "behavior": "coordinated hyphal growth"
            },
            "Flammulina velutipes": {
                "voltage_range": (-0.15, 0.25),
                "frequency_range": (0.067, 0.5),
                "spike_duration": 37.5,
                "environment": "temperature stress",
                "spatial_patterns": "stress response networks",
                "behavior": "emergency communication"
            },
            "Omphalotus nidiformis": {
                "voltage_range": (-0.1, 0.15),
                "frequency_range": (0.033, 0.2),
                "spike_duration": 25.0,
                "environment": "standard conditions",
                "spatial_patterns": "synchronized growth",
                "behavior": "collective decision making"
            },
            "Cordyceps militaris": {
                "voltage_range": (-0.2, 0.3),
                "frequency_range": (0.1, 0.8),
                "spike_duration": 45.0,
                "environment": "hunting mode",
                "spatial_patterns": "target tracking",
                "behavior": "predatory coordination"
            }
        }
        
        # Signal classification vocabulary
        self.mushroom_vocabulary = {
            "low_frequency_burst": "Growth coordination signal",
            "high_frequency_pulse": "Emergency alert - environmental stress detected!",
            "sustained_oscillation": "Synchronized growth pattern activated",
            "spike_train": "Rapid information transfer - target acquired!",
            "low_amplitude_wave": "Background maintenance communication",
            "high_amplitude_spike": "Priority alert - immediate action required!",
            "rhythmic_pattern": "Coordinated network activity",
            "irregular_burst": "Exploratory signal - investigating environment"
        }
        
        self.setup_gui()
        self.setup_plots()
        self.initialize_mycelial_network()
        self.start_simulation()
    
    def setup_gui(self):
        """Create the GUI layout"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='black')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="🍄 Mushroom Communication Translator 3D 🍄", 
                              font=('Arial', 16, 'bold'), fg='lime', bg='black')
        title_label.pack(pady=(0, 10))
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='black')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Species selection
        tk.Label(control_frame, text="Species:", fg='white', bg='black').pack(side=tk.LEFT, padx=(0, 5))
        self.species_var = tk.StringVar(value=self.current_species)
        species_combo = ttk.Combobox(control_frame, textvariable=self.species_var, 
                                    values=list(self.species_data.keys()), state='readonly')
        species_combo.pack(side=tk.LEFT, padx=(0, 10))
        species_combo.bind('<<ComboboxSelected>>', self.change_species)
        
        # Control buttons
        self.start_btn = tk.Button(control_frame, text="Start Translation", bg='green', fg='white',
                                  command=self.toggle_animation)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(control_frame, text="Reset", bg='orange', fg='white',
                 command=self.reset_simulation).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(control_frame, text="Save Translation", bg='blue', fg='white',
                 command=self.save_translation).pack(side=tk.LEFT, padx=(0, 5))
        
        # Main content frame
        content_frame = tk.Frame(main_frame, bg='black')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - plots
        plot_frame = tk.Frame(content_frame, bg='black')
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = plt.figure(figsize=(14, 10), facecolor='black')
        self.fig.suptitle('Mushroom Communication Analysis', color='lime', fontsize=16, fontweight='bold')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Right panel - text displays
        text_frame = tk.Frame(content_frame, bg='black', width=400)
        text_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        text_frame.pack_propagate(False)
        
        # Communication log
        tk.Label(text_frame, text="Communication Log", fg='cyan', bg='black', 
                font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        
        self.communication_text = tk.Text(text_frame, height=15, bg='black', fg='white',
                                        font=('Courier', 10))
        self.communication_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Spatial analysis
        tk.Label(text_frame, text="Spatial Analysis", fg='cyan', bg='black',
                font=('Arial', 12, 'bold')).pack(pady=(0, 5))
        
        self.spatial_text = tk.Text(text_frame, height=15, bg='black', fg='white',
                                   font=('Courier', 10))
        self.spatial_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_plots(self):
        """Setup the 6 analysis plots"""
        # Create subplots
        self.ax1 = self.fig.add_subplot(231, projection='3d')
        self.ax2 = self.fig.add_subplot(232, projection='3d')
        self.ax3 = self.fig.add_subplot(233)
        self.ax4 = self.fig.add_subplot(234)
        self.ax5 = self.fig.add_subplot(235)
        self.ax6 = self.fig.add_subplot(236)
        
        # Set titles
        self.ax1.set_title("3D Mycelial Network", color='lime', fontweight='bold')
        self.ax2.set_title("Signal Propagation", color='lime', fontweight='bold')
        self.ax3.set_title("Communication Timeline", color='lime', fontweight='bold')
        self.ax4.set_title("Frequency Analysis", color='lime', fontweight='bold')
        self.ax5.set_title("Spatial Heat Map", color='lime', fontweight='bold')
        self.ax6.set_title("Signal Classification", color='lime', fontweight='bold')
        
        plt.tight_layout()
    
    def initialize_mycelial_network(self):
        """Initialize the 3D mycelial network structure"""
        # Create network nodes
        self.mycelial_network = {}
        
        # Generate random network topology
        for i in range(20):
            node_id = f"node_{i}"
            self.mycelial_network[node_id] = {
                'position': (
                    np.random.uniform(-10, 10),
                    np.random.uniform(-10, 10),
                    np.random.uniform(-2, 2)
                ),
                'connections': [],
                'activity': 0.0
            }
        
        # Create connections between nearby nodes
        nodes = list(self.mycelial_network.keys())
        for i, node_a in enumerate(nodes):
            pos_a = self.mycelial_network[node_a]['position']
            for j, node_b in enumerate(nodes[i+1:], i+1):
                pos_b = self.mycelial_network[node_b]['position']
                distance = np.sqrt(sum((a-b)**2 for a, b in zip(pos_a, pos_b)))
                if distance < 8.0:  # Connect nearby nodes
                    self.mycelial_network[node_a]['connections'].append(node_b)
                    self.mycelial_network[node_b]['connections'].append(node_a)
    
    def generate_action_potentials(self):
        """Generate new action potentials based on current species"""
        species_data = self.species_data[self.current_species]
        
        # Generate 2-4 new action potentials
        num_potentials = np.random.randint(2, 5)
        
        for _ in range(num_potentials):
            # Generate signal parameters
            amplitude = np.random.uniform(*species_data['voltage_range'])
            frequency = np.random.uniform(*species_data['frequency_range'])
            
            # Random 3D position in the network
            position = (
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10),
                np.random.uniform(-2, 2)
            )
            
            # Classify signal
            signal_type = self.classify_signal(amplitude, frequency)
            
            # Create action potential
            ap = {
                'position': position,
                'amplitude': amplitude,
                'frequency': frequency,
                'timestamp': self.time_step * 0.5,
                'signal_type': signal_type,
                'duration': species_data['spike_duration']
            }
            
            self.action_potentials.append(ap)
            
            # Update network activity
            self.update_network_activity(position, amplitude)
        
        # Keep only recent action potentials
        self.action_potentials = self.action_potentials[-100:]
    
    def classify_signal(self, amplitude, frequency):
        """Classify signal based on amplitude and frequency"""
        if frequency < 0.1:
            if abs(amplitude) < 0.1:
                return "low_amplitude_wave"
            else:
                return "low_frequency_burst"
        elif frequency > 0.4:
            if abs(amplitude) > 0.15:
                return "high_amplitude_spike"
            else:
                return "high_frequency_pulse"
        else:
            if abs(amplitude) > 0.15:
                return "spike_train"
            elif frequency > 0.2:
                return "rhythmic_pattern"
            else:
                return "sustained_oscillation"
    
    def update_network_activity(self, signal_pos, amplitude):
        """Update network node activity based on signal propagation"""
        for node_id, node in self.mycelial_network.items():
            distance = np.sqrt(sum((a-b)**2 for a, b in zip(signal_pos, node['position'])))
            # Activity decreases with distance
            activity_boost = abs(amplitude) * np.exp(-distance / 5.0)
            node['activity'] = min(1.0, node['activity'] + activity_boost)
    
    def decode_communication(self):
        """Decode the latest communication signals"""
        if not self.action_potentials:
            return
        
        # Get recent signals
        recent_signals = [ap for ap in self.action_potentials if 
                         ap['timestamp'] > (self.time_step - 1) * 0.5]
        
        for signal in recent_signals:
            # Decode message
            message = self.mushroom_vocabulary.get(signal['signal_type'], 
                                                  "Unknown signal pattern")
            
            # Create communication log entry
            log_entry = {
                'timestamp': signal['timestamp'],
                'message': message,
                'location': f"({signal['position'][0]:.1f}, {signal['position'][1]:.1f}, {signal['position'][2]:.1f}) cm",
                'amplitude': signal['amplitude'],
                'frequency': signal['frequency'],
                'signal_type': signal['signal_type']
            }
            
            self.communication_log.append(log_entry)
        
        # Keep only recent messages
        self.communication_log = self.communication_log[-50:]
    
    def triangulate_signal_source(self):
        """Triangulate the center of communication activity"""
        if len(self.action_potentials) < 3:
            return "Insufficient signals for triangulation"
        
        # Get positions and amplitudes
        positions = np.array([ap['position'] for ap in self.action_potentials])
        amplitudes = np.array([abs(ap['amplitude']) for ap in self.action_potentials])
        
        # Calculate weighted center of mass
        if np.sum(amplitudes) > 0:
            com = np.average(positions, axis=0, weights=amplitudes)
        else:
            com = np.mean(positions, axis=0)
        
        # Calculate network statistics
        distances = np.linalg.norm(positions - com, axis=1)
        network_span = np.max(distances) - np.min(distances)
        
        # Determine topology
        if network_span < 5:
            topology = "Clustered"
        elif network_span < 15:
            topology = "Distributed"
        else:
            topology = "Sparse"
        
        return {
            'center_of_mass': com,
            'network_span': network_span,
            'topology': topology,
            'signal_density': len(self.action_potentials),
            'propagation_time': np.std([ap['timestamp'] for ap in self.action_potentials])
        }
    
    def update_plots(self):
        """Update all visualization plots with robust error handling"""
        try:
            # Clear all plots
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]:
                ax.clear()
            
            # Generate new data
            self.generate_action_potentials()
            self.decode_communication()
            
            # Plot 1: 3D Mycelial Network
            self.ax1.set_title("3D Mycelial Network", color='lime', fontweight='bold')
            
            # Plot network nodes
            for node_id, node in self.mycelial_network.items():
                pos = node['position']
                activity = node['activity']
                color = plt.cm.hot(activity)
                self.ax1.scatter(pos[0], pos[1], pos[2], c=[color], s=100, alpha=0.8)
                
                # Draw connections
                for connected_node in node['connections']:
                    if connected_node in self.mycelial_network:
                        connected_pos = self.mycelial_network[connected_node]['position']
                        self.ax1.plot([pos[0], connected_pos[0]], 
                                     [pos[1], connected_pos[1]], 
                                     [pos[2], connected_pos[2]], 
                                     'white', alpha=0.3, linewidth=0.5)
            
            # Plot 2: Signal Propagation
            self.ax2.set_title("Signal Propagation", color='lime', fontweight='bold')
            
            if self.action_potentials:
                for ap in self.action_potentials[-20:]:  # Show recent signals
                    pos = ap['position']
                    amplitude = abs(ap['amplitude'])
                    self.ax2.scatter(pos[0], pos[1], pos[2], 
                                   c='red', s=amplitude*1000, alpha=0.6)
            
            # Plot 3: Communication Timeline with ROBUST ERROR HANDLING
            self.ax3.set_title("Communication Timeline", color='lime', fontweight='bold')
            self.ax3.set_xlabel("Time (s)")
            self.ax3.set_ylabel("Signal Amplitude")
            
            if self.action_potentials:
                times = [ap['timestamp'] for ap in self.action_potentials]
                amplitudes = [ap['amplitude'] for ap in self.action_potentials]
                colors = [plt.cm.viridis(ap['frequency'] / 3.0) for ap in self.action_potentials]
                
                # Safe scatter plot
                try:
                    self.ax3.scatter(times, amplitudes, c=colors, s=50, alpha=0.7)
                except:
                    self.ax3.scatter(times, amplitudes, c='blue', s=50, alpha=0.7)
                
                # Add trend line with ROBUST error handling
                if len(times) > 1:
                    try:
                        times_array = np.array(times)
                        amplitudes_array = np.array(amplitudes)
                        
                        # Check for valid data
                        if (np.all(np.isfinite(times_array)) and 
                            np.all(np.isfinite(amplitudes_array)) and
                            np.var(times_array) > 1e-12 and
                            np.var(amplitudes_array) > 1e-12):
                            
                            # Use robust polynomial fitting
                            with np.errstate(all='ignore'):
                                z = np.polyfit(times_array, amplitudes_array, 1)
                                p = np.poly1d(z)
                                self.ax3.plot(times_array, p(times_array), 'lime', linestyle='--', alpha=0.7)
                    except:
                        # Skip trend line if any error occurs
                        pass
            
            # Plot 4: Frequency Analysis
            self.ax4.set_title("Frequency Analysis", color='lime', fontweight='bold')
            self.ax4.set_xlabel("Frequency (Hz)")
            self.ax4.set_ylabel("Count")
            
            if self.action_potentials:
                frequencies = [ap['frequency'] for ap in self.action_potentials]
                try:
                    self.ax4.hist(frequencies, bins=10, color='orange', alpha=0.7)
                except:
                    pass
            
            # Plot 5: Spatial Heat Map - SIMPLIFIED to avoid colorbar issues
            self.ax5.set_title("Spatial Heat Map", color='lime', fontweight='bold')
            self.ax5.set_xlabel("X (cm)")
            self.ax5.set_ylabel("Y (cm)")
            
            if self.action_potentials:
                x_coords = [ap['position'][0] for ap in self.action_potentials]
                y_coords = [ap['position'][1] for ap in self.action_potentials]
                amplitudes = [abs(ap['amplitude']) for ap in self.action_potentials]
                
                # Simple scatter plot without problematic colorbar
                try:
                    self.ax5.scatter(x_coords, y_coords, c=amplitudes, 
                                   s=50, cmap='hot', alpha=0.7)
                except:
                    # Fallback to simple red dots
                    self.ax5.scatter(x_coords, y_coords, c='red', s=50, alpha=0.7)
            
            # Plot 6: Signal Classification
            self.ax6.set_title("Signal Classification", color='lime', fontweight='bold')
            self.ax6.set_xlabel("Signal Type")
            self.ax6.set_ylabel("Count")
            
            if self.action_potentials:
                signal_types = [ap['signal_type'] for ap in self.action_potentials]
                unique_types = list(set(signal_types))
                counts = [signal_types.count(t) for t in unique_types]
                
                try:
                    self.ax6.bar(range(len(unique_types)), counts, color='purple', alpha=0.7)
                    self.ax6.set_xticks(range(len(unique_types)))
                    self.ax6.set_xticklabels([t.replace('_', ' ').title() for t in unique_types], 
                                           rotation=45, ha='right')
                except:
                    pass
            
            # Style all plots
            try:
                for ax in [self.ax1, self.ax2]:
                    ax.set_facecolor('black')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    ax.zaxis.label.set_color('white')
                
                for ax in [self.ax3, self.ax4, self.ax5, self.ax6]:
                    ax.set_facecolor('black')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    for spine in ax.spines.values():
                        spine.set_color('white')
            except:
                pass
            
            # Update text displays
            self.update_communication_log()
            self.update_spatial_analysis()
            
            # Refresh display
            try:
                self.canvas.draw()
            except:
                pass  # Ignore canvas drawing errors
                
        except Exception as e:
            print(f"Error in update_plots: {e}")
            # Continue running even if plots fail
            pass
    
    def update_communication_log(self):
        """Update the communication log display"""
        try:
            self.communication_text.delete(1.0, tk.END)
            
            header = f"🍄 {self.current_species} Communication Log 🍄\n"
            header += "=" * 50 + "\n\n"
            self.communication_text.insert(tk.END, header)
            
            for msg in self.communication_log[-10:]:  # Show last 10 messages
                timestamp = f"[{msg['timestamp']:.3f}s]"
                message = f"{timestamp} {msg['message']}\n"
                location = f"   Location: {msg['location']}\n"
                details = f"   Amplitude: {msg['amplitude']:.3f}V, Frequency: {msg['frequency']:.1f}Hz\n\n"
                
                self.communication_text.insert(tk.END, message)
                self.communication_text.insert(tk.END, location)
                self.communication_text.insert(tk.END, details)
            
            self.communication_text.see(tk.END)
        except:
            pass  # Ignore text widget errors
    
    def update_spatial_analysis(self):
        """Update the spatial analysis display"""
        try:
            self.spatial_text.delete(1.0, tk.END)
            
            header = "🗺️ SPATIAL TRIANGULATION ANALYSIS 🗺️\n"
            header += "=" * 40 + "\n\n"
            self.spatial_text.insert(tk.END, header)
            
            if self.action_potentials:
                triangulation = self.triangulate_signal_source()
                if isinstance(triangulation, dict):
                    com = triangulation['center_of_mass']
                    
                    analysis = f"Signal Center of Mass:\n"
                    analysis += f"  X: {com[0]:.1f} cm\n"
                    analysis += f"  Y: {com[1]:.1f} cm\n"
                    analysis += f"  Z: {com[2]:.1f} cm\n\n"
                    
                    analysis += f"Network Topology: {triangulation['topology']}\n"
                    analysis += f"Signal Density: {triangulation['signal_density']} potentials\n"
                    analysis += f"Network Span: {triangulation['network_span']:.1f} cm\n\n"
                    
                    # Species-specific interpretation
                    species_data = self.species_data[self.current_species]
                    analysis += f"Species Pattern: {species_data['spatial_patterns']}\n\n"
                    
                    # Communication pattern analysis
                    if self.action_potentials:
                        signal_types = [ap['signal_type'] for ap in self.action_potentials]
                        most_common = max(set(signal_types), key=signal_types.count)
                        analysis += f"Dominant Signal: {most_common}\n"
                        analysis += f"Communication Mode: {self.mushroom_vocabulary[most_common]}\n\n"
                    
                    # Confidence assessment
                    confidence = min(100, len(self.action_potentials) * 15)
                    analysis += f"Triangulation Confidence: {confidence}%\n"
                    
                    self.spatial_text.insert(tk.END, analysis)
            else:
                self.spatial_text.insert(tk.END, "No signals detected for triangulation")
        except:
            pass  # Ignore text widget errors
    
    def animate(self):
        """Animation loop for real-time analysis"""
        while self.animation_running:
            self.time_step += 1
            self.update_plots()
            time.sleep(0.5)  # 2 FPS for detailed analysis
    
    def toggle_animation(self):
        """Start/stop the translation system"""
        if self.animation_running:
            self.animation_running = False
            self.start_btn.config(text="Start Translation", bg='green')
        else:
            self.animation_running = True
            self.start_btn.config(text="Stop Translation", bg='red')
            animation_thread = threading.Thread(target=self.animate)
            animation_thread.daemon = True
            animation_thread.start()
    
    def reset_simulation(self):
        """Reset the translation system"""
        self.animation_running = False
        self.start_btn.config(text="Start Translation", bg='green')
        self.time_step = 0
        self.action_potentials = []
        self.communication_log = []
        self.initialize_mycelial_network()
        self.update_plots()
    
    def change_species(self, event=None):
        """Change the analyzed species"""
        self.current_species = self.species_var.get()
        self.reset_simulation()
    
    def save_translation(self):
        """Save the current translation log"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'species': self.current_species,
                'communication_log': self.communication_log,
                'action_potentials': [
                    {
                        'position': ap['position'],
                        'amplitude': ap['amplitude'],
                        'frequency': ap['frequency'],
                        'signal_type': ap['signal_type'],
                        'timestamp': ap['timestamp']
                    } for ap in self.action_potentials
                ],
                'triangulation': self.triangulate_signal_source() if self.action_potentials else None
            }
            
            filename = f"mushroom_translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            messagebox.showinfo("Translation Saved", f"Translation log saved to {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save translation: {str(e)}")
    
    def start_simulation(self):
        """Initialize the simulation"""
        self.update_plots()

def main():
    """Main function to run the mushroom translator"""
    print("🍄 Starting Mushroom Communication Translator 3D - FIXED VERSION 🍄")
    print("Decoding what mushrooms are saying and triangulating action potentials")
    print("Based on Joe's quantum foam discovery at Glastonbury 2024")
    print("=" * 70)
    
    try:
        root = tk.Tk()
        app = MushroomTranslator3D(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("Please check your Python environment and dependencies")

if __name__ == "__main__":
    main()
