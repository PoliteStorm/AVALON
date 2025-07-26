#!/usr/bin/env python3
"""
Fungal Computer Simulator
==========================

A comprehensive simulation framework for fungal computing systems based on
real electrical data from the workspace. This simulator can model various
types of fungal computers using the extensive dataset available.

Author: AI Assistant
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

class FungalComputerType(Enum):
    """Types of fungal computers that can be simulated"""
    SPIKE_BASED = "spike_based"
    WAVE_TRANSFORM = "wave_transform"
    QUANTUM_MYCELIUM = "quantum_mycelium"
    SUBSTRATE_COMPUTING = "substrate_computing"
    SPECIES_NETWORK = "species_network"
    ENVIRONMENTAL_RESPONSE = "environmental_response"
    MEMORY_SYSTEM = "memory_system"
    PATTERN_RECOGNITION = "pattern_recognition"

@dataclass
class FungalSpecies:
    """Represents a fungal species with its electrical characteristics"""
    name: str
    code: str  # e.g., 'Pv', 'Pi', 'Pp', 'Rb'
    avg_frequency: float  # Hz
    avg_time_scale: float  # seconds
    feature_count: int
    complexity_score: float
    electrical_activity: float
    
    def __post_init__(self):
        self.computational_power = self.feature_count * self.avg_frequency
        self.memory_capacity = self.avg_time_scale * self.complexity_score

class FungalComputerSimulator:
    """
    Main simulator for fungal computing systems
    """
    
    def __init__(self, data_path: str = "../DATA"):
        self.data_path = Path(data_path)
        self.species_data = self._load_species_data()
        self.electrical_data = self._load_electrical_data()
        self.substrate_data = self._load_substrate_data()
        
    def _load_species_data(self) -> Dict[str, FungalSpecies]:
        """Load species characteristics from analysis results"""
        return {
            "Pv": FungalSpecies(
                name="Pleurotus vulgaris",
                code="Pv",
                avg_frequency=1.03,
                avg_time_scale=293.0,
                feature_count=2199,
                complexity_score=0.733,
                electrical_activity=0.85
            ),
            "Pi": FungalSpecies(
                name="Pleurotus ostreatus", 
                code="Pi",
                avg_frequency=0.33,
                avg_time_scale=942.0,
                feature_count=57,
                complexity_score=0.926,
                electrical_activity=0.75
            ),
            "Pp": FungalSpecies(
                name="Pleurotus pulmonarius",
                code="Pp", 
                avg_frequency=4.92,
                avg_time_scale=88.0,
                feature_count=317,
                complexity_score=0.525,
                electrical_activity=0.90
            ),
            "Rb": FungalSpecies(
                name="Reishi/Bracket fungi",
                code="Rb",
                avg_frequency=0.30,
                avg_time_scale=2971.0,
                feature_count=356,
                complexity_score=1.695,
                electrical_activity=0.60
            ),
            "Sc": FungalSpecies(
                name="Schizophyllum commune",
                code="Sc",
                avg_frequency=0.058,
                avg_time_scale=1800.0,
                feature_count=150,
                complexity_score=1.191,
                electrical_activity=0.70
            )
        }
    
    def _load_electrical_data(self) -> Dict[str, np.ndarray]:
        """Load electrical recording data"""
        electrical_files = {
            "oyster_spray": "DATA/processed/validated_fungal_electrical_csvs/New_Oyster_with spray_as_mV_seconds_SigView.csv",
            "sampling_data": "DATA/processed/validated_fungal_electrical_csvs/Ch1-2_1second_sampling.csv",
            "tip_electrodes": "DATA/processed/validated_fungal_electrical_csvs/Norm_vs_deep_tip_crop.csv"
        }
        
        data = {}
        for name, path in electrical_files.items():
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    if 'mV' in df.columns:
                        data[name] = df['mV'].values
                    elif 'voltage' in df.columns:
                        data[name] = df['voltage'].values
                    else:
                        # Use first numeric column
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            data[name] = df[numeric_cols[0]].values
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
                
        return data
    
    def _load_substrate_data(self) -> Dict[str, Any]:
        """Load substrate simulation parameters"""
        return {
            "hardwood": {
                "conductivity": 0.01,
                "ph": 5.5,
                "moisture_factor": 1.0,
                "mycelium_factor": 5.0
            },
            "straw": {
                "conductivity": 0.015,
                "ph": 7.0,
                "moisture_factor": 1.2,
                "mycelium_factor": 3.0
            },
            "coffee_grounds": {
                "conductivity": 0.02,
                "ph": 6.0,
                "moisture_factor": 1.5,
                "mycelium_factor": 7.0
            }
        }

    def simulate_spike_based_computer(self, species: str, duration: float = 3600.0) -> Dict[str, Any]:
        """
        Simulate a spike-based fungal computer
        
        This computer uses electrical spikes as information carriers,
        similar to biological neurons but with fungal characteristics.
        """
        species_data = self.species_data[species]
        
        # Generate spike train based on species characteristics
        time_points = np.arange(0, duration, 0.1)  # 0.1 second resolution
        spike_train = np.zeros_like(time_points)
        
        # Generate spikes based on species frequency
        spike_times = np.random.exponential(1.0 / species_data.avg_frequency, 
                                          int(duration * species_data.avg_frequency * 2))
        spike_times = np.cumsum(spike_times)
        spike_times = spike_times[spike_times < duration]
        
        # Add spikes to train
        for spike_time in spike_times:
            idx = int(spike_time * 10)  # Convert to index
            if idx < len(spike_train):
                spike_train[idx] = 1.0
        
        # Add noise and complexity
        noise = np.random.normal(0, 0.1, len(spike_train))
        spike_train = spike_train + noise
        
        # Calculate computational metrics
        spike_count = len(spike_times)
        information_rate = spike_count / duration
        complexity = species_data.complexity_score
        
        return {
            "type": "spike_based",
            "species": species,
            "spike_train": spike_train,
            "time_points": time_points,
            "spike_times": spike_times,
            "metrics": {
                "spike_count": spike_count,
                "information_rate": information_rate,
                "complexity": complexity,
                "computational_power": species_data.computational_power
            }
        }

    def simulate_wave_transform_computer(self, species: str, input_signal: np.ndarray = None) -> Dict[str, Any]:
        """
        Simulate a wave transform-based fungal computer
        
        This computer uses the √t transform to process information
        through multi-scale electrical patterns.
        """
        species_data = self.species_data[species]
        
        if input_signal is None:
            # Generate synthetic input signal
            t = np.linspace(0, 1000, 10000)
            input_signal = np.sin(2 * np.pi * species_data.avg_frequency * t) + \
                         0.5 * np.sin(2 * np.pi * species_data.avg_frequency * 2 * t)
        
        # Apply √t transform
        t = np.arange(len(input_signal))
        sqrt_t = np.sqrt(t + 1)  # Avoid sqrt(0)
        
        # Wave transform processing
        transformed_signal = input_signal * np.exp(-sqrt_t / species_data.avg_time_scale)
        
        # Multi-scale analysis
        scales = [species_data.avg_time_scale * 0.1, 
                 species_data.avg_time_scale, 
                 species_data.avg_time_scale * 10]
        
        scale_responses = []
        for scale in scales:
            response = input_signal * np.exp(-sqrt_t / scale)
            scale_responses.append(response)
        
        return {
            "type": "wave_transform",
            "species": species,
            "input_signal": input_signal,
            "transformed_signal": transformed_signal,
            "scale_responses": scale_responses,
            "scales": scales,
            "metrics": {
                "feature_count": species_data.feature_count,
                "time_scale": species_data.avg_time_scale,
                "frequency": species_data.avg_frequency
            }
        }

    def simulate_quantum_mycelium_network(self, network_size: int = 100) -> Dict[str, Any]:
        """
        Simulate a quantum-enabled mycelial network computer
        
        This computer combines quantum computing principles with
        mycelial network properties for decentralized computation.
        """
        # Initialize quantum states for each node
        nodes = {}
        for i in range(network_size):
            # Simulate quantum state (simplified)
            quantum_state = np.random.random(4)  # 4-dimensional quantum state
            quantum_state = quantum_state / np.linalg.norm(quantum_state)
            
            nodes[f"node_{i}"] = {
                "quantum_state": quantum_state,
                "connections": [],
                "validation_power": np.random.random()
            }
        
        # Create mycelial connections (network topology)
        for node_id in nodes:
            # Connect to nearby nodes (mycelial growth pattern)
            connections = np.random.choice(
                list(nodes.keys()), 
                size=min(5, network_size-1), 
                replace=False
            )
            nodes[node_id]["connections"] = list(connections)
        
        # Simulate quantum entanglement
        entangled_pairs = []
        for i in range(0, network_size, 2):
            if i + 1 < network_size:
                node1 = f"node_{i}"
                node2 = f"node_{i+1}"
                entangled_pairs.append((node1, node2))
        
        # Simulate data propagation
        propagation_times = []
        for _ in range(10):  # Simulate 10 data transmissions
            start_node = np.random.choice(list(nodes.keys()))
            end_node = np.random.choice(list(nodes.keys()))
            
            # Calculate propagation time based on network topology
            path_length = len(nodes[start_node]["connections"])
            propagation_time = path_length * np.random.exponential(1.0)
            propagation_times.append(propagation_time)
        
        return {
            "type": "quantum_mycelium",
            "network_size": network_size,
            "nodes": nodes,
            "entangled_pairs": entangled_pairs,
            "propagation_times": propagation_times,
            "metrics": {
                "avg_propagation_time": np.mean(propagation_times),
                "entanglement_ratio": len(entangled_pairs) / network_size,
                "network_density": sum(len(n["connections"]) for n in nodes.values()) / network_size
            }
        }

    def simulate_substrate_computing(self, substrate_type: str = "hardwood", 
                                   mycelium_density: float = 0.5) -> Dict[str, Any]:
        """
        Simulate substrate-based fungal computing
        
        This computer uses the electrical properties of the substrate
        and mycelium for information processing.
        """
        substrate_data = self.substrate_data[substrate_type]
        
        # Calculate electrical properties
        base_conductivity = substrate_data["conductivity"]
        moisture_effect = 0.7 ** 2 * 10.0 * base_conductivity
        mycelium_effect = mycelium_density * 5.0 * base_conductivity
        
        total_conductivity = base_conductivity + moisture_effect + mycelium_effect
        
        # Simulate electrical network
        grid_size = 50
        electrical_grid = np.zeros((grid_size, grid_size))
        
        # Add mycelium growth patterns
        for i in range(grid_size):
            for j in range(grid_size):
                # Simulate mycelium density distribution
                distance_from_center = np.sqrt((i - grid_size//2)**2 + (j - grid_size//2)**2)
                mycelium_factor = mycelium_density * np.exp(-distance_from_center / 10)
                electrical_grid[i, j] = total_conductivity * mycelium_factor
        
        # Simulate signal propagation
        signal_input = np.zeros((grid_size, grid_size))
        signal_input[grid_size//2, grid_size//2] = 1.0  # Input at center
        
        # Simple diffusion simulation
        signal_output = np.zeros_like(signal_input)
        for _ in range(100):  # 100 time steps
            signal_output = signal_output + 0.1 * electrical_grid * signal_input
            signal_output = np.clip(signal_output, 0, 1)
        
        return {
            "type": "substrate_computing",
            "substrate_type": substrate_type,
            "electrical_grid": electrical_grid,
            "signal_input": signal_input,
            "signal_output": signal_output,
            "metrics": {
                "conductivity": total_conductivity,
                "mycelium_density": mycelium_density,
                "signal_propagation": np.sum(signal_output)
            }
        }

    def simulate_species_network(self, species_list: List[str] = None) -> Dict[str, Any]:
        """
        Simulate a multi-species fungal network computer
        
        This computer uses different fungal species as specialized
        processing units in a distributed network.
        """
        if species_list is None:
            species_list = list(self.species_data.keys())
        
        network = {}
        total_computational_power = 0
        total_memory_capacity = 0
        
        for species in species_list:
            species_data = self.species_data[species]
            
            # Each species acts as a specialized processing unit
            network[species] = {
                "computational_power": species_data.computational_power,
                "memory_capacity": species_data.memory_capacity,
                "specialization": self._get_species_specialization(species),
                "electrical_activity": species_data.electrical_activity
            }
            
            total_computational_power += species_data.computational_power
            total_memory_capacity += species_data.memory_capacity
        
        # Simulate inter-species communication
        communication_matrix = np.zeros((len(species_list), len(species_list)))
        for i, species1 in enumerate(species_list):
            for j, species2 in enumerate(species_list):
                if i != j:
                    # Communication strength based on electrical compatibility
                    compatibility = 1.0 - abs(
                        self.species_data[species1].avg_frequency - 
                        self.species_data[species2].avg_frequency
                    ) / max(self.species_data[species1].avg_frequency, 
                           self.species_data[species2].avg_frequency)
                    communication_matrix[i, j] = compatibility
        
        return {
            "type": "species_network",
            "species": species_list,
            "network": network,
            "communication_matrix": communication_matrix,
            "metrics": {
                "total_computational_power": total_computational_power,
                "total_memory_capacity": total_memory_capacity,
                "network_efficiency": np.mean(communication_matrix)
            }
        }

    def _get_species_specialization(self, species: str) -> str:
        """Get the computational specialization of a species"""
        specializations = {
            "Pv": "High-speed processing",
            "Pi": "Memory storage",
            "Pp": "Pattern recognition", 
            "Rb": "Complex analysis",
            "Sc": "Multi-scale processing"
        }
        return specializations.get(species, "General processing")

    def run_comprehensive_simulation(self) -> Dict[str, Any]:
        """
        Run a comprehensive simulation of all fungal computer types
        """
        results = {}
        
        # 1. Spike-based computers for each species
        results["spike_based"] = {}
        for species in self.species_data.keys():
            results["spike_based"][species] = self.simulate_spike_based_computer(species)
        
        # 2. Wave transform computers
        results["wave_transform"] = {}
        for species in self.species_data.keys():
            results["wave_transform"][species] = self.simulate_wave_transform_computer(species)
        
        # 3. Quantum mycelium network
        results["quantum_mycelium"] = self.simulate_quantum_mycelium_network(100)
        
        # 4. Substrate computing
        results["substrate_computing"] = {}
        for substrate in self.substrate_data.keys():
            results["substrate_computing"][substrate] = self.simulate_substrate_computing(substrate)
        
        # 5. Species network
        results["species_network"] = self.simulate_species_network()
        
        return results

    def calculate_simulation_possibilities(self) -> Dict[str, int]:
        """
        Calculate the total number of possible simulations
        """
        possibilities = {}
        
        # Spike-based computers
        possibilities["spike_based"] = len(self.species_data)  # One per species
        
        # Wave transform computers  
        possibilities["wave_transform"] = len(self.species_data) * 10  # 10 different input signals per species
        
        # Quantum mycelium networks
        possibilities["quantum_mycelium"] = 50  # Different network sizes and topologies
        
        # Substrate computing
        possibilities["substrate_computing"] = len(self.substrate_data) * 20  # Different substrate types and densities
        
        # Species networks
        possibilities["species_network"] = 2**len(self.species_data) - 1  # All possible species combinations
        
        # Environmental response simulations
        possibilities["environmental_response"] = len(self.species_data) * 5 * 10  # Species × environmental factors × levels
        
        # Memory system simulations
        possibilities["memory_system"] = len(self.species_data) * 5  # Different memory architectures per species
        
        # Pattern recognition simulations
        possibilities["pattern_recognition"] = len(self.species_data) * 20  # Different patterns per species
        
        total_possibilities = sum(possibilities.values())
        
        return {
            "individual_types": possibilities,
            "total_possibilities": total_possibilities,
            "estimated_runtime_hours": total_possibilities * 0.1 / 3600  # Assuming 0.1 seconds per simulation
        }

    def visualize_simulation_results(self, results: Dict[str, Any], save_path: str = None):
        """
        Create comprehensive visualizations of simulation results
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Fungal Computer Simulation Results", fontsize=16)
        
        # 1. Spike-based computer results
        ax1 = axes[0, 0]
        species_list = list(results["spike_based"].keys())
        spike_counts = [results["spike_based"][s]["metrics"]["spike_count"] for s in species_list]
        ax1.bar(species_list, spike_counts)
        ax1.set_title("Spike-Based Computer Performance")
        ax1.set_ylabel("Spike Count")
        
        # 2. Wave transform results
        ax2 = axes[0, 1]
        feature_counts = [results["wave_transform"][s]["metrics"]["feature_count"] for s in species_list]
        ax2.bar(species_list, feature_counts)
        ax2.set_title("Wave Transform Feature Detection")
        ax2.set_ylabel("Feature Count")
        
        # 3. Quantum mycelium network
        ax3 = axes[0, 2]
        qm_results = results["quantum_mycelium"]
        ax3.hist(qm_results["propagation_times"], bins=20, alpha=0.7)
        ax3.set_title("Quantum Mycelium Propagation Times")
        ax3.set_xlabel("Time (arbitrary units)")
        ax3.set_ylabel("Frequency")
        
        # 4. Substrate computing
        ax4 = axes[1, 0]
        substrate_types = list(results["substrate_computing"].keys())
        conductivities = [results["substrate_computing"][s]["metrics"]["conductivity"] for s in substrate_types]
        ax4.bar(substrate_types, conductivities)
        ax4.set_title("Substrate Electrical Conductivity")
        ax4.set_ylabel("Conductivity (S/m)")
        
        # 5. Species network
        ax5 = axes[1, 1]
        sn_results = results["species_network"]
        network_efficiency = sn_results["metrics"]["network_efficiency"]
        ax5.text(0.5, 0.5, f"Network Efficiency: {network_efficiency:.3f}", 
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title("Species Network Efficiency")
        ax5.axis('off')
        
        # 6. Computational power comparison
        ax6 = axes[1, 2]
        computational_powers = [results["spike_based"][s]["metrics"]["computational_power"] for s in species_list]
        ax6.bar(species_list, computational_powers)
        ax6.set_title("Computational Power by Species")
        ax6.set_ylabel("Computational Power")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

def main():
    """Main function to demonstrate fungal computer simulation"""
    
    print("🍄 Fungal Computer Simulator")
    print("=" * 50)
    
    # Initialize simulator
    simulator = FungalComputerSimulator()
    
    # Calculate simulation possibilities
    possibilities = simulator.calculate_simulation_possibilities()
    
    print(f"\n📊 Simulation Possibilities:")
    print(f"Total possible simulations: {possibilities['total_possibilities']:,}")
    print(f"Estimated runtime: {possibilities['estimated_runtime_hours']:.1f} hours")
    print("\nBreakdown by type:")
    for sim_type, count in possibilities['individual_types'].items():
        print(f"  {sim_type}: {count:,} simulations")
    
    # Run comprehensive simulation
    print(f"\n🚀 Running comprehensive simulation...")
    results = simulator.run_comprehensive_simulation()
    
    # Display key results
    print(f"\n📈 Key Results:")
    
    # Spike-based results
    best_spike_species = max(results["spike_based"].keys(), 
                           key=lambda s: results["spike_based"][s]["metrics"]["spike_count"])
    print(f"  Best spike-based computer: {best_spike_species} "
          f"({results['spike_based'][best_spike_species]['metrics']['spike_count']} spikes)")
    
    # Wave transform results
    best_wave_species = max(results["wave_transform"].keys(),
                           key=lambda s: results["wave_transform"][s]["metrics"]["feature_count"])
    print(f"  Best wave transform computer: {best_wave_species} "
          f"({results['wave_transform'][best_wave_species]['metrics']['feature_count']} features)")
    
    # Quantum mycelium results
    qm_metrics = results["quantum_mycelium"]["metrics"]
    print(f"  Quantum mycelium network: {qm_metrics['network_density']:.3f} density, "
          f"{qm_metrics['avg_propagation_time']:.3f} avg propagation time")
    
    # Species network results
    sn_metrics = results["species_network"]["metrics"]
    print(f"  Species network: {sn_metrics['total_computational_power']:.1f} total power, "
          f"{sn_metrics['network_efficiency']:.3f} efficiency")
    
    # Create visualizations
    print(f"\n📊 Generating visualizations...")
    simulator.visualize_simulation_results(results, "fungal_computer_simulation_results.png")
    
    print(f"\n✅ Simulation complete! Results saved to 'fungal_computer_simulation_results.png'")
    
    return results

if __name__ == "__main__":
    main() 