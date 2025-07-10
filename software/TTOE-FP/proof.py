import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker, cwt, find_peaks
from scipy.ndimage import uniform_filter1d
import networkx as nx
from collections import defaultdict, Counter
import pandas as pd
from scipy.stats import entropy
import json

# Enhanced configuration for comprehensive analysis
SIM_DURATION_HOURS = 100
SAMPLING_RATE_PER_HOUR = 60
TOTAL_SAMPLES = SIM_DURATION_HOURS * SAMPLING_RATE_PER_HOUR

# Multiple simulation parameters for robust analysis
NUM_SIMULATIONS = 25  # Run multiple simulations for pattern analysis
NUM_NODES = 12
CONNECTIVITY_PROB = 0.35
SCALES = np.arange(1, 15, 0.5)
FREQUENCIES_K = np.linspace(0.1, 8, 30)

# Enhanced fungal vocabulary with more specific patterns
ENHANCED_FUNGAL_VOCABULARY = {
    "SPIKE_GENERIC": "SIGNAL_TRANSMISSION",
    "BURST_HIGH_FREQ": "NUTRIENT_DETECTED", 
    "PROLONGED_LOW_FREQ": "STRESS_CONDITION",
    "ADAMATZKY_ON_SPIKE": "PRESSURE_ON",
    "ADAMATZKY_OFF_SPIKE": "PRESSURE_OFF",
    "RAPID_OSCILLATION": "RESOURCE_ALLOCATION",
    "DECAY_PATTERN": "PATHWAY_DEGRADATION",
    "GROWTH_SURGE": "NETWORK_EXPANSION",
    "SYNCHRONOUS_BURST": "COLLECTIVE_DECISION",
    "ASYMMETRIC_WAVE": "DIRECTIONAL_GROWTH",
    "THRESHOLD_BREACH": "CRITICAL_STATE_CHANGE",
    "HARMONIC_RESONANCE": "NETWORK_SYNCHRONIZATION"
}

NODE_STATES = ["IDLE", "SEARCHING", "DEFENDING", "GROWING", "COMMUNICATING", "RESOURCE_SHARING", "STRESS_RESPONSE"]

class MycelialRosettaStone:
    def __init__(self):
        self.pattern_frequency = defaultdict(int)
        self.state_transitions = defaultdict(lambda: defaultdict(int))
        self.stimulus_response_patterns = defaultdict(list)
        self.network_behavior_correlations = defaultdict(list)
        self.temporal_pattern_sequences = []
        self.cross_node_propagation = defaultdict(list)
        self.rosetta_mappings = {}
        
    def record_pattern(self, pattern_id, node_id, time_sample, context=None):
        """Record detected patterns with context"""
        self.pattern_frequency[pattern_id] += 1
        if context:
            self.stimulus_response_patterns[context].append({
                'pattern': pattern_id,
                'node': node_id,
                'time': time_sample,
                'meaning': ENHANCED_FUNGAL_VOCABULARY.get(pattern_id, "UNKNOWN")
            })
    
    def record_state_transition(self, from_state, to_state, trigger_pattern=None):
        """Record state transitions and their triggers"""
        self.state_transitions[from_state][to_state] += 1
        
    def analyze_temporal_sequences(self, node_patterns, time_window=10):
        """Analyze temporal sequences of patterns"""
        for node_id, patterns in node_patterns.items():
            if len(patterns) < 2:
                continue
            
            # Sort patterns by time
            sorted_patterns = sorted(patterns, key=lambda x: x[0])
            
            # Find sequences within time windows
            for i in range(len(sorted_patterns)-1):
                current_time, current_pattern = sorted_patterns[i]
                next_time, next_pattern = sorted_patterns[i+1]
                
                if next_time - current_time <= time_window * SAMPLING_RATE_PER_HOUR:
                    sequence = f"{current_pattern} -> {next_pattern}"
                    self.temporal_pattern_sequences.append({
                        'sequence': sequence,
                        'node': node_id,
                        'interval': next_time - current_time,
                        'meaning': f"{ENHANCED_FUNGAL_VOCABULARY.get(current_pattern, 'UNKNOWN')} leads to {ENHANCED_FUNGAL_VOCABULARY.get(next_pattern, 'UNKNOWN')}"
                    })
    
    def build_rosetta_stone(self):
        """Build the comprehensive Rosetta Stone mapping"""
        # Pattern frequency analysis
        total_patterns = sum(self.pattern_frequency.values())
        pattern_probabilities = {k: v/total_patterns for k, v in self.pattern_frequency.items()}
        
        # Most common sequences
        sequence_counter = Counter([seq['sequence'] for seq in self.temporal_pattern_sequences])
        common_sequences = dict(sequence_counter.most_common(10))
        
        # State transition probabilities
        transition_probs = {}
        for from_state, transitions in self.state_transitions.items():
            total_transitions = sum(transitions.values())
            if total_transitions > 0:
                transition_probs[from_state] = {
                    to_state: count/total_transitions 
                    for to_state, count in transitions.items()
                }
        
        self.rosetta_mappings = {
            'pattern_frequencies': pattern_probabilities,
            'common_sequences': common_sequences,
            'state_transition_probabilities': transition_probs,
            'stimulus_responses': dict(self.stimulus_response_patterns),
            'vocabulary': ENHANCED_FUNGAL_VOCABULARY
        }
        
        return self.rosetta_mappings

def enhanced_pattern_detection(cwt_matrix, scales, signal, threshold_factor=0.3):
    """Enhanced pattern detection with multiple pattern types"""
    detected_patterns = []
    
    # 1. Spike detection (high frequency, short duration)
    spike_scale_idx = np.where((scales >= 1) & (scales <= 3))[0]
    if len(spike_scale_idx) > 0:
        spike_energy = np.mean(np.abs(cwt_matrix[spike_scale_idx, :]), axis=0)
        spike_peaks, properties = find_peaks(spike_energy, 
                                           height=threshold_factor * np.max(spike_energy), 
                                           distance=SAMPLING_RATE_PER_HOUR // 4,
                                           prominence=0.1 * np.max(spike_energy))
        for peak in spike_peaks:
            detected_patterns.append((peak, "SPIKE_GENERIC"))
    
    # 2. Burst detection (medium frequency, clustered)
    burst_scale_idx = np.where((scales >= 3) & (scales <= 7))[0]
    if len(burst_scale_idx) > 0:
        burst_energy = np.mean(np.abs(cwt_matrix[burst_scale_idx, :]), axis=0)
        burst_peaks, _ = find_peaks(burst_energy, 
                                  height=threshold_factor * np.max(burst_energy),
                                  distance=SAMPLING_RATE_PER_HOUR // 2)
        for peak in burst_peaks:
            # Check if it's a high-frequency burst
            if np.mean(burst_energy[max(0, peak-10):peak+10]) > 0.7 * np.max(burst_energy):
                detected_patterns.append((peak, "BURST_HIGH_FREQ"))
    
    # 3. Prolonged low-frequency patterns
    low_freq_scale_idx = np.where(scales >= 8)[0]
    if len(low_freq_scale_idx) > 0:
        low_freq_energy = np.mean(np.abs(cwt_matrix[low_freq_scale_idx, :]), axis=0)
        # Use moving average to detect prolonged patterns
        smoothed = uniform_filter1d(low_freq_energy, size=SAMPLING_RATE_PER_HOUR)
        prolonged_peaks, _ = find_peaks(smoothed, 
                                      height=threshold_factor * np.max(smoothed),
                                      distance=SAMPLING_RATE_PER_HOUR * 2)
        for peak in prolonged_peaks:
            detected_patterns.append((peak, "PROLONGED_LOW_FREQ"))
    
    # 4. Oscillation detection
    for scale_idx in range(len(scales)):
        scale_signal = cwt_matrix[scale_idx, :]
        # Look for regular oscillations
        autocorr = np.correlate(scale_signal, scale_signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find periodic peaks in autocorrelation
        if len(autocorr) > SAMPLING_RATE_PER_HOUR:
            period_peaks, _ = find_peaks(autocorr[10:SAMPLING_RATE_PER_HOUR], 
                                       height=0.3 * np.max(autocorr))
            if len(period_peaks) >= 2:  # Regular oscillation detected
                # Find time points where this oscillation is strongest
                osc_strength = np.abs(scale_signal)
                osc_peaks, _ = find_peaks(osc_strength, 
                                        height=threshold_factor * np.max(osc_strength),
                                        distance=SAMPLING_RATE_PER_HOUR // 3)
                for peak in osc_peaks:
                    detected_patterns.append((peak, "RAPID_OSCILLATION"))
                break  # Only detect once per signal
    
    return detected_patterns

def generate_enhanced_mycelial_network(num_nodes, connectivity_prob):
    """Generate network with enhanced properties"""
    G = nx.erdos_renyi_graph(num_nodes, connectivity_prob)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        # Add edges to make it connected
        components = list(nx.connected_components(G))
        for i in range(len(components)-1):
            node1 = list(components[i])[0]
            node2 = list(components[i+1])[0]
            G.add_edge(node1, node2)
    
    # Add enhanced edge properties
    for u, v in G.edges():
        G[u][v]['delay'] = np.random.uniform(0.1, 2.0)
        G[u][v]['attenuation'] = np.random.uniform(0.7, 0.95)
        G[u][v]['conductivity'] = np.random.uniform(0.5, 1.0)
        G[u][v]['adaptation_rate'] = np.random.uniform(0.01, 0.1)
    
    return G

def simulate_comprehensive_experiment(rosetta_stone, simulation_id):
    """Run a comprehensive simulation with multiple stimuli and analysis"""
    
    # Generate network and initial signals
    network = generate_enhanced_mycelial_network(NUM_NODES, CONNECTIVITY_PROB)
    
    # Generate base signals with more realistic parameters
    node_voltages = {}
    for node in network.nodes():
        node_voltages[node] = generate_enhanced_v_t(TOTAL_SAMPLES)
    
    # Apply multiple stimuli at different times and locations
    stimuli = [
        ("MECHANICAL_LOAD_ON", np.random.randint(0, NUM_NODES), TOTAL_SAMPLES // 6),
        ("NUTRIENT_GRADIENT", np.random.randint(0, NUM_NODES), TOTAL_SAMPLES // 3),
        ("STRESS_CONDITION", np.random.randint(0, NUM_NODES), TOTAL_SAMPLES // 2),
        ("CHEMICAL_SIGNAL", np.random.randint(0, NUM_NODES), 2 * TOTAL_SAMPLES // 3)
    ]
    
    for stimulus_type, node_idx, start_time in stimuli:
        apply_enhanced_stimulus(node_voltages, stimulus_type, node_idx, start_time, network)
    
    # Propagate signals with enhanced dynamics
    node_voltages = propagate_enhanced_signals(network, node_voltages)
    
    # Compute wavelet transforms
    node_wavelet_transforms = {}
    node_decoded_patterns = {}
    
    for node in network.nodes():
        cwt_matrix = compute_W_kt(node_voltages[node], SCALES)
        node_wavelet_transforms[node] = cwt_matrix
        patterns = enhanced_pattern_detection(cwt_matrix, SCALES, node_voltages[node])
        node_decoded_patterns[node] = patterns
        
        # Record patterns in Rosetta Stone
        for time_sample, pattern_id in patterns:
            context = None
            # Check if pattern is near a stimulus
            for stim_type, stim_node, stim_time in stimuli:
                if abs(time_sample - stim_time) < SAMPLING_RATE_PER_HOUR * 3:  # Within 3 hours
                    context = f"{stim_type}_response"
                    break
            rosetta_stone.record_pattern(pattern_id, node, time_sample, context)
    
    # Analyze temporal sequences
    rosetta_stone.analyze_temporal_sequences(node_decoded_patterns)
    
    # Simulate state transitions
    for node in network.nodes():
        current_state = "IDLE"
        for t in range(0, TOTAL_SAMPLES, SAMPLING_RATE_PER_HOUR):  # Check every hour
            new_state = simulate_enhanced_state_logic(node_decoded_patterns[node], current_state, t, network, node)
            if new_state != current_state:
                rosetta_stone.record_state_transition(current_state, new_state)
                current_state = new_state
    
    return {
        'network': network,
        'voltages': node_voltages,
        'wavelets': node_wavelet_transforms,
        'patterns': node_decoded_patterns,
        'stimuli': stimuli
    }

def generate_enhanced_v_t(total_samples, spike_prob=0.008, noise_level=0.03):
    """Generate more realistic voltage signals"""
    V_t = np.random.normal(0, noise_level, total_samples)
    
    # Add baseline drift
    baseline_drift = 0.01 * np.sin(2 * np.pi * np.arange(total_samples) / (24 * SAMPLING_RATE_PER_HOUR))
    V_t += baseline_drift
    
    # Add spikes with varying characteristics
    for i in range(total_samples):
        if np.random.rand() < spike_prob:
            amplitude = np.random.lognormal(mean=-1, sigma=0.5)  # More realistic amplitude distribution
            duration_samples = int(np.random.exponential(3) * SAMPLING_RATE_PER_HOUR)
            duration_samples = min(duration_samples, total_samples - i)
            
            # Varied spike shapes
            spike_type = np.random.choice(['gaussian', 'exponential', 'oscillatory'])
            
            if spike_type == 'gaussian':
                x = np.linspace(-2, 2, duration_samples)
                spike_shape = amplitude * np.exp(-x**2)
            elif spike_type == 'exponential':
                x = np.linspace(0, 3, duration_samples)
                spike_shape = amplitude * np.exp(-x)
            else:  # oscillatory
                x = np.linspace(0, 2*np.pi, duration_samples)
                spike_shape = amplitude * np.sin(x) * np.exp(-x/5)
            
            end_idx = min(i + duration_samples, total_samples)
            V_t[i:end_idx] += spike_shape[0:end_idx-i]
    
    return V_t

def apply_enhanced_stimulus(node_voltages, stimulus_type, node_idx, start_time_sample, network):
    """Apply stimulus with realistic propagation characteristics"""
    
    stimulus_patterns = {
        "MECHANICAL_LOAD_ON": {"amplitude": 2.0, "duration_hours": 3, "pattern": "step"},
        "NUTRIENT_GRADIENT": {"amplitude": 1.2, "duration_hours": 8, "pattern": "ramp"},
        "STRESS_CONDITION": {"amplitude": 0.8, "duration_hours": 12, "pattern": "oscillatory"},
        "CHEMICAL_SIGNAL": {"amplitude": 1.5, "duration_hours": 2, "pattern": "pulse"}
    }
    
    if stimulus_type not in stimulus_patterns:
        return
    
    params = stimulus_patterns[stimulus_type]
    duration_samples = int(params["duration_hours"] * SAMPLING_RATE_PER_HOUR)
    amplitude = params["amplitude"]
    
    end_idx = min(start_time_sample + duration_samples, TOTAL_SAMPLES)
    t_range = np.arange(end_idx - start_time_sample)
    
    if params["pattern"] == "step":
        stimulus_signal = amplitude * np.ones(len(t_range))
    elif params["pattern"] == "ramp":
        stimulus_signal = amplitude * t_range / len(t_range)
    elif params["pattern"] == "oscillatory":
        stimulus_signal = amplitude * np.sin(2 * np.pi * t_range / (SAMPLING_RATE_PER_HOUR * 2))
    else:  # pulse
        stimulus_signal = amplitude * np.exp(-t_range / (SAMPLING_RATE_PER_HOUR * 0.5))
    
    node_voltages[node_idx][start_time_sample:end_idx] += stimulus_signal

def propagate_enhanced_signals(network, node_voltages):
    """Enhanced signal propagation with adaptation"""
    new_node_voltages = {node: np.copy(node_voltages[node]) for node in network.nodes()}
    
    for t in range(1, TOTAL_SAMPLES):
        for node in network.nodes():
            incoming_signal = 0.0
            for neighbor in network.neighbors(node):
                if network.has_edge(node, neighbor):
                    edge_data = network[node][neighbor]
                    delay_samples = int(edge_data['delay'] * SAMPLING_RATE_PER_HOUR)
                    attenuation = edge_data['attenuation']
                    conductivity = edge_data['conductivity']
                    
                    if t - delay_samples >= 0:
                        signal_strength = node_voltages[neighbor][t - delay_samples]
                        incoming_signal += signal_strength * attenuation * conductivity
            
            # Enhanced integration with nonlinear dynamics
            current_voltage = new_node_voltages[node][t]
            integration_factor = 0.7  # Base integration
            
            # Nonlinear response
            if abs(incoming_signal) > 0.5:  # Threshold for nonlinear response
                integration_factor = 0.9
            
            new_node_voltages[node][t] = (
                current_voltage * (1 - integration_factor) + 
                incoming_signal * integration_factor
            )
    
    return new_node_voltages

def compute_W_kt(V_t_signal, scales):
    """Compute wavelet transform with enhanced processing"""
    # Preprocess signal
    signal = V_t_signal - np.mean(V_t_signal)  # Remove DC component
    
    # Apply Ricker wavelet transform
    cwtmatr = cwt(signal, ricker, scales)
    
    return cwtmatr

def simulate_enhanced_state_logic(node_patterns, current_state, time_sample, network, node_id):
    """Enhanced state logic with more sophisticated decision making"""
    
    # Get recent patterns (within last 2 hours)
    recent_patterns = [pattern_id for t, pattern_id in node_patterns 
                      if abs(t - time_sample) < 2 * SAMPLING_RATE_PER_HOUR]
    
    # State transition rules
    if "BURST_HIGH_FREQ" in recent_patterns:
        if current_state in ["IDLE", "SEARCHING"]:
            return "COMMUNICATING"
    
    if "PROLONGED_LOW_FREQ" in recent_patterns:
        if current_state != "STRESS_RESPONSE":
            return "STRESS_RESPONSE"
    
    if "RAPID_OSCILLATION" in recent_patterns:
        if current_state in ["IDLE", "COMMUNICATING"]:
            return "RESOURCE_SHARING"
    
    if "SPIKE_GENERIC" in recent_patterns:
        if current_state == "IDLE":
            return "SEARCHING"
        elif current_state == "SEARCHING":
            return "GROWING"
    
    # Default transitions back to IDLE
    if len(recent_patterns) == 0 and current_state != "IDLE":
        return "IDLE"
    
    return current_state

# Main execution
def main():
    print("Starting Comprehensive Mycelial Network Analysis...")
    print(f"Running {NUM_SIMULATIONS} simulations...")
    
    # Initialize Rosetta Stone
    rosetta_stone = MycelialRosettaStone()
    
    simulation_results = []
    
    # Run multiple simulations
    for sim_id in range(NUM_SIMULATIONS):
        print(f"Running simulation {sim_id + 1}/{NUM_SIMULATIONS}...")
        result = simulate_comprehensive_experiment(rosetta_stone, sim_id)
        simulation_results.append(result)
    
    # Build the Rosetta Stone
    print("Building Rosetta Stone...")
    rosetta_mappings = rosetta_stone.build_rosetta_stone()
    
    # Generate comprehensive analysis
    analysis = generate_comprehensive_analysis(rosetta_stone, simulation_results)
    
    return rosetta_stone, analysis, simulation_results

def generate_comprehensive_analysis(rosetta_stone, simulation_results):
    """Generate comprehensive analysis of the simulation results"""
    
    analysis = {
        'pattern_analysis': {},
        'network_dynamics': {},
        'temporal_relationships': {},
        'stimulus_response_mapping': {},
        'predictive_framework': {}
    }
    
    # Pattern frequency analysis
    total_patterns = sum(rosetta_stone.pattern_frequency.values())
    analysis['pattern_analysis'] = {
        'total_patterns_detected': total_patterns,
        'patterns_per_simulation': total_patterns / NUM_SIMULATIONS,
        'pattern_distribution': dict(rosetta_stone.pattern_frequency),
        'most_common_patterns': sorted(rosetta_stone.pattern_frequency.items(), 
                                     key=lambda x: x[1], reverse=True)[:5],
        'pattern_diversity': len(rosetta_stone.pattern_frequency)
    }
    
    # Temporal sequence analysis
    sequences = rosetta_stone.temporal_pattern_sequences
    sequence_counter = Counter([seq['sequence'] for seq in sequences])
    analysis['temporal_relationships'] = {
        'total_sequences': len(sequences),
        'unique_sequences': len(sequence_counter),
        'most_common_sequences': dict(sequence_counter.most_common(10)),
        'average_interval': np.mean([seq['interval'] for seq in sequences]) / SAMPLING_RATE_PER_HOUR
    }
    
    # State transition analysis
    analysis['network_dynamics'] = {
        'state_transitions': dict(rosetta_stone.state_transitions),
        'most_stable_states': {},
        'most_active_transitions': {}
    }
    
    # Calculate state stability
    for state, transitions in rosetta_stone.state_transitions.items():
        total_transitions = sum(transitions.values())
        if total_transitions > 0:
            self_transitions = transitions.get(state, 0)
            stability = self_transitions / total_transitions
            analysis['network_dynamics']['most_stable_states'][state] = stability
    
    # Stimulus-response mapping
    analysis['stimulus_response_mapping'] = dict(rosetta_stone.stimulus_response_patterns)
    
    return analysis

# Run the comprehensive analysis
if __name__ == "__main__":
    rosetta_stone, analysis, simulation_results = main()
    
    # Print summary
    print("\n" + "="*80)
    print("MYCELIAL NETWORK ROSETTA STONE - COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    print(f"\nTOTAL SIMULATIONS: {NUM_SIMULATIONS}")
    print(f"TOTAL PATTERNS DETECTED: {analysis['pattern_analysis']['total_patterns_detected']}")
    print(f"PATTERNS PER SIMULATION: {analysis['pattern_analysis']['patterns_per_simulation']:.2f}")
    
    print(f"\nMOST COMMON PATTERNS:")
    for pattern, count in analysis['pattern_analysis']['most_common_patterns']:
        meaning = ENHANCED_FUNGAL_VOCABULARY.get(pattern, "UNKNOWN")
        print(f"  {pattern}: {count} occurrences ({meaning})")
    
    print(f"\nMOST COMMON TEMPORAL SEQUENCES:")
    for sequence, count in list(analysis['temporal_relationships']['most_common_sequences'].items())[:5]:
        print(f"  {sequence}: {count} occurrences")
    
    print(f"\nAVERAGE TIME BETWEEN SEQUENTIAL PATTERNS: {analysis['temporal_relationships']['average_interval']:.2f} hours")
    
    print("\nAnalysis complete! Detailed results stored in analysis dictionary.")