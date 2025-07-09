import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker, cwt # Example wavelets for CWT
from scipy.ndimage import uniform_filter1d
import networkx as nx

# --- Configuration ---
SIM_DURATION_HOURS = 100
SAMPLING_RATE_PER_HOUR = 60 # 60 samples per hour
TOTAL_SAMPLES = SIM_DURATION_HOURS * SAMPLING_RATE_PER_HOUR

# Mycelial network parameters
NUM_NODES = 10
CONNECTIVITY_PROB = 0.3 # For a random graph

# Wavelet parameters (example for Morlet or Ricker)
SCALES = np.arange(1, 10, 0.5) # Range of scales (tau in your equation)
FREQUENCIES_K = np.linspace(0.1, 5, 20) # Range of 'k' (pseudo-frequencies)

# --- 1. Data Acquisition and Preprocessing (Simulated Data) ---
def generate_mycelial_network(num_nodes, connectivity_prob):
    G = nx.erdos_renyi_graph(num_nodes, connectivity_prob)
    # Add attributes like resistance or delay to edges
    for u, v in G.edges():
        G[u][v]['delay'] = np.random.uniform(0.1, 1.0) # hours
        G[u][v]['attenuation'] = np.random.uniform(0.8, 1.0) # signal retention
    return G

def generate_v_t(total_samples, spike_prob=0.01, noise_level=0.05,
                 adamatzky_spike_duration_range_hours=(1, 21),
                 adamatzky_amplitude_range_mV=(0.03, 2.1)):
    V_t = np.random.normal(0, noise_level, total_samples)
    
    # Simulate Adamatzky-like spikes
    for i in range(total_samples):
        if np.random.rand() < spike_prob: # Chance of initiating a spike
            amplitude = np.random.uniform(*adamatzky_amplitude_range_mV)
            duration_samples = int(np.random.uniform(*adamatzky_spike_duration_range_hours) * SAMPLING_RATE_PER_HOUR)
            
            # Simple spike shape (e.g., a Gaussian or a half-sine)
            spike_shape = np.sin(np.linspace(0, np.pi, duration_samples)) * amplitude
            
            end_idx = min(i + duration_samples, total_samples)
            V_t[i:end_idx] += spike_shape[0:end_idx-i]
    
    return V_t

# Simulate a basic mycelial network activity
mycelium_graph = generate_mycelial_network(NUM_NODES, CONNECTIVITY_PROB)
node_voltages = {node: generate_v_t(TOTAL_SAMPLES) for node in mycelium_graph.nodes()}

# --- 2. Simulating Stimulus & Propagation (Conceptual) ---
def apply_stimulus(node_voltages, stimulus_type, node_idx, start_time_sample):
    # This is where you'd inject specific Adamatzky-like patterns
    # e.g., 'Stim_Nutrient_A' might add a specific high-frequency burst
    # For now, let's just add a distinct amplitude increase
    spike_duration_samples = int(5 * SAMPLING_RATE_PER_HOUR) # Example
    amplitude_boost = 1.5 # Example
    
    end_idx = min(start_time_sample + spike_duration_samples, TOTAL_SAMPLES)
    node_voltages[node_idx][start_time_sample:end_idx] += (
        np.sin(np.linspace(0, np.pi, spike_duration_samples)) * amplitude_boost
    )[0:end_idx-start_time_sample]
    
    print(f"Stimulus '{stimulus_type}' applied to node {node_idx} at time {start_time_sample/SAMPLING_RATE_PER_HOUR:.2f} hours.")

# Propagate signals (simplified - not a full biophysical model)
def propagate_signals(mycelium_graph, node_voltages):
    new_node_voltages = {node: np.copy(node_voltages[node]) for node in mycelium_graph.nodes()}
    for t in range(1, TOTAL_SAMPLES):
        for node in mycelium_graph.nodes():
            incoming_signal = 0.0
            for neighbor in mycelium_graph.neighbors(node):
                delay_samples = int(mycelium_graph[node][neighbor]['delay'] * SAMPLING_RATE_PER_HOUR)
                attenuation = mycelium_graph[node][neighbor]['attenuation']
                
                if t - delay_samples >= 0:
                    incoming_signal += node_voltages[neighbor][t - delay_samples] * attenuation
            
            # Simple integration: node_voltage = endogenous + incoming
            new_node_voltages[node][t] = (
                (new_node_voltages[node][t] * 0.5) + (incoming_signal * 0.5) # Example weighted average
            )
    return new_node_voltages

# Initial stimulus for demonstration
apply_stimulus(node_voltages, "Mechanical_Load_ON", 0, TOTAL_SAMPLES // 4)
node_voltages = propagate_signals(mycelium_graph, node_voltages) # Update voltages after propagation

# --- 3. Signal Analysis (W(k,τ) - Wavelet Transform) ---
# For simplicity, we'll use a fixed wavelet for now (Ricker, which is Mexican Hat)
def compute_W_kt(V_t_signal, scales):
    # Using scipy.signal.cwt for Continuous Wavelet Transform
    # 'widths' corresponds to scales (tau) for the Ricker wavelet
    widths_for_cwt = scales # In scipy, scales directly map to 'widths' for ricker
    
    # We need to map 'k' (frequency) from the raw CWT output.
    # For Ricker wavelet, 'k' is intrinsically linked to 'width'.
    # For a full 'k' dimension as in your equation, you'd likely use a Morlet wavelet
    # and define k based on its central frequency, or use a custom implementation.
    # For this simulation, we'll implicitly use the frequency content revealed by the scales.
    
    cwtmatr = cwt(V_t_signal, ricker, widths_for_cwt)
    # The output cwtmatr has dimensions (num_scales, num_samples)
    return cwtmatr

# Store wavelet transforms for all nodes
node_wavelet_transforms = {node: compute_W_kt(node_voltages[node], SCALES)
                           for node in mycelium_graph.nodes()}

# --- 4. Spike/Pattern Detection (Post W(k,τ)) ---
# This is a simplified pattern detection based on energy in specific scales
def detect_patterns(cwt_matrix, scales, threshold=0.5):
    detected_patterns = [] # List of (time_sample, pattern_id)
    # Example: Look for high energy at specific scales corresponding to "spike-like" features
    spike_scale_idx = np.where((scales >= 2) & (scales <= 5))[0] # Example: scales 2-5
    
    # Average energy in these scales
    if len(spike_scale_idx) > 0:
        avg_energy = np.mean(np.abs(cwt_matrix[spike_scale_idx, :]), axis=0)
        
        # Simple peak detection
        peaks, _ = scipy.signal.find_peaks(avg_energy, height=threshold * np.max(avg_energy), distance=SAMPLING_RATE_PER_HOUR // 2)
        
        for peak_time in peaks:
            # Here, you'd apply more sophisticated pattern matching
            # E.g., comparing local CWT coefficients around the peak to pre-defined "word" templates
            # For simplicity, we'll assign a generic 'spike' pattern
            detected_patterns.append((peak_time, "SPIKE_GENERIC"))
            
    return detected_patterns

# Detected patterns for each node
node_decoded_spikes = {node: detect_patterns(node_wavelet_transforms[node], SCALES)
                       for node in mycelium_graph.nodes()}

# --- 5. Behavioral Modeling (S(t)) & Rosetta Stone Logic ---
# This is the core of the "Rosetta Stone" and decision making
# Assume a simple state machine or rule-based system for each node
# Define "fungal words" and their "meanings"
FUNGAL_VOCABULARY = {
    "SPIKE_GENERIC": "SIGNAL_TRANSMISSION",
    "BURST_HIGH_FREQ": "NUTRIENT_DETECTED",
    "PROLONGED_LOW_FREQ": "STRESS_CONDITION",
    "ADAMATZKY_ON_SPIKE": "PRESSURE_ON",
    "ADAMATZKY_OFF_SPIKE": "PRESSURE_OFF"
}

# Define simple node states and transitions
NODE_STATES = ["IDLE", "SEARCHING", "DEFENDING", "GROWING"]

# Simulate a simplified S(t) logic for a single node (node 0) as an example
# In a real model, this would be distributed across nodes and interactions
def simulate_S_t_logic(node_decoded_spikes, current_node_state, time_sample):
    
    # s_i(t) for this node (what it's 'sensing' or 'producing')
    current_messages = [FUNGAL_VOCABULARY.get(pattern_id) for t, pattern_id in node_decoded_spikes[0] if t == time_sample]
    
    # τ(s_i, s_j, φ) - Interaction term (simplified for demo)
    # This would typically involve looking at messages from neighboring nodes (s_j)
    # and using their 'meaning' to influence the current node's state
    # For simplicity, let's assume if any neighbor sends a "NUTRIENT_DETECTED" message,
    # it influences the current node to switch to "SEARCHING"
    
    neighbor_messages = []
    for neighbor in mycelium_graph.neighbors(0):
        # Check if neighbor sent a specific message recently
        for t, pattern_id in node_decoded_spikes[neighbor]:
            if time_sample - t < SAMPLING_RATE_PER_HOUR * 2: # Within last 2 hours
                neighbor_messages.append(FUNGAL_VOCABULARY.get(pattern_id))
    
    new_state = current_node_state
    
    if "NUTRIENT_DETECTED" in current_messages or "NUTRIENT_DETECTED" in neighbor_messages:
        new_state = "SEARCHING"
    elif "STRESS_CONDITION" in current_messages:
        new_state = "DEFENDING"
    elif "PRESSURE_ON" in current_messages:
        print(f"Node 0 detected Pressure ON at {time_sample/SAMPLING_RATE_PER_HOUR:.2f} hours. State: {current_node_state}")
        # Could trigger specific action here, e.g., resource reallocation
        new_state = "IDLE" # Or a pressure-response state
    
    # S(t) = overall network action
    # This would be derived from the collective states of all nodes
    # For now, let's just make S(t) the current state of node 0 as a proxy
    return new_state

current_node_0_state = "IDLE"
network_S_t = [] # Represents the collective S(t) over time
for t in range(TOTAL_SAMPLES):
    current_node_0_state = simulate_S_t_logic(node_decoded_spikes, current_node_0_state, t)
    network_S_t.append(current_node_0_state)

# --- 7. Error Detection and Validation ---

def check_for_errors(V_t_original, cwt_matrix, detected_patterns, simulated_actions, expected_actions):
    errors = {}

    # a) Wavelet Reconstruction Error (for W(k,τ)) - Conceptual, as inverse CWT can be complex
    # You'd need an inverse wavelet transform function here.
    # reconstructed_V_t = inverse_cwt(cwt_matrix, wavelet_function, scales)
    # reconstruction_error = np.mean(np.abs(V_t_original - reconstructed_V_t))
    # errors['reconstruction_error'] = reconstruction_error
    # if reconstruction_error > some_threshold:
    #     print(f"Warning: High reconstruction error for W(k,τ): {reconstruction_error:.4f}")

    # b) Pattern Detection Consistency
    # Do detected patterns align with expected patterns from stimulus inputs?
    # This requires knowing when specific stimuli were applied and what patterns they *should* produce.
    # For our simulated data, we only applied one stimulus to node 0.
    expected_pattern_time = TOTAL_SAMPLES // 4 # Time of 'Mechanical_Load_ON' stimulus
    
    # Check if 'ADAMATZKY_ON_SPIKE' (or a similar pattern) was detected at the stimulus time
    pattern_detected_at_stim = False
    for t, pattern_id in node_decoded_spikes[0]:
        if abs(t - expected_pattern_time) < SAMPLING_RATE_PER_HOUR: # Within ~1 hour of stimulus
            if pattern_id == "SPIKE_GENERIC": # Our current simple pattern
                pattern_detected_at_stim = True
                break
    
    if not pattern_detected_at_stim:
        errors['stimulus_pattern_missed'] = "Expected pattern for 'Mechanical_Load_ON' not detected at node 0."
        print(errors['stimulus_pattern_missed'])
    else:
        print("Pattern for 'Mechanical_Load_ON' successfully detected near stimulus time.")

    # c) Action Decoding Accuracy (for S(t) and Rosetta Stone)
    # This requires a pre-defined 'ground truth' of expected network actions.
    # For simplicity, if we applied 'Mechanical_Load_ON', we might expect a state change to 'IDLE' or 'PRESSURE_RESPONSE'
    # immediately after.

    # Example: If 'Mechanical_Load_ON' was detected, was the system state changed accordingly?
    expected_state_change_time = (TOTAL_SAMPLES // 4) + (SAMPLING_RATE_PER_HOUR * 2) # After some propagation delay
    if expected_state_change_time < TOTAL_SAMPLES:
        if network_S_t[expected_state_change_time] == "IDLE": # Based on our demo logic
            print("S(t) logic appears to be functioning as expected for stimulus response.")
        else:
            errors['incorrect_s_t_action'] = "S(t) did not transition to expected state after stimulus."
            print(errors['incorrect_s_t_action'])
            
    # d) Noise/Garbage Check: Are there false positives (patterns detected where no stimulus/known activity occurred)?
    # This requires examining periods of no expected activity.
    false_positives = 0
    for node in mycelium_graph.nodes():
        for t, pattern_id in node_decoded_spikes[node]:
            # If t is far from any known stimulus and not an endogenous spike time
            # This is hard to do without a more complex ground truth of endogenous activity
            pass # Placeholder
            
    if false_positives > 0:
        errors['false_positives'] = f"Detected {false_positives} false positive patterns."
        print(errors['false_positives'])
        
    if not errors:
        print("Simulation and decoding appear to be internally consistent, further validation needed.")
    return errors

# Run error checks
simulation_errors = check_for_errors(node_voltages[0], node_wavelet_transforms[0], node_decoded_spikes, network_S_t, None)

# --- Visualization (Optional but highly recommended) ---
# Plot raw voltage signal for a node
plt.figure(figsize=(15, 6))
plt.subplot(3, 1, 1)
plt.plot(np.array(range(TOTAL_SAMPLES)) / SAMPLING_RATE_PER_HOUR, node_voltages[0])
plt.title(f'Simulated Raw Voltage Signal for Node 0 (Duration: {SIM_DURATION_HOURS} hours)')
plt.xlabel('Time (hours)')
plt.ylabel('Voltage (mV)')

# Plot wavelet transform (scalogram) for a node
plt.subplot(3, 1, 2)
plt.imshow(np.abs(node_wavelet_transforms[0]), extent=[0, SIM_DURATION_HOURS, SCALES[-1], SCALES[0]],
           cmap='jet', aspect='auto', origin='upper')
plt.colorbar(label='Absolute Wavelet Coefficient')
plt.title('Wavelet Transform (W(k,τ)) for Node 0')
plt.xlabel('Time (hours)')
plt.ylabel('Scale (τ)')

# Plot detected patterns and S(t) state for node 0
plt.subplot(3, 1, 3)
plt.plot(np.array(range(TOTAL_SAMPLES)) / SAMPLING_RATE_PER_HOUR, [NODE_STATES.index(s) for s in network_S_t], drawstyle='steps-post')
spike_times = [t / SAMPLING_RATE_PER_HOUR for t, _ in node_decoded_spikes[0]]
plt.vlines(spike_times, ymin=-0.5, ymax=len(NODE_STATES)-0.5, color='r', linestyle='--', label='Detected Spikes')
plt.yticks(range(len(NODE_STATES)), NODE_STATES)
plt.title('Decoded State (S(t) Proxy) for Node 0')
plt.xlabel('Time (hours)')
plt.ylabel('Node State')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print the final errors
print("\n--- Simulation Error Report ---")
if simulation_errors:
    for k, v in simulation_errors.items():
        print(f"- {k}: {v}")
else:
    print("No immediate errors detected by the internal checks.")
print("-----------------------------")