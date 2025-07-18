import json
from pathlib import Path
import numpy as np

def generate_complex_signal(t, base_freq, num_spikes, spike_amp, noise_level):
    """Generates a more realistic signal with noise and events."""
    base_signal = 0.1 * np.sin(2 * np.pi * base_freq * t)
    spikes = np.zeros_like(t)
    spike_times = np.random.choice(t, num_spikes, replace=False)
    for st in spike_times:
        idx = np.argmin(np.abs(t - st))
        spike_len = np.random.randint(5, 15)
        spikes[idx:idx + spike_len] = spike_amp * np.exp(-np.linspace(0, 3, spike_len))
    noise = np.random.normal(0, noise_level, len(t))
    return base_signal + spikes + noise

def generate_spatial_data(t, n_points, growth_rate):
    """Generates realistic spatial growth data."""
    base_pos = np.random.rand(n_points, 2) * 10
    spatial_coords = np.zeros((len(t), n_points, 2))
    spatial_coords[0] = base_pos
    rng = np.random.default_rng()
    for idx in range(1, len(t)):
        directions = rng.normal(0, 1, (n_points, 2))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9
        spatial_coords[idx] = spatial_coords[idx - 1] + directions * growth_rate * (1 + 0.1 * np.sin(t[idx]))
    return spatial_coords

def main():
    """Generates and saves synthetic 'real' data for three replicates."""
    t = np.linspace(0, 300, 3000) # 300s at 10Hz

    for i in range(1, 4):
        replicate_path = Path(f"data/real/replicate_{i}")
        
        # Electrochemical
        electro_data = {
            "time": t.tolist(),
            "voltage": generate_complex_signal(t, 0.05, 15, 0.4, 0.02).tolist()
        }
        with open(replicate_path / "electro.json", "w") as f:
            json.dump(electro_data, f)

        # Acoustic
        acoustic_data = {
            "time": t.tolist(),
            "signal": generate_complex_signal(t, 0.2, 20, 0.15, 0.01).tolist()
        }
        with open(replicate_path / "acoustic.json", "w") as f:
            json.dump(acoustic_data, f)

        # Spatial
        spatial_data = {
            "time": t.tolist(),
            "coords": generate_spatial_data(t, 25, 0.005).tolist()
        }
        with open(replicate_path / "spatial.json", "w") as f:
            json.dump(spatial_data, f)
        
        print(f"Generated data for replicate {i}")

if __name__ == "__main__":
    main() 