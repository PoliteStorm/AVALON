import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import List, Optional, Tuple

def generate_comparative_visualization(base_dir: str = "/home/kronos/AVALON/15061491/fungal_spikes/wavelet_analysis"):
    base_dir = Path(base_dir)
    species_data = {}
    
    # Load data for each species
    for species_dir in base_dir.iterdir():
        if species_dir.is_dir() and species_dir.name != "pattern_analysis":
            event_times_file = species_dir / "event_times.txt"
            if event_times_file.exists():
                events = np.loadtxt(event_times_file)
                species_data[species_dir.name] = {
                    'events': events,
                    'intervals': np.diff(events) if len(events) > 1 else np.array([])
                }
    
    # Create comparative visualization
    plt.figure(figsize=(20, 15))
    
    # 1. Event Rate Comparison
    plt.subplot(2, 2, 1)
    event_rates = {name: len(data['events'])/data['events'][-1] 
                  for name, data in species_data.items()}
    plt.bar(event_rates.keys(), event_rates.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Event Rate Comparison')
    plt.ylabel('Events per Second')
    
    # 2. Interval Distribution Comparison
    plt.subplot(2, 2, 2)
    for name, data in species_data.items():
        if len(data['intervals']) > 0:
            sns.kdeplot(data['intervals'], label=name)
    plt.title('Interval Distribution Comparison')
    plt.xlabel('Interval (s)')
    plt.ylabel('Density')
    plt.legend()
    
    # 3. Activity Pattern Over Time
    plt.subplot(2, 2, 3)
    for name, data in species_data.items():
        # Plot first hour of activity
        events = data['events']
        mask = events <= 3600  # First hour
        if np.any(mask):
            plt.plot(events[mask], np.arange(np.sum(mask)), label=name)
    plt.title('Cumulative Events (First Hour)')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Events')
    plt.legend()
    
    # 4. Rhythmicity Analysis
    plt.subplot(2, 2, 4)
    for name, data in species_data.items():
        if len(data['intervals']) > 100:
            # Calculate autocorrelation
            intervals = data['intervals'][:1000]  # Use first 1000 intervals
            acf = np.correlate(intervals - np.mean(intervals),
                             intervals - np.mean(intervals), mode='full')
            acf = acf[len(acf)//2:len(acf)//2 + 100]  # Show first 100 lags
            acf = acf / acf[0]  # Normalize
            plt.plot(np.arange(len(acf)), acf, label=name)
    plt.title('Interval Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(base_dir / 'comparative_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_comparative_visualization() 