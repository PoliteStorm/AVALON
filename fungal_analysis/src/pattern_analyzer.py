import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import gc
import psutil
import os
from tqdm import tqdm

class PatternAnalyzer:
    def __init__(self, base_dir: str = "/home/kronos/AVALON/15061491/fungal_spikes/wavelet_analysis"):
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / "pattern_analysis"
        self.output_dir.mkdir(exist_ok=True)
        self.chunk_size = 100000  # Process events in chunks
        
    def _get_memory_usage(self) -> str:
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / 1024 / 1024  # Convert to MB
        return f"{mem:.1f} MB"
        
    def analyze_intervals(self, file_dir: Path) -> Dict:
        """Analyze the temporal patterns in event intervals"""
        print(f"\nMemory usage before loading: {self._get_memory_usage()}")
        
        # Count total lines first
        total_events = sum(1 for _ in open(file_dir / 'event_times.txt'))
        print(f"Total events to process: {total_events:,}")
        
        # Initialize accumulators
        intervals_sum = 0
        intervals_sq_sum = 0
        min_interval = float('inf')
        max_interval = float('-inf')
        n_intervals = 0
        burst_count = 0
        
        # Process events in chunks to calculate basic statistics
        print("Pass 1: Calculating basic statistics...")
        prev_event = None
        chunk_intervals = []
        
        with open(file_dir / 'event_times.txt', 'r') as f:
            for line in tqdm(f, total=total_events, desc="Processing events"):
                event_time = float(line.strip())
                
                if prev_event is not None:
                    interval = event_time - prev_event
                    intervals_sum += interval
                    intervals_sq_sum += interval * interval
                    min_interval = min(min_interval, interval)
                    max_interval = max(max_interval, interval)
                    n_intervals += 1
                    
                    # Store recent intervals for autocorrelation
                    chunk_intervals.append(interval)
                    if len(chunk_intervals) > self.chunk_size:
                        chunk_intervals.pop(0)
                
                prev_event = event_time
                
                # Periodically clear memory
                if n_intervals % 1000000 == 0:
                    gc.collect()
                    print(f"Memory usage: {self._get_memory_usage()}")
        
        # Calculate statistics
        mean_interval = intervals_sum / n_intervals
        std_interval = np.sqrt((intervals_sq_sum / n_intervals) - (mean_interval * mean_interval))
        cv = std_interval / mean_interval
        
        # Calculate burst ratio using the last chunk
        burst_threshold = mean_interval - std_interval
        bursts = np.array(chunk_intervals) < burst_threshold
        burst_ratio = np.sum(bursts) / len(chunk_intervals)
        
        print("Calculating rhythmicity...")
        # Calculate rhythmicity using the last chunk for efficiency
        chunk_intervals = np.array(chunk_intervals)
        chunk_intervals = chunk_intervals - np.mean(chunk_intervals)
        acf = np.correlate(chunk_intervals, chunk_intervals, mode='full')
        acf = acf[len(acf)//2:]
        if len(acf) > 1:
            peaks = np.where((acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:]))[0] + 1
            rhythmicity = acf[peaks[0]] / acf[0] if len(peaks) > 0 else 0
        else:
            rhythmicity = 0
            
        print(f"Memory usage after analysis: {self._get_memory_usage()}")
        
        stats_dict = {
            'mean_interval': mean_interval,
            'std_interval': std_interval,
            'cv': cv,
            'min_interval': min_interval,
            'max_interval': max_interval,
            'burst_ratio': burst_ratio,
            'rhythmicity': rhythmicity,
            'total_events': total_events,
            'total_intervals': n_intervals
        }
        
        return stats_dict, np.array(chunk_intervals)  # Return only the last chunk for visualization
        
    def plot_pattern_analysis(self, name: str, intervals: np.ndarray, stats: Dict):
        """Generate visualizations for pattern analysis"""
        print(f"Generating visualizations for {name}...")
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Interval distribution
        plt.subplot(2, 2, 1)
        sns.histplot(intervals, bins=50, kde=True)
        plt.axvline(stats['mean_interval'], color='r', linestyle='--', label='Mean')
        plt.axvline(stats['mean_interval'] - stats['std_interval'], color='g', 
                   linestyle='--', label='Burst threshold')
        plt.title('Interval Distribution (Sample)')
        plt.xlabel('Interval (s)')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot 2: Event raster (sample)
        plt.subplot(2, 2, 2)
        event_times = np.cumsum(intervals[:1000])
        plt.eventplot(event_times, lineoffsets=0.5, linelengths=0.5)
        plt.title('Event Raster (Sample of 1000 Events)')
        plt.xlabel('Time (s)')
        plt.ylabel('Events')
        
        # Plot 3: Autocorrelation
        plt.subplot(2, 2, 3)
        max_lag = min(1000, len(intervals)//2)
        acf = np.correlate(intervals - np.mean(intervals), 
                          intervals - np.mean(intervals), mode='full')
        acf = acf[len(acf)//2:len(acf)//2 + max_lag]
        acf = acf / acf[0]  # Normalize
        plt.plot(np.arange(max_lag), acf)
        plt.title('Interval Autocorrelation')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        
        # Plot 4: Inter-event interval time series
        plt.subplot(2, 2, 4)
        plt.plot(intervals[:1000])
        plt.axhline(stats['mean_interval'], color='r', linestyle='--', label='Mean')
        plt.axhline(stats['mean_interval'] - stats['std_interval'], color='g', 
                   linestyle='--', label='Burst threshold')
        plt.title('Inter-event Interval Time Series (Sample)')
        plt.xlabel('Event Number')
        plt.ylabel('Interval (s)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{name}_pattern_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clear memory
        gc.collect()
        print(f"Memory usage after plotting: {self._get_memory_usage()}")
        
    def analyze_all_recordings(self):
        """Analyze patterns in all recordings"""
        results = []
        
        # Process each recording
        recordings = [d for d in self.base_dir.iterdir() if d.is_dir() and d != self.output_dir]
        print(f"\nFound {len(recordings)} recordings to analyze")
        
        for recording_dir in recordings:
            try:
                print(f"\n{'='*50}")
                print(f"Analyzing patterns in {recording_dir.name}...")
                print(f"{'='*50}")
                
                stats, intervals = self.analyze_intervals(recording_dir)
                stats['recording'] = recording_dir.name
                results.append(stats)
                
                # Generate visualizations
                self.plot_pattern_analysis(recording_dir.name, intervals, stats)
                
                # Clear memory between recordings
                gc.collect()
                print(f"Memory usage after recording: {self._get_memory_usage()}")
                
            except Exception as e:
                print(f"Error analyzing {recording_dir.name}: {str(e)}")
                continue
        
        # Generate summary report
        if results:
            print("\nGenerating summary report...")
            summary = ["=== Spiking Pattern Analysis Summary ===\n"]
            
            for r in sorted(results, key=lambda x: x['rhythmicity'], reverse=True):
                summary.extend([
                    f"\nRecording: {r['recording']}",
                    f"Pattern Characteristics:",
                    f"- Total Events: {r['total_events']:,}",
                    f"- Total Intervals: {r['total_intervals']:,}",
                    f"- Rhythmicity Index: {r['rhythmicity']:.3f}",
                    f"- Burst Ratio: {r['burst_ratio']:.3f}",
                    f"- Mean Interval: {r['mean_interval']:.2f}s",
                    f"- Interval CV: {r['cv']:.3f}",
                    f"- Min Interval: {r['min_interval']:.2f}s",
                    f"- Max Interval: {r['max_interval']:.2f}s"
                ])
            
            with open(self.output_dir / 'pattern_analysis_report.txt', 'w') as f:
                f.write('\n'.join(summary))
            
            print("\nAnalysis complete! Check the pattern_analysis directory for results.")

def main():
    analyzer = PatternAnalyzer()
    analyzer.analyze_all_recordings()

if __name__ == "__main__":
    main() 