from data_loader import FungalDataLoader
from signal_processing import FungalSignalProcessor
from visualization import SignalVisualizer
import numpy as np
import sys

def analyze_recording(filename: str):
    """
    Analyze a single fungal recording using wavelet transform
    
    Args:
        filename: Name of the recording file to analyze
    """
    # Initialize components
    loader = FungalDataLoader()
    
    try:
        # Load data
        print(f"\nAnalyzing recording: {filename}")
        print("Loading data...")
        data, sampling_rate = loader.load_recording(filename)
        print(f"Loaded {len(data)} samples at {sampling_rate} Hz")
        
        # Initialize processor and visualizer
        processor = FungalSignalProcessor(sampling_rate)
        visualizer = SignalVisualizer(sampling_rate)
        
        # Preprocess signal
        print("Preprocessing signal...")
        processed_data = processor.preprocess_signal(data)
        
        # Detect spikes using wavelet transform
        print("Detecting spikes...")
        spike_indices = processor.detect_spikes_wavelet(processed_data)
        print(f"Found {len(spike_indices)} spikes")
        
        # Compute wavelet transform
        print("Computing wavelet transform...")
        coefficients, frequencies = processor.compute_cwt(processed_data)
        print(f"Analyzed {len(frequencies)} frequency scales")
        
        # Extract features
        if len(spike_indices) > 0:
            print("Extracting spike features...")
            features = processor.extract_spike_features(processed_data, spike_indices)
            print(f"Extracted {features.shape[1]} features from each spike")
        
        # Visualize results
        print("Generating visualizations...")
        
        # Plot original signal with detected spikes
        visualizer.plot_signal_with_spikes(processed_data, spike_indices,
                                         title=f"Detected Spikes in {filename}")
        
        # Plot wavelet transform
        visualizer.plot_wavelet_transform(coefficients, frequencies)
        
        # Plot spike features if spikes were found
        if len(spike_indices) > 0:
            visualizer.plot_spike_features(features, frequencies)
            visualizer.plot_spike_comparison(processed_data, spike_indices)
            
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error analyzing {filename}: {str(e)}")
        
def main():
    """
    Main function to analyze all available recordings
    """
    loader = FungalDataLoader()
    recordings = loader.get_available_recordings()
    
    print(f"Found {len(recordings)} recordings to analyze")
    
    for recording in recordings:
        try:
            analyze_recording(recording)
        except KeyboardInterrupt:
            print("\nAnalysis interrupted by user")
            sys.exit(0)
        except Exception as e:
            print(f"Error analyzing {recording}: {str(e)}")
            continue
            
if __name__ == "__main__":
    main() 