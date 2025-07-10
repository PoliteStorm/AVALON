import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sqrt_wavelet import SqrtWaveletTransform

def process_file(file_path: str):
    """Process a single CSV file and generate wavelet analysis."""
    
    print(f"Processing file: {file_path}")
    
    # Read data
    data = pd.read_csv(file_path)
    signal = data.iloc[:, 1].values  # Assuming second column is voltage
    
    # Initialize wavelet transform
    wavelet = SqrtWaveletTransform(num_scales=50)
    
    # Perform transform
    print("Performing wavelet transform...")
    coeffs = wavelet.transform(signal)
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title(f'Raw Signal - {Path(file_path).name}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 1, 2)
    plt.imshow(np.abs(coeffs), aspect='auto', cmap='jet')
    plt.colorbar(label='Magnitude')
    plt.title('Wavelet Transform')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    
    plt.tight_layout()
    output_file = output_dir / f"{Path(file_path).stem}_analysis.png"
    plt.savefig(output_file)
    plt.close()
    
    print(f"Analysis complete. Results saved to: {output_file}")

if __name__ == "__main__":
    # Process a sample file
    file_path = "/home/kronos/AVALON/15061491/Activity_time_part1.csv"
    process_file(file_path) 