import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqrt_wavelet import SqrtWaveletTransform

def main():
    # Setup paths
    base_dir = Path('/home/kronos/AVALON')
    csv_data_dir = base_dir / 'fungal_networks/csv_data'
    experimental_dir = base_dir / '15061491'
    output_dir = base_dir / 'fungal_wavelet_analysis/results'
    output_dir.mkdir(exist_ok=True)

    # Initialize analyzers
    wavelet = SqrtWaveletTransform()

    print("Starting enhanced wavelet analysis...")
    
    # Process temporal experiments
    print("\n1. Analyzing temporal experiments...")
    temporal_files = [
        experimental_dir / 'Activity_time_part1.csv',
        experimental_dir / 'Activity_time_part2.csv',
        experimental_dir / 'Activity_time_part3.csv'
    ]

    for file in temporal_files:
        if file.exists():
            print(f"\nProcessing {file.name}...")
            try:
                data = pd.read_csv(file)
                signal = data.iloc[:, 1].values  # Assuming second column is voltage
                
                # Perform wavelet transform
                coeffs = wavelet.transform(signal)
                
                # Plot results
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                plt.plot(signal)
                plt.title(f'Raw Signal - {file.name}')
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                
                plt.subplot(2, 1, 2)
                plt.imshow(np.abs(coeffs), aspect='auto', cmap='jet')
                plt.colorbar(label='Magnitude')
                plt.title('Wavelet Transform')
                plt.xlabel('Time')
                plt.ylabel('Scale')
                
                plt.tight_layout()
                plt.savefig(output_dir / f'{file.stem}_analysis.png')
                plt.close()
                
                print("Analysis complete. Results saved.")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

    # Process environmental response data
    print("\n2. Analyzing environmental response data...")
    response_files = [
        experimental_dir / 'Activity_pause_spray.csv',
        experimental_dir / 'New_Oyster_with spray.csv'
    ]

    for file in response_files:
        if file.exists():
            print(f"\nProcessing {file.name}...")
            try:
                data = pd.read_csv(file)
                signal = data.iloc[:, 1].values
                
                # Perform wavelet transform
                coeffs = wavelet.transform(signal)
                
                # Plot results
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                plt.plot(signal)
                plt.title(f'Raw Signal - {file.name}')
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                
                plt.subplot(2, 1, 2)
                plt.imshow(np.abs(coeffs), aspect='auto', cmap='jet')
                plt.colorbar(label='Magnitude')
                plt.title('Wavelet Transform')
                plt.xlabel('Time')
                plt.ylabel('Scale')
                
                plt.tight_layout()
                plt.savefig(output_dir / f'{file.stem}_analysis.png')
                plt.close()
                
                print("Analysis complete. Results saved.")
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

    print("\nAll analyses complete. Results saved in:", output_dir)

if __name__ == "__main__":
    main() 