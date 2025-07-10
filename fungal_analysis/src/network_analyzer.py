import numpy as np
import scipy.io as sio
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class NetworkAnalyzer:
    def __init__(self, network_dir: str = "/home/kronos/AVALON/fungal_networks"):
        self.network_dir = Path(network_dir)
        
    def parse_filename(self, filename: str) -> dict:
        """Parse metadata from filename"""
        parts = filename.replace('.mat', '').split('_')
        return {
            'species': parts[0],
            'scale': parts[1],
            'treatment': parts[2],
            'condition': parts[3],
            'state': parts[4],
            'time': parts[5],
            'replicate': parts[6]
        }
        
    def read_mat_file(self, filepath: Path) -> dict:
        """Read MATLAB file and return its contents"""
        try:
            data = sio.loadmat(filepath)
            return data
        except Exception as e:
            print(f"Error reading {filepath.name}: {str(e)}")
            return None
            
    def analyze_network_structure(self):
        """Analyze network files and their structure"""
        metadata = []
        file_structures = {}
        
        # Process each file
        for file in self.network_dir.glob('*.mat'):
            try:
                # Parse metadata
                meta = self.parse_filename(file.name)
                meta['file_size'] = file.stat().st_size
                metadata.append(meta)
                
                # Read file structure
                data = self.read_mat_file(file)
                if data is not None:
                    structure = {k: type(v).__name__ for k, v in data.items() 
                               if not k.startswith('__')}
                    if str(structure) not in file_structures:
                        file_structures[str(structure)] = []
                    file_structures[str(structure)].append(file.name)
                    
            except Exception as e:
                print(f"Error processing {file.name}: {str(e)}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(metadata)
        
        # Generate report
        report = ["=== Fungal Network Dataset Analysis ===\n"]
        
        # Species summary
        report.extend([
            "\nSpecies Distribution:",
            df['species'].value_counts().to_string()
        ])
        
        # Time ranges
        report.extend([
            "\nTime Ranges:",
            df.groupby('species')['time'].agg(['nunique', 'min', 'max']).to_string()
        ])
        
        # File structures
        report.extend([
            "\nFile Structure Types:",
            *[f"\nStructure {i+1}: {k}\nFiles: {len(v)}"
              for i, (k, v) in enumerate(file_structures.items())]
        ])
        
        # Save report
        with open(self.network_dir / 'network_analysis_report.txt', 'w') as f:
            f.write('\n'.join(report))
            
        return df, file_structures
        
    def convert_to_csv(self, output_dir: Path = None):
        """Convert MAT files to CSV format where possible"""
        if output_dir is None:
            output_dir = self.network_dir / 'csv_data'
        output_dir.mkdir(exist_ok=True)
        
        for file in self.network_dir.glob('*.mat'):
            try:
                data = self.read_mat_file(file)
                if data is not None:
                    # Process each non-metadata field
                    for key, value in data.items():
                        if not key.startswith('__'):
                            if isinstance(value, np.ndarray):
                                # Convert to DataFrame
                                df = pd.DataFrame(value)
                                # Save to CSV
                                csv_name = f"{file.stem}_{key}.csv"
                                df.to_csv(output_dir / csv_name, index=False)
                                print(f"Converted {file.name} - {key} to CSV")
                            
            except Exception as e:
                print(f"Error converting {file.name}: {str(e)}")
                continue

def main():
    analyzer = NetworkAnalyzer()
    df, structures = analyzer.analyze_network_structure()
    print("\nAnalysis complete. Check network_analysis_report.txt for details.")
    
    # Convert sample file to CSV
    analyzer.convert_to_csv()
    print("\nConversion to CSV complete.")

if __name__ == "__main__":
    main() 