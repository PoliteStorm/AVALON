"""
Fungal Trait Integration Module

Integrates electrical activity analysis with the fungaltraits database
to provide ecological context and enhanced statistical analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import json
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import warnings

class FungalTraitIntegrator:
    """Integrate electrical activity data with fungal trait database."""
    
    def __init__(self, traits_data_path: Optional[str] = None):
        """
        Initialize with path to fungaltraits data.
        
        Args:
            traits_data_path: Path to fungaltraits CSV file or URL
        """
        self.traits_data_path = traits_data_path
        self.traits_df = None
        self.electrical_data = {}
        self.species_mapping = {
            # Map your recording filenames to standardized species names
            'Blue_oyster': 'Pleurotus ostreatus',
            'Hericium': 'Hericium erinaceus',
            'New_Oyster': 'Pleurotus ostreatus',
            # Add more mappings as needed
        }
        
    def load_fungal_traits(self) -> pd.DataFrame:
        """Load fungal traits database."""
        if self.traits_data_path is None:
            # Default to GitHub URL for fungaltraits
            url = "https://github.com/traitecoevo/fungaltraits/raw/master/funtothefun.csv"
            try:
                self.traits_df = pd.read_csv(url)
                print(f"Loaded {len(self.traits_df)} trait records from fungaltraits database")
            except Exception as e:
                print(f"Error loading from URL: {e}")
                print("Please download the fungaltraits data manually")
                return None
        else:
            self.traits_df = pd.read_csv(self.traits_data_path)
            
        return self.traits_df
    
    def load_electrical_data(self, analysis_results_dir: str) -> Dict:
        """Load electrical activity analysis results."""
        results_dir = Path(analysis_results_dir)
        
        for json_file in results_dir.glob("*_analysis.json"):
            recording_name = json_file.stem.replace("_analysis", "")
            
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Extract key electrical metrics
            electrical_metrics = {
                'spike_count': data.get('spikes', {}).get('count', 0),
                'mean_interval': data.get('spikes', {}).get('mean_interval', 0),
                'complexity': data.get('linguistic', {}).get('theta_1.0', {}).get('complexity', {}).get('normalized_complexity', 0),
                'entropy': data.get('linguistic', {}).get('theta_1.0', {}).get('complexity', {}).get('entropy', 0),
                'word_length': data.get('linguistic', {}).get('theta_1.0', {}).get('word_stats', {}).get('avg_word_length', 0),
                'vocabulary_size': data.get('linguistic', {}).get('theta_1.0', {}).get('word_stats', {}).get('vocabulary_size', 0)
            }
            
            self.electrical_data[recording_name] = electrical_metrics
            
        print(f"Loaded electrical data for {len(self.electrical_data)} recordings")
        return self.electrical_data
    
    def map_species_to_traits(self) -> pd.DataFrame:
        """Map electrical recordings to trait data by species."""
        if self.traits_df is None:
            self.load_fungal_traits()
            
        mapped_data = []
        
        for recording, electrical_metrics in self.electrical_data.items():
            # Extract species from recording name
            species = self._extract_species(recording)
            
            if species:
                # Find matching trait data
                trait_matches = self.traits_df[
                    self.traits_df['species'].str.contains(species, case=False, na=False)
                ]
                
                if not trait_matches.empty:
                    # Aggregate trait data for this species
                    aggregated_traits = self._aggregate_species_traits(trait_matches)
                    
                    # Combine with electrical data
                    combined_data = {
                        'recording': recording,
                        'species': species,
                        **electrical_metrics,
                        **aggregated_traits
                    }
                    mapped_data.append(combined_data)
                    
        return pd.DataFrame(mapped_data)
    
    def _extract_species(self, recording_name: str) -> Optional[str]:
        """Extract species name from recording filename."""
        # Check manual mapping first
        for key, species in self.species_mapping.items():
            if key.lower() in recording_name.lower():
                return species
                
        # Try to extract from filename patterns
        if 'oyster' in recording_name.lower():
            return 'Pleurotus ostreatus'
        elif 'hericium' in recording_name.lower():
            return 'Hericium erinaceus'
            
        return None
    
    def _aggregate_species_traits(self, trait_matches: pd.DataFrame) -> Dict:
        """Aggregate trait measurements for a species."""
        aggregated = {}
        
        # Numeric traits to aggregate
        numeric_traits = [
            'spore_length', 'spore_width', 'N', 'P', 'K', 'Ca', 'Mg',
            'growth_rate', 'biomass', 'enzyme_activity'
        ]
        
        for trait in numeric_traits:
            if trait in trait_matches.columns:
                values = pd.to_numeric(trait_matches[trait], errors='coerce')
                if not values.isna().all():
                    aggregated[f'{trait}_mean'] = values.mean()
                    aggregated[f'{trait}_std'] = values.std()
                    aggregated[f'{trait}_count'] = values.count()
        
        # Categorical traits
        if 'primary_lifestyle' in trait_matches.columns:
            lifestyle = trait_matches['primary_lifestyle'].mode()
            aggregated['primary_lifestyle'] = lifestyle[0] if len(lifestyle) > 0 else 'unknown'
            
        return aggregated
    
    def analyze_trait_correlations(self, merged_df: pd.DataFrame) -> Dict:
        """Analyze correlations between electrical and trait measurements."""
        correlations = {}
        
        electrical_vars = ['spike_count', 'mean_interval', 'complexity', 'entropy']
        trait_vars = [col for col in merged_df.columns if col.endswith('_mean')]
        
        for elec_var in electrical_vars:
            if elec_var in merged_df.columns:
                for trait_var in trait_vars:
                    if trait_var in merged_df.columns:
                        # Remove NaN values
                        valid_data = merged_df[[elec_var, trait_var]].dropna()
                        
                        if len(valid_data) > 3:  # Minimum for correlation
                            r, p = pearsonr(valid_data[elec_var], valid_data[trait_var])
                            correlations[f'{elec_var}_vs_{trait_var}'] = {
                                'correlation': r,
                                'p_value': p,
                                'n_samples': len(valid_data)
                            }
        
        return correlations
    
    def plot_trait_electrical_correlations(self, merged_df: pd.DataFrame, 
                                         save_path: Optional[str] = None):
        """Create correlation plots between traits and electrical activity."""
        plt.figure(figsize=(15, 10))
        
        # Select key variables for plotting
        electrical_vars = ['spike_count', 'complexity', 'entropy']
        trait_vars = [col for col in merged_df.columns if col.endswith('_mean')][:6]  # Top 6 traits
        
        plot_count = 1
        for i, elec_var in enumerate(electrical_vars):
            for j, trait_var in enumerate(trait_vars):
                if plot_count <= 9:  # 3x3 grid
                    plt.subplot(3, 3, plot_count)
                    
                    # Remove NaN values
                    valid_data = merged_df[[elec_var, trait_var]].dropna()
                    
                    if len(valid_data) > 1:
                        plt.scatter(valid_data[trait_var], valid_data[elec_var], 
                                  alpha=0.7, s=60)
                        
                        # Add trend line if enough points
                        if len(valid_data) > 3:
                            z = np.polyfit(valid_data[trait_var], valid_data[elec_var], 1)
                            p = np.poly1d(z)
                            plt.plot(valid_data[trait_var], p(valid_data[trait_var]), 
                                   "r--", alpha=0.8)
                    
                    plt.xlabel(trait_var.replace('_mean', '').replace('_', ' ').title())
                    plt.ylabel(elec_var.replace('_', ' ').title())
                    plt.title(f'{elec_var} vs {trait_var}', fontsize=10)
                    
                    plot_count += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_integrated_report(self, output_path: str):
        """Generate a comprehensive report integrating traits and electrical data."""
        # Load and merge data
        merged_df = self.map_species_to_traits()
        correlations = self.analyze_trait_correlations(merged_df)
        
        report = ["# Integrated Fungal Traits and Electrical Activity Analysis\n"]
        
        # Summary statistics
        report.append("## Summary Statistics")
        report.append(f"- Total recordings analyzed: {len(self.electrical_data)}")
        report.append(f"- Species with trait data: {merged_df['species'].nunique()}")
        report.append(f"- Significant correlations found: {sum(1 for c in correlations.values() if c['p_value'] < 0.05)}")
        
        # Top correlations
        report.append("\n## Significant Trait-Electrical Correlations")
        significant_corrs = {k: v for k, v in correlations.items() if v['p_value'] < 0.05}
        
        for corr_name, stats in sorted(significant_corrs.items(), 
                                     key=lambda x: abs(x[1]['correlation']), reverse=True):
            report.append(f"- **{corr_name}**: r = {stats['correlation']:.3f}, p = {stats['p_value']:.3f}, n = {stats['n_samples']}")
        
        # Species breakdown
        report.append("\n## Species Analysis")
        for species in merged_df['species'].unique():
            species_data = merged_df[merged_df['species'] == species]
            report.append(f"\n### {species}")
            report.append(f"- Recordings: {len(species_data)}")
            report.append(f"- Mean spike count: {species_data['spike_count'].mean():.1f} ± {species_data['spike_count'].std():.1f}")
            report.append(f"- Mean complexity: {species_data['complexity'].mean():.3f} ± {species_data['complexity'].std():.3f}")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
            
        return merged_df, correlations

def main():
    """Example usage of the trait integration module."""
    integrator = FungalTraitIntegrator()
    
    # Load data
    integrator.load_electrical_data("fungal_analysis/enhanced_analysis_results")
    
    # Generate integrated analysis
    merged_df, correlations = integrator.generate_integrated_report(
        "fungal_analysis/trait_electrical_integration_report.md"
    )
    
    # Create correlation plots
    integrator.plot_trait_electrical_correlations(
        merged_df, 
        "fungal_analysis/visualizations/trait_electrical_correlations.png"
    )
    
    print("Trait integration analysis complete!")

if __name__ == "__main__":
    main() 