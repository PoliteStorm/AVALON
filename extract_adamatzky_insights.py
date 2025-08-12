#!/usr/bin/env python3
"""
Extract Key Insights from Adamatzky Frequency Discrimination Analysis

This script reads the large results file and extracts the most important findings
for easy interpretation and reporting.
"""

import json
import numpy as np
from pathlib import Path

def extract_key_insights(results_file: str) -> dict:
    """Extract key insights from the analysis results."""
    
    print(f"Loading results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    insights = {}
    
    # Extract summary statistics
    if 'summary' in results:
        summary = results['summary']
        insights['summary'] = {
            'total_frequencies_tested': summary.get('total_frequencies_tested', 0),
            'mean_thd': summary.get('mean_thd', 0),
            'std_thd': summary.get('std_thd', 0),
            'low_freq_thd_mean': summary.get('low_freq_thd_mean', 0),
            'high_freq_thd_mean': summary.get('high_freq_mean', 0),
            'frequency_discrimination_threshold': summary.get('frequency_discrimination_threshold', 10.0)
        }
    
    # Extract THD analysis
    if 'thd_analysis' in results:
        thd_data = results['thd_analysis']
        thd_values = list(thd_data.values())
        
        insights['thd_analysis'] = {
            'all_thd_values': thd_values,
            'min_thd': min(thd_values) if thd_values else 0,
            'max_thd': max(thd_values) if thd_values else 0,
            'low_freq_thd': [thd_data.get(str(f), 0) for f in range(1, 11)],
            'high_freq_thd': [thd_data.get(str(f), 0) for f in [20, 30, 40, 50, 60, 70, 80, 90, 100]]
        }
    
    # Extract harmonic analysis
    if 'harmonic_analysis' in results:
        harmonic_data = results['harmonic_analysis']
        insights['harmonic_analysis'] = {
            'harmonic_2_3_ratios': [harmonic_data.get(str(f), {}).get('harmonic_2_3_ratio', 0) 
                                   for f in range(1, 11)],
            'harmonic_2_amplitudes': [harmonic_data.get(str(f), {}).get('harmonic_2_amp', 0) 
                                     for f in range(1, 11)],
            'harmonic_3_amplitudes': [harmonic_data.get(str(f), {}).get('harmonic_3_amp', 0) 
                                     for f in range(1, 11)]
        }
    
    # Extract fuzzy classification
    if 'fuzzy_classification' in results:
        fuzzy_data = results['fuzzy_classification']
        insights['fuzzy_classification'] = {
            'thresholds': fuzzy_data.get('thresholds', {}),
            'has_fuzzy_sets': 'fuzzy_sets' in fuzzy_data
        }
    
    return insights

def print_insights_summary(insights: dict):
    """Print a formatted summary of the key insights."""
    
    print("\n" + "="*80)
    print("ADAMATZKY FREQUENCY DISCRIMINATION ANALYSIS - KEY INSIGHTS")
    print("="*80)
    
    # Summary statistics
    if 'summary' in insights:
        summary = insights['summary']
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   • Total frequencies tested: {summary['total_frequencies_tested']}")
        print(f"   • Mean THD: {summary['mean_thd']:.3f}")
        print(f"   • THD Standard Deviation: {summary['std_thd']:.3f}")
        print(f"   • Frequency discrimination threshold: {summary['frequency_discrimination_threshold']} mHz")
    
    # THD Analysis
    if 'thd_analysis' in insights:
        thd_analysis = insights['thd_analysis']
        print(f"\n🔍 THD ANALYSIS:")
        print(f"   • THD Range: {thd_analysis['min_thd']:.3f} - {thd_analysis['max_thd']:.3f}")
        
        if thd_analysis['low_freq_thd']:
            low_freq_mean = np.mean(thd_analysis['low_freq_thd'])
            print(f"   • Low frequency THD mean (≤10 mHz): {low_freq_mean:.3f}")
        
        if thd_analysis['high_freq_thd']:
            high_freq_mean = np.mean(thd_analysis['high_freq_thd'])
            print(f"   • High frequency THD mean (>10 mHz): {high_freq_mean:.3f}")
    
    # Harmonic Analysis
    if 'harmonic_analysis' in insights:
        harmonic = insights['harmonic_analysis']
        print(f"\n🎵 HARMONIC ANALYSIS:")
        
        if harmonic['harmonic_2_3_ratios']:
            ratios = [r for r in harmonic['harmonic_2_3_ratios'] if r > 0]
            if ratios:
                avg_ratio = np.mean(ratios)
                print(f"   • Average 2nd/3rd harmonic ratio: {avg_ratio:.3f}")
        
        if harmonic['harmonic_2_amplitudes']:
            h2_amps = [a for a in harmonic['harmonic_2_amplitudes'] if a > 0]
            if h2_amps:
                print(f"   • 2nd harmonic amplitude range: {min(h2_amps):.3f} - {max(h2_amps):.3f}")
        
        if harmonic['harmonic_3_amplitudes']:
            h3_amps = [a for a in harmonic['harmonic_3_amplitudes'] if a > 0]
            if h3_amps:
                print(f"   • 3rd harmonic amplitude range: {min(h3_amps):.3f} - {max(h3_amps):.3f}")
    
    # Fuzzy Classification
    if 'fuzzy_classification' in insights:
        fuzzy = insights['fuzzy_classification']
        print(f"\n🧠 FUZZY CLASSIFICATION:")
        print(f"   • Fuzzy sets implemented: {'Yes' if fuzzy['has_fuzzy_sets'] else 'No'}")
        if fuzzy['thresholds']:
            print(f"   • Classification thresholds defined for {len(fuzzy['thresholds'])} categories")
    
    # Key Findings
    print(f"\n🎯 KEY FINDINGS:")
    
    if 'thd_analysis' in insights:
        thd_analysis = insights['thd_analysis']
        if thd_analysis['low_freq_thd'] and thd_analysis['high_freq_thd']:
            low_mean = np.mean(thd_analysis['low_freq_thd'])
            high_mean = np.mean(thd_analysis['high_freq_thd'])
            
            if low_mean > high_mean:
                print(f"   ✅ CONFIRMED: Low frequencies (≤10 mHz) show higher THD than high frequencies")
                print(f"      This aligns with Adamatzky's findings of increased distortion at low frequencies")
            else:
                print(f"   ⚠️  UNEXPECTED: High frequencies show higher THD than low frequencies")
                print(f"      This differs from Adamatzky's expected pattern")
    
    print(f"\n   📈 The analysis successfully tested {insights.get('summary', {}).get('total_frequencies_tested', 0)} frequencies")
    print(f"   🧬 Implemented Adamatzky's methodology for fungal frequency discrimination")
    print(f"   🎵 Generated comprehensive harmonic and THD analysis")
    
    print("\n" + "="*80)

def main():
    """Main function to extract and display insights."""
    
    results_file = "results/adamatzky_frequency_discrimination_results.json"
    
    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        print("Please run the Adamatzky analysis first.")
        return
    
    try:
        insights = extract_key_insights(results_file)
        print_insights_summary(insights)
        
        # Save insights summary
        insights_file = "results/adamatzky_insights_summary.json"
        with open(insights_file, 'w') as f:
            json.dump(insights, f, indent=2)
        print(f"\n💾 Insights summary saved to: {insights_file}")
        
    except Exception as e:
        print(f"Error extracting insights: {str(e)}")

if __name__ == "__main__":
    main() 