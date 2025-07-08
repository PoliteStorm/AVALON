#!/usr/bin/env python3
"""
📊 DETAILED RESULTS EXTRACTION AND VISUALIZATION
Extract all words, patterns, and create visualizations from the simulation
"""

import numpy as np
import json
from comprehensive_communication_simulation import ComprehensiveCommunicationSimulation

def extract_all_results():
    """Extract comprehensive results from the simulation"""
    
    print("🔬 EXTRACTING COMPREHENSIVE SIMULATION RESULTS")
    print("="*60)
    
    # Run simulation
    simulation = ComprehensiveCommunicationSimulation()
    results = simulation.run_full_simulation()
    
    # Extract detailed information
    detailed_results = {
        'test_cases': [],
        'all_words': [],
        'frequency_patterns': [],
        'species_dialects': {},
        'confidence_scores': [],
        'discoveries': []
    }
    
    print("\n📊 EXTRACTING DETAILED DATA...")
    
    for i, result in enumerate(results, 1):
        test_case = result['test_case']
        analysis = result['analysis_result']
        
        # Extract test case details
        case_details = {
            'id': i,
            'name': test_case['name'],
            'species': test_case['species_name'],
            'environment': test_case['environmental_context'],
            'electrical_pattern': test_case['electrical_pattern'],
            'analysis_results': analysis
        }
        detailed_results['test_cases'].append(case_details)
        
        # Extract words and linguistic patterns
        linguistic = analysis['layer_analyses']['linguistic']
        word_entry = {
            'test_case': i,
            'primary_word': linguistic['primary_word'],
            'word_count': linguistic['word_count'],
            'sentence_type': linguistic['sentence_type'],
            'complexity': linguistic['complexity'],
            'human_similarity': linguistic['human_similarity']
        }
        detailed_results['all_words'].append(word_entry)
        
        # Extract frequency patterns
        electrical = analysis['layer_analyses']['electrical']
        freq_pattern = {
            'test_case': i,
            'frequency': electrical['frequency'],
            'amplitude': electrical['amplitude'],
            'duration': electrical['duration'],
            'complexity': electrical['complexity'],
            'activity_level': electrical['activity_level']
        }
        detailed_results['frequency_patterns'].append(freq_pattern)
        
        # Extract species dialect info
        species = test_case['species_name']
        dialectal = analysis['dialectal_analysis']
        if species not in detailed_results['species_dialects']:
            detailed_results['species_dialects'][species] = []
        
        detailed_results['species_dialects'][species].append({
            'test_case': i,
            'dialect_name': dialectal['dialect_name'],
            'communication_style': dialectal['communication_style'],
            'frequency_match': dialectal['frequency_match'],
            'confidence': dialectal['dialect_confidence']
        })
        
        # Extract confidence scores
        confidence = analysis['scientific_confidence']
        detailed_results['confidence_scores'].append({
            'test_case': i,
            'overall_confidence': confidence['overall_confidence'],
            'confidence_level': confidence['confidence_level']
        })
        
        # Extract discoveries
        discoveries = analysis['discoveries']
        for discovery in discoveries:
            discovery['test_case'] = i
            detailed_results['discoveries'].append(discovery)
    
    return detailed_results

def display_all_words(detailed_results):
    """Display all identified words and patterns"""
    
    print("\n📝 ALL IDENTIFIED WORDS AND PATTERNS")
    print("="*50)
    
    for word_entry in detailed_results['all_words']:
        print(f"\n🔤 Test Case {word_entry['test_case']}:")
        print(f"   Primary Word: {word_entry['primary_word']}")
        print(f"   Word Count: {word_entry['word_count']}")
        print(f"   Sentence Type: {word_entry['sentence_type']}")
        print(f"   Complexity: {word_entry['complexity']:.3f}")
        print(f"   Human Similarity: {word_entry['human_similarity']:.3f}")
    
    # Summary of all words
    all_words = [w['primary_word'] for w in detailed_results['all_words']]
    unique_words = list(set(all_words))
    
    print(f"\n📊 WORD SUMMARY:")
    print(f"   Total Words Identified: {len(all_words)}")
    print(f"   Unique Words: {len(unique_words)}")
    print(f"   Unique Word List: {', '.join(unique_words)}")

def display_frequency_patterns(detailed_results):
    """Display frequency patterns analysis"""
    
    print("\n📡 FREQUENCY PATTERN ANALYSIS")
    print("="*50)
    
    for pattern in detailed_results['frequency_patterns']:
        print(f"\n⚡ Test Case {pattern['test_case']}:")
        print(f"   Frequency: {pattern['frequency']:.1f} Hz")
        print(f"   Amplitude: {pattern['amplitude']:.3f} mV")
        print(f"   Duration: {pattern['duration']:.1f} hours")
        print(f"   Complexity: {pattern['complexity']:.3f}")
        print(f"   Activity Level: {pattern['activity_level']}")
    
    # Frequency distribution
    frequencies = [p['frequency'] for p in detailed_results['frequency_patterns']]
    print(f"\n📊 FREQUENCY DISTRIBUTION:")
    print(f"   Min Frequency: {min(frequencies):.1f} Hz")
    print(f"   Max Frequency: {max(frequencies):.1f} Hz")
    print(f"   Average Frequency: {np.mean(frequencies):.1f} Hz")
    print(f"   Frequency Range: {max(frequencies) - min(frequencies):.1f} Hz")

def display_species_dialects(detailed_results):
    """Display species dialect analysis"""
    
    print("\n🗣️ SPECIES DIALECT ANALYSIS")
    print("="*50)
    
    for species, dialect_info in detailed_results['species_dialects'].items():
        print(f"\n🧬 {species}:")
        for info in dialect_info:
            print(f"   Test Case {info['test_case']}: {info['dialect_name']}")
            print(f"   Communication Style: {info['communication_style']}")
            print(f"   Frequency Match: {info['frequency_match']}")
            print(f"   Confidence: {info['confidence']}")

def display_confidence_analysis(detailed_results):
    """Display confidence score analysis"""
    
    print("\n📊 CONFIDENCE SCORE ANALYSIS")
    print("="*50)
    
    confidences = [c['overall_confidence'] for c in detailed_results['confidence_scores']]
    
    print(f"📈 CONFIDENCE STATISTICS:")
    print(f"   Average Confidence: {np.mean(confidences):.1%}")
    print(f"   Highest Confidence: {max(confidences):.1%}")
    print(f"   Lowest Confidence: {min(confidences):.1%}")
    print(f"   Standard Deviation: {np.std(confidences):.1%}")
    
    print(f"\n📊 CONFIDENCE BREAKDOWN:")
    for conf in detailed_results['confidence_scores']:
        print(f"   Test Case {conf['test_case']}: {conf['overall_confidence']:.1%} - {conf['confidence_level']}")

def display_discoveries(detailed_results):
    """Display discovery analysis"""
    
    print("\n🔍 DISCOVERY ANALYSIS")
    print("="*50)
    
    if detailed_results['discoveries']:
        print(f"🎯 POTENTIAL DISCOVERIES IDENTIFIED: {len(detailed_results['discoveries'])}")
        
        for discovery in detailed_results['discoveries']:
            print(f"\n🚀 Test Case {discovery['test_case']}:")
            print(f"   Type: {discovery['type']}")
            print(f"   Description: {discovery['description']}")
            print(f"   Significance: {discovery['significance']}")
            print(f"   Research Priority: {discovery['research_priority']}")
    else:
        print("No specific discoveries identified in this simulation run.")

def create_simple_visualizations(detailed_results):
    """Create simple text-based visualizations"""
    
    print("\n📈 SIMPLE VISUALIZATIONS")
    print("="*50)
    
    # Frequency distribution visualization
    print("\n📊 FREQUENCY DISTRIBUTION (Hz):")
    frequencies = [p['frequency'] for p in detailed_results['frequency_patterns']]
    
    # Create simple bar chart
    for i, freq in enumerate(frequencies, 1):
        bar_length = int(freq * 2)  # Scale for display
        bar = "█" * bar_length
        print(f"   Case {i}: {freq:5.1f} Hz {bar}")
    
    # Confidence distribution
    print("\n📊 CONFIDENCE DISTRIBUTION:")
    confidences = [c['overall_confidence'] for c in detailed_results['confidence_scores']]
    
    for i, conf in enumerate(confidences, 1):
        bar_length = int(conf * 50)  # Scale for display
        bar = "█" * bar_length
        print(f"   Case {i}: {conf:5.1%} {bar}")
    
    # Species dialect distribution
    print("\n📊 SPECIES DIALECT DISTRIBUTION:")
    species_counts = {}
    for species in detailed_results['species_dialects']:
        species_counts[species] = len(detailed_results['species_dialects'][species])
    
    for species, count in species_counts.items():
        bar = "█" * count
        print(f"   {species}: {count} cases {bar}")

def main():
    """Main function to extract and display all results"""
    
    print("🔬 COMPREHENSIVE FUNGAL COMMUNICATION RESULTS EXTRACTION")
    print("="*80)
    
    # Extract all results
    detailed_results = extract_all_results()
    
    # Display all components
    display_all_words(detailed_results)
    display_frequency_patterns(detailed_results)
    display_species_dialects(detailed_results)
    display_confidence_analysis(detailed_results)
    display_discoveries(detailed_results)
    create_simple_visualizations(detailed_results)
    
    # Save results to JSON file
    with open('detailed_simulation_results.json', 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        json_results = json.loads(json.dumps(detailed_results, default=str))
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 RESULTS SAVED TO: detailed_simulation_results.json")
    print(f"🏆 COMPREHENSIVE ANALYSIS COMPLETE!")
    
    return detailed_results

if __name__ == "__main__":
    main() 