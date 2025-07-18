#!/usr/bin/env python3
"""
Enhanced Fungal Language Decoder - Building on Rosetta Stone Findings
=====================================================================

This module extends the fungal_rosetta_stone.py findings to build a comprehensive
fungal communication language by analyzing:
1. Temporal patterns (timing sequences)
2. Amplitude variations (intensity levels)
3. Frequency modulations (pitch changes)
4. Spatial coordination patterns
5. Multi-modal combinations

Author: Fungal Communication Research Team
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.stats import pearsonr
from scipy.signal import find_peaks, hilbert
from fungal_rosetta_stone import FungalRosettaStone

@dataclass
class FungalWord:
    """Represents a decoded fungal communication unit"""
    pattern_name: str
    meaning: str
    confidence: float
    temporal_signature: np.ndarray
    frequency_signature: np.ndarray
    spatial_signature: np.ndarray
    context_dependencies: List[str]
    
@dataclass
class FungalPhrase:
    """Represents a sequence of fungal words"""
    words: List[FungalWord]
    sequence_meaning: str
    temporal_context: str
    confidence: float

class EnhancedFungalLanguageDecoder(FungalRosettaStone):
    """Enhanced decoder that builds complete fungal language vocabulary"""
    
    def __init__(self):
        super().__init__()
        self.vocabulary: Dict[str, FungalWord] = {}
        self.phrases: List[FungalPhrase] = {}
        self.temporal_patterns: Dict = {}
        self.amplitude_patterns: Dict = {}
        self.frequency_patterns: Dict = {}
        
        # Initialize expanded research parameters
        self._initialize_language_parameters()
    
    def _initialize_language_parameters(self):
        """Initialize parameters for comprehensive language analysis based on Adamatzky's research"""
        
        # Temporal pattern vocabulary (based on Adamatzky 2021, 2022)
        self.temporal_vocabulary = {
            'burst_pattern': {
                'duration_range': (3600, 75600),  # 1-21 hours in seconds
                'interval_range': (1800, 7200),   # 30-120 min between bursts
                'meaning': 'urgent_communication',
                'context': 'stress_response'
            },
            'sustained_pattern': {
                'duration_range': (840, 3600),    # 14-60 min
                'interval_range': (3600, 14400),  # 1-4 hours
                'meaning': 'maintenance_signal',
                'context': 'normal_operation'
            },
            'rhythmic_pattern': {
                'duration_range': (156, 840),     # 2.6-14 min (from P. djamor research)
                'interval_range': (300, 1800),    # 5-30 min
                'meaning': 'coordination_signal',
                'context': 'network_synchronization'
            }
        }
        
        # Amplitude-based vocabulary (based on measured ranges 0.03-2.1 mV)
        self.amplitude_vocabulary = {
            'whisper': {'range': (0.03, 0.1), 'meaning': 'local_communication'},
            'normal': {'range': (0.1, 0.5), 'meaning': 'standard_signal'},
            'medium': {'range': (0.5, 1.0), 'meaning': 'medium_range_signal'},
            'strong': {'range': (1.0, 2.1), 'meaning': 'long_range_signal'}
        }
        
        # Frequency modulation vocabulary (based on observed patterns)
        self.frequency_vocabulary = {
            'low_frequency': {'range': (0.001, 0.01), 'meaning': 'maintenance_signal'},     # ~14 min period
            'mid_frequency': {'range': (0.01, 0.05), 'meaning': 'coordination_signal'},     # ~2.6 min period
            'high_frequency': {'range': (0.05, 0.1), 'meaning': 'alert_signal'},           # Faster patterns
        }
        
        # Spatial coordination vocabulary (from research on mycelial networks)
        self.spatial_vocabulary = {
            'expansion': {'pattern': 'radial_growth', 'meaning': 'resource_seeking'},
            'contraction': {'pattern': 'inward_growth', 'meaning': 'conservation'},
            'directional': {'pattern': 'directed_growth', 'meaning': 'targeted_response'},
            'network': {'pattern': 'mesh_growth', 'meaning': 'information_sharing'}
        }
    
    def analyze_temporal_patterns(self, time_array: np.ndarray, signal_array: np.ndarray) -> Dict:
        """Analyze temporal patterns to identify communication sequences"""
        
        # Find signal events (peaks)
        peaks, properties = find_peaks(signal_array, 
                                     height=np.mean(signal_array) + 2*np.std(signal_array),
                                     distance=10)
        
        if len(peaks) < 2:
            return {'patterns': [], 'confidence': 0.0}
        
        # Calculate inter-event intervals
        intervals = np.diff(time_array[peaks])
        durations = properties['peak_heights']
        
        # Classify temporal patterns
        patterns = []
        for i, interval in enumerate(intervals):
            for pattern_name, params in self.temporal_vocabulary.items():
                if (params['duration_range'][0] <= durations[i] <= params['duration_range'][1] and
                    params['interval_range'][0] <= interval <= params['interval_range'][1]):
                    
                    patterns.append({
                        'type': pattern_name,
                        'meaning': params['meaning'],
                        'context': params['context'],
                        'confidence': 0.8,  # Base confidence
                        'timestamp': time_array[peaks[i]],
                        'duration': durations[i],
                        'interval': interval
                    })
        
        return {
            'patterns': patterns,
            'total_events': len(peaks),
            'confidence': len(patterns) / len(peaks) if peaks.size > 0 else 0.0
        }
    
    def analyze_amplitude_patterns(self, signal_array: np.ndarray) -> Dict:
        """Analyze amplitude variations to identify intensity-based communication"""
        
        # Calculate envelope using Hilbert transform
        envelope = np.abs(hilbert(signal_array))
        
        # Classify amplitude levels
        amplitude_words = []
        for i, amp in enumerate(envelope[::100]):  # Sample every 100th point
            for word_type, params in self.amplitude_vocabulary.items():
                if params['range'][0] <= amp <= params['range'][1]:
                    amplitude_words.append({
                        'type': word_type,
                        'meaning': params['meaning'],
                        'amplitude': amp,
                        'confidence': 0.7,
                        'position': i
                    })
                    break
        
        return {
            'words': amplitude_words,
            'envelope': envelope,
            'confidence': len(amplitude_words) / (len(envelope) // 100) if len(envelope) > 0 else 0.0
        }
    
    def analyze_frequency_patterns(self, time_array: np.ndarray, signal_array: np.ndarray) -> Dict:
        """Analyze frequency modulations to identify pitch-based communication"""
        
        # Calculate instantaneous frequency
        analytic_signal = hilbert(signal_array)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi * np.diff(time_array))
        
        # Classify frequency patterns
        frequency_words = []
        for i, freq in enumerate(instantaneous_frequency[::50]):  # Sample every 50th point
            for word_type, params in self.frequency_vocabulary.items():
                if params['range'][0] <= abs(freq) <= params['range'][1]:
                    frequency_words.append({
                        'type': word_type,
                        'meaning': params['meaning'],
                        'frequency': freq,
                        'confidence': 0.6,
                        'position': i
                    })
                    break
        
        return {
            'words': frequency_words,
            'instantaneous_frequency': instantaneous_frequency,
            'confidence': len(frequency_words) / (len(instantaneous_frequency) // 50) if len(instantaneous_frequency) > 0 else 0.0
        }
    
    def build_vocabulary(self, multi_modal_data: Dict) -> Dict[str, FungalWord]:
        """Build vocabulary from multi-modal analysis results"""
        vocabulary = {}
        
        # Extract temporal patterns
        for pattern in multi_modal_data['temporal']['patterns']:
            word = FungalWord(
                pattern_name=pattern['type'],
                meaning=pattern['meaning'],
                confidence=pattern['confidence'],
                temporal_signature=np.array([pattern['duration'], pattern['interval']]),
                frequency_signature=np.array([]),  # Will be updated with frequency data
                spatial_signature=np.array([]),    # Will be updated with spatial data
                context_dependencies=[pattern['context']]
            )
            vocabulary[f"temporal_{pattern['type']}"] = word
        
        # Extract amplitude patterns
        for word in multi_modal_data['amplitude']['words']:
            word_key = f"amplitude_{word['type']}"
            if word_key not in vocabulary:
                vocabulary[word_key] = FungalWord(
                    pattern_name=word['type'],
                    meaning=word['meaning'],
                    confidence=word['confidence'],
                    temporal_signature=np.array([]),
                    frequency_signature=np.array([word['amplitude']]),
                    spatial_signature=np.array([]),
                    context_dependencies=['amplitude_based']
                )
        
        # Extract frequency patterns
        for word in multi_modal_data['frequency']['words']:
            word_key = f"frequency_{word['type']}"
            if word_key not in vocabulary:
                vocabulary[word_key] = FungalWord(
                    pattern_name=word['type'],
                    meaning=word['meaning'],
                    confidence=word['confidence'],
                    temporal_signature=np.array([]),
                    frequency_signature=np.array([word['frequency']]),
                    spatial_signature=np.array([]),
                    context_dependencies=['frequency_based']
                )
        
        return vocabulary
    
    def decode_communication_sequence(self, sequence_data: Dict) -> List[FungalPhrase]:
        """Decode a sequence of fungal communications into phrases"""
        
        phrases = []
        
        # Build vocabulary first
        vocabulary = self.build_vocabulary(sequence_data)
        
        # Find temporal sequences
        for modality, data in sequence_data.items():
            temporal_analysis = self.analyze_temporal_patterns(data['time'], data['signal'])
            
            # Group patterns into phrases
            current_phrase = []
            phrase_start_time = None
            
            for pattern in temporal_analysis['patterns']:
                word_key = f"{modality}_{pattern['type']}"
                if word_key in vocabulary:
                    current_phrase.append(vocabulary[word_key])
                    
                    if phrase_start_time is None:
                        phrase_start_time = pattern['timestamp']
                    
                    # Check if phrase is complete (gap > 30 seconds)
                    if (pattern['timestamp'] - phrase_start_time > 30 and 
                        len(current_phrase) >= 2):
                        
                        # Determine phrase meaning
                        phrase_meaning = self._interpret_phrase_meaning(current_phrase)
                        
                        phrases.append(FungalPhrase(
                            words=current_phrase.copy(),
                            sequence_meaning=phrase_meaning,
                            temporal_context=f"Duration: {pattern['timestamp'] - phrase_start_time:.1f}s",
                            confidence=np.mean([word.confidence for word in current_phrase])
                        ))
                        
                        current_phrase = []
                        phrase_start_time = None
        
        return phrases
    
    def _interpret_phrase_meaning(self, words: List[FungalWord]) -> str:
        """Interpret the meaning of a sequence of words"""
        
        meanings = [word.meaning for word in words]
        
        # Pattern-based interpretation
        if 'urgent_communication' in meanings and 'broadcast_signal' in meanings:
            return "Emergency broadcast to network"
        elif 'coordination_signal' in meanings and 'standard_signal' in meanings:
            return "Network synchronization sequence"
        elif 'maintenance_signal' in meanings:
            return "Routine network maintenance"
        elif 'local_communication' in meanings:
            return "Local neighbor interaction"
        else:
            return f"Complex sequence: {', '.join(set(meanings))}"
    
    def generate_language_report(self, vocabulary: Dict, phrases: List[FungalPhrase]) -> str:
        """Generate comprehensive language analysis report"""
        
        report = f"""
# Fungal Language Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Vocabulary Summary
- **Total Words Discovered**: {len(vocabulary)}
- **Temporal Patterns**: {len([w for w in vocabulary.values() if 'burst' in w.pattern_name or 'sustained' in w.pattern_name or 'rhythmic' in w.pattern_name])}
- **Amplitude Variations**: {len([w for w in vocabulary.values() if any(amp in w.meaning for amp in ['whisper', 'normal', 'shout', 'emergency'])])}
- **Frequency Modulations**: {len([w for w in vocabulary.values() if 'frequency' in w.meaning])}

## Discovered Words
"""
        
        for word_key, word in vocabulary.items():
            report += f"- **{word.pattern_name}**: {word.meaning} (confidence: {word.confidence:.2f})\n"
        
        report += f"""
## Phrase Analysis
- **Total Phrases**: {len(phrases)}
- **Average Phrase Length**: {np.mean([len(p.words) for p in phrases]) if phrases else 0:.1f} words
- **Communication Patterns**:
"""
        
        for phrase in phrases:
            report += f"  - {phrase.sequence_meaning} ({len(phrase.words)} words, {phrase.confidence:.2f} confidence)\n"
        
        return report
    
    def visualize_language_patterns(self, vocabulary: Dict, phrases: List[FungalPhrase], save_path: str = None):
        """Create visualization of discovered language patterns"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Vocabulary distribution
        ax1 = axes[0, 0]
        word_types = [word.meaning for word in vocabulary.values()]
        unique_types, counts = np.unique(word_types, return_counts=True)
        ax1.bar(range(len(unique_types)), counts)
        ax1.set_xticks(range(len(unique_types)))
        ax1.set_xticklabels(unique_types, rotation=45, ha='right')
        ax1.set_title('Vocabulary Distribution')
        ax1.set_ylabel('Count')
        
        # 2. Confidence distribution
        ax2 = axes[0, 1]
        confidences = [word.confidence for word in vocabulary.values()]
        ax2.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Word Confidence Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        
        # 3. Phrase length distribution
        ax3 = axes[1, 0]
        phrase_lengths = [len(phrase.words) for phrase in phrases]
        if phrase_lengths:
            ax3.hist(phrase_lengths, bins=10, alpha=0.7, edgecolor='black')
        ax3.set_title('Phrase Length Distribution')
        ax3.set_xlabel('Number of Words')
        ax3.set_ylabel('Frequency')
        
        # 4. Temporal pattern timeline
        ax4 = axes[1, 1]
        # Create a simple timeline visualization
        x_vals = range(len(phrases))
        y_vals = [phrase.confidence for phrase in phrases]
        ax4.plot(x_vals, y_vals, 'o-', alpha=0.7)
        ax4.set_title('Phrase Confidence Timeline')
        ax4.set_xlabel('Phrase Sequence')
        ax4.set_ylabel('Confidence')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def demo_enhanced_language_analysis():
    """Demonstrate enhanced fungal language analysis"""
    
    print("🍄 ENHANCED FUNGAL LANGUAGE DECODER")
    print("Building on Rosetta Stone findings...")
    
    # Initialize decoder
    decoder = EnhancedFungalLanguageDecoder()
    
    # Create synthetic multi-modal data for demonstration
    t = np.linspace(0, 300, 3000)  # 5 minutes of data
    
    # Simulate complex communication patterns
    electrochemical = {
        'time': t,
        'signal': (0.1 * np.sin(2 * np.pi * 0.1 * t) + 
                  0.3 * np.sin(2 * np.pi * 0.05 * t) * (t > 60) * (t < 120) +  # Burst pattern
                  0.2 * np.random.normal(0, 0.05, len(t)) +  # Noise
                  0.5 * np.exp(-((t - 180) / 20)**2))  # Emergency signal
    }
    
    acoustic = {
        'time': t,
        'signal': (0.05 * np.sin(2 * np.pi * 0.2 * t) + 
                  0.1 * np.sin(2 * np.pi * 0.3 * t) * (t > 100) * (t < 200) +  # Coordination
                  0.1 * np.random.normal(0, 0.02, len(t)))
    }
    
    spatial = {
        'time': t,
        'signal': 0.01 * t + 0.02 * np.sin(2 * np.pi * 0.02 * t)  # Gradual growth with oscillation
    }
    
    multi_modal_data = {
        'electrochemical': electrochemical,
        'acoustic': acoustic,
        'spatial': spatial
    }
    
    # Analyze language patterns
    vocabulary = decoder.build_vocabulary(multi_modal_data)
    phrases = decoder.decode_communication_sequence(multi_modal_data)
    
    # Generate report
    report = decoder.generate_language_report(vocabulary, phrases)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("language_analysis_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save vocabulary
    vocab_dict = {k: {
        'meaning': v.meaning,
        'confidence': v.confidence,
        'context': v.context_dependencies
    } for k, v in vocabulary.items()}
    
    with open(output_dir / f"fungal_vocabulary_{timestamp}.json", 'w') as f:
        json.dump(vocab_dict, f, indent=2)
    
    # Save phrases
    phrases_dict = {
        f"phrase_{i}": {
            'meaning': phrase.sequence_meaning,
            'word_count': len(phrase.words),
            'confidence': phrase.confidence,
            'context': phrase.temporal_context
        } for i, phrase in enumerate(phrases)
    }
    
    with open(output_dir / f"fungal_phrases_{timestamp}.json", 'w') as f:
        json.dump(phrases_dict, f, indent=2)
    
    # Save report
    with open(output_dir / f"language_analysis_report_{timestamp}.md", 'w') as f:
        f.write(report)
    
    # Create visualization
    decoder.visualize_language_patterns(vocabulary, phrases, 
                                      output_dir / f"language_patterns_{timestamp}.png")
    
    print(f"✅ Language analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Vocabulary words discovered: {len(vocabulary)}")
    print(f"📝 Communication phrases identified: {len(phrases)}")
    print(f"🔍 Average confidence: {np.mean([w.confidence for w in vocabulary.values()]):.2f}")

if __name__ == "__main__":
    demo_enhanced_language_analysis() 