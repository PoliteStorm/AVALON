import numpy as np
from scipy import signal
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class LinguisticAnalyzer:
    """
    Implements Adamatzky's methodology for analyzing fungal electrical signals as language.
    """
    
    def __init__(self, sampling_rate: float = 1.0):
        self.sampling_rate = sampling_rate
        
    def group_spikes_to_words(self, spike_times: np.ndarray, 
                             theta_multiplier: float = 1.0) -> List[List[int]]:
        """
        Group spikes into words based on inter-spike intervals.
        Uses Adamatzky's method where spikes closer than theta belong to same word.
        
        Args:
            spike_times: Array of spike timestamps
            theta_multiplier: Multiplier for the average interval threshold (1.0 or 2.0)
            
        Returns:
            List of words, where each word is a list of spike indices
        """
        if len(spike_times) < 2:
            return []
            
        # Calculate intervals
        intervals = np.diff(spike_times)
        avg_interval = np.mean(intervals)
        theta = avg_interval * theta_multiplier
        
        # Group spikes into words
        words = []
        current_word = [0]  # Start with first spike
        
        for i in range(1, len(spike_times)):
            if intervals[i-1] <= theta:
                current_word.append(i)
            else:
                words.append(current_word)
                current_word = [i]
                
        # Add last word if not empty
        if current_word:
            words.append(current_word)
            
        return words
        
    def analyze_word_statistics(self, words: List[List[int]]) -> Dict:
        """
        Analyze statistical properties of fungal words.
        
        Args:
            words: List of words (each word is list of spike indices)
            
        Returns:
            Dictionary containing word statistics
        """
        if not words:
            return {
                'word_lengths': [],
                'avg_word_length': 0,
                'word_length_distribution': {},
                'vocabulary_size': 0
            }
            
        # Calculate word lengths
        word_lengths = [len(word) for word in words]
        
        # Calculate distribution
        length_dist = defaultdict(int)
        for length in word_lengths:
            length_dist[length] += 1
            
        # Convert to probability distribution
        total_words = len(words)
        length_dist = {k: v/total_words for k, v in length_dist.items()}
        
        return {
            'word_lengths': word_lengths,
            'avg_word_length': np.mean(word_lengths),
            'word_length_distribution': dict(length_dist),
            'vocabulary_size': len(set(tuple(word) for word in words))
        }
        
    def analyze_syntax(self, words: List[List[int]]) -> Dict:
        """
        Analyze syntactic patterns in fungal words.
        
        Args:
            words: List of words (each word is list of spike indices)
            
        Returns:
            Dictionary containing syntax analysis results
        """
        if len(words) < 2:
            return {
                'transition_matrix': {},
                'common_sequences': [],
                'sequence_probabilities': {}
            }
            
        # Convert words to tuple representation for hashing
        word_tuples = [tuple(word) for word in words]
        unique_words = list(set(word_tuples))
        
        # Create word-to-index mapping
        word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
        
        # Build transition matrix
        n_words = len(unique_words)
        transitions = np.zeros((n_words, n_words))
        
        for i in range(len(word_tuples)-1):
            curr_idx = word_to_idx[word_tuples[i]]
            next_idx = word_to_idx[word_tuples[i+1]]
            transitions[curr_idx][next_idx] += 1
            
        # Normalize transitions
        row_sums = transitions.sum(axis=1)
        transitions = np.divide(transitions, row_sums[:, np.newaxis],
                              where=row_sums[:, np.newaxis] != 0)
        
        # Find common sequences (bigrams)
        bigrams = defaultdict(int)
        for i in range(len(word_tuples)-1):
            bigram = (word_tuples[i], word_tuples[i+1])
            bigrams[bigram] += 1
            
        # Convert to probabilities
        total_bigrams = sum(bigrams.values())
        bigram_probs = {k: v/total_bigrams for k, v in bigrams.items()}
        
        # Sort by probability
        common_sequences = sorted(bigram_probs.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:10]
        
        return {
            'transition_matrix': transitions.tolist(),
            'common_sequences': common_sequences,
            'sequence_probabilities': dict(bigram_probs)
        }
        
    def compute_complexity_measures(self, words: List[List[int]]) -> Dict:
        """
        Compute complexity measures of the fungal language.
        
        Args:
            words: List of words (each word is list of spike indices)
            
        Returns:
            Dictionary containing complexity metrics
        """
        if not words:
            return {
                'algorithmic_complexity': 0,
                'normalized_complexity': 0,
                'entropy': 0
            }
            
        # Convert words to string representation for complexity calculation
        word_strings = [''.join(map(str, word)) for word in words]
        text = ' '.join(word_strings)
        
        # Calculate Lempel-Ziv complexity
        n = len(text)
        complexity = 1
        substrings = set()
        
        i = 0
        while i < n:
            length = 1
            while i + length <= n and text[i:i+length] in substrings:
                length += 1
            substrings.add(text[i:i+length])
            complexity += 1
            i += length
            
        # Calculate entropy
        word_counts = defaultdict(int)
        total_words = len(words)
        
        for word in words:
            word_counts[tuple(word)] += 1
            
        probabilities = [count/total_words for count in word_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities)
        
        return {
            'algorithmic_complexity': complexity,
            'normalized_complexity': complexity / len(text) if text else 0,
            'entropy': entropy
        }
        
    def analyze_linguistic_features(self, spike_times: np.ndarray) -> Dict:
        """
        Perform complete linguistic analysis of fungal signals.
        
        Args:
            spike_times: Array of spike timestamps
            
        Returns:
            Dictionary containing all linguistic analysis results
        """
        # Analyze with both theta values
        results = {}
        for theta_mult in [1.0, 2.0]:
            # Group spikes into words
            words = self.group_spikes_to_words(spike_times, theta_mult)
            
            # Perform all analyses
            prefix = f'theta_{theta_mult}'
            results[prefix] = {
                'word_stats': self.analyze_word_statistics(words),
                'syntax': self.analyze_syntax(words),
                'complexity': self.compute_complexity_measures(words)
            }
            
        return results 