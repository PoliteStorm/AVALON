"""
🧪 SEMANTIC TESTING FRAMEWORK
===========================

Advanced semantic testing framework for fungal communication analysis.
Based on Dehshibi & Adamatzky (2021) Biosystems Research.

Primary Sources:
- Dehshibi, M.M. & Adamatzky, A. (2021). "Electrical activity of fungi: 
  Spikes detection and complexity analysis" Biosystems 203, 104373
  DOI: 10.1016/j.biosystems.2021.104373
- Adamatzky, A. (2023). "Language of fungi derived from their electrical spiking activity"
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings
from scipy import signal, stats
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
import gc
import json
import os

from fungal_communication_github.research_constants import (
    RESEARCH_CITATION,
    SPECIES_DATABASE,
    ELECTRICAL_PARAMETERS,
    get_research_backed_parameters,
    validate_simulation_against_research,
    ensure_scientific_rigor
)

@dataclass
class SemanticConfig:
    """Configuration for semantic testing"""
    voltage_threshold: float = 0.0001  # V
    frequency_range: Dict[str, float] = None
    pattern_recognition: bool = True
    statistical_validation: bool = True
    complexity_analysis: bool = True
    max_memory_usage: float = 0.8  # Maximum fraction of available memory to use
    chunk_size: int = 1000  # Reduced chunk size for more frequent updates
    save_intermediate: bool = True
    intermediate_dir: str = "intermediate_results"
    
    def __init__(self, **kwargs):
        self.voltage_threshold = kwargs.get('voltage_threshold', 0.0001)
        self.frequency_range = kwargs.get('frequency_range', {'min': 0.01, 'max': 10.0})
        self.pattern_recognition = kwargs.get('pattern_recognition', True)
        self.statistical_validation = kwargs.get('statistical_validation', True)
        self.complexity_analysis = kwargs.get('complexity_analysis', True)
        self.max_memory_usage = kwargs.get('max_memory_usage', 0.8)
        self.chunk_size = kwargs.get('chunk_size', 1000)
        self.save_intermediate = kwargs.get('save_intermediate', True)
        self.intermediate_dir = kwargs.get('intermediate_dir', "intermediate_results")

class SemanticTestingFramework:
    """
    Advanced semantic testing framework for fungal communication.
    
    Features:
    - Research-backed pattern recognition
    - Advanced statistical validation
    - Complexity analysis
    - Semantic interpretation
    - Empirical validation
    """
    
    def __init__(self):
        """Initialize semantic testing framework"""
        print("\n🧪 SEMANTIC TESTING FRAMEWORK INITIALIZED")
        print(f"📊 Research Foundation: {RESEARCH_CITATION['authors']} ({RESEARCH_CITATION['year']})")
        
        # Initialize configuration
        self.config = SemanticConfig()
        print(f"🔬 Voltage Threshold: {self.config.voltage_threshold} V")
        print(f"📈 Frequency Range: {self.config.frequency_range} Hz")
        print(f"🧬 Pattern Recognition: {'Enabled' if self.config.pattern_recognition else 'Disabled'}")
        
        # Create intermediate results directory
        if self.config.save_intermediate:
            os.makedirs(self.config.intermediate_dir, exist_ok=True)
        
        # Initialize semantic protocols
        self.semantic_protocols = {
            'semantic_dictionary': {
                'high_frequency': ['alert', 'active', 'stressed'],
                'low_frequency': ['rest', 'dormant', 'stable'],
                'high_amplitude': ['significant', 'important', 'urgent'],
                'low_amplitude': ['background', 'routine', 'normal']
            }
        }
        
        # Initialize analysis parameters
        self.analysis_params = ELECTRICAL_PARAMETERS.copy()
        
    def analyze_semantic_patterns(self, voltage_data: np.ndarray, time_data: np.ndarray,
                                species: str = "Pleurotus_djamor", 
                                progress_callback: Optional[callable] = None) -> Dict:
        """Analyze semantic patterns with memory optimization and progress tracking."""
        if len(voltage_data) != len(time_data):
            raise ValueError("Voltage and time data must have same length")
            
        if not isinstance(species, str) or species not in SPECIES_DATABASE:
            raise ValueError(f"Invalid species: {species}")
            
        # Create intermediate results directory if needed
        if self.config.save_intermediate:
            os.makedirs(self.config.intermediate_dir, exist_ok=True)
        
        # Initialize results
        results = {
            'timestamp': datetime.now().isoformat(),
            'species': species,
            'data_points': len(voltage_data),
            'duration': time_data[-1] - time_data[0],
            'analysis_layers': {},
            'validation': {}
        }
        
        try:
            # Process data in smaller chunks
            chunk_size = min(self.config.chunk_size, len(voltage_data))
            n_chunks = (len(voltage_data) + chunk_size - 1) // chunk_size
            
            # Layer 1: Pattern Recognition (chunked)
            if self.config.pattern_recognition:
                patterns = []
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, len(voltage_data))
                    
                    # Ensure indices match array dimensions
                    chunk_v = voltage_data[start_idx:end_idx]
                    chunk_t = time_data[start_idx:end_idx]
                    
                    chunk_patterns = self._analyze_patterns(chunk_v, chunk_t)
                    patterns.extend(chunk_patterns.get('patterns', []))
                    
                    # Save intermediate results
                    if self.config.save_intermediate:
                        intermediate = {
                            'chunk': i + 1,
                            'total_chunks': n_chunks,
                            'patterns_found': len(chunk_patterns.get('patterns', [])),
                            'start_time': chunk_t[0],
                            'end_time': chunk_t[-1],
                            'patterns': chunk_patterns.get('patterns', [])
                        }
                        self._save_intermediate('patterns', i, intermediate)
                    
                    if progress_callback:
                        progress = 0.25 * (i + 1) / n_chunks
                        progress_callback(progress, f"Pattern Recognition (Chunk {i+1}/{n_chunks})")
                    
                    # Force garbage collection
                    gc.collect()
                
                results['analysis_layers']['pattern_recognition'] = {
                    'total_patterns': len(patterns),
                    'patterns': patterns
                }
            
            # Layer 2: Statistical Validation (chunked)
            if self.config.statistical_validation:
                statistical_results = {'tests': [], 'validations': []}
                
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, len(voltage_data))
                    
                    # Ensure indices match
                    chunk_v = voltage_data[start_idx:end_idx]
                    chunk_patterns = patterns[start_idx:end_idx] if patterns else []
                    
                    chunk_stats = self._validate_statistics(chunk_v, {'patterns': chunk_patterns})
                    statistical_results['tests'].extend(chunk_stats.get('tests', []))
                    statistical_results['validations'].extend(chunk_stats.get('validations', []))
                    
                    # Save intermediate results
                    if self.config.save_intermediate:
                        intermediate = {
                            'chunk': i + 1,
                            'total_chunks': n_chunks,
                            'tests_performed': len(chunk_stats.get('tests', [])),
                            'validations_performed': len(chunk_stats.get('validations', [])),
                            'results': chunk_stats
                        }
                        self._save_intermediate('statistics', i, intermediate)
                    
                    if progress_callback:
                        progress = 0.5 + 0.25 * (i + 1) / n_chunks
                        progress_callback(progress, f"Statistical Validation (Chunk {i+1}/{n_chunks})")
                    
                    gc.collect()
                
                results['analysis_layers']['statistical_validation'] = statistical_results
            
            # Layer 3: Complexity Analysis (chunked)
            if self.config.complexity_analysis:
                complexity_results = {'entropy_analysis': {}, 'fractal_analysis': {}}
                
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, len(voltage_data))
                    
                    # Ensure indices match
                    chunk_v = voltage_data[start_idx:end_idx]
                    chunk_t = time_data[start_idx:end_idx]
                    
                    chunk_complexity = self._analyze_complexity(chunk_v, chunk_t)
                    
                    # Save intermediate results
                    if self.config.save_intermediate:
                        intermediate = {
                            'chunk': i + 1,
                            'total_chunks': n_chunks,
                            'entropy_results': chunk_complexity.get('entropy_analysis', {}),
                            'fractal_results': chunk_complexity.get('fractal_analysis', {})
                        }
                        self._save_intermediate('complexity', i, intermediate)
                    
                    # Merge results
                    for key in ['entropy_analysis', 'fractal_analysis']:
                        if key not in complexity_results:
                            complexity_results[key] = chunk_complexity[key]
                        else:
                            complexity_results[key].update(chunk_complexity[key])
                    
                    if progress_callback:
                        progress = 0.75 + 0.15 * (i + 1) / n_chunks
                        progress_callback(progress, f"Complexity Analysis (Chunk {i+1}/{n_chunks})")
                    
                    gc.collect()
                
                results['analysis_layers']['complexity_analysis'] = complexity_results
            
            # Layer 4: Semantic Interpretation
            semantic_results = self._interpret_semantics(
                results['analysis_layers'].get('pattern_recognition', {}),
                results['analysis_layers'].get('statistical_validation'),
                results['analysis_layers'].get('complexity_analysis')
            )
            results['analysis_layers']['semantic_interpretation'] = semantic_results
            
            if progress_callback:
                progress_callback(1.0, "Analysis Complete")
            
            return results
            
        except Exception as e:
            # Save error state if possible
            if self.config.save_intermediate:
                self._save_intermediate('error', 0, {
                    'error': str(e),
                    'stage': results.get('current_stage', 'unknown'),
                    'partial_results': results
                })
            raise e
    
    def _analyze_patterns(self, voltage_data: np.ndarray, time_data: np.ndarray) -> Dict:
        """Analyze voltage patterns"""
        patterns = []
        
        # Basic pattern detection
        mean_voltage = np.mean(voltage_data)
        std_voltage = np.std(voltage_data)
        threshold = mean_voltage + 2 * std_voltage
        
        # Find peaks
        peaks, _ = signal.find_peaks(voltage_data, height=threshold)
        
        if len(peaks) > 0:
            patterns.append({
                'type': 'spike',
                'count': len(peaks),
                'times': time_data[peaks].tolist(),
                'amplitudes': voltage_data[peaks].tolist(),
                'confidence': 0.8
            })
        
        # Frequency analysis
        if len(voltage_data) > 10:
            freqs, psd = signal.welch(voltage_data, fs=1/(time_data[1]-time_data[0]))
            peak_freqs = freqs[signal.find_peaks(psd)[0]]
            
            if len(peak_freqs) > 0:
                patterns.append({
                    'type': 'oscillation',
                    'frequencies': peak_freqs.tolist(),
                    'power': psd[signal.find_peaks(psd)[0]].tolist(),
                    'confidence': 0.7
                })
        
        return {'patterns': patterns}
    
    def _validate_statistics(self, voltage_data: np.ndarray, pattern_results: Dict) -> Dict:
        """Validate statistical significance"""
        validations = []
        tests = []
        
        # Basic statistical tests
        if len(voltage_data) > 10:
            # Test for non-randomness
            z_stat, p_value = stats.normaltest(voltage_data)
            tests.append({
                'name': 'normality_test',
                'statistic': float(z_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            })
            
            # Test for stationarity
            if len(voltage_data) > 20:
                chunks = np.array_split(voltage_data, 4)
                means = [np.mean(chunk) for chunk in chunks]
                f_stat, p_value = stats.f_oneway(*chunks)
                tests.append({
                    'name': 'stationarity_test',
                    'statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                })
        
        # Pattern validation
        patterns = pattern_results.get('patterns', [])
        for pattern in patterns:
            if pattern['type'] == 'spike':
                validations.append({
                    'pattern_type': 'spike',
                    'valid': pattern['count'] > 0 and max(pattern['amplitudes']) > self.config.voltage_threshold,
                    'confidence': pattern['confidence']
                })
            elif pattern['type'] == 'oscillation':
                validations.append({
                    'pattern_type': 'oscillation',
                    'valid': any(f > self.config.frequency_range['min'] and 
                               f < self.config.frequency_range['max'] 
                               for f in pattern['frequencies']),
                    'confidence': pattern['confidence']
                })
        
        return {
            'tests': tests,
            'validations': validations
        }
    
    def _analyze_complexity(self, voltage_data: np.ndarray, time_data: np.ndarray) -> Dict:
        """Analyze signal complexity"""
        results = {
            'entropy_analysis': {},
            'fractal_analysis': {}
        }
        
        if len(voltage_data) > 10:
            # Sample entropy
            r = 0.2 * np.std(voltage_data)
            sample_entropy = self._calculate_sample_entropy(voltage_data, m=2, r=r)
            results['entropy_analysis']['sample_entropy'] = float(sample_entropy)
            
            # Approximate entropy
            approx_entropy = self._calculate_approximate_entropy(voltage_data, m=2, r=r)
            results['entropy_analysis']['approximate_entropy'] = float(approx_entropy)
            
            # Fractal dimension (Higuchi)
            if len(voltage_data) > 100:
                fractal_dim = self._calculate_fractal_dimension(voltage_data)
                results['fractal_analysis']['higuchi_fd'] = float(fractal_dim)
        
        return results
    
    def _calculate_sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy"""
        N = len(data)
        B = 0.0
        A = 0.0
        
        # Convert r to absolute units
        r = r * np.std(data)
        
        # Create template vectors
        xmi = np.array([data[i:i+m] for i in range(N-m)])
        xmj = np.array([data[i:i+m] for i in range(N-m)])
        
        # Calculate matches for m
        for i in range(len(xmi)):
            for j in range(len(xmj)):
                if i != j:
                    if np.max(np.abs(xmi[i] - xmj[j])) <= r:
                        B += 1
        
        # Calculate matches for m+1
        xm1i = np.array([data[i:i+m+1] for i in range(N-m-1)])
        xm1j = np.array([data[i:i+m+1] for i in range(N-m-1)])
        
        for i in range(len(xm1i)):
            for j in range(len(xm1j)):
                if i != j:
                    if np.max(np.abs(xm1i[i] - xm1j[j])) <= r:
                        A += 1
        
        # Calculate sample entropy
        if B > 0 and A > 0:
            return -np.log(A/B)
        else:
            return 0.0
    
    def _calculate_approximate_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate approximate entropy"""
        N = len(data)
        phi = np.zeros(2)
        
        for i in range(2):
            template = m + i
            count = np.zeros(N - template + 1)
            
            # Create template vectors
            xmi = np.array([data[j:j+template] for j in range(N-template+1)])
            
            # Count matches
            for j in range(N - template + 1):
                matches = np.max(np.abs(xmi - xmi[j]), axis=1) <= r
                count[j] = np.sum(matches) - 1  # Subtract self-match
            
            # Calculate phi
            phi[i] = np.mean(np.log(count / (N - template + 1)))
        
        return phi[0] - phi[1]
    
    def _calculate_fractal_dimension(self, data: np.ndarray, k_max: int = 8) -> float:
        """Calculate Higuchi fractal dimension"""
        N = len(data)
        L = np.zeros(k_max)
        x = np.arange(N)
        
        for k in range(1, k_max + 1):
            Lk = 0
            
            for m in range(k):
                # Extract subsequence
                subset = data[m::k]
                subset_x = x[m::k]
                
                # Calculate length
                diff = np.abs(np.diff(subset))
                Lm = np.sum(diff) * (N - 1) / (((N - m) // k) * k)
                Lk += Lm
            
            L[k-1] = Lk / k
        
        # Fit line to log-log plot
        x = np.log(np.arange(1, k_max + 1))
        y = np.log(L)
        slope, _ = np.polyfit(x, y, 1)
        
        return -slope
    
    def _interpret_semantics(self, pattern_results: Dict, statistical_results: Dict,
                           complexity_results: Dict) -> Dict:
        """Interpret semantic meaning"""
        interpretation = {
            'semantic_patterns': [],
            'confidence': 0.0,
            'complexity_level': 'unknown'
        }
        
        # Analyze patterns
        patterns = pattern_results.get('patterns', [])
        for pattern in patterns:
            if pattern['type'] == 'spike':
                interpretation['semantic_patterns'].append({
                    'type': 'communication_event',
                    'count': pattern['count'],
                    'confidence': pattern['confidence'],
                    'interpretation': 'Potential signaling event'
                })
            elif pattern['type'] == 'oscillation':
                interpretation['semantic_patterns'].append({
                    'type': 'rhythmic_activity',
                    'frequencies': pattern['frequencies'],
                    'confidence': pattern['confidence'],
                    'interpretation': 'Coordinated network activity'
                })
        
        # Analyze complexity
        if complexity_results:
            entropy = complexity_results.get('entropy_analysis', {}).get('sample_entropy', 0)
            if entropy > 1.5:
                interpretation['complexity_level'] = 'high'
            elif entropy > 0.8:
                interpretation['complexity_level'] = 'medium'
            else:
                interpretation['complexity_level'] = 'low'
        
        # Calculate overall confidence
        pattern_confidence = np.mean([p.get('confidence', 0) for p in patterns]) if patterns else 0
        statistical_confidence = np.mean([t.get('significant', False) 
                                       for t in statistical_results.get('tests', [])]) if statistical_results else 0
        
        interpretation['confidence'] = (pattern_confidence + statistical_confidence) / 2
        
        return interpretation
    
    def _convert_to_json_serializable(self, obj):
        """Helper method to convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        return obj

    def _save_intermediate(self, stage: str, chunk: int, data: Dict):
        """Save intermediate results to file"""
        if not self.config.save_intermediate:
            return
            
        filename = f"{self.config.intermediate_dir}/{stage}_chunk_{chunk}.json"
        # Convert numpy types to Python native types before saving
        serializable_data = self._convert_to_json_serializable(data)
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)

def demo_semantic_testing():
    """Demonstration of semantic testing framework"""
    framework = SemanticTestingFramework()
    
    # Generate sample data
    t = np.linspace(0, 10, 1000)
    v = 0.001 * np.sin(2 * np.pi * 0.5 * t) + 0.0001 * np.random.randn(len(t))
    
    # Run analysis
    results = framework.analyze_semantic_patterns(v, t)
    
    print("\n✅ Demo Results:")
    print(f"Found {results['analysis_layers']['pattern_recognition']['total_patterns']} patterns")
    print(f"Complexity level: {results['analysis_layers']['semantic_interpretation']['complexity_level']}")
    print(f"Confidence: {results['analysis_layers']['semantic_interpretation']['confidence']:.2f}")

if __name__ == "__main__":
    demo_semantic_testing() 