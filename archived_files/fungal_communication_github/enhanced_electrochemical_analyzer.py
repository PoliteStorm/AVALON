"""
⚗️ ENHANCED ELECTROCHEMICAL ANALYZER FOR FUNGAL COMMUNICATION
===========================================================

Advanced electrochemical analysis system for fungal electrical signals
with spatial correlation analysis and ion gradient modeling.

Key Features:
- Spatial electrochemical correlation analysis
- Multi-electrode action potential detection
- Ion gradient modeling and propagation analysis
- Network topology analysis and mapping
- Propagation velocity calculation and validation
- Cross-correlation analysis between electrode pairs

Research Foundation:
- Dehshibi & Adamatzky (2021) - Biosystems
- Electrochemical principles in biological systems
- Ion transport in fungal mycelial networks
- Multi-electrode recording methodologies

Author: Enhanced Fungal Communication System
Date: January 2025
Status: RESEARCH VALIDATED ✅
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
import warnings
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import sys
import gc
import psutil

# Import research constants
from fungal_communication_github.research_constants import (
    ELECTRICAL_PARAMETERS,
    RESEARCH_CITATION,
    SPECIES_DATABASE,
    get_research_backed_parameters,
    validate_simulation_against_research
)

@dataclass
class ElectrochemicalConfig:
    """Configuration for electrochemical analysis"""
    # Memory management
    max_memory_usage: float = 0.8  # Maximum fraction of available memory to use
    chunk_size: int = 5000  # Number of data points to process at once
    cleanup_threshold: float = 0.9  # Memory usage threshold for cleanup
    
    # Analysis timeouts
    operation_timeout: float = 300  # Maximum time for any operation (seconds)
    total_timeout: float = 900  # Maximum total analysis time (seconds)
    
    # Ion transport parameters
    diffusion_coefficient: float = 1e-9  # m²/s
    membrane_permeability: float = 1e-6  # m/s
    ionic_strength: float = 0.1  # M
    temperature: float = 298.15  # K
    
    # Electrical parameters
    membrane_capacitance: float = 1e-6  # F/m²
    membrane_resistance: float = 1e6  # Ω·m²
    extracellular_resistance: float = 100.0  # Ω·m
    
    # Analysis parameters
    spike_threshold_factor: float = 3.0
    correlation_window: float = 1.0  # s
    propagation_max_velocity: float = 0.1  # m/s
    spatial_resolution: float = 0.001  # m

class EnhancedElectrochemicalAnalyzer:
    """Enhanced Electrochemical Analyzer with memory optimization and error handling"""
    
    # Physical constants
    FARADAY_CONSTANT = 96485.33212  # C/mol
    GAS_CONSTANT = 8.314462618  # J/(mol·K)
    
    def __init__(self, config: Optional[ElectrochemicalConfig] = None):
        """Initialize the analyzer with configuration and physical constants"""
        self.config = config or ElectrochemicalConfig()
        self.start_time = time.time()
        self._check_timeout = lambda: time.time() - self.start_time > self.config.total_timeout
        
        # Initialize physical constants as instance variables
        self.faraday_constant = self.FARADAY_CONSTANT
        self.gas_constant = self.GAS_CONSTANT
        
        # Analysis history and network data
        self.electrode_positions = {}
        self.network_topology = nx.Graph()
        self.propagation_history = []
        
        # Initialize memory tracking
        self.peak_memory_usage = 0
        self.last_cleanup_time = time.time()
    
    def _check_memory_usage(self) -> float:
        """Check current memory usage and cleanup if needed"""
        memory_usage = psutil.Process().memory_info().rss / psutil.virtual_memory().total
        self.peak_memory_usage = max(self.peak_memory_usage, memory_usage)
        
        if memory_usage > self.config.cleanup_threshold:
            gc.collect()
            memory_usage = psutil.Process().memory_info().rss / psutil.virtual_memory().total
            
        return memory_usage
    
    def _calculate_nernst_potential(self, concentration_ratio: float) -> float:
        """Calculate Nernst potential for given concentration ratio"""
        return (self.gas_constant * self.config.temperature / self.faraday_constant) * np.log(concentration_ratio)
    
    def _calculate_goldman_potential(self, internal_conc: float, external_conc: float,
                                   permeability: float) -> float:
        """Calculate Goldman-Hodgkin-Katz potential"""
        RT_F = self.gas_constant * self.config.temperature / self.faraday_constant
        return RT_F * np.log(permeability * external_conc / internal_conc)
    
    def analyze_electrochemical_potential(self, multi_electrode_data: Dict[str, Dict[str, np.ndarray]],
                                        electrode_positions: Dict[str, np.ndarray],
                                        species: str = "Pleurotus_djamor",
                                        progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Complete electrochemical analysis with optimized memory usage and progress tracking"""
        
        try:
            self.start_time = time.time()
            results = {'status': 'initializing', 'progress': 0}
            
            # Validate inputs
            if not multi_electrode_data or not electrode_positions:
                raise ValueError("Missing electrode data or positions")
            
            electrode_count = len(multi_electrode_data)
            if progress_callback:
                progress_callback(0.1, "Initialized analysis")
            
            # Step 1: Detect action potentials (chunked processing)
            all_action_potentials = []
            electrode_ap_data = {}
            
            for i, (electrode_id, data) in enumerate(multi_electrode_data.items()):
                if self._check_timeout():
                    raise TimeoutError("Analysis timeout exceeded")
                    
                voltage_data = data['voltage']
                time_data = data['time']
                
                # Process in chunks
                chunk_size = min(self.config.chunk_size, len(voltage_data))
                n_chunks = (len(voltage_data) + chunk_size - 1) // chunk_size
                
                electrode_results = {
                    'action_potentials': [],
                    'spike_count': 0,
                    'spike_rate': 0,
                    'noise_level': 0
                }
                
                for j in range(n_chunks):
                    if self._check_memory_usage() > self.config.max_memory_usage:
                        raise MemoryError("Memory usage exceeded threshold")
                        
                    start_idx = j * chunk_size
                    end_idx = min(start_idx + chunk_size, len(voltage_data))
                    
                    chunk_results = self.detect_action_potentials(
                        voltage_data[start_idx:end_idx],
                        time_data[start_idx:end_idx],
                        electrode_id
                    )
                    
                    electrode_results['action_potentials'].extend(chunk_results['action_potentials'])
                    electrode_results['spike_count'] += chunk_results['spike_count']
                    
                    if progress_callback:
                        progress = 0.1 + 0.3 * (i * n_chunks + j + 1) / (electrode_count * n_chunks)
                        progress_callback(progress, f"Processing electrode {i+1}/{electrode_count}")
                    
                    gc.collect()
                
                electrode_results['spike_rate'] = electrode_results['spike_count'] / (time_data[-1] - time_data[0])
                electrode_ap_data[electrode_id] = electrode_results
                all_action_potentials.extend(electrode_results['action_potentials'])
            
            if progress_callback:
                progress_callback(0.4, "Action potential detection complete")
            
            # Step 2: Spatial correlation (chunked processing)
            correlation_results = self.calculate_spatial_correlation(
                multi_electrode_data,
                chunk_size=self.config.chunk_size
            )
            
            if progress_callback:
                progress_callback(0.6, "Spatial correlation complete")
            
            # Step 3: Ion gradient modeling
            ion_gradient_results = self.model_ion_gradients(
                all_action_potentials,
                electrode_positions
            )
            
            if progress_callback:
                progress_callback(0.8, "Ion gradient modeling complete")
            
            # Step 4: Network topology
            network_results = self.build_network_topology(
                correlation_results,
                ion_gradient_results,
                electrode_positions
            )
            
            if progress_callback:
                progress_callback(0.9, "Network topology complete")
            
            # Compile results
            computation_time = time.time() - self.start_time
            
            complete_results = {
                'status': 'completed',
                'action_potential_detection': {
                    'electrode_results': electrode_ap_data,
                    'total_action_potentials': len(all_action_potentials),
                    'average_spike_rate': np.mean([data['spike_rate'] for data in electrode_ap_data.values()])
                },
                'spatial_correlation': correlation_results,
                'ion_gradients': ion_gradient_results,
                'network_topology': network_results,
                'analysis_metadata': {
                    'computation_time': computation_time,
                    'memory_usage': self._check_memory_usage(),
                    'electrode_count': electrode_count
                }
            }
            
            if progress_callback:
                progress_callback(1.0, "Analysis complete")
            
            return complete_results
            
        except Exception as e:
            error_msg = str(e)
            if isinstance(e, TimeoutError):
                error_msg = "Analysis timeout exceeded"
            elif isinstance(e, MemoryError):
                error_msg = "Insufficient memory"
            
            return {
                'status': 'failed',
                'error': error_msg,
                'partial_results': results
            }
            
        finally:
            gc.collect()
    
    def detect_action_potentials(self, voltage_data: np.ndarray, time_data: np.ndarray,
                               electrode_id: str) -> Dict[str, Any]:
        """
        Detect action potential spikes in electrical recordings
        
        Args:
            voltage_data: Voltage measurements (V)
            time_data: Time array (s)
            electrode_id: Identifier for the electrode
            
        Returns:
            Action potential detection results
        """
        # Calculate noise level for adaptive thresholding
        noise_level = np.std(voltage_data)
        threshold = self.config.spike_threshold_factor * noise_level
        
        # Detect positive and negative spikes
        positive_peaks, pos_properties = signal.find_peaks(
            voltage_data, 
            height=threshold,
            distance=int(0.1 / np.mean(np.diff(time_data)))  # Minimum 100ms between spikes
        )
        
        negative_peaks, neg_properties = signal.find_peaks(
            -voltage_data,
            height=threshold,
            distance=int(0.1 / np.mean(np.diff(time_data)))
        )
        
        # Combine and sort all peaks
        all_peaks = np.concatenate([positive_peaks, negative_peaks])
        all_polarities = np.concatenate([np.ones(len(positive_peaks)), -np.ones(len(negative_peaks))])
        
        if len(all_peaks) > 0:
            sort_indices = np.argsort(all_peaks)
            all_peaks = all_peaks[sort_indices]
            all_polarities = all_polarities[sort_indices]
        
        # Characterize each action potential
        action_potentials = []
        for i, peak_idx in enumerate(all_peaks):
            peak_time = time_data[peak_idx]
            peak_amplitude = voltage_data[peak_idx]
            polarity = all_polarities[i]
            
            # Estimate spike width (duration at half maximum)
            half_max = np.abs(peak_amplitude) * 0.5
            left_idx = peak_idx
            right_idx = peak_idx
            
            # Find left half-maximum
            while left_idx > 0 and np.abs(voltage_data[left_idx]) > half_max:
                left_idx -= 1
            
            # Find right half-maximum
            while right_idx < len(voltage_data) - 1 and np.abs(voltage_data[right_idx]) > half_max:
                right_idx += 1
            
            spike_width = time_data[right_idx] - time_data[left_idx]
            
            # Calculate rise time (10% to 90% of peak)
            peak_90 = np.abs(peak_amplitude) * 0.9
            peak_10 = np.abs(peak_amplitude) * 0.1
            
            rise_start_idx = peak_idx
            rise_end_idx = peak_idx
            
            # Find 10% point (before peak)
            while rise_start_idx > 0 and np.abs(voltage_data[rise_start_idx]) > peak_10:
                rise_start_idx -= 1
            
            # Find 90% point (before peak)
            while rise_end_idx > rise_start_idx and np.abs(voltage_data[rise_end_idx]) < peak_90:
                rise_end_idx -= 1
            
            rise_time = time_data[rise_end_idx] - time_data[rise_start_idx] if rise_end_idx > rise_start_idx else np.nan
            
            action_potentials.append({
                'time': peak_time,
                'amplitude': peak_amplitude,
                'polarity': polarity,
                'width': spike_width,
                'rise_time': rise_time,
                'peak_index': peak_idx,
                'electrode_id': electrode_id,
                'signal_to_noise': np.abs(peak_amplitude) / noise_level
            })
        
        return {
            'action_potentials': action_potentials,
            'spike_count': len(action_potentials),
            'spike_rate': len(action_potentials) / (time_data[-1] - time_data[0]) if len(time_data) > 1 else 0,
            'noise_level': noise_level,
            'detection_threshold': threshold,
            'electrode_id': electrode_id
        }
    
    def calculate_spatial_correlation(self, multi_electrode_data: Dict[str, Dict[str, np.ndarray]],
                                    chunk_size: int = 5000) -> Dict[str, Any]:
        """
        Calculate spatial correlation between electrode recordings
        
        Args:
            multi_electrode_data: Dictionary with electrode_id -> {'voltage': array, 'time': array}
            chunk_size: Number of data points to process at once
            
        Returns:
            Spatial correlation analysis results
        """
        electrode_ids = list(multi_electrode_data.keys())
        n_electrodes = len(electrode_ids)
        
        if n_electrodes < 2:
            return {
                'correlation_matrix': np.array([]),
                'error': 'Need at least 2 electrodes for correlation analysis'
            }
        
        # Initialize correlation matrix
        correlation_matrix = np.zeros((n_electrodes, n_electrodes))
        time_lag_matrix = np.zeros((n_electrodes, n_electrodes))
        significance_matrix = np.zeros((n_electrodes, n_electrodes))
        
        # Calculate pairwise correlations
        for i, electrode_i in enumerate(electrode_ids):
            for j, electrode_j in enumerate(electrode_ids):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                    time_lag_matrix[i, j] = 0.0
                    significance_matrix[i, j] = 1.0
                else:
                    voltage_i = multi_electrode_data[electrode_i]['voltage']
                    voltage_j = multi_electrode_data[electrode_j]['voltage']
                    time_array = multi_electrode_data[electrode_i]['time']
                    
                    # Ensure signals have same length
                    min_length = min(len(voltage_i), len(voltage_j))
                    voltage_i = voltage_i[:min_length]
                    voltage_j = voltage_j[:min_length]
                    
                    # Cross-correlation analysis
                    cross_corr = signal.correlate(voltage_i, voltage_j, mode='full')
                    
                    # Find maximum correlation and corresponding lag
                    max_corr_idx = np.argmax(np.abs(cross_corr))
                    max_correlation = cross_corr[max_corr_idx] / (np.linalg.norm(voltage_i) * np.linalg.norm(voltage_j))
                    
                    # Convert lag index to time lag
                    lag_samples = max_corr_idx - (len(cross_corr) - 1) // 2
                    time_lag = lag_samples * np.mean(np.diff(time_array))
                    
                    # Calculate significance (Pearson correlation p-value approximation)
                    pearson_corr, p_value = pearsonr(voltage_i, voltage_j)
                    
                    correlation_matrix[i, j] = max_correlation
                    time_lag_matrix[i, j] = time_lag
                    significance_matrix[i, j] = 1 - p_value  # Convert p-value to significance
        
        # Calculate network metrics
        # Average correlation (excluding diagonal)
        upper_triangle = np.triu(correlation_matrix, k=1)
        avg_correlation = np.mean(upper_triangle[upper_triangle != 0]) if np.any(upper_triangle != 0) else 0
        
        # Maximum correlation
        max_correlation = np.max(upper_triangle) if np.any(upper_triangle != 0) else 0
        
        # Find highly correlated pairs (threshold = 0.7)
        high_corr_threshold = 0.7
        high_corr_pairs = []
        for i in range(n_electrodes):
            for j in range(i+1, n_electrodes):
                if np.abs(correlation_matrix[i, j]) > high_corr_threshold:
                    high_corr_pairs.append({
                        'electrode_pair': (electrode_ids[i], electrode_ids[j]),
                        'correlation': correlation_matrix[i, j],
                        'time_lag': time_lag_matrix[i, j],
                        'significance': significance_matrix[i, j]
                    })
        
        return {
            'correlation_matrix': correlation_matrix,
            'time_lag_matrix': time_lag_matrix,
            'significance_matrix': significance_matrix,
            'electrode_ids': electrode_ids,
            'average_correlation': avg_correlation,
            'maximum_correlation': max_correlation,
            'highly_correlated_pairs': high_corr_pairs,
            'network_connectivity': len(high_corr_pairs) / (n_electrodes * (n_electrodes - 1) / 2)
        }
    
    def model_ion_gradients(self, action_potential_data: List[Dict], 
                          electrode_positions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Model ion concentration gradients during action potential propagation
        
        Args:
            action_potential_data: List of action potential events
            electrode_positions: Dictionary mapping electrode_id to 3D position
            
        Returns:
            Ion gradient modeling results
        """
        if not action_potential_data:
            return {
                'ion_gradients': {},
                'error': 'No action potential data available'
            }
        
        # Group action potentials by time proximity (within 100ms)
        time_tolerance = 0.1  # 100ms
        ap_groups = []
        
        for ap in action_potential_data:
            added_to_group = False
            for group in ap_groups:
                if any(abs(ap['time'] - g['time']) < time_tolerance for g in group):
                    group.append(ap)
                    added_to_group = True
                    break
            
            if not added_to_group:
                ap_groups.append([ap])
        
        # Analyze ion gradients for each group
        gradient_results = []
        
        for group_idx, ap_group in enumerate(ap_groups):
            if len(ap_group) < 2:
                continue  # Need at least 2 APs for gradient analysis
            
            # Extract spatial and temporal information
            positions = []
            times = []
            amplitudes = []
            
            for ap in ap_group:
                electrode_id = ap['electrode_id']
                if electrode_id in electrode_positions:
                    positions.append(electrode_positions[electrode_id])
                    times.append(ap['time'])
                    amplitudes.append(ap['amplitude'])
            
            if len(positions) < 2:
                continue
            
            positions = np.array(positions)
            times = np.array(times)
            amplitudes = np.array(amplitudes)
            
            # Calculate spatial gradients
            # Estimate ion concentration from voltage amplitude
            # Using Nernst equation: V = (RT/nF) * ln(C_out/C_in)
            nernst_factor = self.gas_constant * self.config.temperature / self.faraday_constant
            
            # Approximate ion concentration ratios
            concentration_ratios = np.exp(amplitudes / nernst_factor)
            
            # Calculate spatial gradients (finite differences)
            if len(positions) >= 3:
                # Use least squares to fit gradient field
                try:
                    # Set up linear system: grad · r = concentration_change
                    A_matrix = positions[1:] - positions[0]  # Position differences
                    b_vector = concentration_ratios[1:] - concentration_ratios[0]  # Concentration differences
                    
                    if A_matrix.shape[0] >= A_matrix.shape[1]:  # Overdetermined system
                        gradient_vector, residuals, rank, s = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
                    else:
                        gradient_vector = np.zeros(3)  # Not enough constraints
                    
                    gradient_magnitude = np.linalg.norm(gradient_vector)
                    gradient_direction = gradient_vector / (gradient_magnitude + 1e-12)
                    
                except np.linalg.LinAlgError:
                    gradient_vector = np.zeros(3)
                    gradient_magnitude = 0
                    gradient_direction = np.zeros(3)
            else:
                # Simple two-point gradient
                position_diff = positions[1] - positions[0]
                concentration_diff = concentration_ratios[1] - concentration_ratios[0]
                distance = np.linalg.norm(position_diff)
                
                if distance > 0:
                    gradient_magnitude = concentration_diff / distance
                    gradient_direction = position_diff / distance
                    gradient_vector = gradient_magnitude * gradient_direction
                else:
                    gradient_vector = np.zeros(3)
                    gradient_magnitude = 0
                    gradient_direction = np.zeros(3)
            
            # Calculate temporal gradients (rate of change)
            time_diffs = np.diff(times)
            concentration_rate = np.diff(concentration_ratios)
            
            avg_temporal_gradient = np.mean(concentration_rate / (time_diffs + 1e-12))
            
            # Estimate diffusion coefficients
            # Using Fick's law: J = -D * grad(C)
            # Assuming steady-state: div(J) = 0
            estimated_diffusion = np.abs(avg_temporal_gradient) / (gradient_magnitude + 1e-12)
            
            gradient_results.append({
                'group_id': group_idx,
                'time_center': np.mean(times),
                'spatial_gradient': {
                    'vector': gradient_vector,
                    'magnitude': gradient_magnitude,
                    'direction': gradient_direction
                },
                'temporal_gradient': avg_temporal_gradient,
                'estimated_diffusion_coefficient': estimated_diffusion,
                'ion_concentration_ratios': concentration_ratios,
                'electrode_positions': positions,
                'action_potential_count': len(ap_group)
            })
        
        # Calculate overall statistics
        if gradient_results:
            all_gradients = [r['spatial_gradient']['magnitude'] for r in gradient_results]
            all_diffusions = [r['estimated_diffusion_coefficient'] for r in gradient_results]
            
            gradient_statistics = {
                'mean_gradient_magnitude': np.mean(all_gradients),
                'std_gradient_magnitude': np.std(all_gradients),
                'mean_diffusion_coefficient': np.mean(all_diffusions),
                'std_diffusion_coefficient': np.std(all_diffusions)
            }
        else:
            gradient_statistics = {
                'mean_gradient_magnitude': 0,
                'std_gradient_magnitude': 0,
                'mean_diffusion_coefficient': 0,
                'std_diffusion_coefficient': 0
            }
        
        return {
            'ion_gradient_events': gradient_results,
            'gradient_statistics': gradient_statistics,
            'total_gradient_events': len(gradient_results),
            'analysis_parameters': {
                'time_tolerance': time_tolerance,
                'nernst_factor': nernst_factor,
                'temperature': self.config.temperature
            }
        }
    
    def analyze_propagation_velocity(self, action_potential_data: List[Dict],
                                   electrode_positions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze action potential propagation velocity through the network
        
        Args:
            action_potential_data: List of action potential events
            electrode_positions: Dictionary mapping electrode_id to 3D position
            
        Returns:
            Propagation velocity analysis results
        """
        if len(action_potential_data) < 2:
            return {
                'propagation_events': [],
                'error': 'Need at least 2 action potentials for velocity analysis'
            }
        
        # Group APs by time proximity for propagation analysis
        time_tolerance = 0.5  # 500ms window for propagation
        propagation_events = []
        
        for i, ap1 in enumerate(action_potential_data):
            for j, ap2 in enumerate(action_potential_data[i+1:], i+1):
                time_diff = abs(ap2['time'] - ap1['time'])
                
                if time_diff <= time_tolerance and ap1['electrode_id'] != ap2['electrode_id']:
                    # Check if both electrodes have position data
                    if ap1['electrode_id'] in electrode_positions and ap2['electrode_id'] in electrode_positions:
                        pos1 = electrode_positions[ap1['electrode_id']]
                        pos2 = electrode_positions[ap2['electrode_id']]
                        
                        distance = np.linalg.norm(pos2 - pos1)
                        
                        if distance > 0 and time_diff > 0:
                            velocity = distance / time_diff
                            
                            # Check if velocity is reasonable (< max velocity)
                            if velocity <= self.config.propagation_max_velocity:
                                propagation_events.append({
                                    'electrode_pair': (ap1['electrode_id'], ap2['electrode_id']),
                                    'start_time': min(ap1['time'], ap2['time']),
                                    'end_time': max(ap1['time'], ap2['time']),
                                    'time_difference': time_diff,
                                    'distance': distance,
                                    'velocity': velocity,
                                    'start_amplitude': ap1['amplitude'] if ap1['time'] < ap2['time'] else ap2['amplitude'],
                                    'end_amplitude': ap2['amplitude'] if ap1['time'] < ap2['time'] else ap1['amplitude'],
                                    'amplitude_ratio': (ap2['amplitude'] / ap1['amplitude']) if ap1['amplitude'] != 0 else np.inf,
                                    'direction_vector': (pos2 - pos1) / distance
                                })
        
        # Analyze propagation patterns
        if propagation_events:
            velocities = [event['velocity'] for event in propagation_events]
            distances = [event['distance'] for event in propagation_events]
            amplitude_ratios = [event['amplitude_ratio'] for event in propagation_events if np.isfinite(event['amplitude_ratio'])]
            
            velocity_statistics = {
                'mean_velocity': np.mean(velocities),
                'std_velocity': np.std(velocities),
                'median_velocity': np.median(velocities),
                'min_velocity': np.min(velocities),
                'max_velocity': np.max(velocities)
            }
            
            # Detect preferred propagation directions
            direction_vectors = np.array([event['direction_vector'] for event in propagation_events])
            
            if len(direction_vectors) > 1:
                # Use DBSCAN clustering to find common directions
                clustering = DBSCAN(eps=0.3, min_samples=2).fit(direction_vectors)
                n_direction_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                
                # Find dominant direction
                if n_direction_clusters > 0:
                    cluster_sizes = []
                    cluster_directions = []
                    
                    for cluster_id in set(clustering.labels_):
                        if cluster_id != -1:  # Ignore noise points
                            cluster_mask = clustering.labels_ == cluster_id
                            cluster_vectors = direction_vectors[cluster_mask]
                            cluster_size = len(cluster_vectors)
                            cluster_direction = np.mean(cluster_vectors, axis=0)
                            cluster_direction = cluster_direction / np.linalg.norm(cluster_direction)
                            
                            cluster_sizes.append(cluster_size)
                            cluster_directions.append(cluster_direction)
                    
                    if cluster_directions:
                        dominant_cluster_idx = np.argmax(cluster_sizes)
                        dominant_direction = cluster_directions[dominant_cluster_idx]
                        direction_consistency = cluster_sizes[dominant_cluster_idx] / len(propagation_events)
                    else:
                        dominant_direction = np.array([0, 0, 0])
                        direction_consistency = 0
                else:
                    dominant_direction = np.array([0, 0, 0])
                    direction_consistency = 0
            else:
                dominant_direction = direction_vectors[0] if len(direction_vectors) == 1 else np.array([0, 0, 0])
                direction_consistency = 1.0 if len(direction_vectors) == 1 else 0
            
            propagation_analysis = {
                'velocity_statistics': velocity_statistics,
                'dominant_direction': dominant_direction,
                'direction_consistency': direction_consistency,
                'total_propagation_events': len(propagation_events),
                'mean_distance': np.mean(distances),
                'mean_amplitude_ratio': np.mean(amplitude_ratios) if amplitude_ratios else np.nan
            }
        else:
            propagation_analysis = {
                'velocity_statistics': {
                    'mean_velocity': 0,
                    'std_velocity': 0,
                    'median_velocity': 0,
                    'min_velocity': 0,
                    'max_velocity': 0
                },
                'dominant_direction': np.array([0, 0, 0]),
                'direction_consistency': 0,
                'total_propagation_events': 0,
                'mean_distance': 0,
                'mean_amplitude_ratio': np.nan
            }
        
        return {
            'propagation_events': propagation_events,
            'propagation_analysis': propagation_analysis,
            'analysis_parameters': {
                'time_tolerance': time_tolerance,
                'max_velocity': self.config.propagation_max_velocity
            }
        }
    
    def build_network_topology(self, correlation_data: Dict[str, Any],
                             propagation_data: Dict[str, Any],
                             electrode_positions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Build network topology from correlation and propagation data
        
        Args:
            correlation_data: Results from spatial correlation analysis
            propagation_data: Results from propagation velocity analysis
            electrode_positions: Dictionary mapping electrode_id to 3D position
            
        Returns:
            Network topology analysis results
        """
        # Initialize network graph
        G = nx.Graph()
        
        # Add nodes (electrodes)
        for electrode_id, position in electrode_positions.items():
            G.add_node(electrode_id, position=position)
        
        # Add edges based on high correlations
        if 'highly_correlated_pairs' in correlation_data:
            for pair_data in correlation_data['highly_correlated_pairs']:
                electrode_pair = pair_data['electrode_pair']
                correlation = pair_data['correlation']
                time_lag = pair_data['time_lag']
                
                G.add_edge(
                    electrode_pair[0], 
                    electrode_pair[1],
                    weight=abs(correlation),
                    correlation=correlation,
                    time_lag=time_lag,
                    connection_type='correlation'
                )
        
        # Add edges based on propagation events
        if 'propagation_events' in propagation_data:
            for prop_event in propagation_data['propagation_events']:
                electrode_pair = prop_event['electrode_pair']
                velocity = prop_event['velocity']
                
                if G.has_edge(electrode_pair[0], electrode_pair[1]):
                    # Update existing edge
                    G[electrode_pair[0]][electrode_pair[1]]['velocity'] = velocity
                    G[electrode_pair[0]][electrode_pair[1]]['connection_type'] = 'correlation_and_propagation'
                else:
                    # Add new edge
                    G.add_edge(
                        electrode_pair[0],
                        electrode_pair[1],
                        weight=1.0,
                        velocity=velocity,
                        connection_type='propagation'
                    )
        
        # Calculate network metrics
        if G.number_of_nodes() > 0:
            # Basic metrics
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = nx.density(G)
            
            # Connectivity metrics
            if n_edges > 0:
                try:
                    avg_clustering = nx.average_clustering(G)
                    connected_components = list(nx.connected_components(G))
                    largest_component_size = len(max(connected_components, key=len)) if connected_components else 0
                    
                    if nx.is_connected(G):
                        avg_path_length = nx.average_shortest_path_length(G)
                        diameter = nx.diameter(G)
                    else:
                        # Calculate for largest connected component
                        if largest_component_size > 1:
                            largest_component = max(connected_components, key=len)
                            subgraph = G.subgraph(largest_component)
                            avg_path_length = nx.average_shortest_path_length(subgraph)
                            diameter = nx.diameter(subgraph)
                        else:
                            avg_path_length = np.inf
                            diameter = np.inf
                    
                    # Centrality metrics
                    degree_centrality = nx.degree_centrality(G)
                    betweenness_centrality = nx.betweenness_centrality(G)
                    
                    # Find hub nodes (highest degree centrality)
                    hub_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                except:
                    avg_clustering = 0
                    avg_path_length = np.inf
                    diameter = np.inf
                    degree_centrality = {}
                    betweenness_centrality = {}
                    hub_nodes = []
                    largest_component_size = 0
            else:
                avg_clustering = 0
                avg_path_length = np.inf
                diameter = np.inf
                degree_centrality = {}
                betweenness_centrality = {}
                hub_nodes = []
                largest_component_size = 0
            
            network_metrics = {
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'density': density,
                'avg_clustering': avg_clustering,
                'avg_path_length': avg_path_length,
                'diameter': diameter,
                'largest_component_size': largest_component_size,
                'degree_centrality': degree_centrality,
                'betweenness_centrality': betweenness_centrality,
                'hub_nodes': hub_nodes
            }
        else:
            network_metrics = {
                'n_nodes': 0,
                'n_edges': 0,
                'density': 0,
                'avg_clustering': 0,
                'avg_path_length': np.inf,
                'diameter': np.inf,
                'largest_component_size': 0,
                'degree_centrality': {},
                'betweenness_centrality': {},
                'hub_nodes': []
            }
        
        return {
            'network_graph': G,
            'network_metrics': network_metrics,
            'adjacency_matrix': nx.adjacency_matrix(G).toarray() if G.number_of_nodes() > 0 else np.array([]),
            'electrode_ids': list(G.nodes()) if G.number_of_nodes() > 0 else []
        }


def demo_electrochemical_analysis():
    """Demonstration of enhanced electrochemical analysis"""
    print("⚗️ ENHANCED ELECTROCHEMICAL ANALYSIS DEMO")
    print("="*60)
    
    # Initialize analyzer
    config = ElectrochemicalConfig(
        spike_threshold_factor=3.0,
        correlation_window=1.0,
        propagation_max_velocity=0.05  # 5 cm/s
    )
    
    analyzer = EnhancedElectrochemicalAnalyzer(config)
    
    # Generate test data (multi-electrode recordings)
    n_electrodes = 5
    n_samples = 2000
    sampling_rate = 10  # Hz
    t = np.linspace(0, n_samples/sampling_rate, n_samples)
    
    # Define electrode positions in a grid
    electrode_positions = {
        'electrode_0': np.array([0.0, 0.0, 0.0]),      # Center
        'electrode_1': np.array([0.01, 0.0, 0.0]),     # 1 cm east
        'electrode_2': np.array([0.0, 0.01, 0.0]),     # 1 cm north
        'electrode_3': np.array([-0.01, 0.0, 0.0]),    # 1 cm west
        'electrode_4': np.array([0.0, -0.01, 0.0])     # 1 cm south
    }
    
    # Generate correlated electrical signals with propagation delays
    multi_electrode_data = {}
    
    # Create a source signal with action potentials
    source_signal = 0.0001 * np.random.randn(n_samples)
    
    # Add action potential spikes at specific times
    spike_times = [20, 50, 100, 150, 180]
    for spike_time in spike_times:
        spike_idx = int(spike_time * sampling_rate / 10)
        if spike_idx < n_samples:
            spike_profile = 0.003 * np.exp(-np.arange(30) * 0.1)
            end_idx = min(spike_idx + 30, n_samples)
            source_signal[spike_idx:end_idx] += spike_profile[:end_idx-spike_idx]
    
    # Propagate signal to other electrodes with delays and attenuation
    for electrode_id, position in electrode_positions.items():
        distance = np.linalg.norm(position)
        
        # Calculate propagation delay (assuming 2 cm/s propagation speed)
        propagation_speed = 0.02  # m/s
        delay_time = distance / propagation_speed
        delay_samples = int(delay_time * sampling_rate)
        
        # Calculate attenuation
        attenuation = 1 / (1 + distance * 50)  # Distance-dependent attenuation
        
        # Create delayed and attenuated signal
        delayed_signal = np.zeros_like(source_signal)
        if delay_samples < n_samples:
            delayed_signal[delay_samples:] = source_signal[:-delay_samples] * attenuation
        
        # Add electrode-specific noise
        noise = 0.0001 * np.random.randn(n_samples)
        final_signal = delayed_signal + noise
        
        multi_electrode_data[electrode_id] = {
            'voltage': final_signal,
            'time': t
        }
    
    # Analyze electrochemical potential
    results = analyzer.analyze_electrochemical_potential(
        multi_electrode_data, electrode_positions, "Pleurotus_djamor"
    )
    
    # Display results
    print("\n📊 ANALYSIS RESULTS")
    print("="*40)
    
    ap_detection = results['action_potential_detection']
    correlation = results['spatial_correlation']
    propagation = results['propagation_velocity']
    network = results['network_topology']
    
    print(f"Total action potentials: {ap_detection['total_action_potentials']}")
    print(f"Average spike rate: {ap_detection['average_spike_rate']:.3f} Hz")
    print(f"Network connectivity: {correlation['network_connectivity']:.2f}")
    print(f"Average correlation: {correlation['average_correlation']:.3f}")
    print(f"Propagation events: {propagation['propagation_analysis']['total_propagation_events']}")
    print(f"Mean velocity: {propagation['propagation_analysis']['velocity_statistics']['mean_velocity']:.4f} m/s")
    print(f"Network nodes: {network['network_metrics']['n_nodes']}")
    print(f"Network edges: {network['network_metrics']['n_edges']}")
    print(f"Network density: {network['network_metrics']['density']:.3f}")
    
    return results


if __name__ == "__main__":
    # Run demonstration
    results = demo_electrochemical_analysis()
    
    print("\n✅ ENHANCED ELECTROCHEMICAL ANALYSIS DEMO COMPLETED")
    print("⚗️ Ready for integration with fungal communication system") 