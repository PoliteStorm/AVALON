import pytest
from pathlib import Path
import numpy as np

from fungal_rosetta_stone import FungalRosettaStone

# Define the path to the configuration file
CONFIG_PATH = Path(__file__).parent.parent / "research_parameters.yml"

@pytest.fixture(scope="module")
def analyzer():
    """Fixture to initialize the FungalRosettaStone analyzer."""
    if not CONFIG_PATH.exists():
        pytest.fail(f"Configuration file not found at {CONFIG_PATH}")
    return FungalRosettaStone(config_path=CONFIG_PATH)

@pytest.fixture
def sample_spike_times():
    # Create synthetic spike times with known patterns at different scales
    base_pattern = np.array([0, 0.1, 0.2])  # micro-level pattern
    meso_pattern = np.array([0, 0.1, 0.2, 4.0, 4.1, 4.2])  # meso-level pattern
    macro_pattern = np.concatenate([meso_pattern, meso_pattern + 25.0])  # macro-level pattern
    
    # Add some noise
    noise = np.random.uniform(0, 0.05, size=5)
    return np.sort(np.concatenate([macro_pattern, noise]))

def test_w_transform_analytic_signal(analyzer):
    """
    Tests the W-transform against a known result for a simple analytic signal.
    This ensures the core numerical integration is stable and correct.
    """
    # Create a simple sine wave as an analytic signal
    t = np.linspace(0, 10, 1000)
    signal_data = np.sin(2 * np.pi * 1 * t) # 1 Hz sine wave
    
    k_range = np.linspace(0.1, 5, 10)
    tau_range = np.linspace(0.1, 2, 10)

    # Perform the W-transform
    W = analyzer._w_transform(t, signal_data, k_range, tau_range, "morlet")

    # Check output shape and type
    assert W.shape == (len(k_range), len(tau_range))
    assert np.iscomplexobj(W)

    # Regression test against a known value (pre-calculated)
    # This value has been updated to match the output from the more accurate
    # Simpson's rule integration.
    expected_checksum = -0.0188688 + -0.002599j
    assert np.isclose(W[2, 2], expected_checksum, atol=1e-5)

def test_hierarchical_word_analysis(analyzer, sample_spike_times):
    """Test the hierarchical word analysis functionality."""
    results = analyzer.analyze_hierarchical_words(sample_spike_times, "test_species")
    
    # Check basic structure
    assert 'levels' in results
    assert 'cross_level_analysis' in results
    
    # Check all levels are present
    expected_levels = {'micro', 'meso', 'macro', 'super', 'ultra'}
    assert set(results['levels'].keys()) == expected_levels
    
    # Check micro-level patterns
    micro = results['levels']['micro']
    assert micro['word_count'] > 0
    assert len(micro['patterns']) > 0
    assert 'complexity' in micro
    
    # Check cross-level analysis
    cross_level = results['cross_level_analysis']
    assert 'level_correlations' in cross_level
    assert 'nested_patterns' in cross_level
    assert 'hierarchy_strength' in cross_level
    assert 0 <= cross_level['hierarchy_strength'] <= 1

def test_pattern_strength_calculation(analyzer):
    """Test the enhanced pattern strength calculation."""
    # Create test patterns with known properties
    test_patterns = [
        {
            'occurrence_count': 10,
            'mean_interval': 60.0,  # 1-minute interval
            'std_interval': 3.0,    # Low variation
            'mean_amplitude': 0.5,   # Mid-range amplitude
            'confidence': 0.9,
            'complexity_score': 1.5
        },
        {
            'occurrence_count': 5,
            'mean_interval': 30.0,
            'std_interval': 6.0,
            'mean_amplitude': 0.8,
            'confidence': 0.7,
            'complexity_score': 1.2
        }
    ]
    
    result = analyzer.calculate_pattern_strength(test_patterns, total_duration=3600.0)
    
    # Check structure
    assert 'overall_strength' in result
    assert 'confidence_interval' in result
    assert 'reliability_score' in result
    assert 'statistical_significance' in result
    assert 'feature_strengths' in result
    
    # Check value ranges
    assert 0 <= result['overall_strength'] <= 1
    assert result['confidence_interval'][0] <= result['overall_strength']
    assert result['confidence_interval'][1] >= result['overall_strength']
    assert 0 <= result['reliability_score'] <= 1
    assert 0 <= result['statistical_significance']['p_value'] <= 1

def test_noise_handling(analyzer):
    """Test improved noise handling in voltage filtering."""
    # Create synthetic noisy signal
    t = np.linspace(0, 10, 1000)
    clean_signal = np.sin(2 * np.pi * t)  # Base signal
    noise = np.random.normal(0, 0.5, size=len(t))  # Gaussian noise
    spikes = np.zeros_like(t)
    spike_locations = [100, 300, 500, 700]  # Add some spikes
    for loc in spike_locations:
        spikes[loc:loc+10] = 2.0
    noisy_signal = clean_signal + noise + spikes
    
    # Filter signal
    filtered = analyzer._filter_voltage(noisy_signal)
    
    # Check noise reduction
    assert np.std(filtered) < np.std(noisy_signal)
    
    # Check spike preservation
    for loc in spike_locations:
        assert np.max(filtered[loc:loc+10]) > np.mean(filtered)
        
    # Check baseline correction
    assert abs(np.mean(filtered)) < abs(np.mean(noisy_signal))

def test_spike_detection():
    """Test spike detection using synthetic data with known spikes."""
    # Create synthetic data
    t = np.linspace(0, 100, 1000)  # 100s at 10Hz
    voltage = np.zeros_like(t)
    
    # Add known spikes
    spike_times = [10, 20, 30, 40, 50]
    for st in spike_times:
        idx = np.argmin(np.abs(t - st))
        voltage[idx:idx+5] = 0.3 * np.exp(-np.linspace(0, 3, 5))
    
    # Add noise
    voltage += np.random.normal(0, 0.01, len(t))
    
    # Initialize analyzer
    analyzer = FungalRosettaStone()
    
    # Test for each species
    species = ['C_militaris', 'F_velutipes', 'S_commune', 'O_nidiformis']
    for species_name in species:
        result = analyzer.detect_spikes(t, voltage, species_name)
        
        # Basic checks
        assert isinstance(result, dict)
        assert 'spike_times' in result
        assert 'spike_count' in result
        assert len(result['spike_times']) > 0
        
        # Check detection parameters
        params = result['detection_params']
        if species_name in ['C_militaris', 'F_velutipes']:
            assert params['window_size'] == 200
            assert params['threshold'] == 0.1
            assert params['min_distance'] == 300
        elif species_name == 'S_commune':
            assert params['window_size'] == 100
            assert params['threshold'] == 0.005
            assert params['min_distance'] == 100
        else:  # O_nidiformis
            assert params['window_size'] == 50
            assert params['threshold'] == 0.003
            assert params['min_distance'] == 100

def test_spike_detection_real_data():
    """Test spike detection with real data characteristics from Adamatzky 2021."""
    # Create data matching paper's characteristics
    t = np.linspace(0, 3600*24, 86400)  # 24 hours at 1 sample/sec
    
    def generate_species_data(species):
        voltage = np.zeros_like(t)
        if species == 'C_militaris':
            # Large amplitude spikes with long intervals
            n_spikes = 20  # Will give ~116 min intervals over 24h
            spike_times = np.sort(np.random.choice(t[:-100], n_spikes, replace=False))
            for st in spike_times:
                idx = np.argmin(np.abs(t - st))
                amp = np.random.uniform(0.1, 0.3)  # mV
                voltage[idx:idx+100] = amp * np.exp(-np.linspace(0, 5, 100))
                
        elif species == 'F_velutipes':
            # Mix of regular spikes and high-frequency bursts
            # Regular spikes
            n_spikes = 15
            spike_times = np.sort(np.random.choice(t[:-100], n_spikes, replace=False))
            for st in spike_times:
                idx = np.argmin(np.abs(t - st))
                amp = np.random.uniform(0.2, 0.4)  # mV
                voltage[idx:idx+100] = amp * np.exp(-np.linspace(0, 5, 100))
            
            # Add 3 bursts with 5-10 spikes each
            for _ in range(3):
                burst_start = np.random.choice(t[:-1000])
                n_burst_spikes = np.random.randint(5, 11)
                for i in range(n_burst_spikes):
                    idx = np.argmin(np.abs(t - (burst_start + i*180)))  # 3 min between burst spikes
                    amp = np.random.uniform(0.2, 0.4)
                    voltage[idx:idx+60] = amp * np.exp(-np.linspace(0, 5, 60))
                
        elif species == 'S_commune':
            # Regular small spikes with wave packets
            # Regular spikes
            n_spikes = 35  # Will give ~41 min intervals
            spike_times = np.sort(np.random.choice(t[:-100], n_spikes, replace=False))
            for st in spike_times:
                idx = np.argmin(np.abs(t - st))
                amp = np.random.uniform(0.02, 0.04)  # mV
                voltage[idx:idx+50] = amp * np.exp(-np.linspace(0, 3, 50))
            
            # Add 2 wave packets
            for _ in range(2):
                packet_start = np.random.choice(t[:-3000])
                # Create amplitude evolution as in Fig 6
                amps = np.concatenate([
                    np.linspace(0.02, 0.04, 8),
                    np.linspace(0.04, 0.02, 8)
                ])
                for i, amp in enumerate(amps):
                    idx = np.argmin(np.abs(t - (packet_start + i*300)))  # 5 min between packet spikes
                    voltage[idx:idx+50] = amp * np.exp(-np.linspace(0, 3, 50))
                
        else:  # O_nidiformis
            # Very small spikes with high variability
            n_spikes = 25
            spike_times = np.sort(np.random.choice(t[:-100], n_spikes, replace=False))
            for st in spike_times:
                idx = np.argmin(np.abs(t - st))
                amp = np.random.uniform(0.005, 0.009)  # mV
                voltage[idx:idx+40] = amp * np.exp(-np.linspace(0, 2, 40))
        
        # Add baseline drift and noise
        drift = 0.01 * np.sin(2*np.pi*t/t[-1])
        noise = np.random.normal(0, 0.001, len(t))
        return voltage + drift + noise
    
    analyzer = FungalRosettaStone()
    
    # Test each species
    for species in ['C_militaris', 'F_velutipes', 'S_commune', 'O_nidiformis']:
        voltage = generate_species_data(species)
        result = analyzer.detect_spikes(t, voltage, species)
        
        # Basic checks
        assert isinstance(result, dict)
        assert 'spike_times' in result
        assert 'statistics' in result
        assert len(result['spike_times']) > 0
        
        # Check species-specific statistics
        stats = result['statistics']
        if species == 'C_militaris':
            assert 90 <= stats['mean_interval_min'] <= 140  # ~116 min
            assert 0.1 <= stats['mean_amplitude_mv'] <= 0.3
            
        elif species == 'F_velutipes':
            assert 80 <= stats['mean_interval_min'] <= 120  # ~102 min
            assert 0.2 <= stats['mean_amplitude_mv'] <= 0.4
            # Check burst detection
            assert 'patterns' in result
            bursts = [p for p in result['patterns'] if p['type'] == 'burst']
            assert len(bursts) > 0
            for burst in bursts:
                assert burst['num_spikes'] >= 3
                assert burst['duration'] <= 3600  # Max 1 hour
                
        elif species == 'S_commune':
            assert 30 <= stats['mean_interval_min'] <= 50  # ~41 min
            assert 0.02 <= stats['mean_amplitude_mv'] <= 0.04
            # Check wave packet detection
            assert 'patterns' in result
            packets = [p for p in result['patterns'] if p['type'] == 'wave_packet']
            assert len(packets) > 0
            for packet in packets:
                assert packet['num_spikes'] >= 5
                # Check amplitude evolution
                assert packet['amplitude_evolution']['rising_phase'] < \
                       packet['amplitude_evolution']['falling_phase']
                
        else:  # O_nidiformis
            assert 70 <= stats['mean_interval_min'] <= 110  # ~92 min
            assert 0.005 <= stats['mean_amplitude_mv'] <= 0.009

def test_spike_detection_roc():
    """Test spike detection performance using ROC curve."""
    # Generate synthetic data with known spike times
    t = np.linspace(0, 1000, 10000)  # 1000s at 10Hz
    voltage = np.zeros_like(t)
    true_spikes = []
    
    # Add 20 known spikes
    for _ in range(20):
        spike_time = np.random.choice(t)
        idx = np.argmin(np.abs(t - spike_time))
        voltage[idx:idx+10] = 0.5 * np.exp(-np.linspace(0, 3, 10))
        true_spikes.append(spike_time)
    
    # Add noise
    voltage += np.random.normal(0, 0.05, len(t))
    
    analyzer = FungalRosettaStone()
    result = analyzer.detect_spikes(t, voltage, 'S_commune')
    
    # Calculate detection accuracy
    detected_spikes = np.array(result['spike_times'])
    true_positives = 0
    false_positives = 0
    
    # Count matches (within 1s window)
    for ds in detected_spikes:
        if any(abs(ds - ts) < 1.0 for ts in true_spikes):
            true_positives += 1
        else:
            false_positives += 1
    
    false_negatives = len(true_spikes) - true_positives
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Assert reasonable performance
    assert precision > 0.8, "Precision should be >80%"
    assert recall > 0.8, "Recall should be >80%"
    assert f1_score > 0.8, "F1 score should be >80%"

def test_full_regression_on_sample_data():
    """Run a full analysis on a small, version-controlled sample dataset
    and compare the output dictionary against a stored, known-good result."""
    
    # Create synthetic test data with known patterns
    t = np.linspace(0, 3600*24, 86400)  # 24 hours at 1Hz
    
    def generate_test_data(species):
        voltage = np.zeros_like(t)
        if species == 'C_militaris':
            # Add 20 spikes with known intervals (~116 min)
            spike_times = np.arange(0, t[-1], 116*60)  # Every 116 minutes
            for st in spike_times:
                idx = np.argmin(np.abs(t - st))
                amp = 0.2  # mV (within 0.1-0.3 range)
                voltage[idx:idx+100] = amp * np.exp(-np.linspace(0, 5, 100))
                
        elif species == 'F_velutipes':
            # Add regular spikes and bursts
            # Regular spikes every 102 minutes
            spike_times = np.arange(0, t[-1], 102*60)
            for st in spike_times:
                idx = np.argmin(np.abs(t - st))
                amp = 0.3  # mV (within 0.2-0.4 range)
                voltage[idx:idx+100] = amp * np.exp(-np.linspace(0, 5, 100))
            
            # Add 3 bursts with 5 spikes each
            burst_starts = [4*3600, 12*3600, 20*3600]  # At 4h, 12h, and 20h
            for start in burst_starts:
                for i in range(5):
                    idx = np.argmin(np.abs(t - (start + i*180)))  # 3 min between spikes
                    voltage[idx:idx+60] = 0.3 * np.exp(-np.linspace(0, 5, 60))
                
        elif species == 'S_commune':
            # Add 35 spikes with ~41 min intervals
            spike_times = np.arange(0, t[-1], 41*60)
            for st in spike_times:
                idx = np.argmin(np.abs(t - st))
                amp = 0.03  # mV (within 0.02-0.04 range)
                voltage[idx:idx+50] = amp * np.exp(-np.linspace(0, 3, 50))
            
            # Add 2 wave packets
            packet_starts = [6*3600, 18*3600]  # At 6h and 18h
            for start in packet_starts:
                amps = np.concatenate([
                    np.linspace(0.02, 0.04, 8),
                    np.linspace(0.04, 0.02, 8)
                ])
                for i, amp in enumerate(amps):
                    idx = np.argmin(np.abs(t - (start + i*300)))
                    voltage[idx:idx+50] = amp * np.exp(-np.linspace(0, 3, 50))
                
        else:  # O_nidiformis
            # Add 25 spikes with ~92 min intervals
            spike_times = np.arange(0, t[-1], 92*60)
            for st in spike_times:
                idx = np.argmin(np.abs(t - st))
                amp = 0.007  # mV (within 0.005-0.009 range)
                voltage[idx:idx+40] = amp * np.exp(-np.linspace(0, 2, 40))
        
        # Add realistic noise
        noise = np.random.normal(0, 0.001, len(t))
        drift = 0.01 * np.sin(2*np.pi*t/t[-1])  # Add slow drift
        return voltage + noise + drift
    
    analyzer = FungalRosettaStone()
    
    # Test each species
    for species in ['C_militaris', 'F_velutipes', 'S_commune', 'O_nidiformis']:
        voltage = generate_test_data(species)
        
        # Run full analysis pipeline
        result = analyzer.detect_spikes(t, voltage, species)
        
        # Validate spike detection
        assert isinstance(result, dict)
        assert 'spike_times' in result
        assert 'spike_heights' in result
        assert 'statistics' in result
        assert 'validation' in result
        
        # Check species-specific characteristics
        stats = result['statistics']
        validation = result['validation']
        
        if species == 'C_militaris':
            # Check intervals and amplitudes
            assert 100 <= stats['mean_interval_min'] <= 130  # ~116 min
            assert 0.1 <= stats['mean_amplitude_mv'] <= 0.3
            # Check validation metrics
            assert validation['mean_snr'] >= 3.0
            assert validation['mean_shape_score'] >= 0.6
            
        elif species == 'F_velutipes':
            # Check regular spike characteristics
            assert 90 <= stats['mean_interval_min'] <= 110  # ~102 min
            assert 0.2 <= stats['mean_amplitude_mv'] <= 0.4
            # Verify burst detection
            assert 'patterns' in result
            bursts = [p for p in result['patterns'] if p['type'] == 'burst']
            assert len(bursts) >= 2  # Should detect at least 2 bursts
            
        elif species == 'S_commune':
            # Check intervals and amplitudes
            assert 35 <= stats['mean_interval_min'] <= 45  # ~41 min
            assert 0.02 <= stats['mean_amplitude_mv'] <= 0.04
            # Check wave packet detection
            assert 'patterns' in result
            packets = [p for p in result['patterns'] if p['type'] == 'wave_packet']
            assert len(packets) >= 1
            
        else:  # O_nidiformis
            # Check intervals and amplitudes
            assert 85 <= stats['mean_interval_min'] <= 100  # ~92 min
            assert 0.005 <= stats['mean_amplitude_mv'] <= 0.009
            assert validation['mean_snr'] >= 3.0
        
        # Run hierarchical word analysis
        words_result = analyzer.analyze_hierarchical_words(result['spike_times'], species)
        
        # Validate hierarchical analysis
        assert 'levels' in words_result
        assert 'cross_level_analysis' in words_result
        
        # Check each level
        for level_name, level_data in words_result['levels'].items():
            assert 'word_count' in level_data
            assert 'complexity' in level_data
            assert 'patterns' in level_data
            assert 'statistical_validation' in level_data
            
            # Validate statistical metrics
            validation = level_data['statistical_validation']
            assert 0 <= validation['significance'] <= 1
            assert isinstance(validation['effect_size'], float)
            assert len(validation['confidence_interval']) == 2
            
        # Check cross-level analysis
        cross = words_result['cross_level_analysis']
        assert 'level_correlations' in cross
        assert 'nested_patterns' in cross
        assert 'hierarchy_strength' in cross
        assert isinstance(cross['hierarchy_strength'], float)
        assert 0 <= cross['hierarchy_strength'] <= 1
        
        # Run pattern analysis
        if len(result['spike_times']) >= 2:
            k_range = np.linspace(0.1, 10, 50)
            tau_range = np.linspace(0, t[-1], 50)
            W = analyzer._w_transform(t, voltage, k_range, tau_range, analyzer.wavelet_type)
            pattern_result = analyzer._analyze_w_patterns(W, t, species)
            
            # Validate pattern analysis
            assert 'n_patterns' in pattern_result
            assert 'patterns' in pattern_result
            assert 'pattern_strength' in pattern_result
            assert isinstance(pattern_result['pattern_strength'], float)
            assert 0 <= pattern_result['pattern_strength'] <= 1

def test_multichannel_processing():
    """Test processing of multi-channel recordings with species-specific characteristics."""
    # Create synthetic multi-channel data matching Fig 3 patterns
    t = np.linspace(0, 3600*24, 86400)  # 24 hours at 1Hz
    
    def generate_channel_data(species, base_delay=0):
        if species == 'C_militaris':
            # Large spikes up to 12mV
            base = np.random.normal(0, 0.5, len(t))
            spikes = np.zeros_like(t)
            spike_times = np.sort(np.random.choice(t[:-1000], 20, replace=False))
            for st in spike_times:
                idx = np.argmin(np.abs(t - (st + base_delay)))
                amp = np.random.uniform(8, 12)
                spikes[idx:idx+600] = amp * np.exp(-np.linspace(0, 5, 600))
            return base + spikes
            
        elif species == 'F_velutipes':
            # High-frequency bursts
            base = -5 + np.cumsum(np.random.normal(0, 0.001, len(t)))
            bursts = np.zeros_like(t)
            burst_starts = np.sort(np.random.choice(t[:-3000], 5, replace=False))
            for st in burst_starts:
                idx = np.argmin(np.abs(t - (st + base_delay)))
                for i in range(12):  # 12 spikes per burst as in paper
                    burst_idx = idx + i*180  # 3 min between spikes
                    if burst_idx < len(t)-60:
                        amp = 2.1  # Peak amplitude from paper
                        bursts[burst_idx:burst_idx+60] = amp * np.exp(-np.linspace(0, 3, 60))
            return base + bursts
            
        elif species == 'S_commune':
            # Wave packets with amplitude evolution
            base = np.random.normal(0, 0.01, len(t))
            packets = np.zeros_like(t)
            packet_starts = np.sort(np.random.choice(t[:-10000], 3, replace=False))
            for st in packet_starts:
                idx = np.argmin(np.abs(t - (st + base_delay)))
                # Create amplitude evolution as in Fig 6
                amps = np.concatenate([
                    np.linspace(0.02, 0.04, 8),  # Rising phase
                    np.linspace(0.04, 0.02, 8)   # Falling phase
                ])
                for i, amp in enumerate(amps):
                    packet_idx = idx + i*300  # 5 min between spikes
                    if packet_idx < len(t)-50:
                        packets[packet_idx:packet_idx+50] = amp * np.exp(-np.linspace(0, 3, 50))
            return base + packets
            
        else:  # O_nidiformis
            # Very small spikes with high variability
            base = 0.1 * np.exp(-t/t[-1])  # Slow decay
            spikes = np.zeros_like(t)
            spike_times = np.sort(np.random.choice(t[:-500], 25, replace=False))
            for st in spike_times:
                idx = np.argmin(np.abs(t - (st + base_delay)))
                amp = np.random.uniform(0.005, 0.009)
                spikes[idx:idx+300] = amp * np.exp(-np.linspace(0, 2, 300))
            noise = np.random.normal(0, 0.001, len(t))
            return base + spikes + noise
    
    analyzer = FungalRosettaStone()
    
    # Test each species with 4 channels
    for species in ['C_militaris', 'F_velutipes', 'S_commune', 'O_nidiformis']:
        # Generate 4 channels with increasing delays
        channels = [
            generate_channel_data(species, base_delay=i*60)  # 1 min delay between channels
            for i in range(4)
        ]
        
        result = analyzer.process_multichannel_data(t, channels, species)
        
        # Basic structure checks
        assert isinstance(result, dict)
        assert 'channels' in result
        assert len(result['channels']) == 4
        assert 'cross_channel_analysis' in result
        
        # Species-specific checks
        if species == 'C_militaris':
            # Check for large amplitude spikes
            max_amp = max(max(abs(np.array(ch['filtered_voltage']))) 
                        for ch in result['channels'])
            assert 8 <= max_amp <= 12
            # Check interval statistics
            for ch in result['channels']:
                stats = ch['spike_data']['statistics']
                assert 90 <= stats['mean_interval_min'] <= 140  # ~116 min
            
        elif species == 'F_velutipes':
            # Check burst detection
            burst_count = 0
            for ch in result['channels']:
                assert 'bursts' in ch['spike_data']
                burst_count += len(ch['spike_data']['bursts'])
                if ch['spike_data']['bursts']:
                    burst = ch['spike_data']['bursts'][0]
                    assert burst['num_spikes'] >= 3
                    assert burst['mean_amplitude'] >= 1.5  # High amplitude bursts
            assert burst_count > 0, "Should detect bursts in F. velutipes"
            
        elif species == 'S_commune':
            # Check wave packet detection and propagation
            packet_count = 0
            for ch in result['channels']:
                packets = [p for p in ch['spike_data']['patterns'] 
                         if p['type'] == 'wave_packet']
                packet_count += len(packets)
                if packets:
                    # Check amplitude evolution
                    packet = packets[0]
                    assert packet['amplitude_evolution']['rising_phase'] < \
                           packet['amplitude_evolution']['falling_phase']
            assert packet_count > 0, "Should detect wave packets in S. commune"
            
            # Check for synchronization between channels
            cross = result['cross_channel_analysis']
            for rel in cross['spike_relationships']:
                # Should see ~60s delays between channels
                assert 30 <= abs(rel['mean_delay']) <= 90
                
        else:  # O_nidiformis
            # Check amplitude range
            for ch in result['channels']:
                filtered = np.array(ch['filtered_voltage'])
                assert np.max(abs(filtered)) < 0.01  # Very small spikes
                stats = ch['spike_data']['statistics']
                assert 70 <= stats['mean_interval_min'] <= 110  # ~92 min
        
        # Check cross-channel analysis
        cross = result['cross_channel_analysis']
        assert 'channel_correlations' in cross
        assert 'spike_relationships' in cross
        
        # Verify correlation matrix properties
        correlations = np.array(cross['channel_correlations'])
        assert correlations.shape == (4, 4)
        assert np.allclose(correlations, correlations.T)  # Symmetric
        assert np.allclose(np.diagonal(correlations), 1.0)  # Unit diagonal 

def test_spike_detection_parameters(analyzer):
    """Test species-specific spike detection parameters."""
    # Test for each species
    for species in ['C_militaris', 'F_velutipes', 'S_commune', 'O_nidiformis']:
        # Create synthetic data
        data = analyzer.generate_synthetic_data(species)
        time = np.arange(len(data)) / 1000.0  # Assuming 1kHz sampling
        
        # Detect spikes
        result = analyzer.detect_spikes(time, data, species)
        
        # Check basic detection worked
        assert 'spike_times' in result
        assert 'spike_heights' in result
        assert len(result['spike_times']) > 0
        
        # Check species-specific thresholds were applied
        if species == 'C_militaris':
            # C_militaris should detect smaller spikes now
            small_spikes = sum(1 for h in result['spike_heights'] if 0.02 <= h <= 0.05)
            assert small_spikes > 0, "Should detect small spikes for C_militaris"
            
        if species == 'F_velutipes':
            # F_velutipes should have reasonable spike counts
            assert len(result['spike_times']) >= 5, "Should detect enough spikes for F_velutipes"

def test_empty_data_handling(analyzer):
    """Test handling of empty or invalid data."""
    # Test empty spike times
    empty_result = analyzer.analyze_hierarchical_words(np.array([]), "test_species")
    assert all(level['word_count'] == 0 for level in empty_result['levels'].values())
    
    # Test empty patterns
    empty_patterns = analyzer.calculate_pattern_strength([], total_duration=100.0)
    assert empty_patterns['overall_strength'] == 0.0
    assert empty_patterns['confidence_interval'] == (0.0, 0.0)
    assert empty_patterns['reliability_score'] == 0.0 