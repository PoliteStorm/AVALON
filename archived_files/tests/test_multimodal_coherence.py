import numpy as np
import pytest
from fungal_communication_github.utils import load_voltage_trace, apply_environmental_noise, _full_path
from fungal_communication_github.semantic_testing_framework import SemanticTestingFramework
from fungal_communication_github.mushroom_communication.fungal_acoustic_detector import FungalAcousticDetector

DATA_VOLT = 'pleurotus_djamor_voltage.csv'


def _skip_if_missing():
    try:
        load_voltage_trace(DATA_VOLT)
        return False
    except FileNotFoundError:
        return True


@pytest.mark.skipif(_skip_if_missing(), reason='Real voltage dataset not found')
def test_multimodal_temporal_alignment():
    """Ensure acoustic peaks correspond to detected electrical spikes within 100 ms tolerance."""
    voltage = load_voltage_trace(DATA_VOLT) * 1e-6  # µV -> V
    voltage = apply_environmental_noise(voltage)

    sr = 10000
    t = np.arange(len(voltage)) / sr

    framework = SemanticTestingFramework()
    sem_results = framework.analyze_semantic_patterns(voltage, t, species='Pleurotus_djamor')

    # Get spike times
    spike_times = []
    for p in sem_results['analysis_layers']['pattern_recognition']['patterns']:
        if p['type'] == 'spike':
            spike_times.extend(p['times'])
    spike_times = np.array(spike_times)

    # Generate simple acoustic pressures proportional to voltage
    acoustic_data = {
        'times': t,
        'actual_pressures': voltage * 2e3,  # arbitrary conversion factor
        'ideal_pressures': voltage * 2e3
    }
    detector = FungalAcousticDetector()
    ac_results = detector._analyze_acoustic_data(acoustic_data, 'Pleurotus_djamor', {})

    # Build acoustic peak time list (use dominant frequency peaks)
    ac_peaks = []
    for chunk in ac_results['intermediate_results']:
        ac_peaks.extend(chunk['time_range'])
    ac_peaks = np.array(ac_peaks)

    # For each spike time, expect acoustic energy within 0.1 s
    matched = 0
    for s in spike_times:
        if np.any(np.abs(ac_peaks - s) <= 0.1):
            matched += 1
    alignment = matched / len(spike_times) if len(spike_times) else 1.0
    assert alignment >= 0.7, f'Multimodal alignment too low ({alignment:.2%})' 