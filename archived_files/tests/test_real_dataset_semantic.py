import numpy as np
import pytest

from fungal_communication_github.semantic_testing_framework import SemanticTestingFramework
from fungal_communication_github.utils import load_voltage_trace, load_spike_times, apply_environmental_noise

DATA_VOLT = 'pleurotus_djamor_voltage.csv'  # 10 kHz sampled raw voltage (µV)
DATA_SPIKES = 'pleurotus_djamor_spike_times.csv'  # ground-truth spike onset times (s)


def _skip_if_no_data():
    try:
        load_voltage_trace(DATA_VOLT)
        load_spike_times(DATA_SPIKES)
        return False
    except FileNotFoundError:
        return True


@pytest.mark.skipif(_skip_if_no_data(), reason='Real voltage dataset not found')
def test_semantic_detection_against_ground_truth():
    """Compare detected spike count and timing to ground-truth within ±50 ms."""
    voltage = load_voltage_trace(DATA_VOLT)
    gt_spikes = load_spike_times(DATA_SPIKES)

    # add realistic environmental noise
    voltage_noisy = apply_environmental_noise(voltage * 1e-6)  # convert µV→V

    sampling_rate = 10000  # Hz ; implicit in the dataset
    t = np.arange(len(voltage_noisy)) / sampling_rate

    framework = SemanticTestingFramework()
    results = framework.analyze_semantic_patterns(voltage_noisy, t,
                                                 species='Pleurotus_djamor')
    detected = []
    for p in results['analysis_layers']['pattern_recognition']['patterns']:
        if p['type'] == 'spike':
            detected.extend(p['times'])
    detected = np.array(detected)

    # For fairness: count spikes within ±0.05 s of ground truth
    matched = 0
    for s in gt_spikes:
        if np.any(np.abs(detected - s) <= 0.05):
            matched += 1
    recall = matched / len(gt_spikes)

    # Accept ≥80 % recall as sufficient
    assert recall >= 0.80, f'Recall too low ({recall:.2%})' 