import numpy as np
import pytest

from fungal_communication_github.semantic_testing_framework import SemanticTestingFramework
from fungal_communication_github.mushroom_communication.fungal_acoustic_detector import FungalAcousticDetector
from fungal_communication_github.research_constants import (
    SPECIES_DATABASE,
    validate_simulation_against_research,
    ensure_scientific_rigor,
)


@pytest.mark.parametrize("species", list(SPECIES_DATABASE.keys()))
def test_semantic_framework_validity(species):
    """Run SemanticTestingFramework on sample data and validate against research constants."""
    # Generate synthetic data based on species voltage patterns
    params = SPECIES_DATABASE[species]
    duration = 10  # seconds
    sampling_rate = 100  # Hz  (keep small for test speed)
    t = np.linspace(0, duration, duration * sampling_rate, endpoint=False)

    # Base noise
    voltage = params.get("baseline_voltage", 1e-3) + params.get("noise_level", 1e-4) * np.random.randn(len(t))

    # Add sine patterns
    for pattern in params.get("voltage_patterns", []):
        if pattern["type"] == "sine":
            voltage += pattern["amplitude"] * np.sin(2 * np.pi * pattern["frequency"] * t)
        elif pattern["type"] == "spike":
            # Insert simple spike events
            indices = np.random.choice(len(t), size=pattern["count"], replace=False)
            voltage[indices] += pattern["amplitude"]

    framework = SemanticTestingFramework()
    results = framework.analyze_semantic_patterns(voltage, t, species=species)

    # Build validation parameters
    validation_params = {
        "species": species,
        "voltage_range": {"min": float(voltage.min()), "max": float(voltage.max())},
        "methods": [
            "pattern_recognition",
            "statistical_validation",
            "complexity_analysis",
            "semantic_interpretation",
        ],
    }
    validation = validate_simulation_against_research(validation_params)

    assert validation["valid"], f"Scientific validation failed: {validation['messages']}"

    # Ensure scientific rigor metadata can be added without error
    ensure_scientific_rigor(results)


@pytest.mark.parametrize("species", list(SPECIES_DATABASE.keys()))
def test_acoustic_detector_validity(species):
    """Run a lightweight acoustic analysis and validate against research constants."""
    # Generate synthetic acoustic data using the same principle as voltage
    params = SPECIES_DATABASE[species]
    duration = 5  # seconds
    sampling_rate = 100  # samples/sec
    t = np.linspace(0, duration, duration * sampling_rate, endpoint=False)
    pressures = params.get("baseline_voltage", 1e-3) * np.sin(2 * np.pi * 0.5 * t)  # simplistic

    acoustic_data = {
        "times": t,
        "actual_pressures": pressures,
        "ideal_pressures": pressures * 1.05,
    }

    detector = FungalAcousticDetector()
    analysis = detector._analyze_acoustic_data(acoustic_data, species, {})

    # Build validation parameters
    validation_params = {
        "species": species,
        "voltage_range": {"min": float(pressures.min()), "max": float(pressures.max())},
        "methods": ["acoustic_conversion", "piezoelectric_modeling"],
    }

    validation = validate_simulation_against_research(validation_params)
    assert validation["valid"], f"Acoustic validation failed: {validation['messages']}"

    # Ensure we can tag with scientific rigor
    ensure_scientific_rigor(analysis)

    # Check that analysis returns expected top-level keys
    expected_keys = {"frequency_analysis", "signal_quality", "statistical_analysis"}
    assert expected_keys.issubset(analysis.keys()), "Incomplete acoustic analysis output" 