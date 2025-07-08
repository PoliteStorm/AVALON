import numpy as np
import pytest

from fungal_communication_github.enhanced_geometric_pattern_detector import EnhancedGeometricPatternDetector
from fungal_communication_github.utils import _full_path
from pathlib import Path

DATA_COORDS = 'pleurotus_djamor_coords.npy'  # Nx3 array of voxel coords


def _skip_if_no_coords():
    return not Path(_full_path(DATA_COORDS)).exists()


@pytest.mark.skipif(_skip_if_no_coords(), reason='Geometric coordinate file not found')
def test_geometric_pattern_detection():
    coords = np.load(_full_path(DATA_COORDS))
    detector = EnhancedGeometricPatternDetector()
    results = detector.analyze_geometric_patterns(coords)

    # Expect at least one high-confidence pattern
    high_conf_patterns = [p for p in results['patterns'] if p['confidence'] > 0.8]
    assert len(high_conf_patterns) > 0, 'No high-confidence geometric pattern detected' 