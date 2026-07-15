import numpy as np
import pytest

from dart.coupling.photosynthesis.iterative import replace_triangle_physiology


def test_cplantbox_state_replaces_triangle_physiology_and_keeps_baleno_diagnostics():
    triangles = [
        {'segment_idx': 1, 'eta': 9.1, 'An_umol': 91.0},
        {'segment_idx': 0, 'eta': 9.0, 'An_umol': 90.0},
    ]

    replace_triangle_physiology(
        triangles,
        eta_per_segment=np.array([0.2, 0.3]),
        an_per_segment=np.array([20.0, 30.0]),
    )

    assert [t['eta'] for t in triangles] == pytest.approx([0.3, 0.2])
    assert [t['An_umol'] for t in triangles] == pytest.approx([30.0, 20.0])
    assert [t['eta_baleno_diag'] for t in triangles] == pytest.approx([9.1, 9.0])
    assert [t['An_baleno_diag'] for t in triangles] == pytest.approx([91.0, 90.0])


def test_triangle_mapping_rejects_unknown_segment():
    with pytest.raises(ValueError, match='segment_idx'):
        replace_triangle_physiology(
            [{'segment_idx': 2, 'eta': 9.0, 'An_umol': 90.0}],
            eta_per_segment=np.array([0.2]),
            an_per_segment=np.array([20.0]),
        )
