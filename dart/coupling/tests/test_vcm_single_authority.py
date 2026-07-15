import numpy as np
import pytest

from dart.coupling.photosynthesis.iterative import replace_triangle_physiology
from dart.coupling.sif.sif_writer import (
    collect_per_triangle_eta,
    compute_sunlit_shaded_summary,
    write_triangle_sif_csv,
)


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


def test_sif_outputs_consume_cplantbox_triangle_state(tmp_path):
    triangles = [
        {'tri_idx': 0, 'segment_idx': 0, 'eta': 9.0, 'An_umol': 90.0,
         'apar_Wm2': 0.5, 'area_cm2': 2.0, 'sunlit_frac': 1.0},
        {'tri_idx': 1, 'segment_idx': 1, 'eta': 9.1, 'An_umol': 91.0,
         'apar_Wm2': 0.25, 'area_cm2': 3.0, 'sunlit_frac': 0.0},
    ]
    replace_triangle_physiology(triangles, [0.2, 0.3], [20.0, 30.0])
    results = [{'tri_data_raw': triangles}]

    assert collect_per_triangle_eta(results) == pytest.approx([0.2, 0.3])
    summary = compute_sunlit_shaded_summary(results, clearsky_par_wm2=600.0)
    assert summary['mean_eta_sunlit'] == pytest.approx(0.2)
    assert summary['mean_eta_shaded'] == pytest.approx(0.3)

    output = tmp_path / 'triangles.csv'
    write_triangle_sif_csv(output, triangles, plant_idx=0, clearsky_par_wm2=600.0)
    assert [float(line.split(',')[4]) for line in output.read_text().splitlines()[1:]] == pytest.approx([0.2, 0.3])
