import sys
import types

import numpy as np
import pytest

from dart.coupling.photosynthesis import iterative


class _Plant:
    def getSegmentIds(self, organ_type):
        return [0]

    def getOrgans(self):
        return []


def _fake_dart(monkeypatch, tmp_path, *, baleno_ok=True):
    dart_pkg = types.ModuleType('dart.coupling.dart')
    dart_pkg.__path__ = []
    baleno = types.ModuleType('dart.coupling.dart.baleno')
    parsers = types.ModuleType('dart.coupling.dart.parsers')

    baleno_home = tmp_path / 'baleno'
    (baleno_home / 'resources').mkdir(parents=True)
    (tmp_path / 'simulation' / 'input').mkdir(parents=True)

    baleno.BALENO_DIR = baleno_home
    baleno.EXTERNAL_GS_PLUGIN = 'external'
    baleno.run_baleno_subprocess = lambda **kwargs: baleno_ok
    baleno.read_baleno_tleaf_multi = lambda *args, **kwargs: [np.array([30.0])]
    baleno.log_baleno_diagnostics = lambda *args, **kwargs: None
    baleno.make_baleno_radiation_config = lambda *args, **kwargs: {}
    baleno.make_baleno_vegetation_config = lambda *args, **kwargs: {}
    baleno.write_baleno_closures_input = lambda *args, **kwargs: None
    baleno.read_baleno_outputs_multi = lambda *args, **kwargs: {
        'eta': [np.array([9.0])],
        'tri_data': [None],
        'tri_data_raw': [[{
            'segment_idx': 0, 'eta': 9.0, 'An_umol': 90.0,
        }]],
    }
    parsers.write_json5 = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, 'dart.coupling.dart', dart_pkg)
    monkeypatch.setitem(sys.modules, 'dart.coupling.dart.baleno', baleno)
    monkeypatch.setitem(sys.modules, 'dart.coupling.dart.parsers', parsers)
    monkeypatch.setattr(iterative, 'build_scene_row_mapping', lambda *args: {
        'plant_to_obj_to_scene': [{}], 'n_scene_rows': 0,
    })
    monkeypatch.setattr(
        iterative, 'segment_gs_to_scene_rcw_multi',
        lambda *args, **kwargs: np.empty(0),
    )
    monkeypatch.setattr(
        iterative, 'get_prospect_params_per_position',
        lambda *args, **kwargs: [{'Cab': 40.0, 'N': 1.5}],
    )
    monkeypatch.setattr(iterative, 'get_prospect_params', lambda *args: {})


def test_sif_final_state_is_resolved_at_final_baleno_temperature(monkeypatch, tmp_path):
    _fake_dart(monkeypatch, tmp_path)
    solve_tleaf = []
    solve_parameters = []

    def solve(*args, tleaf, vcm_parameters=None, **kwargs):
        temperature = float(np.mean(tleaf))
        solve_tleaf.append(temperature)
        solve_parameters.append(vcm_parameters)
        return {
            'An_per_umol': np.array([temperature]),
            'An_total_mmol': temperature,
            'gco2': np.array([temperature / 100.0]),
            'eta': np.array([temperature / 100.0]),
            'psi_leaf_cm': np.array([-500.0]),
            'psi_leaf_MPa': np.array([-0.05]),
        }

    monkeypatch.setattr(iterative, 'run_photosynthesis_solve', solve)
    results = iterative.run_iterative_coupling_multi(
        [_Plant()], sim_time=55, par_umol_per_plant=[np.array([1000.0])],
        mapping_json_paths=['map.json'], reindex_json_paths=['reindex.json'],
        baleno_sim_dir=tmp_path / 'simulation', baleno_simu_name='test',
        n_plants=1, max_iterations=1, initial_tleaf=[np.array([25.0])],
        with_sif=True, vcm_parameters={'vcm_alpha': 0.2},
    )

    assert solve_tleaf == [25.0, 30.0]
    assert solve_parameters == [{'vcm_alpha': 0.2}] * 2
    assert results[0]['tleaf_per_segment'] == pytest.approx([30.0])
    assert results[0]['an_per_segment'] == pytest.approx([30.0])
    assert results[0]['gs_per_segment'] == pytest.approx([0.3])
    assert results[0]['eta_per_segment'] == pytest.approx([0.3])
    assert results[0]['tri_data_raw'][0]['eta'] == pytest.approx(0.3)


def test_sif_baleno_failure_is_not_allowed_to_reuse_old_output(monkeypatch, tmp_path):
    _fake_dart(monkeypatch, tmp_path, baleno_ok=False)
    monkeypatch.setattr(iterative, 'run_photosynthesis_solve', lambda *args, **kwargs: {
        'An_per_umol': np.array([20.0]), 'An_total_mmol': 20.0,
        'gco2': np.array([0.2]), 'eta': np.array([0.2]),
        'psi_leaf_cm': np.array([-500.0]), 'psi_leaf_MPa': np.array([-0.05]),
    })

    with pytest.raises(RuntimeError, match='Baleno subprocess failed'):
        iterative.run_iterative_coupling_multi(
            [_Plant()], sim_time=55,
            par_umol_per_plant=[np.array([1000.0])],
            mapping_json_paths=['map.json'], reindex_json_paths=['reindex.json'],
            baleno_sim_dir=tmp_path / 'simulation', baleno_simu_name='test',
            n_plants=1, max_iterations=1, initial_tleaf=[np.array([25.0])],
            with_sif=True,
        )
