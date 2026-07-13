"""Continuous native-growth snapshot contract."""

import numpy as np

from dart.coupling.growth.grow import grow_plant


XML = "dart/coupling/data/wheat_calibrated.xml"


def _nodes(plant):
    return np.asarray([[node.x, node.y, node.z] for node in plant.getNodes()], float)


def test_fractional_snapshots_are_materialized_from_one_plant():
    snapshots = {}

    def capture(plant, time):
        snapshots[time] = _nodes(plant).copy()

    plant = grow_plant(XML, simulation_time=2.0, seed=42, daily_met={},
                       snapshot_times=[0.5, 1.0, 1.25, 2.0], snapshot_callback=capture)

    assert list(snapshots) == [0.5, 1.0, 1.25, 2.0]
    assert np.array_equal(snapshots[2.0], _nodes(plant))
    assert snapshots[0.5] is not snapshots[2.0]
