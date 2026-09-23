"""``AircraftTable``: per-aircraft arrays matched to rows and columns by key."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bluesky_sandbox.core.aircraft_table import AircraftTable, Column


def _flat() -> AircraftTable:
    return AircraftTable(
        {
            "total_s": Column(0.0, np.float64),
            "substeps": Column(0, np.int32, per_step=True),
        }
    )


def _keyed() -> AircraftTable:
    return AircraftTable(
        {
            "route_index": Column(-1, np.int32),
            "min_nm": Column(math.inf, np.float64, per_step=True),
        },
        keyed_columns=True,
    )


def test_a_new_table_is_empty_with_its_dtypes():
    flat, keyed = _flat(), _keyed()
    assert len(flat) == 0 and flat.ids == ()
    assert flat["total_s"].shape == (0,) and flat["total_s"].dtype == np.float64
    assert flat["substeps"].dtype == np.int32
    assert keyed["route_index"].shape == (0, 0)


def test_newcomers_get_each_arrays_fill():
    table = _keyed()
    table.set_columns(["a", "b"])
    table.sync_rows(["X", "Y"])
    assert table["route_index"].tolist() == [[-1, -1], [-1, -1]]
    assert np.isinf(table["min_nm"]).all()


def test_survivors_keep_their_values_by_callsign_not_position():
    table = _flat()
    table.sync_rows(["A", "B", "C"])
    table["total_s"][:] = [1.0, 2.0, 3.0]
    # B leaves, D arrives, and the order changes.
    table.sync_rows(["C", "D", "A"])
    assert table.ids == ("C", "D", "A")
    assert table.row == {"C": 0, "D": 1, "A": 2}
    assert table["total_s"].tolist() == [3.0, 0.0, 1.0]


def test_sync_returns_where_each_row_came_from():
    table = _flat()
    table.sync_rows(["A", "B", "C"])
    take = table.sync_rows(["C", "D", "A"])
    assert take.tolist() == [2, -1, 0]
    # The same map carries a caller's own per-row data.
    partners = [{"B"}, None, {"A"}]
    carried = [partners[i] if i >= 0 else None for i in take]
    assert carried == [{"A"}, None, {"B"}]


def test_ids_object_changes_only_when_rows_do():
    table = _flat()
    table.sync_rows(["A", "B"])
    before = table.ids
    assert table.sync_rows(["A", "B"]) is None       # an equal, NEW list
    assert table.ids is before
    table.sync_rows(["A"])
    assert table.ids is not before


def test_duplicate_callsigns_are_refused():
    with pytest.raises(ValueError, match="unique"):
        _flat().sync_rows(["A", "A"])


def test_columns_keep_their_values_by_name():
    table = _keyed()
    table.set_columns(["a", "b"])
    table.sync_rows(["X", "Y"])
    table["route_index"][:] = [[1, 2], [3, 4]]
    table.set_columns(["c", "b", "a"])
    assert table.col == {"c": 0, "b": 1, "a": 2}
    assert table["route_index"].tolist() == [[-1, 2, 1], [-1, 4, 3]]


def test_rows_and_columns_remap_together():
    table = _keyed()
    table.set_columns(["a", "b"])
    table.sync_rows(["X", "Y"])
    table["route_index"][:] = [[1, 2], [3, 4]]
    table.sync_rows(["Y", "Z"])
    table.set_columns(["b"])
    assert table["route_index"].tolist() == [[4], [-1]]


def test_a_flat_table_has_no_columns_to_set():
    with pytest.raises(TypeError, match="no keyed columns"):
        _flat().set_columns(["a"])


def test_reset_step_refills_only_per_step_arrays():
    table = _flat()
    table.sync_rows(["A", "B"])
    table["total_s"][:] = [1.0, 2.0]
    table["substeps"][:] = [5, 6]
    table.reset_step()
    assert table["total_s"].tolist() == [1.0, 2.0]
    assert table["substeps"].tolist() == [0, 0]


def test_reset_step_refills_in_place():
    table = _flat()
    table.sync_rows(["A"])
    substeps = table["substeps"]
    table.reset_step()
    assert table["substeps"] is substeps


def test_clear_drops_rows_but_keeps_columns():
    table = _keyed()
    table.set_columns(["a", "b"])
    table.sync_rows(["X", "Y"])
    table.clear()
    assert table.ids == () and table.row == {}
    assert table.columns == ("a", "b")
    assert table["route_index"].shape == (0, 2)
    # A later aircraft with a returning callsign starts from the fill.
    table.sync_rows(["X"])
    assert table["route_index"].tolist() == [[-1, -1]]


def test_augmented_assignment_updates_in_place():
    table = _flat()
    table.sync_rows(["A", "B"])
    total = table["total_s"]
    table["total_s"] += np.array([1.5, 2.5])
    assert table["total_s"] is total
    assert table["total_s"].tolist() == [1.5, 2.5]


def test_an_array_cannot_be_replaced():
    table = _flat()
    table.sync_rows(["A", "B"])
    with pytest.raises(TypeError, match="in place"):
        table["total_s"] = np.zeros(5)
