"""The field picker's categories and choices come from the code.

Each built-in field is filed under the module that defines it, described by
that module's docstring - nothing lists the categories, so a new module is a
new category and a moved field changes category with it. A field whose
constructor takes another field is not offered at all: a transform on the field
it wraps builds it.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

from bluesky_sandbox.interface.fields import actions, observations
from bluesky_sandbox.ui.designer import catalog

_CATALOGS = [
    (catalog.obs_fields, observations),
    (catalog.action_fields, actions),
]


@pytest.mark.parametrize(
    ("entries", "package"), _CATALOGS, ids=["observations", "actions"]
)
def test_every_category_is_a_public_module(entries, package):
    modules = {
        info.name
        for info in pkgutil.iter_modules(package.__path__)
        if not info.name.startswith("_")
    }
    assert {entry["category"] for entry in entries()} <= modules


def test_a_field_wrapping_another_is_left_to_the_transform_that_builds_it():
    # Picked on its own it has nothing to wrap; the field it wraps builds it.
    offered = {entry["name"] for entry in catalog.obs_fields()}
    built = {
        type(observations.AltFt().relative_to_own()),
        type(observations.HdgDeg().relative_to_own()),
        type(observations.AltFt().stacked(depth=2)[1]),
        type(observations.DistToOwnNm().stacked(depth=2)[1]),
    }
    assert {cls.__name__ for cls in built} == {
        "Difference",
        "AngleDifference",
        "LaggedObs",
        "LaggedPair",
    }
    assert not offered & {cls.__name__ for cls in built}


@pytest.mark.parametrize(
    ("entries", "package"), _CATALOGS, ids=["observations", "actions"]
)
def test_a_field_is_filed_under_the_module_that_defines_it(entries, package):
    for entry in entries():
        cls = getattr(package, entry["name"])
        assert cls.__module__ == f"{package.__name__}.{entry['category']}"


@pytest.mark.parametrize(
    ("entries", "package"), _CATALOGS, ids=["observations", "actions"]
)
def test_a_category_is_described_by_its_module_docstring(entries, package):
    for entry in entries():
        module = importlib.import_module(f"{package.__name__}.{entry['category']}")
        first_paragraph = " ".join(module.__doc__.strip().split("\n\n")[0].split())
        assert entry["category_doc"] == first_paragraph != ""


def test_a_switch_offers_no_bounds_and_no_normalizer():
    # Its bounds are fixed at (0, 1) and it takes the value as given, so the
    # designer has nothing to set; a normal action has both.
    entries = {entry["name"]: entry for entry in catalog.action_fields()}
    assert entries["AutopilotLnav"]["params"] == []
    assert entries["AutopilotLnav"]["normalizable"] is False
    assert {"low", "high"} <= {p["name"] for p in entries["SpdKts"]["params"]}
    assert entries["SpdKts"]["normalizable"] is True
