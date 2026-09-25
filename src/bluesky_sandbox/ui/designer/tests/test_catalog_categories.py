"""The field picker's categories come from the package layout.

Each built-in field is filed under the module that defines it, described by
that module's docstring - nothing lists the categories, so a new module is a
new category and a moved field changes category with it.
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
def test_every_public_module_is_a_category(entries, package):
    modules = {
        info.name
        for info in pkgutil.iter_modules(package.__path__)
        if not info.name.startswith("_")
    }
    assert {entry["category"] for entry in entries()} == modules


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
