"""Every name the package exports resolves."""

from __future__ import annotations

import importlib

import pytest

import bluesky_sandbox
from bluesky_sandbox.interface.fields import queryables

_PACKAGES = ("bluesky_sandbox", "bluesky_sandbox.interface.fields")


@pytest.mark.parametrize("package", _PACKAGES)
def test_every_exported_name_resolves(package):
    # ``qobs`` is loaded lazily and pointed at a module that did not exist, so
    # nothing failed until someone reached for it.
    module = importlib.import_module(package)
    missing = []
    for name in module.__all__:
        try:
            getattr(module, name)
        except (AttributeError, ImportError):
            missing.append(name)
    assert missing == []


def test_qobs_is_the_queryable_fields_module():
    assert bluesky_sandbox.qobs is queryables
