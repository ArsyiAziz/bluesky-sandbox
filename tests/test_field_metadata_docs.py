"""A field's docstring Metadata section is rendered from its ``meta``.

Hand-written copies drifted (a wrong unit, missing ``suppresses_when_on``);
rendering at class creation makes that impossible, and these check the source
does not grow a hand-written copy again.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass

import pytest

from bluesky_sandbox.interface.fields import actions, observations
from bluesky_sandbox.interface.fields.base import (
    ActionMeta,
    ObsField,
    ObsMeta,
    ObsQuantity,
    Unit,
    _render_metadata,
)

_MODULES = (observations, actions)
_DOCUMENTED = [
    cls
    for module in _MODULES
    for name, cls in vars(module).items()
    if inspect.isclass(cls)
    and cls.__module__ == module.__name__
    and isinstance(getattr(cls, "meta", None), (ObsMeta, ActionMeta))
    and cls.__dict__.get("__doc__")
]


def test_every_static_meta_field_is_covered():
    assert len(_DOCUMENTED) > 80


@pytest.mark.parametrize("cls", _DOCUMENTED, ids=lambda c: c.__name__)
def test_the_docstring_ends_with_the_rendered_meta(cls):
    doc = inspect.getdoc(cls)
    assert doc.endswith(_render_metadata(cls.meta))
    assert doc.count("Metadata:") == 1


@pytest.mark.parametrize("module", _MODULES, ids=lambda m: m.__name__)
def test_the_source_has_no_hand_written_metadata(module):
    tree = ast.parse(inspect.getsource(module))
    written = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and "Metadata:" in (ast.get_docstring(node) or "")
    ]
    assert written == []


def test_the_rendered_block_lists_only_what_differs_from_the_default():
    meta = ActionMeta(
        "sw",
        Unit.SWITCH,
        requires_on=("a", "b"),
        dynamic_bounds=True,
    )
    assert _render_metadata(meta) == (
        "Metadata:\n"
        "    name: sw\n"
        "    unit: switch\n"
        "    requires_on: a, b\n"
        "    dynamic_bounds: True"
    )


def test_a_hand_written_block_is_replaced_and_the_prose_kept():
    @dataclass(frozen=True)
    class Custom(ObsField):
        """A custom field.

        Metadata:
            name: stale
            unit: ft

        Some prose after the block.
        """

        meta = ObsMeta("custom", Unit.M, ObsQuantity.ALTITUDE)

        def get(self, idx):
            return 0.0

        def bounds(self, idx):
            return 0.0, 1.0

    assert inspect.getdoc(Custom) == (
        "A custom field.\n\n"
        "Some prose after the block.\n\n"
        "Metadata:\n"
        "    name: custom\n"
        "    unit: m\n"
        "    quantity: altitude"
    )
