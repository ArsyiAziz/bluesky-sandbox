"""Layout tree for pygame view composition.

The driver lays out its window through a binary-tree of splits.  Each
node is either a :class:`Leaf` (one :class:`PygameView`) or a
:class:`Split` (orientation + a list of child layouts with fractions).
The tree is computed from the user's ``views=`` spec - a recursive
tuple where each tuple becomes a split, alternating orientation:

* the **outer** tuple defaults to vertical (top-to-bottom stack),
* nested tuples flip to horizontal (left-to-right), then back to
  vertical, and so on.

This matches the natural reading: ``((A, B), C)`` reads as "A and B
stacked on the left, C on the right" because the *outer* is vertical
by default - wait, no, with the alternation rule the outer is vertical
so ``((A, B), C)`` is "A+B vertical group on the *top*, C on the
bottom".  To get "A+B on the left, C on the right", flip the spec:
``((A, B), C)`` with horizontal-first or use an explicit
:class:`Split` directly (see :func:`parse`).

Drag-rearrange mutates the tree in place; layout recomputation is
cheap so we re-run it every frame after a mutation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

import pygame

if TYPE_CHECKING:
    from bluesky_sandbox.ui.drivers.pygame.views.base import PygameView


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------

Orientation = Literal["h", "v"]   # "h" = side-by-side, "v" = stacked


@dataclass
class Leaf:
    """A layout leaf - wraps one :class:`PygameView`."""
    view: PygameView
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))


@dataclass
class Split:
    """A layout split - multiple children divided along ``orientation``."""
    orientation: Orientation
    children: list[Node]
    fractions: list[float]
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))


Node = Union[Leaf, Split]


# ----------------------------------------------------------------------
# Spec helpers - let users tag tuples with explicit orientation
# ----------------------------------------------------------------------


class _SpecTag:
    """Marker class so users can write ``HSplit(A, B)`` / ``VSplit(A, B)``."""
    __slots__ = ("children", "orientation")

    def __init__(self, orientation: Orientation, *children) -> None:
        self.orientation = orientation
        self.children = children


def HSplit(*children) -> _SpecTag:
    """Mark a horizontal (side-by-side) split in a ``views=`` spec."""
    return _SpecTag("h", *children)


def VSplit(*children) -> _SpecTag:
    """Mark a vertical (top-to-bottom) split in a ``views=`` spec."""
    return _SpecTag("v", *children)


# ----------------------------------------------------------------------
# Parsing: spec -> Node tree
# ----------------------------------------------------------------------


def parse(spec, default_orientation: Orientation = "v") -> Node:
    """Parse a ``views=`` spec into a layout tree.

    Accepted forms:

    * A :class:`PygameView` *class* - instantiated as a leaf.
    * A :class:`PygameView` *instance* - wrapped as a leaf.
    * A ``HSplit(...)`` / ``VSplit(...)`` tag with explicit orientation.
    * A plain tuple/list - treated as a split in the *current* default
      orientation.  Nested tuples alternate orientation automatically
      so a flat outer tuple becomes a vertical stack, with horizontal
      sub-splits one level in, etc.

    Parameters
    ----------
    spec:
        The ``views`` argument from the env / driver.
    default_orientation:
        The orientation to assume for an untagged outer tuple.  Defaults
        to ``"v"`` (vertical stack - preserves the legacy default).
    """
    from bluesky_sandbox.ui.drivers.pygame.views.base import PygameView

    # Already a Node? Fine.
    if isinstance(spec, (Leaf, Split)):
        return spec

    # Bare class - instantiate.
    if isinstance(spec, type) and issubclass(spec, PygameView):
        return Leaf(view=spec())

    # Bare instance.
    if isinstance(spec, PygameView):
        return Leaf(view=spec)

    # Tagged split.
    if isinstance(spec, _SpecTag):
        children = [parse(c, _flip(spec.orientation)) for c in spec.children]
        return _make_split(spec.orientation, children)

    # Plain tuple/list.
    if isinstance(spec, (tuple, list)):
        children = [parse(c, _flip(default_orientation)) for c in spec]
        return _make_split(default_orientation, children)

    raise TypeError(
        f"Unsupported view spec: {spec!r} "
        f"(expected PygameView class/instance, tuple, or HSplit/VSplit tag)."
    )


def _flip(o: Orientation) -> Orientation:
    return "h" if o == "v" else "v"


def _make_split(orientation: Orientation, children: list[Node]) -> Node:
    """Build a :class:`Split` from children; collapse trivial 1-child wrappers."""
    if len(children) == 0:
        raise ValueError("Empty split.")
    if len(children) == 1:
        return children[0]
    # Equal initial fractions, normalised by view defaults if all are leaves.
    fracs = []
    for c in children:
        if isinstance(c, Leaf):
            fracs.append(c.view.default_height_fraction)
        else:
            fracs.append(1.0)
    s = sum(fracs)
    fractions = [f / s for f in fracs] if s > 0 else [1.0 / len(children)] * len(children)
    return Split(orientation=orientation, children=children, fractions=fractions)


# ----------------------------------------------------------------------
# Layout: assign rects to every node
# ----------------------------------------------------------------------


def compute(node: Node, rect: pygame.Rect) -> None:
    """Recursively assign rects to every node, given the root rect."""
    node.rect = pygame.Rect(rect)
    if isinstance(node, Leaf):
        node.view.rect = pygame.Rect(rect)
        return
    n = len(node.children)
    if node.orientation == "v":
        y = rect.top
        for i, (child, frac) in enumerate(zip(node.children, node.fractions)):
            h = int(round(rect.height * frac))
            if i == n - 1:
                h = rect.bottom - y
            compute(child, pygame.Rect(rect.left, y, rect.width, h))
            y += h
    else:  # "h"
        x = rect.left
        for i, (child, frac) in enumerate(zip(node.children, node.fractions)):
            w = int(round(rect.width * frac))
            if i == n - 1:
                w = rect.right - x
            compute(child, pygame.Rect(x, rect.top, w, rect.height))
            x += w


# ----------------------------------------------------------------------
# Tree iteration / search
# ----------------------------------------------------------------------


def iter_leaves(node: Node) -> Iterable[Leaf]:
    if isinstance(node, Leaf):
        yield node
    else:
        for c in node.children:
            yield from iter_leaves(c)


def iter_splits(node: Node) -> Iterable[Split]:
    if isinstance(node, Split):
        yield node
        for c in node.children:
            yield from iter_splits(c)


def find_leaf_at(node: Node, pos: tuple[int, int]) -> Leaf | None:
    if not node.rect.collidepoint(pos):
        return None
    if isinstance(node, Leaf):
        return node
    for c in node.children:
        hit = find_leaf_at(c, pos)
        if hit is not None:
            return hit
    return None


def find_parent_split(root: Node, target: Node) -> tuple[Split | None, int]:
    """Locate *target*'s parent Split + its index inside it (or (None, -1) for root)."""
    if isinstance(root, Leaf):
        return None, -1
    for i, c in enumerate(root.children):
        if c is target:
            return root, i
        if isinstance(c, Split):
            parent, idx = find_parent_split(c, target)
            if parent is not None:
                return parent, idx
    return None, -1


# ----------------------------------------------------------------------
# Tree mutation: remove / insert with auto-unwrap
# ----------------------------------------------------------------------


def remove_leaf(root: Node, leaf: Leaf) -> Node:
    """Remove *leaf* from the tree.  Returns the (possibly new) root.

    Cleans up: a Split that loses a child to leave only one child gets
    *unwrapped* into that single child.  A root Split that loses every
    child becomes... unhandled - callers must keep at least one leaf.
    """
    if root is leaf:
        raise ValueError("Cannot remove the only leaf in the tree.")
    parent, idx = find_parent_split(root, leaf)
    if parent is None:
        raise ValueError(f"{leaf} not in tree.")
    parent.children.pop(idx)
    parent.fractions.pop(idx)
    _renormalise(parent)
    return _unwrap(root)


def insert_leaf(
    root: Node, target_leaf: Leaf, new_leaf: Leaf, zone: str,
) -> Node:
    """Insert *new_leaf* relative to *target_leaf*.

    *zone* in ``{"top", "bottom", "left", "right", "center"}`` - for
    ``"center"`` the views swap; for the others *target_leaf* is paired
    with *new_leaf* in a new (or existing) split with the appropriate
    orientation.
    """
    if zone == "center":
        # Swap views in place.  Mutates target_leaf and new_leaf must be
        # a fresh leaf (caller's responsibility).
        target_leaf.view, new_leaf.view = new_leaf.view, target_leaf.view
        return root

    orient: Orientation = "v" if zone in ("top", "bottom") else "h"
    insert_before = zone in ("top", "left")

    parent, idx = find_parent_split(root, target_leaf)
    if parent is not None and parent.orientation == orient:
        # Insert into existing parent split - same orientation, just
        # nudge the child list and fractions.
        new_pos = idx if insert_before else idx + 1
        parent.children.insert(new_pos, new_leaf)
        # Steal half of target_leaf's fraction for the new sibling.
        share = parent.fractions[idx + (0 if insert_before else 0)]
        if insert_before:
            half = share / 2
            parent.fractions[idx] = half
            parent.fractions.insert(idx, half)
        else:
            half = share / 2
            parent.fractions[idx] = half
            parent.fractions.insert(idx + 1, half)
        _renormalise(parent)
        return root

    # Wrap target in a new split.  Two children: target_leaf and new_leaf
    # in the requested order.  Each gets fraction 0.5.
    new_children = [new_leaf, target_leaf] if insert_before else [target_leaf, new_leaf]
    new_split = Split(
        orientation=orient,
        children=new_children,
        fractions=[0.5, 0.5],
    )
    if parent is None:
        # Target was the root -> the new split becomes the new root.
        return new_split
    parent.children[idx] = new_split
    return root


def _renormalise(split: Split) -> None:
    """Re-scale a split's fractions to sum to 1 (after a child was added/removed)."""
    s = sum(split.fractions)
    if s <= 0:
        n = max(len(split.fractions), 1)
        split.fractions = [1.0 / n] * len(split.fractions)
    else:
        split.fractions = [f / s for f in split.fractions]


def _unwrap(node: Node) -> Node:
    """Recursively collapse 1-child Splits and merge same-orientation chains."""
    if isinstance(node, Leaf):
        return node
    # Recursively unwrap children first.
    new_children: list[Node] = []
    new_fractions: list[float] = []
    for c, f in zip(node.children, node.fractions):
        unwrapped = _unwrap(c)
        # Merge same-orientation child split into parent.
        if isinstance(unwrapped, Split) and unwrapped.orientation == node.orientation:
            for cc, ff in zip(unwrapped.children, unwrapped.fractions):
                new_children.append(cc)
                new_fractions.append(f * ff)
        else:
            new_children.append(unwrapped)
            new_fractions.append(f)
    if len(new_children) == 1:
        return new_children[0]
    node.children = new_children
    node.fractions = new_fractions
    _renormalise(node)
    return node


# ----------------------------------------------------------------------
# Drop-zone helpers (used by the driver during drag-rearrange)
# ----------------------------------------------------------------------


def drop_zone(rect: pygame.Rect, pos: tuple[int, int], edge_frac: float = 0.25
              ) -> str | None:
    """Classify a cursor position inside *rect* into a drop zone.

    Zones: ``"top"``, ``"bottom"``, ``"left"``, ``"right"``, ``"center"``,
    or ``None`` if *pos* is outside *rect*.
    """
    if not rect.collidepoint(pos):
        return None
    x, y = pos
    rx = (x - rect.left) / max(rect.width,  1)
    ry = (y - rect.top)  / max(rect.height, 1)
    # Closest edge by relative distance.
    distances = {
        "left":   rx,
        "right":  1 - rx,
        "top":    ry,
        "bottom": 1 - ry,
    }
    name, dist = min(distances.items(), key=lambda it: it[1])
    if dist > edge_frac:
        return "center"
    return name


def drop_preview_rect(target_rect: pygame.Rect, zone: str) -> pygame.Rect:
    """Return the preview rect to highlight when dropping into *zone* of *target_rect*."""
    r = target_rect
    if zone == "top":
        return pygame.Rect(r.left, r.top, r.width, r.height // 2)
    if zone == "bottom":
        return pygame.Rect(r.left, r.top + r.height // 2, r.width, r.height - r.height // 2)
    if zone == "left":
        return pygame.Rect(r.left, r.top, r.width // 2, r.height)
    if zone == "right":
        return pygame.Rect(r.left + r.width // 2, r.top, r.width - r.width // 2, r.height)
    return pygame.Rect(r)  # center -> whole rect (swap)
