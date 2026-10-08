"""Older names of renamed fields, kept working: a design's named shapes were
once its "bounds", an element's shape its ``bounds``."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=type)


def renamed(**old_to_new: str) -> Callable[[T], T]:
    """Class decorator (outside ``@dataclass``): for each ``old=new``, field
    ``new`` is still taken as keyword ``old`` and read as attribute ``old``."""

    def apply(cls: T) -> T:
        init = cls.__init__

        @functools.wraps(init)
        def __init__(self, *args, **kwargs):
            # The older name wins when both come: ``dataclasses.replace(x,
            # old=...)`` passes every field by its new name too.
            for old, new in old_to_new.items():
                if old in kwargs:
                    kwargs[new] = kwargs.pop(old)
            init(self, *args, **kwargs)

        cls.__init__ = __init__
        for old, new in old_to_new.items():
            setattr(
                cls,
                old,
                property(
                    lambda self, _new=new: getattr(self, _new),
                    # Setting it sets the new one - refused on a frozen class.
                    lambda self, value, _new=new: setattr(self, _new, value),
                    doc=f"``{new}``'s older name.",
                ),
            )
        return cls

    return apply
