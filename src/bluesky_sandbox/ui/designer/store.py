"""On-disk persistence for design specs.

A flat directory of ``<name>.json`` documents, or ``<name>-v<version>.json``
when the design carries a ``metadata.version``. Names are slugged to stay
filesystem-safe and to keep the store from escaping its root.

Versions are separate documents, not revisions of one: a design at v2 is a
different task from the same design at v1 - it trains a different policy and a
checkpoint traces to one of them. Saving a bumped version therefore adds an
entry instead of overwriting the one you were working from, which is what the
flat ``<name>.json`` scheme used to do silently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .spec import DesignSpec

_DEFAULT_DIR = Path.home() / ".bluesky_sandbox" / "designs"
_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def slug(name: str) -> str:
    s = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not s:
        raise ValueError(f"name {name!r} slugs to empty.")
    return s


@dataclass
class SpecStore:
    """Read/write design specs under a root directory."""

    root: Path = _DEFAULT_DIR

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{slug(name)}.json"

    @staticmethod
    def _split_stem(stem: str) -> tuple[str, str]:
        """``("point-merge-v2")`` -> ``("point-merge", "2")``; no suffix -> ``("", )``."""
        base, sep, version = stem.rpartition("-v")
        if sep and base and version and version[0].isdigit():
            return base, version
        return stem, ""

    def list(self) -> list[dict[str, str]]:
        out = []
        for p in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                meta = data.get("metadata", {})
            except (OSError, json.JSONDecodeError):
                meta = {}
            base, version = self._split_stem(p.stem)
            out.append(
                {
                    "name": p.stem,          # the load key
                    "title": meta.get("name", p.stem),
                    "base": base,            # groups versions of one design
                    "version": str(meta.get("version", version) or version),
                }
            )
        return out

    def load(self, name: str) -> DesignSpec:
        path = self._path(name)
        if not path.exists():
            raise FileNotFoundError(f"no design named {name!r}.")
        return DesignSpec.from_json(path.read_text())

    def save(self, name: str, spec: DesignSpec) -> str:
        """Write ``spec``; returns the stem it was stored under.

        A design with a ``metadata.version`` is stored beside its other
        versions rather than on top of them.
        """
        version = str((spec.metadata or {}).get("version", "") or "").strip()
        stem = slug(name)
        if version and version != "0":
            stem = f"{stem}-v{slug(version)}"
        path = self.root / f"{stem}.json"
        path.write_text(spec.to_json())
        return path.stem

    def delete(self, name: str) -> None:
        path = self._path(name)
        if path.exists():
            path.unlink()
