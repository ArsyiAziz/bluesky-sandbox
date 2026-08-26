"""Sphinx configuration for the Bluesky Sandbox documentation.

The source directory is ``docs/`` itself rather than ``docs/source/`` so that
``docs/media/`` - already referenced by the README with those exact relative
paths - is inside the Sphinx source tree and usable from these pages as-is.

Read the Docs does not install the package itself. ``pip install .`` runs
``setup.py``, which needs either Node or a committed ``web/dist`` to build the
designer frontend and hard-fails with neither - and a docs builder has no
business compiling TypeScript. So ``src/`` goes on ``sys.path`` directly and
the import-time dependencies are installed from ``docs/requirements.txt``.

Only the render backends are mocked; see ``autodoc_mock_imports`` below for why
BlueSky itself cannot be.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# -- Project information -----------------------------------------------------

project = "Bluesky Sandbox"
author = "Arsyi Aziz"
copyright = "%Y, Arsyi Aziz"  # noqa: A001 - Sphinx expands %Y itself

# Read the version without importing the package: at conf time the mocks below
# are not installed yet, so an import here would pull in the real BlueSky.
_init = (SRC / "bluesky_sandbox" / "__init__.py").read_text(encoding="utf-8")
_match = re.search(r'^__version__ = "([^"]+)"', _init, re.MULTILINE)
if _match is None:
    raise RuntimeError("could not find __version__ in bluesky_sandbox/__init__.py")
release = _match.group(1)
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "attrs_inline",
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3

# -- autodoc -----------------------------------------------------------------

# Only the GUI and render backends are mocked. BlueSky itself deliberately is
# NOT: `speeds.py` evaluates `1.0 / kts` against `bluesky.tools.aero` at import
# time, and a mock has no `__rtruediv__`, so mocking it fails the import of
# every module downstream - which is nearly all of them. It is installed for
# real from docs/requirements.txt instead.
#
# Anything a class *subclasses* must likewise stay real, which is why gymnasium
# and pettingzoo are installed rather than listed here.
autodoc_mock_imports = [
    "direct",  # packaged with panda3d
    "fastapi",
    "imageio",
    "imageio_ffmpeg",
    "OpenGL",
    "panda3d",
    "pygame",
    "PyQt6",
    "starlette",
    "uvicorn",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_preserve_defaults = True
autodoc_class_signature = "separated"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- intersphinx -------------------------------------------------------------

# The point of Sphinx here: `:class:`gymnasium.spaces.Box`` in a docstring
# resolves to the upstream docs instead of rendering as dead code font.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "pettingzoo": ("https://pettingzoo.farama.org/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"Bluesky Sandbox {release}"
html_static_path: list[str] = []
html_theme_options = {
    "source_repository": "https://github.com/ArsyiAziz/bluesky-sandbox/",
    "source_branch": "main",
    "source_directory": "docs/",
}
