# Installation

Bluesky Sandbox requires **Python 3.11 or newer**.

## Base install

The package is not yet on PyPI. For now, please install it from git:

```bash
pip install "bluesky-sandbox @ git+https://github.com/ArsyiAziz/bluesky-sandbox.git"
```

## Optional extras

Rendering backends and the designer are optional.

| Extra | Enables |
|---|---|
| `pygame` | Pygame visualization backend (`render_mode="pygame"`) |
| `qtgl` | BlueSky's native QtGL radar window |
| `panda3d` | 3D rendering via Panda3D (`render_mode="panda3d"`) |
| `designer` | Web-based [Environment Designer](designer.md) |
| `recording` | Video recording in the Pygame driver |
| `all` | All backends plus the designer |

```bash
# Example: the web designer plus the Pygame backend
pip install "bluesky-sandbox[designer,pygame] @ git+https://github.com/ArsyiAziz/bluesky-sandbox.git"
```

## Development setup

```bash
git clone https://github.com/ArsyiAziz/bluesky-sandbox.git
cd bluesky_sandbox

pip install -e ".[dev]"
pytest
```

### Build environment flags

| Variable | Effect |
|---|---|
| `BLUESKY_SANDBOX_SKIP_NPM=1` | Skip npm builds; require pre-built `dist` assets. |
| `BLUESKY_SANDBOX_FORCE_NPM=1` | Require npm to build assets; fail explicitly if missing. |

## Building these docs

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```


