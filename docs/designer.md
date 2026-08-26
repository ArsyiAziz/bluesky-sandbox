# Environment Designer

The designer is a browser tool for laying out an environment and exporting it as a Python
package. To install the environment designer, run the following:
```bash
pip install "bluesky-sandbox[designer] @ git+https://github.com/ArsyiAziz/bluesky-sandbox.git"
python -m bluesky_sandbox.ui.designer --port 8765
```

Then open `http://localhost:8765`.

```{image} media/screenshots/designer.png
:alt: Designing an arrival into JFK.
:width: 620px
:align: center
```

## Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Interface to bind. |
| `--port` | `8765` | Port to serve on. |
| `--reload` | off | Auto-reload on source change, for development. |

The `bluesky-sandbox-designer` console script is equivalent to
`python -m bluesky_sandbox.ui.designer`.

## What the designer generates:

**Generate task** writes a self-contained package:

```text
demo_task/
  design.json   # the design itself
  scenario.py   # geometry / spawn / queryables
  config.py     # observation + action fields, simulator settings
  setup.py      # module-level helpers the hooks lean on
  env.py        # reward, termination, and other task hooks
  __main__.py   # python -m demo_task - a short example rollout
```
