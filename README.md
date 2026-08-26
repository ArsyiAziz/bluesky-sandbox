# Bluesky Sandbox

Highly Modular Air Traffic Control environment framework for Machine Learning.

Bluesky Sandbox is a PettingZoo framework powered by the open-source BlueSky ATM simulator. It transforms complex Air Traffic Control challenges, from sector conflict resolution to trajectory optimization, into customizable machine learning environments with minimal setup.

> ⚠️ Note: Bluesky Sandbox is currently in active development and not yet on PyPI. Install directly via Git (see [Installation](#installation)).


<p align="center">
  <img src="docs/media/point-merge.gif" alt="Twenty aircraft sequencing onto a point merge, with live ETA and distance-to-merge tables" width="620">
  <br>
  <em>A point-merge task: twenty arrivals sequenced onto a single merge point, ETA and distance-to-merge tracked per callsign.</em>
</p>

## Features

* **Native Multi-Agent RL:** Built from the ground up on the PettingZoo ParallelEnv standard.
* **Low Boilerplate:**  Graphically plot airspaces, routes, and spawn sectors right in your browser.
* **Low-Code UI:** Export full, clean Python task packages with pre-configured rewards, actions, and observations.
* **Multi-Backend Rendering:** Choose between lightweight Pygame, immersive 3D Panda3D, or BlueSky’s native QtGL radar interface.

## Quickstart

Instead of selecting from static prebuilt environments, Bluesky Sandbox lets you design custom tasks tailored to your exact research goals. The first step is
always to build one.

**1. Design an environment.** Start the designer and open `http://localhost:8765`:

```bash
pip install "bluesky-sandbox[designer,pygame] @ git+https://github.com/ArsyiAziz/bluesky-sandbox.git"
python -m bluesky_sandbox.ui.designer --port 8765
```

Lay out the airspace, spawn regions, waypoints and routes on the map, pick the
observation and action fields, then press **Generate task**. You get a Python
package:

```
demo_task/
  design.json   # the design itself 
  scenario.py   # geometry / spawn / queryables
  config.py     # observation + action fields, simulator settings
  setup.py      # module-level helpers the hooks lean on
  env.py        # reward, termination, and other task hooks
  __main__.py   # python -m demo_task — a short smoke rollout
```

**2.  Plug in Your RL Algorithm.** The generated package exposes a standard PettingZoo Env interface:


```python
from demo_task import Env

env = Env(render_mode=None)          # "pygame" | "panda3d" | "qtgl" | None
obs, info = env.reset(seed=0)

# PettingZoo parallel API: one action per agent, dicts back keyed by agent.
for _ in range(1000):
    if env.episode_done:
        obs, info = env.reset()
        continue
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

## Rendering

Switch renderers with a single argument (render_mode):


| Mode | Visual | Best For |
| :--- | :--- | :--- |
| `pygame` | ![Pygame](docs/media/screenshots/pygame.png) | High-speed 2D prototyping & quick debugging |
| `panda3d` | ![Panda3D](docs/media/screenshots/panda3d.png) | 3D altitude visualization & spatial analysis |
| `qtgl` | ![QtGL](docs/media/screenshots/qtgl.png) | High-fidelity BlueSky native radar display |




## Installation

Install the base package directly from source:

```bash
pip install "bluesky-sandbox @ git+https://github.com/ArsyiAziz/bluesky-sandbox.git"
```

### Optional Extras

Tailor your installation based on your visualization and design needs:


```bash
# Example: Install with web designer and Pygame support
pip install "bluesky-sandbox[designer,pygame] @ git+https://github.com/ArsyiAziz/bluesky-sandbox.git"
```

| Extra | Enables |
|---|---|
| `pygame` | Pygame visualization backend (`render_mode="pygame"`) |
| `qtgl` | BlueSky's native QtGL radar window |
| `panda3d` | 3D rendering via Panda3D (`render_mode="panda3d"`) |
| `designer` | Web-based Environment Designer |
| `all` | Full suite (all backends + designer tools) |

## Environment Designer

The package includes a built-in web tool for designing and inspecting environments without manual coding:

```bash
python -m bluesky_sandbox.ui.designer --port 8765 --reload
```

Open `http://localhost:8765` in your browser to access the designer.

<p align="center">
  <img src="docs/media/screenshots/designer.png" alt="The Environment Designer with a CAMRN FOUR arrival into JFK: named waypoints, spawn regions and routes on the map, with the geometry outline alongside" width="620">
  <br>
  <em>Designing an arrival into JFK.</em>
</p>

* **Frontend Development:** To work on the UI directly, run the Vite dev server inside `src/bluesky_sandbox/ui/designer/web`.
* **Build Behavior:** The compiled frontend is not committed. A source install builds it with npm (so Node is required).

### Build Environment Flags

| Variable | Effect |
|---|---|
| `BLUESKY_SANDBOX_SKIP_NPM=1` | Skip npm builds; require pre-built `dist` assets. |
| `BLUESKY_SANDBOX_FORCE_NPM=1` | Require npm to build assets; fail explicitly if missing. |

## Verification & Diagnostics

To verify your performance data files (e.g., OpenAP or BADA datasets):

```bash
python -m bluesky_sandbox.doctor
# or
bluesky-sandbox-doctor
```

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ArsyiAziz/bluesky-sandbox.git
cd bluesky_sandbox

# Install editable package with dev dependencies (Node required for UI build)
pip install -e ".[dev]"

# Run test suite
pytest
```

## Technical Notes

### BlueSky Fork

The `bluesky-simulator` dependency is currently pinned to a custom fork of BlueSky containing necessary bug fixes. It will be updated to the standard PyPI release once these fixes are merged into upstream BlueSky's main branch.


## License

Distributed under the MIT License. See [LICENSE](LICENSE).


## Related Projects

Bluesky Sandbox is a sister project to
[BlueSky-Gym](https://github.com/TUDelft-CNS-ATM/bluesky-gym). Please consider visiting their project!

| | BlueSky-Gym | Bluesky Sandbox |
|---|---|---|
| Interface | Gymnasium, single-agent | PettingZoo `ParallelEnv`, multi-agent |
| Environments | A curated, prebuilt set — `gym.make('MergeEnv-v0')` | You design your own |
| Best for | Benchmarking against a standard task set | Building a task that does not exist yet |

