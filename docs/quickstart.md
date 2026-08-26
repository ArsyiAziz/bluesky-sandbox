# Quickstart

Instead of selecting from static prebuilt environments, Bluesky Sandbox lets you design custom tasks tailored to your exact research goals. The first step is always to build one.


## 1. Design an environment

```bash
pip install "bluesky-sandbox[designer,pygame] @ git+https://github.com/ArsyiAziz/bluesky-sandbox.git"
python -m bluesky_sandbox.ui.designer --port 8765
```

Open `http://localhost:8765`. In the interface you can edit the environment configuration, including laying out the airspace, spawn regions, waypoints and routes on the map, picking the observation and action fields. When you are done press **Generate task**. You will get a Python package:

```text
demo_task/
  design.json   # the design itself
  scenario.py   # geometry / spawn / queryables
  config.py     # observation + action fields, simulator settings
  setup.py      # module-level helpers the hooks lean on
  env.py        # reward, termination, and other task hooks
  __main__.py   # python -m demo_task - a short smoke rollout
```

> Note: The generated code is built on the primitives defined in the [API reference](api/index.md).


## 2. Run a rollout

The generated package exposes a standard PettingZoo parallel interface:

```python
from demo_task import Env

env = Env(render_mode=None)          # "pygame" | "panda3d" | "qtgl" | None
obs, info = env.reset(seed=0)

# One action per agent; dicts back, keyed by agent.
for _ in range(1000):
    if env.episode_done:
        obs, info = env.reset()
        continue
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

Swap the random policy for your learner and you are training. 

## Next steps
- [Environment and configuration](api/env.md): Customize core Bluesky Simulator settings, reward hooks, and termination rules.
- [Observation and action fields](api/fields.md): Define agent observations spaces and action spaces.
