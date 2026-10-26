# Environment Designer

The designer is a browser tool for laying out an environment and exporting it as a Python
package. To install the environment designer, run the following:
```bash
pip install "bluesky-sandbox[designer] @ git+https://github.com/ArsyiAziz/bluesky-sandbox.git"
python -m bluesky_sandbox.ui.designer --port 8765
```

Then open `http://localhost:8765`. For a step-by-step walk-through of the
designer, see [Your first task in the designer](tutorials/converging-fix.md).

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
  task_types.py # this design's keys as types, for your editor
  __main__.py   # python -m demo_task - a short example rollout
```

### Templates

The **Generate task** dialog has three templates. The design keeps the selected
template in `metadata["template"]`.

| Template | Adds | Use it to |
|---|---|---|
| **Plain** | nothing | Use the environment with your own trainer. |
| **RL** | `train.py`: a training-loop scaffold for the MDP of the design | Write your own algorithm. The scaffold includes a privileged critic and a cost critic when the design needs them. |
| **SB3** | `train.py`, `evaluate.py`, `masked_ppo.py`, `wrappers.py`, and `networks.py` or `ppo_lagrangian.py` when the design needs them | Train with Stable-Baselines3 PPO, one policy for all agents. |

For the RL and SB3 templates, **processes** sets how many simulators the
training script runs at the same time (`metadata["processes"]`). With the SB3
template, the dialog also shows what SB3 cannot do with the design. For
example, SB3 cannot train a continuous action together with a choice. See the
[SB3 training tutorial](tutorials/train-sb3.md).

The hooks in `env.py` and the task-info functions in `setup.py` are annotated
with the types in `task_types.py`, generated from the design: an editor that
reads annotations (VS Code, PyCharm) completes `context.obs["ownship"]["alt_ft"]`
or `context.query("goal")`, and a type checker flags a key the design does not
have. They are for editors only - nothing in them runs.
