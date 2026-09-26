"""Evaluating a trained policy in a generated package: ``evaluate.py``, which
flies it episode by episode - in a window, when the design asks for one.

A design picks the window with ``metadata["eval_render_mode"]``: one of the
window render modes (:data:`~bluesky_sandbox.ui.drivers.WINDOW_RENDER_MODES`),
or ``None`` for none. Training never draws; this is where the policy is
watched.
"""

from __future__ import annotations

import json
from typing import Any

from bluesky_sandbox.ui.drivers import FRAME_DRIVERS, WINDOW_RENDER_MODES

__all__ = [
    "eval_render_mode",
    "evaluation_catalog",
    "rl_evaluate_py",
    "sb3_evaluate_py",
]

#: The window an evaluation draws in unless the design says otherwise.
DEFAULT_EVAL_RENDER_MODE = next(iter(FRAME_DRIVERS))


def eval_render_mode(metadata: dict[str, Any]) -> str | None:
    """The window a design's evaluation draws in, or ``None`` for none."""
    if "eval_render_mode" not in metadata:
        return DEFAULT_EVAL_RENDER_MODE
    mode = metadata["eval_render_mode"]
    if mode is not None and mode not in WINDOW_RENDER_MODES:
        raise ValueError(
            f"eval_render_mode must be one of {list(WINDOW_RENDER_MODES)} or None, got {mode!r}"
        )
    return mode


def evaluation_catalog() -> dict[str, Any]:
    """What the designer offers for evaluation: the windows it can draw in."""
    return {
        "render_modes": list(WINDOW_RENDER_MODES),
        "default": DEFAULT_EVAL_RENDER_MODE,
    }


def _render_mode_block(render_mode: str | None) -> str:
    modes = ", ".join(f'"{m}"' for m in WINDOW_RENDER_MODES)
    return f"""#: The window an evaluation is drawn in - {modes} - or None for none.
RENDER_MODE = {json.dumps(render_mode) if render_mode else "None"}
"""


_EPISODES = """    returns = []
    try:
        for episode in range(episodes):
            obs, _infos = env.reset(seed=seed + episode)
            total, steps = 0.0, 0
            while not env.unwrapped.episode_done and steps != max_steps:
                steps += 1
                actions = {{agent: {act} for agent, ob in obs.items()}}
                obs, rewards, _terminations, _truncations, _infos = env.step(actions)
                total += sum(float(np.sum(r)) for r in rewards.values())
                if render_mode is not None:
                    env.render()
            returns.append(total)
            print(f"episode {{episode}}: {{steps}} steps, total reward {{total:.3f}}")
    finally:
        env.close()
    return returns
"""


def sb3_evaluate_py(pkg: str, render_mode: str | None) -> str:
    """``evaluate.py`` for the SB3 package: the saved PPO policy, flown."""
    return f'''"""Fly the policy ``python -m {pkg}.train`` saved, episode by episode:

    python -m {pkg}.evaluate

``RENDER_MODE`` is the window it is drawn in.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from .train import make_env

{_render_mode_block(render_mode)}

def evaluate(
    model_path: str = "ppo_{pkg}",
    episodes: int = 3,
    seed: int = 0,
    render_mode: str | None = RENDER_MODE,
    max_steps: int | None = None,
) -> list[float]:
    """Fly ``episodes`` episodes, every agent acting as the policy would
    deterministically - each to its end, or ``max_steps`` steps; return each
    one's total reward."""
    model = PPO.load(model_path)
    env = make_env(render_mode=render_mode)
{_EPISODES.format(act="model.predict(ob, deterministic=True)[0]")}

if __name__ == "__main__":
    evaluate()
'''


def rl_evaluate_py(
    pkg: str,
    class_stem: str,
    render_mode: str | None,
    *,
    privileged: bool,
    batched: bool,
) -> str:
    """``evaluate.py`` for the RL scaffold: the reader's trained actor, flown.
    ``batched``: the scaffold's policy is called on a batch of observations."""
    view = "actor_obs(ob)" if privileged else "ob"
    act = f"policy(one({view}))[0]" if batched else f"policy({view})"
    view_import = "from bluesky_sandbox import actor_obs\n\n" if privileged else ""
    one = (
        '''

def one(ob):
    """One observation as a batch of one, as the policy is called on batches."""
    return {key: value[None] for key, value in ob.items()} if isinstance(ob, dict) else ob[None]
'''
        if batched
        else ""
    )
    return f'''"""Fly your trained actor, episode by episode:

    python -m {pkg}.evaluate

``load_policy`` is yours to fill in; ``RENDER_MODE`` is the window it is drawn
in.
"""

from __future__ import annotations

import numpy as np

{view_import}from .env import {class_stem}Env

{_render_mode_block(render_mode)}

def load_policy():
    """Your trained actor - as ``build_policy`` in train.py makes it."""
    raise NotImplementedError("load_policy: return your trained actor.")
{one}

def evaluate(
    episodes: int = 3,
    seed: int = 0,
    render_mode: str | None = RENDER_MODE,
    max_steps: int | None = None,
) -> list[float]:
    """Fly ``episodes`` episodes with your actor - each to its end, or
    ``max_steps`` steps; return each one's total reward."""
    policy = load_policy()
    env = {class_stem}Env(render_mode=render_mode)
{_EPISODES.format(act=act)}

if __name__ == "__main__":
    evaluate()
'''
