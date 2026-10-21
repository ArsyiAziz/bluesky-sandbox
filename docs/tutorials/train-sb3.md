# Train a speed policy with Stable-Baselines3

This tutorial trains a PPO policy on the [final approach task](final-approach.md).
The policy controls only the speed of each arrival. All aircraft share one policy.

**You will learn to:**

- generate a package with the SB3 template
- give the action space finite bounds on a [-1, 1] scale
- train in parallel processes and read the logs
- fly the trained policy
- use speed choices instead of a continuous range

## Before you start

Install the `sb3` extra (Stable-Baselines3, SuperSuit, TensorBoard):

```bash
pip install "bluesky-sandbox[sb3,designer,pygame] @ git+https://github.com/ArsyiAziz/bluesky-sandbox.git"
```

Use the finished design [`designs/final_approach.json`](designs/final_approach.json),
or build it with the [final approach tutorial](final-approach.md).

## 1. Generate the SB3 package

In the designer, select **Generate task**, then the **SB3** template. Set
**processes** to 4. Each process runs one BlueSky simulator.

From the command line, set the same values in the design `metadata`, then build:

```python
import json

design = json.load(open("final_approach.json"))
design["metadata"].update({"template": "sb3", "processes": 4})
json.dump(design, open("final_approach_sb3.json", "w"), indent=2)
```

```bash
bluesky-sandbox design build final_approach_sb3.json --out . --name final_approach
```

The SB3 template adds these files:

```text
final_approach/
  train.py        # python -m final_approach.train
  evaluate.py     # python -m final_approach.evaluate
  masked_ppo.py   # PPO that does not learn from empty agent slots
  networks.py     # a transformer over the ownship and each intruder
  wrappers.py     # fits the env to SB3, logs each agent's episode
```

These files are yours to change. If the design has a `cost` hook, the package
also contains `ppo_lagrangian.py`. See
[Integrations](../api/integrations.md#what-an-sb3-package-carries).

## 2. Set the action space

The action is `ActiveRouteWaypointSpdDeltaKts`: a speed offset in knots from
the speed restriction of the next fix. Give it `low`, `high` and a
`SymmetricNormalizer`:

```json
{
  "field": "ActiveRouteWaypointSpdDeltaKts",
  "kwargs": {
    "low": -40,
    "high": 20,
    "normalizer": {"type": "normalizer", "name": "SymmetricNormalizer", "kwargs": {}}
  }
}
```

In the designer, set the normalizer in the settings of the action field. Then
check the space:

```python
from final_approach.train import make_env

env = make_env()
env.reset(seed=0)
print(env.action_space(env.possible_agents[0]))  # Box(-1.0, 1.0, (1,), float32)
```

The env maps -1 to -40 kt and 1 to +20 kt. Both settings are necessary:

- **Without `low` and `high`**, the field reads its bounds from the aircraft
  performance at run time, and the space is `Box(-inf, inf)`. PPO stops with
  `Continuous action space must have a finite lower and upper bound`.
- **Without the normalizer**, the space is `Box(-40, 20)`. PPO starts with a
  standard deviation of 1 in action units, which is 1 kt here. That is too
  small to change the spacing, so PPO explores very little.

## 3. Train

Run from the folder that contains the package:

```bash
python -m final_approach.train
tensorboard --logdir runs
```

Or call `train()` with your own values:

```python
from final_approach.train import train

train(total_timesteps=1_200_000, n_processes=4, save_every=400_000)
```

The model is saved as `ppo_final_approach.zip` at the end, or when you push
Ctrl+C. Checkpoints go to `checkpoints/`.

**Steps.** Each agent slot of each process is one entry of the vector env:
4 processes x 10 slots = 40 entries. One PPO update collects 2,048 steps from
each entry, so 81,920 agent steps. On a 10-core laptop this takes approximately
2.5 minutes.

**Logs.** `rollout/ep_rew_mean` is the mean return of one aircraft, from spawn
to the end of its flight. For reference, a policy that does nothing gets
approximately -33 for each aircraft on seeds 0 to 9 (see
[step 7 of the final approach tutorial](final-approach.md#7-check-it-with-a-policy-that-does-nothing)).

## 4. Fly the trained policy

```bash
python -m final_approach.evaluate
```

This flies 3 episodes in a Pygame window with the deterministic policy.
Without a window:

```python
from final_approach.evaluate import evaluate

returns = evaluate(episodes=10, render_mode=None)
```

To fly a checkpoint, give its path:
`evaluate("checkpoints/ppo_final_approach_400000_steps")`.

## 5. Use speed choices

To give fixed speed steps instead of a range, use a `StepNormalizer`:

```json
"normalizer": {
  "type": "normalizer",
  "name": "StepNormalizer",
  "kwargs": {"step": 10, "steps_each_way": 4, "steps_up": 2}
}
```

The choices are -40, -30, -20, -10, 0, +10 and +20 kt, and the space is
`MultiDiscrete([7])`. The package then wraps the env in `MultiDiscreteActions`,
and PPO uses a categorical policy. Generate the package again and train it with
the same commands.

SB3 cannot train a continuous action and a choice together. The designer shows
an error for such a design.

## Things to try

- Add a `cost` hook, for example the seconds in loss of separation:
  `return float(context.separation.los.time.during_step_s)`. The package then
  trains with PPO-Lagrangian.
- Add a frame stack to `IntruderFixArrivalDeltaS`, so that the policy sees
  whether a gap opens or closes.
- Change the PPO settings in `train.py` (`learning_rate`, `n_steps`), or the
  network in `networks.py`.
