# Environment and configuration

## `bluesky_sandbox.env`

The class a task subclasses.

```{eval-rst}
.. automodule:: bluesky_sandbox.env
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.config`

Static configuration: which observation and action fields make up the interface, and the
simulator settings the episode runs under.

```{eval-rst}
.. automodule:: bluesky_sandbox.config
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.core.base_environment`

The PettingZoo `ParallelEnv` implementation underneath. Most tasks never touch this
directly, but the hook protocol and the aircraft lifecycle states are defined here.

```{eval-rst}
.. automodule:: bluesky_sandbox.core.base_environment
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.core.layout`

Which columns of an observation or action belong to which field. The action space
is a `Box` while every action is continuous; with a switch it is a `Dict` of a
`continuous` Box and a `binary` MultiBinary:

```python
env.action_space(agent)     # Dict('binary': MultiBinary(2), 'continuous': Box(-1.0, 1.0, (4,)))
env.action_layout(agent)    # {"continuous": [Slot("active_route_waypoint_alt_delta_ft", 0:1), ...],
                            #  "binary":     [Slot("autopilot_lnav", 0:1), Slot("autopilot_vnav", 1:2)]}
env.step({agent: {"continuous": cont, "binary": bits}})
```

A layout depends only on the config, so `observation_layout(config)` and
`action_layout(config)` give it without an environment - in a vector env's main
process, say.

```{eval-rst}
.. automodule:: bluesky_sandbox.core.layout
   :members:
.. automodule:: bluesky_sandbox.core.slot
   :members:
```

## `bluesky_sandbox.core.step_values`

Raw (unnormalized) values, by field name, from the arrays the observation is
built from - so reading them costs no recomputation:

```python
raw = context.raw_obs                        # in a hook; env.raw_observation(agent) outside
raw["ownship"]["alt_ft"]                     # 12000.0, in the field's unit
raw["intruders"]["dist_to_own_nm"][i]        # intruder i: row i of obs["intruders"]
raw["intruders"]["acid"][i]                  # its callsign
context.raw_action["alt_delta_ft"]           # what the action field was set to, ft
```

State fields are read the same way, but no agent observes them: list them in
`EnvConfig.state_fields` / `intruder_state_fields`, and hooks read them by name,
in raw values. Use them for what a reward, a done condition or an info needs but
the policy must not depend on, whether actor or critic:

```python
config = EnvConfig(
    obs_fields=[obs.CasKts()],
    state_fields=[obs.AltFt()],                      # computed, never observed
    intruder_state_fields=[obs.DistToOwnNm()],
)
context.state["ownship"]["alt_ft"]           # in a hook; env.raw_state(agent) outside
context.state["intruders"]["dist_to_own_nm"][i]
batch.state["ownship"]["alt_ft"]             # (n_agents,), in a batched hook
```

```{eval-rst}
.. automodule:: bluesky_sandbox.core.step_values
   :members: RawObservation, StepValues
```

## `bluesky_sandbox.core.batch`

Batched hooks: `reward_batch`, `cost_batch`, `terminated_batch` and
`truncated_batch` are called once per step with every agent's step stacked into
arrays, instead of once per agent. Each is optional; a task defines a hook one
way or the other, never both.

A reward, and a cost, is a number or a 1-D array of components - per agent from
`reward` / `cost`, or shape `(n_agents,)` / `(n_agents, k)` from the batched
hooks - and keeps one shape from step to step. `cost` is optional: a task that
defines it gets each agent's cost in `info["cost"]`.

```python
class MyEnv(BlueskyEnv):
    def reward(self, obs, action, terminated, truncated, context, info, rng):
        return np.array([progress(context), -effort(action)])   # two components

    def cost_batch(self, batch):                # (n_agents, 2): one column per constraint
        dist = batch.raw_obs["intruders"]["dist_to_own_nm"]
        return np.stack([(dist < 5.0).any(axis=1), (dist < 10.0).sum(axis=1)], axis=1)
```

```python
class MyEnv(BlueskyEnv):
    def reward_batch(self, batch):              # one value per batch.acids
        alt_err = batch.raw_obs["ownship"]["active_route_waypoint_alt_diff_ft"]   # (n_agents,)
        close = batch.raw_obs["intruders"]["dist_to_own_nm"] < 5.0                 # (n_agents, n_intruders)
        return -np.abs(alt_err) / 1000.0 - close.sum(axis=1)
```

A task-info provider with `batched = True` is called the same way, `provider(batch)`,
writing into `batch.infos[k]`.

```{eval-rst}
.. automodule:: bluesky_sandbox.core.batch
   :members: StepBatch, RawBatch
```

## `bluesky_sandbox.core.runtime`

The BlueSky process lifecycle — startup, stepping, teardown.

```{eval-rst}
.. automodule:: bluesky_sandbox.core.runtime
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.core.services`

Services the environment exposes to fields and task hooks during a step.

```{eval-rst}
.. automodule:: bluesky_sandbox.core.services
   :members:
   :show-inheritance:
```
