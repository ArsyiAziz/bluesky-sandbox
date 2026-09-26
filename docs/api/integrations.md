# Integrations

Optional utilities layered on core primitives—enabling space transformations, vectorized execution, and asymmetric state features for Centralized Training with Decentralized Execution (CTDE).


## `bluesky_sandbox.integrations.asymmetric`

Separates critic observations from actor inputs, giving centralized critics access to privileged global state during training without exposing it to individual policy networks.

```{eval-rst}
.. automodule:: bluesky_sandbox.integrations.asymmetric
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.integrations.common.parallel`

```{eval-rst}
.. automodule:: bluesky_sandbox.integrations.common.parallel
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.integrations.vector`

Episodes in parallel: BlueSky is one simulator per process, so each copy of the
env runs in its own worker process, built there from `make_env`. Every agent of
every copy is one entry of the vector; worker `i` is reset with `seed + i`.
Needs the `parallel` extra (`sb3` for Stable-Baselines3).

```python
from bluesky_sandbox.integrations import sb3_vec_env, wrap_parallel_env
from stable_baselines3 import PPO

def make_env(render_mode=None):     # module-level, so each worker can import it
    return wrap_parallel_env(MyTaskEnv(render_mode=render_mode), max_agents=20)

model = PPO("MultiInputPolicy", sb3_vec_env(make_env, n_processes=4))
```

With `watch=True`, one copy is built with `make_env(render_mode="pygame")` and
drawn after every step, the others headless: the copies step together, so a
drawn copy sets the pace for all of them.

With `record=Recording(every=10_000, length=200)`, every copy records its own
clips - named `worker0`, `worker1`, … - built with
`make_env(render_mode="rgb_array")` and drawing only while a clip records. The
workers cannot upload; `Clips(recording).new()` hands their finished clips to
the main process, in memory, once each, and deletes them unless the recording
keeps a `folder` (see [Rendering](../rendering.md#recording-video)):

```python
recording = Recording(every=10_000, length=200)
clips = Clips(recording)
vec = sb3_vec_env(make_env, n_processes=4, record=recording)
...
for clip in clips.new():
    wandb.log({f"video/{clip.name}": wandb.Video(io.BytesIO(clip.data), format="mp4")})
clips.close()  # removes the temporary folder
```

```{eval-rst}
.. automodule:: bluesky_sandbox.integrations.vector
   :members:
```

## `bluesky_sandbox.integrations.common.spaces`

```{eval-rst}
.. automodule:: bluesky_sandbox.integrations.common.spaces
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.integrations.common.wrappers`

```{eval-rst}
.. automodule:: bluesky_sandbox.integrations.common.wrappers
   :members:
   :show-inheritance:
```
