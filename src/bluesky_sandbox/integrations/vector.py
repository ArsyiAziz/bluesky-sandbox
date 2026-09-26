"""Copies of a task env as one vector env, each copy in its own process.

BlueSky is one simulator per process, so running episodes in parallel means
running processes in parallel: each copy of the env is built inside its own
worker from ``make_env`` - never copied from this one - and every agent of
every copy is one entry of the vector. Worker ``i`` is reset with ``seed + i``,
so the copies play different episodes.

Training draws nothing, but with ``record``, every copy records its own clips
(:class:`~bluesky_sandbox.interface.wrappers.RecordVideo`), built with
``make_env(render_mode="rgb_array")`` and named ``worker{i}``: the copies step
together, so their clips cover the same steps. They draw only while a clip
records. :class:`~bluesky_sandbox.interface.wrappers.Clips` finds them from
this process, to log.

Needs SuperSuit (and Stable-Baselines3 for :func:`sb3_vec_env`), which the
library does not depend on otherwise.
"""

from __future__ import annotations

import functools
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from bluesky_sandbox.interface.wrappers.record import Recording, RecordVideo

__all__ = ["sb3_vec_env", "vec_env"]


def _markov(
    make_env: Callable[..., Any],
    record: Recording | None = None,
    worker: int = 0,
) -> Any:
    from supersuit.vector.markov_vector_wrapper import MarkovVectorEnv  # noqa: PLC0415 - optional

    if record is not None:
        env = RecordVideo(
            make_env(render_mode="rgb_array"), record, name=f"worker{worker}"
        )
    else:
        env = make_env()
    return MarkovVectorEnv(env)


def _probe(make_env: Callable[[], Any]) -> tuple[Any, Any, int, dict]:
    """One copy's spaces and agent count - built, read and closed in a worker,
    so this process never starts a simulator of its own."""
    env = _markov(make_env)
    try:
        return env.observation_space, env.action_space, env.num_envs, dict(env.metadata)
    finally:
        env.close()


def vec_env(
    make_env: Callable[..., Any],
    n_processes: int = 1,
    record: Recording | None = None,
) -> Any:
    """``n_processes`` copies of the env from ``make_env`` as one gymnasium
    vector env, one entry per agent of each copy.

    ``make_env`` builds one env as SuperSuit takes it - a PettingZoo parallel
    env with a fixed agent pool and fixed-shape observations, such as
    :func:`~bluesky_sandbox.integrations.wrap_parallel_env` makes. It must be
    importable or picklable: each worker calls it to build its own copy. With
    one process the copy is built here, with no worker. With ``record`` it
    must take ``render_mode``: every copy is built drawing offscreen, and
    records.
    """
    if n_processes < 1:
        raise ValueError(f"n_processes must be at least 1, got {n_processes}.")
    if record is not None:
        record.path  # noqa: B018 - makes a temporary folder here, for every worker to share
    if n_processes == 1:
        return _markov(make_env, record)
    from supersuit.vector.multiproc_vec import ProcConcatVec  # noqa: PLC0415 - optional

    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        obs_space, act_space, per_copy, metadata = pool.submit(_probe, make_env).result()
    builds = [
        functools.partial(_markov, make_env, record, i)
        for i in range(n_processes)
    ]
    return ProcConcatVec(builds, obs_space, act_space, per_copy * n_processes, metadata)


def sb3_vec_env(
    make_env: Callable[..., Any],
    n_processes: int = 1,
    record: Recording | None = None,
) -> Any:
    """:func:`vec_env` as a Stable-Baselines3 ``VecEnv``."""
    from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper  # noqa: PLC0415 - optional

    class _SeededOnReset(SB3VecEnvWrapper):
        # SuperSuit's SB3 wrapper expects a vector env that seeds itself;
        # these seed on reset, so SB3's seed is kept for the next one.
        _seed = None

        def seed(self, seed=None):
            self._seed = seed
            return [seed]

        def reset(self, seed=None, options=None):
            seed, self._seed = (self._seed if seed is None else seed), None
            observations, self.reset_infos = self.venv.reset(seed=seed, options=options)
            return observations

    return _SeededOnReset(vec_env(make_env, n_processes, record))
