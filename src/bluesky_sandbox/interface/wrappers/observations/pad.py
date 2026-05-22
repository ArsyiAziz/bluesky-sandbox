"""Pad the variable-length intruder block into a fixed-shape Box.

Upstream emits per-agent observations as a Dict::

    {"ownship":   ndarray(ownship_dim,),
     "intruders": ndarray(n_actual, n_features)}            # variable n_actual

SB3's replay buffer requires a fixed shape for every component of a Dict
obs, so we pad ``"intruders"`` out to ``max_intruders x (n_features + 1)``
- the trailing ``+ 1`` is a validity flag (1 for real, 0 for padding)
that downstream consumers can use as an attention mask.

This wrapper does **only** the padding step. Feature normalisation /
deltaisation is handled by the configured observation fields; the values
landing here are already in whatever range those field normalizers produce.

``max_intruders`` is read off the base env at construction time
(``env.unwrapped.max_intruders``, which derives from
``SpawnConfig.max_aircraft()``).
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box, Dict, Sequence
from pettingzoo.utils.wrappers import BaseParallelWrapper


class IntruderPaddingWrapper(BaseParallelWrapper):
    """Pad ``obs["intruders"]`` from variable length to a fixed Box.

    No-op (pass-through) when the wrapped env has no intruder fields
    configured (``config.intruder_obs_fields`` is None/empty).
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self._has_intruders = bool(env.unwrapped.config.intruder_obs_fields)
        self._max_intr = env.unwrapped.max_intruders if self._has_intruders else 0
        self._n_features = 0
        self._n_per_intr = 0
        # Per-intruder aux arrays parallel to ``intruders`` (e.g. intruders_keep):
        # (key, feature_shape, dtype). Padded to ``max_intruders`` rows with zeros.
        self._aux_specs: list[tuple[str, tuple[int, ...], np.dtype]] = []
        self._obs_space = None
        self._refresh_from_env()

    def _refresh_from_env(self) -> None:
        """Refresh upstream feature metadata while keeping the padding cap fixed."""
        self._has_intruders = bool(self.env.unwrapped.config.intruder_obs_fields)
        current_max = self.env.unwrapped.max_intruders if self._has_intruders else 0
        if current_max > self._max_intr:
            raise ValueError(
                "current config would require more intruder rows than the "
                f"construction-time padding cap ({current_max} > {self._max_intr})."
            )

        inner = self.env.observation_space(None)
        if not self._has_intruders:
            self._n_features = 0
            self._n_per_intr = 0
            self._aux_specs = []
            self._obs_space = inner
            return

        # Read the upstream's per-intruder Box (after any field normalisation) and
        # build a padded Box of shape (max_intruders, n_features + 1) where
        # the trailing column is the validity flag.
        feature_box = inner["intruders"].feature_space
        self._n_features = int(feature_box.shape[0])
        self._n_per_intr = self._n_features + 1

        feat_low = np.concatenate(
            [feature_box.low, np.array([0.0], dtype=np.float32)]
        )
        feat_high = np.concatenate(
            [feature_box.high, np.array([1.0], dtype=np.float32)]
        )
        intruders_box = Box(
            low =np.tile(feat_low,  (self._max_intr, 1)),
            high=np.tile(feat_high, (self._max_intr, 1)),
            dtype=np.float32,
        )
        spaces = dict(inner.spaces)
        spaces["intruders"] = intruders_box

        # Pad any parallel per-intruder aux Sequence (e.g. intruders_keep) to a
        # fixed (max_intruders, *feature_shape) Box, zero-filling padded rows.
        self._aux_specs = []
        for key, sub in inner.spaces.items():
            if key in ("ownship", "intruders"):
                continue
            if isinstance(sub, Sequence) and isinstance(sub.feature_space, Box):
                feat = sub.feature_space
                fshape = tuple(feat.shape)
                shape = (self._max_intr,) + fshape
                low = np.broadcast_to(feat.low, shape).astype(feat.dtype)
                high = np.broadcast_to(feat.high, shape).astype(feat.dtype)
                spaces[key] = Box(low=low, high=high, dtype=feat.dtype)
                self._aux_specs.append((key, fshape, feat.dtype))
        self._obs_space = Dict(spaces)

    # ---- public dims (used by SB3 features extractors) -------------------
    @property
    def max_intruders(self) -> int:
        return self._max_intr

    @property
    def intruder_feature_dim(self) -> int:
        return self._n_per_intr

    # ---- spaces ----------------------------------------------------------
    def observation_space(self, _agent):
        self._refresh_from_env()
        return self._obs_space

    # ---- lifecycle -------------------------------------------------------
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.agents = self.env.agents
        self._refresh_from_env()
        return self._pad(obs), info

    def step(self, actions):
        obs, rew, term, trunc, info = self.env.step(actions)
        self.agents = self.env.agents
        self._refresh_from_env()
        return self._pad(obs), rew, term, trunc, info

    # ---- core ------------------------------------------------------------
    def _pad(self, observations: dict) -> dict:
        if not self._has_intruders:
            return observations

        out = {}
        for agent, dict_obs in observations.items():
            intr    = np.asarray(dict_obs["intruders"], dtype=np.float32)

            block = np.zeros((self._max_intr, self._n_per_intr), dtype=np.float32)
            n = min(intr.shape[0], self._max_intr) if intr.ndim == 2 else 0
            if n > 0:
                block[:n, :self._n_features] = intr[:n]
                block[:n,  self._n_features] = 1.0

            padded = {**dict_obs, "intruders": block}
            # Pad parallel per-intruder aux arrays (e.g. intruders_keep) to the
            # same max length, zero-filling padded rows so they stay aligned.
            for key, fshape, dtype in self._aux_specs:
                if key not in dict_obs:
                    continue
                src = np.asarray(dict_obs[key], dtype=dtype)
                aux = np.zeros((self._max_intr,) + fshape, dtype=dtype)
                m = min(src.shape[0], self._max_intr) if src.ndim >= 1 else 0
                if m > 0:
                    aux[:m] = src[:m]
                padded[key] = aux
            out[agent] = padded
        return out
