"""Messages between aircraft: what each intruder broadcast last step, as a
learned communication channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import bluesky as bs
import numpy as np

from .. import _state
from .._common import _indices_array
from .._state import _CommBacked, comm_messages
from ..base import ObsMeta, ObsQuantity, PairObsField, Unit


@dataclass(frozen=True)
class IntruderCommMessage(_CommBacked, PairObsField):
    """One channel of an intruder's broadcast communication message.

    Reads the value the intruder emitted through its ``CommBroadcast`` action
    on the previous step (0.0 for an aircraft that has not yet spoken). A pure
    agent-to-agent information channel with no physical semantics - the
    meaning of the signal is whatever the shared policy learns to encode.
    Values live in the action's ``[-1, 1]`` range.

    ``noise_std > 0`` adds receiver-side Gaussian channel noise (clipped back
    to the message range), drawn from a per-episode-seeded RNG. The DIAL/DRU
    grounding pressure: a message must be high-contrast to survive a noisy
    channel, so ambiguous low-amplitude signaling stops being free.
    """

    meta = ObsMeta(
        "intruder_comm_message",
        Unit.UNITLESS,
        ObsQuantity.ACTION,
        is_pair=True,
    )
    channel: Annotated[int, "message channel index"] = 0
    noise_std: Annotated[float, "receiver-side Gaussian channel noise std"] = 0.0
    low: Annotated[float, "message value"] = -1.0
    high: Annotated[float, "message value"] = 1.0

    def get_pair(self, own_idx: int, other_idx: Any) -> Any:
        return float(self.get_pairs(own_idx, [other_idx])[0])

    def get_pairs(self, own_idx: int, other_indices: Any) -> Any:
        others = _indices_array(other_indices).ravel()
        return self._received(np.array([int(own_idx)]), others[None, :])[0]

    def get_pair_matrix(self, own_indices: Any) -> np.ndarray:
        owns = _indices_array(own_indices).ravel()
        n = int(bs.traf.ntraf)
        # Every other aircraft, in order, for each ownship: exactly the pairs
        # get_pairs would be asked for, so the noise draws come in the same order.
        heard = np.arange(n)[None, :] != owns[:, None]
        matrix = np.full((owns.size, n), np.nan)
        cols = np.broadcast_to(np.arange(n), heard.shape)[heard].reshape(owns.size, -1)
        matrix[heard] = self._received(owns, cols).ravel()
        return matrix

    def _received(self, owns: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Each ownship's received messages from aircraft ``cols[row]``.

        The message does not depend on the listener; the receiver noise does,
        drawn row by row so a matrix consumes the noise stream exactly as one
        ``get_pairs`` call per ownship would.
        """
        del owns
        sent = comm_messages(self.channel)
        values = sent[cols]
        if self.noise_std > 0.0 and values.size:
            values = values + _state._COMM_NOISE_RNG.normal(
                0.0, self.noise_std, size=values.shape
            )
            np.clip(values, self.low, self.high, out=values)
        return values

    def _expected_pair(self, own_idx: int, other_idx: int) -> Any:
        # The message as sent; receiver noise (``noise_std``) is random on top.
        del own_idx
        return float(_state._COMM_MESSAGE.read_one(other_idx).get(self.channel, 0.0))

    def bounds(self, own_idx: int) -> tuple[float, float]:
        del own_idx
        return float(self.low), float(self.high)
