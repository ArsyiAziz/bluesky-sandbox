"""Aircraft performance: what a type can fly, and how fast.

:mod:`.envelope` samples feasible (altitude, CAS) targets and fleet-wide
aggregates; :mod:`.speeds` handles the CAS/Mach crossover those targets are
expressed against.
"""

from __future__ import annotations

from .bada import (
    bada_available,
    bada_data_dir,
    bada_coefficients,
    ensure_user_resource_root,
    bada_install_hint,
    bada_aircraft_types,
    load_perf_bada,
)
from .models import (
    MODELS,
    available_types,
    spawnable_types,
    type_limits,
)
from .envelope import (
    EnvelopeSample,
    feasible_alt_cas,
    feasible_alt_for_type,
    feasible_cas_at_alt,
    fleet_ceiling_ft,
    fleet_max_cas_kt,
    fleet_sim_ceiling_ft,
    fleet_speed_band_kt,
)
from .speeds import (
    CrossoverSpeedState,
    cas_ceiling_ms,
    cas_tolerance_as_mach,
    crossover_display,
    crossover_speed_state,
    within_speed_tolerance,
    within_speed_tolerance_many,
)

__all__ = [
    "MODELS",
    "available_types",
    "spawnable_types",
    "type_limits",
    "bada_available",
    "bada_data_dir",
    "bada_coefficients",
    "ensure_user_resource_root",
    "bada_install_hint",
    "bada_aircraft_types",
    "load_perf_bada",
    "CrossoverSpeedState",
    "EnvelopeSample",
    "cas_ceiling_ms",
    "cas_tolerance_as_mach",
    "crossover_display",
    "crossover_speed_state",
    "feasible_alt_cas",
    "feasible_alt_for_type",
    "feasible_cas_at_alt",
    "fleet_ceiling_ft",
    "fleet_max_cas_kt",
    "fleet_sim_ceiling_ft",
    "fleet_speed_band_kt",
    "within_speed_tolerance",
    "within_speed_tolerance_many",
]
