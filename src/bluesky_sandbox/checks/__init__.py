"""Checks for a design's fields: test cases set up by hand (:mod:`.cases`),
in situations placed exactly (:mod:`.placement`)."""

from .cases import Case, CaseResult, CasesReport, Tolerance, run_cases
from .placement import Aircraft, Situation, place, without_traffic

__all__ = [
    "Aircraft",
    "Case",
    "CaseResult",
    "CasesReport",
    "Situation",
    "Tolerance",
    "place",
    "run_cases",
    "without_traffic",
]
