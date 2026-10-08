"""Checks for a design's fields: each field against itself on live traffic
(:mod:`.fields`), and test cases set up by hand (:mod:`.cases`) in situations
placed exactly (:mod:`.placement`)."""

from .cases import Case, CaseResult, CasesReport, Tolerance, fly, run_cases
from .fields import FieldCheck, FieldsReport, check_fields, normalization_findings
from .placement import Aircraft, Situation, place, positions, without_traffic

__all__ = [
    "Aircraft",
    "Case",
    "CaseResult",
    "CasesReport",
    "FieldCheck",
    "FieldsReport",
    "Situation",
    "Tolerance",
    "check_fields",
    "fly",
    "normalization_findings",
    "place",
    "positions",
    "run_cases",
    "without_traffic",
]
