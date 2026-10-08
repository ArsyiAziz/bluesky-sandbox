"""A design's test cases: saved with it (in a folder, tests/cases.json),
checked against its situations, generated into its task package as
cases.py, and run - field checks then cases - in a process of their own."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from bluesky_sandbox.checks import Case, Situation
from bluesky_sandbox.ui.designer import codegen
from bluesky_sandbox.ui.designer import spec as S
from bluesky_sandbox.ui.designer.api import create_app
from bluesky_sandbox.ui.designer.design_tests import cases_module, design_cases
from bluesky_sandbox.ui.designer.folder import CASES_FILE, read_folder, write_folder
from bluesky_sandbox.ui.designer.schema import design_schema

from .test_designer import _example_design_spec

_LEVEL = {"gs_kts": 250, "alt_ft": 10_000, "actype": "B744"}


def _tests() -> dict:
    return {
        "situations": [
            {
                "name": "head-on",
                "aircraft": [
                    {"acid": "OWN", "lat": 52.0, "lon": 4.0, "track_deg": 90, **_LEVEL},
                    {"acid": "INTR", "relative_to": "OWN", "distance_nm": 10, "bearing_deg": 90,
                     "track_deg": 270, **_LEVEL},
                ],
            }
        ],
        "cases": [
            {"situation": "head-on", "field": {"field": "ClosingRateKts"}, "of": "INTR",
             "expected": 500, "tolerance": {"rel": 0.01}},
            {"situation": "head-on", "field": {"field": "AltFt"}, "expected": 9000,
             "tolerance": {"abs": 1}, "note": "wrong on purpose: it flies at 10,000 ft"},
        ],
    }


def _design(tests=None) -> S.DesignSpec:
    spec = _example_design_spec()
    spec.tests = _tests() if tests is None else tests
    return S.DesignSpec.from_json(spec.to_json())


def test_a_design_keeps_its_tests_and_one_without_has_none():
    assert _design().to_dict()["tests"] == _tests()
    assert "tests" not in _design({}).to_dict()


def test_a_folder_keeps_them_beside_its_structure(tmp_path):
    write_folder(_design(), tmp_path)
    assert json.loads((tmp_path / CASES_FILE).read_text()) == _tests()
    assert "tests" not in json.loads((tmp_path / "design.json").read_text())
    assert read_folder(tmp_path).tests == _tests()
    write_folder(_design({}), tmp_path)  # tests dropped: the file goes too
    assert not (tmp_path / CASES_FILE).exists()


def test_the_schema_takes_them():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(design_schema())
    validator.validate(_design().to_dict())
    # Placed one way only: by position and relative both, the schema says no.
    both = _design().to_dict()
    both["tests"]["situations"][0]["aircraft"][1].update(lat=52.0, lon=4.2)
    assert not validator.is_valid(both)


@pytest.mark.parametrize(
    ("break_it", "says"),
    [
        (lambda t: t["situations"][0]["aircraft"][0].update(speed=3), "no speed"),
        (lambda t: t["situations"][0]["aircraft"][1].pop("bearing_deg"), "give one of"),
        (lambda t: t["cases"][0].update(situation="nowhere"), "no situation is named 'nowhere'"),
        (lambda t: t["cases"][0].update(of="GHOST"), "places no aircraft 'GHOST'"),
        (lambda t: t["cases"][0].pop("tolerance"), "needs tolerance"),
        (lambda t: t["cases"][0].update(tolerance={"absolute": 1}), "tolerance"),
    ],
)
def test_broken_tests_say_where(break_it, says):
    tests = _tests()
    break_it(tests)
    with pytest.raises(S.SpecError, match=says):
        design_cases(_design(tests))


def test_they_are_generated_as_the_checks_take_them():
    namespace: dict = {}
    exec(compile(cases_module(_design()), "cases.py", "exec"), namespace)  # noqa: S102
    (situation,) = namespace["SITUATIONS"]
    assert isinstance(situation, Situation) and [a.acid for a in situation.aircraft] == ["OWN", "INTR"]
    first, second = namespace["CASES"]
    assert isinstance(first, Case) and first.of == "INTR" and type(first.field).__name__ == "ClosingRateKts"
    assert second.note.startswith("wrong on purpose")


def test_a_package_carries_them_only_when_there_are_some():
    assert "pkg/cases.py" in codegen.generate_task(_design(), "pkg")
    assert "pkg/cases.py" not in codegen.generate_task(_design({}), "pkg")


def test_they_run_field_checks_then_cases_and_stream_each_result():
    client = TestClient(create_app())
    response = client.post("/api/spec/test", json={"spec": _design().to_dict()})
    assert response.status_code == 200
    lines = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    fields = [r for r in lines if r["kind"] == "field"]
    cases = {r["index"]: r for r in lines if r["kind"] == "case"}
    assert fields and all(r["ok"] for r in fields)
    assert cases[0]["ok"] and cases[0]["got"] == pytest.approx(500.0, rel=0.01)
    assert not cases[1]["ok"] and cases[1]["got"] == pytest.approx(10_000.0, abs=1)
    assert lines[-1]["kind"] == "done"


def test_broken_tests_are_refused_before_anything_runs():
    tests = _tests()
    tests["cases"][0]["situation"] = "nowhere"
    response = TestClient(create_app()).post("/api/spec/test", json={"spec": _design(tests).to_dict()})
    assert response.status_code == 422
    assert "nowhere" in response.json()["detail"]


def test_each_situation_is_placed_for_the_tab_to_draw():
    body = TestClient(create_app()).post("/api/spec/test/situations", json={"spec": _design().to_dict()}).json()
    own, intr = body["situations"]["head-on"]
    assert (own["acid"], own["lat"], own["lon"]) == ("OWN", 52.0, 4.0)
    # 10 nm east of the ownship, as the checks place it.
    assert intr["lat"] == pytest.approx(52.0, abs=1e-3) and intr["lon"] == pytest.approx(4.0 + 10 / 60 / 0.6157, rel=1e-3)
    broken = _tests()
    broken["situations"][0]["aircraft"][1].pop("bearing_deg")
    assert "give one of" in TestClient(create_app()).post(
        "/api/spec/test/situations", json={"spec": _design(broken).to_dict()}
    ).json()["error"]
