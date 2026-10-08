"""A design's test cases: saved with it (in a folder, tests/cases.json),
checked against its situations, generated into its task package as
cases.py, and run - field checks then cases - in a process of their own."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from bluesky_sandbox.checks import Case, Situation
from bluesky_sandbox.cli import main as cli_main
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
    # One being written does not hide the rest: each is placed on its own.
    tests = _tests()
    tests["situations"].append({"name": "unfinished", "aircraft": [{"acid": "A", "lat": 52.0, "lon": 4.0}]})
    body = TestClient(create_app()).post("/api/spec/test/situations", json={"spec": _design(tests).to_dict()}).json()
    assert list(body["situations"]) == ["head-on"]
    assert body["errors"] == ["tests: situation 'unfinished', aircraft 'A' needs track_deg, gs_kts, alt_ft, actype"]


def test_an_action_case_applies_and_reads():
    tests = _tests()
    tests["cases"] = [
        {"situation": "head-on", "apply": {"field": "AltDeltaFt"}, "value": 1000, "field": {"field": "AltDeltaFt"},
         "expected": 11000, "tolerance": {"abs": 1}},
        {"situation": "head-on", "apply": {"field": "AltDeltaFt"}, "value": 1000,
         "field": {"field": "ApAltFt"}, "expected": 11000, "tolerance": {"abs": 1}},
    ]
    design = _design(tests)
    pytest.importorskip("jsonschema").Draft202012Validator(design_schema()).validate(design.to_dict())
    namespace: dict = {}
    exec(compile(cases_module(design), "cases.py", "exec"), namespace)  # noqa: S102
    held, selected = namespace["CASES"]
    assert type(held.apply).__name__ == type(held.field).__name__ == "AltDeltaFt" and held.value == 1000
    assert type(selected.field).__name__ == "ApAltFt"
    response = TestClient(create_app()).post("/api/spec/test", json={"spec": design.to_dict()})
    cases = [json.loads(line) for line in response.text.splitlines() if '"case"' in line]
    assert [c["ok"] for c in cases] == [True, True], cases


def test_half_an_action_case_is_refused():
    tests = _tests()
    tests["cases"][0]["apply"] = {"field": "AltDeltaFt"}
    with pytest.raises(S.SpecError, match="an action and a value, or neither"):
        design_cases(_design(tests))


_MY_TESTS = """import bluesky as bs


def test_it_resets_with_aircraft(design_env):
    design_env.reset(seed=0)
    assert bs.traf.ntraf > 0


def test_wrong_on_purpose(design_env):
    design_env.reset(seed=0)
    assert bs.traf.ntraf == 999, "not 999 aircraft"
"""


def _with_files(files: dict) -> S.DesignSpec:
    tests = _tests()
    tests["files"] = files
    return _design(tests)


def test_a_folder_keeps_test_files_as_files(tmp_path):
    write_folder(_with_files({"test_mine.py": _MY_TESTS}), tmp_path)
    assert (tmp_path / "tests" / "test_mine.py").read_text() == _MY_TESTS
    assert "files" not in json.loads((tmp_path / CASES_FILE).read_text())
    assert read_folder(tmp_path).tests["files"] == {"test_mine.py": _MY_TESTS}
    write_folder(_with_files({}), tmp_path)  # dropped: the file goes too
    assert not (tmp_path / "tests" / "test_mine.py").exists()


@pytest.mark.parametrize(("name", "says"), [("mine.py", "test_<name>.py"), ("test_design.py", "the package's own")])
def test_a_test_file_is_named_as_pytest_finds_it(name, says):
    with pytest.raises(S.SpecError, match=says):
        design_cases(_with_files({name: "pass\n"}))


def test_a_package_runs_its_tests_with_pytest():
    files = codegen.generate_task(_with_files({"test_mine.py": _MY_TESTS}), "pkg")
    assert {"pkg/tests/conftest.py", "pkg/tests/test_design.py", "pkg/tests/test_mine.py"} <= set(files)
    assert "def design_env" in files["pkg/tests/conftest.py"]
    assert "check_fields(design_env)" in files["pkg/tests/test_design.py"]


def test_the_designer_runs_the_test_files_after_the_cases():
    response = TestClient(create_app()).post("/api/spec/test", json={"spec": _with_files({"test_mine.py": _MY_TESTS}).to_dict()})
    tests = {r["name"]: r for r in map(json.loads, response.text.splitlines()) if r["kind"] == "test"}
    assert tests["test_mine.py::test_it_resets_with_aircraft"]["ok"]
    wrong = tests["test_mine.py::test_wrong_on_purpose"]
    assert not wrong["ok"] and wrong["line"] == 11 and "not 999 aircraft" in wrong["error"]


def test_the_command_line_runs_them_and_exits_as_pytest_does(tmp_path):
    fixed = _tests()
    fixed["cases"][1]["expected"] = 10_000  # what it flies at
    write_folder(_design(fixed), tmp_path / "ok")
    assert cli_main(["design", "test", str(tmp_path / "ok"), "-q", "-p", "no:warnings"]) == 0
    write_folder(_with_files({"test_mine.py": _MY_TESTS}), tmp_path / "failing")
    assert cli_main(["design", "test", str(tmp_path / "failing"), "-q", "-p", "no:warnings"]) == 1
