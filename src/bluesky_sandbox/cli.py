"""The ``bluesky-sandbox`` command: work with a design without the designer.

::

    bluesky-sandbox design check   PATH            build it; report every problem
    bluesky-sandbox design preview PATH [--seed N]  one episode's plan: its aircraft
    bluesky-sandbox design build   PATH [--out DIR] [--name NAME]   its task package
    bluesky-sandbox design test    PATH [PYTEST ARGS]   its fields' checks, test cases and test files
    bluesky-sandbox design convert PATH OUT         a file to a folder, or back
    bluesky-sandbox design schema  [--out FILE]     what a design.json may hold

``PATH`` is a design folder (``design.json`` and ``code/``) or one ``.json``
file. ``check`` exits 1 when the design does not build, and ``test`` as pytest
does when a test fails, so either can gate a commit or a CI job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["main"]

# Where each diagnostics block lives in a design folder.
_FOLDER_FILES = {
    "hook": "code/hooks.py",
    "hook_setup": "code/hooks.py",
    "task_info": "code/task_info.py",
    "task_info_setup": "code/task_info.py",
    "scenario": "code/scenario.py",
    "scenario_setup": "code/scenario.py",
    "spawn_source": "code/scenario.py",
}


def _where(block: str, folder: bool) -> str:
    """A diagnostics block, as a person finds it: its file and function in a
    folder, its place in the design otherwise."""
    kind, _, name = block.partition(":")
    if not folder:
        return block
    if kind == "code":
        return f"code/{name}"
    path = _FOLDER_FILES.get(kind, block)
    if not name:
        return f"{path} (setup)"
    function = f"plan_{name}" if kind == "spawn_source" else name
    return f"{path}: {function}()"


def _check(args: argparse.Namespace) -> int:
    from .ui.designer.builder import BuildError, build_design_config, build_scenario
    from .ui.designer.diagnostics import diagnostics
    from .ui.designer.preview import airspace_warnings

    spec, folder = _load(args.path)
    failed = False
    for block, problems in diagnostics(spec).items():
        for p in problems:
            failed |= p.get("severity") == "error"
            line = p.get("line")
            at = f", line {line}" if line else ""
            print(f"{p.get('severity', 'error')}: {_where(block, folder)}{at}: {p.get('message')}")
    try:
        build_design_config(spec)
        support = build_scenario(spec).support()
    except (BuildError, ValueError, TypeError) as e:
        print(f"error: {e}")
        return 1
    for warning in airspace_warnings(support):
        print(f"warning: outside the airspace: {warning}")
    if failed:
        return 1
    print(f"ok: {args.path} builds - at most {int(support.max_aircraft)} aircraft")
    return 0


def _preview(args: argparse.Namespace) -> int:
    from .ui.designer.preview import scenario_preview

    spec, _folder = _load(args.path)
    preview = scenario_preview(spec, seed=args.seed)
    if args.json:
        print(json.dumps(preview, indent=2, default=str))
        return 0
    aircraft = preview["sampled_aircraft"]
    print(f"seed {args.seed}: {len(aircraft)} aircraft (at most {preview['max_aircraft']})")
    for a in sorted(aircraft, key=lambda a: a["spawn_time"]):
        speed = "envelope" if a["spd_kts"] is None else f"{a['spd_kts']:.0f} kt"
        origin = f"  [{a['source']}]" if a.get("source") else ""
        route = " > ".join(str(s if isinstance(s, str) else s.get("waypoint")) for s in a["route"] or [])
        print(
            f"  t={a['spawn_time']:7.1f} s  {a['actype']:<5} {a['lat']:8.4f} {a['lon']:9.4f}  "
            f"{a['alt_ft']:7.0f} ft  {speed:>8}  {route}{origin}"
        )
    for warning in preview.get("airspace_warnings") or []:
        print(f"warning: outside the airspace: {warning}")
    return 0


def _build(args: argparse.Namespace) -> int:
    from .ui.designer import codegen

    spec, _folder = _load(args.path)
    name = args.name or spec.metadata.get("name") or Path(args.path).stem or "designed_task"
    files = codegen.generate_task(spec, name)
    out = Path(args.out)
    package = _write(files, out)
    print(f"wrote {len(files)} files to {out / package}")
    return 0


def _write(files: dict[str, str], out: Path) -> str:
    """Write a generated package's ``files`` under ``out``; its name."""
    for rel, text in files.items():
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return next(iter(files)).split("/", 1)[0]


def _test(args: argparse.Namespace) -> int:
    import subprocess
    import tempfile

    from .ui.designer import codegen

    spec, _folder = _load(args.path)
    # A design with no tests of its own is still checked field by field.
    spec.tests = spec.tests or {"cases": []}
    with tempfile.TemporaryDirectory(prefix="bsd_test_") as tmp:
        package = _write(codegen.generate_task(spec, "designed_task"), Path(tmp))
        env = dict(os.environ)
        # The package, and the library this command runs from - the one it tests against.
        library = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(p for p in (tmp, library, env.get("PYTHONPATH", "")) if p)
        tests = Path(tmp).resolve() / package / "tests"
        command = [sys.executable, "-m", "pytest", str(tests), "--rootdir", str(tests), "-p", "no:cacheprovider"]
        return subprocess.run([*command, *args.pytest_args], cwd=tests, env=env, check=False).returncode


def _convert(args: argparse.Namespace) -> int:
    from .ui.designer.folder import save_design

    spec, _folder = _load(args.path)
    out = Path(args.out)
    if out.exists() and not args.force and (out.is_file() or any(out.iterdir())):
        print(f"error: {out} exists; pass --force to write over it", file=sys.stderr)
        return 1
    save_design(spec, out)
    print(f"wrote {out}")
    return 0


def _schema(args: argparse.Namespace) -> int:
    from .ui.designer.schema import design_schema

    text = json.dumps(design_schema(), indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


def _load(path: str) -> tuple[Any, bool]:
    """The design at ``path``, and whether it is a folder."""
    from .ui.designer.folder import load_design

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: {path} does not exist")
    return load_design(p), p.is_dir() or p.name == "design.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bluesky-sandbox", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    design = commands.add_parser("design", help="check, preview, build or convert a design")
    actions = design.add_subparsers(dest="action", required=True)

    check = actions.add_parser("check", help="build it; report every problem (exit 1 on any error)")
    check.add_argument("path")
    check.set_defaults(run=_check)

    preview = actions.add_parser("preview", help="one episode's plan: its aircraft")
    preview.add_argument("path")
    preview.add_argument("--seed", type=int, default=0)
    preview.add_argument("--json", action="store_true", help="the whole preview, as JSON")
    preview.set_defaults(run=_preview)

    build = actions.add_parser("build", help="generate its task package")
    build.add_argument("path")
    build.add_argument("--out", default=".", help="where to write the package (default: here)")
    build.add_argument("--name", help="the package name (default: the design's)")
    build.set_defaults(run=_build)

    test = actions.add_parser(
        "test", help="run its fields' checks, test cases and test files with pytest (exit as pytest does)"
    )
    test.add_argument("path")
    test.add_argument("pytest_args", nargs=argparse.REMAINDER, help="passed on to pytest: -k, -x, -v ...")
    test.set_defaults(run=_test)

    convert = actions.add_parser("convert", help="a .json file to a folder, or a folder to a .json file")
    convert.add_argument("path")
    convert.add_argument("out", help="a folder, or a path ending in .json")
    convert.add_argument("--force", action="store_true", help="write over what is there")
    convert.set_defaults(run=_convert)

    schema = actions.add_parser("schema", help="the JSON Schema of design.json")
    schema.add_argument("--out", help="write it here (default: print it)")
    schema.set_defaults(run=_schema)

    args = parser.parse_args(argv)
    return int(args.run(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
