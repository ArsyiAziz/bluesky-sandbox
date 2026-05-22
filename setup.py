"""Builds the designer frontend as part of the wheel build.

The frontend at ``ui/designer/web`` is a Vite/TypeScript app whose compiled
output is served by the designer API. Two facts shape how it is packaged:

- Installs happen from a git URL, so pip builds the wheel on the *user's*
  machine. Making ``npm`` a hard build requirement would make Node a hard
  install requirement for everyone.
- ``package-data`` globs match nothing when ``dist`` is absent, so a missing
  frontend produces a perfectly valid wheel that 404s at runtime.

So the build is best-effort and the committed ``dist`` is the fallback: npm
runs when it is available and the sources have changed, and the committed
output is used when it is not. The only hard failure is having neither.

Freshness is decided by hashing the frontend sources rather than comparing
mtimes, because git does not preserve mtimes - every file in a fresh clone
looks equally new, which would rebuild on every install.

Building from an sdist skips npm entirely: the sdist carries the compiled
output but not the TypeScript sources, so the shipped dist is authoritative.

Environment variables:
  BLUESKY_SANDBOX_SKIP_NPM=1   never invoke npm; require a committed dist
  BLUESKY_SANDBOX_FORCE_NPM=1  require npm; fail rather than fall back
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

_WEB = Path(__file__).parent / "src" / "bluesky_sandbox" / "ui" / "designer" / "web"
_DIST = _WEB / "dist"
_STAMP = _DIST / ".build-hash"

# Everything the compiled output depends on. node_modules and dist are
# excluded: one is an input's *installation*, the other is the output.
_SOURCES = ("index.html", "package.json", "package-lock.json", "tsconfig.json",
            "vite.config.ts")


def _source_hash() -> str:
    h = hashlib.sha256()
    paths = [_WEB / name for name in _SOURCES]
    paths += sorted(p for p in (_WEB / "src").rglob("*") if p.is_file())
    for path in paths:
        if not path.is_file():
            continue
        h.update(str(path.relative_to(_WEB)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _built() -> bool:
    return (_DIST / "index.html").is_file() and any(_DIST.glob("assets/*.js"))


def _run(cmd: list[str]) -> None:
    print(f"  running {' '.join(cmd)} in {_WEB}", flush=True)
    proc = subprocess.run(cmd, cwd=_WEB, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        raise SystemExit(
            f"\nThe designer frontend failed to build ({' '.join(cmd)} exited "
            f"{proc.returncode}).\nFix the error above, or set "
            f"BLUESKY_SANDBOX_SKIP_NPM=1 to package the committed dist as-is.\n"
        )


def _fail_unbuilt(reason: str) -> None:
    raise SystemExit(
        f"\nThe designer frontend is not built and {reason}.\n"
        f"Build it before packaging:\n\n    cd {_WEB}\n    npm ci && npm run build\n"
    )


class build_py_with_frontend(build_py):
    def run(self):
        self.build_frontend()
        super().run()

    def build_frontend(self) -> None:
        want = _source_hash()
        have = _STAMP.read_text().strip() if _STAMP.is_file() else None

        if os.environ.get("BLUESKY_SANDBOX_SKIP_NPM"):
            if not _built():
                _fail_unbuilt("BLUESKY_SANDBOX_SKIP_NPM is set")
            print("designer frontend: using committed dist (npm skipped)")
            return

        # Building from an sdist: it ships the compiled dist but deliberately
        # not the TypeScript sources, so there is nothing to build from and the
        # shipped output is authoritative. Without this, the absent sources hash
        # differently from the stamp and npm is invoked in a tree with no
        # package.json.
        if not (_WEB / "package.json").is_file():
            if not _built():
                _fail_unbuilt("the frontend sources are not present either")
            print("designer frontend: using packaged dist (sources not in sdist)")
            return

        if _built() and have == want:
            print("designer frontend: up to date, skipping npm")
            return

        if shutil.which("npm") is None:
            if os.environ.get("BLUESKY_SANDBOX_FORCE_NPM"):
                raise SystemExit(
                    "\nBLUESKY_SANDBOX_FORCE_NPM is set but npm was not found on "
                    "PATH.\nInstall Node, or unset the variable to package the "
                    "committed dist as-is.\n"
                )
            if not _built():
                _fail_unbuilt("npm was not found on PATH")
            # The committed output may predate the current sources, but a
            # stale UI beats refusing to install on a machine without Node.
            print(
                "designer frontend: npm not found, using committed dist "
                "(it may be stale)", flush=True,
            )
            return

        reason = "not built" if not _built() else "sources changed"
        print(f"designer frontend: {reason}, building with npm", flush=True)
        if not (_WEB / "node_modules").is_dir():
            _run(["npm", "ci", "--no-audit", "--no-fund"])
        _run(["npm", "run", "build"])
        if not _built():
            raise SystemExit(
                f"\nnpm run build completed but {_DIST} still has no output.\n"
            )
        _STAMP.write_text(want + "\n")
        print("designer frontend: built", flush=True)


setup(cmdclass={"build_py": build_py_with_frontend})
