"""``python -m bluesky_sandbox.doctor`` - what is installed, and where.

Performance data is the thing most likely to be missing after a fresh install:
OpenAP ships with its package, BADA is licensed and does not. Without a check
like this, a design that names BADA types fails at aircraft resolution, deep
inside env construction, with an error about a type not being available - the
symptom is far from the cause.
"""

from __future__ import annotations

import sys

import bluesky as bs


def report() -> tuple[list[str], bool]:
    """Return ``(lines, ok)`` describing the install."""
    # BADA is found via ``bs.resource()``, whose search roots include the user
    # directory only after init. Probing first reports installed data missing.
    bs.init("sim")

    # Deferred for the same reason: both probe BADA on import path.
    from bluesky_sandbox.config import _available_aircraft  # noqa: PLC0415
    from bluesky_sandbox.sim.performance.bada import (  # noqa: PLC0415
        bada_available,
        bada_data_dir,
    )

    lines: list[str] = []
    ok = True

    configured = str(getattr(bs.settings, "performance_model", "openap")).lower()
    lines.append(f"configured performance model : {configured}")

    for model in ("openap", "bada"):
        mark = " (configured)" if model == configured else ""
        try:
            n = len(_available_aircraft(model))
            lines.append(f"  {model:<7}: {n} aircraft types{mark}")
        except Exception as e:
            lines.append(f"  {model:<7}: UNAVAILABLE{mark}")
            lines.append(f"           {e}")
            if model == configured:
                ok = False

    lines.append("")
    compiled_lines, compiled_ok = _compiled_modules()
    lines.extend(compiled_lines)
    ok = ok and compiled_ok

    lines.append("")
    lines.append(f"BADA data directory : {bada_data_dir()}")
    lines.append(f"BADA data present   : {'yes' if bada_available() else 'no'}")
    lines.append("BlueSky resource search order:")
    try:
        for i, base in enumerate(bs.resource.path.bases()):
            lines.append(f"  {i}: {base}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"  (unavailable: {e})")
    return lines, ok


def _compiled_modules() -> tuple[list[str], bool]:
    """Whether BlueSky's compiled geo functions and conflict detector load.

    Both are built when BlueSky is installed, and both fall back quietly when
    they fail to load. Python geo is only slower, and differs in the last
    digits. Without the compiled detector, the default ``cd_method``
    (``CSTATEBASED``) does not exist: ``CDMETHOD`` fails without raising, and
    BlueSky runs with no conflict detection at all.
    """
    from bluesky.tools import geo  # noqa: PLC0415
    from bluesky.traffic.asas import statebased  # noqa: PLC0415

    from bluesky_sandbox.config import DEFAULT_CD_METHOD  # noqa: PLC0415

    compiled_geo = geo.select_backend().__name__.endswith("_cgeo")
    compiled_cd = hasattr(statebased, "CStateBased")
    lines = [
        f"geo functions      : {'compiled' if compiled_geo else 'Python'}",
        "conflict detection : "
        + ("compiled (CSTATEBASED)" if compiled_cd else "UNAVAILABLE (CSTATEBASED)"),
    ]
    if not compiled_cd:
        lines.append(
            f"                     the default cd_method ({DEFAULT_CD_METHOD}) "
            "needs it; reinstall bluesky-simulator with a C++ compiler"
        )
    return lines, compiled_cd


def main() -> int:
    lines, ok = report()
    print("\n".join(lines))
    if not ok:
        print("\nA check above failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
