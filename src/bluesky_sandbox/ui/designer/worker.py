"""A warm process the designer samples designs in.

BlueSky is a process singleton, so the designer never flies a design in its own
process; it used to start a fresh Python for every sample, and most of each
sample's few seconds went to importing BlueSky and OpenAP again. This process
imports them once, then runs one job at a time from stdin - a JSON line naming
a generated design package and the script to run against it - printing what
the script prints, then :data:`DONE` (and :data:`ERROR` before it, on a
failure). Each job's package has a name of its own and is dropped from
``sys.modules`` after it, so one design never sees another's code.

One process flies one performance model (BlueSky cannot switch in place):
:mod:`.runner` starts another for a design on a different one.
"""

from __future__ import annotations

import json
import sys
import traceback

DONE = "__WORKER_DONE__"
ERROR = "__WORKER_ERROR__"
READY = "__WORKER_READY__"


def _run(job: dict) -> None:
    workdir, package = job["workdir"], job["package"]
    sys.path.insert(0, workdir)
    try:
        with open(job["script"]) as f:
            source = f.read()
        namespace = {"__name__": "_designer_job", "__file__": job["script"]}
        exec(compile(source, job["script"], "exec"), namespace)
        namespace["main"]()
    finally:
        sys.path.remove(workdir)
        for name in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
            del sys.modules[name]


def main() -> None:
    # The imports every job needs, paid once.
    import bluesky_sandbox.env  # noqa: F401, PLC0415
    import bluesky_sandbox.ui.drivers.common.readouts  # noqa: F401, PLC0415

    print(READY, flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            _run(json.loads(line))
        except BaseException as error:  # noqa: BLE001 - reported, and the next job runs
            last = traceback.format_exception_only(type(error), error)[-1].strip()
            print(ERROR + json.dumps({"error": last}), flush=True)
        finally:
            sys.stdout.flush()
            print(DONE, flush=True)


if __name__ == "__main__":
    main()
