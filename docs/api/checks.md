# Checks

A field's bulk code agreeing with its one-aircraft code shows the code is consistent; it cannot show the field means what it should, since a wrong sign or frame is written the same way in both. A test case can. It places aircraft by hand, where the answer can be worked out on paper (an intruder 10 nm dead ahead, closing at 500 kts), and states the value a field should give there, within a tolerance the case chooses.

```python
from bluesky_sandbox.checks import Aircraft, Case, Situation, Tolerance, run_cases

level = dict(gs_kts=250, alt_ft=10_000, actype="B744")
head_on = Situation("head-on", (
    Aircraft("OWN", lat=0.0, lon=0.0, track_deg=90, **level),
    Aircraft("INTR", relative_to="OWN", distance_nm=10, bearing_deg=90, track_deg=270, **level),
))
run_cases(env, [head_on], [
    Case("head-on", obs.ClosingRateKts(), 500.0, Tolerance(rel=0.01), of="INTR"),
]).assert_ok()
```

Cases run in the env's own design geometry (its shapes, queryables and routes) with none of its traffic, so a custom field that reads its environment is checked as it runs. Each kind of field says once how it gives a value in a case (`case_value`); the runner reads built-in and custom fields alike.

## `bluesky_sandbox.checks.cases`

```{eval-rst}
.. automodule:: bluesky_sandbox.checks.cases
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.checks.placement`

```{eval-rst}
.. automodule:: bluesky_sandbox.checks.placement
   :members:
   :show-inheritance:
```
