# Checks

Two kinds of check, for built-in and custom fields alike.

**Each field against itself.** A field may compute the same value several ways - in bulk for every aircraft, one at a time, as a pair matrix, normalized as a batch or one value at a time - and the env reads whichever is quickest where it is. `check_fields(env)` steps the env on live traffic and compares them, and compares a field with its plain statement of its value, `expected(idx)` (`expected_pair(own, other)` for a pair field), where it states one. Every built-in field states one; a custom field may. Over the same run, a value that is not a number (NaN, infinite) fails, and so does one that differs when the episode is flown again with the same seed. A value outside the field's bounds is noted, not failed: a clipped normalizer loses it, and whether that matters is the designer's call. Each field is also timed as the env reads it, every aircraft at once, against the simulation's own step: `FieldCheck.ms`, whether it has a batched path (`FieldCheck.batched`), and `FieldsReport.step_ms`.

```python
from bluesky_sandbox.checks import check_fields

check_fields(env).assert_ok()
```

**Test cases.** A field's bulk code agreeing with its one-aircraft code shows the code is consistent; it cannot show the field means what it should, since a wrong sign or frame is written the same way in both. A test case can. It places aircraft by hand, where the answer can be worked out on paper (an intruder 10 nm dead ahead, closing at 500 kts), and states the value a field should give there, within a tolerance the case chooses.

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

A case can first give an action a value; BlueSky takes the command, and nothing is flown - the case tests what the field says, not how BlueSky flies it. Reading an action gives what it holds:

```python
Case("head-on", act.AltDeltaFt(), 11_000.0, Tolerance(abs=1), apply=act.AltDeltaFt(), value=1000.0)
Case("head-on", obs.ApAltFt(), 11_000.0, Tolerance(abs=1), apply=act.AltDeltaFt(), value=1000.0)
```

A failed case says what it saw on the way: the value given (and on its grid), the command BlueSky took, what the action holds, and the field's value raw and normalized.

Cases run in the env's own design geometry (its shapes, queryables and routes) with none of its traffic, so a custom field that reads its environment is checked as it runs. Each kind of field says once how it gives a value in a case (`case_value`); the runner reads built-in and custom fields alike.

**Probing a field.** A check says whether a field is right; a probe shows how it got where it is. With the env flown to a moment, `probe(env, field, acid)` reads one field for one aircraft: each stage from its inputs to what the policy sees, actual and expected (raw then normalized for an observation; the policy's value, the value on its grid, the target and the BlueSky command for an action), the calls it made into the library and the design's code with what each returned, and each BlueSky traffic value it read. An override sets a traffic value before the field runs (everything that reads it sees it, and it is put back after) or makes a call return a value given, whole or one number in it:

```python
from bluesky_sandbox.checks.probe import Override, probe

probe(env, obs.AltFt(), "KLM12", overrides=[Override("traf:alt", 3048.0)])
probe(env, my_field, "KLM12", overrides=[Override("call:my_design.fields:reach_of", 20.0, leaf="nm")])
```

For a field that is stacked, a `LagRecorder` told to record at each step gives the probe the history to show the lag ring with: the frames the ring holds for the aircraft, and what each lagged field read at each step. The probe also times each way the field computes.

## `bluesky_sandbox.checks.fields`

```{eval-rst}
.. automodule:: bluesky_sandbox.checks.fields
   :members:
   :show-inheritance:
```

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

## `bluesky_sandbox.checks.probe`

```{eval-rst}
.. automodule:: bluesky_sandbox.checks.probe
   :members: probe, Override, Stage, TraceNode, ProbeResult, LagRecorder
   :show-inheritance:
```
