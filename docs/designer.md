# Environment Designer

The designer is a browser tool for laying out an environment and exporting it as a Python
package. To install the environment designer, run the following:
```bash
pip install "bluesky-sandbox[designer] @ git+https://github.com/ArsyiAziz/bluesky-sandbox.git"
python -m bluesky_sandbox.ui.designer --port 8765
```

Then open `http://localhost:8765`.

```{image} media/screenshots/designer.png
:alt: The Environment Designer with JFK's CAMRN FIVE arrival: its fixes, two spawn streams feeding SIE, and the route library on the map
:width: 620px
:align: center
```

## A tour of the designer

The screenshots show JFK's CAMRN FIVE arrival: two streams of traffic feeding the STAR at SIE, descending to cross HOGGS  at FL180 and CAMRN at 11,000 ft. 

**Map.** Lay out the airspace, waypoints (search a navdb fix by name), spawn regions and routes; reseed to preview an episode's aircraft. Each fix shows the altitude and speed its routes cross it at.

**Route.** The route library: routes built from waypoint steps, each with its crossing (altitude, speed, fly-over), and routes made of other routes - here each feeder airway joined to the STAR.

```{image} media/screenshots/designer-routes.png
:alt: The Route tab: the route library, the CAMRN5 route's steps with their crossings, and the routes on the map
:width: 620px
:align: center
```

**Spaces.** What each aircraft observes and what it can be told: every observation and action field, its raw range, its normalizer and what the policy sees.

```{image} media/screenshots/designer-spaces.png
:alt: The Spaces tab: ownship and intruder observation fields with their raw ranges, normalizers and the values a policy sees
:width: 620px
:align: center
```

Expand a field to see how its normalizer maps raw values; open it to probe it - fly the design to a moment and trace the value from the simulator through its normalizer to what the policy sees.

```{image} media/screenshots/field-probe.png
:alt: Probing the track field: the value read from BlueSky, its normalized encoding, and the trace of each call
:width: 620px
:align: center
```

**Code.** The environment's hooks, task information, scenario code and spawn plans, as Python; and the generated package, read-only. 
Here `on_aircraft_spawned` switches VNAV on, so each aircraft flies its route's altitude and speed crossings.

```{image} media/screenshots/designer-code.png
:alt: The Code tab with the on_aircraft_spawned hook switching VNAV on for each aircraft
:width: 620px
:align: center
```

**Config** holds the simulator and conflict-detection settings, and
**Metadata** the design's name, version and note.

## Command-line options

| Flag | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Interface to bind. |
| `--port` | `8765` | Port to serve on. |
| `--reload` | off | Auto-reload on source change, for development. |

The `bluesky-sandbox-designer` console script is equivalent to
`python -m bluesky_sandbox.ui.designer`.

## What the designer generates:

**Generate task** writes a self-contained package:

```text
demo_task/
  design.json   # the design itself
  scenario.py   # geometry / spawn / queryables
  config.py     # observation + action fields, simulator settings
  setup.py      # module-level helpers the hooks lean on
  env.py        # reward, termination, and other task hooks
  task_types.py # this design's keys as types, for your editor
  __main__.py   # python -m demo_task - a short example rollout
  __init__.py   # Env, Scenario, and the design's version
  README.md     # what the design is, from its note.
```

The hooks in `env.py` and the task-info functions in `setup.py` are annotated
with the types in `task_types.py`, generated from the design: an editor that
reads annotations (VS Code, PyCharm) completes `context.obs["ownship"]["alt_ft"]`
or `context.query("goal")`, and a type checker flags a key the design does not
have. They are for editors only - nothing in them runs.
