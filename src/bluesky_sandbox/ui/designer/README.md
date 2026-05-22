# Environment Designer

A declarative-spec layer plus web app for creating, configuring, and iterating
on bluesky-sandbox environments. It sits on the simulation primitives
(`bounds`, `queryables`, `spawn`, `distributions`, `scenarios`) and exposes two
workflows: a **Map** tab (airspace / spawn / query regions and navigation data
over a slippy map) and a **Code** tab (a VS Code-style editor over the design
document).

## Architecture

```
DesignSpec (JSON)  ── the single source of truth
   │
   ├── airspace / queryables / spawn   (structured data → spec.dump/load)
   └── env: fields + "module:attr" code refs  (logic → resolved at build)
        │
   builder.build_scenario        builder.build_design_config
        │
   Scenario/EpisodeSpec          EnvConfig
        │                         │
        └──── bluesky_sandbox runtime (BlueskyEnv)
```

The seam follows the primitives: structured geometry/spawn/queryables/dists are
serialised and GUI-editable; reward/termination/field logic is referenced by
import string and edited as code.

### Backend modules

| Module | Role |
|--------|------|
| `spec.py`    | object ⟷ JSON-able spec dict for every structured primitive; `DesignSpec`/`EnvSpec`/`FieldRef` document model |
| `builder.py` | spec → `Scenario` resources plus static `EnvConfig`; resolves field refs and `"module:attr"` code refs |
| `nav.py`     | navdb query layer (waypoints/airports/runways) scoped to a window, plus global `search` |
| `preview.py` | spec → renderable geometry + seeded sampled traffic for the map |
| `catalog.py` | palette of available footprints / bands / fields / aircraft for the GUI |
| `codegen.py` | spec → runnable task package (design.json + scenario/env/task scaffolding) |
| `runner.py`  | code-gen the spec to a temp package and launch it in a live driver subprocess |
| `store.py`   | on-disk persistence of named designs (`~/.bluesky_sandbox/designs`) |
| `api.py`     | FastAPI server tying it together + static frontend serving |

### Frontend (`web/`)

Vite + React + TypeScript. Both tabs project the same `DesignSpec`; edits
validate and preview against the backend live.

**Map** tab (MapLibre GL over an OSM basemap, deck.gl for 3D primitives):
- **3D see-through bounds** — airspace and queryable regions render as deck.gl
  **wireframe boxes** (no opaque faces), so you see through them. The wireframe's
  top and side edges follow the altitude band per vertex, so varying bands
  (linear/radial/vertex) slope correctly for any footprint shape. A selected
  element is drawn in its **canonical** (unrotated) frame so the edit handles line
  up even while a rotation group rotates the rest of the preview, and each corner
  handle gets a **dotted drop-line** to the shape's base when it sits above the
  ground (elevated bounds).
- **Aircraft & waypoint overlays** — sampled aircraft are drawn as dots, while
  selected task waypoints are drawn as diamond markers with optional reach
  rings so they remain distinct from navdb fixes and stay above the basemap.
- **Composition** — footprints compose via a `boolean` shape (union / intersection
  / difference) with two nested sub-shapes, to any depth.
- **Properties panel** — structured forms for the whole design, grouped into
  **subtabs** (Geometry / Fields / Config / Sampling). Every dropdown is the
  same searchable, category-grouped picker used by the fields/actions lists
  (one `Picker` component, uniform across panel, toolbar, and modals):
  - **All bounds primitives** — every footprint (box / disk / sector /
    annular_sector / polygon / boolean) and altitude band (constant / linear /
    radial / vertex), assignable as the airspace singleton, queryable
    regions/waypoints, or spawn regions. A waypoint can instead **sample its
    position** from a region each episode (toggle "sample position within a
    region"); its static schema-support position is the region centre, and
    reseeding the map previews the draws.
  - **Named bounds are the single source of geometry.** Every shape —
    airspace, query-regions, spawn-regions, and waypoint sample areas — is
    defined once in the **Bounds** section and referenced by name
    (`{"ref": "<bounds>"}`). Define a bounds once and reuse it across consumers;
    editing it (numerically, or by dragging its resize/move/rotate handles on the
    map) updates every consumer that references it. A bounds referenced by more
    than one element shows a **"shared by N: …"** line naming the elements that
    use it, so shared edits are obvious. Specs with inline geometry are
    migrated into the Bounds table automatically on load. (In the spec/codegen
    these live under `regions` / `REGIONS`; the GUI labels them "Bounds" to avoid
    colliding with query-*regions*.)
  - **Observations & Actions** — add/remove built-in fields from the catalog,
    or reference a **custom field by import path** (`module:ClassName`). Each
    chip shows a compact **signature summary** of how it's configured (selected
    queryable, constructor kwargs, normalizer). Intruder observations are added
    through the same picker as any other; the config modal then offers a
    **"relative to ownship"** checkbox that turns the field into an
    intruder-relative pair field via `.relative_to_own()` (emitted as
    `obs.AltFt().relative_to_own()`, with the normalizer stored as a transform
    kwarg). The same modal offers a **"frame stack"** checkbox with a *depth*,
    which emits `obs.ConflictTlosS().stacked(depth=3)` — the live field plus
    `depth-1` `.lagged(steps=k)` copies, each on the live field's own bounds and
    normalizer, so a stacked channel needs no separate calibration. The chip
    carries a `×N` badge, and one entry expands to `N` channels in the
    observation (`EnvConfig` flattens one level). Frame stacking and "relative to
    ownship" are mutually exclusive — a `FieldRef` carries one transform slot —
    so the two checkboxes disable each other.
    A one-line **strategy summary** under each list restates the chosen
    features as a sentence (e.g. "Each agent observes 3 ownship features, plus 4
    features (1 relative to ownship, 1 frame-stacked) for each nearby
    intruder.").
  - **Configuration** — allowed aircraft, `dt` / `simdt`, performance model, and a
    **conflict-detection** group: detection method (`CDMETHOD`), resolution
    method (`RESO`, default *off* so the agent resolves conflicts itself), and
    optional protected-zone radius (`ZONER`) / height (`ZONEDH`) / look-ahead
    (`DTLOOK`) — unset ones use BlueSky's defaults. These are re-applied every
    reset (BlueSky doesn't keep them across `sim.reset()`).
  - **Sampling** — a live readout of one drawn episode (counts + per-aircraft
    type/altitude/speed), with reseed.
- **Nav data** — airports and waypoints from navdb scoped to the airspace
  window; **hover** any fix/airport/selected-waypoint for its name; selected
  features carry persistent name labels.
- **Search** — a global navdb search box flies the map to a fix/airport or adds
  a fix as a waypoint queryable.

  - **Variable-time spawning** — each spawn region has a `spawn_time` (fixed or
    a range), so aircraft arrive staggered.
  - **Spawn heading** — each spawn region sets its **initial heading** via an
    `hdg_deg` value (fixed / range / scipy), defaulting to the full `0-360` range
    (i.e. uniform, as before). Ranges **wrap through north** (e.g. `350-10` is a
    20° arc). **Selecting a spawn region on the map** draws a green **pie wedge**
    out of its centre spanning that heading range (a full disk when `0-360`),
    with a mean-direction arrow, so you can see which way aircraft launch.
  - **Sampled values (any distribution)** — every sampled scalar (spawn count /
    speed / `spawn_time`, each rotation-group angle, and each **waypoint
    constraint/target** — target alt/speed, reach radius, alt/speed tolerances)
    uses a common editor:
    **fixed**, a uniform **range**, or **any `scipy.stats` distribution**. The
    distribution is chosen from a searchable list (each entry hinting its
    signature, e.g. `truncnorm(a, b, loc=0, scale=1)`), and its parameters are
    introspected from scipy and shown as **named fields** — so kwargs can't be
    mistyped. The build validates support (e.g. an unbounded `norm` is rejected
    where bounded support is required).
  - **Route assignment** — a regions × routes matrix (in the route library)
    assigns a named route to each spawn region at a glance; fixed-waypoint
    ordering and route-sampling weights stay editable per region card.
  - **Editable env hooks** — the Code tab's "env hooks" panel lists the
    environment's `@overridable` hooks (discovered by introspection, not a
    hard-coded list); override any by editing just the method body. **reward /
    terminated / truncated are uniform hooks** too — always present (default
    `0.0` / never done), edited in the same panel; there is no separate `task.py`
    or config reward function. Only customised hooks are emitted in the generated
    `env.py` — the rest inherit the base, so there's no `super()` boilerplate.
    Old designs (reward in `task.py`) migrate into hooks automatically on load.
    The override picker is **grouped by category** (task outcome / definitions /
    lifecycle events) and each hook carries introspected **metadata** (category,
    params, return type, the base default `return`) shown as a hint beside the
    editor — none of it hard-coded. Scaffolds use that metadata (e.g.
    `define_agent_context` is seeded with a starter that reads
    `self.episode_queryables`). Completions surface the spec's queryable names
    and temporal queryable tracking is inferred from selected field requirements.
    They also expose the `AgentStepContext`
    surface — `context.acid` / `context.acidx` / `context.data` /
    `context.separation` / `context.query("name")` — and `self.episode_queryables`.
  - **Composable routes (subroutes)** — a route step can be a waypoint *or*
    another route (`{"route": name}`), expanded inline at spawn (cycle-guarded).
    Build a shared `approach` segment once and reuse it across `arrival`s; the
    step picker offers both waypoints and subroutes, and rename/delete cascade
    through references.
  - **Per-aircraft sampled waypoints** — a sampled waypoint's "sample" can be
    *per episode* (one shared position drawn each reset) or *per aircraft* (each
    aircraft draws its **own** target from the region at spawn). A per-aircraft
    waypoint is a normal named queryable: `context.query("goal")` returns *this*
    aircraft's target, so all the usual `Waypoint*` observation fields work by
    name. Reference it in a spawn route to make aircraft fly to their own target.
    The map draws each sampled aircraft's target as a ring with a line from the
    aircraft (reseed to see the spread); a per-aircraft waypoint has **no single
    static marker** (it would be misleading) — only its sample region and the
    per-aircraft rings. Discrete "random one of N" assignment is just
    per-aircraft route sampling (a categorical over single-waypoint routes).
  - **Field parametrisation** — edit any built-in field's constructor kwargs
    (e.g. `low`/`high`) inline; hover a field for its docstring.
  - **Randomization** — per-episode **rotation groups**. A group's members are
    **bounds** (named regions): rotating a bounds rotates every element (airspace
    / queryable / spawn region) that references it, by an angle sampled from a
    range about the members' centre. Groups **nest**: put one group *inside*
    another and its bounds are spun locally first, then carried by the parent's
    rotation ("rotation in rotation", cycle-guarded). The legacy whole-airspace
    rotation migrates into a single all-bounds group on load. Reseed on the map
    to preview different draws.
- **Sampling** — the map shows one sampled episode (aircraft locations +
  rotation); **reseed**/**reset** redraws it.

**Code** tab (Monaco): shows the task's **code structure** as Python, not JSON.
The whole design is emitted as `design.py` (constructing the bounds / queryables /
spawn / fields directly); editable helper modules such as `custom_fields.py`
live in `spec.code`; `scenario.py` / `env.py` / `__main__.py` are read-only.
Editing reward/termination hooks gets **completions** for the design's
observation/action/queryable names. (`spec.json` is still available for raw
edits.)

**Custom fields** — add an observation/action field by import path, or
**+ scaffold** a starter class into `custom_fields.py` and edit it in the Code
tab; remove it from the chip list. The builder execs these modules in-process for
live validation.

**Generate task** (toolbar): turns the current design into a runnable task
package — `design.py` plus your code modules, with references package-qualified,
so the result imports and builds standalone. **Download .zip** saves the whole
project directory.

**Run** (toolbar): launches the *actual* environment in a real driver window —
pick **pygame** (2D multi-view), **panda3d** (3D), or **qtgl** (BlueSky native)
and press ▶ Run. Under the hood the design is code-generated into a throwaway
package (identical to "Generate task", so hooks and custom fields are included)
and run in a detached **subprocess** that constructs the env, resets, and
realtime-steps with sampled actions while rendering. The window opens on the
machine hosting the designer (your local machine), and BlueSky's process-global
state stays isolated from the API. This complements the Map tab's static
*preview* (geometry + one sampled episode) with a live rollout. The Run control
opens a small **dialog** for the driver, its view layout, and route options:

- **all routes** — overlays the design's **defined routes** (resolved once per
  episode from the route definitions + waypoint positions — not recomputed per
  aircraft per step), in addition to the tracked aircraft's live route. All three
  drivers support it: pygame/panda3d draw the overlay per frame; qtgl stacks
  `POLYLINE` shapes (added/removed as the toggle flips). In a running driver,
  press **`O`** to toggle it live.
- **track aircraft** — by default no aircraft is selected (so no live route shows
  until you click one; clicking empty deselects). Enable this to always keep an
  aircraft tracked (the first live one) — its route + HUD info follow it, handy
  for RL eval videos. Settable directly as `driver.auto_track` from eval code.

When the design is invalid, the reason is shown in the (red) status bar.

### Accessing context in hook code

The reward / terminated / truncated hooks (and the other agent hooks) receive an
`AgentStepContext` (`bluesky_sandbox.types`) bound to one aircraft for the step:

```python
# reward hook body (edited in the Code tab's "env hooks" panel)
goal = context.query("goal")
return 1.0 if goal.current.inside else -0.01

# terminated hook body
return context.query("goal").current.inside
```

`context.query("name")` evaluates a design queryable for the current aircraft
(a `QueryRegion` returns `.current`, `.step`, and `.time`; a `Waypoint` returns
`.target`, `.current`, `.route`, `.step`, and `.time`); `context.queryable("name")` returns
the object itself. Other fields: `context.acid` (callsign), `context.acidx`
(BlueSky traffic index), `context.separation` (conflict / LoS info), and
`context.data` — your own per-aircraft payload.

You **define** `context.data` by overriding the `define_agent_context(acid,
acidx)` hook (its return value becomes `context.data`); everything else on the
context is wired from the episode's queryables automatically. The Code editor
offers completions for your design's queryable / observation / action names and
this context surface.

### How code plugs in

Reward/termination/truncation functions and custom observation/action field
classes are *logic*, so they live as Python in `spec.code` (edited in the Code
tab) and are referenced from the spec by import string (`task:reward`,
`custom_fields:MyField`). The builder registers these modules in-process for live
validation, and `codegen` writes them into the generated package.

## Running

### One-time: build the frontend

```bash
cd bluesky_sandbox/designer/web
npm install
npm run build          # emits web/dist/, served by the API at /
```

### Production-ish (API serves the built frontend)

```bash
python -m bluesky_sandbox.designer --port 8765
# open http://127.0.0.1:8765
```

### Dev (hot-reloading frontend + API)

```bash
# terminal 1 — API
python -m bluesky_sandbox.designer --port 8765 --reload
# terminal 2 — Vite dev server (proxies /api to :8765)
cd bluesky_sandbox/designer/web && npm run dev
# open http://127.0.0.1:5173
```

## Tests

No pytest required — the suites run standalone:

```bash
python -m bluesky_sandbox.designer.tests.test_designer   # spec round-trip, build, nav
python -m bluesky_sandbox.designer.tests.test_api        # FastAPI endpoints
```

## API surface

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/api/health` | liveness |
| GET  | `/api/catalog` | GUI palettes |
| POST | `/api/nav/features` | navdb features in a bounds window |
| GET  | `/api/nav/search?q=` | global fix/airport search (prefix-ranked) |
| GET  | `/api/nav/waypoint/{ident}` | resolve a fix |
| GET  | `/api/nav/airport/{icao}` | resolve an airport (+runways) |
| POST | `/api/spec/validate` | build + report ok/errors + summary |
| POST | `/api/spec/preview` | renderable geometry + sampled traffic |
| POST | `/api/spec/run` | launch the design in a live driver window (subprocess) |
| POST | `/api/spec/generate` | generate a runnable task package from the spec |
| GET/PUT/DELETE | `/api/specs[/{name}]` | persist named designs |
