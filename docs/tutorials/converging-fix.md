# Your first task in the designer

```{figure} ../media/tutorials/converging-fix.gif
:alt: Eight arrivals fly from a ring 25 to 35 nm out directly to the fix GIBBZ; aircraft that meet at the fix turn red
:width: 620px
:align: center

Eight arrivals appear on a ring around the fix GIBBZ and fly directly to it at
10,000 ft. Their paths meet at the fix. Red aircraft have lost separation:
with no heading control, they arrive at the fix together.
```

This tutorial builds a complete task with only the controls of the
[Environment Designer](../designer.md). You do not write JSON, and the only code
that you write is the reward and the two done conditions.

The task is near Washington Dulles. Eight arrivals appear 25 to 35 nm from the
fix GIBBZ and fly directly to it. All of them fly at 10,000 ft, so their paths
converge at the fix. The agent controls each arrival. It gives a heading
offset from the direct course, so that the aircraft keep separation and still
reach the fix.

**You will learn to:**

- find a fix in the navigation database and make it a waypoint
- draw the airspace and a spawn region on the map
- set the traffic of a spawn region: count, types, level, speed and spawn time
- build a route and assign it to a spawn region
- select observation and action fields, their bounds and their normalizers
- write the reward, termination and truncation hooks
- look at a sampled episode, fly the design, and generate the task package

The finished design is
[`designs/converging_fix.json`](designs/converging_fix.json). To open it, select
**Import JSON file…** in the **Project…** menu.

## Before you start

Install and start the designer, then open `http://localhost:8765`:

```bash
pip install "bluesky-sandbox[designer,pygame] @ git+https://github.com/ArsyiAziz/bluesky-sandbox.git"
python -m bluesky_sandbox.ui.designer --port 8765
```

The toolbar at the top of the page has these controls:

| Control | What it does |
|---|---|
| **Map**, **Route**, **Spaces**, **Config**, **Code**, **Metadata** | The tabs. This tutorial uses them from left to right. |
| Undo and redo | Undo or redo the last edit (⌘/Ctrl+Z, ⌘/Ctrl+⇧+Z). |
| Project name and **Save** | The name of the design. **Save** keeps the design in `~/.bluesky_sandbox/designs/`. |
| **Project…** | Import a `design.json` file, start a new project, or delete the open one. |
| **▶ Run** | Fly the design in a renderer window. |
| **Generate task** | Write the design as a Python package. |
| **✓ Valid** | The design builds. When it does not build, the status bar shows the reason in red. |

## 1. Name the project

Click the project name (**untitled**) in the toolbar, type `converging_fix` and
press Enter. This is the name of the saved design and of the generated package.

## 2. The fix

On the **Map** tab, type `GIBBZ` in the search box (**search fixes /
airports…**) at the top left. The list shows **FIX GIBBZ**.

1. Click the name **GIBBZ**. The map moves to the fix.
2. Click the **+** button beside the name. The designer adds the waypoint
   `gibbz` to the **Waypoints** list on the right.

```{image} ../media/tutorials/designer/01-search.jpg
:alt: The search box with the result FIX GIBBZ and its + button
:width: 620px
:align: center
```

```{image} ../media/tutorials/designer/02-waypoint.png
:alt: The element list with the waypoint gibbz under Waypoints
:width: 300px
:align: center
```

A waypoint is a named point that the task can ask about. An aircraft that
passes within the reach radius (1 nm by default) of `gibbz` has reached it. To
change the reach radius, click `gibbz` in the list.

```{tip}
New elements appear at the center of the map. Move the map to the fix first,
then add the airspace and the spawn region in steps 3 and 4. Their centers are
then on the fix.
```

## 3. The airspace

1. Under **Shapes** in the right panel, click **+ shape**. A new shape named
   `shape` opens in the panel.
2. Change its name to `sector`.
3. Select **the airspace**.
4. Under **Footprint**, set **type** to **disk**. Expand **coordinates**: the
   center is the fix. Set **radius nm** to `40`.
5. Under **Altitude**, set **altitude** to **constant**, **min ft** to `0`
   and **max ft** to `20000`.
6. Click the zoom out button (**−**) on the map two times to see all of the
   airspace.

```{image} ../media/tutorials/designer/03-airspace.jpg
:alt: The 40 nm airspace disk around GIBBZ, with its settings in the right panel
:width: 620px
:align: center
```

The airspace is the volume of the task. The renderers draw it, and in step 7 a
hook ends the episode of an aircraft that flies out of it. You can also move or
resize a selected shape with the handles on the map.

Click **← all elements** to go back to the list.

## 4. The spawn region

1. Under **Spawn**, click **+ spawn**. The designer adds the spawn region
   `SPAWN`, with its own shape, and opens it.
2. Change its name to `ring`.
3. In the **Shape** section, set **type** to **annular_sector**. Expand
   **coordinates** and set **inner nm** to `25`, **outer nm** to `35` and
   **half angle°** to `180`. A half angle of 180° makes a full ring.

```{image} ../media/tutorials/designer/04-spawn-shape.jpg
:alt: The green spawn ring between 25 and 35 nm from GIBBZ, inside the blue airspace
:width: 620px
:align: center
```

The **Traffic** section sets which aircraft appear, and how:

| Setting | Value | Why |
|---|---|---|
| **aircraft types** | A320, B738 (the default) | Half of the aircraft are of each type. Use **+ add a type to the mix…** for more types. |
| **count** | **fixed**, `8` | Eight aircraft in each episode. |
| **conflict-free spawn** | **always conflict-free** | No aircraft appears in conflict with another. All conflicts then come from the converging paths. |
| **alt ft** | selected, **fixed**, `10000` | All aircraft at the same level. Their conflicts are horizontal, and a heading action can solve them. |
| **speed kt** | **range (uniform)**, `250` to `280` | |
| **heading °** | `0` to `360` (the default) | The route turns each aircraft toward the fix after it appears. |
| **spawn time s** | **range (uniform)**, `0` to `900` | The aircraft appear at different times in the first 15 minutes. |

```{image} ../media/tutorials/designer/05-traffic.png
:alt: The Traffic section of the spawn region
:width: 300px
:align: center
```

When a spawn region is selected, the map shows its heading range as a green
wedge from its center. A range of 0° to 360° is a full disk.

Click **← all elements**.

## 5. The route

1. Open the **Route** tab and click **+ route**.
2. Change the name `route_1` to `direct`.
3. In **add to end**, select `gibbz`.

The graph shows the route from the spawn region `ring` to `gibbz`, and the map
below it shows the route.

```{image} ../media/tutorials/designer/06-route.jpg
:alt: The Route tab with the route direct, from ring to gibbz
:width: 620px
:align: center
```

**from** only sets where the preview starts. It does not give the route to the
spawn region. To do that:

1. Open the **Map** tab and click **route library** under **Routes**.
2. Under **Route assignment**, set `ring` to `direct`.

```{image} ../media/tutorials/designer/07-route-assign.png
:alt: Route assignment, with ring set to direct
:width: 300px
:align: center
```

Each aircraft of `ring` now gets the flight plan `direct`, and flies it in
LNAV. You can also set the route on the spawn region itself, in its **Route**
section (**route override**).

Click **← all elements**.

## 6. Observations and the action

Open the **Spaces** tab. The left side lists the fields. The right side shows
each field's raw range, its normalizer, and the values that the policy sees.

A new design starts with some fields. Replace them:

- To remove a field, move the pointer over its row and click **✕**.
- To add a field, type its name in the **Add an … observation…** or **Add an
  action…** box, then select it in the list.
- To set a field's bounds or normalizer, click its row. Set **low** and
  **high** under **Constructor**, and **strategy** under **Normalization**.
  Then click **done**.

Remove `LatDeg`, `LonDeg`, `AltFt`, `DistToOwnNm`, `HdgDeg` and `SpdKts`, and
add these fields:

| Part | Field | low, high | Normalizer | What it tells the policy |
|---|---|---|---|---|
| ownship | `ActiveRouteWaypointDistanceNm` | 0, `40` | `MinMaxNormalizer` | How far the fix is. |
| ownship | `ActiveRouteWaypointTrackErrorDeg` | default | `SymmetricNormalizer` | The angle between my track and the direct course to the fix. |
| intruder | `DistToOwnNm` | 0, `40` | `MinMaxNormalizer` | How far the intruder is. |
| intruder | `BrgFromOwnRelTrkDeg` | default | `SymmetricNormalizer` | Where the intruder is, relative to my track. 0° is ahead. |
| intruder | `ClosingRateKts` | `-600`, `600` | `SymmetricNormalizer` | How fast we close. A positive value means that we close. |
| action | `ActiveRouteWaypointHdgDeltaDeg` | `-45`, `45` | `SymmetricNormalizer` | My heading is the bearing to the fix plus this value. |

```{image} ../media/tutorials/designer/09-field-editor.png
:alt: The field editor of ActiveRouteWaypointHdgDeltaDeg, with low -45, high 45 and SymmetricNormalizer
:width: 620px
:align: center
```

With `SymmetricNormalizer`, the policy gives a value from -1 to 1, and the
environment scales it to -45° to 45°. An action of 0 flies directly to the fix,
as LNAV does.

```{image} ../media/tutorials/designer/08-spaces.png
:alt: The Spaces tab with two ownship fields, three intruder fields and one action
:width: 620px
:align: center
```

The intruder fields give one row for each other aircraft. Keep **intruder
scale** at **ownship's envelope**.

## 7. Reward and done conditions

Open the **Code** tab. Under **env hooks**, the hooks `reward`, `terminated` and
`truncated` are always present. Click a hook to edit it. The editor shows the
function signature above the code: you write only the body. The **Available
here** panel on the right lists the names that you can use, and the editor
completes them.

```python
# reward
# A small cost for every step, so a long detour costs more than a short one.
reward = -0.01
if context.separation.los:
    reward -= 1.0
if context.query("gibbz").current.satisfied:
    reward += 10.0
return reward
```

```python
# terminated
return context.query("gibbz").current.satisfied
```

```python
# truncated
return not context.airspace.current.inside
```

`context` is the step of one aircraft. `context.separation.los` is true while
the aircraft has lost separation. `context.query("gibbz").current.satisfied`
is true when the aircraft reaches the fix. `context.airspace.current.inside`
is false when the aircraft is out of the airspace from step 3.

```{image} ../media/tutorials/designer/10-code.png
:alt: The reward hook in the Code tab
:width: 620px
:align: center
```

## 8. The step length

Open the **Config** tab and set **step (dt)** to `5`. The policy then gives a
new action every 5 seconds.

Keep the other settings. **Conflict detection** uses a protected zone of 5 nm
and 1,000 ft, and `context.separation` uses the same zone. **resolution** is
**off (agent resolves)**: BlueSky does not resolve conflicts, so the policy
must do it.

```{image} ../media/tutorials/designer/11-config.png
:alt: The Config tab with step (dt) set to 5 seconds
:width: 620px
:align: center
```

## 9. Save and look at an episode

Click **Save** in the toolbar. The status bar shows `saved as converging_fix`.

Open the **Map** tab and select the **aircraft** layer. The map shows the
aircraft of one sampled episode, with the type, level, speed and heading of
each one. Use the **‹** and **›** buttons under **Episode** to see the
episodes of other seeds.

```{image} ../media/tutorials/designer/12-episode.jpg
:alt: One sampled episode, with eight aircraft on the ring around GIBBZ
:width: 620px
:align: center
```

## 10. Fly the design

Click **▶ Run** in the toolbar. In the dialog, select the renderer (**driver**)
and its views. Under **action**, select **0 action**, then click **▶ Run**.
A window opens on the computer that runs the designer, and the aircraft fly
with a policy that always gives 0.

```{image} ../media/tutorials/designer/13-run.png
:alt: The Run dialog with the pygame driver and 0 action selected
:width: 620px
:align: center
```

**random** gives a random action at each step. Use it to see what the action
does.

## 11. Generate the task

Click **Generate task**. The dialog shows the files of the package. Keep the
**Plain** template and click **Download .zip**. The **RL** and **SB3**
templates add training scripts (see the
[SB3 training tutorial](train-sb3.md)).

```{image} ../media/tutorials/designer/14-generate.png
:alt: The Generate task dialog with the Plain template and the files of the package
:width: 620px
:align: center
```

Unzip the package and fly it with a policy that does nothing:

```python
import numpy as np
from converging_fix import Env

env = Env(render_mode=None)
env.reset(seed=1)
total = 0.0
while not env.episode_done:
    _, rewards, *_ = env.step({agent: np.zeros(1, np.float32) for agent in env.agents})
    total += sum(rewards.values())
print(total)
```

With seed 1, all eight aircraft reach GIBBZ in 261 steps. Four of them lose
separation near the fix, and the total return is 37.0. With seeds 0 and 2, two
and four aircraft lose separation. The policy must turn the aircraft before
they come near the fix, so that they arrive one after the other.

## Things to try

- Make the traffic denser: set **count** to `12`, or set **spawn time s** to
  `0` to `600`.
- Add a second spawn region with **uncooperative traffic** selected. Its
  aircraft fly on autopilot. The policy sees them as intruders but cannot
  command them.
- Add conflict fields to the intruders, for example `ConflictTcpaS` and
  `ConflictHorizontalDistAtCpaNm`, so that the policy sees the predicted
  closest point of approach.
- Open `spec.json` under **Design** in the **Code** tab to see the design that
  you made, as JSON. The other tutorials show each step in this form.
- Generate the task with the **SB3** template and train a policy, as in the
  [SB3 training tutorial](train-sb3.md).

## Next: more designer features

The next tutorials use more features of the designer, and show each step as
the `design.json` that it writes:

- [Arrival through moving weather](storm-arrival.md): generated shapes,
  motion, heading clearances.
- [Final approach spacing](final-approach.md): route constraints, VNAV, speed
  control at a merge.
- [En route sectors](enroute-sectors.md): partition generators, sampled
  traffic, conflict detection settings.
