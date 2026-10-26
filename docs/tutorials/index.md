# Tutorials

Each tutorial builds one complete task in the [Environment Designer](../designer.md),
from an empty map to a package you can train on. Each one also shows a different
set of designer features. The last tutorial trains a policy on one of these
tasks with Stable-Baselines3.

## Start here

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Your first task in the designer
:link: converging-fix
:link-type: doc
:img-top: ../media/tutorials/converging-fix.png

Arrivals converge on one fix near Washington Dulles. A policy gives heading
offsets to keep them separated. Every step is a click in the designer, with a
screenshot.

*Search, shapes, spawn traffic, routes, fields, hooks, Run, Generate task.*
:::

::::

## Designer features

The next three tutorials show each step as the `design.json` that the designer
writes, so that you can see how each setting is stored.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Arrival through moving weather
:link: storm-arrival
:link-type: doc
:img-top: ../media/tutorials/storm-arrival.png

Arrivals fly the CAMRN arrival into New York while two storm cells drift and
change shape. A policy issues temporary heading clearances around them.

*Named fixes, routes, generated shapes, motion, clearances.*
:::

:::{grid-item-card} Final approach spacing
:link: final-approach
:link-type: doc
:img-top: ../media/tutorials/final-approach.png

Two arrival streams merge onto the final approach for JFK runway 31R. A policy
adjusts speed to space the aircraft at the merge.

*Route constraints, VNAV, intruder fix-arrival fields, lifecycle hooks.*
:::

:::{grid-item-card} En route sectors
:link: enroute-sectors
:link-type: doc
:img-top: ../media/tutorials/enroute-sectors.png

Four crossing flows pass through a high-altitude center that is cut into new
sectors in each episode. A policy resolves conflicts with vectors.

*Partition generators, sampled traffic, conflict detection settings.*
:::

::::

## Train a policy

::::{grid} 1
:gutter: 3

:::{grid-item-card} Train a speed policy with Stable-Baselines3
:link: train-sb3
:link-type: doc

Generate the final approach task with the SB3 template, train one PPO policy
that every arrival shares, and compare it with a policy that does nothing.

*SB3 template, parallel processes, TensorBoard logs, evaluation, speed choices.*
:::

::::

## How to use the tutorials

Each tutorial comes with its finished design in `docs/tutorials/designs/`. You can
build the design step by step in the designer, or you can load the finished file
and read the tutorial alongside it.

To load a finished design in the designer, select **Import JSON file…** in the
**Project…** menu of the toolbar.

To work without the browser, use the `bluesky-sandbox` command. It reads the same
`design.json`:

```bash
bluesky-sandbox design check   docs/tutorials/designs/storm_arrival.json   # does it build?
bluesky-sandbox design preview docs/tutorials/designs/storm_arrival.json --seed 0
bluesky-sandbox design build   docs/tutorials/designs/storm_arrival.json --out tasks/
bluesky-sandbox design test    docs/tutorials/designs/storm_arrival.json   # field checks
```

`check` reports every problem in the design and exits with 1 if the design does
not build. `preview` lists the aircraft of one sampled episode: the spawn time,
type, position, level, speed and route of each one. `build` writes the task
package. `test` runs the field checks that are described in
[Checks](../api/checks.md).

## The common pattern

All the tasks are built in the same order. The designer tabs follow this order:

1. **Map**: shapes, waypoints, spawn regions. Everything on the map is a
   named *shape*. The airspace, spawn regions and query regions refer to shapes
   by name, so one edit to a shape updates every element that refers to it.
2. **Route**: the route library, and which spawn region flies which route.
3. **Spaces**: the observation and action fields of each agent.
4. **Code**: the reward, termination and truncation hooks, and any other hook
   that the task needs.
5. **Config**: the step length, conflict detection, wind and performance model.

Then **▶ Run** flies the design on a renderer, and **Generate task** writes the
package.

:::{tip}
In every tutorial, first fly the design with a policy that does nothing. If the
aircraft do not complete their routes without your policy, a learning agent
cannot complete them either. Find out why before you train. Each tutorial
includes such a run and its result.
:::

```{toctree}
:hidden:

converging-fix
storm-arrival
final-approach
enroute-sectors
train-sb3
```
