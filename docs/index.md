# Bluesky Sandbox

Highly modular Air Traffic Control environments for machine learning.

Bluesky Sandbox is a [PettingZoo](https://pettingzoo.farama.org/) framework powered by the
open-source [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) ATM simulator. It turns
Air Traffic Control problems — sector conflict resolution, arrival sequencing, trajectory
optimization — into customizable multi-agent environments with minimal setup.

:::{warning}
Bluesky Sandbox is in active development and not yet on PyPI. Install directly from git —
see [Installation](installation.md).
:::

```{figure} media/point-merge.gif
:alt: Twenty aircraft sequencing onto a point merge, with live ETA and distance-to-merge tables
:width: 620px
:align: center

A point-merge task: twenty arrivals sequenced onto a single merge point, ETA and
distance-to-merge tracked per callsign.
```

## What makes it different

Rather than picking from a fixed catalogue of prebuilt environments, you *design* the task:
the airspace, the spawn regions, the routes, and the observation and action fields your
agents see. The [Environment Designer](designer.md) allows you to arange these components in your browser and our system will generate a clean Python package for you to use.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Native multi-agent
Built on the PettingZoo `ParallelEnv` standard from the ground up.
:::

:::{grid-item-card} Low boilerplate
Arange airspaces, routes and spawn sectors graphically, then export a full task package.
:::

:::{grid-item-card} Multi-backend rendering
Lightweight Pygame, 3D Panda3D, or BlueSky's native QtGL radar visualization.
:::

:::{grid-item-card} Composable interface
Observation and action fields, wrappers and normalizers combine instead of forking.
:::

::::

## Getting started

```{toctree}
:maxdepth: 2
:caption: Guide

installation
quickstart
designer
rendering
diagnostics
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Project

GitHub repository <https://github.com/ArsyiAziz/bluesky-sandbox>
Issue tracker <https://github.com/ArsyiAziz/bluesky-sandbox/issues>
```

## Related projects

Bluesky Sandbox is a sister project to
[BlueSky-Gym](https://github.com/TUDelft-CNS-ATM/bluesky-gym).

| | BlueSky-Gym | Bluesky Sandbox |
|---|---|---|
| Interface | Gymnasium, single-agent | PettingZoo `ParallelEnv`, multi-agent |
| Environments | A curated, prebuilt set — `gym.make('MergeEnv-v0')` | You design your own |
| Best for | Benchmarking against a standard task set | Building a task that does not exist yet |

## License

Distributed under the MIT License.
