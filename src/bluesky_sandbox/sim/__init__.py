"""The simulated world and the episodes drawn from it.

Airspace volumes (:mod:`.bounds`), pairwise conflict geometry
(:mod:`.geometry`), aircraft capability (:mod:`.performance`), the queryable
resources a task scores against (:mod:`.queryables`), and the sampling stack
that produces an episode (:mod:`.sampling`, :mod:`.spawn`, :mod:`.scenario`).

Deliberately importless: a group ``__init__`` that pulled its subpackages in
would make a layer-0 import like ``sim.sampling`` drag ``sim.scenario``
behind it, which is a cycle. Import the subpackage you want.
"""
