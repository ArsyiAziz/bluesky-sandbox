// Which bounds the map draws. Elements draw the bounds they are; a bounds no
// element draws - standalone, a waypoint's sample area or anchor, a group's -
// shows only with the "bounds" layer on, or when a selection brings it in:
//   - own: drawn as itself - the selected bounds, everything a selected group
//     holds, a selected waypoint's sample area and anchor;
//   - context: what those (or the selected element's bounds) are drawn,
//     placed or moved by - faint and dashed, for where they may go.
import type { SpecDict } from "../api";
import { dependenciesOf, refsOfLayers } from "../shapeGraph";
import { shapesInGroup, groupsOf } from "../groupTree";
import type { EditTarget } from "./types";

export type ShapesInView = { own: Set<string>; context: Set<string> };

// The bounds name a ref is of: `sectors.0` is `sectors`'s.
const baseOf = (spec: SpecDict, ref: unknown): string | null => {
  if (typeof ref !== "string") return null;
  if (spec.shapes?.[ref]) return ref;
  const base = ref.replace(/\.\d+$/, "");
  return spec.shapes?.[base] ? base : null;
};

export function shapesInView(spec: SpecDict | null, target: EditTarget | null): ShapesInView {
  const own = new Set<string>();
  const context = new Set<string>();
  if (!spec || !target) return { own, context };
  // Bounds whose dependencies to show, besides `own`.
  const users: string[] = [];
  const add = (set: Set<string>, ref: unknown) => {
    const name = baseOf(spec, ref);
    if (name) set.add(name);
  };
  if (target.scope === "region") {
    add(own, target.name);
  } else if (target.scope === "group") {
    for (const name of shapesInGroup(spec, target.id)) own.add(name);
    const group = groupsOf(spec).find((g) => g.id === target.id);
    for (const name of refsOfLayers(spec, group?.motion ?? [])) context.add(name);
  } else if (target.scope === "queryable") {
    const q = spec.queryables?.[target.name];
    add(own, q?.sample?.ref);
    add(own, q?.anchor);
    const b = baseOf(spec, q?.shape?.ref);
    if (b) users.push(b);
  } else if (target.scope === "airspace") {
    const b = baseOf(spec, spec.airspace?.ref);
    if (b) users.push(b);
  } else if (target.scope === "spawn") {
    const b = baseOf(spec, spec.spawn?.regions?.[target.index]?.shape?.ref);
    if (b) users.push(b);
  }
  for (const name of [...own, ...users]) {
    for (const dep of dependenciesOf(spec, name)) if (!own.has(dep)) context.add(dep);
  }
  return { own, context };
}
