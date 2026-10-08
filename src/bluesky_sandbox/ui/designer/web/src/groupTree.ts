// Groups as containers of bounds: a tree. A group holds bounds (`members`) and
// other groups (whose `parent` it is); a bounds is in at most one group, and
// whatever uses a bounds - an element, a waypoint anchored to it - moves with
// it. The operations here are the panel's drags (and their keyboard twins):
// each mutates the spec in place, then prunes groups left empty.
import type { SpecDict } from "./api";
import { newGroupId } from "./specHelpers";

export type Group = SpecDict & { id: string; name?: string; members?: string[]; parent?: string | null };

export const groupsOf = (spec: SpecDict | null): Group[] => (spec?.transform?.groups ?? []) as Group[];

// The group holding bounds `name`, if any.
export const groupOfShape = (spec: SpecDict | null, name: string): Group | undefined =>
  groupsOf(spec).find((g) => (g.members ?? []).includes(name));

// The groups directly inside group `id` (or, with null, at the top).
export const childGroups = (spec: SpecDict | null, id: string | null): Group[] =>
  groupsOf(spec).filter((g) => (g.parent ?? null) === id);

// Whether group `inner` is `outer` or inside it, at any depth.
export function isWithin(spec: SpecDict | null, inner: string, outer: string): boolean {
  const byId = new Map(groupsOf(spec).map((g) => [g.id, g]));
  let at: string | null | undefined = inner;
  for (let i = 0; at && i < 64; i++) {
    if (at === outer) return true;
    at = byId.get(at)?.parent;
  }
  return false;
}

// Every bounds in group `id`, its subgroups' included.
export function shapesInGroup(spec: SpecDict | null, id: string): string[] {
  const out: string[] = [];
  for (const g of groupsOf(spec)) {
    if (isWithin(spec, g.id, id)) {
      for (const m of g.members ?? []) if (!m.startsWith("wp:") && !out.includes(m)) out.push(m);
    }
  }
  return out;
}

// The waypoints anchored to any of `bounds` (and, from older designs, a
// group's own `wp:` members).
export function anchoredWaypoints(spec: SpecDict | null, bounds: string[], groupId?: string): string[] {
  const out = Object.entries(spec?.queryables ?? {})
    .filter(([, q]: [string, any]) => q?.type === "waypoint" && bounds.includes(q.anchor))
    .map(([name]) => name);
  if (groupId) {
    for (const g of groupsOf(spec)) {
      if (!isWithin(spec, g.id, groupId)) continue;
      for (const m of g.members ?? []) if (m.startsWith("wp:") && !out.includes(m.slice(3))) out.push(m.slice(3));
    }
  }
  return out;
}

function setGroups(spec: SpecDict, groups: Group[]) {
  const t = { ...(spec.transform ?? {}) };
  if (groups.length) t.groups = groups;
  else delete t.groups;
  spec.transform = Object.keys(t).length ? t : null;
}

// Drop groups with neither bounds nor subgroups - left over from a drag.
function prune(spec: SpecDict) {
  let groups = groupsOf(spec);
  for (let pass = 0; pass < 8; pass++) {
    const empty = groups.filter(
      (g) => !(g.members ?? []).length && !groups.some((o) => o.parent === g.id),
    );
    if (!empty.length) break;
    const gone = new Set(empty.map((g) => g.id));
    groups = groups.filter((g) => !gone.has(g.id));
  }
  setGroups(spec, groups);
}

function withoutShape(groups: Group[], name: string): Group[] {
  return groups.map((g) => ({ ...g, members: (g.members ?? []).filter((m) => m !== name) }));
}

// Put bounds `name` into group `id` - out of any other; null: out of all.
export function moveShape(spec: SpecDict, name: string, id: string | null) {
  let groups = withoutShape(groupsOf(spec), name);
  if (id) groups = groups.map((g) => (g.id === id ? { ...g, members: [...(g.members ?? []), name] } : g));
  setGroups(spec, groups);
  prune(spec);
}

// Bounds `dragged` dropped on bounds `target`: a new group of the two, where
// `target` was (inside its group, if any). Returns the new group's id.
export function groupTogether(spec: SpecDict, dragged: string, target: string): string | null {
  if (dragged === target) return null;
  const host = groupOfShape(spec, target);
  let groups = withoutShape(withoutShape(groupsOf(spec), dragged), target);
  const id = newGroupId();
  groups = [
    ...groups,
    { id, name: uniqueGroupName(groups, `${target} group`), members: [target, dragged], parent: host?.id ?? null, pivot: null },
  ];
  setGroups(spec, groups);
  prune(spec);
  return id;
}

// Group `id` into group `parent` (null: to the top) - never into itself.
export function moveGroup(spec: SpecDict, id: string, parent: string | null): boolean {
  if (parent && isWithin(spec, parent, id)) return false;
  setGroups(
    spec,
    groupsOf(spec).map((g) => (g.id === id ? { ...g, parent } : g)),
  );
  return true;
}

// Ungroup group `id`: its bounds and subgroups go to where it was.
export function dissolveGroup(spec: SpecDict, id: string) {
  const groups = groupsOf(spec);
  const group = groups.find((g) => g.id === id);
  if (!group) return;
  const parent = group.parent ?? null;
  let next = groups.filter((g) => g.id !== id).map((g) => (g.parent === id ? { ...g, parent } : g));
  if (parent) {
    next = next.map((g) => (g.id === parent ? { ...g, members: [...(g.members ?? []), ...(group.members ?? [])] } : g));
  }
  setGroups(spec, next);
  prune(spec);
}

function uniqueGroupName(groups: Group[], base: string): string {
  const taken = new Set(groups.map((g) => g.name));
  if (!taken.has(base)) return base;
  for (let i = 2; ; i++) if (!taken.has(`${base} ${i}`)) return `${base} ${i}`;
}

// Older designs grouped waypoints directly (`wp:<name>` members): anchor each
// to a bounds of its group instead, where the group has one. Mutates spec.
export function migrateWaypointMembers(spec: SpecDict) {
  const groups = groupsOf(spec);
  if (!groups.some((g) => (g.members ?? []).some((m) => m.startsWith("wp:")))) return;
  setGroups(
    spec,
    groups.map((g) => {
      const bound = shapesInGroup(spec, g.id)[0];
      if (!bound) return g;
      const members = (g.members ?? []).filter((m) => {
        if (!m.startsWith("wp:")) return true;
        const q = spec.queryables?.[m.slice(3)];
        if (q && !q.anchor) q.anchor = bound;
        return !q;
      });
      return { ...g, members };
    }),
  );
}
