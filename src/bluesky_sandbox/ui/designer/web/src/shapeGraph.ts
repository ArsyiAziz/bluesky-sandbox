// Who uses which bounds, and what each depends on - the design's dependencies
// in one place, for the Geometry panel to show and for renames and deletes to
// follow. A bounds is used by:
//   - an element: the airspace, a queryable's bounds or sample area, a spawn
//     region's bounds;
//   - another bounds: a generator's parent, a placement's region to stay
//     within or clear of, a drift's region to bounce in;
//   - a group: one whose moves bounce within it (a group holding a bounds
//     contains it - `group` - but doesn't use it);
//   - a waypoint anchored to it;
//   - code: `ctx.shape("name")`, `shapes["name"]` - or the older `ctx.bounds`, `named_bounds`, `episode_bounds`.
// A ref to one of a partition's shapes (`sectors.0`) uses that partition.
import type { SpecDict } from "./api";
import type { EditTarget } from "./map/types";

export type ShapeUse = {
  // What uses it, for a person: "airspace", "storm", "wx (within)".
  label: string;
  // Where to go to see it, if anywhere.
  target: EditTarget | null;
  // The exact name referred to: the bounds itself or one of its shapes.
  ref: string;
  // The kind of element it is, where its target doesn't say: "waypoint".
  kind?: string;
};

export type ShapeInfo = {
  name: string;
  usedBy: ShapeUse[];
  // The bounds this one's layers refer to: [name, through which param].
  dependsOn: { name: string; via: string }[];
  // The code blocks that read it by name.
  inCode: string[];
  // The group that holds it, if any.
  group: { id: string; name: string } | null;
};

// Whether `ref` names bounds `name`: itself, or one of its shapes.
export const refersTo = (ref: unknown, name: string): boolean =>
  typeof ref === "string" &&
  (ref === name || (ref.startsWith(`${name}.`) && /^\d+$/.test(ref.slice(name.length + 1))));

// The bounds a ref names: `sectors.0` is `sectors`'s.
const shapeOf = (ref: string, names: Set<string>): string | null => {
  if (names.has(ref)) return ref;
  const base = ref.replace(/\.\d+$/, "");
  return base !== ref && names.has(base) ? base : null;
};

// A region's layers that may refer to other bounds: its generator's params,
// its placement, each motion - as [where, the layer's params].
function layers(region: SpecDict): [string, SpecDict][] {
  const out: [string, SpecDict][] = [];
  if (region?.footprint?.type === "generated") out.push(["", region.footprint.params ?? {}]);
  if (region?.placement) out.push(["placement", region.placement]);
  for (const m of region?.motion ?? []) out.push([String(m?.type ?? "motion"), m]);
  return out;
}

// Every ref a value holds - one, or a list of them.
const refsIn = (value: any): string[] =>
  (Array.isArray(value) ? value : [value]).map((v) => v?.ref).filter((r): r is string => typeof r === "string");

// The design's code, block by block: what may read a bounds by name.
function codeBlocks(spec: SpecDict): [string, string][] {
  const out: [string, string][] = [];
  const walk = (label: string, value: any) => {
    if (typeof value === "string") out.push([label, value]);
    else if (Array.isArray(value)) value.forEach((v, i) => walk(`${label}[${i}]`, v));
    else if (value && typeof value === "object") for (const [k, v] of Object.entries(value)) walk(label ? `${label}.${k}` : k, v);
  };
  for (const [file, text] of Object.entries(spec.code ?? {})) walk(file, text);
  for (const [hook, body] of Object.entries(spec.env?.hooks ?? {})) walk(`hook ${hook}`, body);
  walk("hook setup", spec.env?.hook_setup);
  walk("task info setup", spec.env?.task_info_setup);
  walk("task info", spec.env?.task_info);
  walk("scenario setup", spec.scenario_setup);
  for (const [hook, body] of Object.entries(spec.scenario_hooks ?? {})) walk(`scenario ${hook}`, body);
  return out;
}

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export function shapeGraph(spec: SpecDict | null): Record<string, ShapeInfo> {
  const regions: Record<string, SpecDict> = spec?.shapes ?? {};
  const names = new Set(Object.keys(regions));
  const info: Record<string, ShapeInfo> = {};
  for (const name of names) info[name] = { name, usedBy: [], dependsOn: [], inCode: [], group: null };
  if (!spec) return info;
  const use = (ref: unknown, label: string, target: EditTarget | null) => {
    if (typeof ref !== "string") return;
    const name = shapeOf(ref, names);
    const q = target?.scope === "queryable" ? spec.queryables?.[target.name] : null;
    const kind = q?.type === "waypoint" ? "waypoint" : undefined;
    if (name) info[name].usedBy.push({ label: ref === name ? label : `${label} (${ref})`, target, ref, kind });
  };

  use(spec.airspace?.ref, "airspace", { scope: "airspace" });
  for (const [qname, q] of Object.entries(spec.queryables ?? {}) as [string, SpecDict][]) {
    use(q?.shape?.ref, qname, { scope: "queryable", name: qname });
    use(q?.sample?.ref, `${qname} (sample area)`, { scope: "queryable", name: qname });
  }
  (spec.spawn?.regions ?? []).forEach((r: SpecDict, index: number) =>
    use(r?.shape?.ref, r?.name || `spawn ${index + 1}`, { scope: "spawn", index }),
  );
  for (const [rname, region] of Object.entries(regions)) {
    for (const [layer, params] of layers(region)) {
      for (const [key, value] of Object.entries(params)) {
        if (key === "type") continue;
        for (const ref of refsIn(value)) {
          const via = layer ? `${layer} ${key}` : key;
          use(ref, `${rname} (${via})`, { scope: "region", name: rname });
          const dep = shapeOf(ref, names);
          if (dep) info[rname].dependsOn.push({ name: ref, via });
        }
      }
    }
  }
  for (const g of (spec.transform?.groups ?? []) as SpecDict[]) {
    for (const member of g.members ?? []) if (info[member]) info[member].group = { id: g.id, name: g.name ?? g.id };
    // A group's motion may bounce within a bounds.
    for (const m of g.motion ?? []) {
      for (const [key, value] of Object.entries(m ?? {})) {
        if (key !== "type") for (const ref of refsIn(value)) use(ref, `group ${g.name ?? g.id} (${m.type} ${key})`, { scope: "group", id: g.id });
      }
    }
  }
  // A waypoint anchored to a bounds moves with it.
  for (const [qname, q] of Object.entries(spec.queryables ?? {}) as [string, SpecDict][]) {
    if (q?.type === "waypoint") use(q.anchor, `${qname} (anchored)`, { scope: "queryable", name: qname });
  }
  const blocks = codeBlocks(spec);
  for (const name of names) {
    // ctx.shape("x"), shapes["x"] - or the older ctx.bounds("x"), named_bounds["x"].
    const read = new RegExp(`(?:bounds|shapes?)\\s*[([]\\s*["']${escape(name)}(\\.\\d+)?["']`);
    info[name].inCode = blocks.filter(([, text]) => read.test(text)).map(([label]) => label);
  }
  return info;
}

// The bounds a layer list refers to - a generator's parent, a placement's
// region, a drift's - by bounds name (a partition's shape by its partition).
export function refsOfLayers(spec: SpecDict, params: SpecDict[]): string[] {
  const names = new Set(Object.keys(spec?.shapes ?? {}));
  const out: string[] = [];
  for (const p of params) {
    for (const [key, value] of Object.entries(p ?? {})) {
      if (key === "type") continue;
      for (const ref of refsIn(value)) {
        const name = shapeOf(ref, names);
        if (name && !out.includes(name)) out.push(name);
      }
    }
  }
  return out;
}

// The bounds bounds `name` depends on, by name.
export const dependenciesOf = (spec: SpecDict, name: string): string[] =>
  refsOfLayers(spec, layers(spec?.shapes?.[name]).map(([, params]) => params)).filter((n) => n !== name);

// Whether nothing uses a bounds - no element, bounds, group's moves or code.
// Being in a group isn't a use: a group only holds it.
export const isUnused = (b: ShapeInfo | undefined): boolean => !!b && b.usedBy.length === 0 && b.inCode.length === 0;

// Rename bounds `oldName` to `newName` wherever it is referred to - elements,
// other bounds' layers, groups - its shapes' refs (`old.0`) with it. Mutates
// spec; code that reads it by name is the designer's to change.
export function renameShapeRefs(spec: SpecDict, oldName: string, newName: string) {
  const rename = (ref: string) => (refersTo(ref, oldName) ? newName + ref.slice(oldName.length) : ref);
  const fix = (b: any) => (b && typeof b.ref === "string" ? { ...b, ref: rename(b.ref) } : b);
  const fixValue = (v: any) => (Array.isArray(v) ? v.map(fix) : fix(v));
  if (spec.airspace) spec.airspace = fix(spec.airspace);
  for (const q of Object.values(spec.queryables ?? {}) as SpecDict[]) {
    if (q.shape) q.shape = fix(q.shape);
    if (q.sample) q.sample = fix(q.sample);
  }
  for (const r of (spec.spawn?.regions ?? []) as SpecDict[]) if (r.shape) r.shape = fix(r.shape);
  for (const region of Object.values(spec.shapes ?? {}) as SpecDict[]) {
    for (const [, params] of layers(region)) {
      for (const key of Object.keys(params)) if (key !== "type") params[key] = fixValue(params[key]);
    }
  }
  for (const g of (spec.transform?.groups ?? []) as SpecDict[]) {
    if (Array.isArray(g.members)) g.members = g.members.map((m: string) => rename(m));
    for (const m of g.motion ?? []) for (const key of Object.keys(m)) if (key !== "type") m[key] = fixValue(m[key]);
  }
  for (const q of Object.values(spec.queryables ?? {}) as SpecDict[]) {
    if (typeof q?.anchor === "string") q.anchor = rename(q.anchor);
  }
}
