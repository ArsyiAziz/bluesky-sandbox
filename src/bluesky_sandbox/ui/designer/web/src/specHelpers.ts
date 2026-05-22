// Constructors + small immutable helpers for spec fragments the panel edits.
import type { SpecDict } from "./api";

// ---- sampled footprint params ------------------------------------------- //
// Footprint scalar params (radius_nm, half_angle_deg, ...) may be sampled per
// episode: number | {type:"range",low,high} | {type:"scipy",...}. Client-side
// geometry (map preview, summaries) renders the *representative* shape,
// mirroring the backend's representative_value (range -> midpoint,
// scipy -> mean-ish from loc/scale).

export function isSampledValue(v: any): boolean {
  return !!v && typeof v === "object" && (v.type === "range" || v.type === "scipy");
}

export function repValue(v: any): number {
  if (typeof v === "number") return v;
  if (v && typeof v === "object") {
    if (v.type === "range") return (Number(v.low) + Number(v.high)) / 2;
    if (v.type === "scipy") {
      const k = v.kwds ?? {};
      const loc = Number(k.loc ?? 0);
      const scale = Number(k.scale ?? 0);
      return v.name === "uniform" ? loc + scale / 2 : loc;
    }
  }
  return NaN;
}

export function fmtSampled(v: any, digits = 0): string {
  if (typeof v === "number") return v.toFixed(digits);
  if (v && typeof v === "object") {
    if (v.type === "range") return `${Number(v.low).toFixed(digits)}–${Number(v.high).toFixed(digits)}`;
    if (v.type === "scipy") return `~${v.name ?? "dist"}`;
  }
  return "?";
}

export const defaultBox = (lat = 52.0, lon = 4.75): SpecDict => ({
  type: "box",
  lat_min_deg: lat - 0.2,
  lat_max_deg: lat + 0.2,
  lon_min_deg: lon - 0.3,
  lon_max_deg: lon + 0.3,
});

export const defaultDisk = (lat = 52.0, lon = 4.75): SpecDict => ({
  type: "disk",
  center: { lat_deg: lat, lon_deg: lon },
  radius_nm: 20,
  n_vertices: 72,
});

export const defaultSector = (lat = 52.0, lon = 4.75): SpecDict => ({
  type: "sector",
  center: { lat_deg: lat, lon_deg: lon },
  radius_nm: 30,
  bearing_deg: 90,
  half_angle_deg: 30,
  n_vertices: 24,
});

export const defaultAnnular = (lat = 52.0, lon = 4.75): SpecDict => ({
  type: "annular_sector",
  center: { lat_deg: lat, lon_deg: lon },
  inner_radius_nm: 10,
  outer_radius_nm: 30,
  bearing_deg: 90,
  half_angle_deg: 30,
  n_vertices: 48,
});

export const defaultPolygon = (lat = 52.0, lon = 4.75): SpecDict => ({
  type: "polygon",
  coords: [
    [lat + 0.2, lon - 0.2],
    [lat + 0.2, lon + 0.2],
    [lat - 0.2, lon + 0.2],
    [lat - 0.2, lon - 0.2],
  ],
});

export const FOOTPRINT_TYPES = ["box", "disk", "sector", "annular_sector", "polygon", "boolean"];

export const makeFootprint = (type: string, lat = 52.0, lon = 4.75): SpecDict => {
  switch (type) {
    case "box":
      return defaultBox(lat, lon);
    case "disk":
      return defaultDisk(lat, lon);
    case "sector":
      return defaultSector(lat, lon);
    case "annular_sector":
      return defaultAnnular(lat, lon);
    case "polygon":
      return defaultPolygon(lat, lon);
    case "boolean":
      return { type: "boolean", op: "union", left: defaultBox(lat, lon), right: defaultDisk(lat, lon) };
    default:
      return defaultBox();
  }
};

export const defaultConstantBand = (minFt = 0, maxFt = 20000): SpecDict => ({ type: "constant", min_ft: minFt, max_ft: maxFt });

export const defaultLinearBand = (lat = 52.0, lon = 4.5): SpecDict => ({
  type: "linear",
  start: { lat_deg: lat, lon_deg: lon },
  end: { lat_deg: lat, lon_deg: lon + 0.6 },
  start_band_ft: [1000, 5000],
  end_band_ft: [3000, 9000],
});

export const defaultRadialBand = (lat = 52.0, lon = 4.75): SpecDict => ({
  type: "radial",
  center: { lat_deg: lat, lon_deg: lon },
  radius_nm: 20,
  inner_band_ft: [1000, 4000],
  outer_band_ft: [2000, 8000],
});

export const ALTITUDE_TYPES = ["none", "constant", "linear", "radial", "vertex"];

export const makeAltitude = (type: string, lat = 52.0, lon = 4.75): SpecDict | null => {
  switch (type) {
    case "none":
      return null;
    case "constant":
      return defaultConstantBand();
    case "linear":
      return defaultLinearBand(lat, lon);
    case "radial":
      return defaultRadialBand(lat, lon);
    case "vertex":
      return {
        type: "vertex",
        vertices: [
          [lat + 0.2, lon - 0.2],
          [lat + 0.2, lon + 0.2],
          [lat - 0.2, lon],
        ],
        min_values_ft: 0,
        max_values_ft: [8000, 9000, 10000],
      };
    default:
      return null;
  }
};

// A representative center for seeding a new primitive, read from an existing
// footprint when possible.
export const footprintCenter = (fp: SpecDict | undefined): [number, number] => {
  if (!fp) return [52.0, 4.75];
  if (fp.type === "box")
    return [
      (repValue(fp.lat_min_deg) + repValue(fp.lat_max_deg)) / 2,
      (repValue(fp.lon_min_deg) + repValue(fp.lon_max_deg)) / 2,
    ];
  if (fp.center) return [fp.center.lat_deg, fp.center.lon_deg];
  if (fp.start) return [fp.start.lat_deg, fp.start.lon_deg];
  if (fp.coords?.length) {
    const n = fp.coords.length;
    const [lat, lon] = fp.coords.reduce(([a, b]: [number, number], [c, d]: [number, number]) => [a + c, b + d], [0, 0]);
    return [lat / n, lon / n];
  }
  return [52.0, 4.75];
};

export const footprintBbox = (fp: SpecDict | undefined): [number, number, number, number] | null => {
  if (!fp) return null;
  if (fp.type === "box")
    return [repValue(fp.lat_min_deg), repValue(fp.lat_max_deg), repValue(fp.lon_min_deg), repValue(fp.lon_max_deg)];
  if (fp.type === "polygon" && fp.coords?.length) {
    const lats = fp.coords.map(([lat]: [number, number]) => lat);
    const lons = fp.coords.map(([, lon]: [number, number]) => lon);
    return [Math.min(...lats), Math.max(...lats), Math.min(...lons), Math.max(...lons)];
  }
  if (fp.type === "boolean") {
    const left = footprintBbox(fp.left);
    const right = footprintBbox(fp.right);
    if (!left) return right;
    if (!right) return left;
    return [Math.min(left[0], right[0]), Math.max(left[1], right[1]), Math.min(left[2], right[2]), Math.max(left[3], right[3])];
  }
  if (fp.center) {
    const rawRadius = fp.outer_radius_nm ?? fp.radius_nm ?? 1;
    const radiusNm = Number.isFinite(repValue(rawRadius)) ? repValue(rawRadius) : 1;
    const dLat = radiusNm / 60;
    const cosLat = Math.max(0.01, Math.cos((fp.center.lat_deg * Math.PI) / 180));
    const dLon = dLat / cosLat;
    return [fp.center.lat_deg - dLat, fp.center.lat_deg + dLat, fp.center.lon_deg - dLon, fp.center.lon_deg + dLon];
  }
  return null;
};

export const placementCenter = (airspace: SpecDict | null | undefined, viewCenter: [number, number]): [number, number] => {
  const bbox = footprintBbox(airspace?.footprint);
  if (!bbox) return viewCenter;
  const [lat, lon] = viewCenter;
  const [latMin, latMax, lonMin, lonMax] = bbox;
  if (lat >= latMin && lat <= latMax && lon >= lonMin && lon <= lonMax) return viewCenter;
  return footprintCenter(airspace?.footprint);
};

export const placementAltitudeRange = (airspace: SpecDict | null | undefined): [number, number] | null => {
  const altitude = airspace?.altitude;
  if (altitude?.type !== "constant") return null;
  const lo = Number(altitude.min_ft);
  const hi = Number(altitude.max_ft);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  return [Math.min(lo, hi), Math.max(lo, hi)];
};

export const clippedSpawnAltitudeRange = (airspace: SpecDict | null | undefined): [number, number] => {
  const range = placementAltitudeRange(airspace);
  if (!range) return [5000, 15000];
  const [airLo, airHi] = range;
  const lo = Math.max(airLo, Math.min(5000, airHi));
  const hi = Math.min(airHi, Math.max(15000, airLo));
  return lo <= hi ? [lo, hi] : [airLo, airHi];
};

export const defaultRegion = (lat = 52.0, lon = 4.75, altitude?: SpecDict | null): SpecDict => ({
  type: "region",
  footprint: defaultBox(lat, lon),
  altitude: altitude === undefined ? defaultConstantBand() : altitude,
});

export const defaultQueryRegion = (lat = 52.0, lon = 4.75, altitude?: SpecDict | null): SpecDict => ({
  type: "query_region",
  bounds: defaultRegion(lat, lon, altitude),
  color: "orange",
  render_shape: true,
  render_label: true,
  track_temporal_state: false,
});

export const defaultWaypoint = (lat = 52.0, lon = 4.75, ident?: string, altFt = 3000): SpecDict => ({
  type: "waypoint",
  ...(ident ? { waypoint: ident } : { lat, lon }),
  ...(Number.isFinite(altFt) ? { alt_ft: altFt } : {}),
  reach_radius_nm: 1,
  color: "cyan",
  render_shape: true,
  render_label: true,
  track_temporal_state: false,
});

export const defaultSpawnRegion = (lat = 52.0, lon = 4.75, altLo = 5000, altHi = 15000): SpecDict => {
  const lo = Math.min(altLo, altHi);
  const hi = Math.max(altLo, altHi);
  return {
    type: "spawn_region",
    // The bounds' altitude band IS the spawn altitude range — the backend samples
    // spawn altitude from it, so `params` no longer carries a duplicate alt_ft.
    bounds: { type: "region", footprint: defaultBox(lat, lon), altitude: { type: "constant", min_ft: lo, max_ft: hi } },
    n_aircraft: { type: "scipy", name: "randint", args: [2, 6], kwds: {} },
    params: {
      spd_kts: { type: "range", low: 200, high: 280 },
    },
    aircraft_type: null,
    callsign_prefixes: null,
    spawn_time: 0.0,
    route: null,
    name: "SPAWN",
    render_shape: true,
    render_name: true,
  };
};

// randint(low, high) is exclusive of `high`; the panel shows an inclusive count
// range, so convert at the boundary.
export const countRange = (n: any): [number, number] => {
  if (typeof n === "number") return [n, n];
  if (n && n.type === "scipy" && n.name === "randint") return [n.args[0], n.args[1] - 1];
  if (n && n.type === "range") return [n.low, n.high];
  return [1, 1];
};

export const makeCount = (lo: number, hi: number): SpecDict | number =>
  lo === hi ? lo : { type: "scipy", name: "randint", args: [lo, hi + 1], kwds: {} };

export const rangeOf = (v: any): [number, number] =>
  v && v.type === "range" ? [v.low, v.high] : [0, 0];

export const makeRange = (low: number, high: number): SpecDict => ({ type: "range", low, high });

// spawn_time may be a fixed float or a (low, high) range.
export const spawnTimeRange = (v: any): [number, number] => {
  if (typeof v === "number") return [v, v];
  if (v && v.type === "range") return [v.low, v.high];
  return [0, 0];
};

export const makeSpawnTime = (lo: number, hi: number): number | SpecDict =>
  lo === hi ? lo : { type: "range", low: lo, high: hi };

// transform: group rotation sampled from a (min,max) angle range.
export const rotationRange = (spec: SpecDict): [number, number] | null => {
  const r = spec.transform?.rotation;
  if (!r) return null;
  const a = r.angle_deg;
  if (typeof a === "number") return [a, a];
  if (a?.type === "range") return [a.low, a.high];
  return [0, 0];
};

export const makeRotation = (lo: number, hi: number): SpecDict => ({
  rotation: { angle_deg: lo === hi ? lo : { type: "range", low: lo, high: hi }, pivot: null },
});

// --- rotation groups (nestable per-group rotation) -----------------------

// A rotation group's members are **bounds** (named regions): rotating a bounds
// rotates every element (airspace / queryable / spawn region) that references it.
export const designBounds = (spec: SpecDict): string[] => Object.keys(spec?.regions ?? {});

export const angleRange = (angleDeg: any): [number, number] => {
  if (typeof angleDeg === "number") return [angleDeg, angleDeg];
  if (angleDeg?.type === "range") return [angleDeg.low, angleDeg.high];
  return [0, 0];
};

export const makeAngle = (lo: number, hi: number): number | SpecDict =>
  lo === hi ? lo : { type: "range", low: lo, high: hi };

let _gid = 0;
export const newGroupId = (): string => `g${Date.now().toString(36)}${(_gid++).toString(36)}`;

// The bounds (named region) an element id resolves to, for upgrading older
// groups whose members were stored as element ids ("airspace"/"q:…"/"s:…").
const elementToBounds = (spec: SpecDict, eid: string): string | null => {
  if (eid === "airspace") return spec?.airspace?.ref ?? null;
  if (eid.startsWith("q:")) {
    const q = spec?.queryables?.[eid.slice(2)];
    return q?.bounds?.ref ?? q?.sample?.ref ?? null;
  }
  if (eid.startsWith("s:")) {
    const name = eid.slice(2);
    const regions = spec?.spawn?.regions ?? [];
    const r = regions.find((x: SpecDict, i: number) => (x.name || `spawn_${i + 1}`) === name);
    return r?.bounds?.ref ?? null;
  }
  return null;
};

// Normalize a group's members: keep bounds (region) names and ``wp:<name>``
// waypoint members as-is, and translate any leftover element ids from older specs.
const toBoundsMembers = (spec: SpecDict, members: any[]): string[] => {
  const regions = spec?.regions ?? {};
  const queryables = spec?.queryables ?? {};
  const out: string[] = [];
  for (const m of members ?? []) {
    if (typeof m === "string" && m.startsWith("wp:")) {
      if (queryables[m.slice(3)] && !out.includes(m)) out.push(m);
      continue;
    }
    const name = m in regions ? m : elementToBounds(spec, m);
    if (name && !out.includes(name)) out.push(name);
  }
  return out;
};

// Make the rotation model bounds-based and the groups editor the only view:
//   - migrate the legacy single `transform.rotation` into one all-bounds group;
//   - rewrite any existing group whose members are element ids into bounds names.
export const migrateRotationGroups = (spec: SpecDict): SpecDict => {
  const t = spec?.transform;
  if (!t) return spec;
  if (t.groups) {
    const next = structuredClone(spec);
    next.transform.groups = t.groups.map((g: SpecDict) => ({
      ...g,
      members: toBoundsMembers(spec, g.members ?? []),
    }));
    return next;
  }
  if (!t.rotation) return spec;
  const next = structuredClone(spec);
  next.transform = {
    groups: [
      {
        id: newGroupId(),
        name: "all",
        angle_deg: t.rotation.angle_deg,
        pivot: t.rotation.pivot ?? null,
        members: designBounds(spec),
        parent: null,
      },
    ],
  };
  return next;
};

// Overall [min, max] ft of an altitude band spec - used to derive a spawn
// region's `alt_ft` range from its bounds altitude (so it isn't entered twice).
export const altRange = (alt: SpecDict | null | undefined): [number, number] | null => {
  if (!alt) return null;
  switch (alt.type) {
    case "constant":
      return [alt.min_ft, alt.max_ft];
    case "linear":
      return [Math.min(alt.start_band_ft[0], alt.end_band_ft[0]), Math.max(alt.start_band_ft[1], alt.end_band_ft[1])];
    case "radial":
      return [Math.min(alt.inner_band_ft[0], alt.outer_band_ft[0]), Math.max(alt.inner_band_ft[1], alt.outer_band_ft[1])];
    case "vertex": {
      const mins = Array.isArray(alt.min_values_ft) ? alt.min_values_ft : [alt.min_values_ft];
      const maxs = Array.isArray(alt.max_values_ft) ? alt.max_values_ft : [alt.max_values_ft];
      return [Math.min(...mins), Math.max(...maxs)];
    }
    default:
      return null;
  }
};

export const clone = <T,>(o: T): T => structuredClone(o);

const isRef = (b: any): boolean => !!b && typeof b.ref === "string";

// Extract a top-level Python function's body (dedented) from source, or null.
const extractFuncBody = (src: string, name: string): string | null => {
  const lines = src.split("\n");
  const defRe = new RegExp(`^def\\s+${name}\\s*\\(`);
  const i = lines.findIndex((l) => defRe.test(l));
  if (i < 0) return null;
  const body: string[] = [];
  for (let j = i + 1; j < lines.length; j++) {
    const l = lines[j];
    if (l.trim() === "") {
      body.push("");
      continue;
    }
    if (l.length - l.trimStart().length === 0) break; // back to top level
    body.push(l);
  }
  const indents = body.filter((l) => l.trim()).map((l) => l.length - l.trimStart().length);
  const min = indents.length ? Math.min(...indents) : 0;
  return body.map((l) => l.slice(min)).join("\n").trim() || null;
};

// Migrate the old reward/termination/truncation model (task.py functions
// referenced by env.reward_fn) into reward/terminated/truncated env hooks, then
// drop task.py. Idempotent: specs already on the hook model are unchanged.
export const migrateRewardHooks = (spec: SpecDict): SpecDict => {
  const env = spec?.env;
  if (!env || (env.reward_fn == null && env.termination_fn == null && env.truncation_fn == null)) {
    return spec;
  }
  const next = structuredClone(spec);
  const task: string = next.code?.["task.py"] ?? "";
  next.env.hooks = next.env.hooks ?? {};
  for (const [refKey, hook] of [
    ["reward_fn", "reward"],
    ["termination_fn", "terminated"],
    ["truncation_fn", "truncated"],
  ] as const) {
    if (next.env[refKey] != null && next.env.hooks[hook] == null) {
      const body = extractFuncBody(task, hook);
      if (body) next.env.hooks[hook] = body;
    }
    delete next.env[refKey];
  }
  if (next.code) delete next.code["task.py"];
  return next;
};

// The element labels that reference a named bounds (airspace / query bounds +
// sample / spawn). Used to show which elements share a bounds.
export const boundsRefLabels = (spec: SpecDict, name: string): string[] => {
  const labels: string[] = [];
  if (spec?.airspace?.ref === name) labels.push("airspace");
  for (const [qname, q] of Object.entries(spec?.queryables ?? {}) as [string, SpecDict][]) {
    if (q.bounds?.ref === name) labels.push(qname);
    if (q.sample?.ref === name) labels.push(`${qname} (sample)`);
  }
  (spec?.spawn?.regions ?? []).forEach((r: SpecDict, i: number) => {
    if (r.bounds?.ref === name) labels.push(r.name || `spawn_${i + 1}`);
  });
  return labels;
};

// Count how many elements (airspace / query bounds + sample / spawn) reference a
// named bounds — used to warn about shared edits and to garbage-collect orphans.
export const countBoundsRefs = (spec: SpecDict, name: string): number =>
  boundsRefLabels(spec, name).length;

// Drop every named bounds that nothing references (orphan GC). Bounds are only
// created through an element, so any unreferenced one is leftover after a delete
// or reassignment and is swept. Mutates spec.
export const gcOrphanBounds = (spec: SpecDict): void => {
  const regions = spec.regions;
  if (!regions) return;
  for (const name of Object.keys(regions)) {
    if (countBoundsRefs(spec, name) === 0) delete regions[name];
  }
};

// Geometry lives only in named regions. Promote any inline bounds (airspace,
// query-region, spawn-region, waypoint sample) into spec.regions and replace it
// with a {ref: name}. Idempotent — already-ref'd bounds are left untouched.
export const normalizeToRegions = (spec: SpecDict): SpecDict => {
  if (!spec) return spec;
  const next = structuredClone(spec);
  next.regions = next.regions ?? {};
  const used = new Set(Object.keys(next.regions));
  const addRegion = (base: string, bounds: SpecDict): string => {
    const slug = (base || "region").replace(/[^0-9a-zA-Z_]+/g, "_") || "region";
    let name = slug;
    let i = 1;
    while (used.has(name)) name = `${slug}_${i++}`;
    used.add(name);
    next.regions[name] = bounds;
    return name;
  };
  const toRef = (bounds: any, base: string): any =>
    !bounds || isRef(bounds) ? bounds : { ref: addRegion(base, bounds) };

  if (next.airspace && !isRef(next.airspace)) next.airspace = toRef(next.airspace, "airspace");
  for (const [name, q] of Object.entries(next.queryables ?? {}) as [string, any][]) {
    if (q.type === "query_region" && q.bounds) q.bounds = toRef(q.bounds, name);
    if (q.type === "waypoint" && q.sample) q.sample = toRef(q.sample, `${name}_sample`);
  }
  (next.spawn?.regions ?? []).forEach((r: any, i: number) => {
    if (r.bounds) r.bounds = toRef(r.bounds, r.name || `spawn_${i + 1}`);
  });
  return next;
};

// Remove a top-level `class Name(...)` block (and a preceding decorator line)
// from Python source, up to the next top-level class/def/decorator or EOF.
export const stripClass = (source: string, className: string): string => {
  const lines = source.split("\n");
  const startRe = new RegExp(`^class\\s+${className}\\s*[(:]`);
  const classLine = lines.findIndex((l) => startRe.test(l));
  if (classLine < 0) return source;
  // include an immediately-preceding decorator (e.g. @dataclass)
  let start = classLine;
  while (start > 0 && lines[start - 1].trimStart().startsWith("@")) start--;
  let end = classLine + 1;
  while (end < lines.length && !/^(class\s|def\s|@)/.test(lines[end])) end++;
  // drop trailing blank lines left behind
  const before = lines.slice(0, start);
  while (before.length && before[before.length - 1].trim() === "") before.pop();
  const after = lines.slice(end);
  return [...before, "", ...after].join("\n");
};
