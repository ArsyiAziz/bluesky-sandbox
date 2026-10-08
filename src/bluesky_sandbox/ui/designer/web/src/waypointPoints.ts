// A waypoint is where a point shape is: `"shape": {"ref": <point>}`. The
// point says where - fixed, a navdb fix, placed anew each episode, in a group -
// and the waypoint what to ask of an aircraft there (altitude, speed, reach).
// Older designs held the position on the waypoint (lat/lon, a navdb name, a
// region to sample from, a bounds to follow); `migrateWaypointsToPoints`
// moves each onto a point of its own.
import type { SpecDict } from "./api";
import { groupOfShape, groupTogether, moveShape } from "./groupTree";
import { defaultWaypoint, footprintCenter } from "./specHelpers";

export const pointRegion = (lat: number | null, lon: number | null, fix?: string): SpecDict => ({
  type: "region",
  footprint: {
    type: "point",
    ...(Number.isFinite(lat) && Number.isFinite(lon) ? { center: { lat_deg: lat, lon_deg: lon } } : {}),
    ...(fix ? { fix: fix.toUpperCase() } : {}),
  },
});

// A name for a new region, from `base`, not yet taken.
export function freeRegionName(spec: SpecDict, base: string): string {
  const slug = (base || "point").replace(/[^0-9a-zA-Z_]+/g, "_") || "point";
  let name = slug;
  for (let i = 1; spec.shapes?.[name]; i++) name = `${slug}_${i}`;
  return name;
}

// Add point `region` under a name from `base`; its name.
export function addPoint(spec: SpecDict, region: SpecDict, base: string): string {
  spec.shapes = spec.shapes ?? {};
  const name = freeRegionName(spec, base);
  spec.shapes[name] = region;
  return name;
}

// The point a waypoint is at, if it is on one.
export const waypointPoint = (spec: SpecDict | null, q: SpecDict | undefined): SpecDict | null => {
  const ref = q?.shape?.ref;
  const region = typeof ref === "string" ? spec?.shapes?.[ref] : null;
  return region?.footprint?.type === "point" ? region : null;
};

// Where a waypoint is drawn in the design (not an episode's draw): its
// point's position, or an older one's own.
export function waypointLatLon(spec: SpecDict | null, q: SpecDict | undefined): [number, number] | null {
  const c = waypointPoint(spec, q)?.footprint?.center;
  if (c && Number.isFinite(c.lat_deg) && Number.isFinite(c.lon_deg)) return [c.lat_deg, c.lon_deg];
  if (q && Number.isFinite(q.lat) && Number.isFinite(q.lon)) return [q.lat, q.lon];
  return null;
}

// The region a waypoint's position is drawn from: its point's placement
// region (the old sample area), when the point is placed.
export function waypointSampleRef(spec: SpecDict | null, q: SpecDict | undefined): string | null {
  const within = waypointPoint(spec, q)?.placement?.within?.ref;
  return typeof within === "string" ? within : typeof q?.sample?.ref === "string" ? q.sample.ref : null;
}

// Move every waypoint in an older form onto a point of its own. Mutates spec;
// whether anything changed.
export function migrateWaypointsToPoints(spec: SpecDict): boolean {
  let changed = false;
  for (const [name, q] of Object.entries(spec.queryables ?? {}) as [string, SpecDict][]) {
    if (q?.type !== "waypoint" || q.shape) continue;
    changed = true;
    let region: SpecDict;
    const sample = q.sample?.ref;
    if (typeof sample === "string" && spec.shapes?.[sample]) {
      // Drawn from a region each episode: a point placed in it, with its
      // altitude band (the old draw took the altitude from it too).
      const area = spec.shapes[sample];
      const [lat, lon] = footprintCenter(area.footprint);
      region = pointRegion(lat, lon);
      region.placement = { type: "InRegion", within: { ref: sample } };
      if (area.altitude) region.altitude = structuredClone(area.altitude);
    } else {
      region = pointRegion(q.lat ?? null, q.lon ?? null, q.waypoint);
      if (!q.waypoint && !region.footprint.center) region.footprint.center = { lat_deg: 52.0, lon_deg: 4.75 };
    }
    const point = addPoint(spec, region, name);
    // Following a bounds: in a group with it.
    const anchor = q.anchor;
    if (typeof anchor === "string" && spec.shapes?.[anchor]) {
      const group = groupOfShape(spec, anchor);
      if (group) moveShape(spec, point, group.id);
      else groupTogether(spec, point, anchor);
    }
    for (const key of ["lat", "lon", "waypoint", "sample", "anchor"]) delete q[key];
    if (!region.placement) delete q.sample_per;
    q.shape = { ref: point };
  }
  return changed;
}

// Add a waypoint named from `base` on a new point of the same name - at
// lat/lon, or at navdb `fix`. Mutates spec; the waypoint's name.
export function addWaypointAt(
  spec: SpecDict,
  base: string,
  lat: number | null,
  lon: number | null,
  fix?: string,
  altFt?: number,
): string {
  spec.queryables = spec.queryables ?? {};
  let name = base;
  for (let i = 1; spec.queryables[name]; i++) name = `${base}_${i}`;
  const point = addPoint(spec, pointRegion(lat, lon, fix), name);
  spec.queryables[name] = { ...defaultWaypoint(altFt), shape: { ref: point } };
  return name;
}
