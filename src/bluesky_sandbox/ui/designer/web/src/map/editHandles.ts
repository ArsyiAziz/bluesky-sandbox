// Drag-edit handles: derive the draggable control points for the selected
// element (box corners, polygon vertices, centre, move, rotate) and apply a
// handle drag back onto the spec. Pure functions over the spec + geometry.
import type { PreviewResult, SpecDict } from "../api";
import { isSampledValue } from "../specHelpers";
import type { DragState, EditHandle, EditTarget, RGBA } from "./types";
import {
  NAMED,
  WAYPOINT_LIFT_M,
  boundsCenter,
  boundsRadiusDeg,
  boxCorner,
  cssToRgb,
  frontendBoundsGeometry,
  inverseRotateLatLon,
  latLonObj,
  moveFootprint,
  rotateLatLon,
  solveRawPointsForDisplay,
  updateRotatedBoxCorner,
  zMeters,
} from "./geometry";

export function targetKey(target: EditTarget | null | undefined): string {
  if (!target) return "";
  if (target.scope === "airspace") return "airspace";
  if (target.scope === "queryable") return `queryable:${target.name}`;
  if (target.scope === "region") return `region:${target.name}`;
  if (target.scope === "group") return `group:${target.id}`;
  return `spawn:${target.index}`;
}

// The member bounds of a transform group, resolved to their RegionBounds.
// (Waypoint members — ``wp:<name>`` — have no bounds and are skipped.)
export function groupMemberBounds(spec: SpecDict | null, id: string): SpecDict[] {
  const group = (spec?.transform?.groups ?? []).find((g: any) => g.id === id);
  if (!group) return [];
  return (group.members ?? [])
    .filter((m: string) => !m.startsWith("wp:"))
    .map((name: string) => spec?.regions?.[name])
    .filter(Boolean) as SpecDict[];
}

// Lat/lon points of every member of a group: footprint vertices for bounds, the
// position for ``wp:<name>`` waypoint members.
function groupMemberPoints(spec: SpecDict | null, id: string): [number, number][] {
  const group = (spec?.transform?.groups ?? []).find((g: any) => g.id === id);
  if (!group) return [];
  const pts: [number, number][] = [];
  for (const m of group.members ?? []) {
    if (typeof m === "string" && m.startsWith("wp:")) {
      const q = spec?.queryables?.[m.slice(3)];
      if (q && Number.isFinite(q.lat) && Number.isFinite(q.lon)) pts.push([q.lat, q.lon]);
    } else {
      const g = spec?.regions?.[m] ? frontendBoundsGeometry(spec.regions[m]) : null;
      if (g) for (const v of g.vertices) pts.push(v);
    }
  }
  return pts;
}

// Lat/lon bounding box of a group's member geometry, and its centre.
export function groupBbox(spec: SpecDict | null, id: string): { center: [number, number]; latSpan: number; lonSpan: number } | null {
  const pts = groupMemberPoints(spec, id);
  if (!pts.length) return null;
  const lats = pts.map((p) => p[0]);
  const lons = pts.map((p) => p[1]);
  const latMin = Math.min(...lats), latMax = Math.max(...lats);
  const lonMin = Math.min(...lons), lonMax = Math.max(...lons);
  return { center: [(latMin + latMax) / 2, (lonMin + lonMax) / 2], latSpan: latMax - latMin, lonSpan: lonMax - lonMin };
}

export function sameTarget(a: EditTarget | null | undefined, b: EditTarget | null | undefined): boolean {
  return targetKey(a) === targetKey(b);
}

// Geometry lives in named regions; a consumer's bounds may be a {ref: name}.
// Resolve to the underlying region so handles draw on — and edits flow to — it.
export function resolveBounds(spec: SpecDict | null, b: any): SpecDict | null {
  if (!spec || !b) return null;
  if (typeof b.ref === "string") return spec.regions?.[b.ref] ?? null;
  return b;
}

export function buildEditHandles(spec: SpecDict | null, preview?: PreviewResult | null, selected?: EditTarget | null): EditHandle[] {
  if (!spec || !selected) return [];
  const handles: EditHandle[] = [];
  const airspaceBounds = resolveBounds(spec, spec.airspace);
  if (airspaceBounds && sameTarget(selected, { scope: "airspace" })) {
    handles.push(...boundsEditHandles(airspaceBounds, { scope: "airspace" }, "airspace", NAMED.blue));
  }
  for (const [name, q] of Object.entries(spec.queryables ?? {})) {
    const target: EditTarget = { scope: "queryable", name };
    if (!sameTarget(selected, target)) continue;
    const item = q as SpecDict;
    if (item.type === "waypoint") {
      // A sampled waypoint has no fixed point to drag — its position is drawn
      // from its sample region (per episode or per aircraft), so skip the handle.
      if (item.sample) continue;
      const resolved = preview?.queryables.find((p: any) => p.kind === "waypoint" && p.name === name) as any;
      const lon = Number.isFinite(item.lon) ? item.lon : resolved?.lon;
      const lat = Number.isFinite(item.lat) ? item.lat : resolved?.lat;
      const altFt = Number.isFinite(item.alt_ft) ? item.alt_ft : resolved?.alt_ft;
      if (Number.isFinite(lat) && Number.isFinite(lon)) {
        handles.push({
          id: `waypoint:${name}`,
          name,
          lat,
          lon,
          z: zMeters(altFt) + WAYPOINT_LIFT_M,
          target,
          role: "move-shape",
          color: cssToRgb(item.color),
        });
      }
    } else {
      const bounds = resolveBounds(spec, item.bounds);
      if (bounds) handles.push(...boundsEditHandles(bounds, target, name, cssToRgb(item.color)));
    }
  }
  (spec.spawn?.regions ?? []).forEach((region: SpecDict, index: number) => {
    const target: EditTarget = { scope: "spawn", index };
    if (sameTarget(selected, target)) {
      const bounds = resolveBounds(spec, region.bounds);
      if (bounds) handles.push(...boundsEditHandles(bounds, target, region.name ?? `spawn_${index}`, NAMED.green));
    }
  });
  // Standalone named bounds (not yet referenced, or sample-only) edit directly.
  if (selected.scope === "region") {
    const bounds = spec.regions?.[selected.name];
    if (bounds) handles.push(...boundsEditHandles(bounds, selected, selected.name, NAMED.slate));
  }
  // A transform group: a single move handle at the members' centre + a rotate
  // handle, which translate / spin every member bounds together (static edit).
  if (selected.scope === "group") {
    const box = groupBbox(spec, selected.id);
    const group = (spec.transform?.groups ?? []).find((g: any) => g.id === selected.id);
    if (box && group) {
      const name = group.name ?? selected.id;
      const [clat, clon] = box.center;
      handles.push({ id: `group:${selected.id}:move`, name, lat: clat, lon: clon, target: selected, role: "move-shape", color: NAMED.violet });
      const radius = Math.max(0.05, Math.max(box.latSpan, box.lonSpan) * 0.6);
      handles.push({ id: `group:${selected.id}:rotate`, name, lat: clat + radius, lon: clon, target: selected, role: "rotate", color: NAMED.violet });
    }
  }
  return handles;
}

export function boundsEditHandles(bounds: SpecDict, target: EditTarget, name: string, color: RGBA): EditHandle[] {
  const fp = bounds?.footprint;
  if (!fp) return [];
  const base = `${target.scope}:${"name" in target ? target.name : "index" in target ? target.index : "airspace"}`;
  const center = boundsCenter(bounds);
  const rotation = bounds.rotation_deg ?? 0;
  const handle = (role: EditHandle["role"], lat: number, lon: number, extra: Partial<EditHandle> = {}): EditHandle => ({
    id: `${base}:${role}:${extra.index ?? 0}`,
    name,
    ...(() => {
      const spatial = role === "box-corner" || role === "polygon-vertex" || role === "center";
      const p = spatial && center && rotation ? rotateLatLon(lat, lon, center, rotation) : [lat, lon];
      return { lat: p[0], lon: p[1] };
    })(),
    target,
    role,
    color,
    ...extra,
  });
  const handles: EditHandle[] = [];
  switch (fp.type) {
    case "box": {
      // Corner-resize would overwrite a per-episode-sampled edge with a fixed
      // number; a sampled box keeps only move/rotate (which translate the
      // sampled edges meaningfully). Edit the ranges in the panel instead.
      const edges = [fp.lat_min_deg, fp.lat_max_deg, fp.lon_min_deg, fp.lon_max_deg];
      if (edges.some(isSampledValue)) break;
      handles.push(
        handle("box-corner", fp.lat_max_deg, fp.lon_min_deg, { index: 0 }),
        handle("box-corner", fp.lat_max_deg, fp.lon_max_deg, { index: 1 }),
        handle("box-corner", fp.lat_min_deg, fp.lon_max_deg, { index: 2 }),
        handle("box-corner", fp.lat_min_deg, fp.lon_min_deg, { index: 3 }),
      );
      break;
    }
    case "polygon":
      handles.push(...(fp.coords ?? []).map((p: [number, number], index: number) =>
        handle("polygon-vertex", p[0], p[1], { index }),
      ));
      break;
    // disk / sector / annular_sector translate via the single move-shape handle
    // below (a dedicated centre handle would just duplicate it).
    default:
      break;
  }
  if (center) {
    handles.push(handle("move-shape", center[0], center[1]));
    const radius = Math.max(0.05, boundsRadiusDeg(bounds) * 0.65);
    const cosLat = Math.max(0.01, Math.cos((center[0] * Math.PI) / 180));
    // Sit at the shape's "north" point and travel with it: the handle angle uses
    // the same CCW convention as rotateLatLon, so dragging spins the shape the
    // way the pointer moves.
    const angle = ((90 + (bounds.rotation_deg ?? 0)) * Math.PI) / 180;
    handles.push(handle("rotate", center[0] + radius * Math.sin(angle), center[1] + (radius * Math.cos(angle)) / cosLat));
  }
  return handles;
}

export function boundsForTarget(spec: SpecDict, target: EditTarget): SpecDict | null {
  // Resolve through {ref} so handle edits mutate the shared named region in
  // place (every consumer referencing it updates together).
  if (target.scope === "airspace") return resolveBounds(spec, spec.airspace);
  if (target.scope === "queryable") return resolveBounds(spec, spec.queryables?.[target.name]?.bounds);
  if (target.scope === "region") return spec.regions?.[target.name] ?? null;
  if (target.scope === "group") return null; // a group has no single bounds
  return resolveBounds(spec, spec.spawn?.regions?.[target.index]?.bounds);
}

export function dragStateForHandle(handle: EditHandle, spec: SpecDict, startY: number): DragState {
  const startSpec = structuredClone(spec);
  if (handle.target.scope === "group") {
    const box = groupBbox(startSpec, handle.target.id);
    return { handle, startSpec, startY, rotationCenter: box?.center ?? null, rotationDeg: 0 };
  }
  const startBounds = boundsForTarget(startSpec, handle.target);
  return {
    handle,
    startSpec,
    startY,
    rotationCenter: startBounds ? boundsCenter(startBounds) : null,
    rotationDeg: startBounds?.rotation_deg ?? 0,
  };
}

// Translate / rotate every member bounds of a group together, from the drag's
// start snapshot, so the operation is absolute (no per-frame drift). Footprints
// stay parametric: a rotate moves each member's centre about the group centre
// and adds to its own ``rotation_deg``.
function updateGroupFromHandle(next: SpecDict, handle: EditHandle, lon: number, lat: number, drag?: DragState): SpecDict {
  if (handle.target.scope !== "group") return next;
  const base = drag?.startSpec ?? next;
  const box = groupBbox(base, handle.target.id);
  if (!box) return next;
  const [clat, clon] = box.center;
  const group = (next.transform?.groups ?? []).find((g: any) => g.id === (handle.target as any).id);
  const members: string[] = group?.members ?? [];
  if (handle.role === "move-shape") {
    const dLat = lat - clat;
    const dLon = lon - clon;
    for (const m of members) {
      if (m.startsWith("wp:")) {
        const name = m.slice(3);
        const startQ = base.queryables?.[name];
        const q = next.queryables?.[name];
        if (startQ && q && Number.isFinite(startQ.lat) && Number.isFinite(startQ.lon)) {
          q.lat = startQ.lat + dLat;
          q.lon = startQ.lon + dLon;
        }
        continue;
      }
      const startB = base.regions?.[m];
      const b = next.regions?.[m];
      if (!startB || !b) continue;
      b.footprint = structuredClone(startB.footprint);
      moveFootprint(b.footprint, dLat, dLon);
    }
    return next;
  }
  if (handle.role === "rotate") {
    const cosLat = Math.max(0.01, Math.cos((clat * Math.PI) / 180));
    const angleDeg = (Math.atan2(lat - clat, (lon - clon) * cosLat) * 180) / Math.PI;
    const delta = (((angleDeg - 90) % 360) + 360) % 360;
    for (const m of members) {
      if (m.startsWith("wp:")) {
        const name = m.slice(3);
        const startQ = base.queryables?.[name];
        const q = next.queryables?.[name];
        if (startQ && q && Number.isFinite(startQ.lat) && Number.isFinite(startQ.lon)) {
          const [nlat, nlon] = rotateLatLon(startQ.lat, startQ.lon, box.center, delta);
          q.lat = nlat;
          q.lon = nlon;
        }
        continue;
      }
      const startB = base.regions?.[m];
      const b = next.regions?.[m];
      if (!startB || !b) continue;
      b.footprint = structuredClone(startB.footprint);
      const mc = boundsCenter(b); // member centre, from the start snapshot
      if (mc) {
        const [nlat, nlon] = rotateLatLon(mc[0], mc[1], box.center, delta);
        moveFootprint(b.footprint, nlat - mc[0], nlon - mc[1]);
      }
      b.rotation_deg = (((startB.rotation_deg ?? 0) + delta) % 360 + 360) % 360 || undefined;
    }
    return next;
  }
  return next;
}

export function updateSpecFromHandle(
  spec: SpecDict,
  handle: EditHandle,
  lon: number,
  lat: number,
  drag?: DragState,
  pointerY?: number,
): SpecDict {
  void pointerY;
  const next = structuredClone(spec);
  if (handle.target.scope === "group") {
    return updateGroupFromHandle(next, handle, lon, lat, drag);
  }
  if (handle.target.scope === "queryable") {
    const q = next.queryables?.[handle.target.name];
    if ((handle.role === "waypoint" || handle.role === "move-shape") && q?.type === "waypoint") {
      const { waypoint, ...rest } = q;
      next.queryables[handle.target.name] = { ...rest, lat, lon };
      return next;
    }
  }
  const bounds = boundsForTarget(next, handle.target);
  const fp = bounds?.footprint;
  if (!fp) return next;
  const center = boundsCenter(bounds);
  const spatialHandle = handle.role === "box-corner" || handle.role === "polygon-vertex" || handle.role === "center";
  const rotationCenter = drag?.rotationCenter ?? center;
  const rotationDeg = drag?.rotationDeg ?? bounds.rotation_deg ?? 0;
  let editLat = lat;
  let editLon = lon;
  if (spatialHandle && rotationCenter && rotationDeg) {
    [editLat, editLon] = inverseRotateLatLon(lat, lon, rotationCenter, rotationDeg);
  }
  if (handle.role === "move-shape" && drag) {
    const startBounds = boundsForTarget(drag.startSpec, handle.target);
    const startCenter = startBounds ? boundsCenter(startBounds) : null;
    if (startCenter) moveFootprint(fp, lat - startCenter[0], lon - startCenter[1]);
    return next;
  }
  if (handle.role === "rotate") {
    if (center) {
      const cosLat = Math.max(0.01, Math.cos((center[0] * Math.PI) / 180));
      const angleDeg = (Math.atan2(lat - center[0], (lon - center[1]) * cosLat) * 180) / Math.PI;
      bounds.rotation_deg = (((angleDeg - 90) % 360) + 360) % 360;
    }
    return next;
  }
  if (handle.role === "box-corner") {
    const startBounds = drag ? boundsForTarget(drag.startSpec, handle.target) : null;
    const startFp = startBounds?.footprint;
    if (rotationDeg && rotationCenter && startFp?.type === "box" && handle.index != null) {
      updateRotatedBoxCorner(fp, startFp, handle.index, lat, lon, rotationDeg, rotationCenter);
      return next;
    }
    const lats = [fp.lat_min_deg, fp.lat_max_deg];
    const lons = [fp.lon_min_deg, fp.lon_max_deg];
    if (handle.index === 0 || handle.index === 1) lats[1] = editLat;
    if (handle.index === 2 || handle.index === 3) lats[0] = editLat;
    if (handle.index === 0 || handle.index === 3) lons[0] = editLon;
    if (handle.index === 1 || handle.index === 2) lons[1] = editLon;
    fp.lat_min_deg = Math.min(lats[0], lats[1]);
    fp.lat_max_deg = Math.max(lats[0], lats[1]);
    fp.lon_min_deg = Math.min(lons[0], lons[1]);
    fp.lon_max_deg = Math.max(lons[0], lons[1]);
  } else if (handle.role === "polygon-vertex" && handle.index != null && fp.coords?.[handle.index]) {
    const startBounds = drag ? boundsForTarget(drag.startSpec, handle.target) : null;
    const startFp = startBounds?.footprint;
    const startCenter = startBounds ? boundsCenter(startBounds) : null;
    if (rotationDeg && startFp?.type === "polygon" && startCenter) {
      const displayPoints = (startFp.coords ?? []).map(([rawLat, rawLon]: [number, number]) =>
        rotateLatLon(rawLat, rawLon, startCenter, rotationDeg),
      );
      displayPoints[handle.index] = [lat, lon];
      fp.coords = solveRawPointsForDisplay(displayPoints, rotationDeg, startCenter);
    } else {
      fp.coords[handle.index] = [editLat, editLon];
    }
  } else if (handle.role === "center" && fp.center) {
    fp.center = latLonObj(editLat, editLon);
  }
  return next;
}
