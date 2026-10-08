// Drag-edit handles: derive the draggable control points for the selected
// element (box corners, polygon vertices, center, move, rotate) and apply a
// handle drag back onto the spec. Pure functions over the spec + geometry.
import { waypointLatLon, waypointPoint } from "../waypointPoints";
import type { PreviewResult, SpecDict } from "../api";
import { isSampledValue } from "../specHelpers";
import { anchoredWaypoints, shapesInGroup } from "../groupTree";
import type { DragState, EditHandle, EditTarget, RGBA } from "./types";
import {
  NAMED,
  WAYPOINT_LIFT_M,
  shapeCenter,
  shapeRadiusDeg,
  boxCorner,
  cssToRgb,
  frontendShapeGeometry,
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
export function groupMemberShapes(spec: SpecDict | null, id: string): SpecDict[] {
  return shapesInGroup(spec, id)
    .map((name: string) => spec?.shapes?.[name])
    .filter(Boolean) as SpecDict[];
}

// Lat/lon points of every member of a group: footprint vertices for bounds, the
// position for ``wp:<name>`` waypoint members.
function groupMemberPoints(spec: SpecDict | null, id: string): [number, number][] {
  const pts: [number, number][] = [];
  const bounds = shapesInGroup(spec, id);
  for (const m of bounds) {
    const g = spec?.shapes?.[m] ? frontendShapeGeometry(spec.shapes[m]) : null;
    if (g) for (const v of g.vertices) pts.push(v);
  }
  for (const name of anchoredWaypoints(spec, bounds, id)) {
    const q = spec?.queryables?.[name];
    if (q && Number.isFinite(q.lat) && Number.isFinite(q.lon)) pts.push([q.lat, q.lon]);
  }
  return pts;
}

// Move waypoints anchored to a bounds with it: from where the drag started,
// through ``map``.
function carryAnchored(
  next: SpecDict,
  base: SpecDict,
  names: string[],
  map: (lat: number, lon: number) => [number, number],
) {
  for (const name of names) {
    const startQ = base.queryables?.[name];
    const q = next.queryables?.[name];
    if (!startQ || !q || !Number.isFinite(startQ.lat) || !Number.isFinite(startQ.lon)) continue;
    const { waypoint: _fix, ...rest } = q;
    const [lat, lon] = map(startQ.lat, startQ.lon);
    next.queryables[name] = { ...rest, lat, lon };
  }
}

// The named bounds an edit target draws from, if it is one.
function shapeNameOf(spec: SpecDict, target: EditTarget): string | null {
  if (target.scope === "region") return target.name;
  if (target.scope === "airspace") return spec.airspace?.ref ?? null;
  if (target.scope === "queryable") return spec.queryables?.[target.name]?.shape?.ref ?? null;
  if (target.scope === "spawn") return spec.spawn?.regions?.[target.index]?.shape?.ref ?? null;
  return null;
}

// Lat/lon bounding box of a group's member geometry, and its center.
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
export function resolveShape(spec: SpecDict | null, b: any): SpecDict | null {
  if (!spec || !b) return null;
  if (typeof b.ref === "string") return spec.shapes?.[b.ref] ?? null;
  return b;
}

export function buildEditHandles(spec: SpecDict | null, preview?: PreviewResult | null, selected?: EditTarget | null): EditHandle[] {
  if (!spec || !selected) return [];
  const handles: EditHandle[] = [];
  const airspaceShape = resolveShape(spec, spec.airspace);
  if (airspaceShape && sameTarget(selected, { scope: "airspace" })) {
    handles.push(...shapeEditHandles(airspaceShape, { scope: "airspace" }, "airspace", NAMED.blue));
  }
  for (const [name, q] of Object.entries(spec.queryables ?? {})) {
    const target: EditTarget = { scope: "queryable", name };
    if (!sameTarget(selected, target)) continue;
    const item = q as SpecDict;
    if (item.type === "waypoint") {
      // A placed point (or an older sampled waypoint) has no fixed spot to
      // drag - it is drawn anew each episode - so no handle.
      const point = waypointPoint(spec, item);
      if (item.sample || point?.placement) continue;
      const resolved = preview?.queryables.find((p: any) => p.kind === "waypoint" && p.name === name) as any;
      const at = waypointLatLon(spec, item);
      const lat = at ? at[0] : resolved?.lat;
      const lon = at ? at[1] : resolved?.lon;
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
      const bounds = resolveShape(spec, item.shape);
      if (bounds) handles.push(...shapeEditHandles(bounds, target, name, cssToRgb(item.color)));
    }
  }
  (spec.spawn?.regions ?? []).forEach((region: SpecDict, index: number) => {
    const target: EditTarget = { scope: "spawn", index };
    if (sameTarget(selected, target)) {
      const bounds = resolveShape(spec, region.shape);
      if (bounds) handles.push(...shapeEditHandles(bounds, target, region.name ?? `spawn_${index}`, NAMED.green));
    }
  });
  // Standalone named bounds (not yet referenced, or sample-only) edit directly.
  if (selected.scope === "region") {
    const bounds = spec.shapes?.[selected.name];
    if (bounds) handles.push(...shapeEditHandles(bounds, selected, selected.name, NAMED.slate));
  }
  // A transform group: a single move handle at the members' center + a rotate
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

export function shapeEditHandles(bounds: SpecDict, target: EditTarget, name: string, color: RGBA): EditHandle[] {
  const fp = bounds?.footprint;
  if (!fp) return [];
  const base = `${target.scope}:${"name" in target ? target.name : "index" in target ? target.index : "airspace"}`;
  const center = shapeCenter(bounds);
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
    // below (a dedicated center handle would just duplicate it).
    default:
      break;
  }
  // A generated shape is drawn at a random turn each episode: it moves, but a
  // rotation would mean nothing.
  if (center && fp.type === "generated") {
    if (fp.params?.center) handles.push(handle("move-shape", center[0], center[1]));
  } else if (center && fp.type === "point") {
    // A point only moves: turning it about itself changes nothing.
    handles.push(handle("move-shape", center[0], center[1]));
  } else if (center) {
    handles.push(handle("move-shape", center[0], center[1]));
    const radius = Math.max(0.05, shapeRadiusDeg(bounds) * 0.65);
    const cosLat = Math.max(0.01, Math.cos((center[0] * Math.PI) / 180));
    // Sit at the shape's "north" point and travel with it: the handle angle uses
    // the same CCW convention as rotateLatLon, so dragging spins the shape the
    // way the pointer moves.
    const angle = ((90 + (bounds.rotation_deg ?? 0)) * Math.PI) / 180;
    handles.push(handle("rotate", center[0] + radius * Math.sin(angle), center[1] + (radius * Math.cos(angle)) / cosLat));
  }
  return handles;
}

export function shapeForTarget(spec: SpecDict, target: EditTarget): SpecDict | null {
  // Resolve through {ref} so handle edits mutate the shared named region in
  // place (every consumer referencing it updates together).
  if (target.scope === "airspace") return resolveShape(spec, spec.airspace);
  if (target.scope === "queryable") return resolveShape(spec, spec.queryables?.[target.name]?.shape);
  if (target.scope === "region") return spec.shapes?.[target.name] ?? null;
  if (target.scope === "group") return null; // a group has no single bounds
  return resolveShape(spec, spec.spawn?.regions?.[target.index]?.shape);
}

export function dragStateForHandle(handle: EditHandle, spec: SpecDict, startY: number): DragState {
  const startSpec = structuredClone(spec);
  if (handle.target.scope === "group") {
    const box = groupBbox(startSpec, handle.target.id);
    return { handle, startSpec, startY, rotationCenter: box?.center ?? null, rotationDeg: 0 };
  }
  const startShape = shapeForTarget(startSpec, handle.target);
  return {
    handle,
    startSpec,
    startY,
    rotationCenter: startShape ? shapeCenter(startShape) : null,
    rotationDeg: startShape?.rotation_deg ?? 0,
  };
}

// Translate / rotate every member bounds of a group together, from the drag's
// start snapshot, so the operation is absolute (no per-frame drift). Footprints
// stay parametric: a rotate moves each member's center about the group center
// and adds to its own ``rotation_deg``.
function updateGroupFromHandle(next: SpecDict, handle: EditHandle, lon: number, lat: number, drag?: DragState): SpecDict {
  if (handle.target.scope !== "group") return next;
  const base = drag?.startSpec ?? next;
  const box = groupBbox(base, handle.target.id);
  if (!box) return next;
  const [clat, clon] = box.center;
  // Every bounds in it - its subgroups' too - and the waypoints anchored to them.
  const id = (handle.target as any).id as string;
  const members = shapesInGroup(base, id);
  const anchored = anchoredWaypoints(base, members, id);
  if (handle.role === "move-shape") {
    const dLat = lat - clat;
    const dLon = lon - clon;
    carryAnchored(next, base, anchored, (a, b) => [a + dLat, b + dLon]);
    for (const m of members) {
      const startB = base.shapes?.[m];
      const b = next.shapes?.[m];
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
    carryAnchored(next, base, anchored, (a, b) => rotateLatLon(a, b, box.center, delta));
    for (const m of members) {
      const startB = base.shapes?.[m];
      const b = next.shapes?.[m];
      if (!startB || !b) continue;
      b.footprint = structuredClone(startB.footprint);
      const mc = shapeCenter(b); // member center, from the start snapshot
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
      // Moved: its point - off any navdb fix it was at.
      const point = waypointPoint(next, q);
      if (point) {
        point.footprint.center = latLonObj(lat, lon);
        delete point.footprint.fix;
        return next;
      }
      const { waypoint, ...rest } = q;
      next.queryables[handle.target.name] = { ...rest, lat, lon };
      return next;
    }
  }
  const bounds = shapeForTarget(next, handle.target);
  const fp = bounds?.footprint;
  if (!fp) return next;
  const center = shapeCenter(bounds);
  const spatialHandle = handle.role === "box-corner" || handle.role === "polygon-vertex" || handle.role === "center";
  const rotationCenter = drag?.rotationCenter ?? center;
  const rotationDeg = drag?.rotationDeg ?? bounds.rotation_deg ?? 0;
  let editLat = lat;
  let editLon = lon;
  if (spatialHandle && rotationCenter && rotationDeg) {
    [editLat, editLon] = inverseRotateLatLon(lat, lon, rotationCenter, rotationDeg);
  }
  // Waypoints anchored to this bounds move with it.
  const shapeName = shapeNameOf(next, handle.target);
  const anchored = shapeName && drag ? anchoredWaypoints(drag.startSpec, [shapeName]) : [];
  if (handle.role === "move-shape" && drag) {
    const startShape = shapeForTarget(drag.startSpec, handle.target);
    const startCenter = startShape ? shapeCenter(startShape) : null;
    if (startCenter) {
      const dLat = lat - startCenter[0];
      const dLon = lon - startCenter[1];
      moveFootprint(fp, dLat, dLon);
      carryAnchored(next, drag.startSpec, anchored, (a, b) => [a + dLat, b + dLon]);
    }
    return next;
  }
  if (handle.role === "rotate") {
    if (center) {
      const cosLat = Math.max(0.01, Math.cos((center[0] * Math.PI) / 180));
      const angleDeg = (Math.atan2(lat - center[0], (lon - center[1]) * cosLat) * 180) / Math.PI;
      bounds.rotation_deg = (((angleDeg - 90) % 360) + 360) % 360;
      const startShape = drag ? shapeForTarget(drag.startSpec, handle.target) : null;
      const turn = (bounds.rotation_deg ?? 0) - (startShape?.rotation_deg ?? 0);
      if (drag) carryAnchored(next, drag.startSpec, anchored, (a, b) => rotateLatLon(a, b, center, turn));
    }
    return next;
  }
  if (handle.role === "box-corner") {
    const startShape = drag ? shapeForTarget(drag.startSpec, handle.target) : null;
    const startFp = startShape?.footprint;
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
    const startShape = drag ? shapeForTarget(drag.startSpec, handle.target) : null;
    const startFp = startShape?.footprint;
    const startCenter = startShape ? shapeCenter(startShape) : null;
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
