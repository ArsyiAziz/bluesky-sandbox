// Pure geometry helpers shared by the map: colour parsing, 3D region/waypoint
// wireframe construction, footprint math, and route resolution. No React, no
// deck.gl - just data in, data out, so the same primitives drive rendering and
// the drag-edit handles.
import type { PreviewResult, SpecDict } from "../api";
import { altRange, repValue } from "../specHelpers";
import type {
  Edge,
  Face,
  RGBA,
  RoutePath,
  Selectable,
  WaypointEdge,
  WaypointFace,
  WaypointShape,
} from "./types";

export const FT_TO_M = 0.3048;
export const WAYPOINT_LIFT_M = 25;
export const TOLERANCE_DISC_VERTICES = 72;
export const DEFAULT_WAYPOINT_REACH_NM = 1;
export const WAYPOINT_MARKER_NM = 0.18;
export const WAYPOINT_STEM_COLOR = [255, 255, 255, 180] as const;

export const NAMED: Record<string, RGBA> = {
  cyan: [34, 211, 238, 255], orange: [245, 158, 11, 255], red: [239, 68, 68, 255],
  magenta: [217, 70, 239, 255], green: [34, 197, 94, 255], blue: [59, 130, 246, 255],
  yellow: [234, 179, 8, 255], white: [255, 255, 255, 255], lime: [132, 204, 22, 255],
  slate: [148, 163, 184, 255], violet: [167, 139, 250, 255],
};

// A stable palette to colour anonymous/named routes by index.
const ROUTE_PALETTE: RGBA[] = [
  NAMED.cyan, NAMED.orange, NAMED.lime, NAMED.magenta,
  NAMED.yellow, NAMED.green, NAMED.blue, NAMED.red,
];

// The authoritative name→colour palette, populated from the catalog (the same
// list the colour picker shows and the drivers render), so every named colour —
// black, gray, purple, … — resolves here rather than only the small built-in set.
let CATALOG_PALETTE: Record<string, RGBA> = {};

function parseHex(value: string): RGBA | null {
  const m = value.trim().match(/^#?([0-9a-f]{6})$/i);
  if (!m) return null;
  const v = parseInt(m[1], 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255, 255];
}

export function setColorPalette(colors: Record<string, string>) {
  const next: Record<string, RGBA> = {};
  for (const [name, hex] of Object.entries(colors)) {
    const rgb = parseHex(hex);
    if (rgb) next[name.trim().toLowerCase()] = rgb;
  }
  CATALOG_PALETTE = next;
}

export function cssToRgb(name?: string): RGBA {
  if (!name) return NAMED.orange;
  const n = name.trim().toLowerCase();
  const hex = parseHex(n);
  if (hex) return hex;
  // Prefer the catalog palette (matches the drivers); fall back to the built-in
  // names for internal colours and the moment before the catalog has loaded.
  if (CATALOG_PALETTE[n]) return CATALOG_PALETTE[n];
  if (NAMED[n]) return NAMED[n];
  return NAMED.orange;
}

export function point(lon: number, lat: number, props: Record<string, any>): GeoJSON.Feature {
  return { type: "Feature", properties: props, geometry: { type: "Point", coordinates: [lon, lat] } };
}

export function centroid(vertices: [number, number][]): [number, number] {
  const n = vertices.length || 1;
  const [sLat, sLon] = vertices.reduce(([a, b], [lat, lon]) => [a + lat, b + lon], [0, 0]);
  return [sLon / n, sLat / n];
}

// Build a region of any shape as edges (bottom ring, top ring, vertical sides)
// plus translucent shaded faces (sides darker than the top, for a volume cue).
// Per-vertex floors/ceilings make varying bands (linear/radial/vertex) slope on
// the top and the sides; a flat band gives a clean box. Edges use LineLayer
// (which renders vertical segments; PathLayer does not).
export function regionGeometry(
  g: { vertices: [number, number][]; alt_min_ft?: number | null; alt_max_ft?: number | null; per_vertex_alt_ft?: [number, number][] },
  color: RGBA,
  out: { edges: Edge[]; faces: Face[] },
  meta: Selectable = {},
) {
  const verts = g.vertices;
  const n = verts.length;
  if (n < 2) return;
  // Real-world vertical scale (no exaggeration): altitude in true metres, so
  // 0 ft sits exactly on the map plane.
  const k = FT_TO_M;
  const pv = g.per_vertex_alt_ft && g.per_vertex_alt_ft.length === n ? g.per_vertex_alt_ft : null;
  const floor = (i: number) => (pv ? pv[i][0] : g.alt_min_ft ?? 0) * k;
  const ceil = (i: number) => (pv ? pv[i][1] : g.alt_max_ft ?? 0) * k;
  const P = (i: number, z: number) => [verts[i][1], verts[i][0], z];
  const hasHeight = verts.some((_, i) => ceil(i) - floor(i) > 1);

  const edge: RGBA = [color[0], color[1], color[2], 255];
  // A faint floor footprint gives a shading/volume cue. We deliberately use only
  // the (horizontal) floor face: vertical wall faces are degenerate when a
  // polygon layer triangulates in 2D (top/bottom corners share lat/lon), so they
  // render unreliably. The floor is a valid flat polygon, and it sits below the
  // contents so it never hides them. The walls are shown via edges instead.
  out.faces.push({ ...meta, polygon: verts.map((_, i) => P(i, floor(i))), color: [color[0], color[1], color[2], 32] });

  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    out.edges.push({ ...meta, src: P(i, floor(i)), tgt: P(j, floor(j)), color: edge }); // bottom ring
    if (hasHeight) {
      out.edges.push({ ...meta, src: P(i, ceil(i)), tgt: P(j, ceil(j)), color: edge }); // top ring
      out.edges.push({ ...meta, src: P(i, floor(i)), tgt: P(i, ceil(i)), color: edge }); // vertical side
    }
  }
}

export function zMeters(altFt: number): number {
  return Number.isFinite(altFt) ? altFt * FT_TO_M : 0;
}

// A dotted vertical "drop" line at each footprint vertex, from the ground
// (where the DOM edit handle projects) up to the shape's lowest altitude, so an
// elevated bounds' draggable corner is visually tied to the corner it controls.
// Faked as short dashes since the LineLayer has no dash support.
export function addHandleDropStems(
  g: { vertices: [number, number][]; alt_min_ft?: number | null; per_vertex_alt_ft?: [number, number][] },
  color: RGBA,
  edges: Edge[],
) {
  const n = g.vertices.length;
  const pv = g.per_vertex_alt_ft && g.per_vertex_alt_ft.length === n ? g.per_vertex_alt_ft : null;
  const dash: RGBA = [color[0], color[1], color[2], 210];
  const DASHES = 7;
  for (let i = 0; i < n; i++) {
    const [lat, lon] = g.vertices[i];
    const base = (pv ? pv[i][0] : g.alt_min_ft ?? 0) * FT_TO_M;
    if (base <= 1) continue; // shape already sits on the ground — no gap to bridge
    for (let d = 0; d < DASHES; d++) {
      edges.push({
        src: [lon, lat, (base * d) / DASHES],
        tgt: [lon, lat, (base * (d + 0.5)) / DASHES],
        color: dash,
      });
    }
  }
}

export function waypointPosition(q: any): [number, number, number] {
  return [q.lon, q.lat, zMeters(q.alt_ft) + WAYPOINT_LIFT_M];
}

export function waypointDiscVertices(q: any, z: number, radiusNm: number): number[][] {
  const lat = Number(q.lat);
  const lon = Number(q.lon);
  const cosLat = Math.max(0.01, Math.cos((lat * Math.PI) / 180));
  return Array.from({ length: TOLERANCE_DISC_VERTICES }, (_, i) => {
    const a = (2 * Math.PI * i) / TOLERANCE_DISC_VERTICES;
    const dLat = (radiusNm / 60) * Math.cos(a);
    const dLon = ((radiusNm / 60) * Math.sin(a)) / cosLat;
    return [lon + dLon, lat + dLat, z];
  });
}

export function addWaypointStem(q: any, edges: WaypointEdge[]) {
  if (!Number.isFinite(q.alt_ft) || q.alt_ft <= 0) return;
  edges.push({
    name: q.name,
    target: q.target,
    ident: q.ident,
    alt_ft: q.alt_ft,
    speed_kts: q.speed_kts,
    reach_radius_nm: q.reach_radius_nm,
    alt_tolerance_ft: q.alt_tolerance_ft,
    speed_tolerance_kts: q.speed_tolerance_kts,
    src: [q.lon, q.lat, 0],
    tgt: waypointPosition(q),
    color: [...WAYPOINT_STEM_COLOR],
  });
}

export function waypointToleranceGeometry(
  q: any,
  faces: WaypointFace[],
  edges: WaypointEdge[],
) {
  const radiusNm = Number.isFinite(q.reach_radius_nm) && q.reach_radius_nm > 0 ? q.reach_radius_nm : DEFAULT_WAYPOINT_REACH_NM;
  const color = cssToRgb(q.color);
  const props: WaypointShape = {
    name: q.name,
    target: q.target,
    ident: q.ident,
    alt_ft: q.alt_ft,
    speed_kts: q.speed_kts,
    reach_radius_nm: radiusNm,
    alt_tolerance_ft: q.alt_tolerance_ft,
    speed_tolerance_kts: q.speed_tolerance_kts,
  };
  const baseAlt = Number.isFinite(q.alt_ft) ? q.alt_ft : 0;
  const hasHeight = Number.isFinite(q.alt_tolerance_ft) && q.alt_tolerance_ft > 0 && Number.isFinite(q.alt_ft);
  const z0 = zMeters(hasHeight ? baseAlt - q.alt_tolerance_ft : baseAlt) + WAYPOINT_LIFT_M;
  const z1 = zMeters(hasHeight ? baseAlt + q.alt_tolerance_ft : baseAlt) + WAYPOINT_LIFT_M;
  const bottom = waypointDiscVertices(q, z0, radiusNm);
  const top = hasHeight ? waypointDiscVertices(q, z1, radiusNm) : bottom;
  faces.push({ ...props, polygon: bottom, color: [color[0], color[1], color[2], 1] });
  if (hasHeight) {
    faces.push({ ...props, polygon: top, color: [color[0], color[1], color[2], 1] });
  }
  for (let i = 0; i < bottom.length; i++) {
    const j = (i + 1) % bottom.length;
    edges.push({ ...props, src: bottom[i], tgt: bottom[j], color });
    if (hasHeight) {
      edges.push({ ...props, src: top[i], tgt: top[j], color });
      if (i % 9 === 0) edges.push({ ...props, src: bottom[i], tgt: top[i], color });
    }
  }
}

export function latLonObj(lat: number, lon: number) {
  return { lat_deg: lat, lon_deg: lon };
}

export function footprintCoords(fp: SpecDict): [number, number][] {
  // Sampled params (range/dist dicts) render at their representative value,
  // so a per-episode-shaped region still shows on the map.
  if (!fp) return [];
  if (fp.type === "box") {
    return [
      [repValue(fp.lat_min_deg), repValue(fp.lon_min_deg)],
      [repValue(fp.lat_min_deg), repValue(fp.lon_max_deg)],
      [repValue(fp.lat_max_deg), repValue(fp.lon_max_deg)],
      [repValue(fp.lat_max_deg), repValue(fp.lon_min_deg)],
    ];
  }
  if (fp.type === "polygon") return fp.coords ?? [];
  if (fp.type === "disk" && fp.center && Number.isFinite(repValue(fp.radius_nm))) {
    return radialVertices(fp.center, repValue(fp.radius_nm), fp.n_vertices ?? 72);
  }
  if (fp.type === "sector" && fp.center && Number.isFinite(repValue(fp.radius_nm))) {
    const n = fp.n_vertices ?? 24;
    const bearing = repValue(fp.bearing_deg);
    const half = repValue(fp.half_angle_deg);
    return [
      [fp.center.lat_deg, fp.center.lon_deg],
      ...Array.from({ length: n + 1 }, (_, i) =>
        offsetLatLon(fp.center, bearing - half + (2 * half * i) / n, repValue(fp.radius_nm)),
      ),
    ];
  }
  if (fp.type === "annular_sector" && fp.center && Number.isFinite(repValue(fp.outer_radius_nm))) {
    const n = fp.n_vertices ?? 48;
    const bearing = repValue(fp.bearing_deg);
    const half = repValue(fp.half_angle_deg);
    const start = bearing - half;
    const end = bearing + half;
    const step = (end - start) / n;
    const inner = Array.from({ length: n + 1 }, (_, i) => offsetLatLon(fp.center, start + step * i, repValue(fp.inner_radius_nm)));
    const outer = Array.from({ length: n + 1 }, (_, i) => offsetLatLon(fp.center, end - step * i, repValue(fp.outer_radius_nm)));
    return [...inner, ...outer];
  }
  if (fp.type === "boolean") return [...footprintCoords(fp.left), ...footprintCoords(fp.right)];
  if (fp.center) return [[fp.center.lat_deg, fp.center.lon_deg]];
  return [];
}

export function boundsCenter(bounds: SpecDict): [number, number] | null {
  const fp = bounds?.footprint;
  if (!fp) return null;
  const coords = footprintCoords(fp);
  if (!coords.length) return null;
  const lats = coords.map(([lat]) => lat);
  const lons = coords.map(([, lon]) => lon);
  return [
    (Math.min(...lats) + Math.max(...lats)) / 2,
    (Math.min(...lons) + Math.max(...lons)) / 2,
  ];
}

export function boundsRadiusDeg(bounds: SpecDict): number {
  const center = boundsCenter(bounds);
  if (!center) return 0.1;
  const coords = footprintCoords(bounds?.footprint);
  const radius = Math.max(
    ...coords.map(([lat, lon]) => Math.hypot(lat - center[0], lon - center[1])),
    0.06,
  );
  return Number.isFinite(radius) ? radius : 0.1;
}

export function frontendBoundsGeometry(bounds: SpecDict): {
  vertices: [number, number][];
  alt_min_ft?: number;
  alt_max_ft?: number;
  per_vertex_alt_ft?: [number, number][];
} | null {
  const fp = bounds?.footprint;
  if (!fp) return null;
  // Compute the altitude profile on the unrotated footprint vertices; rotation
  // only moves lat/lon, so the per-vertex bands stay index-aligned afterwards.
  const rawVertices = footprintCoords(fp);
  const center = boundsCenter(bounds);
  const vertices =
    center && bounds.rotation_deg
      ? rawVertices.map(([lat, lon]) => rotateLatLon(lat, lon, center, bounds.rotation_deg))
      : rawVertices;
  // An overall band (every altitude type) so the selected region is always drawn
  // elevated; the per-vertex profile (radial / linear / vertex) refines that to
  // match the backend preview's sloped volume when it lines up vertex-for-vertex.
  const band = altRange(bounds.altitude);
  return {
    vertices,
    alt_min_ft: band ? band[0] : undefined,
    alt_max_ft: band ? band[1] : undefined,
    per_vertex_alt_ft: perVertexAltFt(bounds.altitude, rawVertices) ?? undefined,
  };
}

// Per-vertex [min, max] altitude bands matching the backend's altitude models,
// so a selected region's drawn volume follows its radial/linear/vertex profile
// instead of collapsing to a flat band. Returns null for constant/none (the flat
// alt_min/alt_max band already covers those).
function perVertexAltFt(alt: SpecDict | null | undefined, vertices: [number, number][]): [number, number][] | null {
  if (!alt) return null;
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  const mix = (lo: [number, number], hi: [number, number], t: number): [number, number] => [
    lerp(lo[0], hi[0], t),
    lerp(lo[1], hi[1], t),
  ];
  if (alt.type === "linear") {
    const s = alt.start;
    const e = alt.end;
    const cosLat = Math.max(0.01, Math.cos((s.lat_deg * Math.PI) / 180));
    const ex = (e.lon_deg - s.lon_deg) * cosLat;
    const ey = e.lat_deg - s.lat_deg;
    const len2 = ex * ex + ey * ey || 1;
    return vertices.map(([lat, lon]) => {
      const vx = (lon - s.lon_deg) * cosLat;
      const vy = lat - s.lat_deg;
      const t = Math.max(0, Math.min(1, (vx * ex + vy * ey) / len2));
      return mix(alt.start_band_ft, alt.end_band_ft, t);
    });
  }
  if (alt.type === "radial") {
    const c = alt.center;
    const cosLat = Math.max(0.01, Math.cos((c.lat_deg * Math.PI) / 180));
    return vertices.map(([lat, lon]) => {
      const dNm = Math.hypot(lat - c.lat_deg, (lon - c.lon_deg) * cosLat) * 60;
      const t = alt.radius_nm > 0 ? Math.max(0, Math.min(1, dNm / alt.radius_nm)) : 0;
      return mix(alt.inner_band_ft, alt.outer_band_ft, t);
    });
  }
  if (alt.type === "vertex") {
    const mins = Array.isArray(alt.min_values_ft) ? alt.min_values_ft : null;
    const maxs = Array.isArray(alt.max_values_ft) ? alt.max_values_ft : null;
    // Only meaningful when the value arrays align with the footprint vertices;
    // otherwise regionGeometry falls back to the flat band.
    if (!mins || !maxs || mins.length !== vertices.length || maxs.length !== vertices.length) return null;
    return vertices.map((_, i) => [Number(mins[i]), Number(maxs[i])]);
  }
  return null;
}

export function rotateLatLon(lat: number, lon: number, center: [number, number], angleDeg: number): [number, number] {
  const angle = (angleDeg * Math.PI) / 180;
  const cosLat = Math.max(0.01, Math.cos((center[0] * Math.PI) / 180));
  const x = (lon - center[1]) * cosLat;
  const y = lat - center[0];
  const xr = x * Math.cos(angle) - y * Math.sin(angle);
  const yr = x * Math.sin(angle) + y * Math.cos(angle);
  return [center[0] + yr, center[1] + xr / cosLat];
}

export function inverseRotateLatLon(lat: number, lon: number, center: [number, number], angleDeg: number): [number, number] {
  return rotateLatLon(lat, lon, center, -angleDeg);
}

export function offsetLatLon(center: { lat_deg: number; lon_deg: number }, bearingDeg: number, distanceNm: number): [number, number] {
  const angle = (bearingDeg * Math.PI) / 180;
  const cosLat = Math.max(0.01, Math.cos((center.lat_deg * Math.PI) / 180));
  return [
    center.lat_deg + (distanceNm / 60) * Math.cos(angle),
    center.lon_deg + ((distanceNm / 60) * Math.sin(angle)) / cosLat,
  ];
}

export function radialVertices(center: { lat_deg: number; lon_deg: number }, radiusNm: number, n: number): [number, number][] {
  return Array.from({ length: n }, (_, i) => offsetLatLon(center, (360 * i) / n, radiusNm));
}

// Shift a (possibly sampled) scalar param by a delta: a range translates both
// endpoints, a scipy dist translates its loc - so map move-drags stay
// meaningful on per-episode-sampled edges instead of corrupting the dict.
function shiftSampled(v: any, d: number): any {
  if (typeof v === "number") return v + d;
  if (v && typeof v === "object") {
    if (v.type === "range") return { ...v, low: Number(v.low) + d, high: Number(v.high) + d };
    if (v.type === "scipy") return { ...v, kwds: { ...(v.kwds ?? {}), loc: Number(v.kwds?.loc ?? 0) + d } };
  }
  return v;
}

export function moveFootprint(fp: SpecDict, dLat: number, dLon: number) {
  if (fp.type === "box") {
    fp.lat_min_deg = shiftSampled(fp.lat_min_deg, dLat);
    fp.lat_max_deg = shiftSampled(fp.lat_max_deg, dLat);
    fp.lon_min_deg = shiftSampled(fp.lon_min_deg, dLon);
    fp.lon_max_deg = shiftSampled(fp.lon_max_deg, dLon);
  } else if (fp.type === "polygon") {
    fp.coords = (fp.coords ?? []).map(([lat, lon]: [number, number]) => [lat + dLat, lon + dLon]);
  } else if (fp.type === "boolean") {
    moveFootprint(fp.left, dLat, dLon);
    moveFootprint(fp.right, dLat, dLon);
  } else if (fp.center) {
    fp.center = latLonObj(fp.center.lat_deg + dLat, fp.center.lon_deg + dLon);
  }
}

export function latLonBboxCenter(points: [number, number][]): [number, number] | null {
  if (!points.length) return null;
  const lats = points.map(([lat]) => lat);
  const lons = points.map(([, lon]) => lon);
  return [
    (Math.min(...lats) + Math.max(...lats)) / 2,
    (Math.min(...lons) + Math.max(...lons)) / 2,
  ];
}

export function solveRawPointsForDisplay(
  displayPoints: [number, number][],
  rotationDeg: number,
  initialCenter: [number, number],
): [number, number][] {
  let center = initialCenter;
  let rawPoints = displayPoints;
  for (let i = 0; i < 8; i++) {
    rawPoints = displayPoints.map(([lat, lon]) => inverseRotateLatLon(lat, lon, center, rotationDeg));
    const nextCenter = latLonBboxCenter(rawPoints);
    if (!nextCenter) break;
    if (Math.hypot(nextCenter[0] - center[0], nextCenter[1] - center[1]) < 1e-10) break;
    center = nextCenter;
  }
  return rawPoints;
}

export function boxCorner(fp: SpecDict, index: number): [number, number] {
  switch (index) {
    case 0:
      return [repValue(fp.lat_max_deg), repValue(fp.lon_min_deg)];
    case 1:
      return [repValue(fp.lat_max_deg), repValue(fp.lon_max_deg)];
    case 2:
      return [repValue(fp.lat_min_deg), repValue(fp.lon_max_deg)];
    default:
      return [repValue(fp.lat_min_deg), repValue(fp.lon_min_deg)];
  }
}

export function updateRotatedBoxCorner(
  fp: SpecDict,
  startFp: SpecDict,
  index: number,
  pointerLat: number,
  pointerLon: number,
  rotationDeg: number,
  startCenter: [number, number],
) {
  const oppositeIndex = (index + 2) % 4;
  const oppositeRaw = boxCorner(startFp, oppositeIndex);
  const oppositeDisplay = rotateLatLon(oppositeRaw[0], oppositeRaw[1], startCenter, rotationDeg);
  const rawPoints = solveRawPointsForDisplay(
    [oppositeDisplay, [pointerLat, pointerLon]],
    rotationDeg,
    [
      (oppositeDisplay[0] + pointerLat) / 2,
      (oppositeDisplay[1] + pointerLon) / 2,
    ],
  );
  const [a, b] = rawPoints;
  fp.lat_min_deg = Math.min(a[0], b[0]);
  fp.lat_max_deg = Math.max(a[0], b[0]);
  fp.lon_min_deg = Math.min(a[1], b[1]);
  fp.lon_max_deg = Math.max(a[1], b[1]);
}

// Resolve the spec's routes (named library + global/region fixed arrays) into
// drawable polylines through the previewed waypoint positions. A route is kept
// only if at least two of its waypoints resolve to a position.
// Map of waypoint name → drawable position. A sampled waypoint's drawn position
// varies per seed/aircraft, so it's anchored at its sample region's centroid — a
// stable, representative point (the region itself shows the spread).
export function waypointPositions(spec: SpecDict | null, preview: PreviewResult | null): Map<string, [number, number, number]> {
  const positions = new Map<string, [number, number, number]>();
  if (!preview) return positions;
  // Per-aircraft sampled waypoints have no single point; their preview lat/lon
  // is the meaningless template, so skip them here (region-centroid anchored
  // below). Per-episode sampled waypoints DO have their episode position -
  // that must win over any centroid fallback, or every fix collapses onto its
  // sample region's centre and route polylines bend through the wrong place.
  for (const q of preview.queryables) {
    if (
      q.kind === "waypoint" &&
      !(q as any).sample_per_aircraft &&
      Number.isFinite(q.lon) &&
      Number.isFinite(q.lat)
    ) {
      positions.set(q.name, waypointPosition(q));
    }
  }
  for (const [name, q] of Object.entries(spec?.queryables ?? {}) as [string, any][]) {
    if (q?.type !== "waypoint" || !q.sample || positions.has(name)) continue;
    // Fallback anchor (per-aircraft waypoints, or a preview without this
    // waypoint): the sample region's centroid, preferring the sampled-episode
    // region geometry from the preview over the canonical spec shape.
    const ref = q.sample.ref;
    const sampledRegion = ref ? preview.regions?.[ref] : undefined;
    const region = ref ? spec?.regions?.[ref] : q.sample;
    const g = sampledRegion ?? (region?.footprint ? frontendBoundsGeometry(region) : null);
    if (!g?.vertices?.length) continue;
    const [clon, clat] = centroid(g.vertices);
    const z = zMeters(Number.isFinite(repValue(q.alt_ft)) ? repValue(q.alt_ft) : 0) + WAYPOINT_LIFT_M;
    positions.set(name, [clon, clat, z]);
  }
  return positions;
}

// The entry waypoint name(s) of a route spec — the first waypoint reachable along
// each limb (descending subroutes / branches / sampled-route choices). Used to
// connect a spawn region to where its route begins.
export function routeEntryWaypoints(
  routeSpec: any,
  routes: Record<string, any>,
  seen: Set<string> = new Set(),
): string[] {
  if (routeSpec == null) return [];
  if (typeof routeSpec === "string") {
    if (seen.has(routeSpec) || !routes[routeSpec]) return [];
    return routeEntryWaypoints(routes[routeSpec], routes, new Set([...seen, routeSpec]));
  }
  if (!Array.isArray(routeSpec)) {
    if (routeSpec.type === "categorical") {
      return Object.keys(routeSpec.weights ?? {}).flatMap((n) => routeEntryWaypoints(n, routes, seen));
    }
    return [];
  }
  for (const s of routeSpec) {
    if (typeof s === "string") return [s];
    if (s && typeof s === "object" && typeof s.waypoint === "string") return [s.waypoint];
    if (s && typeof s === "object" && typeof s.route === "string") {
      const r = routeEntryWaypoints(s.route, routes, seen);
      if (r.length) return r;
      continue;
    }
    if (s && typeof s === "object" && Array.isArray(s.choice)) {
      const r = s.choice.flatMap((b: any) => routeEntryWaypoints(Array.isArray(b) ? b : [b], routes, seen));
      if (r.length) return r;
      continue;
    }
  }
  return [];
}

// The RoutePath key(s) a spawn region flies, paired with that route's entry
// waypoints — so a connector can be coloured to match the route polyline. Keys
// mirror those `routePaths` assigns: named → ``route:<name>``, a region's fixed
// list → ``route:__region_<i>__``, the global fixed list → ``route:__global__``.
function spawnRouteKeys(region: any, index: number, spec: SpecDict): { key: string; entries: string[] }[] {
  const routes: Record<string, any> = spec.spawn?.routes ?? {};
  const named = (name: string) => ({ key: `route:${name}`, entries: routeEntryWaypoints(name, routes) });
  const resolve = (rs: any, fixedKey: string): { key: string; entries: string[] }[] => {
    if (rs == null) return [];
    if (Array.isArray(rs)) return [{ key: fixedKey, entries: routeEntryWaypoints(rs, routes) }];
    if (typeof rs === "string") return [named(rs)];
    if (rs.type === "categorical") return Object.keys(rs.weights ?? {}).map(named);
    return [];
  };
  return region?.route != null
    ? resolve(region.route, `route:__region_${index}__`)
    : resolve(spec.spawn?.route, "route:__global__");
}

// One connector per spawn region → the entry waypoint(s) of the route it flies
// (its own route override, else the global route), tagged with the route key so
// it can be drawn in the route's colour, so a route reads as a full path from
// spawn to goal on the map.
export function spawnRouteLinks(
  spec: SpecDict | null,
  preview: PreviewResult | null,
): { src: [number, number, number]; tgt: [number, number, number]; spawnIndex: number; routeKey: string }[] {
  if (!spec || !preview) return [];
  const positions = waypointPositions(spec, preview);
  const links: { src: [number, number, number]; tgt: [number, number, number]; spawnIndex: number; routeKey: string }[] = [];
  (spec.spawn?.regions ?? []).forEach((region: any, index: number) => {
    const sr: any = preview.spawn_regions?.[index];
    if (!sr?.vertices?.length) return;
    const [olon, olat] = centroid(sr.vertices);
    for (const { key, entries } of spawnRouteKeys(region, index, spec)) {
      for (const name of entries) {
        const p = positions.get(name);
        if (p) links.push({ src: [olon, olat, 0], tgt: p, spawnIndex: index, routeKey: key });
      }
    }
  });
  return links;
}

export function routePaths(spec: SpecDict | null, preview: PreviewResult | null): RoutePath[] {
  if (!spec || !preview) return [];
  const positions = waypointPositions(spec, preview);
  const out: RoutePath[] = [];
  const seen = new Set<string>();
  let colorIndex = 0;
  const routes: Record<string, any> = spec.spawn?.routes ?? {};
  // Enumerate every concrete waypoint-name path through a (possibly branching)
  // route: {route: name} steps inline (cycle-guarded), {choice: [...]} branches
  // fan out into one path per branch. Mirrors spawn.expand_route_paths so the
  // whole STAR/SID network draws, not just one limb.
  const enumerate = (steps: any, seenRefs: Set<string> = new Set()): string[][] => {
    if (!Array.isArray(steps)) return [[]];
    let paths: string[][] = [[]];
    for (const s of steps) {
      if (s && typeof s === "object" && Array.isArray(s.choice)) {
        const branchPaths: string[][] = [];
        for (const branch of s.choice) {
          const branchSteps = Array.isArray(branch) ? branch : [branch];
          branchPaths.push(...enumerate(branchSteps, seenRefs));
        }
        paths = paths.flatMap((p) => branchPaths.map((bp) => [...p, ...bp]));
      } else if (s && typeof s === "object" && typeof s.route === "string") {
        if (!seenRefs.has(s.route) && routes[s.route]) {
          const sub = enumerate(routes[s.route], new Set([...seenRefs, s.route]));
          paths = paths.flatMap((p) => sub.map((sp) => [...p, ...sp]));
        }
      } else if (s && typeof s === "object" && typeof s.waypoint === "string") {
        // Constrained {waypoint: name, speed_kts?, alt_ft?} step: name only.
        paths = paths.map((p) => [...p, s.waypoint]);
      } else if (typeof s === "string") {
        paths = paths.map((p) => [...p, s]);
      }
    }
    return paths;
  };
  // Drop consecutive duplicate waypoints so a shared junction joins cleanly.
  const collapse = (names: string[]): string[] =>
    names.filter((n, i) => i === 0 || n !== names[i - 1]);
  const add = (key: string, name: string, steps: any) => {
    if (!Array.isArray(steps) || seen.has(key)) return;
    seen.add(key);
    // All limbs of one route share the key (so the eye toggle hides them
    // together) and colour, but are separate polylines.
    const color = ROUTE_PALETTE[colorIndex++ % ROUTE_PALETTE.length];
    const drawn = new Set<string>();
    for (const path of enumerate(steps)) {
      const points = collapse(path)
        .filter((n) => positions.has(n))
        .map((n) => positions.get(n)!);
      if (points.length < 2) continue;
      const ptKey = points.map((p) => `${p[0].toFixed(6)},${p[1].toFixed(6)}`).join("|");
      if (drawn.has(ptKey)) continue;
      drawn.add(ptKey);
      out.push({ key, name, color, points });
    }
  };

  for (const name of Object.keys(routes)) add(`route:${name}`, name, routes[name]);
  if (Array.isArray(spec.spawn?.route)) add("route:__global__", "global route", spec.spawn.route);
  (spec.spawn?.regions ?? []).forEach((region: SpecDict, index: number) => {
    if (Array.isArray(region.route)) {
      const name = region.name || `spawn_${index}`;
      add(`route:__region_${index}__`, `${name} route`, region.route);
    }
  });
  return out;
}
