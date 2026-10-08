// deck.gl layer construction for the map: bounds wireframes, waypoint tolerance
// volumes, route polylines + direction arrows, sampled aircraft, and nav
// context. Category visibility + per-element hide are applied here so the spec
// is never mutated by view toggles.
import { IconLayer, LineLayer, PathLayer, ScatterplotLayer, SolidPolygonLayer, TextLayer } from "@deck.gl/layers";
import type { NavFeatures, PreviewResult, SpawnedAircraft, SpawnTarget, SpecDict } from "../api";
import type {
  CategoryVisibility,
  Edge,
  EditTarget,
  Face,
  Mark,
  RGBA,
  RoutePath,
  Selectable,
  WaypointEdge,
  WaypointFace,
} from "./types";
import {
  NAMED,
  addHandleDropStems,
  atTime,
  motionTrack,
  cssToRgb,
  frontendShapeGeometry,
  regionGeometry,
  spawnRouteLinks,
  waypointPosition,
  waypointPositions,
  zMeters,
  addWaypointStem,
  waypointToleranceGeometry,
} from "./geometry";
import { shapesInView } from "./shapesInView";
import { waypointLatLon, waypointPoint } from "../waypointPoints";
import type { Pick } from "../episode";

// A bounds the selection is drawn, placed or moved by: faint, dashed.
const CONTEXT: RGBA = [148, 163, 184, 150];
import { shapeForTarget, groupBbox, groupMemberShapes, targetKey } from "./editHandles";

export const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

export const LABEL_FONT = ["Open Sans Regular"];

export const ROTATION_HANDLE_ICON =
  "data:image/svg+xml;charset=utf-8," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path d="M24.7 7.3A12 12 0 1 0 27.5 20h-5.2a7.4 7.4 0 1 1-1.2-9.2L17 15h12V3z"/></svg>',
  );

// A 4-way move glyph for the center "drag the whole shape" handle, so it reads
// differently from the corner/vertex resize handles and the rotate handle.
export const MOVE_HANDLE_ICON =
  "data:image/svg+xml;charset=utf-8," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 2l3.5 3.5h-2.5v5h5V8L21.5 11.5 18 15v-2.5h-5v5h2.5L12 21l-3.5-3.5H11v-5H6V15l-3.5-3.5L6 8v2.5h5v-5H8.5z"/></svg>',
  );

// An upward-pointing (north) triangle, tinted per-route via IconLayer mask.
const ROUTE_ARROW_ICON =
  "data:image/svg+xml;charset=utf-8," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 2 L21 21 L12 16 L3 21 Z"/></svg>',
  );

// Tooltip for any hovered deck object (nav fixes/airports, selected waypoints,
// aircraft, routes). Unified here because deck owns pointer events for rendered
// task overlays, so all hover info must come through deck.
// Tags on the map, and how they are drawn - shared with the map's declutter,
// which sizes them to find the ones that fit.
export const TAG_SIZE_PX = 12;
export const TAG_OFFSET_PX = 9;
export const TAG_LINE_HEIGHT = 1.2;

const three = (n: number) => String(Math.round(n)).padStart(3, "0");
const fl = (ft: number) => `FL${three(ft / 100)}`;

// An aircraft's tag: its pygame label, and its heading.
export const aircraftTag = (a: SpawnedAircraft) => [...a.label, `HDG${three(a.hdg_deg % 360)}`].join("\n");

// A target's tag: its name, then the window an aircraft must be in to reach
// it - altitude, speed, reach radius - for what the target constrains.
export function targetTag(t: SpawnTarget): string {
  const window = (at: number, tol: number | null, show: (v: number) => string, unit = "") =>
    tol ? `${show(at - tol)}–${show(at + tol)}${unit}` : `${show(at)}${unit}`;
  const parts: string[] = [];
  if (t.alt_ft != null) parts.push(window(t.alt_ft, t.alt_tolerance_ft, fl));
  if (t.speed_kts != null) {
    parts.push(window(t.speed_kts, t.speed_tolerance_kts, (v) => String(Math.round(v)), " kt"));
    if (t.speed_tolerance_mach) parts.push(`M±${t.speed_tolerance_mach}`.replace("0.", "."));
  }
  if (t.reach_radius_nm) parts.push(`r ${t.reach_radius_nm} nm`);
  return [t.waypoint ?? "target", parts.join(" · ")].filter(Boolean).join("\n");
}

export const spawnTargetKey = (t: SpawnTarget) => `target:${t.waypoint}:${t.lat.toFixed(5)}:${t.lon.toFixed(5)}:${t.alt_ft}`;

// The targets of every aircraft, each once.
export function uniqueTargets(flown: SpawnedAircraft[]): SpawnTarget[] {
  const seen = new Map<string, SpawnTarget>();
  for (const a of flown) for (const t of a.targets) seen.set(spawnTargetKey(t), t);
  return [...seen.values()];
}

export function getTooltip({ object }: any) {
  if (!object) return null;
  if (object.points) return { text: `route: ${object.name}` };
  if (object.routeName) return { text: `route: ${object.routeName}` };
  if (object.awid) return { text: `${object.awid}  ${object.from_id} → ${object.to_id}` };
  if (object.icao) return { text: `${object.icao}${object.name ? " — " + object.name : ""}` };
  // An aircraft as flown: its full label, as the pygame view gives it.
  if (object.callsign && object.label)
    return {
      text: [
        ...object.label,
        `HDG${String(Math.round(object.hdg_deg) % 360).padStart(3, "0")}  spawned t+${Math.round(object.time_s)} s`,
        object.controlled ? "controlled" : "background traffic",
        ...(object.targets ?? []).map((t: SpawnTarget) => `→ ${targetTag(t).replace("\n", "  ")}`),
      ].join("\n"),
    };
  if (object.actype) return { text: `${object.actype}  ${Math.round(object.alt_ft)} ft` };
  const id = object.ident ?? object.name;
  if (id) {
    if (object.role === "rotate") return { text: `${id}  rotate` };
    const typ = object.wptype ? `  (${object.wptype})` : "";
    const alt = Number.isFinite(object.alt_ft) ? `  ${Math.round(object.alt_ft)} ft` : "";
    return { text: `${id}${typ}${alt}` };
  }
  return null;
}

// Bearing of a segment as an IconLayer angle (CCW degrees) so a north-pointing
// icon points along the route direction when drawn flat on the ground plane.
function segmentAngle(a: [number, number, number], b: [number, number, number]): number {
  const cosLat = Math.max(0.01, Math.cos((a[1] * Math.PI) / 180));
  const dEast = (b[0] - a[0]) * cosLat;
  const dNorth = b[1] - a[1];
  const bearing = (Math.atan2(dEast, dNorth) * 180) / Math.PI; // clockwise from north
  return -bearing;
}

// A ring about the picked aircraft: the one whose observation is sampled.
function pickRing(at: [number, number, number][]) {
  return new ScatterplotLayer({
    id: "aircraft-picked",
    data: at,
    getPosition: (p: [number, number, number]) => p,
    filled: false,
    stroked: true,
    getLineColor: [255, 255, 255, 255],
    lineWidthMinPixels: 2,
    getRadius: 9,
    radiusUnits: "pixels",
    billboard: true,
    parameters: { depthTest: false },
  });
}

export function deckLayers(
  preview: PreviewResult,
  nav: NavFeatures | null,
  onSelect: (target: EditTarget | null) => void,
  draftSpec: SpecDict | null,
  selectedTarget: EditTarget | null,
  routes: RoutePath[],
  highlightedRoute: string | null,
  visibility: CategoryVisibility,
  hidden: Set<string>,
  lineScale: number,
  // The live spec (draft during a drag, else the committed spec). Used to draw
  // standalone named bounds; draftSpec alone is null outside a drag.
  spec: SpecDict | null = draftSpec,
  // The episode's aircraft as the environment created them, when its run is
  // in: drawn in place of the preview's draw, with a heading leader each.
  flown: SpawnedAircraft[] | null = null,
  // The callsigns whose tags fit without overlapping; null tags them all.
  tagged: Set<string> | null = null,
  // Seconds into the episode: moving regions are drawn as they are then.
  motionT = 0,
  // The picked aircraft's key (its callsign, or `preview:<i>` before the run
  // names it), ringed; and how a click on an aircraft picks it.
  picked: string | null = null,
  onPickAircraft: ((pick: Pick) => void) | null = null,
) {
  // A preview shape at the time shown, with - if it moves - its track.
  const timed = <G extends { vertices: [number, number][] }>(g: G, color: RGBA, meta: Selectable): G => {
    motionTrack(g as any, color, out.edges, meta);
    return atTime(g as any, motionT) as G;
  };
  const shown = (key: string) => !hidden.has(key);
  // The selected element is drawn from canonical (unrotated) geometry below so
  // its edit handles line up; skip its rotated preview copy here. (Per-episode
  // rotation groups rotate the preview, but handles edit the canonical design.)
  const selKey = selectedTarget ? targetKey(selectedTarget) : null;
  const isSel = (t: EditTarget) => selKey !== null && targetKey(t) === selKey;
  // Global line-width multiplier (the "lines" overlay control).
  const lw = (w: number) => w * lineScale;
  const out = { edges: [] as Edge[], faces: [] as Face[], marks: [] as Mark[] };
  const draftOut = { edges: [] as Edge[], faces: [] as Face[], marks: [] as Mark[] };
  const waypointOut = { edges: [] as WaypointEdge[], faces: [] as WaypointFace[] };
  const draftShape = draftSpec && selectedTarget ? shapeForTarget(draftSpec, selectedTarget) : null;
  if (draftShape) {
    const g = frontendShapeGeometry(draftShape);
    if (g && selectedTarget) regionGeometry(g, [255, 255, 255, 255], draftOut, { target: selectedTarget, name: "draft" });
  }
  if (preview.airspace && visibility.airspace && shown("airspace") && !isSel({ scope: "airspace" })) {
    // The airspace is a role a shape plays: clicking it picks that shape.
    const ref = spec?.airspace?.ref;
    const meta = {
      name: ref ?? "airspace",
      target: (typeof ref === "string" ? { scope: "region", name: ref } : { scope: "airspace" }) as EditTarget,
    };
    regionGeometry(timed(preview.airspace, NAMED.blue, meta), NAMED.blue, out, meta);
  }
  const wpts = visibility.waypoints
    ? preview.queryables
        // Per-aircraft sampled waypoints have no single fixed point — each
        // aircraft draws its own target (shown as rings), so skip the static marker.
        .filter((q) => q.kind === "waypoint" && q.render_shape !== false && !(q as any).sample_per_aircraft && shown(`queryable:${q.name}`))
        // Envelope-altitude waypoints carry no alt_ft; use the preview's
        // representative display altitude so the marker (stem + tolerance
        // volume) renders at a sensible height instead of on the ground.
        .map((q: any) => ({
          ...q,
          alt_ft: q.alt_ft ?? q.display_alt_ft,
          target: { scope: "queryable", name: q.name },
        }))
    : [];
  if (visibility.waypoints && draftSpec && selectedTarget?.scope === "queryable") {
    const draftQ = draftSpec.queryables?.[selectedTarget.name];
    const at = draftQ?.type === "waypoint" ? waypointLatLon(draftSpec, draftQ) : null;
    if (at) {
      wpts.push({
        ...draftQ,
        lat: at[0],
        lon: at[1],
        kind: "waypoint",
        name: selectedTarget.name,
        ident: waypointPoint(draftSpec, draftQ)?.footprint?.fix ?? draftQ.waypoint,
        target: { scope: "queryable", name: selectedTarget.name },
        color: "white",
      });
    }
  }
  for (const q of wpts) addWaypointStem(q, waypointOut.edges);
  for (const q of wpts) waypointToleranceGeometry(q, waypointOut.faces, waypointOut.edges);
  if (visibility.regions) {
    for (const q of preview.queryables.filter((q) => q.kind === "region" && q.render_shape !== false && shown(`queryable:${q.name}`))) {
      if (isSel({ scope: "queryable", name: q.name })) continue;
      const meta = { name: q.name, target: { scope: "queryable", name: q.name } as EditTarget };
      regionGeometry(timed(q, cssToRgb(q.color), meta), cssToRgb(q.color), out, meta);
    }
  }
  // Spawn regions are bounds too - render them as wireframes (green), like the
  // airspace and queryable regions.
  if (visibility.spawnRegions) {
    for (const [index, r] of preview.spawn_regions.entries()) {
      if (r.render_shape === false || !shown(`spawn:${index}`)) continue;
      if (isSel({ scope: "spawn", index })) continue;
      const meta = { name: r.name, target: { scope: "spawn", index } as EditTarget };
      regionGeometry(timed(r, NAMED.green, meta), NAMED.green, out, meta);
    }
  }
  // Named bounds not already drawn by a consumer (airspace / query-region /
  // spawn) — e.g. freshly created, or used only as a waypoint sample area — are
  // drawn standalone (neutral), with the "bounds" layer on or when the
  // selection brings them in (see shapesInView); what those depend on, faint
  // and dashed.
  if (spec) {
    const inView = shapesInView(spec, selectedTarget);
    const refName = (b: any): string | null => (b && typeof b.ref === "string" ? b.ref : null);
    const drawn = new Set<string>();
    const a = refName(spec.airspace);
    if (a) drawn.add(a);
    for (const q of Object.values(spec.queryables ?? {}) as any[]) {
      if (q?.type === "query_region") {
        const r = refName(q.shape);
        if (r) drawn.add(r);
      }
    }
    for (const r of (spec.spawn?.regions ?? []) as any[]) {
      const rb = refName(r?.shape);
      if (rb) drawn.add(rb);
    }
    for (const [name, bounds] of Object.entries(spec.shapes ?? {}) as [string, SpecDict][]) {
      if (!shown(`region:${name}`)) continue;
      if (!visibility.shapes && !inView.own.has(name)) {
        // Context: where the selection may go - unless an element draws it.
        if (!inView.context.has(name) || drawn.has(name)) continue;
        const g = preview.shapes?.[name]?.envelope ?? preview.shapes?.[name] ?? frontendShapeGeometry(bounds);
        if (g) regionGeometry(g as any, CONTEXT, out, { name, target: { scope: "region", name } }, "dashed");
        continue;
      }
      // A generated region: where its draws can fall, faint, behind whatever
      // draws this episode's shape - and a partition as its shapes, name.0 ...
      const envelope = preview.shapes?.[name]?.envelope;
      if (envelope) {
        regionGeometry(envelope, [148, 163, 184, 45], out, {
          name: `${name} (any draw)`,
          target: { scope: "region", name },
        });
        const parts = Object.values(preview.shapes ?? {}).filter(
          (r: any) => r.generated === name && r.name !== name,
        ) as any[];
        for (const part of parts) {
          regionGeometry(part, NAMED.slate, out, { name: part.name, target: { scope: "region", name } });
        }
        if (parts.length) continue;
      }
      if (drawn.has(name)) continue;
      // Prefer the sampled-episode geometry from the preview (shape draw +
      // rotation) so per-episode-randomized regions render as an episode
      // would place them; the selected region stays canonical so its edit
      // handles line up with the panel's parametric shape, with the sampled
      // copy kept visible as a faint ghost.
      const rawSampled = preview.shapes?.[name];
      const meta = { name, target: { scope: "region", name } as EditTarget };
      const sampled = rawSampled ? timed(rawSampled, NAMED.slate, meta) : undefined;
      const selected = isSel({ scope: "region", name });
      const g = (!selected && sampled) || frontendShapeGeometry(bounds);
      if (g) regionGeometry(g, NAMED.slate, out, meta);
      // Selected: the shape you edit is solid; this episode's draw dashed.
      if (selected && sampled) {
        regionGeometry(sampled, NAMED.slate, out, { name: `${name} (this episode)`, target: meta.target }, "dashed");
      }
    }
  }
  // The selected airspace/query-region/spawn bounds: draw it from the canonical
  // (unrotated) design so the edit handles align even while a rotation group
  // rotates the rest of the preview. (Standalone `region` scope is already drawn
  // canonical above; waypoints have no bounds to draw here.)
  const selDropStems: Edge[] = [];
  if (spec && selectedTarget && selectedTarget.scope !== "region" && selectedTarget.scope !== "group") {
    const selShape = shapeForTarget(spec, selectedTarget);
    const g = selShape ? frontendShapeGeometry(selShape) : null;
    if (g) {
      const color =
        selectedTarget.scope === "airspace"
          ? NAMED.blue
          : selectedTarget.scope === "spawn"
            ? NAMED.green
            : cssToRgb(spec.queryables?.[selectedTarget.name]?.color);
      regionGeometry(g, color, out, { name: "selected", target: selectedTarget });
      addHandleDropStems(g, color, selDropStems);
      // And this episode's draw of it, dashed - placed, generated, moving.
      const episode =
        selectedTarget.scope === "airspace"
          ? preview.airspace
          : selectedTarget.scope === "spawn"
            ? preview.spawn_regions[selectedTarget.index]
            : preview.queryables.find((q) => q.name === (selectedTarget as any).name && q.kind === "region");
      if (episode) {
        const meta = { name: "selected (this episode)", target: selectedTarget };
        regionGeometry(timed(episode as any, color, meta), color, out, meta, "dashed");
      }
    }
  }

  // A selected transform group: outline its members' bounding box (violet) and
  // draw each member's footprint highlighted, so the group's extent + the move/
  // rotate handles read as one unit. Drawn from the live (draft during drag) spec.
  if (spec && selectedTarget?.scope === "group") {
    const box = groupBbox(spec, selectedTarget.id);
    if (box) {
      const [clat, clon] = box.center;
      const hLat = Math.max(box.latSpan, 0.02) / 2 + 0.01;
      const hLon = Math.max(box.lonSpan, 0.02) / 2 + 0.01;
      const corners: [number, number][] = [
        [clat + hLat, clon - hLon], [clat + hLat, clon + hLon],
        [clat - hLat, clon + hLon], [clat - hLat, clon - hLon],
      ];
      const violet: RGBA = [...NAMED.violet];
      for (let i = 0; i < 4; i++) {
        const a = corners[i], b = corners[(i + 1) % 4];
        out.edges.push({ target: selectedTarget, name: "group", src: [a[1], a[0], 0], tgt: [b[1], b[0], 0], color: violet });
      }
      for (const bounds of groupMemberShapes(spec, selectedTarget.id)) {
        const g = frontendShapeGeometry(bounds);
        if (g) regionGeometry(g, violet, out, { target: selectedTarget, name: "group" });
      }
    }
  }

  // Spawn directions: when a spawn region is selected, draw a "pie" wedge out of
  // its center spanning the heading range (a full disk when unconstrained), plus
  // a mean-direction arrow.
  const spawnSector: { polygon: number[][]; color: RGBA }[] = [];
  const spawnSectorEdges: Edge[] = [];
  const spawnArrows: { position: number[]; angle: number; color: RGBA }[] = [];
  if (selectedTarget?.scope === "spawn") {
    const r: any = preview.spawn_regions[selectedTarget.index];
    if (r?.bounding_box) {
      const bb = r.bounding_box;
      const cLat = (bb.lat_min + bb.lat_max) / 2;
      const cLon = (bb.lon_min + bb.lon_max) / 2;
      const cosLat = Math.max(0.01, Math.cos((cLat * Math.PI) / 180));
      // A small pie around the center drag handle (not a region-sized wedge).
      const reach = (Math.max(bb.lat_max - bb.lat_min, (bb.lon_max - bb.lon_min) * cosLat) || 0.1) * 0.22;
      const z = zMeters(r.alt_min_ft ?? 0);
      const fill: RGBA = [120, 235, 150, 70];
      const edge: RGBA = [120, 235, 150, 255];
      const pt = (bearing: number): number[] => {
        const rad = (bearing * Math.PI) / 180;
        return [cLon + (Math.sin(rad) * reach) / cosLat, cLat + Math.cos(rad) * reach, z];
      };
      let [lo, hi] = (r.heading as [number, number] | null) ?? [0, 360];
      if (hi < lo) hi += 360; // wrap through north
      const span = hi - lo;
      const mean = (lo + hi) / 2;
      const arc: number[][] = [];
      const steps = Math.max(2, Math.ceil(span / 10));
      for (let i = 0; i <= steps; i++) arc.push(pt(lo + (span * i) / steps));
      if (span >= 359.5) {
        // Unconstrained: a full disk (no apex), outlined.
        spawnSector.push({ polygon: arc, color: fill });
        for (let i = 0; i + 1 < arc.length; i++) spawnSectorEdges.push({ src: arc[i], tgt: arc[i + 1], color: edge });
      } else if (span > 0.5) {
        const poly = [[cLon, cLat, z], ...arc];
        spawnSector.push({ polygon: poly, color: fill });
        for (let i = 0; i + 1 < poly.length; i++) spawnSectorEdges.push({ src: poly[i], tgt: poly[i + 1], color: edge });
        spawnSectorEdges.push({ src: poly[poly.length - 1], tgt: poly[0], color: edge }); // close apex
      } else {
        // Fixed heading: a single radius line + the arrow.
        spawnSectorEdges.push({ src: [cLon, cLat, z], tgt: pt(mean), color: edge });
      }
      spawnArrows.push({ position: pt(mean), angle: -mean, color: edge });
    }
  }

  const navLayers: any[] = [];
  if (nav && visibility.airways) {
    // Real-world airway network (BlueSky aw* legs) as faint connector lines,
    // drawn under the nav fixes so the dots stay legible.
    navLayers.push(
      new LineLayer({
        id: "nav-airways",
        data: nav.airways,
        getSourcePosition: (d: any) => [d.from_lon_deg, d.from_lat_deg, 0],
        getTargetPosition: (d: any) => [d.to_lon_deg, d.to_lat_deg, 0],
        getColor: [120, 150, 200, 130],
        getWidth: lw(1),
        widthUnits: "pixels",
        widthMinPixels: lw(1),
        pickable: true,
        parameters: { depthTest: false },
      }),
    );
  }
  if (nav && visibility.nav) {
    navLayers.push(
          new ScatterplotLayer({
            id: "nav-waypoints",
            data: nav.waypoints,
            getPosition: (w: any) => [w.lon_deg, w.lat_deg, 0],
            getFillColor: [225, 235, 250, 235],
            getLineColor: [40, 60, 90, 255],
            stroked: true,
            lineWidthMinPixels: 1,
            getRadius: 4,
            radiusUnits: "pixels",
            radiusMinPixels: 3,
            billboard: true,
            pickable: true,
            parameters: { depthTest: false },
          }),
          new ScatterplotLayer({
            id: "nav-airports",
            data: nav.airports,
            getPosition: (a: any) => [a.lon_deg, a.lat_deg, 0],
            getFillColor: [224, 160, 48, 255],
            getLineColor: [255, 255, 255, 255],
            stroked: true,
            lineWidthMinPixels: 1,
            getRadius: 4,
            radiusUnits: "pixels",
            billboard: true,
            pickable: true,
            parameters: { depthTest: false },
          }),
    );
  }
  const layers: any[] = [
    ...navLayers,
    new SolidPolygonLayer({
      id: "bound-faces",
      data: out.faces,
      getPolygon: (d: any) => d.polygon,
      getFillColor: (d: any) => d.color,
      pickable: true,
      onClick: (info: any) => {
        if (info.object?.target) onSelect(info.object.target);
        return true;
      },
      parameters: { depthTest: false },
    }),
    new LineLayer({
      id: "handle-drop-stems",
      data: selDropStems,
      getSourcePosition: (d: any) => d.src,
      getTargetPosition: (d: any) => d.tgt,
      getColor: (d: any) => d.color,
      getWidth: lw(1.5),
      widthUnits: "pixels",
      widthMinPixels: lw(1),
      parameters: { depthTest: false },
    }),
    new SolidPolygonLayer({
      id: "spawn-direction-sector",
      data: spawnSector,
      getPolygon: (d: any) => d.polygon,
      getFillColor: (d: any) => d.color,
      parameters: { depthTest: false },
    }),
    new LineLayer({
      id: "spawn-direction-edges",
      data: spawnSectorEdges,
      getSourcePosition: (d: any) => d.src,
      getTargetPosition: (d: any) => d.tgt,
      getColor: (d: any) => d.color,
      getWidth: lw(1.5),
      widthUnits: "pixels",
      widthMinPixels: lw(1),
      parameters: { depthTest: false },
    }),
    new IconLayer({
      id: "spawn-direction-arrows",
      data: spawnArrows,
      getPosition: (d: any) => d.position,
      getIcon: () => ({ url: ROUTE_ARROW_ICON, width: 24, height: 24, mask: true }),
      getSize: 18,
      sizeUnits: "pixels",
      getColor: (d: any) => d.color,
      getAngle: (d: any) => d.angle,
      billboard: false,
      parameters: { depthTest: false },
    }),
    new LineLayer({
      id: "bound-edges",
      data: out.edges,
      getSourcePosition: (d: any) => d.src,
      getTargetPosition: (d: any) => d.tgt,
      getColor: (d: any) => d.color,
      getWidth: lw(2.5),
      widthUnits: "pixels",
      widthMinPixels: lw(2),
      pickable: true,
      onClick: (info: any) => {
        if (info.object?.target) onSelect(info.object.target);
        return true;
      },
      parameters: { depthTest: false },
    }),
    // Point bounds: a ring with a dot, hollow for this episode's draw.
    new ScatterplotLayer({
      id: "bound-marks",
      data: [...out.marks, ...draftOut.marks.map((m) => ({ ...m, color: [255, 255, 255, 245] as RGBA }))],
      getPosition: (d: Mark) => d.position as [number, number, number],
      getFillColor: (d: Mark) => (d.dashed ? [0, 0, 0, 0] : [d.color[0], d.color[1], d.color[2], 90]),
      getLineColor: (d: Mark) => [d.color[0], d.color[1], d.color[2], 255],
      stroked: true,
      filled: true,
      getRadius: 7,
      radiusUnits: "pixels",
      lineWidthMinPixels: lw(2),
      billboard: true,
      pickable: true,
      onClick: (info: any) => {
        if (info.object?.target) onSelect(info.object.target);
        return true;
      },
      parameters: { depthTest: false },
    }),
    new SolidPolygonLayer({
      id: "draft-bound-faces",
      data: draftOut.faces,
      getPolygon: (d: any) => d.polygon,
      getFillColor: [255, 255, 255, 26],
      pickable: false,
      parameters: { depthTest: false },
    }),
    new LineLayer({
      id: "draft-bound-edges",
      data: draftOut.edges,
      getSourcePosition: (d: any) => d.src,
      getTargetPosition: (d: any) => d.tgt,
      getColor: [255, 255, 255, 245],
      getWidth: lw(3),
      widthUnits: "pixels",
      widthMinPixels: lw(2),
      pickable: false,
      parameters: { depthTest: false },
    }),
    // Waypoint tolerance: a real flat lat/lon disc at the waypoint altitude.
    // If altitude tolerance is set, render top/bottom discs and sparse vertical
    // edges so the tolerance has visible height.
    new SolidPolygonLayer({
      id: "waypoint-tolerance-faces",
      data: waypointOut.faces,
      getPolygon: (d: any) => d.polygon,
      getFillColor: (d: any) => d.color,
      pickable: true,
      onClick: (info: any) => {
        if (info.object?.target) onSelect(info.object.target);
        return true;
      },
      parameters: { depthTest: false },
    }),
    new LineLayer({
      id: "waypoint-tolerance-edges",
      data: waypointOut.edges,
      getSourcePosition: (d: any) => d.src,
      getTargetPosition: (d: any) => d.tgt,
      getColor: (d: any) => d.color,
      getWidth: lw(2),
      widthUnits: "pixels",
      widthMinPixels: lw(1.5),
      pickable: true,
      onClick: (info: any) => {
        if (info.object?.target) onSelect(info.object.target);
        return true;
      },
      parameters: { depthTest: false },
    }),
    new ScatterplotLayer({
      id: "waypoint-centers",
      data: wpts,
      getPosition: waypointPosition,
      getFillColor: [0, 0, 0, 0],
      getLineColor: (q: any) => cssToRgb(q.color),
      stroked: true,
      lineWidthMinPixels: lw(2),
      getRadius: 0.18 * 1852,
      radiusUnits: "meters",
      pickable: true,
      onClick: (info: any) => {
        if (info.object?.target) onSelect(info.object.target);
        return true;
      },
      parameters: { depthTest: false },
    }),
  ];

  // Spawn → route entry: a connector from each spawn region to the first
  // waypoint of the route it flies, so a route reads as a full path from spawn
  // to goal. Skipped for hidden spawn regions.
  if (spec && visibility.routes && visibility.spawnRegions) {
    const links = spawnRouteLinks(spec, preview).filter((l) => shown(`spawn:${l.spawnIndex}`));
    // Color each connector to match its route's polyline.
    const colorByKey = new Map(routes.map((r) => [r.key, r.color]));
    const linkColor = (l: any): RGBA => {
      const c = colorByKey.get(l.routeKey) ?? NAMED.green;
      const a = highlightedRoute == null ? 200 : c === colorByKey.get(`route:${highlightedRoute}`) ? 230 : 60;
      return [c[0], c[1], c[2], a];
    };
    if (links.length) {
      layers.push(
        new LineLayer({
          id: "spawn-route-links",
          data: links,
          getSourcePosition: (d: any) => d.src,
          getTargetPosition: (d: any) => d.tgt,
          getColor: linkColor,
          getWidth: lw(1.5),
          widthUnits: "pixels",
          widthMinPixels: lw(1),
          updateTriggers: { getColor: highlightedRoute },
          parameters: { depthTest: false },
        }),
      );
    }
  }

  // Routes: one polyline per route through its waypoints, plus direction arrows
  // at segment midpoints. Highlighted route is full strength; others dim.
  const visibleRoutes = visibility.routes ? routes.filter((r) => shown(r.key)) : [];
  if (visibleRoutes.length) {
    const alphaFor = (r: RoutePath): number =>
      highlightedRoute == null ? 220 : r.name === highlightedRoute ? 255 : 60;
    const widthFor = (r: RoutePath): number => lw(r.name === highlightedRoute ? 4.5 : 3);
    const arrows = visibleRoutes.flatMap((r) =>
      r.points.slice(0, -1).map((p, i) => {
        const q = r.points[i + 1];
        return {
          routeName: r.name,
          position: [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2],
          angle: segmentAngle(p, q),
          color: [r.color[0], r.color[1], r.color[2], alphaFor(r)] as RGBA,
        };
      }),
    );
    layers.push(
      new PathLayer({
        id: "route-paths",
        data: visibleRoutes,
        getPath: (d: RoutePath) => d.points,
        getColor: (d: RoutePath) => [d.color[0], d.color[1], d.color[2], alphaFor(d)],
        getWidth: widthFor,
        widthUnits: "pixels",
        widthMinPixels: lw(2),
        capRounded: true,
        jointRounded: true,
        pickable: true,
        parameters: { depthTest: false },
        updateTriggers: { getColor: highlightedRoute, getWidth: [highlightedRoute, lineScale] },
      }),
      new IconLayer({
        id: "route-arrows",
        data: arrows,
        getPosition: (d: any) => d.position,
        getIcon: () => ({ url: ROUTE_ARROW_ICON, width: 24, height: 24, mask: true }),
        getSize: 18,
        sizeUnits: "pixels",
        getColor: (d: any) => d.color,
        getAngle: (d: any) => d.angle,
        billboard: false,
        pickable: true,
        parameters: { depthTest: false },
        updateTriggers: { getColor: highlightedRoute },
      }),
    );
  }

  // Aircraft: a dot in space at the sampled altitude (toggleable).
  if (visibility.aircraft) {
    // Per-aircraft goal: a faint line to each aircraft's sampled target plus a
    // hollow ring at it, so stepping the episode shows the per-aircraft destination spread.
    // Once the episode run is in, its aircraft's own resolved targets are
    // drawn below instead of the preview's draw of them.
    const targeted = flown ? [] : preview.sampled_aircraft.filter((a: any) => a.target);
    const targetZ = (a: any) => zMeters(a.target.alt_ft ?? a.alt_ft);
    // Episode positions of the shared (per-episode sampled) waypoints, for
    // threading each aircraft's goal line through its intermediate fixes.
    const wpPositions = waypointPositions(spec, preview);
    // Each sampled target drawn as a *waypoint*: a reach-radius tolerance disc
    // in world units and the waypoint's color. The pixel-space ring below
    // stays as the zoomed-out affordance, but at working zoom the disc is what
    // makes the per-aircraft sampled waypoints actually visible.
    const targetShapes = { edges: [] as WaypointEdge[], faces: [] as WaypointFace[] };
    for (const a of targeted as any[]) {
      const t = a.target;
      if (t.reach_radius_nm == null) continue;
      waypointToleranceGeometry(
        {
          name: t.name ?? "target",
          lat: t.lat,
          lon: t.lon,
          alt_ft: t.alt_ft,
          reach_radius_nm: t.reach_radius_nm,
          alt_tolerance_ft: t.alt_tolerance_ft,
          speed_tolerance_kts: t.speed_tolerance_kts,
          color: t.color,
        },
        targetShapes.faces,
        targetShapes.edges,
      );
    }
    layers.push(
      new SolidPolygonLayer({
        id: "aircraft-target-tolerance-faces",
        data: targetShapes.faces,
        getPolygon: (d: any) => d.polygon,
        getFillColor: (d: any) => d.color,
        pickable: false,
        parameters: { depthTest: false },
      }),
      new LineLayer({
        id: "aircraft-target-tolerance-edges",
        data: targetShapes.edges,
        getSourcePosition: (d: any) => d.src,
        getTargetPosition: (d: any) => d.tgt,
        getColor: (d: any) => d.color,
        getWidth: lw(1.5),
        widthUnits: "pixels",
        widthMinPixels: lw(1),
        pickable: false,
        parameters: { depthTest: false },
      }),
      new PathLayer({
        id: "aircraft-target-lines",
        data: targeted,
        // Thread the goal line through the aircraft's *intermediate* route
        // waypoints (the shared sampled fixes, at their episode positions)
        // rather than jumping straight to the final target - a two-leg route
        // draws spawn -> fix -> exit, matching what the aircraft will fly.
        getPath: (a: any) => {
          const path: [number, number, number][] = [[a.lon, a.lat, zMeters(a.alt_ft)]];
          const steps = Array.isArray(a.route) ? a.route : [];
          for (let i = 0; i < steps.length - 1; i++) {
            const s = steps[i];
            const name = typeof s === "string" ? s : s?.waypoint;
            const p = name ? wpPositions.get(name) : undefined;
            if (p) path.push(p);
          }
          path.push([a.target.lon, a.target.lat, targetZ(a)]);
          return path;
        },
        getColor: [239, 68, 68, 110],
        getWidth: lw(1),
        widthUnits: "pixels",
        widthMinPixels: lw(1),
        parameters: { depthTest: false },
      }),
      new ScatterplotLayer({
        id: "aircraft-targets",
        data: targeted,
        getPosition: (a: any) => [a.target.lon, a.target.lat, targetZ(a)],
        getFillColor: [239, 68, 68, 0],
        getLineColor: [239, 68, 68, 230],
        stroked: true,
        lineWidthMinPixels: lw(1.5),
        getRadius: 6,
        radiusUnits: "pixels",
        billboard: true,
        parameters: { depthTest: false },
      }),
    );
    if (flown) {
      // A leader to where the aircraft is a minute on, at its ground speed and
      // heading - as a radar scope draws it.
      const ahead = (a: SpawnedAircraft): [number, number, number] => {
        const nmAhead = a.gs_kts / 60;
        const h = (a.hdg_deg * Math.PI) / 180;
        return [
          a.lon_deg + (nmAhead * Math.sin(h)) / (60 * Math.cos((a.lat_deg * Math.PI) / 180)),
          a.lat_deg + (nmAhead * Math.cos(h)) / 60,
          zMeters(a.alt_ft),
        ];
      };
      const color = (a: SpawnedAircraft): [number, number, number, number] =>
        a.controlled ? [239, 68, 68, 255] : [160, 160, 160, 255];
      layers.push(
        new LineLayer({
          id: "aircraft-leaders",
          data: flown,
          getSourcePosition: (a: SpawnedAircraft) => [a.lon_deg, a.lat_deg, zMeters(a.alt_ft)],
          getTargetPosition: ahead,
          getColor: color,
          getWidth: 1.5,
          widthUnits: "pixels",
          parameters: { depthTest: false },
        }),
        new ScatterplotLayer({
          id: "aircraft",
          data: flown,
          getPosition: (a: SpawnedAircraft) => [a.lon_deg, a.lat_deg, zMeters(a.alt_ft)],
          getFillColor: color,
          getLineColor: [255, 255, 255, 220],
          stroked: true,
          lineWidthMinPixels: 1,
          getRadius: 4,
          radiusUnits: "pixels",
          billboard: true,
          pickable: true,
          onClick: (info) => {
            const a = info.object as SpawnedAircraft | undefined;
            if (a) onPickAircraft?.({ key: a.callsign, at_s: a.time_s, acid: a.callsign, actype: a.actype });
            return !!a;
          },
          parameters: { depthTest: false },
        }),
        pickRing(flown.filter((a) => a.callsign === picked).map((a) => [a.lon_deg, a.lat_deg, zMeters(a.alt_ft)])),
      );
      // Each aircraft's route as resolved for it: a line through its targets,
      // and at each target its constraint window - the reach disc, the
      // altitude band as a cylinder - drawn once however many share it.
      const shapes = { edges: [] as WaypointEdge[], faces: [] as WaypointFace[] };
      for (const t of uniqueTargets(flown)) {
        waypointToleranceGeometry(
          {
            name: t.waypoint ?? "target",
            lat: t.lat,
            lon: t.lon,
            alt_ft: t.alt_ft,
            speed_kts: t.speed_kts,
            reach_radius_nm: t.reach_radius_nm,
            alt_tolerance_ft: t.alt_tolerance_ft,
            speed_tolerance_kts: t.speed_tolerance_kts,
            color: t.color ?? "cyan",
          },
          shapes.faces,
          shapes.edges,
        );
      }
      layers.push(
        new PathLayer({
          id: "aircraft-routes",
          data: flown.filter((a) => a.targets.length),
          getPath: (a: SpawnedAircraft) => {
            let alt = a.alt_ft;
            return [
              [a.lon_deg, a.lat_deg, zMeters(a.alt_ft)],
              ...a.targets.map((t) => {
                alt = t.alt_ft ?? alt;
                return [t.lon, t.lat, zMeters(alt)] as [number, number, number];
              }),
            ];
          },
          getColor: (a: SpawnedAircraft) => (a.controlled ? [239, 68, 68, 110] : [160, 160, 160, 110]),
          getWidth: lw(1),
          widthUnits: "pixels",
          widthMinPixels: lw(1),
          parameters: { depthTest: false },
        }),
        new SolidPolygonLayer({
          id: "aircraft-target-tolerance-faces",
          data: shapes.faces,
          getPolygon: (d: any) => d.polygon,
          getFillColor: (d: any) => d.color,
          pickable: false,
          parameters: { depthTest: false },
        }),
        new LineLayer({
          id: "aircraft-target-tolerance-edges",
          data: shapes.edges,
          getSourcePosition: (d: any) => d.src,
          getTargetPosition: (d: any) => d.tgt,
          getColor: (d: any) => d.color,
          getWidth: lw(1.5),
          widthUnits: "pixels",
          widthMinPixels: lw(1),
          pickable: false,
          parameters: { depthTest: false },
        }),
      );
      // Tags: each aircraft's pygame label, and each target's constraints,
      // for those the map found room for; hovering an aircraft gives its label
      // even where its tag had no room.
      if (visibility.labels) {
        const tagLayer = (id: string, data: any[], getPosition: any, getText: any, getColor: any, getBackgroundColor: any) =>
          new TextLayer({
            id,
            data,
            getPosition,
            getText,
            getColor,
            getSize: TAG_SIZE_PX,
            getTextAnchor: "start",
            getAlignmentBaseline: "center",
            getPixelOffset: [TAG_OFFSET_PX, 0],
            fontFamily: "Helvetica, Arial, sans-serif",
            fontWeight: 600,
            lineHeight: TAG_LINE_HEIGHT,
            characterSet: "auto",
            background: true,
            getBackgroundColor,
            backgroundPadding: [4, 3, 4, 3],
            billboard: true,
            parameters: { depthTest: false },
          });
        const targets = uniqueTargets(flown).filter((t) => targetTag(t).includes("\n"));
        layers.push(
          tagLayer(
            "aircraft-target-labels",
            tagged ? targets.filter((t) => tagged.has(spawnTargetKey(t))) : targets,
            (t: SpawnTarget) => [t.lon, t.lat, zMeters(t.alt_ft ?? 0)],
            targetTag,
            [253, 230, 138, 255],
            [40, 32, 10, 215],
          ),
          tagLayer(
            "aircraft-labels",
            tagged ? flown.filter((a) => tagged.has(a.callsign)) : flown,
            (a: SpawnedAircraft) => [a.lon_deg, a.lat_deg, zMeters(a.alt_ft)],
            aircraftTag,
            (a: SpawnedAircraft) => (a.controlled ? [255, 214, 214, 255] : [225, 225, 225, 255]),
            [20, 20, 22, 215],
          ),
        );
      }
    } else {
      layers.push(
        new ScatterplotLayer({
          id: "aircraft",
          data: preview.sampled_aircraft,
          getPosition: (a: any) => [a.lon, a.lat, zMeters(a.alt_ft)],
          getFillColor: [239, 68, 68, 255],
          getLineColor: [255, 255, 255, 220],
          stroked: true,
          lineWidthMinPixels: 1,
          getRadius: 4,
          radiusUnits: "pixels",
          billboard: true,
          pickable: true,
          onClick: (info) => {
            const a = info.object as any;
            if (a) onPickAircraft?.({ key: `preview:${info.index}`, at_s: a.spawn_time, acid: null, actype: a.actype });
            return !!a;
          },
          parameters: { depthTest: false },
        }),
        pickRing(
          preview.sampled_aircraft
            .filter((_: any, i: number) => `preview:${i}` === picked)
            .map((a: any) => [a.lon, a.lat, zMeters(a.alt_ft)]),
        ),
      );
    }
  }
  return layers;
}
