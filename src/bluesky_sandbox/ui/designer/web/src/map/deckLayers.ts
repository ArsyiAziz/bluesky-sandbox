// deck.gl layer construction for the map: bounds wireframes, waypoint tolerance
// volumes, route polylines + direction arrows, sampled aircraft, and nav
// context. Category visibility + per-element hide are applied here so the spec
// is never mutated by view toggles.
import { IconLayer, LineLayer, PathLayer, ScatterplotLayer, SolidPolygonLayer } from "@deck.gl/layers";
import type { NavFeatures, PreviewResult, SpecDict } from "../api";
import type {
  CategoryVisibility,
  Edge,
  EditTarget,
  Face,
  RGBA,
  RoutePath,
  WaypointEdge,
  WaypointFace,
} from "./types";
import {
  NAMED,
  addHandleDropStems,
  cssToRgb,
  frontendBoundsGeometry,
  regionGeometry,
  spawnRouteLinks,
  waypointPosition,
  waypointPositions,
  zMeters,
  addWaypointStem,
  waypointToleranceGeometry,
} from "./geometry";
import { boundsForTarget, groupBbox, groupMemberBounds, targetKey } from "./editHandles";

export const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

export const LABEL_FONT = ["Open Sans Regular"];

export const ROTATION_HANDLE_ICON =
  "data:image/svg+xml;charset=utf-8," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path d="M24.7 7.3A12 12 0 1 0 27.5 20h-5.2a7.4 7.4 0 1 1-1.2-9.2L17 15h12V3z"/></svg>',
  );

// A 4-way move glyph for the centre "drag the whole shape" handle, so it reads
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
export function getTooltip({ object }: any) {
  if (!object) return null;
  if (object.points) return { text: `route: ${object.name}` };
  if (object.routeName) return { text: `route: ${object.routeName}` };
  if (object.awid) return { text: `${object.awid}  ${object.from_id} → ${object.to_id}` };
  if (object.icao) return { text: `${object.icao}${object.name ? " — " + object.name : ""}` };
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
) {
  const shown = (key: string) => !hidden.has(key);
  // The selected element is drawn from canonical (unrotated) geometry below so
  // its edit handles line up; skip its rotated preview copy here. (Per-episode
  // rotation groups rotate the preview, but handles edit the canonical design.)
  const selKey = selectedTarget ? targetKey(selectedTarget) : null;
  const isSel = (t: EditTarget) => selKey !== null && targetKey(t) === selKey;
  // Global line-width multiplier (the "lines" overlay control).
  const lw = (w: number) => w * lineScale;
  const out = { edges: [] as Edge[], faces: [] as Face[] };
  const draftOut = { edges: [] as Edge[], faces: [] as Face[] };
  const waypointOut = { edges: [] as WaypointEdge[], faces: [] as WaypointFace[] };
  const draftBounds = draftSpec && selectedTarget ? boundsForTarget(draftSpec, selectedTarget) : null;
  if (draftBounds) {
    const g = frontendBoundsGeometry(draftBounds);
    if (g && selectedTarget) regionGeometry(g, [255, 255, 255, 255], draftOut, { target: selectedTarget, name: "draft" });
  }
  if (preview.airspace && visibility.airspace && shown("airspace") && !isSel({ scope: "airspace" })) {
    regionGeometry(preview.airspace, NAMED.blue, out, {
      name: "airspace",
      target: { scope: "airspace" },
    });
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
    if (draftQ?.type === "waypoint" && Number.isFinite(draftQ.lat) && Number.isFinite(draftQ.lon)) {
      wpts.push({
        ...draftQ,
        kind: "waypoint",
        name: selectedTarget.name,
        ident: draftQ.waypoint,
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
      regionGeometry(q, cssToRgb(q.color), out, {
        name: q.name,
        target: { scope: "queryable", name: q.name },
      });
    }
  }
  // Spawn regions are bounds too - render them as wireframes (green), like the
  // airspace and queryable regions.
  if (visibility.spawnRegions) {
    for (const [index, r] of preview.spawn_regions.entries()) {
      if (r.render_shape === false || !shown(`spawn:${index}`)) continue;
      if (isSel({ scope: "spawn", index })) continue;
      regionGeometry(r, NAMED.green, out, {
        name: r.name,
        target: { scope: "spawn", index },
      });
    }
  }
  // Named bounds not already drawn by a consumer (airspace / query-region /
  // spawn) — e.g. freshly created, or used only as a waypoint sample area — are
  // drawn standalone (neutral) so they're visible and editable on the map.
  if (spec) {
    const refName = (b: any): string | null => (b && typeof b.ref === "string" ? b.ref : null);
    const drawn = new Set<string>();
    const a = refName(spec.airspace);
    if (a) drawn.add(a);
    for (const q of Object.values(spec.queryables ?? {}) as any[]) {
      if (q?.type === "query_region") {
        const r = refName(q.bounds);
        if (r) drawn.add(r);
      }
    }
    for (const r of (spec.spawn?.regions ?? []) as any[]) {
      const rb = refName(r?.bounds);
      if (rb) drawn.add(rb);
    }
    for (const [name, bounds] of Object.entries(spec.regions ?? {}) as [string, SpecDict][]) {
      if (drawn.has(name) || !shown(`region:${name}`)) continue;
      // Prefer the sampled-episode geometry from the preview (shape draw +
      // rotation) so per-episode-randomized regions render as an episode
      // would place them; the selected region stays canonical so its edit
      // handles line up with the panel's parametric shape, with the sampled
      // copy kept visible as a faint ghost.
      const sampled = preview.regions?.[name];
      const selected = isSel({ scope: "region", name });
      const g = (!selected && sampled) || frontendBoundsGeometry(bounds);
      if (g) regionGeometry(g, NAMED.slate, out, { name, target: { scope: "region", name } });
      if (selected && sampled) {
        regionGeometry(sampled, [148, 163, 184, 70], out, {
          name: `${name} (sampled)`,
          target: { scope: "region", name },
        });
      }
    }
  }
  // The selected airspace/query-region/spawn bounds: draw it from the canonical
  // (unrotated) design so the edit handles align even while a rotation group
  // rotates the rest of the preview. (Standalone `region` scope is already drawn
  // canonical above; waypoints have no bounds to draw here.)
  const selDropStems: Edge[] = [];
  if (spec && selectedTarget && selectedTarget.scope !== "region" && selectedTarget.scope !== "group") {
    const selBounds = boundsForTarget(spec, selectedTarget);
    const g = selBounds ? frontendBoundsGeometry(selBounds) : null;
    if (g) {
      const color =
        selectedTarget.scope === "airspace"
          ? NAMED.blue
          : selectedTarget.scope === "spawn"
            ? NAMED.green
            : cssToRgb(spec.queryables?.[selectedTarget.name]?.color);
      regionGeometry(g, color, out, { name: "selected", target: selectedTarget });
      addHandleDropStems(g, color, selDropStems);
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
      for (const bounds of groupMemberBounds(spec, selectedTarget.id)) {
        const g = frontendBoundsGeometry(bounds);
        if (g) regionGeometry(g, violet, out, { target: selectedTarget, name: "group" });
      }
    }
  }

  // Spawn directions: when a spawn region is selected, draw a "pie" wedge out of
  // its centre spanning the heading range (a full disk when unconstrained), plus
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
      // A small pie around the centre drag handle (not a region-sized wedge).
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
    // Colour each connector to match its route's polyline.
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
    // hollow ring at it, so reseeding shows the per-aircraft destination spread.
    const targeted = preview.sampled_aircraft.filter((a: any) => a.target);
    const targetZ = (a: any) => zMeters(a.target.alt_ft ?? a.alt_ft);
    // Episode positions of the shared (per-episode sampled) waypoints, for
    // threading each aircraft's goal line through its intermediate fixes.
    const wpPositions = waypointPositions(spec, preview);
    // Each sampled target drawn as a *waypoint*: a reach-radius tolerance disc
    // in world units and the waypoint's colour. The pixel-space ring below
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
        parameters: { depthTest: false },
      }),
    );
  }
  return layers;
}
