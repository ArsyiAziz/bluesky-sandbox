// Read-only map preview for the Route tab: draws the design (regions, waypoints,
// routes) using the same deck layers as the main map, highlights the route being
// edited, and reports waypoint clicks so the graph editor can append them.
import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { LineLayer } from "@deck.gl/layers";
import { api, type SpecDict, type PreviewResult, type ValidateResult } from "../api";
import type { CategoryVisibility, EditTarget } from "../map/types";
import { centroid, frontendBoundsGeometry, routePaths } from "../map/geometry";
import { deckLayers, getTooltip } from "../map/deckLayers";
import { basemapById, DEFAULT_BASEMAP } from "../map/basemaps";

// Extra connector layers for the route view: spawn region → the route's first
// waypoint(s), and any sampled waypoint → the region it draws its position from.
function connectorLayers(
  preview: PreviewResult,
  spec: SpecDict | null,
  routeName: string | null,
  spawnIndex: number,
  routeWaypoints: string[],
): any[] {
  const layers: any[] = [];
  const wpPos = new Map<string, [number, number, number]>();
  for (const q of preview.queryables as any[]) {
    if (q.kind === "waypoint" && Number.isFinite(q.lon) && Number.isFinite(q.lat)) wpPos.set(q.name, [q.lon, q.lat, 0]);
  }

  // Spawn → first waypoint of each enumerated limb of the route. Matched by
  // index (spawn_regions mirror spec order) so it's robust to naming.
  const sr = spawnIndex >= 0 ? (preview.spawn_regions as any[])[spawnIndex] : null;
  if (sr?.vertices?.length) {
    {
      const origin = centroid(sr.vertices);
      const rps = routePaths(spec, preview).filter((rp) => rp.name === routeName && rp.points.length);
      let firsts = rps.map((rp) => rp.points[0]);
      // Fallback for a single-waypoint route (no polyline): use its one waypoint.
      if (!firsts.length && routeWaypoints[0] && wpPos.has(routeWaypoints[0])) firsts = [wpPos.get(routeWaypoints[0])!];
      // Match the route's polyline colour.
      const c = rps[0]?.color ?? [120, 235, 150, 255];
      if (firsts.length) {
        layers.push(
          new LineLayer({
            id: "route-spawn-origin",
            data: firsts.map((p) => ({ src: [origin[0], origin[1], 0], tgt: [p[0], p[1], p[2] ?? 0] })),
            getSourcePosition: (d: any) => d.src,
            getTargetPosition: (d: any) => d.tgt,
            getColor: [c[0], c[1], c[2], 220],
            getWidth: 2,
            widthUnits: "pixels",
            parameters: { depthTest: false },
          }),
        );
      }
    }
  }

  // Sampled waypoint → the region its position is drawn from each episode.
  const sampleLinks: { src: number[]; tgt: number[] }[] = [];
  for (const name of routeWaypoints) {
    const q = spec?.queryables?.[name];
    const ref = q?.sample?.ref;
    const region = ref ? spec?.regions?.[ref] : undefined;
    const wp = wpPos.get(name);
    if (!region || !wp) continue;
    const g = frontendBoundsGeometry(region);
    if (!g?.vertices?.length) continue;
    const c = centroid(g.vertices);
    sampleLinks.push({ src: [wp[0], wp[1], 0], tgt: [c[0], c[1], 0] });
  }
  if (sampleLinks.length) {
    layers.push(
      new LineLayer({
        id: "route-sample-links",
        data: sampleLinks,
        getSourcePosition: (d: any) => d.src,
        getTargetPosition: (d: any) => d.tgt,
        getColor: [255, 210, 90, 200],
        getWidth: 1.5,
        widthUnits: "pixels",
        parameters: { depthTest: false },
      }),
    );
  }
  return layers;
}

// Routes + their waypoints + region context; aircraft/nav off to keep it light.
const VISIBILITY: CategoryVisibility = {
  airspace: true, regions: true, waypoints: true, routes: true, spawnRegions: true,
  aircraft: false, nav: false, airways: false, labels: true,
};
const NO_HIDDEN = new Set<string>();

export default function RouteMap({
  spec,
  highlightRoute,
  spawnIndex,
  routeWaypoints,
  onPickWaypoint,
}: {
  spec: SpecDict | null;
  highlightRoute: string | null;
  spawnIndex: number;
  routeWaypoints: string[];
  onPickWaypoint: (name: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);
  const previewRef = useRef<PreviewResult | null>(null);
  const pickRef = useRef(onPickWaypoint);
  pickRef.current = onPickWaypoint;
  const highlightRef = useRef(highlightRoute);
  highlightRef.current = highlightRoute;
  const spawnRef = useRef(spawnIndex);
  spawnRef.current = spawnIndex;
  const wpRef = useRef(routeWaypoints);
  wpRef.current = routeWaypoints;
  const specRef = useRef(spec);
  specRef.current = spec;
  const [ready, setReady] = useState(false);

  const refresh = () => {
    if (!overlayRef.current || !previewRef.current) return;
    const onSelect = (t: EditTarget | null) => {
      if (t?.scope === "queryable") pickRef.current(t.name);
    };
    const base = deckLayers(
      previewRef.current, null, onSelect, null, null,
      routePaths(specRef.current, previewRef.current),
      highlightRef.current, VISIBILITY, NO_HIDDEN, 1, specRef.current,
    );
    const links = connectorLayers(previewRef.current, specRef.current, highlightRef.current, spawnRef.current, wpRef.current);
    overlayRef.current.setProps({ layers: [...base, ...links] });
  };

  // Create the map once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const saved = window.localStorage.getItem("designer.basemap") ?? DEFAULT_BASEMAP;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: basemapById(saved).style,
      center: [4.75, 52.0],
      zoom: 7,
    });
    const ro = new ResizeObserver(() => map.resize());
    ro.observe(containerRef.current);
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    map.on("load", () => {
      const overlay = new MapboxOverlay({ interleaved: false, layers: [], getTooltip, pickingRadius: 8 });
      map.addControl(overlay as any);
      overlayRef.current = overlay;
      map.resize();
      setReady(true);
    });
    mapRef.current = map;
    return () => {
      ro.disconnect();
      overlayRef.current = null;
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Re-fetch the preview when the spec changes; redraw when ready/highlight change.
  useEffect(() => {
    if (!ready || !spec) return;
    let cancelled = false;
    api
      .preview(spec, 0)
      .then((preview: PreviewResult) => {
        if (cancelled) return;
        previewRef.current = preview;
        refresh();
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec, ready]);

  useEffect(() => {
    if (ready) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [highlightRoute, spawnIndex, routeWaypoints.join(","), ready]);

  return <div ref={containerRef} className="route-map" />;
}

// Re-export so callers needn't import the type separately.
export type { ValidateResult };
