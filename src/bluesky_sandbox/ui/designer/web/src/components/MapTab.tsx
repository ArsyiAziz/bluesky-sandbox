import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { api, type SpecDict, type PreviewResult, type NavFeatures, type SpawnedAircraft } from "../api";
import DesignPanel from "./DesignPanel";
import { EpisodeStepper } from "./EpisodeStepper";
import SearchBox from "./SearchBox";
import type { CategoryVisibility, EditHandle, EditTarget } from "../map/types";
import { centroid, frontendShapeGeometry, point, routePaths, zMeters } from "../map/geometry";
import { buildEditHandles, dragStateForHandle, targetKey, updateSpecFromHandle } from "../map/editHandles";
import {
  EMPTY,
  LABEL_FONT,
  MOVE_HANDLE_ICON,
  ROTATION_HANDLE_ICON,
  TAG_LINE_HEIGHT,
  TAG_OFFSET_PX,
  TAG_SIZE_PX,
  aircraftTag,
  spawnTargetKey,
  targetTag,
  uniqueTargets,
  deckLayers,
  getTooltip,
} from "../map/deckLayers";
import { WebMercatorViewport } from "@deck.gl/core";
import { setColorPalette } from "../map/geometry";
import { BASEMAPS, DEFAULT_BASEMAP, basemapById, type BasemapId } from "../map/basemaps";
import { gcOrphanShapes, placementAltitudeRange } from "../specHelpers";
import { addWaypointAt } from "../waypointPoints";
import { shapeGraph, isUnused } from "../shapeGraph";
import { shapesInView } from "../map/shapesInView";
import { useRefresh } from "../refresh";
import { useEpisode, useEpisodeSpawns } from "../episode";

type DragState = import("../map/types").DragState;

const PANEL_WIDTH_KEY = "designer.panelWidth";
const DEFAULT_PANEL_WIDTH = 360;
const MIN_PANEL_WIDTH = 260;

/** Panel width kept within [MIN, 620] and never past 60% of the window. */
function clampPanelWidth(width: number): number {
  const max = Math.max(MIN_PANEL_WIDTH, Math.min(620, window.innerWidth * 0.6));
  return Math.round(Math.min(max, Math.max(MIN_PANEL_WIDTH, width)));
}
const DESIGNER_LABEL_SOURCE = "designer-labels";
const DESIGNER_LABEL_LAYER = "designer-labels";

const DEFAULT_VISIBILITY: CategoryVisibility = {
  airspace: true,
  regions: true,
  waypoints: true,
  routes: true,
  spawnRegions: true,
  shapes: false,
  aircraft: false,
  nav: false,
  airways: false,
  labels: true,
};

// Order + labels for the map overlay's "layers" toggles.
const CATEGORIES: { key: keyof CategoryVisibility; label: string }[] = [
  { key: "airspace", label: "airspace" },
  { key: "regions", label: "regions" },
  { key: "waypoints", label: "waypoints" },
  { key: "routes", label: "routes" },
  { key: "spawnRegions", label: "spawn" },
  { key: "shapes", label: "all shapes" },
  { key: "aircraft", label: "aircraft" },
  { key: "nav", label: "nav" },
  { key: "airways", label: "airways" },
  { key: "labels", label: "labels" },
];

// True when focus is in a text-entry context, so keyboard delete must stand
// down (the panel + Monaco are full of inputs the user types into).
function isEditableElement(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  return (
    el.tagName === "INPUT" ||
    el.tagName === "TEXTAREA" ||
    el.tagName === "SELECT" ||
    el.isContentEditable
  );
}

export default function MapTab({
  spec,
  onSpecChange,
}: {
  spec: SpecDict | null;
  onSpecChange: (next: SpecDict) => void;
}) {
  // Design-panel width. Persisted per browser so a width you chose survives a
  // reload; clamped on read as well as on drag, because a stored value from a
  // wider monitor would otherwise leave no room for the map.
  const [panelWidth, setPanelWidth] = useState<number>(() => {
    const stored = Number(localStorage.getItem(PANEL_WIDTH_KEY));
    return clampPanelWidth(Number.isFinite(stored) && stored > 0 ? stored : DEFAULT_PANEL_WIDTH);
  });
  const resizingRef = useRef(false);

  useEffect(() => {
    localStorage.setItem(PANEL_WIDTH_KEY, String(panelWidth));
  }, [panelWidth]);

  // Re-clamp when the window shrinks, so the panel can never crowd out the map.
  useEffect(() => {
    const onResize = () => setPanelWidth((w) => clampPanelWidth(w));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const startPanelResize = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    resizingRef.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    // Width is measured from the right edge, so the handle tracks the cursor
    // regardless of where inside it the drag started.
    const onMove = (ev: PointerEvent) => {
      if (!resizingRef.current) return;
      setPanelWidth(clampPanelWidth(window.innerWidth - ev.clientX));
    };
    const onUp = () => {
      resizingRef.current = false;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);
  const previewRef = useRef<PreviewResult | null>(null);
  const specRef = useRef<SpecDict | null>(spec);
  const draftSpecRef = useRef<SpecDict | null>(null);
  const navRef = useRef<NavFeatures | null>(null);
  const dragHandleRef = useRef<EditHandle | null>(null);
  const dragStateRef = useRef<DragState | null>(null);
  const rafRef = useRef<number | null>(null);
  const handleRafRef = useRef<number | null>(null);
  const activeBasemapRef = useRef<BasemapId | null>(null);
  const [selectedTarget, setSelectedTarget] = useState<EditTarget | null>(null);
  const selectedTargetRef = useRef<EditTarget | null>(null);
  const [handleTick, setHandleTick] = useState(0);
  const [ready, setReady] = useState(false);
  const { seed, pick, setPick } = useEpisode();
  // The picked aircraft, for the map to ring and the deck to set.
  const pickRef = useRef(pick);
  pickRef.current = pick;
  const setPickRef = useRef(setPick);
  setPickRef.current = setPick;
  const [viewCenter, setViewCenter] = useState<[number, number]>([52.0, 4.75]);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [basemap, setBasemap] = useState<BasemapId>(() => {
    const saved = window.localStorage.getItem("designer.basemap") ?? DEFAULT_BASEMAP;
    return basemapById(saved).id;
  });

  // View-only visibility (never written to the spec).
  const [visibility, setVisibility] = useState<CategoryVisibility>(DEFAULT_VISIBILITY);
  const visibilityRef = useRef(visibility);
  // The episode run, for the aircraft as flown - only while they are shown.
  const { spawns, loading: flying } = useEpisodeSpawns(spec, visibility.aircraft);
  const flownRef = useRef<SpawnedAircraft[] | null>(null);
  flownRef.current = visibility.aircraft ? spawns?.aircraft ?? null : null;
  const taggedRef = useRef<Set<string> | null>(null);
  const [hiddenElements, setHiddenElements] = useState<Set<string>>(new Set());
  const hiddenRef = useRef(hiddenElements);
  // View-only lock: locked elements show no edit handles and can't be deleted
  // from the map (never written to the spec).
  const [lockedElements, setLockedElements] = useState<Set<string>>(new Set());
  const lockedRef = useRef(lockedElements);
  // Highlight is purely a render concern (no JSX depends on it), so the ref is
  // the single source of truth and we refresh deck imperatively.
  const highlightedRouteRef = useRef<string | null>(null);
  // Global line-width multiplier for the overlay "lines" slider.
  const [lineScale, setLineScale] = useState(1);
  const lineScaleRef = useRef(1);
  // Moving regions: the time they are shown at (s into the episode), and
  // whether it plays. Only offered when the preview has something moving.
  const [motionT, setMotionT] = useState(0);
  const motionTRef = useRef(0);
  const [playing, setPlaying] = useState(false);
  const [hasMotion, setHasMotion] = useState(false);

  useEffect(() => {
    if (!dragStateRef.current) {
      specRef.current = spec;
      draftSpecRef.current = null;
    }
  }, [spec]);

  const scheduleHandleRefresh = () => {
    if (handleRafRef.current != null) return;
    handleRafRef.current = requestAnimationFrame(() => {
      handleRafRef.current = null;
      setHandleTick((v) => v + 1);
    });
  };

  const pointerLonLat = (event: PointerEvent | React.PointerEvent): [number, number] | null => {
    const map = mapRef.current;
    const container = containerRef.current;
    if (!map || !container) return null;
    const rect = container.getBoundingClientRect();
    const p = map.unproject([event.clientX - rect.left, event.clientY - rect.top]);
    return [p.lng, p.lat];
  };

  const startDomDrag = (handle: EditHandle, event: React.PointerEvent) => {
    if (!specRef.current) return;
    event.preventDefault();
    event.stopPropagation();
    dragHandleRef.current = handle;
    dragStateRef.current = dragStateForHandle(handle, specRef.current, event.clientY);
    mapRef.current?.dragPan.disable();
    const move = (ev: PointerEvent) => {
      const coord = pointerLonLat(ev);
      const state = dragStateRef.current;
      if (!coord || !state) return;
      draftSpecRef.current = updateSpecFromHandle(
        state.startSpec,
        state.handle,
        coord[0],
        coord[1],
        state,
        ev.clientY,
      );
      scheduleHandleRefresh();
      scheduleDeckRefresh();
    };
    const up = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      const coord = pointerLonLat(ev);
      const state = dragStateRef.current;
      if (coord && state) {
        const next = updateSpecFromHandle(
          state.startSpec,
          state.handle,
          coord[0],
          coord[1],
          state,
          ev.clientY,
        );
        specRef.current = next;
        draftSpecRef.current = null;
        onSpecChange(next);
      }
      dragHandleRef.current = null;
      dragStateRef.current = null;
      mapRef.current?.dragPan.enable();
      scheduleHandleRefresh();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up, { once: true });
  };

  const selectTarget = (target: EditTarget | null) => {
    selectedTargetRef.current = target;
    setSelectedTarget(target);
    refreshDeck();
    // An unused bounds' flag shows with the bounds: with the selection.
    refreshLabels();
  };

  // A deck object click and the underlying maplibre "click" both fire for the
  // same pointer event. Stamp deck selections so the map's click-to-deselect can
  // skip them - otherwise clicking an element would immediately deselect it.
  const lastDeckClickRef = useRef(0);
  const selectFromDeck = (target: EditTarget | null) => {
    lastDeckClickRef.current = performance.now();
    selectTarget(target);
  };

  const setHighlight = (name: string | null) => {
    highlightedRouteRef.current = name;
    refreshDeck();
  };

  // Delete the currently selected map element from the spec. Wired to
  // Delete/Backspace; airspace is a destructive singleton so it confirms first.
  const deleteSelected = () => {
    const target = selectedTargetRef.current;
    if (!target || !specRef.current) return;
    if (target.scope === "group") return; // groups are removed from the panel, not the map
    if (lockedRef.current.has(targetKey(target))) return;
    if (target.scope === "airspace" && !window.confirm("Delete the airspace?")) return;
    // A bounds others use stays: deleting it would leave their refs dangling.
    if (target.scope === "region") {
      const info = shapeGraph(specRef.current)[target.name];
      const users = (info?.usedBy ?? []).filter((u) => !u.label.endsWith("(sample area)"));
      if (users.length) {
        window.alert(
          `“${target.name}” is used by ${users.map((u) => u.label).join(", ")}. Point them at another shape, or delete them, first.`,
        );
        return;
      }
    }
    const next = structuredClone(specRef.current);
    if (target.scope === "airspace") next.airspace = null;
    else if (target.scope === "queryable") delete next.queryables?.[target.name];
    else if (target.scope === "spawn") next.spawn?.regions?.splice(target.index, 1);
    else if (target.scope === "region") {
      // A standalone-drawn bounds (e.g. a waypoint sample area): drop the bounds
      // and any waypoint sample pointing at it, so no reference dangles.
      delete next.shapes?.[target.name];
      for (const q of Object.values(next.queryables ?? {}) as any[]) {
        if (q?.sample?.ref === target.name) delete q.sample;
      }
    }
    // Sweep bounds this delete left unreferenced.
    gcOrphanShapes(next, specRef.current);
    specRef.current = next;
    selectTarget(null);
    onSpecChange(next);
  };

  // Keyboard delete for the selected element. Guarded so it never fires while
  // the user is typing in the properties panel (inputs / selects / Monaco).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      if (!selectedTargetRef.current || isEditableElement(document.activeElement)) return;
      e.preventDefault();
      deleteSelected();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const showMotionAt = (t: number) => {
    motionTRef.current = t;
    setMotionT(t);
    scheduleDeckRefresh();
  };
  // Play: an hour in about twelve seconds, then round again.
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      const next = motionTRef.current >= 3600 ? 0 : motionTRef.current + 60;
      showMotionAt(next);
    }, 200);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing]);

  const changeLineScale = (v: number) => {
    lineScaleRef.current = v;
    setLineScale(v);
    refreshDeck();
  };

  const toggleHidden = (key: string) =>
    setHiddenElements((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const toggleLocked = (key: string) =>
    setLockedElements((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const refreshDeck = () => {
    rafRef.current = null;
    if (overlayRef.current && previewRef.current) {
      const activeSpec = draftSpecRef.current ?? specRef.current;
      overlayRef.current.setProps({
        layers: deckLayers(
          previewRef.current,
          navRef.current,
          selectFromDeck,
          draftSpecRef.current,
          selectedTargetRef.current,
          routePaths(activeSpec, previewRef.current),
          highlightedRouteRef.current,
          visibilityRef.current,
          hiddenRef.current,
          lineScaleRef.current,
          activeSpec,
          flownRef.current,
          taggedRef.current,
          motionTRef.current,
          pickRef.current?.key ?? null,
          (p) => {
            lastDeckClickRef.current = performance.now();
            setPickRef.current(p);
          },
        ),
      });
    }
  };

  // Which aircraft tags fit: in screen space, as the overlay projects them (at
  // altitude, on the tilted map), each kept unless it would cover a tag kept
  // before it - controlled traffic first. Run when the view settles.
  const declutterTags = () => {
    const map = mapRef.current;
    const flown = flownRef.current;
    if (!map || !flown) {
      taggedRef.current = null;
      return;
    }
    const center = map.getCenter();
    const canvas = map.getCanvas();
    const viewport = new WebMercatorViewport({
      width: canvas.clientWidth,
      height: canvas.clientHeight,
      longitude: center.lng,
      latitude: center.lat,
      zoom: map.getZoom(),
      pitch: map.getPitch(),
      bearing: map.getBearing(),
    });
    // Each tag's box beside its point, as the text layer lays it out.
    const charW = TAG_SIZE_PX * 0.6;
    const lineH = TAG_SIZE_PX * TAG_LINE_HEIGHT;
    const box = (x: number, y: number, text: string): [number, number, number, number] => {
      const lines = text.split("\n");
      const w = Math.max(...lines.map((l) => l.length)) * charW + 8;
      const h = lines.length * lineH + 6;
      const x0 = x + TAG_OFFSET_PX - 4;
      return [x0, y - h / 2, x0 + w, y + h / 2];
    };
    // Controlled aircraft first, then background traffic, then targets.
    const candidates: { id: string; at: [number, number, number]; text: string }[] = [
      ...[...flown]
        .sort((a, b) => Number(b.controlled) - Number(a.controlled))
        .map((a) => ({ id: a.callsign, at: [a.lon_deg, a.lat_deg, zMeters(a.alt_ft)] as [number, number, number], text: aircraftTag(a) })),
      ...uniqueTargets(flown).map((t) => ({
        id: spawnTargetKey(t),
        at: [t.lon, t.lat, zMeters(t.alt_ft ?? 0)] as [number, number, number],
        text: targetTag(t),
      })),
    ];
    const kept: [number, number, number, number][] = [];
    const tagged = new Set<string>();
    for (const c of candidates) {
      const [x, y] = viewport.project(c.at);
      const b = box(x, y, c.text);
      if (!kept.some((k) => b[0] < k[2] && k[0] < b[2] && b[1] < k[3] && k[1] < b[3])) {
        kept.push(b);
        tagged.add(c.id);
      }
    }
    taggedRef.current = tagged;
  };

  const ensureLabelLayer = () => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;
    if (!map.getSource(DESIGNER_LABEL_SOURCE)) {
      map.addSource(DESIGNER_LABEL_SOURCE, { type: "geojson", data: EMPTY });
    }
    if (!map.getLayer(DESIGNER_LABEL_LAYER)) {
      map.addLayer({
        id: DESIGNER_LABEL_LAYER, type: "symbol", source: DESIGNER_LABEL_SOURCE,
        layout: { "text-field": ["get", "label"], "text-font": LABEL_FONT, "text-size": 12, "text-offset": [0, 1.1], "text-anchor": "top" },
        paint: { "text-color": ["coalesce", ["get", "color"], "#fff"], "text-halo-color": "#000", "text-halo-width": 1.2 },
      });
    }
    refreshLabels();
  };

  const scheduleDeckRefresh = () => {
    if (rafRef.current != null) return;
    rafRef.current = requestAnimationFrame(refreshDeck);
  };

  // Rebuild the maplibre text-label source, respecting view visibility/hide so
  // labels disappear together with their geometry.
  const refreshLabels = () => {
    const map = mapRef.current;
    const src = map?.getSource(DESIGNER_LABEL_SOURCE) as maplibregl.GeoJSONSource | undefined;
    if (!src) return;
    const preview = previewRef.current;
    const vis = visibilityRef.current;
    const hidden = hiddenRef.current;
    const labels: GeoJSON.Feature[] = [];
    if (preview && vis.labels) {
      if (vis.spawnRegions) {
        preview.spawn_regions.forEach((r, index) => {
          if (r.render_shape !== false && r.render_name !== false && !hidden.has(`spawn:${index}`)) {
            const [lon, lat] = centroid(r.vertices);
            labels.push(point(lon, lat, { label: r.name }));
          }
        });
      }
      for (const q of preview.queryables) {
        if (hidden.has(`queryable:${q.name}`)) continue;
        if (q.kind === "region" && vis.regions && q.render_shape !== false && q.render_label !== false) {
          const [lon, lat] = centroid(q.vertices);
          labels.push(point(lon, lat, { label: q.name }));
        } else if (q.kind === "waypoint" && vis.waypoints && q.render_shape !== false && q.render_label !== false) {
          labels.push(point(q.lon, q.lat, { label: q.ident ?? q.name }));
        }
      }
    }
    // A bounds nothing uses: flagged where it is, whatever the labels setting -
    // it is a warning, not a name - when it is drawn at all (see shapesInView).
    const spec = specRef.current;
    if (preview && spec) {
      const graph = shapeGraph(spec);
      const inView = shapesInView(spec, selectedTargetRef.current);
      for (const [name, info] of Object.entries(graph)) {
        if (!isUnused(info) || hidden.has(`region:${name}`)) continue;
        if (!vis.shapes && !inView.own.has(name)) continue;
        const g = preview.shapes?.[name] ?? frontendShapeGeometry(spec.shapes?.[name]);
        if (!g?.vertices?.length) continue;
        const [lon, lat] = centroid(g.vertices);
        labels.push(point(lon, lat, { label: `⚠ ${name} · unused`, color: "#fbbf24" }));
      }
    }
    src.setData({ type: "FeatureCollection", features: labels });
  };

  // Fetch navdb features for the current map viewport (not the airspace), so
  // waypoints/airports show for whatever you're looking at. Reads only refs, so
  // it's stable enough to call from the map's moveend handler.
  const loadNav = () => {
    const map = mapRef.current;
    if (!map || (!visibilityRef.current.nav && !visibilityRef.current.airways)) {
      navRef.current = null;
      refreshDeck();
      return;
    }
    const bb = map.getBounds();
    const windowBounds = {
      type: "region",
      footprint: {
        type: "box",
        lat_min_deg: bb.getSouth(),
        lat_max_deg: bb.getNorth(),
        lon_min_deg: bb.getWest(),
        lon_max_deg: bb.getEast(),
      },
      altitude: null,
    };
    api
      .navFeatures(windowBounds, 200, 1500)
      .then((nav) => {
        navRef.current = nav;
        refreshDeck();
      })
      .catch(() => {});
  };

  // Re-render deck + labels whenever view visibility / hide / highlight change.
  // The flown aircraft came in, or went: draw them.
  useEffect(() => {
    if (ready) {
      declutterTags();
      refreshDeck();
      refreshLabels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spawns, ready]);

  // An aircraft was picked - here or in the Spaces tab's sample: ring it.
  useEffect(() => {
    if (ready) refreshDeck();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pick?.key, ready]);

  useEffect(() => {
    visibilityRef.current = visibility;
    if (ready) {
      refreshDeck();
      refreshLabels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visibility, ready]);

  useEffect(() => {
    hiddenRef.current = hiddenElements;
    if (ready) {
      refreshDeck();
      refreshLabels();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hiddenElements, ready]);

  useEffect(() => {
    lockedRef.current = lockedElements;
  }, [lockedElements]);

  // Load the color palette (same list the picker shows + the drivers render) so
  // named colors like black / gray / purple resolve in the map preview.
  const refreshKey = useRefresh();
  useEffect(() => {
    api
      .catalogOnce()
      .then((c) => {
        if (c?.colors) {
          setColorPalette(c.colors);
          if (ready) {
            refreshDeck();
            refreshLabels();
          }
        }
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, refreshKey]);

  // (Re)load nav whenever the nav or airways visibility toggles.
  useEffect(() => {
    if (ready) loadNav();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visibility.nav, visibility.airways, ready]);

  // Create the map once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: basemapById(basemap).style,
      center: [4.75, 52.0],
      zoom: 7,
      pitch: 45,
      bearing: -15,
    });
    const ro = new ResizeObserver(() => map.resize());
    ro.observe(containerRef.current);
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    map.on("load", () => {
      ensureLabelLayer();

      const overlay = new MapboxOverlay({
        interleaved: false,
        layers: [],
        getTooltip,
        pickingRadius: 8,
      });
      map.addControl(overlay as any);
      overlayRef.current = overlay;

      // Refetch nav features for the new viewport after panning/zooming.
      map.on("moveend", () => {
        const center = map.getCenter();
        setViewCenter([center.lat, center.lng]);
        if (visibilityRef.current.nav || visibilityRef.current.airways) loadNav();
        // The view settled: find room for the aircraft tags again.
        if (flownRef.current) {
          declutterTags();
          refreshDeck();
        }
      });
      const center = map.getCenter();
      setViewCenter([center.lat, center.lng]);
      map.on("move", scheduleHandleRefresh);
      // Clicking empty map deselects, but a deck object click fires the same
      // pointer event - defer so a just-stamped deck selection wins.
      map.on("click", () => {
        window.setTimeout(() => {
          if (performance.now() - lastDeckClickRef.current > 150) selectTarget(null);
        }, 0);
      });

      map.resize();
      setReady(true);
    });
    map.on("style.load", ensureLabelLayer);
    mapRef.current = map;
    activeBasemapRef.current = basemap;
    return () => {
      if (rafRef.current != null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (handleRafRef.current != null) {
        cancelAnimationFrame(handleRafRef.current);
        handleRafRef.current = null;
      }
      ro.disconnect();
      overlayRef.current = null;
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    if (activeBasemapRef.current === basemap) return;
    activeBasemapRef.current = basemap;
    window.localStorage.setItem("designer.basemap", basemap);
    map.setStyle(basemapById(basemap).style);
    map.once("style.load", () => {
      ensureLabelLayer();
      refreshDeck();
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [basemap, ready]);

  // Frame a design's airspace when it is first drawn - a design loaded or
  // started, not every edit to it, so editing never moves the view.
  const framedRef = useRef<string | null>(null);
  const frameDesign = (preview: PreviewResult) => {
    const map = mapRef.current;
    const name = String(specRef.current?.metadata?.name ?? "");
    const box = preview.airspace?.bounding_box;
    if (!map || !box || framedRef.current === name) return;
    framedRef.current = name;
    map.fitBounds(
      [
        [box.lon_min, box.lat_min],
        [box.lon_max, box.lat_max],
      ],
      { padding: 40, duration: 0 },
    );
  };

  // Re-render geometry whenever the (valid) spec / seed changes.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !spec) return;
    let canceled = false;
    setLoading(true);

    api
      .preview(spec, seed)
      .then((preview: PreviewResult) => {
        if (canceled) return;
        setError(null);
        previewRef.current = preview;
        frameDesign(preview);
        const moves = (g: any) => Array.isArray(g?.frames) && g.frames.length > 1;
        setHasMotion(
          moves(preview.airspace) ||
            preview.queryables.some(moves) ||
            preview.spawn_regions.some(moves) ||
            Object.values(preview.shapes ?? {}).some(moves),
        );
        refreshDeck();
        refreshLabels();
        setInfo(`${preview.sampled_aircraft.length} aircraft · max ${preview.max_aircraft} · ${preview.queryables.length} queryables`);
        setWarnings(preview.airspace_warnings ?? []);
      })
      .catch((e) => !canceled && setError(String(e)))
      .finally(() => !canceled && setLoading(false));

    return () => {
      canceled = true;
    };
  }, [spec, ready, seed, refreshKey]);

  const focusShape = (bounds: SpecDict) => {
    const map = mapRef.current;
    const fp = bounds?.footprint;
    if (!map || !fp) return;
    if (fp.type === "box") {
      map.fitBounds([[fp.lon_min_deg, fp.lat_min_deg], [fp.lon_max_deg, fp.lat_max_deg]], { padding: 80, duration: 500, maxZoom: 10 });
    } else if (fp.center) {
      map.flyTo({ center: [fp.center.lon_deg, fp.center.lat_deg], zoom: 9 });
    }
  };

  const flyTo = (lon: number, lat: number) => mapRef.current?.flyTo({ center: [lon, lat], zoom: 10 });

  // A navdb fix, added from the search: a waypoint on a point at the fix.
  const addWaypoint = (ident: string, lat: number, lon: number) => {
    if (!spec) return;
    const next = structuredClone(spec);
    const altRange = placementAltitudeRange(next.airspace);
    const altFt = altRange ? (altRange[0] + altRange[1]) / 2 : undefined;
    addWaypointAt(next, ident.toLowerCase(), lat, lon, ident, altFt);
    onSpecChange(next);
  };

  const toggleCategory = (key: keyof CategoryVisibility) =>
    setVisibility((v) => ({ ...v, [key]: !v[key] }));

  const currentHandles = buildEditHandles(
    draftSpecRef.current ?? specRef.current,
    previewRef.current,
    selectedTarget,
  ).filter((handle) => !lockedElements.has(targetKey(handle.target)));
  const handleViews = currentHandles
    .map((handle) => {
      const map = mapRef.current;
      if (!map) return null;
      const p = map.project([handle.lon, handle.lat]);
      return { handle, x: p.x, y: p.y };
    })
    .filter(Boolean) as { handle: EditHandle; x: number; y: number }[];
  void handleTick;

  return (
    <div className="map-tab">
      <div className="map-area">
        <div ref={containerRef} className="map-canvas" />
        <div className="edit-handle-layer">
          {handleViews.map(({ handle, x, y }) => {
            const isRotate = handle.role === "rotate";
            const isMove = handle.role === "move-shape";
            const color = `rgba(${handle.color[0]}, ${handle.color[1]}, ${handle.color[2]}, 1)`;
            const mask = isRotate ? ROTATION_HANDLE_ICON : undefined;
            return (
              <button
                key={handle.id}
                title={isMove ? "drag to move the whole shape" : handle.role}
                className={`edit-handle ${isMove ? "move" : ""} ${isRotate ? "rotation" : ""}`}
                style={{
                  left: x,
                  top: y,
                  borderColor: color,
                  // Move = a solid filled circle in the element color (distinct
                  // from the white-filled corner handles); rotate = masked icon.
                  backgroundColor: mask ? color : isMove ? color : "rgba(255,255,255,0.94)",
                  WebkitMaskImage: mask ? `url("${mask}")` : undefined,
                  maskImage: mask ? `url("${mask}")` : undefined,
                }}
                onPointerDown={(e) => startDomDrag(handle, e)}
                onClick={(e) => {
                  e.stopPropagation();
                  selectTarget(handle.target);
                }}
              >
                {isMove && (
                  <span
                    className="move-glyph"
                    style={{
                      WebkitMaskImage: `url("${MOVE_HANDLE_ICON}")`,
                      maskImage: `url("${MOVE_HANDLE_ICON}")`,
                    }}
                  />
                )}
              </button>
            );
          })}
        </div>
        <div className="map-overlay">
          <SearchBox onFlyTo={flyTo} onAddWaypoint={addWaypoint} near={viewCenter} />
          <div className="overlay-section">
            <div className="overlay-title">map</div>
            <select
              className="basemap-select"
              value={basemap}
              onChange={(e) => setBasemap(e.target.value as BasemapId)}
              title={basemapById(basemap).description}
            >
              {BASEMAPS.map((entry) => (
                <option key={entry.id} value={entry.id}>
                  {entry.label}
                </option>
              ))}
            </select>
          </div>
          <div className="overlay-section motion-control">
            <div className="overlay-title">episode</div>
            <EpisodeStepper busy={loading} label={false} />
            {pick && visibility.aircraft && (
              <div className="muted small episode-pick">
                picked {pick.acid ?? pick.actype} · what it observes: Spaces › Sample
              </div>
            )}
            {hasMotion && (
              <div className="motion-row">
                <button
                  type="button"
                  className="icon-btn"
                  onClick={() => setPlaying((p) => !p)}
                  aria-label={playing ? "Pause the motion" : "Play the motion"}
                  title={playing ? "pause" : "play the next hour"}
                >
                  {playing ? "❚❚" : "▶"}
                </button>
                <input
                  type="range"
                  min={0}
                  max={3600}
                  step={60}
                  value={motionT}
                  aria-label="Time into the episode"
                  onChange={(e) => {
                    setPlaying(false);
                    showMotionAt(Number(e.target.value));
                  }}
                />
                <span className="motion-time">+{Math.round(motionT / 60)} min</span>
              </div>
            )}
          </div>
          <div className="overlay-section">
            <div className="overlay-title">layers</div>
            <div className="layer-toggles">
              {CATEGORIES.map(({ key, label }) => (
                <label key={key} className={visibility[key] ? "" : "off"}>
                  <input type="checkbox" checked={visibility[key]} onChange={() => toggleCategory(key)} /> {label}
                </label>
              ))}
            </div>
          </div>
          <div className="overlay-controls">
            <label className="line-width" title="line thickness">
              lines
              <input
                type="range"
                min={1}
                max={4}
                step={0.5}
                value={lineScale}
                onChange={(e) => changeLineScale(parseFloat(e.target.value))}
              />
            </label>
          </div>
          <div className="map-info">{info}</div>
          {visibility.aircraft && (
            <div className="map-info muted small">
              {flying
                ? `running the episode… ${spawns ? `${spawns.aircraft.length} aircraft so far` : "starting the simulator"}`
                : spawns
                  ? `aircraft as flown: ${spawns.aircraft.length}${spawns.done ? `, up by t+${Math.round(spawns.done.sim_time_s)} s` : ""}`
                  : ""}
            </div>
          )}
          {selectedTarget && (
            <div className="map-hint muted small">selected · press Delete to remove</div>
          )}
          {warnings.length > 0 && (
            <div className="map-warning">⚠ outside airspace: {warnings.join(", ")}</div>
          )}
          {error && <div className="map-error">{error}</div>}
          {!spec && <div className="map-error">spec JSON is invalid — fix it in the Code tab</div>}
        </div>
      </div>
      {spec && (
        <div
          className="panel-resizer"
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize design panel"
          title="Drag to resize · double-click to reset"
          onPointerDown={startPanelResize}
          onDoubleClick={() => setPanelWidth(DEFAULT_PANEL_WIDTH)}
        />
      )}
      {spec && (
        <DesignPanel
          width={panelWidth}
          spec={spec}
          onChange={onSpecChange}
          onFocusShape={focusShape}
          viewCenter={viewCenter}
          hiddenElements={hiddenElements}
          onToggleHidden={toggleHidden}
          lockedElements={lockedElements}
          onToggleLocked={toggleLocked}
          onHighlightRoute={setHighlight}
          selectedKey={selectedTarget ? targetKey(selectedTarget) : null}
          onSelect={selectTarget}
        />
      )}
    </div>
  );
}
