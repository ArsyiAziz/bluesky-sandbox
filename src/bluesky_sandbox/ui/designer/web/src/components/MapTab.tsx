import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import maplibregl from "maplibre-gl";
import { MapboxOverlay } from "@deck.gl/mapbox";
import { api, type SpecDict, type PreviewResult, type NavFeatures, type ValidateResult } from "../api";
import DesignPanel from "./DesignPanel";
import SearchBox from "./SearchBox";
import type { CategoryVisibility, EditHandle, EditTarget } from "../map/types";
import { centroid, point, routePaths } from "../map/geometry";
import { buildEditHandles, dragStateForHandle, targetKey, updateSpecFromHandle } from "../map/editHandles";
import { EMPTY, LABEL_FONT, MOVE_HANDLE_ICON, ROTATION_HANDLE_ICON, deckLayers, getTooltip } from "../map/deckLayers";
import { setColorPalette } from "../map/geometry";
import { BASEMAPS, DEFAULT_BASEMAP, basemapById, type BasemapId } from "../map/basemaps";
import { defaultWaypoint, gcOrphanBounds, placementAltitudeRange } from "../specHelpers";

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
  validation,
}: {
  spec: SpecDict | null;
  onSpecChange: (next: SpecDict) => void;
  validation: ValidateResult | null;
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
  const [seed, setSeed] = useState(0);
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
    const next = structuredClone(specRef.current);
    if (target.scope === "airspace") next.airspace = null;
    else if (target.scope === "queryable") delete next.queryables?.[target.name];
    else if (target.scope === "spawn") next.spawn?.regions?.splice(target.index, 1);
    else if (target.scope === "region") {
      // A standalone-drawn bounds (e.g. a waypoint sample area): drop the bounds
      // and any waypoint sample pointing at it, so no reference dangles.
      delete next.regions?.[target.name];
      for (const q of Object.values(next.queryables ?? {}) as any[]) {
        if (q?.sample?.ref === target.name) delete q.sample;
      }
    }
    // Bounds are created only via elements; sweep any now left unreferenced.
    gcOrphanBounds(next);
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
        ),
      });
    }
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
        paint: { "text-color": "#fff", "text-halo-color": "#000", "text-halo-width": 1.2 },
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

  // Load the colour palette (same list the picker shows + the drivers render) so
  // named colours like black / gray / purple resolve in the map preview.
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
  }, [ready]);

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

  // Re-render geometry whenever the (valid) spec / seed changes.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !spec) return;
    let cancelled = false;
    setLoading(true);

    api
      .preview(spec, seed)
      .then((preview: PreviewResult) => {
        if (cancelled) return;
        setError(null);
        previewRef.current = preview;
        refreshDeck();
        refreshLabels();
        setInfo(`${preview.sampled_aircraft.length} aircraft · max ${preview.max_aircraft} · ${preview.queryables.length} queryables`);
        setWarnings(preview.airspace_warnings ?? []);
      })
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));

    return () => {
      cancelled = true;
    };
  }, [spec, ready, seed]);

  const focusBounds = (bounds: SpecDict) => {
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

  const addWaypoint = (ident: string) => {
    if (!spec) return;
    const next = structuredClone(spec);
    next.queryables = next.queryables ?? {};
    let name = ident.toLowerCase();
    let n = 1;
    while (next.queryables[name]) name = `${ident.toLowerCase()}_${n++}`;
    const altRange = placementAltitudeRange(next.airspace);
    const altFt = altRange ? (altRange[0] + altRange[1]) / 2 : undefined;
    next.queryables[name] = defaultWaypoint(52.0, 4.75, ident.toUpperCase(), altFt);
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
                  // Move = a solid filled circle in the element colour (distinct
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
          <SearchBox onFlyTo={flyTo} onAddWaypoint={addWaypoint} />
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
            <button disabled={loading} onClick={() => setSeed((s) => s + 1)} title="resample aircraft + airspace config">
              reseed
            </button>
            <button onClick={() => setSeed(0)} disabled={loading || seed === 0} title="reset to seed 0">
              reset
            </button>
            <span className="muted small">seed {seed}</span>
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
          onFocusBounds={focusBounds}
          seed={seed}
          onSeedChange={setSeed}
          viewCenter={viewCenter}
          hiddenElements={hiddenElements}
          onToggleHidden={toggleHidden}
          lockedElements={lockedElements}
          onToggleLocked={toggleLocked}
          onHighlightRoute={setHighlight}
          selectedKey={selectedTarget ? targetKey(selectedTarget) : null}
          onSelect={selectTarget}
          validationError={validation && !validation.ok ? validation.error : undefined}
        />
      )}
    </div>
  );
}
