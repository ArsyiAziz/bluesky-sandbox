// The Geometry tab, rebuilt as a master–detail view:
//   - an OUTLINE (compact, one row per element, grouped by kind) for navigation
//     and add/visibility, kept in sync with the map selection; and
//   - an INSPECTOR that edits only the currently-selected element, in place of
//     the outline while one is picked (back returns to it).
// The spec object is the source of truth; every edit yields a new spec via
// onChange (which App also re-serializes into the code editor).
import { Hint } from "./Hint";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, type SpecDict } from "../../api";
import { type ShapeInfo, shapeGraph, isUnused, renameShapeRefs } from "../../shapeGraph";
import ShapeEditor, { MotionEditor } from "../ShapeEditor";
import { ShapesTree, DependencyTree } from "./ShapesTree";
import { Row } from "./OutlineRow";
import { PointShapesProvider, pointShapesOf } from "../../pointShapes";
import { addWaypointAt, pointRegion } from "../../waypointPoints";
import {
  anchoredWaypoints,
  shapesInGroup,
  childGroups,
  dissolveGroup,
  type Group,
  groupOfShape,
  groupTogether,
  isWithin,
  moveShape,
  moveGroup,
} from "../../groupTree";
import {
  shapeRefLabels,
  clippedSpawnAltitudeRange,
  clone,
  countShapeRefs,
  defaultConstantBand,
  defaultQueryRegion,
  defaultRegion,
  defaultSpawnRegion,
  gcOrphanShapes,
  partitionNames,
  placementAltitudeRange,
  placementCenter,
} from "../../specHelpers";
import type { EditTarget } from "../../map/types";
import { targetKey } from "../../map/editHandles";
import { EyeToggle, LockToggle } from "./Section";
import { FieldGroup } from "./FieldGroup";
import { Picker } from "./Picker";
import { ValueField, NumInput } from "./ValueField";
import { QueryableBody } from "./QueryableCard";
import { SpawnBody } from "./SpawnCard";
import { RouteSettings } from "./RouteSettings";
import { useRefresh } from "../../refresh";

const emptySpawn = (): SpecDict => ({ type: "spawn_config", regions: [], aircraft_type: null, route: null, routes: {} });

// Selection covers the map's edit targets plus a panel-only "routes" view (the
// route library has no single map target, so it's selected from the outline).
type Sel = EditTarget | { scope: "routes" } | null;

const selKey = (s: Sel): string =>
  s && s.scope === "routes" ? s.scope : targetKey(s as EditTarget | null);

function parseTargetKey(key: string | null | undefined): EditTarget | null {
  if (!key) return null;
  if (key === "airspace") return { scope: "airspace" };
  if (key.startsWith("queryable:")) return { scope: "queryable", name: key.slice("queryable:".length) };
  if (key.startsWith("region:")) return { scope: "region", name: key.slice("region:".length) };
  if (key.startsWith("spawn:")) return { scope: "spawn", index: Number(key.slice("spawn:".length)) };
  if (key.startsWith("group:")) return { scope: "group", id: key.slice("group:".length) };
  return null;
}

export default function GeometryTab({
  spec,
  onChange,
  onFocusShape,
  viewCenter,
  hiddenElements,
  onToggleHidden,
  lockedElements,
  onToggleLocked,
  onHighlightRoute,
  selectedKey,
  onSelect,
}: {
  spec: SpecDict;
  onChange: (next: SpecDict) => void;
  onFocusShape: (bounds: SpecDict) => void;
  viewCenter?: [number, number];
  hiddenElements: Set<string>;
  onToggleHidden: (key: string) => void;
  lockedElements: Set<string>;
  onToggleLocked: (key: string) => void;
  onHighlightRoute: (name: string | null) => void;
  selectedKey?: string | null;
  onSelect: (target: EditTarget | null) => void;
}) {
  const [filter, setFilter] = useState("");
  const [showAllShapes, setShowAllShapes] = useState(false);
  // The outline: elements by kind, or bounds by what depends on what.
  const [view, setView] = useState<"kind" | "dependency">("kind");
  // Panel-only "routes" selection; a map selection (selectedKey) supersedes it.
  const [routesView, setRoutesView] = useState(false);
  useEffect(() => {
    if (selectedKey) {
      setRoutesView(false);
    }
  }, [selectedKey]);

  // The zone conflict-free spawns are cleared against, from BlueSky's CD.
  const [sep, setSep] = useState<Record<string, number>>({});
  // SpawnConfig's own retry defaults, shown when a field is left empty.
  const [spawnDefaults, setSpawnDefaults] = useState<Record<string, number>>({});
  // The shape generators, for naming a generated partition's shapes.
  const [generators, setGenerators] = useState<any[]>([]);
  const refreshKey = useRefresh();
  useEffect(() => {
    api
      .catalogOnce()
      .then((c) => {
        setSep(c?.separation ?? {});
        setSpawnDefaults(c?.spawn_defaults ?? {});
        setGenerators(c?.generators ?? []);
      })
      .catch(() => setSep({}));
  }, [refreshKey]);
  const sepZone =
    sep.pz_radius_nm === undefined
      ? "the protected zone"
      : `${sep.pz_radius_nm} nm / ${sep.pz_height_ft} ft / ${sep.lookahead_s} s`;

  const has = (n: string) => !filter || n.toLowerCase().includes(filter.toLowerCase());

  const edit = (mut: (s: SpecDict) => void) => {
    const next = clone(spec);
    mut(next);
    // Sweep bounds a delete or reassignment left unreferenced, so orphans
    // never accumulate; bounds added on their own stay.
    gcOrphanShapes(next, spec);
    onChange(next);
  };

  // Which area is the airspace: a role, so moving it keeps the area it leaves
  // (no orphan sweep - an area that was the airspace is still a shape).
  const setAirspaceRef = (ref: string | null) => {
    const next = clone(spec);
    next.airspace = ref ? { ref } : null;
    onChange(next);
  };

  // ---- selection ----------------------------------------------------------
  // The airspace is a role a shape plays: picking it opens that shape.
  const picked = parseTargetKey(selectedKey);
  const airspaceRef: string | undefined = spec.airspace?.ref;
  const sel: Sel = routesView
    ? { scope: "routes" }
    : picked?.scope === "airspace" && airspaceRef
      ? { scope: "region", name: airspaceRef }
      : picked;
  // Elements picked from the list start afresh; ones reached through a
  // dependency (hop) remember the way back.
  const [trail, setTrail] = useState<EditTarget[]>([]);
  const selectTarget = (t: EditTarget) => {
    setRoutesView(false);
    setTrail([]);
    onSelect(t);
  };
  const hop = (t: EditTarget) => {
    const here = parseTargetKey(selectedKey);
    if (here && targetKey(here) !== targetKey(t)) setTrail((prev) => [...prev, here]);
    setRoutesView(false);
    onSelect(t);
  };
  const hopBack = () => {
    const prev = trail[trail.length - 1];
    if (!prev) return;
    setTrail(trail.slice(0, -1));
    onSelect(prev);
  };
  const selectRoutes = () => {
    setRoutesView(true);
    onSelect(null);
  };

  // ---- placement seeds (where a freshly added element lands) ---------------
  const [viewLat, viewLon] = viewCenter ?? [52.0, 4.75];
  const [addLat, addLon] = placementCenter(spec.airspace, [viewLat, viewLon]);
  const addAltRange = placementAltitudeRange(spec.airspace);
  const addRegionAltitude = addAltRange ? defaultConstantBand(addAltRange[0], addAltRange[1]) : undefined;
  const addWaypointAlt = addAltRange ? (addAltRange[0] + addAltRange[1]) / 2 : undefined;
  const [spawnAltLo, spawnAltHi] = clippedSpawnAltitudeRange(spec.airspace);

  // ---- named bounds (shared geometry library) -----------------------------
  const regions: Record<string, SpecDict> = spec.shapes ?? {};
  const namedRegionNames = Object.keys(regions);
  // Who uses which bounds, and what each depends on.
  const graph = useMemo(() => shapeGraph(spec), [spec]);
  // Point bounds: left out wherever an area is needed.
  const pointShapes = useMemo(() => pointShapesOf(spec), [spec]);
  const unusedShapes = namedRegionNames.filter((n) => isUnused(graph[n]));
  // What an element can refer to: every named bounds, and each shape of a
  // generated partition (`sectors.0`, ...) - redrawn each episode.
  const partitions: Record<string, string[]> = Object.fromEntries(
    namedRegionNames.map((n) => [n, partitionNames(n, regions[n], generators)]),
  );
  const refRegionNames = namedRegionNames.flatMap((n) => [n, ...partitions[n]]);
  const resolveRef = (b: any): SpecDict | undefined => (b?.ref ? regions[b.ref] : b);
  const resolveShapeByName = (name: string): SpecDict | undefined => regions[name];
  const shapeRefCount = (name: string): number => countShapeRefs(spec, name);
  const setRegion = (name: string, b: SpecDict) => edit((s) => (s.shapes[name] = b));
  const renameRegion = (oldName: string, newNameRaw: string) => {
    const newName = newNameRaw.trim();
    if (!newName || newName === oldName || spec.shapes?.[newName]) return;
    edit((s) => {
      s.shapes[newName] = s.shapes[oldName];
      delete s.shapes[oldName];
      rewriteRegionRefs(s, oldName, newName);
    });
    // Keep the renamed bounds selected (its key changed).
    onSelect({ scope: "region", name: newName });
  };

  // ---- queryables ---------------------------------------------------------
  const queryableEntries = Object.entries(spec.queryables ?? {}) as [string, SpecDict][];
  const regionEntries = queryableEntries.filter(([, q]) => q.type !== "waypoint");
  const waypointEntries = queryableEntries.filter(([, q]) => q.type === "waypoint");
  const waypointNames = waypointEntries.map(([name]) => name);
  // Query-region names (for the waypoint TSAS-bound picker).
  const regionNames = regionEntries.filter(([, q]) => q.type === "query_region").map(([name]) => name);

  const renameQueryable = (name: string, newNameRaw: string) => {
    const newName = newNameRaw.trim();
    if (!newName || newName === name || spec.queryables?.[newName]) return;
    edit((s) => {
      s.queryables[newName] = s.queryables[name];
      delete s.queryables[name];
      syncShapeName(s, s.queryables[newName].shape?.ref, newName);
      syncShapeName(s, s.queryables[newName].sample?.ref, `${newName}_sample`);
    });
    // Keep the renamed queryable selected (its key changed).
    onSelect({ scope: "queryable", name: newName });
  };

  const spawnRegions: SpecDict[] = spec.spawn?.regions ?? [];
  const routeNames = Object.keys(spec.spawn?.routes ?? {});

  // ---- groups ---------------------------------------------------------------
  // A group holds bounds and other groups (see groupTree): moved, rotated,
  // randomized per episode and moved during one together.
  const groups = (spec.transform?.groups ?? []) as Group[];
  const updateGroup = (id: string, patch: SpecDict) =>
    edit((s) => {
      s.transform.groups = s.transform.groups.map((g: SpecDict) => (g.id === id ? { ...g, ...patch } : g));
    });
  const ungroup = (id: string) => {
    onSelect(null);
    edit((s) => dissolveGroup(s, id));
  };

  // ---- adders (each selects the new element so the inspector opens) --------
  const addRegion = () => {
    let qname = "";
    edit((s) => {
      const qr = defaultQueryRegion(addLat, addLon, addRegionAltitude);
      const inline = qr.shape;
      qname = addQueryable(s, qr, "region");
      s.queryables[qname].shape = { ref: addRegionTo(s, inline, qname) };
    });
    if (qname) selectTarget({ scope: "queryable", name: qname });
  };
  const addWaypoint = () => {
    let qname = "";
    edit((s) => {
      qname = addWaypointAt(s, "waypoint", addLat, addLon, undefined, addWaypointAlt);
    });
    if (qname) selectTarget({ scope: "queryable", name: qname });
  };
  const addSpawn = () => {
    const index = (spec.spawn?.regions ?? []).length;
    edit((s) => {
      s.spawn = s.spawn ?? emptySpawn();
      s.spawn.regions = s.spawn.regions ?? [];
      const sr = defaultSpawnRegion(addLat, addLon, spawnAltLo, spawnAltHi);
      sr.shape = { ref: addRegionTo(s, sr.shape, sr.name || "spawn") };
      s.spawn.regions.push(sr);
    });
    selectTarget({ scope: "spawn", index });
  };
  const addShape = () => {
    let name = "";
    edit((s) => {
      name = addRegionTo(s, defaultRegion(addLat, addLon, addRegionAltitude ?? null), "shape");
    });
    if (name) selectTarget({ scope: "region", name });
  };
  // A queryable region for each shape of a generated partition, so each can
  // be asked "is this aircraft inside?" on its own.
  const addPartitionQueryables = (name: string) =>
    edit((s) => {
      for (const part of partitionNames(name, s.shapes?.[name], generators)) {
        const taken = Object.values(s.queryables ?? {}).some((q: any) => q?.shape?.ref === part);
        if (taken) continue;
        const q = { ...defaultQueryRegion(addLat, addLon, addRegionAltitude), shape: { ref: part } };
        addQueryable(s, q, part.replace(".", "_"));
      }
    });
  // Delete a bounds - but not out from under what uses it.
  const deleteShape = (name: string) => {
    const info = graph[name];
    const users = info?.usedBy ?? [];
    if (users.length) {
      window.alert(
        `“${name}” is used by ${users.map((u) => u.label).join(", ")}. ` +
          "Point them at another shape, or delete them, first.",
      );
      return;
    }
    if (info?.inCode.length && !window.confirm(`Code reads “${name}” (${info.inCode.join(", ")}). Delete it anyway?`)) return;
    onSelect(null);
    edit((s) => {
      delete s.shapes[name];
      moveShape(s, name, null);
    });
  };
  // Shared bounds shown in the outline: those referenced by >1 element (or all,
  // when the user opts in). Single-use bounds are edited inline on their element.
  // Bounds shared by several elements, or by none - added on their own - list
  // here; one used by a single element is edited through it.
  const shapeRows = namedRegionNames
    .filter((n) => showAllShapes || n === airspaceRef || isUnused(graph[n]) || (graph[n]?.usedBy.length ?? 0) !== 1)
    // The airspace first: the sector everything else is in.
    .sort((a, b) => Number(b === airspaceRef) - Number(a === airspaceRef));

  // Open a picked element's editor at its top, not wherever the list was.
  const tabRef = useRef<HTMLDivElement | null>(null);
  const pickedKey = selKey(sel);
  useEffect(() => {
    if (pickedKey) tabRef.current?.closest(".design-panel")?.scrollTo({ top: 0 });
  }, [pickedKey]);

  return (
    // With an element picked, its editor takes the panel; back returns to the list.
    <PointShapesProvider value={pointShapes}>
    <div className={sel ? "geo-tab drilled" : "geo-tab"} ref={tabRef}>
      <div className="geo-outline">
        <input
          className="sec-search geo-search"
          placeholder="filter elements…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        <div className="seg geo-view" role="radiogroup" aria-label="Outline view">
          {(["kind", "dependency"] as const).map((v) => (
            <button
              key={v}
              type="button"
              role="radio"
              aria-checked={view === v}
              className={view === v ? "on" : ""}
              onClick={() => setView(v)}
            >
              by {v}
            </button>
          ))}
        </div>

        {view === "dependency" ? (
          <GeoGroup
            title="Dependencies"
            hint="Each shape under the shape it is drawn, placed or moved by, with the elements that use it beneath. A shape that depends on several shows in full once, then as ↑ shown above."
          >
            <DependencyTree spec={spec} graph={graph} has={has} selectedKey={selKey(sel)} onSelect={selectTarget} />
          </GeoGroup>
        ) : (
        <>

        <GeoGroup title="Regions" hint="Named volumes the task can query - 'is this aircraft inside?' - and that spawns can draw positions from. A region's footprint may carry sampled parameters, redrawn each episode." onAdd={addRegion} addLabel="region">
          {regionEntries.filter(([n]) => has(n)).map(([name]) => (
            <Row
              key={name}
              name={name}
              kind="region"
              selected={selKey(sel) === `queryable:${name}`}
              onClick={() => selectTarget({ scope: "queryable", name })}
              hidden={hiddenElements.has(`queryable:${name}`)}
              onToggleHidden={() => onToggleHidden(`queryable:${name}`)}
              locked={lockedElements.has(`queryable:${name}`)}
              onToggleLocked={() => onToggleLocked(`queryable:${name}`)}
            />
          ))}
          {regionEntries.length === 0 && <div className="muted small">no regions</div>}
        </GeoGroup>

        <GeoGroup title="Waypoints" hint="Named fixes aircraft can be routed to. A waypoint may gate position only, or also an altitude and a crossing speed." onAdd={addWaypoint} addLabel="waypoint">
          {waypointEntries.filter(([n]) => has(n)).map(([name]) => (
            <Row
              key={name}
              name={name}
              kind="waypoint"
              selected={selKey(sel) === `queryable:${name}`}
              onClick={() => selectTarget({ scope: "queryable", name })}
              hidden={hiddenElements.has(`queryable:${name}`)}
              onToggleHidden={() => onToggleHidden(`queryable:${name}`)}
              locked={lockedElements.has(`queryable:${name}`)}
              onToggleLocked={() => onToggleLocked(`queryable:${name}`)}
            />
          ))}
          {waypointEntries.length === 0 && <div className="muted small">no waypoints</div>}
        </GeoGroup>

        <GeoGroup title="Routes" hint="Ordered sequences of waypoints an aircraft flies. A route step can be a single fix, a junction, or a weighted choice between branches.">
          <Row
            name={`route library${routeNames.length ? ` · ${routeNames.length}` : ""}`}
            kind="routes"
            selected={selKey(sel) === "routes"}
            onClick={selectRoutes}
          />
        </GeoGroup>

        <GeoGroup title="Spawn" hint="Where aircraft appear and how many. Each spawn region draws a count per episode and samples positions, types, and entry states inside its shape." onAdd={addSpawn} addLabel="spawn">
          {/* Settings for every spawn region at once, folded away so the
              regions themselves stay in view. */}
          <details className="geo-settings">
            <summary>spawn settings</summary>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={spec.spawn?.conflict_free_spawn === true}
                onChange={(e) =>
                  edit((s) => {
                    s.spawn = s.spawn ?? emptySpawn();
                    s.spawn.conflict_free_spawn = e.target.checked;
                  })
                }
              />
              <span>conflict-free spawn</span>
            </label>
            {spec.spawn?.conflict_free_spawn === true && (
              <div>
                <div className="value-field">
                  <div className="vf-head">
                    <span className="vf-label">buffer horiz nm</span>
                    <span className="vf-spacer" />
                    <NumInput
                      className="vf-input"
                      step="any"
                      placeholder="none"
                      value={
                        (spec.spawn?.conflict_free_margin_nm as
                          | number
                          | undefined) ?? Number.NaN
                      }
                      onChange={(n) =>
                        edit((s) => {
                          s.spawn = s.spawn ?? emptySpawn();
                          s.spawn.conflict_free_margin_nm = n;
                        })
                      }
                      onClear={() =>
                        edit((s) => {
                          s.spawn = s.spawn ?? emptySpawn();
                          delete s.spawn.conflict_free_margin_nm;
                        })
                      }
                    />
                  </div>
                </div>
                <div className="value-field">
                  <div className="vf-head">
                    <span className="vf-label">buffer vert ft</span>
                    <span className="vf-spacer" />
                    <NumInput
                      className="vf-input"
                      step="any"
                      placeholder="none"
                      value={
                        (spec.spawn?.conflict_free_margin_ft as
                          | number
                          | undefined) ?? Number.NaN
                      }
                      onChange={(n) =>
                        edit((s) => {
                          s.spawn = s.spawn ?? emptySpawn();
                          s.spawn.conflict_free_margin_ft = n;
                        })
                      }
                      onClear={() =>
                        edit((s) => {
                          s.spawn = s.spawn ?? emptySpawn();
                          delete s.spawn.conflict_free_margin_ft;
                        })
                      }
                    />
                  </div>
                </div>
                <div className="value-field">
                  <div className="vf-head">
                    <span className="vf-label">buffer time s</span>
                    <span className="vf-spacer" />
                    <NumInput
                      className="vf-input"
                      step="any"
                      placeholder="none"
                      value={
                        (spec.spawn?.conflict_free_margin_s as
                          | number
                          | undefined) ?? Number.NaN
                      }
                      onChange={(n) =>
                        edit((s) => {
                          s.spawn = s.spawn ?? emptySpawn();
                          s.spawn.conflict_free_margin_s = n;
                        })
                      }
                      onClear={() =>
                        edit((s) => {
                          s.spawn = s.spawn ?? emptySpawn();
                          delete s.spawn.conflict_free_margin_s;
                        })
                      }
                    />
                  </div>
                </div>
                <div className="muted small">
                  reject spawns whose predicted CPA comes within zone + buffer
                  ({sepZone}), so clearance holds as aircraft maneuver; blank adds
                  none ↑
                </div>
              </div>
            )}
            {/* Only meaningful while some maintain area actually uses the
                distance guard - i.e. is not (effectively) conflict-free. */}
            {spawnRegions.some(
              (r) =>
                r.maintain === true &&
                (r.conflict_free_spawn === undefined ||
                r.conflict_free_spawn === null
                  ? spec.spawn?.conflict_free_spawn !== true
                  : r.conflict_free_spawn !== true),
            ) && (
              <div>
                <div className="value-field">
                  <div className="vf-head">
                    <span className="vf-label">respawn min sep nm</span>
                    <span className="vf-spacer" />
                    <NumInput
                      className="vf-input"
                      step="any"
                      placeholder="protected zone"
                      value={
                        (spec.spawn?.maintain_min_sep_nm as number | undefined) ??
                        Number.NaN
                      }
                      onChange={(n) =>
                        edit((s) => {
                          s.spawn = s.spawn ?? emptySpawn();
                          s.spawn.maintain_min_sep_nm = n;
                        })
                      }
                      onClear={() =>
                        edit((s) => {
                          s.spawn = s.spawn ?? emptySpawn();
                          delete s.spawn.maintain_min_sep_nm;
                        })
                      }
                    />
                  </div>
                </div>
                <div className="muted small">
                  steady-density top-ups must be this far from live traffic; empty
                  uses the protected zone, 0 disables. Areas spawning
                  conflict-free use the predicted-conflict check instead ↑
                </div>
              </div>
            )}
            <div>
                <div className="value-field">
                  <div className="vf-head">
                    <span className="vf-label">spawn max tries</span>
                    <span className="vf-spacer" />
                    <NumInput
                      className="vf-input"
                      int
                      step={1}
                      placeholder={`${spawnDefaults.spawn_max_tries ?? ""} (default)`}
                      value={(spec.spawn?.spawn_max_tries as number | undefined) ?? Number.NaN}
                      onChange={(n) =>
                        edit((s) => {
                          s.spawn = s.spawn ?? emptySpawn();
                          s.spawn.spawn_max_tries = Math.max(1, Math.round(n));
                        })
                      }
                      onClear={() =>
                        edit((s) => {
                          s.spawn = s.spawn ?? emptySpawn();
                          delete s.spawn.spawn_max_tries;
                        })
                      }
                    />
                  </div>
                </div>
              <div className="muted small">
                clear spawn states to try before deferring to the next step
              </div>
              {spawnRegions.some((r) => r.maintain === true) && (
                <>
                  <div className="value-field">
                    <div className="vf-head">
                      <span className="vf-label">warn after failed top-ups</span>
                      <span className="vf-spacer" />
                      <NumInput
                        className="vf-input"
                        int
                        step={1}
                        placeholder={`${spawnDefaults.spawn_warn_after ?? ""} (default)`}
                        value={(spec.spawn?.spawn_warn_after as number | undefined) ?? Number.NaN}
                        onChange={(n) =>
                          edit((s) => {
                            s.spawn = s.spawn ?? emptySpawn();
                            s.spawn.spawn_warn_after = Math.max(1, Math.round(n));
                          })
                        }
                        onClear={() =>
                          edit((s) => {
                            s.spawn = s.spawn ?? emptySpawn();
                            delete s.spawn.spawn_warn_after;
                          })
                        }
                      />
                    </div>
                  </div>
                  <div className="muted small">
                    warn once when a maintain area misses its count this many times
                    in a row
                  </div>
                </>
              )}
              <div className="value-field">
                <div className="vf-head">
                  <span className="vf-label">aircraft cap</span>
                  <span className="vf-spacer" />
                  <NumInput
                    className="vf-input"
                    int
                    step={1}
                    placeholder="none - from the regions"
                    value={(spec.spawn?.aircraft_cap as number | undefined) ?? Number.NaN}
                    onChange={(n) =>
                      edit((s) => {
                        s.spawn = s.spawn ?? emptySpawn();
                        s.spawn.aircraft_cap = Math.max(1, Math.round(n));
                      })
                    }
                    onClear={() =>
                      edit((s) => {
                        s.spawn = s.spawn ?? emptySpawn();
                        delete s.spawn.aircraft_cap;
                      })
                    }
                  />
                </div>
              </div>
              <div className="muted small">
                at most this many aircraft at once, from the regions or code (<code>env.spawn</code>). None: the most
                the regions can produce, code uncapped. Fixed-size padding sizes from it.
              </div>
            </div>
          </details>
          {spawnRegions
            .map((r, i): [SpecDict, number] => [r, i])
            .filter(([r, i]) => has(r.name || `spawn_${i + 1}`))
            .map(([r, i]) => (
              <Row
                key={i}
                name={r.name || `spawn_${i + 1}`}
                kind="spawn"
                selected={selKey(sel) === `spawn:${i}`}
                onClick={() => selectTarget({ scope: "spawn", index: i })}
                hidden={hiddenElements.has(`spawn:${i}`)}
                onToggleHidden={() => onToggleHidden(`spawn:${i}`)}
                locked={lockedElements.has(`spawn:${i}`)}
                onToggleLocked={() => onToggleLocked(`spawn:${i}`)}
              />
            ))}
          {spawnRegions.length === 0 && <div className="muted small">no spawn regions</div>}
        </GeoGroup>

        <GeoGroup
          title="Shapes"
          hint="Named areas and points elements are on, and the groups that hold them. Drag a shape onto another to group the two, onto a group to join it, a group onto a group to nest it - or to the top to take it out. A group moves, rotates and randomizes all it holds together."
          onAdd={addShape}
          addLabel="shape"
          action={
            namedRegionNames.length > 0 ? (
              <button className="link" onClick={() => setShowAllShapes((v) => !v)}>
                {showAllShapes ? "shared only" : "show all"}
              </button>
            ) : undefined
          }
        >
          {/* The airspace: which area is the sector - a role, set here or on the area. */}
          <label className="numfield inline geo-airspace">
            <span>airspace</span>
            <Picker
              placeholder="— none"
              value={airspaceRef ?? ""}
              onChange={(v) => setAirspaceRef(v || null)}
              options={[
                { value: "", label: "— none", description: "every aircraft counts as inside" },
                ...namedRegionNames
                  .filter((n) => !pointShapes.has(n))
                  .map((n) => ({ value: n, description: n === airspaceRef ? "the sector" : undefined })),
              ]}
            />
            {airspaceRef && (
              <button type="button" className="link" onClick={() => selectTarget({ scope: "region", name: airspaceRef })}>
                open
              </button>
            )}
          </label>
          {!airspaceRef && namedRegionNames.length > 0 && (
            <div className="geo-flag-note" role="note">
              <span aria-hidden="true">⚠</span> no airspace: pick the area that is the sector
            </div>
          )}
          {unusedShapes.length > 0 && (
            <div className="geo-flag-note" role="note">
              <span aria-hidden="true">⚠</span> {unusedShapes.length} not used by anything
            </div>
          )}
          <ShapesTree
            spec={spec}
            graph={graph}
            topShapes={shapeRows}
            has={has}
            selectedKey={selKey(sel)}
            onSelect={selectTarget}
            lockedElements={lockedElements}
            onToggleLocked={onToggleLocked}
            refCount={shapeRefCount}
            edit={edit}
          />
          {namedRegionNames.length === 0 && (
            <div className="muted small">made with the elements above, or + shape</div>
          )}
        </GeoGroup>
        </>
        )}
      </div>

      <div className="geo-inspector">
        {sel != null && (
          <nav className="geo-crumbs" aria-label="Back">
            <button
              className="geo-back"
              onClick={() => {
                setRoutesView(false);
                setTrail([]);
                onSelect(null);
              }}
            >
              ← all elements
            </button>
            {trail.length > 0 && (
              <button className="geo-back" onClick={hopBack}>
                ← back to <strong>{targetLabel(spec, trail[trail.length - 1])}</strong>
              </button>
            )}
          </nav>
        )}

        {sel?.scope === "routes" && (
          <RouteSettings
            spawn={spec.spawn ?? emptySpawn()}
            waypointNames={waypointNames}
            hiddenElements={hiddenElements}
            onToggleHidden={onToggleHidden}
            onHighlightRoute={onHighlightRoute}
            onChange={(spawn) => edit((s) => (s.spawn = spawn))}
          />
        )}

        {sel?.scope === "queryable" && spec.queryables?.[sel.name] && (
          <QueryableInspector
            shapeLink={
              <ShapeLink
                refName={spec.queryables[sel.name].shape?.ref ?? spec.queryables[sel.name].sample?.ref}
                graph={graph}
                onSelect={hop}
                self={{ scope: "queryable", name: sel.name }}
              />
            }
            key={sel.name}
            name={sel.name}
            q={spec.queryables[sel.name]}
            kind={spec.queryables[sel.name].type === "waypoint" ? "waypoint" : "region"}
            hidden={hiddenElements.has(`queryable:${sel.name}`)}
            onToggleHidden={() => onToggleHidden(`queryable:${sel.name}`)}
            locked={lockedElements.has(`queryable:${sel.name}`)}
            onToggleLocked={() => onToggleLocked(`queryable:${sel.name}`)}
            onRename={(n) => renameQueryable(sel.name, n)}
            onDelete={() => {
              onSelect(null);
              edit((s) => delete s.queryables[sel.name]);
            }}
            regionNames={regionNames}
            namedRegions={regions}
            namedRegionNames={refRegionNames}
            resolveRegion={resolveShapeByName}
            onEditRegion={setRegion}
            shapeRefCount={shapeRefCount}
            onNewShape={() =>
              edit((s) => {
                // A waypoint's is a point; a region's, an area.
                const fresh =
                  s.queryables[sel.name].type === "waypoint"
                    ? pointRegion(addLat, addLon)
                    : defaultRegion(addLat, addLon, addRegionAltitude ?? null);
                s.queryables[sel.name].shape = { ref: addRegionTo(s, fresh, sel.name) };
              })
            }
            onChange={(nq) => edit((s) => (s.queryables[sel.name] = nq))}
            onFocus={() => {
              const b = resolveRef(spec.queryables[sel.name].shape);
              if (b) onFocusShape(b);
            }}
          />
        )}

        {sel?.scope === "spawn" && spawnRegions[sel.index] && (
          <>
            <InspectorHead
              kind="spawn"
              name={spawnRegions[sel.index].name ?? ""}
              onRename={(n) =>
                edit((s) => {
                  const old = s.spawn.regions[sel.index];
                  s.spawn.regions[sel.index] = { ...old, name: n };
                  if (n && n !== old?.name) syncShapeName(s, old.shape?.ref, n);
                })
              }
              hidden={hiddenElements.has(`spawn:${sel.index}`)}
              onToggleHidden={() => onToggleHidden(`spawn:${sel.index}`)}
              locked={lockedElements.has(`spawn:${sel.index}`)}
              onToggleLocked={() => onToggleLocked(`spawn:${sel.index}`)}
              onDelete={() => {
                onSelect(null);
                edit((s) => s.spawn.regions.splice(sel.index, 1));
              }}
            />
            <ShapeLink
              refName={spawnRegions[sel.index]?.shape?.ref}
              graph={graph}
              onSelect={hop}
              self={{ scope: "spawn", index: sel.index }}
            />
            <LockableBody locked={lockedElements.has(`spawn:${sel.index}`)}>
              <SpawnBody
                region={spawnRegions[sel.index]}
                globalConflictFree={spec.spawn?.conflict_free_spawn === true}
                globalMaintainMinSepNm={
                  spec.spawn?.maintain_min_sep_nm as number | null | undefined
                }
                routeNames={routeNames}
                waypointNames={waypointNames}
                regionNames={refRegionNames}
                resolveRegion={resolveShapeByName}
                onEditRegion={setRegion}
                shapeRefCount={shapeRefCount}
                onNewRegion={() =>
                  edit((s) => {
                    const r = s.spawn.regions[sel.index];
                    r.shape = {
                      ref: addRegionTo(
                        s,
                        defaultSpawnRegion(addLat, addLon, spawnAltLo, spawnAltHi).shape,
                        r.name || `spawn_${sel.index + 1}`,
                      ),
                    };
                  })
                }
                onChange={(nr) =>
                  edit((s) => {
                    const old = s.spawn.regions[sel.index];
                    s.spawn.regions[sel.index] = nr;
                    if (nr.name && nr.name !== old?.name) syncShapeName(s, nr.shape?.ref, nr.name);
                  })
                }
                onFocus={() => {
                  const b = resolveRef(spawnRegions[sel.index].shape);
                  if (b) onFocusShape(b);
                }}
              />
            </LockableBody>
          </>
        )}

        {sel?.scope === "region" && regions[sel.name] && (
          <>
            <InspectorHead
              kind={`${sel.name === airspaceRef ? "airspace" : pointShapes.has(sel.name) ? "point" : "area"}${shapeRefCount(sel.name) > 1 ? ` ·${shapeRefCount(sel.name)}` : ""}`}
              name={sel.name}
              onRename={(n) => renameRegion(sel.name, n)}
              onDelete={() => deleteShape(sel.name)}
              hidden={sel.name === airspaceRef ? hiddenElements.has("airspace") : undefined}
              onToggleHidden={sel.name === airspaceRef ? () => onToggleHidden("airspace") : undefined}
              locked={lockedElements.has(`region:${sel.name}`)}
              onToggleLocked={() => onToggleLocked(`region:${sel.name}`)}
            />
            {!pointShapes.has(sel.name) && (
              <AirspaceRole
                isAirspace={sel.name === airspaceRef}
                other={airspaceRef && airspaceRef !== sel.name ? airspaceRef : undefined}
                onChange={(on) => setAirspaceRef(on ? sel.name : null)}
              />
            )}
            <ShapeDependencies
              info={graph[sel.name]}
              onSelect={hop}
              onAddQueryable={() => {
                let qname = "";
                edit((s) => {
                  const q = { ...defaultQueryRegion(addLat, addLon, addRegionAltitude), shape: { ref: sel.name } };
                  qname = addQueryable(s, q, sel.name);
                });
                if (qname) selectTarget({ scope: "queryable", name: qname });
              }}
            />
            <GroupPicker
              spec={spec}
              name={sel.name}
              onGroupWith={(other) => edit((s) => groupTogether(s, sel.name, other))}
              onJoin={(id) => edit((s) => moveShape(s, sel.name, id))}
              onOpen={(id) => hop({ scope: "group", id })}
            />
            <LockableBody locked={lockedElements.has(`region:${sel.name}`)}>
              <ShapeEditor
                shape={regions[sel.name]}
                onChange={(b) => setRegion(sel.name, b)}
                onFocus={() => onFocusShape(regions[sel.name])}
                generatorRegionNames={namedRegionNames.filter((n) => n !== sel.name)}
              />
              {partitions[sel.name]?.length > 0 && (
                <div className="partition-actions">
                  <div className="muted small">
                    Shapes {partitions[sel.name].join(", ")} - each a region elements can refer to.
                  </div>
                  <button type="button" onClick={() => addPartitionQueryables(sel.name)}>
                    Add a queryable per shape
                  </button>
                </div>
              )}
            </LockableBody>
          </>
        )}

        {sel?.scope === "group" && groups.find((g) => g.id === sel.id) && (
          <GroupInspector
            spec={spec}
            group={groups.find((g) => g.id === sel.id)!}
            center={[addLat, addLon]}
            regionNames={refRegionNames}
            onRename={(n) => updateGroup(sel.id, { name: n })}
            onUpdate={(patch) => updateGroup(sel.id, patch)}
            onTakeOut={(name) => edit((s) => moveShape(s, name, groups.find((g) => g.id === sel.id)?.parent ?? null))}
            onNest={(parent) => edit((s) => moveGroup(s, sel.id, parent))}
            onUngroup={() => ungroup(sel.id)}
            onSelect={hop}
          />
        )}
      </div>
    </div>
    </PointShapesProvider>
  );
}

// Inspector for a group: what it holds (and takes out), the group it is in,
// its per-episode randomization and its moves during an episode. Selecting it
// on the map shows move + rotate handles that move everything it holds.
function GroupInspector({
  spec,
  group,
  center,
  regionNames,
  onRename,
  onUpdate,
  onTakeOut,
  onNest,
  onUngroup,
  onSelect,
}: {
  spec: SpecDict;
  group: Group;
  center: [number, number];
  regionNames: string[];
  onRename: (n: string) => void;
  onUpdate: (patch: SpecDict) => void;
  onTakeOut: (bound: string) => void;
  onNest: (parent: string | null) => void;
  onUngroup: () => void;
  onSelect: (t: EditTarget) => void;
}) {
  const trans = group.translation ?? {};
  const subs = childGroups(spec, group.id);
  const members = (group.members ?? []).filter((m) => !m.startsWith("wp:"));
  // The waypoints on its points (and, from older designs, anchored to its bounds).
  const held = shapesInGroup(spec, group.id);
  const waypoints = [
    ...Object.entries(spec.queryables ?? {})
      .filter(([, q]: [string, any]) => q?.type === "waypoint" && held.includes(q.shape?.ref))
      .map(([n]) => n),
    ...anchoredWaypoints(spec, held, group.id),
  ];
  // Where it may go: the top, or any group not inside it (no cycles).
  const parents = ((spec.transform?.groups ?? []) as Group[]).filter((o) => !isWithin(spec, o.id, group.id));
  const setTranslation = (patch: SpecDict) => {
    const next = { ...trans, ...patch };
    const empty = next.east_nm == null && next.north_nm == null;
    onUpdate({ translation: empty ? undefined : next });
  };
  // Randomization is opt-in: without it a group only moves what it holds
  // together, while editing (and during an episode, if it has moves).
  const hasRandomization = group.angle_deg != null || group.translation != null || group.scale != null;
  const enableRandomization = () => onUpdate({ angle_deg: { type: "range", low: -30, high: 30 } });
  const disableRandomization = () => onUpdate({ angle_deg: undefined, translation: undefined, scale: undefined });
  const moves: SpecDict[] = group.motion ?? [];
  return (
    <>
      <InspectorHead kind="group" name={group.name} onRename={onRename} onDelete={onUngroup} />
      <div className="deps-panel">
        <div className="deps-row">
          <span className="deps-label">holds</span>
          <span className="chips">
            {subs.map((g) => (
              <DepChip key={g.id} label={g.name || g.id} target={{ scope: "group", id: g.id }} onSelect={onSelect} />
            ))}
            {members.map((m) => (
              <span key={m} className="chip dep with-x" data-kind="bounds">
                <button type="button" className="chip-go" onClick={() => onSelect({ scope: "region", name: m })}>
                  {m}
                </button>
                <button
                  type="button"
                  className="chip-x"
                  title={group.parent ? "take out, into the group around this one" : "take out of the group"}
                  aria-label={`take ${m} out`}
                  onClick={() => onTakeOut(m)}
                >
                  ✕
                </button>
              </span>
            ))}
            {subs.length + members.length === 0 && <span className="muted small">nothing</span>}
          </span>
        </div>
        {waypoints.length > 0 && (
          <div className="deps-row">
            <span className="deps-label">carries</span>
            <span className="chips">
              {waypoints.map((w) => (
                <DepChip key={w} label={w} target={{ scope: "queryable", name: w }} kind="waypoint" onSelect={onSelect} />
              ))}
            </span>
          </div>
        )}
        <div className="deps-row">
          <span className="deps-label">inside</span>
          <Picker
            searchable={false}
            placeholder="— no group (top level)"
            value={group.parent ?? ""}
            onChange={(v) => onNest(v || null)}
            options={[
              { value: "", label: "— no group (top level)" },
              ...parents.map((o) => ({ value: o.id, label: o.name || o.id })),
            ]}
          />
        </div>
        <div className="muted small">
          Drag bounds onto it in the list to add them. Waypoints on its points move with it.
        </div>
      </div>
      <FieldGroup title="Randomize per episode" defaultOpen={hasRandomization}>
        <label className="radio">
          <input
            type="checkbox"
            checked={hasRandomization}
            onChange={(e) => (e.target.checked ? enableRandomization() : disableRandomization())}
          />
          randomize this group each episode
        </label>
        {hasRandomization ? (
          <>
            <div className="muted small">Sampled each episode about the group center. Step the map's episode to see another draw.</div>
            <ValueField label="rotate °" step={5} value={group.angle_deg ?? 0} onChange={(v) => onUpdate({ angle_deg: v })} />
            <ValueField label="east nm" step={1} value={trans.east_nm ?? 0} onChange={(v) => setTranslation({ east_nm: v })} />
            <ValueField label="north nm" step={1} value={trans.north_nm ?? 0} onChange={(v) => setTranslation({ north_nm: v })} />
            <ValueField label="scale ×" step={0.1} value={group.scale ?? 1} onChange={(v) => onUpdate({ scale: v })} />
            {group.parent && <div className="muted small">Inside a group, it is randomized first, then carried by that group.</div>}
          </>
        ) : (
          <div className="muted small">Off - each episode starts with it where it is drawn.</div>
        )}
      </FieldGroup>
      <FieldGroup title="Moves during the episode" defaultOpen={moves.length > 0} hint={moves.length ? `${moves.length}` : undefined}>
        <div className="muted small">
          The group moves as one, rigidly about its center: the shapes it holds keep their distances. Waypoints on its
          points hold still - a fix doesn't move.
        </div>
        <MotionEditor
          motion={moves}
          onChange={(m) => onUpdate({ motion: m ?? undefined, ...(m ? {} : { motion_update: undefined }) })}
          cadence={group.motion_update}
          onCadence={(u) => onUpdate({ motion_update: u ?? undefined })}
          center={center}
          regionNames={regionNames}
        />
      </FieldGroup>
      <div className="geo-inspector-foot">
        <button type="button" onClick={onUngroup} title="Remove the group; what it holds goes to where it was">
          Ungroup
        </button>
      </div>
    </>
  );
}

// The airspace: a role one area plays - the sector. Code reads it as
// `ctx.airspace` and `info["in_airspace"]`; drivers draw it in red; new
// elements are seeded inside it. Content beyond it is a warning, not an error.
function AirspaceRole({
  isAirspace,
  other,
  onChange,
}: {
  isAirspace: boolean;
  other?: string;
  onChange: (on: boolean) => void;
}) {
  return (
    <div className="deps-panel compact">
      <label className="radio" title='the sector: ctx.airspace and info["in_airspace"] ask about it; drivers draw it in red'>
        <input type="checkbox" checked={isAirspace} onChange={(e) => onChange(e.target.checked)} />
        the airspace
        {!isAirspace && other && <span className="muted small"> - instead of “{other}”</span>}
      </label>
    </div>
  );
}

// A bounds' group: the one it is in, to join, or a bounds to group it with -
// the keyboard's way to do what dragging in the list does.
function GroupPicker({
  spec,
  name,
  onGroupWith,
  onJoin,
  onOpen,
}: {
  spec: SpecDict;
  name: string;
  onGroupWith: (other: string) => void;
  onJoin: (id: string | null) => void;
  onOpen: (id: string) => void;
}) {
  const current = groupOfShape(spec, name);
  const groups = (spec.transform?.groups ?? []) as Group[];
  const others = Object.keys(spec.shapes ?? {}).filter((n) => n !== name);
  return (
    <div className="deps-panel compact">
      <div className="deps-row">
        <span className="deps-label">group</span>
        <Picker
          placeholder="— in no group"
          value={current ? `g:${current.id}` : ""}
          onChange={(v) => {
            if (!v) onJoin(null);
            else if (v.startsWith("g:")) onJoin(v.slice(2));
            else if (v.startsWith("b:")) onGroupWith(v.slice(2));
          }}
          options={[
            { value: "", label: "— in no group" },
            ...groups.map((g) => ({ value: `g:${g.id}`, label: g.name || g.id, category: "join a group" })),
            ...others.map((n) => {
              const in_ = groupOfShape(spec, n);
              return {
                value: `b:${n}`,
                label: n,
                category: "group with a shape",
                description: in_ ? `a new group, inside ${in_.name || in_.id}` : "a new group of the two",
              };
            }),
          ]}
        />
        {current && (
          <button type="button" className="link" onClick={() => onOpen(current.id)}>
            open
          </button>
        )}
      </div>
    </div>
  );
}

// A group of outline rows with a header and optional inline "+ add" / action.
function GeoGroup({
  title,
  hint,
  onAdd,
  addLabel,
  action,
  children,
}: {
  title: string;
  /** What this group of geometry is for, shown on hover. */
  hint?: string;
  onAdd?: () => void;
  addLabel?: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="geo-group">
      <div className="geo-group-head">
        <span className="geo-group-title">
          {title}
          <Hint text={hint} />
        </span>
        <span className="spacer" />
        {action}
        {onAdd && (
          <button className="geo-add" title={`add ${addLabel ?? title}`} onClick={onAdd}>
            + {addLabel}
          </button>
        )}
      </div>
      {children}
    </div>
  );
}

// Under an element's header: the bounds it uses (open it), who else uses it,
// and what that bounds depends on - so a change's reach is in view first.
function ShapeLink({
  refName,
  graph,
  onSelect,
  self,
}: {
  refName: string | undefined;
  graph: Record<string, ShapeInfo>;
  onSelect: (t: EditTarget) => void;
  self: EditTarget;
}) {
  if (!refName) return null;
  const name = graph[refName] ? refName : refName.replace(/\.\d+$/, "");
  const info = graph[name];
  if (!info) return null;
  const same = (t: EditTarget | null) => !!t && targetKey(t) === targetKey(self);
  const others = info.usedBy.filter((u) => !same(u.target));
  return (
    <div className="deps-panel compact">
      <div className="deps-row">
        <span className="deps-label">shape</span>
        <span className="chips">
          <DepChip label={refName} target={{ scope: "region", name }} onSelect={onSelect} />
          {refName !== name && <span className="muted small">a shape of {name}, drawn each episode</span>}
        </span>
      </div>
      {others.length > 0 && (
        <div className="deps-row">
          <span className="deps-label">shared with</span>
          <span className="chips">
            {others.map((u, i) => (
              <DepChip key={i} label={u.label} target={u.target} kind={u.kind} onSelect={onSelect} />
            ))}
          </span>
        </div>
      )}
      <DependsOn info={info} onSelect={onSelect} />
    </div>
  );
}

// A chip that goes to an element, colored as its kind is.
function DepChip({
  label,
  target,
  kind: as,
  onSelect,
}: {
  label: string;
  target: EditTarget | null;
  kind?: string;
  onSelect: (t: EditTarget) => void;
}) {
  const kind = as ?? (target ? kindOfTarget(target) : "bounds");
  return target ? (
    <button type="button" className="chip dep" data-kind={kind} onClick={() => onSelect(target)}>
      {label}
    </button>
  ) : (
    <span className="chip dep" data-kind={kind}>
      {label}
    </span>
  );
}

// What a bounds depends on: each bounds once, with every param it is used through.
function DependsOn({ info, onSelect }: { info: ShapeInfo; onSelect: (t: EditTarget) => void }) {
  const merged = new Map<string, string[]>();
  for (const d of info.dependsOn) merged.set(d.name, [...(merged.get(d.name) ?? []), d.via]);
  if (!merged.size) return null;
  return (
    <div className="deps-row">
      <span className="deps-label">depends on</span>
      <span className="chips">
        {[...merged].map(([name, vias]) => (
          <button
            key={name}
            type="button"
            className="chip dep"
            data-kind="bounds"
            title={`through its ${vias.join(", ")}`}
            onClick={() => onSelect({ scope: "region", name: name.replace(/\.\d+$/, "") })}
          >
            {name} <span className="muted">· {vias.join(", ")}</span>
          </button>
        ))}
      </span>
    </div>
  );
}

// The kind an edit target is - for its color and icon.
function kindOfTarget(t: EditTarget): string {
  if (t.scope === "queryable") return "region";
  if (t.scope === "region") return "bounds";
  return t.scope;
}

// How to name a target for a person: "storm", "spawn_1", "airspace".
function targetLabel(spec: SpecDict, t: EditTarget): string {
  if (t.scope === "airspace") return "airspace";
  if (t.scope === "spawn") return spec.spawn?.regions?.[t.index]?.name || `spawn ${t.index + 1}`;
  if (t.scope === "group") return (spec.transform?.groups ?? []).find((g: SpecDict) => g.id === t.id)?.name ?? "group";
  return "name" in t ? t.name : "element";
}

// A bounds' dependencies: what uses it, what it uses, the code that reads it -
// each a way there - and, unused, what it could be.
function ShapeDependencies({
  info,
  onSelect,
  onAddQueryable,
}: {
  info: ShapeInfo | undefined;
  onSelect: (t: EditTarget) => void;
  onAddQueryable: () => void;
}) {
  if (!info) return null;
  const unused = isUnused(info);
  return (
    <div className="deps-panel">
      {unused ? (
        <div className="deps-warning" role="note">
          <strong>
            <span aria-hidden="true">⚠</span> Not used by anything
          </strong>
          <span>
            No element, other shape, group or code refers to it. Pick it as an element's shape, as a generator's parent
            or a placement's region - or read it in code with <code>ctx.shape("{info.name}")</code>.
          </span>
          <span className="deps-actions">

            <button type="button" onClick={onAddQueryable}>
              Add a queryable on it
            </button>
          </span>
        </div>
      ) : (
        <>
          {info.usedBy.length > 0 && (
            <div className="deps-row">
              <span className="deps-label">used by</span>
              <span className="chips">
                {info.usedBy.map((u, i) => (
                  <DepChip key={i} label={u.label} target={u.target} kind={u.kind} onSelect={onSelect} />
                ))}
              </span>
            </div>
          )}
          {info.inCode.length > 0 && (
            <div className="deps-row">
              <span className="deps-label">read in</span>
              <span className="muted small">{info.inCode.join(", ")}</span>
            </div>
          )}
          {info.usedBy.length > 1 && <div className="muted small">Edits here change it for every one of them.</div>}
        </>
      )}
      <DependsOn info={info} onSelect={onSelect} />
    </div>
  );
}

// Each kind of element's icon (24px paths), as its badge shows it.
const KIND_ICONS: Record<string, string> = {
  airspace: "M3 5h18v14H3V5Zm2 2v10h14V7H5Z",
  region: "M12 2 4 6v6c0 5 3.4 8.7 8 10 4.6-1.3 8-5 8-10V6l-8-4Zm0 2.2 6 3V12c0 3.9-2.5 6.9-6 8-3.5-1.1-6-4.1-6-8V7.2l6-3Z",
  waypoint: "M12 2a7 7 0 0 0-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 0 0-7-7Zm0 9.5a2.5 2.5 0 1 1 0-5 2.5 2.5 0 0 1 0 5Z",
  spawn: "M11 2h2v5h-2V2Zm0 15h2v5h-2v-5ZM2 11h5v2H2v-2Zm15 0h5v2h-5v-2Zm-5-2a3 3 0 1 1 0 6 3 3 0 0 1 0-6Z",
  bounds: "M4 4h6v2H6v4H4V4Zm10 0h6v6h-2V6h-4V4ZM4 14h2v4h4v2H4v-6Zm14 0h2v6h-6v-2h4v-4Z",
  area: "M4 4h6v2H6v4H4V4Zm10 0h6v6h-2V6h-4V4ZM4 14h2v4h4v2H4v-6Zm14 0h2v6h-6v-2h4v-4Z",
  point: "M11 2h2v5h-2V2Zm0 15h2v5h-2v-5ZM2 11h5v2H2v-2Zm15 0h5v2h-5v-2Zm-5-2.5a3.5 3.5 0 1 1 0 7 3.5 3.5 0 0 1 0-7Z",
  group: "M3 3h8v8H3V3Zm10 0h8v8h-8V3ZM3 13h8v8H3v-8Zm10 0h8v8h-8v-8Z",
};

// The inspector header: the kind (badge, color), editable name, eye, delete.
function InspectorHead({
  kind,
  name,
  onRename,
  hidden,
  onToggleHidden,
  locked,
  onToggleLocked,
  onDelete,
}: {
  kind: string;
  name?: string;
  onRename?: (n: string) => void;
  hidden?: boolean;
  onToggleHidden?: () => void;
  locked?: boolean;
  onToggleLocked?: () => void;
  onDelete?: () => void;
}) {
  const base = kind.split(/[\s·]/)[0];
  return (
    <div className="geo-inspector-head" data-kind={base}>
      <span className="kind-badge" data-kind={base}>
        <svg viewBox="0 0 24 24" width="13" height="13" aria-hidden="true">
          <path fill="currentColor" d={KIND_ICONS[base] ?? KIND_ICONS.bounds} />
        </svg>
        {kind}
      </span>
      {onRename ? (
        <input className="name-input" defaultValue={name} key={name} onBlur={(e) => onRename(e.target.value)} />
      ) : (
        <span className="geo-inspector-name">{name ?? kind}</span>
      )}
      <span className="spacer" />
      {onToggleLocked && <LockToggle locked={locked ?? false} onToggle={onToggleLocked} />}
      {onToggleHidden && <EyeToggle hidden={hidden ?? false} onToggle={onToggleHidden} />}
      {onDelete && (
        <button className="link danger" title="delete" onClick={onDelete}>
          ✕
        </button>
      )}
    </div>
  );
}

// Disables a selected element's editors when it's locked, so its params can't be
// changed from the panel (a disabled fieldset + pointer-events guard covers both
// native inputs and the custom div-based controls). The header stays enabled so
// the element can still be unlocked.
function LockableBody({ locked, children }: { locked: boolean; children: React.ReactNode }) {
  if (!locked) return <>{children}</>;
  return (
    <fieldset className="geo-locked" disabled>
      {children}
    </fieldset>
  );
}

// Wraps QueryableBody with the inspector header (the body has no card chrome).
function QueryableInspector({
  kind,
  hidden,
  onToggleHidden,
  locked,
  onToggleLocked,
  onRename,
  onDelete,
  shapeLink,
  ...body
}: Parameters<typeof QueryableBody>[0] & {
  shapeLink?: React.ReactNode;
  kind: string;
  hidden: boolean;
  onToggleHidden: () => void;
  locked: boolean;
  onToggleLocked: () => void;
  onRename: (n: string) => void;
  onDelete: () => void;
}) {
  return (
    <>
      <InspectorHead
        kind={kind}
        name={body.name}
        onRename={onRename}
        hidden={hidden}
        onToggleHidden={onToggleHidden}
        locked={locked}
        onToggleLocked={onToggleLocked}
        onDelete={onDelete}
      />
      {shapeLink}
      <LockableBody locked={locked}>
        <QueryableBody {...body} />
      </LockableBody>
    </>
  );
}

// --- spec edit helpers (named bounds bookkeeping) ------------------------

// Add a named bounds (unique name, derived from its role) and return its name.
function addRegionTo(s: SpecDict, bounds: SpecDict, base = "bounds"): string {
  s.shapes = s.shapes ?? {};
  const slug = (base || "bounds").replace(/[^0-9a-zA-Z_]+/g, "_") || "bounds";
  let name = slug;
  let i = 1;
  while (s.shapes[name]) name = `${slug}_${i++}`;
  s.shapes[name] = bounds;
  return name;
}

// When a bounds is referenced by exactly one element, keep its name following
// that element's name. Renames the bounds (deduped) and rewrites the ref.
function syncShapeName(s: SpecDict, name: string | undefined, desired: string) {
  if (!name || !s.shapes?.[name] || countShapeRefs(s, name) !== 1) return;
  const slug = (desired || "bounds").replace(/[^0-9a-zA-Z_]+/g, "_") || "bounds";
  if (name === slug) return;
  let target = slug;
  let i = 1;
  while (s.shapes[target]) target = `${slug}_${i++}`;
  s.shapes[target] = s.shapes[name];
  delete s.shapes[name];
  rewriteRegionRefs(s, name, target);
}

// Rewrite every {"ref": oldName} bounds reference after a region is renamed.
// Every ref to a renamed bounds follows it - elements, other bounds' layers,
// groups, its shapes' refs (see shapeGraph).
function rewriteRegionRefs(s: SpecDict, oldName: string, newName: string) {
  renameShapeRefs(s, oldName, newName);
}

function addQueryable(s: SpecDict, q: SpecDict, prefix: string): string {
  s.queryables = s.queryables ?? {};
  let n = 1;
  let name = prefix;
  while (s.queryables[name]) name = `${prefix}_${n++}`;
  s.queryables[name] = q;
  return name;
}
