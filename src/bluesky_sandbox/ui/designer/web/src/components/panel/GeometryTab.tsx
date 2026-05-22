// The Geometry tab, rebuilt as a master–detail view:
//   - an OUTLINE (compact, one row per element, grouped by kind) for navigation
//     and add/visibility, kept in sync with the map selection; and
//   - an INSPECTOR that edits only the currently-selected element.
// The spec object is the source of truth; every edit yields a new spec via
// onChange (which App also re-serialises into the code editor).
import { Hint } from "./Hint";
import { useEffect, useState } from "react";
import type { SpecDict } from "../../api";
import BoundsEditor from "../BoundsEditor";
import {
  boundsRefLabels,
  clippedSpawnAltitudeRange,
  clone,
  countBoundsRefs,
  defaultConstantBand,
  defaultQueryRegion,
  defaultRegion,
  defaultSpawnRegion,
  defaultWaypoint,
  designBounds,
  gcOrphanBounds,
  newGroupId,
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
  onFocusBounds,
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
  onFocusBounds: (bounds: SpecDict) => void;
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
  const [showAllBounds, setShowAllBounds] = useState(false);
  // Panel-only "routes" selection; a map selection (selectedKey) supersedes it.
  const [routesView, setRoutesView] = useState(false);
  useEffect(() => {
    if (selectedKey) {
      setRoutesView(false);
    }
  }, [selectedKey]);

  const has = (n: string) => !filter || n.toLowerCase().includes(filter.toLowerCase());

  const edit = (mut: (s: SpecDict) => void) => {
    const next = clone(spec);
    mut(next);
    // Bounds are only created via elements; sweep any left unreferenced after a
    // delete or reassignment so orphans never accumulate.
    gcOrphanBounds(next);
    onChange(next);
  };

  // ---- selection ----------------------------------------------------------
  const sel: Sel = routesView ? { scope: "routes" } : parseTargetKey(selectedKey);
  const selectTarget = (t: EditTarget) => {
    setRoutesView(false);
    onSelect(t);
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
  const regions: Record<string, SpecDict> = spec.regions ?? {};
  const namedRegionNames = Object.keys(regions);
  const resolveRef = (b: any): SpecDict | undefined => (b?.ref ? regions[b.ref] : b);
  const resolveRegionBounds = (name: string): SpecDict | undefined => regions[name];
  const boundsRefCount = (name: string): number => countBoundsRefs(spec, name);
  const setRegion = (name: string, b: SpecDict) => edit((s) => (s.regions[name] = b));
  const renameRegion = (oldName: string, newNameRaw: string) => {
    const newName = newNameRaw.trim();
    if (!newName || newName === oldName || spec.regions?.[newName]) return;
    edit((s) => {
      s.regions[newName] = s.regions[oldName];
      delete s.regions[oldName];
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
  // Fixed lat/lon waypoints can join a group directly (navdb fixes and
  // sampled-from-bounds waypoints are excluded: the former are anchored to a fix,
  // the latter already group via their sample bounds).
  const groupableWaypoints = waypointEntries
    .filter(([, q]) => q.waypoint == null && q.sample == null)
    .map(([name]) => name);
  // Query-region names (for the waypoint TSAS-bound picker).
  const regionNames = regionEntries.filter(([, q]) => q.type === "query_region").map(([name]) => name);

  const renameQueryable = (name: string, newNameRaw: string) => {
    const newName = newNameRaw.trim();
    if (!newName || newName === name || spec.queryables?.[newName]) return;
    edit((s) => {
      s.queryables[newName] = s.queryables[name];
      delete s.queryables[name];
      syncBoundsName(s, s.queryables[newName].bounds?.ref, newName);
      syncBoundsName(s, s.queryables[newName].sample?.ref, `${newName}_sample`);
    });
    // Keep the renamed queryable selected (its key changed).
    onSelect({ scope: "queryable", name: newName });
  };

  const spawnRegions: SpecDict[] = spec.spawn?.regions ?? [];
  const routeNames = Object.keys(spec.spawn?.routes ?? {});

  // ---- transform groups ---------------------------------------------------
  // A group is a named set of bounds moved/rotated together (and randomised per
  // episode). Members are bounds names; a bounds belongs to at most one group.
  const groups: SpecDict[] = spec.transform?.groups ?? [];
  const boundsList = designBounds(spec);
  const groupOwnerOf = (name: string) => groups.find((g) => (g.members ?? []).includes(name));
  const setGroups = (next: SpecDict[]) =>
    edit((s) => {
      const t = { ...(s.transform ?? {}) };
      if (next.length) t.groups = next;
      else delete t.groups;
      s.transform = Object.keys(t).length ? t : null;
    });
  const addGroup = () => {
    const id = newGroupId();
    // A plain organizational group (move/rotate members together while editing);
    // per-episode randomization is opt-in from the inspector.
    setGroups([...groups, { id, name: `group ${groups.length + 1}`, members: [], parent: null, pivot: null }]);
    selectTarget({ scope: "group", id });
  };
  const updateGroup = (id: string, patch: SpecDict) => setGroups(groups.map((g) => (g.id === id ? { ...g, ...patch } : g)));
  const removeGroup = (id: string) => {
    onSelect(null);
    setGroups(groups.filter((g) => g.id !== id).map((g) => (g.parent === id ? { ...g, parent: null } : g)));
  };
  // Membership is exclusive: assigning a bounds removes it from any other group.
  const toggleMember = (id: string, name: string) => {
    const had = (groups.find((g) => g.id === id)?.members ?? []).includes(name);
    setGroups(
      groups.map((g) => {
        const members = (g.members ?? []).filter((m: string) => m !== name);
        if (g.id === id && !had) members.push(name);
        return { ...g, members };
      }),
    );
  };

  // ---- adders (each selects the new element so the inspector opens) --------
  const addRegion = () => {
    let qname = "";
    edit((s) => {
      const qr = defaultQueryRegion(addLat, addLon, addRegionAltitude);
      const inline = qr.bounds;
      qname = addQueryable(s, qr, "region");
      s.queryables[qname].bounds = { ref: addRegionTo(s, inline, qname) };
    });
    if (qname) selectTarget({ scope: "queryable", name: qname });
  };
  const addWaypoint = () => {
    let qname = "";
    edit((s) => {
      qname = addQueryable(s, defaultWaypoint(addLat, addLon, undefined, addWaypointAlt), "waypoint");
    });
    if (qname) selectTarget({ scope: "queryable", name: qname });
  };
  const addSpawn = () => {
    const index = (spec.spawn?.regions ?? []).length;
    edit((s) => {
      s.spawn = s.spawn ?? emptySpawn();
      s.spawn.regions = s.spawn.regions ?? [];
      const sr = defaultSpawnRegion(addLat, addLon, spawnAltLo, spawnAltHi);
      sr.bounds = { ref: addRegionTo(s, sr.bounds, sr.name || "spawn") };
      s.spawn.regions.push(sr);
    });
    selectTarget({ scope: "spawn", index });
  };
  const setAirspace = () => {
    edit((s) => {
      const r = addRegionTo(s, defaultRegion(addLat, addLon, addRegionAltitude ?? null), "airspace");
      s.airspace = { ref: r };
    });
    selectTarget({ scope: "airspace" });
  };

  // Shared bounds shown in the outline: those referenced by >1 element (or all,
  // when the user opts in). Single-use bounds are edited inline on their element.
  const boundsRows = namedRegionNames.filter((n) => showAllBounds || boundsRefCount(n) > 1);

  return (
    <div className="geo-tab">
      <div className="geo-outline">
        <input
          className="sec-search geo-search"
          placeholder="filter elements…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />

        <GeoGroup title="Airspace" hint="The outer boundary of the simulated sector. Aircraft leaving it end their episode; every other shape here is expected to sit inside it." onAdd={spec.airspace ? undefined : setAirspace} addLabel="set">
          {spec.airspace ? (
            <Row
              name="airspace"
              kind="airspace"
              selected={selKey(sel) === "airspace"}
              onClick={() => selectTarget({ scope: "airspace" })}
              hidden={hiddenElements.has("airspace")}
              onToggleHidden={() => onToggleHidden("airspace")}
              locked={lockedElements.has("airspace")}
              onToggleLocked={() => onToggleLocked("airspace")}
            />
          ) : (
            <div className="muted small">no airspace</div>
          )}
        </GeoGroup>

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

        <GeoGroup title="Spawn" hint="Where aircraft appear and how many. Each spawn region draws a count per episode and samples positions, types, and entry states inside its bounds." onAdd={addSpawn} addLabel="spawn">
          <label className="checkbox-row" style={{ padding: "2px 8px" }}>
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
            <div style={{ padding: "0 8px 4px 24px" }}>
              <div className="value-field">
                <div className="vf-head">
                  <span className="vf-label">buffer horiz nm</span>
                  <span className="vf-spacer" />
                  <NumInput
                    className="vf-input"
                    step={0.5}
                    value={spec.spawn?.conflict_free_margin_nm ?? 0}
                    onChange={(n) =>
                      edit((s) => {
                        s.spawn = s.spawn ?? emptySpawn();
                        s.spawn.conflict_free_margin_nm = n;
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
                    step={100}
                    value={spec.spawn?.conflict_free_margin_ft ?? 0}
                    onChange={(n) =>
                      edit((s) => {
                        s.spawn = s.spawn ?? emptySpawn();
                        s.spawn.conflict_free_margin_ft = n;
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
                    step={30}
                    value={spec.spawn?.conflict_free_margin_s ?? 0}
                    onChange={(n) =>
                      edit((s) => {
                        s.spawn = s.spawn ?? emptySpawn();
                        s.spawn.conflict_free_margin_s = n;
                      })
                    }
                  />
                </div>
              </div>
              <div className="muted small">
                reject spawns whose predicted CPA comes within PZ + buffer (space)
                or lookahead + buffer (time), so clearance holds as aircraft
                maneuver ↑
              </div>
            </div>
          )}
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

        <GeoGroup title="Groups" hint="Per-episode randomisation. Add the bounds a group covers, then set how far it may rotate, shift or scale each episode - the whole group moves together, preserving the geometry between its members." onAdd={addGroup} addLabel="group">
          {groups.filter((g) => has(g.name || g.id)).map((g) => (
            <Row
              key={g.id}
              name={`${g.name || g.id}${(g.members?.length ?? 0) ? ` · ${g.members.length}` : ""}`}
              kind="group"
              selected={selKey(sel) === `group:${g.id}`}
              onClick={() => selectTarget({ scope: "group", id: g.id })}
            />
          ))}
          {groups.length === 0 && <div className="muted small">no groups</div>}
        </GeoGroup>

        <GeoGroup
          title="Bounds"
          action={
            namedRegionNames.length > 0 ? (
              <button className="link" onClick={() => setShowAllBounds((v) => !v)}>
                {showAllBounds ? "shared only" : "show all"}
              </button>
            ) : undefined
          }
        >
          {boundsRows.filter((n) => has(n)).map((name) => (
            <Row
              key={name}
              name={name}
              kind={boundsRefCount(name) > 1 ? `bounds ·${boundsRefCount(name)}` : "bounds"}
              selected={selKey(sel) === `region:${name}`}
              onClick={() => selectTarget({ scope: "region", name })}
              locked={lockedElements.has(`region:${name}`)}
              onToggleLocked={() => onToggleLocked(`region:${name}`)}
            />
          ))}
          {boundsRows.length === 0 && (
            <div className="muted small">
              {namedRegionNames.length ? "no shared bounds" : "created with the elements above"}
            </div>
          )}
        </GeoGroup>
      </div>

      <div className="geo-inspector">
        {sel == null && (
          <div className="geo-inspector-empty muted small">
            Select an element on the map or in the list to edit it.
          </div>
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

        {sel?.scope === "airspace" && spec.airspace && (
          <>
            <InspectorHead
              kind="airspace"
              onDelete={() => {
                onSelect(null);
                edit((s) => (s.airspace = null));
              }}
              hidden={hiddenElements.has("airspace")}
              onToggleHidden={() => onToggleHidden("airspace")}
              locked={lockedElements.has("airspace")}
              onToggleLocked={() => onToggleLocked("airspace")}
            />
            <LockableBody locked={lockedElements.has("airspace")}>
              <BoundsEditor
                bounds={spec.airspace}
                onChange={(b) => edit((s) => (s.airspace = b))}
                onFocus={() => {
                  const b = resolveRef(spec.airspace);
                  if (b) onFocusBounds(b);
                }}
                regionNames={namedRegionNames}
                requireRef
                resolveRegion={resolveRegionBounds}
                onEditRegion={setRegion}
                refCount={spec.airspace?.ref ? boundsRefCount(spec.airspace.ref) : undefined}
                onNewRegion={() =>
                  edit((s) => {
                    const r = addRegionTo(s, defaultRegion(addLat, addLon, addRegionAltitude ?? null), "airspace");
                    s.airspace = { ref: r };
                  })
                }
              />
            </LockableBody>
          </>
        )}

        {sel?.scope === "queryable" && spec.queryables?.[sel.name] && (
          <QueryableInspector
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
            namedRegionNames={namedRegionNames}
            resolveRegion={resolveRegionBounds}
            onEditRegion={setRegion}
            boundsRefCount={boundsRefCount}
            onNewBoundsRegion={() =>
              edit((s) => {
                s.queryables[sel.name].bounds = {
                  ref: addRegionTo(s, defaultRegion(addLat, addLon, addRegionAltitude ?? null), sel.name),
                };
              })
            }
            onNewSampleRegion={() =>
              edit((s) => {
                s.queryables[sel.name].sample = {
                  ref: addRegionTo(s, defaultRegion(addLat, addLon, addRegionAltitude ?? null), `${sel.name}_sample`),
                };
              })
            }
            onChange={(nq) => edit((s) => (s.queryables[sel.name] = nq))}
            onFocus={() => {
              const b = resolveRef(spec.queryables[sel.name].bounds);
              if (b) onFocusBounds(b);
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
                  if (n && n !== old?.name) syncBoundsName(s, old.bounds?.ref, n);
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
            <LockableBody locked={lockedElements.has(`spawn:${sel.index}`)}>
              <SpawnBody
                region={spawnRegions[sel.index]}
                routeNames={routeNames}
                waypointNames={waypointNames}
                regionNames={namedRegionNames}
                resolveRegion={resolveRegionBounds}
                onEditRegion={setRegion}
                boundsRefCount={boundsRefCount}
                onNewRegion={() =>
                  edit((s) => {
                    const r = s.spawn.regions[sel.index];
                    r.bounds = {
                      ref: addRegionTo(
                        s,
                        defaultSpawnRegion(addLat, addLon, spawnAltLo, spawnAltHi).bounds,
                        r.name || `spawn_${sel.index + 1}`,
                      ),
                    };
                  })
                }
                onChange={(nr) =>
                  edit((s) => {
                    const old = s.spawn.regions[sel.index];
                    s.spawn.regions[sel.index] = nr;
                    if (nr.name && nr.name !== old?.name) syncBoundsName(s, nr.bounds?.ref, nr.name);
                  })
                }
                onFocus={() => {
                  const b = resolveRef(spawnRegions[sel.index].bounds);
                  if (b) onFocusBounds(b);
                }}
              />
            </LockableBody>
          </>
        )}

        {sel?.scope === "region" && regions[sel.name] && (
          <>
            <InspectorHead
              kind={boundsRefCount(sel.name) > 1 ? `bounds ·${boundsRefCount(sel.name)}` : "bounds"}
              name={sel.name}
              onRename={(n) => renameRegion(sel.name, n)}
              locked={lockedElements.has(`region:${sel.name}`)}
              onToggleLocked={() => onToggleLocked(`region:${sel.name}`)}
            />
            {boundsRefCount(sel.name) > 1 && (
              <div className="muted small">
                shared by {boundsRefCount(sel.name)}: {boundsRefLabels(spec, sel.name).join(", ")} — edits affect all.
              </div>
            )}
            <LockableBody locked={lockedElements.has(`region:${sel.name}`)}>
              <BoundsEditor
                bounds={regions[sel.name]}
                onChange={(b) => setRegion(sel.name, b)}
                onFocus={() => onFocusBounds(regions[sel.name])}
              />
            </LockableBody>
          </>
        )}

        {sel?.scope === "group" && groups.find((g) => g.id === sel.id) && (
          <GroupInspector
            group={groups.find((g) => g.id === sel.id)!}
            groups={groups}
            boundsList={boundsList}
            waypointList={groupableWaypoints}
            ownerOf={groupOwnerOf}
            onRename={(n) => updateGroup(sel.id, { name: n })}
            onUpdate={(patch) => updateGroup(sel.id, patch)}
            onToggleMember={(name) => toggleMember(sel.id, name)}
            onRemove={() => removeGroup(sel.id)}
          />
        )}
      </div>
    </div>
  );
}

// Inspector for a transform group: members (bounds chips), nesting parent, and
// the per-episode rotation / translation / scale ranges. Selecting it on the map
// shows move + rotate handles that statically transform the members.
function GroupInspector({
  group,
  groups,
  boundsList,
  waypointList,
  ownerOf,
  onRename,
  onUpdate,
  onToggleMember,
  onRemove,
}: {
  group: SpecDict;
  groups: SpecDict[];
  boundsList: string[];
  waypointList: string[];
  ownerOf: (id: string) => SpecDict | undefined;
  onRename: (n: string) => void;
  onUpdate: (patch: SpecDict) => void;
  onToggleMember: (id: string) => void;
  onRemove: () => void;
}) {
  const trans = group.translation ?? {};
  const members: string[] = group.members ?? [];
  // A member id is a bounds name or ``wp:<name>`` for a waypoint.
  const memberLabel = (id: string) => (id.startsWith("wp:") ? id.slice(3) : id);
  const isWp = (id: string) => id.startsWith("wp:");
  // Valid parents: any other group that isn't a descendant of this one (no cycles).
  const descendants = (id: string): Set<string> => {
    const out = new Set<string>();
    const walk = (pid: string) => {
      for (const g of groups) if (g.parent === pid && !out.has(g.id)) { out.add(g.id); walk(g.id); }
    };
    walk(id);
    return out;
  };
  const banned = descendants(group.id);
  const setTranslation = (patch: SpecDict) => {
    const next = { ...trans, ...patch };
    const empty = next.east_nm == null && next.north_nm == null;
    onUpdate({ translation: empty ? undefined : next });
  };
  // Randomization is opt-in: a group with no transform fields is purely
  // organizational (its members still drag/rotate together on the map).
  const hasRandomization = group.angle_deg != null || group.translation != null || group.scale != null;
  const enableRandomization = () => onUpdate({ angle_deg: { type: "range", low: -30, high: 30 } });
  const disableRandomization = () => onUpdate({ angle_deg: undefined, translation: undefined, scale: undefined, parent: null });
  // Candidates for the "add member" picker: every bounds + groupable waypoint not
  // already in this group; ones held by another group note their owner (picking
  // moves them, since membership is exclusive).
  const candidates = [
    ...boundsList.map((n) => ({ value: n, label: n, category: "bounds" })),
    ...waypointList.map((n) => ({ value: `wp:${n}`, label: n, category: "waypoints" })),
  ]
    .filter((c) => !members.includes(c.value))
    .map((c) => {
      const owner = ownerOf(c.value);
      return owner && owner.id !== group.id ? { ...c, description: `in "${owner.name || owner.id}"` } : c;
    });
  return (
    <>
      <InspectorHead kind="group" name={group.name} onRename={onRename} onDelete={onRemove} />
      <FieldGroup title="Members" defaultOpen hint={`${members.length}`}>
        <div className="muted small">Bounds and waypoints moved/rotated together. Each joins at most one group.</div>
        <div className="chips">
          {members.length === 0 && <span className="muted small">no members</span>}
          {members.map((id) => (
            <span key={id} className="chip member on" title={id}>
              {memberLabel(id)}
              {isWp(id) ? <span className="muted"> ·wp</span> : null}
              <button className="chip-x" title="remove" onClick={() => onToggleMember(id)}>✕</button>
            </span>
          ))}
        </div>
        <Picker
          placeholder="+ add member…"
          onChange={(v) => v && onToggleMember(v)}
          options={candidates}
        />
      </FieldGroup>
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
            <div className="muted small">Sampled each episode about the group centre. Reseed on the map to preview.</div>
            <ValueField label="rotate °" step={5} value={group.angle_deg ?? 0} onChange={(v) => onUpdate({ angle_deg: v })} />
            <ValueField label="east nm" step={1} value={trans.east_nm ?? 0} onChange={(v) => setTranslation({ east_nm: v })} />
            <ValueField label="north nm" step={1} value={trans.north_nm ?? 0} onChange={(v) => setTranslation({ north_nm: v })} />
            <ValueField label="scale ×" step={0.1} value={group.scale ?? 1} onChange={(v) => onUpdate({ scale: v })} />
            <label className="numfield inline">
              <span>inside</span>
              <Picker
                searchable={false}
                placeholder="— none (top level)"
                value={group.parent ?? ""}
                onChange={(v) => onUpdate({ parent: v || null })}
                options={[
                  { value: "", label: "— none (top level)" },
                  ...groups
                    .filter((o) => o.id !== group.id && !banned.has(o.id))
                    .map((o) => ({ value: o.id, label: o.name || o.id })),
                ]}
              />
            </label>
            <div className="muted small">A nested group spins locally first, then is carried by its parent.</div>
          </>
        ) : (
          <div className="muted small">Off — this group only moves/rotates its members together while editing.</div>
        )}
      </FieldGroup>
    </>
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

// One compact outline row: kind tag + name, selection highlight, optional eye.
function Row({
  name,
  kind,
  selected,
  onClick,
  hidden,
  onToggleHidden,
  locked,
  onToggleLocked,
}: {
  name: string;
  kind: string;
  selected: boolean;
  onClick: () => void;
  hidden?: boolean;
  onToggleHidden?: () => void;
  locked?: boolean;
  onToggleLocked?: () => void;
}) {
  return (
    <div className={selected ? "geo-row selected" : "geo-row"} onClick={onClick}>
      <span className="geo-row-kind">{kind}</span>
      <span className="geo-row-name" title={name}>{name}</span>
      <span className="spacer" />
      {onToggleLocked && <LockToggle locked={locked ?? false} onToggle={onToggleLocked} />}
      {onToggleHidden && <EyeToggle hidden={hidden ?? false} onToggle={onToggleHidden} />}
    </div>
  );
}

// The inspector header: editable name (when applicable), kind tag, eye, delete.
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
  return (
    <div className="geo-inspector-head">
      {onRename ? (
        <input className="name-input" defaultValue={name} key={name} onBlur={(e) => onRename(e.target.value)} />
      ) : (
        <span className="geo-inspector-name">{name ?? kind}</span>
      )}
      <span className="kind-tag">{kind}</span>
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
  ...body
}: Parameters<typeof QueryableBody>[0] & {
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
      <LockableBody locked={locked}>
        <QueryableBody {...body} />
      </LockableBody>
    </>
  );
}

// --- spec edit helpers (named bounds bookkeeping) ------------------------

// Add a named bounds (unique name, derived from its role) and return its name.
function addRegionTo(s: SpecDict, bounds: SpecDict, base = "bounds"): string {
  s.regions = s.regions ?? {};
  const slug = (base || "bounds").replace(/[^0-9a-zA-Z_]+/g, "_") || "bounds";
  let name = slug;
  let i = 1;
  while (s.regions[name]) name = `${slug}_${i++}`;
  s.regions[name] = bounds;
  return name;
}

// When a bounds is referenced by exactly one element, keep its name following
// that element's name. Renames the bounds (deduped) and rewrites the ref.
function syncBoundsName(s: SpecDict, name: string | undefined, desired: string) {
  if (!name || !s.regions?.[name] || countBoundsRefs(s, name) !== 1) return;
  const slug = (desired || "bounds").replace(/[^0-9a-zA-Z_]+/g, "_") || "bounds";
  if (name === slug) return;
  let target = slug;
  let i = 1;
  while (s.regions[target]) target = `${slug}_${i++}`;
  s.regions[target] = s.regions[name];
  delete s.regions[name];
  rewriteRegionRefs(s, name, target);
}

// Rewrite every {"ref": oldName} bounds reference after a region is renamed.
function rewriteRegionRefs(s: SpecDict, oldName: string, newName: string) {
  const fix = (b: any) => (b && b.ref === oldName ? { ref: newName } : b);
  if (s.airspace) s.airspace = fix(s.airspace);
  for (const q of Object.values(s.queryables ?? {}) as SpecDict[]) {
    if (q.bounds) q.bounds = fix(q.bounds);
    if (q.sample) q.sample = fix(q.sample);
  }
  for (const r of (s.spawn?.regions ?? []) as SpecDict[]) {
    if (r.bounds) r.bounds = fix(r.bounds);
  }
}

function addQueryable(s: SpecDict, q: SpecDict, prefix: string): string {
  s.queryables = s.queryables ?? {};
  let n = 1;
  let name = prefix;
  while (s.queryables[name]) name = `${prefix}_${n++}`;
  s.queryables[name] = q;
  return name;
}
