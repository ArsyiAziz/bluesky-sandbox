// The Route tab: a guided node-graph editor for a route (top) over a read-only
// map preview (bottom). The graph is decompiled from the route's nested
// step-list; edits go through pure step-list transforms (routeGraph ops), so the
// route stays valid. Drag is for pan/zoom only — structure is edited via the
// inspector and by clicking waypoints on the map.
import { useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  type Node,
  type Edge,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { SpecDict } from "../api";
import { Picker } from "../components/panel/Picker";
import RouteMap from "./RouteMap";
import {
  type Addr,
  type RouteStep,
  addrKey,
  appendAfter,
  appendToEnd,
  addOption,
  branchAfter,
  deleteAt,
  removeOption,
  routeToGraph,
  setOptionWeight,
} from "./routeGraph";

const emptySpawn = (): SpecDict => ({ type: "spawn_config", regions: [], aircraft_type: null, route: null, routes: {} });

type NodeData = {
  label: string;
  kind: string;
  selected: boolean;
  onDelete?: () => void;
};

function StepNode({ data }: NodeProps<Node<NodeData>>) {
  return (
    <div className={`rf-node rf-${data.kind} ${data.selected ? "sel" : ""}`}>
      <Handle type="target" position={Position.Left} className="rf-handle" />
      <span className="rf-node-label">{data.label}</span>
      {data.onDelete && (
        <button className="rf-node-x" title="remove" onClick={(e) => { e.stopPropagation(); data.onDelete!(); }}>✕</button>
      )}
      <Handle type="source" position={Position.Right} className="rf-handle" />
    </div>
  );
}

const NODE_TYPES = { step: StepNode };

export default function RouteTab({
  spec,
  onSpecChange,
}: {
  spec: SpecDict | null;
  onSpecChange: (next: SpecDict) => void;
}) {
  const routes: Record<string, RouteStep[]> = spec?.spawn?.routes ?? {};
  const routeNames = Object.keys(routes);
  const [active, setActive] = useState<string | null>(routeNames[0] ?? null);
  const [selId, setSelId] = useState<string | null>(null);
  const current = active && routes[active] ? active : routeNames[0] ?? null;
  const steps = current ? routes[current] ?? [] : [];

  const waypointNames = useMemo(
    () =>
      Object.entries(spec?.queryables ?? {})
        .filter(([, q]) => (q as SpecDict).type === "waypoint")
        .map(([name]) => name),
    [spec],
  );

  // Which spawn region this route is viewed as originating from. A named route
  // can feed several spawn regions; this just sets the preview's origin. Default
  // to a region already assigned this route, else the first.
  const spawnRegions: SpecDict[] = spec?.spawn?.regions ?? [];
  const spawnNames = spawnRegions.map((r, i) => r.name || `spawn_${i + 1}`);
  const [spawnSel, setSpawnSel] = useState<string | null>(null);
  const defaultSpawn = (() => {
    const idx = spawnRegions.findIndex((r) => r.route === current);
    return idx >= 0 ? spawnNames[idx] : spawnNames[0] ?? null;
  })();
  const spawnName = spawnSel && spawnNames.includes(spawnSel) ? spawnSel : defaultSpawn;
  const spawnIdx = spawnName ? spawnNames.indexOf(spawnName) : -1;

  const writeRoutes = (next: Record<string, RouteStep[]>) => {
    if (!spec) return;
    const nextSpec = structuredClone(spec);
    nextSpec.spawn = nextSpec.spawn ?? emptySpawn();
    nextSpec.spawn.routes = next;
    onSpecChange(nextSpec);
  };
  const setSteps = (next: RouteStep[]) => {
    if (!current) return;
    writeRoutes({ ...routes, [current]: next });
  };

  const addRoute = () => {
    let i = routeNames.length + 1;
    let name = `route_${i}`;
    while (routes[name]) name = `route_${++i}`;
    writeRoutes({ ...routes, [name]: [] });
    setActive(name);
    setSelId(null);
  };
  const renameRoute = (oldName: string, raw: string) => {
    const name = raw.trim();
    if (!name || name === oldName || routes[name]) return;
    const next: Record<string, RouteStep[]> = {};
    for (const [k, v] of Object.entries(routes)) next[k === oldName ? name : k] = v;
    writeRoutes(next);
    setActive(name);
  };
  const deleteRoute = (name: string) => {
    const next = { ...routes };
    delete next[name];
    writeRoutes(next);
    setActive(Object.keys(next)[0] ?? null);
    setSelId(null);
  };

  // Decompile the active route into a laid-out graph.
  const graph = useMemo(() => routeToGraph(steps), [steps]);
  const selNode = graph.nodes.find((n) => n.id === selId) ?? null;
  const selAddr: Addr | null = selNode?.addr ?? null;

  // Waypoint names appearing in this route (for the map's sampled-waypoint links).
  const routeWaypoints = graph.nodes.filter((n) => n.kind === "waypoint").map((n) => n.label);

  const rfNodes: Node<NodeData>[] = graph.nodes.map((n) => ({
    id: n.id,
    type: "step",
    position: { x: n.x, y: n.y },
    data: {
      label: n.kind === "start" ? spawnName ?? "spawn" : n.label || (n.kind === "merge" ? "●" : ""),
      kind: n.kind,
      selected: n.id === selId,
      onDelete: n.addr && (n.kind === "waypoint" || n.kind === "subroute") ? () => setSteps(deleteAt(steps, n.addr!)) : undefined,
    },
    draggable: false,
    selectable: n.kind !== "merge",
  }));
  const rfEdges: Edge[] = graph.edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    label: e.label,
    animated: false,
  }));

  // A step to add, chosen from the waypoint / subroute picker.
  const stepFromValue = (v: string): RouteStep => (v.startsWith("rt:") ? { route: v.slice(3) } : v);
  const addOptions = [
    ...waypointNames.map((n) => ({ value: n, label: n, category: "waypoints" })),
    ...routeNames.filter((n) => n !== current).map((n) => ({ value: `rt:${n}`, label: `⤷ ${n}`, category: "subroutes" })),
  ];

  const appendStep = (v: string) => {
    const step = stepFromValue(v);
    if (selAddr && (selNode?.kind === "waypoint" || selNode?.kind === "subroute")) setSteps(appendAfter(steps, selAddr, step));
    else setSteps(appendToEnd(steps, step));
  };

  // Clicking a waypoint on the map appends it after the selection (or at the end).
  const onPickWaypoint = (name: string) => {
    if (!current) return;
    appendStep(name);
  };

  return (
    <div className="route-tab">
      <div className="route-top">
        <div className="route-toolbar">
          <Picker
            className="route-select"
            placeholder={routeNames.length ? "select route…" : "no routes"}
            value={current ?? ""}
            onChange={(v) => { setActive(v); setSelId(null); }}
            options={routeNames.map((n) => ({ value: n }))}
          />
          <button onClick={addRoute}>+ route</button>
          {current && (
            <>
              <input className="name-input" key={current} defaultValue={current} onBlur={(e) => renameRoute(current, e.target.value)} />
              <button className="link danger" title="delete route" onClick={() => deleteRoute(current)}>✕</button>
              <span className="spacer" />
              <label className="numfield inline" title="which spawn region this route is viewed from">
                <span>from spawn</span>
                <Picker
                  searchable={spawnNames.length > 6}
                  placeholder={spawnNames.length ? "spawn region…" : "no spawn regions"}
                  value={spawnName ?? ""}
                  onChange={(v) => setSpawnSel(v || null)}
                  options={spawnNames.map((n) => ({ value: n }))}
                />
              </label>
            </>
          )}
        </div>

        <div className="route-graph-area">
          {current ? (
            <div className="route-graph">
              <ReactFlow
                nodes={rfNodes}
                edges={rfEdges}
                nodeTypes={NODE_TYPES}
                onNodeClick={(_, n) => setSelId(n.id)}
                onPaneClick={() => setSelId(null)}
                nodesConnectable={false}
                nodesDraggable={false}
                fitView
                proOptions={{ hideAttribution: true }}
              >
                <Background />
                <Controls showInteractive={false} />
              </ReactFlow>
            </div>
          ) : (
            <div className="route-empty muted">Create a route to start composing it.</div>
          )}

          {current && (
            <div className="route-inspector">
              <RouteStepInspector
                selNode={selNode}
                steps={steps}
                addOptions={addOptions}
                appendLabel={selNode && (selNode.kind === "waypoint" || selNode.kind === "subroute") ? "add after" : "add to end"}
                onAppend={appendStep}
                onBranch={(v) => selAddr && setSteps(branchAfter(steps, selAddr, stepFromValue(v)))}
                onAddOption={(v) => selAddr && setSteps(addOption(steps, selAddr, stepFromValue(v)))}
                onRemoveOption={(oi) => selAddr && setSteps(removeOption(steps, selAddr, oi))}
                onSetWeight={(oi, w) => selAddr && setSteps(setOptionWeight(steps, selAddr, oi, w))}
                onDelete={() => selAddr && setSteps(deleteAt(steps, selAddr))}
              />
            </div>
          )}
        </div>
      </div>

      <div className="route-bottom">
        <RouteMap
          spec={spec}
          highlightRoute={current}
          spawnIndex={spawnIdx}
          routeWaypoints={routeWaypoints}
          onPickWaypoint={onPickWaypoint}
        />
      </div>
    </div>
  );
}

// The right-hand inspector: actions for the selected graph node (add after,
// branch, delete) or the route as a whole (add to end). Branch nodes get option
// + weight controls.
function RouteStepInspector({
  selNode,
  steps,
  addOptions,
  appendLabel,
  onAppend,
  onBranch,
  onAddOption,
  onRemoveOption,
  onSetWeight,
  onDelete,
}: {
  selNode: ReturnType<typeof routeToGraph>["nodes"][number] | null;
  steps: RouteStep[];
  addOptions: { value: string; label: string; category: string }[];
  appendLabel: string;
  onAppend: (v: string) => void;
  onBranch: (v: string) => void;
  onAddOption: (v: string) => void;
  onRemoveOption: (oi: number) => void;
  onSetWeight: (oi: number, w: number) => void;
  onDelete: () => void;
}) {
  const isBranch = selNode?.kind === "branch";
  const isStep = selNode?.kind === "waypoint" || selNode?.kind === "subroute";
  // Find the choice step for a selected branch node, to list its options.
  const choice = useMemo(() => {
    if (!isBranch || !selNode?.addr) return null;
    let list: any = steps;
    const a = selNode.addr;
    for (let i = 0; i + 1 < a.length; i += 2) list = list[a[i]].choice[a[i + 1]];
    return list[a[a.length - 1]];
  }, [isBranch, selNode, steps]);

  return (
    <div className="route-inspector-body">
      {isStep && (
        <>
          <div className="sub-label">{selNode!.label}</div>
          <label className="numfield inline">
            <span>{appendLabel}</span>
            <Picker placeholder="+ waypoint…" onChange={onAppend} options={addOptions} />
          </label>
          <label className="numfield inline">
            <span>branch</span>
            <Picker placeholder="+ split to…" onChange={onBranch} options={addOptions} />
          </label>
          <button className="link danger" onClick={onDelete}>✕ delete step</button>
        </>
      )}

      {isBranch && choice && (
        <>
          <div className="sub-label">branch · {choice.choice.length} options</div>
          {choice.choice.map((opt: RouteStep[], oi: number) => (
            <div className="route-opt-row" key={oi}>
              <span className="muted small">opt {oi + 1}</span>
              <input
                type="number"
                min={0}
                step={0.25}
                className="route-weight"
                title="relative likelihood"
                key={`${oi}:${choice.weights?.[oi] ?? 1}`}
                defaultValue={Number(choice.weights?.[oi] ?? 1)}
                onBlur={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) onSetWeight(oi, v); }}
              />
              <button className="chip-x" title="remove option" onClick={() => onRemoveOption(oi)}>✕</button>
            </div>
          ))}
          <label className="numfield inline">
            <span>add option</span>
            <Picker placeholder="+ option…" onChange={onAddOption} options={addOptions} />
          </label>
        </>
      )}

      {!isStep && !isBranch && (
        <>
          <div className="muted small">Select a node to edit it, or add to the end of the route.</div>
          <label className="numfield inline">
            <span>add to end</span>
            <Picker placeholder="+ waypoint…" onChange={onAppend} options={addOptions} />
          </label>
        </>
      )}
    </div>
  );
}
