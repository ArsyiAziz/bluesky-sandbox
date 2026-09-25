// The Route tab: a guided node-graph editor for a route (top) over a read-only
// map preview (bottom). The graph is decompiled from the route's nested
// step-list; every edit is a pure step-list transform (routeGraph ops), so the
// route stays valid. Besides the inspector and clicking waypoints on the map,
// a drag edits it: from a node's dot into space adds a step after it, onto a
// later node on its path branches around the steps between; a node dragged
// along its path moves there; a waypoint dropped on a line is inserted there.
import { useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  BaseEdge,
  Controls,
  Handle,
  Position,
  getBezierPath,
  useReactFlow,
  type Connection,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeChange,
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
  addFrom,
  addOption,
  branchAfter,
  canSkip,
  deleteAt,
  isChoice,
  dropIndex,
  insertOnEdge,
  moveWithin,
  removeOption,
  routeToGraph,
  setOptionWeight,
  skipBetween,
} from "./routeGraph";

// What a dragged waypoint chip carries.
const STEP_MIME = "application/x-route-step";

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
      <Handle type="target" position={Position.Left} className="rf-handle" isConnectableStart={false} />
      <span className="rf-node-label">{data.label}</span>
      {data.onDelete && (
        <button className="rf-node-x" title="remove" onClick={(e) => { e.stopPropagation(); data.onDelete!(); }}>✕</button>
      )}
      <Handle type="source" position={Position.Right} className="rf-handle" />
    </div>
  );
}

type EdgeData = {
  over: boolean;
  onOver: (over: boolean) => void;
  onDropStep: (value: string) => void;
};

// An edge with a wide, invisible band a dragged waypoint can be dropped on.
function InsertEdge(props: EdgeProps<Edge<EdgeData>>) {
  const { id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, label, data } = props;
  const [path, labelX, labelY] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  const accepts = (e: React.DragEvent) => e.dataTransfer.types.includes(STEP_MIME);
  return (
    <>
      <BaseEdge id={id} path={path} label={label} labelX={labelX} labelY={labelY} className={data?.over ? "rf-edge-over" : ""} />
      <path
        d={path}
        className="rf-edge-drop"
        onDragOver={(e) => {
          if (!accepts(e)) return;
          e.preventDefault();
          e.dataTransfer.dropEffect = "copy";
          data?.onOver(true);
        }}
        onDragLeave={() => data?.onOver(false)}
        onDrop={(e) => {
          if (!accepts(e)) return;
          e.preventDefault();
          data?.onOver(false);
          data?.onDropStep(e.dataTransfer.getData(STEP_MIME));
        }}
      />
    </>
  );
}

// Fit the graph in view when its nodes come or go, so a step added past the
// edge is not left out of sight.
const FIT = { maxZoom: 1, padding: 0.25 };
function FitOnChange({ signature }: { signature: string }) {
  const { fitView } = useReactFlow();
  useEffect(() => {
    // After the new nodes are measured.
    const handle = requestAnimationFrame(() => fitView({ ...FIT, duration: 150 }));
    return () => cancelAnimationFrame(handle);
  }, [signature, fitView]);
  return null;
}

const NODE_TYPES = { step: StepNode };
const EDGE_TYPES = { insert: InsertEdge };

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
  // A node being dragged along its path, and where it is.
  const [drag, setDrag] = useState<{ id: string; x: number } | null>(null);
  // The edge a waypoint is being dragged over.
  const [overEdge, setOverEdge] = useState<string | null>(null);
  // A drag from a node's dot let go in empty space: what to add there.
  const [adding, setAdding] = useState<{ from: string; x: number; y: number } | null>(null);
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

  const movable = (kind: string) => kind === "waypoint" || kind === "subroute" || kind === "branch";
  const rfNodes: Node<NodeData>[] = graph.nodes.map((n) => ({
    id: n.id,
    type: "step",
    // A dragged node follows the pointer along its row only: it moves within
    // its path, not out of it.
    position: { x: drag?.id === n.id ? drag.x : n.x, y: n.y },
    data: {
      label: n.kind === "start" ? spawnName ?? "spawn" : n.label || (n.kind === "merge" ? "●" : ""),
      kind: n.kind,
      selected: n.id === selId,
      onDelete: n.addr && (n.kind === "waypoint" || n.kind === "subroute") ? () => setSteps(deleteAt(steps, n.addr!)) : undefined,
    },
    draggable: movable(n.kind),
    selectable: n.kind !== "merge",
  }));
  const nodeById = (id: string | null | undefined) => graph.nodes.find((n) => n.id === id);
  const rfEdges: Edge<EdgeData>[] = graph.edges.map((e) => ({
    id: e.id,
    type: "insert",
    source: e.source,
    target: e.target,
    label: e.label,
    animated: false,
    data: {
      over: overEdge === e.id,
      onOver: (over: boolean) => setOverEdge(over ? e.id : null),
      onDropStep: (value: string) => setSteps(insertOnEdge(steps, graph.nodes, e, stepFromValue(value))),
    },
  }));

  const onNodesChange = (changes: NodeChange<Node<NodeData>>[]) => {
    for (const c of changes) if (c.type === "position" && c.position) setDrag({ id: c.id, x: c.position.x });
  };
  const onNodeDragStop = (_: unknown, node: Node) => {
    const g = nodeById(node.id);
    setDrag(null);
    if (g?.addr) setSteps(moveWithin(steps, g.addr, dropIndex(graph.nodes, g, node.position.x)));
  };
  const isValidConnection = (c: Edge | Connection) => {
    const from = nodeById(c.source);
    const to = nodeById(c.target);
    return Boolean(from && to && canSkip(steps, from, to));
  };
  const onConnect = (c: Connection) => {
    const from = nodeById(c.source);
    const to = nodeById(c.target);
    if (from && to) setSteps(skipBetween(steps, from, to));
  };
  const onConnectEnd = (event: MouseEvent | TouchEvent, state: { fromNode?: { id: string } | null; toNode?: unknown }) => {
    if (state.toNode || !state.fromNode) return;
    const point = "changedTouches" in event ? event.changedTouches[0] : event;
    setAdding({ from: state.fromNode.id, x: point.clientX, y: point.clientY });
  };

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

  // Steps in a route, counted through its branches.
  const stepCount = (list: RouteStep[]): number =>
    list.reduce((n: number, st) => n + (isChoice(st) ? st.choice.reduce((m, o) => m + stepCount(o), 0) : 1), 0);
  const usedBy = (name: string) => spawnRegions.filter((r) => r.route === name).length;

  return (
    <div className="route-tab">
      <aside className="route-list">
        <div className="route-list-head">
          <span>routes</span>
          <button onClick={addRoute} title="add a route">
            + route
          </button>
        </div>
        {routeNames.map((n) => (
          <div
            key={n}
            className={n === current ? "route-list-row on" : "route-list-row"}
            onClick={() => {
              setActive(n);
              setSelId(null);
            }}
          >
            <span className="route-list-name">{n}</span>
            <span className="muted small">
              {stepCount(routes[n] ?? [])} step{stepCount(routes[n] ?? []) === 1 ? "" : "s"}
              {usedBy(n) ? ` · ${usedBy(n)} spawn${usedBy(n) === 1 ? "" : "s"}` : ""}
            </span>
          </div>
        ))}
        {routeNames.length === 0 && <p className="muted small">No routes yet.</p>}
      </aside>

      <div className="route-main">
        {current && (
          <div className="route-head">
            <input
              className="route-name"
              key={current}
              defaultValue={current}
              title="route name"
              onBlur={(e) => renameRoute(current, e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
            />
            <label className="route-from" title="which spawn region this route is viewed from">
              <span className="muted">from</span>
              <Picker
                searchable={spawnNames.length > 6}
                placeholder={spawnNames.length ? "spawn region…" : "no spawn regions"}
                value={spawnName ?? ""}
                onChange={(v) => setSpawnSel(v || null)}
                options={spawnNames.map((n) => ({ value: n }))}
              />
            </label>
            <span className="spacer" />
            <button className="danger-btn" title="delete this route" onClick={() => deleteRoute(current)}>
              Delete route
            </button>
          </div>
        )}

        <div className="route-graph-area">
          {current ? (
            <div className="route-graph">
              <ReactFlow
                nodes={rfNodes}
                edges={rfEdges}
                nodeTypes={NODE_TYPES}
                edgeTypes={EDGE_TYPES}
                colorMode="dark"
                onNodeClick={(_, n) => setSelId(n.id)}
                onPaneClick={() => setSelId(null)}
                onNodesChange={onNodesChange}
                onNodeDragStop={onNodeDragStop}
                isValidConnection={isValidConnection}
                onConnect={onConnect}
                onConnectEnd={onConnectEnd}
                fitView
                fitViewOptions={FIT}
                proOptions={{ hideAttribution: true }}
              >
                <Background />
                <Controls showInteractive={false} />
                <FitOnChange signature={`${current}:${graph.nodes.map((n) => n.id).join(",")}`} />
              </ReactFlow>
              {adding && (
                <AddPopover
                  x={adding.x}
                  y={adding.y}
                  options={addOptions}
                  onPick={(v) => {
                    const from = nodeById(adding.from);
                    if (from) setSteps(addFrom(steps, from, stepFromValue(v)));
                    setAdding(null);
                  }}
                  onClose={() => setAdding(null)}
                />
              )}
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
          <button className="danger-btn route-delete-step" onClick={onDelete}>
            Delete step
          </button>
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

      <div className="sub-label">drag onto a line</div>
      <div className="route-palette">
        {addOptions.map((o) => (
          <span
            key={o.value}
            className="geo-row-kind route-palette-item"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData(STEP_MIME, o.value);
              e.dataTransfer.effectAllowed = "copy";
            }}
            title={`drag onto a line in the graph to insert ${o.label} there`}
          >
            {o.label}
          </span>
        ))}
      </div>
      <p className="muted small">
        Drag a node's dot into space to add after it, or onto a later node on its path to branch around
        the steps between. Drag a node sideways to move it along its path.
      </p>
    </div>
  );
}

// The steps to add, opened where a drag from a node's dot was let go.
function AddPopover({
  x,
  y,
  options,
  onPick,
  onClose,
}: {
  x: number;
  y: number;
  options: { value: string; label: string; category: string }[];
  onPick: (value: string) => void;
  onClose: () => void;
}) {
  const [filter, setFilter] = useState("");
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const shown = options.filter((o) => o.label.toLowerCase().includes(filter.toLowerCase()));
  return (
    <>
      <div className="route-popover-backdrop" onClick={onClose} />
      <div className="route-popover" style={{ left: x, top: y }}>
        <input
          autoFocus
          placeholder="add after…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && shown[0] && onPick(shown[0].value)}
        />
        <div className="route-popover-list">
          {shown.map((o) => (
            <button key={o.value} onClick={() => onPick(o.value)}>
              {o.label}
            </button>
          ))}
          {shown.length === 0 && <span className="muted small">nothing matches</span>}
        </div>
      </div>
    </>
  );
}
