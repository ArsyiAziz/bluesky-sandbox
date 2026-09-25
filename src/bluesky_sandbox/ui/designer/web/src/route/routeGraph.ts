// Route ⇄ graph translation for the Route tab.
//
// A route is a nested step-list (the spec model): a waypoint name, a constrained
// waypoint {waypoint,speed_kts?,alt_ft?}, a subroute reference {route}, or a
// branch {choice:[steps…][], weights?}. This module DECOMPILES that into a laid-
// out node/edge graph for React Flow, and provides guided EDIT operations that
// are pure transforms on the step-list — so every edit yields a valid
// series-parallel route (no fragile graph→step compilation).

export type WpStep = { waypoint: string; speed_kts?: number; alt_ft?: number };
export type Subroute = { route: string };
export type Choice = { choice: RouteStep[][]; weights?: number[] };
export type RouteStep = string | Subroute | Choice | WpStep;

export const isSubroute = (s: RouteStep): s is Subroute =>
  typeof s === "object" && s != null && typeof (s as any).route === "string";
export const isChoice = (s: RouteStep): s is Choice =>
  typeof s === "object" && s != null && Array.isArray((s as any).choice);
export const isWpStep = (s: RouteStep): s is WpStep =>
  typeof s === "object" && s != null && typeof (s as any).waypoint === "string";
export const stepName = (s: RouteStep): string => (isWpStep(s) ? s.waypoint : (s as string));

// An address into the nested step-list: pairs of (choiceIndex, optionIndex)
// descend into branch options, and the final element is the index in the
// deepest list. So [2] is the 3rd top-level step; [1,0,2] is the 3rd step of
// option 0 of the choice at top-level index 1.
export type Addr = number[];

const addrEq = (a: Addr, b: Addr) => a.length === b.length && a.every((v, i) => v === b[i]);
const addrKey = (a: Addr) => a.join(".");

// Resolve an address to its containing list + index within a (already-cloned)
// route, so an op can splice in place.
function locate(route: RouteStep[], addr: Addr): { list: RouteStep[]; index: number } {
  let list = route;
  for (let i = 0; i + 1 < addr.length; i += 2) {
    const step = list[addr[i]] as Choice;
    list = step.choice[addr[i + 1]];
  }
  return { list, index: addr[addr.length - 1] };
}

// --- Graph model (React-Flow-friendly, but library-agnostic here) -----------

export type GNodeKind = "start" | "waypoint" | "subroute" | "branch" | "merge";
export type GNode = {
  id: string;
  kind: GNodeKind;
  label: string;
  addr: Addr | null; // the step this node represents (null for start/merge)
  x: number;
  y: number;
  // Branch-only: per-option weights + the option entry node ids (for labels).
  optionCount?: number;
  // Merge-only: the branch step it closes.
  of?: Addr;
};
export type GEdge = {
  id: string;
  source: string;
  target: string;
  label?: string;
  // For an edge leaving a branch node: which option it enters (to edit weight).
  choiceAddr?: Addr;
  optionIndex?: number;
};

const COL = 140;
const ROW = 60;

type Sub = {
  nodes: GNode[];
  edges: GEdge[];
  entries: string[];
  exits: string[];
  cols: number;
  rows: number;
};

// Lay out one step at grid (col,row-top). Returns its sub-graph with grid spans.
function layoutStep(step: RouteStep, addr: Addr, col: number, rowTop: number): Sub {
  if (isChoice(step)) {
    const branchId = `b:${addrKey(addr)}`;
    const mergeId = `m:${addrKey(addr)}`;
    const nodes: GNode[] = [];
    const edges: GEdge[] = [];
    let row = rowTop;
    let maxOptCols = 1;
    const options = step.choice.length ? step.choice : [[]];
    const optExits: string[][] = [];
    const optEntries: string[][] = [];
    options.forEach((opt, oi) => {
      const sub = layoutSequence(opt, [...addr, oi], col + 1, row);
      nodes.push(...sub.nodes);
      edges.push(...sub.edges);
      optEntries.push(sub.entries);
      optExits.push(sub.exits);
      maxOptCols = Math.max(maxOptCols, sub.cols);
      row += Math.max(1, sub.rows);
    });
    const rows = Math.max(1, row - rowTop);
    const midRow = rowTop + rows / 2 - 0.5;
    const mergeCol = col + 1 + maxOptCols;
    nodes.push({ id: branchId, kind: "branch", label: "⎇", addr, x: col * COL, y: midRow * ROW, optionCount: options.length });
    nodes.push({ id: mergeId, kind: "merge", label: "", addr: null, of: addr, x: mergeCol * COL, y: midRow * ROW });
    options.forEach((_, oi) => {
      const w = step.weights?.[oi];
      const entries = optEntries[oi].length ? optEntries[oi] : [mergeId];
      for (const e of entries) {
        edges.push({
          id: `e:${branchId}->${e}`,
          source: branchId,
          target: e,
          label: w != null ? `w ${w}` : undefined,
          choiceAddr: addr,
          optionIndex: oi,
        });
      }
      for (const x of optExits[oi]) edges.push({ id: `e:${x}->${mergeId}:${oi}`, source: x, target: mergeId });
    });
    return { nodes, edges, entries: [branchId], exits: [mergeId], cols: maxOptCols + 2, rows };
  }
  // A plain waypoint or subroute: a single node.
  const id = `n:${addrKey(addr)}`;
  const kind: GNodeKind = isSubroute(step) ? "subroute" : "waypoint";
  const label = isSubroute(step) ? `⤷ ${step.route}` : stepName(step);
  return {
    nodes: [{ id, kind, label, addr, x: col * COL, y: rowTop * ROW }],
    edges: [],
    entries: [id],
    exits: [id],
    cols: 1,
    rows: 1,
  };
}

// Lay out a step list left-to-right, chaining each step's exits to the next's
// entries. `col`/`rowTop` are grid coordinates (multiplied by COL/ROW).
function layoutSequence(steps: RouteStep[], addrPrefix: Addr, col: number, rowTop: number): Sub {
  const nodes: GNode[] = [];
  const edges: GEdge[] = [];
  let curCol = col;
  let prevExits: string[] = [];
  let entries: string[] = [];
  let maxRows = 1;
  steps.forEach((step, i) => {
    const sub = layoutStep(step, [...addrPrefix, i], curCol, rowTop);
    nodes.push(...sub.nodes);
    edges.push(...sub.edges);
    if (i === 0) entries = sub.entries;
    for (const p of prevExits) for (const e of sub.entries) edges.push({ id: `seq:${p}->${e}`, source: p, target: e });
    prevExits = sub.exits;
    curCol += sub.cols;
    maxRows = Math.max(maxRows, sub.rows);
  });
  return { nodes, edges, entries, exits: prevExits, cols: Math.max(1, curCol - col), rows: maxRows };
}

// Decompile a route into a laid-out graph with a synthetic START node.
export function routeToGraph(steps: RouteStep[]): { nodes: GNode[]; edges: GEdge[] } {
  const seq = layoutSequence(steps ?? [], [], 1, 0);
  const start: GNode = { id: "start", kind: "start", label: "spawn", addr: null, x: 0, y: (seq.rows / 2 - 0.5) * ROW };
  const edges = [...seq.edges];
  for (const e of seq.entries) edges.push({ id: `start->${e}`, source: "start", target: e });
  return { nodes: [start, ...seq.nodes], edges };
}

// --- Guided edit operations (pure: route in, new route out) ------------------

const clone = (r: RouteStep[]): RouteStep[] => structuredClone(r);

export function appendAfter(route: RouteStep[], addr: Addr, step: RouteStep): RouteStep[] {
  const next = clone(route);
  const { list, index } = locate(next, addr);
  list.splice(index + 1, 0, step);
  return next;
}

// Insert a step at the end of the top-level route (used when nothing is selected).
export function appendToEnd(route: RouteStep[], step: RouteStep): RouteStep[] {
  const next = clone(route);
  next.push(step);
  return next;
}

export function insertBefore(route: RouteStep[], addr: Addr, step: RouteStep): RouteStep[] {
  const next = clone(route);
  const { list, index } = locate(next, addr);
  list.splice(index, 0, step);
  return next;
}

export function deleteAt(route: RouteStep[], addr: Addr): RouteStep[] {
  const next = clone(route);
  const { list, index } = locate(next, addr);
  list.splice(index, 1);
  return next;
}

// Replace the step at an address (e.g. editing a crossing restriction).
export function replaceAt(route: RouteStep[], addr: Addr, step: RouteStep): RouteStep[] {
  const next = clone(route);
  const { list, index } = locate(next, addr);
  list[index] = step;
  return next;
}

// Split after a node: wrap the steps following it (in the same list) into a
// branch, with the existing tail as option 1 and a new step as option 2.
export function branchAfter(route: RouteStep[], addr: Addr, newStep: RouteStep): RouteStep[] {
  const next = clone(route);
  const { list, index } = locate(next, addr);
  const tail = list.splice(index + 1);
  const choice: Choice = { choice: [tail, [newStep]] };
  list.push(choice);
  return next;
}

export function addOption(route: RouteStep[], choiceAddr: Addr, step: RouteStep): RouteStep[] {
  const next = clone(route);
  const { list, index } = locate(next, choiceAddr);
  const choice = list[index] as Choice;
  if (!isChoice(choice)) return route;
  choice.choice.push([step]);
  if (choice.weights) choice.weights.push(1);
  return next;
}

export function removeOption(route: RouteStep[], choiceAddr: Addr, optionIndex: number): RouteStep[] {
  const next = clone(route);
  const { list, index } = locate(next, choiceAddr);
  const choice = list[index] as Choice;
  if (!isChoice(choice)) return route;
  choice.choice.splice(optionIndex, 1);
  if (choice.weights) choice.weights.splice(optionIndex, 1);
  // Collapse a 1-option branch back into its inline steps.
  if (choice.choice.length === 1) list.splice(index, 1, ...choice.choice[0]);
  return next;
}

export function setOptionWeight(route: RouteStep[], choiceAddr: Addr, optionIndex: number, weight: number): RouteStep[] {
  const next = clone(route);
  const { list, index } = locate(next, choiceAddr);
  const choice = list[index] as Choice;
  if (!isChoice(choice)) return route;
  const weights = choice.choice.map((_, i) => (i === optionIndex ? Math.max(0, weight) : Number(choice.weights?.[i] ?? 1)));
  // Equal weights mean uniform → store none, keeping the spec minimal.
  choice.weights = weights.every((w) => w === weights[0]) ? undefined : weights;
  return next;
}

// --- Gestures: what a drag on the graph means for the step-list ---------------

// Where a node sits: the list holding it (by address prefix) and its index
// there. The spawn node sits before the route's first step; a merge node sits
// where its branch step does, so what follows a merge follows the branch.
function place(node: GNode): { prefix: Addr; index: number } | null {
  if (node.kind === "start") return { prefix: [], index: -1 };
  const addr = node.kind === "merge" ? node.of : node.addr;
  if (!addr) return null;
  return { prefix: addr.slice(0, -1), index: addr[addr.length - 1] };
}

// Add `step` right after `node`. From a branch node, it becomes a new option.
export function addFrom(route: RouteStep[], node: GNode, step: RouteStep): RouteStep[] {
  if (node.kind === "branch" && node.addr) return addOption(route, node.addr, step);
  const at = place(node);
  if (!at) return route;
  const next = clone(route);
  const { list } = locate(next, [...at.prefix, 0]);
  list.splice(at.index + 1, 0, step);
  return next;
}

// Put `step` on the edge `edge`: before the node it enters, or - on an edge
// into a merge - at the end of the option it closes.
export function insertOnEdge(route: RouteStep[], nodes: GNode[], edge: GEdge, step: RouteStep): RouteStep[] {
  const byId = (id: string) => nodes.find((n) => n.id === id);
  const source = byId(edge.source);
  const target = byId(edge.target);
  if (!source || !target) return route;
  if (target.kind !== "merge" && target.addr) return insertBefore(route, target.addr, step);
  // An empty option: the edge runs from the branch straight to its merge.
  if (source.kind === "branch" && source.addr && edge.optionIndex != null) {
    const next = clone(route);
    const { list, index } = locate(next, source.addr);
    (list[index] as Choice).choice[edge.optionIndex].push(step);
    return next;
  }
  return addFrom(route, source, step);
}

// Where in `to`'s list a skip from `from` would land, or null when there is no
// such skip: both must be on one path, with at least one step between them. A
// merge closing the branch `from` is inside counts as that path's end.
function skipSpan(from: GNode, to: GNode, route: RouteStep[]): { prefix: Addr; start: number; end: number } | null {
  const a = place(from);
  if (!a) return null;
  let end: number;
  if (to.kind === "merge") {
    if (!to.of || !addrEq(a.prefix.slice(0, -1), to.of)) return null;
    end = locate(route, [...a.prefix, 0]).list.length;
  } else {
    const b = to.addr && place(to);
    if (!b || !addrEq(a.prefix, b.prefix)) return null;
    end = b.index;
  }
  return end - a.index >= 2 ? { prefix: a.prefix, start: a.index + 1, end } : null;
}

export function canSkip(route: RouteStep[], from: GNode, to: GNode): boolean {
  return skipSpan(from, to, route) !== null;
}

// A branch around the steps between `from` and `to`: one option takes them,
// the other goes straight on.
export function skipBetween(route: RouteStep[], from: GNode, to: GNode): RouteStep[] {
  const span = skipSpan(from, to, route);
  if (!span) return route;
  const next = clone(route);
  const { list } = locate(next, [...span.prefix, 0]);
  const between = list.splice(span.start, span.end - span.start);
  list.splice(span.start, 0, { choice: [between, []] });
  return next;
}

// Move the step at `addr` to `toIndex` within its own list.
export function moveWithin(route: RouteStep[], addr: Addr, toIndex: number): RouteStep[] {
  const next = clone(route);
  const { list, index } = locate(next, addr);
  if (toIndex === index) return route;
  const [step] = list.splice(index, 1);
  list.splice(toIndex, 0, step);
  return next;
}

// The index a step dragged to x lands at among the steps of its own list.
export function dropIndex(nodes: GNode[], node: GNode, x: number): number {
  if (!node.addr) return 0;
  const prefix = node.addr.slice(0, -1);
  const siblings = nodes.filter(
    (n) =>
      n.id !== node.id &&
      n.addr &&
      n.kind !== "merge" &&
      n.addr.length === node.addr!.length &&
      addrEq(n.addr.slice(0, -1), prefix),
  );
  return siblings.filter((n) => n.x < x).length;
}

export { addrEq, addrKey };
