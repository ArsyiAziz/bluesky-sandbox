// The Geometry outline's two views of the bounds:
//   - ShapesTree: by kind - groups as folders holding bounds and other groups.
//     Drag a bounds onto a bounds to group the two, onto a group to join it; a
//     group onto a group to nest it; either onto the top to take it out. (The
//     inspectors' "group" pickers do the same from the keyboard.)
//   - DependencyTree: by dependency - each bounds under the bounds it is drawn,
//     placed or moved by, with the elements that use it beneath.
import { type DragEvent, useState } from "react";
import type { SpecDict } from "../../api";
import { type ShapeInfo, isUnused } from "../../shapeGraph";
import { childGroups, type Group, groupOfShape, groupsOf, groupTogether, isWithin, moveShape, moveGroup } from "../../groupTree";
import { targetKey } from "../../map/editHandles";
import type { EditTarget } from "../../map/types";
import { Row } from "./OutlineRow";

type Drag = { type: "bound"; name: string } | { type: "group"; id: string };

export function ShapesTree({
  spec,
  graph,
  topShapes,
  has,
  selectedKey,
  onSelect,
  lockedElements,
  onToggleLocked,
  refCount,
  edit,
}: {
  spec: SpecDict;
  graph: Record<string, ShapeInfo>;
  // The bounds in no group to list (shared or standalone ones; all, asked).
  topShapes: string[];
  has: (name: string) => boolean;
  selectedKey: string;
  onSelect: (t: EditTarget) => void;
  lockedElements: Set<string>;
  onToggleLocked: (key: string) => void;
  refCount: (name: string) => number;
  edit: (mut: (s: SpecDict) => void) => void;
}) {
  const [drag, setDrag] = useState<Drag | null>(null);
  const [over, setOver] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  // Whether what is dragged may be dropped on `on`.
  const takes = (on: Drag | "top"): boolean => {
    if (!drag) return false;
    if (on === "top") return drag.type === "bound" ? !!groupOfShape(spec, drag.name) : !!groupById(spec, drag.id)?.parent;
    if (on.type === "bound") return drag.type === "bound" && drag.name !== on.name;
    if (drag.type === "bound") return groupOfShape(spec, drag.name)?.id !== on.id;
    return drag.id !== on.id && !isWithin(spec, on.id, drag.id) && groupById(spec, drag.id)?.parent !== on.id;
  };
  const drop = (on: Drag | "top") => {
    const d = drag;
    setDrag(null);
    setOver(null);
    if (!d || !takes(on)) return;
    edit((s) => {
      if (on === "top") {
        if (d.type === "bound") moveShape(s, d.name, null);
        else moveGroup(s, d.id, null);
      } else if (on.type === "bound") {
        if (d.type === "bound") groupTogether(s, d.name, on.name);
      } else if (d.type === "bound") moveShape(s, d.name, on.id);
      else moveGroup(s, d.id, on.id);
    });
  };
  // The handlers that make a row draggable as `self` and a drop target.
  const dnd = (self: Drag, key: string) => ({
    draggable: true,
    onDragStart: (e: DragEvent<HTMLDivElement>) => {
      e.stopPropagation();
      e.dataTransfer.effectAllowed = "move";
      e.dataTransfer.setData("text/plain", self.type === "bound" ? self.name : self.id);
      setDrag(self);
    },
    onDragEnd: () => {
      setDrag(null);
      setOver(null);
    },
    onDragOver: (e: DragEvent<HTMLDivElement>) => {
      if (!takes(self)) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      if (over !== key) setOver(key);
    },
    onDragLeave: (e: DragEvent<HTMLDivElement>) => {
      if (!e.currentTarget.contains(e.relatedTarget as Node | null) && over === key) setOver(null);
    },
    onDrop: (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      drop(self);
    },
  });

  // Whether a group, or anything in it, passes the filter.
  const shows = (g: Group): boolean =>
    has(g.name || g.id) ||
    (g.members ?? []).some((m) => has(m)) ||
    childGroups(spec, g.id).some(shows);

  const shapeRow = (name: string, depth: number) => (
    <Row
      key={`b:${name}`}
      name={name}
      depth={depth}
      flag={isUnused(graph[name]) ? "unused" : undefined}
      kind={`${name === spec.airspace?.ref ? "airspace" : spec.shapes?.[name]?.footprint?.type === "point" ? "point" : "area"}${refCount(name) > 1 ? ` ·${refCount(name)}` : ""}`}
      selected={selectedKey === `region:${name}`}
      onClick={() => onSelect({ scope: "region", name })}
      locked={lockedElements.has(`region:${name}`)}
      onToggleLocked={() => onToggleLocked(`region:${name}`)}
      dropping={over === `b:${name}`}
      dnd={dnd({ type: "bound", name }, `b:${name}`)}
    />
  );

  const groupNode = (g: Group, depth: number): React.ReactNode => {
    if (!shows(g)) return null;
    const open = !collapsed.has(g.id);
    const subs = childGroups(spec, g.id);
    const members = (g.members ?? []).filter((m) => graph[m]);
    const count = members.length + subs.length;
    const toggle = () =>
      setCollapsed((prev) => {
        const next = new Set(prev);
        if (next.has(g.id)) next.delete(g.id);
        else next.add(g.id);
        return next;
      });
    return (
      <div key={`g:${g.id}`} className="geo-tree-group" role="group" aria-label={g.name || g.id}>
        <Row
          name={g.name || g.id}
          kind="group"
          depth={depth}
          note={`${count} item${count === 1 ? "" : "s"}`}
          selected={selectedKey === `group:${g.id}`}
          onClick={() => onSelect({ scope: "group", id: g.id })}
          lead={
            <button
              type="button"
              className="geo-caret"
              aria-expanded={open}
              aria-label={open ? "collapse" : "expand"}
              onClick={(e) => {
                e.stopPropagation();
                toggle();
              }}
            >
              {open ? "▾" : "▸"}
            </button>
          }
          dropping={over === `g:${g.id}`}
          dnd={dnd({ type: "group", id: g.id }, `g:${g.id}`)}
        />
        {open && (
          <>
            {subs.map((sub) => groupNode(sub, depth + 1))}
            {members.filter((m) => has(m) || has(g.name || g.id)).map((m) => shapeRow(m, depth + 1))}
          </>
        )}
      </div>
    );
  };

  const grouped = new Set(groupsOf(spec).flatMap((g) => g.members ?? []));
  const top = topShapes.filter((n) => !grouped.has(n) && has(n));
  const tops = childGroups(spec, null);
  return (
    <div className={drag ? "geo-tree dragging" : "geo-tree"}>
      {drag && (
        <div
          className={["geo-drop-top", takes("top") && "takes", over === "top" && "dropping"].filter(Boolean).join(" ")}
          onDragOver={(e) => {
            if (!takes("top")) return;
            e.preventDefault();
            if (over !== "top") setOver("top");
          }}
          onDragLeave={() => over === "top" && setOver(null)}
          onDrop={(e) => {
            e.preventDefault();
            drop("top");
          }}
        >
          {takes("top") ? "drop here to take it out of its group" : "drop on a shape to group the two, on a group to join it"}
        </div>
      )}
      {tops.map((g) => groupNode(g, 0))}
      {top.map((n) => shapeRow(n, 0))}
      {tops.length === 0 && top.length === 0 && Object.keys(graph).length > 0 && (
        <div className="muted small">no shared or standalone shapes</div>
      )}
    </div>
  );
}

const groupById = (spec: SpecDict, id: string): Group | undefined => groupsOf(spec).find((g) => g.id === id);

// ---- by dependency --------------------------------------------------------

// The kind of element a target is, for its tag: a queryable is a region or a
// waypoint.
function kindOf(spec: SpecDict, t: EditTarget): string {
  if (t.scope === "queryable") return spec.queryables?.[t.name]?.type === "waypoint" ? "waypoint" : "region";
  if (t.scope === "region") return kindOfShape(spec, t.name);
  return t.scope;
}

// A shape's kind: a point, or an area.
const kindOfShape = (spec: SpecDict, name: string): string =>
  spec.shapes?.[name]?.footprint?.type === "point" ? "point" : "area";

// How a shape depends on the one above it, briefly: "placed in", "avoids",
// "drifts in", "split from" - from the layer param it refers through.
function relation(via: string): string {
  const [layer, key] = via.includes(" ") ? via.split(" ", 2) : ["", via];
  if (layer === "placement") return key === "avoid" ? "avoids" : "placed in";
  if (layer) return `${layer.toLowerCase()}s ${key === "within" ? "in" : key}`;
  if (key === "parent") return "split from";
  if (key === "within") return "drawn in";
  return key.replace(/_/g, " ");
}

export function DependencyTree({
  spec,
  graph,
  has,
  selectedKey,
  onSelect,
}: {
  spec: SpecDict;
  graph: Record<string, ShapeInfo>;
  has: (name: string) => boolean;
  selectedKey: string;
  onSelect: (t: EditTarget) => void;
}) {
  const names = Object.keys(graph);
  const base = (ref: string) => (graph[ref] ? ref : ref.replace(/\.\d+$/, ""));
  // The shapes drawn, placed or moved by `name` - each with how.
  const dependents = (name: string): [string, string][] =>
    names.flatMap((n) => {
      const vias = graph[n].dependsOn.filter((d) => base(d.name) === name && n !== name).map((d) => relation(d.via));
      return vias.length ? [[n, [...new Set(vias)].join(", ")] as [string, string]] : [];
    });
  // Roots: shapes that depend on no other. A cycle has none, so whatever is
  // left once the roots are drawn is drawn from its first shape.
  const roots = names.filter((n) => !graph[n].dependsOn.some((d) => graph[base(d.name)] && base(d.name) !== n));

  // Whether `name`'s subtree has anything that passes the filter.
  const matches = (name: string, seen = new Set<string>()): boolean => {
    if (seen.has(name)) return false;
    seen.add(name);
    return (
      has(name) ||
      graph[name].usedBy.some((u) => has(u.label)) ||
      dependents(name).some(([n]) => matches(n, seen))
    );
  };

  const seen = new Set<string>();
  const node = (name: string, how?: string): React.ReactNode => {
    const info = graph[name];
    const kind = kindOfShape(spec, name);
    const head = (
      <button
        type="button"
        className={selectedKey === `region:${name}` ? "dep-name on" : "dep-name"}
        onClick={() => onSelect({ scope: "region", name })}
        title={`open ${name}`}
      >
        <span className="dep-kind" data-kind={kind}>
          {kind}
        </span>
        {name}
      </button>
    );
    if (seen.has(name)) {
      return (
        <li key={`${name}:again`} className="dep-node again">
          <div className="dep-row">
            {how && <span className="dep-how">{how}</span>}
            {head}
            <span className="dep-note">↑ above</span>
          </div>
        </li>
      );
    }
    seen.add(name);
    const users = info.usedBy.filter((u) => u.target && u.target.scope !== "region");
    const children = dependents(name);
    return (
      <li key={name} className="dep-node">
        <div className="dep-row">
          {how && <span className="dep-how">{how}</span>}
          {head}
          {info.group && (
            <button
              type="button"
              className="chip dep group"
              data-kind="group"
              title={`in group ${info.group.name}`}
              onClick={() => onSelect({ scope: "group", id: info.group!.id })}
            >
              {info.group.name}
            </button>
          )}
          {isUnused(info) && (
            <span className="geo-row-flag" title="Nothing uses this shape: no element, other shape, group's moves or code">
              <span aria-hidden="true">⚠</span> unused
            </span>
          )}
        </div>
        {(children.length > 0 || users.length > 0 || info.inCode.length > 0) && (
          <ul className="dep-children">
            {children.map(([n, rel]) => node(n, rel))}
            {users.map((u, i) => {
              const t = u.target as EditTarget;
              const kind = u.kind ?? kindOf(spec, t);
              return (
                <li key={`use:${i}`} className="dep-node leaf">
                  <button
                    type="button"
                    className={selectedKey === targetKey(t) ? "dep-row dep-use on" : "dep-row dep-use"}
                    onClick={() => onSelect(t)}
                  >
                    <span className="dep-kind" data-kind={kind}>
                      {kind}
                    </span>
                    {u.label}
                  </button>
                </li>
              );
            })}
            {info.inCode.map((where) => (
              <li key={`code:${where}`} className="dep-node leaf">
                <span className="dep-row dep-use code">
                  <span className="dep-kind" data-kind="code">
                    code
                  </span>
                  {where}
                </span>
              </li>
            ))}
          </ul>
        )}
      </li>
    );
  };
  const tops = [...roots, ...names.filter((n) => !roots.includes(n))];
  const rows: React.ReactNode[] = [];
  for (const r of tops) if (!seen.has(r) && matches(r)) rows.push(node(r));

  // Elements on no shape: older waypoints at a position of their own.
  const loose = Object.entries(spec.queryables ?? {}).filter(
    ([n, q]: [string, any]) => q?.type === "waypoint" && !q.shape && !q.anchor && !q.sample?.ref && has(n),
  );
  return (
    <div className="dep-tree">
      {rows.length > 0 && <ul className="dep-children top">{rows}</ul>}
      {names.length === 0 && <div className="muted small">no shapes yet</div>}
      {loose.length > 0 && (
        <>
          <div className="geo-tree-sub">on no shape</div>
          {loose.map(([n]) => (
            <Row
              key={`wp:${n}`}
              name={n}
              kind="waypoint"
              selected={selectedKey === `queryable:${n}`}
              onClick={() => onSelect({ scope: "queryable", name: n })}
            />
          ))}
        </>
      )}
    </div>
  );
}
