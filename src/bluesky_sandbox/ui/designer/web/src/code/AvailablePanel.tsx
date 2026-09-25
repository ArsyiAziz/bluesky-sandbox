// "Available here": what the code in the open block can use - its parameters
// and its module's names - each expandable into its attributes, items and keyed
// calls, read from the design's code intel. Clicking a row inserts its full
// expression at the cursor.
import { useState } from "react";
import { api } from "../api";
import type { Intel, Member } from "./intel";

// A row: what it is, the expression it inserts, and how it reads under its parent.
type Row = { member: Member; expr: string; label: string };

function makeRow(member: Member, parent: string | null, step: string): Row {
  return { member, expr: parent === null ? member.name : `${parent}${step}`, label: parent === null ? member.name : step };
}

function children(intel: Intel, parent: Row): Row[] {
  const { member, expr } = parent;
  // A keyed call's keys: query("goal") and on from its result.
  const keys = member.returns_by_key ? intel.types[member.returns_by_key]?.items ?? [] : [];
  if (keys.length) return keys.map((m) => makeRow(m, expr, `(${JSON.stringify(m.name)})`));
  if (member.kind === "function") return [];
  const type = member.type ? intel.types[member.type] : undefined;
  if (!type) return [];
  const items = (type.items ?? []).map((m) => makeRow(m, expr, `[${JSON.stringify(m.name)}]`));
  const attrs = type.attrs.map((m) => makeRow(m, expr, `.${m.name}`));
  return [...items, ...attrs];
}

function Tree({
  intel,
  row: current,
  depth,
  onInsert,
}: {
  intel: Intel;
  row: Row;
  depth: number;
  onInsert: (text: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [moduleRows, setModuleRows] = useState<Row[] | null>(null);
  const { member } = current;
  const expandable = Boolean(member.module) || children(intel, current).length > 0;
  const toggle = () => {
    if (member.module && !moduleRows) {
      api
        .pythonModuleMembers(member.module)
        .then((result) =>
          setModuleRows(
            (result.members as Member[]).map((m) => makeRow(m, current.expr, `.${m.name}`)),
          ),
        )
        .catch(() => setModuleRows([]));
    }
    setOpen(!open);
  };
  const rows = member.module ? moduleRows ?? [] : open ? children(intel, current) : [];
  return (
    <div className="avail-node">
      <div className="avail-row" style={{ paddingLeft: depth * 12 }} title={member.doc || member.detail || ""}>
        <button className="avail-toggle" onClick={toggle} disabled={!expandable} aria-label="expand">
          {expandable ? (open ? "▾" : "▸") : ""}
        </button>
        <button className={`avail-name avail-${member.kind}`} onClick={() => onInsert(current.expr)}>
          {current.label}
        </button>
        {member.detail && <span className="avail-detail">{member.detail}</span>}
      </div>
      {open &&
        rows.map((child) => (
          <Tree key={child.expr} intel={intel} row={child} depth={depth + 1} onInsert={onInsert} />
        ))}
    </div>
  );
}

export function AvailablePanel({
  intel,
  scope,
  onInsert,
}: {
  intel: Intel | null;
  scope: string | null;
  onInsert: (text: string) => void;
}) {
  const [filter, setFilter] = useState("");
  const current = intel && scope ? intel.scopes[scope] : undefined;
  if (!intel || !current) return null;
  const match = (m: Member) => !filter || m.name.toLowerCase().includes(filter.toLowerCase());
  const params = current.params.filter(match);
  const names = (intel.names[current.names] ?? []).filter(match);
  return (
    <div className="avail">
      <h3>Available here</h3>
      <input
        className="avail-filter"
        placeholder="filter names…"
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
      />
      {params.length > 0 && <div className="sub-label">parameters</div>}
      {params.map((m) => (
        <Tree key={`p:${m.name}`} intel={intel} row={makeRow(m, null, "")} depth={0} onInsert={onInsert} />
      ))}
      {names.length > 0 && <div className="sub-label">module names</div>}
      {names.map((m) => (
        <Tree key={`n:${m.name}`} intel={intel} row={makeRow(m, null, "")} depth={0} onInsert={onInsert} />
      ))}
    </div>
  );
}
