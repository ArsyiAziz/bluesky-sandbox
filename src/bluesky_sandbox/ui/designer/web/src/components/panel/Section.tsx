// Small shared panel primitives: a collapsible section header, a collapsible
// element card (with map-selection highlight), and the per-element "hide from
// view" eye toggle (view-only; never touches the spec).
import { Hint } from "./Hint";
import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

export function Section({
  title,
  subtitle,
  hint,
  children,
}: {
  title: string;
  subtitle?: string;
  /** What this section is for, shown on hover. A marker is rendered beside the
   *  title so a reader can tell a hint exists rather than having to hover
   *  every heading to find out. */
  hint?: string;
  children: any;
}) {
  const [open, setOpen] = useState(true);
  return (
    <section className="panel-section">
      <h4 onClick={() => setOpen(!open)}>
        <span className="chev">{open ? "▾" : "▸"}</span> {title}
        <Hint text={hint} />
        {subtitle && <em>{subtitle}</em>}
      </h4>
      {open && <div className="section-body">{children}</div>}
    </section>
  );
}

// A collapsible element card: header row (chevron + caller-supplied content) and
// a body shown only when expanded. Selecting the element on the map (``selected``)
// highlights it, auto-expands it, and scrolls it into view; clicking the chevron
// toggles and selects it (so panel <-> map selection stay in sync).
export function CollapsibleCard({
  selected = false,
  onSelect,
  header,
  children,
}: {
  selected?: boolean;
  onSelect?: () => void;
  header: ReactNode;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (selected) {
      setOpen(true);
      ref.current?.scrollIntoView({ block: "nearest" });
    }
  }, [selected]);
  return (
    <div ref={ref} className={selected ? "card selected" : "card"}>
      <div className="row between">
        <button
          className="chev-btn"
          title={open ? "collapse" : "expand"}
          onClick={() => {
            setOpen((o) => !o);
            onSelect?.();
          }}
        >
          {open ? "▾" : "▸"}
        </button>
        {header}
      </div>
      {open && <div className="card-body">{children}</div>}
    </div>
  );
}

export function EyeToggle({ hidden, onToggle }: { hidden: boolean; onToggle: () => void }) {
  return (
    <button
      className={`eye-toggle ${hidden ? "off" : ""}`}
      title={hidden ? "hidden from map — click to show" : "shown on map — click to hide"}
      onClick={(e) => {
        e.stopPropagation();
        onToggle();
      }}
    >
      {hidden ? "⊘" : "👁"}
    </button>
  );
}

// View-only lock (never touches the spec): when locked, an element shows no map
// edit handles and can't be deleted from the map, guarding it from accidental
// drags/edits while still selectable for inspection.
export function LockToggle({ locked, onToggle }: { locked: boolean; onToggle: () => void }) {
  return (
    <button
      className={`lock-toggle ${locked ? "on" : ""}`}
      title={locked ? "locked — click to allow map edits" : "unlocked — click to lock map edits"}
      onClick={(e) => {
        e.stopPropagation();
        onToggle();
      }}
    >
      {locked ? "🔒" : "🔓"}
    </button>
  );
}
