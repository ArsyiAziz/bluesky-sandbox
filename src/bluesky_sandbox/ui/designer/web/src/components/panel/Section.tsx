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
  defaultOpen = true,
  children,
}: {
  title: string;
  defaultOpen?: boolean;
  subtitle?: string;
  /** What this section is for, shown on hover. A marker is rendered beside the
   *  title so a reader can tell a hint exists rather than having to hover
   *  every heading to find out. */
  hint?: string;
  children: any;
}) {
  const [open, setOpen] = useState(defaultOpen);
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
      <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
        {hidden ? (
          <path
            fill="currentColor"
            d="M12 7a5 5 0 0 1 5 5c0 .65-.13 1.26-.36 1.83l2.92 2.92A11.8 11.8 0 0 0 23 12c-1.73-4.39-6-7.5-11-7.5-1.4 0-2.74.25-3.98.7l2.16 2.16A4.9 4.9 0 0 1 12 7zM2 4.27l2.28 2.28.46.46A11.8 11.8 0 0 0 1 12c1.73 4.39 6 7.5 11 7.5 1.55 0 3.03-.3 4.38-.84l.42.42L19.73 22 21 20.73 3.27 3 2 4.27zM7.53 9.8l1.55 1.55A2.8 2.8 0 0 0 9 12a3 3 0 0 0 3 3c.22 0 .44-.03.65-.08l1.55 1.55A5 5 0 0 1 7 12c0-.79.2-1.53.53-2.2z"
          />
        ) : (
          <path
            fill="currentColor"
            d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17a5 5 0 1 1 0-10 5 5 0 0 1 0 10zm0-8a3 3 0 1 0 0 6 3 3 0 0 0 0-6z"
          />
        )}
      </svg>
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
      <svg viewBox="0 0 24 24" width="13" height="13" aria-hidden="true">
        <path
          fill="currentColor"
          d={
            locked
              ? "M18 8h-1V6a5 5 0 0 0-10 0v2H6a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V10a2 2 0 0 0-2-2zM9 6a3 3 0 0 1 6 0v2H9V6zm3 11a2 2 0 1 1 0-4 2 2 0 0 1 0 4z"
              : "M18 8h-1V6a5 5 0 0 0-9.9-1l1.94.5A3 3 0 0 1 15 6v2H6a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V10a2 2 0 0 0-2-2zm-6 9a2 2 0 1 1 0-4 2 2 0 0 1 0 4z"
          }
        />
      </svg>
    </button>
  );
}
