// A collapsible sub-group inside the inspector: a small header that toggles a
// body of related fields. Lets a dense element editor (waypoint, spawn region)
// be split into Position / Constraints / Appearance / Advanced groups so only
// the part you're working on is open at once.
import { useState } from "react";
import type { ReactNode } from "react";

export function FieldGroup({
  title,
  hint,
  defaultOpen = false,
  children,
}: {
  title: string;
  hint?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={open ? "field-group open" : "field-group"}>
      <button type="button" className="field-group-head" onClick={() => setOpen((o) => !o)}>
        <span className="chev">{open ? "▾" : "▸"}</span>
        <span className="field-group-title">{title}</span>
        {hint && <span className="field-group-hint">{hint}</span>}
      </button>
      {open && <div className="field-group-body">{children}</div>}
    </div>
  );
}
