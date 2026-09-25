// Searchable, category-grouped picker for observation/action fields. Replaces
// the flat native <select>: filters by name/doc, groups by the module each field
// is defined in (the catalog's `category`), and shows each field's docstring
// inline, tagged with its quantity (obs) or control axis (action).
import { useEffect, useMemo, useRef, useState } from "react";
import type { FieldOption } from "./FieldList";

function metaLabel(value: any): string {
  if (value == null) return "";
  if (Array.isArray(value)) return value.map(String).join(", ");
  return String(value);
}

// What an option measures or commands: its quantity (obs) or control axis
// (action). Searchable, and shown as a tag unless its group already says it.
function tagOf(option: FieldOption, kind: "obs" | "action"): string {
  const meta = option.profile?.meta ?? {};
  return metaLabel(kind === "action" ? meta.control_axis : meta.quantity);
}

function shownTag(option: FieldOption, kind: "obs" | "action"): string {
  const tag = tagOf(option, kind);
  return tag === option.category ? "" : tag;
}

export function FieldPicker({
  kind,
  placeholder,
  options,
  onAdd,
}: {
  kind: "obs" | "action";
  placeholder: string;
  options: FieldOption[];
  onAdd: (name: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const byCat = new Map<string, FieldOption[]>();
    const docs = new Map<string, string>();
    for (const o of options) {
      const cat = o.category || "other";
      const haystack = [o.name, o.doc, cat, tagOf(o, kind)].join(" ").toLowerCase();
      if (q && !haystack.includes(q)) continue;
      (byCat.get(cat) ?? byCat.set(cat, []).get(cat)!).push(o);
      if (o.category_doc) docs.set(cat, o.category_doc);
    }
    return [...byCat.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([cat, opts]) => ({
        cat,
        doc: docs.get(cat) ?? "",
        opts: opts.sort((x, y) => x.name.localeCompare(y.name)),
      }));
  }, [options, query, kind]);

  // Flat order matches what's rendered, so arrow-key navigation lines up.
  const flat = useMemo(() => groups.flatMap((g) => g.opts), [groups]);

  useEffect(() => setActive(0), [query, open]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const choose = (name: string) => {
    onAdd(name);
    setQuery("");
    setOpen(false);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setActive((a) => Math.min(a + 1, flat.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === "Enter") {
      if (!open || !flat[active]) return;
      e.preventDefault();
      choose(flat[active].name);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <div className="field-picker" ref={rootRef}>
      <input
        className="field-picker-input"
        value={query}
        placeholder={placeholder}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
      />
      {open && (
        <div className="field-picker-menu">
          {flat.length === 0 ? (
            <div className="field-picker-empty muted small">no matches</div>
          ) : (
            groups.map((g) => (
              <div className="field-picker-group" key={g.cat}>
                <div className="field-picker-group-label" title={g.doc}>
                  {g.cat}
                </div>
                {g.opts.map((o) => {
                  const idx = flat.indexOf(o);
                  return (
                    <button
                      type="button"
                      key={o.name}
                      className={idx === active ? "field-picker-opt active" : "field-picker-opt"}
                      onMouseEnter={() => setActive(idx)}
                      onMouseDown={(e) => {
                        e.preventDefault();
                        choose(o.name);
                      }}
                    >
                      <span className="field-picker-opt-name">
                        {o.name}
                        {o.pair_only ? " (pair)" : ""}
                        {shownTag(o, kind) && (
                          <span className="field-picker-opt-tag">{shownTag(o, kind)}</span>
                        )}
                      </span>
                      {o.doc && <span className="field-picker-opt-doc muted small">{o.doc}</span>}
                    </button>
                  );
                })}
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
