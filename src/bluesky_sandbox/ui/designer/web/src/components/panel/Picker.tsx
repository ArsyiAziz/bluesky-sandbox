// Generic searchable, category-grouped dropdown — the same look & feel as the
// fields/actions picker, reused for every "choose one" control so the dropdowns
// are uniform across the designer. Two modes:
//   - select  (a `value` is passed): the trigger shows the current label.
//   - add     (no `value`): the trigger shows the placeholder and resets after a
//             pick (e.g. "+ override hook…").
// The menu is rendered in a portal with fixed positioning anchored to the
// trigger, so it never clips inside scrolling panels/modals.
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

export interface PickerOption {
  value: string;
  label?: string;
  /** Short qualifier shown beside the label in its own colour - a
   *  version, a state, a count. Kept generic: every option list here
   *  uses one Picker, so this is a Picker affordance, not a
   *  version-specific one. */
  badge?: string;
  description?: string;
  category?: string;
  disabled?: boolean;
}

export function Picker({
  value,
  placeholder,
  options,
  onChange,
  searchable = true,
  disabled = false,
  className = "",
  title,
}: {
  value?: string;
  placeholder: string;
  options: PickerOption[];
  onChange: (value: string) => void;
  searchable?: boolean;
  disabled?: boolean;
  className?: string;
  title?: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [pos, setPos] = useState<{ left: number; top: number; width: number } | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const isSelect = value !== undefined;

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const byCat = new Map<string, PickerOption[]>();
    for (const o of options) {
      const hay = `${o.value} ${o.label ?? ""} ${o.badge ?? ""} ${o.description ?? ""}`.toLowerCase();
      if (q && !hay.includes(q)) continue;
      const cat = o.category ?? "";
      (byCat.get(cat) ?? byCat.set(cat, []).get(cat)!).push(o);
    }
    return [...byCat.entries()]
      .sort(([a], [b]) => (a === "other" ? 1 : b === "other" ? -1 : a.localeCompare(b)))
      .map(([cat, opts]) => ({ cat, opts }));
  }, [options, query]);

  const flat = useMemo(() => groups.flatMap((g) => g.opts), [groups]);
  const selected = options.find((o) => o.value === value);

  const reposition = useCallback(() => {
    const el = triggerRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setPos({ left: r.left, top: r.bottom + 2, width: r.width });
  }, []);

  useEffect(() => setActive(0), [query, open]);
  useLayoutEffect(() => {
    if (open) reposition();
  }, [open, reposition]);
  useEffect(() => {
    if (open && searchable) inputRef.current?.focus();
  }, [open, searchable]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (triggerRef.current?.contains(t) || menuRef.current?.contains(t)) return;
      setOpen(false);
    };
    const onScrollOrResize = () => reposition();
    document.addEventListener("mousedown", onDown);
    // Capture-phase scroll catches scrolling of any ancestor container.
    window.addEventListener("scroll", onScrollOrResize, true);
    window.addEventListener("resize", onScrollOrResize);
    return () => {
      document.removeEventListener("mousedown", onDown);
      window.removeEventListener("scroll", onScrollOrResize, true);
      window.removeEventListener("resize", onScrollOrResize);
    };
  }, [open, reposition]);

  const choose = (opt: PickerOption) => {
    if (opt.disabled) return;
    onChange(opt.value);
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
      choose(flat[active]);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  const menu = open && pos && (
    <div
      ref={menuRef}
      className="picker-menu"
      style={{ left: pos.left, top: pos.top, minWidth: pos.width }}
    >
      {searchable && (
        <input
          ref={inputRef}
          className="picker-search"
          value={query}
          placeholder="filter…"
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
        />
      )}
      <div className="picker-scroll">
        {flat.length === 0 ? (
          <div className="picker-empty muted small">no matches</div>
        ) : (
          groups.map((g) => (
            <div className="picker-group" key={g.cat || "_"}>
              {g.cat && <div className="picker-group-label">{g.cat}</div>}
              {g.opts.map((o) => {
                const idx = flat.indexOf(o);
                const cls = [
                  "picker-opt",
                  idx === active ? "active" : "",
                  o.disabled ? "disabled" : "",
                  o.value === value ? "current" : "",
                ]
                  .filter(Boolean)
                  .join(" ");
                return (
                  <button
                    type="button"
                    key={o.value}
                    className={cls}
                    disabled={o.disabled}
                    onMouseEnter={() => setActive(idx)}
                    onMouseDown={(e) => {
                      e.preventDefault();
                      choose(o);
                    }}
                  >
                    <span className="picker-opt-name">
                      {o.label ?? o.value}
                      {o.badge && <span className="picker-opt-badge">{o.badge}</span>}
                    </span>
                    {o.description && (
                      <span className="picker-opt-doc muted small">{o.description}</span>
                    )}
                  </button>
                );
              })}
            </div>
          ))
        )}
      </div>
    </div>
  );

  return (
    <div className={`picker ${className}`} title={title}>
      <button
        type="button"
        ref={triggerRef}
        className={`picker-trigger ${open ? "open" : ""}`}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={onKeyDown}
      >
        <span className={isSelect && selected ? "picker-value" : "picker-placeholder"}>
          {isSelect ? (
            <>
              {selected?.label ?? selected?.value ?? placeholder}
              {selected?.badge && (
                <span className="picker-opt-badge">{selected.badge}</span>
              )}
            </>
          ) : (
            placeholder
          )}
        </span>
        <span className="picker-caret">▾</span>
      </button>
      {menu && createPortal(menu, document.body)}
    </div>
  );
}
