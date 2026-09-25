// A draggable divider between two panes, and the size it sets - remembered per
// browser, clamped to what the window allows, reset by double-clicking.
import { useEffect, useState, type PointerEvent as ReactPointerEvent } from "react";

const read = (key: string): number | null => {
  try {
    const v = Number(window.localStorage.getItem(key));
    return Number.isFinite(v) && v > 0 ? v : null;
  } catch {
    return null;
  }
};

const write = (key: string, value: number) => {
  try {
    window.localStorage.setItem(key, String(value));
  } catch {
    /* storage may be blocked: the size just is not remembered */
  }
};

/** A size stored under `key`, kept within [min, max()] as the window changes. */
export function useStoredSize(key: string, initial: number, min: number, max: () => number) {
  const clamp = (v: number) => Math.min(Math.max(min, max()), Math.max(min, v));
  const [size, setSize] = useState(() => clamp(read(key) ?? initial));
  useEffect(() => write(key, size), [key, size]);
  useEffect(() => {
    const onResize = () => setSize((s) => clamp(s));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return [size, (v: number) => setSize(clamp(v)), () => setSize(clamp(initial))] as const;
}

/**
 * The handle. `onDrag` gets the pointer's position along the resize axis -
 * clientX for a vertical divider (`axis="x"`), clientY for a horizontal one.
 */
export function Resizer({
  axis,
  onDrag,
  onReset,
  label,
}: {
  axis: "x" | "y";
  onDrag: (pos: number) => void;
  onReset: () => void;
  label: string;
}) {
  const start = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    const target = e.currentTarget;
    target.setPointerCapture(e.pointerId);
    target.classList.add("dragging");
    const move = (ev: PointerEvent) => onDrag(axis === "x" ? ev.clientX : ev.clientY);
    const up = () => {
      target.classList.remove("dragging");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  return (
    <div
      className={`resizer resizer-${axis}`}
      role="separator"
      aria-orientation={axis === "x" ? "vertical" : "horizontal"}
      aria-label={label}
      title="Drag to resize · double-click to reset"
      onPointerDown={start}
      onDoubleClick={onReset}
    />
  );
}
