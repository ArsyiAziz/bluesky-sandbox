import { useCallback, useRef, useState } from "react";

/**
 * The ⓘ marker beside a heading or label, with its explanation on hover.
 *
 * The bubble is a `position: fixed` element measured and placed on hover,
 * rather than the CSS `::after` this started as. A pseudo-element cannot be
 * measured, so it can only ever open in one direction - which runs off-screen
 * for anything near the bottom or right of the panel. Fixed positioning also
 * escapes the sidebar's scroll clipping for free.
 */
export function Hint({ text }: { text?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [box, setBox] = useState<{ top: number; left: number } | null>(null);

  const show = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const W = 280;                    // must match max-width below
    const EST_H = 84;                 // enough for ~4 wrapped lines
    const M = 8;                      // keep off the very edge

    // Below by default; above when the bottom would overflow and there is more
    // room up top. Left-clamped so a marker near the right edge stays visible.
    const below = r.bottom + 6;
    const flip = below + EST_H > window.innerHeight - M && r.top > EST_H + M;
    setBox({
      top: flip ? Math.max(M, r.top - 6 - EST_H) : below,
      left: Math.min(Math.max(M, r.left), window.innerWidth - W - M),
    });
  }, []);

  if (!text) return null;
  return (
    <span
      ref={ref}
      className="hint-mark"
      /* No ``title``: the browser would paint its own tooltip a second after
         the bubble, so every hint appeared twice. ``aria-label`` carries the
         text for assistive tech without rendering anything. */
      aria-label={text}
      onMouseEnter={show}
      onMouseLeave={() => setBox(null)}
    >
      ⓘ
      {box && (
        <span className="hint-bubble" style={{ top: box.top, left: box.left }}>
          {text}
        </span>
      )}
    </span>
  );
}
