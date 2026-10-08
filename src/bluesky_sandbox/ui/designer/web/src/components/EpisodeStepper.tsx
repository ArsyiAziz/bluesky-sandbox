// Which episode every view shows: ‹ n ›. Each number draws everything sampled
// per episode - shapes, placements, aircraft, wind - and the same number always
// draws the same episode (it is the seed).
import { useEffect, useState } from "react";
import { useEpisode } from "../episode";
import { Spinner } from "./Spinner";

// ``label``: name it "episode" - off where a heading already does.
export function EpisodeStepper({ busy = false, label = true }: { busy?: boolean; label?: boolean }) {
  const { seed, setSeed } = useEpisode();
  const [text, setText] = useState(String(seed));
  useEffect(() => setText(String(seed)), [seed]);
  const commit = () => {
    const n = Math.floor(Number(text));
    if (Number.isFinite(n) && n >= 0) setSeed(n);
    else setText(String(seed));
  };
  return (
    <span
      className="episode-stepper"
      role="group"
      aria-label="Episode"
      title="Draw another episode: everything sampled per episode (shapes, placements, aircraft, wind) is drawn again. The same number always gives the same episode."
    >
      {label && <span className="episode-label">episode</span>}
      <span className="episode-controls">
        <button
          type="button"
          disabled={busy || seed === 0}
          onClick={() => setSeed((s) => Math.max(0, s - 1))}
          aria-label="Previous episode"
        >
          ‹
        </button>
        <input
          inputMode="numeric"
          aria-label="Episode number"
          value={text}
          onChange={(e) => setText(e.target.value.replace(/[^0-9]/g, ""))}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") commit();
            else if (e.key === "ArrowUp") {
              e.preventDefault();
              setSeed((s) => s + 1);
            } else if (e.key === "ArrowDown") {
              e.preventDefault();
              setSeed((s) => Math.max(0, s - 1));
            }
          }}
          style={{ width: `${Math.max(2, text.length) + 1}ch` }}
        />
        <button type="button" disabled={busy} onClick={() => setSeed((s) => s + 1)} aria-label="Next episode">
          ›
        </button>
      </span>
      {busy && <Spinner label="drawing this episode" />}
      {seed !== 0 && (
        <button type="button" className="link episode-zero" onClick={() => setSeed(0)} title="Back to episode 0">
          ↺ 0
        </button>
      )}
    </span>
  );
}
