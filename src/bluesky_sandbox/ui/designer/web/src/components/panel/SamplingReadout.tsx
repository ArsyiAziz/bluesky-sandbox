// The sampled episode's aircraft, in spawn order; picking one shows what it
// observes below. The seed is the map's reseed/reset, so both show one episode.
import { useEffect, useState } from "react";
import { api, type PreviewResult, type SpecDict } from "../../api";
import { useRefresh } from "../../refresh";
import { useEpisode } from "../../episode";

export function SamplingReadout({
  spec,
  seed,
  onSeedChange,
}: {
  spec: SpecDict;
  seed: number;
  onSeedChange: (seed: number) => void;
}) {
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { pick, setPick } = useEpisode();

  const refreshKey = useRefresh();
  useEffect(() => {
    let canceled = false;
    setLoading(true);
    api
      .preview(spec, seed)
      .then((p) => !canceled && (setPreview(p), setError(null)))
      .catch((e) => !canceled && setError(String(e)))
      .finally(() => !canceled && setLoading(false));
    return () => {
      canceled = true;
    };
  }, [spec, seed, refreshKey]);

  if (error) return <div className="error-text">{error}</div>;
  if (!preview) return <p className="muted">…</p>;

  // In spawn order, each with its place in the drawn list.
  const ac = preview.sampled_aircraft
    .map((a, index) => ({ ...a, index }))
    .sort((x, y) => x.spawn_time - y.spawn_time);
  const picked = pick?.index ?? ac[0]?.index;
  return (
    <div className={loading ? "sampling loading" : "sampling"}>
      <div className="sampling-head">
        <span>
          {ac.length} aircraft <span className="muted">· max {preview.max_aircraft} at once</span>
        </span>
        <span className="sampling-seed">
          <span className="muted">seed {seed}</span>
          <button disabled={loading} onClick={() => onSeedChange(seed + 1)} title="draw another episode">
            reseed
          </button>
          <button disabled={loading || seed === 0} onClick={() => onSeedChange(0)} title="back to seed 0">
            reset
          </button>
        </span>
      </div>
      <div className="sampling-list" role="listbox" aria-label="aircraft in this episode">
        <div className="sampling-row sampling-row-h">
          <span>spawns</span>
          <span>type</span>
          <span className="num">alt ft</span>
          <span className="num">spd kt</span>
        </div>
        {ac.map((a) => (
          <div
            key={a.index}
            role="option"
            aria-selected={a.index === picked}
            className={a.index === picked ? "sampling-row on" : "sampling-row"}
            onClick={() => setPick({ index: a.index, at_s: a.spawn_time, type: a.actype })}
            title="see what this aircraft observes as it spawns"
          >
            <span>t+{Math.round(a.spawn_time)} s</span>
            <span>{a.actype}</span>
            <span className="num">{Math.round(a.alt_ft).toLocaleString("en-US")}</span>
            <span className="num">{Math.round(a.spd_kts)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
