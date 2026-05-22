// Live readout of one sampled episode: counts + per-aircraft alt/speed/type.
// The seed is shared with the map's reseed/reset, so both stay in sync.
import { useEffect, useState } from "react";
import { api, type PreviewResult, type SpecDict } from "../../api";

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

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .preview(spec, seed)
      .then((p) => !cancelled && (setPreview(p), setError(null)))
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [spec, seed]);

  if (error) return <div className="error-text">{error}</div>;
  if (!preview) return <p className="muted">…</p>;

  const ac = preview.sampled_aircraft;
  return (
    <div className="sampling">
      <div className="row between">
        <span>
          {ac.length} aircraft · max {preview.max_aircraft}
        </span>
        <span className="row">
          <button className="link" disabled={loading} onClick={() => onSeedChange(seed + 1)}>
            reseed ({seed})
          </button>
          <button className="link" disabled={loading || seed === 0} onClick={() => onSeedChange(0)}>
            reset
          </button>
        </span>
      </div>
      <table className="sample-table">
        <thead>
          <tr>
            <th>type</th>
            <th>alt ft</th>
            <th>spd kt</th>
            <th>t+s</th>
          </tr>
        </thead>
        <tbody>
          {ac.slice(0, 12).map((a, i) => (
            <tr key={i}>
              <td>{a.actype}</td>
              <td>{Math.round(a.alt_ft)}</td>
              <td>{Math.round(a.spd_kts)}</td>
              <td>{Math.round(a.spawn_time)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {ac.length > 12 && <div className="muted">+{ac.length - 12} more…</div>}
    </div>
  );
}
