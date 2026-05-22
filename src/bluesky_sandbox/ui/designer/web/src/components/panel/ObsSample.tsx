// Inspect the *labeled* observation + a sampled action the policy receives, for
// the drawn episode in the Sampling tab. Building the env runs BlueSky in a
// one-shot subprocess, so it's button-triggered (not live like the preview).
// Makes the obs layout explicit - field order, normalizer-expanded columns
// (e.g. a circular angle -> cos/sin), and raw vs normalized values.
import { useEffect, useRef, useState } from "react";
import { api, type SampleResult, type SpecDict } from "../../api";

const fmt = (v: number) =>
  Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01) ? v.toExponential(2) : v.toFixed(3);

export function ObsSample({ spec, seed }: { spec: SpecDict; seed: number }) {
  const [result, setResult] = useState<SampleResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // Building the env is heavy (BlueSky subprocess), so we don't sample on mount
  // or on every spec edit. Once "armed" by a first click, reseeds (and explicit
  // resamples via the nonce) reload with a loading state, like the preview.
  const [armed, setArmed] = useState(false);
  const [nonce, setNonce] = useState(0);
  const specRef = useRef(spec);
  specRef.current = spec;

  useEffect(() => {
    if (!armed) return;
    let cancelled = false;
    setLoading(true);
    setError("");
    api
      .sample(specRef.current, seed, 3, 25)
      .then((r) => !cancelled && setResult(r))
      .catch((e) => !cancelled && setError(String(e?.message ?? e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true; // ignore a stale reseed's result
    };
  }, [seed, armed, nonce]);

  const run = () => (armed ? setNonce((n) => n + 1) : setArmed(true));

  return (
    <div className="obs-sample">
      <div className="row between">
        <span className="muted small">obs + action for this episode (seed {seed})</span>
        <button className="link" onClick={run} disabled={loading}>
          {loading ? "sampling…" : result ? "resample" : "sample obs"}
        </button>
      </div>
      {error && <div className="error-text small">{error}</div>}

      {result?.agents.map((a) => (
        <div className="sample-agent" key={a.acid}>
          <div className="sub-label">{a.acid} · {a.n_intruders} intruder(s)</div>

          <div className="muted small">ownship</div>
          <table className="sample-table">
            <tbody>
              {a.ownship.map((f, i) => (
                <tr key={i}><td>{f.name}</td><td className="num">{fmt(f.value)}</td></tr>
              ))}
            </tbody>
          </table>

          <div className="muted small">action (sampled)</div>
          <table className="sample-table">
            <tbody>
              {a.action.map((f, i) => (
                <tr key={i}><td>{f.name}</td><td className="num">{fmt(f.value)}</td></tr>
              ))}
            </tbody>
          </table>

          {a.intruder_fields.length > 0 && a.intruders.length > 0 && (
            <>
              <div className="muted small">intruders (row each; columns = fields)</div>
              <div className="obs-sample-scroll">
                <table className="sample-table">
                  <thead>
                    <tr>{a.intruder_fields.map((n, i) => <th key={i}>{n}</th>)}</tr>
                  </thead>
                  <tbody>
                    {a.intruders.map((row, ri) => (
                      <tr key={ri}>{row.map((v, ci) => <td key={ci} className="num">{fmt(v)}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      ))}
    </div>
  );
}
