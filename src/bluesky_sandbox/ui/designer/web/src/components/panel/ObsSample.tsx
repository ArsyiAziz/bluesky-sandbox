// What the picked aircraft of the sampled episode observes as it spawns: each
// field's raw value and the columns the policy sees, part by part, and an
// action drawn from its space. The episode is the map's (same seed), picked in
// the aircraft list above; it runs the environment, so it takes a few seconds.
import { useState } from "react";
import type { SampleAgent, SampleField, SpecDict } from "../../api";
import { useEpisode, useEpisodeSample } from "../../episode";

function num(v: unknown): string {
  if (v === null || v === undefined) return "–";
  if (Array.isArray(v)) return v.map(num).join(", ");
  const x = Number(v);
  if (!Number.isFinite(x)) return String(v);
  const abs = Math.abs(x);
  if (abs >= 1e5 || (abs !== 0 && abs < 1e-3)) return x.toExponential(2);
  if (abs >= 100) return Math.round(x).toLocaleString("en-US");
  return x.toFixed(abs >= 10 ? 1 : 2);
}

const range = (f: SampleField) =>
  f.low === null || f.high === null ? "" : `range for this aircraft: [${num(f.low)}, ${num(f.high)}] ${f.unit}`;

export function ObsSample({ spec }: { spec: SpecDict }) {
  const { pick } = useEpisode();
  const { result, loading, error } = useEpisodeSample(spec);
  const [part, setPart] = useState("ownship");
  const [row, setRow] = useState(0);
  const agent: SampleAgent | undefined = result?.agents[0];

  const head = loading
    ? `stepping to ${pick ? `${pick.acid ?? "the aircraft"} at t+${Math.round(pick.at_s)} s` : "the first aircraft"}…`
    : agent
      ? `${agent.acid} · ${agent.type} · at t = ${Math.round(result!.sim_time_s)} s`
      : "";
  if (error) return <div className="error-text small">{error}</div>;
  if (!agent)
    return <p className="muted small">{loading ? head : "No aircraft is up in this episode yet."}</p>;

  const parts = [...Object.keys(agent.parts), "action"];
  const shown = parts.includes(part) ? part : "ownship";
  const entry = agent.parts[shown];
  const acids = entry?.acids;
  const r = acids ? Math.min(row, Math.max(0, acids.length - 1)) : 0;

  return (
    <div className={loading ? "obs-sample loading" : "obs-sample"}>
      <div className="obs-sample-head">{head}</div>
      <div className="seg">
        {parts.map((p) => (
          <button key={p} className={p === shown ? "on" : ""} onClick={() => setPart(p)}>
            {p}
            {agent.parts[p]?.acids ? ` · ${agent.parts[p].acids!.length}` : ""}
          </button>
        ))}
      </div>

      {acids && acids.length > 0 && (
        <label className="obs-sample-row-pick">
          <span className="muted small">intruder</span>
          <select value={r} onChange={(e) => setRow(Number(e.target.value))}>
            {acids.map((a, i) => (
              <option key={i} value={i}>
                {i + 1}. {a}
              </option>
            ))}
          </select>
        </label>
      )}
      {acids && acids.length === 0 && <p className="muted small">No intruders in view.</p>}

      <div className="obs-grid">
        <span className="obs-grid-h">field</span>
        <span className="obs-grid-h num">raw</span>
        <span className="obs-grid-h num">policy sees</span>
        {shown === "action"
          ? agent.action.map((f) => (
              <Row key={f.name} name={f.name} raw="" seen={num(f.value)} title="drawn from the action space" />
            ))
          : (!acids || acids.length > 0) &&
            entry.fields.map((f) => (
              <Row
                key={f.name}
                name={f.lag ? `└ lag ${f.lag}` : f.name}
                lag={Boolean(f.lag)}
                raw={`${num(acids ? f.raw?.[r] : f.raw)}${f.unit ? ` ${f.unit}` : ""}`}
                seen={num(acids ? f.obs?.[r] : f.obs)}
                title={range(f)}
              />
            ))}
      </div>
    </div>
  );
}

function Row({
  name,
  raw,
  seen,
  title,
  lag = false,
}: {
  name: string;
  raw: string;
  seen: string;
  title: string;
  lag?: boolean;
}) {
  return (
    <>
      <span className={lag ? "obs-grid-name lag" : "obs-grid-name"} title={`${name}${title ? `\n${title}` : ""}`}>
        {name}
      </span>
      <span className="num">{raw}</span>
      <span className="num muted">{seen}</span>
    </>
  );
}
