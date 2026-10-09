// The Spaces tab's field report: the design's episodes for the seeds given,
// what a step costs at each count of aircraft - phase by phase and field by
// field - and each field's values against its bounds. See checks.report.
import { Fragment, useEffect, useMemo, useState } from "react";
import { api, type FieldReportResult, type ReportSpread, type SpecDict } from "../../api";
import { niceTicks } from "../MdpTab";
import { Spinner } from "../Spinner";

type Report = FieldReportResult;
type Bounds = Report["bounds"][number];

// Each phase of a step as the report names it, in the order a step runs them.
const PHASES: [key: string, label: string][] = [
  ["actions", "actions"],
  ["simulation", "BlueSky simulation"],
  ["bookkeeping", "bookkeeping"],
  ["lifecycle", "aircraft lifecycle"],
  ["fields", "fields"],
  ["packing", "normalizing and packing"],
  ["hooks", "hooks"],
  ["wrappers", "wrappers"],
  ["other", "other"],
];

// The parts of a field list as the Spaces tab titles them.
const PARTS: Record<string, string> = {
  obs_fields: "ownship",
  intruder_obs_fields: "intruder",
  critic_obs_fields: "critic ownship",
  critic_intruder_obs_fields: "critic intruder",
  state_fields: "state ownship",
  intruder_state_fields: "state intruder",
};

// The last report and what it sampled: kept while the design is unchanged.
export type ReportRun = { seeds: [number, number]; steps: number; report: Report };

export function FieldReport({
  spec,
  last,
  onReport,
  onClose,
  onOpenField,
}: {
  spec: SpecDict;
  last: ReportRun | null;
  onReport: (run: ReportRun) => void;
  onClose: () => void;
  onOpenField: (list: string, entry: number) => void;
}) {
  const [tab, setTab] = useState<"cost" | "bounds" | "checks">("cost");
  // What to sample: in the inputs, for the designer to set.
  const [firstSeed, setFirstSeed] = useState(last?.seeds[0] ?? 0);
  const [lastSeed, setLastSeed] = useState(last?.seeds[1] ?? 4);
  const [steps, setSteps] = useState(last?.steps ?? 100);
  // A run asked for: none on opening when the last one still holds.
  const [run, setRun] = useState(last ? 0 : 1);
  const [report, setReport] = useState<Report | null>(last?.report ?? null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!run) return;
    let live = true;
    const seeds: [number, number] = [firstSeed, Math.max(firstSeed, lastSeed)];
    const n = Math.max(1, steps);
    setLoading(true);
    setError(null);
    api
      .report(spec, seeds, n)
      .then((r) => {
        if (!live) return;
        setReport(r);
        onReport({ seeds, steps: n, report: r });
      })
      .catch((e) => live && setError(String(e?.message ?? e)))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
    // Run on opening and on asking again: not on each keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run]);

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal field-report" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-title">Field report</div>
            <div className="muted small">{spec.metadata?.name ?? "this design"}</div>
          </div>
          <div className="pane-tabs report-tabs" role="tablist">
            <button role="tab" aria-selected={tab === "cost"} className={tab === "cost" ? "on" : ""} onClick={() => setTab("cost")}>
              Cost
            </button>
            <button role="tab" aria-selected={tab === "bounds"} className={tab === "bounds" ? "on" : ""} onClick={() => setTab("bounds")}>
              Bounds
            </button>
            <button role="tab" aria-selected={tab === "checks"} className={tab === "checks" ? "on" : ""} onClick={() => setTab("checks")}>
              Checks{report && failures(report) ? <span className="bad"> · {failures(report)} ✗</span> : null}
            </button>
          </div>
          <button className="link" onClick={onClose}>
            done
          </button>
        </div>
        <div className="report-body">
          <div className="report-controls">
            <label className="numfield inline">
              <span>seeds</span>
              <input type="number" value={firstSeed} aria-label="first seed" onChange={(e) => setFirstSeed(Math.trunc(Number(e.target.value) || 0))} />
            </label>
            <label className="numfield inline">
              <span>to</span>
              <input type="number" value={lastSeed} aria-label="last seed" onChange={(e) => setLastSeed(Math.trunc(Number(e.target.value) || 0))} />
            </label>
            <label className="numfield inline">
              <span>steps each</span>
              <input type="number" min={1} value={steps} aria-label="steps per episode" onChange={(e) => setSteps(Math.max(1, Math.trunc(Number(e.target.value) || 1)))} />
            </label>
            <button onClick={() => setRun((n) => n + 1)} disabled={loading}>
              ↻ run
            </button>
          </div>
          {loading && (
            <div className="muted small">
              <Spinner label="flying the sampled episodes" /> flying seeds {firstSeed} to {Math.max(firstSeed, lastSeed)}, {steps} steps each…
            </div>
          )}
          {error && <div className="probe-error">{error}</div>}
          {report && !loading && tab === "cost" && <CostTab report={report} onOpen={onOpenField} />}
          {report && !loading && tab === "bounds" && <BoundsTab report={report} onOpen={onOpenField} />}
          {report && !loading && tab === "checks" && <ChecksTab report={report} seed={firstSeed} steps={steps} onOpen={onOpenField} />}
        </div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------- cost --

function CostTab({ report, onOpen }: { report: Report; onOpen: (list: string, entry: number) => void }) {
  const cost = report.cost;
  const last = cost.aircraft.length - 1;
  const [hover, setHover] = useState<number | null>(null);
  const at = hover ?? last;
  const total = cost.total.median[at] ?? 0;
  const phases = PHASES.filter(([key]) => cost.phases[key]).map(([key, label]) => ({
    key,
    label,
    ms: cost.phases[key].median[at] ?? 0,
  }));
  const fieldTotal = Object.values(cost.fields).reduce((sum, ts) => sum + (ts[at] ?? 0), 0);
  const rows = Object.entries(cost.fields)
    .map(([key, ts]) => ({ key, ms: ts[at], info: report.fields[key] }))
    .filter((r) => r.info)
    .sort((a, b) => (b.ms ?? -1) - (a.ms ?? -1));
  const slowest = rows[0]?.ms ?? 0;
  return (
    <>
      <div className="sub-label">processing time per step</div>
      <div className="report-cost-top">
        <CostChart report={report} at={at} hover={hover} onHover={setHover} />
        <div className="report-at">
          <div className="report-total">
            <span className="muted">
              total step · at {cost.aircraft[at]} aircraft
            </span>
            <span className="report-total-value">{ms(total)}</span>
            <span className="muted mono">
              {pct(cost.percentiles[0])} – {pct(cost.percentiles[1])}: {ms(cost.total.low[at])} – {ms(cost.total.high[at])}
            </span>
          </div>
          <div className="report-phase-bar" role="img" aria-label="the step, phase by phase">
            {phases.map((p) => (
              <span key={p.key} className={`phase-${p.key}`} style={{ width: `${total ? (p.ms / total) * 100 : 0}%` }} title={p.label} />
            ))}
          </div>
          <div className="report-phases">
            {[...phases]
              .sort((a, b) => b.ms - a.ms)
              .map((p) => (
                <Fragment key={p.key}>
                  <span className={`report-swatch phase-${p.key}`} />
                  <span className={total && p.ms / total < 0.05 ? "muted" : ""}>{p.label}</span>
                  <span className="mono">{ms(p.ms)}</span>
                  <span className="mono muted">{total ? `${Math.round((p.ms / total) * 100)}%` : ""}</span>
                </Fragment>
              ))}
          </div>
        </div>
      </div>
      <div className="report-table-box grow">
        <table className="report-table">
          <thead>
            <tr>
              <th className="num">#</th>
              <th>field</th>
              <th>part</th>
              <th>read as</th>
              <th className="num">time</th>
              <th>share of fields</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.key}>
                <td className="num muted">{i + 1}</td>
                <td>
                  <FieldLink info={r.info} onOpen={onOpen} />
                </td>
                <td className="muted">{PARTS[r.info.list] ?? r.info.list}</td>
                <td className={r.info.batched === false ? "warn" : "muted"}>{r.info.batched === false ? "one at a time" : "batched"}</td>
                <td className="num mono">{ms(r.ms)}</td>
                <td>
                  <div className="report-share">
                    <span className="report-share-track">
                      <span className={r.info.batched === false ? "warn-fill" : ""} style={{ width: `${slowest && r.ms ? Math.max(1, (r.ms / slowest) * 100) : 0}%` }} />
                    </span>
                    <span className="mono muted">{fieldTotal && r.ms != null ? `${((r.ms / fieldTotal) * 100).toFixed(1)}%` : "—"}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

// The total step, the fields and the simulation at each count, each with its
// spread; past the most the sampled episodes flew, dotted.
function CostChart({
  report,
  at,
  hover,
  onHover,
}: {
  report: Report;
  at: number;
  hover: number | null;
  onHover: (k: number | null) => void;
}) {
  const cost = report.cost;
  const W = 560;
  const H = 230;
  const M = { l: 56, r: 14, t: 10, b: 40 };
  const series: { key: string; spread: ReportSpread; cls: string }[] = [
    { key: "fields", spread: cost.phases.fields, cls: "phase-fields" },
    { key: "simulation", spread: cost.phases.simulation, cls: "phase-simulation" },
    { key: "total", spread: cost.total, cls: "report-total-line" },
  ].filter((s) => s.spread);
  const ys = series.flatMap((s) => [...s.spread.high, ...s.spread.median]).filter((v): v is number => v != null);
  const xMax = Math.max(...cost.aircraft);
  const yMax = Math.max(...ys, 0) || 1;
  const yTicks = niceTicks(0, yMax, 4);
  const top = Math.max(yMax, yTicks[yTicks.length - 1] ?? yMax);
  const sx = (n: number) => M.l + (n / (xMax || 1)) * (W - M.l - M.r);
  const sy = (v: number) => H - M.b - (v / top) * (H - M.t - M.b);
  // The counts the sampled episodes flew: solid; placed below them or topped
  // up above, dotted.
  const firstSampled = cost.sampled.indexOf(true);
  const lastSampled = cost.sampled.lastIndexOf(true);
  const segment = (values: (number | null)[], from: number, to: number) =>
    cost.aircraft
      .slice(from, to + 1)
      .map((n, k) => [n, values[from + k]] as const)
      .filter((p): p is readonly [number, number] => p[1] != null)
      .map(([n, v]) => `${sx(n).toFixed(1)},${sy(v).toFixed(1)}`)
      .join(" ");
  const area = (s: ReportSpread, from: number, to: number) => {
    const idx = cost.aircraft.map((_, k) => k).slice(from, to + 1).filter((k) => s.low[k] != null && s.high[k] != null);
    const up = idx.map((k) => `${sx(cost.aircraft[k]).toFixed(1)},${sy(s.high[k]!).toFixed(1)}`);
    const down = idx.reverse().map((k) => `${sx(cost.aircraft[k]).toFixed(1)},${sy(s.low[k]!).toFixed(1)}`);
    return [...up, ...down].join(" ");
  };
  const lastIdx = cost.aircraft.length - 1;
  const parts: { from: number; to: number; beyond: boolean }[] =
    firstSampled < 0
      ? [{ from: 0, to: lastIdx, beyond: true }]
      : [
          { from: 0, to: firstSampled, beyond: true },
          { from: firstSampled, to: lastSampled, beyond: false },
          { from: lastSampled, to: lastIdx, beyond: true },
        ].filter((p) => p.to > p.from || (!p.beyond && p.to === p.from));
  const lo = firstSampled >= 0 ? sx(cost.aircraft[firstSampled]) : 0;
  const hi = lastSampled >= 0 ? sx(cost.aircraft[lastSampled]) : 0;
  const roomy = hi - lo > 60;
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - box.left) / box.width) * W;
    let best = 0;
    cost.aircraft.forEach((n, k) => {
      if (Math.abs(sx(n) - x) < Math.abs(sx(cost.aircraft[best]) - x)) best = k;
    });
    if (best !== hover) onHover(best);
  };
  return (
    <svg className="report-chart" viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="processing time per step against aircraft in the air" onMouseMove={onMove} onMouseLeave={() => onHover(null)}>
      {yTicks.map((v) => (
        <g key={`y${v}`}>
          <line className="mdp-grid" x1={M.l} x2={W - M.r} y1={sy(v)} y2={sy(v)} />
          <text className="mdp-tick" x={M.l - 6} y={sy(v) + 3} textAnchor="end">
            {ms(v)}
          </text>
        </g>
      ))}
      {niceTicks(0, xMax, 5).map((n) => (
        <text key={`x${n}`} className="mdp-tick" x={sx(n)} y={H - M.b + 14} textAnchor="middle">
          {n}
        </text>
      ))}
      <line className="mdp-axis" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
      <line className="mdp-axis" x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} />
      <text className="mdp-tick" x={(M.l + W - M.r) / 2} y={H - 6} textAnchor="middle">
        aircraft in the air
      </text>
      <text className="mdp-tick" x={12} y={(M.t + H - M.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(M.t + H - M.b) / 2})`}>
        ms per step
      </text>
      {series.map((s) => (
        <g key={s.key} className={`report-series ${s.cls}`}>
          {parts.map((p) => (
            <Fragment key={`${p.from}-${p.to}`}>
              <polygon className={p.beyond ? "report-band beyond" : "report-band"} points={area(s.spread, p.from, p.to)} />
              <polyline className={p.beyond ? "report-line beyond" : "report-line"} points={segment(s.spread.median, p.from, p.to)} />
            </Fragment>
          ))}
        </g>
      ))}
      {firstSampled >= 0 && (firstSampled > 0 || lastSampled < lastIdx) && (
        <>
          {[firstSampled, lastSampled]
            .filter((k, i, all) => all.indexOf(k) === i)
            .map((k) => (
              <line key={`m${k}`} className="report-reached" x1={sx(cost.aircraft[k])} x2={sx(cost.aircraft[k])} y1={M.t} y2={H - M.b} />
            ))}
          <text className="mdp-tick" x={roomy ? (lo + hi) / 2 : hi + 4} y={M.t + 10} textAnchor={roomy ? "middle" : "start"}>
            {roomy ? "sampled" : "◂ sampled"}
          </text>
        </>
      )}
      {hover != null && (
        <g>
          <line className="report-hover" x1={sx(cost.aircraft[at])} x2={sx(cost.aircraft[at])} y1={M.t} y2={H - M.b} />
          {series.map((s) =>
            s.spread.median[at] == null ? null : (
              <circle key={s.key} className={`report-dot ${s.cls}`} cx={sx(cost.aircraft[at])} cy={sy(s.spread.median[at]!)} r={s.key === "total" ? 5 : 3} />
            ),
          )}
        </g>
      )}
    </svg>
  );
}

// ----------------------------------------------------------------- bounds --

function BoundsTab({ report, onOpen }: { report: Report; onOpen: (list: string, entry: number) => void }) {
  const rows = useMemo(
    () =>
      [...report.bounds]
        .filter((b) => b.samples > 0)
        .sort((a, b) => (b.below + b.above) / b.samples - (a.below + a.above) / a.samples),
    [report],
  );
  const [picked, setPicked] = useState<string | null>(rows[0]?.field ?? null);
  const sel = rows.find((r) => r.field === picked) ?? rows[0];
  return (
    <>
      <div className="report-table-box capped">
        <table className="report-table">
          <thead>
            <tr>
              <th>field</th>
              <th>bounds</th>
              <th>seen</th>
              <th className="num">below</th>
              <th className="num">above</th>
              <th className="num">furthest out</th>
              <th>normalizer</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((b) => {
              const out = b.below + b.above > 0;
              return (
                <tr key={b.field} className={b.field === sel?.field ? "picked" : ""}>
                  <td>
                    <button className="report-row-pick mono" aria-pressed={b.field === sel?.field} onClick={() => setPicked(b.field)}>
                      {b.field === sel?.field ? "▾" : "▸"} {b.name}
                    </button>
                  </td>
                  <td className="mono muted">{b.fixed_bounds ? `${num(b.fixed_bounds[0])} – ${num(b.fixed_bounds[1])}` : "each aircraft's"}</td>
                  <td className="mono">
                    {num(b.min)} – {num(b.max)}
                  </td>
                  <td className={`num mono ${b.below ? "warn" : "muted"}`}>{share(b.below, b.samples)}</td>
                  <td className={`num mono ${b.above ? "warn" : "muted"}`}>{share(b.above, b.samples)}</td>
                  <td className="num mono">{furthest(b)}</td>
                  <td className={out && b.clips ? "warn" : "muted"}>
                    {b.normalizer ? b.normalizer.replace(/Normalizer$/, "") + (b.clips ? ", clipped" : "") : "raw"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {sel && <Histogram tally={sel} report={report} onOpen={onOpen} />}
    </>
  );
}

function Histogram({ tally, report, onOpen }: { tally: Bounds; report: Report; onOpen: (list: string, entry: number) => void }) {
  const [hover, setHover] = useState<number | null>(null);
  const bins = Object.entries(tally.histogram)
    .map(([k, n]) => [Number(k), n] as const)
    .sort((a, b) => a[0] - b[0]);
  const W = 620;
  const H = 210;
  const M = { l: 44, r: 10, t: 22, b: 40 };
  const fixed = tally.fixed_bounds;
  // A bin's edges: in the field's unit where every aircraft has the same
  // bounds, else as a share of each aircraft's own.
  const edge = (k: number) => (fixed ? fixed[0] + (k / tally.bins) * (fixed[1] - fixed[0]) : k / tally.bins);
  const lowK = Math.min(0, ...bins.map(([k]) => k));
  const highK = Math.max(tally.bins, ...bins.map(([k]) => k + 1));
  const x0 = edge(lowK);
  const x1 = edge(highK);
  const sx = (v: number) => M.l + ((v - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
  const shares = bins.map(([k, n]) => [k, (n / tally.samples) * 100] as const);
  const yTicks = niceTicks(0, Math.max(...shares.map(([, s]) => s), 1), 4);
  const yTop = yTicks[yTicks.length - 1] || 1;
  const sy = (s: number) => H - M.b - (s / yTop) * (H - M.t - M.b);
  const hovered = shares.find(([k]) => k === hover);
  const unit = fixed ? "" : " of its bounds";
  const label = (v: number) => (fixed ? num(v) : `${Math.round(v * 100)}%`);
  // The end bins hold all that is further out than the bounds' width again.
  const below = -tally.bins - 1;
  const above = 2 * tally.bins;
  const binText = (k: number) =>
    k === below
      ? `below ${label(edge(k + 1))}${unit}`
      : k === above
        ? `above ${label(edge(k))}${unit}`
        : `${label(edge(k))} – ${label(edge(k + 1))}${unit}`;
  const inside = tally.samples - tally.below - tally.above;
  const info = report.fields[tally.field];
  return (
    <div className="report-hist">
      <div className="report-hist-head">
        <span className="sub-label inline">{tally.name} · raw values seen</span>
        <span className={`mono small ${hovered ? "" : "muted"}`}>
          {hovered ? `${binText(hovered[0])} · ${hovered[1].toFixed(1)}% of samples` : "hover a bar to read it"}
        </span>
      </div>
      <div className="report-hist-body">
        <svg className="report-chart" viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="raw values against the bounds" onMouseLeave={() => setHover(null)}>
          {yTicks.map((s) => (
            <g key={`y${s}`}>
              <line className="mdp-grid" x1={M.l} x2={W - M.r} y1={sy(s)} y2={sy(s)} />
              <text className="mdp-tick" x={M.l - 6} y={sy(s) + 3} textAnchor="end">
                {Number(s.toPrecision(3))}%
              </text>
            </g>
          ))}
          {shares.map(([k, s]) => (
            <rect
              key={k}
              className={k < 0 || k >= tally.bins ? "report-bin out" : "report-bin"}
              opacity={hover == null || hover === k ? 1 : 0.45}
              x={sx(edge(k)) + 1}
              y={sy(s)}
              width={Math.max(1, sx(edge(k + 1)) - sx(edge(k)) - 2)}
              height={H - M.b - sy(s)}
              onMouseEnter={() => setHover(k)}
            />
          ))}
          {shares
            .filter(([k]) => k === below || k === above)
            .map(([k, s]) => (
              <text key={`o${k}`} className="mdp-tick" x={sx(edge(k + 0.5))} y={sy(s) - 4} textAnchor={k === below ? "start" : "end"}>
                {k === below ? `< ${label(edge(k + 1))}` : `> ${label(edge(k))}`}
              </text>
            ))}
          {[0, tally.bins].map((k) => (
            <g key={`b${k}`}>
              <line className="report-bound" x1={sx(edge(k))} x2={sx(edge(k))} y1={M.t - 8} y2={H - M.b} />
              <text className="mdp-tick" x={sx(edge(k)) + 4} y={M.t - 10}>
                {k === 0 ? "low" : "high"}
              </text>
            </g>
          ))}
          {niceTicks(edge(Math.max(lowK, below + 1)), edge(Math.min(highK, above)), 5).map((v) => (
            <text key={`x${v}`} className="mdp-tick" x={sx(v)} y={H - M.b + 14} textAnchor="middle">
              {label(v)}
            </text>
          ))}
          <line className="mdp-axis" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
          <line className="mdp-axis" x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} />
          <text className="mdp-tick" x={(M.l + W - M.r) / 2} y={H - 6} textAnchor="middle">
            {fixed ? "raw" : "raw, as a share of each aircraft's bounds"}
          </text>
          <text className="mdp-tick" x={12} y={(M.t + H - M.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(M.t + H - M.b) / 2})`}>
            share of samples
          </text>
        </svg>
        <div className="report-hist-stats">
          <span className="muted">inside</span>
          <span className="mono">{share(inside, tally.samples)}</span>
          <span className="muted">below bound</span>
          <span className="mono">{share(tally.below, tally.samples)}</span>
          <span className="muted">above bound</span>
          <span className="mono">{share(tally.above, tally.samples)}</span>
          <span className="muted sep">normalized seen</span>
          <span className="mono sep">
            {tally.normalized_min == null ? "—" : `${num(tally.normalized_min)} – ${num(tally.normalized_max)}`}
          </span>
          <span className="muted">clipped</span>
          <span className="mono">{!tally.normalizer ? "—" : tally.clips ? share(tally.below + tally.above, tally.samples) : "not clipped"}</span>
        </div>
      </div>
      <div className="report-hist-foot">
        <span className="report-legend">
          <span className="report-bin-swatch" /> inside
        </span>
        <span className="report-legend">
          <span className="report-bin-swatch out" /> outside
        </span>
        {info && <FieldLink info={info} onOpen={onOpen} label="open in its Probe tab" />}
      </div>
    </div>
  );
}

// ----------------------------------------------------------------- checks --

// What fails a field or an action: the count the tab shows.
function failures(report: Report): number {
  const checks = report.checks;
  return (
    checks.episode.length +
    checks.fields.filter((f) => f.findings.length).length +
    checks.actions.filter((a) => a.agrees === false).length
  );
}

function ChecksTab({
  report,
  seed,
  steps,
  onOpen,
}: {
  report: Report;
  seed: number;
  steps: number;
  onOpen: (list: string, entry: number) => void;
}) {
  const checks = report.checks;
  // Each field: what fails it first, then what is worth knowing, then what holds.
  const rank = (f: { findings: string[]; notes: string[] }) => (f.findings.length ? 0 : f.notes.length ? 1 : 2);
  const fields = [...checks.fields].sort((a, b) => rank(a) - rank(b));
  const actionRank = (a: Report["checks"]["actions"][number]) => (a.agrees === false ? 0 : a.error ? 1 : 2);
  const actions = [...checks.actions].sort((a, b) => actionRank(a) - actionRank(b));
  return (
    <>
      <div className="muted small">
        seed {seed} · {steps} steps · flown twice
      </div>
      {checks.episode.map((f) => (
        <div key={f} className="probe-error">
          {f}
        </div>
      ))}
      <div className="report-table-box grow">
        <table className="report-table">
          <thead>
            <tr>
              <th>field</th>
              <th>part</th>
              <th>result</th>
              <th>what it found</th>
            </tr>
          </thead>
          <tbody>
            {fields.map((f) => {
              const info = report.fields[f.field];
              const said = f.findings.length ? f.findings : f.notes;
              return (
                <tr key={f.field}>
                  <td>{info ? <FieldLink info={info} onOpen={onOpen} /> : <span className="mono">{f.field}</span>}</td>
                  <td className="muted">{info ? PARTS[info.list] ?? info.list : ""}</td>
                  <td className={f.findings.length ? "bad" : f.notes.length ? "warn" : "ok"}>
                    {f.findings.length ? "✗ fails" : f.notes.length ? "⚠ note" : "✓ holds"}
                  </td>
                  <td className="report-found">
                    {said.map((line) => (
                      <div key={line}>{line}</div>
                    ))}
                  </td>
                </tr>
              );
            })}
            {actions.map((a) => {
              const info = report.fields[a.field];
              return (
                <tr key={a.field}>
                  <td>{info ? <FieldLink info={info} onOpen={onOpen} /> : <span className="mono">{a.name}</span>}</td>
                  <td className="muted">action</td>
                  <td className={a.agrees === false ? "bad" : a.error ? "muted" : "ok"}>
                    {a.agrees === false ? "✗ fails" : a.error ? "not probed" : a.agrees ? "✓ holds" : "—"}
                  </td>
                  <td className="report-found">
                    {a.error ? (
                      <div>{a.error}</div>
                    ) : a.agrees === false ? (
                      <div>
                        commands {num(a.actual)}, states {num(a.expected)}
                      </div>
                    ) : (
                      <div className="mono muted">{a.command.join("; ")}</div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

// ------------------------------------------------------------------ parts --

function FieldLink({
  info,
  onOpen,
  label,
}: {
  info: { name: string; list: string; entry: number | null };
  onOpen: (list: string, entry: number) => void;
  label?: string;
}) {
  if (info.entry == null) return <span className="mono">{label ?? info.name}</span>;
  return (
    <button className={label ? "link" : "link mono report-field"} onClick={() => onOpen(info.list, info.entry!)}>
      {label ?? info.name}
    </button>
  );
}

function ms(x: number | null | undefined): string {
  if (x == null) return "—";
  if (x === 0) return "0";
  if (x < 1) return `${Number((x * 1000).toPrecision(3))} µs`;
  return `${Number(x.toPrecision(3))} ms`;
}

function num(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return "—";
  return Number(x.toPrecision(4)).toLocaleString("en-US");
}

function share(n: number, of: number): string {
  if (!of || !n) return "0";
  return `${((n / of) * 100).toFixed(1)}%`;
}

// A percentile as an ordinal: 10th, 90th, 1st, 2nd, 3rd.
function pct(p: number): string {
  const n = Math.round(p);
  const tail = n % 100 >= 11 && n % 100 <= 13 ? "th" : ({ 1: "st", 2: "nd", 3: "rd" } as Record<number, string>)[n % 10] ?? "th";
  return `${n}${tail}`;
}

// How far past its bounds a field went, where every aircraft has the same.
function furthest(b: Bounds): string {
  if (!b.below && !b.above) return "—";
  if (!b.fixed_bounds || b.min == null || b.max == null) return "—";
  const under = b.below ? b.fixed_bounds[0] - b.min : 0;
  const over = b.above ? b.max - b.fixed_bounds[1] : 0;
  return over >= under ? `+${num(over)}` : `−${num(under)}`;
}
