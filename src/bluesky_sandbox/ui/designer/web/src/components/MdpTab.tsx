// The design's spaces (ui/designer/mdp.py), part by part: the part's columns
// drawn as a strip colored by the normalizer each goes through, and a table of
// its fields - each lag listed under the field it lags - where a row opens to
// its normalizer's settings and mapping.
import { Fragment, useEffect, useMemo, useState } from "react";
import { api, type SampleResult, type SpecDict } from "../api";
import { normalizerColor } from "../normColors";
import { useEpisodeSample } from "../episode";
import { useRefresh } from "../refresh";

type Curve = { x: number[]; series: number[][]; x_label: string; y_label: string };
type Field = {
  name: string;
  class: string;
  doc: string;
  columns: [number, number];
  binary: boolean;
  lag: { steps: number; of: string | null; inner: string } | null;
  raw: { width: number; unit: string; low: number | null; high: number | null; per_aircraft: boolean };
  normalizer: { name: string; params: Record<string, unknown> } | null;
  output: { low: (number | null)[]; high: (number | null)[] };
  curve: Curve | null;
  curve_note?: string;
};
type Part = { part: string; rows?: string; width: number; fields: Field[] };
type Summary = {
  ok: boolean;
  error?: string;
  normalizers: string[];
  observation: Part[];
  action: { space: string; parts: Part[] };
};
type Role = "observation" | "action";

const colorOf = normalizerColor;

const shortName = (name: string) => name.replace(/Normalizer$/, "");
const span = (f: Field) => f.columns[1] - f.columns[0];
const columnsText = (f: Field) => (span(f) > 1 ? `${f.columns[0]}–${f.columns[1] - 1}` : String(f.columns[0]));

function number(v: number | null | undefined, open = "∞"): string {
  if (v === null || v === undefined) return open;
  if (Math.abs(v - Math.round(v)) < 1e-9) v = Math.round(v);
  const abs = Math.abs(v);
  if (abs !== 0 && (abs >= 1e7 || abs < 1e-3)) return v.toExponential(1);
  if (abs >= 1000) return Math.round(v).toLocaleString("en-US");
  return Number.isInteger(v) ? String(v) : v.toFixed(abs < 10 ? 2 : 1);
}

const range = (low: number | null, high: number | null) => `[${number(low, "−∞")}, ${number(high)}]`;
const rawText = (f: Field) =>
  f.binary
    ? "{0, 1}"
    : `${f.raw.per_aircraft ? "per aircraft" : range(f.raw.low, f.raw.high)}${f.raw.unit ? ` ${f.raw.unit}` : ""}`;
const outputText = (f: Field) =>
  f.binary
    ? "{0, 1}"
    : !f.normalizer && f.raw.per_aircraft
    ? "per aircraft"
    : f.output.low.length > 1
      ? `${f.output.low.length} × ${range(f.output.low[0], f.output.high[0])}`
      : range(f.output.low[0], f.output.high[0]);
const paramsText = (f: Field) =>
  f.normalizer
    ? Object.entries(f.normalizer.params)
        .map(([k, v]) => `${k}=${Array.isArray(v) ? `(${v.join(", ")})` : String(v)}`)
        .join(" · ")
    : "";

// Each field followed by its lags, the lags in step order; a lag whose field
// is not in the part keeps its own place.
function grouped(fields: Field[]): { field: Field; lags: Field[] }[] {
  const lagsOf = new Map<string, Field[]>();
  for (const f of fields) {
    if (f.lag?.of) lagsOf.set(f.lag.of, [...(lagsOf.get(f.lag.of) ?? []), f]);
  }
  return fields
    .filter((f) => !f.lag?.of)
    .map((field) => ({ field, lags: (lagsOf.get(field.name) ?? []).sort((a, b) => a.lag!.steps - b.lag!.steps) }));
}

export default function MdpTab({ spec }: { spec: SpecDict | null }) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const refreshKey = useRefresh();

  useEffect(() => {
    if (!spec) return;
    let canceled = false;
    const handle = setTimeout(() => {
      api
        .mdp(spec)
        .then((result) => !canceled && setSummary(result))
        .catch((e) => !canceled && setSummary({ ok: false, error: String(e) } as Summary));
    }, 300);
    return () => {
      canceled = true;
      clearTimeout(handle);
    };
  }, [spec, refreshKey]);

  if (!spec) return <div className="mdp-tab muted">The spec has a JSON error; fix it in the Code tab.</div>;
  if (!summary) return <div className="mdp-tab muted">Loading…</div>;
  if (!summary.ok)
    return (
      <div className="mdp-tab">
        <pre className="error-text">{summary.error}</pre>
      </div>
    );

  const every = [...summary.observation, ...summary.action.parts].flatMap((p) => p.fields);
  const used = new Set(every.map((f) => f.normalizer?.name ?? ""));
  const legend = [...summary.normalizers.filter((n) => used.has(n)), ...(used.has("") ? [""] : [])];

  return (
    <div className="mdp-tab">
      <div className="mdp-legend" aria-label="normalizers">
        {legend.map((name) => (
          <span key={name || "raw"}>
            <span className="mdp-dot" style={{ background: colorOf(summary.normalizers, name) }} />
            {name ? shortName(name) : "raw"}
          </span>
        ))}
      </div>
      <h3>observation</h3>
      {summary.observation.map((p) => (
        <PartView key={p.part} part={p} role="observation" normalizers={summary.normalizers} spec={spec} />
      ))}
      <h3>action · {summary.action.space}</h3>
      {summary.action.parts.map((p) => (
        <PartView key={p.part} part={p} role="action" normalizers={summary.normalizers} spec={spec} />
      ))}
    </div>
  );
}

function PartView({
  part,
  role,
  normalizers,
  spec,
}: {
  part: Part;
  role: Role;
  normalizers: string[];
  spec: SpecDict;
}) {
  const [hovered, setHovered] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const groups = useMemo(() => grouped(part.fields), [part]);
  // Hovering a field lights its lags too, in the strip and the table.
  const groupOf = useMemo(() => {
    const out = new Map<string, string>();
    for (const g of groups) for (const f of [g.field, ...g.lags]) out.set(f.name, g.field.name);
    return out;
  }, [groups]);
  const lit = (f: Field) => hovered !== null && groupOf.get(f.name) === groupOf.get(hovered);
  const hover = (f: Field) => ({ onMouseEnter: () => setHovered(f.name), onMouseLeave: () => setHovered(null) });
  const color = (f: Field) => colorOf(normalizers, f.normalizer?.name);

  // A lag that reads, scales and lands like its field is folded into the
  // field's row; one that differs keeps a row of its own under it.
  const alike = (f: Field, base: Field) =>
    rawText(f) === rawText(base) && f.normalizer?.name === base.normalizer?.name && paramsText(f) === paramsText(base);

  const row = (f: Field, base: Field | null, folded: Field[] = []) => {
    const cls = ["mdp-row", base ? "lag" : "", lit(f) ? "lit" : "", open === f.name ? "open" : ""].join(" ");
    const cols = [f, ...folded].map(columnsText);
    const contiguous = folded.every((l, i) => l.columns[0] === (i === 0 ? f : folded[i - 1]).columns[1]);
    const colsText =
      folded.length && contiguous ? `${f.columns[0]}–${folded[folded.length - 1].columns[1] - 1}` : cols.join(", ");
    return (
      <Fragment key={f.name}>
        <tr className={cls} onClick={() => setOpen(open === f.name ? null : f.name)} title={f.doc} {...hover(f)}>
          <td className="mdp-cols">{colsText}</td>
          <td className="mdp-name">
            {base ? (
              <span className="mdp-lag">
                └ lag {f.lag!.steps}
              </span>
            ) : (
              <>
                {f.name}
                {f.lag && <span className="muted"> · lag {f.lag.steps} of {f.lag.inner}</span>}
              </>
            )}
            {folded.map((l) => (
              <span key={l.name} className="geo-row-kind mdp-lag-tag" title={`${l.name} · column ${columnsText(l)}`}>
                lag {l.lag!.steps}
              </span>
            ))}
          </td>
          <td>{rawText(f)}</td>
          <td>
            <span className="mdp-dot" style={{ background: color(f) }} />
            {f.normalizer ? shortName(f.normalizer.name) : <span className="muted">raw</span>}
          </td>
          <td>{outputText(f)}</td>
          <td className="mdp-spark-cell">{f.curve && <Sparkline curve={f.curve} color={color(f)} />}</td>
        </tr>
        {open === f.name && (
          <tr className="mdp-open">
            <td />
            <td colSpan={5}>
              <Detail
                field={f}
                role={role}
                color={color(f)}
                spec={spec}
                rangeKey={role === "action" ? "action" : part.part}
              />
            </td>
          </tr>
        )}
      </Fragment>
    );
  };

  return (
    <section className="mdp-part">
      <div className="mdp-part-head">
        <span className="mdp-part-name">{part.part}</span>
        <span className="muted">
          {part.width} column{part.width === 1 ? "" : "s"}
          {part.rows === "per intruder" ? ", a row per intruder" : part.rows ? ", one per agent" : ""}
        </span>
      </div>
      <div className="mdp-cells" role="img" aria-label={`${part.part}: ${part.width} columns`}>
        {part.fields.map((f) => (
          <div
            key={f.name}
            className={lit(f) ? "mdp-field lit" : "mdp-field"}
            title={`${columnsText(f)}  ${f.name}  ·  ${f.normalizer ? shortName(f.normalizer.name) : "raw"}`}
            onClick={() => setOpen(open === f.name ? null : f.name)}
            {...hover(f)}
          >
            {Array.from({ length: span(f) }, (_, i) => (
              <span key={i} style={{ background: color(f) }} />
            ))}
          </div>
        ))}
      </div>
      <table className="mdp-table">
        <thead>
          <tr>
            <th>col</th>
            <th>field</th>
            <th>raw</th>
            <th>normalizer</th>
            <th>{role === "observation" ? "policy sees" : "policy gives"}</th>
            <th>mapping</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((g) => {
            const folded = g.lags.filter((l) => alike(l, g.field));
            const own = g.lags.filter((l) => !alike(l, g.field));
            return [row(g.field, null, folded), ...own.map((l) => row(l, g.field))];
          })}
        </tbody>
      </table>
    </section>
  );
}

function Detail({
  field: f,
  role,
  color,
  spec,
  rangeKey,
}: {
  field: Field;
  role: Role;
  color: string;
  spec: SpecDict;
  rangeKey: string;
}) {
  const params = f.normalizer ? Object.entries(f.normalizer.params) : [];
  // A range each aircraft resolves is read from the sampled episode, so the
  // plot can be in the field's units, one line per aircraft.
  const perAircraft = f.raw.per_aircraft && f.curve !== null;
  const { result, loading } = useEpisodeSample(spec, perAircraft);
  const aircraft = perAircraft ? aircraftRanges(result, rangeKey, f.name) : [];
  const unit = f.raw.unit ? ` (${f.raw.unit})` : "";
  let lines: Line[] = [];
  let labels = { x: f.curve?.x_label ?? "", y: f.curve?.y_label ?? "" };
  if (f.curve && aircraft.length) {
    // The curve is sampled over the position in the range: scale it onto
    // each aircraft's own range, on whichever axis is the raw value.
    const onX = role === "observation" || f.curve.x_label !== "policy's value";
    lines = aircraft.flatMap((a, i) =>
      f.curve!.series.map((ys, s) => {
        const scale = (v: number) => a.low + v * (a.high - a.low);
        return {
          xs: onX ? f.curve!.x.map(scale) : f.curve!.x,
          ys: onX ? ys : ys.map(scale),
          label: s === 0 ? a.type : undefined,
          strong: i === 0,
          dashed: s > 0,
        };
      }),
    );
    labels = onX ? { x: `raw${unit}`, y: labels.y } : { x: labels.x, y: `command${unit}` };
  } else if (f.curve) {
    lines = f.curve.series.map((ys, s) => ({
      xs: f.curve!.x,
      ys,
      label: f.curve!.series.length > 1 ? `col ${f.columns[0] + s}` : undefined,
      strong: true,
      dashed: s > 0,
    }));
  }
  const ends = f.curve && !aircraft.length ? rangeEnds(f.curve.x) : [];
  return (
    <div className="mdp-detail">
      <dl className="summary">
        <dt>class</dt>
        <dd>{f.class}</dd>
        {f.doc && (
          <>
            <dt>about</dt>
            <dd>{f.doc}</dd>
          </>
        )}
        <dt>normalizer</dt>
        <dd>{f.normalizer ? f.normalizer.name : "none, the raw value is passed as is"}</dd>
        {params.map(([k, v]) => (
          <Fragment key={k}>
            <dt className="mdp-param">{k}</dt>
            <dd>{Array.isArray(v) ? `(${v.join(", ")})` : String(v)}</dd>
          </Fragment>
        ))}
        <dt>{role === "observation" ? "policy sees" : "policy gives"}</dt>
        <dd>{outputText(f)}</dd>
        {perAircraft && (
          <>
            <dt>range</dt>
            <dd>
              {aircraft.length
                ? aircraft.map((a) => `${a.type} [${number(a.low)}, ${number(a.high)}]`).join(" · ")
                : loading
                  ? "reading each aircraft's range from the sampled episode…"
                  : "each aircraft's own; no aircraft is up in the sampled episode"}
            </dd>
          </>
        )}
      </dl>
      {lines.length > 0 ? (
        <figure className="mdp-plot">
          <Plot lines={lines} ends={ends} color={color} xLabel={labels.x} yLabel={labels.y} />
          {aircraft.length > 0 && (
            <figcaption className="muted small">
              One line per aircraft type up at t = {Math.round(result!.sim_time_s)} s in the sampled episode (seed{" "}
              {result!.seed}); the picked aircraft's is bold.
            </figcaption>
          )}
        </figure>
      ) : (
        f.curve_note && <div className="muted">{f.curve_note}</div>
      )}
    </div>
  );
}

type Line = { xs: number[]; ys: number[]; label?: string; strong: boolean; dashed: boolean };

// The picked aircraft's range first, then one per other type; a few at most.
function aircraftRanges(result: SampleResult | null, key: string, name: string) {
  const rows = result?.ranges?.[key]?.[name] ?? [];
  const first = result?.agents[0]?.acid;
  const ordered = [...rows.filter((r) => r[0] === first), ...rows.filter((r) => r[0] !== first)];
  const seen = new Set<string>();
  const out: { type: string; low: number; high: number }[] = [];
  for (const [, type, low, high] of ordered) {
    if (low === null || high === null || seen.has(type)) continue;
    seen.add(type);
    out.push({ type, low, high });
  }
  return out.slice(0, 5);
}

// The range's own ends in a curve's samples, which reach a tenth past each.
function rangeEnds(x: number[]): number[] {
  const x0 = x[0];
  const x1 = x[x.length - 1];
  const reach = (x1 - x0) / 1.2;
  return [x0 + reach * 0.1, x1 - reach * 0.1];
}

function Sparkline({ curve, color }: { curve: Curve; color: string }) {
  const W = 96;
  const H = 22;
  const all = curve.series.flat();
  const lo = Math.min(...all);
  const hi = Math.max(...all) - lo < 1e-9 ? lo + 1 : Math.max(...all);
  const x0 = curve.x[0];
  const x1 = curve.x[curve.x.length - 1];
  const sx = (x: number) => 1 + ((x - x0) / (x1 - x0 || 1)) * (W - 2);
  const sy = (y: number) => H - 2 - ((y - lo) / (hi - lo)) * (H - 4);
  return (
    <svg width={W} height={H} aria-hidden="true">
      {curve.series.map((series, i) => (
        <polyline
          key={i}
          points={series.map((y, j) => `${sx(curve.x[j])},${sy(y)}`).join(" ")}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
          strokeDasharray={i === 0 ? undefined : "3 2"}
        />
      ))}
    </svg>
  );
}

// About n round-numbered ticks across [lo, hi].
function niceTicks(lo: number, hi: number, n = 4): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const rough = span / n;
  const mag = 10 ** Math.floor(Math.log10(rough));
  const f = rough / mag;
  const step = (f >= 7.5 ? 10 : f >= 3.5 ? 5 : f >= 1.5 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(Math.round(v / step) * step);
  return out;
}

// Label heights pushed apart so none overlap, kept in their order.
function spread(ys: number[], gap: number): number[] {
  const order = ys.map((y, i) => [y, i] as const).sort((a, b) => a[0] - b[0]);
  const out = [...ys];
  let last = -Infinity;
  for (const [y, i] of order) {
    out[i] = Math.max(y, last + gap);
    last = out[i];
  }
  return out;
}

// Lines on one pair of axes, with dashed marks at the range's ends when there
// is one range, and a crosshair reading every line on hover.
function Plot({
  lines,
  ends,
  color,
  xLabel,
  yLabel,
}: {
  lines: Line[];
  ends: number[];
  color: string;
  xLabel: string;
  yLabel: string;
}) {
  const [hoverX, setHoverX] = useState<number | null>(null);
  const W = 420;
  const H = 200;
  const M = { l: 52, r: 56, t: 10, b: 34 };
  const box = useMemo(() => {
    const xs = lines.flatMap((l) => l.xs);
    const ys = lines.flatMap((l) => l.ys);
    let lo = Math.min(...ys);
    let hi = Math.max(...ys);
    if (hi - lo < 1e-9) {
      lo -= 1;
      hi += 1;
    }
    return { x0: Math.min(...xs), x1: Math.max(...xs), lo, hi };
  }, [lines]);
  const { x0, x1, lo, hi } = box;
  const sx = (x: number) => M.l + ((x - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
  const sy = (y: number) => H - M.b - ((y - lo) / (hi - lo)) * (H - M.t - M.b);
  const midY = (M.t + H - M.b) / 2;
  const xTicks = ends.length ? ends : niceTicks(x0, x1, 5);
  const yTicks = niceTicks(lo, hi, 4);
  // A line's value at x: its nearest sample.
  const at = (l: Line, x: number) => {
    let best = 0;
    for (let i = 1; i < l.xs.length; i++) if (Math.abs(l.xs[i] - x) < Math.abs(l.xs[best] - x)) best = i;
    return l.ys[best];
  };
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const frac = ((e.clientX - r.left) * (W / r.width) - M.l) / (W - M.l - M.r);
    setHoverX(x0 + Math.max(0, Math.min(1, frac)) * (x1 - x0));
  };
  const labeled = lines.filter((l) => l.label);
  const labelYs = spread(
    labeled.map((l) => sy(l.ys[l.ys.length - 1]) + 4),
    11,
  );
  return (
    <>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        onMouseMove={onMove}
        onMouseLeave={() => setHoverX(null)}
        role="img"
        aria-label={`${xLabel} to ${yLabel}`}
      >
        <line className="mdp-axis" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
        <line className="mdp-axis" x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} />
        {yTicks.map((y, i) => (
          <g key={`y${i}`}>
            <line className="mdp-grid" x1={M.l} x2={W - M.r} y1={sy(y)} y2={sy(y)} />
            <text className="mdp-tick" x={M.l - 6} y={sy(y) + 4} textAnchor="end">
              {number(y)}
            </text>
          </g>
        ))}
        {lo < 0 && hi > 0 && <line className="mdp-grid" x1={M.l} x2={W - M.r} y1={sy(0)} y2={sy(0)} />}
        {ends.map((x, i) => (
          <line key={`e${i}`} className="mdp-end" x1={sx(x)} x2={sx(x)} y1={M.t} y2={H - M.b} />
        ))}
        {xTicks.map((x, i) => (
          <text key={`x${i}`} className="mdp-tick" x={sx(x)} y={H - M.b + 14} textAnchor="middle">
            {number(x)}
          </text>
        ))}
        <text className="mdp-tick" x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle">
          {xLabel}
        </text>
        <text className="mdp-tick" x={12} y={midY} textAnchor="middle" transform={`rotate(-90 12 ${midY})`}>
          {yLabel}
        </text>
        {lines.map((l, i) => (
          <polyline
            key={i}
            points={l.xs.map((x, j) => `${sx(x)},${sy(l.ys[j])}`).join(" ")}
            fill="none"
            stroke={color}
            strokeWidth={l.strong ? 2 : 1.25}
            strokeOpacity={l.strong ? 1 : 0.55}
            strokeDasharray={l.dashed ? "5 4" : undefined}
            strokeLinejoin="round"
          />
        ))}
        {labeled.map((l, i) => (
          <text key={`l${i}`} className="mdp-tick" x={W - M.r + 6} y={labelYs[i]}>
            {l.label}
          </text>
        ))}
        {hoverX !== null && (
          <g>
            <line className="mdp-cross" x1={sx(hoverX)} x2={sx(hoverX)} y1={M.t} y2={H - M.b} />
            {lines.map((l, i) => (
              <circle key={i} cx={sx(hoverX)} cy={sy(at(l, hoverX))} r={l.strong ? 4 : 3} fill={color} stroke="var(--bg)" strokeWidth={2} />
            ))}
          </g>
        )}
      </svg>
      <div className="small mdp-readout">
        {hoverX === null ? (
          <span className="muted">hover to read</span>
        ) : (
          <span>
            {xLabel} {number(hoverX)} →{" "}
            {lines
              .filter((l) => !l.dashed || lines.length <= 2)
              .map((l) => `${l.label && labeled.length > 1 ? `${l.label} ` : ""}${number(at(l, hoverX))}`)
              .join(", ")}
          </span>
        )}
      </div>
    </>
  );
}
