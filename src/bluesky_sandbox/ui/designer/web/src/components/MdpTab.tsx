// The design's spaces (ui/designer/mdp.py), part by part: the part's columns
// drawn as a strip colored by the normalizer each goes through, and a table of
// its fields - each lag listed under the field it lags - where a row opens to
// its normalizer's settings and mapping.
import { Fragment, useEffect, useMemo, useState } from "react";
import { api, type SpecDict } from "../api";
import { normalizerColor } from "../normColors";
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
  if (abs !== 0 && (abs >= 1e5 || abs < 1e-3)) return v.toExponential(1);
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
        <PartView key={p.part} part={p} role="observation" normalizers={summary.normalizers} />
      ))}
      <h3>action · {summary.action.space}</h3>
      {summary.action.parts.map((p) => (
        <PartView key={p.part} part={p} role="action" normalizers={summary.normalizers} />
      ))}
    </div>
  );
}

function PartView({ part, role, normalizers }: { part: Part; role: Role; normalizers: string[] }) {
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
              <Detail field={f} role={role} color={color(f)} />
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

function Detail({ field: f, role, color }: { field: Field; role: Role; color: string }) {
  const params = f.normalizer ? Object.entries(f.normalizer.params) : [];
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
      </dl>
      {f.curve ? <Plot curve={f.curve} color={color} first={f.columns[0]} /> : f.curve_note && <div className="muted">{f.curve_note}</div>}
    </div>
  );
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

// The mapping, with axes at the sampled ends, dashed marks at the range's ends
// (the samples reach a tenth past each, see mdp.py) and a crosshair on hover.
function Plot({ curve, color, first }: { curve: Curve; color: string; first: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 400;
  const H = 200;
  const M = { l: 44, r: 44, t: 10, b: 34 };
  const xs = curve.x;
  const { lo, hi } = useMemo(() => {
    const all = curve.series.flat();
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    return hi - lo < 1e-9 ? { lo: lo - 1, hi: hi + 1 } : { lo, hi };
  }, [curve]);
  const x0 = xs[0];
  const x1 = xs[xs.length - 1];
  const sx = (x: number) => M.l + ((x - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
  const sy = (y: number) => H - M.b - ((y - lo) / (hi - lo)) * (H - M.t - M.b);
  const reach = (x1 - x0) / 1.2;
  const ends = [x0 + reach * 0.1, x1 - reach * 0.1];
  const midY = (M.t + H - M.b) / 2;
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const frac = ((e.clientX - box.left) * (W / box.width) - M.l) / (W - M.l - M.r);
    setHover(Math.max(0, Math.min(xs.length - 1, Math.round(frac * (xs.length - 1)))));
  };
  return (
    <figure className="mdp-plot">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        role="img"
        aria-label={`${curve.x_label} to ${curve.y_label}`}
      >
        <line className="mdp-axis" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
        <line className="mdp-axis" x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} />
        {[lo, hi].map((y, i) => (
          <g key={`y${i}`}>
            <line className="mdp-grid" x1={M.l} x2={W - M.r} y1={sy(y)} y2={sy(y)} />
            <text className="mdp-tick" x={M.l - 6} y={sy(y) + 4} textAnchor="end">
              {number(y)}
            </text>
          </g>
        ))}
        {lo < 0 && hi > 0 && <line className="mdp-grid" x1={M.l} x2={W - M.r} y1={sy(0)} y2={sy(0)} />}
        {ends.map((x, i) => (
          <g key={`x${i}`}>
            <line className="mdp-end" x1={sx(x)} x2={sx(x)} y1={M.t} y2={H - M.b} />
            <text className="mdp-tick" x={sx(x)} y={H - M.b + 14} textAnchor="middle">
              {number(x)}
            </text>
          </g>
        ))}
        <text className="mdp-tick" x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle">
          {curve.x_label}
        </text>
        <text className="mdp-tick" x={12} y={midY} textAnchor="middle" transform={`rotate(-90 12 ${midY})`}>
          {curve.y_label}
        </text>
        {curve.series.map((series, i) => (
          <g key={i}>
            <polyline
              points={series.map((y, j) => `${sx(xs[j])},${sy(y)}`).join(" ")}
              fill="none"
              stroke={color}
              strokeWidth={2}
              strokeDasharray={i === 0 ? undefined : "5 4"}
              strokeLinejoin="round"
            />
            {curve.series.length > 1 && (
              <text className="mdp-tick" x={W - M.r + 6} y={sy(series[series.length - 1]) + 4}>
                col {first + i}
              </text>
            )}
          </g>
        ))}
        {hover !== null && (
          <g>
            <line className="mdp-cross" x1={sx(xs[hover])} x2={sx(xs[hover])} y1={M.t} y2={H - M.b} />
            {curve.series.map((series, i) => (
              <circle key={i} cx={sx(xs[hover])} cy={sy(series[hover])} r={4} fill={color} stroke="var(--bg)" strokeWidth={2} />
            ))}
          </g>
        )}
      </svg>
      <figcaption className="small">
        {hover === null ? (
          <span className="muted">hover to read</span>
        ) : (
          <span>
            {curve.x_label} {number(xs[hover])} → {curve.series.map((s) => number(s[hover])).join(", ")}
          </span>
        )}
      </figcaption>
    </figure>
  );
}
