// The field editor's Probe tab: the design flown to a moment, one aircraft,
// and one field read there - each stage actual and expected, the calls it made
// and the traffic it read (any of them overridable), its lag ring and its cost.
// See bluesky_sandbox.checks.probe.
import { Fragment, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  api,
  type ProbeCell,
  type ProbeLag,
  type ProbeNode,
  type ProbeOverride,
  type ProbeResult,
  type ProbeResultBody,
  type ProbeStage,
  type SpecDict,
} from "../../api";
import { niceTicks } from "../MdpTab";

// How long the probe waits after an edit before it flies again (ms).
const DEBOUNCE_MS = 450;
// A progression column needs this much room (px) before it folds to a grid.
const COLUMN_PX = 132;
// Frames and chips shown either side of a fold.
const FOLD_KEEP = 2;

export function FieldProbe({
  spec,
  listKey,
  entry,
  kind,
  stepNormalizer,
}: {
  spec: SpecDict;
  listKey: string;
  entry: number;
  kind: "obs" | "action";
  // The step normalizer's kwargs, for an action that takes a choice.
  stepNormalizer?: SpecDict | null;
}) {
  const dt = Number(spec.env?.dt) || 1;
  const [seed, setSeed] = useState(0);
  const [atS, setAtS] = useState(0);
  const [acid, setAcid] = useState<string | null>(null);
  const [other, setOther] = useState<string | null>(null);
  const [overrides, setOverrides] = useState<ProbeOverride[]>([]);
  const [give, setGive] = useState<number>(() => defaultGive(stepNormalizer));
  const [flight, setFlight] = useState(0);
  const [out, setOut] = useState<ProbeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const request = JSON.stringify({ spec, listKey, entry, seed, atS, acid, other, overrides, give, flight });
  useEffect(() => {
    let live = true;
    const timer = window.setTimeout(() => {
      setLoading(true);
      api
        .probe(spec, {
          list: listKey,
          entry,
          seed,
          at_s: atS,
          acid,
          other,
          overrides,
          give: kind === "action" ? give : null,
        })
        .then((r) => {
          if (!live) return;
          setOut(r);
          setError(r.error ?? null);
        })
        .catch((e) => live && setError(String(e?.message ?? e)))
        .finally(() => live && setLoading(false));
    }, DEBOUNCE_MS);
    return () => {
      live = false;
      window.clearTimeout(timer);
    };
    // The request, as one value: the spec is a new object on every edit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request]);

  const result = out?.result;
  const own = result?.aircraft ?? out?.acid ?? null;
  const setOverride = (o: ProbeOverride) =>
    setOverrides((list) => [...list.filter((x) => !sameOverride(x, o)), o]);
  const clearOverride = (o: ProbeOverride) => setOverrides((list) => list.filter((x) => !sameOverride(x, o)));

  return (
    <div className="probe">
      <div className="probe-controls">
        <label className="numfield inline">
          <span>seed</span>
          <input type="number" value={seed} onChange={(e) => setSeed(Math.trunc(Number(e.target.value) || 0))} />
        </label>
        <label className="numfield inline">
          <span>at</span>
          <input
            type="number"
            min={0}
            step={dt}
            value={atS}
            onChange={(e) => setAtS(Math.max(0, Number(e.target.value) || 0))}
          />
          <span className="muted">
            s · step {Math.round(atS / dt)} (dt {fmt(dt)} s)
          </span>
        </label>
        <label className="numfield inline">
          <span>aircraft</span>
          <select value={own ?? ""} onChange={(e) => setAcid(e.target.value || null)}>
            {!own && <option value="">the newest</option>}
            {(out?.aircraft ?? []).map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </label>
        {result?.kind === "pair" && (
          <label className="numfield inline">
            <span>about</span>
            <select value={result.other ?? ""} onChange={(e) => setOther(e.target.value || null)}>
              {(out?.aircraft ?? [])
                .filter((a) => a !== own)
                .map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
            </select>
          </label>
        )}
        <button onClick={() => setFlight((n) => n + 1)} title="fly the episode to this moment again">
          ↻ re-fly
        </button>
        {out && (
          <span className="muted small">
            {out.aircraft.length} aircraft in the air{loading ? " · probing…" : ""}
          </span>
        )}
        {!out && loading && <span className="muted small">probing…</span>}
      </div>

      {error && <div className="probe-error">{error}</div>}

      {result && (
        <>
          {result.kind === "action" ? (
            <GivePanel result={result} give={give} onGive={setGive} stepped={!!stepNormalizer} />
          ) : (
            result.curve.length > 1 && <NormalizerChart result={result} />
          )}
          <div className="sub-label">stages</div>
          <StageTable result={result} />
          {result.notes.map((n) => (
            <div key={n} className="probe-note muted small">
              {n}
            </div>
          ))}
          <div className="sub-label">trace</div>
          <div className="probe-tree">
            {result.trace.map((node, i) => (
              <TraceRow
                key={i}
                node={node}
                depth={0}
                own={result.aircraft}
                overrides={overrides}
                onSet={setOverride}
                onClear={clearOverride}
              />
            ))}
            {result.lag && <LagRing lag={result.lag} result={result} dt={dt} />}
          </div>
          <Cost result={result} />
          <div className="probe-foot">
            <span className="muted small">
              {overrides.length === 0
                ? "no overrides"
                : `${overrides.length} ${overrides.length === 1 ? "override" : "overrides"}`}
            </span>
            <button className="link" disabled={overrides.length === 0} onClick={() => setOverrides([])}>
              clear overrides
            </button>
          </div>
        </>
      )}
    </div>
  );
}

// -------------------------------------------------------------- normalizer --

// The normalizer across the field's bounds and a fifth either side, and where
// this aircraft's value falls on it.
function NormalizerChart({ result }: { result: ProbeResultBody }) {
  const W = 420;
  const H = 170;
  const M = { l: 46, r: 14, t: 12, b: 34 };
  const raw = scalar(stageOf(result, "raw")?.actual);
  const norm = scalar(stageOf(result, "normalized")?.actual);
  const xs = result.curve.map(([x]) => x);
  const ys = result.curve.map(([, y]) => y);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (x: number) => M.l + ((Math.min(Math.max(x, x0), x1) - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
  const sy = (y: number) => H - M.b - ((y - y0) / (y1 - y0 || 1)) * (H - M.t - M.b);
  const points = result.curve.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(" ");
  const bounds = result.bounds ?? [x0, x1];
  const unit = unitSuffix(result.unit).trim();
  const px = raw == null ? null : sx(raw);
  const py = norm == null ? null : sy(norm);
  // The point's label on whichever side has room.
  const right = px != null && px < W - M.r - 130;
  return (
    <>
      <div className="sub-label">normalizer</div>
      <svg className="probe-chart" viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="normalized against raw">
        {niceTicks(y0, y1, 4).map((y) => (
          <g key={`y${y}`}>
            <line className="mdp-grid" x1={M.l} x2={W - M.r} y1={sy(y)} y2={sy(y)} />
            <text className="mdp-tick" x={M.l - 6} y={sy(y) + 3} textAnchor="end">
              {fmt(y)}
            </text>
          </g>
        ))}
        {niceTicks(x0, x1, 5).map((x) => (
          <text key={`x${x}`} className="mdp-tick" x={sx(x)} y={H - M.b + 14} textAnchor="middle">
            {fmt(x)}
          </text>
        ))}
        {bounds.map((b, i) => (
          <line key={`b${i}`} className="probe-chart-bound" x1={sx(b)} x2={sx(b)} y1={M.t} y2={H - M.b} />
        ))}
        <line className="mdp-axis" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
        <line className="mdp-axis" x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} />
        <text className="mdp-tick" x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle">
          raw{unit ? ` (${unit})` : ""}
        </text>
        <text className="mdp-tick" x={11} y={(M.t + H - M.b) / 2} textAnchor="middle"
          transform={`rotate(-90 11 ${(M.t + H - M.b) / 2})`}>
          normalized
        </text>
        <polyline points={points} className="probe-chart-line" />
        {px != null && py != null && (
          <>
            <circle cx={px} cy={py} r={4} className="probe-chart-dot" />
            <text className="probe-chart-label" x={right ? px + 8 : px - 8} y={py - 8} textAnchor={right ? "start" : "end"}>
              {withUnit(raw, result.unit)} → {fmt(norm)}
            </text>
          </>
        )}
      </svg>
    </>
  );
}

// ------------------------------------------------------------------ action --

function GivePanel({
  result,
  give,
  onGive,
  stepped,
}: {
  result: ProbeResultBody;
  give: number;
  onGive: (v: number) => void;
  stepped: boolean;
}) {
  const target = stageOf(result, "target") ?? stageOf(result, "on grid");
  if (stepped && result.choices.length) {
    return (
      <div className="probe-give">
        <div className="sub-label">give a choice</div>
        <div className="probe-choices" style={{ gridTemplateColumns: `repeat(${Math.min(result.choices.length, 7)}, minmax(0, 1fr))` }}>
          {result.choices.map((c) => (
            <button
              key={c.choice}
              className={c.choice === give ? "probe-choice on" : "probe-choice"}
              onClick={() => onGive(c.choice)}
            >
              <span className="muted small">choice {c.choice + 1}</span>
              <span className="mono">
                {signed(c.steps)} {Math.abs(c.steps) === 1 ? "step" : "steps"}
              </span>
              {c.target != null && <span className="mono muted small">→ {withUnit(c.target, result.unit)}</span>}
            </button>
          ))}
        </div>
      </div>
    );
  }
  return (
    <div className="probe-give">
      <label className="numfield inline">
        <span>policy gives</span>
        <input type="number" step={0.1} value={give} onChange={(e) => onGive(Number(e.target.value) || 0)} />
        {target && (
          <span className="muted">
            → {target.name} {withUnit(scalar(target.actual), result.unit)}
          </span>
        )}
      </label>
    </div>
  );
}

// ------------------------------------------------------------------ stages --

function StageTable({ result }: { result: ProbeResultBody }) {
  return (
    <table className="probe-stages">
      <thead>
        <tr>
          <th />
          <th title="what the env computes for the policy">actual</th>
          <th title="the field's plain statement of it, one value at a time">expected</th>
          <th />
          <th />
        </tr>
      </thead>
      <tbody>
        {result.stages.map((s) =>
          s.name === "BS command" ? (
            <tr key={s.name}>
              <td className="muted">{s.name}</td>
              <td className="mono" colSpan={4}>
                {s.actual ?? "—"}
              </td>
            </tr>
          ) : (
            <tr key={s.name}>
              <td className="muted">{s.name}</td>
              <td className="mono">{stageText(s, result)}</td>
              <td className="mono">{s.expected == null ? "" : stageText({ ...s, actual: s.expected }, result)}</td>
              <td className={s.agrees === false ? "bad" : "ok"}>{s.agrees == null ? "" : s.agrees ? "✓" : "✗"}</td>
              <td className="muted small">
                {s.name === "normalized" ? clipNote(result) : s.agrees === false ? differNote(s, result) : s.note}
              </td>
            </tr>
          ),
        )}
      </tbody>
    </table>
  );
}

function stageText(s: ProbeStage, result: ProbeResultBody): string {
  const v = s.actual;
  if (v == null) return "—";
  if (s.name === "policy") return result.choices.length ? `choice ${Number(v) + 1} of ${result.choices.length}` : fmt(Number(v));
  if (s.name === "steps") return signed(Number(v));
  if (s.name === "normalized") return Array.isArray(v) ? v.map((x) => fmt(x)).join(", ") : fmt(Number(v));
  if (Array.isArray(v)) return v.map((x) => fmt(x)).join(", ");
  if (typeof v === "string") return v;
  return withUnit(Number(v), result.unit);
}

function differNote(s: ProbeStage, result: ProbeResultBody): string {
  const a = scalar(s.actual);
  const e = scalar(s.expected);
  return a == null || e == null ? "" : `differ by ${withUnit(Math.abs(a - e), result.unit)}`;
}

function clipNote(result: ProbeResultBody): string {
  const raw = scalar(stageOf(result, "raw")?.actual);
  const norm = scalar(stageOf(result, "normalized")?.actual);
  const ys = result.curve.map(([, y]) => y);
  if (raw == null || norm == null || !result.bounds || !ys.length) return "";
  const [low, high] = result.bounds;
  if (raw > high && norm >= Math.max(...ys) - 1e-9) return `clipped at ${fmt(high)}`;
  if (raw < low && norm <= Math.min(...ys) + 1e-9) return `clipped at ${fmt(low)}`;
  return "";
}

// ------------------------------------------------------------------- trace --

function TraceRow({
  node,
  depth,
  own,
  overrides,
  onSet,
  onClear,
}: {
  node: ProbeNode;
  depth: number;
  own: string;
  overrides: ProbeOverride[];
  onSet: (o: ProbeOverride) => void;
  onClear: (o: ProbeOverride) => void;
}) {
  const [open, setOpen] = useState(depth < 2);
  const traffic = node.key?.startsWith("traf:") ?? false;
  const nested = node.children.length > 0 || node.leaves.length > 0;
  const whole = node.number == null ? null : overrideOf(node.key, node.number, own);
  return (
    <>
      <OverridableRow
        depth={depth}
        caret={nested ? (open ? "▾" : "▸") : ""}
        onToggle={nested ? () => setOpen(!open) : undefined}
        label={node.label}
        value={node.value}
        override={whole}
        traffic={traffic}
        overrides={overrides}
        onSet={onSet}
        onClear={onClear}
        head={depth === 0}
      />
      {open &&
        node.leaves.map(([leaf, value, number]) => (
          <OverridableRow
            key={leaf}
            depth={depth + 1}
            caret=""
            label={leaf}
            value={value}
            override={node.key ? { target: node.key, value: number, leaf } : null}
            traffic={false}
            overrides={overrides}
            onSet={onSet}
            onClear={onClear}
          />
        ))}
      {open &&
        node.children.map((child, i) => (
          <TraceRow
            key={i}
            node={child}
            depth={depth + 1}
            own={own}
            overrides={overrides}
            onSet={onSet}
            onClear={onClear}
          />
        ))}
    </>
  );
}

function OverridableRow({
  depth,
  caret,
  onToggle,
  label,
  value,
  override,
  traffic,
  overrides,
  onSet,
  onClear,
  head,
}: {
  depth: number;
  caret: string;
  onToggle?: () => void;
  label: string;
  value: string;
  override: ProbeOverride | null;
  traffic: boolean;
  overrides: ProbeOverride[];
  onSet: (o: ProbeOverride) => void;
  onClear: (o: ProbeOverride) => void;
  head?: boolean;
}) {
  const set = override ? overrides.find((o) => sameOverride(o, override)) : undefined;
  const indent = { paddingLeft: 12 + depth * 20 };
  if (set) {
    return (
      <div className="probe-row overridden" style={indent}>
        <span>{label} =</span>
        <input
          type="number"
          value={set.value}
          aria-label={`override ${label}`}
          onChange={(e) => onSet({ ...set, value: Number(e.target.value) || 0 })}
        />
        <span className="probe-override-note small">
          {traffic ? "overridden in BlueSky" : "overridden"}
        </span>
        <button className="link probe-row-end" aria-label="clear override" onClick={() => onClear(set)}>
          ✕
        </button>
      </div>
    );
  }
  return (
    <div className={head ? "probe-row head" : "probe-row"} style={indent}>
      <span className={onToggle ? "probe-toggle" : ""} onClick={onToggle}>
        {caret && <span className="probe-caret">{caret}</span>}
        {label}
        {value !== "" && <> {label.startsWith("bs.traf") || !onToggle ? "=" : "→"} {value}</>}
      </span>
      {override && (
        <button className="link probe-row-end small" onClick={() => onSet(override)}>
          override
        </button>
      )}
    </div>
  );
}

// What overriding a node sets, starting from the value it read.
function overrideOf(key: string | null, value: number, own: string): ProbeOverride | null {
  if (!key) return null;
  if (key.startsWith("traf:")) {
    const [path, aircraft] = key.split("@");
    return { target: path, value, aircraft: aircraft || own, leaf: "" };
  }
  return { target: key, value, leaf: "" };
}

// A traffic override carries the value read; a call's, what it returned.
function sameOverride(a: ProbeOverride, b: ProbeOverride): boolean {
  return a.target === b.target && (a.leaf ?? "") === (b.leaf ?? "") && (a.aircraft ?? "") === (b.aircraft ?? "");
}

// --------------------------------------------------------------------- lag --

function LagRing({ lag, result, dt }: { lag: ProbeLag; result: ProbeResultBody; dt: number }) {
  const [open, setOpen] = useState(true);
  const depth = lag.depth ?? lag.lags.length + 1;
  const now = result.sim_time_s;
  const steps = (s: number) => Math.round((now - s) / dt);
  const ago = (n: number) => `${n} ${n === 1 ? "step" : "steps"} ago`;
  let since = "";
  if (lag.frames.length < depth && lag.first_s != null) {
    since =
      lag.start_s != null && lag.first_s > lag.start_s
        ? ` · ${result.aircraft} spawned ${ago(steps(lag.first_s))}`
        : steps(lag.first_s) === 0
          ? " · episode start"
          : ` · episode started ${ago(steps(lag.first_s))}`;
  }
  const name = (k: number) => (k === 0 ? lag.inner : `_lag${k}`);
  const frames = fold(lag.frames, FOLD_KEEP);
  return (
    <>
      <button className="probe-row head probe-lag-head" onClick={() => setOpen(!open)}>
        <span className="probe-caret">{open ? "▾" : "▸"}</span>
        lag ring of {lag.inner} · {lag.frames.length} of {depth} frames{since}
      </button>
      {open && (
        <div className="probe-lag">
          {frames.map((f, i) =>
            f === null ? (
              <div key={`fold${i}`} className="probe-frame fold muted small">
                … {lag.frames.length - 2 * FOLD_KEEP} more frames
              </div>
            ) : (
              <div key={f.back} className="probe-frame">
                <span className="muted">{f.sim_time_s != null ? `at ${fmt(f.sim_time_s)} s` : ago(f.back)}</span>
                <span className="mono">{withUnit(f.value, result.unit)}</span>
                <span className="probe-readers">
                  {fold(f.read_by, FOLD_KEEP).map((k, j) =>
                    k === null ? (
                      <span key={`f${j}`} className="muted">
                        …
                      </span>
                    ) : (
                      <span key={k} className={f.placeholder.includes(k) ? "probe-reader placeholder" : "probe-reader"}>
                        {name(k)}
                      </span>
                    ),
                  )}
                </span>
              </div>
            ),
          )}
          <Progression lag={lag} name={name} />
          <div className="probe-legend small muted">
            <span className="probe-swatch" />
            placeholder: too little history yet
          </div>
        </div>
      )}
    </>
  );
}

function Progression({ lag, name }: { lag: ProbeLag; name: (k: number) => string }) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const measure = () => setWidth(el.clientWidth);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  const columns = [0, ...lag.lags];
  const compact = width > 0 && 96 + columns.length * COLUMN_PX > width;
  const values = lag.progression.flatMap((r) => r.cells.map((c) => c.value)).filter((v): v is number => v != null);
  const [lo, hi] = [Math.min(...values), Math.max(...values)];
  const cellText = (c: ProbeCell) => (c.value == null ? "—" : fmt(c.value));
  const mark = (c: ProbeCell) => (c.placeholder || c.agrees == null ? "" : c.agrees ? " ✓" : " ✗");
  return (
    <div ref={box} className="probe-progression">
      {compact ? (
        <div className="probe-grid" style={{ gridTemplateColumns: `72px repeat(${columns.length}, minmax(10px, 1fr))` }}>
          <span className="muted small">pushed</span>
          {columns.map((k) => (
            <span key={k} className="muted small probe-grid-head" title={name(k)}>
              {k === 0 || k === columns[columns.length - 1] ? (k === 0 ? "0" : `${k}`) : ""}
            </span>
          ))}
          {lag.progression.map((row) => (
            <Fragment key={row.sim_time_s}>
              <span className="muted small">{fmt(row.sim_time_s)} s</span>
              {row.cells.map((c) => (
                <span
                  key={c.steps}
                  className={c.placeholder ? "probe-grid-cell placeholder" : c.agrees === false ? "probe-grid-cell bad" : "probe-grid-cell"}
                  style={c.placeholder ? undefined : { opacity: 0.25 + 0.75 * shade(c.value, lo, hi) }}
                  title={`${name(c.steps)} at ${fmt(row.sim_time_s)} s: ${cellText(c)}${mark(c)}`}
                />
              ))}
            </Fragment>
          ))}
        </div>
      ) : (
        <div className="probe-table" style={{ gridTemplateColumns: `96px repeat(${columns.length}, ${COLUMN_PX}px)` }}>
          <span className="muted small">pushed</span>
          {columns.map((k) => (
            <span key={k} className="muted small">
              {name(k)}
            </span>
          ))}
          {lag.progression.map((row) => (
            <Fragment key={row.sim_time_s}>
              <span className="muted">{fmt(row.sim_time_s)} s</span>
              {row.cells.map((c) => (
                <span key={c.steps} className="mono">
                  <span className={c.placeholder ? "probe-cell placeholder" : c.agrees === false ? "probe-cell bad" : "probe-cell"}>
                    {cellText(c)}
                    {mark(c)}
                  </span>
                </span>
              ))}
            </Fragment>
          ))}
        </div>
      )}
    </div>
  );
}

function shade(v: number | null, lo: number, hi: number): number {
  if (v == null || !Number.isFinite(lo) || hi <= lo) return 1;
  return (v - lo) / (hi - lo);
}

// The first and last `keep` of a long list, a null where the rest fold.
function fold<T>(items: T[], keep: number): (T | null)[] {
  return items.length <= 2 * keep + 1 ? items : [...items.slice(0, keep), null, ...items.slice(-keep)];
}

// -------------------------------------------------------------------- cost --

function Cost({ result }: { result: ProbeResultBody }) {
  const [open, setOpen] = useState(false);
  const cost = result.cost;
  const paths = Object.keys(cost.ms ?? {});
  if (!paths.length) return null;
  const line = paths.map((p) => `${ms(cost.ms![p])} ${p}`).join(" · ");
  return (
    <div className="probe-cost">
      <button className="probe-row" onClick={() => setOpen(!open)}>
        <span className="probe-caret">{open ? "▾" : "▸"}</span>
        <span className="sub-label inline">cost</span> <span className="muted">{line}</span>
        {cost.batched_path === false && <span className="warn"> · no batched path</span>}
      </button>
      {open && (
        <div className="probe-cost-table">
          <span className="muted">path</span>
          <span className="muted">{cost.aircraft} aircraft</span>
          <span className="muted">{cost.max_aircraft} aircraft</span>
          {paths.map((p) => (
            <Fragment key={p}>
              <span>{p}</span>
              <span className="mono">{ms(cost.ms![p])}</span>
              <span className="mono">{ms(cost.ms_at_max?.[p])}</span>
            </Fragment>
          ))}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ format --

function stageOf(result: ProbeResultBody, name: string): ProbeStage | undefined {
  return result.stages.find((s) => s.name === name);
}

function scalar(v: any): number | null {
  const x = Array.isArray(v) ? v[0] : v;
  return x == null || !Number.isFinite(Number(x)) ? null : Number(x);
}

function fmt(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return "—";
  const rounded = Math.abs(x) >= 1000 ? Math.round(x * 10) / 10 : Number(x.toPrecision(4));
  return rounded.toLocaleString("en-US", { maximumFractionDigits: 4 });
}

function signed(x: number): string {
  return x > 0 ? `+${fmt(x)}` : x < 0 ? `−${fmt(-x)}` : "±0";
}

function unitSuffix(unit: string): string {
  return !unit ? "" : unit === "deg" ? "°" : ` ${unit}`;
}

function withUnit(x: number | null, unit: string): string {
  return x == null ? "—" : `${fmt(x)}${unitSuffix(unit)}`;
}

function ms(x: number | undefined): string {
  if (x == null) return "—";
  return x < 0.01 ? "<0.01 ms" : `${x < 0.1 ? x.toPrecision(2) : x < 10 ? x.toFixed(2) : x.toFixed(1)} ms`;
}

// The choice in the middle for a step action - no change, where zero is one,
// else one step up - and zero for a continuous one.
function defaultGive(step?: SpecDict | null): number {
  const each = Number(step?.kwargs?.steps_each_way ?? 10);
  return step && Number.isFinite(each) ? each : 0;
}
