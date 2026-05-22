// Editor for a "sampled value" — anything the API accepts where a scenario can
// randomise a scalar: a fixed number, a uniform (low, high) range, or **any
// scipy.stats distribution**. The distribution's name is picked from the catalog
// (searchable, with its signature as a hint) and its parameters are shown as
// named fields (introspected from scipy), so kwargs can't be mistyped. Emits the
// same tagged encoding the spec (de)serialiser uses:
//   number | {type:"range",low,high} | {type:"scipy",name,args,kwds}
import { Hint } from "./Hint";
import { useEffect, useState } from "react";
import { api } from "../../api";
import { Picker } from "./Picker";

export type SampledValue = number | { type: string; [k: string]: any } | undefined;

type Mode = "fixed" | "range" | "dist" | "envelope" | "choice";
type DistInfo = { name: string; params: string[]; discrete: boolean; unbounded?: boolean; signature: string };

function modeOf(v: SampledValue): Mode {
  if (v && typeof v === "object") {
    if (v.type === "range") return "range";
    if (v.type === "scipy") return "dist";
    if (v.type === "envelope") return "envelope";
    if (v.type === "categorical") return "choice";
  }
  return "fixed";
}

const paramDefault = (p: string): number => (p === "scale" ? 1 : p === "high" ? 10 : 0);

// A number input that keeps the raw text while focused, so you can type
// negatives / decimals / clear without the value snapping to a default. It only
// emits when the text parses to a finite number.
export function NumInput({
  value,
  onChange,
  step,
  int = false,
  className = "",
  placeholder,
  onClear,
}: {
  value: number;
  onChange: (n: number) => void;
  step?: number | string;
  int?: boolean;
  className?: string;
  placeholder?: string;
  // When provided, clearing the input (empty text) calls this instead of holding
  // the last value — lets an optional field fall back to "unset" / inherit.
  onClear?: () => void;
}) {
  const [focused, setFocused] = useState(false);
  const [text, setText] = useState("");
  const display = focused ? text : Number.isFinite(value) ? String(value) : "";
  return (
    <input
      className={className}
      type="number"
      step={step}
      placeholder={placeholder}
      value={display}
      onFocus={() => {
        setText(Number.isFinite(value) ? String(value) : "");
        setFocused(true);
      }}
      onBlur={() => setFocused(false)}
      onChange={(e) => {
        setText(e.target.value);
        if (e.target.value === "" && onClear) {
          onClear();
          return;
        }
        const n = parseFloat(e.target.value);
        if (Number.isFinite(n)) onChange(int ? Math.round(n) : n);
      }}
    />
  );
}

// Seed kwds for a distribution: keep any existing values for params it still has.
function defaultsFor(info: DistInfo | undefined, existing: Record<string, number> = {}): Record<string, number> {
  const out: Record<string, number> = {};
  for (const p of info?.params ?? ["loc", "scale"]) out[p] = existing[p] ?? paramDefault(p);
  return out;
}

// An optional ValueField: a checkbox enables the field; when on it shows the
// fixed/range/distribution editor, when off the value is cleared (undefined).
export function OptValueField({
  label,
  value,
  defaultValue,
  onChange,
  step = 1,
  int = false,
  allowEnvelope = false,
  disabled = false,
  disabledReason,
  hint,
}: {
  label: string;
  /** What this value controls, shown on hover. */
  hint?: string;
  value: SampledValue;
  defaultValue?: SampledValue;
  onChange: (v: SampledValue) => void;
  step?: number;
  int?: boolean;
  allowEnvelope?: boolean;
  disabled?: boolean;
  disabledReason?: string;
}) {
  const [draftEnabled, setDraftEnabled] = useState(false);
  const enabled = value !== undefined && value !== null;
  const showEditor = enabled || draftEnabled;
  const checkboxDisabled = disabled && !enabled;
  return (
    <div className="opt-value-field">
      <label className="radio">
        <input
          type="checkbox"
          checked={showEditor}
          disabled={checkboxDisabled}
          onChange={(e) => {
            if (!e.target.checked) {
              setDraftEnabled(false);
              onChange(undefined);
              return;
            }
            if (defaultValue === undefined) {
              setDraftEnabled(true);
              return;
            }
            onChange(defaultValue);
          }}
        />
        {label}
        <Hint text={hint} />
      </label>
      {checkboxDisabled && disabledReason && (
        <div className="muted small">{disabledReason}</div>
      )}
      {showEditor && (
        <ValueField label="" value={value} onChange={onChange} step={step} int={int} allowEnvelope={allowEnvelope} />
      )}
    </div>
  );
}

export function ValueField({
  label,
  value,
  onChange,
  step = 1,
  int = false,
  allowEnvelope = false,
  allowChoice = false,
}: {
  label: string;
  value: SampledValue;
  onChange: (v: SampledValue) => void;
  step?: number;
  int?: boolean;
  // When set, offers an "aircraft envelope" mode for per-aircraft draws within
  // the flight envelope (no fixed value to edit).
  allowEnvelope?: boolean;
  // When set, offers a weighted-choice mode: a categorical over *numeric*
  // values (e.g. a count-scale mixture: 90% x1.0, 10% x0.08).
  allowChoice?: boolean;
}) {
  const [dists, setDists] = useState<DistInfo[]>([]);
  useEffect(() => {
    api.catalogOnce().then((c) => setDists(c?.distributions ?? [])).catch(() => setDists([]));
  }, []);
  const distByName = (name: string) => dists.find((d) => d.name === name);

  const mode = modeOf(value);
  const fixed = typeof value === "number" ? value : Number.NaN;
  const finiteFixed = Number.isFinite(fixed) ? fixed : 0;
  const range = mode === "range" ? (value as any) : { low: finiteFixed, high: finiteFixed };
  const dist = mode === "dist" ? (value as any) : null;
  const distInfo = dist ? distByName(dist.name) : undefined;

  const choiceWeights: Record<string, number> =
    mode === "choice" ? ((value as any).weights ?? {}) : {};

  const switchMode = (m: Mode) => {
    if (m === "fixed") {
      // Always commit a finite number so switching from a range/distribution
      // works (the current value isn't numeric then, so `fixed` is NaN).
      const seed = Number.isFinite(fixed) ? fixed : Number(range.low) || 0;
      onChange(int ? Math.round(seed) : seed);
    }
    else if (m === "range") onChange({ type: "range", low: range.low ?? 0, high: range.high ?? 0 });
    else if (m === "envelope") onChange({ type: "envelope" });
    else if (m === "choice") {
      const seed = Number.isFinite(fixed) ? fixed : 1;
      onChange({ type: "categorical", weights: { [String(seed)]: 1 } });
    }
    else {
      const name = int ? "randint" : "uniform";
      onChange({ type: "scipy", name, args: [], kwds: defaultsFor(distByName(name)) });
    }
  };

  const setChoice = (entries: [string, number][]) =>
    onChange({ type: "categorical", weights: Object.fromEntries(entries) });

  const pickDist = (name: string) =>
    onChange({ type: "scipy", name, args: [], kwds: defaultsFor(distByName(name), dist?.kwds ?? {}) });

  const setKwd = (p: string, v: number) =>
    onChange({ ...dist, args: [], kwds: { ...(dist.kwds ?? {}), [p]: v } });

  // Optional Bounded(...) wrapper: restrict an (often unbounded) distribution to
  // a finite [lo, hi] so it can size an observation space. Emits `bounds`/`mode`
  // on the scipy value, which the spec (de)serialiser + codegen understand.
  const bounded = Array.isArray(dist?.bounds);
  const bnd = bounded ? dist.bounds : int ? [1, 100] : [0, 1];
  const bmode: string = dist?.mode === "clip" ? "clip" : "truncate";
  const toggleBounded = (on: boolean) => {
    if (on) onChange({ ...dist, bounds: bnd, mode: bmode });
    else {
      const { bounds: _b, mode: _m, ...rest } = dist;
      onChange(rest);
    }
  };
  const setBound = (i: 0 | 1, n: number) => {
    const b = [Number(bnd[0]), Number(bnd[1])];
    b[i] = n;
    onChange({ ...dist, bounds: b, mode: bmode });
  };
  const setBMode = (m: string) => onChange({ ...dist, bounds: bnd, mode: m });

  return (
    <div className="value-field">
      <div className="vf-head">
        {label && <span className="vf-label">{label}</span>}
        <span className="vf-spacer" />
        <Picker
          className="vf-mode"
          searchable={false}
          value={mode}
          placeholder="mode"
          onChange={(m) => switchMode(m as Mode)}
          options={[
            { value: "fixed", label: "fixed" },
            { value: "range", label: "range (uniform)" },
            { value: "dist", label: "distribution" },
            ...(allowChoice
              ? [{ value: "choice", label: "choice (weighted)", description: "weighted mixture over fixed values" }]
              : []),
            ...(allowEnvelope
              ? [{ value: "envelope", label: "aircraft envelope", description: "per-aircraft draw within the flight envelope" }]
              : []),
          ]}
        />
      </div>
      {mode === "envelope" && (
        <div className="muted small">sampled per aircraft within the flight envelope</div>
      )}
      {mode === "choice" && (
        <div className="vf-dist">
          {Object.entries(choiceWeights).map(([k, w], i, entries) => (
            <div className="vf-row" key={i}>
              <NumInput
                className="vf-input"
                step="any"
                value={parseFloat(k)}
                onChange={(n) =>
                  setChoice(entries.map(([ek, ew], j) => (j === i ? [String(n), ew] : [ek, ew])))
                }
              />
              <span className="vf-sep">×</span>
              <NumInput
                className="vf-input"
                step="any"
                value={w}
                onChange={(n) =>
                  setChoice(entries.map(([ek, ew], j) => (j === i ? [ek, Math.max(n, 1e-9)] : [ek, ew])))
                }
              />
              <button
                className="link danger"
                disabled={entries.length <= 1}
                onClick={() => setChoice(entries.filter((_, j) => j !== i))}
              >
                ✕
              </button>
            </div>
          ))}
          <button
            className="link"
            onClick={() => setChoice([...Object.entries(choiceWeights), ["1", 1]])}
          >
            + value
          </button>
          <div className="muted small">value × relative weight — one value is drawn per episode</div>
        </div>
      )}
      {mode === "fixed" && (
        <NumInput
          className="vf-input"
          step={step}
          int={int}
          value={fixed}
          onChange={(n) => onChange(n)}
        />
      )}
      {mode === "range" && (
        <div className="vf-row">
          <NumInput
            className="vf-input"
            step={step}
            int={int}
            value={range.low}
            onChange={(n) => onChange({ type: "range", low: n, high: range.high })}
          />
          <span className="vf-sep">–</span>
          <NumInput
            className="vf-input"
            step={step}
            int={int}
            value={range.high}
            onChange={(n) => onChange({ type: "range", low: range.low, high: n })}
          />
        </div>
      )}
      {mode === "dist" && dist && (
        <div className="vf-dist">
          <Picker
            className="vf-distname"
            placeholder="distribution"
            value={dist.name ?? ""}
            onChange={pickDist}
            options={dists.map((d) => ({ value: d.name, label: d.name, description: d.signature }))}
          />
          {distInfo ? (
            <div className="vf-params">
              {distInfo.params.map((p) => (
                <label className="vf-param" key={p}>
                  <span>{p}</span>
                  <NumInput
                    step="any"
                    value={dist.kwds?.[p] ?? paramDefault(p)}
                    placeholder={String(paramDefault(p))}
                    onChange={(n) => setKwd(p, n)}
                  />
                </label>
              ))}
            </div>
          ) : (
            <div className="muted small">unknown distribution — pick one from the list above</div>
          )}
          {distInfo && (
            <div className="vf-bounds">
              <label className="vf-bound-toggle">
                <input
                  type="checkbox"
                  checked={bounded}
                  onChange={(e) => toggleBounded(e.target.checked)}
                />
                <span>bound to [lo, hi]</span>
              </label>
              {!bounded && distInfo.unbounded && (
                <span className="muted small">
                  unbounded support — set bounds to use as a count / size
                </span>
              )}
              {bounded && (
                <div className="vf-row">
                  <NumInput
                    className="vf-input"
                    step={step}
                    int={int}
                    value={bnd[0]}
                    onChange={(n) => setBound(0, n)}
                  />
                  <span className="vf-sep">–</span>
                  <NumInput
                    className="vf-input"
                    step={step}
                    int={int}
                    value={bnd[1]}
                    onChange={(n) => setBound(1, n)}
                  />
                  <Picker
                    className="vf-mode"
                    searchable={false}
                    value={bmode}
                    placeholder="mode"
                    onChange={setBMode}
                    options={[
                      { value: "truncate", label: "truncate", description: "renormalise onto [lo, hi]" },
                      { value: "clip", label: "clip", description: "clamp samples into [lo, hi]" },
                    ]}
                  />
                </div>
              )}
            </div>
          )}
          {distInfo && (
            <div className="muted small vf-sig">
              <code>
                {bounded && "Bounded("}
                {dist.name}({distInfo.params.map((p) => `${p}=${dist.kwds?.[p] ?? paramDefault(p)}`).join(", ")})
                {bounded && `, ${bnd[0]}, ${bnd[1]}, mode='${bmode}')`}
              </code>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
