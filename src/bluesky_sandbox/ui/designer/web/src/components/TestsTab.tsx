// The design's tests: situations placed by hand, and the value each field
// should give in them (spec.tests - see design_tests.py), run with every
// field's check of itself (bluesky_sandbox.checks). What an aircraft holds - its
// inputs, which are required, the ways it can be placed - is read from the
// design schema, and the fields from the design and the catalog: nothing here
// names one.
import { useEffect, useMemo, useRef, useState } from "react";
import { api, type CaseRunResult, type FieldCheckResult, type PlacedAircraft, type SpecDict } from "../api";
import { Page } from "./form";
import { Picker, type PickerOption } from "./panel/Picker";
import { Spinner } from "./Spinner";

type Prop = { type?: string | string[] };
type AircraftSchema = {
  properties: Record<string, Prop>;
  required: string[];
  placements: string[][];
};
// An aircraft is named by this key: what a case's `of` and `aircraft` refer to.
const NAME = "acid";

function aircraftSchema(schema: any): AircraftSchema | null {
  const items = schema?.properties?.tests?.properties?.situations?.items?.properties?.aircraft?.items;
  if (!items) return null;
  return {
    properties: items.properties ?? {},
    required: items.required ?? [],
    placements: (items.oneOf ?? []).map((o: any) => o.required as string[]),
  };
}

const types = (p?: Prop) => [p?.type ?? []].flat();
const isNumber = (p?: Prop) => types(p).some((t) => t === "number" || t === "integer");

// The design's own fields - every list of field references its env holds,
// whatever it is called - then the library's, from the catalog.
function fieldOptions(spec: SpecDict, catalog: any): PickerOption[] {
  const out: PickerOption[] = [];
  for (const [list, refs] of Object.entries(spec.env ?? {})) {
    if (!Array.isArray(refs)) continue;
    for (const ref of refs) {
      if (!ref || typeof ref !== "object" || typeof ref.field !== "string") continue;
      out.push({
        value: JSON.stringify(ref),
        label: refLabel(ref),
        category: `this design: ${list}`,
      });
    }
  }
  for (const option of catalog?.obs_fields ?? []) {
    out.push({
      value: JSON.stringify({ field: option.name }),
      label: option.name,
      description: option.doc,
      category: option.category ?? "library",
    });
  }
  return out;
}

function refLabel(ref: SpecDict): string {
  const kwargs = Object.entries(ref.kwargs ?? {})
    .map(([k, v]) => `${k}=${typeof v === "object" ? ((v as any)?.name ?? "…") : JSON.stringify(v)}`)
    .join(", ");
  return kwargs ? `${ref.field}(${kwargs})` : ref.field;
}

function unitOf(catalog: any, ref: SpecDict | undefined): string {
  const option = (catalog?.obs_fields ?? []).find((o: any) => o.name === ref?.field);
  const unit = option?.profile?.meta?.unit;
  return unit && unit !== "unitless" ? unit : "";
}

const fmt = (v: number | number[] | null | undefined) =>
  v == null ? "—" : Array.isArray(v) ? v.map((x) => +x.toPrecision(6)).join(", ") : String(+v.toPrecision(6));

export default function TestsTab({ spec, onChange }: { spec: SpecDict | null; onChange: (next: SpecDict) => void }) {
  const [schema, setSchema] = useState<AircraftSchema | null>(null);
  const [catalog, setCatalog] = useState<any>(null);
  const [placed, setPlaced] = useState<Record<string, PlacedAircraft[]>>({});
  const [placeError, setPlaceError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [fieldResults, setFieldResults] = useState<FieldCheckResult[] | null>(null);
  const [caseResults, setCaseResults] = useState<Record<number, CaseRunResult>>({});
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    api
      .designSchema()
      .then((s) => setSchema(aircraftSchema(s)))
      .catch(() => setSchema(null));
    api
      .catalogOnce()
      .then(setCatalog)
      .catch(() => setCatalog(null));
    return () => abort.current?.abort();
  }, []);

  const tests: SpecDict = spec?.tests ?? {};
  const situations: SpecDict[] = tests.situations ?? [];
  const cases: SpecDict[] = tests.cases ?? [];
  const situationsKey = JSON.stringify(situations);

  // Where each situation's aircraft are, placed as the checks place them.
  useEffect(() => {
    if (!spec || situations.length === 0) {
      setPlaced({});
      setPlaceError(null);
      return;
    }
    const timer = setTimeout(() => {
      api
        .testSituations(spec)
        .then((r) => {
          setPlaced(r.situations ?? {});
          setPlaceError(r.ok ? null : (r.errors ?? []).join("\n") || null);
        })
        .catch((e) => setPlaceError(String(e?.message ?? e)));
    }, 300);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [situationsKey]);

  const options = useMemo(() => (spec ? fieldOptions(spec, catalog) : []), [spec, catalog]);

  if (!spec) return <div className="form-page muted">Spec has a JSON error; fix it in the Code tab.</div>;

  const setTests = (patch: SpecDict) => {
    const next = { ...tests, ...patch };
    const empty = !(next.situations ?? []).length && !(next.cases ?? []).length;
    const { tests: _drop, ...rest } = spec;
    onChange(empty ? rest : { ...spec, tests: next });
  };
  const setSituations = (next: SpecDict[]) => setTests({ situations: next });
  const setCases = (next: SpecDict[]) => setTests({ cases: next });

  const run = async () => {
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    setRunning(true);
    setRunError(null);
    setFieldResults([]);
    setCaseResults({});
    try {
      await api.testDesign(
        spec,
        (r) => {
          if (r.kind === "field") setFieldResults((prev) => [...(prev ?? []), r]);
          else setCaseResults((prev) => ({ ...prev, [r.index]: r }));
        },
        controller.signal,
      );
    } catch (e: any) {
      if (!controller.signal.aborted) setRunError(String(e?.message ?? e));
    } finally {
      if (abort.current === controller) setRunning(false);
    }
  };

  const caseCount = Object.keys(caseResults).length;
  const failedCases = Object.values(caseResults).filter((r) => !r.ok).length;
  const failedFields = (fieldResults ?? []).filter((r) => !r.ok).length;

  return (
    <Page
      title="Tests"
      intro="Set up aircraft, then say what a field should read. Running the tests also checks each field against itself."
    >
      <div className="tests-page">
        <div className="tests-actions">
          <button className="run-btn" onClick={run} disabled={running}>
            {running ? <Spinner label="running the tests" /> : "▶"} Run tests
          </button>
          {fieldResults && !running && (
            <span className={failedCases + failedFields ? "error-text small" : "muted small"}>
              {caseCount} case{caseCount === 1 ? "" : "s"}, {failedCases} failed · {fieldResults.length} field
              {fieldResults.length === 1 ? "" : "s"} checked, {failedFields} failed
            </span>
          )}
        </div>
        {runError && <div className="error-text small">{runError}</div>}

        <section className="tests-section">
          <h3>Situations</h3>
          {placeError && <div className="error-text small">{placeError}</div>}
          {!schema && <div className="muted small">Loading…</div>}
          {schema &&
            situations.map((s, i) => (
              <SituationCard
                key={i}
                situation={s}
                schema={schema}
                placed={placed[s.name]}
                onChange={(next) => setSituations(situations.map((o, j) => (j === i ? next : o)))}
                onRemove={() => setSituations(situations.filter((_, j) => j !== i))}
              />
            ))}
          <button onClick={() => setSituations([...situations, { name: uniqueName(situations), aircraft: [] }])}>
            + situation
          </button>
        </section>

        <section className="tests-section">
          <h3>Cases</h3>
          {cases.length === 0 ? (
            <div className="muted small">No cases yet. Add one: a situation, a field, and the value it should read.</div>
          ) : (
            <table className="sample-table tests-cases">
              <thead>
                <tr>
                  <th>situation</th>
                  <th>field</th>
                  <th title="For pair fields: the other aircraft.">about</th>
                  <th title="Which aircraft to read. Default: the first one.">for</th>
                  <th>expected</th>
                  <th title="Passes if within either: an absolute difference, or a fraction of the expected value.">
                    tolerance abs / rel
                  </th>
                  <th>note</th>
                  <th>got</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {cases.map((c, i) => (
                  <CaseRow
                    key={i}
                    value={c}
                    situations={situations}
                    options={options}
                    unit={unitOf(catalog, c.field)}
                    result={caseResults[i]}
                    onChange={(next) => setCases(cases.map((o, j) => (j === i ? next : o)))}
                    onRemove={() => setCases(cases.filter((_, j) => j !== i))}
                  />
                ))}
              </tbody>
            </table>
          )}
          <button
            disabled={situations.length === 0}
            title={situations.length ? undefined : "add a situation first"}
            onClick={() =>
              setCases([
                ...cases,
                {
                  situation: situations[0]?.name,
                  expected: null,
                  tolerance: {},
                },
              ])
            }
          >
            + case
          </button>
        </section>

        <section className="tests-section">
          <h3>Field checks</h3>
          {fieldResults === null ? (
            <div className="muted small">
              Not run yet. Each field's batch values are compared with its one-at-a-time values, and with its{" "}
              <code>expected</code> if it defines one.
            </div>
          ) : (
            <ul className="tests-fields">
              {fieldResults.map((r) => (
                <li key={r.field} className={r.ok ? "ok" : "failed"}>
                  <span aria-hidden="true">{r.ok ? "✓" : "✗"}</span> {r.field}
                  {!r.ok && <div className="error-text small">{r.findings.join("; ")}</div>}
                </li>
              ))}
              {running && (
                <li className="muted">
                  <Spinner label="checking" />
                </li>
              )}
            </ul>
          )}
        </section>
      </div>
    </Page>
  );
}

function uniqueName(situations: SpecDict[]): string {
  const taken = new Set(situations.map((s) => s.name));
  let n = situations.length + 1;
  while (taken.has(`situation ${n}`)) n += 1;
  return `situation ${n}`;
}

function SituationCard({
  situation,
  schema,
  placed,
  onChange,
  onRemove,
}: {
  situation: SpecDict;
  schema: AircraftSchema;
  placed?: PlacedAircraft[];
  onChange: (next: SpecDict) => void;
  onRemove: () => void;
}) {
  const aircraft: SpecDict[] = situation.aircraft ?? [];
  const setAircraft = (next: SpecDict[]) => onChange({ ...situation, aircraft: next });
  const add = () => {
    // A copy of the one before it, named anew: the next aircraft is mostly the same.
    const last = aircraft[aircraft.length - 1];
    const taken = new Set(aircraft.map((a) => a[NAME]));
    let n = aircraft.length + 1;
    while (taken.has(`AC${n}`)) n += 1;
    setAircraft([...aircraft, { ...(last ?? {}), [NAME]: `AC${n}` }]);
  };
  return (
    <div className="tests-situation">
      <div className="tests-situation-head">
        <input
          className="form-input"
          aria-label="situation name"
          value={situation.name ?? ""}
          onChange={(e) => onChange({ ...situation, name: e.target.value })}
        />
        <label className="numfield inline" title="Which episode of the design to use.">
          <span>seed</span>
          <input
            type="number"
            value={situation.seed ?? 0}
            onChange={(e) => onChange({ ...situation, seed: Number(e.target.value) || 0 })}
          />
        </label>
        <button className="chip-x" title="remove the situation" onClick={onRemove}>
          ✕
        </button>
      </div>
      <div className="tests-situation-body">
        <div className="tests-aircraft">
          {aircraft.map((a, i) => (
            <AircraftRow
              key={i}
              value={a}
              schema={schema}
              first={i === 0}
              onChange={(next) => setAircraft(aircraft.map((o, j) => (j === i ? next : o)))}
              onRemove={() => setAircraft(aircraft.filter((_, j) => j !== i))}
            />
          ))}
          <button onClick={add}>+ aircraft</button>
        </div>
        <SituationSketch aircraft={placed ?? []} />
      </div>
    </div>
  );
}

function AircraftRow({
  value,
  schema,
  first,
  onChange,
  onRemove,
}: {
  value: SpecDict;
  schema: AircraftSchema;
  first: boolean;
  onChange: (next: SpecDict) => void;
  onRemove: () => void;
}) {
  const groups = schema.placements;
  const placing = new Set(groups.flat());
  // The way it is placed: the one whose inputs it holds - or, before any is
  // filled in, the one picked.
  const holds = groups.findIndex((g) => g.some((k) => value[k] != null));
  const [picked, setPicked] = useState<number>(Math.max(0, holds));
  const group = groups[holds >= 0 ? holds : picked] ?? [];
  const rest = Object.keys(schema.properties).filter((k) => k !== NAME && !placing.has(k));
  const set = (key: string, v: unknown) => {
    const next = { ...value };
    if (v === null || v === undefined || v === "") delete next[key];
    else next[key] = v;
    onChange(next);
  };
  const place = (index: number) => {
    const next = { ...value };
    for (const k of placing) delete next[k];
    setPicked(index);
    onChange(next);
  };
  const input = (key: string) => {
    const prop = schema.properties[key];
    const required = schema.required.includes(key) || group.includes(key);
    return (
      <label key={key} className={`numfield inline${required && value[key] == null ? " missing" : ""}`}>
        <span>{key}</span>
        <input
          type={isNumber(prop) ? "number" : "text"}
          value={value[key] ?? ""}
          onChange={(e) =>
            set(key, isNumber(prop) ? (e.target.value === "" ? null : Number(e.target.value)) : e.target.value)
          }
        />
      </label>
    );
  };
  return (
    <div className={first ? "tests-aircraft-row ownship" : "tests-aircraft-row"}>
      <input
        className="form-input tests-acid"
        aria-label="callsign"
        title={first ? "The ownship. Cases read this aircraft unless they name another." : undefined}
        value={value[NAME] ?? ""}
        onChange={(e) => set(NAME, e.target.value)}
      />
      <select
        className="form-input"
        aria-label="placed by"
        value={groups.indexOf(group)}
        onChange={(e) => place(Number(e.target.value))}
      >
        {groups.map((g, i) => (
          <option key={i} value={i}>
            by {g.join(", ")}
          </option>
        ))}
      </select>
      {group.map(input)}
      {rest.map(input)}
      <button className="chip-x" title="remove the aircraft" onClick={onRemove}>
        ✕
      </button>
    </div>
  );
}

// The situation from above, in nautical miles around its first aircraft: each
// aircraft where the checks place it, pointing along its track.
function SituationSketch({ aircraft }: { aircraft: PlacedAircraft[] }) {
  if (aircraft.length === 0) return <div className="tests-sketch muted small">No aircraft yet.</div>;
  const [origin] = aircraft;
  const coslat = Math.cos((origin.lat * Math.PI) / 180);
  const at = aircraft.map((a) => ({
    ...a,
    x: (a.lon - origin.lon) * 60 * coslat,
    y: (a.lat - origin.lat) * 60,
  }));
  const reach = Math.max(2, ...at.map((a) => Math.max(Math.abs(a.x), Math.abs(a.y)))) * 1.6;
  const size = 180;
  const px = (v: number) => (v / reach) * (size / 2);
  return (
    <svg
      className="tests-sketch"
      viewBox={`${-size / 2} ${-size / 2} ${size} ${size}`}
      role="img"
      aria-label="the situation from above"
    >
      <circle r={px(reach / 1.6)} className="tests-sketch-range" />
      {at.map((a, i) => (
        <g key={a.acid} transform={`translate(${px(a.x)} ${-px(a.y)}) rotate(${a.track_deg})`}>
          <path d="M0 -7 L5 6 L0 3 L-5 6 Z" className={i === 0 ? "tests-sketch-own" : "tests-sketch-other"} />
          <text transform={`rotate(${-a.track_deg})`} x={8} y={4} className="tests-sketch-label">
            {a.acid}
          </text>
        </g>
      ))}
      <text x={-size / 2 + 4} y={size / 2 - 4} className="tests-sketch-scale">
        {(reach / 1.6).toFixed(reach > 10 ? 0 : 1)} nm
      </text>
    </svg>
  );
}

function CaseRow({
  value,
  situations,
  options,
  unit,
  result,
  onChange,
  onRemove,
}: {
  value: SpecDict;
  situations: SpecDict[];
  options: PickerOption[];
  unit: string;
  result?: CaseRunResult;
  onChange: (next: SpecDict) => void;
  onRemove: () => void;
}) {
  const situation = situations.find((s) => s.name === value.situation);
  const names: string[] = (situation?.aircraft ?? []).map((a: SpecDict) => a[NAME]).filter(Boolean);
  const set = (key: string, v: unknown) => {
    const next = { ...value };
    if (v === null || v === undefined || v === "") delete next[key];
    else next[key] = v;
    onChange(next);
  };
  const tolerance = value.tolerance ?? {};
  const setTolerance = (key: string, v: string) => {
    const next = { ...tolerance };
    if (v === "") delete next[key];
    else next[key] = Number(v);
    onChange({ ...value, tolerance: next });
  };
  const [expectedText, setExpectedText] = useState<string>(fmtExpected(value.expected));
  useEffect(() => setExpectedText(fmtExpected(value.expected)), [value.expected]);
  const fieldKey = value.field ? JSON.stringify(value.field) : "";
  const fieldOptions =
    options.some((o) => o.value === fieldKey) || !value.field
      ? options
      : [
          {
            value: fieldKey,
            label: refLabel(value.field),
            category: "this case",
          },
          ...options,
        ];
  return (
    <tr className={result ? (result.ok ? "ok" : "failed") : undefined}>
      <td>
        <select className="form-input" value={value.situation ?? ""} onChange={(e) => set("situation", e.target.value)}>
          {situations.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name}
            </option>
          ))}
        </select>
      </td>
      <td>
        <Picker
          placeholder="+ field…"
          value={fieldKey}
          options={fieldOptions}
          onChange={(v) => v && set("field", JSON.parse(v))}
        />
      </td>
      <td>
        <select className="form-input" value={value.of ?? ""} onChange={(e) => set("of", e.target.value)}>
          <option value="">—</option>
          {names.map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
      </td>
      <td>
        <select className="form-input" value={value.aircraft ?? ""} onChange={(e) => set("aircraft", e.target.value)}>
          <option value="">{names[0] ?? "—"}</option>
          {names.slice(1).map((n) => (
            <option key={n}>{n}</option>
          ))}
        </select>
      </td>
      <td>
        <span className="form-unit">
          <input
            className="form-input"
            aria-label="expected"
            placeholder="a number, or several: 1, 2"
            value={expectedText}
            onChange={(e) => {
              setExpectedText(e.target.value);
              const parsed = parseExpected(e.target.value);
              if (parsed !== undefined) set("expected", parsed);
            }}
          />
          {unit && <span className="unit">{unit}</span>}
        </span>
      </td>
      <td className="tests-tolerance">
        <input
          type="number"
          className="form-input"
          aria-label="absolute tolerance"
          placeholder="0"
          value={tolerance.abs ?? ""}
          onChange={(e) => setTolerance("abs", e.target.value)}
        />
        <input
          type="number"
          className="form-input"
          aria-label="relative tolerance"
          placeholder="0"
          step={0.01}
          value={tolerance.rel ?? ""}
          onChange={(e) => setTolerance("rel", e.target.value)}
        />
      </td>
      <td>
        <input
          className="form-input"
          aria-label="note"
          value={value.note ?? ""}
          onChange={(e) => set("note", e.target.value)}
        />
      </td>
      <td className="tests-got" title={result?.error ?? undefined}>
        {result &&
          (result.error ? (
            <span className="error-text small">⚠ {result.error}</span>
          ) : (
            <>
              <span aria-hidden="true">{result.ok ? "✓" : "✗"}</span> {fmt(result.got)}
              {!result.ok && result.got != null && (
                <button
                  className="tests-take"
                  title="Use this value as the expected one. Only do this after checking it yourself."
                  onClick={() => set("expected", result.got)}
                >
                  use
                </button>
              )}
            </>
          ))}
      </td>
      <td>
        <button className="chip-x" title="remove the case" onClick={onRemove}>
          ✕
        </button>
      </td>
    </tr>
  );
}

function fmtExpected(v: unknown): string {
  if (v == null) return "";
  return Array.isArray(v) ? v.join(", ") : String(v);
}

// "500" -> 500, "1, 2" -> [1, 2], "" -> null; undefined while it is not a number yet.
function parseExpected(text: string): number | number[] | null | undefined {
  const parts = text.split(",").map((p) => p.trim());
  if (parts.length === 1 && parts[0] === "") return null;
  const numbers = parts.map(Number);
  if (parts.some((p) => p === "") || numbers.some((n) => !Number.isFinite(n))) return undefined;
  return numbers.length === 1 ? numbers[0] : numbers;
}
