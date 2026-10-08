import { useEffect, useState } from "react";
import { api, type NavWaypoint, type SpecDict } from "../api";
import {
  ALTITUDE_TYPES,
  FOOTPRINT_TYPES,
  clone,
  defaultRegion,
  fmtSampled,
  footprintCenter,
  makeAltitude,
  makeFootprint,
  repValue,
} from "../specHelpers";
import { Picker } from "./panel/Picker";
import { usePointShapes } from "../pointShapes";
import { ValueField } from "./panel/ValueField";
import type { SampledValue } from "./panel/ValueField";

// One-line description of a footprint's shape + size, shown next to the shape
// picker so the geometry reads at a glance while the numeric fields stay folded.
// Sampled params show as their range/dist (e.g. "r 40–60 nm").
export function summarizeFootprint(fp: SpecDict | undefined): string {
  if (!fp) return "";
  switch (fp.type) {
    case "box":
      return `${(repValue(fp.lat_max_deg) - repValue(fp.lat_min_deg)).toFixed(2)}×${(repValue(fp.lon_max_deg) - repValue(fp.lon_min_deg)).toFixed(2)}°`;
    case "disk":
      return `r ${fmtSampled(fp.radius_nm)} nm`;
    case "sector":
      return `r ${fmtSampled(fp.radius_nm)} nm · 2×${fmtSampled(fp.half_angle_deg)}°`;
    case "annular_sector":
      return `${fmtSampled(fp.inner_radius_nm)}…${fmtSampled(fp.outer_radius_nm)} nm`;
    case "polygon":
      return `${(fp.coords ?? []).length} pts`;
    case "boolean":
      return `${fp.op} (A∘B)`;
    case "generated":
      return `${fp.generator} · each episode`;
    case "point":
      return fp.center ? `${Number(fp.center.lat_deg).toFixed(3)}, ${Number(fp.center.lon_deg).toFixed(3)}` : "point";
    default:
      return fp.type;
  }
}

// Editors for every bounds primitive: all footprints (box / disk / sector /
// annular_sector / polygon) and altitude bands (constant / linear /
// radial / vertex). Used for the airspace, queryable regions, and spawn regions.

export function NumField({
  label,
  value,
  onChange,
  step = 0.01,
  min,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
}) {
  return (
    <label className="numfield">
      <span>{label}</span>
      <input
        type="number"
        step={step}
        min={min}
        value={Number.isFinite(value) ? value : ""}
        onChange={(e) => {
          let v = parseFloat(e.target.value);
          if (!Number.isFinite(v)) return;
          if (min != null && v < min) v = min;
          onChange(v);
        }}
      />
    </label>
  );
}

function LatLonField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: { lat_deg: number; lon_deg: number };
  onChange: (v: { lat_deg: number; lon_deg: number }) => void;
}) {
  return (
    <div className="latlon">
      <span className="latlon-label">{label}</span>
      <NumField label="lat" value={value?.lat_deg} onChange={(v) => onChange({ ...value, lat_deg: v })} />
      <NumField label="lon" value={value?.lon_deg} onChange={(v) => onChange({ ...value, lon_deg: v })} />
    </div>
  );
}

// A point: at a lat/lon, or at a navdb fix - found by name, nearest the point
// first (names repeat worldwide), and drawn where it is. Moving it off the fix
// makes it a lat/lon again.
function PointFields({ fp, set }: { fp: SpecDict; set: (mut: (f: SpecDict) => void) => void }) {
  const near: [number, number] | undefined = fp.center ? [fp.center.lat_deg, fp.center.lon_deg] : undefined;
  return (
    <>
      <label className="numfield inline">
        <span>navdb fix</span>
        <FixSearch
          value={fp.fix ?? ""}
          near={near}
          onPick={(w) =>
            set((f) => {
              f.fix = w.ident;
              f.center = { lat_deg: w.lat_deg, lon_deg: w.lon_deg };
            })
          }
          onClear={() => set((f) => delete f.fix)}
        />
      </label>
      <LatLonField
        label="at"
        value={fp.center}
        onChange={(c) =>
          set((f) => {
            f.center = c;
            delete f.fix;
          })
        }
      />
    </>
  );
}

// Type a fix's name; pick from the navdb's matches - exact, then prefix, then
// the rest, each nearest `near` first. Arrows move, Enter picks, Escape closes.
function FixSearch({
  value,
  near,
  onPick,
  onClear,
}: {
  value: string;
  near?: [number, number];
  onPick: (w: NavWaypoint) => void;
  onClear: () => void;
}) {
  const [text, setText] = useState(value);
  const [hits, setHits] = useState<NavWaypoint[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  useEffect(() => setText(value), [value]);
  useEffect(() => {
    const q = text.trim();
    if (!open || q.length < 2 || q === value) {
      setHits([]);
      return;
    }
    let live = true;
    const t = window.setTimeout(() => {
      api
        .search(q, 8, near)
        .then((r) => live && (setHits(r.waypoints), setActive(0)))
        .catch(() => live && setHits([]));
    }, 200);
    return () => {
      live = false;
      window.clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, open]);
  const pick = (w: NavWaypoint) => {
    onPick(w);
    setText(w.ident);
    setOpen(false);
  };
  const nm = (w: NavWaypoint) => {
    if (!near) return "";
    const cos = Math.cos((near[0] * Math.PI) / 180);
    const d = 60 * Math.hypot(w.lat_deg - near[0], (w.lon_deg - near[1]) * cos);
    return d < 10 ? `${d.toFixed(1)} nm` : `${Math.round(d)} nm`;
  };
  return (
    <span className="fix-search">
      <input
        value={text}
        placeholder="search - none: a lat/lon"
        role="combobox"
        aria-expanded={open && hits.length > 0}
        aria-autocomplete="list"
        onChange={(e) => {
          setText(e.target.value.toUpperCase());
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => window.setTimeout(() => (setOpen(false), setText(value)), 150)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setActive((a) => Math.min(a + 1, hits.length - 1));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setActive((a) => Math.max(a - 1, 0));
          } else if (e.key === "Enter" && hits[active]) {
            e.preventDefault();
            pick(hits[active]);
          } else if (e.key === "Escape") {
            setOpen(false);
            setText(value);
          }
        }}
      />
      {value && (
        <button type="button" className="link" onClick={onClear} title="not a fix: a lat/lon">
          clear
        </button>
      )}
      {open && hits.length > 0 && (
        <ul className="fix-hits" role="listbox">
          {hits.map((w, i) => (
            <li
              key={`${w.ident}:${w.lat_deg}:${w.lon_deg}:${i}`}
              role="option"
              aria-selected={i === active}
              className={i === active ? "on" : ""}
              onMouseDown={(e) => {
                e.preventDefault();
                pick(w);
              }}
              onMouseEnter={() => setActive(i)}
            >
              <b>{w.ident}</b>
              {w.wptype && <span className="muted"> {w.wptype}</span>}
              <span className="spacer" />
              <span className="muted">{nm(w) || `${w.lat_deg.toFixed(2)}, ${w.lon_deg.toFixed(2)}`}</span>
            </li>
          ))}
        </ul>
      )}
    </span>
  );
}

function BandField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: [number, number];
  onChange: (v: [number, number]) => void;
}) {
  const [lo, hi] = value ?? [0, 0];
  return (
    <div className="latlon">
      <span className="latlon-label">{label}</span>
      <NumField label="min ft" step={500} min={0} value={lo} onChange={(v) => onChange([v, hi])} />
      <NumField label="max ft" step={500} min={0} value={hi} onChange={(v) => onChange([lo, v])} />
    </div>
  );
}

function offsetLatLon(center: { lat_deg: number; lon_deg: number }, bearingDeg: number, distanceNm: number): [number, number] {
  const angle = (bearingDeg * Math.PI) / 180;
  const cosLat = Math.max(0.01, Math.cos((center.lat_deg * Math.PI) / 180));
  return [
    center.lat_deg + (distanceNm / 60) * Math.cos(angle),
    center.lon_deg + ((distanceNm / 60) * Math.sin(angle)) / cosLat,
  ];
}

function footprintVertices(fp: SpecDict | undefined): [number, number][] {
  // Sampled params contribute their representative value, matching the map.
  if (!fp) return [];
  switch (fp.type) {
    case "box":
      return [
        [repValue(fp.lat_min_deg), repValue(fp.lon_max_deg)],
        [repValue(fp.lat_max_deg), repValue(fp.lon_max_deg)],
        [repValue(fp.lat_max_deg), repValue(fp.lon_min_deg)],
        [repValue(fp.lat_min_deg), repValue(fp.lon_min_deg)],
      ];
    case "polygon":
      return fp.coords ?? [];
    case "disk": {
      const n = fp.n_vertices ?? 72;
      return Array.from({ length: n }, (_, i) => offsetLatLon(fp.center, (360 * i) / n, repValue(fp.radius_nm)));
    }
    case "sector": {
      const n = fp.n_vertices ?? 24;
      const bearing = repValue(fp.bearing_deg);
      const half = repValue(fp.half_angle_deg);
      return [
        [fp.center.lat_deg, fp.center.lon_deg],
        ...Array.from({ length: n + 1 }, (_, i) =>
          offsetLatLon(fp.center, bearing - half + (2 * half * i) / n, repValue(fp.radius_nm)),
        ),
      ];
    }
    case "annular_sector": {
      const n = fp.n_vertices ?? 48;
      const bearing = repValue(fp.bearing_deg);
      const half = repValue(fp.half_angle_deg);
      const start = bearing - half;
      const end = bearing + half;
      const step = (end - start) / n;
      const inner = Array.from({ length: n + 1 }, (_, i) => offsetLatLon(fp.center, start + step * i, repValue(fp.inner_radius_nm)));
      const outer = Array.from({ length: n + 1 }, (_, i) => offsetLatLon(fp.center, end - step * i, repValue(fp.outer_radius_nm)));
      return [...inner, ...outer];
    }
    default:
      return [];
  }
}

function syncVertexAltitude(alt: SpecDict | null | undefined, fp: SpecDict | undefined): SpecDict | null | undefined {
  if (!alt || alt.type !== "vertex") return alt;
  const vertices = footprintVertices(fp);
  if (vertices.length < 3) return alt;
  return {
    ...alt,
    vertices,
    min_values_ft: vertexValues(alt.min_values_ft, vertices.length, 0),
    max_values_ft: vertexValues(alt.max_values_ft, vertices.length, 10000),
  };
}

// A footprint editor: shape-type selector + type-specific fields. Recursive,
// so a "boolean" footprint composes two nested footprint editors. At the top
// level (`foldFields`) the numeric coordinate fields collapse behind a summary,
// since the map drag handles are the primary way to edit the shape.
export function FootprintEditor({
  footprint,
  onChange,
  center,
  foldFields = false,
  regionNames = [],
  allowPoint = false,
  onlyPoint = false,
}: {
  footprint: SpecDict;
  onChange: (f: SpecDict) => void;
  center: [number, number];
  foldFields?: boolean;
  // Named regions a generator can refer to (a parent, a region to stay in).
  regionNames?: string[];
  // Offer a point - not where an area is needed, nor inside a combined shape.
  allowPoint?: boolean;
  // Only a point: a waypoint's.
  onlyPoint?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const set = (mut: (f: SpecDict) => void) => {
    const next = clone(footprint);
    mut(next);
    onChange(next);
  };
  const picker = (
    <label className="numfield inline">
      <span>type</span>
      <Picker
        searchable={false}
        placeholder="type"
        value={footprint.type}
        onChange={(v) => onChange(makeFootprint(v, center[0], center[1]))}
        options={FOOTPRINT_TYPES.filter((t) =>
          onlyPoint ? t === "point" || t === footprint.type : allowPoint || t !== "point" || footprint.type === "point",
        ).map((t) => ({
          value: t,
          description: t === "point" ? "a position, no area: a fix, a spawn point" : undefined,
        }))}
      />
    </label>
  );
  if (!foldFields) {
    return (
      <div className="footprint-editor">
        {picker}
        <FootprintFields fp={footprint} set={set} center={center} regionNames={regionNames} />
      </div>
    );
  }
  return (
    <div className="footprint-editor">
      {picker}
      <button type="button" className="coord-disclosure" onClick={() => setOpen((o) => !o)}>
        <span className="chev">{open ? "▾" : "▸"}</span>
        coordinates
        <span className="coord-summary">{summarizeFootprint(footprint)}</span>
      </button>
      {open && <FootprintFields fp={footprint} set={set} center={center} regionNames={regionNames} />}
    </div>
  );
}

function FootprintFields({
  fp,
  set,
  center,
  regionNames = [],
}: {
  fp: SpecDict;
  set: (mut: (f: SpecDict) => void) => void;
  center: [number, number];
  regionNames?: string[];
}) {
  if (fp.type === "generated") {
    return <GeneratedFields fp={fp} set={set} center={center} regionNames={regionNames} />;
  }
  if (fp.type === "boolean") {
    return (
      <div className="boolean-fp">
        <label className="numfield inline">
          <span>op</span>
          <Picker
            searchable={false}
            placeholder="op"
            value={fp.op}
            onChange={(v) => set((f) => (f.op = v))}
            options={[
              { value: "union", label: "union (A ∪ B)" },
              { value: "intersection", label: "intersection (A ∩ B)" },
              { value: "difference", label: "difference (A − B)" },
            ]}
          />
        </label>
        <div className="operand">
          <div className="operand-tag">A</div>
          <FootprintEditor footprint={fp.left} onChange={(l) => set((f) => (f.left = l))} center={center} />
        </div>
        <div className="operand">
          <div className="operand-tag">B</div>
          <FootprintEditor footprint={fp.right} onChange={(r) => set((f) => (f.right = r))} center={center} />
        </div>
      </div>
    );
  }
  // Scalar shape params are ValueFields: fixed, per-episode range, or a scipy
  // distribution - the same sampled-value encoding the spec accepts everywhere
  // else (n_aircraft, rotation angle). Centers stay plain lat/lon.
  const sampled = (
    label: string,
    key: string,
    step = 0.5,
  ) => (
    <ValueField
      label={label}
      step={step}
      value={fp[key] as SampledValue}
      onChange={(v) => set((f) => (f[key] = v))}
    />
  );
  switch (fp.type) {
    case "box":
      return (
        <div className="fp-params">
          {sampled("lat min", "lat_min_deg", 0.01)}
          {sampled("lat max", "lat_max_deg", 0.01)}
          {sampled("lon min", "lon_min_deg", 0.01)}
          {sampled("lon max", "lon_max_deg", 0.01)}
        </div>
      );
    case "disk":
      return (
        <>
          <LatLonField label="center" value={fp.center} onChange={(c) => set((f) => (f.center = c))} />
          <div className="fp-params">{sampled("radius nm", "radius_nm")}</div>
        </>
      );
    case "point":
      return <PointFields fp={fp} set={set} />;
    case "sector":
      return (
        <>
          <LatLonField label="center" value={fp.center} onChange={(c) => set((f) => (f.center = c))} />
          <div className="fp-params">
            {sampled("radius nm", "radius_nm")}
            {sampled("bearing°", "bearing_deg", 1)}
            {sampled("half angle°", "half_angle_deg", 1)}
          </div>
        </>
      );
    case "annular_sector":
      return (
        <>
          <LatLonField label="center" value={fp.center} onChange={(c) => set((f) => (f.center = c))} />
          <div className="fp-params">
            {sampled("inner nm", "inner_radius_nm")}
            {sampled("outer nm", "outer_radius_nm")}
            {sampled("bearing°", "bearing_deg", 1)}
            {sampled("half angle°", "half_angle_deg", 1)}
          </div>
        </>
      );
    case "polygon":
      return <CoordList coords={fp.coords ?? []} onChange={(c) => set((f) => (f.coords = c))} />;
    default:
      return <div className="muted">{fp.type}</div>;
  }
}

// A generator's or a placement's parameters, as the catalog describes them
// (read from its code) - each by the kind of input it takes.
type ClassParam = { name: string; kind: string; default: any };
type GeneratorInfo = { name: string; doc: string; params: ClassParam[]; partition?: string | null };
type PlacementInfo = { name: string; doc: string; params: ClassParam[] };
type MotionInfo = PlacementInfo;
const CONSTRAINTS = new Set(["contains", "within", "min_area_nm2", "max_tries"]);

function useCatalogList<T>(key: "generators" | "placements" | "motions"): T[] {
  const [items, setItems] = useState<T[]>([]);
  useEffect(() => {
    api
      .catalogOnce()
      .then((c: any) => setItems(c?.[key] ?? []))
      .catch(() => undefined);
  }, [key]);
  return items;
}

// One editor per param, by its kind: a point, a named region (or several), a
// number / range / distribution, a whole number, a flag. Points (``contains``)
// are set in the design's JSON or in code.
function ParamFields({
  params,
  values,
  onChange,
  center,
  regionNames,
}: {
  params: ClassParam[];
  values: SpecDict;
  onChange: (name: string, value: any) => void;
  center: [number, number];
  regionNames: string[];
}) {
  const points = usePointShapes();
  const field = (p: ClassParam) => {
    const value = values[p.name];
    const label = p.name.replace(/_/g, " ");
    switch (p.kind) {
      case "latlon":
        return (
          <LatLonField
            key={p.name}
            label={label}
            value={value ?? p.default ?? { lat_deg: center[0], lon_deg: center[1] }}
            onChange={(v) => onChange(p.name, v)}
          />
        );
      case "region":
        // A region to be in: an area - a point has no inside.
        return (
          <label key={p.name} className="numfield inline">
            <span>{label}</span>
            <Picker
              searchable={false}
              placeholder="none"
              value={value?.ref ?? ""}
              onChange={(v) => onChange(p.name, v ? { ref: v } : undefined)}
              options={[
                { value: "", label: "none" },
                ...regionNames.filter((n) => !points.has(n.replace(/\.\d+$/, ""))).map((n) => ({ value: n })),
              ]}
            />
          </label>
        );
      case "regions": {
        const chosen = new Set(((value ?? []) as SpecDict[]).map((r) => r?.ref));
        return (
          <div key={p.name} className="param-regions">
            <span className="sub-label">{label}</span>
            <div className="chips">
              {regionNames.length === 0 && <span className="muted small">no other regions</span>}
              {regionNames.map((n) => (
                <button
                  key={n}
                  type="button"
                  className={chosen.has(n) ? "chip member on" : "chip member"}
                  aria-pressed={chosen.has(n)}
                  onClick={() => {
                    const next = chosen.has(n) ? [...chosen].filter((c) => c !== n) : [...chosen, n];
                    onChange(p.name, next.length ? next.map((ref) => ({ ref })) : undefined);
                  }}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
        );
      }
      case "value":
        // A param that may be left out (None): on or off, then its value.
        if (p.default === null) {
          return (
            <div key={p.name} className="param-optional">
              <label className="radio modal-check">
                <input
                  type="checkbox"
                  checked={value != null}
                  onChange={(e) => onChange(p.name, e.target.checked ? { type: "range", low: 0, high: 360 } : undefined)}
                />
                {label}
              </label>
              {value != null && (
                <ValueField label={label} step={1} value={value as SampledValue} onChange={(v) => onChange(p.name, v)} />
              )}
            </div>
          );
        }
        return (
          <ValueField
            key={p.name}
            label={label}
            step={0.5}
            value={(value ?? p.default) as SampledValue}
            onChange={(v) => onChange(p.name, v)}
          />
        );
      case "bool":
        return (
          <label key={p.name} className="radio modal-check">
            <input
              type="checkbox"
              checked={Boolean(value ?? p.default)}
              onChange={(e) => onChange(p.name, e.target.checked === Boolean(p.default) ? undefined : e.target.checked)}
            />
            {label}
          </label>
        );
      case "int":
      case "float":
        return (
          <NumField
            key={p.name}
            label={label}
            step={p.kind === "int" ? 1 : 0.5}
            value={value ?? p.default ?? 0}
            onChange={(v) => onChange(p.name, p.kind === "int" ? Math.round(v) : v)}
          />
        );
      default:
        return null;
    }
  };
  return <>{params.map(field)}</>;
}

function GeneratedFields({
  fp,
  set,
  center,
  regionNames,
}: {
  fp: SpecDict;
  set: (mut: (f: SpecDict) => void) => void;
  center: [number, number];
  regionNames: string[];
}) {
  const generators = useCatalogList<GeneratorInfo>("generators");
  const info = generators.find((g) => g.name === fp.generator);
  const params: SpecDict = fp.params ?? {};
  const setParam = (name: string, value: any) =>
    set((f) => {
      f.params = { ...(f.params ?? {}) };
      if (value === undefined || value === null || value === "") delete f.params[name];
      else f.params[name] = value;
    });
  const own = (info?.params ?? []).filter((p) => !CONSTRAINTS.has(p.name));
  const constraints = (info?.params ?? []).filter((p) => CONSTRAINTS.has(p.name));
  const fields = (list: ClassParam[]) => (
    <ParamFields params={list} values={params} onChange={setParam} center={center} regionNames={regionNames} />
  );
  return (
    <div className="generated-fp">
      <label className="numfield inline">
        <span>generator</span>
        <Picker
          searchable={false}
          placeholder="generator"
          value={fp.generator}
          onChange={(v) =>
            set((f) => {
              f.generator = v;
              // Keep what the new generator also takes; drop the rest.
              const takes = new Set((generators.find((g) => g.name === v)?.params ?? []).map((p) => p.name));
              f.params = Object.fromEntries(Object.entries(f.params ?? {}).filter(([k]) => takes.has(k)));
            })
          }
          options={generators.map((g) => ({ value: g.name, description: g.doc }))}
        />
      </label>
      {info && <div className="muted small field-doc">{info.doc.replace(/``/g, "")}</div>}
      <div className="fp-params generated-params">{fields(own)}</div>
      {constraints.length > 0 && (
        <details className="generated-constraints">
          <summary>constraints</summary>
          <div className="fp-params">{fields(constraints)}</div>
        </details>
      )}
      <div className="muted small">
        Drawn anew each episode
        {info?.partition
          ? `, as ${params[info.partition] ?? info.params.find((p) => p.name === info.partition)?.default} shapes - “name.0”, “name.1”, … - each a region elements can refer to`
          : ""}
        .
      </div>
    </div>
  );
}

// How the region moves during the episode: motions, in order - each applied to
// the shape the ones before it left. None: it stays put.
export function MotionEditor({
  motion,
  onChange,
  cadence,
  onCadence,
  center,
  regionNames,
}: {
  motion: SpecDict[] | undefined;
  onChange: (m: SpecDict[] | null) => void;
  // How often the environment moves it: every step (the default) or substep.
  cadence: string | undefined;
  onCadence: (u: string | null) => void;
  center: [number, number];
  regionNames: string[];
}) {
  const motions = useCatalogList<MotionInfo>("motions");
  const list = motion ?? [];
  const update = (i: number, next: SpecDict | null) => {
    const out = list.flatMap((m, j) => (j !== i ? [m] : next ? [next] : []));
    onChange(out.length ? out : null);
  };
  return (
    <div className="motion-editor">
      <div className="be-head">
        <span className="be-title">moves</span>
        <button
          type="button"
          className="link"
          disabled={!motions.length}
          onClick={() => onChange([...list, { type: motions[0]?.name ?? "Drift" }])}
        >
          + motion
        </button>
      </div>
      {list.length === 0 && <div className="muted small">stays put during the episode</div>}
      {list.map((m, i) => {
        const info = motions.find((x) => x.name === m.type);
        const setParam = (name: string, value: any) => {
          const next: SpecDict = { ...m };
          if (value === undefined || value === null || value === "") delete next[name];
          else next[name] = value;
          update(i, next);
        };
        return (
          <div className="motion-item" key={i}>
            <div className="row between">
              <Picker
                searchable={false}
                placeholder="motion"
                value={m.type}
                onChange={(v) => update(i, { type: v })}
                options={motions.map((x) => ({ value: x.name, description: x.doc.replace(/``/g, "") }))}
              />
              <button type="button" className="chip-x" aria-label={`remove ${m.type}`} onClick={() => update(i, null)}>
                ✕
              </button>
            </div>
            {info && (
              <div className="fp-params">
                <ParamFields params={info.params} values={m} onChange={setParam} center={center} regionNames={regionNames} />
              </div>
            )}
          </div>
        );
      })}
      {list.length > 0 && (
        <>
          <div className="muted small">Each motion's ranges are drawn once an episode. The map shows the next hour, faint.</div>
          <details className="motion-advanced">
            <summary>advanced</summary>
            <label className="numfield inline">
              <span>update</span>
              <Picker
                searchable={false}
                placeholder="every step"
                value={cadence ?? "step"}
                onChange={(v) => onCadence(v === "step" ? null : v)}
                options={[
                  {
                    value: "step",
                    label: "every step",
                    description: "moved once each environment step, at its end - the default; between, it holds",
                  },
                  {
                    value: "substep",
                    label: "every substep",
                    description: "moved each simulation substep: what is inside is of the shape at each, at more compute",
                  },
                ]}
              />
            </label>
          </details>
        </>
      )}
    </div>
  );
}

// Where the region sits each episode: where it is drawn (fixed), or a
// placement's draw - after its shape, so a shape can be both drawn and placed.
function PlacementEditor({
  placement,
  onChange,
  center,
  regionNames,
}: {
  placement: SpecDict | null | undefined;
  onChange: (p: SpecDict | null) => void;
  center: [number, number];
  regionNames: string[];
}) {
  const placements = useCatalogList<PlacementInfo>("placements");
  const info = placements.find((p) => p.name === placement?.type);
  const setParam = (name: string, value: any) => {
    const next: SpecDict = { ...(placement ?? {}) };
    if (value === undefined || value === null || value === "") delete next[name];
    else next[name] = value;
    onChange(next);
  };
  return (
    <div className="placement-editor">
      <label className="numfield inline">
        <span>placed</span>
        <Picker
          searchable={false}
          placeholder="where drawn"
          value={placement?.type ?? ""}
          onChange={(v) => onChange(v ? { type: v } : null)}
          options={[
            { value: "", label: "where drawn", description: "the same spot every episode" },
            ...placements.map((p) => ({ value: p.name, description: p.doc.replace(/``/g, "") })),
          ]}
        />
      </label>
      {placement && info && (
        <>
          <div className="fp-params">
            <ParamFields params={info.params} values={placement} onChange={setParam} center={center} regionNames={regionNames} />
          </div>
          <div className="muted small">Placed anew each episode. Step the map's episode to see another spot.</div>
        </>
      )}
    </div>
  );
}

function CoordList({
  coords,
  onChange,
}: {
  coords: [number, number][];
  onChange: (c: [number, number][]) => void;
}) {
  return (
    <div className="coordlist">
      {coords.map((c, i) => (
        <div className="latlon" key={i}>
          <span className="latlon-label">v{i}</span>
          <NumField label="lat" value={c[0]} onChange={(v) => onChange(coords.map((p, j) => (j === i ? [v, p[1]] : p)))} />
          <NumField label="lon" value={c[1]} onChange={(v) => onChange(coords.map((p, j) => (j === i ? [p[0], v] : p)))} />
          <button className="link danger" disabled={coords.length <= 3} onClick={() => onChange(coords.filter((_, j) => j !== i))}>
            ✕
          </button>
        </div>
      ))}
      <button className="link" onClick={() => onChange([...coords, coords[coords.length - 1] ?? [52, 4.75]])}>
        + vertex
      </button>
    </div>
  );
}

function vertexValue(value: number | number[] | undefined, index: number, fallback: number): number {
  if (Array.isArray(value)) return Number.isFinite(value[index]) ? value[index] : fallback;
  return Number.isFinite(value) ? (value as number) : fallback;
}

function vertexValues(value: number | number[] | undefined, n: number, fallback: number): number[] {
  return Array.from({ length: n }, (_, i) => vertexValue(value, i, fallback));
}

function VertexBandFields({ alt, footprint, set }: { alt: SpecDict; footprint: SpecDict; set: (a: SpecDict) => void }) {
  const footprintVerts = footprintVertices(footprint);
  const vertices: [number, number][] = footprintVerts.length >= 3 ? footprintVerts : alt.vertices ?? [];
  const minValues = vertexValues(alt.min_values_ft, vertices.length, 0);
  const maxValues = vertexValues(alt.max_values_ft, vertices.length, 10000);

  const updateMin = (index: number, value: number) => {
    set({ ...alt, vertices, min_values_ft: minValues.map((v, i) => (i === index ? value : v)), max_values_ft: maxValues });
  };
  const updateMax = (index: number, value: number) => {
    set({ ...alt, vertices, min_values_ft: minValues, max_values_ft: maxValues.map((v, i) => (i === index ? value : v)) });
  };

  if (vertices.length < 3) return <div className="muted">vertex altitude follows the current footprint vertices</div>;

  return (
    <div className="coordlist vertex-alt-list">
      {vertices.map((vertex, i) => (
        <div className="vertex-alt-row" key={i} title={`${vertex[0].toFixed(6)}, ${vertex[1].toFixed(6)}`}>
          <span className="latlon-label">v{i}</span>
          <NumField label="min ft" step={500} min={0} value={minValues[i]} onChange={(v) => updateMin(i, v)} />
          <NumField label="max ft" step={500} min={0} value={maxValues[i]} onChange={(v) => updateMax(i, v)} />
        </div>
      ))}
    </div>
  );
}

function AltitudeFields({ alt, footprint, set }: { alt: SpecDict; footprint: SpecDict; set: (a: SpecDict) => void }) {
  switch (alt.type) {
    case "constant":
      return (
        <div className="grid2">
          <NumField label="min ft" step={500} min={0} value={alt.min_ft} onChange={(v) => set({ ...alt, min_ft: v })} />
          <NumField label="max ft" step={500} min={0} value={alt.max_ft} onChange={(v) => set({ ...alt, max_ft: v })} />
        </div>
      );
    case "linear":
      return (
        <>
          <LatLonField label="start" value={alt.start} onChange={(c) => set({ ...alt, start: c })} />
          <LatLonField label="end" value={alt.end} onChange={(c) => set({ ...alt, end: c })} />
          <BandField label="start band" value={alt.start_band_ft} onChange={(b) => set({ ...alt, start_band_ft: b })} />
          <BandField label="end band" value={alt.end_band_ft} onChange={(b) => set({ ...alt, end_band_ft: b })} />
        </>
      );
    case "radial":
      return (
        <>
          <LatLonField label="center" value={alt.center} onChange={(c) => set({ ...alt, center: c })} />
          <NumField label="radius nm" step={0.5} value={alt.radius_nm} onChange={(v) => set({ ...alt, radius_nm: v })} />
          <BandField label="inner band" value={alt.inner_band_ft} onChange={(b) => set({ ...alt, inner_band_ft: b })} />
          <BandField label="outer band" value={alt.outer_band_ft} onChange={(b) => set({ ...alt, outer_band_ft: b })} />
        </>
      );
    case "vertex":
      return <VertexBandFields alt={alt} footprint={footprint} set={set} />;
    default:
      return <div className="muted">{alt.type} band - edit via Code/JSON</div>;
  }
}

export default function ShapeEditor({
  shape,
  onChange,
  onFocus,
  regionNames,
  generatorRegionNames,
  requireRef,
  onNewRegion,
  resolveRegion,
  onEditRegion,
  refCount,
  areaOnly = false,
  pointOnly = false,
}: {
  // An area is needed (the airspace, a query region): no point shapes.
  areaOnly?: boolean;
  // A point is needed (a waypoint's position): only point shapes.
  pointOnly?: boolean;
  shape: SpecDict;
  onChange: (b: SpecDict) => void;
  onFocus?: () => void;
  regionNames?: string[];
  // Named regions a generated shape may refer to (a parent, a region to stay
  // within) - the design's others. Defaults to regionNames.
  generatorRegionNames?: string[];
  // Region-only mode: this shape must reference a named one (no inline).
  requireRef?: boolean;
  // Atomically create a new region and point this shape at it (the host owns
  // spec.shapes, so creation + ref must happen in one edit).
  onNewRegion?: () => void;
  // Resolve a referenced region's shape + persist edits to it, so a consumer
  // can edit the shared shape geometry in place from its own card.
  resolveRegion?: (name: string) => SpecDict | undefined;
  onEditRegion?: (name: string, shape: SpecDict) => void;
  // How many elements reference this shape (to warn that edits are shared).
  refCount?: number;
}) {
  const points = usePointShapes();
  // When the host offers named regions, allow (or require) this shape to
  // reference one ({"ref": name}) instead of carrying an inline footprint.
  const isRef = typeof shape?.ref === "string";
  const showPicker = requireRef || (regionNames && regionNames.length > 0) || onNewRegion;
  const onPick = (value: string) => {
    if (value === "__new__") onNewRegion?.();
    else if (value) onChange({ ref: value });
    else if (!requireRef) onChange(defaultRegion());
  };
  const refPicker = showPicker && (
    <label className="numfield inline">
      <span>shape</span>
      <Picker
        placeholder={requireRef ? "choose a shape…" : "inline"}
        value={isRef ? shape.ref : ""}
        onChange={onPick}
        options={[
          { value: "", label: requireRef ? "choose a shape…" : "inline" },
          ...(regionNames ?? [])
            .filter((n) => (areaOnly ? !points.has(n) : pointOnly ? points.has(n) : true) || (isRef && shape.ref === n))
            .map((n) => ({ value: n, description: points.has(n) ? "a point" : undefined })),
          ...(onNewRegion ? [{ value: "__new__", label: pointOnly ? "+ new point…" : "+ new shape…" }] : []),
        ]}
      />
    </label>
  );

  // The shape actually edited here: the referenced region (edited in place) or
  // the inline shape. In region-only mode an unset/inline value has no editor.
  const working: SpecDict | undefined = isRef
    ? resolveRegion?.(shape.ref)
    : requireRef
      ? undefined
      : shape;
  const onWorking = (b: SpecDict) => (isRef ? onEditRegion?.(shape.ref, b) : onChange(b));

  if (!working || !working.footprint) {
    return (
      <div className="shape-editor">
        {refPicker}
        <div className="muted small">
          {isRef && /^(.+)\.\d+$/.test(shape.ref) && (regionNames ?? []).includes(shape.ref)
            ? `one of “${shape.ref.replace(/\.\d+$/, "")}”'s shapes, drawn each episode - edit “${shape.ref.replace(/\.\d+$/, "")}” to change how.`
            : isRef
              ? `shape “${shape.ref}” not found.`
              : "pick a shape, or make one."}
        </div>
      </div>
    );
  }

  const fp = working.footprint ?? {};
  const alt = working.altitude;
  // What its layers may refer to: the others - never itself (or its shapes).
  const otherNames = (generatorRegionNames ?? regionNames ?? []).filter(
    (n) => !isRef || n.replace(/\.\d+$/, "") !== shape.ref,
  );
  const [clat, clon] = footprintCenter(fp);

  return (
    <div className="shape-editor">
      {refPicker}
      <section className="be-section">
        <div className="be-head">
          <span className="be-title">footprint</span>
          {onFocus && (
            <button className="link" onClick={onFocus}>
              show on map
            </button>
          )}
        </div>
      {areaOnly && fp.type === "point" && (
        <div className="error-text small">This needs an area: a point has no inside. Pick another shape.</div>
      )}
      {pointOnly && fp.type !== "point" && (
        <div className="error-text small">A waypoint is at a point: pick a point, or make this one.</div>
      )}
      <FootprintEditor
        footprint={fp}
        allowPoint={!areaOnly}
        onlyPoint={pointOnly}
        foldFields={fp.type !== "generated"}
        regionNames={otherNames}
        onChange={(f) => {
          const next: SpecDict = { ...clone(working), footprint: f };
          next.altitude = syncVertexAltitude(next.altitude, f);
          onWorking(next);
        }}
        center={[clat, clon]}
      />
      </section>
      <section className="be-section">
        <div className="be-head">
          <span className="be-title">where</span>
        </div>
      {fp.type !== "point" && (
      <label className="numfield inline">
        <span>rotate°</span>
        <input
          type="number"
          step={5}
          value={working.rotation_deg ?? 0}
          onChange={(e) => onWorking({ ...clone(working), rotation_deg: Number(e.target.value) || undefined })}
        />
      </label>
      )}
      <PlacementEditor
        placement={working.placement}
        onChange={(placement) => {
          const next: SpecDict = clone(working);
          if (placement) next.placement = placement;
          else delete next.placement;
          onWorking(next);
        }}
        center={[clat, clon]}
        regionNames={otherNames}
      />
      </section>
      <section className="be-section">
      <MotionEditor
        motion={working.motion}
        onChange={(motion) => {
          const next: SpecDict = clone(working);
          if (motion) next.motion = motion;
          else {
            delete next.motion;
            delete next.motion_update;
          }
          onWorking(next);
        }}
        cadence={working.motion_update}
        onCadence={(u) => {
          const next: SpecDict = clone(working);
          if (u) next.motion_update = u;
          else delete next.motion_update;
          onWorking(next);
        }}
        center={[clat, clon]}
        regionNames={otherNames}
      />
      </section>
      <section className="be-section">
        <div className="be-head">
          <span className="be-title">altitude</span>
        </div>
      <label className="numfield inline alt-select">
        <span>altitude</span>
        <Picker
          searchable={false}
          placeholder="none"
          value={alt ? alt.type : "none"}
          onChange={(v) => {
            const nextAlt = syncVertexAltitude(makeAltitude(v, clat, clon), fp);
            onWorking({ ...clone(working), altitude: nextAlt });
          }}
          options={ALTITUDE_TYPES.map((t) => ({ value: t }))}
        />
      </label>
      {alt && (
        <AltitudeFields
          alt={alt}
          footprint={fp}
          set={(a) => onWorking({ ...clone(working), altitude: syncVertexAltitude(a, fp) })}
        />
      )}
      </section>
    </div>
  );
}
