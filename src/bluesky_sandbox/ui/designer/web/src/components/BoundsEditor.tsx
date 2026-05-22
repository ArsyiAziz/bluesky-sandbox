import { useState } from "react";
import type { SpecDict } from "../api";
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
}: {
  footprint: SpecDict;
  onChange: (f: SpecDict) => void;
  center: [number, number];
  foldFields?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const set = (mut: (f: SpecDict) => void) => {
    const next = clone(footprint);
    mut(next);
    onChange(next);
  };
  const picker = (
    <label className="numfield inline">
      <span>shape</span>
      <Picker
        searchable={false}
        placeholder="shape"
        value={footprint.type}
        onChange={(v) => onChange(makeFootprint(v, center[0], center[1]))}
        options={FOOTPRINT_TYPES.map((t) => ({ value: t }))}
      />
    </label>
  );
  if (!foldFields) {
    return (
      <div className="footprint-editor">
        {picker}
        <FootprintFields fp={footprint} set={set} center={center} />
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
      {open && <FootprintFields fp={footprint} set={set} center={center} />}
    </div>
  );
}

function FootprintFields({
  fp,
  set,
  center,
}: {
  fp: SpecDict;
  set: (mut: (f: SpecDict) => void) => void;
  center: [number, number];
}) {
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

export default function BoundsEditor({
  bounds,
  onChange,
  onFocus,
  regionNames,
  requireRef,
  onNewRegion,
  resolveRegion,
  onEditRegion,
  refCount,
}: {
  bounds: SpecDict;
  onChange: (b: SpecDict) => void;
  onFocus?: () => void;
  regionNames?: string[];
  // Region-only mode: this bounds must reference a named region (no inline).
  requireRef?: boolean;
  // Atomically create a new region and point this bounds at it (the host owns
  // spec.regions, so creation + ref must happen in one edit).
  onNewRegion?: () => void;
  // Resolve a referenced region's bounds + persist edits to it, so a consumer
  // can edit the shared bounds geometry in place from its own card.
  resolveRegion?: (name: string) => SpecDict | undefined;
  onEditRegion?: (name: string, bounds: SpecDict) => void;
  // How many elements reference this bounds (to warn that edits are shared).
  refCount?: number;
}) {
  // When the host offers named regions, allow (or require) this bounds to
  // reference one ({"ref": name}) instead of carrying an inline footprint.
  const isRef = typeof bounds?.ref === "string";
  const showPicker = requireRef || (regionNames && regionNames.length > 0) || onNewRegion;
  const onPick = (value: string) => {
    if (value === "__new__") onNewRegion?.();
    else if (value) onChange({ ref: value });
    else if (!requireRef) onChange(defaultRegion());
  };
  const refPicker = showPicker && (
    <label className="numfield inline">
      <span>bounds</span>
      <Picker
        placeholder={requireRef ? "choose bounds…" : "inline"}
        value={isRef ? bounds.ref : ""}
        onChange={onPick}
        options={[
          { value: "", label: requireRef ? "choose bounds…" : "inline" },
          ...(regionNames ?? []).map((n) => ({ value: n })),
          ...(onNewRegion ? [{ value: "__new__", label: "+ new bounds…" }] : []),
        ]}
      />
    </label>
  );

  // The bounds actually edited here: the referenced region (edited in place) or
  // the inline bounds. In region-only mode an unset/inline value has no editor.
  const working: SpecDict | undefined = isRef
    ? resolveRegion?.(bounds.ref)
    : requireRef
      ? undefined
      : bounds;
  const onWorking = (b: SpecDict) => (isRef ? onEditRegion?.(bounds.ref, b) : onChange(b));

  if (!working || !working.footprint) {
    return (
      <div className="bounds-editor">
        {refPicker}
        <div className="muted small">
          {isRef ? `bounds “${bounds.ref}” not found.` : "pick a bounds, or create one."}
        </div>
      </div>
    );
  }

  const fp = working.footprint ?? {};
  const alt = working.altitude;
  const [clat, clon] = footprintCenter(fp);

  return (
    <div className="bounds-editor">
      {refPicker}
      {isRef && typeof refCount === "number" && refCount > 1 && (
        <div className="muted small">shared by {refCount} elements — edits affect all.</div>
      )}
      {onFocus && (
        <div className="row between">
          <span className="kind-tag">footprint</span>
          <button className="link" onClick={onFocus}>
            focus
          </button>
        </div>
      )}
      <FootprintEditor
        footprint={fp}
        foldFields
        onChange={(f) => {
          const next: SpecDict = { ...clone(working), footprint: f };
          next.altitude = syncVertexAltitude(next.altitude, f);
          onWorking(next);
        }}
        center={[clat, clon]}
      />
      <NumField
        label="rotate°"
        step={5}
        value={working.rotation_deg ?? 0}
        onChange={(v) => onWorking({ ...clone(working), rotation_deg: v || undefined })}
      />

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
    </div>
  );
}
