// Editor for a single queryable (goal/restricted region or waypoint). The body
// is exposed as `QueryableBody` (used by the geometry inspector) grouped into
// Shape/Position · Constraints · Appearance · Advanced sub-groups; `QueryableCard`
// wraps that body in a collapsible card for any legacy list view.
import type { SpecDict } from "../../api";
import BoundsEditor, { NumField } from "../BoundsEditor";
import { footprintCenter } from "../../specHelpers";
import { CollapsibleCard, EyeToggle } from "./Section";
import { FieldGroup } from "./FieldGroup";
import { Picker } from "./Picker";
import { ColorPicker } from "./ColorPicker";
import { NumInput, OptValueField } from "./ValueField";

type QueryableBodyProps = {
  name: string;
  q: SpecDict;
  regionNames: string[];
  namedRegions: Record<string, SpecDict>;
  namedRegionNames: string[];
  onNewBoundsRegion: () => void;
  onNewSampleRegion: () => void;
  resolveRegion: (name: string) => SpecDict | undefined;
  onEditRegion: (name: string, bounds: SpecDict) => void;
  boundsRefCount: (name: string) => number;
  onChange: (q: SpecDict) => void;
  onFocus: () => void;
};

// The grouped field editor for one queryable (no card chrome). The host supplies
// the name/eye/delete header.
export function QueryableBody({
  name,
  q,
  regionNames,
  namedRegions,
  namedRegionNames,
  onNewBoundsRegion,
  onNewSampleRegion,
  resolveRegion,
  onEditRegion,
  boundsRefCount,
  onChange,
  onFocus,
}: QueryableBodyProps) {
  const appearance = (
    <FieldGroup title="Appearance">
      <label className="numfield inline">
        <span>color</span>
        <ColorPicker value={q.color} onChange={(color) => onChange({ ...q, color })} />
      </label>
      <VisualizationToggles
        shape={q.render_shape !== false}
        label={q.render_label !== false}
        onShape={(render_shape) => onChange({ ...q, render_shape })}
        onLabel={(render_label) => onChange({ ...q, render_label })}
      />
      {q.type === "waypoint" && (
        <WaypointTsas q={q} regionNames={regionNames} onChange={onChange} />
      )}
    </FieldGroup>
  );
  const advanced = (
    <FieldGroup title="Advanced">
      <label className="radio">
        <input
          type="checkbox"
          checked={q.track_temporal_state === true}
          onChange={(e) => onChange({ ...q, track_temporal_state: e.target.checked })}
        />
        track temporal state
      </label>
      <div className="muted small">
        Enables per-substep tracking for this queryable when fields use during-step, time, or step-minimum values.
      </div>
    </FieldGroup>
  );

  if (q.type === "waypoint") {
    return (
      <div className="queryable-body">
        <FieldGroup title="Position" defaultOpen>
          <WaypointPosition
            q={q}
            namedRegions={namedRegions}
            namedRegionNames={namedRegionNames}
            onNewSampleRegion={onNewSampleRegion}
            resolveRegion={resolveRegion}
            onEditRegion={onEditRegion}
            boundsRefCount={boundsRefCount}
            onChange={onChange}
          />
        </FieldGroup>
        <FieldGroup title="Constraints">
          <WaypointConstraints q={q} onChange={onChange} />
        </FieldGroup>
        {appearance}
        {advanced}
      </div>
    );
  }
  return (
    <div className="queryable-body">
      <FieldGroup title="Shape" defaultOpen>
        <BoundsEditor
          bounds={q.bounds}
          onChange={(b) => onChange({ ...q, bounds: b })}
          onFocus={onFocus}
          regionNames={namedRegionNames}
          requireRef
          onNewRegion={onNewBoundsRegion}
          resolveRegion={resolveRegion}
          onEditRegion={onEditRegion}
          refCount={q.bounds?.ref ? boundsRefCount(q.bounds.ref) : undefined}
        />
      </FieldGroup>
      {appearance}
      {advanced}
    </div>
  );
}

export function QueryableCard({
  hidden,
  onToggleHidden,
  onRename,
  selected,
  onSelectCard,
  onRemove,
  ...body
}: QueryableBodyProps & {
  hidden: boolean;
  onToggleHidden: () => void;
  onRename: (n: string) => void;
  selected?: boolean;
  onSelectCard?: () => void;
  onRemove: () => void;
}) {
  return (
    <CollapsibleCard
      selected={selected}
      onSelect={onSelectCard}
      header={<>
        <input className="name-input" defaultValue={body.name} onBlur={(e) => onRename(e.target.value)} />
        <span className="kind-tag">{body.q.type === "waypoint" ? "waypoint" : "region"}</span>
        <EyeToggle hidden={hidden} onToggle={onToggleHidden} />
        <button className="link danger" onClick={onRemove}>
          ✕
        </button>
      </>}
    >
      <QueryableBody {...body} />
    </CollapsibleCard>
  );
}

export function VisualizationToggles({
  shape,
  label,
  onShape,
  onLabel,
}: {
  shape: boolean;
  label: boolean;
  onShape: (v: boolean) => void;
  onLabel: (v: boolean) => void;
}) {
  return (
    <div className="row">
      <label className="radio">
        <input type="checkbox" checked={shape} onChange={(e) => onShape(e.target.checked)} />
        show shape
      </label>
      <label className="radio">
        <input type="checkbox" checked={label} disabled={!shape} onChange={(e) => onLabel(e.target.checked)} />
        show label
      </label>
    </div>
  );
}

// Position / shape source for a waypoint: a navdb fix, a fixed lat/lon, or a
// position sampled within a region (per episode or per aircraft).
function WaypointPosition({
  q,
  namedRegions,
  namedRegionNames,
  onNewSampleRegion,
  resolveRegion,
  onEditRegion,
  boundsRefCount,
  onChange,
}: {
  q: SpecDict;
  namedRegions: Record<string, SpecDict>;
  namedRegionNames: string[];
  onNewSampleRegion: () => void;
  resolveRegion: (name: string) => SpecDict | undefined;
  onEditRegion: (name: string, bounds: SpecDict) => void;
  boundsRefCount: (name: string) => number;
  onChange: (q: SpecDict) => void;
}) {
  const named = q.waypoint != null;
  const sampled = q.sample != null;
  // Sampled waypoint specs keep lat/lon in sync with the sample region centre
  // so the static query target stays schema-stable. Per-aircraft samples are
  // compiled onto route steps when this waypoint is used in a spawn route.
  const setSampleRegion = (bounds: SpecDict) => {
    const resolved = bounds?.ref ? namedRegions[bounds.ref] : bounds;
    const fp = resolved?.footprint ?? resolved;
    const [lat, lon] = footprintCenter(fp);
    const next: SpecDict = { ...q, sample: bounds };
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      next.lat = lat;
      next.lon = lon;
    }
    onChange(next);
  };
  return (
    <div>
      <div className="row">
        <label className="radio">
          <input type="radio" checked={named} onChange={() => onChange({ ...stripPos(q), waypoint: q.waypoint ?? "EKROS" })} />
          navdb fix
        </label>
        <label className="radio">
          <input type="radio" checked={!named} onChange={() => onChange({ ...stripPos(q), lat: q.lat ?? 52.0, lon: q.lon ?? 4.75 })} />
          lat/lon
        </label>
      </div>
      {!named && (
        <label className="radio">
          <input
            type="checkbox"
            checked={sampled}
            onChange={(e) => {
              if (e.target.checked) onNewSampleRegion();
              else onChange(stripSample(q));
            }}
          />
          sample position within a region
        </label>
      )}
      {named ? (
        <label className="numfield inline">
          <span>fix</span>
          <input value={q.waypoint ?? ""} onChange={(e) => onChange({ ...q, waypoint: e.target.value.toUpperCase() })} />
        </label>
      ) : sampled ? (
        <div className="sample-region">
          <label className="numfield inline">
            <span>sample</span>
            <Picker
              searchable={false}
              placeholder="per episode (shared)"
              value={q.sample_per === "aircraft" ? "aircraft" : "episode"}
              onChange={(v) => {
                const next = { ...q };
                if (v === "aircraft") next.sample_per = "aircraft";
                else delete next.sample_per;
                onChange(next);
              }}
              options={[
                { value: "episode", label: "per episode (shared)" },
                { value: "aircraft", label: "per aircraft (unique)" },
              ]}
            />
          </label>
          <div className="muted small">
            {q.sample_per === "aircraft"
              ? "when this waypoint appears in a spawn route, each aircraft draws its route target from these bounds at spawn; query fields then read that target from BlueSky."
              : "one position (lat/lon + altitude) drawn from these bounds each episode, shared by all aircraft; reseed to preview."}
          </div>
          <BoundsEditor
            bounds={q.sample}
            onChange={setSampleRegion}
            regionNames={namedRegionNames}
            requireRef
            onNewRegion={onNewSampleRegion}
            resolveRegion={resolveRegion}
            onEditRegion={onEditRegion}
            refCount={q.sample?.ref ? boundsRefCount(q.sample.ref) : undefined}
          />
        </div>
      ) : (
        <div className="grid2">
          <NumField label="lat" value={q.lat} onChange={(v) => onChange({ ...q, lat: v })} />
          <NumField label="lon" value={q.lon} onChange={(v) => onChange({ ...q, lon: v })} />
        </div>
      )}
    </div>
  );
}

// Optional crossing constraints for a waypoint: altitude (and tolerance), speed
// (and tolerance, both requiring altitude), and reach radius.
function WaypointConstraints({ q, onChange }: { q: SpecDict; onChange: (q: SpecDict) => void }) {
  const altitudeEnabled = q.alt_ft != null;
  const speedRequiresAltitude = q.speed_kts != null && q.alt_ft == null;
  return (
    <div>
      <OptValueField
        label="alt ft"
        step={500}
        allowEnvelope
        value={q.alt_ft}
        defaultValue={3000}
        onChange={(v) => {
          const next = { ...q };
          if (v == null) {
            delete next.alt_ft;
            delete next.alt_tolerance_ft;
            delete next.speed_kts;
            delete next.speed_tolerance_kts;
            delete next.reachable_from_spawn;
            delete next.reachable_vs_fraction;
            onChange(next);
            return;
          }
          next.alt_ft = v;
          // reachable-from-spawn only applies to an envelope-sampled altitude.
          if (typeof v !== "object" || v.type !== "envelope") {
            delete next.reachable_from_spawn;
            delete next.reachable_vs_fraction;
          }
          onChange(next);
        }}
      />
      {typeof q.alt_ft === "object" && q.alt_ft?.type === "envelope" && (
        <>
          <label className="radio">
            <input
              type="checkbox"
              checked={q.reachable_from_spawn === true}
              onChange={(e) => {
                const next = { ...q };
                if (e.target.checked) {
                  next.reachable_from_spawn = true;
                } else {
                  delete next.reachable_from_spawn;
                  delete next.reachable_vs_fraction;
                }
                onChange(next);
              }}
            />
            reachable from spawn
          </label>
          <div className="muted small">
            Bounds the per-aircraft envelope altitude to what the aircraft can
            climb/descend to before reaching the fix (from its max vertical rate
            and distance to go), so every episode is altitude-feasible.
          </div>
          {q.reachable_from_spawn === true && (
            <>
              <div className="value-field">
                <div className="vf-head">
                  <span className="vf-label">vs fraction</span>
                  <span className="vf-spacer" />
                  <NumInput
                    className="vf-input"
                    step={0.05}
                    value={
                      typeof q.reachable_vs_fraction === "number"
                        ? q.reachable_vs_fraction
                        : 1
                    }
                    onChange={(v) => {
                      const next = { ...q };
                      // 1.0 is the runtime default -> omit; store only a tighter
                      // margin (validation requires 0 < f <= 1).
                      if (Number.isFinite(v) && v > 0 && v < 1) {
                        next.reachable_vs_fraction = v;
                      } else {
                        delete next.reachable_vs_fraction;
                      }
                      onChange(next);
                    }}
                  />
                </div>
              </div>
              <div className="muted small">
                Fraction of max climb/descent rate to assume (0–1, default 1).
                Lower reserves margin for the turn-to-fix and speed matching.
              </div>
            </>
          )}
        </>
      )}
      <OptValueField
        label="speed kt"
        step={5}
        allowEnvelope
        value={q.speed_kts}
        defaultValue={220}
        disabled={!altitudeEnabled}
        disabledReason="enable altitude before setting a waypoint speed"
        onChange={(v) => {
          if (v == null) {
            const next = { ...q };
            delete next.speed_kts;
            delete next.speed_tolerance_kts;
            onChange(next);
            return;
          }
          if (q.alt_ft == null) return;
          onChange({ ...q, speed_kts: v });
        }}
      />
      {speedRequiresAltitude && (
        <div className="error-text small">waypoint speed requires altitude</div>
      )}
      <OptValueField
        label="reach radius nm"
        step={0.1}
        value={q.reach_radius_nm}
        defaultValue={1}
        onChange={(v) => onChange({ ...q, reach_radius_nm: v })}
      />
      {altitudeEnabled && (
        <OptValueField
          label="alt tol ft"
          step={100}
          value={q.alt_tolerance_ft}
          onChange={(v) => onChange({ ...q, alt_tolerance_ft: v })}
        />
      )}
      <OptValueField
        label="speed tol kt"
        step={5}
        value={q.speed_tolerance_kts}
        defaultValue={20}
        disabled={!altitudeEnabled}
        disabledReason="enable altitude before setting speed tolerance"
        onChange={(v) => {
          if (v == null) {
            const next = { ...q };
            delete next.speed_tolerance_kts;
            onChange(next);
            return;
          }
          if (q.alt_ft == null) return;
          onChange({ ...q, speed_tolerance_kts: v });
        }}
      />
      <OptValueField
        label="speed tol M"
        step={0.01}
        value={q.speed_tolerance_mach}
        defaultValue={0.02}
        disabled={!altitudeEnabled}
        disabledReason="enable altitude before setting speed tolerance"
        onChange={(v) => {
          if (v == null) {
            const next = { ...q };
            delete next.speed_tolerance_mach;
            onChange(next);
            return;
          }
          if (q.alt_ft == null) return;
          onChange({ ...q, speed_tolerance_mach: v });
        }}
      />
      <div className="muted small">
        Optional Mach tolerance, applied above the CAS/Mach crossover altitude.
        Leave unset to auto-derive it from “speed tol kt” there, so a single CAS
        tolerance stays well-defined at any sampled altitude.
      </div>
    </div>
  );
}

// The TSAS strip toggle + which region's aircraft it tracks.
function WaypointTsas({
  q,
  regionNames,
  onChange,
}: {
  q: SpecDict;
  regionNames: string[];
  onChange: (q: SpecDict) => void;
}) {
  return (
    <>
      <label className="radio">
        <input
          type="checkbox"
          checked={q.render_tsas ?? q.render_shape !== false}
          onChange={(e) => onChange({ ...q, render_tsas: e.target.checked })}
        />
        show TSAS strip
      </label>
      {(q.render_tsas ?? q.render_shape !== false) && (
        <label className="numfield inline">
          <span>TSAS bound</span>
          <Picker
            placeholder="all aircraft"
            value={q.tsas_region ?? ""}
            onChange={(v) => onChange({ ...q, tsas_region: v || undefined })}
            options={[{ value: "", label: "all aircraft" }, ...regionNames.map((name) => ({ value: name }))]}
          />
        </label>
      )}
    </>
  );
}

// An optional numeric constraint: a checkbox toggles it on/off. Off -> the
// value becomes undefined (dropped from the spec), so it can be cleared again.
export function OptNumField({
  label,
  value,
  step,
  min,
  defaultValue,
  onChange,
}: {
  label: string;
  value: number | null | undefined;
  step?: number;
  min?: number;
  defaultValue: number;
  onChange: (v: number | undefined) => void;
}) {
  const on = value != null && Number.isFinite(value);
  return (
    <div className="opt-field">
      <label className="radio">
        <input type="checkbox" checked={on} onChange={(e) => onChange(e.target.checked ? defaultValue : undefined)} />
        {label}
      </label>
      {on && <NumField label="" value={value as number} step={step} min={min} onChange={(v) => onChange(v)} />}
    </div>
  );
}

function stripPos(q: SpecDict): SpecDict {
  const { lat, lon, waypoint, sample, ...rest } = q;
  return rest;
}

function stripSample(q: SpecDict): SpecDict {
  const { sample, ...rest } = q;
  return rest;
}
