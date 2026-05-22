// Editor for one spawn region. The body is exposed as `SpawnBody` (used by the
// geometry inspector) grouped into Shape · Traffic · Route · Appearance;
// `SpawnCard` wraps it in a collapsible card for any legacy list view.
import type { SpecDict } from "../../api";
import BoundsEditor from "../BoundsEditor";
import { NumInput, OptValueField, ValueField } from "./ValueField";
import { Picker } from "./Picker";
import { CollapsibleCard, EyeToggle } from "./Section";
import { FieldGroup } from "./FieldGroup";
import { VisualizationToggles } from "./QueryableCard";
import { RouteSpecControl } from "./RouteSettings";

type SpawnBodyProps = {
  region: SpecDict;
  routeNames: string[];
  waypointNames: string[];
  regionNames: string[];
  onNewRegion: () => void;
  resolveRegion: (name: string) => SpecDict | undefined;
  onEditRegion: (name: string, bounds: SpecDict) => void;
  boundsRefCount: (name: string) => number;
  onChange: (r: SpecDict) => void;
  onFocus: () => void;
};

// The grouped field editor for one spawn region (no card chrome). Spawn altitude
// defaults to the bounds altitude band, but can be overridden explicitly.
export function SpawnBody({
  region,
  routeNames,
  waypointNames,
  regionNames,
  onNewRegion,
  resolveRegion,
  onEditRegion,
  boundsRefCount,
  onChange,
  onFocus,
}: SpawnBodyProps) {
  const setBounds = (b: SpecDict) => onChange({ ...region, bounds: b });
  return (
    <div className="spawn-body">
      <FieldGroup title="Shape" defaultOpen>
        <BoundsEditor
          bounds={region.bounds}
          onChange={setBounds}
          onFocus={onFocus}
          regionNames={regionNames}
          requireRef
          onNewRegion={onNewRegion}
          resolveRegion={resolveRegion}
          onEditRegion={onEditRegion}
          refCount={region.bounds?.ref ? boundsRefCount(region.bounds.ref) : undefined}
        />
        <div className="muted small">when spawn altitude is unset, it follows the bounds altitude band ↑</div>
      </FieldGroup>
      <FieldGroup title="Traffic" defaultOpen>
        <ValueField
          label={region.maintain ? "target in airspace" : "count"}
          int
          value={region.n_aircraft}
          onChange={(v) => onChange({ ...region, n_aircraft: v })}
        />
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={region.maintain === true}
            onChange={(e) => onChange({ ...region, maintain: e.target.checked })}
          />
          <span>maintain steady density</span>
        </label>
        <div className="muted small">
          continuously respawn (clear of traffic) to hold the count live ↑
        </div>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={region.controlled === false}
            onChange={(e) => onChange({ ...region, controlled: !e.target.checked })}
          />
          <span>uncooperative traffic</span>
        </label>
        <div className="muted small">
          fly on autopilot, seen as intruders, but never commanded by the policy ↑
        </div>
        <div className="value-field">
          <div className="vf-head">
            <span className="vf-label">conflict-free spawn</span>
            <span className="vf-spacer" />
            <Picker
              className="vf-mode"
              searchable={false}
              placeholder="inherit"
              value={
                region.conflict_free_spawn === true
                  ? "on"
                  : region.conflict_free_spawn === false
                    ? "off"
                    : "inherit"
              }
              onChange={(v) => {
                const next = { ...region };
                if (v === "on") next.conflict_free_spawn = true;
                else if (v === "off") next.conflict_free_spawn = false;
                else delete next.conflict_free_spawn;
                onChange(next);
              }}
              options={[
                {
                  value: "inherit",
                  label: "inherit (global)",
                  description: "use the global conflict-free-spawn default",
                },
                {
                  value: "on",
                  label: "conflict-free",
                  description:
                    "spawn this area's aircraft clear of predicted conflicts",
                },
                {
                  value: "off",
                  label: "as sampled",
                  description: "spawn as sampled, even when the global is on",
                },
              ]}
            />
          </div>
        </div>
        <div className="muted small">
          override the global conflict-free-spawn default for this area ↑
        </div>
        {(
          [
            ["conflict_free_margin_nm", "buffer horiz nm", 0.5],
            ["conflict_free_margin_ft", "buffer vert ft", 100],
            ["conflict_free_margin_s", "buffer time s", 30],
          ] as const
        ).map(([key, label, step]) => (
          <div className="value-field" key={key}>
            <div className="vf-head">
              <span className="vf-label">{label}</span>
              <span className="vf-spacer" />
              <NumInput
                className="vf-input"
                step={step}
                placeholder="inherit"
                value={(region[key] as number | undefined) ?? Number.NaN}
                onChange={(n) => onChange({ ...region, [key]: n })}
                onClear={() => {
                  const next = { ...region };
                  delete next[key];
                  onChange(next);
                }}
              />
            </div>
          </div>
        ))}
        <div className="muted small">
          spawn buffer over the protected zone; empty inherits the global ↑
        </div>
        <OptValueField
          label="alt ft"
          step={500}
          allowEnvelope
          value={region.params?.alt_ft}
          defaultValue={{ type: "envelope" }}
          onChange={(v) => {
            const params = { ...(region.params ?? {}) };
            if (v == null) delete params.alt_ft;
            else params.alt_ft = v;
            onChange({ ...region, params });
          }}
        />
        <ValueField
          label="speed kt"
          step={5}
          allowEnvelope
          value={region.params?.spd_kts}
          onChange={(v) => onChange({ ...region, params: { ...region.params, spd_kts: v } })}
        />
        <ValueField
          label="heading °"
          step={10}
          value={region.params?.hdg_deg ?? { type: "range", low: 0, high: 360 }}
          onChange={(v) => onChange({ ...region, params: { ...region.params, hdg_deg: v } })}
        />
        <ValueField
          label="spawn time s"
          step={5}
          value={region.spawn_time}
          onChange={(v) => onChange({ ...region, spawn_time: v })}
        />
      </FieldGroup>
      <FieldGroup title="Route">
        <RouteSpecControl
          label="route override"
          route={region.route}
          routeNames={routeNames}
          waypointNames={waypointNames}
          allowInherit
          onChange={(route) => onChange({ ...region, route })}
        />
      </FieldGroup>
      <FieldGroup title="Appearance">
        <VisualizationToggles
          shape={region.render_shape !== false}
          label={region.render_name !== false}
          onShape={(render_shape) => onChange({ ...region, render_shape })}
          onLabel={(render_name) => onChange({ ...region, render_name })}
        />
      </FieldGroup>
    </div>
  );
}

export function SpawnCard({
  hidden,
  onToggleHidden,
  selected,
  onSelectCard,
  onRemove,
  ...body
}: SpawnBodyProps & {
  hidden: boolean;
  onToggleHidden: () => void;
  selected?: boolean;
  onSelectCard?: () => void;
  onRemove: () => void;
}) {
  return (
    <CollapsibleCard
      selected={selected}
      onSelect={onSelectCard}
      header={<>
        <input className="name-input" value={body.region.name ?? ""} onChange={(e) => body.onChange({ ...body.region, name: e.target.value })} />
        <EyeToggle hidden={hidden} onToggle={onToggleHidden} />
        <button className="link danger" onClick={onRemove}>
          ✕
        </button>
      </>}
    >
      <SpawnBody {...body} />
    </CollapsibleCard>
  );
}
