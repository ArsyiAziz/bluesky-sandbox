// Environment configuration: aircraft whitelist, timing, CD method, performance
// model, the reward/termination/truncation code references, and the per-episode
// airspace rotation editor.
import type { SpecDict } from "../../api";
import { NumField } from "../BoundsEditor";
import { Picker } from "./Picker";
import { ValueField } from "./ValueField";
import { designBounds, newGroupId } from "../../specHelpers";

// An optional numeric override: empty leaves BlueSky's default (sends null), a
// typed value overrides it. The placeholder shows the default — no checkbox.
function DefaultedNumField({
  label,
  value,
  placeholder,
  step,
  onChange,
}: {
  label: string;
  value: number | null | undefined;
  placeholder: string;
  step?: number;
  onChange: (v: number | null) => void;
}) {
  return (
    <label className="numfield inline">
      <span>{label}</span>
      <input
        type="number"
        step={step}
        min={0}
        placeholder={placeholder}
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
      />
    </label>
  );
}

// Per-episode rotation groups. Each group rotates a chosen set of **bounds**
// (named regions) by a sampled angle — rotating a bounds rotates every element
// (airspace / queryable / spawn region) that references it. Groups nest: a group
// whose parent is another group is rotated locally first, then carried by the
// parent's rotation ("rotation in rotation").
export function RotationEditor({
  spec,
  onChange,
}: {
  spec: SpecDict;
  onChange: (transform: SpecDict | null) => void;
}) {
  const groups: SpecDict[] = spec.transform?.groups ?? [];
  const bounds = designBounds(spec);
  const ownerOf = (eid: string) => groups.find((g) => (g.members ?? []).includes(eid));

  const setGroups = (next: SpecDict[]) => onChange(next.length ? { groups: next } : null);
  const update = (i: number, patch: SpecDict) =>
    setGroups(groups.map((g, j) => (j === i ? { ...g, ...patch } : g)));

  const addGroup = () =>
    setGroups([
      ...groups,
      {
        id: newGroupId(),
        name: `group ${groups.length + 1}`,
        angle_deg: { type: "range", low: -30, high: 30 },
        pivot: null,
        members: [],
        parent: null,
      },
    ]);

  const removeGroup = (i: number) => {
    const id = groups[i].id;
    setGroups(groups.filter((_, j) => j !== i).map((g) => (g.parent === id ? { ...g, parent: null } : g)));
  };

  // An element belongs to at most one group; assigning it here removes it from
  // any other group (so rotations don't double-apply outside the nesting).
  const toggleMember = (i: number, eid: string) =>
    setGroups(
      groups.map((g, j) => {
        const members = (g.members ?? []).filter((m: string) => m !== eid);
        if (j === i && !(groups[i].members ?? []).includes(eid)) members.push(eid);
        return { ...g, members };
      }),
    );

  // Valid parents for group i: any other group that isn't a descendant of i
  // (prevents cycles).
  const descendants = (id: string): Set<string> => {
    const out = new Set<string>();
    const walk = (pid: string) => {
      for (const g of groups) if (g.parent === pid && !out.has(g.id)) { out.add(g.id); walk(g.id); }
    };
    walk(id);
    return out;
  };

  return (
    <div className="groups-editor">
      {groups.length === 0 && (
        <div className="muted small">No rotation groups — the airspace is fixed each episode.</div>
      )}
      {groups.map((g, i) => {
        const banned = descendants(g.id);
        const parentOptions = [
          { value: "", label: "— none (top level)" },
          ...groups
            .filter((o) => o.id !== g.id && !banned.has(o.id))
            .map((o) => ({ value: o.id, label: o.name || o.id })),
        ];
        return (
          <div className="card" key={g.id}>
            <div className="row between">
              <input
                className="name-input"
                value={g.name ?? ""}
                onChange={(e) => update(i, { name: e.target.value })}
              />
              <button className="chip-x" title="remove group" onClick={() => removeGroup(i)}>✕</button>
            </div>
            <ValueField
              label="angle°"
              step={5}
              value={g.angle_deg}
              onChange={(v) => update(i, { angle_deg: v })}
            />
            <label className="numfield inline">
              <span>inside</span>
              <Picker
                searchable={false}
                placeholder="— none (top level)"
                value={g.parent ?? ""}
                onChange={(v) => update(i, { parent: v || null })}
                options={parentOptions}
              />
            </label>
            <div className="sub-label">bounds</div>
            <div className="chips">
              {bounds.length === 0 && <span className="muted small">no bounds yet</span>}
              {bounds.map((name) => {
                const owner = ownerOf(name);
                const mine = owner?.id === g.id;
                const elsewhere = owner && !mine;
                return (
                  <button
                    key={name}
                    className={mine ? "chip member on" : "chip member"}
                    title={elsewhere ? `currently in "${owner!.name || owner!.id}"` : name}
                    onClick={() => toggleMember(i, name)}
                  >
                    {name}
                    {elsewhere ? <span className="muted"> ·{owner!.name || owner!.id}</span> : null}
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
      <button onClick={addGroup}>+ rotation group</button>
      {groups.length > 0 && (
        <div className="muted small">
          Each group rotates its members by an angle sampled per episode about their centre. Put a
          group <em>inside</em> another to compose rotations (local spin, then carried). Reseed on the
          map to preview.
        </div>
      )}
    </div>
  );
}

// Full env configuration: aircraft whitelist, timing, CD method, performance
// model, and the reward/termination/truncation code references.
export function ConfigEditor({
  env,
  code,
  catalog,
  onChange,
  onCodeChange,
}: {
  env: SpecDict;
  code: Record<string, string>;
  catalog: any;
  onChange: (env: SpecDict) => void;
  onCodeChange: (code: Record<string, string>) => void;
}) {
  const set = (patch: SpecDict) => onChange({ ...env, ...patch });
  const aircraft: string[] = env.allowed_aircraft ?? [];
  const aircraftSet = new Set(aircraft.map((ac) => ac.toUpperCase()));
  // Available aircraft for the *selected* performance model; an object with an
  // `error` means that model's database isn't installed (e.g. BADA).
  const model = env.performance_model ?? "openap";
  const avail = catalog?.aircraft?.[model];
  const availError: string | null = avail && !Array.isArray(avail) ? avail.error : null;
  const available: string[] = Array.isArray(avail) ? avail : [];
  const allSelected = available.length > 0 && available.every((a) => aircraftSet.has(a.toUpperCase()));
  return (
    <div className="config-editor">
      <div className="sub-label">allowed aircraft</div>
      {availError ? (
        <div className="error-text small">{model}: {availError}</div>
      ) : (
        <>
          <label className="radio" title="sample uniformly across every aircraft type in the model">
            <input
              type="checkbox"
              checked={allSelected}
              disabled={available.length === 0}
              onChange={(e) => set({ allowed_aircraft: e.target.checked ? [...available] : [] })}
            />
            use all available ({available.length} {model})
          </label>
          {!allSelected && (
            <>
              <div className="chips">
                {aircraft.map((ac, i) => (
                  <span className="chip" key={`${ac}-${i}`}>
                    {ac}
                    <button className="chip-x" onClick={() => set({ allowed_aircraft: aircraft.filter((_, j) => j !== i) })}>
                      ✕
                    </button>
                  </span>
                ))}
                {aircraft.length === 0 && <span className="muted">none</span>}
              </div>
              <Picker
                placeholder="+ add aircraft…"
                onChange={(v) => set({ allowed_aircraft: [...aircraft, v] })}
                options={available
                  .filter((a) => !aircraftSet.has(a.toUpperCase()))
                  .map((a) => ({ value: a }))}
              />
            </>
          )}
        </>
      )}

      <div className="grid2">
        <NumField label="dt (s)" step={0.1} value={env.dt} onChange={(v) => set({ dt: v })} />
        <NumField label="simdt (s)" step={0.01} value={env.simdt} onChange={(v) => set({ simdt: v })} />
      </div>
      <label className="numfield inline">
        <span>perf model</span>
        <Picker
          searchable={false}
          placeholder="openap"
          value={env.performance_model ?? "openap"}
          onChange={(v) => set({ performance_model: v })}
          options={[{ value: "openap" }, { value: "bada" }]}
        />
      </label>

      <div className="sub-label">conflict detection</div>
      <label className="numfield inline">
        <span>cd method</span>
        <Picker
          searchable={false}
          placeholder="CSTATEBASED"
          value={env.cd_method ?? "CSTATEBASED"}
          onChange={(v) => set({ cd_method: v })}
          options={(catalog?.conflict?.cd_methods ?? ["CSTATEBASED", "STATEBASED"]).map((m: string) => ({ value: m }))}
        />
      </label>
      <label className="numfield inline">
        <span>resolution</span>
        <Picker
          searchable={false}
          placeholder="off"
          value={env.reso_method ?? "OFF"}
          onChange={(v) => set({ reso_method: v === "OFF" ? null : v })}
          options={(catalog?.conflict?.reso_methods ?? ["OFF", "MVP"]).map((m: string) => ({
            value: m,
            label: m === "OFF" ? "off (agent resolves)" : m,
          }))}
        />
      </label>
      <DefaultedNumField label="PZ radius (nm)" step={0.5} placeholder="5 (default)"
        value={env.pz_radius_nm} onChange={(v) => set({ pz_radius_nm: v })} />
      <DefaultedNumField label="PZ height (ft)" step={100} placeholder="1000 (default)"
        value={env.pz_height_ft} onChange={(v) => set({ pz_height_ft: v })} />
      <DefaultedNumField label="lookahead (s)" step={30} placeholder="300 (default)"
        value={env.lookahead_s} onChange={(v) => set({ lookahead_s: v })} />
      <div className="muted small">
        Detection runs always (for observations); resolution defaults off so the agent resolves
        conflicts. PZ / lookahead left unset use BlueSky's defaults. Re-applied each reset.
      </div>

      <div className="sub-label">wind</div>
      <div className="grid2">
        <NumField label="direction (° from)" step={5}
          value={env.wind_dir_deg ?? 270} onChange={(v) => set({ wind_dir_deg: v ?? 270 })} />
        <NumField label="speed (kt)" step={5}
          value={env.wind_kts ?? 0} onChange={(v) => set({ wind_kts: v ?? 0 })} />
      </div>
      <div className="grid2">
        <NumField label="turbulence (kt RMS)" step={1}
          value={env.turbulence_kts ?? 0} onChange={(v) => set({ turbulence_kts: v ?? 0 })} />
        <NumField label="gust τ (s)" step={5}
          value={env.gust_tau_s ?? 30} onChange={(v) => set({ gust_tau_s: v ?? 30 })} />
      </div>
      <div className="muted small">
        Uniform wind, aviation-standard: direction is where it blows <em>from</em> (270 = westerly,
        pushing aircraft east). Speed 0 = no wind. Turbulence adds a time-correlated gust (RMS kt)
        decorrelating over gust τ. Re-applied each reset.
      </div>

      <div className="muted small">
        Reward / termination / truncation are env hooks — edit them in the Code tab's “env hooks” panel.
      </div>
    </div>
  );
}
