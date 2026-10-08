// Environment configuration: aircraft whitelist, timing, CD method, performance
// model, the reward/termination/truncation code references, and the per-episode
// airspace rotation editor.
import { useEffect, useState } from "react";
import type { SpecDict } from "../../api";
import { Picker } from "./Picker";
import { FormCard, FormColumn, FormRow, NumberInput } from "../form";
import { ValueField } from "./ValueField";
import { designBounds, newGroupId } from "../../specHelpers";

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
          Each group rotates its members by an angle sampled per episode about their center. Put a
          group <em>inside</em> another to compose rotations (local spin, then carried). Reseed on the
          map to preview.
        </div>
      )}
    </div>
  );
}

// How strong the wind is: calm (none), steady (a mean wind), or gusting (with
// turbulence) - read from the speeds themselves, nothing else saved.
type WindMode = "calm" | "steady" | "gusting";
const WIND_RANK: Record<WindMode, number> = { calm: 0, steady: 1, gusting: 2 };
const windOf = (env: SpecDict): WindMode =>
  (env.turbulence_kts ?? 0) > 0 ? "gusting" : (env.wind_kts ?? 0) > 0 ? "steady" : "calm";

// The wind: none unless chosen. The fields show for the mode picked, and stay
// while one is being typed - clearing the speed to retype it does not hide it;
// only picking another mode does. A design whose values ask for more (one
// loaded, or edited in the Code tab) shows its fields.
function WindCard({ env, set }: { env: SpecDict; set: (patch: SpecDict) => void }) {
  const [mode, setMode] = useState<WindMode>(() => windOf(env));
  const saved = windOf(env);
  useEffect(() => {
    if (WIND_RANK[saved] > WIND_RANK[mode]) setMode(saved);
  }, [saved, mode]);
  const choose = (next: WindMode) => {
    setMode(next);
    const speed = (env.wind_kts ?? 0) > 0 ? env.wind_kts : 20;
    const gust = (env.turbulence_kts ?? 0) > 0 ? env.turbulence_kts : 5;
    if (next === "calm") set({ wind_kts: 0, turbulence_kts: 0 });
    else if (next === "steady") set({ wind_kts: speed, turbulence_kts: 0 });
    else set({ wind_kts: speed, turbulence_kts: gust });
  };
  return (
    <FormCard
      title="Wind"
      help={
        mode === "calm" ? undefined : (
          <>Uniform, aviation-standard: the direction is where it blows <em>from</em> (270 is westerly, pushing aircraft east).{mode === "gusting" && " Turbulence adds a gust with that RMS, decorrelating over τ."} Applied at each reset.</>
        )
      }
    >
      <FormRow label="wind">
        <Picker
          searchable={false}
          placeholder="calm"
          value={mode}
          onChange={(v) => choose((v || "calm") as WindMode)}
          options={[
            { value: "calm", description: "no wind" },
            { value: "steady", description: "a uniform wind from one direction" },
            { value: "gusting", description: "a uniform wind with turbulence" },
          ]}
        />
      </FormRow>
      {mode !== "calm" && (
        <>
          <FormRow label="direction" help="where it blows from">
            <NumberInput unit="°" step={5} value={env.wind_dir_deg ?? 270} onChange={(v) => set({ wind_dir_deg: v ?? 270 })} />
          </FormRow>
          <FormRow label="speed" help="the mean">
            <NumberInput unit="kt" step={5} min={0} value={env.wind_kts ?? 0} onChange={(v) => set({ wind_kts: v ?? 0 })} />
          </FormRow>
        </>
      )}
      {mode === "gusting" && (
        <>
          <FormRow label="turbulence" help="RMS">
            <NumberInput unit="kt" step={1} min={0} value={env.turbulence_kts ?? 0} onChange={(v) => set({ turbulence_kts: v ?? 0 })} />
          </FormRow>
          <FormRow label="gust τ" help="how long a gust holds">
            <NumberInput unit="s" step={5} min={0} value={env.gust_tau_s ?? 30} onChange={(v) => set({ gust_tau_s: v ?? 30 })} />
          </FormRow>
        </>
      )}
    </FormCard>
  );
}

// Full env configuration, in two columns: what is flown - the simulation's
// timing, the aircraft and their performance model - and the world it is flown
// in: conflict detection, wind, automation.
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
  const bs = catalog?.bluesky_defaults ?? {};
  return (
    <>
      <FormColumn>
        <FormCard title="Simulation" help="The step is how long one environment step lasts; the simulator advances in sim steps within it.">
          <FormRow label="step (dt)" help="per environment step">
            <NumberInput unit="s" step={0.1} min={0} value={env.dt} onChange={(v) => set({ dt: v })} />
          </FormRow>
          <FormRow label="sim step" help="per BlueSky update">
            <NumberInput unit="s" step={0.01} min={0} placeholder={`${bs.simdt ?? "BlueSky"} (default)`}
              value={env.simdt} onChange={(v) => set({ simdt: v })} />
          </FormRow>
          <FormRow label="performance">
            <Picker
              searchable={false}
              placeholder="openap"
              value={env.performance_model ?? "openap"}
              onChange={(v) => set({ performance_model: v })}
              options={[{ value: "openap" }, { value: "bada" }]}
            />
          </FormRow>
        </FormCard>

        <FormCard title="Aircraft" help="Each spawned aircraft's type is drawn from these.">
          {availError ? (
            <div className="error-text small">{model}: {availError}</div>
          ) : (
            <>
              <label className="form-check" title="sample uniformly across every aircraft type in the model">
                <input
                  type="checkbox"
                  checked={allSelected}
                  disabled={available.length === 0}
                  onChange={(e) => set({ allowed_aircraft: e.target.checked ? [...available] : [] })}
                />
                every type in {model} ({available.length})
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
                    {aircraft.length === 0 && <span className="muted">none yet</span>}
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
        </FormCard>
      </FormColumn>

      <FormColumn>
        <FormCard title="Conflict detection" help="Detection always runs, for observations; resolution is off by default so the agent resolves. Unset values use BlueSky's. Applied at each reset.">
          <FormRow label="method">
            <Picker
              searchable={false}
              placeholder="CSTATEBASED"
              value={env.cd_method ?? "CSTATEBASED"}
              onChange={(v) => set({ cd_method: v })}
              options={(catalog?.conflict?.cd_methods ?? ["CSTATEBASED", "STATEBASED"]).map((m: string) => ({ value: m }))}
            />
          </FormRow>
          <FormRow label="resolution">
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
          </FormRow>
          <FormRow label="check every" help="between detections; divides dt, a multiple of the sim step">
            <NumberInput unit="s" step={env.simdt ?? bs.simdt} min={0} placeholder={`${bs.asas_dt ?? "BlueSky"} (default)`}
              value={env.asas_dt} onChange={(v) => set({ asas_dt: v })} />
          </FormRow>
          <FormRow label="zone radius">
            <NumberInput unit="nm" step={0.5} min={0} placeholder="5 (default)" value={env.pz_radius_nm} onChange={(v) => set({ pz_radius_nm: v })} />
          </FormRow>
          <FormRow label="zone height">
            <NumberInput unit="ft" step={100} min={0} placeholder="1000 (default)" value={env.pz_height_ft} onChange={(v) => set({ pz_height_ft: v })} />
          </FormRow>
          <FormRow label="lookahead">
            <NumberInput unit="s" step={30} min={0} placeholder="300 (default)" value={env.lookahead_s} onChange={(v) => set({ lookahead_s: v })} />
          </FormRow>
        </FormCard>

        <WindCard env={env} set={set} />

        <FormCard title="Automation" help={<>What an aircraft on own navigation (LNAV + VNAV) does by itself.</>}>
          <label className="form-check" title="a fix's arrival time (arrival slack on the waypoint) becomes a BlueSky RTA">
            <input
              type="checkbox"
              checked={env.fly_arrival_times ?? false}
              onChange={(e) => set({ fly_arrival_times: e.target.checked })}
            />
            meet arrival times (RTA): own navigation adjusts speed to be over each fix on time
          </label>
        </FormCard>
      </FormColumn>
    </>
  );
}
