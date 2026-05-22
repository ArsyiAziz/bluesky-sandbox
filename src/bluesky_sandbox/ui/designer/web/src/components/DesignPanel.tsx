import { useEffect, useState } from "react";
import { api, type SpecDict } from "../api";
import { clone, gcOrphanBounds, stripClass } from "../specHelpers";
import type { EditTarget } from "../map/types";
import { Section } from "./panel/Section";
import { FieldList } from "./panel/FieldList";
import { ConfigEditor } from "./panel/ConfigEditor";
import { SamplingReadout } from "./panel/SamplingReadout";
import { ObsSample } from "./panel/ObsSample";
import GeometryTab from "./panel/GeometryTab";

const FIELD_LISTS = [
  "obs_fields",
  "intruder_obs_fields",
  "critic_obs_fields",
  "critic_intruder_obs_fields",
  "action_fields",
];

// Structured editor for the whole design: geometry + role assignments, fields,
// full env configuration, and a live sampling readout. The spec object is the
// source of truth; every edit yields a new spec via onChange, which App also
// re-serialises into the code editor.
export default function DesignPanel({
  spec,
  onChange,
  onFocusBounds,
  seed,
  onSeedChange,
  viewCenter,
  hiddenElements,
  onToggleHidden,
  lockedElements,
  onToggleLocked,
  onHighlightRoute,
  selectedKey,
  onSelect,
  validationError,
  width,
}: {
  spec: SpecDict;
  onChange: (next: SpecDict) => void;
  onFocusBounds: (bounds: SpecDict) => void;
  seed: number;
  onSeedChange: (seed: number) => void;
  viewCenter?: [number, number];
  hiddenElements: Set<string>;
  onToggleHidden: (key: string) => void;
  lockedElements: Set<string>;
  onToggleLocked: (key: string) => void;
  onHighlightRoute: (name: string | null) => void;
  selectedKey?: string | null;
  onSelect?: (target: EditTarget | null) => void;
  validationError?: string;
  width?: number;
}) {
  const [catalog, setCatalog] = useState<any>(null);
  const [tab, setTab] = useState<"geometry" | "fields" | "env" | "sampling">("geometry");
  useEffect(() => {
    api.catalogOnce().then(setCatalog).catch(() => setCatalog(null));
  }, []);

  const edit = (mut: (s: SpecDict) => void) => {
    const next = clone(spec);
    mut(next);
    // Bounds are only created via elements; sweep any left unreferenced after
    // a delete or reassignment so orphans never accumulate.
    gcOrphanBounds(next);
    onChange(next);
  };

  const env = spec.env ?? {};
  const scaffolds = catalog?.scaffolds;

  // Add a scaffolded custom field: append a class to custom_fields.py and a ref.
  const addScaffold = (kind: "obs" | "action", listKey: string) => {
    if (!scaffolds) return;
    edit((s) => {
      const code = { ...(s.code ?? {}) };
      let src = code["custom_fields.py"] ?? scaffolds.module_header;
      const base = kind === "obs" ? "CustomObs" : "CustomAct";
      let n = 1;
      while (src.includes(`class ${base}${n}(`)) n++;
      const name = `${base}${n}`;
      const snake = name.replace(/([a-z])([A-Z])/g, "$1_$2").toLowerCase();
      const tmpl = kind === "obs" ? scaffolds.obs_field : scaffolds.action_field;
      src += tmpl.split("{name}").join(name).split("{snake}").join(snake);
      code["custom_fields.py"] = src;
      s.code = code;
      const ref = { field: `custom_fields:${name}` };
      s.env[listKey] = [...(s.env[listKey] ?? []), ref];
    });
  };

  // Remove a field; if it was a custom field now unused, delete its class too.
  const removeField = (listKey: string, idx: number) =>
    edit((s) => {
      const list = s.env[listKey] ?? [];
      const ref: string | undefined = list[idx]?.field;
      s.env[listKey] = list.filter((_: any, j: number) => j !== idx);
      if (ref?.startsWith("custom_fields:")) {
        const stillUsed = FIELD_LISTS.some((k) => (s.env[k] ?? []).some((f: any) => f.field === ref));
        const file = s.code?.["custom_fields.py"];
        if (!stillUsed && file) {
          s.code = { ...s.code, "custom_fields.py": stripClass(file, ref.split(":")[1]) };
        }
      }
    });

  const TABS: { id: typeof tab; label: string }[] = [
    { id: "geometry", label: "Geometry" },
    { id: "fields", label: "Fields" },
    { id: "env", label: "Config" },
    { id: "sampling", label: "Sampling" },
  ];

  return (
    <div className="design-panel" style={width ? { width } : undefined}>
      <nav className="panel-tabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "panel-tab active" : "panel-tab"} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "geometry" && (
        <GeometryTab
          spec={spec}
          onChange={onChange}
          onFocusBounds={onFocusBounds}
          viewCenter={viewCenter}
          hiddenElements={hiddenElements}
          onToggleHidden={onToggleHidden}
          lockedElements={lockedElements}
          onToggleLocked={onToggleLocked}
          onHighlightRoute={onHighlightRoute}
          selectedKey={selectedKey}
          onSelect={onSelect ?? (() => {})}
        />
      )}

      {tab === "fields" && (<>
      {/* ------------------------------------------------------ observations */}
      <Section title="Observations" subtitle="ownship + intruder features" hint="What the policy sees each step. Ownship fields describe the aircraft being controlled; intruder fields are repeated once per other aircraft in view. Critic-only fields are visible to the value network but never to the policy.">
        <FieldList
          label="ownship"
          fields={env.obs_fields ?? []}
          options={(catalog?.obs_fields ?? []).filter((f: any) => !f.pair_only)}
          normalizers={catalog?.normalizers ?? []}
          queryables={spec.queryables ?? {}}
          validationError={validationError}
          code={spec.code ?? {}}
          onCodeChange={(code) => edit((s) => (s.code = code))}
          onChange={(fl) => edit((s) => (s.env.obs_fields = fl))}
          onRemove={(i) => removeField("obs_fields", i)}
          onAddScaffold={() => addScaffold("obs", "obs_fields")}
        />
        <div className="intruder-block">
          <label className="radio">
            <input
              type="checkbox"
              checked={env.intruder_obs_fields != null}
              onChange={(e) => edit((s) => (s.env.intruder_obs_fields = e.target.checked ? [] : null))}
            />
            intruder observations
          </label>
          {env.intruder_obs_fields != null && (
            <FieldList
              label="intruder"
              fields={env.intruder_obs_fields}
              options={catalog?.obs_fields ?? []}
              normalizers={catalog?.normalizers ?? []}
              queryables={spec.queryables ?? {}}
              validationError={validationError}
              code={spec.code ?? {}}
              onCodeChange={(code) => edit((s) => (s.code = code))}
              onChange={(fl) => edit((s) => (s.env.intruder_obs_fields = fl))}
              onRemove={(i) => removeField("intruder_obs_fields", i)}
              onAddScaffold={() => addScaffold("obs", "intruder_obs_fields")}
              allowRelative
            />
          )}
        </div>
        <p className="strategy-note muted small">{observationStrategy(env)}</p>
      </Section>

      {/* --------------------------------------------- critic-only (privileged) */}
      {/* Asymmetric actor-critic / CTDE: these fields are folded into the value
          function's observation at training time but never reach the actor, so
          the deployed policy stays a function of the actor-side fields above. */}
      <Section
        title="Critic-only observations"
        subtitle="privileged — training only, hidden from the actor"
      >
        <p className="strategy-note muted small">
          Extra features the value function may exploit at training time (e.g.
          other aircraft's route intent or global state). They never reach the
          actor, so the deployed policy is unchanged. Leave both off for a
          symmetric actor-critic.
        </p>
        <div className="critic-block">
          <label className="radio">
            <input
              type="checkbox"
              checked={env.critic_obs_fields != null}
              onChange={(e) => edit((s) => (s.env.critic_obs_fields = e.target.checked ? [] : null))}
            />
            ownship (privileged)
          </label>
          {env.critic_obs_fields != null && (
            <FieldList
              label="critic ownship"
              fields={env.critic_obs_fields}
              options={(catalog?.obs_fields ?? []).filter((f: any) => !f.pair_only)}
              normalizers={catalog?.normalizers ?? []}
              queryables={spec.queryables ?? {}}
              validationError={validationError}
              code={spec.code ?? {}}
              onCodeChange={(code) => edit((s) => (s.code = code))}
              onChange={(fl) => edit((s) => (s.env.critic_obs_fields = fl))}
              onRemove={(i) => removeField("critic_obs_fields", i)}
              onAddScaffold={() => addScaffold("obs", "critic_obs_fields")}
            />
          )}
        </div>
        <div className="critic-block">
          <label className="radio">
            <input
              type="checkbox"
              checked={env.critic_intruder_obs_fields != null}
              onChange={(e) => edit((s) => (s.env.critic_intruder_obs_fields = e.target.checked ? [] : null))}
            />
            intruder (privileged)
          </label>
          {env.critic_intruder_obs_fields != null && (
            <FieldList
              label="critic intruder"
              fields={env.critic_intruder_obs_fields}
              options={catalog?.obs_fields ?? []}
              normalizers={catalog?.normalizers ?? []}
              queryables={spec.queryables ?? {}}
              validationError={validationError}
              code={spec.code ?? {}}
              onCodeChange={(code) => edit((s) => (s.code = code))}
              onChange={(fl) => edit((s) => (s.env.critic_intruder_obs_fields = fl))}
              onRemove={(i) => removeField("critic_intruder_obs_fields", i)}
              onAddScaffold={() => addScaffold("obs", "critic_intruder_obs_fields")}
              allowRelative
            />
          )}
        </div>
      </Section>

      {/* ----------------------------------------------------------- actions */}
      <Section title="Actions" subtitle="agent control axes" hint="What the policy can command each step. Each field is one continuous axis, normalised to [-1, 1]; the field decides what that maps to (a heading delta, an altitude delta, a speed target).">
        <FieldList
          label="action"
          fields={env.action_fields ?? []}
          options={catalog?.action_fields ?? []}
          normalizers={catalog?.normalizers ?? []}
          queryables={spec.queryables ?? {}}
          validationError={validationError}
          code={spec.code ?? {}}
          onCodeChange={(code) => edit((s) => (s.code = code))}
          onChange={(fl) => edit((s) => (s.env.action_fields = fl))}
          onRemove={(i) => removeField("action_fields", i)}
          onAddScaffold={() => addScaffold("action", "action_fields")}
        />
        <p className="strategy-note muted small">{actionStrategy(env, catalog)}</p>
      </Section>

      </>)}

      {tab === "env" && (<>
      {/* ----------------------------------------------------- configuration */}
      <Section title="Configuration" subtitle="env + code references" hint="Simulator settings and the code the task supplies: step length, aircraft types, performance model, and references to reward / termination / task-info hooks.">
        <ConfigEditor
          env={env}
          code={spec.code ?? {}}
          catalog={catalog}
          onChange={(ne) => edit((s) => (s.env = ne))}
          onCodeChange={(code) => edit((s) => (s.code = code))}
        />
      </Section>

      </>)}

      {tab === "sampling" && (
        <>
          <Section title="Sampling" subtitle="a drawn episode" hint="One episode drawn with the seed below - the concrete geometry, spawns and routes the design produces. Change the seed to see how much the design varies.">
            <SamplingReadout spec={spec} seed={seed} onSeedChange={onSeedChange} />
          </Section>
          <Section title="Observations" subtitle="what the policy sees" hint="The observation vector for a sampled agent, field by field, with the values it would actually receive. Use it to check normalisation and ordering.">
            <ObsSample spec={spec} seed={seed} />
          </Section>
        </>
      )}
    </div>
  );
}

// Channels one field ref contributes to the observation vector. A `stacked`
// ref is ONE chip but `depth` channels (live + depth-1 lagged copies), so a
// count of list entries understates the real width - which is the number the
// sentence below is claiming to state.
function fieldChannels(f: SpecDict): number {
  if (f.transform !== "stacked") return 1;
  return Math.max(1, Number(f.transform_kwargs?.depth) || 3);
}

function channelCount(fields: SpecDict[]): number {
  return fields.reduce((sum, f) => sum + fieldChannels(f), 0);
}

// One-line plain-English summary of the observation strategy, shown under the
// observation fields so the chosen features read as a sentence.
function observationStrategy(env: any): string {
  const ownFields = (env.obs_fields ?? []) as SpecDict[];
  const own = channelCount(ownFields);
  const ownStacked = ownFields.filter((f) => f.transform === "stacked").length;
  const intr = env.intruder_obs_fields;
  const ownPart =
    own === 0
      ? "Each agent observes no ownship features yet"
      : `Each agent observes ${own} ownship feature${own === 1 ? "" : "s"}` +
        (ownStacked ? ` (${ownStacked} frame-stacked)` : "");
  if (intr == null) return `${ownPart}; intruder observations are off (single-agent view).`;
  const intrFields = intr as SpecDict[];
  const rel = intrFields.filter((f) => f.transform === "relative_to_own").length;
  const stacked = intrFields.filter((f) => f.transform === "stacked").length;
  const n = channelCount(intrFields);
  if (intrFields.length === 0) {
    return `${ownPart}, plus intruder observations (no intruder features added yet).`;
  }
  const notes = [
    rel ? `${rel} relative to ownship` : "",
    stacked ? `${stacked} frame-stacked` : "",
  ].filter(Boolean);
  const notePart = notes.length ? ` (${notes.join(", ")})` : "";
  return `${ownPart}, plus ${n} feature${n === 1 ? "" : "s"}${notePart} for each nearby intruder.`;
}

// One-line summary of the action strategy, derived from each action field's
// control axis + mode (absolute / delta / switch) in the catalog.
function actionStrategy(env: any, catalog: any): string {
  const fields = env.action_fields ?? [];
  if (fields.length === 0) return "No actions yet — the agent cannot command the aircraft.";
  const byName = new Map((catalog?.action_fields ?? []).map((o: any) => [o.name, o]));
  const parts = fields.map((f: any) => {
    const meta = (byName.get(f.field) as any)?.profile?.meta ?? {};
    const axis = meta.control_axis ?? f.field;
    const mode = meta.mode ? ` (${meta.mode})` : "";
    return `${axis}${mode}`;
  });
  return `Each agent commands: ${parts.join(", ")}.`;
}
