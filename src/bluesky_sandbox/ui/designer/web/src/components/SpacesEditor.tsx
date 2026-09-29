import { useEffect, useRef, useState } from "react";
import { api, type SpecDict } from "../api";
import { clone, gcOrphanBounds, scaffoldClass, stripClass } from "../specHelpers";
import { Section } from "./panel/Section";
import { FieldList, namedActions } from "./panel/FieldList";
import { Picker } from "./panel/Picker";
import { useRefresh } from "../refresh";

const FIELD_LISTS = [
  "obs_fields",
  "intruder_obs_fields",
  "critic_obs_fields",
  "critic_intruder_obs_fields",
  "action_fields",
];

// The observation and action fields, edited: what the policy sees and what it
// commands. The Spaces tab shows it beside the spaces it builds.
export default function SpacesEditor({
  spec,
  onChange,
  validationError,
}: {
  spec: SpecDict;
  onChange: (next: SpecDict) => void;
  validationError?: string;
}) {
  const [catalog, setCatalog] = useState<any>(null);
  const refreshKey = useRefresh();
  useEffect(() => {
    api.catalogOnce().then(setCatalog).catch(() => setCatalog(null));
  }, [refreshKey]);

  // Edits build on the latest spec, not this render's: one gesture can make
  // several (a rename changes custom_fields.py and the field's ref), and each
  // must see the one before it.
  const latest = useRef(spec);
  latest.current = spec;
  const edit = (mut: (s: SpecDict) => void) => {
    const next = clone(latest.current);
    mut(next);
    gcOrphanBounds(next);
    latest.current = next;
    onChange(next);
  };

  const env = spec.env ?? {};
  const scaffolds = catalog?.scaffolds;
  // What a field's referring parameter can name: this design's own actions.
  const references = { action: namedActions(env.action_fields ?? [], catalog?.action_fields ?? []) };

  // Add a scaffolded custom field: append a class to custom_fields.py and a ref.
  const addScaffold = (kind: "obs" | "action", listKey: string) => {
    if (!scaffolds) return;
    edit((s) => {
      const code = { ...(s.code ?? {}) };
      let src = code["custom_fields.py"] ?? scaffolds.module_header;
      const base = kind === "obs" ? "CustomObs" : "CustomAct";
      let n = 1;
      while (src.includes(`class ${base}${n}(`)) n++;
      src += scaffoldClass(scaffolds, kind, `${base}${n}`);
      const name = `${base}${n}`;
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

  return (
    <div className="spaces-editor">
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
            scaffolds={scaffolds}
            references={references}
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
                scaffolds={scaffolds}
                references={references}
                onAddScaffold={() => addScaffold("obs", "intruder_obs_fields")}
                allowRelative
              />
            )}
            {env.intruder_obs_fields != null && (
              <label
                className="numfield inline"
                title={
                  "Whose envelope normalizes a non-pair field with per-aircraft bounds " +
                  "(e.g. CAS, altitude) in an intruder row. Only differs with mixed " +
                  "aircraft types; fixed low/high bounds make it irrelevant."
                }
              >
                <span>intruder scale</span>
                <Picker
                  searchable={false}
                  placeholder="ownship's envelope"
                  value={env.intruder_obs_bounds ?? "ownship"}
                  onChange={(v) => edit((s) => (s.env.intruder_obs_bounds = v))}
                  options={[
                    { value: "ownship", label: "ownship's envelope" },
                    { value: "intruder", label: "intruder's own envelope" },
                  ]}
                />
              </label>
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
                scaffolds={scaffolds}
                references={references}
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
                scaffolds={scaffolds}
                references={references}
                onAddScaffold={() => addScaffold("obs", "critic_intruder_obs_fields")}
                allowRelative
              />
            )}
          </div>
        </Section>

        {/* ----------------------------------------------------------- actions */}
        <Section title="Actions" subtitle="agent control axes" hint="What the policy can command each step. A continuous action takes a range, usually normalized to [-1, 1]; the field decides what that maps to (a heading delta, an altitude delta, a speed target). A switch takes 0 or 1. With any switch, the action space is a Dict of the two parts: continuous and binary.">
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
            scaffolds={scaffolds}
            references={references}
            onAddScaffold={() => addScaffold("action", "action_fields")}
          />
          <p className="strategy-note muted small">{actionStrategy(env, catalog)}</p>
        </Section>
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
  // Built-in actions say which part of the action space they are in; a custom
  // one's class does, which the designer cannot see.
  const kinds = fields.map((f: any) => (byName.get(f.field) as any)?.kind);
  const binary = kinds.filter((k: string) => k === "binary").length;
  const space = binary
    ? ` Action space: Dict of continuous (${fields.length - binary}) and binary (${binary}).`
    : "";
  return `Each agent commands: ${parts.join(", ")}.${space}`;
}
