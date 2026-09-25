// Observation/action field lists: add/remove/parametrize field refs, edit
// constructor kwargs + normalization in a modal, and scaffold/edit custom field
// classes in custom_fields.py.
import { Fragment, useEffect, useState } from "react";
import Editor from "@monaco-editor/react";
import type { SpecDict } from "../../api";
import { scaffoldClass, type Scaffolds } from "../../specHelpers";
import { registerPythonIntel } from "../../code/pythonEditor";
import { FieldPicker } from "./FieldPicker";
import { Picker } from "./Picker";
import { normalizerColor } from "../../normColors";

export interface FieldOption {
  name: string;
  doc?: string;
  // The module the field is defined in, and that module's description.
  category?: string;
  category_doc?: string;
  // Whether the constructor takes a normalizer (a switch action does not).
  normalizable?: boolean;
  pair_only?: boolean;
  params?: { name: string; type: string; default: any }[];
  queryable_spec?: QueryableFieldSpec | null;
  profile?: {
    module?: string;
    class_name?: string;
    signature?: string;
    meta?: Record<string, any>;
    queryable_spec?: QueryableFieldSpec | null;
    source?: string;
    trail?: TrailEntry[];
  };
}

// One function the field's value is computed with (see ui/designer/trail.py).
type TrailEntry = {
  name: string;
  location: string;
  depth: number;
  source: string | null;
  external: boolean;
};

type QueryableFieldSpec = {
  kind: "any" | "region" | "waypoint";
  path: string;
  label: string;
  description?: string;
  requirements?: string[];
  cardinality?: "single" | "multiple" | "active";
  allow_empty_selection?: boolean;
};

// Add/remove/parametrize field refs ({field, kwargs}). Built-ins come from the
// catalog (with docstrings on hover and editable constructor params); custom
// fields are referenced by import path or scaffolded into custom_fields.py.
export function FieldList({
  label,
  fields,
  options,
  normalizers,
  queryables,
  validationError,
  code,
  onCodeChange,
  onChange,
  onRemove,
  onAddScaffold,
  scaffolds,
  allowRelative,
}: {
  label: string;
  fields: SpecDict[];
  options: FieldOption[];
  normalizers: FieldOption[];
  queryables: Record<string, SpecDict>;
  validationError?: string;
  code: Record<string, string>;
  onCodeChange: (code: Record<string, string>) => void;
  onChange: (fields: SpecDict[]) => void;
  onRemove: (index: number) => void;
  onAddScaffold?: () => void;
  scaffolds?: Scaffolds;
  // When set (intruder list), offer a second picker that turns any ownship
  // observation into an intruder-relative pair field via `.relative_to_own()`.
  allowRelative?: boolean;
}) {
  const [editing, setEditing] = useState<number | null>(null);
  const normalizerOrder = normalizers.map((n) => n.name);
  const optByName = (n: string) => options.find((o) => o.name === n);
  const pickerLabel =
    label === "action"
      ? "Add an action…"
      : label === "intruder"
        ? "Add an intruder observation…"
        : "Add an ownship observation…";

  const setKwargs = (i: number, kwargs: SpecDict) =>
    onChange(fields.map((f, j) => (j === i ? { ...f, kwargs } : f)));
  const setField = (i: number, field: SpecDict) =>
    onChange(fields.map((f, j) => (j === i ? field : f)));

  const addField = (name: string) => {
    const kwargs = defaultKwargsForField(optByName(name), queryables);
    onChange([...fields, Object.keys(kwargs).length ? { field: name, kwargs } : { field: name }]);
  };

  return (
    <div className="field-list">
      <div className="field-rows">
        {fields.map((f, i) => {
          const opt = optByName(f.field);
          const custom = f.field.includes(":");
          const problem = fieldHasValidationProblem(f.field, validationError);
          const norm = fieldNormalizer(f);
          const params = fieldParams(f);
          const depth = Number(f.transform_kwargs?.depth) || 3;
          return (
            <div
              className={problem ? "field-row invalid" : "field-row"}
              key={`${f.field}-${i}`}
              title={problem ? validationError : opt?.doc || f.field}
              onClick={() => setEditing(i)}
            >
              <span className="field-row-name">{custom ? `⚙ ${f.field.split(":").pop()}` : f.field}</span>
              {f.transform === "relative_to_own" && (
                <span className="geo-row-kind" title="intruder value − ownship value">
                  − own
                </span>
              )}
              {f.transform === "stacked" && (
                <span className="geo-row-kind" title={`the live value and ${depth - 1} lagged copies`}>
                  lags {depth - 1}
                </span>
              )}
              <span className="field-row-norm">
                {norm && (
                  <>
                    <span className="mdp-dot" style={{ background: normalizerColor(normalizerOrder, norm) }} />
                    {norm.replace(/Normalizer$/, "")}
                  </>
                )}
              </span>
              <button
                className="field-row-x"
                title="remove"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove(i);
                }}
              >
                ✕
              </button>
              {params && (
                <span className="field-row-params" title={params}>
                  {params}
                </span>
              )}
            </div>
          );
        })}
        {fields.length === 0 && <span className="muted">no {label} fields</span>}
      </div>

      {editing != null && fields[editing] && (
        <FieldConfigModal
          field={fields[editing]}
          option={optByName(fields[editing].field)}
          kind={label === "action" ? "action" : "obs"}
          normalizers={normalizers}
          queryables={queryables}
          code={code}
          onCodeChange={onCodeChange}
          scaffolds={scaffolds}
          kwargs={fields[editing].kwargs ?? {}}
          allowRelative={allowRelative}
          onChange={(kw) => setKwargs(editing, kw)}
          onFieldChange={(nextField) => setField(editing, nextField)}
          onClose={() => setEditing(null)}
        />
      )}

      <div className="field-add">
        <FieldPicker
          kind={label === "action" ? "action" : "obs"}
          placeholder={pickerLabel}
          options={options}
          onAdd={addField}
        />
        {onAddScaffold && (
          <button onClick={onAddScaffold} title="create a new custom field in custom_fields.py and edit it in the Code tab">
            + custom
          </button>
        )}
      </div>
    </div>
  );
}

// A field's configured arguments in one line, e.g. `goal, low=-1, high=1`:
// queryable selection and constructor kwargs; the normalizer is shown apart.
function fieldParams(f: SpecDict): string {
  const kw: SpecDict = f.kwargs ?? {};
  const parts: string[] = [];
  if (kw.query_name) parts.push(String(kw.query_name));
  if (Array.isArray(kw.query_names) && kw.query_names.length) parts.push(kw.query_names.join(","));
  for (const [k, v] of Object.entries(kw)) {
    if (k === "query_name" || k === "query_names" || k === "normalizer" || v == null) continue;
    parts.push(`${k}=${v}`);
  }
  return parts.join(", ");
}

// The normalizer's name; an intruder-relative field keeps it in transform_kwargs.
function fieldNormalizer(f: SpecDict): string | null {
  const norm = f.transform_kwargs?.normalizer ?? f.kwargs?.normalizer;
  return norm?.name ? String(norm.name) : null;
}

function fieldHasValidationProblem(fieldName: string, error?: string): boolean {
  if (!error || !fieldName) return false;
  const shortName = fieldName.includes(":") ? fieldName.split(":").pop() ?? fieldName : fieldName;
  return quotedValues(error).includes(fieldName) || quotedValues(error).includes(shortName);
}

function referencedQueryableNames(
  spec: QueryableFieldSpec,
  kwargs: SpecDict,
  queryables: Record<string, SpecDict>,
): string[] {
  const compatible = compatibleQueryables(spec, queryables).map(({ name }) => name);
  const cardinality = spec.cardinality ?? "single";
  if (cardinality === "active" || cardinality === "multiple") {
    if (Array.isArray(kwargs.query_names) && kwargs.query_names.length > 0) {
      return kwargs.query_names.filter((name: string) => compatible.includes(name));
    }
    return spec.allow_empty_selection === false ? [] : compatible;
  }
  return kwargs.query_name ? [String(kwargs.query_name)] : [];
}

function quotedValues(text: string): string[] {
  const values: string[] = [];
  const re = /'([^']+)'|"([^"]+)"/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text))) values.push(match[1] ?? match[2]);
  return values;
}

// Edit a field/action in a modal: constructor kwargs, normalization, and source profile.
function FieldConfigModal({
  field,
  option,
  kind,
  normalizers,
  queryables,
  code,
  onCodeChange,
  scaffolds,
  kwargs,
  allowRelative,
  onChange,
  onFieldChange,
  onClose,
}: {
  field: SpecDict;
  option?: FieldOption;
  kind: "obs" | "action";
  normalizers: FieldOption[];
  queryables: Record<string, SpecDict>;
  code: Record<string, string>;
  onCodeChange: (code: Record<string, string>) => void;
  scaffolds?: Scaffolds;
  kwargs: SpecDict;
  allowRelative?: boolean;
  onChange: (kwargs: SpecDict) => void;
  onFieldChange: (field: SpecDict) => void;
  onClose: () => void;
}) {
  const queryableSpec = option?.queryable_spec ?? option?.profile?.queryable_spec ?? null;
  const params = (option?.params ?? []).filter((p) => p.name !== "normalizer");
  const genericParams = params.filter((p) => !isQueryableParam(p.name, queryableSpec));
  const effectiveParams = option ? params : CUSTOM_FIELD_PARAMS;
  // Intruder-relative fields are built via `.relative_to_own(...)`: the
  // normalizer is a transform kwarg, and base constructor bounds don't apply.
  const isRelative = field.transform === "relative_to_own";
  // Only a non-pair, non-queryable observation can be made relative (the
  // `.relative_to_own()` method lives on plain ObsFields).
  const relativeEligible = !!allowRelative && !option?.pair_only && !queryableSpec;
  const setRelative = (on: boolean) => {
    if (on) {
      onFieldChange({ ...field, transform: "relative_to_own" });
    } else {
      const { transform: _t, transform_kwargs: _tk, ...rest } = field;
      onFieldChange(rest);
    }
  };
  // Frame stacking via `.stacked(depth=n)`, which expands to the live field plus
  // n-1 `.lagged(steps=k)` copies. FieldRef carries ONE transform slot, so this
  // and `relative_to_own` are mutually exclusive - the toggles disable each
  // other rather than silently overwriting.
  const isStacked = field.transform === "stacked";
  const stackDepth = Number(field.transform_kwargs?.depth) || 3;
  const stackEligible = kind === "obs" && !isRelative;
  const setStacked = (on: boolean) => {
    if (on) {
      onFieldChange({ ...field, transform: "stacked", transform_kwargs: { depth: 3 } });
    } else {
      const { transform: _t, transform_kwargs: _tk, ...rest } = field;
      onFieldChange(rest);
    }
  };
  const setStackDepth = (depth: number) => {
    const d = Math.max(1, Math.min(9, Math.round(depth) || 3));
    onFieldChange({ ...field, transform: "stacked", transform_kwargs: { depth: d } });
  };
  const transformKwargs: SpecDict = field.transform_kwargs ?? {};
  const normalizer = (isRelative ? transformKwargs.normalizer : kwargs.normalizer)?.type === "normalizer"
    ? (isRelative ? transformKwargs.normalizer : kwargs.normalizer)
    : null;
  const setNormalizer = (value: SpecDict | null) => {
    if (isRelative) {
      const tk = { ...transformKwargs };
      if (!value) delete tk.normalizer;
      else tk.normalizer = value;
      onFieldChange({ ...field, transform_kwargs: tk });
    } else {
      const next = { ...kwargs };
      if (!value) delete next.normalizer;
      else next.normalizer = value;
      onChange(next);
    }
  };
  const custom = !option && field.field?.includes(":");
  const source =
    option?.profile?.source ||
    customFieldSource(field.field, code) ||
    customFieldTemplate(field.field, kind, scaffolds);
  // A switch action takes its value as given; nothing to normalize.
  const normalizable = option?.normalizable !== false;
  // Fixed for the modal's lifetime, so renaming the class does not swap the
  // editor's model out from under the cursor.
  const [editorPath] = useState(() => `custom_field_${field.field}.py`);
  const rename = (name: string) => {
    const renamed = renameCustomField(field.field, name, code, kind, scaffolds);
    if (!renamed) return false;
    onCodeChange(renamed.code);
    onFieldChange({ ...field, field: renamed.ref });
    return true;
  };
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className="modal field-config-modal" onMouseDown={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <div>
            <div className="modal-title">{option?.name ?? field.field}</div>
            <div className="muted small">{option?.doc ?? "custom field/action"}</div>
          </div>
          <button className="link" onClick={onClose}>done</button>
        </div>

        <div className="modal-grid">
          <div className="modal-pane">
            {custom && <CustomNameInput name={field.field.split(":")[1] ?? ""} onRename={rename} />}
            {relativeEligible && (
              <label className="radio modal-check rel-toggle">
                <input
                  type="checkbox"
                  checked={isRelative}
                  disabled={isStacked}
                  onChange={(e) => setRelative(e.target.checked)}
                />
                relative to ownship (intruder − ownship)
              </label>
            )}
            {stackEligible && (
              <>
                <label className="radio modal-check rel-toggle">
                  <input
                    type="checkbox"
                    checked={isStacked}
                    onChange={(e) => setStacked(e.target.checked)}
                  />
                  frame stack (live + lagged copies)
                </label>
                {isStacked && (
                  <>
                    <label className="numfield inline">
                      <span>depth</span>
                      <input
                        type="number"
                        min={1}
                        max={9}
                        value={stackDepth}
                        onChange={(e) => setStackDepth(Number(e.target.value))}
                      />
                    </label>
                    <div className="muted small field-doc">
                      Emits {stackDepth} channels: the live value plus{" "}
                      {stackDepth - 1} lagged {stackDepth === 2 ? "copy" : "copies"}{" "}
                      (t−1{stackDepth > 2 ? ` … t−${stackDepth - 1}` : ""}), each on
                      the same normalizer and bounds as the live one.
                    </div>
                  </>
                )}
              </>
            )}
            <div className="sub-label">constructor</div>
            {isRelative ? (
              <div className="muted small field-doc">
                Intruder-relative: <code>{field.field}(intruder) − {field.field}(ownship)</code>.
                Bounds are derived from the base field, so there are no constructor params to set.
              </div>
            ) : (
              <>
                {queryableSpec && (
                  <QueryableParamControls
                    spec={queryableSpec}
                    queryables={queryables}
                    kwargs={kwargs}
                    onChange={onChange}
                  />
                )}
                {(option ? genericParams : effectiveParams).length === 0 && !queryableSpec && (
                  <span className="muted small">no editable constructor params</span>
                )}
                {(option ? genericParams : effectiveParams).map((p) => (
                  <ParamInput
                    key={p.name}
                    param={p}
                    value={kwargs[p.name]}
                    onChange={(value) => {
                      const next = { ...kwargs };
                      if (value === undefined) delete next[p.name];
                      else next[p.name] = value;
                      onChange(next);
                    }}
                  />
                ))}
              </>
            )}

            {normalizable && (<>
            <div className="sub-label normalizer-label">normalization</div>
            <label className="numfield inline">
              <span>strategy</span>
              <Picker
                searchable={false}
                placeholder="Raw"
                value={normalizer?.name ?? ""}
                onChange={(v) =>
                  setNormalizer(v ? { type: "normalizer", name: v, kwargs: {} } : null)
                }
                options={[
                  { value: "", label: "Raw" },
                  ...normalizers.map((n) => ({ value: n.name, description: n.doc })),
                ]}
              />
            </label>
            {normalizer && (
              <NormalizerParams
                option={normalizers.find((n) => n.name === normalizer.name)}
                value={normalizer}
                onChange={(value) => setNormalizer(value)}
              />
            )}
            </>)}
          </div>

          <div className="modal-pane">
            <div className="sub-label">{option ? "frozen code profile" : "custom code profile"}</div>
            <ProfileBlock option={option} field={field} />
            {custom ? (
              <div className="custom-code-editor">
                <Editor
                  height="100%"
                  path={editorPath}
                  language="python"
                  value={source}
                  beforeMount={registerPythonIntel}
                  onChange={(value) => {
                    const nextSource = value ?? "";
                    // Renaming the class here renames the field, as the name
                    // box does; otherwise the ref would point at a class that
                    // is gone.
                    const typed = nextSource.match(/^class\s+([A-Za-z_]\w*)\s*[(:]/m)?.[1];
                    const [moduleName, current] = field.field.split(":", 2);
                    onCodeChange(updateCustomFieldSource(field.field, code, nextSource, kind, scaffolds));
                    if (typed && typed !== current && !classDefined(code, field.field, typed)) {
                      onFieldChange({ ...field, field: `${moduleName}:${typed}` });
                    }
                  }}
                  theme="vs-dark"
                  options={{
                    minimap: { enabled: false },
                    fontSize: 12,
                    tabSize: 4,
                    scrollBeyondLastLine: false,
                    automaticLayout: true,
                  }}
                />
              </div>
            ) : source ? (
              <>
                <pre className="source-profile" aria-readonly="true"><code>{source}</code></pre>
                <CallTrail trail={option?.profile?.trail ?? []} />
              </>
            ) : (
              <div className="muted small">source profile unavailable for this custom reference</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// The custom field's class name. Edits apply as they are typed while the name
// is a valid, unused class name; anything else is kept as a draft and flagged.
function CustomNameInput({ name, onRename }: { name: string; onRename: (name: string) => boolean }) {
  const [draft, setDraft] = useState(name);
  useEffect(() => setDraft(name), [name]);
  const problem =
    draft === name
      ? ""
      : !/^[A-Za-z_]\w*$/.test(draft)
        ? "not a valid class name"
        : "already defined in this module";
  return (
    <>
      <label className="numfield inline">
        <span>name</span>
        <input
          value={draft}
          aria-invalid={problem !== ""}
          onChange={(e) => {
            const next = e.target.value;
            setDraft(next);
            onRename(next);
          }}
          onBlur={() => setDraft(name)}
        />
      </label>
      {problem && <div className="muted small field-doc">{problem}; still named {name}</div>}
    </>
  );
}

function isQueryableParam(name: string, spec: QueryableFieldSpec | null): boolean {
  if (!spec) return false;
  if (spec.cardinality === "active" || spec.cardinality === "multiple") {
    return name === "query_names";
  }
  return name === "query_name";
}

function QueryableParamControls({
  spec,
  queryables,
  kwargs,
  onChange,
}: {
  spec: QueryableFieldSpec;
  queryables: Record<string, SpecDict>;
  kwargs: SpecDict;
  onChange: (kwargs: SpecDict) => void;
}) {
  const compatible = compatibleQueryables(spec, queryables);
  const path = `context.query("name").${spec.path}`;
  const cardinality = spec.cardinality ?? "single";
  if (cardinality === "active" || cardinality === "multiple") {
    return (
      <div className="queryable-param">
        <div className="muted small field-doc">{spec.description || spec.label} · {path}</div>
        {compatible.length === 0 ? (
          <label className="numfield inline">
            <span>query_names</span>
            <Picker
              disabled
              placeholder={missingQueryableReason(spec)}
              value=""
              onChange={() => {}}
              options={[{ value: "", label: missingQueryableReason(spec) }]}
            />
          </label>
        ) : (
          <>
            <div className="muted small">
              {(kwargs.query_names?.length ?? 0) === 0 && spec.allow_empty_selection !== false
                ? "using all compatible queryables"
                : "using selected queryables"}
            </div>
            <div className="queryable-checks">
              {compatible.map(({ name }) => {
                const explicit = Array.isArray(kwargs.query_names) && kwargs.query_names.length > 0;
                const selected = explicit ? kwargs.query_names.includes(name) : true;
                return (
                  <label className="radio modal-check" key={name}>
                    <input
                      type="checkbox"
                      checked={selected}
                      onChange={(e) => {
                        const allNames = compatible.map((q) => q.name);
                        const current = explicit ? [...kwargs.query_names] : allNames;
                        const nextNames = e.target.checked
                          ? [...new Set([...current, name])]
                          : current.filter((n: string) => n !== name);
                        const next = { ...kwargs };
                        if (
                          spec.allow_empty_selection !== false &&
                          (nextNames.length === 0 || nextNames.length === allNames.length)
                        ) {
                          delete next.query_names;
                        }
                        else next.query_names = nextNames;
                        onChange(next);
                      }}
                    />
                    {name}
                  </label>
                );
              })}
            </div>
          </>
        )}
      </div>
    );
  }

  const value = kwargs.query_name ?? "";
  return (
    <div className="queryable-param">
      <div className="muted small field-doc">{spec.description || spec.label} · {path}</div>
      <label className="numfield inline">
        <span>query_name</span>
        <Picker
          disabled={compatible.length === 0}
          placeholder={compatible.length ? "select queryable" : missingQueryableReason(spec)}
          value={value}
          onChange={(v) => {
            const next = { ...kwargs };
            if (!v) delete next.query_name;
            else next.query_name = v;
            onChange(next);
          }}
          options={[
            { value: "", label: compatible.length ? "select queryable" : missingQueryableReason(spec) },
            ...compatible.map(({ name }) => ({ value: name })),
          ]}
        />
      </label>
    </div>
  );
}

function compatibleQueryables(
  spec: QueryableFieldSpec,
  queryables: Record<string, SpecDict>,
): { name: string; queryable: SpecDict }[] {
  return Object.entries(queryables ?? {})
    .filter(([, q]) => queryableKind(q) === spec.kind || spec.kind === "any")
    .filter(([, q]) => queryableSatisfiesRequirements(q, spec.requirements ?? []))
    .map(([name, queryable]) => ({ name, queryable }));
}

function queryableKind(q: SpecDict): "region" | "waypoint" {
  return q?.type === "waypoint" ? "waypoint" : "region";
}

function queryableSatisfiesRequirements(q: SpecDict, requirements: string[]): boolean {
  for (const requirement of requirements) {
    if (requirement === "altitude" && !Number.isFinite(q.alt_ft)) return false;
    if (requirement === "speed" && !Number.isFinite(q.speed_kts)) return false;
    if (
      requirement === "tolerance" &&
      q.reach_radius_nm == null &&
      q.alt_tolerance_ft == null &&
      q.speed_tolerance_kts == null &&
      q.speed_tolerance_mach == null
    ) {
      return false;
    }
  }
  return true;
}

function missingQueryableReason(spec: QueryableFieldSpec): string {
  const kind = spec.kind === "any" ? "queryable" : spec.kind;
  const req = (spec.requirements ?? []).filter((r) => !["route", "step", "time"].includes(r));
  return req.length ? `no compatible ${kind} (${req.join(", ")})` : `no compatible ${kind}`;
}

function defaultKwargsForField(option: FieldOption | undefined, queryables: Record<string, SpecDict>): SpecDict {
  const spec = option?.queryable_spec ?? option?.profile?.queryable_spec ?? null;
  if (!spec) return {};
  const cardinality = spec.cardinality ?? "single";
  const compatible = compatibleQueryables(spec, queryables);
  if (cardinality !== "single") {
    return spec.allow_empty_selection === false && compatible.length
      ? { query_names: compatible.map((q) => q.name) }
      : {};
  }
  return compatible.length ? { query_name: compatible[0].name } : {};
}

function ParamInput({
  param,
  value,
  onChange,
}: {
  param: { name: string; type: string; default: any };
  value: any;
  onChange: (value: any | undefined) => void;
}) {
  if (typeof param.default === "boolean") {
    return (
      <label className="radio modal-check">
        <input
          type="checkbox"
          checked={value ?? param.default}
          onChange={(e) => onChange(e.target.checked)}
        />
        {param.name}
      </label>
    );
  }
  const isNum = typeof param.default === "number" || param.default === null;
  return (
    <label className="numfield inline">
      <span>{param.name}</span>
      <input
        type={isNum ? "number" : "text"}
        placeholder={param.default === null ? "default" : String(param.default)}
        value={value ?? ""}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") onChange(undefined);
          else onChange(isNum ? parseFloat(raw) : raw);
        }}
      />
    </label>
  );
}

const CUSTOM_FIELD_PARAMS = [
  { name: "low", type: "float", default: null },
  { name: "high", type: "float", default: null },
];

function NormalizerParams({
  option,
  value,
  onChange,
}: {
  option?: FieldOption;
  value: SpecDict;
  onChange: (value: SpecDict) => void;
}) {
  const params = option?.params ?? [];
  if (!params.length) return option?.doc ? <div className="muted small field-doc">{option.doc}</div> : null;
  return (
    <div className="normalizer-params">
      {option?.doc && <div className="muted small field-doc">{option.doc}</div>}
      {params.map((p) => (
        <ParamInput
          key={p.name}
          param={p}
          value={value.kwargs?.[p.name]}
          onChange={(paramValue) => {
            const kwargs = { ...(value.kwargs ?? {}) };
            if (paramValue === undefined) delete kwargs[p.name];
            else kwargs[p.name] = paramValue;
            onChange({ ...value, kwargs });
          }}
        />
      ))}
    </div>
  );
}

// The functions a built-in field's value is computed with, in call order and
// indented by call depth: its class source is often a single call into these.
function CallTrail({ trail }: { trail: TrailEntry[] }) {
  if (!trail.length) return null;
  return (
    <div className="call-trail">
      <div className="sub-label">computed with</div>
      {trail.map((entry, i) =>
        entry.source ? (
          <details key={i} className="trail-entry" style={{ marginLeft: entry.depth * 14 }}>
            <summary>
              <code>{entry.name}</code>
              <span className="trail-location">{entry.location}</span>
            </summary>
            <pre className="source-profile" aria-readonly="true"><code>{entry.source}</code></pre>
          </details>
        ) : (
          <div key={i} className="trail-entry trail-leaf" style={{ marginLeft: entry.depth * 14 }}>
            <code>{entry.name}</code>
            <span className="trail-location">{entry.external ? `BlueSky · ${entry.location}` : entry.location}</span>
          </div>
        ),
      )}
    </div>
  );
}

function ProfileBlock({ option, field }: { option?: FieldOption; field: SpecDict }) {
  const profile = option?.profile;
  const meta = profile?.meta ?? {};
  return (
    <dl className="profile-list">
      <dt>ref</dt><dd>{field.field}</dd>
      {profile?.module && <><dt>module</dt><dd>{profile.module}</dd></>}
      {profile?.signature && <><dt>signature</dt><dd>{profile.signature}</dd></>}
      {Object.entries(meta).map(([key, value]) => (
        <Fragment key={key}>
          <dt>{key}</dt>
          <dd>{Array.isArray(value) ? value.join(", ") : String(value)}</dd>
        </Fragment>
      ))}
    </dl>
  );
}

function customFieldSource(ref: string, code: Record<string, string>): string {
  if (!ref?.includes(":")) return "";
  const [moduleName, className] = ref.split(":", 2);
  const source = code[`${moduleName}.py`] ?? "";
  if (!source) return "";
  const block = findClassBlock(source, className);
  if (!block) return source;
  return source.split("\n").slice(block.start, block.end).join("\n");
}

function findClassBlock(source: string, className: string): { start: number; classLine: number; end: number } | null {
  const lines = source.split("\n");
  const startRe = new RegExp(`^class\\s+${className}\\s*[(:]`);
  const classLine = lines.findIndex((line) => startRe.test(line));
  if (classLine < 0) return null;
  let start = classLine;
  while (start > 0 && lines[start - 1].trimStart().startsWith("@")) start--;
  let end = classLine + 1;
  while (end < lines.length && !/^(class\s|def\s|@)/.test(lines[end])) end++;
  return { start, classLine, end };
}

function replaceClassName(source: string, oldName: string, newName: string): string {
  const oldSnake = snakeCase(oldName);
  const newSnake = snakeCase(newName);
  return source
    .replace(new RegExp(`class\\s+${oldName}\\s*\\(`), `class ${newName}(`)
    .replace(new RegExp(`"${oldSnake}"`), `"${newSnake}"`);
}

// Whether the module behind `ref` already defines a class `name`.
function classDefined(code: Record<string, string>, ref: string, name: string): boolean {
  const moduleName = ref.split(":", 2)[0];
  return findClassBlock(code[`${moduleName}.py`] ?? "", name) !== null;
}

function renameCustomField(
  ref: string,
  newName: string,
  code: Record<string, string>,
  kind: "obs" | "action",
  scaffolds?: Scaffolds,
): { ref: string; code: Record<string, string> } | null {
  if (!ref?.includes(":") || !/^[A-Za-z_]\w*$/.test(newName)) return null;
  const [moduleName, oldName] = ref.split(":", 2);
  if (newName === oldName) return { ref, code };
  if (classDefined(code, ref, newName)) return null;
  const oldSource = customFieldSource(ref, code) || customFieldTemplate(ref, kind, scaffolds);
  const newRef = `${moduleName}:${newName}`;
  const newSource = replaceClassName(oldSource, oldName, newName);
  return {
    ref: newRef,
    code: updateCustomFieldSource(ref, code, newSource, kind, scaffolds),
  };
}

function snakeCase(name: string): string {
  return name.replace(/([a-z0-9])([A-Z])/g, "$1_$2").replace(/[^0-9a-zA-Z_]+/g, "_").toLowerCase();
}

// A new class for a custom ref, from the catalog's template.
function customFieldTemplate(ref: string, kind: "obs" | "action", scaffolds?: Scaffolds): string {
  if (!scaffolds) return "";
  const className = ref?.includes(":") ? ref.split(":", 2)[1] : "CustomField";
  return scaffoldClass(scaffolds, kind, className).trim() + "\n";
}

function updateCustomFieldSource(
  ref: string,
  code: Record<string, string>,
  classSource: string,
  kind: "obs" | "action",
  scaffolds?: Scaffolds,
): Record<string, string> {
  if (!ref?.includes(":")) return code;
  const [moduleName, className] = ref.split(":", 2);
  const fileName = `${moduleName}.py`;
  const existing = code[fileName] ?? scaffolds?.module_header ?? "";
  const replacement = classSource.trimEnd() || customFieldTemplate(ref, kind, scaffolds).trimEnd();
  const block = findClassBlock(existing, className);
  if (!block) {
    return { ...code, [fileName]: `${existing.trimEnd()}\n\n${replacement}\n` };
  }
  const lines = existing.split("\n");
  const next = [...lines.slice(0, block.start), ...replacement.split("\n"), ...lines.slice(block.end)].join("\n");
  return { ...code, [fileName]: next.endsWith("\n") ? next : `${next}\n` };
}
