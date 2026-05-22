// Observation/action field lists: add/remove/parametrise field refs, edit
// constructor kwargs + normalization in a modal, and scaffold/edit custom field
// classes in custom_fields.py.
import { Fragment, useState } from "react";
import Editor from "@monaco-editor/react";
import type { SpecDict } from "../../api";
import { registerModalCompletions } from "./monacoCompletions";
import { FieldPicker } from "./FieldPicker";
import { Picker } from "./Picker";

export interface FieldOption {
  name: string;
  doc?: string;
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
  };
}

type QueryableFieldSpec = {
  kind: "any" | "region" | "waypoint";
  path: string;
  label: string;
  description?: string;
  requirements?: string[];
  cardinality?: "single" | "multiple" | "active";
  allow_empty_selection?: boolean;
};

// Add/remove/parametrise field refs ({field, kwargs}). Built-ins come from the
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
  // When set (intruder list), offer a second picker that turns any ownship
  // observation into an intruder-relative pair field via `.relative_to_own()`.
  allowRelative?: boolean;
}) {
  const [editing, setEditing] = useState<number | null>(null);
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
      <div className="chips">
        {fields.map((f, i) => {
          const opt = optByName(f.field);
          const custom = f.field.includes(":");
          const validationProblem = fieldHasValidationProblem(f.field, validationError);
          const problem = validationProblem;
          return (
            <span
              className={problem ? "chip invalid-chip" : "chip"}
              key={`${f.field}-${i}`}
              title={
                validationProblem
                    ? validationError
                    : opt?.doc || f.field
              }
            >
              {custom ? `⚙ ${f.field.split(":").pop()}` : f.field}
              <span className="chip-sig">{fieldSummary(f)}</span>
              {f.transform === "relative_to_own" && <span className="rel-badge" title="intruder value − ownship value"> −own</span>}
              {f.transform === "stacked" && (
                <span
                  className="rel-badge"
                  title={`frame stack: live + ${(Number(f.transform_kwargs?.depth) || 3) - 1} lagged copies`}
                >
                  {` ×${Number(f.transform_kwargs?.depth) || 3}`}
                </span>
              )}
              <button className="chip-cfg" title="configure" onClick={() => setEditing(i)}>
                ⚙
              </button>
              <button className="chip-x" onClick={() => onRemove(i)}>
                ✕
              </button>
            </span>
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
          kwargs={fields[editing].kwargs ?? {}}
          allowRelative={allowRelative}
          onChange={(kw) => setKwargs(editing, kw)}
          onFieldChange={(nextField) => setField(editing, nextField)}
          onClose={() => setEditing(null)}
        />
      )}

      <FieldPicker
        kind={label === "action" ? "action" : "obs"}
        placeholder={pickerLabel}
        options={options}
        onAdd={addField}
      />

      {onAddScaffold && (
        <div className="row">
          <button onClick={onAddScaffold} title="create a new custom field in custom_fields.py and edit it in the Code tab">
            + custom {label === "action" ? "action" : "observation"}
          </button>
        </div>
      )}
    </div>
  );
}

// A compact summary of a field's configured "signature" for its chip, e.g.
// `(goal) · MinMax` or `(low=-1, high=1)`. Covers queryable selection,
// constructor kwargs, and the normalizer (which lives in transform_kwargs for
// intruder-relative fields).
function fieldSummary(f: SpecDict): string {
  const kw: SpecDict = f.kwargs ?? {};
  const parts: string[] = [];
  if (kw.query_name) parts.push(String(kw.query_name));
  if (Array.isArray(kw.query_names) && kw.query_names.length) parts.push(kw.query_names.join(","));
  for (const [k, v] of Object.entries(kw)) {
    if (k === "query_name" || k === "query_names" || k === "normalizer" || v == null) continue;
    parts.push(`${k}=${v}`);
  }
  const inner = parts.length ? `(${parts.join(", ")})` : "";
  const norm = f.transform_kwargs?.normalizer ?? kw.normalizer;
  const normTag = norm?.name ? ` · ${String(norm.name).replace(/Normalizer$/, "")}` : "";
  return `${inner}${normTag}`;
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
  const source = option?.profile?.source || customFieldSource(field.field, code) || customFieldTemplate(field.field, kind);
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
            {custom && (
              <label className="numfield inline">
                <span>name</span>
                <input
                  value={field.field.split(":")[1] ?? ""}
                  onChange={(e) => {
                    const renamed = renameCustomField(field.field, e.target.value, code, kind);
                    if (!renamed) return;
                    onCodeChange(renamed.code);
                    onFieldChange({ ...field, field: renamed.ref });
                  }}
                />
              </label>
            )}
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
          </div>

          <div className="modal-pane">
            <div className="sub-label">{option ? "frozen code profile" : "custom code profile"}</div>
            <ProfileBlock option={option} field={field} />
            {custom ? (
              <div className="custom-code-editor">
                <Editor
                  height="100%"
                  path={`custom_field_${field.field}.py`}
                  language="python"
                  value={source}
                  beforeMount={registerModalCompletions}
                  onChange={(value) => {
                    const nextSource = value ?? "";
                    onCodeChange(updateCustomFieldSource(field.field, code, nextSource, kind));
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
              <pre className="source-profile" aria-readonly="true"><code>{source}</code></pre>
            ) : (
              <div className="muted small">source profile unavailable for this custom reference</div>
            )}
          </div>
        </div>
      </div>
    </div>
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

function renameCustomField(
  ref: string,
  newNameRaw: string,
  code: Record<string, string>,
  kind: "obs" | "action",
): { ref: string; code: Record<string, string> } | null {
  if (!ref?.includes(":")) return null;
  const newName = newNameRaw.replace(/[^0-9a-zA-Z_]/g, "");
  if (!newName || /^[0-9]/.test(newName)) return null;
  const [moduleName, oldName] = ref.split(":", 2);
  if (newName === oldName) return { ref, code };
  const oldSource = customFieldSource(ref, code) || customFieldTemplate(ref, kind);
  const newRef = `${moduleName}:${newName}`;
  const newSource = replaceClassName(oldSource, oldName, newName);
  return {
    ref: newRef,
    code: updateCustomFieldSource(ref, code, newSource, kind),
  };
}

const CUSTOM_MODULE_HEADER = `"""Custom observation/action fields for this design.

Classes in this module are referenced as import strings, e.g.
\`custom_fields:MyField\`. Use the designer's field configuration modal to set
constructor bounds and normalizers for each referenced class.
"""

from __future__ import annotations

from dataclasses import dataclass

import bluesky as bs

from bluesky_sandbox.fields.base import (
    ActionField, ActionMeta, ActionMode, ControlAxis,
    ObsField, ObsMeta, ObsQuantity, Unit,
)
`;

function snakeCase(name: string): string {
  return name.replace(/([a-z0-9])([A-Z])/g, "$1_$2").replace(/[^0-9a-zA-Z_]+/g, "_").toLowerCase();
}

function customFieldTemplate(ref: string, kind: "obs" | "action"): string {
  const className = ref?.includes(":") ? ref.split(":", 2)[1] : "CustomField";
  const snake = snakeCase(className);
  if (kind === "action") {
    return `@dataclass(frozen=True)
class ${className}(ActionField):
    """Custom action: maps one agent action scalar to a BlueSky command."""

    meta = ActionMeta(
        "${snake}",
        Unit.UNITLESS,
        control_axis=ControlAxis.HEADING,
        mode=ActionMode.ABSOLUTE,
    )
    low: float = 0.0
    high: float = 1.0

    def set(self, idx: int, value: float) -> None:
        value = min(max(float(value), self.low), self.high)
        bs.stack.stack(f"HDG {bs.traf.id[idx]} {value:.6f}")

    def bounds(self, idx: int):
        return self._configured_bounds()
`;
  }
  return `@dataclass(frozen=True)
class ${className}(ObsField):
    """Custom observation: one scalar value per aircraft."""

    meta = ObsMeta(
        "${snake}",
        Unit.UNITLESS,
        ObsQuantity.DISTANCE,
    )
    low: float = -1.0
    high: float = 1.0

    def get(self, idx: int):
        return 0.0

    def bounds(self, idx: int):
        return self._configured_bounds()
`;
}

function updateCustomFieldSource(
  ref: string,
  code: Record<string, string>,
  classSource: string,
  kind: "obs" | "action",
): Record<string, string> {
  if (!ref?.includes(":")) return code;
  const [moduleName, className] = ref.split(":", 2);
  const fileName = `${moduleName}.py`;
  const existing = code[fileName] ?? CUSTOM_MODULE_HEADER;
  const replacement = classSource.trimEnd() || customFieldTemplate(ref, kind).trimEnd();
  const block = findClassBlock(existing, className);
  if (!block) {
    return { ...code, [fileName]: `${existing.trimEnd()}\n\n${replacement}\n` };
  }
  const lines = existing.split("\n");
  const next = [...lines.slice(0, block.start), ...replacement.split("\n"), ...lines.slice(block.end)].join("\n");
  return { ...code, [fileName]: next.endsWith("\n") ? next : `${next}\n` };
}
