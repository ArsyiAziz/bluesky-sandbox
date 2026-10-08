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
import { editorTheme, useTheme } from "../../theme";

export interface FieldOption {
  name: string;
  doc?: string;
  // The module the field is defined in, and that module's description.
  category?: string;
  category_doc?: string;
  // Whether the constructor takes a normalizer (a switch action does not).
  normalizable?: boolean;
  // Whether an action takes a grid (a switch does not).
  griddable?: boolean;
  pair_only?: boolean;
  params?: FieldParam[];
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

// A constructor parameter. `refers` says what it names - "action": one of the
// design's actions - so it is picked from the design's own (`references`).
// `blank` says what leaving an optional one blank means, as its code says it.
type FieldParam = { name: string; type: string; default: any; refers?: string; blank?: string };

// A choice a referring parameter can take.
export type Choice = { value: string; label?: string };

// The design's configured actions another field can name, each by its meta
// name, read from the catalog: not a custom field (its name is unknown here),
// nor one that itself names an action.
export function namedActions(fields: SpecDict[], options: FieldOption[]): Choice[] {
  const out: Choice[] = [];
  for (const f of fields) {
    const option = options.find((o) => o.name === f.field);
    const name = option?.profile?.meta?.name;
    if (!name || option?.params?.some((p) => p.refers) || out.some((c) => c.value === name)) continue;
    out.push({ value: name, label: `${name} · ${f.field}` });
  }
  return out;
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
  references = {},
  addLabel,
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
  // What a referring parameter can name, by what it refers to (see FieldParam).
  references?: Record<string, Choice[]>;
  // The add picker's placeholder, when the label alone does not say it.
  addLabel?: string;
}) {
  const [editing, setEditing] = useState<number | null>(null);
  // A row being dragged, and the gap it would drop into (0 is before the first).
  const [dragFrom, setDragFrom] = useState<number | null>(null);
  const [dropAt, setDropAt] = useState<number | null>(null);
  // Move the field at `from` into the gap `gap`, keeping the rest in order.
  // Reordering changes the observation's column order, which a trained policy
  // depends on - so a drag starts only from a row's grip, and every move says
  // what it did, with a way back.
  const [armed, setArmed] = useState<number | null>(null);
  const [moved, setMoved] = useState<{ name: string; from: number; to: number; before: SpecDict[] } | null>(null);
  const move = (from: number, gap: number) => {
    const to = gap > from ? gap - 1 : gap;
    if (to === from || to < 0 || to >= fields.length) return;
    const next = [...fields];
    const [field] = next.splice(from, 1);
    next.splice(to, 0, field);
    setMoved({ name: field.field.split(":").pop() ?? field.field, from, to, before: fields });
    onChange(next);
  };
  // The gap a pointer over row i points at: before it in its upper half.
  const gapAt = (e: React.DragEvent<HTMLElement>, i: number) => {
    const box = e.currentTarget.getBoundingClientRect();
    return e.clientY < box.top + box.height / 2 ? i : i + 1;
  };
  const endDrag = () => {
    setDragFrom(null);
    setDropAt(null);
    setArmed(null);
  };
  const normalizerOrder = normalizers.map((n) => n.name);
  const optByName = (n: string) => options.find((o) => o.name === n);
  const pickerLabel =
    addLabel ??
    (label === "action"
      ? "Add an action…"
      : label === "intruder"
        ? "Add an intruder observation…"
        : "Add an ownship observation…");

  const setKwargs = (i: number, kwargs: SpecDict) =>
    onChange(fields.map((f, j) => (j === i ? { ...f, kwargs } : f)));
  const setField = (i: number, field: SpecDict) =>
    onChange(fields.map((f, j) => (j === i ? field : f)));

  const addField = (name: string) => {
    const kwargs = defaultKwargsForField(optByName(name), queryables, references);
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
              className={[
                "field-row",
                problem ? "invalid" : "",
                dragFrom === i ? "dragging" : "",
                dragFrom !== null && dropAt === i ? "drop-before" : "",
                dragFrom !== null && dropAt === i + 1 && i === fields.length - 1 ? "drop-after" : "",
              ].join(" ")}
              key={`${f.field}-${i}`}
              title={`${problem ? validationError : opt?.doc || f.field}\n\nDrag, or Alt+↑/↓, to reorder.`}
              tabIndex={0}
              draggable={armed === i}
              onClick={() => setEditing(i)}
              onKeyDown={(e) => {
                if (e.key === "Enter") setEditing(i);
                else if (e.altKey && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
                  e.preventDefault();
                  move(i, e.key === "ArrowUp" ? i - 1 : i + 2);
                }
              }}
              onDragStart={(e) => {
                if (armed !== i) {
                  e.preventDefault();
                  return;
                }
                e.dataTransfer.effectAllowed = "move";
                e.dataTransfer.setData("text/plain", f.field);
                setDragFrom(i);
              }}
              onDragOver={(e) => {
                if (dragFrom === null) return; // another list's row, or not a row
                e.preventDefault();
                setDropAt(gapAt(e, i));
              }}
              onDrop={(e) => {
                e.preventDefault();
                if (dragFrom !== null) move(dragFrom, gapAt(e, i));
                endDrag();
              }}
              onDragEnd={endDrag}
            >
              <span
                className="field-row-grip"
                title="drag to reorder"
                onPointerDown={() => setArmed(i)}
                onPointerUp={() => setArmed(null)}
                onClick={(e) => e.stopPropagation()}
              >
                ⠿
              </span>
              <span className="field-row-num">{i + 1}</span>
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
      {moved && (
        <div className="field-moved" role="status">
          <span>
            Moved <b>{moved.name}</b> from {moved.from + 1} to {moved.to + 1}. The observation's columns reorder, so a
            policy trained on the old order will not match.
          </span>
          <button
            onClick={() => {
              onChange(moved.before);
              setMoved(null);
            }}
          >
            Undo
          </button>
          <button className="link" onClick={() => setMoved(null)} aria-label="dismiss">
            ✕
          </button>
        </div>
      )}

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
          references={references}
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
  if (f.clearance) {
    const c = f.clearance;
    const dur = Array.isArray(c.duration) ? ` ${c.duration[0]}-${c.duration[1]} s` : "";
    const lock = c.lock ? `, lock ${c.lock === "duration" ? "for duration" : "until captured"}` : "";
    parts.push(`clearance${dur}${lock}`);
  }
  if (kw.grid?.type === "grid") {
    parts.push(`grid ${kw.grid.step}${kw.grid.on === "target" ? " on the target" : ""}`);
  }
  if (kw.query_name) parts.push(String(kw.query_name));
  if (Array.isArray(kw.query_names) && kw.query_names.length) parts.push(kw.query_names.join(","));
  for (const [k, v] of Object.entries(kw)) {
    if (["query_name", "query_names", "normalizer", "grid"].includes(k) || v == null) continue;
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
  references,
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
  references: Record<string, Choice[]>;
  onChange: (kwargs: SpecDict) => void;
  onFieldChange: (field: SpecDict) => void;
  onClose: () => void;
}) {
  const { theme } = useTheme();
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
                    choices={p.refers ? references[p.refers] ?? [] : undefined}
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

            {kind === "action" && option?.griddable !== false && (
              <GridControls
                value={kwargs.grid?.type === "grid" ? kwargs.grid : null}
                onChange={(grid) => {
                  const next = { ...kwargs };
                  const moved = regrid(normalizer, kwargs.grid ?? null, grid);
                  if (moved.normalizer) next.normalizer = moved.normalizer;
                  if (moved.grid) next.grid = moved.grid;
                  else delete next.grid;
                  onChange(next);
                }}
              />
            )}

            {kind === "action" && (
              <ClearanceControls
                normalizers={normalizers}
                value={field.clearance ?? null}
                onChange={(clearance) => {
                  const next = { ...field };
                  if (clearance) next.clearance = clearance;
                  else delete next.clearance;
                  onFieldChange(next);
                }}
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
                  theme={editorTheme(theme)}
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

function defaultKwargsForField(
  option: FieldOption | undefined,
  queryables: Record<string, SpecDict>,
  references: Record<string, Choice[]>,
): SpecDict {
  // A parameter naming one of the design's own starts on the first of them.
  const named: SpecDict = {};
  for (const p of option?.params ?? []) {
    const first = p.refers ? references[p.refers]?.[0] : undefined;
    if (first) named[p.name] = first.value;
  }
  const spec = option?.queryable_spec ?? option?.profile?.queryable_spec ?? null;
  if (!spec) return named;
  const cardinality = spec.cardinality ?? "single";
  const compatible = compatibleQueryables(spec, queryables);
  if (cardinality !== "single") {
    return spec.allow_empty_selection === false && compatible.length
      ? { query_names: compatible.map((q) => q.name) }
      : {};
  }
  return compatible.length ? { ...named, query_name: compatible[0].name } : named;
}

function ParamInput({
  param,
  choices,
  value,
  onChange,
}: {
  param: FieldParam;
  // Set for a referring parameter: what it can name in this design.
  choices?: Choice[];
  value: any;
  onChange: (value: any | undefined) => void;
}) {
  if (choices) {
    // A value no longer in the design stays listed, so it is visible to fix.
    const stale = value && !choices.some((c) => c.value === value);
    const none = choices.length ? `select ${param.refers}` : `no ${param.refers} in this design`;
    return (
      <label className="numfield inline">
        <span>{param.name}</span>
        <Picker
          disabled={choices.length === 0 && !stale}
          placeholder={none}
          value={value ?? ""}
          onChange={(v) => onChange(v || undefined)}
          options={[
            { value: "", label: none },
            ...choices,
            ...(stale ? [{ value: String(value), label: `${value} (not in this design)` }] : []),
          ]}
        />
      </label>
    );
  }
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
        placeholder={param.default === null ? param.blank ?? "default" : String(param.default)}
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

const CUSTOM_FIELD_PARAMS: FieldParam[] = [
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


// A grid set or taken away, and the step normalizer with it: on a grid its
// step counts grid steps, so its step (in the action's unit) becomes the grid's
// when one is set, and grid step × count when it is taken away - the steps the
// same either way. Any other normalizer is left as it is.
function regrid(
  normalizer: SpecDict | null,
  from: SpecDict | null,
  to: SpecDict | null,
): { normalizer: SpecDict | null; grid: SpecDict | null } {
  const steps = normalizer?.name === "StepNormalizer" ? normalizer : null;
  if (!steps) return { normalizer, grid: to };
  const kwargs: SpecDict = { ...(steps.kwargs ?? {}) };
  if (to && !from) {
    if (kwargs.step != null) to = { ...to, step: kwargs.step };
    delete kwargs.step;
    return { normalizer: { ...steps, kwargs }, grid: to };
  }
  if (!to && from?.step != null) {
    const step = Number(from.step) * Number(kwargs.step ?? 1);
    return { normalizer: { ...steps, kwargs: { ...kwargs, step } }, grid: null };
  }
  return { normalizer, grid: to };
}

// An action's grid (actions.Grid): whole steps, after the normalizer - of the
// action's own value (a delta's change) or of the target it commands (flight
// levels), in the action's unit. `target` offers the choice between them: an
// absolute value (a duration) has only the one.
function GridControls({
  value,
  onChange,
  label = "grid",
  startStep = 1000,
  target = true,
}: {
  value: SpecDict | null;
  onChange: (value: SpecDict | null) => void;
  label?: string;
  startStep?: number;
  target?: boolean;
}) {
  return (
    <>
      <div className="sub-label">{label}</div>
      <label className="radio modal-check" title="applied after the normalizer, within what the aircraft can be commanded">
        <input
          type="checkbox"
          checked={value != null}
          onChange={(e) => onChange(e.target.checked ? { type: "grid", step: startStep, on: "value" } : null)}
        />
        on a grid: whole steps
      </label>
      {value != null && (
        <>
          <label className="numfield inline">
            <span>step</span>
            <input
              type="number"
              min={0}
              value={value.step ?? ""}
              onChange={(e) => onChange({ ...value, step: Number(e.target.value) })}
            />
          </label>
          {target && <label className="numfield inline">
            <span>on</span>
            <Picker
              searchable={false}
              placeholder="the value"
              value={value.on ?? "value"}
              onChange={(v) => onChange({ ...value, on: v || "value" })}
              options={[
                { value: "value", label: "the value", description: "the action's own value on the grid: a delta's change - +1,000 ft from 23,344 ft is 24,344 ft" },
                { value: "target", label: "the target", description: "what it commands on the grid: flight levels - +1,000 ft from 23,344 ft is FL240" },
              ]}
            />
          </label>}
          <div className="muted small">In the action's unit. A step normalizer steps along it.</div>
        </>
      )}
    </>
  );
}

// An action as a clearance (actions.Clearance): the policy gives it when it
// decides to - its mask - for a duration it chooses, after which own navigation
// takes the axis back; the lock sets how long nothing else is accepted on the
// axis. What the policy observes of it (ActionLocked, ClearanceTimeLeftS) is
// added like any other observation.
function ClearanceControls({
  value,
  normalizers,
  onChange,
}: {
  value: SpecDict | null;
  normalizers: FieldOption[];
  onChange: (value: SpecDict | null) => void;
}) {
  const duration: [number, number] | null = Array.isArray(value?.duration)
    ? [Number(value!.duration[0]), Number(value!.duration[1])]
    : null;
  const set = (patch: SpecDict) => {
    const next: SpecDict = { ...(value ?? {}), ...patch };
    if (!Array.isArray(next.duration)) {
      if (next.lock === "duration") next.lock = null;
      delete next.duration_normalizer;
      delete next.duration_grid;
      delete next.duration_from;
    }
    onChange(next);
  };
  const normalizer: SpecDict | null = value?.duration_normalizer ?? null;
  return (
    <>
      <div className="sub-label">clearance</div>
      <label className="radio modal-check" title="the policy decides when to give it; otherwise the aircraft flies on">
        <input type="checkbox" checked={value != null} onChange={(e) => onChange(e.target.checked ? { duration: null, lock: null } : null)} />
        a clearance: given only when the policy decides
      </label>
      {value != null && (
        <>
          <label className="radio modal-check" title="after it, own navigation takes the axis back">
            <input
              type="checkbox"
              checked={duration != null}
              onChange={(e) => set({ duration: e.target.checked ? [0, 600] : null })}
            />
            temporary: lasts a duration the policy chooses
          </label>
          {duration && (
            <>
              <label className="numfield inline">
                <span>shortest s</span>
                <input type="number" min={0} step={10} value={duration[0]} onChange={(e) => set({ duration: [Math.max(0, Number(e.target.value)), duration[1]] })} />
              </label>
              <label className="numfield inline">
                <span>longest s</span>
                <input type="number" min={0} step={10} value={duration[1]} onChange={(e) => set({ duration: [duration[0], Math.max(duration[0], Number(e.target.value))] })} />
              </label>
              <label className="numfield inline">
                <span>duration normalizer</span>
                <Picker
                  searchable={false}
                  placeholder="Raw"
                  value={normalizer?.name ?? ""}
                  onChange={(v) => set({ duration_normalizer: v ? { type: "normalizer", name: v, kwargs: {} } : null })}
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
                  onChange={(v) => set({ duration_normalizer: v })}
                />
              )}
              <GridControls
                label="duration grid"
                startStep={30}
                target={false}
                value={value.duration_grid ?? null}
                onChange={(grid) => {
                  const moved = regrid(normalizer, value.duration_grid ?? null, grid);
                  set({ duration_normalizer: moved.normalizer, duration_grid: moved.grid });
                }}
              />
              <label className="numfield inline">
                <span>counted from</span>
                <Picker
                  searchable={false}
                  placeholder="given"
                  value={value.duration_from ?? "issued"}
                  onChange={(v) => set({ duration_from: v === "issued" ? null : v })}
                  options={[
                    { value: "issued", label: "given", description: "the duration starts when the clearance is given" },
                    { value: "captured", label: "flown", description: "the duration is a hold that starts once the aircraft has flown the command" },
                  ]}
                />
              </label>
            </>
          )}
          <label className="numfield inline">
            <span>lock</span>
            <Picker
              searchable={false}
              placeholder="none"
              value={value.lock ?? ""}
              onChange={(v) => set({ lock: v || null })}
              options={[
                { value: "", label: "none", description: "another clearance may follow at any step" },
                { value: "captured", label: "until captured", description: "nothing on the axis until the command is flown" },
                ...(duration
                  ? [{ value: "duration", label: "for the duration", description: "nothing on the axis - resume included - until it runs out" }]
                  : []),
              ]}
            />
          </label>
          <div className="muted small">
            To let the policy see it, add ActionLocked and ClearanceTimeLeftS to
            the observation, with this action as their target.
          </div>
        </>
      )}
    </>
  );
}
