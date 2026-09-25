import { useEffect, useMemo, useRef, useState } from "react";
import Editor from "@monaco-editor/react";
import { api, type SpecDict, type ValidateResult } from "../api";
import type { Intel } from "../code/intel";
import { AvailablePanel } from "../code/AvailablePanel";
import { scopeKey } from "../code/intel";
import { registerPythonIntel, useIntel } from "../code/pythonEditor";
import { Picker } from "./panel/Picker";
import { useRefresh } from "../refresh";
import { Resizer, useStoredSize } from "./Resizer";

const TASK_INFO_BODY_TEMPLATE = `task = info["task"]
task["metric"] = 0.0
`;

type TaskInfoEntry = {
  name: string;
  body: string;
};

type TaskInfoType = {
  name: string;
  doc?: string;
  category?: string;
  params?: { name: string; type?: string; required?: boolean; default?: any }[];
  scaffold?: {
    name: string;
    provider_var: string;
    setup: string;
    body: string;
  };
};

function pythonIdentifier(name: string): string {
  const cleaned = name.replace(/[^A-Za-z0-9_]/g, "_");
  return /^[A-Za-z_]/.test(cleaned) ? cleaned : `_${cleaned}`;
}

function replaceEvery(source: string, search: string, replacement: string): string {
  return source.split(search).join(replacement);
}

function appendSetupBlock(existing: string, block: string): string {
  const seenImports = new Set(
    existing
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.startsWith("import ") || line.startsWith("from ")),
  );
  const lines = block
    .trim()
    .split("\n")
    .filter((line) => {
      const trimmed = line.trim();
      if (!(trimmed.startsWith("import ") || trimmed.startsWith("from "))) return true;
      if (seenImports.has(trimmed)) return false;
      seenImports.add(trimmed);
      return true;
    });
  const next = lines.join("\n").trim();
  if (!next) return existing;
  return `${existing.trimEnd()}${existing.trim() ? "\n\n" : ""}${next}\n`;
}

function specializeTaskInfoScaffold(type: TaskInfoType, providerName: string): { setup: string; body: string } {
  const scaffold = type.scaffold;
  if (!scaffold) return { setup: "", body: TASK_INFO_BODY_TEMPLATE };
  const sourceName = scaffold.name;
  const targetName = pythonIdentifier(providerName);
  const targetVar = `${targetName.toUpperCase()}_TASK_INFO_PROVIDER`;
  return {
    setup: replaceEvery(
      replaceEvery(scaffold.setup, scaffold.provider_var, targetVar),
      `${sourceName}_`,
      `${targetName}_`,
    ),
    body: replaceEvery(scaffold.body, scaffold.provider_var, targetVar),
  };
}

// The setup value a task-info entry names, when its body is only that name -
// then the entry is that provider object, as the builder reads it.
function taskInfoProvider(entry: TaskInfoEntry | null, intel: Intel | null) {
  const body = entry?.body.trim() ?? "";
  return intel?.names.setup?.find((member) => member.name === body);
}

// The provider type of such an entry: the class of the object it names.
function taskInfoTypeForEntry(
  entry: TaskInfoEntry | null,
  types: TaskInfoType[],
  intel: Intel | null,
): TaskInfoType | undefined {
  const provider = taskInfoProvider(entry, intel);
  return provider ? types.find((type) => type.name === provider.detail) : undefined;
}

// VS Code-like view of the task's *code structure* (not raw JSON). Editable
// helper modules such as custom_fields.py live in spec.code and are edited here
// as Python; the rest of the package (scenario/env/__main__) is generated
// read-only so you can see the full structure. A "spec.json" entry keeps the
// raw document available for power edits.
const SPEC_FILE = "spec.json";

function langOf(path: string): string {
  if (path.endsWith(".py")) return "python";
  if (path.endsWith(".json")) return "json";
  if (path.endsWith(".md")) return "markdown";
  return "plaintext";
}

export default function CodeTab({
  spec,
  specText,
  onSpecChange,
  onSpecTextChange,
  validation,
  onShowSpaces,
}: {
  spec: SpecDict | null;
  specText: string;
  onSpecChange: (next: SpecDict) => void;
  onSpecTextChange: (text: string) => void;
  validation: ValidateResult | null;
  onShowSpaces: () => void;
}) {
  const [generated, setGenerated] = useState<Record<string, string>>({});
  const codeTabRef = useRef<HTMLDivElement | null>(null);
  const [treeW, setTreeW, resetTreeW] = useStoredSize("designer.code.treeWidth", 220, 160, () => window.innerWidth * 0.3);
  const [sideW, setSideW, resetSideW] = useStoredSize("designer.code.sideWidth", 300, 220, () => window.innerWidth * 0.45);
  const [pkg, setPkg] = useState<string>("");
  const [selected, setSelected] = useState<string>("design.py");
  const [genError, setGenError] = useState<string | null>(null);
  // The per-agent hook a batched one just replaced, and its code, until it is
  // restored or dismissed.
  const [replacedHook, setReplacedHook] = useState<
    { batch: string; perAgent: string; body: string } | null
  >(null);
  const [hookCatalog, setHookCatalog] = useState<any[]>([]);
  // Scenario hooks come from the backend (catalog.scenario_hooks, derived from
  // spec.SCENARIO_HOOKS) rather than a list duplicated here, so adding a hook
  // server-side surfaces it in the editor with no frontend change.
  const [scenarioHookCatalog, setScenarioHookCatalog] = useState<any[]>([]);
  const [taskInfoTypes, setTaskInfoTypes] = useState<TaskInfoType[]>([]);
  // The header a new custom code module starts with (catalog.scaffolds).
  const [moduleHeader, setModuleHeader] = useState<string>("");

  const refreshKey = useRefresh();
  useEffect(() => {
    api
      .catalogOnce()
      .then((c) => {
        setHookCatalog(c?.hooks ?? []);
        setScenarioHookCatalog(c?.scenario_hooks ?? []);
        setTaskInfoTypes(c?.task_info_types ?? []);
        setModuleHeader(c?.scaffolds?.module_header ?? "");
      })
      .catch(() => {
        setHookCatalog([]);
        setScenarioHookCatalog([]);
        setTaskInfoTypes([]);
      });
  }, [refreshKey]);

  const intel = useIntel();
  // The Monaco editor, for inserting what the "Available here" panel offers.
  const editorRef = useRef<any>(null);
  const insertAtCursor = (text: string) => {
    const editor = editorRef.current;
    // Generated files are read-only: nothing to insert into.
    if (!editor || editorReadOnly) return;
    editor.executeEdits("available", [{ range: editor.getSelection(), text, forceMoveMarkers: true }]);
    editor.focus();
  };

  const codeFiles = useMemo(() => Object.keys(spec?.code ?? {}), [spec]);
  const hooks: Record<string, string> = spec?.env?.hooks ?? {};
  const hookSetup: string = spec?.env?.hook_setup ?? "";
  const taskInfoSetup: string = spec?.env?.task_info_setup ?? "";
  const taskInfo: TaskInfoEntry[] = spec?.env?.task_info ?? [];
  const externalTaskInfoProviders: string[] = spec?.env?.task_info_providers ?? [];
  // reward/terminated/truncated always exist (default hooks); shown first and
  // not deletable. Other customized hooks follow and can be removed.
  const DEFAULT_HOOKS = ["reward", "terminated", "truncated"];
  const DEFAULT_BODIES: Record<string, string> = {
    reward: "return 0.0",
    terminated: "return False",
    truncated: "return False",
  };
  // A batched hook (reward_batch) replaces its per-agent one, which is then
  // hidden - unless it came back with code of its own, so the clash shows.
  const replaced = (name: string) => `${name}_batch` in hooks && !(name in hooks);
  const shownHooks = [
    ...DEFAULT_HOOKS.filter((n) => !replaced(n)),
    ...Object.keys(hooks).filter((n) => !DEFAULT_HOOKS.includes(n)),
  ];
  const selectedHookSetup = selected === "hooksetup";
  const selectedHook = selected.startsWith("hook:") ? selected.slice(5) : null;
  const selectedTaskInfoSetup = selected === "taskinfo:setup";
  const selectedTaskInfoText = selected.startsWith("taskinfo:") ? selected.slice(9) : null;
  const selectedTaskInfo = selectedTaskInfoText && /^\d+$/.test(selectedTaskInfoText)
    ? Number(selectedTaskInfoText)
    : null;
  const selectedTaskInfoEntry =
    selectedTaskInfo != null && Number.isInteger(selectedTaskInfo)
      ? taskInfo[selectedTaskInfo]
      : null;
  const selectedTaskInfoProviderReference = Boolean(taskInfoProvider(selectedTaskInfoEntry, intel));
  const selectedTaskInfoType = taskInfoTypeForEntry(selectedTaskInfoEntry, taskInfoTypes, intel);
  const hookMeta = (name: string) => hookCatalog.find((h) => h.name === name);
  const hookBody = (name: string) => hooks[name] ?? DEFAULT_BODIES[name] ?? "";

  const scaffoldHook = (meta: any): string => {
    // Prefer a curated, hook-specific scaffold when the backend provides one.
    if (meta?.scaffold) return `${meta.scaffold}\n`;
    const doc = meta?.doc ? `# ${meta.doc}\n` : "";
    const body = meta?.returns_none
      ? "return None"
      : `return ${meta?.default ?? "None"}  # TODO: return ${meta?.returns || "the value this hook expects"}`;
    return `${doc}${body}\n`;
  };
  const setHook = (name: string, body: string) => {
    if (!spec) return;
    onSpecChange({ ...spec, env: { ...spec.env, hooks: { ...hooks, [name]: body } } });
  };
  // A hook body without its comments or blank lines: what it actually does.
  const hookCode = (body: string) =>
    body
      .split("\n")
      .filter((line) => line.trim() && !line.trim().startsWith("#"))
      .join("\n")
      .trim();

  const removeHook = (name: string) => {
    if (!spec || DEFAULT_HOOKS.includes(name)) return;
    if (replacedHook?.batch === name) setReplacedHook(null);
    const next = { ...hooks };
    delete next[name];
    onSpecChange({ ...spec, env: { ...spec.env, hooks: next } });
    if (selected === `hook:${name}`) setSelected(`hook:reward`);
  };
  const addHook = (name: string) => {
    if (!spec || !name || hooks[name]) return;
    const next = { ...hooks, [name]: scaffoldHook(hookMeta(name)) };
    // A batched hook replaces its per-agent one: a design defines one of them.
    const perAgent = name.endsWith("_batch") ? name.slice(0, -"_batch".length) : "";
    if (DEFAULT_HOOKS.includes(perAgent)) {
      const body = hooks[perAgent] ?? "";
      const code = hookCode(body);
      const custom = Boolean(code) && code !== DEFAULT_BODIES[perAgent];
      if (custom) {
        const ok = window.confirm(
          `${name} replaces ${perAgent}: a design defines one of the two, so ` +
            `${perAgent} and its code will be removed. You can restore it from ` +
            `the notice above the editor until you dismiss it. Continue?`,
        );
        if (!ok) return;
      }
      delete next[perAgent];
      setReplacedHook(custom ? { batch: name, perAgent, body } : null);
    }
    onSpecChange({ ...spec, env: { ...spec.env, hooks: next } });
    setSelected(`hook:${name}`);
  };

  // Put the replaced per-agent hook back, with its code. The two cannot both
  // be defined, so this removes the batched hook - and whatever it holds.
  const restoreReplacedHook = () => {
    if (!spec || !replacedHook) return;
    const { batch, perAgent, body } = replacedHook;
    const next = { ...hooks, [perAgent]: body };
    delete next[batch];
    onSpecChange({ ...spec, env: { ...spec.env, hooks: next } });
    setReplacedHook(null);
    setSelected(`hook:${perAgent}`);
  };

  const setHookSetup = (body: string) => {
    if (!spec) return;
    onSpecChange({ ...spec, env: { ...spec.env, hook_setup: body } });
  };

  // Scenario code lives at the top level of the spec, not under env: it is
  // emitted into scenario.py and also compiled by the builder for live preview.
  const scenarioSetup: string = spec?.scenario_setup ?? "";
  const scenarioHooks: Record<string, string> = spec?.scenario_hooks ?? {};
  const selectedScenarioSetup = selected === "scenariosetup";
  const selectedScenarioHook = selected.startsWith("scenariohook:")
    ? selected.slice("scenariohook:".length)
    : null;
  const scenarioHookMeta = (name: string) => scenarioHookCatalog.find((h) => h.name === name);
  const setScenarioSetup = (body: string) => {
    if (!spec) return;
    onSpecChange({ ...spec, scenario_setup: body });
  };
  const setScenarioHook = (name: string, body: string) => {
    if (!spec) return;
    onSpecChange({ ...spec, scenario_hooks: { ...scenarioHooks, [name]: body } });
  };
  const removeScenarioHook = (name: string) => {
    if (!spec) return;
    const next = { ...scenarioHooks };
    delete next[name];
    onSpecChange({ ...spec, scenario_hooks: next });
    if (selected === `scenariohook:${name}`) setSelected(SPEC_FILE);
  };
  const addScenarioHook = (name: string) => {
    if (!name || scenarioHooks[name]) return;
    const meta = scenarioHookMeta(name);
    setScenarioHook(name, meta?.scaffold ?? "return geometry\n");
    setSelected(`scenariohook:${name}`);
  };

  const setTaskInfoSetup = (body: string) => {
    if (!spec) return;
    onSpecChange({ ...spec, env: { ...spec.env, task_info_setup: body } });
  };

  const setTaskInfoBody = (index: number, body: string) => {
    if (!spec) return;
    const next = [...taskInfo];
    if (!next[index]) return;
    next[index] = { ...next[index], body };
    onSpecChange({ ...spec, env: { ...spec.env, task_info: next } });
  };

  const removeTaskInfoProvider = (index: number) => {
    if (!spec) return;
    const next = taskInfo.filter((_, i) => i !== index);
    onSpecChange({ ...spec, env: { ...spec.env, task_info: next } });
    if (selected === `taskinfo:${index}`) setSelected(SPEC_FILE);
  };

  const uniqueTaskInfoName = (base: string): string => {
    const used = new Set(taskInfo.map((entry) => entry.name));
    if (!used.has(base)) return base;
    let i = 2;
    while (used.has(`${base}_${i}`)) i += 1;
    return `${base}_${i}`;
  };

  const addTaskInfoProvider = (typeName: string) => {
    if (!spec) return;
    const taskInfoType = taskInfoTypes.find((type) => type.name === typeName);
    const baseName = taskInfoType?.scaffold?.name ?? "task_info";
    const providerName = uniqueTaskInfoName(baseName);
    const scaffold = taskInfoType ? specializeTaskInfoScaffold(taskInfoType, providerName) : { setup: "", body: TASK_INFO_BODY_TEMPLATE };
    const nextProviders = [
      ...taskInfo,
      {
        name: providerName,
        body: scaffold.body,
      },
    ];
    onSpecChange({
      ...spec,
      env: {
        ...spec.env,
        task_info_setup: appendSetupBlock(taskInfoSetup, scaffold.setup),
        task_info: nextProviders,
      },
    });
    setSelected(taskInfoType ? "taskinfo:setup" : `taskinfo:${nextProviders.length - 1}`);
  };

  // Regenerate the structural files when the (valid) spec settles.
  useEffect(() => {
    if (!spec) return;
    const name = spec.metadata?.name || "designed_task";
    const handle = setTimeout(() => {
      api
        .generate(spec, name)
        .then((r) => {
          setPkg(r.package);
          setGenerated(r.files);
          setGenError(null);
        })
        .catch((e) => setGenError(String(e)));
    }, 500);
    return () => clearTimeout(handle);
  }, [spec, refreshKey]);

  // Generated files that are NOT editable design code (those come from spec.code).
  const structuralFiles = useMemo(() => {
    const prefix = pkg ? `${pkg}/` : "";
    return Object.keys(generated)
      .filter((p) => {
        const base = p.startsWith(prefix) ? p.slice(prefix.length) : p;
        return !codeFiles.includes(base);
      })
      .sort();
  }, [generated, codeFiles, pkg]);

  const addFile = () => {
    const name = window.prompt("new module filename", "custom_fields.py");
    if (!name || !name.endsWith(".py") || !spec) return;
    const initial = name === "custom_fields.py" && moduleHeader ? moduleHeader : `# ${name}\n`;
    onSpecChange({ ...spec, code: { ...spec.code, [name]: initial } });
    setSelected(name);
  };

  const updateCode = (file: string, value: string) => {
    if (!spec) return;
    onSpecChange({ ...spec, code: { ...spec.code, [file]: value } });
  };

  const deleteCode = (file: string) => {
    if (!spec) return;
    const next = { ...spec.code };
    delete next[file];
    onSpecChange({ ...spec, code: next });
    if (selected === file) setSelected(SPEC_FILE);
  };

  const isSpec = selected === SPEC_FILE;
  const isCode = codeFiles.includes(selected);
  const structuralKey = pkg && structuralFiles.includes(`${pkg}/${selected}`) ? `${pkg}/${selected}` : selected;
  const value = selectedHook
    ? hookBody(selectedHook)
    : selectedHookSetup
      ? hookSetup
    : selectedTaskInfoSetup
      ? taskInfoSetup
    : selectedTaskInfoEntry != null
      ? selectedTaskInfoEntry.body
    : selectedScenarioSetup
      ? scenarioSetup
    : selectedScenarioHook
      ? (scenarioHooks[selectedScenarioHook] ?? "")
    : isSpec
      ? specText
      : isCode
        ? (spec?.code?.[selected] ?? "")
        : (generated[structuralKey] ?? generated[selected] ?? "");
  const editorPath = selectedHook
    ? `hook_${selectedHook}.py`
    : selectedHookSetup
      ? "hook_setup.py"
      : selectedTaskInfoEntry
        ? `task_info_${selectedTaskInfoEntry.name}.py`
      : selectedTaskInfoSetup
        ? "task_info_setup.py"
      : selectedScenarioSetup
        ? "scenario_setup.py"
      : selectedScenarioHook
        ? `scenario_${selectedScenarioHook}.py`
      : selected;
  const editorLanguage =
    selectedHook
    || selectedHookSetup
    || selectedTaskInfoEntry
    || selectedTaskInfoSetup
    || selectedScenarioSetup
    || selectedScenarioHook
      ? "python"
      : langOf(selected);
  const editorReadOnly =
    !isSpec
    && !isCode
    && !selectedHook
    && !selectedHookSetup
    && !selectedTaskInfoSetup
    && !selectedScenarioSetup
    && !selectedScenarioHook
    && (selectedTaskInfoEntry == null || selectedTaskInfoProviderReference);

  return (
    <div className="code-tab" ref={codeTabRef}>
      <aside className="file-tree" style={{ width: treeW }}>
        <div className="tree-group">design</div>
        <FileItem path={SPEC_FILE} active={isSpec} onClick={() => setSelected(SPEC_FILE)} />

        <div className="tree-group">
          code <button className="link" onClick={addFile}>+ file</button>
        </div>
        {codeFiles.map((f) => (
          <FileItem
            key={f}
            path={f}
            active={selected === f}
            editable
            onClick={() => setSelected(f)}
            onDelete={() => deleteCode(f)}
          />
        ))}

        <div className="tree-group">task info</div>
        <FileItem
          path="setup"
          active={selectedTaskInfoSetup}
          editable
          onClick={() => setSelected("taskinfo:setup")}
        />
        <Picker
          className="hook-add"
          placeholder="+ task info…"
          title="add task-info provider"
          onChange={addTaskInfoProvider}
          options={[
            { value: "custom", label: "custom", description: "Free-form task diagnostics provider.", category: "custom" },
            ...taskInfoTypes.map((type) => ({
              value: type.name,
              label: type.name,
              description: type.doc,
              category: type.category ?? "task info",
            })),
          ]}
        />
        {taskInfo.map((provider, i) => (
          <FileItem
            key={`${provider.name}-${i}`}
            path={provider.name}
            active={selected === `taskinfo:${i}`}
            editable
            onClick={() => setSelected(`taskinfo:${i}`)}
            onDelete={() => removeTaskInfoProvider(i)}
          />
        ))}
        {taskInfo.length === 0 && <div className="muted small file-empty">none</div>}
        {externalTaskInfoProviders.length > 0 && (
          <>
            <div className="tree-group">task info imports</div>
            {externalTaskInfoProviders.map((ref, i) => (
              <FileItem
                key={`${ref}-${i}`}
                path={ref}
                active={false}
                onClick={() => setSelected(SPEC_FILE)}
              />
            ))}
          </>
        )}

        <div className="tree-group">env hooks</div>
        <FileItem
          path="setup"
          active={selectedHookSetup}
          editable
          onClick={() => setSelected("hooksetup")}
        />
        {shownHooks.map((h) => (
          <FileItem
            key={h}
            path={h}
            active={selected === `hook:${h}`}
            editable
            onClick={() => setSelected(`hook:${h}`)}
            onDelete={DEFAULT_HOOKS.includes(h) ? undefined : () => removeHook(h)}
          />
        ))}
        {hookCatalog.length > 0 && (
          <Picker
            className="hook-add"
            placeholder="+ override hook…"
            title="override an environment hook"
            onChange={addHook}
            options={hookCatalog
              .filter((h) => !DEFAULT_HOOKS.includes(h.name) && !hooks[h.name])
              .map((h) => ({
                value: h.name,
                label: h.name,
                description: h.doc,
                category: h.category ?? "other",
              }))}
          />
        )}

        <div className="tree-group">scenario code</div>
        <FileItem
          path="setup"
          active={selectedScenarioSetup}
          editable
          onClick={() => setSelected("scenariosetup")}
        />
        {Object.keys(scenarioHooks).map((h) => (
          <FileItem
            key={h}
            path={h}
            active={selected === `scenariohook:${h}`}
            editable
            onClick={() => setSelected(`scenariohook:${h}`)}
            onDelete={() => removeScenarioHook(h)}
          />
        ))}
        {scenarioHookCatalog.length > 0 && (
          <Picker
            className="hook-add"
            placeholder="+ scenario hook…"
            title="sample geometry per episode from the design"
            onChange={addScenarioHook}
            options={scenarioHookCatalog
              .filter((h) => !scenarioHooks[h.name])
              .map((h) => ({
                value: h.name,
                label: h.name,
                description: h.doc,
                category: "scenario",
              }))}
          />
        )}

        <div className="tree-group">generated{pkg ? ` · ${pkg}/` : ""}</div>
        {structuralFiles.map((p) => {
          const base = pkg && p.startsWith(`${pkg}/`) ? p.slice(pkg.length + 1) : p;
          return <FileItem key={p} path={base} active={selected === base} onClick={() => setSelected(base)} />;
        })}
        {genError && <div className="error-text small">{genError}</div>}
      </aside>

      <Resizer
        axis="x"
        label="Resize the file list"
        onDrag={(x) => setTreeW(x - (codeTabRef.current?.getBoundingClientRect().left ?? 0))}
        onReset={resetTreeW}
      />
      <div className="editor-pane">
        {replacedHook && replacedHook.batch in hooks && (
          <div className="hook-notice small">
            <span>
              <code>{replacedHook.perAgent}</code> was replaced by{" "}
              <code>{replacedHook.batch}</code>, and its code removed. Restoring it
              removes <code>{replacedHook.batch}</code>.
            </span>
            <button
              onClick={restoreReplacedHook}
              title={`Bring back ${replacedHook.perAgent} with its code; ${replacedHook.batch} is removed.`}
            >
              Restore {replacedHook.perAgent}
            </button>
            <button className="link" onClick={() => setReplacedHook(null)}>
              dismiss
            </button>
          </div>
        )}
        {selectedHook && (
          <div className="hook-sig small">
            <code className="hook-sig-def">
              def <span className="hook-sig-name">{selectedHook}</span>
              {hookMeta(selectedHook)?.def_signature ?? "(self, …)"}:
            </code>
            {hookMeta(selectedHook)?.doc ? (
              <span className="hook-sig-doc"> — {hookMeta(selectedHook).doc}</span>
            ) : null}
          </div>
        )}
        {selectedHookSetup && (
          <div className="hook-sig small">
            <code className="hook-sig-def"># env-hook imports, constants, and helpers</code>
          </div>
        )}
        {selectedTaskInfoEntry && (
          <div className="hook-sig small">
            {selectedTaskInfoProviderReference ? (
              <code className="hook-sig-def">
                provider <span className="hook-sig-name">{selectedTaskInfoEntry.body.trim()}</span>
              </code>
            ) : (
              <code className="hook-sig-def">
                def <span className="hook-sig-name">{selectedTaskInfoEntry.name}</span>
                (obs, action, info, context, rng) -&gt; None:
              </code>
            )}
          </div>
        )}
        {selectedTaskInfoSetup && (
          <div className="hook-sig small">
            <code className="hook-sig-def"># task-info imports, constants, and helpers</code>
          </div>
        )}
        {selectedScenarioSetup && (
          <div className="hook-sig small">
            <code className="hook-sig-def">
              # scenario.py module scope: imports, constants and helpers for the scenario hooks
            </code>
          </div>
        )}
        {selectedScenarioHook && (
          <div className="hook-sig small">
            <code className="hook-sig-def">
              def _{selectedScenarioHook}
              {scenarioHookMeta(selectedScenarioHook)?.signature ?? "(geometry, rng)"}:
            </code>
            {scenarioHookMeta(selectedScenarioHook)?.doc && (
              <span className="hook-doc"> {scenarioHookMeta(selectedScenarioHook)?.doc}</span>
            )}
          </div>
        )}
        <Editor
          key={editorPath}
          height="100%"
          path={editorPath}
          language={editorLanguage}
          value={value}
          beforeMount={registerPythonIntel}
          onMount={(editor) => (editorRef.current = editor)}
          onChange={(v) => {
            if (selectedHook) setHook(selectedHook, v ?? "");
            else if (selectedHookSetup) setHookSetup(v ?? "");
            else if (selectedScenarioSetup) setScenarioSetup(v ?? "");
            else if (selectedScenarioHook) setScenarioHook(selectedScenarioHook, v ?? "");
            else if (selectedTaskInfoSetup) setTaskInfoSetup(v ?? "");
            else if (selectedTaskInfoEntry != null && selectedTaskInfo != null) setTaskInfoBody(selectedTaskInfo, v ?? "");
            else if (isSpec) onSpecTextChange(v ?? "");
            else if (isCode) updateCode(selected, v ?? "");
          }}
          theme="vs-dark"
          options={{
            minimap: { enabled: false },
            fontSize: 13,
            tabSize: isSpec ? 2 : 4,
            readOnly: editorReadOnly,
            scrollBeyondLastLine: false,
            automaticLayout: true,
            "semanticHighlighting.enabled": true,
          }}
        />
      </div>

      <Resizer axis="x" label="Resize the side panel" onDrag={(x) => setSideW(window.innerWidth - x)} onReset={resetSideW} />
      <aside className="inspector" style={{ width: sideW }}>
        <AvailablePanel intel={intel} scope={scopeKey(editorPath)} onInsert={insertAtCursor} />
        <h3>Validation</h3>
        {!validation && <p className="muted">…</p>}
        {validation && !validation.ok && <pre className="error-text">{validation.error}</pre>}
        {validation?.ok && validation.summary && (
          <dl className="summary">
            <dt>max aircraft</dt>
            <dd>{validation.summary.max_aircraft}</dd>
            <dt>fields</dt>
            <dd>
              {validation.summary.obs_fields.length} ownship
              {validation.summary.intruder_obs_fields ? ` · ${validation.summary.intruder_obs_fields.length} intruder` : ""}
              {` · ${validation.summary.action_fields.length} action`}{" "}
              <button className="link" onClick={onShowSpaces}>
                see spaces
              </button>
            </dd>
            <dt>aircraft</dt>
            <dd title={validation.summary.allowed_aircraft.join(", ")}>
              {validation.summary.allowed_aircraft.length} types
            </dd>
            <dt>queryables</dt>
            <dd>{validation.summary.queryables.join(", ") || "none"}</dd>
          </dl>
        )}
        {selectedHook && (() => {
          const m = hookMeta(selectedHook);
          return (
            <div className="hook-hint">
              <h3>Hook</h3>
              {m?.doc && <p className="small">{m.doc}</p>}
              <dl className="summary">
                {m?.category && (<><dt>category</dt><dd>{m.category}</dd></>)}
                {m?.params?.length ? (<><dt>params</dt><dd>{m.params.join(", ")}</dd></>) : null}
                {m?.returns && (<><dt>returns</dt><dd><code>{m.returns}</code></dd></>)}
                {m?.default != null && (<><dt>default</dt><dd><code>return {m.default}</code></dd></>)}
              </dl>
              {m?.params?.includes("context") && (
                <p className="muted small">
                  Tip: <code>context.query("name")</code> reads a queryable for this aircraft.
                </p>
              )}
            </div>
          );
        })()}
        {selectedHookSetup && (
          <div className="hook-hint">
            <h3>Hook Setup</h3>
            <p className="small">
              Module-level Python emitted before the env class. Put imports, constants, or helper functions used by hook methods here.
            </p>
          </div>
        )}
        {selectedTaskInfoEntry != null && (
          <div className="hook-hint">
            <h3>Task Info</h3>
            {selectedTaskInfoProviderReference ? (
              <>
                <p className="small">
                  Constructor-backed provider object. Edit its constructor arguments and callback functions in <code>task info/setup</code>.
                </p>
                {selectedTaskInfoType?.params?.length ? (
                  <dl className="summary">
                    <dt>type</dt>
                    <dd>{selectedTaskInfoType.name}</dd>
                    <dt>constructor</dt>
                    <dd>{selectedTaskInfoType.params.map((param: any) => param.name).join(", ")}</dd>
                  </dl>
                ) : null}
              </>
            ) : (
              <>
                <p className="small">
                  Runs once per controlled agent after observations and base info are built. Write public diagnostics under <code>info["task"]</code>.
                </p>
                <p className="muted small">
                  Useful inputs: <code>context.acid</code>, <code>context.query("name")</code>, <code>obs</code>, <code>action</code>, and <code>rng</code>.
                </p>
              </>
            )}
          </div>
        )}
        {selectedTaskInfoSetup && (
          <div className="hook-hint">
            <h3>Task Info Setup</h3>
            <p className="small">
              Module-level Python emitted before task-info providers. Put imports, constants, or helper functions here.
            </p>
          </div>
        )}
        {!isSpec && !isCode && !selectedHook && !selectedHookSetup && selectedTaskInfoEntry == null && !selectedTaskInfoSetup && <p className="muted small">generated · read-only</p>}
      </aside>
    </div>
  );
}

function FileItem({
  path,
  active,
  editable,
  onClick,
  onDelete,
}: {
  path: string;
  active: boolean;
  editable?: boolean;
  onClick: () => void;
  onDelete?: () => void;
}) {
  return (
    <div className={active ? "file-item active" : "file-item"} onClick={onClick}>
      <span className="file-name">
        {editable ? "✎ " : ""}
        {path}
      </span>
      {onDelete && (
        <button
          className="chip-x"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
        >
          ✕
        </button>
      )}
    </div>
  );
}
