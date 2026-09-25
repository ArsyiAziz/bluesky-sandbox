import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type SpecDict, type ValidateResult } from "./api";
import { forgetModuleMembers, useCodeIntel } from "./code/pythonEditor";
import { DEFAULT_SPEC } from "./defaultSpec";
import { migrateRewardHooks, migrateRotationGroups, normalizeToRegions } from "./specHelpers";
import MapTab from "./components/MapTab";
import CodeTab from "./components/CodeTab";
import RouteTab from "./route/RouteTab";
import GenerateModal from "./components/GenerateModal";
import MdpTab from "./components/MdpTab";
import { RefreshContext } from "./refresh";
import MetadataTab from "./components/MetadataTab";
import RunModal from "./components/RunModal";
import { Picker } from "./components/panel/Picker";

type Tab = "map" | "route" | "code" | "metadata" | "mdp";

const normalizeSpec = (spec: SpecDict): SpecDict =>
  migrateRotationGroups(migrateRewardHooks(normalizeToRegions(spec)));

const IMPORT_JSON_VALUE = "__import_json__";
const NEW_PROJECT_VALUE = "__new_project__";
const DELETE_PROJECT_VALUE = "__delete_project__";

export default function App() {
  const [tab, setTab] = useState<Tab>("map");
  const [specText, setSpecText] = useState<string>(() => JSON.stringify(normalizeSpec(DEFAULT_SPEC), null, 2));
  const [validation, setValidation] = useState<ValidateResult | null>(null);
  const [saveName, setSaveName] = useState("untitled");
  const [currentSavedName, setCurrentSavedName] = useState<string | null>(null);
  const [savedSpecs, setSavedSpecs] = useState<
    { name: string; title: string; base?: string; version?: string }[]
  >([]);
  const [status, setStatus] = useState<string>("");
  const [generateOpen, setGenerateOpen] = useState(false);
  const [runOpen, setRunOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const importInputRef = useRef<HTMLInputElement | null>(null);

  // Parse the editor text into a spec object; null while the JSON is invalid.
  const { spec, parseError } = useMemo<{ spec: SpecDict | null; parseError: string | null }>(() => {
    try {
      return { spec: JSON.parse(specText), parseError: null };
    } catch (e) {
      return { spec: null, parseError: (e as Error).message };
    }
  }, [specText]);
  // What the design's code can use, for every code editor. App is outside the
  // refresh context it provides, so it passes the count itself.
  useCodeIntel(spec, refreshKey);


  // The spec object is the source of truth; structured edits (the properties
  // panel) re-serialize it back into the editor text so both views stay in sync.
  const updateSpec = useCallback((next: SpecDict) => {
    setSpecText(JSON.stringify(next, null, 2));
  }, []);

  // ---- Undo / redo over the spec text -----------------------------------
  // History of settled spec snapshots. Edits are recorded debounced (so typing
  // and slider drags coalesce into one step); ⌘/Ctrl+Z / ⇧+Z time-travel them.
  // The Monaco code editor keeps its own text undo while it has focus.
  const specTextRef = useRef(specText);
  specTextRef.current = specText;
  const historyRef = useRef<string[]>([specText]);
  const histIndexRef = useRef(0);
  const timeTravelRef = useRef(false);
  const recordTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [, setHistVersion] = useState(0);

  const recordNow = useCallback(() => {
    if (recordTimerRef.current) {
      clearTimeout(recordTimerRef.current);
      recordTimerRef.current = null;
    }
    const idx = histIndexRef.current;
    if (historyRef.current[idx] === specTextRef.current) return;
    const next = historyRef.current.slice(0, idx + 1);
    next.push(specTextRef.current);
    if (next.length > 100) next.shift();
    historyRef.current = next;
    histIndexRef.current = next.length - 1;
    setHistVersion((v) => v + 1);
  }, []);

  // Snapshot settled states (skip the change that came from time-travel itself).
  useEffect(() => {
    if (timeTravelRef.current) {
      timeTravelRef.current = false;
      return;
    }
    if (recordTimerRef.current) clearTimeout(recordTimerRef.current);
    recordTimerRef.current = setTimeout(recordNow, 350);
  }, [specText, recordNow]);

  const goTo = useCallback((idx: number) => {
    if (idx < 0 || idx >= historyRef.current.length) return;
    histIndexRef.current = idx;
    timeTravelRef.current = true;
    setSpecText(historyRef.current[idx]);
    setHistVersion((v) => v + 1);
  }, []);

  const resetHistory = useCallback((text: string) => {
    if (recordTimerRef.current) clearTimeout(recordTimerRef.current);
    historyRef.current = [text];
    histIndexRef.current = 0;
    setHistVersion((v) => v + 1);
  }, []);

  // Undo records any just-typed (still-pending) edit first, so a quick
  // type-then-undo reverts that edit rather than skipping it.
  const undo = useCallback(() => {
    recordNow();
    goTo(histIndexRef.current - 1);
  }, [recordNow, goTo]);
  const redo = useCallback(() => goTo(histIndexRef.current + 1), [goTo]);

  const canUndo =
    histIndexRef.current > 0 || historyRef.current[histIndexRef.current] !== specText;
  const canRedo = histIndexRef.current < historyRef.current.length - 1;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey)) return;
      const k = e.key.toLowerCase();
      if (k !== "z" && k !== "y") return;
      // Let the code editor handle its own text undo when it's focused.
      const ae = document.activeElement as HTMLElement | null;
      if (ae?.closest?.(".monaco-editor")) return;
      e.preventDefault();
      if (k === "y" || (k === "z" && e.shiftKey)) redo();
      else undo();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo]);

  const refreshSaved = useCallback(() => {
    api.listSpecs().then(setSavedSpecs).catch(() => setSavedSpecs([]));
  }, []);

  useEffect(() => {
    refreshSaved();
  }, [refreshSaved, refreshKey]);

  // Read everything from the backend again - the catalog, previews, code intel,
  // validation, saved projects - keeping the design and its undo history.
  const onRefresh = useCallback(() => {
    setRefreshing(true);
    forgetModuleMembers();
    api
      .refresh()
      .catch(() => undefined)
      .finally(() => {
        setRefreshKey((k) => k + 1);
        setRefreshing(false);
        setStatus("refreshed");
      });
  }, []);

  // Validate against the backend (debounced) whenever the parsed spec changes.
  useEffect(() => {
    if (!spec) {
      setValidation({ ok: false, error: parseError ?? "invalid JSON" });
      return;
    }
    const handle = setTimeout(() => {
      api
        .validate(spec)
        .then(setValidation)
        .catch((e) => setValidation({ ok: false, error: String(e) }));
    }, 400);
    return () => clearTimeout(handle);
  }, [spec, parseError, refreshKey]);

  const onSave = useCallback(() => {
    if (!spec) return;
    // The project name is the source of truth for metadata.name (and the
    // generated package name), so persist it into the design on save.
    const named = { ...spec, metadata: { ...(spec.metadata ?? {}), name: saveName } };
    updateSpec(named);
    api
      .saveSpec(saveName, named)
      .then((r) => {
        setCurrentSavedName(r.name);
        setStatus(`saved as ${r.name}`);
        refreshSaved();
      })
      .catch((e) => setStatus(`save failed: ${e}`));
  }, [spec, saveName, refreshSaved, updateSpec]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== "s") return;
      e.preventDefault();
      onSave();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onSave]);

  const onLoad = useCallback((name: string) => {
    if (!name) return;
    if (name === NEW_PROJECT_VALUE) {
      onNewTaskRef.current();
      return;
    }
    if (name === DELETE_PROJECT_VALUE) {
      onDeleteRef.current();
      return;
    }
    if (name === IMPORT_JSON_VALUE) {
      importInputRef.current?.click();
      return;
    }
    api
      .getSpec(name)
      .then((s) => {
        const text = JSON.stringify(normalizeSpec(s), null, 2);
        timeTravelRef.current = true; // loading a project starts a fresh history
        setSpecText(text);
        resetHistory(text);
        setCurrentSavedName(name);
        setSaveName(s?.metadata?.name || name);
        setStatus(`loaded ${name}`);
      })
      .catch((e) => setStatus(`load failed: ${e}`));
  }, [resetHistory]);

  // Start a fresh task from the default design. Same shape as onLoad: replace
  // the text, reset history (the previous project's steps are not undo-able
  // into a different design), and drop the saved-name binding so a later Save
  // writes a new project instead of overwriting the one that was open.
  const onNewTask = useCallback(() => {
    if (currentSavedName || saveName !== "untitled") {
      const ok = window.confirm(
        "Start a new task? Unsaved changes to the current design will be lost."
      );
      if (!ok) return;
    }
    const text = JSON.stringify(normalizeSpec(DEFAULT_SPEC), null, 2);
    timeTravelRef.current = true;
    setSpecText(text);
    resetHistory(text);
    setCurrentSavedName(null);
    setSaveName("untitled");
    setStatus("new task");
  }, [resetHistory, currentSavedName, saveName]);

  const onImportFile = useCallback((file: File | null) => {
    if (!file) return;
    file
      .text()
      .then((text) => {
        const parsed = JSON.parse(text);
        const normalized = normalizeSpec(parsed);
        const nextText = JSON.stringify(normalized, null, 2);
        timeTravelRef.current = true;
        setSpecText(nextText);
        resetHistory(nextText);
        setCurrentSavedName(null);
        const fallbackName = file.name.replace(/\.json$/i, "") || "imported";
        setSaveName(normalized?.metadata?.name || fallbackName);
        setStatus(`imported ${file.name}`);
      })
      .catch((e) => setStatus(`import failed: ${(e as Error).message}`));
  }, [resetHistory]);

  const deleteName = useMemo(() => {
    if (savedSpecs.some((s) => s.name === saveName)) return saveName;
    if (currentSavedName && savedSpecs.some((s) => s.name === currentSavedName)) {
      return currentSavedName;
    }
    return null;
  }, [currentSavedName, saveName, savedSpecs]);

  const loadOptions = useMemo(() => [
    { value: NEW_PROJECT_VALUE, label: "New project", description: "Start from the default design", category: "project" },
    {
      value: DELETE_PROJECT_VALUE,
      label: deleteName ? `Delete ${deleteName}` : "Delete",
      description: deleteName ? "Remove this saved project" : "The open design is not saved",
      category: "project",
      disabled: !deleteName,
    },
    {
      value: IMPORT_JSON_VALUE,
      label: "Import JSON file…",
      description: "Load a generated design.json or saved spec JSON",
      category: "file",
    },
    // Versions of one design are separate saves, so sort them together and
    // newest-first within a design: the list is a history, and the version you
    // just bumped to is the one you are most likely to reload.
    ...[...savedSpecs]
      .sort((a, b) =>
        (a.base ?? a.name).localeCompare(b.base ?? b.name) ||
        (b.version ?? "").localeCompare(a.version ?? "", undefined, { numeric: true }),
      )
      .map((s) => ({
        value: s.name,
        label: s.title,
        badge: s.version ? `v${s.version}` : undefined,
        description: s.name,
        category: "saved",
      })),
  ], [savedSpecs, deleteName]);

  const onDelete = useCallback(() => {
    if (!deleteName) return;
    if (!window.confirm(`Delete saved project "${deleteName}"? This cannot be undone.`)) return;
    api
      .deleteSpec(deleteName)
      .then(() => {
        setStatus(`deleted ${deleteName}`);
        if (currentSavedName === deleteName) setCurrentSavedName(null);
        refreshSaved();
      })
      .catch((e) => setStatus(`delete failed: ${e}`));
  }, [currentSavedName, deleteName, refreshSaved]);

  // onLoad is declared before these; it reaches them through refs.
  const onNewTaskRef = useRef(onNewTask);
  onNewTaskRef.current = onNewTask;
  const onDeleteRef = useRef(onDelete);
  onDeleteRef.current = onDelete;

  return (
    <RefreshContext.Provider value={refreshKey}>
    <div className="app">
      <header className="toolbar">
        <strong className="brand">Environment Designer</strong>
        <div className="tabs">
          <button className={tab === "map" ? "tab active" : "tab"} onClick={() => setTab("map")}>
            Map
          </button>
          <button className={tab === "route" ? "tab active" : "tab"} onClick={() => setTab("route")}>
            Route
          </button>
          <button className={tab === "code" ? "tab active" : "tab"} onClick={() => setTab("code")}>
            Code
          </button>
          <button className={tab === "metadata" ? "tab active" : "tab"} onClick={() => setTab("metadata")}>
            Metadata
          </button>
          <button className={tab === "mdp" ? "tab active" : "tab"} onClick={() => setTab("mdp")}>
            MDP
          </button>
        </div>
        <div className="undo-redo">
          <button onClick={undo} disabled={!canUndo} title="Undo (⌘/Ctrl+Z)" aria-label="Undo">
            <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
              <path
                fill="currentColor"
                d="M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z"
              />
            </svg>
          </button>
          <button onClick={redo} disabled={!canRedo} title="Redo (⌘/Ctrl+⇧+Z)" aria-label="Redo">
            <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
              <path
                fill="currentColor"
                d="M18.4 10.6C16.55 8.99 14.15 8 11.5 8c-4.65 0-8.58 3.03-9.96 7.22L3.9 16c1.05-3.19 4.05-5.5 7.6-5.5 1.95 0 3.73.72 5.12 1.88L13 16h9V7l-3.6 3.6z"
              />
            </svg>
          </button>
        </div>
        <div className="spacer" />
        <div className="toolbar-group">
          <input
            className="name-input"
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="untitled"
            title="Project name: the saved design's name and the generated package's"
            aria-label="project name"
          />
          <button onClick={onSave} disabled={!spec} title="Save (⌘/Ctrl+S)">
            Save
          </button>
          <Picker className="load-select" placeholder="Project…" onChange={onLoad} options={loadOptions} />
          <input
            ref={importInputRef}
            className="hidden-file-input"
            type="file"
            accept="application/json,.json"
            onChange={(e) => {
              onImportFile(e.currentTarget.files?.[0] ?? null);
              e.currentTarget.value = "";
            }}
          />
          <button
            className="icon-btn"
            onClick={onRefresh}
            disabled={refreshing}
            title="Refresh: read the catalog, previews, code help and saved projects from the server again. Changes to library code need the server restarted, or --reload."
            aria-label="Refresh"
          >
            <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true" className={refreshing ? "spin" : ""}>
              <path
                fill="currentColor"
                d="M17.65 6.35A7.95 7.95 0 0 0 12 4a8 8 0 1 0 7.73 10h-2.08A6 6 0 1 1 12 6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"
              />
            </svg>
          </button>
        </div>
        <span className="toolbar-divider" />
        <button
          className="run-btn"
          onClick={() => setRunOpen(true)}
          disabled={!spec || !validation?.ok}
          title="launch the design in a real driver window"
        >
          ▶ Run
        </button>
        <button className="generate-btn" onClick={() => setGenerateOpen(true)} disabled={!spec || !validation?.ok}>
          Generate task
        </button>
        <ValidationBadge validation={validation} />
      </header>

      <main className="content">
        {tab === "map" ? (
          <MapTab spec={spec} onSpecChange={updateSpec} validation={validation} />
        ) : tab === "route" ? (
          <RouteTab spec={spec} onSpecChange={updateSpec} />
        ) : tab === "metadata" ? (
          <MetadataTab spec={spec} onChange={updateSpec} />
        ) : tab === "mdp" ? (
          <MdpTab spec={spec} />
        ) : (
          <CodeTab
            spec={spec}
            specText={specText}
            onSpecChange={updateSpec}
            onSpecTextChange={setSpecText}
            validation={validation}
          />
        )}
      </main>

      <footer className={validation && !validation.ok ? "statusbar invalid" : "statusbar"}>
        {validation && !validation.ok ? (
          <span className="invalid-reason" title={validation.error}>
            ⚠ {validation.error}
          </span>
        ) : (
          <span>{status}</span>
        )}
        <span className="spacer" />
        {validation?.ok && validation.summary && (
          <span>
            max aircraft: {validation.summary.max_aircraft} · obs:{" "}
            {validation.summary.obs_fields.length} · actions:{" "}
            {validation.summary.action_fields.length} · queryables:{" "}
            {validation.summary.queryables.length}
          </span>
        )}
      </footer>

      {generateOpen && spec && (
        <GenerateModal spec={spec} defaultName={saveName} onClose={() => setGenerateOpen(false)} />
      )}
      {runOpen && spec && <RunModal spec={spec} onClose={() => setRunOpen(false)} />}
    </div>
    </RefreshContext.Provider>
  );
}

function ValidationBadge({ validation }: { validation: ValidateResult | null }) {
  if (!validation) return <span className="badge pending">…</span>;
  if (validation.ok) return <span className="badge ok">valid</span>;
  return (
    <span className="badge error" title={validation.error}>
      invalid
    </span>
  );
}
