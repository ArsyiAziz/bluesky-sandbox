import { type KeyboardEvent as ReactKeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type SpecDict, type ValidateResult } from "./api";
import { forgetModuleMembers, useCodeIntel } from "./code/pythonEditor";
import { DEFAULT_SPEC } from "./defaultSpec";
import { migrateRewardHooks, migrateRotationGroups, normalizeToRegions } from "./specHelpers";
import MapTab from "./components/MapTab";
import CodeTab from "./components/CodeTab";
import RouteTab from "./route/RouteTab";
import GenerateModal from "./components/GenerateModal";
import ConfigTab from "./components/ConfigTab";
import SpacesTab from "./components/SpacesTab";
import { RefreshContext } from "./refresh";
import { EpisodeContext, useEpisodeState } from "./episode";
import MetadataTab from "./components/MetadataTab";
import RunModal from "./components/RunModal";
import AboutModal from "./components/AboutModal";
import { Picker } from "./components/panel/Picker";
import brandMark from "@brand/bluesky-sandbox-icon-small.svg";
import { setThemePreference, type ThemePreference, useTheme } from "./theme";

type Tab = "map" | "route" | "spaces" | "config" | "code" | "metadata";

const normalizeSpec = (spec: SpecDict): SpecDict =>
  migrateRotationGroups(migrateRewardHooks(normalizeToRegions(spec)));

const IMPORT_JSON_VALUE = "__import_json__";
// Each tab: its label, what it is for (its tooltip), and its icon's path (24px).
const TABS: { id: Tab; label: string; hint: string; icon: string }[] = [
  {
    id: "map",
    label: "Map",
    hint: "The airspace, regions, waypoints and spawns, on the map",
    icon: "M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2Zm0 2.2 6 2v11.6l-6-2V6.2Z",
  },
  {
    id: "route",
    label: "Route",
    hint: "The routes aircraft fly, leg by leg",
    icon: "M6 3a3 3 0 1 0 0 6 3 3 0 0 0 0-6Zm12 12a3 3 0 1 0 0 6 3 3 0 0 0 0-6ZM9 6h7a3 3 0 0 1 0 6H8a3 3 0 0 0 0 6h7v-2H8a1 1 0 0 1 0-2h8a5 5 0 0 0 0-10H9v2Z",
  },
  {
    id: "spaces",
    label: "Spaces",
    hint: "What the policy observes and what it does: the observation and action spaces",
    icon: "M4 4h7v7H4V4Zm9 0h7v7h-7V4ZM4 13h7v7H4v-7Zm9 0h7v7h-7v-7Z",
  },
  {
    id: "config",
    label: "Config",
    hint: "How the simulator runs: timing, aircraft, conflict detection, wind",
    icon: "M4 6h10v2H4V6Zm14 0h2v2h-2v2h-2V4h2v2ZM4 16h4v2H4v-2Zm8 0h8v2h-8v-2Zm-2-2h2v6h-2v-6Zm-6-5h8v2H4V9Zm12 0h4v2h-4V9Z",
  },
  {
    id: "code",
    label: "Code",
    hint: "The design's code: hooks, task info, custom fields, and the spec itself",
    icon: "m8.6 16.6-4.6-4.6 4.6-4.6L7.2 6 1.2 12l6 6 1.4-1.4Zm6.8 0 4.6-4.6-4.6-4.6L16.8 6l6 6-6 6-1.4-1.4Z",
  },
  {
    id: "metadata",
    label: "Metadata",
    hint: "The design's name, description and notes",
    icon: "M6 2h9l5 5v15H6V2Zm8 1.5V8h4.5L14 3.5ZM8 12v2h8v-2H8Zm0 4v2h8v-2H8Z",
  },
];
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
  // About opens from the brand, the status bar, or a link to #about.
  const [aboutOpen, setAboutOpen] = useState(() => window.location.hash === "#about");
  const [refreshKey, setRefreshKey] = useState(0);
  const episode = useEpisodeState();
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
    <EpisodeContext.Provider value={episode}>
    <div className="app">
      <a className="skip-link" href="#design">
        Skip to the design
      </a>
      <header className="toolbar">
        <div className="toolbar-main">
        <button
          type="button"
          className="brand"
          title="About BlueSky Sandbox"
          aria-label="About BlueSky Sandbox"
          onClick={() => setAboutOpen(true)}
        >
          <img className="brand-mark" src={brandMark} alt="" width={28} height={28} />
          <span className="brand-text" aria-hidden="true">
            <span className="brand-name">
              <span className="brand-name-blue">BlueSky</span> Sandbox
            </span>
            <span className="brand-app">Environment Designer</span>
          </span>
        </button>
        <nav className="tabs" aria-label="Design sections">
          <TabList tab={tab} onSelect={setTab} />
        </nav>
        </div>
        <div className="toolbar-tools">
        <div className="undo-redo joined" role="group" aria-label="History">
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
        <div className="toolbar-group" role="group" aria-label="Project">
          <div className="joined project-save">
          <input
            className="name-input"
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="untitled"
            title="Project name: the saved design's name and the generated package's"
            aria-label="Project name"
          />
          <button onClick={onSave} disabled={!spec} title="Save (⌘/Ctrl+S)">
            Save
          </button>
          </div>
          <div className="joined project-open">
          <Picker
            className="load-select"
            placeholder="Project…"
            ariaLabel="Open, import or delete a project"
            onChange={onLoad}
            options={loadOptions}
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
        </div>
        <span className="toolbar-divider" aria-hidden="true" />
        <div className="toolbar-actions" role="group" aria-label="Design">
          <button
            className="run-btn"
            onClick={() => setRunOpen(true)}
            disabled={!spec || !validation?.ok}
            title="Launch the design in a real driver window"
          >
            <span aria-hidden="true">▶</span> Run
          </button>
          <button
            className="generate-btn"
            onClick={() => setGenerateOpen(true)}
            disabled={!spec || !validation?.ok}
            title="Generate the design as a task package"
          >
            Generate<span className="generate-more"> task</span>
          </button>
          <ValidationBadge validation={validation} />
        </div>
        </div>
      </header>

      <main
        id="design"
        className="content"
        role="tabpanel"
        aria-labelledby={`tab-${tab}`}
        tabIndex={-1}
      >
        {tab === "map" ? (
          <MapTab spec={spec} onSpecChange={updateSpec} />
        ) : tab === "route" ? (
          <RouteTab spec={spec} onSpecChange={updateSpec} />
        ) : tab === "metadata" ? (
          <MetadataTab spec={spec} onChange={updateSpec} />
        ) : tab === "spaces" ? (
          <SpacesTab
            spec={spec}
            onChange={updateSpec}
            validationError={validation && !validation.ok ? validation.error : undefined}
          />
        ) : tab === "config" ? (
          <ConfigTab spec={spec} onChange={updateSpec} />
        ) : (
          <CodeTab
            spec={spec}
            specText={specText}
            onSpecChange={updateSpec}
            onSpecTextChange={setSpecText}
            validation={validation}
            onShowSpaces={() => setTab("spaces")}
          />
        )}
      </main>

      <footer className={validation && !validation.ok ? "statusbar invalid" : "statusbar"}>
        {validation && !validation.ok ? (
          <span className="invalid-reason" title={validation.error}>
            <span aria-hidden="true">⚠</span> {validation.error}
          </span>
        ) : (
          <span role="status" aria-live="polite">
            {status}
          </span>
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
        <button type="button" className="statusbar-btn" onClick={() => setAboutOpen(true)}>
          About
        </button>
        <ThemeSwitch />
      </footer>

      {generateOpen && spec && (
        <GenerateModal
          spec={spec}
          defaultName={saveName}
          onSpecChange={updateSpec}
          onClose={() => setGenerateOpen(false)}
        />
      )}
      {runOpen && spec && <RunModal spec={spec} onClose={() => setRunOpen(false)} />}
      {aboutOpen && <AboutModal onClose={() => setAboutOpen(false)} />}
    </div>
    </EpisodeContext.Provider>
    </RefreshContext.Provider>
  );
}

// The design sections as tabs. Arrow keys (and Home / End) move between them,
// Enter or Space opens one - a tab builds its whole view, so focus alone does
// not switch it - and Tab leaves the list for the toolbar.
function TabList({ tab, onSelect }: { tab: Tab; onSelect: (tab: Tab) => void }) {
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});
  const [focused, setFocused] = useState<Tab>(tab);
  useEffect(() => setFocused(tab), [tab]);
  const move = (to: number) => {
    const next = TABS[(to + TABS.length) % TABS.length].id;
    setFocused(next);
    refs.current[next]?.focus();
  };
  const onKeyDown = (e: ReactKeyboardEvent) => {
    const at = TABS.findIndex((t) => t.id === focused);
    const keys: Record<string, () => void> = {
      ArrowRight: () => move(at + 1),
      ArrowLeft: () => move(at - 1),
      Home: () => move(0),
      End: () => move(TABS.length - 1),
    };
    if (keys[e.key]) {
      e.preventDefault();
      keys[e.key]();
    }
  };
  return (
    <div role="tablist" aria-label="Design sections" onKeyDown={onKeyDown}>
      {TABS.map((t) => (
        <button
          key={t.id}
          id={`tab-${t.id}`}
          ref={(el) => (refs.current[t.id] = el)}
          role="tab"
          type="button"
          aria-selected={tab === t.id}
          aria-controls="design"
          tabIndex={t.id === focused ? 0 : -1}
          className={tab === t.id ? "tab active" : "tab"}
          title={t.hint}
          onClick={() => onSelect(t.id)}
          onFocus={() => setFocused(t.id)}
        >
          <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
            <path fill="currentColor" d={t.icon} />
          </svg>
          <span className="tab-label">{t.label}</span>
        </button>
      ))}
    </div>
  );
}

// Light, dark, or the system's: three buttons, one pressed.
const THEMES: { id: ThemePreference; label: string; icon: string }[] = [
  {
    id: "light",
    label: "Light theme",
    icon: "M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10ZM11 1h2v3h-2V1Zm0 19h2v3h-2v-3ZM3.5 4.9l1.4-1.4 2.1 2.1-1.4 1.4-2.1-2.1Zm13.4 13.4 1.4-1.4 2.1 2.1-1.4 1.4-2.1-2.1ZM1 11h3v2H1v-2Zm19 0h3v2h-3v-2ZM3.5 19.1l2.1-2.1 1.4 1.4-2.1 2.1-1.4-1.4ZM16.9 5.6 19 3.5l1.4 1.4-2.1 2.1-1.4-1.4Z",
  },
  {
    id: "dark",
    label: "Dark theme",
    icon: "M21 14.5A8.5 8.5 0 0 1 9.5 3a8.5 8.5 0 1 0 11.5 11.5Z",
  },
  {
    id: "system",
    label: "Theme of the system",
    icon: "M3 4h18v12H3V4Zm2 2v8h14V6H5Zm3 12h8v2H8v-2Z",
  },
];

function ThemeSwitch() {
  const { preference } = useTheme();
  return (
    <div className="theme-switch" role="group" aria-label="Theme">
      {THEMES.map((t) => (
        <button
          key={t.id}
          type="button"
          className={preference === t.id ? "on" : ""}
          aria-pressed={preference === t.id}
          aria-label={t.label}
          title={t.label}
          onClick={() => setThemePreference(t.id)}
        >
          <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
            <path fill="currentColor" d={t.icon} />
          </svg>
        </button>
      ))}
    </div>
  );
}

// Whether the design builds: a word and a sign, not color alone.
function ValidationBadge({ validation }: { validation: ValidateResult | null }) {
  if (!validation)
    return (
      <span className="badge pending" title="Checking the design…">
        <span aria-hidden="true">…</span>
        <span className="visually-hidden">Checking the design</span>
      </span>
    );
  if (validation.ok)
    return (
      <span className="badge ok" title="The design builds">
        <span aria-hidden="true">✓</span> Valid
      </span>
    );
  return (
    <span className="badge error" title={validation.error}>
      <span aria-hidden="true">⚠</span> Invalid
    </span>
  );
}
