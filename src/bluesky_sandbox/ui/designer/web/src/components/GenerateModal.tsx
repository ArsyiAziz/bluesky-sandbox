import { useEffect, useState } from "react";
import { api, type GenerateResult, type SpecDict } from "../api";

// A design's recording settings, as kept in its metadata.
type RecordMeta = {
  every: number;
  length: number;
  fps: number;
  driver: string;
  views: string[];
  folder?: string | null;
};

// "Generate task structure": turns the current design into a runnable task
// package (design.json + scenario/env/task scaffolding) the user can download
// and keep iterating on in code.
// The packages it can make: the environment alone, or with a training-loop
// scaffold laid out around the design's MDP. The choice is kept in the design.
const TEMPLATES = [
  { id: "plain", label: "Plain", title: "The environment, and a smoke rollout (python -m <package>)" },
  {
    id: "rl",
    label: "RL",
    title:
      "Adds train.py: a training-loop scaffold for this MDP - the privileged critic's views when the design has critic-only fields, a cost critic when it defines a cost",
  },
  {
    id: "sb3",
    label: "SB3",
    title: "Adds train.py: every agent trained with one shared Stable-Baselines3 PPO policy, flagging what SB3 cannot do with this design",
  },
];

export default function GenerateModal({
  spec,
  defaultName,
  onSpecChange,
  onClose,
}: {
  spec: SpecDict;
  defaultName: string;
  onSpecChange: (next: SpecDict) => void;
  onClose: () => void;
}) {
  const template = String(spec.metadata?.template ?? "plain");
  const setMeta = (patch: Record<string, unknown>) =>
    onSpecChange({ ...spec, metadata: { ...(spec.metadata ?? {}), ...patch } });
  const chooseTemplate = (id: string) => setMeta({ template: id });
  // How the training script runs the env: in how many processes, one copy watched.
  const processes = Math.max(1, Number(spec.metadata?.processes ?? 1) || 1);
  const watch = Boolean(spec.metadata?.watch);
  // Video clips of training, drawn offscreen and uploaded; null when off.
  const record = (spec.metadata?.record ?? null) as RecordMeta | null;
  const trains = template !== "plain";
  const [name, setName] = useState(defaultName);
  const [result, setResult] = useState<GenerateResult | null>(null);
  const [selected, setSelected] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("");
  const canSaveFolder = typeof window !== "undefined" && !!window.showDirectoryPicker;

  const generate = () => {
    setError(null);
    api
      .generate(spec, name)
      .then((r) => {
        setResult(r);
        const files = Object.keys(r.files);
        setSelected(
          files.find((f) => f.endsWith(template === "plain" ? "env.py" : "train.py")) ?? files[0],
        );
      })
      .catch((e) => setError(String(e)));
  };

  // Generate on opening, and again when the template or how it runs changes.
  useEffect(() => {
    const handle = setTimeout(generate, 250);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [template, processes, watch, JSON.stringify(record)]);

  // What can record, from the server: each driver's views and the defaults.
  const catalog = result?.recording;
  const setRecord = (patch: Partial<RecordMeta> | null) =>
    setMeta(patch === null ? { record: null } : { record: { ...(record ?? {}), ...patch } });
  const startRecording = () => {
    if (!catalog) return;
    const { driver, ...defaults } = catalog.defaults;
    // Both draw worker 0 - one in a window, one offscreen - so recording
    // replaces watching.
    setMeta({
      watch: false,
      record: { ...defaults, driver, views: catalog.drivers[driver].default },
    });
  };
  const recordDriver = record?.driver ?? catalog?.defaults.driver ?? "";
  const offeredViews = catalog?.drivers[recordDriver]?.views ?? [];
  const recordViews = record?.views ?? catalog?.drivers[recordDriver]?.default ?? [];
  const toggleView = (view: string) => {
    const next = recordViews.includes(view)
      ? recordViews.filter((v) => v !== view)
      : [...recordViews, view];
    // Kept in the driver's order: pygame stacks them top to bottom in it.
    if (next.length) setRecord({ views: offeredViews.filter((v) => next.includes(v)) });
  };

  const writeFolder = async (
    root: FileSystemDirectoryHandle,
    files: Record<string, string>,
  ) => {
    for (const [path, source] of Object.entries(files)) {
      const parts = path.split("/").filter(Boolean);
      if (parts.length === 0) continue;
      let dir = root;
      for (const part of parts.slice(0, -1)) {
        dir = await dir.getDirectoryHandle(part, { create: true });
      }
      const file = await dir.getFileHandle(parts[parts.length - 1], { create: true });
      const writable = await file.createWritable();
      await writable.write(source);
      await writable.close();
    }
  };

  const saveFolder = async () => {
    if (!window.showDirectoryPicker) {
      setError("Folder save is not supported in this browser. Use Download .zip.");
      return;
    }
    setError(null);
    setStatus("");
    try {
      const next = await api.generate(spec, name);
      setResult(next);
      setSelected(Object.keys(next.files).find((f) => f.endsWith("env.py")) ?? Object.keys(next.files)[0]);
      const dir = await window.showDirectoryPicker({ mode: "readwrite" });
      await writeFolder(dir, next.files);
      setStatus(`saved ${next.package}/`);
    } catch (e) {
      setError(String(e));
    }
  };

  const download = () => {
    if (!result) return;
    // Download the whole project as a real .zip of the package directory.
    api
      .generateZip(spec, name)
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${result.package}.zip`;
        a.click();
        URL.revokeObjectURL(url);
      })
      .catch((e) => setError(String(e)));
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <header className="modal-head">
          <strong>Generate task package</strong>
          <span className="spacer" />
          <input value={name} onChange={(e) => setName(e.target.value)} />
          <button onClick={generate}>Regenerate</button>
          {canSaveFolder && (
            <button onClick={saveFolder}>
              Save folder
            </button>
          )}
          <button onClick={download} disabled={!result}>
            Download .zip
          </button>
          <button onClick={onClose}>Close</button>
        </header>
        <div className="generate-options">
          <div className="seg" role="radiogroup" aria-label="template">
            {TEMPLATES.map((t) => (
              <button
                key={t.id}
                role="radio"
                aria-checked={template === t.id}
                className={template === t.id ? "on" : ""}
                title={t.title}
                onClick={() => chooseTemplate(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>
          {trains && (
            <div className="generate-run" title="How train.py runs the env. BlueSky is one simulator per process, so each process runs its own copy; the copies step together.">
              <label>
                processes
                <input
                  type="number"
                  min={1}
                  max={256}
                  value={processes}
                  onChange={(e) => setMeta({ processes: Math.max(1, Math.round(Number(e.target.value) || 1)) })}
                />
              </label>
              {result?.cpus && <span className="muted small">of {result.cpus} cores</span>}
              <label title="Draw one copy in a pygame window; the others run headless. The copies step together, so the drawn one sets the pace.">
                <input
                  type="checkbox"
                  checked={watch}
                  onChange={(e) => setMeta(e.target.checked ? { watch: true, record: null } : { watch: false })}
                />
                watch one copy
              </label>
              <label title="Every copy records video clips offscreen, uploaded to wandb as they finish. Nothing is drawn between clips.">
                <input
                  type="checkbox"
                  checked={record !== null}
                  disabled={!catalog}
                  onChange={(e) => (e.target.checked ? startRecording() : setRecord(null))}
                />
                record video
              </label>
            </div>
          )}
        </div>
        {trains && record && catalog && (
          <div className="generate-options generate-run generate-record">
            <label title="A clip starts every this many steps of each copy">
              every
              <input
                type="number"
                min={1}
                value={record.every}
                onChange={(e) => setRecord({ every: Math.max(1, Math.round(Number(e.target.value) || 1)) })}
              />
              steps
            </label>
            <label title="Steps per clip: one frame each">
              clip
              <input
                type="number"
                min={1}
                max={record.every}
                value={record.length}
                onChange={(e) => setRecord({ length: Math.max(1, Math.round(Number(e.target.value) || 1)) })}
              />
              steps
            </label>
            <label title="Playback frame rate">
              <input
                type="number"
                min={1}
                max={60}
                value={record.fps}
                onChange={(e) => setRecord({ fps: Math.max(1, Math.round(Number(e.target.value) || 1)) })}
              />
              fps
            </label>
            <label title="Who draws the clips">
              drawn by
              <select
                value={recordDriver}
                onChange={(e) => setRecord({ driver: e.target.value, views: catalog.drivers[e.target.value].default })}
              >
                {Object.keys(catalog.drivers).map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </label>
            <span className="generate-views" title="The views a clip shows">
              {offeredViews.map((view) => (
                <label key={view}>
                  <input type="checkbox" checked={recordViews.includes(view)} onChange={() => toggleView(view)} />
                  {view}
                </label>
              ))}
            </span>
            <label title="Keep every clip in this folder too. Blank: clips are uploaded from memory and none is kept.">
              keep in
              <input
                className="generate-folder"
                placeholder="(upload only)"
                value={record.folder ?? ""}
                onChange={(e) => setRecord({ folder: e.target.value || null })}
              />
            </label>
          </div>
        )}
        {error && <pre className="error-text">{error}</pre>}
        {result?.notes && result.notes.length > 0 && (
          <ul className="generate-notes">
            {result.notes.map((n, i) => (
              <li key={i} className={`generate-note ${n.level}`}>
                <b>{n.level === "error" ? "Cannot train" : n.level === "warning" ? "Differs" : "Note"}</b>
                <span>{n.message}</span>
              </li>
            ))}
          </ul>
        )}
        {status && <p className="muted small generate-status">{status}</p>}
        {result && (
          <div className="modal-body">
            <ul className="file-list">
              {Object.keys(result.files).map((path) => (
                <li
                  key={path}
                  className={path === selected ? "active" : ""}
                  onClick={() => setSelected(path)}
                >
                  {path}
                </li>
              ))}
            </ul>
            <pre className="file-view">{selected ? result.files[selected] : ""}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
