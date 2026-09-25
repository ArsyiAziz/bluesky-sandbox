import { useEffect, useState } from "react";
import { api, type GenerateResult, type SpecDict } from "../api";

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
  const chooseTemplate = (id: string) =>
    onSpecChange({ ...spec, metadata: { ...(spec.metadata ?? {}), template: id } });
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
          files.find((f) => f.endsWith(template === "rl" ? "train.py" : "env.py")) ?? files[0],
        );
      })
      .catch((e) => setError(String(e)));
  };

  // Generate on opening, and again when the template changes.
  useEffect(() => {
    generate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [template]);

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
        {error && <pre className="error-text">{error}</pre>}
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
