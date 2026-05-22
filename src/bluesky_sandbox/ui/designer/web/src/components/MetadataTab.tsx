import type { SpecDict } from "../api";

/**
 * The design's metadata: what it is, which version, and why.
 *
 * Both live in `spec.metadata`, which the designer previously surfaced only as
 * the project name - so the rationale behind a design had nowhere to go except
 * a comment inside the generated code, where it is emitted into every package
 * and cannot be read without opening one. Written here, it lands in the
 * generated README instead, and `version` lands in the package's
 * `__version__` so a run can be traced back to the design that produced it.
 */
export default function MetadataTab({
  spec,
  onChange,
}: {
  spec: SpecDict | null;
  onChange: (next: SpecDict) => void;
}) {
  if (!spec) return <div className="metadata-tab muted">Spec has a JSON error; fix it in the Code tab.</div>;

  const metadata = (spec.metadata ?? {}) as Record<string, unknown>;
  const edit = (patch: Record<string, unknown>) =>
    onChange({ ...spec, metadata: { ...metadata, ...patch } });

  return (
    <div className="metadata-tab">
      <label className="metadata-field">
        <span className="metadata-label">version</span>
        <input
          className="name-input"
          value={String(metadata.version ?? "")}
          placeholder="e.g. 61.2"
          onChange={(e) => edit({ version: e.target.value })}
        />
        <span className="muted small">
          Written to the package's <code>__version__</code> and its README. Bump it when the
          design changes so a checkpoint can be traced to what produced it.
        </span>
      </label>

      <label className="metadata-field">
        <span className="metadata-label">notes</span>
        <textarea
          className="metadata-note"
          value={String(metadata.note ?? "")}
          placeholder={
            "What this design changes and why.\n\n" +
            "Goes into the generated README."
          }
          spellCheck={false}
          onChange={(e) => edit({ note: e.target.value })}
        />
      </label>
    </div>
  );
}
