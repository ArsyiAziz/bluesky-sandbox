import type { SpecDict } from "../api";
import { FormCard, FormRow, Page } from "./form";

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
  if (!spec) return <div className="form-page muted">Spec has a JSON error; fix it in the Code tab.</div>;

  const metadata = (spec.metadata ?? {}) as Record<string, unknown>;
  const edit = (patch: Record<string, unknown>) =>
    onChange({ ...spec, metadata: { ...metadata, ...patch } });

  return (
    <Page title="Metadata" intro="What the design is and why. Both go into the generated package.">
      <FormCard title="Version">
        <FormRow
          label="version"
          help={
            <>
              Written to the package's <code>__version__</code> and README. Bump it when the design changes, so a
              checkpoint can be traced to what produced it.
            </>
          }
        >
          <input
            className="form-input"
            value={String(metadata.version ?? "")}
            placeholder="e.g. 61.2"
            onChange={(e) => edit({ version: e.target.value })}
          />
        </FormRow>
        <FormRow label="project" help="Set in the toolbar; the saved design's name and the package's.">
          <span className="form-value">{String(metadata.name ?? "untitled")}</span>
        </FormRow>
      </FormCard>
      <FormCard title="Notes" help="Goes into the generated README." wide className="notes-card">
        <textarea
          className="metadata-note"
          value={String(metadata.note ?? "")}
          placeholder={"What this design changes and why."}
          spellCheck={false}
          onChange={(e) => edit({ note: e.target.value })}
        />
      </FormCard>
    </Page>
  );
}
