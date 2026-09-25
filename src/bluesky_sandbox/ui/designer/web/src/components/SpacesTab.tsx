// The design's spaces: the fields edited on the left, and on the right the
// observation and action they build, column by column.
import type { SpecDict } from "../api";
import MdpTab from "./MdpTab";
import SpacesEditor from "./SpacesEditor";

export default function SpacesTab({
  spec,
  onChange,
  validationError,
}: {
  spec: SpecDict | null;
  onChange: (next: SpecDict) => void;
  validationError?: string;
}) {
  return (
    <div className="spaces-tab">
      <aside className="spaces-side">
        {spec ? (
          <SpacesEditor spec={spec} onChange={onChange} validationError={validationError} />
        ) : (
          <p className="muted">The spec has a JSON error; fix it in the Code tab.</p>
        )}
      </aside>
      <MdpTab spec={spec} />
    </div>
  );
}
