// The design's spaces: the fields edited on the left, and on the right the
// observation and action they build, column by column.
import type { SpecDict } from "../api";
import MdpTab from "./MdpTab";
import SpacesEditor from "./SpacesEditor";
import { Resizer, useStoredSize } from "./Resizer";

export default function SpacesTab({
  spec,
  onChange,
  validationError,
}: {
  spec: SpecDict | null;
  onChange: (next: SpecDict) => void;
  validationError?: string;
}) {
  const [sideW, setSideW, resetSideW] = useStoredSize("designer.spaces.editorWidth", 450, 320, () => window.innerWidth * 0.6);
  return (
    <div className="spaces-tab">
      <aside className="spaces-side" style={{ width: sideW }}>
        {spec ? (
          <SpacesEditor spec={spec} onChange={onChange} validationError={validationError} />
        ) : (
          <p className="muted">The spec has a JSON error; fix it in the Code tab.</p>
        )}
      </aside>
      <Resizer axis="x" label="Resize the field editor" onDrag={(x) => setSideW(x)} onReset={resetSideW} />
      <MdpTab spec={spec} />
    </div>
  );
}
