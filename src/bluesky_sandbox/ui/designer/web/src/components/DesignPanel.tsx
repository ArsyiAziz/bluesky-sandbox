import type { SpecDict } from "../api";
import type { EditTarget } from "../map/types";
import GeometryTab from "./panel/GeometryTab";

// The map's side panel: the design's geometry - its elements by kind or by
// dependency, and the picked one's inspector. Fields are edited in the Spaces
// tab (with a sample of what an aircraft observes), settings in Config. The
// spec object is the source of truth; every edit yields a new spec via
// onChange, which App also re-serializes into the code editor.
export default function DesignPanel({
  spec,
  onChange,
  onFocusShape,
  viewCenter,
  hiddenElements,
  onToggleHidden,
  lockedElements,
  onToggleLocked,
  onHighlightRoute,
  selectedKey,
  onSelect,
  width,
}: {
  spec: SpecDict;
  onChange: (next: SpecDict) => void;
  onFocusShape: (bounds: SpecDict) => void;
  viewCenter?: [number, number];
  hiddenElements: Set<string>;
  onToggleHidden: (key: string) => void;
  lockedElements: Set<string>;
  onToggleLocked: (key: string) => void;
  onHighlightRoute: (name: string | null) => void;
  selectedKey?: string | null;
  onSelect?: (target: EditTarget | null) => void;
  width?: number;
}) {
  return (
    <div className="design-panel" style={width ? { width } : undefined}>
      <GeometryTab
        spec={spec}
        onChange={onChange}
        onFocusShape={onFocusShape}
        viewCenter={viewCenter}
        hiddenElements={hiddenElements}
        onToggleHidden={onToggleHidden}
        lockedElements={lockedElements}
        onToggleLocked={onToggleLocked}
        onHighlightRoute={onHighlightRoute}
        selectedKey={selectedKey}
        onSelect={onSelect ?? (() => {})}
      />
    </div>
  );
}
