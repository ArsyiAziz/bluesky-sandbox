import { useState } from "react";
import type { SpecDict } from "../api";
import type { EditTarget } from "../map/types";
import { Section } from "./panel/Section";
import { SamplingReadout } from "./panel/SamplingReadout";
import { ObsSample } from "./panel/ObsSample";
import GeometryTab from "./panel/GeometryTab";

// The map's side panel: the design's geometry and role assignments, and a live
// sampling readout. Fields are edited in the Spaces tab, settings in Config. The spec object is the
// source of truth; every edit yields a new spec via onChange, which App also
// re-serializes into the code editor.
export default function DesignPanel({
  spec,
  onChange,
  onFocusBounds,
  seed,
  onSeedChange,
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
  onFocusBounds: (bounds: SpecDict) => void;
  seed: number;
  onSeedChange: (seed: number) => void;
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
  const [tab, setTab] = useState<"geometry" | "sampling">("geometry");

  const TABS: { id: typeof tab; label: string }[] = [
    { id: "geometry", label: "Geometry" },
    { id: "sampling", label: "Sampling" },
  ];

  return (
    <div className="design-panel" style={width ? { width } : undefined}>
      <nav className="panel-tabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "panel-tab active" : "panel-tab"} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "geometry" && (
        <GeometryTab
          spec={spec}
          onChange={onChange}
          onFocusBounds={onFocusBounds}
          viewCenter={viewCenter}
          hiddenElements={hiddenElements}
          onToggleHidden={onToggleHidden}
          lockedElements={lockedElements}
          onToggleLocked={onToggleLocked}
          onHighlightRoute={onHighlightRoute}
          selectedKey={selectedKey}
          onSelect={onSelect ?? (() => {})}
        />
      )}

      {tab === "sampling" && (
        <>
          <Section title="Sampling" subtitle="a drawn episode" hint="One episode drawn with the seed below - the concrete geometry, spawns and routes the design produces. Change the seed to see how much the design varies.">
            <SamplingReadout spec={spec} seed={seed} onSeedChange={onSeedChange} />
          </Section>
          <Section title="Observations" subtitle="what the policy sees" hint="The observation vector for a sampled agent, field by field, with the values it would actually receive. Use it to check normalization and ordering.">
            <ObsSample spec={spec} seed={seed} />
          </Section>
        </>
      )}
    </div>
  );
}
