// The simulator settings: step length, aircraft types, performance model,
// conflict detection and wind.
import { useEffect, useRef, useState } from "react";
import { api, type SpecDict } from "../api";
import { clone } from "../specHelpers";
import { useRefresh } from "../refresh";
import { ConfigEditor } from "./panel/ConfigEditor";
import { FormCard, FormRow, NumberInput, Page } from "./form";

export default function ConfigTab({ spec, onChange }: { spec: SpecDict | null; onChange: (next: SpecDict) => void }) {
  const [catalog, setCatalog] = useState<any>(null);
  const refreshKey = useRefresh();
  useEffect(() => {
    api.catalogOnce().then(setCatalog).catch(() => setCatalog(null));
  }, [refreshKey]);
  const latest = useRef(spec);
  latest.current = spec;
  if (!spec) return <div className="form-page muted">The spec has a JSON error; fix it in the Code tab.</div>;
  const edit = (mut: (s: SpecDict) => void) => {
    const next = clone(latest.current ?? spec);
    mut(next);
    latest.current = next;
    onChange(next);
  };
  return (
    <Page
      title="Config"
      intro="How the simulator runs the design. Reward, termination and truncation are code: see the Code tab's env hooks."
    >
      <ConfigEditor
        env={spec.env ?? {}}
        code={spec.code ?? {}}
        catalog={catalog}
        onChange={(env) => edit((s) => (s.env = env))}
        onCodeChange={(code) => edit((s) => (s.code = code))}
      />
      <GridCard
        grid={spec.grid ?? {}}
        onChange={(grid) =>
          edit((s) => {
            if (Object.keys(grid).length) s.grid = grid;
            else delete s.grid;
          })
        }
      />
    </Page>
  );
}

// The design's grid: one step per quantity, applied everywhere it belongs (the
// builder's apply_grid) - target altitudes on its levels, every step action's
// step. Blank leaves a quantity without a grid.
function GridCard({
  grid,
  onChange,
}: {
  grid: Record<string, number>;
  onChange: (grid: Record<string, number>) => void;
}) {
  const set = (key: string, v: number | null) => {
    const next = { ...grid };
    if (v == null || !(v > 0)) delete next[key];
    else next[key] = v;
    onChange(next);
  };
  return (
    <FormCard
      title="Grid"
      help={
        <>
          One step per quantity for the whole design. Altitude: the levels every
          target altitude is on - fixes, spawn areas marked "on the grid's
          levels", and every altitude action's command (step action or not), so
          every aircraft flies the same levels. Every step action takes its
          quantity's step. Blank: no grid for it.
        </>
      }
    >
      <FormRow label="altitude" help="flight levels">
        <NumberInput unit="ft" step={500} min={0} placeholder="none" value={grid.alt_ft ?? null} onChange={(v) => set("alt_ft", v)} />
      </FormRow>
      <FormRow label="speed" help="a speed step">
        <NumberInput unit="kt" step={5} min={0} placeholder="none" value={grid.spd_kts ?? null} onChange={(v) => set("spd_kts", v)} />
      </FormRow>
      <FormRow label="heading" help="a turn step">
        <NumberInput unit="deg" step={5} min={0} placeholder="none" value={grid.hdg_deg ?? null} onChange={(v) => set("hdg_deg", v)} />
      </FormRow>
    </FormCard>
  );
}
