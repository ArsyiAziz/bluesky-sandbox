// The simulator settings: step length, aircraft types, performance model,
// conflict detection and wind.
import { useEffect, useRef, useState } from "react";
import { api, type SpecDict } from "../api";
import { clone } from "../specHelpers";
import { useRefresh } from "../refresh";
import { ConfigEditor } from "./panel/ConfigEditor";
import { Page } from "./form";

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
    </Page>
  );
}
