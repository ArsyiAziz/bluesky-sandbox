// "Run" dialog: launches the design in a real driver window (pygame / panda3d /
// qtgl) with view + route options. Self-contained — owns the launch + status
// polling so the toolbar stays uncluttered. Tracking continues for the whole
// run (not just until the window opens), surfacing the child's output and, on a
// crash, its traceback.
import { useCallback, useEffect, useRef, useState } from "react";
import { api, type SpecDict } from "../api";
import { Picker } from "./panel/Picker";

type Driver = { render_mode: string; views: string[]; default_views: string[] };
type Phase = "idle" | "launching" | "running" | "exited";

export default function RunModal({ spec, onClose }: { spec: SpecDict; onClose: () => void }) {
  const [drivers, setDrivers] = useState<Driver[]>([]);
  const [mode, setMode] = useState("pygame");
  const [views, setViews] = useState<string[]>([]);
  const [showAllRoutes, setShowAllRoutes] = useState(true);
  const [autoTrack, setAutoTrack] = useState(false);

  /* "all routes" and "track aircraft" are two ways to decide what the view
     follows, so they are exclusive: turning one on turns the other off.
     Clicking the active one turns it off again, which leaves neither - a
     valid third state (nothing overlaid, nothing selected until you click). */
  const selectRouteMode = (mode: "all" | "track") => {
    if (mode === "all") {
      const next = !showAllRoutes;
      setShowAllRoutes(next);
      if (next) setAutoTrack(false);
    } else {
      const next = !autoTrack;
      setAutoTrack(next);
      if (next) setShowAllRoutes(false);
    }
  };
  const [zeroAction, setZeroAction] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [crashed, setCrashed] = useState(false);
  const [status, setStatus] = useState("");
  const [log, setLog] = useState("");

  const pollTimer = useRef<number | null>(null);
  const clearPoll = () => {
    if (pollTimer.current != null) {
      clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }
  };

  useEffect(() => {
    api.catalogOnce().then((c) => setDrivers(c?.drivers ?? [])).catch(() => setDrivers([]));
  }, []);

  const modeDriver = drivers.find((d) => d.render_mode === mode);
  useEffect(() => {
    setViews(modeDriver?.default_views ?? []);
  }, [mode, modeDriver?.render_mode]);

  // Poll the run's status until it exits, so a crash *after* the window opened
  // is caught too (not just startup failures).
  const poll = useCallback(() => {
    api
      .runStatus()
      .then((s) => {
        if (!s.active) {
          setPhase("idle");
          return;
        }
        if (!s.alive) {
          const rc = s.returncode ?? null;
          const didCrash = rc != null && rc !== 0;
          setCrashed(didCrash);
          setPhase("exited");
          setStatus(didCrash ? `run crashed (exit ${rc})` : "run finished");
          setLog(s.error ?? s.log ?? "");
          return; // stop polling — process is gone
        }
        if (s.ready) {
          setPhase("running");
          setCrashed(false);
          setStatus(`running in ${s.render_mode} (pid ${s.pid}) — see the new window`);
          if (s.log) setLog(s.log);
          pollTimer.current = window.setTimeout(poll, 1500);
        } else {
          setPhase("launching");
          pollTimer.current = window.setTimeout(poll, 500);
        }
      })
      .catch((e) => {
        setStatus(`run status check failed: ${e}`);
        setPhase("idle");
      });
  }, []);

  // Resume tracking if a run is already active (e.g. the modal was reopened),
  // and always stop the timer when the modal unmounts.
  useEffect(() => {
    let cancelled = false;
    api
      .runStatus()
      .then((s) => {
        if (!cancelled && s.active) poll();
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      clearPoll();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onRun = useCallback(() => {
    clearPoll();
    setPhase("launching");
    setCrashed(false);
    setLog("");
    setStatus(`launching ${mode}…`);
    api
      .run(spec, mode, views, showAllRoutes, autoTrack, 0, zeroAction ? "zero" : "random")
      .then(() => poll())
      .catch((e) => {
        setStatus(`launch failed: ${e}`);
        setPhase("idle");
      });
  }, [spec, mode, views, showAllRoutes, autoTrack, zeroAction, poll]);

  const onStop = useCallback(() => {
    clearPoll();
    api
      .runStop()
      .then(() => {
        setPhase("idle");
        setStatus("stopped the run");
      })
      .catch(() => {});
  }, []);

  const busy = phase === "launching" || phase === "running";

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal run-modal" onClick={(e) => e.stopPropagation()}>
        <header className="modal-head">
          <strong>Run on a driver</strong>
          <span className="spacer" />
          <button onClick={onClose}>Close</button>
        </header>
        <div className="run-modal-body">
          <label className="numfield inline">
            <span>driver</span>
            <Picker
              searchable={false}
              value={mode}
              placeholder="pygame"
              onChange={setMode}
              options={[
                { value: "pygame", description: "2D multi-view (default)" },
                { value: "panda3d", description: "3D world view" },
                { value: "qtgl", description: "BlueSky native Qt/GL" },
              ]}
            />
          </label>

          {(modeDriver?.views.length ?? 0) > 0 && (
            <div className="run-row">
              <span className="run-row-label">views</span>
              <div className="view-toggles">
                {modeDriver!.views.map((v) => {
                  const on = views.includes(v);
                  return (
                    <button
                      key={v}
                      className={on ? "view-toggle on" : "view-toggle"}
                      onClick={() => setViews((cur) => (on ? cur.filter((x) => x !== v) : [...cur, v]))}
                    >
                      {v.replace(/View$/, "")}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          <div className="run-row">
            <span className="run-row-label">routes</span>
            <button
              className={showAllRoutes ? "view-toggle on" : "view-toggle"}
              onClick={() => selectRouteMode("all")}
              title="overlay the design's defined routes — turns off aircraft tracking"
            >
              all routes
            </button>
            <button
              className={autoTrack ? "view-toggle on" : "view-toggle"}
              onClick={() => selectRouteMode("track")}
              title="keep an aircraft selected (the first live one) and follow its route — for eval videos; turns off the all-routes overlay"
            >
              track aircraft
            </button>
          </div>

          <div className="run-row">
            <span className="run-row-label">action</span>
            <button
              className={!zeroAction ? "view-toggle on" : "view-toggle"}
              onClick={() => setZeroAction(false)}
              title="sample the action space each step (default) — random maneuvers"
            >
              random
            </button>
            <button
              className={zeroAction ? "view-toggle on" : "view-toggle"}
              onClick={() => setZeroAction(true)}
              title="the null action (all zeros) every step — in the waypoint-relative frame that flies the nominal route directly, so you see the un-controlled dynamics"
            >
              0 action
            </button>
          </div>

          <div className="run-actions">
            <button className="run-btn" onClick={onRun} disabled={busy}>
              {phase === "launching" ? <span className="spinner" /> : "▶"} Run
            </button>
            {busy && (
              <button className="link danger" onClick={onStop} title="stop the run">
                stop
              </button>
            )}
            {status && (
              <span className={`run-status muted small${crashed ? " error" : ""}`}>
                {phase === "running" && <span className="run-dot" />}
                {status}
              </span>
            )}
          </div>

          {log && (
            <div className={`run-log-panel${crashed ? " crashed" : ""}`}>
              <div className="run-log-head">
                <span>{crashed ? "error output" : "output"}</span>
                <span className="spacer" />
                <button className="link" onClick={() => navigator.clipboard?.writeText(log)} title="copy">
                  copy
                </button>
              </div>
              <pre className="run-log">{log}</pre>
            </div>
          )}

          <p className="muted small">
            Opens a window on the machine running the designer. Only one driver runs at a time.
          </p>
        </div>
      </div>
    </div>
  );
}
