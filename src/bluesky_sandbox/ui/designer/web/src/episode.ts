// The sampled episode every view shares: its number (the seed it is drawn
// with, stepped by the episode stepper), and the aircraft picked in it. The
// map and the Spaces tab (its sample, its per-aircraft ranges) all read one
// episode, so what one shows the others agree with.
import { createContext, useContext, useEffect, useState } from "react";
import { api, type EpisodeRunDone, type SampleResult, type SpawnedAircraft, type SpecDict } from "./api";
import { useRefresh } from "./refresh";

// A picked aircraft: when it spawns, and its callsign once the episode run
// has named it.
export type Pick = { at_s: number; acid: string | null; key: string; actype: string };

export type Episode = {
  seed: number;
  setSeed: (seed: number | ((s: number) => number)) => void;
  // The picked aircraft: its place in the drawn list and when it spawns. The
  // observation is sampled as it spawns; null samples the first to spawn.
  pick: Pick | null;
  setPick: (pick: Pick | null) => void;
};

export const EpisodeContext = createContext<Episode>({
  seed: 0,
  setSeed: () => {},
  pick: null,
  setPick: () => {},
});

export const useEpisode = () => useContext(EpisodeContext);

// One run per design, seed and time, shared by whoever asks: sampling runs
// the environment in a subprocess, which takes seconds.
const runs = new Map<string, Promise<SampleResult>>();

export function useEpisodeSample(spec: SpecDict | null, enabled = true) {
  const { seed, pick } = useEpisode();
  const refreshKey = useRefresh();
  const [state, setState] = useState<{ result: SampleResult | null; loading: boolean; error: string }>({
    result: null,
    loading: false,
    error: "",
  });
  const at_s = pick?.at_s ?? 0;
  const acid = pick?.acid ?? null;
  useEffect(() => {
    if (!spec || !enabled) return;
    let canceled = false;
    setState((s) => ({ ...s, loading: true, error: "" }));
    const handle = setTimeout(() => {
      const key = JSON.stringify([spec, seed, at_s, acid, refreshKey]);
      let run = runs.get(key);
      if (!run) {
        run = api.sample(spec, seed, at_s, acid);
        runs.set(key, run);
        run.catch(() => runs.delete(key));
      }
      run
        .then((result) => !canceled && setState({ result, loading: false, error: "" }))
        .catch((e) => !canceled && setState({ result: null, loading: false, error: String(e?.message ?? e) }));
    }, 500);
    return () => {
      canceled = true;
      clearTimeout(handle);
    };
  }, [spec, seed, at_s, acid, enabled, refreshKey]);
  return state;
}

// The episode's aircraft as the environment creates them - one run per design
// and seed, shared by whoever shows them, streamed as they are created. A run
// nobody watches any more is stopped; the last few finished ones are kept.
export type EpisodeSpawns = { aircraft: SpawnedAircraft[]; done: EpisodeRunDone | null };

type Run = {
  aircraft: SpawnedAircraft[];
  done: EpisodeRunDone | null;
  error: string;
  watchers: Set<() => void>;
  stop: AbortController;
};

const spawnRuns = new Map<string, Run>();
const KEEP_FINISHED = 6;

function startRun(key: string, spec: SpecDict, seed: number): Run {
  const run: Run = { aircraft: [], done: null, error: "", watchers: new Set(), stop: new AbortController() };
  spawnRuns.set(key, run);
  const tell = () => run.watchers.forEach((w) => w());
  api
    .episode(
      spec,
      seed,
      (a) => {
        run.aircraft = [...run.aircraft, a];
        tell();
      },
      run.stop.signal,
    )
    .then((done) => {
      run.done = done;
      tell();
      const finished = [...spawnRuns].filter(([, r]) => r.done);
      for (const [k] of finished.slice(0, Math.max(0, finished.length - KEEP_FINISHED))) spawnRuns.delete(k);
    })
    .catch((e) => {
      spawnRuns.delete(key);
      if (run.stop.signal.aborted) return;
      run.error = String(e?.message ?? e);
      tell();
    });
  return run;
}

export function useEpisodeSpawns(spec: SpecDict | null, enabled = true) {
  const { seed } = useEpisode();
  const refreshKey = useRefresh();
  const key = JSON.stringify([spec, seed, refreshKey]);
  const snapshot = (run: Run | undefined) => ({
    spawns: run && run.aircraft.length ? { aircraft: run.aircraft, done: run.done } : null,
    loading: Boolean(run && !run.done && !run.error),
    error: run?.error ?? "",
  });
  const [state, setState] = useState(() => snapshot(spawnRuns.get(key)));
  useEffect(() => {
    if (!spec || !enabled) return;
    let run: Run | undefined;
    const watch = () => setState(snapshot(run));
    const handle = setTimeout(
      () => {
        run = spawnRuns.get(key) ?? startRun(key, spec, seed);
        run.watchers.add(watch);
        watch();
      },
      spawnRuns.has(key) ? 0 : 250,
    );
    setState((s) => ({ ...snapshot(spawnRuns.get(key)), loading: true, error: s.error && "" }));
    return () => {
      clearTimeout(handle);
      if (!run) return;
      run.watchers.delete(watch);
      // Nobody is watching an unfinished run: stop it, and forget it.
      if (!run.watchers.size && !run.done) {
        run.stop.abort();
        spawnRuns.delete(key);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);
  return state;
}

export function useEpisodeState(): Episode {
  const [seed, setSeedRaw] = useState(0);
  const [pick, setPick] = useState<Episode["pick"]>(null);
  // A new seed draws new aircraft, so the pick no longer names one.
  const setSeed: Episode["setSeed"] = (next) => {
    setSeedRaw(next);
    setPick(null);
  };
  return { seed, setSeed, pick, setPick };
}
