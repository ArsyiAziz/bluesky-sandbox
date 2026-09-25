// The sampled episode every view shares: the seed the map's reseed draws it
// with, and the aircraft picked in it. The map, the Sampling panel and the
// Spaces tab all read one episode, so what one shows the others agree with.
import { createContext, useContext, useEffect, useState } from "react";
import { api, type EpisodeSpawns, type SampleResult, type SpecDict } from "./api";
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

// The episode's aircraft as the environment creates them - run once per
// design and seed, and shared: it runs the whole episode, which takes seconds.
const spawnRuns = new Map<string, Promise<EpisodeSpawns>>();
const spawnResults = new Map<string, EpisodeSpawns>();

export function useEpisodeSpawns(spec: SpecDict | null, enabled = true) {
  const { seed } = useEpisode();
  const refreshKey = useRefresh();
  const key = JSON.stringify([spec, seed, refreshKey]);
  const [state, setState] = useState<{ spawns: EpisodeSpawns | null; loading: boolean; error: string }>(() => ({
    spawns: spawnResults.get(key) ?? null,
    loading: false,
    error: "",
  }));
  useEffect(() => {
    if (!spec || !enabled) return;
    let canceled = false;
    // A run already in: no wait, no request.
    const done = spawnResults.get(key);
    if (done) {
      setState({ spawns: done, loading: false, error: "" });
      return;
    }
    // Keep the last run on screen while the next one comes.
    setState((s) => ({ ...s, loading: true, error: "" }));
    const handle = setTimeout(() => {
      let run = spawnRuns.get(key);
      if (!run) {
        run = api.episode(spec, seed);
        spawnRuns.set(key, run);
        run.then((r) => spawnResults.set(key, r)).catch(() => spawnRuns.delete(key));
      }
      run
        .then((spawns) => !canceled && setState({ spawns, loading: false, error: "" }))
        .catch((e) => !canceled && setState({ spawns: null, loading: false, error: String(e?.message ?? e) }));
    }, 800);
    return () => {
      canceled = true;
      clearTimeout(handle);
    };
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
