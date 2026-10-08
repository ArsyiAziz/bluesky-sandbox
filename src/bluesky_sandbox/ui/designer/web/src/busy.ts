// What the designer is waiting on: every API request, tracked once here, so
// the status bar can say what is loading - and a request quick enough not to
// notice (under SHOW_AFTER_MS) says nothing.
import { useSyncExternalStore } from "react";

const SHOW_AFTER_MS = 300;

// What each request is doing, by its path: in the words the status bar uses.
const LABELS: [RegExp, string][] = [
  [/^\/api\/catalog/, "Loading the aircraft types and fields…"],
  [/^\/api\/spec\/test\/situations/, "Placing the test situations…"],
  [/^\/api\/spec\/test/, "Running the design's tests…"],
  [/^\/api\/spec\/sample/, "Flying the sampled episode…"],
  [/^\/api\/spec\/episode/, "Flying the episode…"],
  [/^\/api\/spec\/validate/, "Building the design…"],
  [/^\/api\/spec\/preview/, "Drawing the design…"],
  [/^\/api\/spec\/generate/, "Generating the task…"],
  [/^\/api\/spec\/run/, "Starting the run…"],
  [/^\/api\/spec\/(mdp|code-intel|diagnostics)/, "Reading the design…"],
  [/^\/api\/specs/, "Opening the design…"],
  [/^\/api\/nav/, "Searching the navigation data…"],
];

type Job = { id: number; label: string; since: number; shown: boolean };

let jobs: Job[] = [];
let nextId = 1;
const listeners = new Set<() => void>();
let snapshot: string | null = null;

const publish = () => {
  const shown = jobs.filter((j) => j.shown);
  // The longest-running first: what the wait is mostly for.
  const next = shown.length ? shown[0].label + (shown.length > 1 ? ` (+${shown.length - 1})` : "") : null;
  if (next !== snapshot) {
    snapshot = next;
    listeners.forEach((l) => l());
  }
};

export const labelFor = (path: string): string => LABELS.find(([re]) => re.test(path))?.[1] ?? "Loading…";

// Run ``work`` as something the designer waits on, labeled ``label``.
export async function during<T>(label: string, work: () => Promise<T>): Promise<T> {
  const job: Job = { id: nextId++, label, since: Date.now(), shown: false };
  jobs.push(job);
  const timer = setTimeout(() => {
    job.shown = true;
    publish();
  }, SHOW_AFTER_MS);
  try {
    return await work();
  } finally {
    clearTimeout(timer);
    jobs = jobs.filter((j) => j.id !== job.id);
    publish();
  }
}

// ``fetch``, tracked: labeled by its path.
export const trackedFetch = (url: string, init?: RequestInit): Promise<Response> =>
  during(labelFor(url), () => fetch(url, init));

// What the designer is waiting on now, or null.
export function useBusy(): string | null {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => snapshot,
  );
}
