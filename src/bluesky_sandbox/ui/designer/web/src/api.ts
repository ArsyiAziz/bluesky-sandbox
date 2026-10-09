import { during, labelFor, trackedFetch } from "./busy";
import type { Intel, Problem } from "./code/intel";

// Typed client for the designer API. Paths are relative so the Vite dev proxy
// (and the static-served production build) both work without configuration.

// A test situation's aircraft, placed (see /api/spec/test/situations).
export type PlacedAircraft = { acid: string; lat: number; lon: number; track_deg: number; alt_ft: number };
export type TestSituations = { ok: boolean; errors?: string[]; situations?: Record<string, PlacedAircraft[]> };
// One result of a test run: a field checked against itself, or a case.
export type FieldCheckResult = { kind: "field"; field: string; ok: boolean; findings: string[]; notes?: string[] };
export type CaseRunResult = {
  kind: "case";
  index: number;
  ok: boolean;
  got: number | number[] | null;
  error: string | null;
  // What it saw on the way: the action given, the command, what it holds, the value read.
  saw?: string[];
};
// One of the design's own pytest tests (tests.files).
export type TestFileResult = {
  kind: "test";
  name: string;
  ok: boolean;
  skipped?: boolean;
  error: string | null;
  line?: number | null;
};
export type TestResult = FieldCheckResult | CaseRunResult | TestFileResult;

export type SpecDict = Record<string, any>;

export interface ValidateResult {
  ok: boolean;
  error?: string;
  // What builds but is worth knowing: content outside the airspace.
  warnings?: string[];
  summary?: {
    obs_fields: string[];
    intruder_obs_fields: string[] | null;
    critic_obs_fields: string[] | null;
    critic_intruder_obs_fields: string[] | null;
    state_fields: string[];
    intruder_state_fields: string[];
    action_fields: string[];
    // Every type the spawn regions name.
    aircraft_types: string[];
    max_aircraft: number;
    has_airspace: boolean;
    queryables: string[];
  };
}

export interface ShapeGeometry {
  vertices: [number, number][];
  bounding_box: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
  alt_min_ft?: number | null;
  alt_max_ft?: number | null;
  per_vertex_alt_ft?: [number, number][]; // [min,max] per vertex for varying bands
}

export interface PreviewResult {
  airspace: ShapeGeometry | null;
  queryables: any[];
  spawn_regions: (ShapeGeometry & {
    name: string;
    max_aircraft: number;
    render_shape?: boolean;
    render_name?: boolean;
    heading?: [number, number] | null;
  })[];
  // Named regions in the sampled episode's frame (shape draw + rotation), so
  // the map can render per-episode-randomized bounds; keyed by region name.
  // Named regions in this episode. A generated one carries `generated` (the
  // region it belongs to - itself, or for a partition's shape its partition)
  // and, on the region itself, `envelope`: where every draw lies.
  // A moving one carries `trail`: where it is over the next hour, a quarter apart.
  shapes?: Record<
    string,
    ShapeGeometry & { name: string; generated?: string; envelope?: ShapeGeometry; trail?: ShapeGeometry[] }
  >;
  sampled_aircraft: {
    lat: number;
    lon: number;
    alt_ft: number;
    // null: drawn from its flight envelope as it spawns.
    spd_kts: number | null;
    actype: string;
    spawn_time: number;
    // The source that planned it; absent or null for a spawn region's.
    source?: string | null;
    target?: { lat: number; lon: number; alt_ft: number | null } | null;
  }[];
  max_aircraft: number;
  seed: number;
  airspace_warnings?: string[];
}

export interface NavWaypoint {
  ident: string;
  lat_deg: number;
  lon_deg: number;
  wptype?: string;
  desc?: string;
}

export interface NavAirport {
  icao: string;
  lat_deg: number;
  lon_deg: number;
  name?: string;
  runways?: { name: string; lat_deg: number; lon_deg: number }[];
}

export interface NavAirwayLeg {
  awid: string;
  from_id: string;
  to_id: string;
  from_lat_deg: number;
  from_lon_deg: number;
  to_lat_deg: number;
  to_lon_deg: number;
}

export interface NavFeatures {
  window: [number, number, number, number];
  waypoints: NavWaypoint[];
  airports: (NavAirport & { runways: { name: string; lat_deg: number; lon_deg: number }[] })[];
  airways: NavAirwayLeg[];
}

export interface SearchResult {
  waypoints: NavWaypoint[];
  airports: NavAirport[];
}

export interface GenerateResult {
  package: string;
  files: Record<string, string>;
  // What the chosen template cannot do with this design (SB3's limits).
  notes?: { level: "error" | "warning" | "info"; message: string }[];
  // The server machine's cores, for choosing how many processes to train in.
  cpus?: number;
  // What clips can be recorded with: each driver's views, and the defaults.
  recording?: RecordingCatalog;
  // The windows evaluate.py can draw the trained policy in.
  evaluation?: { render_modes: string[]; default: string };
}

export interface RecordingCatalog {
  drivers: Record<string, { views: string[]; default: string[] }>;
  destinations: string[];
  defaults: {
    every: number;
    length: number;
    fps: number;
    driver: string;
    upload: string;
    folder: string;
  };
}


// A field in a sampled observation: its raw value (per intruder row in an
// intruder part), the columns the policy sees, and its range for this aircraft.
export interface SampleField {
  name: string;
  unit: string;
  // Steps back, for a lag of the field before it.
  lag: number | null;
  raw: any;
  obs: any;
  low: number | null;
  high: number | null;
}

export interface SampleAgent {
  acid: string;
  type: string;
  parts: Record<string, { fields: SampleField[]; acids?: string[] }>;
  action: { name: string; value: number[] }[];
}

// One field probed at one moment (see bluesky_sandbox.checks.probe).
export interface ProbeOverride {
  target: string; // "traf:<path>" or "call:<module>:<qualname>"
  value: number;
  aircraft?: string | null;
  leaf?: string;
}

export interface ProbeStage {
  name: string;
  actual: any;
  expected: any;
  note: string;
  agrees: boolean | null;
}

export interface ProbeNode {
  label: string;
  value: string;
  key: string | null;
  overridden: boolean;
  leaves: [string, string][];
  children: ProbeNode[];
}

export interface ProbeCell {
  steps: number;
  value: number | null;
  expected: number | null;
  placeholder: boolean;
  agrees: boolean | null;
}

export interface ProbeLag {
  inner: string;
  lags: number[];
  depth: number | null;
  frames: { back: number; value: number; read_by: number[]; placeholder: number[]; sim_time_s?: number }[];
  progression: { sim_time_s: number; cells: ProbeCell[] }[];
  first_s: number | null;
  start_s: number | null;
  fields: string[];
}

export interface ProbeCost {
  aircraft?: number;
  max_aircraft?: number;
  ms?: Record<string, number>;
  ms_at_max?: Record<string, number>;
  batched_path?: boolean;
}

export interface ProbeResultBody {
  field: string;
  kind: "observation" | "pair" | "action";
  aircraft: string;
  other: string | null;
  sim_time_s: number;
  stages: ProbeStage[];
  trace: ProbeNode[];
  unit: string;
  bounds: [number, number] | null;
  curve: [number, number][];
  choices: { choice: number; steps: number; value: number; on_grid: number; target: number | null }[];
  command: string[];
  lag: ProbeLag | null;
  cost: ProbeCost;
  notes: string[];
}

export interface ProbeResult {
  seed: number;
  sim_time_s: number;
  aircraft: string[];
  acid: string | null;
  result?: ProbeResultBody;
  error?: string;
}

export interface ProbeRequest {
  list: string;
  entry: number;
  part?: number;
  seed?: number;
  at_s?: number;
  acid?: string | null;
  other?: string | null;
  overrides?: ProbeOverride[];
  give?: number | null;
}

// Each field checked against itself and timed, and each action probed once,
// by spec entry (see runner.check_design_fields).
export interface FieldEntryCheck {
  entry: number;
  fields: string[];
  findings: string[];
  notes: string[];
  ms: number | null;
  batched: boolean;
}

export interface ActionEntryCheck {
  entry: number;
  field?: string;
  agrees?: boolean | null;
  actual?: number | null;
  expected?: number | null;
  command?: string[];
  error?: string;
}

export interface FieldsCheck {
  entries: Record<string, FieldEntryCheck[]>;
  actions: ActionEntryCheck[];
  episode_findings: string[];
  aircraft: number;
  step_ms: number | null;
  fields_ms: number | null;
}

export interface SampleResult {
  seed: number;
  sim_time_s: number;
  agents: SampleAgent[];
  // Every aircraft up at that time: [acid, type, low, high] by part and field.
  ranges: Record<string, Record<string, [string, string, number | null, number | null][]>>;
}

// One aircraft of the seeded episode as the environment created it, with its
// label in the pygame view.
export interface SpawnedAircraft {
  callsign: string;
  actype: string;
  time_s: number;
  lat_deg: number;
  lon_deg: number;
  alt_ft: number;
  hdg_deg: number;
  cas_kts: number;
  gs_kts: number;
  mach: number;
  controlled: boolean;
  region_index: number;
  route: string[] | null;
  label: string[];
  // The route's waypoints as resolved for this aircraft, with their constraints.
  targets: SpawnTarget[];
}

export interface SpawnTarget {
  waypoint: string | null;
  lat: number;
  lon: number;
  alt_ft: number | null;
  speed_kts: number | null;
  reach_radius_nm: number | null;
  alt_tolerance_ft: number | null;
  speed_tolerance_kts: number | null;
  speed_tolerance_mach: number | null;
  color: string | null;
}

// The run's summary, once it is over.
export interface EpisodeRunDone {
  seed: number;
  sim_time_s: number;
  complete: boolean;
  scheduled: number;
}

export interface RunResult {
  ok: boolean;
  pid: number;
  package: string;
  render_mode: string;
  workdir: string;
  log: string;
}

export interface RunStatus {
  active: boolean;
  alive: boolean;
  ready: boolean;
  pid?: number;
  render_mode?: string;
  returncode?: number | null;
  log?: string;
  error?: string;
}

export interface PythonMember {
  name: string;
  kind: "module" | "class" | "function" | "value" | string;
  detail?: string;
  doc?: string;
}


async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

let catalogCache: Promise<any> | null = null;

export const api = {
  health: () => fetch("/api/health").then((r) => jsonOrThrow<{ status: string }>(r)),

  catalog: () => trackedFetch("/api/catalog").then((r) => jsonOrThrow<any>(r)),
  designSchema: () => trackedFetch("/api/design/schema").then((r) => jsonOrThrow<any>(r)),

  validate: (spec: SpecDict) =>
    trackedFetch("/api/spec/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    }).then((r) => jsonOrThrow<ValidateResult>(r)),

  preview: (spec: SpecDict, seed = 0) =>
    trackedFetch("/api/spec/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, seed }),
    }).then((r) => jsonOrThrow<PreviewResult>(r)),

  navFeatures: (boundsSpec: SpecDict, airportLimit = 200, waypointLimit = 1500) =>
    trackedFetch("/api/nav/features", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bounds: boundsSpec,
        airport_limit: airportLimit,
        waypoint_limit: waypointLimit,
      }),
    }).then((r) => jsonOrThrow<NavFeatures>(r)),

  // A navdb fix by name: where it is (404, not found).
  navWaypoint: (ident: string) =>
    trackedFetch(`/api/nav/waypoint/${encodeURIComponent(ident)}`).then((r) => jsonOrThrow<NavWaypoint>(r)),

  // ``near``: [lat, lon] - each kind of match nearest it first.
  search: (q: string, limit = 20, near?: [number, number]) =>
    trackedFetch(
      `/api/nav/search?q=${encodeURIComponent(q)}&limit=${limit}` +
        (near ? `&lat=${near[0]}&lon=${near[1]}` : ""),
    ).then((r) =>
      jsonOrThrow<SearchResult>(r),
    ),

  generate: (spec: SpecDict, packageName: string) =>
    trackedFetch("/api/spec/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, package_name: packageName }),
    }).then((r) => jsonOrThrow<GenerateResult>(r)),

  run: (
    spec: SpecDict,
    renderMode: string,
    views: string[],
    showAllRoutes = false,
    autoTrack = false,
    seed = 0,
    actionMode: "random" | "zero" = "random",
  ) =>
    trackedFetch("/api/spec/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        spec,
        render_mode: renderMode,
        views,
        show_all_routes: showAllRoutes,
        auto_track: autoTrack,
        seed,
        action_mode: actionMode,
      }),
    }).then((r) => jsonOrThrow<RunResult>(r)),

  // The seeded episode run until its aircraft are all up, streamed: each
  // aircraft is handed over as the simulation creates it, then the summary.
  episode: async (
    spec: SpecDict,
    seed: number,
    onAircraft: (a: SpawnedAircraft) => void,
    signal: AbortSignal,
    untilS = 3600,
  ): Promise<EpisodeRunDone> =>
    during(labelFor("/api/spec/episode"), async () => {
    const r = await fetch("/api/spec/episode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, seed, until_s: untilS }),
      signal,
    });
    if (!r.ok || !r.body) await jsonOrThrow(r);
    const reader = r.body!.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      let nl: number;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        const item = JSON.parse(line);
        if (item.aircraft) onAircraft(item.aircraft);
        else if (item.done) return item.done as EpisodeRunDone;
        else if (item.error) throw new Error(item.error);
      }
    }
    throw new Error("the episode run ended early");
    }),

  // Where each test situation's aircraft are, for the Tests tab to draw.
  testSituations: (spec: SpecDict) =>
    trackedFetch("/api/spec/test/situations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    }).then((r) => jsonOrThrow<TestSituations>(r)),

  // The design's field checks, then its test cases, streamed: each result is
  // handed over as it is known.
  testDesign: async (spec: SpecDict, onResult: (r: TestResult) => void, signal: AbortSignal): Promise<void> =>
    during(labelFor("/api/spec/test"), async () => {
      const r = await fetch("/api/spec/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec }),
        signal,
      });
      if (!r.ok || !r.body) await jsonOrThrow(r);
      const reader = r.body!.pipeThrough(new TextDecoderStream()).getReader();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += value;
        let nl: number;
        while ((nl = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (!line) continue;
          const item = JSON.parse(line) as TestResult | { kind: "done" } | { kind: "error"; error: string };
          if (item.kind === "done") return;
          if (item.kind === "error") throw new Error(item.error);
          onResult(item);
        }
      }
      throw new Error("the test run ended early");
    }),

  sample: (spec: SpecDict, seed = 0, atS = 0, acid: string | null = null, maxAgents = 3, maxIntruders = 25) =>
    trackedFetch("/api/spec/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, seed, at_s: atS, acid, max_agents: maxAgents, max_intruders: maxIntruders }),
    }).then((r) => jsonOrThrow<SampleResult>(r)),

  checkFields: (spec: SpecDict) =>
    trackedFetch("/api/spec/check-fields", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    }).then((r) => jsonOrThrow<FieldsCheck>(r)),

  probe: (spec: SpecDict, request: ProbeRequest) =>
    trackedFetch("/api/spec/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, ...request }),
    }).then((r) => jsonOrThrow<ProbeResult>(r)),

  runStatus: () => fetch("/api/spec/run/status").then((r) => jsonOrThrow<RunStatus>(r)),

  runStop: () =>
    fetch("/api/spec/run/stop", { method: "POST" }).then((r) => jsonOrThrow<{ ok: boolean }>(r)),

  generateZip: (spec: SpecDict, packageName: string) =>
    trackedFetch("/api/spec/generate/zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, package_name: packageName }),
    }).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status}: ${r.statusText}`);
      return r.blob();
    }),

  catalogOnce: () => (catalogCache ??= trackedFetch("/api/catalog").then((r) => jsonOrThrow<any>(r))),

  // Drop what the backend and this page have cached, so the next reads are fresh.
  refresh: () => {
    catalogCache = null;
    return fetch("/api/refresh", { method: "POST" }).then((r) => jsonOrThrow<{ ok: boolean }>(r));
  },

  // A type the code editor reached, described one level down (its members'
  // types named, `partial`), by its key in the code intel.
  pythonType: (key: string) =>
    fetch(`/api/python/type?key=${encodeURIComponent(key)}`).then((r) =>
      jsonOrThrow<{ types: Record<string, any> }>(r),
    ),

  pythonModuleMembers: (moduleName: string) =>
    fetch(`/api/python/module-members?module=${encodeURIComponent(moduleName)}`).then((r) =>
      jsonOrThrow<{ module: string; members: PythonMember[] }>(r),
    ),

  // The design's MDP: its spaces field by field, and each normalizer's mapping.
  mdp: (spec: SpecDict) =>
    trackedFetch("/api/spec/mdp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    }).then((r) => jsonOrThrow<any>(r)),
  // Problems in the design's code, by block.
  diagnostics: (spec: SpecDict) =>
    trackedFetch("/api/spec/diagnostics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    }).then((r) => jsonOrThrow<{ ok: boolean; problems: Record<string, Problem[]> }>(r)),
  // What the design's code can use: types, scopes and design keys.
  codeIntel: (spec: SpecDict) =>
    trackedFetch("/api/spec/code-intel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    }).then((r) => jsonOrThrow<Intel | { ok: false; error: string }>(r)),

  listSpecs: () => trackedFetch("/api/specs").then((r) => jsonOrThrow<{ name: string; title: string }[]>(r)),

  getSpec: (name: string) =>
    trackedFetch(`/api/specs/${encodeURIComponent(name)}`).then((r) => jsonOrThrow<SpecDict>(r)),

  saveSpec: (name: string, spec: SpecDict) =>
    trackedFetch(`/api/specs/${encodeURIComponent(name)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    }).then((r) => jsonOrThrow<{ name: string }>(r)),

  deleteSpec: (name: string) =>
    trackedFetch(`/api/specs/${encodeURIComponent(name)}`, { method: "DELETE" }).then((r) =>
      jsonOrThrow<{ name: string }>(r),
    ),
};
