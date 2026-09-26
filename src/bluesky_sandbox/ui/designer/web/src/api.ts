import type { Intel, Problem } from "./code/intel";

// Typed client for the designer API. Paths are relative so the Vite dev proxy
// (and the static-served production build) both work without configuration.

export type SpecDict = Record<string, any>;

export interface ValidateResult {
  ok: boolean;
  error?: string;
  summary?: {
    obs_fields: string[];
    intruder_obs_fields: string[] | null;
    critic_obs_fields: string[] | null;
    critic_intruder_obs_fields: string[] | null;
    action_fields: string[];
    allowed_aircraft: string[];
    max_aircraft: number;
    has_airspace: boolean;
    queryables: string[];
  };
}

export interface BoundsGeometry {
  vertices: [number, number][];
  bounding_box: { lat_min: number; lat_max: number; lon_min: number; lon_max: number };
  alt_min_ft?: number | null;
  alt_max_ft?: number | null;
  per_vertex_alt_ft?: [number, number][]; // [min,max] per vertex for varying bands
}

export interface PreviewResult {
  airspace: BoundsGeometry | null;
  queryables: any[];
  spawn_regions: (BoundsGeometry & {
    name: string;
    max_aircraft: number;
    render_shape?: boolean;
    render_name?: boolean;
    heading?: [number, number] | null;
  })[];
  // Named regions in the sampled episode's frame (shape draw + rotation), so
  // the map can render per-episode-randomized bounds; keyed by region name.
  regions?: Record<string, BoundsGeometry & { name: string }>;
  sampled_aircraft: {
    lat: number;
    lon: number;
    alt_ft: number;
    spd_kts: number;
    actype: string;
    spawn_time: number;
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
}

export interface RecordingCatalog {
  drivers: Record<string, { views: string[]; default: string[] }>;
  defaults: { every: number; length: number; fps: number; driver: string };
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

  catalog: () => fetch("/api/catalog").then((r) => jsonOrThrow<any>(r)),

  validate: (spec: SpecDict) =>
    fetch("/api/spec/validate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    }).then((r) => jsonOrThrow<ValidateResult>(r)),

  preview: (spec: SpecDict, seed = 0) =>
    fetch("/api/spec/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, seed }),
    }).then((r) => jsonOrThrow<PreviewResult>(r)),

  navFeatures: (boundsSpec: SpecDict, airportLimit = 200, waypointLimit = 1500) =>
    fetch("/api/nav/features", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        bounds: boundsSpec,
        airport_limit: airportLimit,
        waypoint_limit: waypointLimit,
      }),
    }).then((r) => jsonOrThrow<NavFeatures>(r)),

  search: (q: string, limit = 20) =>
    fetch(`/api/nav/search?q=${encodeURIComponent(q)}&limit=${limit}`).then((r) =>
      jsonOrThrow<SearchResult>(r),
    ),

  generate: (spec: SpecDict, packageName: string) =>
    fetch("/api/spec/generate", {
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
    fetch("/api/spec/run", {
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
  ): Promise<EpisodeRunDone> => {
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
  },

  sample: (spec: SpecDict, seed = 0, atS = 0, acid: string | null = null, maxAgents = 3, maxIntruders = 25) =>
    fetch("/api/spec/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, seed, at_s: atS, acid, max_agents: maxAgents, max_intruders: maxIntruders }),
    }).then((r) => jsonOrThrow<SampleResult>(r)),

  runStatus: () => fetch("/api/spec/run/status").then((r) => jsonOrThrow<RunStatus>(r)),

  runStop: () =>
    fetch("/api/spec/run/stop", { method: "POST" }).then((r) => jsonOrThrow<{ ok: boolean }>(r)),

  generateZip: (spec: SpecDict, packageName: string) =>
    fetch("/api/spec/generate/zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, package_name: packageName }),
    }).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status}: ${r.statusText}`);
      return r.blob();
    }),

  catalogOnce: () => (catalogCache ??= fetch("/api/catalog").then((r) => jsonOrThrow<any>(r))),

  // Drop what the backend and this page have cached, so the next reads are fresh.
  refresh: () => {
    catalogCache = null;
    return fetch("/api/refresh", { method: "POST" }).then((r) => jsonOrThrow<{ ok: boolean }>(r));
  },

  pythonModuleMembers: (moduleName: string) =>
    fetch(`/api/python/module-members?module=${encodeURIComponent(moduleName)}`).then((r) =>
      jsonOrThrow<{ module: string; members: PythonMember[] }>(r),
    ),

  // The design's MDP: its spaces field by field, and each normalizer's mapping.
  mdp: (spec: SpecDict) =>
    fetch("/api/spec/mdp", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    }).then((r) => jsonOrThrow<any>(r)),
  // Problems in the design's code, by block.
  diagnostics: (spec: SpecDict) =>
    fetch("/api/spec/diagnostics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    }).then((r) => jsonOrThrow<{ ok: boolean; problems: Record<string, Problem[]> }>(r)),
  // What the design's code can use: types, scopes and design keys.
  codeIntel: (spec: SpecDict) =>
    fetch("/api/spec/code-intel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    }).then((r) => jsonOrThrow<Intel | { ok: false; error: string }>(r)),

  listSpecs: () => fetch("/api/specs").then((r) => jsonOrThrow<{ name: string; title: string }[]>(r)),

  getSpec: (name: string) =>
    fetch(`/api/specs/${encodeURIComponent(name)}`).then((r) => jsonOrThrow<SpecDict>(r)),

  saveSpec: (name: string, spec: SpecDict) =>
    fetch(`/api/specs/${encodeURIComponent(name)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    }).then((r) => jsonOrThrow<{ name: string }>(r)),

  deleteSpec: (name: string) =>
    fetch(`/api/specs/${encodeURIComponent(name)}`, { method: "DELETE" }).then((r) =>
      jsonOrThrow<{ name: string }>(r),
    ),
};
