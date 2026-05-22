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
}

export interface SampleAgent {
  acid: string;
  ownship: { name: string; value: number }[];
  intruder_fields: string[];
  intruders: number[][];
  n_intruders: number;
  action: { name: string; value: number }[];
}

export interface SampleResult {
  seed: number;
  agents: SampleAgent[];
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

export interface CompletionSymbol {
  name: string;
  kind: "module" | "class" | "function" | "value" | "variable" | "property" | "field" | string;
  detail?: string;
  doc?: string;
  insert?: string;
  color?: string;
  access?: "attribute" | "item" | string;
}

export interface CompletionContext {
  ok: boolean;
  error?: string;
  hook_setup?: { symbols: CompletionSymbol[]; imports: Record<string, string> };
  task_info_setup?: { symbols: CompletionSymbol[]; imports: Record<string, string> };
  queryables?: CompletionSymbol[];
  query_calls?: CompletionSymbol[];
  airspace_result_members?: CompletionSymbol[];
  airspace_result_nested_members?: Record<string, CompletionSymbol[]>;
  query_result_members?: Record<string, CompletionSymbol[]>;
  query_result_nested_members?: Record<string, Record<string, CompletionSymbol[]>>;
  queryable_members?: Record<string, CompletionSymbol[]>;
  obs_fields?: CompletionSymbol[];
  intruder_obs_fields?: CompletionSymbol[];
  action_fields?: CompletionSymbol[];
  hooks?: Record<string, { params: CompletionSymbol[]; members: Record<string, CompletionSymbol[]> }>;
  task_info?: { params: CompletionSymbol[]; members: Record<string, CompletionSymbol[]> };
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

  sample: (spec: SpecDict, seed = 0, maxAgents = 3, maxIntruders = 25) =>
    fetch("/api/spec/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec, seed, max_agents: maxAgents, max_intruders: maxIntruders }),
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

  catalogOnce: (() => {
    let cache: Promise<any> | null = null;
    return () => (cache ??= fetch("/api/catalog").then((r) => jsonOrThrow<any>(r)));
  })(),

  pythonModuleMembers: (moduleName: string) =>
    fetch(`/api/python/module-members?module=${encodeURIComponent(moduleName)}`).then((r) =>
      jsonOrThrow<{ module: string; members: PythonMember[] }>(r),
    ),

  completions: (spec: SpecDict) =>
    fetch("/api/spec/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spec }),
    }).then((r) => jsonOrThrow<CompletionContext>(r)),

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
