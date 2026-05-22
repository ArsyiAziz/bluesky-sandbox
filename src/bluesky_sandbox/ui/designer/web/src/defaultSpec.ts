import type { SpecDict } from "./api";

// Empty starting design: no airspace, no queryables, no spawn regions.
// Geometry is added explicitly from the Geometry tab.
export const DEFAULT_SPEC: SpecDict = {
  version: 1,
  nav_cycle: null,
  metadata: { name: "untitled" },
  // Airspace starts empty - add it from the Geometry tab ("+ set airspace").
  airspace: null,
  queryables: {},
  spawn: {
    type: "spawn_config",
    regions: [],
    aircraft_type: null,
    route: null,
    routes: {},
  },
  env: {
    obs_fields: [{ field: "LatDeg" }, { field: "LonDeg" }, { field: "AltFt" }],
    intruder_obs_fields: [{ field: "DistToOwnNm" }],
    action_fields: [{ field: "HdgDeg" }, { field: "SpdKts" }],
    task_info_setup: "",
    task_info: [],
    task_info_providers: [],
    // reward / terminated / truncated are always-present env hooks (edit them in
    // the Code tab's "env hooks" panel). context.query("goal") evaluates a
    // queryable for the current aircraft (QueryRegion -> .current.inside,
    // Waypoint -> .target/.current/.route/.step/.time).
    hook_setup: "",
    hooks: {
      reward: "# goal = context.query(\"goal\")\n# return 1.0 if goal.current.inside else -0.01\nreturn 0.0",
      terminated: "# return context.query(\"goal\").current.inside\nreturn False",
      truncated: "return False",
    },
    allowed_aircraft: ["A320", "B738"],
    dt: 1.0,
    simdt: 0.05,
    cd_method: "CSTATEBASED",
    performance_model: "openap",
    wind_dir_deg: 270.0,
    wind_kts: 0.0,
    turbulence_kts: 0.0,
    gust_tau_s: 30.0,
  },
  // Custom observation/action fields can live in e.g. custom_fields.py and be
  // referenced as "custom_fields:MyField".
  code: {},
};
