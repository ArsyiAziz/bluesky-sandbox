// Shared map types: edit targets/handles and the intermediate geometry shapes
// (edges/faces) that the geometry builders emit and the deck layers consume.
import type { SpecDict } from "../api";

export type RGBA = [number, number, number, number];

export type EditTarget =
  | { scope: "airspace" }
  | { scope: "queryable"; name: string }
  | { scope: "spawn"; index: number }
  | { scope: "region"; name: string }
  // A transform group: a named set of bounds moved/rotated together (and
  // randomised per episode). Selected from the panel, dragged on the map.
  | { scope: "group"; id: string };

export type Selectable = { target?: EditTarget; name?: string };
export type Edge = { src: number[]; tgt: number[]; color: RGBA } & Selectable;
export type Face = { polygon: number[][]; color: RGBA } & Selectable;

export type WaypointShape = {
  name?: string;
  target?: EditTarget;
  ident?: string;
  alt_ft?: number;
  speed_kts?: number;
  reach_radius_nm?: number;
  alt_tolerance_ft?: number;
  speed_tolerance_kts?: number;
};
export type WaypointFace = Face & WaypointShape;
export type WaypointEdge = Edge & WaypointShape;

export type EditHandle = {
  id: string;
  name: string;
  lat: number;
  lon: number;
  z?: number;
  target: EditTarget;
  role: "waypoint" | "box-corner" | "polygon-vertex" | "center" | "move-shape" | "rotate";
  index?: number;
  color: RGBA;
};

export type DragState = {
  handle: EditHandle;
  startSpec: SpecDict;
  startY: number;
  rotationCenter?: [number, number] | null;
  rotationDeg?: number;
};

// A resolved route ready to draw: ordered waypoint positions + identity/colour.
export type RoutePath = {
  key: string;
  name: string;
  color: RGBA;
  points: [number, number, number][];
};

// View-only visibility: which element categories are shown, and which individual
// elements have been hidden (by element key) from the map without touching the spec.
export type CategoryVisibility = {
  airspace: boolean;
  regions: boolean;
  waypoints: boolean;
  routes: boolean;
  spawnRegions: boolean;
  aircraft: boolean;
  nav: boolean;
  airways: boolean;
  labels: boolean;
};
