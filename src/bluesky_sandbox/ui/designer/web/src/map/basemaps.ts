export type BasemapId = "osm" | "light" | "dark" | "satellite" | "terrain" | "buildings";

export type BasemapSpec = {
  id: BasemapId;
  label: string;
  description: string;
  style: any;
};

const GLYPHS = "https://fonts.openmaptiles.org/{fontstack}/{range}.pbf";

function rasterStyle(
  background: string,
  tiles: string[],
  attribution: string,
  opacity = 1,
): any {
  return {
    version: 8,
    glyphs: GLYPHS,
    sources: {
      base: {
        type: "raster",
        tiles,
        tileSize: 256,
        maxzoom: 19,
        attribution,
      },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": background } },
      { id: "base", type: "raster", source: "base", paint: { "raster-opacity": opacity } },
    ],
  };
}

export const BASEMAPS: BasemapSpec[] = [
  {
    id: "osm",
    label: "OpenStreetMap",
    description: "Standard street map.",
    style: rasterStyle(
      "#0b1021",
      ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      "© OpenStreetMap contributors",
      0.85,
    ),
  },
  {
    id: "light",
    label: "Light",
    description: "Pale street map for editing geometry.",
    style: rasterStyle(
      "#f4f1ea",
      ["https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"],
      "© OpenStreetMap contributors © CARTO",
    ),
  },
  {
    id: "dark",
    label: "Dark",
    description: "Dark street map with low visual weight.",
    style: rasterStyle(
      "#111827",
      ["https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"],
      "© OpenStreetMap contributors © CARTO",
    ),
  },
  {
    id: "satellite",
    label: "Satellite",
    description: "Imagery basemap.",
    style: rasterStyle(
      "#050505",
      ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
      "Tiles © Esri",
    ),
  },
  {
    id: "terrain",
    label: "Terrain",
    description: "Street map with terrain hillshade.",
    style: {
      version: 8,
      glyphs: GLYPHS,
      sources: {
        base: {
          type: "raster",
          tiles: ["https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png"],
          tileSize: 256,
          maxzoom: 19,
          attribution: "© OpenStreetMap contributors © CARTO",
        },
        terrain: {
          type: "raster-dem",
          tiles: ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"],
          tileSize: 256,
          encoding: "terrarium",
          attribution: "Elevation tiles © Mapzen",
        },
      },
      layers: [
        { id: "bg", type: "background", paint: { "background-color": "#d9ded6" } },
        { id: "hillshade", type: "hillshade", source: "terrain", paint: { "hillshade-shadow-color": "#6b7280", "hillshade-highlight-color": "#ffffff", "hillshade-accent-color": "#9ca3af" } },
        { id: "base", type: "raster", source: "base", paint: { "raster-opacity": 0.72 } },
      ],
    },
  },
  {
    id: "buildings",
    label: "Vector / buildings",
    description: "Public vector map style with richer urban/building detail.",
    style: "https://tiles.openfreemap.org/styles/liberty",
  },
];

export const DEFAULT_BASEMAP = "osm" satisfies BasemapId;

export function basemapById(id: string): BasemapSpec {
  return BASEMAPS.find((basemap) => basemap.id === id) ?? BASEMAPS[0];
}
