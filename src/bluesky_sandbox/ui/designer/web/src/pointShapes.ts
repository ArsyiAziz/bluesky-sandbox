// The design's point bounds - a fix, a spawn position: a bounds with no area.
// Wherever an area is needed (the airspace, a query region, a region to place
// or drift within, a generator's region) a picker leaves them out; the
// builder refuses them there too (builder._check_points).
import { createContext, useContext } from "react";
import type { SpecDict } from "./api";

export const isPointShape = (bounds: SpecDict | null | undefined): boolean => bounds?.footprint?.type === "point";

export const pointShapesOf = (spec: SpecDict | null): Set<string> =>
  new Set(
    Object.entries(spec?.shapes ?? {})
      .filter(([, b]) => isPointShape(b as SpecDict))
      .map(([name]) => name),
  );

const PointShapes = createContext<Set<string>>(new Set());
export const PointShapesProvider = PointShapes.Provider;
export const usePointShapes = () => useContext(PointShapes);
