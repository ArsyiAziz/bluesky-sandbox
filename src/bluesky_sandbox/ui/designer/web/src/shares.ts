// Weights as shares: each weight over the sum, as a whole percentage. One
// rule for every weighted mix - aircraft types, route branches, a spawn
// region's routes - so each says how often it is drawn the same way.
export function shares(weights: number[]): number[] {
  const total = weights.reduce((sum, w) => sum + (Number.isFinite(w) && w > 0 ? w : 0), 0);
  return weights.map((w) => (total > 0 && w > 0 ? Math.round((w / total) * 100) : 0));
}
