// A normalizer's color, wherever one is shown: the categorical slots in fixed
// order (validated on this surface), taken by the normalizer's place in the
// catalog so it keeps its color whatever the design uses. Past the slots, it is
// "other"; no normalizer is "raw".
const SLOTS = 8;

export function normalizerColor(catalogOrder: string[], name: string | null | undefined): string {
  if (!name) return "var(--norm-raw)";
  const slot = catalogOrder.indexOf(name);
  return slot >= 0 && slot < SLOTS ? `var(--norm-${slot + 1})` : "var(--norm-other)";
}
