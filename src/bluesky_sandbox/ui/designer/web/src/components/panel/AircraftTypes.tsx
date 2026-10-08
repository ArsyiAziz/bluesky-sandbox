// The aircraft types a spawn region (or source) flies: one type, or a weighted
// mix - each shown with what it is (fixed-wing, helicopter, drone; propulsion,
// wake, mass), from the performance model's own data (catalog.aircraft).
import { shares } from "../../shares";
import { useEffect, useState } from "react";
import { api, type SpecDict } from "../../api";
import { useRefresh } from "../../refresh";
import { Picker } from "./Picker";
import { Spinner } from "../Spinner";

export type AircraftOption = { type: string; name?: string | null; tags: string[] };

// The model's types, or why there are none (its database is not installed).
export function useAircraftTypes(model: string): { types: AircraftOption[]; error: string | null; loading: boolean } {
  // undefined: still loading; null: it failed.
  const [catalog, setCatalog] = useState<any>(undefined);
  const refreshKey = useRefresh();
  useEffect(() => {
    api.catalogOnce().then(setCatalog).catch(() => setCatalog(null));
  }, [refreshKey]);
  const avail = catalog?.aircraft?.[model];
  if (avail && !Array.isArray(avail)) return { types: [], error: avail.error ?? "unavailable", loading: false };
  if (catalog === null) return { types: [], error: "the designer could not load its catalog", loading: false };
  return { types: Array.isArray(avail) ? avail : [], error: null, loading: catalog === undefined };
}

// A region's value - "B744", or {type: "categorical", weights} - as weights.
function weightsOf(value: SpecDict | string | null | undefined): Record<string, number> {
  if (!value) return {};
  if (typeof value === "string") return { [value.toUpperCase()]: 1 };
  if (value.type === "categorical") return { ...(value.weights ?? {}) };
  return {};
}

// Weights back as the value: one type as its code, several as a categorical.
function valueOf(weights: Record<string, number>): SpecDict | string | null {
  const types = Object.keys(weights);
  if (types.length === 0) return null;
  if (types.length === 1) return types[0];
  return { type: "categorical", weights };
}

export function TypeTags({ tags }: { tags: string[] }) {
  return (
    <span className="type-tags">
      {tags.map((t) => (
        <span className="type-tag" key={t}>{t}</span>
      ))}
    </span>
  );
}

export function AircraftTypes({
  value,
  model,
  onChange,
  emptyText = "names no type: pick the types it spawns",
}: {
  value: SpecDict | string | null | undefined;
  model: string;
  onChange: (value: SpecDict | string | null) => void;
  emptyText?: string;
}) {
  const { types, error, loading } = useAircraftTypes(model);
  const byType = new Map(types.map((t) => [t.type, t]));
  const weights = weightsOf(value);
  const chosen = Object.keys(weights);
  // Each type's share of the region's aircraft: its weight over the sum.
  const pct = shares(chosen.map((t) => weights[t] ?? 0));
  const share = (t: string) => pct[chosen.indexOf(t)];
  const set = (next: Record<string, number>) => onChange(valueOf(next));
  return (
    <div className="aircraft-types">
      {chosen.length === 0 && <div className="error-text small">{emptyText}</div>}
      {chosen.map((t) => {
        const info = byType.get(t);
        return (
          <div className="aircraft-type-row" key={t} title={info?.name ?? undefined}>
            <span className="aircraft-type-what">
              <span className={info ? "type-code" : "type-code missing"}>{t}</span>
              {info ? <TypeTags tags={info.tags} /> : <span className="error-text small">not in {model}</span>}
            </span>
            {chosen.length > 1 && (
              <label className="type-share" title="its weight: shares are each weight over the sum, so 3, 1, 1 is 60 %, 20 %, 20 %">
                <input
                  type="number"
                  min={0}
                  step={1}
                  value={weights[t]}
                  onChange={(e) => set({ ...weights, [t]: Math.max(0, Number(e.target.value) || 0) })}
                />
                <span className="type-share-pct">{share(t)} %</span>
              </label>
            )}
            <button
              className="chip-x"
              title="remove"
              onClick={() => {
                const next = { ...weights };
                delete next[t];
                set(next);
              }}
            >
              ✕
            </button>
          </div>
        );
      })}
      {error ? (
        <div className="error-text small">{model}: {error}</div>
      ) : loading ? (
        <div className="muted small"><Spinner /> loading {model}'s aircraft types…</div>
      ) : (
        <Picker
          placeholder={chosen.length ? "+ add a type to the mix…" : "+ aircraft type…"}
          onChange={(v) => v && set({ ...weights, [v]: 1 })}
          options={types
            .filter((t) => !(t.type in weights))
            .map((t) => ({
              value: t.type,
              label: t.type,
              description: [t.name, ...t.tags.slice(1)].filter(Boolean).join(" · "),
              category: t.tags[0] ?? "other",
            }))}
        />
      )}
    </div>
  );
}
