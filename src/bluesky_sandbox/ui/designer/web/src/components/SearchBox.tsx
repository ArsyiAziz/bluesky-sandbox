import { useEffect, useRef, useState } from "react";
import { api, type SearchResult } from "../api";

// Global navdb feature search: fly the map to a fix/airport, or add a fix as a
// waypoint queryable. Debounced; results rank prefix hits first (backend).
export default function SearchBox({
  onFlyTo,
  onAddWaypoint,
}: {
  onFlyTo: (lon: number, lat: number) => void;
  onAddWaypoint: (ident: string) => void;
}) {
  const [q, setQ] = useState("");
  const [res, setRes] = useState<SearchResult | null>(null);
  const [open, setOpen] = useState(false);
  const timer = useRef<number | undefined>(undefined);

  useEffect(() => {
    window.clearTimeout(timer.current);
    if (q.trim().length < 2) {
      setRes(null);
      return;
    }
    timer.current = window.setTimeout(() => {
      api
        .search(q, 12)
        .then((r) => {
          setRes(r);
          setOpen(true);
        })
        .catch(() => setRes(null));
    }, 250);
    return () => window.clearTimeout(timer.current);
  }, [q]);

  const hasResults = res && (res.waypoints.length > 0 || res.airports.length > 0);

  return (
    <div className="search-box">
      <input
        placeholder="search fixes / airports…"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onFocus={() => setOpen(true)}
      />
      {open && hasResults && (
        <ul className="search-results" onMouseLeave={() => setOpen(false)}>
          {res!.airports.map((a) => (
            <li key={`a-${a.icao}`}>
              <button className="result" onClick={() => onFlyTo(a.lon_deg, a.lat_deg)}>
                <span className="tag apt">APT</span> {a.icao}
                {a.name ? ` — ${a.name}` : ""}
              </button>
            </li>
          ))}
          {res!.waypoints.map((w) => (
            <li key={`w-${w.ident}`}>
              <button className="result" onClick={() => onFlyTo(w.lon_deg, w.lat_deg)}>
                <span className="tag wpt">FIX</span> {w.ident}
              </button>
              <button className="add" title="add as waypoint queryable" onClick={() => onAddWaypoint(w.ident)}>
                +
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
