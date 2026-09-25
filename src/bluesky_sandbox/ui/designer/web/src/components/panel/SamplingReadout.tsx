// The sampled episode's aircraft, in spawn order; picking one shows what it
// observes below. The seed is the map's reseed/reset, so both show one episode.
import { useEffect, useState } from "react";
import { api, type PreviewResult, type SpecDict } from "../../api";
import { useRefresh } from "../../refresh";
import { useEpisode, useEpisodeSpawns } from "../../episode";

export function SamplingReadout({
  spec,
  seed,
  onSeedChange,
}: {
  spec: SpecDict;
  seed: number;
  onSeedChange: (seed: number) => void;
}) {
  const [preview, setPreview] = useState<PreviewResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { pick, setPick } = useEpisode();
  const { spawns, loading: running, error: spawnError } = useEpisodeSpawns(spec);

  const refreshKey = useRefresh();
  useEffect(() => {
    let canceled = false;
    setLoading(true);
    api
      .preview(spec, seed)
      .then((p) => !canceled && (setPreview(p), setError(null)))
      .catch((e) => !canceled && setError(String(e)))
      .finally(() => !canceled && setLoading(false));
    return () => {
      canceled = true;
    };
  }, [spec, seed, refreshKey]);

  // The episode's aircraft as the environment created them, once its run is
  // in; until then, the preview's draw of them, which names none.
  type Row = { key: string; at_s: number; acid: string | null; actype: string; alt_ft: number; speed: string };
  const rows: Row[] = spawns
    ? spawns.aircraft.map((a) => ({
        key: a.callsign,
        at_s: a.time_s,
        acid: a.callsign,
        actype: a.actype,
        alt_ft: a.alt_ft,
        speed: a.label[1]?.split(/\s+/).slice(1).join(" ") ?? "",
      }))
    : (preview?.sampled_aircraft ?? [])
        .map((a, i) => ({
          key: `preview:${i}`,
          at_s: a.spawn_time,
          acid: null,
          actype: a.actype,
          alt_ft: a.alt_ft,
          speed: `${Math.round(a.spd_kts)} kt`,
        }))
        .sort((x, y) => x.at_s - y.at_s);
  // A pick made from the preview's rows carries over to the flown aircraft of
  // its type spawning then (a spawn can be put off, never brought forward).
  const pickedRow = (key: string | undefined) => rows.find((r) => r.key === key);
  const picked = !pick
    ? rows[0]?.key
    : rows.some((r) => r.key === pick.key)
      ? pick.key
      : rows.find((r) => r.actype === pick.actype && r.at_s >= pick.at_s - 1)?.key;
  // Once the carried-over row names its aircraft, sample that very aircraft.
  const carried = pick && picked !== pick.key ? pickedRow(picked) : undefined;
  useEffect(() => {
    if (carried?.acid) setPick({ key: carried.key, at_s: carried.at_s, acid: carried.acid, actype: carried.actype });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [carried?.key]);

  if (error) return <div className="error-text">{error}</div>;
  if (!preview) return <p className="muted">…</p>;

  return (
    <div className={loading ? "sampling loading" : "sampling"}>
      <div className="sampling-head">
        <span>
          {rows.length} aircraft <span className="muted">· max {preview.max_aircraft} at once</span>
        </span>
        <span className="sampling-seed">
          <span className="muted">seed {seed}</span>
          <button disabled={loading} onClick={() => onSeedChange(seed + 1)} title="draw another episode">
            reseed
          </button>
          <button disabled={loading || seed === 0} onClick={() => onSeedChange(0)} title="back to seed 0">
            reset
          </button>
        </span>
      </div>
      {!spawns && (
        <div className="muted small sampling-note">
          {running ? "Running the episode for callsigns, headings and speeds as flown…" : spawnError}
        </div>
      )}
      <div className="sampling-list" role="listbox" aria-label="aircraft in this episode">
        <div className="sampling-row sampling-row-h">
          <span>spawns</span>
          <span>aircraft</span>
          <span className="num">FL</span>
          <span className="num">speed</span>
        </div>
        {rows.map((r) => (
          <div
            key={r.key}
            role="option"
            aria-selected={r.key === picked}
            className={r.key === picked ? "sampling-row on" : "sampling-row"}
            onClick={() => setPick({ key: r.key, at_s: r.at_s, acid: r.acid, actype: r.actype })}
            title="see what this aircraft observes as it spawns"
          >
            <span>t+{Math.round(r.at_s)} s</span>
            <span className="sampling-ac">
              {r.acid && <b>{r.acid}</b>} {r.actype}
            </span>
            <span className="num">{String(Math.round(r.alt_ft / 100)).padStart(3, "0")}</span>
            <span className="num">{r.speed}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
