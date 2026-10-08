// Color selector for design elements (queryables, …). The renderers accept a
// palette name ("cyan") or a "#rrggbb" literal, so this offers named swatches
// plus a native custom-color input. The palette comes from the catalog - as the
// drivers draw it; names near an alert's color (red, orange, purple) are drawn
// in another, so they are not offered.
import { useEffect, useState } from "react";
import { api } from "../../api";
import { useRefresh } from "../../refresh";

const FALLBACK = "#888888";

function toHex(value: string | undefined, palette: Record<string, string>): string {
  if (!value) return FALLBACK;
  if (value.startsWith("#")) return value;
  return palette[value.toLowerCase()] ?? FALLBACK;
}

export function ColorPicker({
  value,
  onChange,
}: {
  value: string | undefined;
  onChange: (color: string) => void;
}) {
  const [palette, setPalette] = useState<Record<string, string>>({});
  const [hidden, setHidden] = useState<string[]>([]);

  const refreshKey = useRefresh();
  useEffect(() => {
    api
      .catalogOnce()
      .then((c) => {
        setPalette(c?.colors ?? {});
        setHidden(c?.alert_colors ?? []);
      })
      .catch(() => setPalette({}));
  }, [refreshKey]);

  const isSelected = (name: string, hex: string) =>
    value === name || (value?.startsWith("#") && value.toLowerCase() === hex.toLowerCase());

  return (
    <div className="color-picker">
      {Object.entries(palette).filter(([name]) => !hidden.includes(name)).map(([name, hex]) => (
        <button
          key={name}
          type="button"
          className={isSelected(name, hex) ? "swatch on" : "swatch"}
          style={{ background: hex }}
          title={name}
          onClick={() => onChange(name)}
        />
      ))}
      <label className="swatch custom" title="custom color">
        <input
          type="color"
          value={toHex(value, palette)}
          onChange={(e) => onChange(e.target.value)}
        />
      </label>
    </div>
  );
}
