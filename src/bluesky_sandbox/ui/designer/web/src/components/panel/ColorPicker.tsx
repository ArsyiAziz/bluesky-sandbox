// Colour selector for design elements (queryables, …). The renderers accept a
// palette name ("red") or a "#rrggbb" literal, so this offers named swatches
// plus a native custom-colour input. The palette comes from the catalog.
import { useEffect, useState } from "react";
import { api } from "../../api";

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

  useEffect(() => {
    api.catalogOnce().then((c) => setPalette(c?.colors ?? {})).catch(() => setPalette({}));
  }, []);

  const isSelected = (name: string, hex: string) =>
    value === name || (value?.startsWith("#") && value.toLowerCase() === hex.toLowerCase());

  return (
    <div className="color-picker">
      {Object.entries(palette).map(([name, hex]) => (
        <button
          key={name}
          type="button"
          className={isSelected(name, hex) ? "swatch on" : "swatch"}
          style={{ background: hex }}
          title={name}
          onClick={() => onChange(name)}
        />
      ))}
      <label className="swatch custom" title="custom colour">
        <input
          type="color"
          value={toHex(value, palette)}
          onChange={(e) => onChange(e.target.value)}
        />
      </label>
    </div>
  );
}
