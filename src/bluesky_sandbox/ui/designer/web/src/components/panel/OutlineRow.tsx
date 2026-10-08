// One compact outline row of the Geometry panel: kind tag + name, selection
// highlight, an optional warning, lock and eye. In a tree it is indented
// (`depth`), may lead with a caret, and may take drag-and-drop handlers.
import type { HTMLAttributes, ReactNode } from "react";
import { EyeToggle, LockToggle } from "./Section";

export function Row({
  name,
  kind,
  flag,
  selected,
  onClick,
  hidden,
  onToggleHidden,
  locked,
  onToggleLocked,
  depth,
  lead,
  note,
  dropping,
  dnd,
}: {
  name: string;
  kind: string;
  // A warning to show beside it: "unused".
  flag?: string;
  selected: boolean;
  onClick: () => void;
  hidden?: boolean;
  onToggleHidden?: () => void;
  locked?: boolean;
  onToggleLocked?: () => void;
  // How deep in a tree it sits.
  depth?: number;
  // Before the kind tag: a tree's caret.
  lead?: ReactNode;
  // After the name, muted: "in group 2", "via within".
  note?: ReactNode;
  // Something is being dragged over it that it would take.
  dropping?: boolean;
  dnd?: HTMLAttributes<HTMLDivElement>;
}) {
  const cls = ["geo-row", selected && "selected", dropping && "dropping", depth && "nested"].filter(Boolean).join(" ");
  return (
    <div
      className={cls}
      onClick={onClick}
      style={depth ? ({ "--depth": depth } as React.CSSProperties) : undefined}
      {...dnd}
    >
      {lead}
      <span className="geo-row-kind" data-kind={kind.split(/[\s·]/)[0]}>{kind}</span>
      <span className="geo-row-name" title={name}>{name}</span>
      {note && <span className="geo-row-note">{note}</span>}
      {flag && (
        <span className="geo-row-flag" title="Nothing uses this shape: no element, other shape, group's moves or code">
          <span aria-hidden="true">⚠</span> {flag}
        </span>
      )}
      <span className="spacer" />
      {onToggleLocked && <LockToggle locked={locked ?? false} onToggle={onToggleLocked} />}
      {onToggleHidden && <EyeToggle hidden={hidden ?? false} onToggle={onToggleHidden} />}
    </div>
  );
}
