// The designer's form layout: a page of outlined cards, each a titled group
// of rows - the label in a fixed column, the control beside it, help under the
// control. Config, Metadata and the side panels share it, so a setting looks
// and lines up the same wherever it is.
import type { ReactNode } from "react";

export function Page({ title, intro, children }: { title: string; intro?: ReactNode; children: ReactNode }) {
  return (
    <div className="form-page">
      <header className="form-page-head">
        <h2>{title}</h2>
        {intro && <p>{intro}</p>}
      </header>
      <div className="form-page-body">{children}</div>
    </div>
  );
}

// A column of cards, stacked without gaps: a page of two keeps each card where
// it is put at any width, one column under the other when narrow.
export function FormColumn({ children }: { children: ReactNode }) {
  return <div className="form-column">{children}</div>;
}

export function FormCard({
  title,
  help,
  wide = false,
  className = "",
  children,
}: {
  title: string;
  help?: ReactNode;
  // Span the page's columns, for a card that needs the width.
  wide?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={`form-card${wide ? " wide" : ""} ${className}`}>
      <h3>{title}</h3>
      {children}
      {help && <p className="form-card-help">{help}</p>}
    </section>
  );
}

export function FormRow({
  label,
  help,
  title,
  children,
}: {
  label: ReactNode;
  help?: ReactNode;
  title?: string;
  children: ReactNode;
}) {
  return (
    <div className="form-row" title={title}>
      <span className="form-label">{label}</span>
      <div className="form-control">
        {children}
        {help && <div className="form-help">{help}</div>}
      </div>
    </div>
  );
}

// A number input that is empty while unset, showing the default it falls back to.
export function NumberInput({
  value,
  onChange,
  step,
  min,
  placeholder,
  unit,
}: {
  value: number | null | undefined;
  onChange: (v: number | null) => void;
  step?: number;
  min?: number;
  placeholder?: string;
  unit?: string;
}) {
  const input = (
    <input
      type="number"
      className="form-input"
      step={step}
      min={min}
      placeholder={placeholder}
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
    />
  );
  if (!unit) return input;
  return (
    <span className="form-unit">
      {input}
      <span>{unit}</span>
    </span>
  );
}
