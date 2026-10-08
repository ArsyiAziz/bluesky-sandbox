// A small spinning ring: something is loading. Sized to the text it sits in.
export function Spinner({ label }: { label?: string }) {
  return <span className="spinner" role="progressbar" aria-label={label ?? "loading"} />;
}
