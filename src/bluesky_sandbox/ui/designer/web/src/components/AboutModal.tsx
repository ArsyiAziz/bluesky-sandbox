// About: what the designer is, who made it, and whose work it stands on.
// Partner logos are read from the repository's assets/partners at build time
// (see its README); a partner without one is shown by name.
import { useEffect, useRef, useState } from "react";
import lockupOnDark from "@brand/bluesky-sandbox-lockup-on-dark.svg";
import lockupOnLight from "@brand/bluesky-sandbox-lockup.svg";
import { api } from "../api";
import { useTheme } from "../theme";

const LOGOS = import.meta.glob("@partners/*.{svg,png}", {
  eager: true,
  query: "?url",
  import: "default",
}) as Record<string, string>;

// The logo for `key` - its dark-background version in the dark theme, when
// there is one.
function logo(key: string, dark: boolean): string | undefined {
  const find = (name: string) =>
    Object.entries(LOGOS).find(([path]) => /[^/]+$/.exec(path)?.[0].replace(/\.(svg|png)$/, "") === name)?.[1];
  return (dark && find(`${key}-on-dark`)) || find(key);
}

// A group that developed the designer: the unit, within its university. Each
// shows whichever logos assets/partners has - the unit's, its university's.
type Partner = {
  unit: string;
  university: string;
  // Each logo's height, px: set so the logos weigh about the same - a wide one
  // shorter than a square one - in a row of one height.
  logos: { key: string; alt: string; height: number }[];
  href?: string;
};

const DEVELOPED_AT: Partner[] = [
  {
    unit: "Intelligent Aerospace Systems Lab (IASL)",
    university: "The George Washington University",
    logos: [
      { key: "iasl", alt: "Intelligent Aerospace Systems Lab", height: 52 },
      { key: "gwu", alt: "The George Washington University", height: 52 },
    ],
    href: "https://www.gwu.edu",
  },
  {
    unit: "Operations and Environment Section",
    university: "Delft University of Technology (TU Delft)",
    logos: [{ key: "tudelft", alt: "Delft University of Technology", height: 38 }],
    href: "https://www.tudelft.nl",
  },
];

function PartnerCard({ partner, dark }: { partner: Partner; dark: boolean }) {
  const logos = partner.logos
    .map((l) => ({ ...l, src: logo(l.key, dark) }))
    .filter((l): l is { key: string; alt: string; height: number; src: string } => !!l.src);
  const body = (
    <>
      {logos.length > 0 && (
        <span className="about-partner-logos">
          {logos.map((l) => (
            <img key={l.key} className="about-partner-logo" src={l.src} alt={l.alt} style={{ height: l.height }} />
          ))}
        </span>
      )}
      <span className="about-partner-name">
        <strong>{partner.unit}</strong>
        <span>{partner.university}</span>
      </span>
    </>
  );
  return partner.href ? (
    <a className="about-partner" href={partner.href} target="_blank" rel="noreferrer">
      {body}
    </a>
  ) : (
    <span className="about-partner">{body}</span>
  );
}

export default function AboutModal({ onClose }: { onClose: () => void }) {
  const { theme } = useTheme();
  const dark = theme === "dark";
  const [version, setVersion] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    api
      .health()
      .then((h: any) => setVersion(h?.version ?? null))
      .catch(() => undefined);
  }, []);

  // A dialog: focus moves into it, Tab stays in it, Escape closes it, and
  // focus goes back to whatever opened it.
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>("a[href], button:not(:disabled)"),
      );
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      opener?.focus?.();
    };
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        ref={dialogRef}
        className="modal about-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="modal-head">
          <h2 id="about-title" className="modal-title">
            About
          </h2>
          <button ref={closeRef} onClick={onClose}>
            Close
          </button>
        </header>
        <div className="about-body">
          <img className="about-lockup" src={dark ? lockupOnDark : lockupOnLight} alt="BlueSky Sandbox" />
          <p className="about-lead">
            Environment Designer{version ? <span className="about-version"> · version {version}</span> : null}
          </p>
          <p>Enabling rapid, modular prototyping for air traffic control research.</p>

          <section className="about-section" aria-labelledby="about-developed">
            <h3 id="about-developed">Development</h3>
            <p>
              Developed in the <strong>Intelligent Aerospace Systems Lab (IASL)</strong> at{" "}
              <strong>The George Washington University (GWU)</strong>, in collaboration with the{" "}
              <strong>Operations and Environment Section</strong> at{" "}
              <strong>Delft University of Technology (TU Delft)</strong>.
            </p>
            <div className="about-partners">
              {DEVELOPED_AT.map((p) => (
                <PartnerCard key={p.unit} partner={p} dark={dark} />
              ))}
            </div>
          </section>

          <section className="about-section" aria-labelledby="about-thanks">
            <h3 id="about-thanks">Acknowledgments</h3>
            <p>
              BlueSky Sandbox runs on{" "}
              <a href="https://github.com/TUDelft-CNS-ATM/bluesky" target="_blank" rel="noreferrer">
                BlueSky
              </a>
              , the open-source air traffic simulator developed at <strong>TU Delft</strong>, and is a sister
              project to{" "}
              <a href="https://github.com/TUDelft-CNS-ATM/bluesky-gym" target="_blank" rel="noreferrer">
                BlueSky-Gym
              </a>
              . We thank their authors and the continuous support from the community for making them open.
            </p>
          </section>

          <p className="about-foot">
            <a href="https://github.com/ArsyiAziz/bluesky-sandbox" target="_blank" rel="noreferrer">
              Source on GitHub
            </a>{" "}
            · MIT License
          </p>
        </div>
      </div>
    </div>
  );
}
