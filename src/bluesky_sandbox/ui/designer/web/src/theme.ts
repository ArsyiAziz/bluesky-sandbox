// Light or dark: the designer's theme. A preference - light, dark, or the
// system's - kept in this browser; the theme it resolves to is set on
// <html data-theme>, which styles.css reads every color from.
import { useSyncExternalStore } from "react";

export type ThemePreference = "system" | "light" | "dark";
export type Theme = "light" | "dark";

const KEY = "designer.theme";
const listeners = new Set<() => void>();
const media = window.matchMedia("(prefers-color-scheme: light)");

function stored(): ThemePreference {
  try {
    const value = window.localStorage.getItem(KEY);
    return value === "light" || value === "dark" || value === "system" ? value : "system";
  } catch {
    return "system";
  }
}

let preference: ThemePreference = stored();

function resolve(p: ThemePreference): Theme {
  if (p === "system") return media.matches ? "light" : "dark";
  return p;
}

function apply() {
  const theme = resolve(preference);
  document.documentElement.dataset.theme = theme;
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme === "light" ? "#ffffff" : "#0e1626");
  listeners.forEach((listener) => listener());
}

// The system's change reaches a "system" preference at once.
media.addEventListener("change", () => {
  if (preference === "system") apply();
});

/** Set the theme before the first paint, so the page never flashes the other. */
export function initTheme() {
  apply();
}

export function setThemePreference(next: ThemePreference) {
  preference = next;
  try {
    window.localStorage.setItem(KEY, next);
  } catch {
    // Storage blocked: the choice holds for this page only.
  }
  apply();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** The preference and the theme it resolves to, re-rendering on a change. */
export function useTheme(): { preference: ThemePreference; theme: Theme } {
  const pref = useSyncExternalStore(subscribe, () => preference);
  const theme = useSyncExternalStore(subscribe, () => resolve(preference));
  return { preference: pref, theme };
}

/** The code editor's theme for a designer theme. */
export function editorTheme(theme: Theme): string {
  return theme === "light" ? "vs" : "vs-dark";
}
