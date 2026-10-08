import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The project's brand assets (the logo) live once, at the repository's
// assets/brand; the designer imports them from there as `@brand/...`, and the
// build copies what it uses into ./dist.
const brand = fileURLToPath(new URL("../../../../../assets/brand", import.meta.url));
// Partner logos for the About dialog (assets/partners/README.md).
const partners = fileURLToPath(new URL("../../../../../assets/partners", import.meta.url));

// Dev server proxies /api to the FastAPI backend so the frontend can call it
// without CORS friction. `build` emits to ./dist, which the API serves at /.
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@brand": brand, "@partners": partners } },
  build: { outDir: "dist" },
  server: {
    port: 5173,
    fs: { allow: [".", brand, partners] },
    proxy: {
      "/api": "http://127.0.0.1:8765",
    },
  },
});
