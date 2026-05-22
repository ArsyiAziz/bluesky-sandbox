import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api to the FastAPI backend so the frontend can call it
// without CORS friction. `build` emits to ./dist, which the API serves at /.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765",
    },
  },
});
