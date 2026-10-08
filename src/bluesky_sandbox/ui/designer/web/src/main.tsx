import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource-variable/inter";
import "@fontsource-variable/space-grotesk";
import icon from "@brand/bluesky-sandbox-icon-small.svg";
import App from "./App";
import { initTheme } from "./theme";
import "./styles.css";
import "maplibre-gl/dist/maplibre-gl.css";

initTheme();

// The tab's icon: the logo's small mark, bundled with the page.
const favicon = document.createElement("link");
favicon.rel = "icon";
favicon.type = "image/svg+xml";
favicon.href = icon;
document.head.appendChild(favicon);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
