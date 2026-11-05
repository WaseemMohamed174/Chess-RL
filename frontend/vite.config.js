import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the website on :5173 and forwards API + WebSocket calls to the gateway.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
  build: { outDir: "dist", chunkSizeWarningLimit: 900 },
});
