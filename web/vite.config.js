import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Dev-only mirror of the nginx /api reverse-proxy (web/nginx.conf) so
    // the same relative-URL code path works under `vite dev` against a
    // local API on :8000. The rewrite strips /api, matching proxy_pass
    // http://api:8000/ behaviour in production.
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
