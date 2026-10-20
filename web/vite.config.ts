import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    // changeOrigin: API проверяет Host; Origin браузера остаётся http://127.0.0.1:5173
    proxy: { "/api": { target: "http://127.0.0.1:8765", changeOrigin: true } },
  },
});
