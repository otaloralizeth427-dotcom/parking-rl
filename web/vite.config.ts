// Web de Parking RL: TypeScript + Vite + three.js (versiones fijas en package-lock.json).
//   npm ci && npm run build   → dist/ (lo publica Vercel; en local lo sirve `python run.py serve`)
//   npm run dev               → servidor de desarrollo con recarga; la API se toma del servidor local
import { defineConfig } from "vite";

const API = process.env.PARKING_API ?? "http://127.0.0.1:8765";

export default defineConfig({
  build: {
    target: "es2022",                 // main.ts usa await en el nivel superior
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,       // three.js ocupa ~600 kB sin comprimir
  },
  server: {
    proxy: {
      "/api": API,
      "/ws": { target: API.replace(/^http/, "ws"), ws: true },
    },
  },
});
