// Construcción de la web para Vercel: `npm ci && npm run build` → dist/
// three.js se carga desde el CDN con el importmap de index.html (misma versión fija, 0.160.0),
// así la web sigue funcionando igual cuando la sirve el servidor local sin construir.
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    target: "es2022",                 // app.js usa await en el nivel superior
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: { external: [/^three(\/.*)?$/] },
  },
});
