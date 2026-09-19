import { defineConfig } from "vite";
import { malecnsPlugin } from "./server/malecns-plugin.ts";
import { physicsPlugin } from './server/physics-plugin.ts';
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react(), malecnsPlugin(), physicsPlugin()],
  server: { host: "127.0.0.1" },
});
