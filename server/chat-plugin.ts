import type { Plugin, ViteDevServer, PreviewServer } from "vite";
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
export function chatPlugin(): Plugin {
  const root = process.cwd();
  let active = 0;
  function install(server: ViteDevServer | PreviewServer) {
    server.middlewares.use("/api/chat", async (req, res, next) => {
      const url = (req.url ?? "").split("?")[0];
      if (!["/status", "/generate"].includes(url)) {
        next();
        return;
      }
      const host = req.headers.host ?? "";
      if (
        !/^(127\.0\.0\.1|localhost|\[::1\]):\d+$/.test(host) ||
        (req.headers.origin && req.headers.origin !== `http://${host}`)
      ) {
        res.writeHead(403);
        res.end();
        return;
      }
      const json = (code: number, value: unknown) => {
        if (res.writableEnded) return;
        res.writeHead(code, {
          "Content-Type": "application/json",
          "Cache-Control": "no-store",
        });
        res.end(JSON.stringify(value));
      };
      const manifest = resolve(root, "models/malecns-chat/manifest.json");
      if (url === "/status" && req.method === "GET") {
        if (!existsSync(manifest)) {
          json(200, { ready: false });
          return;
        }
        const m = JSON.parse(readFileSync(manifest, "utf8"));
        json(200, {
          ready: true,
          modelId: m.modelId,
          neurons: m.neurons.length,
          edges: m.edges,
          architecture: m.architecture,
          training: m.training,
          weightsSha256: m.weightsSha256,
        });
        return;
      }
      if (url !== "/generate" || req.method !== "POST") {
        json(405, { error: "Method not allowed" });
        return;
      }
      if (active >= 1) {
        json(429, {
          error: "A response is already running. Try again when it finishes.",
        });
        return;
      }
      try {
        let body = "";
        for await (const chunk of req) {
          body += chunk;
          if (body.length > 12000) throw new Error("Request too large");
        }
        const input = JSON.parse(body);
        if (
          typeof input.message !== "string" ||
          !input.message.trim() ||
          input.message.length > 1000
        )
          throw new Error("Enter 1–1,000 characters.");
        const history = input.history ?? [];
        if (
          !Array.isArray(history) ||
          history.length > 4 ||
          history.some(
            (t) =>
              !["user", "assistant"].includes(t.role) ||
              typeof t.content !== "string" ||
              t.content.length > 1000,
          )
        )
          throw new Error("Invalid conversation history");
        if (!existsSync(manifest)) {
          json(503, { error: "The checkpoint is still training." });
          return;
        }
        active++;
        const child = spawn(
          resolve(root, ".venv-physics/bin/python"),
          [resolve(root, "scripts/language/runtime.py")],
          {
            cwd: root,
            env: {
              ...process.env,
              VECLIB_MAXIMUM_THREADS: "1",
              OPENBLAS_NUM_THREADS: "1",
            },
            stdio: ["pipe", "pipe", "pipe"],
          },
        );
        let output = "",
          error = "";
        const timer = setTimeout(() => child.kill(), 60000);
        child.stdout.on("data", (data) => {
          output += data;
          if (output.length > 4_000_000) child.kill();
        });
        child.stderr.on("data", (data) => {
          error = (error + data).slice(-2000);
        });
        child.on("error", (e) => {
          json(500, { error: e.message });
        });
        child.on("close", (code) => {
          clearTimeout(timer);
          active--;
          if (code !== 0) {
            json(500, { error: error || "Inference stopped" });
            return;
          }
          try {
            json(200, JSON.parse(output));
          } catch {
            json(500, { error: "Invalid model output" });
          }
        });
        child.stdin.end(
          JSON.stringify({
            message: input.message,
            history,
            maxTokens: 40,
            ablated: input.ablated === true,
          }),
        );
      } catch (e) {
        json(400, { error: (e as Error).message });
      }
    });
  }
  return {
    name: "flyscope-chat",
    configureServer(s) {
      install(s);
    },
    configurePreviewServer(s) {
      install(s);
    },
  };
}
