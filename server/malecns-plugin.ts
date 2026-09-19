import {
  mkdir,
  readFile,
  writeFile,
  rename,
  readdir,
  stat,
  unlink,
  utimes,
} from "node:fs/promises";
import { resolve } from "node:path";
import type { Plugin } from "vite";
const cache = resolve("data/malecns/skeleton-cache");
const BASE =
  "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-precomputed/";
const pending = new Map<string, Promise<Buffer>>();
let active = 0;
const queue: (() => void)[] = [];
async function getSkeleton(id: string): Promise<Buffer> {
  await mkdir(cache, { recursive: true });
  const file = resolve(cache, id + ".bin");
  try {
    const b = await readFile(file);
    await utimes(file, new Date(), new Date());
    return b;
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== "ENOENT") throw e;
  }
  if (pending.has(id)) return pending.get(id)!;
  if (queue.length >= 32)
    throw new Error("Too many queued skeleton requests. Retry shortly.");
  const operation = (async () => {
    if (active >= 4) await new Promise<void>((r) => queue.push(r));
    active++;
    try {
      const r = await fetch(BASE + id, { signal: AbortSignal.timeout(30000) });
      if (!r.ok) throw new Error(`Source skeleton HTTP ${r.status}`);
      const limit = 32 * 1024 * 1024;
      if (Number(r.headers.get("content-length")) > limit)
        throw new Error("Skeleton exceeds 32 MB.");
      const reader = r.body!.getReader();
      const chunks: Uint8Array[] = [];
      let length = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        length += value.byteLength;
        if (length > limit) {
          await reader.cancel();
          throw new Error("Skeleton exceeds 32 MB.");
        }
        chunks.push(value);
      }
      const result = Buffer.concat(chunks);
      const temp = file + ".tmp";
      await writeFile(temp, result);
      await rename(temp, file);
      // 512 MB persistent cache; evict least recently accessed files.
      const files = await Promise.all(
        (await readdir(cache))
          .filter((n) => n.endsWith(".bin"))
          .map(async (name) => ({
            name,
            ...(await stat(resolve(cache, name))),
          })),
      );
      let total = files.reduce((s, f) => s + f.size, 0);
      for (const f of files.sort((a, b) => a.mtimeMs - b.mtimeMs)) {
        if (total <= 512 * 1024 * 1024) break;
        if (f.name === id + ".bin") continue;
        await unlink(resolve(cache, f.name)).catch(() => {});
        total -= f.size;
      }
      return result;
    } finally {
      active--;
      queue.shift()?.();
    }
  })();
  pending.set(id, operation);
  try {
    return await operation;
  } finally {
    pending.delete(id);
  }
}
export function malecnsPlugin(): Plugin {
  const install = (server: any) => {
    server.middlewares.use(async (req: any, res: any, next: () => void) => {
      const path = (req.url ?? "").split("?")[0];
      if (!path.startsWith("/api/malecns/skeleton/")) return next();
      const id = path.slice("/api/malecns/skeleton/".length);
      if (!/^\d{1,15}$/.test(id) || req.method !== "GET") {
        res.statusCode = 400;
        res.end("Expected a numeric MaleCNS ID and GET.");
        return;
      }
      try {
        const bytes = await getSkeleton(id);
        res.setHeader("Content-Type", "application/octet-stream");
        res.setHeader("Cache-Control", "public, max-age=86400");
        res.setHeader("Content-Length", bytes.length);
        res.end(bytes);
      } catch (e) {
        res.statusCode = 502;
        res.end((e as Error).message);
      }
    });
  };
  return {
    name: "malecns-fixed-source-cache",
    configureServer: install,
    configurePreviewServer: install,
  };
}
