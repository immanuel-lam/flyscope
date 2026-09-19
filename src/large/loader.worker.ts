/// <reference lib="webworker" />
import type { Dataset, Neuron } from "../data";
import type { LargeMessage } from "./types";
const BASE = "/api/malecns/skeleton/";
let catalog: Dataset | undefined;
const details = new Map<string, ArrayBuffer>();
let cachedBytes = 0;
const shards = new Map<number, ArrayBuffer>();
const pending = new Map<string, Promise<ArrayBuffer>>();
async function buffer(url: string) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Source returned HTTP ${r.status}. ${url}`);
  return r.arrayBuffer();
}
async function cachedSkeleton(id: string) {
  if (details.has(id)) return details.get(id)!;
  if (pending.has(id)) return pending.get(id)!;
  const p = buffer(BASE + id);
  pending.set(id, p);
  try {
    const b = await p;
    if (b.byteLength > 32 * 1024 * 1024)
      throw new Error("Skeleton exceeds the 32 MB per-neuron detail limit.");
    while (cachedBytes + b.byteLength > 64 * 1024 * 1024 && details.size) {
      const key = details.keys().next().value!;
      cachedBytes -= details.get(key)!.byteLength;
      details.delete(key);
    }
    details.set(id, b);
    cachedBytes += b.byteLength;
    return b;
  } finally {
    pending.delete(id);
  }
}
function send(message: LargeMessage, transfer: Transferable[] = []) {
  self.postMessage(message, transfer);
}
self.onmessage = async ({ data }: { data: { kind: string; id?: string } }) => {
  const start = performance.now();
  try {
    if (data.kind === "catalog") {
      const [response, countsBuffer, graphResponse] = await Promise.all([
        fetch("/malecns/catalog.json"),
        buffer("/malecns/outgoing-counts.bin"),
        fetch("/malecns/graph.json"),
      ]);
      if (!response.ok || !graphResponse.ok)
        throw new Error(
          "Run npm run prepare:malecns to create the catalog and graph assets.",
        );
      const text = await response.text();
      const raw = JSON.parse(text);
      const graph = await graphResponse.json();
      const neurons: Neuron[] = raw.rows.map(
        (r: [string, string, string, number[] | null, string, string]) => ({
          id: r[0],
          label: r[1],
          region: r[2].replaceAll("_", " "),
          position: r[3] ?? [0, 0, 0],
          positionKnown: r[3] !== null,
          positionKind: r[4],
          status: r[5],
        }),
      );
      catalog = {
        schemaVersion: 1,
        id: raw.id,
        name: raw.name,
        source: raw.source,
        version: raw.version,
        geometry: "reconstruction",
        coordinateSpace: raw.coordinateSpace,
        units: raw.units,
        neurons,
        edges: [],
      };
      const counts = new Uint32Array(countsBuffer);
      if (counts.length !== neurons.length)
        throw new Error(
          "Graph and catalog sizes do not match. Rebuild both assets.",
        );
      send({
        kind: "catalog",
        dataset: catalog,
        info: {
          total: raw.total,
          positioned: raw.positioned,
          inclusion: raw.inclusion,
          loadMs: performance.now() - start,
          bytes:
            new TextEncoder().encode(text).length + countsBuffer.byteLength,
        },
        counts,
        graph,
      });
    } else if (data.kind === "detail" && data.id) {
      if (!/^\d+$/.test(data.id)) throw new Error("Invalid MaleCNS body ID.");
      const bytes = await cachedSkeleton(data.id);
      const view = new DataView(bytes);
      const nv = view.getUint32(0, true),
        ne = view.getUint32(4, true);
      if (8 + nv * 12 + ne * 8 > bytes.byteLength)
        throw new Error("Truncated source skeleton.");
      const vertices = new Float32Array(bytes, 8, nv * 3),
        edges = new Uint32Array(bytes, 8 + nv * 12, ne * 2);
      const count = Math.min(ne, 150000);
      const segments = new Float32Array(count * 6);
      const stride = ne / count;
      for (let i = 0; i < count; i++) {
        const e = Math.floor(i * stride);
        for (let end = 0; end < 2; end++) {
          const v = edges[e * 2 + end];
          if (v >= nv) throw new Error("Invalid skeleton edge.");
          for (let k = 0; k < 3; k++)
            segments[i * 6 + end * 3 + k] = vertices[v * 3 + k] / 8;
        }
      }
      send(
        {
          kind: "detail",
          detail: {
            id: data.id,
            segments,
            totalSegments: ne,
            bytes: bytes.byteLength,
            loadMs: performance.now() - start,
          },
        },
        [segments.buffer],
      );
    } else if (data.kind === "connections" && data.id) {
      if (!catalog) throw new Error("Load the catalog first.");
      const index = catalog.neurons.findIndex((n) => n.id === data.id);
      if (index < 0) throw new Error("Neuron not in catalog.");
      const shard = index % 128;
      let b = shards.get(shard);
      if (!b) {
        b = await buffer(`/malecns/connections/${shard}.bin`);
        if (shards.size >= 4) shards.delete(shards.keys().next().value!);
        shards.set(shard, b);
      }
      const records = new Uint32Array(b);
      const found: { source: string; target: string; weight: number }[] = [];
      for (let i = 0; i < records.length; i += 3)
        if (records[i] === index)
          found.push({
            source: data.id,
            target: catalog.neurons[records[i + 1]].id,
            weight: records[i + 2],
          });
      found.sort((a, b) => b.weight - a.weight);
      send({
        kind: "connections",
        result: {
          id: data.id,
          edges: found.slice(0, 2000),
          total: found.length,
          loadMs: performance.now() - start,
        },
      });
    }
  } catch (e) {
    send({
      kind: "error",
      request: data.kind,
      id: data.id,
      message: (e as Error).message,
    });
  }
};
