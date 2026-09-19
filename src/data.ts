import type { MotorTrack } from "./motor/types.ts";
import { validateMotorTrack } from "./motor/validation.ts";
export type Vec3 = [number, number, number];
export interface Neuron {
  id: string;
  label: string;
  region: string;
  position: Vec3;
  skeleton?: number[];
  positionKnown?: boolean;
  positionKind?: string;
  status?: string;
}
export interface Edge {
  source: string;
  target: string;
  weight: number;
}
export interface Activity {
  kind: "synthetic" | "simulation" | "recording";
  unit: string;
  times: number[];
  values: Record<string, number[]>;
}
export interface Dataset {
  schemaVersion: 1;
  id: string;
  name: string;
  source: string;
  version: string;
  geometry: "synthetic" | "reconstruction";
  coordinateSpace: string;
  units: string;
  neurons: Neuron[];
  edges: Edge[];
  activity?: Activity;
  motor?: MotorTrack;
}
export const palette = [
  "#b6cf71",
  "#e8b67d",
  "#b698d7",
  "#79b9aa",
  "#d68d8c",
  "#8aaed7",
];
export function parseDataset(input: unknown): Dataset {
  const d = input as Dataset;
  const fail = (s: string): never => {
    throw new Error(s);
  };
  if (!d || d.schemaVersion !== 1)
    fail("Expected schemaVersion: 1. Use the example export as a template.");
  for (const key of [
    "id",
    "name",
    "source",
    "version",
    "coordinateSpace",
    "units",
  ] as const)
    if (typeof d[key] !== "string" || !d[key].trim()) fail(`Missing ${key}.`);
  if (!["synthetic", "reconstruction"].includes(d.geometry))
    fail("Geometry must be synthetic or reconstruction.");
  if (
    !Array.isArray(d.neurons) ||
    !d.neurons.length ||
    d.neurons.length > 20000
  )
    fail("Load between 1 and 20,000 neurons per dataset.");
  const ids = new Set<string>();
  let segments = 0;
  for (const n of d.neurons) {
    if (!n || typeof n.id !== "string" || !n.id || ids.has(n.id))
      fail(
        "Each neuron needs a unique string ID. Keep large source IDs as strings.",
      );
    ids.add(n.id);
    if (typeof n.label !== "string" || typeof n.region !== "string")
      fail("Each neuron needs a label and region.");
    if (
      !Array.isArray(n.position) ||
      n.position.length !== 3 ||
      !n.position.every(Number.isFinite)
    )
      fail(`Invalid position for ${n.id}.`);
    if (
      n.skeleton !== undefined &&
      (!Array.isArray(n.skeleton) ||
        n.skeleton.length % 6 !== 0 ||
        !n.skeleton.every(Number.isFinite))
    )
      fail(`Invalid skeleton segments for ${n.id}.`);
    segments += (n.skeleton?.length ?? 0) / 6;
  }
  if (segments > 500000)
    fail(
      "This viewer supports up to 500,000 skeleton segments. Load a smaller subset.",
    );
  if (!Array.isArray(d.edges) || d.edges.length > 100000)
    fail("Edges must be an array with at most 100,000 entries.");
  for (const e of d.edges)
    if (
      !e ||
      !ids.has(e.source) ||
      !ids.has(e.target) ||
      !Number.isFinite(e.weight) ||
      e.weight < 0
    )
      fail(
        "Each edge must reference loaded neuron IDs and have a non-negative finite weight.",
      );
  if (d.activity) {
    const a = d.activity;
    if (
      !["synthetic", "simulation", "recording"].includes(a.kind) ||
      typeof a.unit !== "string" ||
      !a.unit
    )
      fail("Activity needs a kind and unit.");
    if (
      !Array.isArray(a.times) ||
      !a.times.length ||
      a.times.length > 100000 ||
      !a.times.every(
        (t, i) =>
          Number.isFinite(t) && t >= 0 && (i === 0 || t > a.times[i - 1]),
      )
    )
      fail("Activity times must be increasing, non-negative seconds.");
    if (
      !a.values ||
      typeof a.values !== "object" ||
      Array.isArray(a.values) ||
      !Object.keys(a.values).length
    )
      fail("Activity needs neuron value arrays.");
    for (const [id, values] of Object.entries(a.values))
      if (
        !ids.has(id) ||
        !Array.isArray(values) ||
        values.length !== a.times.length ||
        !values.every(Number.isFinite)
      )
        fail(`Activity values do not match neuron ${id} or the time axis.`);
  }
  if (d.motor !== undefined) validateMotorTrack(d.motor, d.id, d.version);
  return d;
}
export function parseSWC(
  text: string,
  id: string,
  label = id,
  region = "Imported",
): Neuron {
  const nodes = new Map<number, { p: Vec3; parent: number }>();
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim() || line.trimStart().startsWith("#")) continue;
    const v = line.trim().split(/\s+/).map(Number);
    if (
      v.length < 7 ||
      !v.slice(0, 7).every(Number.isFinite) ||
      !Number.isInteger(v[0]) ||
      !Number.isInteger(v[6]) ||
      nodes.has(v[0])
    )
      throw new Error("Invalid or duplicate SWC node.");
    nodes.set(v[0], { p: [v[2], v[3], v[4]], parent: v[6] });
  }
  if (!nodes.size) throw new Error("The SWC file has no nodes.");
  const skeleton: number[] = [];
  for (const n of nodes.values())
    if (n.parent !== -1) {
      const parent = nodes.get(n.parent);
      if (!parent) throw new Error(`Missing SWC parent ${n.parent}.`);
      skeleton.push(...parent.p, ...n.p);
    }
  const root =
    [...nodes.values()].find((n) => n.parent === -1) ??
    nodes.values().next().value!;
  return { id, label, region, position: root.p, skeleton };
}
export function demoDataset(): Dataset {
  let seed = 7281;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  const neurons: Neuron[] = [];
  const edges: Edge[] = [];
  const regions = [
    "Optic lobe · L",
    "Optic lobe · R",
    "Central brain",
    "Mushroom body",
    "Antennal lobe",
    "Descending",
  ];
  const centres: Vec3[] = [
    [-2.0, 0, 0],
    [2.0, 0, 0],
    [0, 0, 0],
    [0, 0.85, -0.2],
    [0, -0.8, 0.45],
    [0, -1.55, -0.2],
  ];
  for (let i = 0; i < 720; i++) {
    const g = i % 6;
    const c = centres[g];
    const a = rand() * Math.PI * 2;
    const z = rand() * 2 - 1;
    const r = Math.cbrt(rand());
    const p: Vec3 = [
      c[0] + Math.cos(a) * Math.sqrt(1 - z * z) * r * (g < 2 ? 0.78 : 0.95),
      c[1] + z * r * (g < 2 ? 1.05 : 0.55),
      c[2] + Math.sin(a) * Math.sqrt(1 - z * z) * r * 0.62,
    ];
    const skeleton: number[] = [];
    let prev = p;
    const target: Vec3 = [
      c[0] * 0.35 + (rand() - 0.5) * 0.8,
      c[1] * 0.3 + (rand() - 0.5) * 0.8,
      (rand() - 0.5) * 0.7,
    ];
    for (let j = 1; j <= 9; j++) {
      const t = j / 9;
      const q: Vec3 = p.map(
        (v, k) =>
          v * (1 - t) +
          target[k] * t +
          Math.sin(t * Math.PI) * (rand() - 0.5) * 0.3,
      ) as Vec3;
      skeleton.push(...prev, ...q);
      if (j > 4) {
        const b: Vec3 = q.map((v) => v + (rand() - 0.5) * 0.38) as Vec3;
        skeleton.push(...q, ...b);
      }
      prev = q;
    }
    neurons.push({
      id: `demo-${i}`,
      label: `${["OL-L", "OL-R", "CB", "MB", "AL", "DN"][g]} ${String(Math.floor(i / 6) + 1).padStart(3, "0")}`,
      region: regions[g],
      position: p,
      skeleton,
    });
    if (i > 5)
      edges.push({
        source: `demo-${i - 6}`,
        target: `demo-${i}`,
        weight: 1 + Math.floor(rand() * 12),
      });
  }
  const times = Array.from({ length: 301 }, (_, i) => i / 10);
  const values: Record<string, number[]> = {};
  neurons.forEach((n, i) => {
    values[n.id] = times.map((t) => demoValue(i, t));
  });
  return {
    schemaVersion: 1,
    id: "synthetic-workbench",
    name: "Synthetic fly circuit",
    source:
      "Procedural fixture. Not measured anatomy or a biological simulation.",
    version: "1.0",
    geometry: "synthetic",
    coordinateSpace: "Illustrative brain-local coordinates",
    units: "arbitrary",
    neurons,
    edges,
    activity: { kind: "synthetic", unit: "a.u.", times, values },
  };
}
export function demoValue(index: number, time: number) {
  return (
    Math.max(0, Math.sin(time * 1.8 - (index % 6) * 0.9 + index * 0.08)) ** 8 *
    (0.4 + 0.6 * (Math.sin(index * 17.3) * 0.5 + 0.5))
  );
}
export function frameAt(a: Activity | undefined, time: number) {
  if (!a) return -1;
  let low = 0,
    high = a.times.length - 1;
  while (low < high) {
    const m = Math.ceil((low + high) / 2);
    if (a.times[m] <= time) low = m;
    else high = m - 1;
  }
  return low;
}
export function activityAt(
  d: Dataset,
  id: string,
  time: number,
): number | undefined {
  const a = d.activity;
  if (!a || time < a.times[0]) return undefined;
  return a.values[id]?.[frameAt(a, time)];
}
export function activityRange(d: Dataset): [number, number] {
  let min = Infinity,
    max = -Infinity;
  for (const vals of Object.values(d.activity?.values ?? {}))
    for (const v of vals) {
      min = Math.min(min, v);
      max = Math.max(max, v);
    }
  return Number.isFinite(min)
    ? [Math.min(0, min), max === min ? min + 1 : max]
    : [0, 1];
}
