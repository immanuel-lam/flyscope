import type { Dataset } from "../data";
export interface CatalogInfo {
  total: number;
  positioned: number;
  inclusion: string;
  loadMs: number;
  bytes: number;
}
export interface Detail {
  id: string;
  segments: Float32Array;
  totalSegments: number;
  bytes: number;
  loadMs: number;
}
export interface GraphInfo {
  edges: number;
  synapseCount: number;
  rawEdges: number;
  bytes: number;
}
export interface ConnectionResult {
  id: string;
  edges: { source: string; target: string; weight: number }[];
  total: number;
  loadMs: number;
}
export type LargeMessage =
  | {
      kind: "catalog";
      dataset: Dataset;
      info: CatalogInfo;
      counts: Uint32Array;
      graph: GraphInfo;
    }
  | { kind: "detail"; detail: Detail }
  | { kind: "connections"; result: ConnectionResult }
  | { kind: "error"; request: string; id?: string; message: string };
