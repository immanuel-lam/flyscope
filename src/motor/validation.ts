import type { MotorTrack } from "./types.ts";
import { JOINTS, RIG_ID } from "./types.ts";
export function validateMotorTrack(
  input: unknown,
  datasetId: string,
  datasetVersion: string,
): MotorTrack {
  const t = input as MotorTrack;
  const fail = (message: string): never => {
    throw new Error(`Motor track: ${message}`);
  };
  if (!t || t.schemaVersion !== 1 || t.rig !== RIG_ID)
    fail(`expected schemaVersion 1 and rig ${RIG_ID}.`);
  if (t.datasetId !== datasetId || t.datasetVersion !== datasetVersion)
    fail("dataset ID/version does not match the loaded structure.");
  if (
    !["simulation", "recording"].includes(t.kind) ||
    typeof t.source !== "string" ||
    !t.source.trim() ||
    typeof t.model !== "string" ||
    !t.model.trim()
  )
    fail("kind, source and model are required.");
  if (
    t.units?.position !== "mm" ||
    t.units?.angle !== "rad" ||
    t.units?.time !== "s"
  )
    fail("convert to mm, radians and seconds before import.");
  if (
    !Array.isArray(t.times) ||
    !t.times.length ||
    t.times.length > 18001 ||
    !t.times.every(
      (v, i) => Number.isFinite(v) && v >= 0 && (i === 0 || v > t.times[i - 1]),
    )
  )
    fail("expected 1–18,001 strictly increasing non-negative timestamps.");
  if (!Array.isArray(t.poses) || t.poses.length !== t.times.length)
    fail("pose count must match timestamps.");
  for (const p of t.poses) {
    if (
      !p ||
      !Array.isArray(p.position) ||
      p.position.length !== 3 ||
      !p.position.every((v) => Number.isFinite(v) && Math.abs(v) <= 10000) ||
      !Number.isFinite(p.heading)
    )
      fail("invalid root pose.");
    if (
      !p.joints ||
      Object.keys(p.joints).length !== JOINTS.length ||
      !JOINTS.every(
        (j) => Number.isFinite(p.joints[j]) && Math.abs(p.joints[j]) <= Math.PI,
      )
    )
      fail("all 20 named joints are required, each within ±π radians.");
  }
  if (t.generator !== undefined) {
    const g = t.generator;
    if (
      !g ||
      typeof g.controllerId !== "string" ||
      !g.controllerId.trim() ||
      !Number.isFinite(g.stepSeconds) ||
      g.stepSeconds <= 0 ||
      !g.parameters ||
      ![g.parameters.forward, g.parameters.turn, g.parameters.wing].every(
        Number.isFinite,
      )
    )
      fail("invalid generator configuration.");
  }
  return t;
}
