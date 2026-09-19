import type { Activity, Dataset, Vec3 } from "../data.ts";
export const LEGS = ["LF", "LM", "LH", "RF", "RM", "RH"] as const;
export type Leg = (typeof LEGS)[number];
export type JointName =
  `${Leg}.${"sweep" | "lift" | "knee"}` | "wing.L" | "wing.R";
export const JOINTS: JointName[] = [
  ...LEGS.flatMap(
    (l) => [`${l}.sweep`, `${l}.lift`, `${l}.knee`] as JointName[],
  ),
  "wing.L",
  "wing.R",
];
export type JointAngles = Record<JointName, number>;
export const RIG_ID = "flyscope-procedural-v1" as const;
export interface MotorPose {
  position: Vec3;
  heading: number;
  joints: JointAngles;
}
export interface MotorTrack {
  schemaVersion: 1;
  rig: typeof RIG_ID;
  kind: "simulation" | "recording";
  source: string;
  model: string;
  datasetId: string;
  datasetVersion: string;
  units: { position: "mm"; angle: "rad"; time: "s" };
  times: number[];
  poses: MotorPose[];
  generator?: {
    controllerId: string;
    parameters: MotorCommand;
    stepSeconds: number;
  };
}
export interface MotorCommand {
  forward: number;
  turn: number;
  wing: number;
  neural?: Record<string, number>;
}
export interface MotorObservation {
  time: number;
  dt: number;
  pose: Readonly<MotorPose>;
  /** No fabricated contact or sensory measurements. */
  activity: (neuronId: string) => number | undefined;
}
export interface ControllerContext {
  dataset: Dataset;
  parameters: Readonly<MotorCommand>;
}
export interface MotorController {
  id: string;
  label: string;
  description: string;
  activity?: Pick<Activity, "kind" | "unit">;
  /** Return a reason to disable this controller for an unsupported dataset. */
  unsupported: (dataset: Dataset) => string | null;
  /** Create fresh state for every replay; no shared mutable controller state. */
  create: (context: ControllerContext) => {
    step: (observation: MotorObservation) => MotorCommand;
  };
}
export function neutralPose(): MotorPose {
  return {
    position: [0, 0, 0],
    heading: 0,
    joints: Object.fromEntries(JOINTS.map((j) => [j, 0])) as JointAngles,
  };
}
