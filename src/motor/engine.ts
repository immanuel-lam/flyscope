import type { Dataset } from "../data.ts";
import { activityAt } from "../data.ts";
import { JOINTS, LEGS, RIG_ID, neutralPose } from "./types.ts";
import type {
  MotorCommand,
  MotorController,
  MotorPose,
  MotorTrack,
} from "./types.ts";
import { validateMotorTrack } from "./validation.ts";
export const DT = 1 / 120;
export const MAX_SECONDS = 60;
const clamp = (v: number, min: number, max: number) =>
  Math.max(min, Math.min(max, v));
/** This is a stable servo + kinematic locomotion model, not contact physics. */
export function simulateMotor(
  dataset: Dataset,
  controller: MotorController,
  parameters: MotorCommand,
  duration = 30,
): MotorTrack {
  const reason = controller.unsupported(dataset);
  if (reason) throw new Error(reason);
  if (!Number.isFinite(duration) || duration <= 0 || duration > MAX_SECONDS)
    throw new Error(
      "Local motor runs must be longer than 0 and no longer than 60 seconds.",
    );
  const driver = controller.create({ dataset, parameters });
  let pose = neutralPose();
  const velocities = neutralPose().joints;
  let forward = 0,
    turn = 0,
    phase = 0;
  const times = [0];
  const poses = [structuredClone(pose)];
  const steps = Math.ceil(duration / DT);
  for (let i = 1; i <= steps; i++) {
    const time = Math.min(i * DT, duration);
    const dt = time - times[i - 1];
    const command = driver.step({
      time: times[i - 1],
      dt,
      pose: structuredClone(pose),
      activity: (id) => activityAt(dataset, id, times[i - 1]),
    });
    if (
      !command ||
      ![command.forward, command.turn, command.wing].every(Number.isFinite)
    )
      throw new Error(
        `Controller ${controller.id} returned a non-finite command at ${time.toFixed(3)} s.`,
      );
    const f = clamp(command.forward, -3, 3),
      r = clamp(command.turn, -2, 2),
      wing = clamp(command.wing, 0, 1);
    const response = 1 - Math.exp(-dt / 0.12);
    forward += (f - forward) * response;
    turn += (r - turn) * response;
    pose.heading += turn * dt;
    pose.position = [
      pose.position[0] + Math.sin(pose.heading) * forward * dt,
      0,
      pose.position[2] + Math.cos(pose.heading) * forward * dt,
    ];
    const effort = Math.min(
      1,
      (Math.abs(forward) + Math.abs(turn) * 0.5) / 1.5,
    );
    phase += Math.PI * 2 * (1 + effort * 3) * dt * (effort > 1e-4 ? 1 : 0);
    const target = neutralPose().joints;
    LEGS.forEach((leg, index) => {
      const side = leg.startsWith("L") ? -1 : 1;
      const tripod = [0, 1, 0, 1, 0, 1][index] * Math.PI;
      const wave = Math.sin(phase + tripod);
      const lift = Math.max(0, Math.cos(phase + tripod));
      target[`${leg}.sweep`] =
        wave * 0.42 * clamp((forward + side * turn * 0.5) / 1.5, -1, 1);
      target[`${leg}.lift`] = lift * 0.3 * effort;
      target[`${leg}.knee`] = lift * 0.52 * effort;
    });
    target["wing.L"] = target["wing.R"] = wing * 0.9;
    for (const j of JOINTS) {
      const acceleration =
        900 * (target[j] - pose.joints[j]) - 60 * velocities[j];
      velocities[j] += acceleration * dt;
      pose.joints[j] = clamp(
        pose.joints[j] + velocities[j] * dt,
        -Math.PI,
        Math.PI,
      );
    }
    times.push(time);
    poses.push(structuredClone(pose));
  }
  return validateMotorTrack(
    {
      schemaVersion: 1,
      rig: RIG_ID,
      kind: "simulation",
      source: `Flyscope ${controller.id}; ${controller.description}`,
      model: "servo-kinematic-v1 (no ground-contact or aerodynamic solver)",
      datasetId: dataset.id,
      datasetVersion: dataset.version,
      units: { position: "mm", angle: "rad", time: "s" },
      times,
      poses,
      generator: {
        controllerId: controller.id,
        parameters: { ...parameters },
        stepSeconds: DT,
      },
    },
    dataset.id,
    dataset.version,
  );
}
/** Hold poses after the final frame; before the first frame show the neutral pose. */
export function poseAt(track: MotorTrack | undefined, time: number): MotorPose {
  if (!track || time < track.times[0]) return neutralPose();
  let lo = 0,
    hi = track.times.length - 1;
  while (lo < hi) {
    const m = Math.ceil((lo + hi) / 2);
    if (track.times[m] <= time) lo = m;
    else hi = m - 1;
  }
  const a = track.poses[lo],
    b = track.poses[Math.min(lo + 1, track.poses.length - 1)];
  const fraction =
    lo === track.times.length - 1
      ? 0
      : (time - track.times[lo]) / (track.times[lo + 1] - track.times[lo]);
  // Heading is continuous/unwrapped, as required by the adapter contract.
  return {
    position: a.position.map((v, k) => v + (b.position[k] - v) * fraction) as [
      number,
      number,
      number,
    ],
    heading: a.heading + (b.heading - a.heading) * fraction,
    joints: Object.fromEntries(
      JOINTS.map((j) => [
        j,
        a.joints[j] + (b.joints[j] - a.joints[j]) * fraction,
      ]),
    ) as MotorPose["joints"],
  };
}
