import { test } from "node:test";
import assert from "node:assert/strict";
import { demoDataset, parseDataset } from "../src/data.ts";
import { simulateMotor, poseAt } from "../src/motor/engine.ts";
import { neutralPose, JOINTS } from "../src/motor/types.ts";
import { controller as manual } from "../src/controllers/manual/controller.ts";
import { controller as neural } from "../src/controllers/neural-readout/controller.ts";
const d = demoDataset();
test("zero command stays still; movement and turn commands change the root and articulated joints", () => {
  const rest = simulateMotor(d, manual, { forward: 0, turn: 0, wing: 0 }, 1);
  assert.deepEqual(rest.poses.at(-1), neutralPose());
  const run = simulateMotor(d, manual, { forward: 1, turn: 0.5, wing: 1 }, 2);
  const end = run.poses.at(-1)!;
  assert.ok(end.position[0] > 0.2);
  assert.ok(end.position[2] > 0.5);
  assert.ok(end.heading > 0.5);
  assert.ok(end.joints["wing.L"] > 0.8);
  assert.ok(run.poses.some((p) => Math.abs(p.joints["LF.sweep"]) > 0.1));
  assert.equal(JOINTS.length, 20);
  assert.deepEqual(run.generator?.parameters, {
    forward: 1,
    turn: 0.5,
    wing: 1,
  });
});
test("fixed-step simulation is deterministic and seeking is independent of playback order", () => {
  const params = { forward: 1, turn: -0.5, wing: 0 };
  const a = simulateMotor(d, manual, params, 2);
  const b = simulateMotor(d, manual, params, 2);
  assert.deepEqual(a, b);
  const first = poseAt(a, 0.73);
  poseAt(a, 1.9);
  assert.deepEqual(first, poseAt(a, 0.73));
  assert.deepEqual(poseAt(a, -1), neutralPose());
  assert.deepEqual(poseAt(a, 5), a.poses.at(-1));
});
test("neural drive depends on mapped data and refuses unmatched real datasets", () => {
  const moving = simulateMotor(d, neural, { forward: 0, turn: 0, wing: 0 }, 2);
  assert.ok(moving.poses.at(-1)!.position[2] > 0.01);
  const quiet = structuredClone(d);
  for (const id of Object.keys(quiet.activity!.values))
    quiet.activity!.values[id].fill(0);
  assert.deepEqual(
    simulateMotor(quiet, neural, { forward: 0, turn: 0, wing: 0 }, 1).poses.at(
      -1,
    ),
    neutralPose(),
  );
  assert.throws(
    () =>
      simulateMotor(
        { ...d, id: "male-cns" },
        neural,
        { forward: 0, turn: 0, wing: 0 },
        1,
      ),
    /Requires/,
  );
});
test("rejects malformed motor data, mismatched provenance, unsupported units and invalid controller output", () => {
  const motor = simulateMotor(d, manual, { forward: 1, turn: 0, wing: 0 }, 1);
  assert.equal(parseDataset({ ...d, motor }).motor, motor);
  assert.throws(
    () => parseDataset({ ...d, motor: { ...motor, datasetVersion: "wrong" } }),
    /version/,
  );
  assert.throws(
    () =>
      parseDataset({
        ...d,
        motor: { ...motor, units: { position: "m", angle: "rad", time: "s" } },
      }),
    /convert/,
  );
  const bad = structuredClone(motor);
  delete (bad.poses[0].joints as Partial<(typeof bad.poses)[0]["joints"]>)[
    "LF.knee"
  ];
  assert.throws(() => parseDataset({ ...d, motor: bad }), /20 named/);
  assert.throws(
    () =>
      simulateMotor(
        d,
        {
          ...manual,
          create: () => ({ step: () => ({ forward: NaN, turn: 0, wing: 0 }) }),
        },
        { forward: 0, turn: 0, wing: 0 },
        1,
      ),
    /non-finite/,
  );
  assert.throws(
    () => simulateMotor(d, manual, { forward: 0, turn: 0, wing: 0 }, 61),
    /60 seconds/,
  );
});
test("motor-only dataset needs no fabricated neural activity", () => {
  const structure = { ...d, activity: undefined };
  const motor = simulateMotor(
    structure,
    manual,
    { forward: 1, turn: 0, wing: 0 },
    1,
  );
  assert.ok(parseDataset({ ...structure, motor }).motor);
  assert.equal(structure.activity, undefined);
});
