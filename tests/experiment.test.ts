import { test } from "node:test";
import assert from "node:assert/strict";
import { demoDataset, parseDataset } from "../src/data.ts";
import { simulateExperiment } from "../src/motor/experiment.ts";
import { controller } from "../src/controllers/walking-circuit/controller.ts";
test("walking circuit records the rates that drive motion, silence stops both", () => {
  const d = demoDataset();
  const run = simulateExperiment(
    d,
    controller,
    { forward: 1, turn: 0, wing: 0 },
    1,
  );
  assert.ok(run.activity);
  assert.ok(new Set(run.activity.values["demo-5"]).size > 10);
  assert.ok(run.motor.poses.at(-1)!.position[2] > 0.5);
  assert.equal(run.activity.times.at(-1), run.motor.times.at(-1));
  assert.doesNotThrow(() =>
    parseDataset({ ...d, activity: run.activity, motor: run.motor }),
  );
  const rest = simulateExperiment(
    d,
    controller,
    { forward: 0, turn: 0, wing: 0 },
    1,
  );
  assert.ok(rest.activity!.values["demo-5"].every((v) => v === 0));
  assert.equal(rest.motor.poses.at(-1)!.position[2], 0);
  assert.throws(
    () =>
      simulateExperiment(
        { ...d, id: "male-cns-v1-full" },
        controller,
        { forward: 1, turn: 0, wing: 0 },
        1,
      ),
    /synthetic/,
  );
});
