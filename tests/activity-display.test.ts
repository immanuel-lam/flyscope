import { test } from "node:test";
import assert from "node:assert/strict";
import {
  recordedPeaks,
  relativeRate,
  activityPointOrder,
} from "../src/physics/activity-display.ts";
test("activity display preserves silence and steady signals, and retains recorded cells in LOD", () => {
  const peaks = recordedPeaks({
    kind: "simulation",
    unit: "a.u.",
    times: [0, 1, 2],
    values: { "7": [0, 0.01, 0.02], "8": [0.4, 0.4, 0.4], "9": [0, 0, 0] },
  });
  assert.equal(relativeRate(0.01, peaks.get("7")!), 0.5);
  assert.equal(relativeRate(0.4, peaks.get("8")!), 1);
  assert.equal(relativeRate(0, peaks.get("9")!), 0);
  assert.equal(relativeRate(undefined, 1), 0);
  const neurons = Array.from({ length: 10 }, (_, i) => ({
    id: String(i),
    label: "",
    region: "",
    position: [0, 0, 0] as [number, number, number],
  }));
  const order = activityPointOrder(neurons, peaks, 4);
  assert.deepEqual(order.slice(0, 3), [7, 8, 9]);
  assert.equal(new Set(order).size, 10);
});
