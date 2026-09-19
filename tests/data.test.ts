import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  parseDataset,
  parseSWC,
  demoDataset,
  activityAt,
} from "../src/data.ts";
test("bundled source skeletons parse without missing parents", () => {
  for (const id of ["12781", "556329"]) {
    const n = parseSWC(readFileSync(`public/${id}.swc`, "utf8"), id);
    assert.ok(n.skeleton!.length > 6000);
    assert.ok(n.position.every(Number.isFinite));
  }
});
test("rejects invalid IDs, references and activity rather than rendering misleading values", () => {
  const d = demoDataset();
  assert.equal(parseDataset(d), d);
  assert.throws(
    () => parseDataset({ ...d, neurons: [d.neurons[0], d.neurons[0]] }),
    /unique/,
  );
  assert.throws(
    () =>
      parseDataset({
        ...d,
        edges: [{ source: "missing", target: d.neurons[0].id, weight: 1 }],
      }),
    /reference/,
  );
  assert.throws(
    () => parseDataset({ ...d, activity: { ...d.activity, times: [1, 0] } }),
    /increasing/,
  );
  assert.throws(
    () =>
      parseDataset({
        ...d,
        activity: { ...d.activity, values: { missing: [0] } },
      }),
    /match/,
  );
});
test("holds previous activity sample without inventing data for unrecorded neurons", () => {
  const d = demoDataset();
  d.activity = {
    kind: "recording",
    unit: "mV",
    times: [1, 2, 3],
    values: { "demo-0": [-70, -50, -65] },
  };
  assert.equal(activityAt(d, "demo-0", 0), undefined);
  assert.equal(activityAt(d, "demo-0", 1.5), -70);
  assert.equal(activityAt(d, "demo-0", 2), -50);
  assert.equal(activityAt(d, "demo-1", 2), undefined);
});
test("SWC malformed input fails clearly", () => {
  assert.throws(() => parseSWC("1 0 1 2 3 1 8", "id"), /Missing/);
  assert.throws(
    () => parseSWC("1 0 1 2 3 1 -1\n1 0 1 2 3 1 -1", "id"),
    /duplicate/,
  );
});
