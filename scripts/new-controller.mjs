import { mkdir, writeFile, access } from "node:fs/promises";
import { resolve } from "node:path";
const id = process.argv[2];
if (
  !id ||
  !/^[a-z][a-z0-9-]{1,39}$/.test(id) ||
  ["off", "replay", "manual", "neural-readout"].includes(id)
) {
  console.error(
    "Usage: npm run new:controller -- my-controller (2–40 lowercase letters, numbers, hyphens; reserved names excluded)",
  );
  process.exit(1);
}
const folder = resolve("src/controllers", id),
  test = resolve("tests", `${id}.test.ts`);
for (const target of [folder, test]) {
  try {
    await access(target);
    console.error(`Refusing to overwrite ${target}`);
    process.exit(1);
  } catch (e) {
    if (e.code !== "ENOENT") throw e;
  }
}
await mkdir(folder, { recursive: true });
await mkdir(resolve("tests"), { recursive: true });
await writeFile(
  resolve(folder, "controller.ts"),
  `import type { MotorController } from '../../motor/types.ts';

export const controller: MotorController = {
  id: '${id}',
  label: '${id}',
  description: 'Synthetic controller scaffold. Replace this with model assumptions and source provenance.',
  // Replace this gate with an explicit dataset/version/channel contract if using neurons.
  unsupported: () => null,
  create: (_context) => ({
    step: (_observation) => {
      // forward: [-3,3] mm/s; turn: [-2,2] rad/s; wing: [0,1].
      // Use observation.activity('string-id'); undefined means missing data, NOT zero.
      // This neutral template deliberately produces no motion.
      return { forward: 0, turn: 0, wing: 0 };
    },
  }),
};
`,
);
await writeFile(
  test,
  `import { test } from 'node:test';
import assert from 'node:assert/strict';
import { demoDataset } from '../src/data.ts';
import { simulateMotor } from '../src/motor/engine.ts';
import { controller } from '../src/controllers/${id}/controller.ts';

test('${id}: deterministic valid motor output', () => {
  const dataset = demoDataset();
  const params = {forward: 0, turn: 0, wing: 0};
  const a = simulateMotor(dataset, controller, params, 1);
  const b = simulateMotor(dataset, controller, params, 1);
  assert.deepEqual(a, b);
  assert.equal(a.times.at(-1), 1);
  // Add task-specific assertions, including a missing-input test if using neurons.
});
`,
);
console.log(
  `Created ${folder}/controller.ts and ${test}. Vite discovers the controller automatically. Read docs/AGENT_INTEGRATION.md, implement the task, and run npm test && npm run build.`,
);
