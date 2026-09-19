import { mkdirSync, writeFileSync } from "node:fs";
import { demoDataset, parseDataset } from "../src/data.ts";
import { simulateMotor } from "../src/motor/engine.ts";
import { controller } from "../src/controllers/neural-readout/controller.ts";
const dataset = demoDataset();
dataset.motor = simulateMotor(
  dataset,
  controller,
  { forward: 0, turn: 0, wing: 0 },
  30,
);
parseDataset(dataset);
mkdirSync("examples", { recursive: true });
writeFileSync("examples/neural-motor-run.json", JSON.stringify(dataset));
console.log(
  `Exported examples/neural-motor-run.json: ${dataset.neurons.length} neurons, ${dataset.motor.poses.length} motor poses. Import it into the viewer.`,
);
