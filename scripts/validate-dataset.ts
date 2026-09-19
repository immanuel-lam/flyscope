import { readFileSync, statSync } from "node:fs";
import { parseDataset } from "../src/data.ts";
const file = process.argv[2];
if (!file) {
  console.error("Usage: npm run validate:dataset -- path/to/run.json");
  process.exit(1);
}
try {
  if (statSync(file).size > 30 * 1024 * 1024)
    throw new Error("File exceeds browser import limit of 30 MB.");
  const d = parseDataset(JSON.parse(readFileSync(file, "utf8")));
  console.log(
    `Valid: ${d.id}@${d.version}; ${d.neurons.length} neurons; ${d.motor?.poses.length ?? 0} motor poses.`,
  );
} catch (e) {
  console.error((e as Error).message);
  process.exit(1);
}
